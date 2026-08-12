//! Replay a strict FP32 PyTorch oracle on the production WGSL WGPU path.

#![recursion_limit = "512"]

use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use burn::{
    backend::wgpu::{
        AutoCompiler, MemoryConfiguration, RuntimeOptions, WgpuDevice, WgpuRuntime,
        graphics::AutoGraphicsApi, init_setup,
    },
    tensor::{Bool, Int, Tensor, TensorData},
};
use clap::{Parser, ValueEnum};
use cubecl::prelude::Runtime;
use irodori_tts_burn::{
    AuxConditionInput, CfgGuidanceMode, ConditioningSignal, EncodedCondition, GuidanceConfig,
    InferenceBuilder, InferenceEngine, SamplerForwardEvaluation, SamplerForwardLane, SamplerMethod,
    SamplerParams, SamplerWorkReport, SamplingRequest, WgslInferenceEngine, codec::DacVaeCodec,
    inference::Ready, load_codec, unpatchify_latent, validation::AudioMetrics,
};
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing_subscriber::EnvFilter;

type WgpuRt = WgpuRuntime<AutoCompiler>;

const ORACLE_FORMAT: &str = "irodori-v4-precision-oracle-v1";
const UPSTREAM_COMMIT: &str = "9f19d9a9048099a4b978a762d0509228fe624e3f";
const MODEL_REVISION: &str = "e4aaac4df355ff560dcd35e0dae272c3a759317b";
const MODEL_SHA256: &str = "5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593";
const CODEC_REVISION: &str = "47376ee24834d7a05a48ebabfe3cde29b3c5e214";
const CODEC_SHA256: &str = "db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5";
const CONVERTED_CODEC_SHA256: &str =
    "4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1";
const TEXT: &str = "こんにちは。";
const DEFAULT_TASKS_MAX: usize = 32;
const MAX_REPEATS: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Precision {
    Fp32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Execution {
    /// Burn's unfused WGPU tensor graph, retained as a same-backend oracle.
    Burn,
    Wgsl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum MemoryConfig {
    SubSlices,
    ExclusivePages,
}

impl Execution {
    const fn label(self) -> &'static str {
        match self {
            Self::Burn => "burn",
            Self::Wgsl => "wgsl",
        }
    }
}

impl MemoryConfig {
    const fn label(self) -> &'static str {
        match self {
            Self::SubSlices => "sub-slices",
            Self::ExclusivePages => "exclusive-pages",
        }
    }

    fn runtime(self) -> MemoryConfiguration {
        match self {
            Self::SubSlices => MemoryConfiguration::SubSlices,
            Self::ExclusivePages => MemoryConfiguration::ExclusivePages,
        }
    }
}

impl Precision {
    const fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
        }
    }

    const fn native_dtype(self) -> &'static str {
        match self {
            Self::Fp32 => "float32",
        }
    }

    const fn safetensors_dtype(self) -> Dtype {
        match self {
            Self::Fp32 => Dtype::F32,
        }
    }
}

fn parse_repeats(value: &str) -> std::result::Result<usize, String> {
    let repeats = value
        .parse::<usize>()
        .map_err(|error| format!("invalid repetition count {value:?}: {error}"))?;
    if !(1..=MAX_REPEATS).contains(&repeats) {
        return Err(format!(
            "repetition count {repeats} is outside the supported range 1..={MAX_REPEATS}"
        ));
    }
    Ok(repeats)
}

fn parse_tasks_max(value: &str) -> std::result::Result<usize, String> {
    let tasks_max = value
        .parse::<usize>()
        .map_err(|error| format!("invalid task aggregation limit {value:?}: {error}"))?;
    if tasks_max == 0 {
        return Err("task aggregation limit must be a positive integer".to_owned());
    }
    Ok(tasks_max)
}

#[derive(Clone, Copy, Debug)]
struct GateArgs {
    max_abs: Option<f64>,
    mean_abs: Option<f64>,
    rmse: Option<f64>,
    min_snr_db: Option<f64>,
    min_cosine: Option<f64>,
}

impl GateArgs {
    fn resolve(self, enforce: bool, label: &str) -> Result<Option<Gates>> {
        let values = [
            self.max_abs,
            self.mean_abs,
            self.rmse,
            self.min_snr_db,
            self.min_cosine,
        ];
        if !enforce {
            ensure!(
                values.iter().all(Option::is_none),
                "{label} thresholds require --enforce; report-only mode never applies hidden gates"
            );
            return Ok(None);
        }
        let (Some(max_abs), Some(mean_abs), Some(rmse), Some(min_snr_db), Some(min_cosine)) = (
            self.max_abs,
            self.mean_abs,
            self.rmse,
            self.min_snr_db,
            self.min_cosine,
        ) else {
            anyhow::bail!(
                "--enforce requires all five {label} gates: max_abs, mean_abs, rmse, min_snr_db, and min_cosine"
            );
        };
        let gates = Gates {
            max_abs,
            mean_abs,
            rmse,
            min_snr_db,
            min_cosine,
        };
        gates.validate(label)?;
        Ok(Some(gates))
    }
}

#[derive(Clone, Copy, Debug)]
struct Gates {
    max_abs: f64,
    mean_abs: f64,
    rmse: f64,
    min_snr_db: f64,
    min_cosine: f64,
}

impl Gates {
    fn validate(self, label: &str) -> Result<()> {
        for (name, value) in [
            ("max_abs", self.max_abs),
            ("mean_abs", self.mean_abs),
            ("rmse", self.rmse),
        ] {
            ensure!(
                value.is_finite() && value >= 0.0,
                "{label} {name} threshold must be finite and non-negative"
            );
        }
        ensure!(
            self.min_snr_db.is_finite(),
            "{label} min_snr_db threshold must be finite"
        );
        ensure!(
            self.min_cosine.is_finite() && (-1.0..=1.0).contains(&self.min_cosine),
            "{label} min_cosine threshold must be finite and within [-1, 1]"
        );
        Ok(())
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "validate_v4_precision",
    about = "Replay a strict FP32 v4 PyTorch oracle through production WGSL WGPU"
)]
struct Args {
    /// Execution policy. This branch exposes production WGSL only.
    #[arg(long, value_enum, default_value = "wgsl")]
    execution: Execution,

    /// Backend element precision. This branch exposes strict FP32 only.
    #[arg(long, value_enum, default_value = "fp32")]
    precision: Precision,

    /// Fixture produced by scripts/export_v4_precision_oracle.py.
    #[arg(long)]
    fixture: PathBuf,

    /// Required out-of-band SHA-256 printed by the exporter.
    #[arg(long)]
    fixture_sha256: String,

    /// Official Aratako/Irodori-TTS-v4-Small model.safetensors.
    #[arg(long)]
    checkpoint: PathBuf,

    /// Rust-converted Semantic-DACVAE weights.
    #[arg(long, default_value = "target/v4_dacvae_weights.safetensors")]
    codec_weights: PathBuf,

    /// Optional PCM16 WAV written from the final Rust decoder repetition.
    #[arg(long)]
    output_wav: Option<PathBuf>,

    /// Persistent CubeCL cache root, uniquely namespaced for this adapter.
    #[arg(long, value_name = "DIR")]
    cubecl_cache_dir: Option<PathBuf>,

    /// Explicit WGPU discrete-adapter enumeration index.
    #[arg(long, default_value_t = 0)]
    adapter_index: usize,

    /// Maximum CubeCL compute tasks aggregated into one GPU command.
    #[arg(
        long,
        default_value_t = DEFAULT_TASKS_MAX,
        value_parser = parse_tasks_max
    )]
    tasks_max: usize,

    /// CubeCL WGPU memory-management preset.
    #[arg(long, value_enum, default_value = "sub-slices")]
    memory_config: MemoryConfig,

    /// Number of RF and codec repetitions using the same loaded models.
    #[arg(long, default_value_t = 1, value_parser = parse_repeats)]
    repeats: usize,

    /// Apply explicit numerical gates. Without this flag the run is report-only.
    #[arg(long)]
    enforce: bool,

    #[arg(long)]
    latent_max_abs: Option<f64>,
    #[arg(long)]
    latent_mean_abs: Option<f64>,
    #[arg(long)]
    latent_rmse: Option<f64>,
    #[arg(long)]
    latent_min_snr_db: Option<f64>,
    #[arg(long)]
    latent_min_cosine: Option<f64>,

    #[arg(long)]
    waveform_max_abs: Option<f64>,
    #[arg(long)]
    waveform_mean_abs: Option<f64>,
    #[arg(long)]
    waveform_rmse: Option<f64>,
    #[arg(long)]
    waveform_min_snr_db: Option<f64>,
    #[arg(long)]
    waveform_min_cosine: Option<f64>,
}

impl Args {
    fn validate_execution_policy(&self) -> Result<()> {
        ensure!(
            matches!(self.execution, Execution::Burn | Execution::Wgsl)
                && self.precision == Precision::Fp32,
            "only strict-FP32 WGPU execution is supported"
        );
        Ok(())
    }

    fn gates(&self) -> Result<AcceptancePolicy> {
        let latent = GateArgs {
            max_abs: self.latent_max_abs,
            mean_abs: self.latent_mean_abs,
            rmse: self.latent_rmse,
            min_snr_db: self.latent_min_snr_db,
            min_cosine: self.latent_min_cosine,
        }
        .resolve(self.enforce, "latent")?;
        let waveform = GateArgs {
            max_abs: self.waveform_max_abs,
            mean_abs: self.waveform_mean_abs,
            rmse: self.waveform_rmse,
            min_snr_db: self.waveform_min_snr_db,
            min_cosine: self.waveform_min_cosine,
        }
        .resolve(self.enforce, "waveform")?;
        let policy = match (latent, waveform) {
            (Some(latent), Some(waveform)) => AcceptancePolicy::Enforce { latent, waveform },
            (None, None) => AcceptancePolicy::ReportOnly,
            _ => anyhow::bail!("latent and waveform acceptance modes are inconsistent"),
        };
        Ok(policy)
    }
}

#[derive(Clone, Copy, Debug)]
enum AcceptancePolicy {
    ReportOnly,
    Enforce { latent: Gates, waveform: Gates },
}

#[derive(Debug, Deserialize)]
struct OraclePayload {
    format: String,
    upstream_commit: String,
    model_revision: String,
    model_sha256: String,
    codec_revision: String,
    codec_sha256: String,
    precision: String,
    native_dtype: String,
    math_policy: MathPolicy,
    noise_contract: NoiseContract,
    parameters: OracleParameters,
    config: OracleConfig,
    tensor_manifest: HashMap<String, TensorManifestEntry>,
}

#[derive(Debug, Deserialize)]
struct MathPolicy {
    autocast: bool,
    cuda_matmul_allow_tf32: bool,
    cudnn_allow_tf32: bool,
    float32_matmul_precision: String,
}

#[derive(Debug, Deserialize)]
struct NoiseContract {
    source_fixture_sha256: String,
    source_key: String,
    source_dtype: String,
    source_shape: Vec<usize>,
    source_tensor_sha256: String,
    effective_key: String,
    effective_dtype: String,
    effective_tensor_sha256: String,
    cast_count: usize,
    sampler_randn_interceptions: usize,
}

#[derive(Debug, Deserialize)]
struct OracleParameters {
    text: String,
    caption: Option<String>,
    no_ref: bool,
    seconds: f64,
    num_steps: usize,
    seed: u64,
    model_precision: String,
    codec_precision: String,
    cfg_guidance_mode: String,
    cfg_scale_text: f64,
    cfg_scale_caption: f64,
    cfg_scale_speaker: f64,
    cfg_min_t: f64,
    cfg_max_t: f64,
    t_schedule_mode: String,
    context_kv_cache: bool,
    compile_model: bool,
    trim_tail: bool,
    watermark: bool,
}

#[derive(Debug, Deserialize)]
struct OracleConfig {
    sample_rate: usize,
    target_samples: usize,
    #[serde(default)]
    decoded_samples: Option<usize>,
    latent_steps: usize,
    patched_steps: usize,
    euler_recurrence_max_abs: f64,
}

#[derive(Debug, Deserialize)]
struct TensorManifestEntry {
    shape: Vec<usize>,
    dtype: String,
    elements: usize,
    bytes: usize,
    sha256: String,
}

#[derive(Debug)]
struct Fixture {
    metadata: OraclePayload,
    text_ids: Vec<i32>,
    text_mask: Vec<bool>,
    caption_ids: Vec<i32>,
    caption_mask: Vec<bool>,
    source_noise: Vec<f32>,
    effective_noise: Vec<f32>,
    expected_text_state: Vec<f32>,
    expected_patched: Vec<f32>,
    expected_waveform: Vec<f32>,
}

#[derive(Debug)]
struct WgpuErrorMonitor {
    errors: Arc<Mutex<Vec<String>>>,
}

impl WgpuErrorMonitor {
    fn new() -> Self {
        Self {
            errors: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn callback_sink(&self) -> Arc<Mutex<Vec<String>>> {
        Arc::clone(&self.errors)
    }

    fn check(&self, stage: &str) -> Result<()> {
        let mut errors = self
            .errors
            .lock()
            .map_err(|_| anyhow::anyhow!("WGPU error monitor lock was poisoned after {stage}"))?;
        if errors.is_empty() {
            return Ok(());
        }
        let count = errors.len();
        let details = errors.drain(..).collect::<Vec<_>>().join("\n---\n");
        anyhow::bail!(
            "WGPU reported {count} uncaptured error(s) during {stage}; results are invalid:\n{details}"
        )
    }
}

fn initialize_tracing() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init()
        .map_err(|error| anyhow::anyhow!("failed to initialize tracing: {error}"))
}

fn normalized_sha256(value: &str, label: &str) -> Result<String> {
    let normalized = value.trim().to_ascii_lowercase();
    ensure!(
        normalized.len() == 64 && normalized.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{label} must be exactly 64 hexadecimal characters"
    );
    Ok(normalized)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Hash backend readback values using a host-independent byte contract.
///
/// PyTorch's `*_f32_sha256` fields hash contiguous IEEE-754 binary32 storage.
/// Serializing every bit pattern explicitly as little endian makes the Rust
/// repeat contract equivalent on little-endian hosts and deterministic on any
/// host architecture.
fn sha256_f32_le(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    values
        .iter()
        .for_each(|value| hasher.update(value.to_bits().to_le_bytes()));
    format!("{:x}", hasher.finalize())
}

fn repeat_tensor_sha256_line(name: &str, repetition: usize, values: &[f32]) -> String {
    format!(
        "repeat_tensor_sha256 name={name} repeat={repetition} encoding=ieee754-f32-le sha256={}",
        sha256_f32_le(values)
    )
}

fn verify_file_sha256(label: &str, path: &Path, expected: &str) -> Result<()> {
    let expected = normalized_sha256(expected, label)?;
    let file = File::open(path)
        .with_context(|| format!("failed to open {label} for SHA-256: {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .with_context(|| format!("failed to hash {label}: {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let actual = format!("{:x}", hasher.finalize());
    ensure!(
        actual == expected,
        "{label} SHA-256 mismatch for {}: got {actual}, expected {expected}",
        path.display()
    );
    println!("sha256: {label}={actual} path={}", path.display());
    Ok(())
}

fn dtype_label(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::BOOL => "bool",
        Dtype::I64 => "int64",
        Dtype::F16 => "float16",
        Dtype::BF16 => "bfloat16",
        Dtype::F32 => "float32",
        _ => "unsupported",
    }
}

fn checked_view<'data>(
    tensors: &SafeTensors<'data>,
    key: &str,
    dtype: Dtype,
    shape: &[usize],
) -> Result<safetensors::tensor::TensorView<'data>> {
    let view = tensors
        .tensor(key)
        .with_context(|| format!("fixture tensor {key:?} is missing"))?;
    ensure!(
        view.dtype() == dtype,
        "fixture tensor {key:?} has dtype {:?}, expected {dtype:?}",
        view.dtype()
    );
    ensure!(
        view.shape() == shape,
        "fixture tensor {key:?} has shape {:?}, expected {shape:?}",
        view.shape()
    );
    Ok(view)
}

fn read_float(
    tensors: &SafeTensors<'_>,
    key: &str,
    dtype: Dtype,
    shape: &[usize],
) -> Result<Vec<f32>> {
    let data = checked_view(tensors, key, dtype, shape)?.data();
    ensure!(
        dtype == Dtype::F32,
        "strict production fixture tensor {key:?} must be F32, got {dtype:?}"
    );
    data.chunks_exact(size_of::<f32>())
        .map(|chunk| {
            let bytes: [u8; size_of::<f32>()] = chunk
                .try_into()
                .map_err(|_| anyhow::anyhow!("invalid f32 bytes in {key:?}"))?;
            Ok(f32::from_le_bytes(bytes))
        })
        .collect()
}

fn read_i64_as_i32(tensors: &SafeTensors<'_>, key: &str, shape: &[usize]) -> Result<Vec<i32>> {
    checked_view(tensors, key, Dtype::I64, shape)?
        .data()
        .chunks_exact(size_of::<i64>())
        .map(|chunk| {
            let bytes: [u8; size_of::<i64>()] = chunk
                .try_into()
                .map_err(|_| anyhow::anyhow!("invalid i64 bytes in {key:?}"))?;
            i32::try_from(i64::from_le_bytes(bytes))
                .with_context(|| format!("token ID in fixture tensor {key:?} exceeds i32"))
        })
        .collect()
}

fn read_bool(tensors: &SafeTensors<'_>, key: &str, shape: &[usize]) -> Result<Vec<bool>> {
    checked_view(tensors, key, Dtype::BOOL, shape)?
        .data()
        .iter()
        .map(|&value| match value {
            0 => Ok(false),
            1 => Ok(true),
            other => anyhow::bail!("invalid boolean byte {other} in {key:?}"),
        })
        .collect()
}

fn validate_tensor_manifest(
    tensors: &SafeTensors<'_>,
    manifest: &HashMap<String, TensorManifestEntry>,
) -> Result<()> {
    let names = tensors.names();
    ensure!(
        names.len() == manifest.len(),
        "tensor manifest count {} does not match fixture tensor count {}",
        manifest.len(),
        names.len()
    );
    for name in names {
        let view = tensors.tensor(name)?;
        let entry = manifest
            .get(name)
            .with_context(|| format!("tensor manifest lacks {name:?}"))?;
        let actual_dtype = dtype_label(view.dtype());
        ensure!(
            actual_dtype != "unsupported",
            "fixture tensor {name:?} has unsupported dtype {:?}",
            view.dtype()
        );
        let elements = view
            .shape()
            .iter()
            .try_fold(1_usize, |product, &dimension| {
                product
                    .checked_mul(dimension)
                    .context("tensor element count overflow")
            })?;
        ensure!(
            entry.shape == view.shape(),
            "manifest shape mismatch for {name:?}"
        );
        ensure!(
            entry.dtype == actual_dtype,
            "manifest dtype mismatch for {name:?}: metadata={:?}, tensor={:?}",
            entry.dtype,
            view.dtype()
        );
        ensure!(
            entry.elements == elements,
            "manifest element count mismatch for {name:?}"
        );
        ensure!(
            entry.bytes == view.data().len(),
            "manifest byte count mismatch for {name:?}"
        );
        let expected = normalized_sha256(&entry.sha256, &format!("tensor {name}"))?;
        ensure!(
            sha256_bytes(view.data()) == expected,
            "manifest SHA-256 mismatch for tensor {name:?}"
        );
    }
    Ok(())
}

fn validate_metadata(metadata: &OraclePayload, precision: Precision) -> Result<()> {
    ensure!(
        metadata.format == ORACLE_FORMAT,
        "unsupported oracle format"
    );
    ensure!(
        metadata.upstream_commit == UPSTREAM_COMMIT,
        "upstream mismatch"
    );
    ensure!(
        metadata.model_revision == MODEL_REVISION,
        "model revision mismatch"
    );
    ensure!(metadata.model_sha256 == MODEL_SHA256, "model SHA mismatch");
    ensure!(
        metadata.codec_revision == CODEC_REVISION,
        "codec revision mismatch"
    );
    ensure!(metadata.codec_sha256 == CODEC_SHA256, "codec SHA mismatch");
    ensure!(
        metadata.precision == precision.label(),
        "oracle precision mismatch"
    );
    ensure!(
        metadata.native_dtype == precision.native_dtype(),
        "oracle native dtype mismatch"
    );
    ensure!(!metadata.math_policy.autocast, "oracle used autocast");
    ensure!(
        !metadata.math_policy.cuda_matmul_allow_tf32,
        "oracle enabled CUDA matmul TF32"
    );
    ensure!(
        !metadata.math_policy.cudnn_allow_tf32,
        "oracle enabled cuDNN TF32"
    );
    ensure!(
        metadata.math_policy.float32_matmul_precision == "highest",
        "oracle float32 matmul policy is not strict"
    );

    let noise = &metadata.noise_contract;
    ensure!(
        noise.source_fixture_sha256.len() == 64
            && noise
                .source_fixture_sha256
                .bytes()
                .all(|value| value.is_ascii_hexdigit() && !value.is_ascii_uppercase()),
        "noise source fixture SHA-256 is not canonical lowercase hexadecimal"
    );
    ensure!(
        noise.source_key == "initial_noise",
        "noise source key mismatch"
    );
    ensure!(
        noise.source_dtype == "float32",
        "noise source dtype mismatch"
    );
    ensure!(
        noise.source_shape == [1, metadata.config.patched_steps, 32],
        "noise source shape mismatch"
    );
    ensure!(
        noise.effective_key == "noise/effective",
        "effective noise key mismatch"
    );
    ensure!(
        noise.effective_dtype == precision.native_dtype(),
        "effective noise dtype mismatch"
    );
    normalized_sha256(&noise.source_tensor_sha256, "source noise tensor")?;
    normalized_sha256(&noise.effective_tensor_sha256, "effective noise tensor")?;
    let source_manifest = metadata
        .tensor_manifest
        .get("noise/source_fp32")
        .context("tensor manifest lacks source noise")?;
    let effective_manifest = metadata
        .tensor_manifest
        .get(&noise.effective_key)
        .context("tensor manifest lacks effective noise")?;
    ensure!(
        source_manifest.sha256 == noise.source_tensor_sha256,
        "noise contract source hash differs from the tensor manifest"
    );
    ensure!(
        effective_manifest.sha256 == noise.effective_tensor_sha256,
        "noise contract effective hash differs from the tensor manifest"
    );
    ensure!(noise.cast_count == 1, "noise must be cast exactly once");
    ensure!(
        noise.sampler_randn_interceptions == 1,
        "sampler must receive exactly one intercepted randn"
    );

    let parameters = &metadata.parameters;
    ensure!(parameters.text == TEXT, "oracle text mismatch");
    ensure!(parameters.caption.is_none(), "oracle must have no caption");
    ensure!(parameters.no_ref, "oracle must be no-reference");
    ensure!(
        parameters.seconds.is_finite() && parameters.seconds > 0.0,
        "oracle duration must be finite and positive"
    );
    ensure!(
        parameters.num_steps == 4,
        "oracle Euler step count mismatch"
    );
    ensure!(parameters.seed == 0, "oracle seed mismatch");
    ensure!(
        parameters.model_precision == precision.label()
            && parameters.codec_precision == precision.label(),
        "oracle model/codec precision mismatch"
    );
    ensure!(
        parameters.cfg_guidance_mode == "independent",
        "oracle CFG mode mismatch"
    );
    ensure!(parameters.cfg_scale_text == 3.0, "text CFG mismatch");
    ensure!(parameters.cfg_scale_caption == 3.0, "caption CFG mismatch");
    ensure!(parameters.cfg_scale_speaker == 5.0, "speaker CFG mismatch");
    ensure!(parameters.cfg_min_t == 0.5, "CFG minimum timestep mismatch");
    ensure!(parameters.cfg_max_t == 1.0, "CFG maximum timestep mismatch");
    ensure!(parameters.t_schedule_mode == "linear", "schedule mismatch");
    ensure!(
        parameters.context_kv_cache,
        "context KV cache must be enabled"
    );
    ensure!(!parameters.compile_model, "oracle model must be eager");
    ensure!(!parameters.trim_tail, "oracle tail must not be trimmed");
    ensure!(!parameters.watermark, "oracle must be pre-watermark");

    let config = &metadata.config;
    ensure!(config.sample_rate == 48_000, "sample rate mismatch");
    let expected_samples = samples_for_duration(parameters.seconds, config.sample_rate)?;
    ensure!(
        config.target_samples == expected_samples && config.target_samples > 0,
        "sample count mismatch"
    );
    let expected_latent_steps = config.target_samples.div_ceil(1_920);
    ensure!(
        config.latent_steps == expected_latent_steps,
        "latent length mismatch"
    );
    ensure!(
        config.patched_steps == config.latent_steps,
        "released v4 patch-size-one length mismatch"
    );
    let decoded_samples = config.decoded_samples.unwrap_or(config.target_samples);
    ensure!(
        decoded_samples == config.latent_steps * 1_920 && decoded_samples >= config.target_samples,
        "decoded sample extent mismatch"
    );
    ensure!(
        config.euler_recurrence_max_abs == 0.0,
        "oracle Euler recurrence is not exact"
    );
    Ok(())
}

fn samples_for_duration(seconds: f64, sample_rate: usize) -> Result<usize> {
    ensure!(
        seconds.is_finite() && seconds > 0.0 && sample_rate > 0,
        "duration and sample rate must be finite and positive"
    );
    let samples = (seconds * sample_rate as f64).round();
    ensure!(
        samples >= 1.0 && samples <= usize::MAX as f64,
        "duration sample count is outside usize range"
    );
    Ok(samples as usize)
}

fn load_fixture(path: &Path, precision: Precision) -> Result<Fixture> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read precision fixture {}", path.display()))?;
    let (_, header) = SafeTensors::read_metadata(&bytes)
        .with_context(|| format!("malformed precision fixture {}", path.display()))?;
    let oracle_json = header
        .metadata()
        .as_ref()
        .context("fixture has no metadata")?
        .get("oracle_json")
        .context("fixture metadata lacks oracle_json")?;
    let metadata: OraclePayload =
        serde_json::from_str(oracle_json).context("invalid oracle_json metadata")?;
    validate_metadata(&metadata, precision)?;
    let tensors = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("malformed precision fixture {}", path.display()))?;
    validate_tensor_manifest(&tensors, &metadata.tensor_manifest)?;

    let native_dtype = precision.safetensors_dtype();
    let patched_steps = metadata.config.patched_steps;
    let target_samples = metadata.config.target_samples;
    let source_noise = read_float(
        &tensors,
        "noise/source_fp32",
        Dtype::F32,
        &[1, patched_steps, 32],
    )?;
    let effective_noise = read_float(
        &tensors,
        "noise/effective",
        native_dtype,
        &[1, patched_steps, 32],
    )?;
    let ref_latent = read_float(
        &tensors,
        "inputs/ref_latent_dummy",
        native_dtype,
        &[1, 1, 32],
    )?;
    let ref_mask = read_bool(&tensors, "inputs/ref_mask_dummy", &[1, 1])?;
    ensure!(
        ref_latent.iter().all(|&value| value == 0.0),
        "reference sentinel must be zero"
    );
    ensure!(
        ref_mask.iter().all(|&value| !value),
        "reference mask must be false"
    );

    let fixture = Fixture {
        text_ids: read_i64_as_i32(&tensors, "inputs/text_input_ids", &[1, 256])?,
        text_mask: read_bool(&tensors, "inputs/text_mask", &[1, 256])?,
        caption_ids: read_i64_as_i32(&tensors, "inputs/caption_input_ids", &[1, 512])?,
        caption_mask: read_bool(&tensors, "inputs/caption_mask", &[1, 512])?,
        source_noise,
        effective_noise,
        expected_text_state: read_float(
            &tensors,
            "conditions/text_state_cond",
            native_dtype,
            &[1, 256, 512],
        )?,
        expected_patched: read_float(
            &tensors,
            "final_patched_latent",
            native_dtype,
            &[1, patched_steps, 32],
        )?,
        expected_waveform: read_float(
            &tensors,
            "raw_decoded_waveform",
            native_dtype,
            &[1, target_samples],
        )?,
        metadata,
    };
    for (label, values) in [
        ("source noise", fixture.source_noise.as_slice()),
        ("effective noise", fixture.effective_noise.as_slice()),
        ("final latent", fixture.expected_patched.as_slice()),
        ("raw waveform", fixture.expected_waveform.as_slice()),
    ] {
        ensure!(
            values.iter().all(|value| value.is_finite()),
            "{label} contains non-finite values"
        );
    }
    ensure!(
        fixture.caption_mask.iter().all(|&value| !value),
        "caption mask must be all false"
    );
    ensure!(
        fixture.text_mask.iter().any(|&value| value),
        "text mask must contain a valid token"
    );
    Ok(fixture)
}

fn initialize_wgpu(
    adapter_index: usize,
    tasks_max: usize,
    memory_config: MemoryConfig,
) -> (WgpuDevice, WgpuErrorMonitor) {
    let device = WgpuDevice::DiscreteGpu(adapter_index);
    let setup = init_setup::<AutoGraphicsApi>(
        &device,
        RuntimeOptions {
            tasks_max,
            memory_config: memory_config.runtime(),
        },
    );
    let monitor = WgpuErrorMonitor::new();
    let callback_errors = monitor.callback_sink();
    setup.device.on_uncaptured_error(Arc::new(move |error| {
        let message = error.to_string();
        tracing::error!(target: "wgpu_uncaptured", error = %message, "uncaptured WGPU error");
        if let Ok(mut errors) = callback_errors.lock() {
            errors.push(message);
        }
    }));
    let info = setup.adapter.get_info();
    println!(
        "wgpu_adapter: index={adapter_index} name={:?} backend={:?} device_type={:?} tasks_max={tasks_max} memory_config={}",
        info.name,
        info.backend,
        info.device_type,
        memory_config.label()
    );
    (device, monitor)
}

fn synchronize_and_check_wgpu(
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    stage: &str,
) -> Result<()> {
    let client = WgpuRt::client(device);
    let sync_result = cubecl::future::block_on(client.sync());
    monitor.check(stage)?;
    sync_result.with_context(|| format!("CubeCL synchronization failed after {stage}"))
}

fn cleanup_unused_wgpu_memory(
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    stage: &str,
) -> Result<()> {
    let client = WgpuRt::client(device);
    let before = client
        .memory_usage()
        .with_context(|| format!("failed to query WGPU memory before {stage}"))?;
    client.memory_cleanup();
    synchronize_and_check_wgpu(device, monitor, stage)?;
    let after = client
        .memory_usage()
        .with_context(|| format!("failed to query WGPU memory after {stage}"))?;
    ensure!(
        after.bytes_in_use <= before.bytes_in_use,
        "WGPU cleanup increased live bytes at {stage}: before={}, after={}",
        before.bytes_in_use,
        after.bytes_in_use
    );
    println!(
        "wgpu_memory_cleanup stage={stage:?} before_allocs={} before_in_use_bytes={} before_reserved_bytes={} after_allocs={} after_in_use_bytes={} after_reserved_bytes={}",
        before.number_allocs,
        before.bytes_in_use,
        before.bytes_reserved,
        after.number_allocs,
        after.bytes_in_use,
        after.bytes_reserved
    );
    Ok(())
}

fn ensure_metrics_finite(label: &str, metrics: &AudioMetrics) -> Result<()> {
    ensure!(metrics.sample_count > 0, "{label} has no samples");
    for (name, value) in [
        ("max_abs", metrics.max_abs_error),
        ("mean_abs", metrics.mean_abs_error),
        ("rmse", metrics.root_mean_square_error),
        ("cosine", metrics.cosine_similarity),
    ] {
        ensure!(value.is_finite(), "{label} {name} is non-finite ({value})");
    }
    ensure!(
        metrics.signal_to_noise_db.is_finite() || metrics.signal_to_noise_db == f64::INFINITY,
        "{label} snr_db is invalid ({})",
        metrics.signal_to_noise_db
    );
    Ok(())
}

fn print_metrics(label: &str, metrics: &AudioMetrics) {
    println!(
        "{label}: count={} max_abs={:.9e} mean_abs={:.9e} rmse={:.9e} snr_db={:.6} cosine={:.12}",
        metrics.sample_count,
        metrics.max_abs_error,
        metrics.mean_abs_error,
        metrics.root_mean_square_error,
        metrics.signal_to_noise_db,
        metrics.cosine_similarity,
    );
}

fn enforce_metrics(label: &str, metrics: &AudioMetrics, gates: Gates) -> Result<()> {
    ensure!(
        metrics.max_abs_error <= gates.max_abs,
        "{label} max_abs {:.9e} exceeds {:.9e}",
        metrics.max_abs_error,
        gates.max_abs
    );
    ensure!(
        metrics.mean_abs_error <= gates.mean_abs,
        "{label} mean_abs {:.9e} exceeds {:.9e}",
        metrics.mean_abs_error,
        gates.mean_abs
    );
    ensure!(
        metrics.root_mean_square_error <= gates.rmse,
        "{label} RMSE {:.9e} exceeds {:.9e}",
        metrics.root_mean_square_error,
        gates.rmse
    );
    ensure!(
        metrics.signal_to_noise_db >= gates.min_snr_db,
        "{label} SNR {:.6} is below {:.6}",
        metrics.signal_to_noise_db,
        gates.min_snr_db
    );
    ensure!(
        metrics.cosine_similarity >= gates.min_cosine,
        "{label} cosine {:.12} is below {:.12}",
        metrics.cosine_similarity,
        gates.min_cosine
    );
    Ok(())
}

fn compare(label: &str, reference: &[f32], candidate: &[f32], gates: Option<Gates>) -> Result<()> {
    let metrics = AudioMetrics::compare(reference, candidate)
        .with_context(|| format!("failed to compare {label}"))?;
    ensure_metrics_finite(label, &metrics)?;
    print_metrics(label, &metrics);
    if let Some(gates) = gates {
        enforce_metrics(label, &metrics, gates)?;
    }
    Ok(())
}

fn same_f32_bits(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(&left, &right)| left.to_bits() == right.to_bits())
}

fn f32_to_pcm16(sample: f32) -> i16 {
    let scaled = (sample.clamp(-1.0, 1.0) * 32768.0).round();
    scaled.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
}

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create WAV directory {}", parent.display()))?;
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("failed to create WAV {}", path.display()))?;
    samples.iter().try_for_each(|&sample| {
        writer
            .write_sample(f32_to_pcm16(sample))
            .context("failed to write PCM sample")
    })?;
    writer.finalize().context("failed to finalize WAV")?;
    Ok(())
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
struct RfTimingBoundaryReport {
    schema_version: u32,
    clock: &'static str,
    pre_start_device_sync: bool,
    work_report_inside_timed_region: bool,
    enqueue_return_s: f64,
    sample_device_complete_s: f64,
    primary_includes_final_latent_readback: bool,
    final_latent_readback_elements: usize,
    sample_and_readback_s: f64,
    cpu_readback_dtype: &'static str,
    cpu_readback_owned: bool,
    cpu_readback_contiguous: bool,
    secondary_stops_after_readback_sync: bool,
    primary_metric: &'static str,
    secondary_metric: &'static str,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
struct CodecTimingBoundaryReport {
    schema_version: u32,
    clock: &'static str,
    pre_start_device_sync: bool,
    enqueue_return_s: f64,
    decode_device_complete_s: f64,
    primary_includes_waveform_readback: bool,
    waveform_readback_elements: usize,
    decode_and_readback_s: f64,
    cpu_readback_dtype: &'static str,
    cpu_readback_owned: bool,
    cpu_readback_contiguous: bool,
    secondary_stops_after_readback_sync: bool,
    primary_metric: &'static str,
    secondary_metric: &'static str,
}

fn validate_product_work_report(
    report: &SamplerWorkReport,
    speaker_patch_size: usize,
    latent_sequence: usize,
    specialized_wgsl: bool,
) -> Result<()> {
    let expected_schedule = [0x3f7f_be77, 0x3f3f_ced9, 0x3eff_be77, 0x3e7f_be77, 0];
    ensure!(report.schema_version == 1, "RF work report schema mismatch");
    ensure!(
        report.method == SamplerMethod::Euler,
        "RF work report sampler mismatch"
    );
    ensure!(
        report.guidance_mode == CfgGuidanceMode::Independent,
        "RF work report guidance mode mismatch"
    );
    ensure!(report.num_steps == 4, "RF work report step count mismatch");
    ensure!(
        report.schedule_f32_bits == expected_schedule,
        "RF work report schedule bits mismatch: got {:?}, expected {:?}",
        report.schedule_f32_bits,
        expected_schedule
    );
    ensure!(
        report.requested.batch_rows == 1
            && report.requested.latent_sequence == latent_sequence
            && report.requested.latent_dim == 32
            && report.requested.text_tokens == 256
            && report.requested.speaker_tokens == Some(speaker_patch_size)
            && report.requested.caption_tokens == Some(512)
            && report.requested.joint_axis == latent_sequence + 256 + speaker_patch_size + 512,
        "RF requested geometry mismatch: {:?}",
        report.requested
    );
    for (label, geometry) in [
        ("compacted", &report.compacted),
        ("encoded", &report.encoded),
    ] {
        ensure!(
            geometry.batch_rows == 1
                && geometry.latent_sequence == latent_sequence
                && geometry.latent_dim == 32
                && geometry.text_tokens == 3
                && geometry.speaker_tokens.is_none()
                && geometry.caption_tokens.is_none()
                && geometry.joint_axis == latent_sequence + 3,
            "RF {label} geometry mismatch: {geometry:?}"
        );
    }
    ensure!(
        report.conditioned_text_mask_all_valid,
        "compacted text mask must be all-valid"
    );
    ensure!(
        report.enabled_cfg == [ConditioningSignal::Text],
        "effective CFG signals mismatch: {:?}",
        report.enabled_cfg
    );
    ensure!(
        !report.has_speaker_context && !report.has_caption_context,
        "masked auxiliary contexts unexpectedly remained active"
    );
    ensure!(
        report.context_kv.enabled,
        "context K/V cache must be enabled"
    );
    ensure!(
        report.context_kv.conditional_layers == 12 && report.context_kv.batched_cfg_layers == 12,
        "context K/V layer counts mismatch: {:?}",
        report.context_kv
    );
    if specialized_wgsl {
        ensure!(
            report.context_kv.derived_text_cfg_pair_used,
            "derived text CFG cache selector mismatch: {:?}",
            report.context_kv
        );
    }

    ensure!(
        report.forwards.len() == 4,
        "expected four whole-model forwards, got {}",
        report.forwards.len()
    );
    ensure!(
        report.effective_model_rows() == 6,
        "expected six effective model rows, got {}",
        report.effective_model_rows()
    );
    ensure!(
        report.model_layers == 12
            && report.whole_model_forwards == 4
            && report.model_block_calls == 48,
        "RF model/block work mismatch: layers={} forwards={} block_calls={}",
        report.model_layers,
        report.whole_model_forwards,
        report.model_block_calls
    );
    let expected_batches = [2, 2, 1, 1];
    let expected_cfg = [true, true, false, false];
    for (index, forward) in report.forwards.iter().enumerate() {
        ensure!(
            forward.step_index == index,
            "RF forward {index} step index mismatch"
        );
        ensure!(
            forward.evaluation == SamplerForwardEvaluation::Primary,
            "RF forward {index} was not a primary Euler evaluation"
        );
        ensure!(
            forward.timestep_f32_bits == expected_schedule[index],
            "RF forward {index} timestep bits mismatch"
        );
        ensure!(
            forward.cfg_active == expected_cfg[index],
            "RF forward {index} CFG activity mismatch"
        );
        ensure!(
            forward.batch_rows == expected_batches[index],
            "RF forward {index} batch mismatch: got {}",
            forward.batch_rows
        );
        let expected_lane = if expected_cfg[index] {
            SamplerForwardLane::BatchedIndependent
        } else {
            SamplerForwardLane::Conditional
        };
        ensure!(
            forward.lane == expected_lane,
            "RF forward {index} lane mismatch"
        );
        ensure!(
            forward.latent_sequence == latent_sequence
                && forward.latent_dim == 32
                && forward.text_tokens == 3
                && forward.speaker_tokens.is_none()
                && forward.caption_tokens.is_none()
                && forward.joint_axis == latent_sequence + 3
                && forward.context_kv_layers == 12,
            "RF forward {index} geometry/cache mismatch: {forward:?}"
        );
        if specialized_wgsl {
            ensure!(
                forward.fixed_cond_lookup_attempted
                    && forward.fixed_cond_lookup_hit
                    && forward.precomputed_cond_forward_used,
                "RF forward {index} fixed-condition selector mismatch: {forward:?}"
            );
        }
    }

    if specialized_wgsl {
        let fixed = &report.fixed_timestep_condition;
        ensure!(
            fixed.engine_cache_supplied
                && fixed.request_selected
                && fixed.lookup_attempts == 4
                && fixed.lookup_hits == 4
                && fixed.precomputed_forward_hits == 4
                && fixed.ordinary_cond_forwards == 0,
            "WGSL fixed timestep-condition work mismatch: {fixed:?}"
        );
    }
    Ok(())
}

trait ValidationExecution {
    type Engine;

    const LABEL: &'static str;
    const SPECIALIZED_WGSL: bool;
    fn build_engine(builder: InferenceBuilder<Ready>) -> Self::Engine;

    fn sample(
        engine: &Self::Engine,
        request: SamplingRequest,
    ) -> irodori_tts_burn::Result<(Tensor<3>, SamplerWorkReport)>;

    fn encode_conditions(
        engine: &Self::Engine,
        text_ids: Tensor<2, Int>,
        text_mask: Tensor<2, Bool>,
    ) -> irodori_tts_burn::Result<EncodedCondition>;

    fn prepare_codec(codec: &mut DacVaeCodec);

    fn decode(codec: &DacVaeCodec, latent: Tensor<3>) -> Tensor<3>;
}

struct WgslExecution;

struct BurnExecution;

impl ValidationExecution for BurnExecution {
    type Engine = InferenceEngine;

    const LABEL: &'static str = "burn";
    const SPECIALIZED_WGSL: bool = false;
    fn build_engine(builder: InferenceBuilder<Ready>) -> Self::Engine {
        builder.build()
    }

    fn sample(
        engine: &Self::Engine,
        request: SamplingRequest,
    ) -> irodori_tts_burn::Result<(Tensor<3>, SamplerWorkReport)> {
        engine.sample_with_work_report(request)
    }

    fn encode_conditions(
        engine: &Self::Engine,
        text_ids: Tensor<2, Int>,
        text_mask: Tensor<2, Bool>,
    ) -> irodori_tts_burn::Result<EncodedCondition> {
        engine
            .model()
            .encode_conditions(text_ids, text_mask, AuxConditionInput::None)
    }

    fn prepare_codec(_codec: &mut DacVaeCodec) {}

    fn decode(codec: &DacVaeCodec, latent: Tensor<3>) -> Tensor<3> {
        codec.decode(latent)
    }
}

impl ValidationExecution for WgslExecution {
    type Engine = WgslInferenceEngine;

    const LABEL: &'static str = "wgsl";
    const SPECIALIZED_WGSL: bool = true;
    fn build_engine(builder: InferenceBuilder<Ready>) -> Self::Engine {
        builder.build_wgsl()
    }

    fn sample(
        engine: &Self::Engine,
        request: SamplingRequest,
    ) -> irodori_tts_burn::Result<(Tensor<3>, SamplerWorkReport)> {
        engine.sample_with_work_report(request)
    }

    fn encode_conditions(
        engine: &Self::Engine,
        text_ids: Tensor<2, Int>,
        text_mask: Tensor<2, Bool>,
    ) -> irodori_tts_burn::Result<EncodedCondition> {
        engine.encode_conditions(text_ids, text_mask, AuxConditionInput::None)
    }

    fn prepare_codec(codec: &mut DacVaeCodec) {
        codec.prepare_decoder_for_wgsl();
    }

    fn decode(codec: &DacVaeCodec, latent: Tensor<3>) -> Tensor<3> {
        codec.decode_wgsl(latent)
    }
}

fn run_backend<E>(
    args: &Args,
    fixture: Fixture,
    policy: AcceptancePolicy,
    device: WgpuDevice,
    monitor: &WgpuErrorMonitor,
) -> Result<()>
where
    E: ValidationExecution,
{
    let (latent_gates, waveform_gates) = match policy {
        AcceptancePolicy::ReportOnly => (None, None),
        AcceptancePolicy::Enforce { latent, waveform } => (Some(latent), Some(waveform)),
    };

    let params = SamplerParams {
        num_steps: 4,
        method: SamplerMethod::Euler,
        guidance: GuidanceConfig {
            mode: CfgGuidanceMode::Independent,
            scale_text: 3.0,
            scale_caption: 3.0,
            scale_speaker: 5.0,
            min_t: 0.5,
            max_t: 1.0,
        },
        truncation_factor: None,
        temporal_rescale: None,
        speaker_kv: None,
        use_context_kv_cache: true,
    };

    let tensor_device = irodori_tts_burn::backend_config::strict_fp32_device(&device)?;
    let load_started = Instant::now();
    let loaded = InferenceBuilder::<_>::new(tensor_device.clone())
        .load_weights(&args.checkpoint)
        .with_context(|| format!("failed to load model {}", args.checkpoint.display()))?;
    let model_config = loaded.model_config().clone();
    ensure!(model_config.latent_dim == 32, "v4 latent_dim must be 32");
    ensure!(
        model_config.latent_patch_size == 1,
        "v4 patch size must be 1"
    );
    let speaker_patch_size = model_config
        .speaker_patch_size
        .context("v4 speaker_patch_size is missing")?;
    let engine = E::build_engine(loaded.with_sampling(params));
    synchronize_and_check_wgpu(&device, monitor, "model load and build")?;
    println!(
        "model_load_build_s={:.3} backend=WgpuRaw (no fusion, f32) execution={} precision={} repeats={}",
        load_started.elapsed().as_secs_f64(),
        E::LABEL,
        args.precision.label(),
        args.repeats
    );

    let text_ids =
        Tensor::<2, Int>::from_data(TensorData::new(fixture.text_ids, [1, 256]), &tensor_device);
    let text_mask =
        Tensor::<2, Bool>::from_data(TensorData::new(fixture.text_mask, [1, 256]), &tensor_device);
    let caption_ids = Tensor::<2, Int>::from_data(
        TensorData::new(fixture.caption_ids, [1, 512]),
        &tensor_device,
    );
    let caption_mask = Tensor::<2, Bool>::from_data(
        TensorData::new(fixture.caption_mask, [1, 512]),
        &tensor_device,
    );
    let ref_latent = Tensor::<3>::from_data(
        TensorData::new(
            vec![0.0_f32; speaker_patch_size * model_config.latent_dim],
            [1, speaker_patch_size, model_config.latent_dim],
        ),
        &tensor_device,
    );
    let ref_mask = Tensor::<2, Bool>::from_data(
        TensorData::new(vec![false; speaker_patch_size], [1, speaker_patch_size]),
        &tensor_device,
    );
    // This is the contract boundary: the canonical CPU fp32 values enter the
    // target backend exactly once. Burn converts them to B::FloatElem here.
    let patched_steps = fixture.metadata.config.patched_steps;
    let initial_noise = Tensor::<3>::from_data(
        TensorData::new(fixture.source_noise, [1, patched_steps, 32]),
        &tensor_device,
    );
    let rust_effective_noise = initial_noise
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .context("failed to read effective initial noise")?;
    synchronize_and_check_wgpu(
        &device,
        monitor,
        "effective initial-noise cast and readback",
    )?;
    ensure!(
        rust_effective_noise.iter().all(|value| value.is_finite()),
        "Rust effective initial noise contains non-finite values"
    );
    ensure!(
        same_f32_bits(&rust_effective_noise, &fixture.effective_noise),
        "Rust and PyTorch effective initial noise differ after their one-time target cast"
    );
    println!("noise_contract: source=f32 cast_count=1 effective_match=bit-exact");

    let encoded = E::encode_conditions(&engine, text_ids.clone(), text_mask.clone())?;
    let encoded_text = encoded
        .text_state
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .context("failed to read encoded text condition")?;
    compare(
        "encoded_text_condition",
        &fixture.expected_text_state,
        &encoded_text,
        latent_gates,
    )?;

    let request = SamplingRequest {
        text_ids,
        text_mask,
        ref_latent: Some(ref_latent),
        ref_mask: Some(ref_mask),
        sequence_length: patched_steps,
        caption_ids: Some(caption_ids),
        caption_mask: Some(caption_mask),
        initial_noise: Some(initial_noise),
    };
    let mut final_patched = None;
    let mut first_work_report: Option<SamplerWorkReport> = None;
    for repetition in 1..=args.repeats {
        synchronize_and_check_wgpu(
            &device,
            monitor,
            &format!("RF pre-timer synchronization repetition {repetition}"),
        )?;
        let started = Instant::now();
        let (actual, work_report) = E::sample(&engine, request.clone()).with_context(|| {
            format!("{} RF sampling failed at repetition {repetition}", E::LABEL)
        })?;
        let enqueue_return_s = started.elapsed().as_secs_f64();
        let sample_sync_result = cubecl::future::block_on(WgpuRt::client(&device).sync());
        let sample_device_complete_s = started.elapsed().as_secs_f64();
        monitor.check(&format!(
            "RF device-complete synchronization repetition {repetition}"
        ))?;
        sample_sync_result.with_context(|| {
            format!("CubeCL RF device synchronization failed at repetition {repetition}")
        })?;
        let values = actual
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .with_context(|| format!("failed to read RF repetition {repetition}"))?;
        synchronize_and_check_wgpu(
            &device,
            monitor,
            &format!("RF sampling and readback repetition {repetition}"),
        )?;
        let sample_and_readback_s = started.elapsed().as_secs_f64();
        validate_product_work_report(
            &work_report,
            speaker_patch_size,
            patched_steps,
            E::SPECIALIZED_WGSL,
        )?;
        if let Some(first) = &first_work_report {
            ensure!(
                &work_report == first,
                "RF work manifest changed between repetitions: first={first:?}, repetition {repetition}={work_report:?}"
            );
        } else {
            first_work_report = Some(work_report.clone());
        }
        let timing_report = RfTimingBoundaryReport {
            schema_version: 1,
            clock: "std::time::Instant",
            pre_start_device_sync: true,
            work_report_inside_timed_region: true,
            enqueue_return_s,
            sample_device_complete_s,
            primary_includes_final_latent_readback: false,
            final_latent_readback_elements: values.len(),
            sample_and_readback_s,
            cpu_readback_dtype: "float32",
            cpu_readback_owned: true,
            cpu_readback_contiguous: true,
            secondary_stops_after_readback_sync: true,
            primary_metric: "sample_device_complete_s",
            secondary_metric: "sample_and_readback_s",
        };
        println!(
            "rf_repeat={repetition}/{} sample_device_complete_s={:.6} sample_and_readback_s={:.6}",
            args.repeats, sample_device_complete_s, sample_and_readback_s,
        );
        println!(
            "rf_work_manifest={}",
            serde_json::to_string(&work_report).context("failed to serialize RF work manifest")?
        );
        println!(
            "rf_timing_manifest={}",
            serde_json::to_string(&timing_report)
                .context("failed to serialize RF timing manifest")?
        );
        println!(
            "{}",
            repeat_tensor_sha256_line("final_patched_latent", repetition, &values)
        );
        compare(
            &format!("final_patched_latent[{repetition}]"),
            &fixture.expected_patched,
            &values,
            latent_gates,
        )?;
        final_patched = Some(actual);
    }
    let final_patched = final_patched.context("RF repetitions produced no latent")?;
    drop(request);
    drop(engine);
    cleanup_unused_wgpu_memory(&device, monitor, "RF-to-codec explicit allocator cleanup")?;

    let final_unpatched = unpatchify_latent(
        final_patched,
        model_config.latent_patch_size,
        model_config.latent_dim,
    );
    let codec_started = Instant::now();
    let mut codec = load_codec(&args.codec_weights, &tensor_device)
        .with_context(|| format!("failed to load codec {}", args.codec_weights.display()))?;
    E::prepare_codec(&mut codec);
    ensure!(
        codec.sample_rate() == fixture.metadata.config.sample_rate,
        "codec sample rate mismatch"
    );
    synchronize_and_check_wgpu(&device, monitor, "codec load")?;
    println!("codec_load_s={:.3}", codec_started.elapsed().as_secs_f64());

    let mut final_waveform = None;
    let decoded_samples = fixture
        .metadata
        .config
        .decoded_samples
        .unwrap_or(fixture.metadata.config.target_samples);
    for repetition in 1..=args.repeats {
        synchronize_and_check_wgpu(
            &device,
            monitor,
            &format!("codec pre-timer synchronization repetition {repetition}"),
        )?;
        let started = Instant::now();
        let decoded = E::decode(&codec, final_unpatched.clone());
        let enqueue_return_s = started.elapsed().as_secs_f64();
        ensure!(
            decoded.dims() == [1, 1, decoded_samples],
            "decoded waveform shape mismatch at repetition {repetition}"
        );
        let decode_sync_result = cubecl::future::block_on(WgpuRt::client(&device).sync());
        let decode_device_complete_s = started.elapsed().as_secs_f64();
        monitor.check(&format!(
            "codec device-complete synchronization repetition {repetition}"
        ))?;
        decode_sync_result.with_context(|| {
            format!("CubeCL codec device synchronization failed at repetition {repetition}")
        })?;
        let values = decoded
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .with_context(|| format!("failed to read codec repetition {repetition}"))?;
        synchronize_and_check_wgpu(
            &device,
            monitor,
            &format!("codec decode and readback repetition {repetition}"),
        )?;
        let decode_and_readback_s = started.elapsed().as_secs_f64();
        let target_values = values[..fixture.metadata.config.target_samples].to_vec();
        let timing_report = CodecTimingBoundaryReport {
            schema_version: 1,
            clock: "std::time::Instant",
            pre_start_device_sync: true,
            enqueue_return_s,
            decode_device_complete_s,
            primary_includes_waveform_readback: false,
            waveform_readback_elements: values.len(),
            decode_and_readback_s,
            cpu_readback_dtype: "float32",
            cpu_readback_owned: true,
            cpu_readback_contiguous: true,
            secondary_stops_after_readback_sync: true,
            primary_metric: "decode_device_complete_s",
            secondary_metric: "decode_and_readback_s",
        };
        println!(
            "codec_repeat={repetition}/{} decode_device_complete_s={:.6} decode_and_readback_s={:.6}",
            args.repeats, decode_device_complete_s, decode_and_readback_s,
        );
        println!(
            "codec_timing_manifest={}",
            serde_json::to_string(&timing_report)
                .context("failed to serialize codec timing manifest")?
        );
        println!(
            "{}",
            repeat_tensor_sha256_line("raw_decoded_waveform", repetition, &target_values)
        );
        compare(
            &format!("raw_decoded_waveform[{repetition}]"),
            &fixture.expected_waveform,
            &target_values,
            waveform_gates,
        )?;
        final_waveform = Some(target_values);
    }
    let final_waveform = final_waveform.context("codec repetitions produced no waveform")?;
    if let Some(path) = &args.output_wav {
        write_wav(
            path,
            &final_waveform,
            u32::try_from(codec.sample_rate()).context("codec sample rate exceeds u32")?,
        )?;
        println!("output_wav={}", path.display());
    }
    synchronize_and_check_wgpu(&device, monitor, "validation completion")?;
    Ok(())
}

fn main() -> Result<()> {
    initialize_tracing()?;
    let args = Args::parse();
    if let Some(cache_dir) = args.cubecl_cache_dir.as_ref() {
        irodori_tts_burn::backend_config::configure_cubecl_persistent_cache(cache_dir)?;
    }
    args.validate_execution_policy()?;
    let policy = args.gates()?;
    match policy {
        AcceptancePolicy::ReportOnly => println!(
            "acceptance_mode=report-only numerical_drift_gates=unset structural_failures=fail-closed"
        ),
        AcceptancePolicy::Enforce { .. } => println!(
            "acceptance_mode=enforce numerical_drift_gates=explicit structural_failures=fail-closed"
        ),
    }
    ensure!(
        args.fixture.is_file(),
        "fixture not found: {}",
        args.fixture.display()
    );
    ensure!(
        args.checkpoint.is_file(),
        "checkpoint not found: {}",
        args.checkpoint.display()
    );
    ensure!(
        args.codec_weights.is_file(),
        "codec weights not found: {}",
        args.codec_weights.display()
    );
    verify_file_sha256("precision_fixture", &args.fixture, &args.fixture_sha256)?;
    verify_file_sha256("official_model", &args.checkpoint, MODEL_SHA256)?;
    verify_file_sha256(
        "converted_codec",
        &args.codec_weights,
        CONVERTED_CODEC_SHA256,
    )?;
    let fixture = load_fixture(&args.fixture, args.precision)?;
    println!(
        "oracle: format={} execution={} precision={} dtype={} upstream={} source_noise_sha256={}",
        fixture.metadata.format,
        args.execution.label(),
        fixture.metadata.precision,
        fixture.metadata.native_dtype,
        fixture.metadata.upstream_commit,
        fixture.metadata.noise_contract.source_tensor_sha256,
    );

    let (device, monitor) = initialize_wgpu(args.adapter_index, args.tasks_max, args.memory_config);
    match args.execution {
        Execution::Burn => run_backend::<BurnExecution>(&args, fixture, policy, device, &monitor),
        Execution::Wgsl => run_backend::<WgslExecution>(&args, fixture, policy, device, &monitor),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_samples_round_fractional_seconds_to_nearest_sample() -> Result<()> {
        ensure!(samples_for_duration(0.5, 48_000)? == 24_000);
        ensure!(samples_for_duration(10.2, 48_000)? == 489_600);
        ensure!(samples_for_duration(19.56, 48_000)? == 938_880);
        Ok(())
    }

    #[test]
    fn cli_defaults_to_production_fp32_report_only() -> Result<()> {
        let args = Args::try_parse_from([
            "validate_v4_precision",
            "--fixture",
            "oracle.safetensors",
            "--fixture-sha256",
            "0000000000000000000000000000000000000000000000000000000000000000",
            "--checkpoint",
            "model.safetensors",
        ])?;
        ensure!(
            args.precision == Precision::Fp32,
            "unexpected default precision"
        );
        ensure!(
            args.execution == Execution::Wgsl,
            "unexpected default execution"
        );
        ensure!(
            args.tasks_max == DEFAULT_TASKS_MAX,
            "unexpected default task aggregation limit"
        );
        ensure!(
            args.memory_config == MemoryConfig::SubSlices,
            "unexpected default memory configuration"
        );
        args.validate_execution_policy()?;
        ensure!(
            matches!(args.gates()?, AcceptancePolicy::ReportOnly),
            "CLI must default to report-only"
        );
        Ok(())
    }

    #[test]
    fn wgsl_execution_accepts_explicit_fp32() -> Result<()> {
        let args = Args::try_parse_from([
            "validate_v4_precision",
            "--execution",
            "wgsl",
            "--precision",
            "fp32",
            "--fixture",
            "oracle.safetensors",
            "--fixture-sha256",
            "0000000000000000000000000000000000000000000000000000000000000000",
            "--checkpoint",
            "model.safetensors",
        ])?;
        ensure!(args.execution == Execution::Wgsl, "WGSL was not selected");
        ensure!(args.precision == Precision::Fp32, "fp32 was not selected");
        args.validate_execution_policy()
    }

    #[test]
    fn cli_rejects_fp16_before_runtime() -> Result<()> {
        let args = Args::try_parse_from([
            "validate_v4_precision",
            "--execution",
            "wgsl",
            "--precision",
            "fp16",
            "--fixture",
            "oracle.safetensors",
            "--fixture-sha256",
            "0000000000000000000000000000000000000000000000000000000000000000",
            "--checkpoint",
            "model.safetensors",
        ]);
        ensure!(args.is_err(), "fp16 must fail closed during CLI parsing");
        Ok(())
    }

    #[test]
    fn cli_rejects_unknown_execution() {
        let parsed = Args::try_parse_from([
            "validate_v4_precision",
            "--execution",
            "cuda",
            "--fixture",
            "oracle.safetensors",
            "--fixture-sha256",
            "0000000000000000000000000000000000000000000000000000000000000000",
            "--checkpoint",
            "model.safetensors",
        ]);
        assert!(parsed.is_err());
    }

    #[test]
    fn cli_rejects_bf16_before_runtime() {
        let parsed = Args::try_parse_from([
            "validate_v4_precision",
            "--execution",
            "wgsl",
            "--precision",
            "bf16",
            "--fixture",
            "fp16-oracle-is-never-opened.safetensors",
            "--fixture-sha256",
            "0000000000000000000000000000000000000000000000000000000000000000",
            "--checkpoint",
            "model-is-never-opened.safetensors",
        ]);
        assert!(parsed.is_err());
    }

    #[test]
    fn cli_rejects_zero_tasks_max() {
        let parsed = Args::try_parse_from([
            "validate_v4_precision",
            "--tasks-max",
            "0",
            "--fixture",
            "oracle.safetensors",
            "--fixture-sha256",
            "0000000000000000000000000000000000000000000000000000000000000000",
            "--checkpoint",
            "model.safetensors",
        ]);
        assert!(parsed.is_err());
    }

    #[test]
    fn cli_accepts_single_task_commands() -> Result<()> {
        let args = Args::try_parse_from([
            "validate_v4_precision",
            "--tasks-max",
            "1",
            "--fixture",
            "oracle.safetensors",
            "--fixture-sha256",
            "0000000000000000000000000000000000000000000000000000000000000000",
            "--checkpoint",
            "model.safetensors",
        ])?;
        ensure!(args.tasks_max == 1, "explicit tasks_max was not preserved");
        Ok(())
    }

    #[test]
    fn cli_accepts_twelve_repetitions_for_two_plus_ten_protocol() -> Result<()> {
        let args = Args::try_parse_from([
            "validate_v4_precision",
            "--repeats",
            "12",
            "--fixture",
            "oracle.safetensors",
            "--fixture-sha256",
            "0000000000000000000000000000000000000000000000000000000000000000",
            "--checkpoint",
            "model.safetensors",
        ])?;
        ensure!(args.repeats == 12, "twelve repetitions were not preserved");
        Ok(())
    }

    #[test]
    fn cli_rejects_zero_repetitions() {
        let parsed = Args::try_parse_from([
            "validate_v4_precision",
            "--repeats",
            "0",
            "--fixture",
            "oracle.safetensors",
            "--fixture-sha256",
            "0000000000000000000000000000000000000000000000000000000000000000",
            "--checkpoint",
            "model.safetensors",
        ]);
        assert!(parsed.is_err());
    }

    #[test]
    fn cli_rejects_more_than_twelve_repetitions() {
        let parsed = Args::try_parse_from([
            "validate_v4_precision",
            "--repeats",
            "13",
            "--fixture",
            "oracle.safetensors",
            "--fixture-sha256",
            "0000000000000000000000000000000000000000000000000000000000000000",
            "--checkpoint",
            "model.safetensors",
        ]);
        assert!(parsed.is_err());
    }

    #[test]
    fn cli_accepts_both_explicit_memory_configs() -> Result<()> {
        for (value, expected) in [
            ("sub-slices", MemoryConfig::SubSlices),
            ("exclusive-pages", MemoryConfig::ExclusivePages),
        ] {
            let args = Args::try_parse_from([
                "validate_v4_precision",
                "--memory-config",
                value,
                "--fixture",
                "oracle.safetensors",
                "--fixture-sha256",
                "0000000000000000000000000000000000000000000000000000000000000000",
                "--checkpoint",
                "model.safetensors",
            ])?;
            ensure!(
                args.memory_config == expected,
                "explicit memory configuration {value:?} was not preserved"
            );
        }
        Ok(())
    }

    #[test]
    fn cli_rejects_unknown_memory_config() {
        let parsed = Args::try_parse_from([
            "validate_v4_precision",
            "--memory-config",
            "pooled",
            "--fixture",
            "oracle.safetensors",
            "--fixture-sha256",
            "0000000000000000000000000000000000000000000000000000000000000000",
            "--checkpoint",
            "model.safetensors",
        ]);
        assert!(parsed.is_err());
    }

    #[test]
    fn report_only_rejects_accidental_thresholds() {
        let gates = GateArgs {
            max_abs: Some(1.0),
            mean_abs: None,
            rmse: None,
            min_snr_db: None,
            min_cosine: None,
        };
        assert!(gates.resolve(false, "test").is_err());
    }

    #[test]
    fn enforcement_requires_every_metric() {
        let incomplete = GateArgs {
            max_abs: Some(1.0),
            mean_abs: Some(1.0),
            rmse: Some(1.0),
            min_snr_db: Some(0.0),
            min_cosine: None,
        };
        assert!(incomplete.resolve(true, "test").is_err());
    }

    #[test]
    fn f32_bit_comparison_distinguishes_signed_zero() {
        assert!(same_f32_bits(&[1.0], &[1.0]));
        assert!(!same_f32_bits(&[0.0], &[-0.0]));
    }

    #[test]
    fn f32_repeat_hash_uses_canonical_little_endian_bits() {
        let values = [0.0_f32, -0.0, 1.0, -2.5];
        assert_eq!(
            sha256_f32_le(&values),
            "283e4f49b9351bde5277c7018f4a353063da06644e860cfbaba79fea476349ec"
        );
    }

    #[test]
    fn repeat_hash_output_contract_is_stable() {
        assert_eq!(
            repeat_tensor_sha256_line("final_patched_latent", 7, &[1.0]),
            concat!(
                "repeat_tensor_sha256 name=final_patched_latent repeat=7 ",
                "encoding=ieee754-f32-le ",
                "sha256=e00e5eb9444182f352323374ef4e08ebcb784725fdd4fd612d7730540b3e0c8c"
            )
        );
    }

    #[test]
    fn codec_timing_manifest_lexical_contract_is_stable() -> Result<()> {
        let report = CodecTimingBoundaryReport {
            schema_version: 1,
            clock: "std::time::Instant",
            pre_start_device_sync: true,
            enqueue_return_s: 0.001,
            decode_device_complete_s: 0.04,
            primary_includes_waveform_readback: false,
            waveform_readback_elements: 96_000,
            decode_and_readback_s: 0.05,
            cpu_readback_dtype: "float32",
            cpu_readback_owned: true,
            cpu_readback_contiguous: true,
            secondary_stops_after_readback_sync: true,
            primary_metric: "decode_device_complete_s",
            secondary_metric: "decode_and_readback_s",
        };
        ensure!(
            serde_json::to_string(&report)?
                == concat!(
                    r#"{"schema_version":1,"clock":"std::time::Instant","#,
                    r#""pre_start_device_sync":true,"enqueue_return_s":0.001,"#,
                    r#""decode_device_complete_s":0.04,"#,
                    r#""primary_includes_waveform_readback":false,"#,
                    r#""waveform_readback_elements":96000,"#,
                    r#""decode_and_readback_s":0.05,"#,
                    r#""cpu_readback_dtype":"float32","#,
                    r#""cpu_readback_owned":true,"#,
                    r#""cpu_readback_contiguous":true,"#,
                    r#""secondary_stops_after_readback_sync":true,"#,
                    r#""primary_metric":"decode_device_complete_s","#,
                    r#""secondary_metric":"decode_and_readback_s"}"#,
                ),
            "codec timing manifest JSON changed"
        );
        Ok(())
    }
}
