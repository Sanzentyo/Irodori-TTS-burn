//! Replay the pinned Irodori-TTS v4 text-only oracle on WgpuRaw.
//!
//! The RF model is loaded from the official v4-Small checkpoint through the
//! production `InferenceBuilder::build_wgsl` policy. The initial noise and every conditioning input come from the
//! authoritative PyTorch fixture; tokenizer, RNG, and duration-predictor
//! validation belong to the separate production CLI checks.

#![recursion_limit = "512"]

use std::{
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use burn::{
    backend::wgpu::{
        MemoryConfiguration, RuntimeOptions, WgpuDevice, WgpuRuntime, graphics::AutoGraphicsApi,
        init_setup,
    },
    tensor::{Bool, Int, Tensor, TensorData},
};
use clap::{Parser, ValueEnum};
use cubecl::prelude::Runtime;
use irodori_tts_wgpu::{
    CfgGuidanceMode, GuidanceConfig, InferenceBuilder, SamplerMethod, SamplerParams,
    SamplingRequest, WgpuRaw, load_codec, unpatchify_latent, validation::AudioMetrics,
};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tracing_subscriber::EnvFilter;

const ORACLE_FORMAT: &str = "irodori-v4-e2e-oracle-v1";
const UPSTREAM_COMMIT: &str = "9f19d9a9048099a4b978a762d0509228fe624e3f";
const MODEL_REVISION: &str = "e4aaac4df355ff560dcd35e0dae272c3a759317b";
const MODEL_SHA256: &str = "5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593";
const CODEC_REVISION: &str = "47376ee24834d7a05a48ebabfe3cde29b3c5e214";
const CODEC_SHA256: &str = "db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5";
const FIXTURE_SHA256: &str = "8022b2baeed05e68dd2d335bebb10392b5817d1251e006413294ff597d363fc8";
const CONVERTED_CODEC_SHA256: &str =
    "4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1";
const LATENT_COSINE_MIN: f64 = 0.999_999;
const WAVEFORM_COSINE_MIN: f64 = 0.999_99;
const TEXT: &str = "こんにちは。";
const DEFAULT_TASKS_MAX: usize = 32;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Execution {
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

fn parse_repeats(value: &str) -> std::result::Result<usize, String> {
    let repeats = value
        .parse::<usize>()
        .map_err(|error| format!("invalid repetition count {value:?}: {error}"))?;
    if !(1..=10).contains(&repeats) {
        return Err(format!(
            "repetition count {repeats} is outside the supported range 1..=10"
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

#[derive(Debug, Parser)]
#[command(
    name = "validate_v4_e2e",
    about = "Replay the pinned v4 PyTorch E2E oracle through WGPU"
)]
struct Args {
    /// RF hot-path execution policy.
    #[arg(long, value_enum, default_value = "wgsl")]
    execution: Execution,

    /// Authoritative fixture produced by scripts/export_v4_e2e_oracle.py.
    #[arg(long, default_value = "/tmp/irodori-v4-e2e-oracle.safetensors")]
    fixture: PathBuf,

    /// Official Aratako/Irodori-TTS-v4-Small model.safetensors.
    #[arg(long)]
    checkpoint: PathBuf,

    /// Rust-converted Semantic-DACVAE weights.
    #[arg(long, default_value = "target/v4_dacvae_weights.safetensors")]
    codec_weights: PathBuf,

    /// PCM16 WAV written from the Rust decoder output.
    #[arg(long, default_value = "/tmp/irodori-v4-e2e-wgpu.wav")]
    output_wav: PathBuf,

    /// Explicit WGPU discrete-adapter enumeration index.
    ///
    /// This is intentionally not a CUDA/NVML ordinal. On the validation host,
    /// WGPU adapter 0 is the otherwise idle RTX 3060 Ti.
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

    /// Number of timed RF and codec repetitions using the same loaded models.
    #[arg(long, default_value_t = 1, value_parser = parse_repeats)]
    repeats: usize,

    /// Override the fail-closed final patched-latent maximum absolute error gate.
    #[arg(long, default_value_t = 5.0e-4)]
    latent_max_abs: f64,

    /// Override the fail-closed raw-waveform maximum absolute error gate.
    #[arg(long, default_value_t = 2.0e-3)]
    waveform_max_abs: f64,
}

#[derive(Debug, Deserialize)]
struct OraclePayload {
    format: String,
    upstream_commit: String,
    model_revision: String,
    model_sha256: String,
    codec_revision: String,
    codec_sha256: String,
    raw_wav_sha256: String,
    parameters: OracleParameters,
    config: OracleConfig,
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
    trim_tail: bool,
    watermark: bool,
}

#[derive(Debug, Deserialize)]
struct OracleConfig {
    sample_rate: usize,
    target_samples: usize,
    latent_steps: usize,
    patched_steps: usize,
    euler_recurrence_max_abs: f64,
    latent_redecode_max_abs: f64,
}

#[derive(Debug)]
struct Fixture {
    metadata: OraclePayload,
    text_ids: Vec<i32>,
    text_mask: Vec<bool>,
    caption_ids: Vec<i32>,
    caption_mask: Vec<bool>,
    initial_noise: Vec<f32>,
    expected_patched: Vec<f32>,
    expected_unpatched: Vec<f32>,
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
            "WGPU reported {count} uncaptured error(s) during {stage}; GPU results are invalid:\n{details}"
        )
    }
}

fn initialize_tracing() -> Result<()> {
    let filter = match EnvFilter::try_from_default_env() {
        Ok(filter) => filter,
        Err(_) => EnvFilter::new("warn"),
    };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init()
        .map_err(|error| anyhow::anyhow!("failed to initialize tracing: {error}"))
}

fn verify_file_sha256(label: &str, path: &Path, expected: &str) -> Result<()> {
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

fn validate_metadata(metadata: &OraclePayload) -> Result<()> {
    ensure!(
        metadata.format == ORACLE_FORMAT,
        "unsupported oracle format {:?}",
        metadata.format
    );
    ensure!(
        metadata.upstream_commit == UPSTREAM_COMMIT,
        "oracle upstream commit mismatch"
    );
    ensure!(
        metadata.model_revision == MODEL_REVISION,
        "oracle model revision mismatch"
    );
    ensure!(
        metadata.model_sha256 == MODEL_SHA256,
        "oracle model SHA-256 mismatch"
    );
    ensure!(
        metadata.codec_revision == CODEC_REVISION,
        "oracle codec revision mismatch"
    );
    ensure!(
        metadata.codec_sha256 == CODEC_SHA256,
        "oracle codec SHA-256 mismatch"
    );

    let parameters = &metadata.parameters;
    ensure!(
        parameters.text == TEXT,
        "oracle text mismatch: {:?}",
        parameters.text
    );
    ensure!(
        parameters.caption.is_none(),
        "oracle must not contain caption text"
    );
    ensure!(parameters.no_ref, "oracle must be a no-reference request");
    ensure!(
        parameters.seconds == 2.0,
        "oracle duration must be 2.0 seconds"
    );
    ensure!(
        parameters.num_steps == 4,
        "oracle must use four Euler steps"
    );
    ensure!(parameters.seed == 0, "oracle seed must be zero");
    ensure!(
        parameters.model_precision == "fp32",
        "oracle model precision must be fp32"
    );
    ensure!(
        parameters.codec_precision == "fp32",
        "oracle codec precision must be fp32"
    );
    ensure!(
        parameters.cfg_guidance_mode == "independent",
        "oracle CFG mode mismatch"
    );
    ensure!(parameters.cfg_scale_text == 3.0, "oracle text CFG mismatch");
    ensure!(
        parameters.cfg_scale_caption == 3.0,
        "oracle caption CFG mismatch"
    );
    ensure!(
        parameters.cfg_scale_speaker == 5.0,
        "oracle speaker CFG mismatch"
    );
    ensure!(
        parameters.cfg_min_t == 0.5,
        "oracle CFG minimum timestep mismatch"
    );
    ensure!(
        parameters.cfg_max_t == 1.0,
        "oracle CFG maximum timestep mismatch"
    );
    ensure!(
        parameters.t_schedule_mode == "linear",
        "oracle schedule must be linear"
    );
    ensure!(
        parameters.context_kv_cache,
        "oracle must enable context KV cache"
    );
    ensure!(!parameters.trim_tail, "oracle must not trim its tail");
    ensure!(!parameters.watermark, "oracle must not be watermarked");

    let config = &metadata.config;
    ensure!(config.sample_rate == 48_000, "oracle sample rate mismatch");
    ensure!(
        config.target_samples == 96_000,
        "oracle sample count mismatch"
    );
    ensure!(config.latent_steps == 50, "oracle latent length mismatch");
    ensure!(config.patched_steps == 50, "oracle patched length mismatch");
    ensure!(
        config.euler_recurrence_max_abs == 0.0,
        "oracle Euler recurrence is not exact"
    );
    ensure!(
        config.latent_redecode_max_abs == 0.0,
        "oracle latent re-decode is not exact"
    );
    Ok(())
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

fn read_f32(tensors: &SafeTensors<'_>, key: &str, shape: &[usize]) -> Result<Vec<f32>> {
    checked_view(tensors, key, Dtype::F32, shape)?
        .data()
        .chunks_exact(size_of::<f32>())
        .map(|chunk| {
            let bytes: [u8; size_of::<f32>()] = chunk
                .try_into()
                .map_err(|_| anyhow::anyhow!("invalid f32 bytes in fixture tensor {key:?}"))?;
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
                .map_err(|_| anyhow::anyhow!("invalid i64 bytes in fixture tensor {key:?}"))?;
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
            other => anyhow::bail!("invalid boolean byte {other} in fixture tensor {key:?}"),
        })
        .collect()
}

fn load_fixture(path: &Path) -> Result<Fixture> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read oracle fixture {}", path.display()))?;
    let (_, header) = SafeTensors::read_metadata(&bytes)
        .with_context(|| format!("malformed oracle fixture {}", path.display()))?;
    let metadata_values = header
        .metadata()
        .as_ref()
        .context("fixture has no metadata")?;
    let oracle_json = metadata_values
        .get("oracle_json")
        .context("fixture metadata key 'oracle_json' is missing")?;
    let metadata: OraclePayload =
        serde_json::from_str(oracle_json).context("invalid oracle_json metadata")?;
    validate_metadata(&metadata)?;

    let tensors = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("malformed oracle fixture {}", path.display()))?;

    let ref_latent = read_f32(&tensors, "inputs/ref_latent_dummy", &[1, 1, 32])?;
    let ref_mask = read_bool(&tensors, "inputs/ref_mask_dummy", &[1, 1])?;
    ensure!(
        ref_latent.iter().all(|&value| value == 0.0),
        "reference sentinel must contain only zeros"
    );
    ensure!(
        ref_mask.iter().all(|&value| !value),
        "reference sentinel mask must be all false"
    );

    let fixture = Fixture {
        metadata,
        text_ids: read_i64_as_i32(&tensors, "inputs/text_input_ids", &[1, 256])?,
        text_mask: read_bool(&tensors, "inputs/text_mask", &[1, 256])?,
        caption_ids: read_i64_as_i32(&tensors, "inputs/caption_input_ids", &[1, 512])?,
        caption_mask: read_bool(&tensors, "inputs/caption_mask", &[1, 512])?,
        initial_noise: read_f32(&tensors, "initial_noise", &[1, 50, 32])?,
        expected_patched: read_f32(&tensors, "final_patched_latent", &[1, 50, 32])?,
        expected_unpatched: read_f32(&tensors, "final_unpatched_latent", &[1, 50, 32])?,
        expected_waveform: read_f32(&tensors, "raw_decoded_waveform", &[1, 96_000])?,
    };
    ensure!(
        fixture.caption_mask.iter().all(|&value| !value),
        "caption mask must be all false for the text-only oracle"
    );
    ensure!(
        fixture.text_mask.iter().any(|&value| value),
        "text mask must contain at least one valid token"
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
    let error_monitor = WgpuErrorMonitor::new();
    let callback_errors = error_monitor.callback_sink();
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
    (device, error_monitor)
}

fn synchronize_and_check_wgpu(
    device: &WgpuDevice,
    error_monitor: &WgpuErrorMonitor,
    stage: &str,
) -> Result<()> {
    let client = WgpuRuntime::client(device);
    let sync_result = cubecl::future::block_on(client.sync());

    // WGPU validation errors do not flow through CubeCL's `sync` result: the
    // stream only scopes internal errors. Check the device-wide callback after
    // the synchronization has driven all pending callbacks.
    error_monitor.check(stage)?;
    sync_result.with_context(|| format!("CubeCL synchronization failed after {stage}"))
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

/// Convert normalized float audio to the production pipeline's signed PCM16 mapping.
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
        let pcm = f32_to_pcm16(sample);
        writer
            .write_sample(pcm)
            .context("failed to write PCM sample")
    })?;
    writer.finalize().context("failed to finalize WAV")?;
    Ok(())
}

fn ensure_metrics_finite(label: &str, metrics: &AudioMetrics) -> Result<()> {
    ensure!(metrics.sample_count > 0, "{label} has no samples");
    for (metric, value) in [
        ("max_abs", metrics.max_abs_error),
        ("mean_abs", metrics.mean_abs_error),
        ("rmse", metrics.root_mean_square_error),
        ("cosine", metrics.cosine_similarity),
    ] {
        ensure!(
            value.is_finite(),
            "{label} {metric} is non-finite ({value})"
        );
    }
    // An exact replay has zero error energy and therefore +infinite SNR.
    // Reject NaN and -infinity while accepting that mathematically valid case.
    ensure!(
        metrics.signal_to_noise_db.is_finite() || metrics.signal_to_noise_db == f64::INFINITY,
        "{label} snr_db is invalid ({})",
        metrics.signal_to_noise_db
    );
    Ok(())
}

fn enforce_acceptance(
    label: &str,
    metrics: &AudioMetrics,
    max_abs_limit: f64,
    cosine_min: f64,
) -> Result<()> {
    ensure_metrics_finite(label, metrics)?;
    ensure!(
        max_abs_limit.is_finite() && max_abs_limit >= 0.0,
        "{label} max_abs threshold must be finite and non-negative"
    );
    ensure!(
        metrics.max_abs_error <= max_abs_limit,
        "{label} max_abs={:.9e} exceeds threshold {max_abs_limit:.9e}",
        metrics.max_abs_error
    );
    ensure!(
        metrics.cosine_similarity >= cosine_min,
        "{label} cosine={:.12} is below threshold {cosine_min:.12}",
        metrics.cosine_similarity
    );
    Ok(())
}

fn validate_output(
    label: &str,
    expected: &[f32],
    actual: &[f32],
    max_abs_limit: f64,
    cosine_min: f64,
) -> Result<()> {
    let metrics = AudioMetrics::compare(expected, actual)
        .with_context(|| format!("failed to compare {label}"))?;
    print_metrics(label, &metrics);
    enforce_acceptance(label, &metrics, max_abs_limit, cosine_min)
}

fn main() -> Result<()> {
    initialize_tracing()?;
    let args = Args::parse();
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
    verify_file_sha256("oracle_fixture", &args.fixture, FIXTURE_SHA256)?;
    verify_file_sha256("official_model", &args.checkpoint, MODEL_SHA256)?;
    verify_file_sha256(
        "converted_codec",
        &args.codec_weights,
        CONVERTED_CODEC_SHA256,
    )?;

    let fixture_started = Instant::now();
    let fixture = load_fixture(&args.fixture)?;
    println!(
        "oracle: format={} upstream={} model_revision={} codec_revision={} raw_wav_sha256={}",
        fixture.metadata.format,
        fixture.metadata.upstream_commit,
        fixture.metadata.model_revision,
        fixture.metadata.codec_revision,
        fixture.metadata.raw_wav_sha256,
    );
    println!(
        "fixture_load_s={:.3} valid_text_tokens={} active_cfg=text-only",
        fixture_started.elapsed().as_secs_f64(),
        fixture.text_mask.iter().filter(|&&value| value).count(),
    );

    let (device, wgpu_errors) =
        initialize_wgpu(args.adapter_index, args.tasks_max, args.memory_config);

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

    let model_load_started = Instant::now();
    let loaded = InferenceBuilder::<WgpuRaw, _>::new(device.clone())
        .load_weights(&args.checkpoint)
        .with_context(|| format!("failed to load model {}", args.checkpoint.display()))?;
    let model_config = loaded.model_config().clone();
    ensure!(model_config.latent_dim == 32, "v4 latent_dim must be 32");
    ensure!(
        model_config.latent_patch_size == 1,
        "v4 latent_patch_size must be 1"
    );
    ensure!(
        model_config.use_pretrained_text_encoder(),
        "v4 must use pretrained text frontend"
    );
    ensure!(
        model_config.use_speaker_condition(),
        "v4 must expose speaker conditioning"
    );
    ensure!(
        model_config.use_caption_condition,
        "v4 must expose caption conditioning"
    );
    let speaker_patch_size = model_config
        .speaker_patch_size
        .context("v4 speaker_patch_size is missing")?;
    ensure!(
        speaker_patch_size > 0,
        "speaker_patch_size must be positive"
    );
    let ready = loaded.with_sampling(params);
    let engine = ready.build_wgsl();
    synchronize_and_check_wgpu(&device, &wgpu_errors, "model load and build")?;
    println!(
        "model_load_build_s={:.3} model_dim={} layers={} heads={} execution={} kv_cache=true repeats={}",
        model_load_started.elapsed().as_secs_f64(),
        model_config.model_dim,
        model_config.num_layers,
        model_config.num_heads,
        args.execution.label(),
        args.repeats,
    );

    let text_ids =
        Tensor::<WgpuRaw, 2, Int>::from_data(TensorData::new(fixture.text_ids, [1, 256]), &device);
    let text_mask = Tensor::<WgpuRaw, 2, Bool>::from_data(
        TensorData::new(fixture.text_mask, [1, 256]),
        &device,
    );
    let caption_ids = Tensor::<WgpuRaw, 2, Int>::from_data(
        TensorData::new(fixture.caption_ids, [1, 512]),
        &device,
    );
    let caption_mask = Tensor::<WgpuRaw, 2, Bool>::from_data(
        TensorData::new(fixture.caption_mask, [1, 512]),
        &device,
    );
    let ref_latent = Tensor::<WgpuRaw, 3>::from_data(
        TensorData::new(
            vec![0.0_f32; speaker_patch_size * model_config.latent_dim],
            [1, speaker_patch_size, model_config.latent_dim],
        ),
        &device,
    );
    let ref_mask = Tensor::<WgpuRaw, 2, Bool>::from_data(
        TensorData::new(vec![false; speaker_patch_size], [1, speaker_patch_size]),
        &device,
    );
    let initial_noise = Tensor::<WgpuRaw, 3>::from_data(
        TensorData::new(fixture.initial_noise, [1, 50, 32]),
        &device,
    );

    let oracle_request = SamplingRequest {
        text_ids,
        text_mask,
        ref_latent: Some(ref_latent),
        ref_mask: Some(ref_mask),
        sequence_length: 50,
        caption_ids: Some(caption_ids),
        caption_mask: Some(caption_mask),
        initial_noise: Some(initial_noise),
    };
    let mut last_patched_values = None;
    for repetition in 1..=args.repeats {
        let sample_started = Instant::now();
        let actual_patched = engine
            .sample(oracle_request.clone())
            .with_context(|| format!("production sampling failed in RF repetition {repetition}"))?;
        let actual_patched_values = actual_patched
            .into_data()
            .to_vec::<f32>()
            .with_context(|| format!("failed to read latent in RF repetition {repetition}"))?;
        let stage = format!("RF sampling and latent readback repetition {repetition}");
        synchronize_and_check_wgpu(&device, &wgpu_errors, &stage)?;
        println!(
            "rf_repeat={repetition}/{} sample_and_readback_s={:.3}",
            args.repeats,
            sample_started.elapsed().as_secs_f64()
        );
        let label = format!("final_patched_latent[{repetition}]");
        validate_output(
            &label,
            &fixture.expected_patched,
            &actual_patched_values,
            args.latent_max_abs,
            LATENT_COSINE_MIN,
        )?;
        last_patched_values = Some(actual_patched_values);
    }
    let actual_patched_values = last_patched_values.context("RF repetitions produced no latent")?;

    // Release the 3 GB RF model before allocating the codec on the 8 GB card.
    drop(engine);

    let actual_patched = Tensor::<WgpuRaw, 3>::from_data(
        TensorData::new(actual_patched_values, [1, 50, 32]),
        &device,
    );
    let actual_unpatched = unpatchify_latent(
        actual_patched,
        model_config.latent_patch_size,
        model_config.latent_dim,
    );
    let actual_unpatched_values = actual_unpatched
        .clone()
        .into_data()
        .to_vec::<f32>()
        .context("failed to read unpatched latent")?;
    synchronize_and_check_wgpu(&device, &wgpu_errors, "latent unpatchify and readback")?;
    let unpatched_metrics =
        AudioMetrics::compare(&fixture.expected_unpatched, &actual_unpatched_values)
            .context("failed to compare final unpatched latent")?;
    print_metrics("final_unpatched_latent", &unpatched_metrics);
    ensure_metrics_finite("final unpatched latent", &unpatched_metrics)?;

    let codec_load_started = Instant::now();
    let mut codec = load_codec::<WgpuRaw>(&args.codec_weights, &device)
        .with_context(|| format!("failed to load codec {}", args.codec_weights.display()))?;
    codec.prepare_decoder_for_wgsl();
    ensure!(
        codec.sample_rate() == fixture.metadata.config.sample_rate,
        "codec sample rate mismatch"
    );
    synchronize_and_check_wgpu(&device, &wgpu_errors, "codec load")?;
    println!(
        "codec_load_s={:.3}",
        codec_load_started.elapsed().as_secs_f64()
    );

    let mut last_decoded_values = None;
    for repetition in 1..=args.repeats {
        let decode_started = Instant::now();
        let decoded = codec.decode_wgsl(actual_unpatched.clone());
        let [batch, channels, samples] = decoded.dims();
        ensure!(
            [batch, channels, samples] == [1, 1, fixture.metadata.config.target_samples],
            "decoded waveform shape mismatch in codec repetition {repetition}: got [{batch}, {channels}, {samples}]"
        );
        let decoded_values = decoded.into_data().to_vec::<f32>().with_context(|| {
            format!("failed to read decoded waveform in codec repetition {repetition}")
        })?;
        let stage = format!("codec decode and waveform readback repetition {repetition}");
        synchronize_and_check_wgpu(&device, &wgpu_errors, &stage)?;
        println!(
            "codec_repeat={repetition}/{} decode_and_readback_s={:.3}",
            args.repeats,
            decode_started.elapsed().as_secs_f64()
        );
        let label = format!("raw_decoded_waveform[{repetition}]");
        validate_output(
            &label,
            &fixture.expected_waveform,
            &decoded_values,
            args.waveform_max_abs,
            WAVEFORM_COSINE_MIN,
        )?;
        last_decoded_values = Some(decoded_values);
    }
    let decoded_values = last_decoded_values.context("codec repetitions produced no waveform")?;

    write_wav(
        &args.output_wav,
        &decoded_values,
        u32::try_from(codec.sample_rate()).context("codec sample rate exceeds u32")?,
    )?;
    println!(
        "output_wav={} source_codec_repeat={}",
        args.output_wav.display(),
        args.repeats
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn required_cli_args() -> [&'static str; 3] {
        ["validate_v4_e2e", "--checkpoint", "model.safetensors"]
    }

    #[test]
    fn cli_runtime_defaults_match_production() {
        let args = Args::try_parse_from(required_cli_args()).unwrap();
        assert_eq!(args.tasks_max, DEFAULT_TASKS_MAX);
        assert_eq!(args.memory_config, MemoryConfig::SubSlices);
    }

    #[test]
    fn cli_accepts_explicit_runtime_options() {
        for (value, expected) in [
            ("sub-slices", MemoryConfig::SubSlices),
            ("exclusive-pages", MemoryConfig::ExclusivePages),
        ] {
            let args = Args::try_parse_from([
                "validate_v4_e2e",
                "--checkpoint",
                "model.safetensors",
                "--tasks-max",
                "1",
                "--memory-config",
                value,
            ])
            .unwrap();
            assert_eq!(args.tasks_max, 1);
            assert_eq!(args.memory_config, expected);
        }
    }

    #[test]
    fn cli_rejects_invalid_runtime_options() {
        for arguments in [vec!["--tasks-max", "0"], vec!["--memory-config", "pooled"]] {
            let parsed = Args::try_parse_from(required_cli_args().into_iter().chain(arguments));
            assert!(parsed.is_err());
        }
    }

    #[test]
    fn exact_replay_positive_infinite_snr_is_valid() {
        let metrics = AudioMetrics::compare(&[0.25, -0.5], &[0.25, -0.5]).unwrap();
        assert_eq!(metrics.signal_to_noise_db, f64::INFINITY);
        ensure_metrics_finite("exact replay", &metrics).unwrap();
    }

    #[test]
    fn negative_infinite_snr_is_rejected() {
        let metrics = AudioMetrics {
            sample_count: 1,
            max_abs_error: 1.0,
            mean_abs_error: 1.0,
            root_mean_square_error: 1.0,
            signal_to_noise_db: f64::NEG_INFINITY,
            cosine_similarity: 0.0,
        };
        assert!(ensure_metrics_finite("invalid", &metrics).is_err());
    }

    #[test]
    fn pcm16_quantization_matches_production_pipeline() {
        let samples = [-1.0, -0.5, 0.0, 0.5, 1.0];
        assert_eq!(
            samples.map(f32_to_pcm16),
            [i16::MIN, -16_384, 0, 16_384, i16::MAX]
        );
        assert_eq!(f32_to_pcm16(-2.0), i16::MIN);
        assert_eq!(f32_to_pcm16(2.0), i16::MAX);
    }
}
