//! Fail-closed WGPU residency and phase-batch measurement harness.
//!
//! This binary intentionally composes the production engine, codec, and
//! `PhaseBatch` API without changing their operations or synchronization.

#![recursion_limit = "512"]

use std::{
    collections::HashSet,
    fs,
    mem::size_of,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use burn::{
    backend::wgpu::{
        AutoCompiler, MemoryConfiguration, RuntimeOptions, WgpuDevice, WgpuRuntime,
        graphics::AutoGraphicsApi, init_setup,
    },
    tensor::{Bool, Device, Int, Tensor, TensorData},
};
use clap::{Parser, ValueEnum};
use cubecl::prelude::Runtime;
use irodori_tts_burn::{
    BatchAudio, BatchItemId, CfgGuidanceMode, GuidanceConfig, InferenceBuilder, IrodoriError,
    OutputGeometry, PhaseBatch, PlannedSynthesis, SamplerMethod, SamplerParams, SamplerWorkReport,
    SamplingRequest, SpeakerKey, VoiceIdentity, WgslWeightProfile,
    codec::{
        CapturedCodecOutput, CapturedDacVaeDecoder, DacVaeCodec, DacVaeDecoder,
        Fixed112DacVaeDecoder,
    },
    load_codec, load_decoder, unpatchify_latent,
};
use safetensors::{Dtype, SafeTensors};
use serde::Serialize;
use sha2::{Digest, Sha256};

const MODEL_SHA256: &str = "5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593";
const CODEC_SHA256: &str = "4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1";
const DECODER_ONLY_CODEC_SHA256: &str =
    "1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231";
type WgpuRt = WgpuRuntime<AutoCompiler>;

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    AllResident,
    PhaseBatch,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum StartupWarmup {
    None,
    DryRun,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum LoadStrategy {
    Sequential,
    Parallel,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum SpeakerMode {
    Same,
    Alternating,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum LengthMode {
    Same,
    Mixed,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum AllocatorMode {
    SubSlices,
    ExclusivePages,
}

impl AllocatorMode {
    const fn configuration(self) -> MemoryConfiguration {
        match self {
            Self::SubSlices => MemoryConfiguration::SubSlices,
            Self::ExclusivePages => MemoryConfiguration::ExclusivePages,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum CodecResidency {
    Full,
    DecodeOnly,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum CodecExecution {
    Eager,
    CapturedGraph,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum DurationResidency {
    Predictive,
    ExactOnly,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum RfWeightResidency {
    PortableFallback,
    Fixed112OneLayout,
    Fixed112PackedOnly,
}

impl RfWeightResidency {
    const fn requires_fixed_112(self) -> bool {
        !matches!(self, Self::PortableFallback)
    }
}

impl From<RfWeightResidency> for WgslWeightProfile {
    fn from(value: RfWeightResidency) -> Self {
        match value {
            RfWeightResidency::PortableFallback => Self::PortableFallback,
            RfWeightResidency::Fixed112OneLayout => Self::Fixed112OneLayout,
            RfWeightResidency::Fixed112PackedOnly => Self::Fixed112PackedOnly,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum CodecWeightResidency {
    PortableFallback,
    Fixed112PackedOnly,
}

impl CodecWeightResidency {
    const fn requires_fixed_112(self) -> bool {
        matches!(self, Self::Fixed112PackedOnly)
    }
}

enum ResidentDecoder {
    Full(Box<DacVaeCodec>),
    DecodeOnly(Box<DacVaeDecoder>),
    Fixed112(Box<Fixed112DacVaeDecoder>),
    Captured(Box<CapturedDacVaeDecoder>),
}

impl ResidentDecoder {
    fn prepare_for_wgsl(&mut self) {
        match self {
            Self::Full(codec) => codec.prepare_decoder_for_wgsl(),
            Self::DecodeOnly(codec) => codec.prepare_for_wgsl(),
            Self::Fixed112(_) => {}
            Self::Captured(_) => {}
        }
    }

    fn decode_wgsl(&self, latent: Tensor<3>) -> Result<Tensor<3>> {
        match self {
            Self::Full(codec) => Ok(codec.decode_wgsl(latent)),
            Self::DecodeOnly(codec) => Ok(codec.decode_wgsl(latent)),
            Self::Fixed112(codec) => codec.decode_wgsl(latent).map_err(Into::into),
            Self::Captured(_) => anyhow::bail!(
                "captured codec must use enqueue_captured to preserve timing boundaries"
            ),
        }
    }

    fn into_captured(
        self,
        input_geometries: impl IntoIterator<Item = [usize; 3]>,
        device: &Device,
    ) -> Result<Self> {
        match self {
            Self::DecodeOnly(codec) => Ok(Self::Captured(Box::new(
                (*codec).into_captured_decode_wgsl(input_geometries, device)?,
            ))),
            Self::Captured(codec) => Ok(Self::Captured(codec)),
            Self::Full(_) => anyhow::bail!("captured codec requires decode-only residency"),
            Self::Fixed112(_) => {
                anyhow::bail!("captured codec does not yet support a fixed112 decoder owner")
            }
        }
    }
}

fn load_resident_decoder(
    path: &Path,
    residency: CodecResidency,
    device: &Device,
) -> Result<ResidentDecoder> {
    match residency {
        CodecResidency::Full => Ok(ResidentDecoder::Full(Box::new(load_codec(path, device)?))),
        CodecResidency::DecodeOnly => Ok(ResidentDecoder::DecodeOnly(Box::new(load_decoder(
            path, device,
        )?))),
    }
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, value_enum)]
    mode: Mode,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    codec_weights: PathBuf,
    /// Oracle fixtures. Same-length mode uses the first; mixed mode cycles all.
    #[arg(long, required = true)]
    fixture: Vec<PathBuf>,
    /// Exactly two freshly prepared reference latents.
    #[arg(long, required = true, num_args = 2)]
    reference: Vec<PathBuf>,
    #[arg(long, default_value_t = 1)]
    requests: usize,
    /// Number of leading requests excluded from steady-state summaries.
    #[arg(long, default_value_t = 0)]
    warmups: usize,
    /// Compile/autotune all planned request shapes before ordinary dispatch.
    #[arg(long, value_enum, default_value = "none")]
    startup_warmup: StartupWarmup,
    /// Use the official no-reference sentinel instead of a prepared speaker.
    #[arg(long)]
    unconditioned: bool,
    #[arg(long, value_enum, default_value = "same")]
    speaker_mode: SpeakerMode,
    #[arg(long, value_enum, default_value = "same")]
    length_mode: LengthMode,
    #[arg(long)]
    output_json: PathBuf,
    #[arg(long, default_value_t = 0)]
    adapter_index: usize,
    #[arg(long, value_enum, default_value = "exclusive-pages")]
    allocator: AllocatorMode,
    #[arg(long, value_enum, default_value = "decode-only")]
    codec_residency: CodecResidency,
    /// Execute codec operators eagerly or replay a process-local fixed-shape graph.
    #[arg(long, value_enum, default_value = "eager")]
    codec_execution: CodecExecution,
    /// Load RF and codec checkpoints sequentially or overlap their I/O/uploads.
    #[arg(long, value_enum, default_value = "sequential")]
    load_strategy: LoadStrategy,
    /// Keep learned duration prediction resident or require exact frame counts.
    #[arg(long, value_enum, default_value = "predictive")]
    duration_residency: DurationResidency,
    /// Keep portable RF fallback weights, drop only the unused QKV layout, or
    /// release all source projections unused by the fixed 112-frame route.
    #[arg(long, value_enum, default_value = "portable-fallback")]
    rf_weight_residency: RfWeightResidency,
    /// Keep the codec source upsampler or release it after preparing the exact
    /// 112-frame polyphase route.
    #[arg(long, value_enum, default_value = "portable-fallback")]
    codec_weight_residency: CodecWeightResidency,
    /// Release completely unused allocator pages after the warmup boundary.
    #[arg(long)]
    cleanup_after_warmup: bool,
    /// Add synchronized allocator snapshots around the first measured request.
    /// This diagnostic mode is excluded from latency comparisons.
    #[arg(long)]
    trace_memory: bool,
    /// Persistent CubeCL cache root, uniquely namespaced for this adapter.
    #[arg(long, value_name = "DIR")]
    cubecl_cache_dir: Option<PathBuf>,
    /// Import a previously exported CubeCL environment before WGPU initialization.
    #[arg(long, value_name = "PATH", requires = "cubecl_cache_dir")]
    cubecl_bundle_in: Option<PathBuf>,
    /// Export the active CubeCL environment after the run; the path must be new.
    #[arg(long, value_name = "PATH", requires = "cubecl_cache_dir")]
    cubecl_bundle_out: Option<PathBuf>,
}

#[derive(Clone)]
struct Fixture {
    path: PathBuf,
    frames: usize,
    text_ids: Vec<i32>,
    text_mask: Vec<bool>,
    caption_ids: Vec<i32>,
    caption_mask: Vec<bool>,
    noise: Vec<f32>,
}

#[derive(Clone)]
struct Reference {
    path: PathBuf,
    frames: usize,
    values: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct MemorySnapshot {
    stage: String,
    number_allocs: u64,
    bytes_in_use: u64,
    bytes_reserved: u64,
}

#[derive(Debug, Serialize)]
struct ItemResult {
    id: String,
    speaker: String,
    frames: usize,
    samples: usize,
    audio_f32_sha256: String,
}

#[derive(Debug, Serialize)]
struct PhaseTiming {
    rf_phase_wall_seconds: f64,
    codec_phase_wall_seconds: f64,
    rf_item_device_complete_seconds: Vec<f64>,
    codec_item_device_complete_seconds: Vec<f64>,
    codec_item_consumer_complete_seconds: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct LoadTiming {
    rf_checkpoint_seconds: f64,
    rf_profile_preparation_seconds: f64,
    request_preparation_seconds: f64,
    codec_checkpoint_seconds: Option<f64>,
    codec_kernel_preparation_seconds: Option<f64>,
    codec_profile_lock_seconds: Option<f64>,
    codec_graph_capture_seconds: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ResidentRequestTiming {
    request: usize,
    warmup: bool,
    rf_device_complete_seconds: f64,
    codec_device_complete_seconds: f64,
    consumer_complete_seconds: f64,
    audio_f32_sha256: String,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    mode: Mode,
    speaker_mode: SpeakerMode,
    length_mode: LengthMode,
    requests: usize,
    adapter_index: usize,
    allocator: AllocatorMode,
    codec_residency: CodecResidency,
    codec_execution: CodecExecution,
    load_strategy: LoadStrategy,
    duration_residency: DurationResidency,
    rf_weight_residency: RfWeightResidency,
    codec_weight_residency: CodecWeightResidency,
    cleanup_after_warmup: bool,
    trace_memory: bool,
    cubecl_cache_dir: Option<PathBuf>,
    cubecl_cache_receipt: Option<irodori_tts_burn::backend_config::CubeClCacheReceipt>,
    cubecl_bundle_import: Option<irodori_tts_burn::backend_config::CubeClBundleImportReceipt>,
    cubecl_bundle_in: Option<PathBuf>,
    cubecl_bundle_out: Option<PathBuf>,
    cubecl_bundle_out_sha256: Option<String>,
    model_sha256: String,
    codec_sha256: String,
    fixture_sha256: Vec<String>,
    reference_sha256: Vec<String>,
    strict_fp32: bool,
    autocast: bool,
    tf32: bool,
    euler_evaluations: usize,
    forward_batches: [usize; 4],
    effective_rows: usize,
    layers: usize,
    block_calls: usize,
    warmups: usize,
    startup_warmup: StartupWarmup,
    startup_dry_run_seconds: Option<f64>,
    measured: usize,
    unconditioned: bool,
    load_wall_seconds: f64,
    codec_load_wall_seconds: Option<f64>,
    load_timing: LoadTiming,
    /// Wall time for every dispatched request, including request warmups.
    execution_wall_seconds: f64,
    /// Sum of consumer-complete latency for measured requests only.
    measured_execution_wall_seconds: f64,
    total_wall_seconds: f64,
    output_seconds: f64,
    requests_per_second: f64,
    end_to_end_requests_per_second: f64,
    audio_seconds_per_wall_second: f64,
    end_to_end_audio_seconds_per_wall_second: f64,
    items: Vec<ItemResult>,
    memory: Vec<MemorySnapshot>,
    phase_timing: Option<PhaseTiming>,
    resident_request_timings: Vec<ResidentRequestTiming>,
    /// Per-request manifests. Mixed-length campaigns must retain the changing
    /// geometry instead of pretending one representative report describes all
    /// requests.
    work_reports: Vec<SamplerWorkReport>,
    /// Backward-compatible representative for same-length campaigns only.
    work_report: Option<SamplerWorkReport>,
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn verify_sha(path: &Path, expected: &str, label: &str) -> Result<()> {
    let actual = sha256_file(path)?;
    ensure!(
        actual == expected,
        "{label} SHA mismatch: expected {expected}, got {actual}"
    );
    Ok(())
}

fn verify_codec_sha(path: &Path, residency: CodecResidency) -> Result<()> {
    let actual = sha256_file(path)?;
    let valid = match residency {
        CodecResidency::Full => actual == CODEC_SHA256,
        CodecResidency::DecodeOnly => actual == CODEC_SHA256 || actual == DECODER_ONLY_CODEC_SHA256,
    };
    ensure!(
        valid,
        "codec SHA mismatch for {residency:?}: expected {}, got {actual}",
        match residency {
            CodecResidency::Full => CODEC_SHA256,
            CodecResidency::DecodeOnly => "released full or pinned decoder-only codec",
        }
    );
    Ok(())
}

fn view<'a>(
    tensors: &SafeTensors<'a>,
    name: &str,
    dtype: Dtype,
) -> Result<safetensors::tensor::TensorView<'a>> {
    let value = tensors
        .tensor(name)
        .with_context(|| format!("missing tensor {name:?}"))?;
    ensure!(value.dtype() == dtype, "tensor {name:?} has wrong dtype");
    Ok(value)
}

fn read_f32(tensors: &SafeTensors<'_>, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let value = view(tensors, name, Dtype::F32)?;
    let shape = value.shape().to_vec();
    let values = value
        .data()
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32 chunk")))
        .collect();
    Ok((shape, values))
}

fn read_i32(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<i32>> {
    view(tensors, name, Dtype::I64)?
        .data()
        .chunks_exact(size_of::<i64>())
        .map(|bytes| {
            i32::try_from(i64::from_le_bytes(bytes.try_into().expect("i64 chunk")))
                .context("token ID exceeds i32")
        })
        .collect()
}

fn read_bool(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<bool>> {
    view(tensors, name, Dtype::BOOL)?
        .data()
        .iter()
        .map(|value| match value {
            0 => Ok(false),
            1 => Ok(true),
            _ => anyhow::bail!("non-canonical bool in tensor {name:?}"),
        })
        .collect()
}

fn load_fixture(path: &Path) -> Result<Fixture> {
    let bytes = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let (shape, noise) = read_f32(&tensors, "noise/source_fp32")?;
    ensure!(
        shape.len() == 3 && shape[0] == 1 && shape[2] == 32,
        "invalid noise shape in {}",
        path.display()
    );
    Ok(Fixture {
        path: path.to_owned(),
        frames: shape[1],
        text_ids: read_i32(&tensors, "inputs/text_input_ids")?,
        text_mask: read_bool(&tensors, "inputs/text_mask")?,
        caption_ids: read_i32(&tensors, "inputs/caption_input_ids")?,
        caption_mask: read_bool(&tensors, "inputs/caption_mask")?,
        noise,
    })
}

fn load_reference(path: &Path) -> Result<Reference> {
    let bytes = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let (shape, values) = read_f32(&tensors, "latent")?;
    ensure!(
        shape.len() == 3 && shape[0] == 1 && shape[2] == 32,
        "invalid reference shape in {}",
        path.display()
    );
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "non-finite reference values"
    );
    Ok(Reference {
        path: path.to_owned(),
        frames: shape[1],
        values,
    })
}

fn sync(device: &WgpuDevice) -> Result<()> {
    cubecl::future::block_on(WgpuRt::client(device).sync()).context("WGPU sync failed")
}

fn snapshot(device: &WgpuDevice, stage: &str) -> Result<MemorySnapshot> {
    sync(device)?;
    let usage = WgpuRt::client(device)
        .memory_usage()
        .context("WGPU memory query failed")?;
    Ok(MemorySnapshot {
        stage: stage.to_owned(),
        number_allocs: usage.number_allocs,
        bytes_in_use: usage.bytes_in_use,
        bytes_reserved: usage.bytes_reserved,
    })
}

fn make_request(
    fixture: &Fixture,
    reference: Option<&Reference>,
    speaker_patch_size: usize,
    device: &Device,
) -> SamplingRequest {
    let (reference_values, reference_frames, reference_mask) = match reference {
        Some(reference) => (
            reference.values.clone(),
            reference.frames,
            vec![true; reference.frames],
        ),
        None => (
            vec![0.0; speaker_patch_size * 32],
            speaker_patch_size,
            vec![false; speaker_patch_size],
        ),
    };
    SamplingRequest {
        text_ids: Tensor::<2, Int>::from_data(
            TensorData::new(fixture.text_ids.clone(), [1, 256]),
            device,
        ),
        text_mask: Tensor::<2, Bool>::from_data(
            TensorData::new(fixture.text_mask.clone(), [1, 256]),
            device,
        ),
        ref_latent: Some(Tensor::<3>::from_data(
            TensorData::new(reference_values, [1, reference_frames, 32]),
            device,
        )),
        ref_mask: Some(Tensor::<2, Bool>::from_data(
            TensorData::new(reference_mask, [1, reference_frames]),
            device,
        )),
        sequence_length: fixture.frames,
        caption_ids: Some(Tensor::<2, Int>::from_data(
            TensorData::new(fixture.caption_ids.clone(), [1, 512]),
            device,
        )),
        caption_mask: Some(Tensor::<2, Bool>::from_data(
            TensorData::new(fixture.caption_mask.clone(), [1, 512]),
            device,
        )),
        initial_noise: Some(Tensor::<3>::from_data(
            TensorData::new(fixture.noise.clone(), [1, fixture.frames, 32]),
            device,
        )),
    }
}

fn audio_values_result(
    id: BatchItemId,
    voice: VoiceIdentity,
    frames: usize,
    values: Vec<f32>,
) -> Result<ItemResult> {
    let mut digest = Sha256::new();
    for value in &values {
        digest.update(value.to_le_bytes());
    }
    let speaker = match &voice {
        VoiceIdentity::Unconditioned => "unconditioned".to_owned(),
        VoiceIdentity::Clone(key) => key.as_str().to_owned(),
        VoiceIdentity::Designed(key) => key.as_str().to_owned(),
    };
    Ok(ItemResult {
        id: id.as_str().to_owned(),
        speaker,
        frames,
        samples: values.len(),
        audio_f32_sha256: format!("{:x}", digest.finalize()),
    })
}

fn audio_result(audio: BatchAudio, frames: usize) -> Result<ItemResult> {
    let values = audio.tensor.into_data().convert::<f32>().to_vec::<f32>()?;
    audio_values_result(audio.id, audio.voice, frames, values)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cubecl_cache_receipt = args
        .cubecl_cache_dir
        .as_ref()
        .map(irodori_tts_burn::backend_config::configure_cubecl_persistent_cache)
        .transpose()?;
    let cubecl_bundle_import = args
        .cubecl_bundle_in
        .as_ref()
        .map(irodori_tts_burn::backend_config::import_cubecl_environment_bundle)
        .transpose()?;
    ensure!(args.requests > 0, "--requests must be positive");
    if matches!(args.codec_execution, CodecExecution::CapturedGraph) {
        ensure!(
            matches!(args.mode, Mode::AllResident),
            "captured codec execution is only available in all-resident mode"
        );
        ensure!(
            matches!(args.codec_residency, CodecResidency::DecodeOnly),
            "captured codec execution requires decode-only codec residency"
        );
        ensure!(
            matches!(
                args.codec_weight_residency,
                CodecWeightResidency::PortableFallback
            ),
            "captured codec execution currently requires portable codec weights"
        );
    }
    if args.rf_weight_residency.requires_fixed_112()
        || args.codec_weight_residency.requires_fixed_112()
    {
        ensure!(
            args.fixture
                .iter()
                .all(|path| load_fixture(path).is_ok_and(|fixture| fixture.frames == 112)),
            "fixed112 RF/codec residency requires only 112-frame fixtures"
        );
    }
    ensure!(
        args.warmups < args.requests,
        "--warmups must be less than --requests"
    );
    ensure!(
        args.reference.len() == 2,
        "exactly two references are required"
    );
    ensure!(
        matches!(args.mode, Mode::AllResident) || args.warmups == 0,
        "phase-batch mode does not accept warmups"
    );
    ensure!(
        matches!(args.mode, Mode::AllResident)
            || matches!(args.load_strategy, LoadStrategy::Sequential),
        "parallel checkpoint loading requires all-resident mode"
    );
    ensure!(
        matches!(args.mode, Mode::AllResident)
            || matches!(args.startup_warmup, StartupWarmup::None),
        "compile-only startup warmup requires all-resident mode"
    );
    ensure!(
        !args.cleanup_after_warmup || matches!(args.mode, Mode::AllResident) && args.warmups > 0,
        "--cleanup-after-warmup requires all-resident mode and at least one warmup"
    );
    if matches!(args.length_mode, LengthMode::Mixed) {
        ensure!(
            args.fixture.len() >= 2,
            "mixed length mode requires multiple fixtures"
        );
    }
    ensure!(
        !args.output_json.exists(),
        "refusing to overwrite {}",
        args.output_json.display()
    );
    if let Some(path) = args.cubecl_bundle_in.as_ref() {
        ensure!(
            path.is_file(),
            "CubeCL input bundle not found: {}",
            path.display()
        );
    }
    if let Some(path) = args.cubecl_bundle_out.as_ref() {
        ensure!(
            !path.exists(),
            "refusing to overwrite CubeCL output bundle {}",
            path.display()
        );
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
    }
    verify_sha(&args.checkpoint, MODEL_SHA256, "model")?;
    verify_codec_sha(&args.codec_weights, args.codec_residency)?;
    let fixtures = args
        .fixture
        .iter()
        .map(|path| load_fixture(path))
        .collect::<Result<Vec<_>>>()?;
    let references = args
        .reference
        .iter()
        .map(|path| load_reference(path))
        .collect::<Result<Vec<_>>>()?;

    let device = WgpuDevice::DiscreteGpu(args.adapter_index);
    init_setup::<AutoGraphicsApi>(
        &device,
        RuntimeOptions {
            tasks_max: 32,
            memory_config: args.allocator.configuration(),
        },
    );
    let tensor_device = irodori_tts_burn::backend_config::strict_fp32_device(&device)?;
    let mut memory = vec![snapshot(&device, "initialized")?];
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
    let total_started = Instant::now();
    let load_started = Instant::now();
    let builder = InferenceBuilder::<_>::new(tensor_device.clone());
    let (loaded, mut early_codec, rf_checkpoint_seconds, early_codec_checkpoint_seconds) =
        match args.load_strategy {
            LoadStrategy::Sequential => {
                let rf_checkpoint_started = Instant::now();
                let loaded = match args.duration_residency {
                    DurationResidency::Predictive => builder.load_weights(&args.checkpoint)?,
                    DurationResidency::ExactOnly => {
                        builder.load_weights_exact_only(&args.checkpoint)?
                    }
                };
                (
                    loaded,
                    None,
                    rf_checkpoint_started.elapsed().as_secs_f64(),
                    None,
                )
            }
            LoadStrategy::Parallel => std::thread::scope(|scope| -> Result<_> {
                let codec_path = args.codec_weights.clone();
                let codec_residency = args.codec_residency;
                let codec_device = tensor_device.clone();
                let codec_handle = scope.spawn(move || {
                    let started = Instant::now();
                    let codec = load_resident_decoder(&codec_path, codec_residency, &codec_device);
                    (codec, started.elapsed().as_secs_f64())
                });
                let rf_checkpoint_started = Instant::now();
                let loaded = match args.duration_residency {
                    DurationResidency::Predictive => builder.load_weights(&args.checkpoint)?,
                    DurationResidency::ExactOnly => {
                        builder.load_weights_exact_only(&args.checkpoint)?
                    }
                };
                let rf_checkpoint_seconds = rf_checkpoint_started.elapsed().as_secs_f64();
                let (codec, codec_checkpoint_seconds) = codec_handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("parallel codec loader panicked"))?;
                Ok((
                    loaded,
                    Some(codec?),
                    rf_checkpoint_seconds,
                    Some(codec_checkpoint_seconds),
                ))
            })?,
        };
    memory.push(snapshot(&device, "rf_source_resident")?);
    let config = loaded.model_config().clone();
    ensure!(
        config.latent_dim == 32 && config.latent_patch_size == 1,
        "unexpected v4 geometry"
    );
    let rf_profile_preparation_started = Instant::now();
    let engine = loaded
        .with_sampling(params)
        .build_wgsl_with_profile(args.rf_weight_residency.into())?;
    sync(&device)?;
    let rf_profile_preparation_seconds = rf_profile_preparation_started.elapsed().as_secs_f64();
    memory.push(snapshot(&device, "rf_resident")?);

    let request_preparation_started = Instant::now();
    let mut planned = Vec::with_capacity(args.requests);
    let mut item_frames = Vec::with_capacity(args.requests);
    for index in 0..args.requests {
        let fixture_index = match args.length_mode {
            LengthMode::Same => 0,
            LengthMode::Mixed => index % fixtures.len(),
        };
        let reference_index = match args.speaker_mode {
            SpeakerMode::Same => 0,
            SpeakerMode::Alternating => index % references.len(),
        };
        let fixture = &fixtures[fixture_index];
        let reference = (!args.unconditioned).then_some(&references[reference_index]);
        let id = BatchItemId::new(format!("request-{index:02}"))?;
        let voice = if args.unconditioned {
            VoiceIdentity::Unconditioned
        } else {
            VoiceIdentity::Clone(SpeakerKey::new(format!("ref{}", reference_index + 1))?)
        };
        planned.push(PlannedSynthesis::new(
            id,
            voice,
            OutputGeometry::new(fixture.frames, 1, 32)?,
            make_request(
                fixture,
                reference,
                config.speaker_patch_size.unwrap_or(1).max(1),
                &tensor_device,
            ),
        )?);
        item_frames.push(fixture.frames);
    }
    let request_preparation_seconds = request_preparation_started.elapsed().as_secs_f64();

    let mut items = Vec::with_capacity(args.requests);
    let mut phase_timing = None;
    let mut resident_request_timings = Vec::new();
    let mut work_reports = Vec::new();
    let mut work_report = None;
    let mut codec_load_wall_seconds = None;
    let mut codec_checkpoint_seconds = early_codec_checkpoint_seconds;
    let mut codec_kernel_preparation_seconds = None;
    let mut codec_profile_lock_seconds = None;
    let mut codec_graph_capture_seconds = None;
    let mut startup_dry_run_seconds = None;
    let load_wall_seconds;
    let execution_started;
    match args.mode {
        Mode::AllResident => {
            let mut codec = if let Some(codec) = early_codec.take() {
                codec
            } else {
                let codec_checkpoint_started = Instant::now();
                let codec = load_resident_decoder(
                    &args.codec_weights,
                    args.codec_residency,
                    &tensor_device,
                )?;
                codec_checkpoint_seconds = Some(codec_checkpoint_started.elapsed().as_secs_f64());
                codec
            };
            memory.push(snapshot(&device, "rf_duration_codec_source_resident")?);
            let codec_kernel_preparation_started = Instant::now();
            codec.prepare_for_wgsl();
            codec_kernel_preparation_seconds =
                Some(codec_kernel_preparation_started.elapsed().as_secs_f64());
            if args.codec_weight_residency.requires_fixed_112() {
                let codec_profile_lock_started = Instant::now();
                codec = match codec {
                    ResidentDecoder::DecodeOnly(codec) => {
                        ResidentDecoder::Fixed112(Box::new((*codec).into_fixed_112_for_wgsl()?))
                    }
                    ResidentDecoder::Full(_) => anyhow::bail!(
                        "fixed112 weight residency requires decode-only codec residency"
                    ),
                    ResidentDecoder::Fixed112(codec) => ResidentDecoder::Fixed112(codec),
                    ResidentDecoder::Captured(_) => anyhow::bail!(
                        "codec graph capture must occur after fixed112 profile locking"
                    ),
                };
                codec_profile_lock_seconds =
                    Some(codec_profile_lock_started.elapsed().as_secs_f64());
            }
            sync(&device)?;
            if matches!(args.startup_warmup, StartupWarmup::DryRun) {
                let prepared = planned
                    .iter()
                    .map(|item| engine.prepare_sampling_request(item.request.clone()))
                    .collect::<irodori_tts_burn::Result<Vec<_>>>()?;
                let started = Instant::now();
                {
                    let _dry_run = cubecl::dry_run::DryRun::new();
                    for request in prepared {
                        {
                            let _patched = engine.sample_prepared(request)?;
                        }
                        tensor_device.memory_cleanup();
                    }
                    let mut codec_geometries = HashSet::with_capacity(item_frames.len());
                    for &frames in &item_frames {
                        if !codec_geometries.insert((1, frames, 32)) {
                            continue;
                        }
                        {
                            let latent = Tensor::<3>::zeros([1, frames, 32], &tensor_device);
                            let _audio = codec.decode_wgsl(latent)?;
                        }
                        tensor_device.memory_cleanup();
                    }
                }
                sync(&device)?;
                startup_dry_run_seconds = Some(started.elapsed().as_secs_f64());
                memory.push(snapshot(&device, "all_resident_after_startup_dry_run")?);
            }
            if matches!(args.codec_execution, CodecExecution::CapturedGraph) {
                memory.push(snapshot(
                    &device,
                    "all_resident_before_codec_graph_capture",
                )?);
                let mut geometries = HashSet::with_capacity(item_frames.len());
                geometries.extend(item_frames.iter().map(|&frames| [1, frames, 32]));
                let started = Instant::now();
                codec = codec.into_captured(geometries, &tensor_device)?;
                sync(&device)?;
                codec_graph_capture_seconds = Some(started.elapsed().as_secs_f64());
                memory.push(snapshot(&device, "all_resident_after_codec_graph_capture")?);
            }
            load_wall_seconds = load_started.elapsed().as_secs_f64();
            memory.push(snapshot(&device, "rf_duration_codec_resident")?);
            execution_started = Instant::now();
            for (index, one) in planned.into_iter().enumerate() {
                sync(&device)?;
                let request_started = Instant::now();
                let (patched, report) = engine.sample_with_work_report(one.request)?;
                sync(&device)?;
                ensure!(
                    report.num_steps == 4
                        && report.schedule_f32_bits
                            == [1065336439, 1061146329, 1056947831, 1048559223, 0]
                        && report.whole_model_forwards == 4
                        && report.model_layers == 12
                        && report.model_block_calls == 48
                        && report
                            .forwards
                            .iter()
                            .map(|forward| forward.batch_rows)
                            .eq([2, 2, 1, 1]),
                    "all-resident RF work manifest mismatch: {report:?}"
                );
                if matches!(args.length_mode, LengthMode::Same) {
                    if let Some(first) = &work_report {
                        ensure!(
                            first == &report,
                            "same-length RF work manifest changed between requests"
                        );
                    } else {
                        work_report = Some(report.clone());
                    }
                }
                work_reports.push(report);
                let rf_device_complete_seconds = request_started.elapsed().as_secs_f64();
                if args.trace_memory && index == args.warmups {
                    memory.push(snapshot(&device, "trace_after_rf_device_complete")?);
                }
                if index == 0 {
                    memory.push(snapshot(&device, "all_resident_after_first_rf")?);
                }
                let codec_started = Instant::now();
                let latent = unpatchify_latent(patched, 1, 32);
                let (item, codec_device_complete_seconds) = match &mut codec {
                    ResidentDecoder::Captured(codec) => {
                        let output: CapturedCodecOutput<'_> = codec.enqueue(latent)?;
                        sync(&device)?;
                        let device_complete = codec_started.elapsed().as_secs_f64();
                        if args.trace_memory && index == args.warmups {
                            memory.push(snapshot(&device, "trace_after_codec_device_complete")?);
                        }
                        (
                            audio_values_result(
                                one.id,
                                one.voice,
                                item_frames[index],
                                output.to_cpu_f32()?,
                            )?,
                            device_complete,
                        )
                    }
                    eager => {
                        let decoded = eager.decode_wgsl(latent)?;
                        sync(&device)?;
                        let device_complete = codec_started.elapsed().as_secs_f64();
                        if args.trace_memory && index == args.warmups {
                            memory.push(snapshot(&device, "trace_after_codec_device_complete")?);
                        }
                        (
                            audio_result(
                                BatchAudio {
                                    id: one.id,
                                    voice: one.voice,
                                    tensor: decoded,
                                },
                                item_frames[index],
                            )?,
                            device_complete,
                        )
                    }
                };
                sync(&device)?;
                if args.trace_memory && index == args.warmups {
                    memory.push(snapshot(&device, "trace_after_consumer_complete")?);
                }
                let consumer_complete_seconds = request_started.elapsed().as_secs_f64();
                resident_request_timings.push(ResidentRequestTiming {
                    request: index + 1,
                    warmup: index < args.warmups,
                    rf_device_complete_seconds,
                    codec_device_complete_seconds,
                    consumer_complete_seconds,
                    audio_f32_sha256: item.audio_f32_sha256.clone(),
                });
                items.push(item);
                if args.cleanup_after_warmup && index + 1 == args.warmups {
                    tensor_device.memory_cleanup();
                    memory.push(snapshot(&device, "all_resident_after_warmup_cleanup")?);
                }
            }
            memory.push(snapshot(&device, "all_resident_after_consumer")?);
            drop(codec);
            drop(engine);
        }
        Mode::PhaseBatch => {
            load_wall_seconds = load_started.elapsed().as_secs_f64();
            execution_started = Instant::now();
            let latents = PhaseBatch::new(engine, planned)?.sample_all()?;
            memory.push(snapshot(&device, "latents_resident_rf_released")?);
            let codec_load_started = Instant::now();
            let codec = match args.codec_residency {
                CodecResidency::Full => {
                    let codec = load_codec(&args.codec_weights, &tensor_device)?;
                    sync(&device)?;
                    codec_load_wall_seconds = Some(codec_load_started.elapsed().as_secs_f64());
                    latents.with_codec(codec)
                }
                CodecResidency::DecodeOnly => {
                    let decoder = load_decoder(&args.codec_weights, &tensor_device)?;
                    sync(&device)?;
                    codec_load_wall_seconds = Some(codec_load_started.elapsed().as_secs_f64());
                    latents.with_decoder(decoder)
                }
            };
            memory.push(snapshot(&device, "latents_codec_resident")?);
            let mut consumed = 0_usize;
            let complete = codec.decode_all(|audio| {
                let frames = item_frames[consumed];
                items.push(audio_result(audio, frames).map_err(|error| {
                    IrodoriError::Config(format!("phase-batch consumer failed: {error:#}"))
                })?);
                consumed += 1;
                Ok(())
            })?;
            let metrics = complete.into_metrics();
            phase_timing = Some(PhaseTiming {
                rf_phase_wall_seconds: metrics.rf_phase_wall.as_secs_f64(),
                codec_phase_wall_seconds: metrics.codec_phase_wall.as_secs_f64(),
                rf_item_device_complete_seconds: metrics
                    .rf_items
                    .iter()
                    .map(|item| item.device_complete.as_secs_f64())
                    .collect(),
                codec_item_device_complete_seconds: metrics
                    .codec_items
                    .iter()
                    .map(|item| item.device_complete.as_secs_f64())
                    .collect(),
                codec_item_consumer_complete_seconds: metrics
                    .codec_items
                    .iter()
                    .map(|item| item.consumer_complete.as_secs_f64())
                    .collect(),
            });
            memory.push(snapshot(&device, "complete")?);
        }
    }
    let execution_wall_seconds = execution_started.elapsed().as_secs_f64();
    let total_wall_seconds = total_started.elapsed().as_secs_f64();
    let measured = args.requests - args.warmups;
    let measured_execution_wall_seconds = if resident_request_timings.is_empty() {
        execution_wall_seconds
    } else {
        resident_request_timings
            .iter()
            .filter(|timing| !timing.warmup)
            .map(|timing| timing.consumer_complete_seconds)
            .sum()
    };
    let output_seconds = items
        .iter()
        .skip(args.warmups)
        .map(|item| item.samples as f64 / 48_000.0)
        .sum::<f64>();
    let cubecl_bundle_out_sha256 = if let Some(path) = args.cubecl_bundle_out.as_ref() {
        irodori_tts_burn::backend_config::export_cubecl_environment_bundle(path)?;
        Some(sha256_file(path)?)
    } else {
        None
    };
    let report = Report {
        schema_version: 5,
        mode: args.mode,
        speaker_mode: args.speaker_mode,
        length_mode: args.length_mode,
        requests: args.requests,
        adapter_index: args.adapter_index,
        allocator: args.allocator,
        codec_residency: args.codec_residency,
        codec_execution: args.codec_execution,
        load_strategy: args.load_strategy,
        duration_residency: args.duration_residency,
        rf_weight_residency: args.rf_weight_residency,
        codec_weight_residency: args.codec_weight_residency,
        cleanup_after_warmup: args.cleanup_after_warmup,
        trace_memory: args.trace_memory,
        cubecl_cache_dir: args.cubecl_cache_dir.clone(),
        cubecl_cache_receipt,
        cubecl_bundle_import,
        cubecl_bundle_in: args.cubecl_bundle_in.clone(),
        cubecl_bundle_out: args.cubecl_bundle_out.clone(),
        cubecl_bundle_out_sha256,
        model_sha256: sha256_file(&args.checkpoint)?,
        codec_sha256: sha256_file(&args.codec_weights)?,
        fixture_sha256: fixtures
            .iter()
            .map(|fixture| sha256_file(&fixture.path))
            .collect::<Result<_>>()?,
        reference_sha256: references
            .iter()
            .map(|reference| sha256_file(&reference.path))
            .collect::<Result<_>>()?,
        strict_fp32: true,
        autocast: false,
        tf32: false,
        euler_evaluations: 4,
        forward_batches: [2, 2, 1, 1],
        effective_rows: 6,
        layers: 12,
        block_calls: 48,
        warmups: args.warmups,
        startup_warmup: args.startup_warmup,
        startup_dry_run_seconds,
        measured,
        unconditioned: args.unconditioned,
        load_wall_seconds,
        codec_load_wall_seconds,
        load_timing: LoadTiming {
            rf_checkpoint_seconds,
            rf_profile_preparation_seconds,
            request_preparation_seconds,
            codec_checkpoint_seconds,
            codec_kernel_preparation_seconds,
            codec_profile_lock_seconds,
            codec_graph_capture_seconds,
        },
        execution_wall_seconds,
        measured_execution_wall_seconds,
        total_wall_seconds,
        output_seconds,
        requests_per_second: measured as f64 / measured_execution_wall_seconds,
        end_to_end_requests_per_second: measured as f64 / total_wall_seconds,
        audio_seconds_per_wall_second: output_seconds / measured_execution_wall_seconds,
        end_to_end_audio_seconds_per_wall_second: output_seconds / total_wall_seconds,
        items,
        memory,
        phase_timing,
        resident_request_timings,
        work_reports,
        work_report,
    };
    if let Some(parent) = args.output_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.output_json, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
