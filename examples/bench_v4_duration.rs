//! Measure the released v4 duration predictor through the production WGSL engine.
//!
//! Two scopes are alternated in one process:
//! - `head`: duration head from an already encoded condition;
//! - `full`: condition encoding followed by the duration head.
//!
//! Both record the wall interval from a pre-synchronized device through GPU
//! completion, then a secondary interval through an owned contiguous f32 scalar
//! readback. The first five observations per scope are warmups so lazy WGPU
//! pipeline compilation cannot leak into the ten measured observations.

#![recursion_limit = "512"]

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, Write},
    mem::size_of,
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
use clap::Parser;
use cubecl::prelude::Runtime;
use irodori_tts_wgpu::{
    AuxConditionInput, EncodedCondition, InferenceBuilder, WgpuRaw, WgslInferenceEngine,
};
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

const FORMAT: &str = "irodori-v4-wgpu-duration-benchmark-v1";
const PYTHON_FORMAT: &str = "irodori-v4-python-duration-benchmark-v1";
const ORACLE_FORMAT: &str = "irodori-v4-precision-oracle-v1";
const DURATION_FIXTURE_FORMAT: &str = "irodori-v4-duration-fixture-v1";
const MODEL_SHA256: &str = "5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593";
const TEXT_TOKENS: usize = 256;
const CAPTION_TOKENS: usize = 512;
const LATENT_DIM: usize = 32;
const DURATION_FEATURES: usize = 14;
const TASKS_MAX: usize = 32;
const WARMUPS: usize = 5;
const MEASURED: usize = 10;
const REPEATS: usize = WARMUPS + MEASURED;
const PYTHON_ABS_TOLERANCE: f32 = 1.0e-4;
const SAMPLE_RATE: usize = 48_000;
const HOP_LENGTH: usize = 1_920;
const LATENT_PATCH_SIZE: usize = 1;
const MIN_SECONDS: f64 = 0.5;
const MAX_SECONDS: f64 = 30.0;

type Backend = WgpuRaw;

#[derive(Debug, Parser)]
#[command(about = "Benchmark the production WGSL v4 duration predictor")]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long, default_value = MODEL_SHA256)]
    checkpoint_sha256: String,
    #[arg(long)]
    fixture: PathBuf,
    #[arg(long)]
    fixture_sha256: String,
    #[arg(long)]
    python_json: PathBuf,
    #[arg(long)]
    python_json_sha256: String,
    #[arg(long, default_value_t = 0)]
    adapter_index: usize,
    #[arg(long, default_value_t = REPEATS)]
    repeats: usize,
    #[arg(long)]
    json_out: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Scope {
    Head,
    Full,
}

impl Scope {
    const fn label(self) -> &'static str {
        match self {
            Self::Head => "head",
            Self::Full => "full",
        }
    }
}

#[derive(Debug, Serialize)]
struct Timing {
    device_complete_seconds: f64,
    readback_complete_seconds: f64,
    readback_elements: usize,
    readback_dtype: &'static str,
    readback_owned: bool,
    readback_contiguous: bool,
}

#[derive(Debug, Serialize)]
struct ScopeResult {
    repeat: usize,
    cold: bool,
    scope: Scope,
    timing: Timing,
    log_frames: f32,
    predicted_frames: f32,
    output_sha256: String,
}

#[derive(Debug, Serialize)]
struct ScopeSummary {
    warmups: usize,
    measured: usize,
    device_complete_min_seconds: f64,
    device_complete_median_seconds: f64,
    device_complete_max_seconds: f64,
    readback_complete_min_seconds: f64,
    readback_complete_median_seconds: f64,
    readback_complete_max_seconds: f64,
    output_hashes_equal: bool,
    log_frames: f32,
    predicted_frames: f32,
}

#[derive(Debug, Serialize)]
struct AdapterRecord {
    index: usize,
    name: String,
    backend: String,
    device_type: String,
    tasks_max: usize,
    memory_config: &'static str,
}

#[derive(Debug, Serialize)]
struct Payload {
    format: &'static str,
    pins: BTreeMap<&'static str, String>,
    adapter: AdapterRecord,
    input: InputRecord,
    timer_contract: TimerContract,
    resolved_length: ResolvedLengthRecord,
    python_reference: PythonReferenceRecord,
    scopes: BTreeMap<&'static str, ScopeSummary>,
    repeats: Vec<ScopeResult>,
    wgpu_uncaptured_errors: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
struct ResolvedLengthRecord {
    duration_scale: f64,
    min_seconds: f64,
    max_seconds: f64,
    sample_rate: usize,
    hop_length: usize,
    latent_patch_size: usize,
    latent_frames: usize,
    patched_frames: usize,
    target_samples: usize,
    seconds: f64,
}

#[derive(Debug, Serialize)]
struct InputRecord {
    text: String,
    text_shape: [usize; 2],
    text_valid_tokens: usize,
    requested_speaker_shape: [usize; 3],
    caption_shape: [usize; 2],
    requested_joint_condition_tokens: usize,
    encoded_text_shape: [usize; 3],
    encoded_aux_removed: bool,
    duration_features_f32: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct TimerContract {
    primary: &'static str,
    secondary: &'static str,
    pre_start_sync: bool,
    cpu_readback_in_primary: bool,
    cpu_readback_elements: usize,
}

#[derive(Debug, Serialize)]
struct PythonReferenceRecord {
    format: String,
    head_log_frames: f32,
    full_log_frames: f32,
    rust_log_frames: f32,
    max_abs_error: f32,
    tolerance: f32,
    passed: bool,
}

#[derive(Debug)]
struct Fixture {
    text_ids: Vec<i32>,
    text_mask: Vec<bool>,
}

struct DeviceInputs {
    text_ids: Tensor<Backend, 2, Int>,
    text_mask: Tensor<Backend, 2, Bool>,
    duration_features: Tensor<Backend, 2>,
    has_speaker: Tensor<Backend, 1, Bool>,
    has_caption: Tensor<Backend, 1, Bool>,
    text_valid_tokens: usize,
}

impl DeviceInputs {
    fn compact_no_aux_condition(
        &self,
    ) -> (
        Tensor<Backend, 2, Int>,
        Tensor<Backend, 2, Bool>,
        AuxConditionInput<Backend>,
    ) {
        (
            self.text_ids.clone().narrow(1, 0, self.text_valid_tokens),
            self.text_mask.clone().narrow(1, 0, self.text_valid_tokens),
            AuxConditionInput::None,
        )
    }
}

#[derive(Clone)]
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

    fn count(&self) -> Result<usize> {
        Ok(self
            .errors
            .lock()
            .map_err(|_| anyhow::anyhow!("WGPU error monitor lock poisoned"))?
            .len())
    }

    fn check(&self, stage: &str) -> Result<()> {
        let errors = self
            .errors
            .lock()
            .map_err(|_| anyhow::anyhow!("WGPU error monitor lock poisoned"))?;
        ensure!(errors.is_empty(), "WGPU errors after {stage}: {errors:?}");
        Ok(())
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(sha256_bytes(&bytes))
}

fn normalize_sha(value: &str, label: &str) -> Result<String> {
    let normalized = value.trim().to_ascii_lowercase();
    ensure!(
        normalized.len() == 64 && normalized.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{label} must be a 64-character SHA-256"
    );
    Ok(normalized)
}

fn verify_file(path: &Path, expected: &str, label: &str) -> Result<String> {
    ensure!(
        path.is_file(),
        "{label} is not a regular file: {}",
        path.display()
    );
    let expected = normalize_sha(expected, label)?;
    let actual = sha256_file(path)?;
    ensure!(
        actual == expected,
        "{label} SHA mismatch: {actual} != {expected}"
    );
    Ok(actual)
}

fn checked_view<'a>(
    tensors: &SafeTensors<'a>,
    key: &str,
    dtype: Dtype,
    shape: &[usize],
) -> Result<safetensors::tensor::TensorView<'a>> {
    let view = tensors
        .tensor(key)
        .with_context(|| format!("fixture tensor {key:?} is missing"))?;
    ensure!(
        view.dtype() == dtype,
        "fixture tensor {key:?} dtype mismatch"
    );
    ensure!(
        view.shape() == shape,
        "fixture tensor {key:?} shape mismatch"
    );
    Ok(view)
}

fn read_i64_as_i32(tensors: &SafeTensors<'_>, key: &str, shape: &[usize]) -> Result<Vec<i32>> {
    checked_view(tensors, key, Dtype::I64, shape)?
        .data()
        .chunks_exact(size_of::<i64>())
        .map(|chunk| {
            let bytes: [u8; size_of::<i64>()] = chunk.try_into().expect("exact i64 chunk");
            i32::try_from(i64::from_le_bytes(bytes))
                .with_context(|| format!("token ID in {key:?} exceeds i32"))
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
            other => anyhow::bail!("invalid bool byte {other} in {key:?}"),
        })
        .collect()
}

fn load_fixture(path: &Path) -> Result<Fixture> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let (_, header) = SafeTensors::read_metadata(&bytes).context("invalid fixture metadata")?;
    let metadata = header
        .metadata()
        .as_ref()
        .context("fixture metadata is missing")?;
    let fixture_metadata: Value = if let Some(value) = metadata.get("duration_json") {
        serde_json::from_str(value).context("fixture duration_json is invalid")?
    } else {
        serde_json::from_str(
            metadata
                .get("oracle_json")
                .context("fixture metadata lacks duration_json/oracle_json")?,
        )
        .context("fixture oracle_json is invalid")?
    };
    ensure!(
        fixture_metadata["format"] == DURATION_FIXTURE_FORMAT
            || fixture_metadata["format"] == ORACLE_FORMAT,
        "fixture format mismatch"
    );
    let tensors = SafeTensors::deserialize(&bytes).context("invalid fixture tensors")?;
    let caption_ids = read_i64_as_i32(&tensors, "inputs/caption_input_ids", &[1, CAPTION_TOKENS])?;
    let caption_mask = read_bool(&tensors, "inputs/caption_mask", &[1, CAPTION_TOKENS])?;
    let fixture = Fixture {
        text_ids: read_i64_as_i32(&tensors, "inputs/text_input_ids", &[1, TEXT_TOKENS])?,
        text_mask: read_bool(&tensors, "inputs/text_mask", &[1, TEXT_TOKENS])?,
    };
    ensure!(
        fixture.text_mask.iter().any(|&value| value),
        "text mask is empty"
    );
    ensure!(
        caption_ids.len() == CAPTION_TOKENS && caption_mask.iter().all(|&value| !value),
        "canonical duration fixture must have no caption"
    );
    Ok(fixture)
}

#[derive(Debug, Deserialize)]
struct PythonScope {
    log_frames: f32,
    output_hashes_equal: bool,
    measured: usize,
    warmups: usize,
}

#[derive(Debug)]
struct PythonReference {
    format: String,
    text: String,
    text_valid_tokens: usize,
    features: Vec<f32>,
    resolved_length: ResolvedLengthRecord,
    head: PythonScope,
    full: PythonScope,
}

fn load_python_reference(path: &Path, fixture_sha: &str) -> Result<PythonReference> {
    let value: Value = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(
        value["format"] == PYTHON_FORMAT,
        "Python duration format mismatch"
    );
    ensure!(
        value["pins"]["model_sha256"] == MODEL_SHA256,
        "Python model pin mismatch"
    );
    ensure!(
        value["pins"]["fixture_sha256"] == fixture_sha,
        "Python fixture pin mismatch"
    );
    let features: Vec<f32> =
        serde_json::from_value(value["input"]["duration_features_f32"].clone())?;
    ensure!(
        features.len() == DURATION_FEATURES,
        "duration feature length mismatch"
    );
    ensure!(
        features.iter().all(|value| value.is_finite()),
        "non-finite duration feature"
    );
    let head: PythonScope = serde_json::from_value(value["scopes"]["head"].clone())?;
    let full: PythonScope = serde_json::from_value(value["scopes"]["full"].clone())?;
    let resolved_length: ResolvedLengthRecord =
        serde_json::from_value(value["resolved_length"].clone())
            .context("Python resolved duration record is missing or invalid")?;
    for (label, scope) in [("head", &head), ("full", &full)] {
        ensure!(
            scope.warmups == WARMUPS && scope.measured == REPEATS - WARMUPS,
            "Python {label} repeat contract mismatch"
        );
        ensure!(
            scope.output_hashes_equal,
            "Python {label} output is nondeterministic"
        );
        ensure!(
            scope.log_frames.is_finite(),
            "Python {label} output is non-finite"
        );
    }
    ensure!(
        head.log_frames.to_bits() == full.log_frames.to_bits(),
        "Python head/full duration outputs differ"
    );
    Ok(PythonReference {
        format: value["format"].as_str().unwrap_or_default().to_string(),
        text: value["input"]["text"]
            .as_str()
            .context("Python input text is missing")?
            .to_string(),
        text_valid_tokens: value["input"]["text_valid_tokens"]
            .as_u64()
            .context("Python valid-token count is missing")? as usize,
        features,
        resolved_length,
        head,
        full,
    })
}

fn initialize_wgpu(adapter_index: usize) -> (WgpuDevice, WgpuErrorMonitor, AdapterRecord) {
    let device = WgpuDevice::DiscreteGpu(adapter_index);
    let setup = init_setup::<AutoGraphicsApi>(
        &device,
        RuntimeOptions {
            tasks_max: TASKS_MAX,
            memory_config: MemoryConfiguration::SubSlices,
        },
    );
    let monitor = WgpuErrorMonitor::new();
    let callback_errors = monitor.callback_sink();
    setup.device.on_uncaptured_error(Arc::new(move |error| {
        if let Ok(mut errors) = callback_errors.lock() {
            errors.push(error.to_string());
        }
    }));
    let info = setup.adapter.get_info();
    let record = AdapterRecord {
        index: adapter_index,
        name: info.name,
        backend: format!("{:?}", info.backend),
        device_type: format!("{:?}", info.device_type),
        tasks_max: TASKS_MAX,
        memory_config: "sub-slices",
    };
    println!("wgpu_adapter={record:?}");
    io::stdout().flush().expect("flush WGPU adapter identity");
    (device, monitor, record)
}

fn synchronize(device: &WgpuDevice, monitor: &WgpuErrorMonitor, stage: &str) -> Result<()> {
    let client = WgpuRuntime::client(device);
    cubecl::future::block_on(client.sync())
        .with_context(|| format!("CubeCL synchronization failed after {stage}"))?;
    monitor.check(stage)
}

fn encode(
    engine: &WgslInferenceEngine,
    inputs: &DeviceInputs,
) -> Result<EncodedCondition<Backend>> {
    let (text_ids, text_mask, aux_input) = inputs.compact_no_aux_condition();
    engine
        .model()
        .encode_conditions(text_ids, text_mask, aux_input)
        .context("duration condition encoding failed")
}

fn predict(
    engine: &WgslInferenceEngine,
    condition: &EncodedCondition<Backend>,
    inputs: &DeviceInputs,
) -> Result<Tensor<Backend, 1>> {
    engine
        .model()
        .predict_duration_compact_no_aux_wgsl(
            condition,
            inputs.duration_features.clone(),
            inputs.has_speaker.clone(),
            inputs.has_caption.clone(),
        )
        .context("duration prediction failed")
}

fn execute_scope(
    engine: &WgslInferenceEngine,
    cached: &EncodedCondition<Backend>,
    inputs: &DeviceInputs,
    scope: Scope,
) -> Result<Tensor<Backend, 1>> {
    match scope {
        Scope::Head => predict(engine, cached, inputs),
        Scope::Full => {
            let condition = encode(engine, inputs)?;
            predict(engine, &condition, inputs)
        }
    }
}

fn measure(
    engine: &WgslInferenceEngine,
    cached: &EncodedCondition<Backend>,
    inputs: &DeviceInputs,
    scope: Scope,
    repeat: usize,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
) -> Result<ScopeResult> {
    synchronize(device, monitor, "duration pre-timer")?;
    let started = Instant::now();
    let output = execute_scope(engine, cached, inputs, scope)?;
    synchronize(device, monitor, "duration device completion")?;
    let device_complete_seconds = started.elapsed().as_secs_f64();
    ensure!(
        output.dims() == [1],
        "duration output shape mismatch: {:?}",
        output.dims()
    );
    let values = output
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .context("duration scalar readback failed")?;
    synchronize(device, monitor, "duration scalar readback")?;
    let readback_complete_seconds = started.elapsed().as_secs_f64();
    ensure!(
        values.len() == 1 && values[0].is_finite(),
        "invalid duration scalar"
    );
    let log_frames = values[0];
    let output_sha256 = sha256_bytes(&log_frames.to_le_bytes());
    let row = ScopeResult {
        repeat,
        cold: repeat <= WARMUPS,
        scope,
        timing: Timing {
            device_complete_seconds,
            readback_complete_seconds,
            readback_elements: values.len(),
            readback_dtype: "float32",
            readback_owned: true,
            readback_contiguous: true,
        },
        log_frames,
        predicted_frames: log_frames.exp_m1(),
        output_sha256,
    };
    println!("duration_repeat={}", serde_json::to_string(&row)?);
    io::stdout().flush().context("flush duration repeat")?;
    Ok(row)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let midpoint = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[midpoint - 1] + values[midpoint]) / 2.0
    } else {
        values[midpoint]
    }
}

fn summarize(rows: &[ScopeResult], scope: Scope) -> Result<ScopeSummary> {
    let selected: Vec<_> = rows.iter().filter(|row| row.scope == scope).collect();
    ensure!(
        selected.len() == REPEATS,
        "{} repeat count mismatch",
        scope.label()
    );
    let measured = &selected[WARMUPS..];
    let mut device: Vec<_> = measured
        .iter()
        .map(|row| row.timing.device_complete_seconds)
        .collect();
    let mut readback: Vec<_> = measured
        .iter()
        .map(|row| row.timing.readback_complete_seconds)
        .collect();
    let device_min = device.iter().copied().fold(f64::INFINITY, f64::min);
    let device_max = device.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let readback_min = readback.iter().copied().fold(f64::INFINITY, f64::min);
    let readback_max = readback.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(ScopeSummary {
        warmups: WARMUPS,
        measured: measured.len(),
        device_complete_min_seconds: device_min,
        device_complete_median_seconds: median(&mut device),
        device_complete_max_seconds: device_max,
        readback_complete_min_seconds: readback_min,
        readback_complete_median_seconds: median(&mut readback),
        readback_complete_max_seconds: readback_max,
        output_hashes_equal: selected
            .iter()
            .map(|row| row.output_sha256.as_str())
            .all(|hash| hash == selected[0].output_sha256),
        log_frames: selected[0].log_frames,
        predicted_frames: selected[0].predicted_frames,
    })
}

fn resolve_predicted_length(predicted_frames: f64) -> Result<ResolvedLengthRecord> {
    ensure!(
        predicted_frames.is_finite() && predicted_frames >= 0.0,
        "invalid predicted frames: {predicted_frames}"
    );
    let min_frames = ((MIN_SECONDS * SAMPLE_RATE as f64) / HOP_LENGTH as f64)
        .ceil()
        .max(1.0) as usize;
    let max_frames = ((MAX_SECONDS * SAMPLE_RATE as f64) / HOP_LENGTH as f64)
        .floor()
        .max(1.0) as usize;
    let latent_frames =
        (predicted_frames.round_ties_even().max(0.0) as usize).clamp(min_frames, max_frames);
    let target_samples = latent_frames
        .checked_mul(HOP_LENGTH)
        .context("resolved duration sample count overflow")?;
    Ok(ResolvedLengthRecord {
        duration_scale: 1.0,
        min_seconds: MIN_SECONDS,
        max_seconds: MAX_SECONDS,
        sample_rate: SAMPLE_RATE,
        hop_length: HOP_LENGTH,
        latent_patch_size: LATENT_PATCH_SIZE,
        latent_frames,
        patched_frames: latent_frames.div_ceil(LATENT_PATCH_SIZE),
        target_samples,
        seconds: target_samples as f64 / SAMPLE_RATE as f64,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.repeats == REPEATS,
        "duration benchmark requires exactly {REPEATS} repeats"
    );
    ensure!(
        !args.json_out.exists(),
        "output already exists: {}",
        args.json_out.display()
    );
    let checkpoint_sha = verify_file(&args.checkpoint, &args.checkpoint_sha256, "checkpoint")?;
    ensure!(
        checkpoint_sha == MODEL_SHA256,
        "checkpoint is not released v4-Small"
    );
    let fixture_sha = verify_file(&args.fixture, &args.fixture_sha256, "fixture")?;
    let python_sha = verify_file(&args.python_json, &args.python_json_sha256, "Python JSON")?;
    let fixture = load_fixture(&args.fixture)?;
    let python = load_python_reference(&args.python_json, &fixture_sha)?;

    let (device, monitor, adapter) = initialize_wgpu(args.adapter_index);
    let loaded = InferenceBuilder::<Backend, _>::new(device.clone())
        .load_weights(&args.checkpoint)
        .context("failed to load v4 model")?;
    let config = loaded.model_config().clone();
    ensure!(
        config.use_duration_predictor,
        "checkpoint duration predictor is disabled"
    );
    ensure!(
        config.duration_aux_dim == DURATION_FEATURES,
        "duration feature width mismatch"
    );
    ensure!(
        config.latent_dim == LATENT_DIM && config.latent_patch_size == 1,
        "latent config mismatch"
    );
    let speaker_tokens = config
        .speaker_patch_size
        .context("checkpoint has no speaker_patch_size")?
        .max(1);
    let engine = loaded.with_default_sampling().build_wgsl();
    synchronize(&device, &monitor, "model load and WGSL preparation")?;

    let inputs = DeviceInputs {
        text_ids: Tensor::from_data(TensorData::new(fixture.text_ids, [1, TEXT_TOKENS]), &device),
        text_mask: Tensor::from_data(
            TensorData::new(fixture.text_mask.clone(), [1, TEXT_TOKENS]),
            &device,
        ),
        duration_features: Tensor::from_data(
            TensorData::new(python.features.clone(), [1, DURATION_FEATURES]),
            &device,
        ),
        has_speaker: Tensor::<Backend, 1, Bool>::from_data(
            TensorData::new(vec![false], [1]),
            &device,
        ),
        has_caption: Tensor::<Backend, 1, Bool>::from_data(
            TensorData::new(vec![false], [1]),
            &device,
        ),
        text_valid_tokens: fixture.text_mask.iter().filter(|&&value| value).count(),
    };
    ensure!(
        inputs.text_valid_tokens == python.text_valid_tokens,
        "fixture/Python valid-token count mismatch"
    );
    let cached = encode(&engine, &inputs)?;
    synchronize(&device, &monitor, "cached duration condition")?;
    let encoded_text_shape = cached.text_state.dims();
    let encoded_aux_removed = cached.aux.is_none();

    let mut rows = Vec::with_capacity(REPEATS * 2);
    for repeat in 1..=REPEATS {
        let order = if repeat % 2 == 1 {
            [Scope::Head, Scope::Full]
        } else {
            [Scope::Full, Scope::Head]
        };
        for scope in order {
            rows.push(measure(
                &engine, &cached, &inputs, scope, repeat, &device, &monitor,
            )?);
        }
    }
    synchronize(&device, &monitor, "duration benchmark completion")?;

    let head = summarize(&rows, Scope::Head)?;
    let full = summarize(&rows, Scope::Full)?;
    let resolved_length = resolve_predicted_length(f64::from(full.predicted_frames))?;
    ensure!(
        resolved_length == python.resolved_length,
        "Python/WGPU resolved duration mismatch: Python={:?}, WGPU={:?}",
        python.resolved_length,
        resolved_length
    );
    ensure!(
        head.output_hashes_equal && full.output_hashes_equal,
        "WGPU duration output is nondeterministic"
    );
    ensure!(
        head.log_frames.to_bits() == full.log_frames.to_bits(),
        "WGPU head/full outputs differ"
    );
    let python_max_abs = (head.log_frames - python.head.log_frames)
        .abs()
        .max((head.log_frames - python.full.log_frames).abs());
    ensure!(
        python_max_abs <= PYTHON_ABS_TOLERANCE,
        "WGPU duration output differs from Python by {python_max_abs}, tolerance={PYTHON_ABS_TOLERANCE}"
    );

    let mut scopes = BTreeMap::new();
    scopes.insert("head", head);
    scopes.insert("full", full);
    let mut pins = BTreeMap::new();
    pins.insert("checkpoint_sha256", checkpoint_sha);
    pins.insert("fixture_sha256", fixture_sha);
    pins.insert("python_json_sha256", python_sha);
    let payload = Payload {
        format: FORMAT,
        pins,
        adapter,
        input: InputRecord {
            text: python.text,
            text_shape: [1, TEXT_TOKENS],
            text_valid_tokens: fixture.text_mask.iter().filter(|&&value| value).count(),
            requested_speaker_shape: [1, speaker_tokens, LATENT_DIM],
            caption_shape: [1, CAPTION_TOKENS],
            requested_joint_condition_tokens: TEXT_TOKENS + speaker_tokens + CAPTION_TOKENS,
            encoded_text_shape,
            encoded_aux_removed,
            duration_features_f32: python.features,
        },
        timer_contract: TimerContract {
            primary: "pre-sync to device complete; scalar readback excluded",
            secondary: "owned contiguous float32 one-element CPU readback complete",
            pre_start_sync: true,
            cpu_readback_in_primary: false,
            cpu_readback_elements: 1,
        },
        resolved_length,
        python_reference: PythonReferenceRecord {
            format: python.format,
            head_log_frames: python.head.log_frames,
            full_log_frames: python.full.log_frames,
            rust_log_frames: rows[0].log_frames,
            max_abs_error: python_max_abs,
            tolerance: PYTHON_ABS_TOLERANCE,
            passed: true,
        },
        scopes,
        repeats: rows,
        wgpu_uncaptured_errors: monitor.count()?,
    };
    if let Some(parent) = args.json_out.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut output = File::options()
        .write(true)
        .create_new(true)
        .open(&args.json_out)?;
    serde_json::to_writer_pretty(&mut output, &payload)?;
    output.write_all(b"\n")?;
    output.flush()?;
    output.sync_all()?;
    println!(
        "duration_summary={}",
        serde_json::to_string(&payload.scopes)?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_lengths_match_released_duration_rounding() {
        let cases = [
            (45.381_015_214_336_86, 45, 86_400, 1.8),
            (111.602_249_616_249_18, 112, 215_040, 4.48),
            (333.443_053_490_291_8, 333, 639_360, 13.32),
            (685.135_738_441_183_7, 685, 1_315_200, 27.4),
        ];
        for (predicted, frames, samples, seconds) in cases {
            let resolved = resolve_predicted_length(predicted).expect("valid prediction");
            assert_eq!(resolved.latent_frames, frames);
            assert_eq!(resolved.patched_frames, frames);
            assert_eq!(resolved.target_samples, samples);
            assert_eq!(resolved.seconds, seconds);
        }
    }

    #[test]
    fn resolved_length_uses_ties_even_and_released_bounds() {
        assert_eq!(
            resolve_predicted_length(44.5)
                .expect("valid even tie")
                .latent_frames,
            44
        );
        assert_eq!(
            resolve_predicted_length(45.5)
                .expect("valid odd tie")
                .latent_frames,
            46
        );
        assert_eq!(
            resolve_predicted_length(0.0)
                .expect("valid lower clamp")
                .latent_frames,
            13
        );
        assert_eq!(
            resolve_predicted_length(1_000_000.0)
                .expect("valid upper clamp")
                .latent_frames,
            750
        );
        assert!(resolve_predicted_length(f64::NAN).is_err());
        assert!(resolve_predicted_length(-1.0).is_err());
    }
}
