//! Compare the real Irodori v4 ModernBERT stage against the pinned PyTorch oracle.
//!
//! Generate the oracle first:
//!
//! ```text
//! uv run scripts/export_v4_modern_oracle.py
//! ```
//!
//! Then run this binary with the exact v4 `model.safetensors` checkpoint.

#![recursion_limit = "512"]

use std::{
    error::Error,
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
    time::Instant,
};

use burn::{
    backend::NdArray,
    backend::wgpu::{
        MemoryConfiguration, RuntimeOptions, WgpuDevice, graphics::AutoGraphicsApi, init_setup,
    },
    tensor::{Bool, Int, Tensor, TensorData, backend::Backend},
};
use irodori_tts_wgpu::{
    BackendConfig, WgpuRaw, model::modern_bert::SharedModernBertConditioner, weights::TensorStore,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const MODEL_REVISION: &str = "e4aaac4df355ff560dcd35e0dae272c3a759317b";
const MODEL_SHA256: &str = "5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593";
const MODEL_SIZE: u64 = 3_064_295_596;
const METADATA_SHA256: &str = "5ac9c3f766ad08facdebac69be272993fcce5444d31a1e4f6dd86a64a03b7c89";
const BACKBONE_SHA256: &str = "df906be62a2fcdb3ae56cc064cb25a0fc583cc36643e543d618691a6b98a1bc0";
const TEXT_PROJECTOR_SHA256: &str =
    "8d0ebc4ff38f62d4dd87357cf8ec78ce1fffcbae9abecc6b245525e3f772d9c2";
const CAPTION_PROJECTOR_SHA256: &str =
    "682fa7592d5322fbb1aa432b19d23aedb2c4d6133f88274d6a19ebf5b851e37f";
const MAX_ABS_LIMIT: f64 = 5.0e-5;
const COSINE_MIN: f64 = 0.999_999;
const DEFAULT_TASKS_MAX: usize = 32;

#[derive(Clone, Copy, Debug)]
enum BackendChoice {
    NdArray,
    WgpuRaw,
    Both,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemoryConfig {
    SubSlices,
    ExclusivePages,
}

impl MemoryConfig {
    const fn label(self) -> &'static str {
        match self {
            Self::SubSlices => "sub-slices",
            Self::ExclusivePages => "exclusive-pages",
        }
    }

    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "sub-slices" => Ok(Self::SubSlices),
            "exclusive-pages" => Ok(Self::ExclusivePages),
            other => Err(format!(
                "unsupported memory configuration {other:?}; expected sub-slices or exclusive-pages"
            )),
        }
    }

    fn runtime(self) -> MemoryConfiguration {
        match self {
            Self::SubSlices => MemoryConfiguration::SubSlices,
            Self::ExclusivePages => MemoryConfiguration::ExclusivePages,
        }
    }
}

#[derive(Debug)]
struct Args {
    checkpoint: PathBuf,
    oracle_dir: PathBuf,
    backend: BackendChoice,
    gpu_id: u32,
    wgpu_adapter_index: Option<usize>,
    tasks_max: usize,
    memory_config: MemoryConfig,
}

#[derive(Debug, Deserialize)]
struct OracleMetadata {
    format_version: u32,
    model_revision: String,
    checkpoint_bytes: u64,
    checkpoint_sha256: String,
    transformers_version: String,
    attention_implementation: String,
    input_ids: Vec<Vec<i32>>,
    attention_mask: Vec<Vec<bool>>,
    input_shape: Vec<usize>,
    outputs: OracleOutputs,
}

#[derive(Debug, Deserialize)]
struct OracleOutputs {
    backbone: OracleTensor,
    text_projector: OracleTensor,
    caption_projector: OracleTensor,
}

#[derive(Debug, Deserialize)]
struct OracleTensor {
    file: String,
    dtype: String,
    shape: Vec<usize>,
    elements: usize,
    bytes: usize,
}

#[derive(Clone, Copy, Debug)]
struct Metrics {
    max_abs: f64,
    mae: f64,
    rmse: f64,
    cosine: f64,
}

fn usage() -> &'static str {
    "usage: validate_v4_modern --checkpoint PATH [--oracle-dir PATH] \
     [--backend ndarray|wgpu-raw|both] [--gpu-id N] [--wgpu-adapter-index N] \
     [--tasks-max N] [--memory-config sub-slices|exclusive-pages]"
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    parse_args_from(std::env::args().skip(1))
}

fn parse_tasks_max(value: &str) -> Result<usize, String> {
    let tasks_max = value
        .parse::<usize>()
        .map_err(|error| format!("invalid task aggregation limit {value:?}: {error}"))?;
    if tasks_max == 0 {
        return Err("task aggregation limit must be a positive integer".to_owned());
    }
    Ok(tasks_max)
}

fn parse_args_from<I>(mut arguments: I) -> Result<Args, Box<dyn Error>>
where
    I: Iterator<Item = String>,
{
    let mut checkpoint = None;
    let mut oracle_dir = PathBuf::from("/tmp/irodori-v4-modern-oracle");
    let mut backend = BackendChoice::NdArray;
    let mut gpu_id = 0_u32;
    let mut wgpu_adapter_index = None;
    let mut tasks_max = DEFAULT_TASKS_MAX;
    let mut memory_config = MemoryConfig::SubSlices;

    while let Some(argument) = arguments.next() {
        let value = |arguments: &mut I| {
            arguments
                .next()
                .ok_or_else(|| format!("{argument} requires a value"))
        };
        match argument.as_str() {
            "--checkpoint" => checkpoint = Some(PathBuf::from(value(&mut arguments)?)),
            "--oracle-dir" => oracle_dir = PathBuf::from(value(&mut arguments)?),
            "--backend" => {
                backend = match value(&mut arguments)?.as_str() {
                    "ndarray" => BackendChoice::NdArray,
                    "wgpu-raw" => BackendChoice::WgpuRaw,
                    "both" => BackendChoice::Both,
                    other => return Err(format!("unsupported backend {other:?}").into()),
                };
            }
            "--gpu-id" => gpu_id = value(&mut arguments)?.parse()?,
            "--wgpu-adapter-index" => wgpu_adapter_index = Some(value(&mut arguments)?.parse()?),
            "--tasks-max" => tasks_max = parse_tasks_max(&value(&mut arguments)?)?,
            "--memory-config" => memory_config = MemoryConfig::parse(&value(&mut arguments)?)?,
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}; {}", usage()).into()),
        }
    }

    Ok(Args {
        checkpoint: checkpoint.ok_or_else(|| format!("--checkpoint is required; {}", usage()))?,
        oracle_dir,
        backend,
        gpu_id,
        wgpu_adapter_index,
        tasks_max,
        memory_config,
    })
}

fn read_metadata(oracle_dir: &Path) -> Result<OracleMetadata, Box<dyn Error>> {
    let path = oracle_dir.join("metadata.json");
    let metadata: OracleMetadata = serde_json::from_slice(&fs::read(&path)?)?;
    if metadata.format_version != 1 {
        return Err(format!("unsupported oracle format {}", metadata.format_version).into());
    }
    if metadata.model_revision != MODEL_REVISION
        || metadata.checkpoint_sha256 != MODEL_SHA256
        || metadata.checkpoint_bytes != MODEL_SIZE
    {
        return Err("oracle does not describe the pinned v4 checkpoint".into());
    }
    if metadata.transformers_version != "5.12.1" {
        return Err(format!(
            "oracle used transformers {}, expected 5.12.1",
            metadata.transformers_version
        )
        .into());
    }
    if metadata.input_shape.len() != 2 {
        return Err("oracle input_shape must have rank two".into());
    }
    Ok(metadata)
}

fn verify_sha256(path: &Path, expected: &str) -> Result<(), Box<dyn Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected {
        return Err(format!(
            "SHA-256 mismatch for {}: expected {expected}, got {actual}",
            path.display()
        )
        .into());
    }
    Ok(())
}

fn read_raw_f32(oracle_dir: &Path, tensor: &OracleTensor) -> Result<Vec<f32>, Box<dyn Error>> {
    if tensor.dtype != "f32-le" {
        return Err(format!("unsupported oracle dtype {:?}", tensor.dtype).into());
    }
    if tensor.shape.iter().product::<usize>() != tensor.elements
        || tensor.bytes != tensor.elements * size_of::<f32>()
    {
        return Err(format!("inconsistent oracle tensor metadata for {}", tensor.file).into());
    }
    let bytes = fs::read(oracle_dir.join(&tensor.file))?;
    if bytes.len() != tensor.bytes {
        return Err(format!(
            "{}: expected {} bytes, got {}",
            tensor.file,
            tensor.bytes,
            bytes.len()
        )
        .into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn flatten_rectangular<T: Copy>(
    rows: &[Vec<T>],
    shape: [usize; 2],
) -> Result<Vec<T>, Box<dyn Error>> {
    if rows.len() != shape[0] || rows.iter().any(|row| row.len() != shape[1]) {
        return Err(format!("input rows do not match shape {shape:?}").into());
    }
    Ok(rows.iter().flatten().copied().collect())
}

fn metrics(reference: &[f32], actual: &[f32]) -> Result<Metrics, Box<dyn Error>> {
    if reference.len() != actual.len() || reference.is_empty() {
        return Err(format!(
            "metric length mismatch: reference={}, actual={}",
            reference.len(),
            actual.len()
        )
        .into());
    }

    let mut max_abs = 0.0_f64;
    let mut abs_sum = 0.0_f64;
    let mut square_sum = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut reference_square_sum = 0.0_f64;
    let mut actual_square_sum = 0.0_f64;
    for (&reference, &actual) in reference.iter().zip(actual) {
        if !reference.is_finite() || !actual.is_finite() {
            return Err("non-finite value found while measuring parity".into());
        }
        let reference = f64::from(reference);
        let actual = f64::from(actual);
        let difference = (reference - actual).abs();
        max_abs = max_abs.max(difference);
        abs_sum += difference;
        square_sum += difference * difference;
        dot += reference * actual;
        reference_square_sum += reference * reference;
        actual_square_sum += actual * actual;
    }
    let count = reference.len() as f64;
    let denominator = (reference_square_sum * actual_square_sum).sqrt();
    Ok(Metrics {
        max_abs,
        mae: abs_sum / count,
        rmse: (square_sum / count).sqrt(),
        cosine: dot / denominator,
    })
}

fn print_metrics(name: &str, values: Metrics) {
    println!(
        "  {name:18} max_abs={:.9e}  MAE={:.9e}  RMSE={:.9e}  cosine={:.12}",
        values.max_abs, values.mae, values.rmse, values.cosine
    );
}

fn enforce_metrics(name: &str, values: Metrics) -> Result<(), Box<dyn Error>> {
    if !values.max_abs.is_finite()
        || !values.mae.is_finite()
        || !values.rmse.is_finite()
        || !values.cosine.is_finite()
    {
        return Err(format!("{name} produced non-finite parity metrics").into());
    }
    if values.max_abs > MAX_ABS_LIMIT || values.cosine < COSINE_MIN {
        return Err(format!(
            "{name} parity gate failed: max_abs={:.9e} (limit {MAX_ABS_LIMIT:.9e}), cosine={:.12} (minimum {COSINE_MIN:.12})",
            values.max_abs, values.cosine
        )
        .into());
    }
    Ok(())
}

fn initialize_wgpu(device: &WgpuDevice, tasks_max: usize, memory_config: MemoryConfig) {
    let setup = init_setup::<AutoGraphicsApi>(
        device,
        RuntimeOptions {
            tasks_max,
            memory_config: memory_config.runtime(),
        },
    );
    let info = setup.adapter.get_info();
    println!(
        "wgpu adapter: logical={device:?} name={:?} backend={:?} device_type={:?} tasks_max={tasks_max} memory_config={}",
        info.name,
        info.backend,
        info.device_type,
        memory_config.label(),
    );
}

fn validate_backend<B: Backend + BackendConfig>(
    checkpoint: &Path,
    oracle_dir: &Path,
    metadata: &OracleMetadata,
    device: B::Device,
) -> Result<(), Box<dyn Error>> {
    B::check_requirements(&device).map_err(|message| format!("backend unavailable: {message}"))?;
    let [batch, sequence] = metadata.input_shape[..]
        .try_into()
        .map_err(|_| "oracle input_shape must have two dimensions")?;
    let ids = flatten_rectangular(&metadata.input_ids, [batch, sequence])?;
    let mask = flatten_rectangular(&metadata.attention_mask, [batch, sequence])?;

    println!("{}", B::backend_label());
    let load_started = Instant::now();
    let store = TensorStore::load(checkpoint)?;
    let record = store.v4_modern_bert_conditioner::<B>(&device)?;
    drop(store);
    let model = SharedModernBertConditioner::<B>::v4_small_from_record(record, &device);
    println!("  load={:.3}s", load_started.elapsed().as_secs_f64());

    let input_ids =
        Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [batch, sequence]), &device);
    let attention_mask =
        Tensor::<B, 2, Bool>::from_data(TensorData::new(mask, [batch, sequence]), &device);
    let forward_started = Instant::now();
    let (backbone, text, caption) = model.forward_all(input_ids, attention_mask);
    let backbone = backbone.to_data().to_vec::<f32>()?;
    let text = text.to_data().to_vec::<f32>()?;
    let caption = caption.to_data().to_vec::<f32>()?;
    println!(
        "  forward+readback={:.3}s",
        forward_started.elapsed().as_secs_f64()
    );

    let reference_backbone = read_raw_f32(oracle_dir, &metadata.outputs.backbone)?;
    let reference_text = read_raw_f32(oracle_dir, &metadata.outputs.text_projector)?;
    let reference_caption = read_raw_f32(oracle_dir, &metadata.outputs.caption_projector)?;
    for (name, values) in [
        ("backbone", metrics(&reference_backbone, &backbone)?),
        ("text_projector", metrics(&reference_text, &text)?),
        ("caption_projector", metrics(&reference_caption, &caption)?),
    ] {
        print_metrics(name, values);
        enforce_metrics(name, values)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    verify_sha256(&args.checkpoint, MODEL_SHA256)?;
    for (file, digest) in [
        ("metadata.json", METADATA_SHA256),
        ("backbone.f32le", BACKBONE_SHA256),
        ("text_projector.f32le", TEXT_PROJECTOR_SHA256),
        ("caption_projector.f32le", CAPTION_PROJECTOR_SHA256),
    ] {
        verify_sha256(&args.oracle_dir.join(file), digest)?;
    }
    let checkpoint_size = fs::metadata(&args.checkpoint)?.len();
    if checkpoint_size != MODEL_SIZE {
        return Err(format!(
            "checkpoint size mismatch: expected {MODEL_SIZE}, got {checkpoint_size}"
        )
        .into());
    }
    let metadata = read_metadata(&args.oracle_dir)?;
    println!(
        "oracle: transformers={} attention={} revision={}",
        metadata.transformers_version, metadata.attention_implementation, metadata.model_revision
    );
    println!("checkpoint_sha256: {}", metadata.checkpoint_sha256);

    match args.backend {
        BackendChoice::NdArray => validate_backend::<NdArray<f32>>(
            &args.checkpoint,
            &args.oracle_dir,
            &metadata,
            NdArray::<f32>::device_from_id(args.gpu_id),
        )?,
        BackendChoice::WgpuRaw => {
            let device = args
                .wgpu_adapter_index
                .map(WgpuDevice::DiscreteGpu)
                .unwrap_or_else(|| WgpuRaw::device_from_id(args.gpu_id));
            initialize_wgpu(&device, args.tasks_max, args.memory_config);
            validate_backend::<WgpuRaw>(&args.checkpoint, &args.oracle_dir, &metadata, device)?
        }
        BackendChoice::Both => {
            validate_backend::<NdArray<f32>>(
                &args.checkpoint,
                &args.oracle_dir,
                &metadata,
                NdArray::<f32>::device_from_id(args.gpu_id),
            )?;
            let device = args
                .wgpu_adapter_index
                .map(WgpuDevice::DiscreteGpu)
                .unwrap_or_else(|| WgpuRaw::device_from_id(args.gpu_id));
            initialize_wgpu(&device, args.tasks_max, args.memory_config);
            validate_backend::<WgpuRaw>(&args.checkpoint, &args.oracle_dir, &metadata, device)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(arguments: &[&str]) -> Result<Args, Box<dyn Error>> {
        parse_args_from(arguments.iter().map(|argument| (*argument).to_owned()))
    }

    #[test]
    fn cli_runtime_defaults_match_production() {
        let args = parse(&["--checkpoint", "model.safetensors"]).unwrap();
        assert_eq!(args.tasks_max, DEFAULT_TASKS_MAX);
        assert_eq!(args.memory_config, MemoryConfig::SubSlices);
    }

    #[test]
    fn cli_accepts_explicit_runtime_options() {
        for (value, expected) in [
            ("sub-slices", MemoryConfig::SubSlices),
            ("exclusive-pages", MemoryConfig::ExclusivePages),
        ] {
            let args = parse(&[
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
        assert!(parse(&["--checkpoint", "model.safetensors", "--tasks-max", "0"]).is_err());
        assert!(
            parse(&[
                "--checkpoint",
                "model.safetensors",
                "--memory-config",
                "pooled",
            ])
            .is_err()
        );
    }
}
