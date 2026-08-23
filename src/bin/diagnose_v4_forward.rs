//! Fail-closed same-input/same-condition per-block DiT diagnostic.

#![recursion_limit = "512"]

use std::{
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    mem::size_of,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use burn::{
    backend::wgpu::{
        AutoCompiler, MemoryConfiguration, RuntimeOptions, WgpuDevice, WgpuRuntime,
        graphics::AutoGraphicsApi, init_setup,
    },
    tensor::{Bool, Tensor, TensorData},
};
use clap::Parser;
use cubecl::prelude::Runtime;
use irodori_tts_burn::{
    AuxConditionState, DiagnosticForwardInput, EncodedCondition, InferenceBuilder,
    ModelCheckpointLoader, SamplerParams, WgslWeightProfile, backend_config::WgpuFloatPrecision,
};
use safetensors::{Dtype, SafeTensors};
use serde::Serialize;
use sha2::{Digest, Sha256};

const MODEL_SHA256: &str = "5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593";
type WgpuRt = WgpuRuntime<AutoCompiler>;

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,
    /// Python diagnostic safetensors containing `rf_selected_*` tensors.
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    input_sha256: String,
    /// Fresh directory receiving raw f32 tensors and `report.json`.
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long)]
    cubecl_cache_dir: PathBuf,
    #[arg(long)]
    cubecl_bundle_in: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    adapter_index: usize,
}

#[derive(Debug, Serialize)]
struct TensorArtifact {
    name: String,
    path: PathBuf,
    shape: Vec<usize>,
    elements: usize,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    diagnostic_only: bool,
    latency_results_valid: bool,
    model_sha256: String,
    input: PathBuf,
    input_sha256: String,
    adapter_name: String,
    adapter_backend: String,
    block_count: usize,
    tensors: Vec<TensorArtifact>,
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn read_f32(tensors: &SafeTensors<'_>, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let tensor = tensors
        .tensor(name)
        .with_context(|| format!("missing tensor {name:?}"))?;
    ensure!(tensor.dtype() == Dtype::F32, "tensor {name:?} must be F32");
    let values = tensor
        .data()
        .chunks_exact(size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32 chunk")))
        .collect::<Vec<_>>();
    ensure!(
        values.len() == tensor.shape().iter().product::<usize>(),
        "tensor {name:?} byte count does not match its shape"
    );
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "tensor {name:?} contains a non-finite value"
    );
    Ok((tensor.shape().to_vec(), values))
}

fn shape3(shape: &[usize], name: &str) -> Result<[usize; 3]> {
    shape
        .try_into()
        .with_context(|| format!("tensor {name:?} must have rank 3, got {shape:?}"))
}

fn shape2(shape: &[usize], name: &str) -> Result<[usize; 2]> {
    shape
        .try_into()
        .with_context(|| format!("tensor {name:?} must have rank 2, got {shape:?}"))
}

fn f32_tensor3(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &burn::tensor::Device,
) -> Result<Tensor<3>> {
    let (shape, values) = read_f32(tensors, name)?;
    Ok(Tensor::from_data(
        TensorData::new(values, shape3(&shape, name)?),
        device,
    ))
}

fn compact_context(
    tensors: &SafeTensors<'_>,
    state_name: &str,
    mask_name: &str,
    optional: bool,
    device: &burn::tensor::Device,
) -> Result<Option<(Tensor<3>, Tensor<2, Bool>)>> {
    let state_present = tensors.names().contains(&state_name);
    let mask_present = tensors.names().contains(&mask_name);
    ensure!(
        state_present == mask_present,
        "context state/mask must be both present or both absent"
    );
    if !state_present {
        ensure!(optional, "required context {state_name:?} is absent");
        return Ok(None);
    }
    let (state_shape, state_values) = read_f32(tensors, state_name)?;
    let (mask_shape, mask_values) = read_f32(tensors, mask_name)?;
    let [batch, tokens, width] = shape3(&state_shape, state_name)?;
    ensure!(
        shape2(&mask_shape, mask_name)? == [batch, tokens],
        "context {state_name:?} state/mask shape mismatch"
    );
    let mut used_columns = 0;
    for row in 0..batch {
        for column in 0..tokens {
            if mask_values[row * tokens + column] > 0.5 {
                used_columns = used_columns.max(column + 1);
            }
        }
    }
    if used_columns == 0 && optional {
        return Ok(None);
    }
    used_columns = used_columns.max(1);
    let mut compact_state = Vec::with_capacity(batch * used_columns * width);
    let mut compact_mask = Vec::with_capacity(batch * used_columns);
    for row in 0..batch {
        let state_start = row * tokens * width;
        compact_state
            .extend_from_slice(&state_values[state_start..state_start + used_columns * width]);
        let mask_start = row * tokens;
        compact_mask.extend_from_slice(&mask_values[mask_start..mask_start + used_columns]);
    }
    Ok(Some((
        Tensor::from_data(
            TensorData::new(compact_state, [batch, used_columns, width]),
            device,
        ),
        Tensor::<2>::from_data(TensorData::new(compact_mask, [batch, used_columns]), device)
            .greater_elem(0.5),
    )))
}

fn write_tensor(directory: &Path, name: &str, tensor: Tensor<3>) -> Result<TensorArtifact> {
    let shape = tensor.dims();
    let values = tensor.into_data().convert::<f32>().to_vec::<f32>()?;
    let path = directory.join(format!("{name}.f32le"));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for value in &values {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(TensorArtifact {
        name: name.to_owned(),
        path: path.clone(),
        shape: shape.to_vec(),
        elements: values.len(),
        sha256: sha256_file(&path)?,
    })
}

fn main() -> Result<()> {
    irodori_tts_burn::backend_config::initialize_cli_tracing("info")?;
    let args = Args::parse();
    ensure!(args.checkpoint.is_file(), "checkpoint is not a file");
    ensure!(args.input.is_file(), "input is not a file");
    ensure!(
        !args.output_dir.exists() && !args.output_dir.is_symlink(),
        "refusing to reuse output directory {}",
        args.output_dir.display()
    );
    ensure!(
        sha256_file(&args.checkpoint)? == MODEL_SHA256,
        "model SHA mismatch"
    );
    ensure!(
        sha256_file(&args.input)? == args.input_sha256,
        "input SHA mismatch"
    );
    irodori_tts_burn::backend_config::configure_cubecl_persistent_cache_for_precision(
        &args.cubecl_cache_dir,
        WgpuFloatPrecision::Fp32,
    )?;
    if let Some(bundle) = args.cubecl_bundle_in.as_ref() {
        irodori_tts_burn::backend_config::import_cubecl_environment_bundle(bundle)?;
    }

    let wgpu_device = WgpuDevice::DiscreteGpu(args.adapter_index);
    let setup = init_setup::<AutoGraphicsApi>(
        &wgpu_device,
        RuntimeOptions {
            tasks_max: 32,
            memory_config: MemoryConfiguration::ExclusivePages,
        },
    );
    let device = irodori_tts_burn::backend_config::strict_fp32_device(&wgpu_device)?;
    let input_bytes = fs::read(&args.input)?;
    let tensors = SafeTensors::deserialize(&input_bytes)?;

    let latent = f32_tensor3(&tensors, "rf_selected_x_t", &device)?;
    let timestep = {
        let (shape, values) = read_f32(&tensors, "rf_selected_t")?;
        ensure!(shape.len() == 1, "selected timestep must have rank 1");
        Tensor::<1>::from_data(TensorData::new(values, [shape[0]]), &device)
    };
    let (text_state, text_mask) = compact_context(
        &tensors,
        "rf_selected_text_state",
        "rf_selected_text_mask",
        false,
        &device,
    )?
    .expect("required text context");
    let speaker = compact_context(
        &tensors,
        "rf_selected_speaker_state",
        "rf_selected_speaker_mask",
        true,
        &device,
    )?;
    let caption = compact_context(
        &tensors,
        "rf_selected_caption_state",
        "rf_selected_caption_mask",
        true,
        &device,
    )?;
    let aux = match (speaker, caption) {
        (None, None) => None,
        (Some((state, mask)), None) => Some(AuxConditionState::Speaker { state, mask }),
        (None, Some((state, mask))) => Some(AuxConditionState::Caption { state, mask }),
        (Some((speaker_state, speaker_mask)), Some((caption_state, caption_mask))) => {
            Some(AuxConditionState::Both {
                speaker_state,
                speaker_mask,
                caption_state,
                caption_mask,
            })
        }
    };
    let condition = EncodedCondition {
        text_state,
        text_mask,
        aux,
    };
    let engine = InferenceBuilder::<_>::new(device.clone())
        .load_weights_with_loader(&args.checkpoint, ModelCheckpointLoader::IndexedFile)?
        .with_sampling(SamplerParams::default())
        .build_wgsl_with_profile(WgslWeightProfile::ProductionPrepared)?;
    let trace =
        engine.diagnostic_forward(DiagnosticForwardInput::new(latent, timestep, condition))?;
    cubecl::future::block_on(WgpuRt::client(&wgpu_device).sync())?;

    fs::create_dir(&args.output_dir)?;
    let mut artifacts = Vec::with_capacity(trace.block_outputs.len() + 2);
    artifacts.push(write_tensor(
        &args.output_dir,
        "rf_selected_after_input_projection",
        trace.after_input_projection,
    )?);
    for (index, block) in trace.block_outputs.into_iter().enumerate() {
        artifacts.push(write_tensor(
            &args.output_dir,
            &format!("rf_selected_block_{index:02}"),
            block,
        )?);
    }
    artifacts.push(write_tensor(
        &args.output_dir,
        "rf_selected_output",
        trace.output,
    )?);
    let report = Report {
        schema_version: 1,
        diagnostic_only: true,
        latency_results_valid: false,
        model_sha256: MODEL_SHA256.to_owned(),
        input: args.input.canonicalize()?,
        input_sha256: args.input_sha256,
        adapter_name: setup.adapter.get_info().name,
        adapter_backend: format!("{:?}", setup.adapter.get_info().backend),
        block_count: artifacts.len() - 2,
        tensors: artifacts,
    };
    let report_path = args.output_dir.join("report.json");
    let report_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&report_path)?;
    serde_json::to_writer_pretty(&report_file, &report)?;
    report_file.sync_all()?;
    Ok(())
}
