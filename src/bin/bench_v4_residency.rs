//! Fail-closed WGPU residency and phase-batch measurement harness.
//!
//! This binary intentionally composes the production engine, codec, and
//! `PhaseBatch` API without changing their operations or synchronization.

#![recursion_limit = "512"]

use std::{
    fs,
    mem::size_of,
    path::{Path, PathBuf},
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
use irodori_tts_burn::{
    BatchAudio, BatchItemId, CfgGuidanceMode, GuidanceConfig, InferenceBuilder, IrodoriError,
    OutputGeometry, PhaseBatch, PlannedSynthesis, SamplerMethod, SamplerParams, SamplingRequest,
    SpeakerKey, VoiceIdentity, WgpuRaw, load_codec, unpatchify_latent,
};
use safetensors::{Dtype, SafeTensors};
use serde::Serialize;
use sha2::{Digest, Sha256};

const MODEL_SHA256: &str = "5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593";
const CODEC_SHA256: &str = "4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1";

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    AllResident,
    PhaseBatch,
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
    #[arg(long, value_enum, default_value = "same")]
    speaker_mode: SpeakerMode,
    #[arg(long, value_enum, default_value = "same")]
    length_mode: LengthMode,
    #[arg(long)]
    output_json: PathBuf,
    #[arg(long, default_value_t = 0)]
    adapter_index: usize,
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
struct Report {
    schema_version: u32,
    mode: Mode,
    speaker_mode: SpeakerMode,
    length_mode: LengthMode,
    requests: usize,
    adapter_index: usize,
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
    load_wall_seconds: f64,
    codec_load_wall_seconds: Option<f64>,
    execution_wall_seconds: f64,
    total_wall_seconds: f64,
    output_seconds: f64,
    requests_per_second: f64,
    end_to_end_requests_per_second: f64,
    audio_seconds_per_wall_second: f64,
    end_to_end_audio_seconds_per_wall_second: f64,
    items: Vec<ItemResult>,
    memory: Vec<MemorySnapshot>,
    phase_timing: Option<PhaseTiming>,
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
    cubecl::future::block_on(WgpuRuntime::client(device).sync()).context("WGPU sync failed")
}

fn snapshot(device: &WgpuDevice, stage: &str) -> Result<MemorySnapshot> {
    sync(device)?;
    let usage = WgpuRuntime::client(device)
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
    reference: &Reference,
    device: &WgpuDevice,
) -> SamplingRequest<WgpuRaw> {
    SamplingRequest {
        text_ids: Tensor::<WgpuRaw, 2, Int>::from_data(
            TensorData::new(fixture.text_ids.clone(), [1, 256]),
            device,
        ),
        text_mask: Tensor::<WgpuRaw, 2, Bool>::from_data(
            TensorData::new(fixture.text_mask.clone(), [1, 256]),
            device,
        ),
        ref_latent: Some(Tensor::<WgpuRaw, 3>::from_data(
            TensorData::new(reference.values.clone(), [1, reference.frames, 32]),
            device,
        )),
        ref_mask: Some(Tensor::<WgpuRaw, 2, Bool>::from_data(
            TensorData::new(vec![true; reference.frames], [1, reference.frames]),
            device,
        )),
        sequence_length: fixture.frames,
        caption_ids: Some(Tensor::<WgpuRaw, 2, Int>::from_data(
            TensorData::new(fixture.caption_ids.clone(), [1, 512]),
            device,
        )),
        caption_mask: Some(Tensor::<WgpuRaw, 2, Bool>::from_data(
            TensorData::new(fixture.caption_mask.clone(), [1, 512]),
            device,
        )),
        initial_noise: Some(Tensor::<WgpuRaw, 3>::from_data(
            TensorData::new(fixture.noise.clone(), [1, fixture.frames, 32]),
            device,
        )),
    }
}

fn audio_result(audio: BatchAudio, frames: usize) -> Result<ItemResult> {
    let values = audio.tensor.into_data().convert::<f32>().to_vec::<f32>()?;
    let mut digest = Sha256::new();
    for value in &values {
        digest.update(value.to_le_bytes());
    }
    let speaker = match &audio.voice {
        VoiceIdentity::Unconditioned => "unconditioned".to_owned(),
        VoiceIdentity::Clone(key) => key.as_str().to_owned(),
        VoiceIdentity::Designed(key) => key.as_str().to_owned(),
    };
    Ok(ItemResult {
        id: audio.id.as_str().to_owned(),
        speaker,
        frames,
        samples: values.len(),
        audio_f32_sha256: format!("{:x}", digest.finalize()),
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.requests > 0, "--requests must be positive");
    ensure!(
        args.reference.len() == 2,
        "exactly two references are required"
    );
    if matches!(args.mode, Mode::AllResident) {
        ensure!(
            args.requests == 1,
            "all-resident probe requires --requests 1"
        );
    }
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
    verify_sha(&args.checkpoint, MODEL_SHA256, "model")?;
    verify_sha(&args.codec_weights, CODEC_SHA256, "codec")?;
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
            memory_config: MemoryConfiguration::SubSlices,
        },
    );
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
    let loaded =
        InferenceBuilder::<WgpuRaw, _>::new(device.clone()).load_weights(&args.checkpoint)?;
    let config = loaded.model_config().clone();
    ensure!(
        config.latent_dim == 32 && config.latent_patch_size == 1,
        "unexpected v4 geometry"
    );
    let engine = loaded.with_sampling(params).build_wgsl();
    sync(&device)?;
    memory.push(snapshot(&device, "rf_resident")?);

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
        let reference = &references[reference_index];
        let id = BatchItemId::new(format!("request-{index:02}"))?;
        let voice = VoiceIdentity::Clone(SpeakerKey::new(format!("ref{}", reference_index + 1))?);
        planned.push(PlannedSynthesis::new(
            id,
            voice,
            OutputGeometry::new(fixture.frames, 1, 32)?,
            make_request(fixture, reference, &device),
        )?);
        item_frames.push(fixture.frames);
    }

    let mut items = Vec::with_capacity(args.requests);
    let mut phase_timing = None;
    let mut codec_load_wall_seconds = None;
    let load_wall_seconds;
    let execution_started;
    match args.mode {
        Mode::AllResident => {
            let mut codec = load_codec::<WgpuRaw>(&args.codec_weights, &device)?;
            codec.prepare_decoder_for_wgsl();
            sync(&device)?;
            load_wall_seconds = load_started.elapsed().as_secs_f64();
            memory.push(snapshot(&device, "rf_duration_codec_resident")?);
            execution_started = Instant::now();
            let one = planned.pop().context("missing request")?;
            sync(&device)?;
            let patched = engine.sample(one.request)?;
            sync(&device)?;
            memory.push(snapshot(&device, "all_resident_after_rf")?);
            let latent = unpatchify_latent(patched, 1, 32);
            let decoded = codec.decode_wgsl(latent);
            sync(&device)?;
            let audio = BatchAudio {
                id: one.id,
                voice: one.voice,
                tensor: decoded,
            };
            items.push(audio_result(audio, item_frames[0])?);
            sync(&device)?;
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
            let mut codec = load_codec::<WgpuRaw>(&args.codec_weights, &device)?;
            codec.prepare_decoder_for_wgsl();
            sync(&device)?;
            codec_load_wall_seconds = Some(codec_load_started.elapsed().as_secs_f64());
            memory.push(snapshot(&device, "latents_codec_resident")?);
            let mut consumed = 0_usize;
            let complete = latents.with_codec(codec).decode_all(|audio| {
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
    let output_seconds = items
        .iter()
        .map(|item| item.samples as f64 / 48_000.0)
        .sum::<f64>();
    let report = Report {
        schema_version: 1,
        mode: args.mode,
        speaker_mode: args.speaker_mode,
        length_mode: args.length_mode,
        requests: args.requests,
        adapter_index: args.adapter_index,
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
        load_wall_seconds,
        codec_load_wall_seconds,
        execution_wall_seconds,
        total_wall_seconds,
        output_seconds,
        requests_per_second: args.requests as f64 / execution_wall_seconds,
        end_to_end_requests_per_second: args.requests as f64 / total_wall_seconds,
        audio_seconds_per_wall_second: output_seconds / execution_wall_seconds,
        end_to_end_audio_seconds_per_wall_second: output_seconds / total_wall_seconds,
        items,
        memory,
        phase_timing,
    };
    if let Some(parent) = args.output_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.output_json, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
