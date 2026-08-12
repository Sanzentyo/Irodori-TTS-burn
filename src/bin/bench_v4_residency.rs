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
    tensor::{Bool, Int, Tensor, TensorData, backend::Backend},
};
use clap::{Parser, ValueEnum};
use cubecl::prelude::Runtime;
use irodori_tts_burn::{
    BatchAudio, BatchItemId, CfgGuidanceMode, GuidanceConfig, InferenceBuilder, IrodoriError,
    OutputGeometry, PhaseBatch, PlannedSynthesis, SamplerMethod, SamplerParams, SamplerWorkReport,
    SamplingRequest, SpeakerKey, VoiceIdentity, WgpuRaw,
    codec::{DacVaeCodec, DacVaeDecoder},
    load_codec, load_decoder, unpatchify_latent,
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

enum ResidentDecoder {
    Full(Box<DacVaeCodec<WgpuRaw>>),
    DecodeOnly(Box<DacVaeDecoder<WgpuRaw>>),
}

impl ResidentDecoder {
    fn prepare_for_wgsl(&mut self) {
        match self {
            Self::Full(codec) => codec.prepare_decoder_for_wgsl(),
            Self::DecodeOnly(codec) => codec.prepare_for_wgsl(),
        }
    }

    fn decode_wgsl(&self, latent: Tensor<WgpuRaw, 3>) -> Tensor<WgpuRaw, 3> {
        match self {
            Self::Full(codec) => codec.decode_wgsl(latent),
            Self::DecodeOnly(codec) => codec.decode_wgsl(latent),
        }
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
    /// Release completely unused allocator pages after the warmup boundary.
    #[arg(long)]
    cleanup_after_warmup: bool,
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
    cleanup_after_warmup: bool,
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
    measured: usize,
    unconditioned: bool,
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
    resident_request_timings: Vec<ResidentRequestTiming>,
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
    reference: Option<&Reference>,
    speaker_patch_size: usize,
    device: &WgpuDevice,
) -> SamplingRequest<WgpuRaw> {
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
        text_ids: Tensor::<WgpuRaw, 2, Int>::from_data(
            TensorData::new(fixture.text_ids.clone(), [1, 256]),
            device,
        ),
        text_mask: Tensor::<WgpuRaw, 2, Bool>::from_data(
            TensorData::new(fixture.text_mask.clone(), [1, 256]),
            device,
        ),
        ref_latent: Some(Tensor::<WgpuRaw, 3>::from_data(
            TensorData::new(reference_values, [1, reference_frames, 32]),
            device,
        )),
        ref_mask: Some(Tensor::<WgpuRaw, 2, Bool>::from_data(
            TensorData::new(reference_mask, [1, reference_frames]),
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
            memory_config: args.allocator.configuration(),
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
                &device,
            ),
        )?);
        item_frames.push(fixture.frames);
    }

    let mut items = Vec::with_capacity(args.requests);
    let mut phase_timing = None;
    let mut resident_request_timings = Vec::new();
    let mut work_report = None;
    let mut codec_load_wall_seconds = None;
    let load_wall_seconds;
    let execution_started;
    match args.mode {
        Mode::AllResident => {
            let mut codec = match args.codec_residency {
                CodecResidency::Full => {
                    ResidentDecoder::Full(Box::new(load_codec(&args.codec_weights, &device)?))
                }
                CodecResidency::DecodeOnly => ResidentDecoder::DecodeOnly(Box::new(load_decoder(
                    &args.codec_weights,
                    &device,
                )?)),
            };
            codec.prepare_for_wgsl();
            sync(&device)?;
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
                if let Some(first) = &work_report {
                    ensure!(
                        first == &report,
                        "RF work manifest changed between requests"
                    );
                } else {
                    work_report = Some(report);
                }
                let rf_device_complete_seconds = request_started.elapsed().as_secs_f64();
                if index == 0 {
                    memory.push(snapshot(&device, "all_resident_after_first_rf")?);
                }
                let codec_started = Instant::now();
                let latent = unpatchify_latent(patched, 1, 32);
                let decoded = codec.decode_wgsl(latent);
                sync(&device)?;
                let codec_device_complete_seconds = codec_started.elapsed().as_secs_f64();
                let audio = BatchAudio {
                    id: one.id,
                    voice: one.voice,
                    tensor: decoded,
                };
                let item = audio_result(audio, item_frames[index])?;
                sync(&device)?;
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
                    <WgpuRaw as Backend>::memory_cleanup(&device);
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
                    let codec = load_codec::<WgpuRaw>(&args.codec_weights, &device)?;
                    sync(&device)?;
                    codec_load_wall_seconds = Some(codec_load_started.elapsed().as_secs_f64());
                    latents.with_codec(codec)
                }
                CodecResidency::DecodeOnly => {
                    let decoder = load_decoder::<WgpuRaw>(&args.codec_weights, &device)?;
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
        allocator: args.allocator,
        codec_residency: args.codec_residency,
        cleanup_after_warmup: args.cleanup_after_warmup,
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
        measured: args.requests - args.warmups,
        unconditioned: args.unconditioned,
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
        resident_request_timings,
        work_report,
    };
    if let Some(parent) = args.output_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.output_json, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
