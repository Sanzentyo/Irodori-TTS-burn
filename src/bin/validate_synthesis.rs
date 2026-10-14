//! Replay official RF/MF fixtures and compare whole versus windowed decoding.
use anyhow::{Result, ensure};
use burn::tensor::{Bool, DType, Device, Int, Tensor, TensorData};
use clap::Parser;
use irodori_tts_burn::{
    FlowParameterization, InferenceBuilder, SamplerParams, SamplingRequest,
    TimestepConditionCachePolicy, WgpuFloatPrecision, backend_config,
    codec::{DecodeWindow, load_decoder_with_float_dtype},
};
use safetensors::{Dtype, SafeTensors};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, io::Write, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    codec: PathBuf,
    #[arg(long)]
    fixture: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum, default_value = "fp32")]
    precision: WgpuFloatPrecision,
    #[arg(long, value_enum, default_value = "model")]
    sampler_state: irodori_tts_burn::SamplerStatePrecision,
    #[arg(long)]
    adapter: Option<usize>,
    #[arg(long, default_value = "discrete:0", conflicts_with = "adapter")]
    device: backend_config::DeviceSelector,
    #[arg(long, default_value_t = 3)]
    repeat: usize,
    #[arg(long, default_value_t = 2)]
    warmups: usize,
    #[arg(long, value_enum, default_value = "full")]
    cache_policy: CachePolicy,
    /// Install an exact profiling candidate before any model weight preparation.
    #[cfg(feature = "profile")]
    #[arg(long, conflicts_with_all = ["portable", "autotune"])]
    route_profile: Option<PathBuf>,
    /// Use the portable graph as the control for a new adapter.
    #[arg(long)]
    portable: bool,
    /// Export the first forward's exact text condition and sampler stages.
    #[arg(long)]
    dump_stages: bool,
    /// Export compacted ModernBERT inputs and synchronized layer boundaries.
    #[cfg(feature = "profile")]
    #[arg(long)]
    dump_text_layers: bool,
    /// Diagnose RF/MF and production codec GPU stages in a separate request.
    #[cfg(feature = "profile")]
    #[arg(long)]
    device_profile: bool,
    /// Measure and persist accuracy-checked attention and MLP routes.
    #[arg(long, conflicts_with = "portable")]
    autotune: bool,
    #[arg(long)]
    autotune_record: Option<PathBuf>,
    /// Compare fixed-request sampler and complete audio graph replay.
    #[arg(long)]
    graph_replay: bool,
    /// Compare graph reuse across changed text, reference voice and noise.
    #[arg(long)]
    condition_graph: bool,
    /// Additional complete sampler plus streaming-decode latency measurements.
    #[arg(long, default_value_t = 0)]
    stream_repeat: usize,
    /// Compare bounded codec graph reuse with the same streaming consumer.
    #[cfg(feature = "streaming-graphs")]
    #[arg(long, requires = "stream_repeat")]
    stream_graph_entries: Option<std::num::NonZeroUsize>,
    /// Subsequent streaming window size; production synthesis uses 96 frames.
    #[arg(long, default_value_t = 17)]
    stream_window_frames: usize,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum CachePolicy {
    Disabled,
    Condition,
    Full,
}
fn floats(file: &SafeTensors<'_>, key: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let view = file.tensor(key)?;
    ensure!(view.dtype() == Dtype::F32, "{key} must be FP32");
    let values = view
        .data()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four bytes")))
        .collect();
    Ok((values, view.shape().to_vec()))
}
fn float_tensor<const D: usize>(
    file: &SafeTensors<'_>,
    key: &str,
    device: &Device,
    dtype: DType,
) -> Result<Tensor<D>> {
    let (values, shape) = floats(file, key)?;
    ensure!(shape.len() == D, "{key} rank mismatch");
    Ok(Tensor::from_data(
        TensorData::new(values, shape),
        (device, dtype),
    ))
}
fn ints(file: &SafeTensors<'_>, key: &str, device: &Device) -> Result<Tensor<2, Int>> {
    let view = file.tensor(key)?;
    ensure!(
        view.dtype() == Dtype::I32 && view.shape().len() == 2,
        "{key} must be rank-two I32"
    );
    let values = view
        .data()
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("four bytes")))
        .collect::<Vec<_>>();
    Ok(Tensor::from_data(
        TensorData::new(values, view.shape()),
        device,
    ))
}
fn mask(file: &SafeTensors<'_>, key: &str, device: &Device) -> Result<Tensor<2, Bool>> {
    let view = file.tensor(key)?;
    ensure!(
        view.dtype() == Dtype::BOOL && view.shape().len() == 2,
        "{key} must be rank-two BOOL"
    );
    Ok(Tensor::from_data(
        TensorData::new(
            view.data()
                .iter()
                .map(|value| *value != 0)
                .collect::<Vec<_>>(),
            view.shape(),
        ),
        device,
    ))
}
fn read<const D: usize>(tensor: Tensor<D>) -> Result<Vec<f32>> {
    tensor
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .map_err(|error| anyhow::anyhow!("readback: {error}"))
}
fn compare(expected: &[f32], actual: &[f32]) -> Result<Value> {
    ensure!(
        expected.len() == actual.len() && !actual.is_empty(),
        "comparison length mismatch"
    );
    ensure!(
        expected.iter().chain(actual).all(|v| v.is_finite()),
        "non-finite output"
    );
    let mut signal = 0.0;
    let mut actual_signal = 0.0;
    let mut product = 0.0;
    let mut error = 0.0;
    let mut absolute_error = 0.0;
    let mut max = 0.0_f64;
    for (a, b) in expected.iter().zip(actual) {
        let (a, b) = (f64::from(*a), f64::from(*b));
        signal += a * a;
        actual_signal += b * b;
        product += a * b;
        error += (a - b) * (a - b);
        absolute_error += (a - b).abs();
        max = max.max((a - b).abs());
    }
    Ok(
        json!({"elements":actual.len(),"max_abs":max,"mean_abs":absolute_error/actual.len() as f64,
        "rmse":(error/actual.len() as f64).sqrt(),
        "cosine":if signal == 0.0 && actual_signal == 0.0 {1.0} else if signal == 0.0 || actual_signal == 0.0 {0.0} else {(product/(signal*actual_signal).sqrt()).clamp(-1.0,1.0)},
        "snr_db":if error>0.0 {Some(10.0*(signal/error).log10())} else {None},
        "bitwise_equal":expected.iter().zip(actual).all(|(a,b)|a.to_bits()==b.to_bits())}),
    )
}
fn save(path: PathBuf, values: &[f32]) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    for value in values {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn save_stage(
    directory: &std::path::Path,
    name: &str,
    shape: &[usize],
    values: &[f32],
) -> Result<Value> {
    let filename = format!("{name}.f32le");
    let path = directory.join(&filename);
    save(path.clone(), values)?;
    Ok(
        json!({"name": name, "file": filename, "shape": shape, "elements": values.len(),
        "sha256": irodori_tts_burn::sha256_file(&path)?}),
    )
}
fn main() -> Result<()> {
    backend_config::initialize_cli_tracing("warn")?;
    let args = Args::parse();
    #[cfg(feature = "profile")]
    if let Some(path) = args.route_profile.as_ref() {
        ensure!(
            !args.portable,
            "route-profile and portable are mutually exclusive"
        );
        let profile = irodori_tts_burn::UnsealedRouteProfile::load(path)?;
        irodori_tts_burn::install_unsealed_route_profile(&profile)?;
    }
    if args.portable {
        irodori_tts_burn::install_portable_route_table(
            irodori_tts_burn::RouteCacheMissReason::PortableRequested,
        )?;
    }
    ensure!(args.stream_repeat <= 20, "stream-repeat must be at most 20");
    ensure!(
        args.stream_window_frames > 0,
        "stream-window-frames must be positive"
    );
    #[cfg(feature = "streaming-graphs")]
    ensure!(
        args.stream_graph_entries.is_none() || args.stream_repeat > 0,
        "stream-graph-entries requires a positive stream-repeat"
    );
    ensure!((1..=20).contains(&args.repeat), "repeat must be 1..20");
    ensure!((1..=20).contains(&args.warmups), "warmups must be 1..20");
    ensure!(!args.output.exists(), "output directory must be new");
    let bytes = fs::read(&args.fixture)?;
    let file = SafeTensors::deserialize(&bytes)?;
    let (_, header) = SafeTensors::read_metadata(&bytes)?;
    let metadata = header
        .metadata()
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing metadata"))?;
    let checkpoint_sha256 = irodori_tts_burn::sha256_file(&args.checkpoint)?;
    let source_checkpoint_sha256 = if args
        .checkpoint
        .extension()
        .is_some_and(|extension| extension == "wgslpack")
    {
        #[cfg(feature = "packed-weights")]
        {
            irodori_tts_burn::weights::packed::source_checkpoint_sha256(&args.checkpoint)?
        }
        #[cfg(not(feature = "packed-weights"))]
        anyhow::bail!("WGSLPack validation requires the packed-weights feature");
    } else {
        checkpoint_sha256.clone()
    };
    ensure!(
        metadata["checkpoint_sha256"] == source_checkpoint_sha256,
        "source checkpoint identity mismatch"
    );
    let initialization_started = Instant::now();
    let selected = args
        .adapter
        .map_or(args.device, backend_config::DeviceSelector::AdapterIndex)
        .device();
    backend_config::configure_cubecl_persistent_cache_for_precision_with_record(
        backend_config::default_cubecl_cache_root()?
            .join(backend_config::wgpu_cache_namespace(&selected)?),
        args.precision,
        args.autotune_record,
    )?;
    if args.autotune {
        use burn::backend::wgpu::{RuntimeOptions, graphics::AutoGraphicsApi, init_setup};
        let setup = init_setup::<AutoGraphicsApi>(&selected, RuntimeOptions::default());
        let info = setup.adapter.get_info();
        irodori_tts_burn::install_autotuned_route_table(
            info.vendor,
            &format!("{:?}", info.backend),
            std::env::consts::OS,
        )?;
    }
    let device = backend_config::wgpu_device_with_precision(&selected, args.precision)?;
    device.sync()?;
    let device_seconds = initialization_started.elapsed().as_secs_f64();
    let model_started = Instant::now();
    let loaded = InferenceBuilder::new(device.clone())
        .with_timestep_condition_cache(match args.cache_policy {
            CachePolicy::Disabled => TimestepConditionCachePolicy::Disabled,
            CachePolicy::Condition => TimestepConditionCachePolicy::ConditionOnly,
            CachePolicy::Full => TimestepConditionCachePolicy::ConditionAndAdaLn,
        })
        .load_weights_with_float_dtype(&args.checkpoint, args.precision.tensor_dtype())?;
    let steps = metadata["steps"].parse()?;
    let mut params =
        if loaded.model_config().flow_parameterization == FlowParameterization::Meanflow {
            SamplerParams::meanflow(steps)
        } else {
            let mut params = SamplerParams {
                num_steps: steps,
                ..Default::default()
            };
            params.guidance.scale_speaker = if metadata
                .get("reference")
                .is_some_and(|value| !value.is_empty())
            {
                5.0
            } else {
                0.0
            };
            params.guidance.scale_caption = if metadata["caption"].is_empty() {
                0.0
            } else {
                3.0
            };
            params
        };
    params.state_precision = args.sampler_state;
    let engine = loaded.with_sampling(params).build_wgsl();
    device.sync()?;
    let model_seconds = model_started.elapsed().as_secs_f64();
    let noise: Tensor<3> = float_tensor(
        &file,
        "initial_noise",
        &device,
        args.sampler_state.dtype(args.precision.tensor_dtype()),
    )?;
    let request = SamplingRequest {
        text_ids: ints(&file, "text_ids", &device)?,
        text_mask: mask(&file, "text_mask", &device)?,
        ref_latent: Some(float_tensor(
            &file,
            "ref_latent",
            &device,
            args.precision.tensor_dtype(),
        )?),
        ref_mask: Some(mask(&file, "ref_mask", &device)?),
        caption_ids: Some(ints(&file, "caption_ids", &device)?),
        caption_mask: Some(mask(&file, "caption_mask", &device)?),
        sequence_length: noise.dims()[1],
        initial_noise: Some(noise),
    };
    let codec_started = Instant::now();
    let mut codec = load_decoder_with_float_dtype(&args.codec, &device, DType::F32)?;
    codec.prepare_for_wgsl();
    device.sync()?;
    let codec_seconds = codec_started.elapsed().as_secs_f64();
    let ready_seconds = initialization_started.elapsed().as_secs_f64();
    let mut times = Vec::new();
    let mut warmup_times = Vec::new();
    let mut last = None;
    let mut waveform = Vec::new();
    for index in 0..args.warmups + args.repeat {
        let start = Instant::now();
        let latent = engine.sample(request.clone())?;
        let values = read(latent.clone())?;
        let rf = start.elapsed().as_secs_f64();
        waveform = read(codec.decode_wgsl(latent.clone().cast(DType::F32)))?;
        let timing = json!({"rf_seconds":rf,"audio_seconds":start.elapsed().as_secs_f64()});
        if index >= args.warmups {
            times.push(timing);
        } else {
            warmup_times.push(timing);
        }
        last = Some((latent, values));
    }
    let (latent, values) = last.expect("positive repeat");
    let latent_shape = latent.dims();
    let mut streamed = Vec::new();
    let stream = codec.decode_stream(
        latent.clone(),
        None,
        DecodeWindow {
            first_frames: 12,
            subsequent_frames: 17,
        },
        |chunk| {
            streamed.extend_from_slice(chunk);
            Ok(std::ops::ControlFlow::Continue(()))
        },
    )?;
    let mut trimmed = Vec::new();
    let limit = waveform.len() - 317;
    codec.decode_stream(latent, Some(limit), DecodeWindow::default(), |chunk| {
        trimmed.extend_from_slice(chunk);
        Ok(std::ops::ControlFlow::Continue(()))
    })?;
    let (expected_latent, _) = floats(&file, "expected_latent")?;
    let (expected_audio, _) = floats(&file, "expected_audio")?;
    let mut report = BTreeMap::from([
        (
            "device",
            json!(
                args.adapter
                    .map_or(args.device, backend_config::DeviceSelector::AdapterIndex)
            ),
        ),
        ("autotune", json!(args.autotune)),
        (
            "kernel_environment",
            json!(args.precision.environment_name()),
        ),
        ("precision", json!(args.precision)),
        ("warmups", json!(args.warmups)),
        (
            "startup",
            json!({"device_seconds": device_seconds, "model_seconds": model_seconds,
            "codec_seconds": codec_seconds, "ready_seconds": ready_seconds,
            "warmup_timings": warmup_times}),
        ),
        ("fixture", json!(args.fixture)),
        ("checkpoint_sha256", json!(checkpoint_sha256)),
        (
            "declared_source_checkpoint_sha256",
            json!(source_checkpoint_sha256),
        ),
        (
            "cache_policy",
            json!(match args.cache_policy {
                CachePolicy::Disabled => "disabled",
                CachePolicy::Condition => "condition",
                CachePolicy::Full => "full",
            }),
        ),
        ("timings", json!(times)),
        ("latent", compare(&expected_latent, &values)?),
        ("audio", compare(&expected_audio, &waveform)?),
        ("streamed_vs_full", compare(&waveform, &streamed)?),
        ("trimmed_vs_full", compare(&waveform[..limit], &trimmed)?),
        (
            "stream_first_audio_seconds",
            json!(stream.first_audio_seconds),
        ),
    ]);
    #[cfg(feature = "profile")]
    if args.device_profile {
        let (profiled_latent, rf) =
            irodori_tts_burn::profile_rf_device(|| engine.sample(request.clone()))?;
        let latent_comparison = compare(&values, &read(profiled_latent.clone())?)?;
        let (audio, codec_stages) =
            codec.decode_wgsl_device_profiled(profiled_latent.cast(DType::F32))?;
        let audio_comparison = compare(&waveform, &read(audio)?)?;
        ensure!(
            latent_comparison["bitwise_equal"] == true && audio_comparison["bitwise_equal"] == true,
            "device profiling changed the ordinary inference output"
        );
        let codec_stages = codec_stages.iter().map(|stage| {
            json!({"stage": stage.label, "duration_ns": stage.duration.as_nanos(),
                "source": match stage.source {
                    irodori_tts_burn::codec::CodecTimingSource::DeviceTimestamp => "device_timestamp",
                    irodori_tts_burn::codec::CodecTimingSource::SynchronizedSystemClock => "synchronized_system_clock",
                }})
        }).collect::<Vec<_>>();
        report.insert(
            "device_profile",
            json!({
                "diagnostic_only": true, "latency_results_valid": false,
                "rf": rf, "codec": codec_stages,
                "latent_vs_ordinary": latent_comparison, "audio_vs_ordinary": audio_comparison,
            }),
        );
    }
    if args.stream_repeat > 0 {
        let window = DecodeWindow {
            first_frames: 12,
            subsequent_frames: args.stream_window_frames,
        };
        #[cfg(feature = "streaming-graphs")]
        let mut graph_codec = args
            .stream_graph_entries
            .map(|entries| {
                let max_frames = args
                    .stream_window_frames
                    .max(window.first_frames)
                    .checked_add(20)
                    .and_then(std::num::NonZeroUsize::new)
                    .ok_or_else(|| anyhow::anyhow!("streaming graph frame limit overflow"))?;
                Ok::<_, anyhow::Error>(irodori_tts_burn::codec::StreamingDacVaeDecoder::new(
                    codec.clone(),
                    irodori_tts_burn::codec::DecoderGraphLimits::new(entries, max_frames)?,
                ))
            })
            .transpose()?;
        let mut timings = Vec::new();
        let mut cold_timings = Vec::new();
        for index in 0..args.warmups + args.stream_repeat {
            let started = Instant::now();
            let latent = engine.sample(request.clone())?;
            let mut first_audio = None;
            let mut audio = Vec::new();
            let sink = |chunk: &[f32]| {
                first_audio.get_or_insert_with(|| started.elapsed().as_secs_f64());
                audio.extend_from_slice(chunk);
                Ok::<_, irodori_tts_burn::IrodoriError>(std::ops::ControlFlow::Continue(()))
            };
            #[cfg(feature = "streaming-graphs")]
            match graph_codec.as_mut() {
                Some(captured) => captured.decode_stream(latent, None, window, sink)?,
                None => codec.decode_stream(latent, None, window, sink)?,
            };
            #[cfg(not(feature = "streaming-graphs"))]
            codec.decode_stream(latent, None, window, sink)?;
            let seconds = started.elapsed().as_secs_f64();
            ensure!(
                compare(&waveform, &audio)?["bitwise_equal"] == true,
                "streaming request changed audio"
            );
            if index >= args.warmups {
                timings.push(json!({"first_audio_seconds": first_audio, "audio_seconds": seconds}));
            } else {
                cold_timings
                    .push(json!({"first_audio_seconds": first_audio, "audio_seconds": seconds}));
            }
        }
        report.insert("streaming_requests", json!(timings));
        report.insert("streaming_warmups", json!(cold_timings));
        let primitive = cubecl::Device::from(selected.clone());
        let usage = primitive
            .client()
            .memory_report(cubecl::MemoryScope::Device)
            .usage();
        report.insert("streaming_memory", json!({"device_bytes_in_use": usage.bytes_in_use,
            "device_bytes_reserved": usage.bytes_reserved,
            "memory_boundary": "whole device allocator after completed streaming replay; not peak"}));
        #[cfg(feature = "streaming-graphs")]
        if let Some(captured) = graph_codec {
            report.insert(
                "streaming_graph_cache",
                json!({"entries": captured.cached_graphs(),
                "limit": args.stream_graph_entries, "window_frames": args.stream_window_frames}),
            );
        }
    }
    if args.graph_replay {
        let example = request
            .initial_noise
            .clone()
            .expect("fixture supplies noise");
        let other = example.clone() * 0.5;
        let mut other_request = request.clone();
        other_request.initial_noise = Some(other.clone());
        let other_latent = engine.sample(other_request)?;
        let other_values = read(other_latent.clone())?;
        let other_audio = read(codec.decode_wgsl(other_latent.cast(DType::F32)))?;
        let start = Instant::now();
        let mut captured = engine.capture_sampling(request.clone())?;
        let sampling_capture_seconds = start.elapsed().as_secs_f64();
        let mut replay_times = Vec::new();
        for index in 0..args.warmups + args.repeat {
            let start = Instant::now();
            let actual = captured.sample_to_cpu_f32(example.clone())?;
            let seconds = start.elapsed().as_secs_f64();
            ensure!(
                compare(&values, &actual)?["bitwise_equal"] == true,
                "captured sampler changed the trajectory"
            );
            if index >= args.warmups {
                replay_times.push(seconds);
            }
        }
        ensure!(
            compare(&other_values, &captured.sample_to_cpu_f32(other.clone())?)?["bitwise_equal"]
                == true,
            "captured sampler retained stale noise"
        );
        drop(captured);
        let start = Instant::now();
        let mut captured = engine.capture_synthesis(request.clone(), &codec)?;
        let synthesis_capture_seconds = start.elapsed().as_secs_f64();
        let mut audio_times = Vec::new();
        for index in 0..args.warmups + args.repeat {
            let start = Instant::now();
            let actual = captured.synthesize_to_cpu_f32(example.clone())?;
            let seconds = start.elapsed().as_secs_f64();
            ensure!(
                compare(&waveform, &actual)?["bitwise_equal"] == true,
                "captured synthesis changed audio"
            );
            if index >= args.warmups {
                audio_times.push(seconds);
            }
        }
        ensure!(
            compare(&other_audio, &captured.synthesize_to_cpu_f32(other)?)?["bitwise_equal"]
                == true,
            "captured synthesis retained stale noise"
        );
        report.insert("graph_replay", json!({"sampling_capture_seconds": sampling_capture_seconds,
            "synthesis_capture_seconds": synthesis_capture_seconds, "sampling_seconds": replay_times,
            "audio_seconds": audio_times, "both_noises_bitwise_equal": true}));
    }
    if args.condition_graph {
        let mut alternate = request.clone();
        alternate.text_ids = alternate.text_ids.flip([1]);
        alternate.ref_latent = alternate.ref_latent.map(|latent| latent * 0.5);
        alternate.caption_ids = alternate.caption_ids.map(|ids| ids.flip([1]));
        alternate.initial_noise = alternate.initial_noise.map(|noise| noise * 0.75);
        let requests = [request.clone(), alternate];
        let prepared = requests
            .iter()
            .cloned()
            .map(|request| engine.prepare_sampling_request(request))
            .collect::<irodori_tts_burn::Result<Vec<_>>>()?;
        let conditions = prepared
            .iter()
            .map(|request| engine.encode_prepared_conditions(request))
            .collect::<irodori_tts_burn::Result<Vec<_>>>()?;
        let expected = prepared
            .iter()
            .cloned()
            .map(|request| read(engine.sample_prepared(request)?))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            compare(&expected[0], &expected[1])?["bitwise_equal"] == false,
            "alternate request must change output"
        );
        let start = Instant::now();
        let mut graph = engine.capture_conditioned_sampling(request.clone())?;
        let capture_seconds = start.elapsed().as_secs_f64();
        let mut measurements = Vec::new();
        for session in 0..args.warmups + args.repeat {
            for index in 0..2 {
                // Include condition encoding and mask validation in both paths.
                // Preparation is shared and excluded, as in sample_prepared.
                for captured in if session % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let started = Instant::now();
                    let actual = if captured {
                        let condition = engine.encode_prepared_conditions(&prepared[index])?;
                        graph.sample_to_cpu_f32(
                            requests[index]
                                .initial_noise
                                .clone()
                                .expect("fixture noise"),
                            &condition,
                        )?
                    } else {
                        read(engine.sample_prepared(prepared[index].clone())?)?
                    };
                    let seconds = started.elapsed().as_secs_f64();
                    ensure!(
                        compare(&expected[index], &actual)?["bitwise_equal"] == true,
                        "condition graph changed output or retained stale conditions"
                    );
                    if session >= args.warmups {
                        measurements.push(json!({"session":session-args.warmups, "request":index, "captured":captured, "seconds":seconds}));
                    }
                }
            }
        }
        // Reuse after switching back must refresh every input again.
        ensure!(
            compare(
                &expected[0],
                &graph.sample_to_cpu_f32(
                    requests[0].initial_noise.clone().expect("fixture noise"),
                    &conditions[0]
                )?
            )?["bitwise_equal"]
                == true
        );
        report.insert("condition_graph", json!({"capture_seconds":capture_seconds,"measurements":measurements,"changed_conditions_bitwise_equal":true}));
    }
    fs::create_dir_all(&args.output)?;
    #[cfg(feature = "profile")]
    if args.dump_text_layers {
        let standard = engine
            .encode_prepared_conditions(&engine.prepare_sampling_request(request.clone())?)?;
        let trace = engine.trace_text_encoder(request.clone())?;
        let final_state = trace
            .tensors
            .last()
            .ok_or_else(|| anyhow::anyhow!("empty text trace"))?;
        ensure!(
            final_state.name == "text_state"
                && compare(&read(standard.text_state)?, &final_state.values)?["bitwise_equal"]
                    == true,
            "text layer tracing changed the production encoder output"
        );
        let directory = args.output.join("text-layers");
        fs::create_dir(&directory)?;
        let tensors = trace
            .tensors
            .iter()
            .map(|tensor| save_stage(&directory, &tensor.name, &tensor.shape, &tensor.values))
            .collect::<Result<Vec<_>>>()?;
        fs::write(
            directory.join("trace.json"),
            serde_json::to_vec_pretty(&json!({
                "input_shape": trace.input_shape, "input_ids": trace.input_ids,
                "attention_mask": trace.attention_mask, "tensors": tensors,
                "precision": args.precision, "checkpoint": args.checkpoint,
            }))?,
        )?;
    }
    let mut stages = Vec::new();
    if args.dump_stages {
        let noise = request.initial_noise.clone().expect("fixture has noise");
        stages.push(save_stage(
            &args.output,
            "input-noise",
            &noise.dims(),
            &read(noise)?,
        )?);
    }
    let (traced_latent, work, trace) = engine.sample_with_diagnostic_trace(request)?;
    if args.dump_stages {
        ensure!(
            compare(&values, &read(traced_latent)?)?["bitwise_equal"] == true,
            "stage tracing changed the ordinary latent output"
        );
        let condition = &trace
            .forwards
            .first()
            .ok_or_else(|| anyhow::anyhow!("diagnostic trace contains no forward"))?
            .condition;
        stages.push(save_stage(
            &args.output,
            "text-state",
            &condition.text_state.dims(),
            &read(condition.text_state.clone())?,
        )?);
    }
    report.insert("sampler_work", serde_json::to_value(work)?);
    let mut forwards = Vec::new();
    for forward in trace.forwards {
        let key = format!("forward_{:03}", forward.ordinal);
        let (expected, _) = floats(&file, &key)?;
        let shape = forward.output.dims();
        let actual = read(forward.output)?;
        if args.dump_stages {
            stages.push(save_stage(&args.output, &key, &shape, &actual)?);
        }
        forwards.push(compare(&expected, &actual)?);
    }
    report.insert("forwards", json!(forwards));
    if args.dump_stages {
        stages.push(save_stage(&args.output, "latent", &latent_shape, &values)?);
        stages.push(save_stage(
            &args.output,
            "audio",
            &[1, 1, waveform.len()],
            &waveform,
        )?);
        report.insert(
            "stage_capture",
            json!({"schema_version": 2,
            "text_state_boundary": "exact first-forward condition, including compacted tokens and CFG rows",
            "fixture_sha256": irodori_tts_burn::sha256_file(&args.fixture)?,
            "codec_sha256": irodori_tts_burn::sha256_file(&args.codec)?, "tensors": stages}),
        );
    } else {
        save(args.output.join("latent.f32le"), &values)?;
        save(args.output.join("audio.f32le"), &waveform)?;
    }
    fs::write(
        args.output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
