//! Isolated stage profiler for the exact production WGSL DACVAE decoder path.

use std::{
    collections::BTreeMap,
    fs,
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
    tensor::{FloatDType, Tensor, TensorData},
};
use clap::{Parser, ValueEnum};
use cubecl::prelude::Runtime;
use irodori_tts_burn::{
    backend_config::{WgpuFloatPrecision, wgpu_device_with_precision},
    codec::{CodecK7Algorithm, CodecTimingSource, load_codec},
    validation::AudioMetrics,
};
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};

type WgpuRt = WgpuRuntime<AutoCompiler>;

#[derive(Debug, Parser)]
#[command(about = "Profile exact production WGSL codec stages from a precision oracle")]
struct Args {
    /// WGPU storage precision used by the codec and handwritten kernels.
    #[arg(long, value_enum, default_value = "fp32")]
    precision: WgpuFloatPrecision,

    /// Native dtype stored in the oracle. Defaults to `--precision`.
    /// This permits an F16 execution to be checked against an independently
    /// pinned F32 oracle without rewriting that source artifact.
    #[arg(long, value_enum)]
    fixture_precision: Option<WgpuFloatPrecision>,

    /// Precision-oracle fixture containing the exact final latent.
    #[arg(long)]
    fixture: PathBuf,

    /// Required out-of-band SHA-256 for the fixture.
    #[arg(long)]
    fixture_sha256: String,

    /// Rust-converted Semantic-DACVAE weights.
    #[arg(long)]
    codec_weights: PathBuf,

    /// Explicit WGPU discrete-adapter enumeration index.
    #[arg(long, default_value_t = 0)]
    adapter_index: usize,

    /// Untimed production-path warmups.
    #[arg(long, default_value_t = 2)]
    warmup: usize,

    /// Timed unchanged production-path repetitions.
    #[arg(long, default_value_t = 10)]
    repeats: usize,

    /// Timed stage profiling repetitions.
    #[arg(long, default_value_t = 5)]
    profile_repeats: usize,

    /// Stage measurement method. Device timestamps avoid per-stage waits.
    #[arg(long, value_enum, default_value_t = StageProfileMethod::Device)]
    stage_profile_method: StageProfileMethod,

    /// k=7 implementation used by the timed decode and stage profiler.
    #[arg(long, value_enum, default_value_t = K7ProfileAlgorithm::Production)]
    k7_algorithm: K7ProfileAlgorithm,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum StageProfileMethod {
    #[default]
    Device,
    Synchronized,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum K7ProfileAlgorithm {
    #[default]
    Production,
    PackedResidue,
    ImplicitGemm,
}

impl From<K7ProfileAlgorithm> for CodecK7Algorithm {
    fn from(value: K7ProfileAlgorithm) -> Self {
        match value {
            K7ProfileAlgorithm::Production => Self::AccuracyApproved,
            K7ProfileAlgorithm::PackedResidue => Self::PackedResidue,
            K7ProfileAlgorithm::ImplicitGemm => Self::CubeClImplicitGemm,
        }
    }
}

#[derive(Clone, Default)]
struct WgpuErrorMonitor {
    errors: Arc<Mutex<Vec<String>>>,
}

impl WgpuErrorMonitor {
    fn callback_sink(&self) -> Arc<Mutex<Vec<String>>> {
        Arc::clone(&self.errors)
    }

    fn check(&self, stage: &str) -> Result<()> {
        let errors = self
            .errors
            .lock()
            .map_err(|_| anyhow::anyhow!("WGPU error monitor lock poisoned after {stage}"))?;
        ensure!(errors.is_empty(), "WGPU errors after {stage}: {errors:?}");
        Ok(())
    }
}

fn initialize_wgpu(adapter_index: usize) -> (WgpuDevice, WgpuErrorMonitor) {
    let device = WgpuDevice::DiscreteGpu(adapter_index);
    let setup = init_setup::<AutoGraphicsApi>(
        &device,
        RuntimeOptions {
            tasks_max: 32,
            memory_config: MemoryConfiguration::SubSlices,
        },
    );
    let monitor = WgpuErrorMonitor::default();
    let callback_errors = monitor.callback_sink();
    setup.device.on_uncaptured_error(Arc::new(move |error| {
        if let Ok(mut errors) = callback_errors.lock() {
            errors.push(error.to_string());
        }
    }));
    let info = setup.adapter.get_info();
    println!(
        "wgpu_adapter: index={adapter_index} name={:?} backend={:?} device_type={:?} tasks_max=32 memory_config=sub-slices",
        info.name, info.backend, info.device_type
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

fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    ensure!(
        expected.len() == 64,
        "fixture SHA-256 must have 64 hex digits"
    );
    ensure!(
        expected.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "fixture SHA-256 contains a non-hex digit"
    );
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read fixture for hashing: {}", path.display()))?;
    let actual = format!("{:x}", Sha256::digest(bytes));
    ensure!(
        actual == expected.to_ascii_lowercase(),
        "fixture SHA-256 mismatch: got {actual}, expected {expected}"
    );
    println!("sha256: precision_fixture={actual} path={}", path.display());
    Ok(())
}

fn read_float_tensor(
    tensors: &SafeTensors<'_>,
    key: &str,
    precision: WgpuFloatPrecision,
) -> Result<(Vec<usize>, Vec<f32>)> {
    let view = tensors
        .tensor(key)
        .with_context(|| format!("fixture tensor {key:?} is missing"))?;
    let expected_dtype = match precision {
        WgpuFloatPrecision::Fp32 => Dtype::F32,
        WgpuFloatPrecision::Fp16 => Dtype::F16,
    };
    ensure!(
        view.dtype() == expected_dtype,
        "fixture tensor {key:?} has dtype {:?}, expected {expected_dtype:?}",
        view.dtype()
    );
    let shape = view.shape().to_vec();
    let values = match precision {
        WgpuFloatPrecision::Fp32 => view
            .data()
            .chunks_exact(size_of::<f32>())
            .map(|chunk| {
                let bytes: [u8; size_of::<f32>()] = chunk
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("invalid f32 bytes in {key:?}"))?;
                Ok(f32::from_le_bytes(bytes))
            })
            .collect::<Result<Vec<_>>>()?,
        WgpuFloatPrecision::Fp16 => view
            .data()
            .chunks_exact(size_of::<half::f16>())
            .map(|chunk| {
                let bytes: [u8; size_of::<half::f16>()] = chunk
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("invalid f16 bytes in {key:?}"))?;
                Ok(half::f16::from_le_bytes(bytes).to_f32())
            })
            .collect::<Result<Vec<_>>>()?,
    };
    Ok((shape, values))
}

fn load_oracle_tensors(
    path: &Path,
    precision: WgpuFloatPrecision,
) -> Result<(Vec<f32>, usize, Vec<f32>)> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read precision fixture {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("malformed precision fixture {}", path.display()))?;
    let (latent_shape, latent) = read_float_tensor(&tensors, "final_patched_latent", precision)?;
    let (waveform_shape, waveform) =
        read_float_tensor(&tensors, "raw_decoded_waveform", precision)?;
    ensure!(
        latent_shape.len() == 3
            && latent_shape[0] == 1
            && latent_shape[1] > 0
            && latent_shape[2] == 32,
        "final_patched_latent shape {latent_shape:?} must be [1, positive_steps, 32]"
    );
    ensure!(
        waveform_shape.len() == 2 && waveform_shape[0] == 1 && waveform_shape[1] > 0,
        "raw_decoded_waveform shape {waveform_shape:?} must be [1, positive_samples]"
    );
    ensure!(
        latent
            .iter()
            .chain(&waveform)
            .all(|value| value.is_finite()),
        "oracle tensors contain non-finite values"
    );
    Ok((latent, latent_shape[1], waveform))
}

fn sha256_f32_le(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    values
        .iter()
        .for_each(|value| hasher.update(value.to_bits().to_le_bytes()));
    format!("{:x}", hasher.finalize())
}

fn waveform_gate(
    reference: &[f32],
    actual: &[f32],
    label: &str,
    precision: WgpuFloatPrecision,
) -> Result<()> {
    let metrics = AudioMetrics::compare(reference, actual)?;
    println!(
        "{label}: count={} max_abs={:.9e} mean_abs={:.9e} rmse={:.9e} snr_db={:.6} cosine={:.12}",
        metrics.sample_count,
        metrics.max_abs_error,
        metrics.mean_abs_error,
        metrics.root_mean_square_error,
        metrics.signal_to_noise_db,
        metrics.cosine_similarity
    );
    let (max_abs, mean_abs, rmse, snr, cosine) = match precision {
        WgpuFloatPrecision::Fp32 => (0.00015, 0.000005, 0.00001, 85.0, 0.99999999),
        WgpuFloatPrecision::Fp16 => (0.005, 0.0005, 0.001, 50.0, 0.99999),
    };
    ensure!(
        metrics.max_abs_error <= max_abs,
        "{label} max_abs gate failed"
    );
    ensure!(
        metrics.mean_abs_error <= mean_abs,
        "{label} mean_abs gate failed"
    );
    ensure!(
        metrics.root_mean_square_error <= rmse,
        "{label} RMSE gate failed"
    );
    ensure!(metrics.signal_to_noise_db >= snr, "{label} SNR gate failed");
    ensure!(
        metrics.cosine_similarity >= cosine,
        "{label} cosine gate failed"
    );
    Ok(())
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    } else {
        sorted[middle]
    }
}

fn print_summary(label: &str, values_ms: &[f64]) {
    let minimum = values_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "timing_summary stage={label} median_ms={:.6} range_ms=[{minimum:.6},{maximum:.6}] samples={}",
        median(values_ms),
        values_ms.len()
    );
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.warmup > 0, "--warmup must be positive");
    ensure!(args.repeats > 0, "--repeats must be positive");
    ensure!(
        args.profile_repeats > 0,
        "--profile-repeats must be positive"
    );
    ensure!(
        args.stage_profile_method == StageProfileMethod::Device
            || args.k7_algorithm == K7ProfileAlgorithm::Production,
        "explicit k7 algorithm comparison requires --stage-profile-method device"
    );
    verify_sha256(&args.fixture, &args.fixture_sha256)?;
    let fixture_precision = args.fixture_precision.unwrap_or(args.precision);
    let (latent_values, latent_steps, expected_waveform) =
        load_oracle_tensors(&args.fixture, fixture_precision)?;
    println!(
        "profile_shape latent_steps={latent_steps} waveform_samples={} execution_precision={} fixture_precision={}",
        expected_waveform.len(),
        args.precision.label(),
        fixture_precision.label()
    );
    let (device, monitor) = initialize_wgpu(args.adapter_index);
    let tensor_device = wgpu_device_with_precision(&device, args.precision)?;

    let mut codec = load_codec(&args.codec_weights, &tensor_device)
        .with_context(|| format!("failed to load codec {}", args.codec_weights.display()))?;
    codec.prepare_decoder_for_wgsl_with_k7_algorithm(args.k7_algorithm.into());
    synchronize_and_check_wgpu(&device, &monitor, "codec load and preparation")?;
    let latent = Tensor::<3>::from_data(
        TensorData::new(latent_values, [1, latent_steps, 32]),
        &tensor_device,
    );

    let decode_selected = |latent| match args.k7_algorithm {
        K7ProfileAlgorithm::Production => codec.decode_wgsl(latent),
        K7ProfileAlgorithm::PackedResidue | K7ProfileAlgorithm::ImplicitGemm => {
            codec.decode_wgsl_with_k7_algorithm(latent, args.k7_algorithm.into())
        }
    };

    for warmup in 1..=args.warmup {
        let output = decode_selected(latent.clone());
        synchronize_and_check_wgpu(&device, &monitor, &format!("warmup {warmup}"))?;
        drop(output);
    }

    let mut production_device_ms = Vec::with_capacity(args.repeats);
    let mut production_readback_ms = Vec::with_capacity(args.repeats);
    let mut production_hash = None;
    for repetition in 1..=args.repeats {
        let started = Instant::now();
        let output = decode_selected(latent.clone());
        synchronize_and_check_wgpu(
            &device,
            &monitor,
            &format!("production device completion {repetition}"),
        )?;
        let device_complete_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let values = output
            .cast(FloatDType::F32)
            .into_data()
            .to_vec::<f32>()
            .with_context(|| format!("failed production readback {repetition}"))?;
        synchronize_and_check_wgpu(
            &device,
            &monitor,
            &format!("production repetition {repetition}"),
        )?;
        let readback_complete_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let hash = sha256_f32_le(&values);
        if let Some(expected_hash) = &production_hash {
            ensure!(
                &hash == expected_hash,
                "production waveform changed at repetition {repetition}"
            );
        } else {
            production_hash = Some(hash.clone());
        }
        waveform_gate(
            &expected_waveform,
            &values,
            &format!("production_waveform[{repetition}]"),
            args.precision,
        )?;
        println!(
            "production_repeat={repetition}/{} decode_device_complete_ms={device_complete_ms:.6} decode_and_readback_ms={readback_complete_ms:.6} sha256={hash}",
            args.repeats
        );
        production_device_ms.push(device_complete_ms);
        production_readback_ms.push(readback_complete_ms);
    }
    print_summary("production_decode_device_complete", &production_device_ms);
    print_summary("production_decode_and_readback", &production_readback_ms);

    if args.k7_algorithm == K7ProfileAlgorithm::ImplicitGemm {
        for warmup in 1..=args.warmup {
            let (output, _) = codec.decode_wgsl_device_profiled_with_k7_algorithm(
                latent.clone(),
                args.k7_algorithm.into(),
            )?;
            let values = output
                .cast(FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .with_context(|| format!("failed implicit-gemm warmup readback {warmup}"))?;
            synchronize_and_check_wgpu(
                &device,
                &monitor,
                &format!("implicit-gemm warmup {warmup}"),
            )?;
            waveform_gate(
                &expected_waveform,
                &values,
                &format!("implicit_gemm_warmup[{warmup}]"),
                args.precision,
            )?;
            println!(
                "candidate_warmup={warmup}/{} algorithm=implicit-gemm sha256={}",
                args.warmup,
                sha256_f32_le(&values)
            );
        }
    }

    let mut stage_samples: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    let mut profiled_total_ms = Vec::with_capacity(args.profile_repeats);
    for repetition in 1..=args.profile_repeats {
        let started = Instant::now();
        let (output, timings) = match args.stage_profile_method {
            StageProfileMethod::Device => codec.decode_wgsl_device_profiled_with_k7_algorithm(
                latent.clone(),
                args.k7_algorithm.into(),
            )?,
            StageProfileMethod::Synchronized => codec
                .decode_wgsl_profiled(latent.clone(), |stage| {
                    synchronize_and_check_wgpu(&device, &monitor, stage)
                })?,
        };
        let device_complete_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let values = output
            .cast(FloatDType::F32)
            .into_data()
            .to_vec::<f32>()
            .with_context(|| format!("failed profiled readback {repetition}"))?;
        synchronize_and_check_wgpu(
            &device,
            &monitor,
            &format!("profiled repetition {repetition}"),
        )?;
        waveform_gate(
            &expected_waveform,
            &values,
            &format!("profiled_waveform[{repetition}]"),
            args.precision,
        )?;
        let profiled_hash = sha256_f32_le(&values);
        if args.k7_algorithm == K7ProfileAlgorithm::Production {
            ensure!(
                production_hash.as_deref() == Some(&profiled_hash),
                "profiled waveform differs bitwise from production"
            );
        }
        for timing in timings {
            let source = match timing.source {
                CodecTimingSource::DeviceTimestamp => "device-timestamp",
                CodecTimingSource::SynchronizedSystemClock => "synchronized-system-clock",
            };
            println!(
                "stage_profile repetition={repetition} stage={} source={source} duration_ms={:.6}",
                timing.label,
                timing.duration.as_secs_f64() * 1_000.0
            );
            stage_samples
                .entry(timing.label)
                .or_default()
                .push(timing.duration.as_secs_f64() * 1_000.0);
        }
        let readback_complete_ms = started.elapsed().as_secs_f64() * 1_000.0;
        println!(
            "profiled_repeat={repetition}/{} method={:?} k7_algorithm={:?} profile_wall_complete_ms={device_complete_ms:.6} profile_and_readback_ms={readback_complete_ms:.6} sha256={profiled_hash}",
            args.profile_repeats, args.stage_profile_method, args.k7_algorithm
        );
        profiled_total_ms.push(device_complete_ms);
    }

    let mut summaries: Vec<_> = stage_samples
        .iter()
        .map(|(&label, values)| (label, median(values), values))
        .collect();
    summaries.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));
    for (label, _, values) in summaries {
        print_summary(label, values);
    }
    print_summary("profiled_wall_complete", &profiled_total_ms);
    monitor.check("profile completion")?;
    println!("wgpu_uncaptured_errors=0");
    Ok(())
}
