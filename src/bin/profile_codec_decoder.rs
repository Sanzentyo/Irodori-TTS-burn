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
    tensor::{Tensor, TensorData},
};
use clap::Parser;
use cubecl::prelude::Runtime;
use irodori_tts_burn::{codec::load_codec, validation::AudioMetrics};
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};

type WgpuRt = WgpuRuntime<AutoCompiler>;

#[derive(Debug, Parser)]
#[command(about = "Profile exact production WGSL codec stages from a precision oracle")]
struct Args {
    /// Strict FP32 precision-oracle fixture containing the exact final latent.
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

    /// Timed stage-synchronized profiling repetitions.
    #[arg(long, default_value_t = 5)]
    profile_repeats: usize,
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

fn read_f32_tensor(tensors: &SafeTensors<'_>, key: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let view = tensors
        .tensor(key)
        .with_context(|| format!("fixture tensor {key:?} is missing"))?;
    ensure!(
        view.dtype() == Dtype::F32,
        "fixture tensor {key:?} has dtype {:?}, expected F32",
        view.dtype()
    );
    let shape = view.shape().to_vec();
    let values = view
        .data()
        .chunks_exact(size_of::<f32>())
        .map(|chunk| {
            let bytes: [u8; size_of::<f32>()] = chunk
                .try_into()
                .map_err(|_| anyhow::anyhow!("invalid f32 bytes in {key:?}"))?;
            Ok(f32::from_le_bytes(bytes))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((shape, values))
}

fn load_oracle_tensors(path: &Path) -> Result<(Vec<f32>, usize, Vec<f32>)> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read precision fixture {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("malformed precision fixture {}", path.display()))?;
    let (latent_shape, latent) = read_f32_tensor(&tensors, "final_patched_latent")?;
    let (waveform_shape, waveform) = read_f32_tensor(&tensors, "raw_decoded_waveform")?;
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

fn strict_waveform_gate(reference: &[f32], actual: &[f32], label: &str) -> Result<()> {
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
    ensure!(
        metrics.max_abs_error <= 0.00015,
        "{label} max_abs gate failed"
    );
    ensure!(
        metrics.mean_abs_error <= 0.000005,
        "{label} mean_abs gate failed"
    );
    ensure!(
        metrics.root_mean_square_error <= 0.00001,
        "{label} RMSE gate failed"
    );
    ensure!(
        metrics.signal_to_noise_db >= 85.0,
        "{label} SNR gate failed"
    );
    ensure!(
        metrics.cosine_similarity >= 0.99999999,
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
    verify_sha256(&args.fixture, &args.fixture_sha256)?;
    let (latent_values, latent_steps, expected_waveform) = load_oracle_tensors(&args.fixture)?;
    println!(
        "profile_shape latent_steps={latent_steps} waveform_samples={}",
        expected_waveform.len()
    );
    let (device, monitor) = initialize_wgpu(args.adapter_index);
    let tensor_device = irodori_tts_burn::backend_config::strict_fp32_device(&device)?;

    let mut codec = load_codec(&args.codec_weights, &tensor_device)
        .with_context(|| format!("failed to load codec {}", args.codec_weights.display()))?;
    codec.prepare_decoder_for_wgsl();
    synchronize_and_check_wgpu(&device, &monitor, "codec load and preparation")?;
    let latent = Tensor::<3>::from_data(
        TensorData::new(latent_values, [1, latent_steps, 32]),
        &tensor_device,
    );

    for warmup in 1..=args.warmup {
        let output = codec.decode_wgsl(latent.clone());
        synchronize_and_check_wgpu(&device, &monitor, &format!("warmup {warmup}"))?;
        drop(output);
    }

    let mut production_device_ms = Vec::with_capacity(args.repeats);
    let mut production_readback_ms = Vec::with_capacity(args.repeats);
    let mut production_hash = None;
    for repetition in 1..=args.repeats {
        let started = Instant::now();
        let output = codec.decode_wgsl(latent.clone());
        synchronize_and_check_wgpu(
            &device,
            &monitor,
            &format!("production device completion {repetition}"),
        )?;
        let device_complete_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let values = output
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
        strict_waveform_gate(
            &expected_waveform,
            &values,
            &format!("production_waveform[{repetition}]"),
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

    let mut stage_samples: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    let mut profiled_total_ms = Vec::with_capacity(args.profile_repeats);
    for repetition in 1..=args.profile_repeats {
        let started = Instant::now();
        let (output, timings) = codec.decode_wgsl_profiled(latent.clone(), |stage| {
            synchronize_and_check_wgpu(&device, &monitor, stage)
        })?;
        let device_complete_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let values = output
            .into_data()
            .to_vec::<f32>()
            .with_context(|| format!("failed profiled readback {repetition}"))?;
        synchronize_and_check_wgpu(
            &device,
            &monitor,
            &format!("profiled repetition {repetition}"),
        )?;
        strict_waveform_gate(
            &expected_waveform,
            &values,
            &format!("profiled_waveform[{repetition}]"),
        )?;
        ensure!(
            production_hash.as_deref() == Some(&sha256_f32_le(&values)),
            "profiled waveform differs bitwise from production"
        );
        for (label, elapsed) in timings {
            stage_samples
                .entry(label)
                .or_default()
                .push(elapsed.as_secs_f64() * 1_000.0);
        }
        let readback_complete_ms = started.elapsed().as_secs_f64() * 1_000.0;
        println!(
            "profiled_repeat={repetition}/{} stage_sync_device_complete_ms={device_complete_ms:.6} stage_sync_and_readback_ms={readback_complete_ms:.6}",
            args.profile_repeats
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
    print_summary("profiled_stage_sync_device_complete", &profiled_total_ms);
    monitor.check("profile completion")?;
    println!("wgpu_uncaptured_errors=0");
    Ok(())
}
