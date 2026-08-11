//! Production-shape SDPA A/B across the supported audio-length sweep.
//!
//! The primary timer is identical for every WGPU path: a device sync before
//! launch, repeated enqueue, then a device sync. CPU readback, hashing, and
//! full-output accuracy checks happen only after the primary measurement.

use std::{
    hint::black_box,
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use burn::{
    backend::wgpu::{
        MemoryConfiguration, RuntimeOptions, WgpuDevice, WgpuRuntime, graphics::AutoGraphicsApi,
        init_setup,
    },
    tensor::{
        Bool, Distribution, Tensor, TensorData, TensorPrimitive, backend::Backend,
        module::attention as burn_attention, ops::AttentionModuleOptions,
    },
};
use clap::Parser;
use cubecl::prelude::Runtime;
use irodori_tts_wgpu::{
    WgpuRaw,
    kernels::fused_sdpa_native::{NativeFaConfig, native_fa_sdpa_wgsl},
};
use sha2::{Digest, Sha256};

type B = WgpuRaw;

const HEADS: usize = 20;
const HEAD_DIM: usize = 64;
const CONTEXT: usize = 3;
const SEQUENCES: [usize; 5] = [13, 25, 50, 100, 200];
const SEED: u64 = 0;
const PATHS: [Path; 4] = [
    Path::Production,
    Path::NativeQ8Kv32,
    Path::NativeQ16Kv16,
    Path::NativeQ32Kv8,
];

#[derive(Debug, Parser)]
#[command(about = "Production-shape Burn SDPA versus native WGSL FlashAttention")]
struct Args {
    #[arg(long, default_value_t = 0)]
    adapter_index: usize,
    #[arg(long, default_value_t = 10)]
    warmup: usize,
    #[arg(long, default_value_t = 100)]
    iterations: usize,
    #[arg(long, default_value_t = 5)]
    trials: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Path {
    Production,
    NativeQ8Kv32,
    NativeQ16Kv16,
    NativeQ32Kv8,
}

impl Path {
    const fn label(self) -> &'static str {
        match self {
            Self::Production => "production_burn",
            Self::NativeQ8Kv32 => "native_q8_kv32",
            Self::NativeQ16Kv16 => "native_q16_kv16",
            Self::NativeQ32Kv8 => "native_q32_kv8",
        }
    }

    const fn config(self) -> Option<NativeFaConfig> {
        match self {
            Self::Production => None,
            Self::NativeQ8Kv32 => Some(NativeFaConfig::Q8_KV32),
            Self::NativeQ16Kv16 => Some(NativeFaConfig::Q16_KV16),
            Self::NativeQ32Kv8 => Some(NativeFaConfig::Q32_KV8),
        }
    }
}

#[derive(Clone)]
struct Inputs {
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    production_mask: Option<Tensor<B, 4, Bool>>,
    custom_mask: Tensor<B, 2>,
}

#[derive(Clone, Copy, Debug)]
struct Timing {
    median_us: f64,
    min_us: f64,
    max_us: f64,
}

#[derive(Clone, Debug)]
struct Accuracy {
    hash: String,
    max_abs: f64,
    mean_abs: f64,
    rmse: f64,
    cosine: f64,
}

#[derive(Default)]
struct WgpuErrorMonitor {
    errors: Arc<Mutex<Vec<String>>>,
}

impl WgpuErrorMonitor {
    fn callback_sink(&self) -> Arc<Mutex<Vec<String>>> {
        Arc::clone(&self.errors)
    }

    fn check(&self, stage: &str) -> Result<()> {
        let mut errors = self
            .errors
            .lock()
            .map_err(|_| anyhow::anyhow!("WGPU error monitor lock poisoned after {stage}"))?;
        ensure!(errors.is_empty(), "WGPU errors after {stage}: {errors:?}");
        errors.clear();
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
    let errors = monitor.callback_sink();
    setup.device.on_uncaptured_error(Arc::new(move |error| {
        if let Ok(mut errors) = errors.lock() {
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

fn synchronize(device: &WgpuDevice, monitor: &WgpuErrorMonitor, stage: &str) -> Result<()> {
    let result = cubecl::future::block_on(WgpuRuntime::client(device).sync());
    monitor.check(stage)?;
    result.with_context(|| format!("CubeCL synchronization failed after {stage}"))
}

fn build_inputs(batch: usize, sequence: usize, device: &WgpuDevice) -> Inputs {
    let total = sequence + CONTEXT;
    B::seed(device, SEED ^ ((batch as u64) << 32) ^ sequence as u64);
    let q = Tensor::random(
        [batch, HEADS, sequence, HEAD_DIM],
        Distribution::Uniform(-0.25, 0.25),
        device,
    );
    let k = Tensor::random(
        [batch, HEADS, total, HEAD_DIM],
        Distribution::Uniform(-0.25, 0.25),
        device,
    );
    let v = Tensor::random(
        [batch, HEADS, total, HEAD_DIM],
        Distribution::Uniform(-0.25, 0.25),
        device,
    );

    let mut custom_values = vec![1.0_f32; batch * total];
    let production_mask = if batch == 1 {
        None
    } else {
        let mut native_values = vec![false; batch * total];
        for position in sequence..total {
            native_values[total + position] = true;
            custom_values[total + position] = 0.0;
        }
        Some(
            Tensor::<B, 2, Bool>::from_data(TensorData::new(native_values, [batch, total]), device)
                .unsqueeze_dim::<3>(1)
                .unsqueeze_dim::<4>(2),
        )
    };
    let custom_mask =
        Tensor::<B, 2>::from_data(TensorData::new(custom_values, [batch, total]), device);
    Inputs {
        q,
        k,
        v,
        production_mask,
        custom_mask,
    }
}

fn run_path(path: Path, inputs: &Inputs) -> Tensor<B, 4> {
    if let Some(config) = path.config() {
        let output = native_fa_sdpa_wgsl(
            inputs.q.clone().into_primitive().tensor(),
            inputs.k.clone().into_primitive().tensor(),
            inputs.v.clone().into_primitive().tensor(),
            inputs.custom_mask.clone().into_primitive().tensor(),
            (HEAD_DIM as f64).powf(-0.5),
            &config,
        );
        Tensor::from_primitive(TensorPrimitive::Float(output))
    } else {
        burn_attention(
            inputs.q.clone(),
            inputs.k.clone(),
            inputs.v.clone(),
            inputs.production_mask.clone(),
            None,
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: false,
            },
        )
    }
}

fn summarize(samples: &[f64]) -> Result<Timing> {
    ensure!(
        !samples.is_empty() && samples.iter().all(|sample| sample.is_finite()),
        "timing samples must be non-empty and finite"
    );
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    Ok(Timing {
        median_us: sorted[sorted.len() / 2],
        min_us: sorted[0],
        max_us: sorted[sorted.len() - 1],
    })
}

fn benchmark(
    inputs: &Inputs,
    args: &Args,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
) -> Result<[Timing; PATHS.len()]> {
    for path in PATHS {
        let mut output = None;
        for _ in 0..args.warmup {
            output = Some(run_path(path, inputs));
        }
        black_box(output.context("warmup count must be positive")?);
        synchronize(device, monitor, path.label())?;
    }

    let mut samples: [Vec<f64>; PATHS.len()] =
        std::array::from_fn(|_| Vec::with_capacity(args.trials));
    for trial in 0..args.trials {
        for offset in 0..PATHS.len() {
            let index = (trial + offset) % PATHS.len();
            let path = PATHS[index];
            synchronize(device, monitor, "pre-timer")?;
            let started = Instant::now();
            let mut output = None;
            for _ in 0..args.iterations {
                output = Some(run_path(path, inputs));
            }
            synchronize(device, monitor, path.label())?;
            let elapsed_us = started.elapsed().as_secs_f64() * 1_000_000.0;
            black_box(output.context("iteration count must be positive")?);
            samples[index].push(elapsed_us / args.iterations as f64);
            println!(
                "timing_sample trial={trial} position={offset} path={} device_complete_us={:.6}",
                path.label(),
                elapsed_us / args.iterations as f64
            );
        }
    }
    let timings = samples
        .iter()
        .map(|values| summarize(values))
        .collect::<Result<Vec<_>>>()?;
    timings
        .try_into()
        .map_err(|_| anyhow::anyhow!("all timing paths are required"))
}

fn readback(
    path: Path,
    inputs: &Inputs,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
) -> Result<Vec<f32>> {
    synchronize(device, monitor, "pre-readback-run")?;
    let output = run_path(path, inputs);
    synchronize(device, monitor, "pre-readback")?;
    let values = output.into_data().to_vec::<f32>()?;
    monitor.check("readback")?;
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "{} produced non-finite output",
        path.label()
    );
    Ok(values)
}

fn hash_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn accuracy(reference: &[f32], candidate: &[f32]) -> Result<Accuracy> {
    ensure!(
        reference.len() == candidate.len() && !reference.is_empty(),
        "accuracy vectors must be non-empty and equal length"
    );
    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut ref_sq = 0.0_f64;
    let mut candidate_sq = 0.0_f64;
    for (&reference, &candidate) in reference.iter().zip(candidate) {
        let reference = f64::from(reference);
        let candidate = f64::from(candidate);
        let difference = candidate - reference;
        max_abs = max_abs.max(difference.abs());
        sum_abs += difference.abs();
        sum_sq += difference * difference;
        dot += reference * candidate;
        ref_sq += reference * reference;
        candidate_sq += candidate * candidate;
    }
    let count = reference.len() as f64;
    let cosine = dot / (ref_sq.sqrt() * candidate_sq.sqrt());
    Ok(Accuracy {
        hash: hash_f32(candidate),
        max_abs,
        mean_abs: sum_abs / count,
        rmse: (sum_sq / count).sqrt(),
        cosine,
    })
}

fn run_scenario(
    batch: usize,
    sequence: usize,
    args: &Args,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
) -> Result<[Timing; PATHS.len()]> {
    let inputs = build_inputs(batch, sequence, device);
    println!(
        "scenario batch={batch} sequence={sequence} kv_sequence={} heads={HEADS} head_dim={HEAD_DIM}",
        sequence + CONTEXT
    );
    let timings = benchmark(&inputs, args, device, monitor)?;
    let reference = readback(Path::Production, &inputs, device, monitor)?;
    let reference_hash = hash_f32(&reference);
    println!(
        "accuracy path={} elements={} hash={reference_hash} reference=true",
        Path::Production.label(),
        reference.len()
    );
    let baseline = timings[0];
    println!(
        "timing_summary path={} median_us={:.6} min_us={:.6} max_us={:.6}",
        Path::Production.label(),
        baseline.median_us,
        baseline.min_us,
        baseline.max_us
    );
    for (index, path) in PATHS.iter().copied().enumerate().skip(1) {
        let candidate = readback(path, &inputs, device, monitor)?;
        let accuracy = accuracy(&reference, &candidate)?;
        ensure!(
            accuracy.max_abs <= 1.0e-3 && accuracy.rmse <= 1.0e-4 && accuracy.cosine >= 0.999_99,
            "{} accuracy outside diagnostic envelope: {accuracy:?}",
            path.label()
        );
        let timing = timings[index];
        let strict_nonoverlap = timing.max_us < baseline.min_us;
        println!(
            "accuracy path={} elements={} hash={} max_abs={:.9e} mean_abs={:.9e} rmse={:.9e} cosine={:.12}",
            path.label(),
            candidate.len(),
            accuracy.hash,
            accuracy.max_abs,
            accuracy.mean_abs,
            accuracy.rmse,
            accuracy.cosine
        );
        println!(
            "timing_summary path={} median_us={:.6} min_us={:.6} max_us={:.6} speedup={:.6} saving_us={:.6} strict_nonoverlap={strict_nonoverlap}",
            path.label(),
            timing.median_us,
            timing.min_us,
            timing.max_us,
            baseline.median_us / timing.median_us,
            baseline.median_us - timing.median_us
        );
    }
    Ok(timings)
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.warmup > 0, "warmup must be positive");
    ensure!(args.iterations > 0, "iterations must be positive");
    ensure!(args.trials > 0, "trials must be positive");
    let (device, monitor) = initialize_wgpu(args.adapter_index);
    println!(
        "protocol warmup={} iterations={} trials={} paths={} sequences={:?} batches=[1,2] primary=pre_sync_to_device_complete readback_in_primary=false accuracy_readback=separate",
        args.warmup,
        args.iterations,
        args.trials,
        PATHS.len(),
        SEQUENCES
    );

    for sequence in SEQUENCES {
        let b1 = run_scenario(1, sequence, &args, &device, &monitor)?;
        let b2 = run_scenario(2, sequence, &args, &device, &monitor)?;
        for (index, path) in PATHS.iter().copied().enumerate() {
            let projected_us = 24.0 * (b1[index].median_us + b2[index].median_us);
            println!(
                "rf_projection sequence={sequence} path={} evaluations_b1=24 evaluations_b2=24 median_sum_us={projected_us:.6}",
                path.label()
            );
        }
    }
    synchronize(&device, &monitor, "final")?;
    println!("wgpu_uncaptured_errors=0");
    Ok(())
}
