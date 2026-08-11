//! Benchmark fixed-schedule batching/caching of v4 timestep conditioning.
//!
//! The pinned four-step Euler request evaluates the model at
//! `[0.999, 0.74925, 0.4995, 0.24975]`. Independent CFG is active for the
//! first two evaluations only, so the current `CondModule` sees effective
//! batches `[2, 2, 1, 1]`, even though each batch contains duplicate timestep
//! rows. This isolated benchmark compares:
//!
//! - current: timestep embedding + `CondModule` four times (six total rows);
//! - B4 per request: compute four unique rows once and materialize two B2 rows;
//! - unique cache: build B4 once, materialize the two B2 rows on every hit;
//! - materialized cache: build B4 and both B2 outputs once, clone views on hit.
//!
//! Timestep inputs are preallocated outside timings, as in the production
//! sampler. The frequency construction inside timestep embedding remains in
//! every measured condition evaluation because that is current model behavior.
//! This binary and its candidate module are not production-connected.
//!
//! After explicit GPU authorization, run once with seeded random weights:
//! `cargo run --release --bin bench_fixed_euler_cond_cache -- 0`
//! or with the pinned official checkpoint's actual `CondModule` weights:
//! `cargo run --release --bin bench_fixed_euler_cond_cache -- 0 --checkpoint /path/model.safetensors`

use std::{
    error::Error,
    hint::black_box,
    io,
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex},
    time::Instant,
};

use burn::{
    backend::wgpu::{
        RuntimeOptions, WgpuDevice, WgpuRuntime, graphics::AutoGraphicsApi, init_setup,
    },
    tensor::{Tensor, backend::Backend},
};
use cubecl::prelude::Runtime;
use irodori_tts_wgpu::WgpuRaw;

#[path = "../model/fixed_euler_cond_cache_candidate.rs"]
mod fixed_euler_cond_cache_candidate;

use fixed_euler_cond_cache_candidate::{
    COND_WIDTH, CondWeights, EFFECTIVE_BATCHES, EULER_STEPS, FixedEulerCondCacheKey,
    FixedEulerCondOutputs, FixedEulerTimestepInputs, LOGICAL_MATERIALIZED_BYTES, MODEL_DIM,
    MaterializedFixedEulerCondCache, TIMESTEP_EMBED_DIM, UNIQUE_CACHE_BYTES,
    UniqueFixedEulerCondCache, baseline_fixed_request, batched_fixed_request, pinned_schedule,
};

type B = WgpuRaw;

const DEFAULT_WARMUP: usize = 10;
const DEFAULT_ITERATIONS: usize = 100;
const DEFAULT_TRIALS: usize = 5;
const SEED: u64 = 0;
const MODEL_GENERATION: u64 = 1;
const F32_BYTES: usize = size_of::<f32>();
const BASELINE_LOGICAL_DISPATCHES: usize = EULER_STEPS * 9;
const B4_BUILD_LOGICAL_DISPATCHES: usize = 9;
const MATERIALIZE_LOGICAL_DISPATCHES: usize = 2;
const MATERIALIZED_RETAINED_BYTES: usize =
    UNIQUE_CACHE_BYTES + (EFFECTIVE_BATCHES[0] + EFFECTIVE_BATCHES[1]) * COND_WIDTH * F32_BYTES;
const MAX_ACCEPTED_ABS: f32 = 1.0e-5;
const OFFICIAL_MODEL_SHA256: &str =
    "5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593";
const OFFICIAL_MODEL_BYTES: u64 = 3_064_295_596;
const EXPECTED_COMPARISON_ELEMENTS: usize = 23_040;

#[derive(Debug)]
struct Args {
    adapter_index: usize,
    checkpoint: Option<PathBuf>,
    warmup: usize,
    iterations: usize,
    trials: usize,
}

enum ParseOutcome {
    Run(Args),
    Help,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Workload {
    Baseline,
    BatchedPerRequest,
    UniqueCacheBuild,
    UniqueCacheHit,
    MaterializedCacheBuild,
    MaterializedCacheHit,
}

impl Workload {
    const fn label(self) -> &'static str {
        match self {
            Self::Baseline => "current-4-forward",
            Self::BatchedPerRequest => "b4-build+materialize/request",
            Self::UniqueCacheBuild => "unique-b4-cache-build",
            Self::UniqueCacheHit => "unique-b4-cache-hit+materialize",
            Self::MaterializedCacheBuild => "materialized-cache-build",
            Self::MaterializedCacheHit => "materialized-cache-hit",
        }
    }
}

const WORKLOADS: [Workload; 6] = [
    Workload::Baseline,
    Workload::BatchedPerRequest,
    Workload::UniqueCacheBuild,
    Workload::UniqueCacheHit,
    Workload::MaterializedCacheBuild,
    Workload::MaterializedCacheHit,
];

enum BenchProduct {
    Outputs(FixedEulerCondOutputs),
    Unique(UniqueFixedEulerCondCache),
    Materialized(MaterializedFixedEulerCondCache),
}

impl BenchProduct {
    fn terminal(&self) -> Tensor<B, 3> {
        match self {
            Self::Outputs(outputs) => outputs.last(),
            Self::Unique(cache) => cache.last(),
            Self::Materialized(cache) => cache.last(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Timing {
    median_us: f64,
    min_us: f64,
    max_us: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Comparison {
    elements: usize,
    bit_mismatches: usize,
    max_abs: f32,
}

impl Comparison {
    fn merge(&mut self, other: Self) {
        self.elements += other.elements;
        self.bit_mismatches += other.bit_mismatches;
        self.max_abs = self.max_abs.max(other.max_abs);
    }
}

#[derive(Debug)]
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

    fn check(&self, stage: &str) -> Result<(), Box<dyn Error>> {
        let mut errors = self.errors.lock().map_err(|_| {
            io::Error::other(format!(
                "WGPU error monitor lock was poisoned after {stage}"
            ))
        })?;
        if errors.is_empty() {
            return Ok(());
        }
        let count = errors.len();
        let details = errors.drain(..).collect::<Vec<_>>().join("\n---\n");
        Err(io::Error::other(format!(
            "WGPU reported {count} uncaptured error(s) during {stage}:\n{details}"
        ))
        .into())
    }
}

struct Fixture {
    key: FixedEulerCondCacheKey,
    inputs: FixedEulerTimestepInputs,
    weights: CondWeights,
    unique_cache: UniqueFixedEulerCondCache,
    materialized_cache: MaterializedFixedEulerCondCache,
}

fn usage() -> &'static str {
    "usage: bench_fixed_euler_cond_cache <adapter-index> [--checkpoint PATH] \
     [--warmup N] [--iterations N] [--trials N]"
}

fn next_positive_usize(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<usize, Box<dyn Error>> {
    let text = args
        .next()
        .ok_or_else(|| io::Error::other(format!("{option} requires a value")))?;
    let value = text.parse::<usize>().map_err(|error| {
        io::Error::other(format!("invalid value {text:?} for {option}: {error}"))
    })?;
    if value == 0 {
        return Err(io::Error::other(format!("{option} must be greater than zero")).into());
    }
    Ok(value)
}

fn parse_args() -> Result<ParseOutcome, Box<dyn Error>> {
    let mut adapter_index = None;
    let mut checkpoint = None;
    let mut warmup = DEFAULT_WARMUP;
    let mut iterations = DEFAULT_ITERATIONS;
    let mut trials = DEFAULT_TRIALS;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--checkpoint" => {
                let path = args
                    .next()
                    .ok_or_else(|| io::Error::other("--checkpoint requires a value"))?;
                if checkpoint.replace(PathBuf::from(path)).is_some() {
                    return Err(io::Error::other("--checkpoint may only be specified once").into());
                }
            }
            "--warmup" => warmup = next_positive_usize(&mut args, "--warmup")?,
            "--iterations" => iterations = next_positive_usize(&mut args, "--iterations")?,
            "--trials" => trials = next_positive_usize(&mut args, "--trials")?,
            "--help" | "-h" => return Ok(ParseOutcome::Help),
            _ if argument.starts_with('-') => {
                return Err(
                    io::Error::other(format!("unknown option {argument:?}; {}", usage())).into(),
                );
            }
            _ if adapter_index.is_none() => {
                adapter_index = Some(argument.parse::<usize>().map_err(|error| {
                    io::Error::other(format!(
                        "invalid adapter index {argument:?}: {error}; {}",
                        usage()
                    ))
                })?);
            }
            _ => {
                return Err(io::Error::other(format!(
                    "unexpected positional argument {argument:?}; {}",
                    usage()
                ))
                .into());
            }
        }
    }

    Ok(ParseOutcome::Run(Args {
        adapter_index: adapter_index
            .ok_or_else(|| io::Error::other(format!("missing adapter index; {}", usage())))?,
        checkpoint,
        warmup,
        iterations,
        trials,
    }))
}

fn verify_official_checkpoint(path: &Path) -> Result<(), Box<dyn Error>> {
    let metadata = path.metadata().map_err(|error| {
        io::Error::other(format!(
            "failed to inspect checkpoint {}: {error}",
            path.display()
        ))
    })?;
    if !metadata.is_file() {
        return Err(io::Error::other(format!(
            "checkpoint {} is not a regular file",
            path.display()
        ))
        .into());
    }
    if metadata.len() != OFFICIAL_MODEL_BYTES {
        return Err(io::Error::other(format!(
            "official checkpoint byte length mismatch: expected {OFFICIAL_MODEL_BYTES}, got {} \
             for {}",
            metadata.len(),
            path.display()
        ))
        .into());
    }

    // `sha2` is intentionally optional in this crate, while this isolated
    // target must also compile without the CLI feature. Invoke the platform
    // verifier directly (without a shell); absence or malformed output fails
    // closed before WGPU initialization or checkpoint deserialization.
    let output = Command::new("sha256sum")
        .arg("--")
        .arg(path)
        .output()
        .map_err(|error| {
            io::Error::other(format!(
                "failed to execute sha256sum for {}: {error}",
                path.display()
            ))
        })?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "sha256sum failed for {} with status {}: {}",
            path.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ))
        .into());
    }
    let stdout = std::str::from_utf8(&output.stdout).map_err(|error| {
        io::Error::other(format!("sha256sum emitted non-UTF-8 output: {error}"))
    })?;
    let actual = stdout
        .split_whitespace()
        .next()
        .ok_or_else(|| io::Error::other("sha256sum emitted an empty digest"))?;
    if actual.len() != OFFICIAL_MODEL_SHA256.len()
        || !actual
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(
            io::Error::other(format!("sha256sum emitted malformed digest {actual:?}")).into(),
        );
    }
    if actual != OFFICIAL_MODEL_SHA256 {
        return Err(io::Error::other(format!(
            "official checkpoint SHA-256 mismatch: expected {OFFICIAL_MODEL_SHA256}, got {actual}"
        ))
        .into());
    }
    Ok(())
}

fn initialize_wgpu(adapter_index: usize) -> (WgpuDevice, WgpuErrorMonitor) {
    let device = WgpuDevice::DiscreteGpu(adapter_index);
    let setup = init_setup::<AutoGraphicsApi>(&device, RuntimeOptions::default());
    let monitor = WgpuErrorMonitor::new();
    let callback_errors = monitor.callback_sink();
    setup.device.on_uncaptured_error(Arc::new(move |error| {
        if let Ok(mut errors) = callback_errors.lock() {
            errors.push(error.to_string());
        }
    }));
    let info = setup.adapter.get_info();
    println!(
        "wgpu_adapter: index={adapter_index} name={:?} backend={:?} device_type={:?}",
        info.name, info.backend, info.device_type
    );
    (device, monitor)
}

fn synchronize_and_check_wgpu(
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    stage: &str,
) -> Result<(), Box<dyn Error>> {
    let client = WgpuRuntime::client(device);
    let sync_result = cubecl::future::block_on(client.sync());
    monitor.check(stage)?;
    sync_result.map_err(|error| {
        io::Error::other(format!(
            "CubeCL synchronization failed after {stage}: {error}"
        ))
        .into()
    })
}

fn load_weights(
    checkpoint: Option<&Path>,
    device: &WgpuDevice,
) -> Result<CondWeights, Box<dyn Error>> {
    match checkpoint {
        Some(path) => CondWeights::from_verified_official_checkpoint(path, device),
        None => Ok(CondWeights::random(device)),
    }
}

fn build_fixture(device: &WgpuDevice, weights: CondWeights) -> Result<Fixture, Box<dyn Error>> {
    let key = FixedEulerCondCacheKey::pinned(MODEL_GENERATION)?;
    let inputs = FixedEulerTimestepInputs::new(key, device)?;
    weights.validate(device)?;
    let unique_cache = UniqueFixedEulerCondCache::build(key, &inputs, &weights, device)?;
    let materialized_cache =
        MaterializedFixedEulerCondCache::build(key, &inputs, &weights, device)?;
    Ok(Fixture {
        key,
        inputs,
        weights,
        unique_cache,
        materialized_cache,
    })
}

fn execute_workload(
    workload: Workload,
    fixture: &Fixture,
    device: &WgpuDevice,
) -> Result<BenchProduct, Box<dyn Error>> {
    match workload {
        Workload::Baseline => Ok(BenchProduct::Outputs(baseline_fixed_request(
            fixture.key,
            &fixture.inputs,
            &fixture.weights,
            device,
        )?)),
        Workload::BatchedPerRequest => Ok(BenchProduct::Outputs(batched_fixed_request(
            fixture.key,
            &fixture.inputs,
            &fixture.weights,
            device,
        )?)),
        Workload::UniqueCacheBuild => Ok(BenchProduct::Unique(UniqueFixedEulerCondCache::build(
            fixture.key,
            &fixture.inputs,
            &fixture.weights,
            device,
        )?)),
        Workload::UniqueCacheHit => Ok(BenchProduct::Outputs(
            fixture.unique_cache.materialize(fixture.key, device)?,
        )),
        Workload::MaterializedCacheBuild => Ok(BenchProduct::Materialized(
            MaterializedFixedEulerCondCache::build(
                fixture.key,
                &fixture.inputs,
                &fixture.weights,
                device,
            )?,
        )),
        Workload::MaterializedCacheHit => Ok(BenchProduct::Outputs(
            fixture.materialized_cache.get(fixture.key, device)?,
        )),
    }
}

fn compare_tensors(
    expected: Tensor<B, 3>,
    actual: Tensor<B, 3>,
) -> Result<Comparison, Box<dyn Error>> {
    if expected.dims() != actual.dims() {
        return Err(io::Error::other(format!(
            "comparison shape mismatch: expected {:?}, actual {:?}",
            expected.dims(),
            actual.dims()
        ))
        .into());
    }
    let expected = expected.into_data().to_vec::<f32>()?;
    let actual = actual.into_data().to_vec::<f32>()?;
    if expected.len() != actual.len() {
        return Err(io::Error::other(format!(
            "comparison length mismatch: expected {}, actual {}",
            expected.len(),
            actual.len()
        ))
        .into());
    }

    let mut comparison = Comparison {
        elements: expected.len(),
        ..Comparison::default()
    };
    for (index, (&expected, &actual)) in expected.iter().zip(&actual).enumerate() {
        if !expected.is_finite() || !actual.is_finite() {
            return Err(io::Error::other(format!(
                "non-finite output at element {index}: expected={expected:?}, actual={actual:?}"
            ))
            .into());
        }
        comparison.bit_mismatches += usize::from(expected.to_bits() != actual.to_bits());
        comparison.max_abs = comparison.max_abs.max((expected - actual).abs());
    }
    Ok(comparison)
}

fn compare_outputs(
    label: &str,
    expected: &FixedEulerCondOutputs,
    actual: &FixedEulerCondOutputs,
) -> Result<Comparison, Box<dyn Error>> {
    let mut comparison = Comparison::default();
    for (step, &batch) in EFFECTIVE_BATCHES.iter().enumerate() {
        let step_comparison = compare_tensors(expected.step(step)?, actual.step(step)?)?;
        let expected_elements = batch * COND_WIDTH;
        if step_comparison.elements != expected_elements {
            return Err(io::Error::other(format!(
                "{label} step {step} comparison element mismatch: expected {expected_elements}, \
                 got {}",
                step_comparison.elements
            ))
            .into());
        }
        println!(
            "correctness {label} step={step}: batch={} elements={} bit_mismatches={} \
             max_abs={:.9e} finite=true",
            batch,
            step_comparison.elements,
            step_comparison.bit_mismatches,
            step_comparison.max_abs
        );
        comparison.merge(step_comparison);
    }
    if comparison.elements != EXPECTED_COMPARISON_ELEMENTS {
        return Err(io::Error::other(format!(
            "{label} aggregate comparison element mismatch: expected \
             {EXPECTED_COMPARISON_ELEMENTS}, got {}",
            comparison.elements
        ))
        .into());
    }
    println!(
        "correctness {label} aggregate: elements={} bit_mismatches={} max_abs={:.9e} \
         finite=true bit_exact={}",
        comparison.elements,
        comparison.bit_mismatches,
        comparison.max_abs,
        comparison.bit_mismatches == 0
    );
    if comparison.max_abs > MAX_ACCEPTED_ABS {
        return Err(io::Error::other(format!(
            "{label} condition-cache max_abs={:.9e} exceeds fail-closed benchmark gate {:.9e}",
            comparison.max_abs, MAX_ACCEPTED_ABS
        ))
        .into());
    }
    Ok(comparison)
}

fn correctness_report(fixture: &Fixture, device: &WgpuDevice) -> Result<(), Box<dyn Error>> {
    let expected = baseline_fixed_request(fixture.key, &fixture.inputs, &fixture.weights, device)?;
    let batched = batched_fixed_request(fixture.key, &fixture.inputs, &fixture.weights, device)?;
    let unique_hit = fixture.unique_cache.materialize(fixture.key, device)?;
    let materialized_hit = fixture.materialized_cache.get(fixture.key, device)?;

    for (label, actual) in [
        ("b4-build+materialize", batched),
        ("unique-cache-hit", unique_hit),
        ("materialized-cache-hit", materialized_hit),
    ] {
        compare_outputs(label, &expected, &actual)?;
    }
    Ok(())
}

fn warm_up_workload(
    workload: Workload,
    fixture: &Fixture,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    count: usize,
) -> Result<(), Box<dyn Error>> {
    let mut output = None;
    for _ in 0..count {
        output = Some(execute_workload(workload, fixture, device)?);
    }
    let output = output.ok_or_else(|| io::Error::other("warmup count must be non-zero"))?;
    black_box(output.terminal().dims());
    synchronize_and_check_wgpu(device, monitor, &format!("{} warmup", workload.label()))
}

fn measure_workload_once(
    workload: Workload,
    fixture: &Fixture,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    iterations: usize,
) -> Result<f64, Box<dyn Error>> {
    let started = Instant::now();
    let mut output = None;
    for _ in 0..iterations {
        output = Some(execute_workload(workload, fixture, device)?);
    }
    let output = output.ok_or_else(|| io::Error::other("iteration count must be non-zero"))?;
    black_box(output.terminal().dims());
    synchronize_and_check_wgpu(device, monitor, workload.label())?;
    Ok(started.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64)
}

fn summarize_samples(samples: &[f64]) -> Result<Timing, Box<dyn Error>> {
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return Err(io::Error::other("timing samples must be non-empty and finite").into());
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    Ok(Timing {
        median_us: sorted[sorted.len() / 2],
        min_us: sorted[0],
        max_us: sorted[sorted.len() - 1],
    })
}

fn benchmark_workloads(
    fixture: &Fixture,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    args: &Args,
) -> Result<[Timing; WORKLOADS.len()], Box<dyn Error>> {
    for workload in WORKLOADS {
        warm_up_workload(workload, fixture, device, monitor, args.warmup)?;
    }

    let mut samples: [Vec<f64>; WORKLOADS.len()] =
        std::array::from_fn(|_| Vec::with_capacity(args.trials));
    for trial in 0..args.trials {
        for offset in 0..WORKLOADS.len() {
            let index = (trial + offset) % WORKLOADS.len();
            samples[index].push(measure_workload_once(
                WORKLOADS[index],
                fixture,
                device,
                monitor,
                args.iterations,
            )?);
        }
    }
    let timings = samples
        .iter()
        .map(|samples| summarize_samples(samples))
        .collect::<Result<Vec<_>, _>>()?;
    timings
        .try_into()
        .map_err(|_| io::Error::other("all workload timing sets are required").into())
}

fn cache_break_even_requests(build_us: f64, hit_us: f64, baseline_us: f64) -> Option<usize> {
    let savings = baseline_us - hit_us;
    if savings > 0.0 {
        Some((build_us / savings).ceil().max(1.0) as usize)
    } else {
        None
    }
}

fn print_static_accounting() {
    let macs_per_row = CondWeights::parameter_macs_per_row();
    let baseline_rows = EFFECTIVE_BATCHES.iter().sum::<usize>();
    let baseline_macs = baseline_rows * macs_per_row;
    let batched_macs = EULER_STEPS * macs_per_row;
    let mac_reduction = 100.0 * (baseline_macs - batched_macs) as f64 / baseline_macs as f64;
    let schedule = pinned_schedule();
    let bits = schedule.map(f32::to_bits);
    println!(
        "contract: D={MODEL_DIM} t_embed={TIMESTEP_EMBED_DIM} out={COND_WIDTH} \
         schedule={schedule:?} schedule_bits={bits:x?} effective_batches={EFFECTIVE_BATCHES:?}"
    );
    println!(
        "MAC accounting: per_row={macs_per_row} current_rows={baseline_rows} \
         current={baseline_macs} B4_rows={EULER_STEPS} B4={batched_macs} \
         reduction={mac_reduction:.3}%"
    );
    println!(
        "logical Burn GPU ops: current={BASELINE_LOGICAL_DISPATCHES} \
         B4_build={B4_BUILD_LOGICAL_DISPATCHES} materialize={MATERIALIZE_LOGICAL_DISPATCHES} \
         B4/request={} unique_cache_hit={MATERIALIZE_LOGICAL_DISPATCHES} materialized_cache_hit=0; \
         backend autotune/packing may emit additional physical dispatches",
        B4_BUILD_LOGICAL_DISPATCHES + MATERIALIZE_LOGICAL_DISPATCHES
    );
    println!(
        "cache bytes: unique={} ({:.3} KiB), logical materialized={} ({:.3} KiB), \
         retained materialized={} ({:.3} KiB; B4 backing + two B2 repeat buffers)",
        UNIQUE_CACHE_BYTES,
        UNIQUE_CACHE_BYTES as f64 / 1024.0,
        LOGICAL_MATERIALIZED_BYTES,
        LOGICAL_MATERIALIZED_BYTES as f64 / 1024.0,
        MATERIALIZED_RETAINED_BYTES,
        MATERIALIZED_RETAINED_BYTES as f64 / 1024.0
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = match parse_args()? {
        ParseOutcome::Run(args) => args,
        ParseOutcome::Help => {
            println!("{}", usage());
            return Ok(());
        }
    };

    if let Some(checkpoint) = args.checkpoint.as_deref() {
        let started = Instant::now();
        verify_official_checkpoint(checkpoint)?;
        println!(
            "official checkpoint verified: path={} bytes={OFFICIAL_MODEL_BYTES} sha256={} \
             elapsed={:.3}s",
            checkpoint.display(),
            OFFICIAL_MODEL_SHA256,
            started.elapsed().as_secs_f64()
        );
    }

    let (device, monitor) = initialize_wgpu(args.adapter_index);
    B::seed(&device, SEED);

    let weights_started = Instant::now();
    let weights = load_weights(args.checkpoint.as_deref(), &device)?;
    synchronize_and_check_wgpu(&device, &monitor, "condition weight load")?;
    println!(
        "condition weights: source={} keys={:?} load_and_upload={:.3}s (outside timings)",
        weights.source_label(),
        weights.checkpoint_keys(),
        weights_started.elapsed().as_secs_f64()
    );

    let fixture_started = Instant::now();
    let fixture = build_fixture(&device, weights)?;
    synchronize_and_check_wgpu(&device, &monitor, "fixture and reusable cache construction")?;
    println!(
        "fixture/cache construction: {:.3}s (outside timings)",
        fixture_started.elapsed().as_secs_f64()
    );
    println!(
        "fixed Euler CondModule cache benchmark: source={} warmup={} iterations={} trials={} \
         seed={SEED}",
        fixture.weights.source_label(),
        args.warmup,
        args.iterations,
        args.trials
    );
    print_static_accounting();

    correctness_report(&fixture, &device)?;
    synchronize_and_check_wgpu(&device, &monitor, "correctness")?;

    let timings = benchmark_workloads(&fixture, &device, &monitor, &args)?;
    for (workload, timing) in WORKLOADS.into_iter().zip(timings) {
        println!(
            "timing {:>36}: median={:.3} us range=[{:.3}, {:.3}] us",
            workload.label(),
            timing.median_us,
            timing.min_us,
            timing.max_us
        );
    }

    let baseline = timings[0].median_us;
    let batched = timings[1].median_us;
    let unique_build = timings[2].median_us;
    let unique_hit = timings[3].median_us;
    let materialized_build = timings[4].median_us;
    let materialized_hit = timings[5].median_us;
    println!(
        "speedup B4/request={:.3}x unique-cache-hit={:.3}x materialized-cache-hit={:.3}x",
        baseline / batched,
        baseline / unique_hit,
        baseline / materialized_hit
    );
    println!(
        "cache break-even requests: unique={:?} materialized={:?}",
        cache_break_even_requests(unique_build, unique_hit, baseline),
        cache_break_even_requests(materialized_build, materialized_hit, baseline)
    );
    monitor.check("final")?;
    println!("WGPU uncaptured errors: 0");
    Ok(())
}
