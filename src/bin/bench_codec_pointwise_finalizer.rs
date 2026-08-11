//! Isolated benchmark for a codec pointwise residual finalizer candidate.
//!
//! The current path is:
//! `packed matmul [1,L,C] -> bias add -> NCL view -> residual add ->`
//! `implicit contiguous copy -> existing Snake`.
//! The candidate keeps the packed matmul and existing Snake unchanged, but
//! replaces the two adds and layout materialization with one SourceKernel that
//! emits contiguous `[1,C,L]`.
//!
//! The candidate side calls the same production launcher used by the codec,
//! so this remains a direct parity and performance guard for that boundary.

use std::{
    error::Error,
    io,
    sync::{Arc, Mutex},
    time::Instant,
};

use burn::{
    backend::wgpu::{
        RuntimeOptions, WgpuDevice, WgpuRuntime, graphics::AutoGraphicsApi, init_setup,
    },
    tensor::{Distribution, Tensor, TensorPrimitive, backend::Backend},
};
use cubecl::prelude::Runtime;
use irodori_tts_wgpu::{
    WgpuRaw,
    kernels::{pointwise_residual_finalizer::pointwise_residual_finalizer_wgsl, snake::snake_wgsl},
};

type B = WgpuRaw;

const DEFAULT_WARMUP: usize = 10;
const DEFAULT_ITERATIONS: usize = 20;
const DEFAULT_TRIALS: usize = 5;
const F32_BYTES: usize = core::mem::size_of::<f32>();

#[derive(Clone, Copy, Debug)]
struct PointwiseCase {
    stage: usize,
    unit: usize,
    channels: usize,
    length: usize,
}

const CASES: [PointwiseCase; 12] = [
    PointwiseCase {
        stage: 0,
        unit: 0,
        channels: 768,
        length: 600,
    },
    PointwiseCase {
        stage: 0,
        unit: 1,
        channels: 768,
        length: 600,
    },
    PointwiseCase {
        stage: 0,
        unit: 2,
        channels: 768,
        length: 600,
    },
    PointwiseCase {
        stage: 1,
        unit: 0,
        channels: 384,
        length: 6_000,
    },
    PointwiseCase {
        stage: 1,
        unit: 1,
        channels: 384,
        length: 6_000,
    },
    PointwiseCase {
        stage: 1,
        unit: 2,
        channels: 384,
        length: 6_000,
    },
    PointwiseCase {
        stage: 2,
        unit: 0,
        channels: 192,
        length: 48_000,
    },
    PointwiseCase {
        stage: 2,
        unit: 1,
        channels: 192,
        length: 48_000,
    },
    PointwiseCase {
        stage: 2,
        unit: 2,
        channels: 192,
        length: 48_000,
    },
    PointwiseCase {
        stage: 3,
        unit: 0,
        channels: 96,
        length: 96_000,
    },
    PointwiseCase {
        stage: 3,
        unit: 1,
        channels: 96,
        length: 96_000,
    },
    PointwiseCase {
        stage: 3,
        unit: 2,
        channels: 96,
        length: 96_000,
    },
];

#[derive(Debug)]
struct Args {
    adapter_index: usize,
    warmup: usize,
    iterations: usize,
    trials: usize,
}

enum ParseOutcome {
    Run(Args),
    Help,
}

#[derive(Clone, Copy, Debug)]
struct Timing {
    median_us: f64,
    min_us: f64,
    max_us: f64,
}

#[derive(Clone, Copy, Debug)]
struct TensorComparison {
    elements: usize,
    bit_mismatches: usize,
    max_abs: f32,
}

#[derive(Clone, Copy, Debug)]
struct BoundaryComparison {
    raw: TensorComparison,
    activated: TensorComparison,
}

#[derive(Debug)]
struct CaseResult {
    case: PointwiseCase,
    baseline: Timing,
    candidate: Timing,
    comparison: BoundaryComparison,
}

#[derive(Clone)]
struct BoundaryOutputs {
    raw: Tensor<B, 3>,
    activated: Tensor<B, 3>,
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

fn usage() -> &'static str {
    "usage: bench_codec_pointwise_finalizer <adapter-index> [--warmup N] \
     [--iterations N] [--trials N]"
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
    let mut warmup = DEFAULT_WARMUP;
    let mut iterations = DEFAULT_ITERATIONS;
    let mut trials = DEFAULT_TRIALS;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
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
        warmup,
        iterations,
        trials,
    }))
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

fn pointwise_matmul(input_ncl: &Tensor<B, 3>, packed_weight: &Tensor<B, 3>) -> Tensor<B, 3> {
    input_ncl
        .clone()
        .swap_dims(1, 2)
        .matmul(packed_weight.clone())
}

fn baseline_forward(
    input_ncl: &Tensor<B, 3>,
    packed_weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    residual_ncl: &Tensor<B, 3>,
    alpha: &Tensor<B, 3>,
    case: PointwiseCase,
) -> BoundaryOutputs {
    let branch_nlc = pointwise_matmul(input_ncl, packed_weight);
    let biased_nlc = branch_nlc + bias.clone().reshape([1, 1, case.channels]);
    let branch_ncl = biased_nlc
        .swap_dims(1, 2)
        .reshape([1, case.channels, case.length]);
    let raw = branch_ncl + residual_ncl.clone();
    let activated = snake_wgsl(
        raw.clone().into_primitive().tensor(),
        alpha.clone().into_primitive().tensor(),
    );
    BoundaryOutputs {
        raw,
        activated: Tensor::from_primitive(TensorPrimitive::Float(activated)),
    }
}

fn candidate_forward(
    input_ncl: &Tensor<B, 3>,
    packed_weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    residual_ncl: &Tensor<B, 3>,
    alpha: &Tensor<B, 3>,
) -> Result<BoundaryOutputs, Box<dyn Error>> {
    let branch_nlc = pointwise_matmul(input_ncl, packed_weight);
    let raw = pointwise_residual_finalizer_wgsl(
        branch_nlc.into_primitive().tensor(),
        bias.clone().into_primitive().tensor(),
        residual_ncl.clone().into_primitive().tensor(),
    )?;
    let raw: Tensor<B, 3> = Tensor::from_primitive(TensorPrimitive::Float(raw));
    let activated = snake_wgsl(
        raw.clone().into_primitive().tensor(),
        alpha.clone().into_primitive().tensor(),
    );
    Ok(BoundaryOutputs {
        raw,
        activated: Tensor::from_primitive(TensorPrimitive::Float(activated)),
    })
}

fn compare_tensor(
    name: &str,
    expected: Tensor<B, 3>,
    actual: Tensor<B, 3>,
) -> Result<TensorComparison, Box<dyn Error>> {
    let expected_shape = expected.dims();
    let actual_shape = actual.dims();
    if expected_shape != actual_shape {
        return Err(io::Error::other(format!(
            "{name} shape mismatch: baseline={expected_shape:?} candidate={actual_shape:?}"
        ))
        .into());
    }
    let expected = expected.into_data().to_vec::<f32>()?;
    let actual = actual.into_data().to_vec::<f32>()?;
    if expected.len() != actual.len() {
        return Err(io::Error::other(format!(
            "{name} length mismatch: baseline={} candidate={}",
            expected.len(),
            actual.len()
        ))
        .into());
    }
    let mut bit_mismatches = 0;
    let mut max_abs = 0.0_f32;
    for (index, (&expected, &actual)) in expected.iter().zip(&actual).enumerate() {
        if !expected.is_finite() || !actual.is_finite() {
            return Err(io::Error::other(format!(
                "{name} contains non-finite values at {index}: baseline={expected:?} candidate={actual:?}"
            ))
            .into());
        }
        if expected.to_bits() != actual.to_bits() {
            bit_mismatches += 1;
        }
        max_abs = max_abs.max((expected - actual).abs());
    }
    if !max_abs.is_finite() {
        return Err(io::Error::other(format!("{name} max_abs is non-finite")).into());
    }
    Ok(TensorComparison {
        elements: expected.len(),
        bit_mismatches,
        max_abs,
    })
}

fn compare_outputs(
    baseline: BoundaryOutputs,
    candidate: BoundaryOutputs,
) -> Result<BoundaryComparison, Box<dyn Error>> {
    Ok(BoundaryComparison {
        raw: compare_tensor("raw residual", baseline.raw, candidate.raw)?,
        activated: compare_tensor("Snake output", baseline.activated, candidate.activated)?,
    })
}

fn synchronize(outputs: BoundaryOutputs, case: PointwiseCase) -> Result<(), Box<dyn Error>> {
    if outputs.raw.dims() != [1, case.channels, case.length]
        || outputs.activated.dims() != [1, case.channels, case.length]
    {
        return Err(io::Error::other(format!(
            "synchronization output shape mismatch for C={} L={}",
            case.channels, case.length
        ))
        .into());
    }
    let values = outputs
        .activated
        .slice([
            0..1,
            case.channels - 1..case.channels,
            case.length - 1..case.length,
        ])
        .into_data()
        .to_vec::<f32>()?;
    let value = values
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("synchronization readback returned no activated value"))?;
    if !value.is_finite() {
        return Err(
            io::Error::other(format!("synchronization readback is non-finite: {value:?}")).into(),
        );
    }
    Ok(())
}

fn warm_up<F>(count: usize, case: PointwiseCase, operation: &mut F) -> Result<(), Box<dyn Error>>
where
    F: FnMut() -> Result<BoundaryOutputs, Box<dyn Error>>,
{
    let mut last = None;
    for _ in 0..count {
        last = Some(operation()?);
    }
    let output = last.ok_or_else(|| io::Error::other("warmup count must be non-zero"))?;
    synchronize(output, case)
}

fn measure<F>(
    iterations: usize,
    case: PointwiseCase,
    operation: &mut F,
) -> Result<f64, Box<dyn Error>>
where
    F: FnMut() -> Result<BoundaryOutputs, Box<dyn Error>>,
{
    let started = Instant::now();
    let mut last = None;
    for _ in 0..iterations {
        last = Some(operation()?);
    }
    let output = last.ok_or_else(|| io::Error::other("iteration count must be non-zero"))?;
    synchronize(output, case)?;
    Ok(started.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64)
}

fn summarize_samples(samples: &[f64]) -> Result<Timing, Box<dyn Error>> {
    if samples.is_empty() {
        return Err(io::Error::other("cannot summarize an empty timing sample set").into());
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(io::Error::other("timing samples contain a non-finite value").into());
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    Ok(Timing {
        median_us: sorted[sorted.len() / 2],
        min_us: sorted[0],
        max_us: sorted[sorted.len() - 1],
    })
}

fn output_elements(case: PointwiseCase) -> Result<usize, Box<dyn Error>> {
    case.channels
        .checked_mul(case.length)
        .ok_or_else(|| io::Error::other("official C*L overflows usize").into())
}

fn output_bytes(case: PointwiseCase) -> Result<usize, Box<dyn Error>> {
    output_elements(case)?
        .checked_mul(F32_BYTES)
        .ok_or_else(|| io::Error::other("official output byte count overflows usize").into())
}

fn benchmark_case(
    device: &<B as Backend>::Device,
    case: PointwiseCase,
    args: &Args,
) -> Result<CaseResult, Box<dyn Error>> {
    let input_ncl = Tensor::<B, 3>::random(
        [1, case.channels, case.length],
        Distribution::Uniform(-0.25, 0.25),
        device,
    );
    let packed_weight = Tensor::<B, 3>::random(
        [1, case.channels, case.channels],
        Distribution::Uniform(-0.025, 0.025),
        device,
    );
    let bias = Tensor::<B, 1>::random([case.channels], Distribution::Uniform(-0.05, 0.05), device);
    let residual_ncl = Tensor::<B, 3>::random(
        [1, case.channels, case.length],
        Distribution::Uniform(-0.5, 0.5),
        device,
    );
    let alpha = Tensor::<B, 3>::random(
        [1, case.channels, 1],
        Distribution::Uniform(0.25, 2.0),
        device,
    );

    // Compile both exact paths and validate both observable tensors before
    // timing. The candidate has no numerical tolerance: its elementwise f32
    // operation order must be bit-identical to the current path.
    let baseline_output = baseline_forward(
        &input_ncl,
        &packed_weight,
        &bias,
        &residual_ncl,
        &alpha,
        case,
    );
    let candidate_output =
        candidate_forward(&input_ncl, &packed_weight, &bias, &residual_ncl, &alpha)?;
    let comparison = compare_outputs(baseline_output, candidate_output)?;
    println!(
        "stage={} unit={} C={:4} L={:6}",
        case.stage, case.unit, case.channels, case.length
    );
    println!(
        "  correctness: raw elements={} bit_mismatch={} max_abs={:.9e}; Snake elements={} bit_mismatch={} max_abs={:.9e}",
        comparison.raw.elements,
        comparison.raw.bit_mismatches,
        comparison.raw.max_abs,
        comparison.activated.elements,
        comparison.activated.bit_mismatches,
        comparison.activated.max_abs,
    );
    if comparison.raw.bit_mismatches != 0 || comparison.activated.bit_mismatches != 0 {
        return Err(io::Error::other(format!(
            "candidate is not bit-identical for stage={} unit={}: raw mismatches={}, Snake mismatches={}",
            case.stage,
            case.unit,
            comparison.raw.bit_mismatches,
            comparison.activated.bit_mismatches
        ))
        .into());
    }

    let mut baseline_operation = || {
        Ok(baseline_forward(
            &input_ncl,
            &packed_weight,
            &bias,
            &residual_ncl,
            &alpha,
            case,
        ))
    };
    let mut candidate_operation =
        || candidate_forward(&input_ncl, &packed_weight, &bias, &residual_ncl, &alpha);
    warm_up(args.warmup, case, &mut baseline_operation)?;
    warm_up(args.warmup, case, &mut candidate_operation)?;

    let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::with_capacity(args.trials));
    for trial in 0..args.trials {
        for offset in 0..2 {
            let variant = (trial + offset) % 2;
            let sample = if variant == 0 {
                measure(args.iterations, case, &mut baseline_operation)?
            } else {
                measure(args.iterations, case, &mut candidate_operation)?
            };
            samples[variant].push(sample);
        }
    }
    let baseline = summarize_samples(&samples[0])?;
    let candidate = summarize_samples(&samples[1])?;

    let tensor_bytes = output_bytes(case)?;
    let baseline_boundary_bytes = tensor_bytes
        .checked_mul(9)
        .ok_or_else(|| io::Error::other("baseline boundary traffic overflows usize"))?;
    let candidate_boundary_bytes = tensor_bytes
        .checked_mul(5)
        .ok_or_else(|| io::Error::other("candidate boundary traffic overflows usize"))?;
    let saved_bytes = baseline_boundary_bytes - candidate_boundary_bytes;
    println!(
        "  baseline : median={:10.1} us range=[{:10.1},{:10.1}] full_dispatch=5 boundary_dispatch=4 traffic={:.3} MiB",
        baseline.median_us,
        baseline.min_us,
        baseline.max_us,
        baseline_boundary_bytes as f64 / (1024.0 * 1024.0),
    );
    println!(
        "  candidate: median={:10.1} us range=[{:10.1},{:10.1}] full_dispatch=3 boundary_dispatch=2 traffic={:.3} MiB speedup={:.3}x save={:.1} us/{:.3} MiB",
        candidate.median_us,
        candidate.min_us,
        candidate.max_us,
        candidate_boundary_bytes as f64 / (1024.0 * 1024.0),
        baseline.median_us / candidate.median_us,
        baseline.median_us - candidate.median_us,
        saved_bytes as f64 / (1024.0 * 1024.0),
    );

    Ok(CaseResult {
        case,
        baseline,
        candidate,
        comparison,
    })
}

fn print_static_traffic() -> Result<(), Box<dyn Error>> {
    let elements =
        CASES
            .iter()
            .try_fold(0_usize, |total, case| -> Result<usize, Box<dyn Error>> {
                total.checked_add(output_elements(*case)?).ok_or_else(|| {
                    Box::<dyn Error>::from(io::Error::other(
                        "aggregate element count overflows usize",
                    ))
                })
            })?;
    let tensor_bytes = elements
        .checked_mul(F32_BYTES)
        .ok_or_else(|| io::Error::other("aggregate tensor bytes overflow usize"))?;
    let baseline_bytes = tensor_bytes
        .checked_mul(9)
        .ok_or_else(|| io::Error::other("aggregate baseline traffic overflows usize"))?;
    let candidate_bytes = tensor_bytes
        .checked_mul(5)
        .ok_or_else(|| io::Error::other("aggregate candidate traffic overflows usize"))?;
    let saved_bytes = baseline_bytes - candidate_bytes;
    println!("static twelve-ResidualUnit pointwise-boundary accounting:");
    println!(
        "  elements={elements}, one materialization={tensor_bytes} B ({:.3} MiB)",
        tensor_bytes as f64 / (1024.0 * 1024.0),
    );
    println!(
        "  boundary traffic=9N -> 5N f32: {baseline_bytes} B ({:.3} MiB) -> {candidate_bytes} B ({:.3} MiB), saved={saved_bytes} B ({:.3} MiB)",
        baseline_bytes as f64 / (1024.0 * 1024.0),
        candidate_bytes as f64 / (1024.0 * 1024.0),
        saved_bytes as f64 / (1024.0 * 1024.0),
    );
    println!("  common packed matmul excluded; full dispatch=60 -> 36, boundary dispatch=48 -> 24");
    Ok(())
}

fn print_summary(results: &[CaseResult]) -> Result<(), Box<dyn Error>> {
    if results.len() != CASES.len() {
        return Err(io::Error::other(format!(
            "summary requires {} cases, got {}",
            CASES.len(),
            results.len()
        ))
        .into());
    }
    let baseline_median_us = results
        .iter()
        .map(|result| result.baseline.median_us)
        .sum::<f64>();
    let baseline_min_us = results
        .iter()
        .map(|result| result.baseline.min_us)
        .sum::<f64>();
    let baseline_max_us = results
        .iter()
        .map(|result| result.baseline.max_us)
        .sum::<f64>();
    let candidate_median_us = results
        .iter()
        .map(|result| result.candidate.median_us)
        .sum::<f64>();
    let candidate_min_us = results
        .iter()
        .map(|result| result.candidate.min_us)
        .sum::<f64>();
    let candidate_max_us = results
        .iter()
        .map(|result| result.candidate.max_us)
        .sum::<f64>();
    let raw_bit_mismatches = results
        .iter()
        .map(|result| result.comparison.raw.bit_mismatches)
        .sum::<usize>();
    let activated_bit_mismatches = results
        .iter()
        .map(|result| result.comparison.activated.bit_mismatches)
        .sum::<usize>();
    let raw_max_abs = results
        .iter()
        .map(|result| result.comparison.raw.max_abs)
        .fold(0.0_f32, f32::max);
    let activated_max_abs = results
        .iter()
        .map(|result| result.comparison.activated.max_abs)
        .fold(0.0_f32, f32::max);
    let covered_elements =
        results
            .iter()
            .try_fold(0_usize, |total, result| -> Result<usize, Box<dyn Error>> {
                total
                    .checked_add(output_elements(result.case)?)
                    .ok_or_else(|| {
                        Box::<dyn Error>::from(io::Error::other(
                            "summary element count overflows usize",
                        ))
                    })
            })?;
    println!("twelve-ResidualUnit measured-median summary:");
    println!(
        "  baseline={baseline_median_us:.1} us sum-range=[{baseline_min_us:.1},{baseline_max_us:.1}], candidate={candidate_median_us:.1} us sum-range=[{candidate_min_us:.1},{candidate_max_us:.1}]"
    );
    println!(
        "  save={:.1} us speedup={:.3}x full_dispatch=60 -> 36 covered_elements={covered_elements}",
        baseline_median_us - candidate_median_us,
        baseline_median_us / candidate_median_us,
    );
    println!(
        "  raw bit_mismatch={raw_bit_mismatches} max_abs={raw_max_abs:.9e}; Snake bit_mismatch={activated_bit_mismatches} max_abs={activated_max_abs:.9e}"
    );
    Ok(())
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let (device, monitor) = initialize_wgpu(args.adapter_index);
    B::seed(&device, 0);
    println!(
        "isolated codec pointwise finalizer benchmark: warmup={} iterations={} trials={} cases={}",
        args.warmup,
        args.iterations,
        args.trials,
        CASES.len()
    );
    print_static_traffic()?;
    let mut results = Vec::with_capacity(CASES.len());
    for case in CASES {
        results.push(benchmark_case(&device, case, &args)?);
        synchronize_and_check_wgpu(
            &device,
            &monitor,
            &format!(
                "stage={} unit={} C={} L={} benchmark",
                case.stage, case.unit, case.channels, case.length
            ),
        )?;
    }
    print_summary(&results)?;
    synchronize_and_check_wgpu(&device, &monitor, "benchmark completion")?;
    println!("wgpu_uncaptured_errors=0");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    match parse_args()? {
        ParseOutcome::Run(args) => run(args),
        ParseOutcome::Help => {
            println!("{}", usage());
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_cover_exact_decoder_instances() {
        assert_eq!(CASES.len(), 12);
        for (stage, channels, length) in [
            (0, 768, 600),
            (1, 384, 6_000),
            (2, 192, 48_000),
            (3, 96, 96_000),
        ] {
            let stage_cases = CASES
                .iter()
                .filter(|case| {
                    case.stage == stage && case.channels == channels && case.length == length
                })
                .collect::<Vec<_>>();
            assert_eq!(stage_cases.len(), 3);
            assert!(
                stage_cases
                    .iter()
                    .enumerate()
                    .all(|(unit, case)| case.unit == unit)
            );
        }
    }

    #[test]
    fn static_boundary_accounting_matches_graph() -> Result<(), Box<dyn Error>> {
        let elements =
            CASES
                .iter()
                .try_fold(0_usize, |total, case| -> Result<usize, Box<dyn Error>> {
                    total.checked_add(output_elements(*case)?).ok_or_else(|| {
                        Box::<dyn Error>::from(io::Error::other(
                            "test element count overflows usize",
                        ))
                    })
                })?;
        assert_eq!(elements, 63_590_400);
        let tensor_bytes = elements * F32_BYTES;
        assert_eq!(tensor_bytes, 254_361_600);
        assert_eq!(9 * tensor_bytes, 2_289_254_400);
        assert_eq!(5 * tensor_bytes, 1_271_808_000);
        assert_eq!(4 * tensor_bytes, 1_017_446_400);
        Ok(())
    }
}
