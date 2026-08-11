//! Isolated benchmark for a codec pointwise-finalizer prepared pair.
//!
//! The measured production boundary is:
//! `packed 1x1 matmul -> pointwise residual finalizer -> next act0 Snake`.
//! The candidate keeps the same packed matmul and emits both the raw NCL
//! residual and its Snake-activated NCL tensor from one finalizer dispatch.
//!
//! Only eight graph boundaries qualify: `res0 -> res1` and `res1 -> res2` in
//! each of four decoder blocks. At those boundaries the consumer needs the raw
//! tensor as its identity shortcut and the activated tensor as its k=7 input.
//! The four `res2` outputs are intentionally excluded: three feed the next
//! block's upsample Snake/ConvTranspose and one feeds the WmHead Snake/Conv1d,
//! not another `ResidualUnit` prepared pair.
//!
//! The full paths include the common packed matmul. Traffic accounting covers
//! only the finalizer/Snake boundary so the one eliminated raw-tensor read is
//! explicit. The candidate side calls the same production launcher used by the
//! decoder, making this a direct parity and performance guard.

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
    kernels::{
        pointwise_residual_finalizer::pointwise_residual_finalizer_wgsl,
        pointwise_residual_snake_pair::pointwise_residual_snake_pair_wgsl, snake::snake_wgsl,
    },
};

type B = WgpuRaw;

const DEFAULT_WARMUP: usize = 10;
const DEFAULT_ITERATIONS: usize = 20;
const DEFAULT_TRIALS: usize = 5;
const F32_BYTES: usize = size_of::<f32>();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PairCase {
    stage: usize,
    producer_unit: usize,
    consumer_unit: usize,
    channels: usize,
    length: usize,
}

const CASES: [PairCase; 8] = [
    PairCase {
        stage: 0,
        producer_unit: 0,
        consumer_unit: 1,
        channels: 768,
        length: 600,
    },
    PairCase {
        stage: 0,
        producer_unit: 1,
        consumer_unit: 2,
        channels: 768,
        length: 600,
    },
    PairCase {
        stage: 1,
        producer_unit: 0,
        consumer_unit: 1,
        channels: 384,
        length: 6_000,
    },
    PairCase {
        stage: 1,
        producer_unit: 1,
        consumer_unit: 2,
        channels: 384,
        length: 6_000,
    },
    PairCase {
        stage: 2,
        producer_unit: 0,
        consumer_unit: 1,
        channels: 192,
        length: 48_000,
    },
    PairCase {
        stage: 2,
        producer_unit: 1,
        consumer_unit: 2,
        channels: 192,
        length: 48_000,
    },
    PairCase {
        stage: 3,
        producer_unit: 0,
        consumer_unit: 1,
        channels: 96,
        length: 96_000,
    },
    PairCase {
        stage: 3,
        producer_unit: 1,
        consumer_unit: 2,
        channels: 96,
        length: 96_000,
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExcludedBoundary {
    stage: usize,
    producer_unit: usize,
    next_consumer: &'static str,
}

const EXCLUDED_BOUNDARIES: [ExcludedBoundary; 4] = [
    ExcludedBoundary {
        stage: 0,
        producer_unit: 2,
        next_consumer: "decoder block 1 upsample Snake -> ConvTranspose",
    },
    ExcludedBoundary {
        stage: 1,
        producer_unit: 2,
        next_consumer: "decoder block 2 upsample Snake -> ConvTranspose",
    },
    ExcludedBoundary {
        stage: 2,
        producer_unit: 2,
        next_consumer: "decoder block 3 upsample Snake -> ConvTranspose",
    },
    ExcludedBoundary {
        stage: 3,
        producer_unit: 2,
        next_consumer: "WmHead Snake -> Conv1d -> Tanh",
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
    finite: bool,
}

#[derive(Clone, Copy, Debug)]
struct PairComparison {
    raw: TensorComparison,
    activated: TensorComparison,
}

#[derive(Debug)]
struct CaseResult {
    case: PairCase,
    baseline: Timing,
    candidate: Timing,
    comparison: PairComparison,
}

#[derive(Clone)]
struct PairOutputs {
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
    "usage: bench_codec_pointwise_snake_pair <adapter-index> [--warmup N] \
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
    next_alpha: &Tensor<B, 3>,
) -> Result<PairOutputs, Box<dyn Error>> {
    let branch_nlc = pointwise_matmul(input_ncl, packed_weight);
    let raw = pointwise_residual_finalizer_wgsl(
        branch_nlc.into_primitive().tensor(),
        bias.clone().into_primitive().tensor(),
        residual_ncl.clone().into_primitive().tensor(),
    )?;
    let raw: Tensor<B, 3> = Tensor::from_primitive(TensorPrimitive::Float(raw));
    let activated = snake_wgsl(
        raw.clone().into_primitive().tensor(),
        next_alpha.clone().into_primitive().tensor(),
    );
    Ok(PairOutputs {
        raw,
        activated: Tensor::from_primitive(TensorPrimitive::Float(activated)),
    })
}

fn candidate_forward(
    input_ncl: &Tensor<B, 3>,
    packed_weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    residual_ncl: &Tensor<B, 3>,
    next_alpha: &Tensor<B, 3>,
) -> Result<PairOutputs, Box<dyn Error>> {
    let branch_nlc = pointwise_matmul(input_ncl, packed_weight);
    let pair = pointwise_residual_snake_pair_wgsl(
        branch_nlc.into_primitive().tensor(),
        bias.clone().into_primitive().tensor(),
        residual_ncl.clone().into_primitive().tensor(),
        next_alpha.clone().into_primitive().tensor(),
    )?;
    let (raw, activated) = pair.into_tensors();
    Ok(PairOutputs {
        raw: Tensor::from_primitive(TensorPrimitive::Float(raw)),
        activated: Tensor::from_primitive(TensorPrimitive::Float(activated)),
    })
}

fn compare_tensor(
    label: &str,
    expected: Tensor<B, 3>,
    actual: Tensor<B, 3>,
) -> Result<TensorComparison, Box<dyn Error>> {
    let expected_shape = expected.dims();
    let actual_shape = actual.dims();
    if expected_shape != actual_shape {
        return Err(io::Error::other(format!(
            "{label} shape mismatch: baseline={expected_shape:?}, candidate={actual_shape:?}"
        ))
        .into());
    }
    let expected = expected.into_data().to_vec::<f32>()?;
    let actual = actual.into_data().to_vec::<f32>()?;
    if expected.len() != actual.len() {
        return Err(io::Error::other(format!(
            "{label} length mismatch: baseline={}, candidate={}",
            expected.len(),
            actual.len()
        ))
        .into());
    }

    let mut bit_mismatches = 0_usize;
    let mut max_abs = 0.0_f32;
    for (index, (&expected, &actual)) in expected.iter().zip(&actual).enumerate() {
        if !expected.is_finite() || !actual.is_finite() {
            return Err(io::Error::other(format!(
                "{label} contains non-finite data at {index}: baseline={expected:?}, candidate={actual:?}"
            ))
            .into());
        }
        if expected.to_bits() != actual.to_bits() {
            bit_mismatches += 1;
        }
        max_abs = max_abs.max((expected - actual).abs());
    }
    if !max_abs.is_finite() {
        return Err(io::Error::other(format!("{label} max_abs is non-finite")).into());
    }
    Ok(TensorComparison {
        elements: expected.len(),
        bit_mismatches,
        max_abs,
        finite: true,
    })
}

fn compare_outputs(
    baseline: PairOutputs,
    candidate: PairOutputs,
) -> Result<PairComparison, Box<dyn Error>> {
    Ok(PairComparison {
        raw: compare_tensor("raw residual", baseline.raw, candidate.raw)?,
        activated: compare_tensor("activated Snake", baseline.activated, candidate.activated)?,
    })
}

fn synchronize(outputs: PairOutputs, case: PairCase) -> Result<(), Box<dyn Error>> {
    let expected_shape = [1, case.channels, case.length];
    if outputs.raw.dims() != expected_shape || outputs.activated.dims() != expected_shape {
        return Err(io::Error::other(format!(
            "output shape mismatch for stage={} {}->{}: expected {expected_shape:?}",
            case.stage, case.producer_unit, case.consumer_unit
        ))
        .into());
    }
    // Keep `raw` alive through the synchronization because the next
    // ResidualUnit retains it as its shortcut. Reading one activated scalar
    // synchronizes the shared queue without adding a second readback to both
    // timed variants; full raw/activated finiteness is checked before timing.
    let _raw = outputs.raw;
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
        .ok_or_else(|| io::Error::other("activated synchronization was empty"))?;
    if !value.is_finite() {
        return Err(io::Error::other(format!(
            "activated synchronization value is non-finite: {value:?}"
        ))
        .into());
    }
    Ok(())
}

fn warm_up<F>(count: usize, case: PairCase, operation: &mut F) -> Result<(), Box<dyn Error>>
where
    F: FnMut() -> Result<PairOutputs, Box<dyn Error>>,
{
    let mut output = None;
    for _ in 0..count {
        output = Some(operation()?);
    }
    synchronize(
        output.ok_or_else(|| io::Error::other("warmup count must be non-zero"))?,
        case,
    )
}

fn measure<F>(iterations: usize, case: PairCase, operation: &mut F) -> Result<f64, Box<dyn Error>>
where
    F: FnMut() -> Result<PairOutputs, Box<dyn Error>>,
{
    let started = Instant::now();
    let mut output = None;
    for _ in 0..iterations {
        output = Some(operation()?);
    }
    synchronize(
        output.ok_or_else(|| io::Error::other("iteration count must be non-zero"))?,
        case,
    )?;
    Ok(started.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64)
}

fn summarize(samples: &[f64]) -> Result<Timing, Box<dyn Error>> {
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

fn elements(case: PairCase) -> Result<usize, Box<dyn Error>> {
    case.channels
        .checked_mul(case.length)
        .ok_or_else(|| io::Error::other("C*L overflows usize").into())
}

fn tensor_bytes(case: PairCase) -> Result<usize, Box<dyn Error>> {
    elements(case)?
        .checked_mul(F32_BYTES)
        .ok_or_else(|| io::Error::other("tensor byte count overflows usize").into())
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn print_case_accounting(case: PairCase) -> Result<(), Box<dyn Error>> {
    let bytes = tensor_bytes(case)?;
    let baseline_traffic = bytes
        .checked_mul(7)
        .ok_or_else(|| io::Error::other("baseline traffic overflows usize"))?;
    let candidate_traffic = bytes
        .checked_mul(6)
        .ok_or_else(|| io::Error::other("candidate traffic overflows usize"))?;
    let pair_live = bytes
        .checked_mul(2)
        .ok_or_else(|| io::Error::other("prepared-pair bytes overflow usize"))?;
    println!(
        "  boundary accounting: dispatch=2 -> 1 (full incl. matmul=3 -> 2); logical traffic=7N -> 6N f32 ({:.3} -> {:.3} MiB), saved raw read={:.3} MiB",
        mib(baseline_traffic),
        mib(candidate_traffic),
        mib(bytes),
    );
    println!(
        "  live outputs: raw+activated={} bytes ({:.3} MiB) for both paths; persistent delta=0; unique bias+alpha={} bytes",
        pair_live,
        mib(pair_live),
        2 * case.channels * F32_BYTES,
    );
    Ok(())
}

fn benchmark_case(
    device: &<B as Backend>::Device,
    case: PairCase,
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
    let next_alpha = Tensor::<B, 3>::random(
        [1, case.channels, 1],
        Distribution::Uniform(0.25, 2.0),
        device,
    );

    let baseline_output = baseline_forward(
        &input_ncl,
        &packed_weight,
        &bias,
        &residual_ncl,
        &next_alpha,
    )?;
    let candidate_output = candidate_forward(
        &input_ncl,
        &packed_weight,
        &bias,
        &residual_ncl,
        &next_alpha,
    )?;
    let comparison = compare_outputs(baseline_output, candidate_output)?;
    println!(
        "stage={} res{}->res{} C={:4} L={:6}",
        case.stage, case.producer_unit, case.consumer_unit, case.channels, case.length
    );
    println!(
        "  correctness raw: elements={} bit_mismatch={} max_abs={:.9e} finite={}; activated: elements={} bit_mismatch={} max_abs={:.9e} finite={}",
        comparison.raw.elements,
        comparison.raw.bit_mismatches,
        comparison.raw.max_abs,
        comparison.raw.finite,
        comparison.activated.elements,
        comparison.activated.bit_mismatches,
        comparison.activated.max_abs,
        comparison.activated.finite,
    );
    print_case_accounting(case)?;

    let mut baseline_operation = || {
        baseline_forward(
            &input_ncl,
            &packed_weight,
            &bias,
            &residual_ncl,
            &next_alpha,
        )
    };
    let mut candidate_operation = || {
        candidate_forward(
            &input_ncl,
            &packed_weight,
            &bias,
            &residual_ncl,
            &next_alpha,
        )
    };
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
    let baseline = summarize(&samples[0])?;
    let candidate = summarize(&samples[1])?;
    println!(
        "  baseline : median={:10.3} us range=[{:10.3},{:10.3}]",
        baseline.median_us, baseline.min_us, baseline.max_us
    );
    println!(
        "  candidate: median={:10.3} us range=[{:10.3},{:10.3}] speedup={:.3}x saving={:.3} us",
        candidate.median_us,
        candidate.min_us,
        candidate.max_us,
        baseline.median_us / candidate.median_us,
        baseline.median_us - candidate.median_us,
    );

    Ok(CaseResult {
        case,
        baseline,
        candidate,
        comparison,
    })
}

fn aggregate_elements() -> Result<usize, Box<dyn Error>> {
    CASES
        .iter()
        .try_fold(0_usize, |total, case| -> Result<usize, Box<dyn Error>> {
            total.checked_add(elements(*case)?).ok_or_else(|| {
                Box::<dyn Error>::from(io::Error::other("aggregate element count overflows usize"))
            })
        })
}

fn print_graph_and_static_accounting() -> Result<(), Box<dyn Error>> {
    println!("eligible prepared-pair boundaries: {}", CASES.len());
    for case in CASES {
        println!(
            "  include stage={} res{}->res{} C={} L={}",
            case.stage, case.producer_unit, case.consumer_unit, case.channels, case.length
        );
    }
    println!("excluded pointwise-finalizer boundaries:");
    for boundary in EXCLUDED_BOUNDARIES {
        println!(
            "  exclude stage={} res{} -> {}",
            boundary.stage, boundary.producer_unit, boundary.next_consumer
        );
    }

    let elements = aggregate_elements()?;
    let bytes = elements
        .checked_mul(F32_BYTES)
        .ok_or_else(|| io::Error::other("aggregate tensor byte count overflows usize"))?;
    let baseline_traffic = bytes
        .checked_mul(7)
        .ok_or_else(|| io::Error::other("aggregate baseline traffic overflows usize"))?;
    let candidate_traffic = bytes
        .checked_mul(6)
        .ok_or_else(|| io::Error::other("aggregate candidate traffic overflows usize"))?;
    println!("static eligible-eight accounting:");
    println!(
        "  elements={elements}, one raw materialization={bytes} bytes ({:.3} MiB)",
        mib(bytes)
    );
    println!(
        "  boundary traffic: {baseline_traffic} -> {candidate_traffic} bytes ({:.3} -> {:.3} MiB), saved={} bytes ({:.3} MiB)",
        mib(baseline_traffic),
        mib(candidate_traffic),
        baseline_traffic - candidate_traffic,
        mib(baseline_traffic - candidate_traffic),
    );
    println!(
        "  dispatch: boundary 16 -> 8; full including common packed matmul 24 -> 16; persistent bytes +0"
    );
    println!(
        "  peak observable output pair is 2N: largest shape C=192/96 has {} bytes ({:.3} MiB), unchanged between paths",
        2 * 9_216_000 * F32_BYTES,
        mib(2 * 9_216_000 * F32_BYTES),
    );
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
    let baseline_median = results
        .iter()
        .map(|result| result.baseline.median_us)
        .sum::<f64>();
    let baseline_min = results
        .iter()
        .map(|result| result.baseline.min_us)
        .sum::<f64>();
    let baseline_max = results
        .iter()
        .map(|result| result.baseline.max_us)
        .sum::<f64>();
    let candidate_median = results
        .iter()
        .map(|result| result.candidate.median_us)
        .sum::<f64>();
    let candidate_min = results
        .iter()
        .map(|result| result.candidate.min_us)
        .sum::<f64>();
    let candidate_max = results
        .iter()
        .map(|result| result.candidate.max_us)
        .sum::<f64>();
    let raw_mismatches = results
        .iter()
        .map(|result| result.comparison.raw.bit_mismatches)
        .sum::<usize>();
    let activated_mismatches = results
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
    let finite = results
        .iter()
        .all(|result| result.comparison.raw.finite && result.comparison.activated.finite);
    let covered_elements =
        results
            .iter()
            .try_fold(0_usize, |total, result| -> Result<usize, Box<dyn Error>> {
                total.checked_add(elements(result.case)?).ok_or_else(|| {
                    Box::<dyn Error>::from(io::Error::other(
                        "summary element count overflows usize",
                    ))
                })
            })?;

    println!("eligible-eight measured-median summary:");
    println!(
        "  baseline={baseline_median:.3} us sum-range=[{baseline_min:.3},{baseline_max:.3}], candidate={candidate_median:.3} us sum-range=[{candidate_min:.3},{candidate_max:.3}]"
    );
    println!(
        "  saving={:.3} us speedup={:.3}x full dispatch=24 -> 16 covered_elements={covered_elements}",
        baseline_median - candidate_median,
        baseline_median / candidate_median,
    );
    println!(
        "  raw bit_mismatch={raw_mismatches} max_abs={raw_max_abs:.9e}; activated bit_mismatch={activated_mismatches} max_abs={activated_max_abs:.9e}; finite={finite}"
    );
    Ok(())
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let (device, monitor) = initialize_wgpu(args.adapter_index);
    B::seed(&device, 0);
    println!(
        "isolated pointwise residual+next-Snake pair: warmup={} iterations={} trials={} eligible_cases={}",
        args.warmup,
        args.iterations,
        args.trials,
        CASES.len()
    );
    print_graph_and_static_accounting()?;

    let mut results = Vec::with_capacity(CASES.len());
    for case in CASES {
        results.push(benchmark_case(&device, case, &args)?);
        synchronize_and_check_wgpu(
            &device,
            &monitor,
            &format!(
                "stage={} res{}->res{} C={} L={} benchmark",
                case.stage, case.producer_unit, case.consumer_unit, case.channels, case.length
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
    fn cases_cover_only_next_residual_unit_prepared_pairs() {
        assert_eq!(CASES.len(), 8);
        for (stage, channels, length) in [
            (0, 768, 600),
            (1, 384, 6_000),
            (2, 192, 48_000),
            (3, 96, 96_000),
        ] {
            let stage_cases = CASES
                .into_iter()
                .filter(|case| {
                    case.stage == stage && case.channels == channels && case.length == length
                })
                .collect::<Vec<_>>();
            assert_eq!(stage_cases.len(), 2);
            assert_eq!(stage_cases[0].producer_unit, 0);
            assert_eq!(stage_cases[0].consumer_unit, 1);
            assert_eq!(stage_cases[1].producer_unit, 1);
            assert_eq!(stage_cases[1].consumer_unit, 2);
        }
        assert_eq!(EXCLUDED_BOUNDARIES.len(), 4);
        assert!(
            EXCLUDED_BOUNDARIES
                .into_iter()
                .all(|boundary| boundary.producer_unit == 2)
        );
    }

    #[test]
    fn defaults_match_measurement_contract() {
        assert_eq!(DEFAULT_WARMUP, 10);
        assert_eq!(DEFAULT_ITERATIONS, 20);
        assert_eq!(DEFAULT_TRIALS, 5);
    }

    #[test]
    fn static_accounting_matches_eight_boundaries() -> Result<(), Box<dyn Error>> {
        let elements = aggregate_elements()?;
        assert_eq!(elements, 42_393_600);
        let bytes = elements * F32_BYTES;
        assert_eq!(bytes, 169_574_400);
        assert_eq!(7 * bytes, 1_187_020_800);
        assert_eq!(6 * bytes, 1_017_446_400);
        assert_eq!(7 * bytes - 6 * bytes, bytes);
        assert_eq!(CASES.len() * 3, 24);
        assert_eq!(CASES.len() * 2, 16);
        Ok(())
    }
}
