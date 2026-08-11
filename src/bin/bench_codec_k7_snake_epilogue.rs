//! Measure an isolated fusion of ResidualUnit `k7 Conv1d -> act1 Snake1d`.
//!
//! The baseline is the current production tile selector followed by the
//! standalone `snake.wgsl` launcher. The candidate keeps the same convolution
//! tile and reduction order but evaluates the identical Snake expression in
//! the output-store epilogue. Production modules and routing remain untouched.
//!
//! Once this binary is explicitly registered, run all twelve exact decoder
//! shapes with:
//! `cargo run --release --bin bench_codec_k7_snake_epilogue -- 0`

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
        conv1d_k7_snake_epilogue::{
            Conv1dK7SnakeTile, conv1d_k7_same_snake_epilogue_wgsl,
            device_supports_conv1d_k7_snake_epilogue,
        },
        conv1d_k7_tiled::{Conv1dK7Dilation, conv1d_k7_same_tiled_wgsl},
        conv1d_k7_tiled_o32::{conv1d_k7_same_tiled_o32_wgsl, device_supports_conv1d_k7_tiled_o32},
        snake::snake_wgsl,
    },
};

type B = WgpuRaw;

const KERNEL_SIZE: usize = 7;
const SNAKE_WORKGROUP_SIZE: usize = 256;
const DEFAULT_WARMUP: usize = 3;
const DEFAULT_ITERATIONS: usize = 5;
const DEFAULT_TRIALS: usize = 7;
const MAX_ABS_TOLERANCE: f32 = 2.0e-6;
const F32_BYTES: usize = core::mem::size_of::<f32>();

// Same-card PyTorch FP32 measurement captured on 2026-08-10 in
// /tmp/irodori-python-fp32.json (PyTorch 2.10.0+cu128, RTX 3060 Ti PCI 07).
// Upstream Irodori-v4 has no torch.jit/TorchScript codec path; these are eager
// and torch.compile(fullgraph) values and must not be labelled TorchScript.
const PYTORCH_EAGER_SNAKE_C96_T96000_US: f64 = 992.225_265_502_929_7;
const PYTORCH_COMPILED_SNAKE_C96_T96000_US: f64 = 180.551_681_518_554_7;
const PYTORCH_COMPILE_FIRST_CALL_MS: f64 = 322.951_43;
const PYTORCH_COMPILED_MAX_ABS: f32 = 4.768_371_6e-7;

#[derive(Clone, Copy, Debug)]
struct ConvCase {
    channels: usize,
    length: usize,
    dilation: Conv1dK7Dilation,
}

const CASES: [ConvCase; 12] = [
    ConvCase {
        channels: 768,
        length: 600,
        dilation: Conv1dK7Dilation::One,
    },
    ConvCase {
        channels: 768,
        length: 600,
        dilation: Conv1dK7Dilation::Three,
    },
    ConvCase {
        channels: 768,
        length: 600,
        dilation: Conv1dK7Dilation::Nine,
    },
    ConvCase {
        channels: 384,
        length: 6_000,
        dilation: Conv1dK7Dilation::One,
    },
    ConvCase {
        channels: 384,
        length: 6_000,
        dilation: Conv1dK7Dilation::Three,
    },
    ConvCase {
        channels: 384,
        length: 6_000,
        dilation: Conv1dK7Dilation::Nine,
    },
    ConvCase {
        channels: 192,
        length: 48_000,
        dilation: Conv1dK7Dilation::One,
    },
    ConvCase {
        channels: 192,
        length: 48_000,
        dilation: Conv1dK7Dilation::Three,
    },
    ConvCase {
        channels: 192,
        length: 48_000,
        dilation: Conv1dK7Dilation::Nine,
    },
    ConvCase {
        channels: 96,
        length: 96_000,
        dilation: Conv1dK7Dilation::One,
    },
    ConvCase {
        channels: 96,
        length: 96_000,
        dilation: Conv1dK7Dilation::Three,
    },
    ConvCase {
        channels: 96,
        length: 96_000,
        dilation: Conv1dK7Dilation::Nine,
    },
];

#[derive(Debug)]
struct Args {
    adapter_index: usize,
    warmup: usize,
    iterations: usize,
    trials: usize,
}

#[derive(Clone, Copy, Debug)]
struct Timing {
    median_us: f64,
    min_us: f64,
    max_us: f64,
}

#[derive(Clone, Copy, Debug)]
struct Comparison {
    elements: usize,
    mismatched_bits: usize,
    max_abs: f32,
}

#[derive(Clone, Copy, Debug)]
struct CaseResult {
    case: ConvCase,
    baseline_tile: Conv1dK7SnakeTile,
    candidate_tile: Conv1dK7SnakeTile,
    baseline: Timing,
    candidate: Timing,
    comparison: Comparison,
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
    "usage: bench_codec_k7_snake_epilogue <adapter-index> [--warmup N] \
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

fn parse_args() -> Result<Args, Box<dyn Error>> {
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
            "--help" | "-h" => {
                println!("{}", usage());
                std::process::exit(0);
            }
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
    Ok(Args {
        adapter_index: adapter_index
            .ok_or_else(|| io::Error::other(format!("missing adapter index; {}", usage())))?,
        warmup,
        iterations,
        trials,
    })
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

const fn prefers_o32(case: ConvCase) -> bool {
    !(case.channels == 768 && matches!(case.dilation, Conv1dK7Dilation::Three))
}

fn baseline_tile(input: &Tensor<B, 3>, case: ConvCase) -> Conv1dK7SnakeTile {
    if !prefers_o32(case) {
        return Conv1dK7SnakeTile::Output16;
    }
    let input = input.clone().into_primitive().tensor();
    if device_supports_conv1d_k7_tiled_o32(&input, case.dilation) {
        Conv1dK7SnakeTile::Output32
    } else {
        Conv1dK7SnakeTile::Output16
    }
}

fn candidate_tile(
    input: &Tensor<B, 3>,
    case: ConvCase,
) -> Result<Conv1dK7SnakeTile, Box<dyn Error>> {
    let input = input.clone().into_primitive().tensor();
    if prefers_o32(case)
        && device_supports_conv1d_k7_snake_epilogue(
            &input,
            case.dilation,
            Conv1dK7SnakeTile::Output32,
        )
    {
        return Ok(Conv1dK7SnakeTile::Output32);
    }
    if device_supports_conv1d_k7_snake_epilogue(&input, case.dilation, Conv1dK7SnakeTile::Output16)
    {
        return Ok(Conv1dK7SnakeTile::Output16);
    }
    let hardware = &input.client.properties().hardware;
    Err(io::Error::other(format!(
        "fused candidate unavailable for C={} L={} d={} (max_bindings={}, max_shared={} B, max_units={}): production must retain accepted k7 -> Snake fallback",
        case.channels,
        case.length,
        case.dilation.value(),
        hardware.max_bindings,
        hardware.max_shared_memory_size,
        hardware.max_units_per_cube,
    ))
    .into())
}

fn baseline_forward(
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    alpha: &Tensor<B, 3>,
    dilation: Conv1dK7Dilation,
    tile: Conv1dK7SnakeTile,
) -> Tensor<B, 3> {
    let input = input.clone().into_primitive().tensor();
    let weight = weight.clone().into_primitive().tensor();
    let bias = bias.clone().into_primitive().tensor();
    let conv_output = match tile {
        Conv1dK7SnakeTile::Output16 => conv1d_k7_same_tiled_wgsl(input, weight, bias, dilation),
        Conv1dK7SnakeTile::Output32 => conv1d_k7_same_tiled_o32_wgsl(input, weight, bias, dilation),
    };
    let output = snake_wgsl(conv_output, alpha.clone().into_primitive().tensor());
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn candidate_forward(
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    alpha: &Tensor<B, 3>,
    dilation: Conv1dK7Dilation,
    tile: Conv1dK7SnakeTile,
) -> Tensor<B, 3> {
    let output = conv1d_k7_same_snake_epilogue_wgsl(
        input.clone().into_primitive().tensor(),
        weight.clone().into_primitive().tensor(),
        bias.clone().into_primitive().tensor(),
        alpha.clone().into_primitive().tensor(),
        dilation,
        tile,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn compare_outputs(
    baseline: Tensor<B, 3>,
    candidate: Tensor<B, 3>,
) -> Result<Comparison, Box<dyn Error>> {
    let baseline = baseline.into_data().to_vec::<f32>()?;
    let candidate = candidate.into_data().to_vec::<f32>()?;
    if baseline.len() != candidate.len() {
        return Err(io::Error::other(format!(
            "output length mismatch: baseline={} candidate={}",
            baseline.len(),
            candidate.len()
        ))
        .into());
    }
    let mut mismatched_bits = 0;
    let mut max_abs = 0.0_f32;
    for (&expected, &actual) in baseline.iter().zip(&candidate) {
        if !expected.is_finite() || !actual.is_finite() {
            return Err(io::Error::other(format!(
                "non-finite output pair: baseline={expected:?} candidate={actual:?}"
            ))
            .into());
        }
        if expected.to_bits() != actual.to_bits() {
            mismatched_bits += 1;
        }
        max_abs = max_abs.max((expected - actual).abs());
    }
    if !max_abs.is_finite() || max_abs > MAX_ABS_TOLERANCE {
        return Err(io::Error::other(format!(
            "fused candidate max_abs={max_abs:.9e} exceeds {MAX_ABS_TOLERANCE:.9e}"
        ))
        .into());
    }
    Ok(Comparison {
        elements: baseline.len(),
        mismatched_bits,
        max_abs,
    })
}

fn synchronize(output: Tensor<B, 3>, case: ConvCase) {
    let _ = output
        .slice([
            0..1,
            case.channels - 1..case.channels,
            case.length - 1..case.length,
        ])
        .into_data();
}

fn warm_up<F>(count: usize, case: ConvCase, operation: &mut F)
where
    F: FnMut() -> Tensor<B, 3>,
{
    let output = (0..count)
        .map(|_| operation())
        .reduce(|_, output| output)
        .expect("warmup count must be non-zero");
    synchronize(output, case);
}

fn measure<F>(iterations: usize, case: ConvCase, operation: &mut F) -> f64
where
    F: FnMut() -> Tensor<B, 3>,
{
    let started = Instant::now();
    let output = (0..iterations)
        .map(|_| operation())
        .reduce(|_, output| output)
        .expect("iteration count must be non-zero");
    synchronize(output, case);
    started.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64
}

fn summarize_samples(samples: &[f64]) -> Timing {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    Timing {
        median_us: sorted[sorted.len() / 2],
        min_us: sorted[0],
        max_us: sorted[sorted.len() - 1],
    }
}

fn output_elements(case: ConvCase) -> usize {
    case.channels
        .checked_mul(case.length)
        .expect("official output element count must not overflow")
}

fn output_bytes(case: ConvCase) -> usize {
    output_elements(case)
        .checked_mul(F32_BYTES)
        .expect("official output byte count must not overflow")
}

fn useful_macs(case: ConvCase) -> u128 {
    let length = case.length as u128;
    let dilation = case.dilation.value() as u128;
    let valid_positions = (0..KERNEL_SIZE)
        .map(|kernel_index| {
            let offset = (kernel_index as i128 - 3) * dilation as i128;
            length.saturating_sub(offset.unsigned_abs())
        })
        .sum::<u128>();
    (case.channels as u128).pow(2) * valid_positions
}

fn conv_workgroups(case: ConvCase, tile: Conv1dK7SnakeTile) -> usize {
    case.length.div_ceil(64) * (case.channels / tile.output_channel_tile())
}

fn snake_workgroups(case: ConvCase) -> usize {
    output_elements(case).div_ceil(SNAKE_WORKGROUP_SIZE)
}

fn benchmark_case(
    device: &<B as Backend>::Device,
    case: ConvCase,
    args: &Args,
) -> Result<CaseResult, Box<dyn Error>> {
    let input = Tensor::<B, 3>::random(
        [1, case.channels, case.length],
        Distribution::Uniform(-1.0, 1.0),
        device,
    );
    let weight = Tensor::<B, 3>::random(
        [case.channels, case.channels, KERNEL_SIZE],
        Distribution::Uniform(-0.025, 0.025),
        device,
    );
    let bias = Tensor::<B, 1>::random([case.channels], Distribution::Uniform(-0.05, 0.05), device);
    let alpha = Tensor::<B, 3>::random(
        [1, case.channels, 1],
        Distribution::Uniform(0.25, 2.0),
        device,
    );
    let baseline_tile = baseline_tile(&input, case);
    let candidate_tile = candidate_tile(&input, case)?;
    let raw_input = input.clone().into_primitive().tensor();
    let hardware = &raw_input.client.properties().hardware;
    let max_bindings = hardware.max_bindings;
    let max_shared_memory_size = hardware.max_shared_memory_size;
    let baseline_o32_supported = device_supports_conv1d_k7_tiled_o32(&raw_input, case.dilation);
    let fused_o16_supported = device_supports_conv1d_k7_snake_epilogue(
        &raw_input,
        case.dilation,
        Conv1dK7SnakeTile::Output16,
    );
    let fused_o32_supported = device_supports_conv1d_k7_snake_epilogue(
        &raw_input,
        case.dilation,
        Conv1dK7SnakeTile::Output32,
    );

    // First launches compile both exact shaders and measure correctness outside
    // the timed samples.
    let expected = baseline_forward(&input, &weight, &bias, &alpha, case.dilation, baseline_tile);
    let actual = candidate_forward(
        &input,
        &weight,
        &bias,
        &alpha,
        case.dilation,
        candidate_tile,
    );
    let comparison = compare_outputs(expected, actual)?;

    let mut baseline_operation =
        || baseline_forward(&input, &weight, &bias, &alpha, case.dilation, baseline_tile);
    let mut candidate_operation = || {
        candidate_forward(
            &input,
            &weight,
            &bias,
            &alpha,
            case.dilation,
            candidate_tile,
        )
    };
    warm_up(args.warmup, case, &mut baseline_operation);
    warm_up(args.warmup, case, &mut candidate_operation);

    let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::with_capacity(args.trials));
    for trial in 0..args.trials {
        for offset in 0..2 {
            let variant = (trial + offset) % 2;
            let sample = if variant == 0 {
                measure(args.iterations, case, &mut baseline_operation)
            } else {
                measure(args.iterations, case, &mut candidate_operation)
            };
            samples[variant].push(sample);
        }
    }
    let baseline = summarize_samples(&samples[0]);
    let candidate = summarize_samples(&samples[1]);

    let bytes = output_bytes(case);
    let baseline_boundary_bytes = 3 * bytes;
    let candidate_boundary_bytes = bytes;
    let saved_bytes = baseline_boundary_bytes - candidate_boundary_bytes;
    println!(
        "C={:4} L={:6} d={} MAC={:.6}G routes={}+Snake -> {}",
        case.channels,
        case.length,
        case.dilation.value(),
        useful_macs(case) as f64 / 1.0e9,
        baseline_tile.label(),
        candidate_tile.label(),
    );
    println!(
        "  device: max_bindings={max_bindings} max_shared={max_shared_memory_size}B current-o32={baseline_o32_supported} fused5-o16={fused_o16_supported} fused5-o32={fused_o32_supported} selected-shared={}B",
        candidate_tile.shared_memory_bytes(case.dilation),
    );
    println!(
        "  correctness: elements={} bit_mismatch={} max_abs={:.9e}",
        comparison.elements, comparison.mismatched_bits, comparison.max_abs
    );
    println!(
        "  baseline : median={:10.1} us range=[{:10.1},{:10.1}] dispatch=2 workgroups={}+{} boundary={:.3} MiB",
        baseline.median_us,
        baseline.min_us,
        baseline.max_us,
        conv_workgroups(case, baseline_tile),
        snake_workgroups(case),
        baseline_boundary_bytes as f64 / (1024.0 * 1024.0),
    );
    println!(
        "  epilogue : median={:10.1} us range=[{:10.1},{:10.1}] dispatch=1 workgroups={} boundary={:.3} MiB speedup={:.3}x save={:.1} us/{:.3} MiB",
        candidate.median_us,
        candidate.min_us,
        candidate.max_us,
        conv_workgroups(case, candidate_tile),
        candidate_boundary_bytes as f64 / (1024.0 * 1024.0),
        baseline.median_us / candidate.median_us,
        baseline.median_us - candidate.median_us,
        saved_bytes as f64 / (1024.0 * 1024.0),
    );

    Ok(CaseResult {
        case,
        baseline_tile,
        candidate_tile,
        baseline,
        candidate,
        comparison,
    })
}

fn print_static_traffic() {
    let elements = CASES.iter().copied().map(output_elements).sum::<usize>();
    let materialized_intermediate_bytes = elements * F32_BYTES;
    let baseline_boundary_bytes = 3 * materialized_intermediate_bytes;
    let candidate_boundary_bytes = materialized_intermediate_bytes;
    let saved_bytes = baseline_boundary_bytes - candidate_boundary_bytes;
    let removed_workgroups = CASES.iter().copied().map(snake_workgroups).sum::<usize>();
    println!("static twelve-ResidualUnit boundary accounting:");
    println!(
        "  baseline dispatch=24, epilogue dispatch=12, removed Snake dispatch=12, removed Snake workgroups={removed_workgroups}"
    );
    println!(
        "  output elements={elements}, materialized intermediate={materialized_intermediate_bytes} B ({:.3} MiB)",
        materialized_intermediate_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  boundary traffic baseline={baseline_boundary_bytes} B ({:.3} MiB), epilogue={candidate_boundary_bytes} B ({:.3} MiB), saved={saved_bytes} B ({:.3} MiB)",
        baseline_boundary_bytes as f64 / (1024.0 * 1024.0),
        candidate_boundary_bytes as f64 / (1024.0 * 1024.0),
        saved_bytes as f64 / (1024.0 * 1024.0),
    );
    let peak_extra_bytes = CASES
        .iter()
        .copied()
        .map(output_bytes)
        .max()
        .expect("CASES must be non-empty");
    println!(
        "  cumulative materialization above; peak extra live intermediate per unit={peak_extra_bytes} B ({:.3} MiB); common alpha reads are excluded",
        peak_extra_bytes as f64 / (1024.0 * 1024.0),
    );
}

fn print_pytorch_characteristic() {
    println!("PyTorch FP32 Snake characteristic (same RTX 3060 Ti, B1/C96/T96000):");
    println!(
        "  eager={PYTORCH_EAGER_SNAKE_C96_T96000_US:.3} us, torch.compile(fullgraph)={PYTORCH_COMPILED_SNAKE_C96_T96000_US:.3} us ({:.3}x), compiled max_abs={PYTORCH_COMPILED_MAX_ABS:.9e}",
        PYTORCH_EAGER_SNAKE_C96_T96000_US / PYTORCH_COMPILED_SNAKE_C96_T96000_US,
    );
    println!(
        "  first compile={PYTORCH_COMPILE_FIRST_CALL_MS:.3} ms; upstream exposes eager/torch.compile, not a TorchScript/torch.jit codec route"
    );
    println!(
        "  implication: graph compilation fuses Snake elementwise work, but only the Conv epilogue removes its input materialization and second dispatch"
    );
}

fn print_summary(results: &[CaseResult]) {
    let baseline_us = results
        .iter()
        .map(|result| result.baseline.median_us)
        .sum::<f64>();
    let candidate_us = results
        .iter()
        .map(|result| result.candidate.median_us)
        .sum::<f64>();
    let bit_mismatches = results
        .iter()
        .map(|result| result.comparison.mismatched_bits)
        .sum::<usize>();
    let max_abs = results
        .iter()
        .map(|result| result.comparison.max_abs)
        .fold(0.0_f32, f32::max);
    let baseline_workgroups = results
        .iter()
        .map(|result| {
            conv_workgroups(result.case, result.baseline_tile) + snake_workgroups(result.case)
        })
        .sum::<usize>();
    let candidate_workgroups = results
        .iter()
        .map(|result| conv_workgroups(result.case, result.candidate_tile))
        .sum::<usize>();
    println!("twelve-ResidualUnit measured-median summary:");
    println!(
        "  baseline={baseline_us:.1} us, epilogue={candidate_us:.1} us, save={:.1} us, speedup={:.3}x",
        baseline_us - candidate_us,
        baseline_us / candidate_us,
    );
    println!(
        "  dispatch=24 -> 12, workgroups={baseline_workgroups} -> {candidate_workgroups}, bit_mismatch={bit_mismatches}, max_abs={max_abs:.9e}"
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let (device, monitor) = initialize_wgpu(args.adapter_index);
    B::seed(&device, 0);
    println!(
        "isolated k7+act1 epilogue benchmark: warmup={} iterations={} trials={} cases={}",
        args.warmup,
        args.iterations,
        args.trials,
        CASES.len()
    );
    print_static_traffic();
    print_pytorch_characteristic();

    let mut results = Vec::with_capacity(CASES.len());
    for case in CASES {
        results.push(benchmark_case(&device, case, &args)?);
        synchronize_and_check_wgpu(
            &device,
            &monitor,
            &format!(
                "C={} L={} d={} benchmark",
                case.channels,
                case.length,
                case.dilation.value()
            ),
        )?;
    }
    print_summary(&results);
    synchronize_and_check_wgpu(&device, &monitor, "benchmark completion")?;
    println!("wgpu_uncaptured_errors=0");
    Ok(())
}
