//! Exactness and timing harness for the production T64/O64/Cin16 k=7 kernel.
//!
//! The default is a four-case representative screen. Run all twelve exact
//! DACVAE residual cases with `--all`. The reference is the previous production
//! selector (O16 only for C768/d3, O32-preferred otherwise); the measured O64
//! path calls the production module directly. Every case requires bitwise exact
//! output, alternates timing order, and fails on uncaptured WGPU errors.

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
        conv1d_k7_tiled::{Conv1dK7Dilation, conv1d_k7_same_tiled_wgsl},
        conv1d_k7_tiled_o32::{
            conv1d_k7_same_tiled_o32_wgsl, device_supports_conv1d_k7_tiled_o32,
            required_shared_memory_bytes as o32_shared_memory_bytes,
        },
        conv1d_k7_tiled_o64::{
            INPUT_CHANNEL_TILE as O64_INPUT_CHANNEL_TILE,
            OUTPUT_CHANNEL_TILE as O64_OUTPUT_CHANNEL_TILE, TIME_TILE as O64_TIME_TILE,
            WORKGROUP_SIZE as O64_WORKGROUP_SIZE, conv1d_k7_same_tiled_o64_wgsl,
            conv1d_k7_tiled_o64_contract_is_compatible, output_channel_tiles,
            required_shared_memory_bytes as o64_shared_memory_bytes,
        },
    },
};

type B = WgpuRaw;

const KERNEL_SIZE: usize = 7;
const DEFAULT_WARMUP: usize = 3;
const DEFAULT_ITERATIONS: usize = 5;
const DEFAULT_TRIALS: usize = 7;
const SCREEN_INDICES: [usize; 4] = [1, 3, 8, 10];

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
    all_cases: bool,
    case_index: Option<usize>,
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

impl Comparison {
    fn require_exact(self) -> Result<Self, Box<dyn Error>> {
        if self.mismatched_bits == 0 && self.max_abs == 0.0 {
            Ok(self)
        } else {
            Err(io::Error::other(format!(
                "production-o64 is not bitwise exact: bit_mismatch={} max_abs={:.9e}",
                self.mismatched_bits, self.max_abs,
            ))
            .into())
        }
    }
}

#[derive(Debug)]
struct CaseResult {
    case: ConvCase,
    reference_tile: ReferenceTile,
    reference: Timing,
    o64: Timing,
    comparison: Comparison,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReferenceTile {
    Output16,
    Output32,
}

impl ReferenceTile {
    const fn label(self) -> &'static str {
        match self {
            Self::Output16 => "prior-o16-c16",
            Self::Output32 => "prior-o32-c16",
        }
    }

    const fn output_channel_tile(self) -> usize {
        match self {
            Self::Output16 => 16,
            Self::Output32 => 32,
        }
    }

    const fn workgroup_size(self) -> usize {
        match self {
            Self::Output16 => 128,
            Self::Output32 => 256,
        }
    }

    fn shared_memory_bytes(self, dilation: Conv1dK7Dilation) -> usize {
        match self {
            Self::Output16 => {
                let input = 16 * (O64_TIME_TILE + 6 * dilation.value());
                let weight = 16 * 16 * KERNEL_SIZE;
                (input + weight) * core::mem::size_of::<f32>()
            }
            Self::Output32 => o32_shared_memory_bytes(dilation),
        }
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

fn usage() -> &'static str {
    "usage: bench_conv1d_k7_tile_candidates_v2 <adapter-index> [--all | --case N] \
     [--warmup N] [--iterations N] [--trials N]"
}

fn next_usize(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<usize, Box<dyn Error>> {
    let text = args
        .next()
        .ok_or_else(|| io::Error::other(format!("{option} requires a value")))?;
    text.parse::<usize>().map_err(|error| {
        io::Error::other(format!("invalid value {text:?} for {option}: {error}")).into()
    })
}

fn next_positive_usize(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<usize, Box<dyn Error>> {
    let value = next_usize(args, option)?;
    if value == 0 {
        return Err(io::Error::other(format!("{option} must be greater than zero")).into());
    }
    Ok(value)
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut adapter_index = None;
    let mut all_cases = false;
    let mut case_index = None;
    let mut warmup = DEFAULT_WARMUP;
    let mut iterations = DEFAULT_ITERATIONS;
    let mut trials = DEFAULT_TRIALS;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--all" => all_cases = true,
            "--case" => case_index = Some(next_usize(&mut args, "--case")?),
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
    if all_cases && case_index.is_some() {
        return Err(io::Error::other("--all and --case are mutually exclusive").into());
    }
    Ok(Args {
        adapter_index: adapter_index
            .ok_or_else(|| io::Error::other(format!("missing adapter index; {}", usage())))?,
        all_cases,
        case_index,
        warmup,
        iterations,
        trials,
    })
}

fn selected_cases(args: &Args) -> Result<Vec<ConvCase>, Box<dyn Error>> {
    if args.all_cases {
        return Ok(CASES.to_vec());
    }
    if let Some(index) = args.case_index {
        return CASES
            .get(index)
            .copied()
            .map(|case| vec![case])
            .ok_or_else(|| {
                io::Error::other(format!(
                    "--case index {index} is out of range; valid range is 0..{}",
                    CASES.len()
                ))
                .into()
            });
    }
    Ok(SCREEN_INDICES.map(|index| CASES[index]).to_vec())
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

const fn prior_prefers_o32(case: ConvCase) -> bool {
    !(case.channels == 768 && matches!(case.dilation, Conv1dK7Dilation::Three))
}

const fn production_prefers_o64(case: ConvCase) -> bool {
    matches!(
        (case.channels, case.length, case.dilation),
        (768, 600, Conv1dK7Dilation::Nine)
            | (384, 6_000, Conv1dK7Dilation::One)
            | (384, 6_000, Conv1dK7Dilation::Three)
            | (192, 48_000, Conv1dK7Dilation::One)
            | (192, 48_000, Conv1dK7Dilation::Three)
            | (96, 96_000, Conv1dK7Dilation::One)
            | (96, 96_000, Conv1dK7Dilation::Three)
    )
}

fn reference_tile(input: &Tensor<B, 3>, case: ConvCase) -> ReferenceTile {
    if !prior_prefers_o32(case) {
        return ReferenceTile::Output16;
    }
    let input = input.clone().into_primitive().tensor();
    if device_supports_conv1d_k7_tiled_o32(&input, case.dilation) {
        ReferenceTile::Output32
    } else {
        ReferenceTile::Output16
    }
}

fn reference_forward(
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    dilation: Conv1dK7Dilation,
    tile: ReferenceTile,
) -> Tensor<B, 3> {
    let input = input.clone().into_primitive().tensor();
    let weight = weight.clone().into_primitive().tensor();
    let bias = bias.clone().into_primitive().tensor();
    let output = match tile {
        ReferenceTile::Output16 => conv1d_k7_same_tiled_wgsl(input, weight, bias, dilation),
        ReferenceTile::Output32 => conv1d_k7_same_tiled_o32_wgsl(input, weight, bias, dilation),
    };
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn o64_forward(
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    dilation: Conv1dK7Dilation,
) -> Tensor<B, 3> {
    let output = conv1d_k7_same_tiled_o64_wgsl(
        input.clone().into_primitive().tensor(),
        weight.clone().into_primitive().tensor(),
        bias.clone().into_primitive().tensor(),
        dilation,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn variant_forward(
    variant: usize,
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    case: ConvCase,
    reference_tile: ReferenceTile,
) -> Tensor<B, 3> {
    match variant {
        0 => reference_forward(input, weight, bias, case.dilation, reference_tile),
        1 => o64_forward(input, weight, bias, case.dilation),
        _ => unreachable!("benchmark has exactly two variants"),
    }
}

fn tensor_values(tensor: Tensor<B, 3>) -> Result<Vec<f32>, Box<dyn Error>> {
    Ok(tensor.into_data().to_vec::<f32>()?)
}

fn compare_outputs(expected: &[f32], actual: &[f32]) -> Result<Comparison, Box<dyn Error>> {
    if expected.len() != actual.len() {
        return Err(io::Error::other(format!(
            "output length mismatch: reference={} O64={}",
            expected.len(),
            actual.len()
        ))
        .into());
    }
    let mut mismatched_bits = 0;
    let mut max_abs = 0.0_f32;
    for (&expected, &actual) in expected.iter().zip(actual) {
        if !expected.is_finite() || !actual.is_finite() {
            return Err(io::Error::other(format!(
                "non-finite output pair: reference={expected:?} O64={actual:?}"
            ))
            .into());
        }
        if expected.to_bits() != actual.to_bits() {
            mismatched_bits += 1;
        }
        max_abs = max_abs.max((expected - actual).abs());
    }
    Ok(Comparison {
        elements: expected.len(),
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
    let mut output = None;
    for _ in 0..count {
        output = Some(operation());
    }
    synchronize(output.expect("warmup count must be non-zero"), case);
}

fn measure<F>(iterations: usize, case: ConvCase, operation: &mut F) -> f64
where
    F: FnMut() -> Tensor<B, 3>,
{
    let started = Instant::now();
    let mut output = None;
    for _ in 0..iterations {
        output = Some(operation());
    }
    synchronize(output.expect("iteration count must be non-zero"), case);
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

fn workgroups(case: ConvCase, output_channel_tile: usize) -> usize {
    case.length.div_ceil(O64_TIME_TILE) * case.channels.div_ceil(output_channel_tile)
}

fn print_timing(label: &str, timing: Timing, reference_us: f64, macs: u128) {
    let gflops = 2.0 * macs as f64 / (timing.median_us * 1_000.0);
    println!(
        "  {label:20} median={:10.1} us range=[{:10.1},{:10.1}] speedup={:6.3}x GFLOP/s={gflops:8.2}",
        timing.median_us,
        timing.min_us,
        timing.max_us,
        reference_us / timing.median_us,
    );
}

fn check_o64_resources(
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    case: ConvCase,
) -> Result<(), Box<dyn Error>> {
    let input = input.clone().into_primitive().tensor();
    let weight = weight.clone().into_primitive().tensor();
    let bias = bias.clone().into_primitive().tensor();
    if conv1d_k7_tiled_o64_contract_is_compatible(&input, &weight, &bias, case.dilation) {
        return Ok(());
    }
    let hardware = &input.client.properties().hardware;
    Err(io::Error::other(format!(
        "production-o64 unavailable for C={} L={} d={}: needs shared={}B/wg={}/dim=(16,16,1)/bindings=4/contiguous-f32-same-device; device max_shared={}B max_units={} max_dim={:?} max_count={:?} max_bindings={}",
        case.channels,
        case.length,
        case.dilation.value(),
        o64_shared_memory_bytes(case.dilation),
        O64_WORKGROUP_SIZE,
        hardware.max_shared_memory_size,
        hardware.max_units_per_cube,
        hardware.max_cube_dim,
        hardware.max_cube_count,
        hardware.max_bindings,
    ))
    .into())
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
    let reference_tile = reference_tile(&input, case);
    check_o64_resources(&input, &weight, &bias, case)?;

    let raw = input.clone().into_primitive().tensor();
    let hardware = &raw.client.properties().hardware;
    println!(
        "C={:4} L={:6} d={} MAC={:.6}G reference={} production_selects_o64={}",
        case.channels,
        case.length,
        case.dilation.value(),
        useful_macs(case) as f64 / 1.0e9,
        reference_tile.label(),
        production_prefers_o64(case),
    );
    println!(
        "  device max_bindings={} max_shared={}B max_units={} max_dim={:?} max_count={:?}",
        hardware.max_bindings,
        hardware.max_shared_memory_size,
        hardware.max_units_per_cube,
        hardware.max_cube_dim,
        hardware.max_cube_count,
    );

    let reference_values = tensor_values(reference_forward(
        &input,
        &weight,
        &bias,
        case.dilation,
        reference_tile,
    ))?;
    let o64_values = tensor_values(o64_forward(&input, &weight, &bias, case.dilation))?;
    let comparison = compare_outputs(&reference_values, &o64_values)?.require_exact()?;
    println!(
        "  correctness {:20} elements={} bit_mismatch={} max_abs={:.9e}",
        "production-o64", comparison.elements, comparison.mismatched_bits, comparison.max_abs,
    );

    for variant in 0..2 {
        let mut operation =
            || variant_forward(variant, &input, &weight, &bias, case, reference_tile);
        warm_up(args.warmup, case, &mut operation);
    }

    let mut samples = [
        Vec::with_capacity(args.trials),
        Vec::with_capacity(args.trials),
    ];
    for trial in 0..args.trials {
        for offset in 0..2 {
            let variant = (trial + offset) % 2;
            let mut operation =
                || variant_forward(variant, &input, &weight, &bias, case, reference_tile);
            samples[variant].push(measure(args.iterations, case, &mut operation));
        }
    }
    let reference = summarize_samples(&samples[0]);
    let o64 = summarize_samples(&samples[1]);
    let macs = useful_macs(case);
    let reduction_tiles = case.channels / O64_INPUT_CHANNEL_TILE;
    println!(
        "  resources {:20} tile={}x{} Cin=16 wg={} acc=8 shared={}B workgroups={} barrier_pairs/wg={}",
        reference_tile.label(),
        O64_TIME_TILE,
        reference_tile.output_channel_tile(),
        reference_tile.workgroup_size(),
        reference_tile.shared_memory_bytes(case.dilation),
        workgroups(case, reference_tile.output_channel_tile()),
        reduction_tiles,
    );
    print_timing(reference_tile.label(), reference, reference.median_us, macs);

    let output_tiles = output_channel_tiles(case.channels)
        .expect("production O64 contract guarantees compatible channels");
    let output_extent = output_tiles * O64_OUTPUT_CHANNEL_TILE;
    let output_utilization = case.channels as f64 / output_extent as f64;
    println!(
        "  resources {:20} tile={}x{} Cin={} wg={} acc=16 shared={}B workgroups={} barrier_pairs/wg={} output_util={:.1}%",
        "production-o64",
        O64_TIME_TILE,
        O64_OUTPUT_CHANNEL_TILE,
        O64_INPUT_CHANNEL_TILE,
        O64_WORKGROUP_SIZE,
        o64_shared_memory_bytes(case.dilation),
        workgroups(case, O64_OUTPUT_CHANNEL_TILE),
        reduction_tiles,
        100.0 * output_utilization,
    );
    print_timing("production-o64", o64, reference.median_us, macs);

    Ok(CaseResult {
        case,
        reference_tile,
        reference,
        o64,
        comparison,
    })
}

fn print_static_resources() {
    println!("production O64 static resource matrix:");
    for dilation in [
        Conv1dK7Dilation::One,
        Conv1dK7Dilation::Three,
        Conv1dK7Dilation::Nine,
    ] {
        println!(
            "  d={}: shared={}B tile={}x{} Cin={} wg={} acc=16",
            dilation.value(),
            o64_shared_memory_bytes(dilation),
            O64_TIME_TILE,
            O64_OUTPUT_CHANNEL_TILE,
            O64_INPUT_CHANNEL_TILE,
            O64_WORKGROUP_SIZE,
        );
    }
}

fn print_summary(results: &[CaseResult], all_cases: bool) {
    let reference_us = results
        .iter()
        .map(|result| result.reference.median_us)
        .sum::<f64>();
    let o64_us = results
        .iter()
        .map(|result| result.o64.median_us)
        .sum::<f64>();
    let selected_us = results
        .iter()
        .map(|result| {
            if production_prefers_o64(result.case) {
                result.o64.median_us
            } else {
                result.reference.median_us
            }
        })
        .sum::<f64>();
    let mismatched_bits = results
        .iter()
        .map(|result| result.comparison.mismatched_bits)
        .sum::<usize>();
    let max_abs = results
        .iter()
        .map(|result| result.comparison.max_abs)
        .fold(0.0_f32, f32::max);
    let reference_workgroups = results
        .iter()
        .map(|result| workgroups(result.case, result.reference_tile.output_channel_tile()))
        .sum::<usize>();
    let o64_workgroups = results
        .iter()
        .map(|result| workgroups(result.case, O64_OUTPUT_CHANNEL_TILE))
        .sum::<usize>();
    let scope = if all_cases {
        "all-twelve"
    } else {
        "selected-screen"
    };
    println!("{scope} sums of independently measured medians:");
    println!(
        "  prior-selector       median_sum={reference_us:.1} us workgroups={reference_workgroups}"
    );
    println!(
        "  production-o64       median_sum={o64_us:.1} us speedup={:.3}x workgroups={o64_workgroups} bit_mismatch={mismatched_bits} max_abs={max_abs:.9e}",
        reference_us / o64_us,
    );
    println!(
        "  production-selector  median_sum={selected_us:.1} us speedup={:.3}x",
        reference_us / selected_us,
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let cases = selected_cases(&args)?;
    let (device, monitor) = initialize_wgpu(args.adapter_index);
    B::seed(&device, 0);
    println!(
        "production O64 exact k7 benchmark: warmup={} iterations={} trials={} cases={} mode={}",
        args.warmup,
        args.iterations,
        args.trials,
        cases.len(),
        if args.all_cases {
            "all-twelve"
        } else if args.case_index.is_some() {
            "single-case"
        } else {
            "representative-first"
        },
    );
    print_static_resources();

    let mut results = Vec::with_capacity(cases.len());
    for case in cases {
        results.push(benchmark_case(&device, case, &args)?);
        synchronize_and_check_wgpu(
            &device,
            &monitor,
            &format!(
                "C={} L={} d={} production O64 benchmark",
                case.channels,
                case.length,
                case.dilation.value()
            ),
        )?;
    }
    print_summary(&results, args.all_cases);
    synchronize_and_check_wgpu(&device, &monitor, "production O64 benchmark completion")?;
    println!("wgpu_uncaptured_errors=0");
    Ok(())
}
