//! Production A/B for T128 Conv1d followed by act1 Snake.
//!
//! Baseline: production T128 raw output followed by production `snake.wgsl`.
//! Fused: the same T128 tile and reduction body with the identical scalar
//! Snake expression applied immediately before each output store.

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
        conv1d_k7_t128::{
            Conv1dK7T128Tile, LaunchGeometry, OUTPUT_CHANNEL_TILE, TIME_TILE,
            conv1d_k7_same_t128_wgsl, conv1d_k7_t128_contract_is_compatible,
            production_tile_for_shape,
        },
        conv1d_k7_t128_snake_epilogue::{
            conv1d_k7_same_t128_snake_epilogue_wgsl,
            conv1d_k7_t128_snake_epilogue_contract_is_compatible,
        },
        conv1d_k7_tiled::Conv1dK7Dilation,
        snake::snake_wgsl,
    },
};

type B = WgpuRaw;

const KERNEL_SIZE: usize = 7;
const F32_BYTES: usize = core::mem::size_of::<f32>();
const SNAKE_WORKGROUP_SIZE: usize = 256;
const DEFAULT_WARMUP: usize = 10;
const DEFAULT_ITERATIONS: usize = 50;
const DEFAULT_TRIALS: usize = 5;
const VARIANT_COUNT: usize = 2;

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

impl Comparison {
    fn require_exact(self) -> Result<Self, Box<dyn Error>> {
        if self.mismatched_bits == 0 && self.max_abs == 0.0 {
            Ok(self)
        } else {
            Err(io::Error::other(format!(
                "T128 Snake epilogue is not bit-exact: bit_mismatch={} max_abs={:.9e}",
                self.mismatched_bits, self.max_abs,
            ))
            .into())
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CaseResult {
    case: ConvCase,
    tile: Conv1dK7T128Tile,
    baseline: Timing,
    fused: Timing,
    comparison: Comparison,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct BufferTraffic {
    input_read_bytes: u128,
    weight_read_bytes: u128,
    bias_read_bytes: u128,
    output_write_bytes: u128,
}

impl BufferTraffic {
    fn total_bytes(self) -> u128 {
        self.input_read_bytes
            + self.weight_read_bytes
            + self.bias_read_bytes
            + self.output_write_bytes
    }

    fn accumulate(&mut self, other: Self) {
        self.input_read_bytes += other.input_read_bytes;
        self.weight_read_bytes += other.weight_read_bytes;
        self.bias_read_bytes += other.bias_read_bytes;
        self.output_write_bytes += other.output_write_bytes;
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
    "usage: bench_conv1d_k7_t128_snake_epilogue <adapter-index> \
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

fn selected_tile(case: ConvCase) -> Conv1dK7T128Tile {
    production_tile_for_shape(case.channels, case.length, case.dilation)
        .expect("all benchmark cases must be exact released T128 shapes")
}

fn baseline_forward(
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    alpha: &Tensor<B, 3>,
    case: ConvCase,
    tile: Conv1dK7T128Tile,
) -> Tensor<B, 3> {
    let raw = conv1d_k7_same_t128_wgsl(
        input.clone().into_primitive().tensor(),
        weight.clone().into_primitive().tensor(),
        bias.clone().into_primitive().tensor(),
        case.dilation,
        tile,
    );
    let output = snake_wgsl(raw, alpha.clone().into_primitive().tensor());
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn fused_forward(
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    alpha: &Tensor<B, 3>,
    case: ConvCase,
    tile: Conv1dK7T128Tile,
) -> Tensor<B, 3> {
    let output = conv1d_k7_same_t128_snake_epilogue_wgsl(
        input.clone().into_primitive().tensor(),
        weight.clone().into_primitive().tensor(),
        bias.clone().into_primitive().tensor(),
        alpha.clone().into_primitive().tensor(),
        case.dilation,
        tile,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn variant_forward(
    variant: usize,
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    alpha: &Tensor<B, 3>,
    case: ConvCase,
    tile: Conv1dK7T128Tile,
) -> Tensor<B, 3> {
    match variant {
        0 => baseline_forward(input, weight, bias, alpha, case, tile),
        1 => fused_forward(input, weight, bias, alpha, case, tile),
        _ => unreachable!("benchmark has exactly two variants"),
    }
}

fn tensor_values(tensor: Tensor<B, 3>) -> Result<Vec<f32>, Box<dyn Error>> {
    Ok(tensor.into_data().to_vec::<f32>()?)
}

fn compare_outputs(expected: &[f32], actual: &[f32]) -> Result<Comparison, Box<dyn Error>> {
    if expected.len() != actual.len() {
        return Err(io::Error::other(format!(
            "output length mismatch: baseline={} fused={}",
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
                "non-finite output pair: baseline={expected:?} fused={actual:?}"
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
        .expect("released output element count must not overflow")
}

fn output_bytes(case: ConvCase) -> usize {
    output_elements(case)
        .checked_mul(F32_BYTES)
        .expect("released output byte count must not overflow")
}

fn model_macs(case: ConvCase) -> u128 {
    (case.channels as u128).pow(2) * KERNEL_SIZE as u128 * case.length as u128
}

fn valid_input_window_positions(case: ConvCase) -> u128 {
    let padding = 3 * case.dilation.value();
    (0..case.length)
        .step_by(TIME_TILE)
        .map(|time_base| {
            let start = time_base.saturating_sub(padding);
            let end = time_base
                .checked_add(TIME_TILE)
                .and_then(|value| value.checked_add(padding))
                .expect("released input-window endpoint must not overflow")
                .min(case.length);
            (end - start) as u128
        })
        .sum()
}

fn conv_buffer_traffic(case: ConvCase) -> BufferTraffic {
    let time_tiles = case.length.div_ceil(TIME_TILE) as u128;
    let output_channel_tiles = (case.channels / OUTPUT_CHANNEL_TILE) as u128;
    let channels = case.channels as u128;
    let bytes = F32_BYTES as u128;
    BufferTraffic {
        input_read_bytes: valid_input_window_positions(case)
            * channels
            * output_channel_tiles
            * bytes,
        weight_read_bytes: time_tiles * channels * channels * KERNEL_SIZE as u128 * bytes,
        bias_read_bytes: time_tiles * channels * 16 * bytes,
        output_write_bytes: channels * case.length as u128 * bytes,
    }
}

fn conv_workgroups(case: ConvCase, tile: Conv1dK7T128Tile) -> usize {
    LaunchGeometry::new(case.channels, case.length, case.dilation, tile)
        .and_then(LaunchGeometry::workgroups)
        .expect("released T128 workgroup count must be representable")
}

fn snake_workgroups(case: ConvCase) -> usize {
    output_elements(case).div_ceil(SNAKE_WORKGROUP_SIZE)
}

fn mib(bytes: u128) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn tmac_per_second(macs: u128, timing_us: f64) -> f64 {
    macs as f64 / (timing_us * 1.0e6)
}

fn check_contracts(
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    alpha: &Tensor<B, 3>,
    case: ConvCase,
    tile: Conv1dK7T128Tile,
) -> Result<(), Box<dyn Error>> {
    let input = input.clone().into_primitive().tensor();
    let weight = weight.clone().into_primitive().tensor();
    let bias = bias.clone().into_primitive().tensor();
    let alpha = alpha.clone().into_primitive().tensor();
    if !conv1d_k7_t128_contract_is_compatible(&input, &weight, &bias, case.dilation, tile) {
        return Err(io::Error::other(format!(
            "production T128 contract failed for C={} L={} d={} tile={}",
            case.channels,
            case.length,
            case.dilation.value(),
            tile.label(),
        ))
        .into());
    }
    if !conv1d_k7_t128_snake_epilogue_contract_is_compatible(
        &input,
        &weight,
        &bias,
        &alpha,
        case.dilation,
        tile,
    ) {
        let hardware = &input.client.properties().hardware;
        return Err(io::Error::other(format!(
            "fused T128+Snake contract failed for C={} L={} d={} tile={} max_bindings={} max_shared={}B max_units={}",
            case.channels,
            case.length,
            case.dilation.value(),
            tile.label(),
            hardware.max_bindings,
            hardware.max_shared_memory_size,
            hardware.max_units_per_cube,
        ))
        .into());
    }
    Ok(())
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
    let tile = selected_tile(case);
    check_contracts(&input, &weight, &bias, &alpha, case, tile)?;

    let expected = tensor_values(baseline_forward(&input, &weight, &bias, &alpha, case, tile))?;
    let actual = tensor_values(fused_forward(&input, &weight, &bias, &alpha, case, tile))?;
    let comparison = compare_outputs(&expected, &actual)?.require_exact()?;

    for variant in 0..VARIANT_COUNT {
        let mut operation = || variant_forward(variant, &input, &weight, &bias, &alpha, case, tile);
        warm_up(args.warmup, case, &mut operation);
    }

    let mut samples: [Vec<f64>; VARIANT_COUNT] =
        std::array::from_fn(|_| Vec::with_capacity(args.trials));
    for trial in 0..args.trials {
        for offset in 0..VARIANT_COUNT {
            let variant = (trial + offset) % VARIANT_COUNT;
            let mut operation =
                || variant_forward(variant, &input, &weight, &bias, &alpha, case, tile);
            samples[variant].push(measure(args.iterations, case, &mut operation));
        }
    }
    let baseline = summarize_samples(&samples[0]);
    let fused = summarize_samples(&samples[1]);

    let macs = model_macs(case);
    let bytes = output_bytes(case) as u128;
    let baseline_boundary = 3 * bytes;
    let fused_boundary = bytes;
    println!(
        "C={:4} L={:6} d={} tile={} model_MAC={:.6}G shared={}B",
        case.channels,
        case.length,
        case.dilation.value(),
        tile.label(),
        macs as f64 / 1.0e9,
        tile.shared_memory_bytes(case.dilation),
    );
    println!(
        "  correctness: elements={} bit_mismatch={} max_abs={:.9e} finite=true",
        comparison.elements, comparison.mismatched_bits, comparison.max_abs,
    );
    println!(
        "  T128+Snake: median={:10.3} us range=[{:10.3},{:10.3}] dispatch=2 workgroups={}+{} boundary={:.3}MiB TMAC/s={:.3}",
        baseline.median_us,
        baseline.min_us,
        baseline.max_us,
        conv_workgroups(case, tile),
        snake_workgroups(case),
        mib(baseline_boundary),
        tmac_per_second(macs, baseline.median_us),
    );
    println!(
        "  fused      : median={:10.3} us range=[{:10.3},{:10.3}] dispatch=1 workgroups={} boundary={:.3}MiB TMAC/s={:.3} speedup={:.3}x save={:.3}us/{:.3}MiB",
        fused.median_us,
        fused.min_us,
        fused.max_us,
        conv_workgroups(case, tile),
        mib(fused_boundary),
        tmac_per_second(macs, fused.median_us),
        baseline.median_us / fused.median_us,
        baseline.median_us - fused.median_us,
        mib(baseline_boundary - fused_boundary),
    );

    Ok(CaseResult {
        case,
        tile,
        baseline,
        fused,
        comparison,
    })
}

fn print_static_accounting() {
    let mut common_conv = BufferTraffic::default();
    let aggregate_output_bytes = CASES.iter().copied().map(output_bytes).sum::<usize>() as u128;
    for case in CASES {
        common_conv.accumulate(conv_buffer_traffic(case));
    }
    let alpha_read_bytes = aggregate_output_bytes;
    let baseline_total = common_conv.total_bytes() + 3 * aggregate_output_bytes;
    let fused_total = common_conv.total_bytes() + alpha_read_bytes;
    let conv_workgroups = CASES
        .iter()
        .copied()
        .map(|case| conv_workgroups(case, selected_tile(case)))
        .sum::<usize>();
    let standalone_snake_workgroups = CASES.iter().copied().map(snake_workgroups).sum::<usize>();
    println!("static exact12 T128 + Snake accounting:");
    println!(
        "  dispatch=24->12 workgroups={}->{} (removed Snake workgroups={standalone_snake_workgroups})",
        conv_workgroups + standalone_snake_workgroups,
        conv_workgroups,
    );
    println!(
        "  common conv traffic={:.3}MiB logical alpha reads={:.3}MiB; full logical traffic={:.3}->{:.3}MiB saved={:.3}MiB",
        mib(common_conv.total_bytes()),
        mib(alpha_read_bytes),
        mib(baseline_total),
        mib(fused_total),
        mib(baseline_total - fused_total),
    );
    let peak_extra_bytes = CASES
        .iter()
        .copied()
        .map(output_bytes)
        .max()
        .expect("cases must not be empty");
    println!(
        "  raw intermediate={}B ({:.3}MiB aggregate), peak extra live={}B ({:.3}MiB), fused persistent=0B",
        aggregate_output_bytes,
        mib(aggregate_output_bytes),
        peak_extra_bytes,
        mib(peak_extra_bytes as u128),
    );
}

fn print_summary(results: &[CaseResult]) {
    let baseline_us = results
        .iter()
        .map(|result| result.baseline.median_us)
        .sum::<f64>();
    let fused_us = results
        .iter()
        .map(|result| result.fused.median_us)
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
        .map(|result| conv_workgroups(result.case, result.tile) + snake_workgroups(result.case))
        .sum::<usize>();
    let fused_workgroups = results
        .iter()
        .map(|result| conv_workgroups(result.case, result.tile))
        .sum::<usize>();
    println!("exact12 sums of independently measured medians:");
    println!(
        "  production T128+Snake={baseline_us:.3}us fused={fused_us:.3}us save={:.3}us speedup={:.3}x",
        baseline_us - fused_us,
        baseline_us / fused_us,
    );
    println!(
        "  dispatch=24->12 workgroups={baseline_workgroups}->{fused_workgroups} bit_mismatch={bit_mismatches} max_abs={max_abs:.9e} finite=true WGPU_errors=0"
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let (device, monitor) = initialize_wgpu(args.adapter_index);
    B::seed(&device, 0);
    println!(
        "production T128+Snake epilogue A/B: warmup={} iterations={} trials={} cases=12 seed=0",
        args.warmup, args.iterations, args.trials,
    );
    print_static_accounting();

    let mut results = Vec::with_capacity(CASES.len());
    for case in CASES {
        results.push(benchmark_case(&device, case, &args)?);
        synchronize_and_check_wgpu(
            &device,
            &monitor,
            &format!(
                "C={} L={} d={} T128+Snake benchmark",
                case.channels,
                case.length,
                case.dilation.value()
            ),
        )?;
    }
    print_summary(&results);
    synchronize_and_check_wgpu(&device, &monitor, "T128+Snake benchmark completion")?;
    println!("wgpu_uncaptured_errors=0");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact12_selector_uses_six_of_each_production_tile() {
        let tiles = CASES.map(selected_tile);
        assert_eq!(
            tiles
                .into_iter()
                .filter(|tile| *tile == Conv1dK7T128Tile::Cin16)
                .count(),
            6
        );
        assert_eq!(
            tiles
                .into_iter()
                .filter(|tile| *tile == Conv1dK7T128Tile::Cin8)
                .count(),
            6
        );
    }

    #[test]
    fn exact12_dispatch_and_boundary_traffic_are_fixed() {
        let output_bytes = CASES.iter().copied().map(output_bytes).sum::<usize>();
        let conv_workgroups = CASES
            .iter()
            .copied()
            .map(|case| conv_workgroups(case, selected_tile(case)))
            .sum::<usize>();
        let snake_workgroups = CASES.iter().copied().map(snake_workgroups).sum::<usize>();
        assert_eq!(output_bytes, 254_361_600);
        assert_eq!(conv_workgroups, 15_552);
        assert_eq!(snake_workgroups, 248_400);
        assert_eq!(3 * output_bytes, 763_084_800);
        assert_eq!(output_bytes, 254_361_600);
        assert_eq!(2 * output_bytes, 508_723_200);
    }

    #[test]
    fn exact12_common_conv_traffic_matches_production_audit() {
        let mut traffic = BufferTraffic::default();
        for case in CASES {
            traffic.accumulate(conv_buffer_traffic(case));
        }
        assert_eq!(traffic.input_read_bytes, 1_750_678_272);
        assert_eq!(traffic.weight_read_bytes, 2_571_706_368);
        assert_eq!(traffic.bias_read_bytes, 31_850_496);
        assert_eq!(traffic.output_write_bytes, 254_361_600);
        assert_eq!(traffic.total_bytes(), 4_608_596_736);
    }
}
