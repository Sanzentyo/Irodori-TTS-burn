//! Exactness and timing harness for production T128/O32 k=7 Conv1d tiles.
//!
//! This binary compares the current measured O16/O32/O64 selector with
//! T128/O32/Cin16 and T128/O32/Cin8 across all twelve DACVAE ResidualUnits.
//! It measures the production launcher directly without invoking codec routing.

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
            ACCUMULATORS_PER_INVOCATION, Conv1dK7T128Tile, LaunchGeometry,
            OUTPUT_CHANNEL_TILE as T128_OUTPUT_TILE, TIME_TILE as T128_TIME_TILE,
            WORKGROUP_SIZE as T128_WORKGROUP_SIZE, conv1d_k7_same_t128_wgsl,
            conv1d_k7_t128_contract_is_compatible, production_tile_for_shape,
        },
        conv1d_k7_tiled::{Conv1dK7Dilation, conv1d_k7_same_tiled_wgsl},
        conv1d_k7_tiled_o32::{
            conv1d_k7_same_tiled_o32_wgsl, device_supports_conv1d_k7_tiled_o32,
            required_shared_memory_bytes as o32_shared_memory_bytes,
        },
        conv1d_k7_tiled_o64::{
            conv1d_k7_same_tiled_o64_wgsl, conv1d_k7_tiled_o64_contract_is_compatible,
            required_shared_memory_bytes as o64_shared_memory_bytes,
        },
    },
};

type B = WgpuRaw;

const KERNEL_SIZE: usize = 7;
const F32_BYTES: usize = core::mem::size_of::<f32>();
const CURRENT_TIME_TILE: usize = 64;
const LOCAL_TIME_LANES: usize = 16;
const DEFAULT_WARMUP: usize = 10;
const DEFAULT_ITERATIONS: usize = 50;
const DEFAULT_TRIALS: usize = 5;
const VARIANT_COUNT: usize = 3;
const T128_TILES: [Conv1dK7T128Tile; 2] = [Conv1dK7T128Tile::Cin16, Conv1dK7T128Tile::Cin8];

#[derive(Clone, Copy, Debug)]
struct ConvCase {
    channels: usize,
    length: usize,
    dilation: Conv1dK7Dilation,
    pytorch_strict_us: f64,
}

const CASES: [ConvCase; 12] = [
    ConvCase {
        channels: 768,
        length: 600,
        dilation: Conv1dK7Dilation::One,
        pytorch_strict_us: 1_049.436_188,
    },
    ConvCase {
        channels: 768,
        length: 600,
        dilation: Conv1dK7Dilation::Three,
        pytorch_strict_us: 1_036.123_505,
    },
    ConvCase {
        channels: 768,
        length: 600,
        dilation: Conv1dK7Dilation::Nine,
        pytorch_strict_us: 1_028.894_730,
    },
    ConvCase {
        channels: 384,
        length: 6_000,
        dilation: Conv1dK7Dilation::One,
        pytorch_strict_us: 9_614.229_736,
    },
    ConvCase {
        channels: 384,
        length: 6_000,
        dilation: Conv1dK7Dilation::Three,
        pytorch_strict_us: 1_654.394_836,
    },
    ConvCase {
        channels: 384,
        length: 6_000,
        dilation: Conv1dK7Dilation::Nine,
        pytorch_strict_us: 1_670.022_430,
    },
    ConvCase {
        channels: 192,
        length: 48_000,
        dilation: Conv1dK7Dilation::One,
        pytorch_strict_us: 3_203.891_296,
    },
    ConvCase {
        channels: 192,
        length: 48_000,
        dilation: Conv1dK7Dilation::Three,
        pytorch_strict_us: 3_207.761_841,
    },
    ConvCase {
        channels: 192,
        length: 48_000,
        dilation: Conv1dK7Dilation::Nine,
        pytorch_strict_us: 3_211.653_137,
    },
    ConvCase {
        channels: 96,
        length: 96_000,
        dilation: Conv1dK7Dilation::One,
        pytorch_strict_us: 1_715.527_649,
    },
    ConvCase {
        channels: 96,
        length: 96_000,
        dilation: Conv1dK7Dilation::Three,
        pytorch_strict_us: 1_716.428_833,
    },
    ConvCase {
        channels: 96,
        length: 96_000,
        dilation: Conv1dK7Dilation::Nine,
        pytorch_strict_us: 1_717.125_092,
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
    fn require_exact(self, label: &str) -> Result<Self, Box<dyn Error>> {
        if self.mismatched_bits == 0 && self.max_abs == 0.0 {
            Ok(self)
        } else {
            Err(io::Error::other(format!(
                "{label} is not bitwise exact: bit_mismatch={} max_abs={:.9e}",
                self.mismatched_bits, self.max_abs,
            ))
            .into())
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BaselineTile {
    Output16,
    Output32,
    Output64,
}

impl BaselineTile {
    const fn label(self) -> &'static str {
        match self {
            Self::Output16 => "selected-t64-o16-c16",
            Self::Output32 => "selected-t64-o32-c16",
            Self::Output64 => "selected-t64-o64-c16",
        }
    }

    const fn output_channel_tile(self) -> usize {
        match self {
            Self::Output16 => 16,
            Self::Output32 => 32,
            Self::Output64 => 64,
        }
    }

    const fn workgroup_size(self) -> usize {
        match self {
            Self::Output16 => 128,
            Self::Output32 | Self::Output64 => 256,
        }
    }

    const fn accumulators_per_invocation(self) -> usize {
        match self {
            Self::Output16 | Self::Output32 => 8,
            Self::Output64 => 16,
        }
    }

    const fn shared_memory_bytes(self, dilation: Conv1dK7Dilation) -> usize {
        match self {
            Self::Output16 => {
                let input = 16 * (CURRENT_TIME_TILE + 6 * dilation.value());
                let weight = 16 * 16 * KERNEL_SIZE;
                (input + weight) * F32_BYTES
            }
            Self::Output32 => o32_shared_memory_bytes(dilation),
            Self::Output64 => o64_shared_memory_bytes(dilation),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
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
struct CaseResult {
    case: ConvCase,
    baseline_tile: BaselineTile,
    timings: [Timing; VARIANT_COUNT],
    comparisons: [Comparison; 2],
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
    "usage: bench_conv1d_k7_t128 <adapter-index> \
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

const fn production_route_tile(case: ConvCase) -> BaselineTile {
    if matches!(
        (case.channels, case.length, case.dilation),
        (768, 600, Conv1dK7Dilation::Nine)
            | (384, 6_000, Conv1dK7Dilation::One)
            | (384, 6_000, Conv1dK7Dilation::Three)
            | (192, 48_000, Conv1dK7Dilation::One)
            | (192, 48_000, Conv1dK7Dilation::Three)
            | (96, 96_000, Conv1dK7Dilation::One)
            | (96, 96_000, Conv1dK7Dilation::Three)
    ) {
        BaselineTile::Output64
    } else if case.channels == 768 && matches!(case.dilation, Conv1dK7Dilation::Three) {
        BaselineTile::Output16
    } else {
        BaselineTile::Output32
    }
}

fn selected_baseline_tile(
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    case: ConvCase,
) -> Result<BaselineTile, Box<dyn Error>> {
    let requested = production_route_tile(case);
    let input = input.clone().into_primitive().tensor();
    let weight = weight.clone().into_primitive().tensor();
    let bias = bias.clone().into_primitive().tensor();
    match requested {
        BaselineTile::Output64
            if conv1d_k7_tiled_o64_contract_is_compatible(
                &input,
                &weight,
                &bias,
                case.dilation,
            ) =>
        {
            Ok(BaselineTile::Output64)
        }
        BaselineTile::Output32 | BaselineTile::Output64
            if device_supports_conv1d_k7_tiled_o32(&input, case.dilation) =>
        {
            Ok(BaselineTile::Output32)
        }
        BaselineTile::Output16 | BaselineTile::Output32 | BaselineTile::Output64 => {
            if input.client.properties().hardware.max_bindings < 4 {
                return Err(io::Error::other(
                    "current O16 fallback requires four storage bindings",
                )
                .into());
            }
            Ok(BaselineTile::Output16)
        }
    }
}

fn baseline_forward(
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    dilation: Conv1dK7Dilation,
    tile: BaselineTile,
) -> Tensor<B, 3> {
    let input = input.clone().into_primitive().tensor();
    let weight = weight.clone().into_primitive().tensor();
    let bias = bias.clone().into_primitive().tensor();
    let output = match tile {
        BaselineTile::Output16 => conv1d_k7_same_tiled_wgsl(input, weight, bias, dilation),
        BaselineTile::Output32 => conv1d_k7_same_tiled_o32_wgsl(input, weight, bias, dilation),
        BaselineTile::Output64 => conv1d_k7_same_tiled_o64_wgsl(input, weight, bias, dilation),
    };
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn t128_forward(
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    dilation: Conv1dK7Dilation,
    tile: Conv1dK7T128Tile,
) -> Tensor<B, 3> {
    let output = conv1d_k7_same_t128_wgsl(
        input.clone().into_primitive().tensor(),
        weight.clone().into_primitive().tensor(),
        bias.clone().into_primitive().tensor(),
        dilation,
        tile,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn variant_forward(
    variant: usize,
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    case: ConvCase,
    baseline_tile: BaselineTile,
) -> Tensor<B, 3> {
    match variant {
        0 => baseline_forward(input, weight, bias, case.dilation, baseline_tile),
        1 | 2 => t128_forward(input, weight, bias, case.dilation, T128_TILES[variant - 1]),
        _ => unreachable!("benchmark has exactly three variants"),
    }
}

fn tensor_values(tensor: Tensor<B, 3>) -> Result<Vec<f32>, Box<dyn Error>> {
    Ok(tensor.into_data().to_vec::<f32>()?)
}

fn compare_outputs(expected: &[f32], actual: &[f32]) -> Result<Comparison, Box<dyn Error>> {
    if expected.len() != actual.len() {
        return Err(io::Error::other(format!(
            "output length mismatch: baseline={} T128={}",
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
                "non-finite output pair: baseline={expected:?} T128={actual:?}"
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

fn model_macs(case: ConvCase) -> u128 {
    (case.channels as u128).pow(2) * KERNEL_SIZE as u128 * case.length as u128
}

fn workgroups(case: ConvCase, time_tile: usize, output_channel_tile: usize) -> usize {
    case.length.div_ceil(time_tile) * case.channels.div_ceil(output_channel_tile)
}

fn barrier_encounters(case: ConvCase, workgroups: usize, input_channel_tile: usize) -> usize {
    workgroups * 2 * (case.channels / input_channel_tile)
}

fn valid_input_window_positions(case: ConvCase, time_tile: usize) -> u128 {
    let padding = 3 * case.dilation.value();
    (0..case.length)
        .step_by(time_tile)
        .map(|time_base| {
            let start = time_base.saturating_sub(padding);
            let end = time_base
                .checked_add(time_tile)
                .and_then(|value| value.checked_add(padding))
                .expect("official input-window endpoint must not overflow")
                .min(case.length);
            (end - start) as u128
        })
        .sum()
}

fn buffer_traffic(case: ConvCase, time_tile: usize, output_channel_tile: usize) -> BufferTraffic {
    let time_tiles = case.length.div_ceil(time_tile) as u128;
    let output_channel_tiles = case.channels.div_ceil(output_channel_tile) as u128;
    let channels = case.channels as u128;
    let bytes = F32_BYTES as u128;
    BufferTraffic {
        input_read_bytes: valid_input_window_positions(case, time_tile)
            * channels
            * output_channel_tiles
            * bytes,
        weight_read_bytes: time_tiles * channels * channels * KERNEL_SIZE as u128 * bytes,
        // Sixteen local x lanes each seed the outputs they own once per time tile.
        bias_read_bytes: time_tiles * channels * LOCAL_TIME_LANES as u128 * bytes,
        output_write_bytes: channels * case.length as u128 * bytes,
    }
}

fn mib(bytes: u128) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn tmac_per_second(macs: u128, timing_us: f64) -> f64 {
    macs as f64 / (timing_us * 1.0e6)
}

fn print_timing(label: &str, timing: Timing, baseline_us: f64, macs: u128) {
    println!(
        "  {label:24} median={:10.3} us range=[{:10.3},{:10.3}] speedup={:6.3}x TMAC/s={:6.3}",
        timing.median_us,
        timing.min_us,
        timing.max_us,
        baseline_us / timing.median_us,
        tmac_per_second(macs, timing.median_us),
    );
}

fn check_t128_contracts(
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    case: ConvCase,
) -> Result<(), Box<dyn Error>> {
    let input = input.clone().into_primitive().tensor();
    let weight = weight.clone().into_primitive().tensor();
    let bias = bias.clone().into_primitive().tensor();
    for tile in T128_TILES {
        if !conv1d_k7_t128_contract_is_compatible(&input, &weight, &bias, case.dilation, tile) {
            let hardware = &input.client.properties().hardware;
            return Err(io::Error::other(format!(
                "{} unavailable for C={} L={} d={}: needs shared={}B/wg={}/dim=(16,16,1)/bindings=4/contiguous-B1-F32-same-device; device max_shared={}B max_units={} max_dim={:?} max_count={:?} max_bindings={}",
                tile.label(),
                case.channels,
                case.length,
                case.dilation.value(),
                tile.shared_memory_bytes(case.dilation),
                T128_WORKGROUP_SIZE,
                hardware.max_shared_memory_size,
                hardware.max_units_per_cube,
                hardware.max_cube_dim,
                hardware.max_cube_count,
                hardware.max_bindings,
            ))
            .into());
        }
    }
    Ok(())
}

fn benchmark_case(
    device: &<B as Backend>::Device,
    case: ConvCase,
    args: &Args,
) -> Result<CaseResult, Box<dyn Error>> {
    // The seed in main makes these non-zero random tensors repeatable. Shapes
    // reproduce the checkpoint's exact contiguous NCL/OIK physical contracts.
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
    let baseline_tile = selected_baseline_tile(&input, &weight, &bias, case)?;
    check_t128_contracts(&input, &weight, &bias, case)?;

    // Compile and compare every shader outside the timed samples.
    let expected = tensor_values(baseline_forward(
        &input,
        &weight,
        &bias,
        case.dilation,
        baseline_tile,
    ))?;
    let mut comparisons = Vec::with_capacity(T128_TILES.len());
    for tile in T128_TILES {
        let actual = tensor_values(t128_forward(&input, &weight, &bias, case.dilation, tile))?;
        comparisons.push(compare_outputs(&expected, &actual)?.require_exact(tile.label())?);
    }
    let comparisons: [Comparison; 2] = comparisons
        .try_into()
        .expect("benchmark has exactly two T128 tiles");

    for variant in 0..VARIANT_COUNT {
        let mut operation =
            || variant_forward(variant, &input, &weight, &bias, case, baseline_tile);
        warm_up(args.warmup, case, &mut operation);
    }

    let mut samples: [Vec<f64>; VARIANT_COUNT] =
        std::array::from_fn(|_| Vec::with_capacity(args.trials));
    for trial in 0..args.trials {
        for offset in 0..VARIANT_COUNT {
            let variant = (trial + offset) % VARIANT_COUNT;
            let mut operation =
                || variant_forward(variant, &input, &weight, &bias, case, baseline_tile);
            samples[variant].push(measure(args.iterations, case, &mut operation));
        }
    }
    let timings = samples.each_ref().map(|samples| summarize_samples(samples));

    let baseline_workgroups =
        workgroups(case, CURRENT_TIME_TILE, baseline_tile.output_channel_tile());
    let t128_geometry = LaunchGeometry::new(
        case.channels,
        case.length,
        case.dilation,
        Conv1dK7T128Tile::Cin16,
    )
    .expect("official T128 geometry must be valid");
    let t128_workgroups = t128_geometry
        .workgroups()
        .expect("official T128 workgroup count must fit usize");
    let macs = model_macs(case);
    println!(
        "C={:4} L={:6} d={} model_MAC={:.6}G PyTorch-strict={:.3} us",
        case.channels,
        case.length,
        case.dilation.value(),
        macs as f64 / 1.0e9,
        case.pytorch_strict_us,
    );
    println!(
        "  resources {:24} T64/O{}/Cin16 wg={} acc={} shared={}B workgroups={} barriers/wg={} buffer={:.3}MiB",
        baseline_tile.label(),
        baseline_tile.output_channel_tile(),
        baseline_tile.workgroup_size(),
        baseline_tile.accumulators_per_invocation(),
        baseline_tile.shared_memory_bytes(case.dilation),
        baseline_workgroups,
        2 * (case.channels / 16),
        mib(
            buffer_traffic(case, CURRENT_TIME_TILE, baseline_tile.output_channel_tile(),)
                .total_bytes()
        ),
    );
    print_timing(
        baseline_tile.label(),
        timings[0],
        timings[0].median_us,
        macs,
    );
    for (index, tile) in T128_TILES.into_iter().enumerate() {
        let comparison = comparisons[index];
        println!(
            "  correctness {:24} elements={} bit_mismatch={} max_abs={:.9e}",
            tile.label(),
            comparison.elements,
            comparison.mismatched_bits,
            comparison.max_abs,
        );
        println!(
            "  resources {:24} T128/O32/Cin{} wg={} acc={} shared={}B workgroups={} barriers/wg={} buffer={:.3}MiB persistent=0",
            tile.label(),
            tile.input_channel_tile(),
            T128_WORKGROUP_SIZE,
            ACCUMULATORS_PER_INVOCATION,
            tile.shared_memory_bytes(case.dilation),
            t128_workgroups,
            2 * (case.channels / tile.input_channel_tile()),
            mib(buffer_traffic(case, T128_TIME_TILE, T128_OUTPUT_TILE).total_bytes()),
        );
        print_timing(tile.label(), timings[index + 1], timings[0].median_us, macs);
    }

    Ok(CaseResult {
        case,
        baseline_tile,
        timings,
        comparisons,
    })
}

fn print_static_accounting() {
    let mut baseline_traffic = BufferTraffic::default();
    let mut t128_traffic = BufferTraffic::default();
    let mut baseline_workgroups = 0;
    let mut t128_workgroups = 0;
    let mut baseline_barriers = 0;
    let mut t128_c16_barriers = 0;
    let mut t128_c8_barriers = 0;
    let mut total_macs = 0_u128;
    for case in CASES {
        let tile = production_route_tile(case);
        let current_workgroups = workgroups(case, CURRENT_TIME_TILE, tile.output_channel_tile());
        let case_t128_workgroups = workgroups(case, T128_TIME_TILE, T128_OUTPUT_TILE);
        baseline_workgroups += current_workgroups;
        t128_workgroups += case_t128_workgroups;
        baseline_barriers += barrier_encounters(case, current_workgroups, 16);
        t128_c16_barriers += barrier_encounters(case, case_t128_workgroups, 16);
        t128_c8_barriers += barrier_encounters(case, case_t128_workgroups, 8);
        baseline_traffic.accumulate(buffer_traffic(
            case,
            CURRENT_TIME_TILE,
            tile.output_channel_tile(),
        ));
        t128_traffic.accumulate(buffer_traffic(case, T128_TIME_TILE, T128_OUTPUT_TILE));
        total_macs += model_macs(case);
    }
    println!("static all-twelve accounting (guarded zero loads excluded):");
    println!(
        "  model_MAC={:.6}G workgroups={baseline_workgroups}->{t128_workgroups} ({:.1}% reduction)",
        total_macs as f64 / 1.0e9,
        100.0 * (baseline_workgroups - t128_workgroups) as f64 / baseline_workgroups as f64,
    );
    println!(
        "  barrier encounters: current={baseline_barriers}, Cin16={t128_c16_barriers}, Cin8={t128_c8_barriers}"
    );
    println!(
        "  input read={:.3}->{:.3}MiB, weight read={:.3}->{:.3}MiB, bias read={:.3}->{:.3}MiB, output write={:.3}MiB",
        mib(baseline_traffic.input_read_bytes),
        mib(t128_traffic.input_read_bytes),
        mib(baseline_traffic.weight_read_bytes),
        mib(t128_traffic.weight_read_bytes),
        mib(baseline_traffic.bias_read_bytes),
        mib(t128_traffic.bias_read_bytes),
        mib(t128_traffic.output_write_bytes),
    );
    println!(
        "  total buffer traffic={:.3}->{:.3}MiB ({:.1}% reduction), persistent T128 bytes=0",
        mib(baseline_traffic.total_bytes()),
        mib(t128_traffic.total_bytes()),
        100.0 * (baseline_traffic.total_bytes() - t128_traffic.total_bytes()) as f64
            / baseline_traffic.total_bytes() as f64,
    );
    for dilation in [
        Conv1dK7Dilation::One,
        Conv1dK7Dilation::Three,
        Conv1dK7Dilation::Nine,
    ] {
        println!(
            "  shared d={}: Cin16={}B Cin8={}B",
            dilation.value(),
            Conv1dK7T128Tile::Cin16.shared_memory_bytes(dilation),
            Conv1dK7T128Tile::Cin8.shared_memory_bytes(dilation),
        );
    }
}

fn print_summary(results: &[CaseResult]) {
    let sums = std::array::from_fn::<_, VARIANT_COUNT, _>(|variant| {
        results
            .iter()
            .map(|result| result.timings[variant].median_us)
            .sum::<f64>()
    });
    let pytorch_sum = results
        .iter()
        .map(|result| result.case.pytorch_strict_us)
        .sum::<f64>();
    let total_macs = results
        .iter()
        .map(|result| model_macs(result.case))
        .sum::<u128>();
    let bit_mismatches = std::array::from_fn::<_, 2, _>(|tile| {
        results
            .iter()
            .map(|result| result.comparisons[tile].mismatched_bits)
            .sum::<usize>()
    });
    let max_abs = std::array::from_fn::<_, 2, _>(|tile| {
        results
            .iter()
            .map(|result| result.comparisons[tile].max_abs)
            .fold(0.0_f32, f32::max)
    });
    let selected_min_sum = results
        .iter()
        .map(|result| {
            result
                .timings
                .iter()
                .map(|timing| timing.median_us)
                .min_by(f64::total_cmp)
                .expect("every case has three timings")
        })
        .sum::<f64>();
    let production_t128_sum = results
        .iter()
        .map(|result| {
            let tile = production_tile_for_shape(
                result.case.channels,
                result.case.length,
                result.case.dilation,
            )
            .expect("all benchmark cases are released production shapes");
            let timing_index = T128_TILES
                .iter()
                .position(|measured| *measured == tile)
                .expect("production tile must be one of the two measured T128 tiles")
                + 1;
            result.timings[timing_index].median_us
        })
        .sum::<f64>();

    println!("all-twelve sums of independently measured medians:");
    println!(
        "  current-selector median_sum={:.3} us TMAC/s={:.3}",
        sums[0],
        tmac_per_second(total_macs, sums[0]),
    );
    for (index, tile) in T128_TILES.into_iter().enumerate() {
        println!(
            "  {:<16} median_sum={:.3} us speedup={:.3}x TMAC/s={:.3} bit_mismatch={} max_abs={:.9e}",
            tile.label(),
            sums[index + 1],
            sums[0] / sums[index + 1],
            tmac_per_second(total_macs, sums[index + 1]),
            bit_mismatches[index],
            max_abs[index],
        );
    }
    println!(
        "  production conservative T128 selector median_sum={production_t128_sum:.3} us speedup={:.3}x",
        sums[0] / production_t128_sum,
    );
    println!(
        "  per-shape selected-min median_sum={selected_min_sum:.3} us speedup={:.3}x",
        sums[0] / selected_min_sum,
    );
    println!(
        "  PyTorch strict FP32 same-card threshold={pytorch_sum:.3} us; current/threshold={:.3}x, production-T128/threshold={:.3}x, selected-min/threshold={:.3}x",
        sums[0] / pytorch_sum,
        production_t128_sum / pytorch_sum,
        selected_min_sum / pytorch_sum,
    );

    let routed = results
        .iter()
        .map(|result| result.baseline_tile.label())
        .collect::<Vec<_>>();
    println!("  verified current routes={routed:?}");
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let (device, monitor) = initialize_wgpu(args.adapter_index);
    B::seed(&device, 0);
    println!(
        "production T128/O32 k7 benchmark: warmup={} iterations={} trials={} variants=3 cases=12 seed=0",
        args.warmup, args.iterations, args.trials,
    );
    println!(
        "vector contract: four logical vec4 accumulators/thread; component-wise fma only; scalar output scatter; backend scalarisation is allowed and must be decided by timing"
    );
    print_static_accounting();

    let mut results = Vec::with_capacity(CASES.len());
    for case in CASES {
        results.push(benchmark_case(&device, case, &args)?);
        synchronize_and_check_wgpu(
            &device,
            &monitor,
            &format!(
                "C={} L={} d={} T128 benchmark",
                case.channels,
                case.length,
                case.dilation.value()
            ),
        )?;
    }
    print_summary(&results);
    synchronize_and_check_wgpu(&device, &monitor, "T128 benchmark completion")?;
    println!("wgpu_uncaptured_errors=0");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_accounting_matches_audit_totals() {
        let current_workgroups = CASES
            .iter()
            .copied()
            .map(|case| {
                workgroups(
                    case,
                    CURRENT_TIME_TILE,
                    production_route_tile(case).output_channel_tile(),
                )
            })
            .sum::<usize>();
        let t128_workgroups = CASES
            .iter()
            .copied()
            .map(|case| workgroups(case, T128_TIME_TILE, T128_OUTPUT_TILE))
            .sum::<usize>();
        assert_eq!(current_workgroups, 22_596);
        assert_eq!(t128_workgroups, 15_552);

        let current_barriers = CASES
            .iter()
            .copied()
            .map(|case| {
                let groups = workgroups(
                    case,
                    CURRENT_TIME_TILE,
                    production_route_tile(case).output_channel_tile(),
                );
                barrier_encounters(case, groups, 16)
            })
            .sum::<usize>();
        let t128_c16_barriers = CASES
            .iter()
            .copied()
            .map(|case| {
                barrier_encounters(case, workgroups(case, T128_TIME_TILE, T128_OUTPUT_TILE), 16)
            })
            .sum::<usize>();
        let t128_c8_barriers = CASES
            .iter()
            .copied()
            .map(|case| {
                barrier_encounters(case, workgroups(case, T128_TIME_TILE, T128_OUTPUT_TILE), 8)
            })
            .sum::<usize>();
        assert_eq!(current_barriers, 530_928);
        assert_eq!(t128_c16_barriers, 358_776);
        assert_eq!(t128_c8_barriers, 717_552);
    }

    #[test]
    fn strict_python_threshold_is_the_recorded_same_card_sum() {
        let sum = CASES.iter().map(|case| case.pytorch_strict_us).sum::<f64>();
        assert!((sum - 30_825.489_273).abs() < 1.0e-9);
        assert_eq!(
            CASES.iter().copied().map(model_macs).sum::<u128>(),
            81_749_606_400
        );
    }

    #[test]
    fn exact_guarded_buffer_traffic_matches_audit() {
        let mut current = BufferTraffic::default();
        let mut t128 = BufferTraffic::default();
        for case in CASES {
            current.accumulate(buffer_traffic(
                case,
                CURRENT_TIME_TILE,
                production_route_tile(case).output_channel_tile(),
            ));
            t128.accumulate(buffer_traffic(case, T128_TIME_TILE, T128_OUTPUT_TILE));
        }
        assert_eq!(current.weight_read_bytes, 5_143_412_736);
        assert_eq!(t128.weight_read_bytes, 2_571_706_368);
        assert_eq!(current.input_read_bytes, 1_583_910_144);
        assert_eq!(t128.input_read_bytes, 1_750_678_272);
        assert_eq!(current.bias_read_bytes, 63_700_992);
        assert_eq!(t128.bias_read_bytes, 31_850_496);
        assert_eq!(current.output_write_bytes, 254_361_600);
        assert_eq!(t128.output_write_bytes, 254_361_600);
        assert!(t128.total_bytes() < current.total_bytes());
    }
}
