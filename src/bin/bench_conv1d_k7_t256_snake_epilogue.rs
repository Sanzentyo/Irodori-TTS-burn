//! Fair isolated A/B for production T256/O32 k=7 Conv1d + act1 Snake tiles.
//!
//! Baseline: the directly imported production T128+Snake fused kernel.
//! Comparands: the directly imported production T256/O32 Cin8+Snake and
//! Cin16+Snake launchers.

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
            Conv1dK7T128Tile, LaunchGeometry as T128LaunchGeometry,
            conv1d_k7_t128_contract_is_compatible, production_tile_for_shape,
        },
        conv1d_k7_t128_snake_epilogue::{
            conv1d_k7_same_t128_snake_epilogue_wgsl,
            conv1d_k7_t128_snake_epilogue_contract_is_compatible,
        },
        conv1d_k7_t256_snake_epilogue::{
            ACCUMULATORS_PER_INVOCATION as T256_ACCUMULATORS, Conv1dK7T256Tile,
            LaunchGeometry as T256LaunchGeometry, conv1d_k7_same_t256_snake_epilogue_wgsl,
            conv1d_k7_t256_snake_epilogue_contract_is_compatible,
        },
        conv1d_k7_tiled::Conv1dK7Dilation,
    },
};

type B = WgpuRaw;

const KERNEL_SIZE: usize = 7;
const DEFAULT_WARMUP: usize = 10;
const DEFAULT_ITERATIONS: usize = 50;
const DEFAULT_TRIALS: usize = 5;
const VARIANT_COUNT: usize = 3;
const T256_TILES: [Conv1dK7T256Tile; 2] = [Conv1dK7T256Tile::Cin8, Conv1dK7T256Tile::Cin16];

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
    fn require_exact(self, label: &str) -> Result<Self, Box<dyn Error>> {
        if self.mismatched_bits == 0 && self.max_abs == 0.0 {
            Ok(self)
        } else {
            Err(io::Error::other(format!(
                "{label} is not bit-exact: bit_mismatch={} max_abs={:.9e}",
                self.mismatched_bits, self.max_abs,
            ))
            .into())
        }
    }
}

#[derive(Debug)]
struct CaseResult {
    case: ConvCase,
    production_tile: Conv1dK7T128Tile,
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
    "usage: bench_conv1d_k7_t256_snake_epilogue <adapter-index> \
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

fn production_tile(case: ConvCase) -> Conv1dK7T128Tile {
    production_tile_for_shape(case.channels, case.length, case.dilation)
        .expect("all benchmark cases are exact released decoder shapes")
}

fn production_t128_snake_forward(
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

fn t256_forward(
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    alpha: &Tensor<B, 3>,
    case: ConvCase,
    tile: Conv1dK7T256Tile,
) -> Tensor<B, 3> {
    let output = conv1d_k7_same_t256_snake_epilogue_wgsl(
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
        0 => production_t128_snake_forward(input, weight, bias, alpha, case, tile),
        1 | 2 => t256_forward(input, weight, bias, alpha, case, T256_TILES[variant - 1]),
        _ => unreachable!("benchmark has exactly three variants"),
    }
}

fn tensor_values(tensor: Tensor<B, 3>, case: ConvCase) -> Result<Vec<f32>, Box<dyn Error>> {
    let expected_shape = [1, case.channels, case.length];
    let actual_shape = tensor.dims();
    if actual_shape != expected_shape {
        return Err(io::Error::other(format!(
            "output shape mismatch: expected={expected_shape:?} actual={actual_shape:?}"
        ))
        .into());
    }
    Ok(tensor.into_data().to_vec::<f32>()?)
}

fn compare_outputs(expected: &[f32], actual: &[f32]) -> Result<Comparison, Box<dyn Error>> {
    if expected.len() != actual.len() {
        return Err(io::Error::other(format!(
            "output length mismatch: T128={} T256={}",
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
                "non-finite output pair: T128={expected:?} T256={actual:?}"
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

fn tmac_per_second(macs: u128, timing_us: f64) -> f64 {
    macs as f64 / (timing_us * 1.0e6)
}

fn check_contracts(
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    alpha: &Tensor<B, 3>,
    case: ConvCase,
    production_tile: Conv1dK7T128Tile,
) -> Result<(), Box<dyn Error>> {
    let input = input.clone().into_primitive().tensor();
    let weight = weight.clone().into_primitive().tensor();
    let bias = bias.clone().into_primitive().tensor();
    let alpha = alpha.clone().into_primitive().tensor();
    if !conv1d_k7_t128_contract_is_compatible(
        &input,
        &weight,
        &bias,
        case.dilation,
        production_tile,
    ) || !conv1d_k7_t128_snake_epilogue_contract_is_compatible(
        &input,
        &weight,
        &bias,
        &alpha,
        case.dilation,
        production_tile,
    ) {
        return Err(io::Error::other(format!(
            "production T128+Snake contract failed for C={} L={} d={} tile={}",
            case.channels,
            case.length,
            case.dilation.value(),
            production_tile.label(),
        ))
        .into());
    }
    for tile in T256_TILES {
        if !conv1d_k7_t256_snake_epilogue_contract_is_compatible(
            &input,
            &weight,
            &bias,
            &alpha,
            case.dilation,
            tile,
        ) {
            let hardware = &input.client.properties().hardware;
            return Err(io::Error::other(format!(
                "{}+Snake contract failed for C={} L={} d={} max_bindings={} max_shared={}B max_units={}",
                tile.label(),
                case.channels,
                case.length,
                case.dilation.value(),
                hardware.max_bindings,
                hardware.max_shared_memory_size,
                hardware.max_units_per_cube,
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
    let production_tile = production_tile(case);
    check_contracts(&input, &weight, &bias, &alpha, case, production_tile)?;

    let expected = tensor_values(
        production_t128_snake_forward(&input, &weight, &bias, &alpha, case, production_tile),
        case,
    )?;
    let t256_cin8 = tensor_values(
        t256_forward(&input, &weight, &bias, &alpha, case, Conv1dK7T256Tile::Cin8),
        case,
    )?;
    let t256_cin16 = tensor_values(
        t256_forward(
            &input,
            &weight,
            &bias,
            &alpha,
            case,
            Conv1dK7T256Tile::Cin16,
        ),
        case,
    )?;
    let comparisons = [
        compare_outputs(&expected, &t256_cin8)?.require_exact("T256/Cin8+Snake")?,
        compare_outputs(&expected, &t256_cin16)?.require_exact("T256/Cin16+Snake")?,
    ];

    for variant in 0..VARIANT_COUNT {
        let mut operation = || {
            variant_forward(
                variant,
                &input,
                &weight,
                &bias,
                &alpha,
                case,
                production_tile,
            )
        };
        warm_up(args.warmup, case, &mut operation);
    }
    let mut samples: [Vec<f64>; VARIANT_COUNT] =
        std::array::from_fn(|_| Vec::with_capacity(args.trials));
    for trial in 0..args.trials {
        for offset in 0..VARIANT_COUNT {
            let variant = (trial + offset) % VARIANT_COUNT;
            let mut operation = || {
                variant_forward(
                    variant,
                    &input,
                    &weight,
                    &bias,
                    &alpha,
                    case,
                    production_tile,
                )
            };
            samples[variant].push(measure(args.iterations, case, &mut operation));
        }
    }
    let timings = std::array::from_fn(|index| summarize_samples(&samples[index]));

    let production_geometry =
        T128LaunchGeometry::new(case.channels, case.length, case.dilation, production_tile)
            .expect("released T128 geometry must be valid");
    let t256_geometry = T256LaunchGeometry::new(
        case.channels,
        case.length,
        case.dilation,
        Conv1dK7T256Tile::Cin8,
    )
    .expect("released T256 geometry must be valid");
    let production_workgroups = production_geometry
        .workgroups()
        .expect("released T128 workgroups must fit usize");
    let t256_workgroups = t256_geometry
        .workgroups()
        .expect("released T256 workgroups must fit usize");
    let macs = model_macs(case);
    println!(
        "C={:4} L={:6} d={} shape=[1,{},{}] model_MAC={:.6}G",
        case.channels,
        case.length,
        case.dilation.value(),
        case.channels,
        case.length,
        macs as f64 / 1.0e9,
    );
    println!(
        "  production {:24}+Snake median={:10.3}us range=[{:10.3},{:10.3}] dispatch=1 workgroups={} shared={}B TMAC/s={:.3}",
        production_tile.label(),
        timings[0].median_us,
        timings[0].min_us,
        timings[0].max_us,
        production_workgroups,
        production_tile.shared_memory_bytes(case.dilation),
        tmac_per_second(macs, timings[0].median_us),
    );
    for (index, tile) in T256_TILES.into_iter().enumerate() {
        let timing = timings[index + 1];
        let comparison = comparisons[index];
        println!(
            "  production {:24}+Snake median={:10.3}us range=[{:10.3},{:10.3}] dispatch=1 workgroups={} shared={}B TMAC/s={:.3} vs-T128={:.3}x",
            tile.label(),
            timing.median_us,
            timing.min_us,
            timing.max_us,
            t256_workgroups,
            tile.shared_memory_bytes(case.dilation),
            tmac_per_second(macs, timing.median_us),
            timings[0].median_us / timing.median_us,
        );
        println!(
            "    correctness shape=ok elements={} bit_mismatch={} max_abs={:.9e} finite=true",
            comparison.elements, comparison.mismatched_bits, comparison.max_abs,
        );
    }

    Ok(CaseResult {
        case,
        production_tile,
        timings,
        comparisons,
    })
}

fn variant_label(variant: usize) -> &'static str {
    match variant {
        0 => "production-t128+snake",
        1 => "t256-cin8+snake",
        2 => "t256-cin16+snake",
        _ => unreachable!("benchmark has exactly three variants"),
    }
}

fn print_static_accounting() {
    let t128_workgroups = CASES
        .into_iter()
        .map(|case| {
            T128LaunchGeometry::new(
                case.channels,
                case.length,
                case.dilation,
                production_tile(case),
            )
            .and_then(T128LaunchGeometry::workgroups)
            .expect("released T128 workgroups must fit usize")
        })
        .sum::<usize>();
    let t256_workgroups = CASES
        .into_iter()
        .map(|case| {
            T256LaunchGeometry::new(
                case.channels,
                case.length,
                case.dilation,
                Conv1dK7T256Tile::Cin8,
            )
            .and_then(T256LaunchGeometry::workgroups)
            .expect("released T256 workgroups must fit usize")
        })
        .sum::<usize>();
    let output_elements = CASES
        .into_iter()
        .map(|case| case.channels * case.length)
        .sum::<usize>();
    println!("static exact12 fused accounting:");
    println!(
        "  dispatch=12 for every fixed route; workgroups T128={t128_workgroups} T256={t256_workgroups}; T256 accumulators={T256_ACCUMULATORS} f32/thread; standalone-Snake dispatch/materialization=0"
    );
    println!(
        "  logical alpha reads={}B output writes={}B for every fused route; persistent bytes=0",
        output_elements * core::mem::size_of::<f32>(),
        output_elements * core::mem::size_of::<f32>(),
    );
    for dilation in [
        Conv1dK7Dilation::One,
        Conv1dK7Dilation::Three,
        Conv1dK7Dilation::Nine,
    ] {
        println!(
            "  shared d={}: T256-Cin8={}B T256-Cin16={}B",
            dilation.value(),
            Conv1dK7T256Tile::Cin8.shared_memory_bytes(dilation),
            Conv1dK7T256Tile::Cin16.shared_memory_bytes(dilation),
        );
    }
}

fn print_summary(results: &[CaseResult]) {
    let total_macs = results
        .iter()
        .map(|result| model_macs(result.case))
        .sum::<u128>();
    let sums: [Timing; VARIANT_COUNT] = std::array::from_fn(|variant| Timing {
        median_us: results
            .iter()
            .map(|result| result.timings[variant].median_us)
            .sum(),
        min_us: results
            .iter()
            .map(|result| result.timings[variant].min_us)
            .sum(),
        max_us: results
            .iter()
            .map(|result| result.timings[variant].max_us)
            .sum(),
    });
    println!("exact12 sums of independently measured per-shape statistics:");
    for variant in 0..VARIANT_COUNT {
        let timing = sums[variant];
        println!(
            "  {:24} median-sum={:.3}us range-sum=[{:.3},{:.3}] TMAC/s={:.3} vs-production={:.3}x",
            variant_label(variant),
            timing.median_us,
            timing.min_us,
            timing.max_us,
            tmac_per_second(total_macs, timing.median_us),
            sums[0].median_us / timing.median_us,
        );
    }
    for (index, tile) in T256_TILES.into_iter().enumerate() {
        let mismatched_bits = results
            .iter()
            .map(|result| result.comparisons[index].mismatched_bits)
            .sum::<usize>();
        let max_abs = results
            .iter()
            .map(|result| result.comparisons[index].max_abs)
            .fold(0.0_f32, f32::max);
        println!(
            "  correctness {:24}+Snake shapes=12/12 bit_mismatch={mismatched_bits} max_abs={max_abs:.9e} finite=true",
            tile.label(),
        );
    }
    let selected_sum = results
        .iter()
        .map(|result| {
            result
                .timings
                .iter()
                .map(|timing| timing.median_us)
                .fold(f64::INFINITY, f64::min)
        })
        .sum::<f64>();
    println!(
        "  per-shape selected minimum median-sum={selected_sum:.3}us vs fixed production={:.3}x",
        sums[0].median_us / selected_sum,
    );
    for result in results {
        let (winner, winner_us) = result
            .timings
            .iter()
            .enumerate()
            .map(|(variant, timing)| (variant_label(variant), timing.median_us))
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .expect("three benchmark variants are present");
        println!(
            "  route C={} L={} d={} production_tile={} winner={} {:.3}us",
            result.case.channels,
            result.case.length,
            result.case.dilation.value(),
            result.production_tile.label(),
            winner,
            winner_us,
        );
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let (device, monitor) = initialize_wgpu(args.adapter_index);
    B::seed(&device, 0);
    println!(
        "isolated T256/O32+Snake benchmark: warmup={} iterations={} trials={} variants=3 cases=12 seed=0",
        args.warmup, args.iterations, args.trials,
    );
    println!(
        "fairness: direct production T128+Snake fused baseline; identical alpha/input/weight/bias; rotating variant order; full-output shape/finite/bit0/maxabs0"
    );
    print_static_accounting();

    let mut results = Vec::with_capacity(CASES.len());
    for case in CASES {
        results.push(benchmark_case(&device, case, &args)?);
        synchronize_and_check_wgpu(
            &device,
            &monitor,
            &format!(
                "C={} L={} d={} T256+Snake benchmark",
                case.channels,
                case.length,
                case.dilation.value()
            ),
        )?;
    }
    print_summary(&results);
    synchronize_and_check_wgpu(&device, &monitor, "T256+Snake benchmark completion")?;
    println!("wgpu_uncaptured_errors=0");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact12_selector_and_workgroups_are_fixed() {
        let tiles = CASES.map(production_tile);
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
        let t128_workgroups = CASES
            .into_iter()
            .map(|case| {
                T128LaunchGeometry::new(
                    case.channels,
                    case.length,
                    case.dilation,
                    production_tile(case),
                )
                .and_then(T128LaunchGeometry::workgroups)
                .expect("released T128 workgroups")
            })
            .sum::<usize>();
        let t256_workgroups = CASES
            .into_iter()
            .map(|case| {
                T256LaunchGeometry::new(
                    case.channels,
                    case.length,
                    case.dilation,
                    Conv1dK7T256Tile::Cin8,
                )
                .and_then(T256LaunchGeometry::workgroups)
                .expect("released T256 workgroups")
            })
            .sum::<usize>();
        assert_eq!(t128_workgroups, 15_552);
        assert_eq!(t256_workgroups, 7_839);
    }

    #[test]
    fn tile_order_and_c96_tail_contract_are_fixed() {
        assert_eq!(T256_TILES[0], Conv1dK7T256Tile::Cin8);
        assert_eq!(T256_TILES[1], Conv1dK7T256Tile::Cin16);
        for case in CASES.into_iter().filter(|case| case.channels == 96) {
            assert_eq!(case.channels % 32, 0);
            assert_eq!(case.length % 256, 0);
            assert_eq!(
                T256LaunchGeometry::new(
                    case.channels,
                    case.length,
                    case.dilation,
                    Conv1dK7T256Tile::Cin8,
                )
                .and_then(T256LaunchGeometry::workgroups),
                Some(1_125)
            );
        }
    }

    #[test]
    fn aggregate_output_and_alpha_bytes_are_exact() {
        let elements = CASES
            .into_iter()
            .map(|case| case.channels * case.length)
            .sum::<usize>();
        assert_eq!(elements, 63_590_400);
        assert_eq!(elements * core::mem::size_of::<f32>(), 254_361_600);
    }
}
