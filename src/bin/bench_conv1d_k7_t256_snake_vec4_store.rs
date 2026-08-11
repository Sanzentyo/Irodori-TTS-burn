//! Direct production A/B of T256+Snake scalar stores versus vec4 stores.
//!
//! All nine physically compatible routes remain in the audit matrix so the
//! C768/L600/d9 loss stays visible; the production codec selector promotes only
//! the other eight measured wins.

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
        conv1d_k7_t256_snake_epilogue::{
            Conv1dK7T256Tile, LaunchGeometry, conv1d_k7_same_t256_snake_epilogue_wgsl,
            conv1d_k7_t256_snake_epilogue_contract_is_compatible, production_tile_for_shape,
        },
        conv1d_k7_t256_snake_vec4_store::{
            conv1d_k7_t256_snake_vec4_store_contract_is_compatible,
            try_conv1d_k7_same_t256_snake_vec4_store_wgsl,
        },
        conv1d_k7_tiled::Conv1dK7Dilation,
    },
};

type B = WgpuRaw;

const KERNEL_SIZE: usize = 7;
const DEFAULT_WARMUP: usize = 10;
const DEFAULT_ITERATIONS: usize = 50;
const DEFAULT_TRIALS: usize = 5;
const VARIANT_COUNT: usize = 2;

#[derive(Clone, Copy, Debug)]
struct ConvCase {
    channels: usize,
    length: usize,
    dilation: Conv1dK7Dilation,
    tile: Conv1dK7T256Tile,
}

const CASES: [ConvCase; 9] = [
    ConvCase {
        channels: 768,
        length: 600,
        dilation: Conv1dK7Dilation::One,
        tile: Conv1dK7T256Tile::Cin16,
    },
    ConvCase {
        channels: 768,
        length: 600,
        dilation: Conv1dK7Dilation::Nine,
        tile: Conv1dK7T256Tile::Cin16,
    },
    ConvCase {
        channels: 384,
        length: 6_000,
        dilation: Conv1dK7Dilation::One,
        tile: Conv1dK7T256Tile::Cin16,
    },
    ConvCase {
        channels: 192,
        length: 48_000,
        dilation: Conv1dK7Dilation::One,
        tile: Conv1dK7T256Tile::Cin16,
    },
    ConvCase {
        channels: 192,
        length: 48_000,
        dilation: Conv1dK7Dilation::Three,
        tile: Conv1dK7T256Tile::Cin16,
    },
    ConvCase {
        channels: 192,
        length: 48_000,
        dilation: Conv1dK7Dilation::Nine,
        tile: Conv1dK7T256Tile::Cin8,
    },
    ConvCase {
        channels: 96,
        length: 96_000,
        dilation: Conv1dK7Dilation::One,
        tile: Conv1dK7T256Tile::Cin16,
    },
    ConvCase {
        channels: 96,
        length: 96_000,
        dilation: Conv1dK7Dilation::Three,
        tile: Conv1dK7T256Tile::Cin16,
    },
    ConvCase {
        channels: 96,
        length: 96_000,
        dilation: Conv1dK7Dilation::Nine,
        tile: Conv1dK7T256Tile::Cin8,
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
                "production vec4 store is not bit-exact: bit_mismatch={} max_abs={:.9e}",
                self.mismatched_bits, self.max_abs,
            ))
            .into())
        }
    }
}

#[derive(Debug)]
struct CaseResult {
    case: ConvCase,
    timings: [Timing; VARIANT_COUNT],
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
    "usage: bench_conv1d_k7_t256_snake_vec4_store <adapter-index> \
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

fn production_forward(
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    alpha: &Tensor<B, 3>,
    case: ConvCase,
) -> Tensor<B, 3> {
    let output = conv1d_k7_same_t256_snake_epilogue_wgsl(
        input.clone().into_primitive().tensor(),
        weight.clone().into_primitive().tensor(),
        bias.clone().into_primitive().tensor(),
        alpha.clone().into_primitive().tensor(),
        case.dilation,
        case.tile,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn vec4_store_forward(
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    alpha: &Tensor<B, 3>,
    case: ConvCase,
) -> Tensor<B, 3> {
    let output = try_conv1d_k7_same_t256_snake_vec4_store_wgsl(
        input.clone().into_primitive().tensor(),
        weight.clone().into_primitive().tensor(),
        bias.clone().into_primitive().tensor(),
        alpha.clone().into_primitive().tensor(),
        case.dilation,
        case.tile,
    )
    .expect("preflighted measured T256 vec4-store launch must succeed");
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn variant_forward(
    variant: usize,
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    alpha: &Tensor<B, 3>,
    case: ConvCase,
) -> Tensor<B, 3> {
    match variant {
        0 => production_forward(input, weight, bias, alpha, case),
        1 => vec4_store_forward(input, weight, bias, alpha, case),
        _ => unreachable!("benchmark has exactly two variants"),
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
            "output length mismatch: scalar-store={} vec4-store={}",
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
                "non-finite output pair: scalar-store={expected:?} vec4-store={actual:?}"
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
) -> Result<(), Box<dyn Error>> {
    let input = input.clone().into_primitive().tensor();
    let weight = weight.clone().into_primitive().tensor();
    let bias = bias.clone().into_primitive().tensor();
    let alpha = alpha.clone().into_primitive().tensor();
    if production_tile_for_shape(case.channels, case.length, case.dilation) != Some(case.tile) {
        return Err(
            io::Error::other("benchmark case is not an accepted production T256 route").into(),
        );
    }
    if !conv1d_k7_t256_snake_epilogue_contract_is_compatible(
        &input,
        &weight,
        &bias,
        &alpha,
        case.dilation,
        case.tile,
    ) || !conv1d_k7_t256_snake_vec4_store_contract_is_compatible(
        &input,
        &weight,
        &bias,
        &alpha,
        case.dilation,
        case.tile,
    ) {
        let properties = input.client.properties();
        return Err(io::Error::other(format!(
            "contract failed for C={} L={} d={} tile={} allocator_alignment={} max_bindings={} max_shared={}B max_units={}",
            case.channels,
            case.length,
            case.dilation.value(),
            case.tile.label(),
            properties.memory.alignment,
            properties.hardware.max_bindings,
            properties.hardware.max_shared_memory_size,
            properties.hardware.max_units_per_cube,
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
    check_contracts(&input, &weight, &bias, &alpha, case)?;

    let expected = tensor_values(
        production_forward(&input, &weight, &bias, &alpha, case),
        case,
    )?;
    let actual = tensor_values(
        vec4_store_forward(&input, &weight, &bias, &alpha, case),
        case,
    )?;
    let comparison = compare_outputs(&expected, &actual)?.require_exact()?;

    for variant in 0..VARIANT_COUNT {
        let mut operation = || variant_forward(variant, &input, &weight, &bias, &alpha, case);
        warm_up(args.warmup, case, &mut operation);
    }
    let mut samples: [Vec<f64>; VARIANT_COUNT] =
        std::array::from_fn(|_| Vec::with_capacity(args.trials));
    for trial in 0..args.trials {
        for offset in 0..VARIANT_COUNT {
            let variant = (trial + offset) % VARIANT_COUNT;
            let mut operation = || variant_forward(variant, &input, &weight, &bias, &alpha, case);
            samples[variant].push(measure(args.iterations, case, &mut operation));
        }
    }
    let timings = std::array::from_fn(|index| summarize_samples(&samples[index]));

    let geometry = LaunchGeometry::new(case.channels, case.length, case.dilation, case.tile)
        .expect("accepted T256 route geometry must be valid");
    let workgroups = geometry
        .workgroups()
        .expect("accepted T256 route workgroups must fit usize");
    let output_elements = case.channels * case.length;
    let output_bytes = output_elements * core::mem::size_of::<f32>();
    let macs = model_macs(case);
    println!(
        "C={:4} L={:6} d={} tile={} shape=[1,{},{}] model_MAC={:.6}G workgroups={} shared={}B",
        case.channels,
        case.length,
        case.dilation.value(),
        case.tile.label(),
        case.channels,
        case.length,
        macs as f64 / 1.0e9,
        workgroups,
        geometry.shared_bytes,
    );
    println!(
        "  output bytes={} scalar_WGSL_store_executions={} vec4_WGSL_store_executions={} bytes/WGSL-store=4->16 execution_ratio=4.000x",
        output_bytes,
        output_elements,
        output_elements / 4,
    );
    println!(
        "  production scalar-store median={:10.3}us range=[{:10.3},{:10.3}] TMAC/s={:.3}",
        timings[0].median_us,
        timings[0].min_us,
        timings[0].max_us,
        tmac_per_second(macs, timings[0].median_us),
    );
    println!(
        "  production vec4-store   median={:10.3}us range=[{:10.3},{:10.3}] TMAC/s={:.3} speedup={:.3}x",
        timings[1].median_us,
        timings[1].min_us,
        timings[1].max_us,
        tmac_per_second(macs, timings[1].median_us),
        timings[0].median_us / timings[1].median_us,
    );
    println!(
        "    correctness shape=ok elements={} bit_mismatch={} max_abs={:.9e} finite=true",
        comparison.elements, comparison.mismatched_bits, comparison.max_abs,
    );

    Ok(CaseResult {
        case,
        timings,
        comparison,
    })
}

fn variant_label(variant: usize) -> &'static str {
    match variant {
        0 => "production-scalar-store",
        1 => "production-vec4-store",
        _ => unreachable!("benchmark has exactly two variants"),
    }
}

fn print_static_accounting() {
    let workgroups = CASES
        .into_iter()
        .map(|case| {
            LaunchGeometry::new(case.channels, case.length, case.dilation, case.tile)
                .and_then(LaunchGeometry::workgroups)
                .expect("accepted route workgroups must fit usize")
        })
        .sum::<usize>();
    let output_elements = CASES
        .into_iter()
        .map(|case| case.channels * case.length)
        .sum::<usize>();
    let output_bytes = output_elements * core::mem::size_of::<f32>();
    println!("static accepted9 accounting:");
    println!(
        "  dispatches=9 workgroups={workgroups} output_elements={output_elements} output_bytes={output_bytes}"
    );
    println!(
        "  scalar dynamic WGSL store executions={output_elements}; vec4 dynamic WGSL store executions={}; reduction=4.000x; written bytes identical",
        output_elements / 4,
    );
    println!(
        "  WGSL output assignment sites=4 scalar versus 1 vec4 per store helper; hardware lowering/store-instruction count is not assumed; conv/FMA/shared/barrier/alpha sequence unchanged; persistent bytes=0"
    );
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
    println!("accepted9 sums of independently measured per-shape statistics:");
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
    let mismatched_bits = results
        .iter()
        .map(|result| result.comparison.mismatched_bits)
        .sum::<usize>();
    let max_abs = results
        .iter()
        .map(|result| result.comparison.max_abs)
        .fold(0.0_f32, f32::max);
    let elements = results
        .iter()
        .map(|result| result.comparison.elements)
        .sum::<usize>();
    println!(
        "  correctness shapes=9/9 elements={elements} bit_mismatch={mismatched_bits} max_abs={max_abs:.9e} finite=true hard_gate=pass"
    );
    for result in results {
        println!(
            "  route C={} L={} d={} tile={} production={:.3}us vec4={:.3}us speedup={:.3}x",
            result.case.channels,
            result.case.length,
            result.case.dilation.value(),
            result.case.tile.label(),
            result.timings[0].median_us,
            result.timings[1].median_us,
            result.timings[0].median_us / result.timings[1].median_us,
        );
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let (device, monitor) = initialize_wgpu(args.adapter_index);
    B::seed(&device, 0);
    let allocator_alignment = WgpuRuntime::client(&device).properties().memory.alignment;
    println!(
        "isolated T256+Snake output-store A/B: warmup={} iterations={} trials={} variants=2 cases=9 seed=0 allocator_alignment={}B",
        args.warmup, args.iterations, args.trials, allocator_alignment,
    );
    println!(
        "fairness: direct production T256+Snake baseline; identical selected tile/alpha/input/weight/bias; rotating order; full-output shape/finite/bit0/maxabs0 hard gate"
    );
    print_static_accounting();

    let mut results = Vec::with_capacity(CASES.len());
    for case in CASES {
        results.push(benchmark_case(&device, case, &args)?);
        synchronize_and_check_wgpu(
            &device,
            &monitor,
            &format!(
                "C={} L={} d={} vec4-store A/B",
                case.channels,
                case.length,
                case.dilation.value()
            ),
        )?;
    }
    print_summary(&results);
    synchronize_and_check_wgpu(&device, &monitor, "vec4-store A/B completion")?;
    println!("wgpu_uncaptured_errors=0");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_matrix_is_exactly_the_nine_production_t256_routes() {
        assert_eq!(CASES.len(), 9);
        for case in CASES {
            assert_eq!(
                production_tile_for_shape(case.channels, case.length, case.dilation),
                Some(case.tile)
            );
            assert!(case.length.is_multiple_of(4));
        }
        assert_eq!(
            CASES
                .into_iter()
                .filter(|case| case.tile == Conv1dK7T256Tile::Cin16)
                .count(),
            7
        );
        assert_eq!(
            CASES
                .into_iter()
                .filter(|case| case.tile == Conv1dK7T256Tile::Cin8)
                .count(),
            2
        );
    }

    #[test]
    fn accepted9_store_and_workgroup_accounting_is_exact() {
        let elements = CASES
            .into_iter()
            .map(|case| case.channels * case.length)
            .sum::<usize>();
        let workgroups = CASES
            .into_iter()
            .map(|case| {
                LaunchGeometry::new(case.channels, case.length, case.dilation, case.tile)
                    .and_then(LaunchGeometry::workgroups)
                    .expect("accepted route workgroups")
            })
            .sum::<usize>();
        assert_eq!(elements, 58_521_600);
        assert_eq!(elements * core::mem::size_of::<f32>(), 234_086_400);
        assert_eq!(elements / 4, 14_630_400);
        assert_eq!(workgroups, 7_191);
    }

    #[test]
    fn defaults_and_rotation_are_fixed() {
        assert_eq!(
            (DEFAULT_WARMUP, DEFAULT_ITERATIONS, DEFAULT_TRIALS),
            (10, 50, 5)
        );
        let orders = (0..DEFAULT_TRIALS)
            .map(|trial| {
                (0..VARIANT_COUNT)
                    .map(|offset| (trial + offset) % VARIANT_COUNT)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            orders,
            vec![vec![0, 1], vec![1, 0], vec![0, 1], vec![1, 0], vec![0, 1]]
        );
    }
}
