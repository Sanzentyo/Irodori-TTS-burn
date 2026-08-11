//! Production acceptance replay for residue-class d1 decomposition.
//!
//! The baseline is the prior fused T256+Snake vec4 route for the exact
//! C192/L48000 d3 and d9 decoder calls. The production residue path includes one compact
//! residue pack and one dilation-one fused-Snake core per call, with direct NCL
//! scatter and no output unpack. Both paths are imported directly from the
//! production kernel registry.

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
        conv1d_k7_residue_d1_snake::{
            ResidueDilation, ResidueLaunchGeometry,
            conv1d_k7_residue_d1_snake_contract_is_compatible,
            try_conv1d_k7_same_residue_d1_snake_wgsl, try_pack_conv1d_k7_residue_input_wgsl,
        },
        conv1d_k7_t256_snake_epilogue::{Conv1dK7T256Tile, LaunchGeometry},
        conv1d_k7_t256_snake_vec4_store::{
            conv1d_k7_t256_snake_vec4_store_contract_is_compatible, production_tile_for_shape,
            try_conv1d_k7_same_t256_snake_vec4_store_wgsl,
        },
        conv1d_k7_tiled::Conv1dK7Dilation,
    },
};

type B = WgpuRaw;

const CHANNELS: usize = 192;
const LENGTH: usize = 48_000;
const KERNEL_SIZE: usize = 7;
const DEFAULT_WARMUP: usize = 10;
const DEFAULT_ITERATIONS: usize = 50;
const DEFAULT_TRIALS: usize = 5;
const VARIANT_COUNT: usize = 2;
const AUDITED_CURRENT_COMBINED_US: f64 = 11_705.138;
const REQUIRED_SAVING_US: f64 = 2_000.0;
const ADOPTION_GATE_US: f64 = 9_705.138;

#[derive(Clone, Copy, Debug)]
struct ConvCase {
    production_dilation: Conv1dK7Dilation,
    residue_dilation: ResidueDilation,
    production_tile: Conv1dK7T256Tile,
}

const CASES: [ConvCase; 2] = [
    ConvCase {
        production_dilation: Conv1dK7Dilation::Three,
        residue_dilation: ResidueDilation::Three,
        production_tile: Conv1dK7T256Tile::Cin16,
    },
    ConvCase {
        production_dilation: Conv1dK7Dilation::Nine,
        residue_dilation: ResidueDilation::Nine,
        production_tile: Conv1dK7T256Tile::Cin8,
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
    fn require_exact(self, case: ConvCase) -> Result<Self, Box<dyn Error>> {
        if self.mismatched_bits == 0 && self.max_abs == 0.0 {
            Ok(self)
        } else {
            Err(io::Error::other(format!(
                "{} production residue path is not bit-exact: bit_mismatch={} max_abs={:.9e}",
                case.residue_dilation.label(),
                self.mismatched_bits,
                self.max_abs,
            ))
            .into())
        }
    }
}

#[derive(Debug)]
struct CaseResult {
    case: ConvCase,
    timings: [Timing; VARIANT_COUNT],
    pack_timing: Timing,
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
    "usage: bench_conv1d_k7_residue_d1 <adapter-index> \
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

fn prior_forward(
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
        case.production_dilation,
        case.production_tile,
    )
    .expect("preflighted prior vec4 route must launch");
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn residue_forward(
    input: &Tensor<B, 3>,
    weight: &Tensor<B, 3>,
    bias: &Tensor<B, 1>,
    alpha: &Tensor<B, 3>,
    case: ConvCase,
) -> Tensor<B, 3> {
    let output = try_conv1d_k7_same_residue_d1_snake_wgsl(
        input.clone().into_primitive().tensor(),
        weight.clone().into_primitive().tensor(),
        bias.clone().into_primitive().tensor(),
        alpha.clone().into_primitive().tensor(),
        case.residue_dilation,
    )
    .expect("preflighted production residue-d1 path must launch");
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn pack_forward(input: &Tensor<B, 3>, case: ConvCase) -> Tensor<B, 1> {
    let packed = try_pack_conv1d_k7_residue_input_wgsl(
        input.clone().into_primitive().tensor(),
        case.residue_dilation,
    )
    .expect("preflighted compact residue pack must launch");
    Tensor::from_primitive(TensorPrimitive::Float(packed))
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
        0 => prior_forward(input, weight, bias, alpha, case),
        1 => residue_forward(input, weight, bias, alpha, case),
        _ => unreachable!("benchmark has exactly two full-path variants"),
    }
}

fn tensor_values(tensor: Tensor<B, 3>) -> Result<Vec<f32>, Box<dyn Error>> {
    let expected_shape = [1, CHANNELS, LENGTH];
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
            "output length mismatch: prior={} residue={}",
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
                "non-finite output pair: prior={expected:?} residue={actual:?}"
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

fn synchronize_output(output: Tensor<B, 3>) {
    let _ = output
        .slice([0..1, CHANNELS - 1..CHANNELS, LENGTH - 1..LENGTH])
        .into_data();
}

fn synchronize_pack(output: Tensor<B, 1>) {
    synchronize_output(output.reshape([1, CHANNELS, LENGTH]));
}

fn warm_up_full<F>(count: usize, operation: &mut F)
where
    F: FnMut() -> Tensor<B, 3>,
{
    let mut output = None;
    for _ in 0..count {
        output = Some(operation());
    }
    synchronize_output(output.expect("warmup count must be non-zero"));
}

fn warm_up_pack<F>(count: usize, operation: &mut F)
where
    F: FnMut() -> Tensor<B, 1>,
{
    let mut output = None;
    for _ in 0..count {
        output = Some(operation());
    }
    synchronize_pack(output.expect("warmup count must be non-zero"));
}

fn measure_full<F>(iterations: usize, operation: &mut F) -> f64
where
    F: FnMut() -> Tensor<B, 3>,
{
    let started = Instant::now();
    let mut output = None;
    for _ in 0..iterations {
        output = Some(operation());
    }
    synchronize_output(output.expect("iteration count must be non-zero"));
    started.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64
}

fn measure_pack<F>(iterations: usize, operation: &mut F) -> f64
where
    F: FnMut() -> Tensor<B, 1>,
{
    let started = Instant::now();
    let mut output = None;
    for _ in 0..iterations {
        output = Some(operation());
    }
    synchronize_pack(output.expect("iteration count must be non-zero"));
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

fn model_macs() -> u128 {
    (CHANNELS as u128).pow(2) * KERNEL_SIZE as u128 * LENGTH as u128
}

fn tmac_per_second(timing_us: f64) -> f64 {
    model_macs() as f64 / (timing_us * 1.0e6)
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
    let final_packed_index = case
        .residue_dilation
        .packed_index(CHANNELS - 1, LENGTH - 1)
        .ok_or_else(|| io::Error::other("final compact residue index is not representable"))?;
    if final_packed_index >= ResidueLaunchGeometry::new(case.residue_dilation).packed_elements {
        return Err(io::Error::other(format!(
            "final compact residue index {final_packed_index} exceeds the exact pack"
        ))
        .into());
    }
    if production_tile_for_shape(CHANNELS, LENGTH, case.production_dilation)
        != Some(case.production_tile)
    {
        return Err(io::Error::other("case is not an accepted prior vec4 route").into());
    }
    if !conv1d_k7_t256_snake_vec4_store_contract_is_compatible(
        &input,
        &weight,
        &bias,
        &alpha,
        case.production_dilation,
        case.production_tile,
    ) || !conv1d_k7_residue_d1_snake_contract_is_compatible(
        &input,
        &weight,
        &bias,
        &alpha,
        case.residue_dilation,
    ) {
        let properties = input.client.properties();
        return Err(io::Error::other(format!(
            "contract failed for C={CHANNELS} L={LENGTH} d={} residue={} allocator_alignment={} max_page={} max_bindings={} max_shared={}B max_units={}",
            case.production_dilation.value(),
            case.residue_dilation.label(),
            properties.memory.alignment,
            properties.memory.max_page_size,
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
        [1, CHANNELS, LENGTH],
        Distribution::Uniform(-1.0, 1.0),
        device,
    );
    let weight = Tensor::<B, 3>::random(
        [CHANNELS, CHANNELS, KERNEL_SIZE],
        Distribution::Uniform(-0.025, 0.025),
        device,
    );
    let bias = Tensor::<B, 1>::random([CHANNELS], Distribution::Uniform(-0.05, 0.05), device);
    let alpha = Tensor::<B, 3>::random([1, CHANNELS, 1], Distribution::Uniform(0.25, 2.0), device);
    check_contracts(&input, &weight, &bias, &alpha, case)?;

    let expected = tensor_values(prior_forward(&input, &weight, &bias, &alpha, case))?;
    let actual = tensor_values(residue_forward(&input, &weight, &bias, &alpha, case))?;
    let comparison = compare_outputs(&expected, &actual)?.require_exact(case)?;

    for variant in 0..VARIANT_COUNT {
        let mut operation = || variant_forward(variant, &input, &weight, &bias, &alpha, case);
        warm_up_full(args.warmup, &mut operation);
    }
    let mut pack_operation = || pack_forward(&input, case);
    warm_up_pack(args.warmup, &mut pack_operation);

    let mut samples: [Vec<f64>; VARIANT_COUNT] =
        std::array::from_fn(|_| Vec::with_capacity(args.trials));
    let mut pack_samples = Vec::with_capacity(args.trials);
    for trial in 0..args.trials {
        if trial.is_multiple_of(2) {
            let mut operation = || pack_forward(&input, case);
            pack_samples.push(measure_pack(args.iterations, &mut operation));
        }
        for offset in 0..VARIANT_COUNT {
            let variant = (trial + offset) % VARIANT_COUNT;
            let mut operation = || variant_forward(variant, &input, &weight, &bias, &alpha, case);
            samples[variant].push(measure_full(args.iterations, &mut operation));
        }
        if !trial.is_multiple_of(2) {
            let mut operation = || pack_forward(&input, case);
            pack_samples.push(measure_pack(args.iterations, &mut operation));
        }
    }
    let timings = std::array::from_fn(|index| summarize_samples(&samples[index]));
    let pack_timing = summarize_samples(&pack_samples);

    let production_geometry = LaunchGeometry::new(
        CHANNELS,
        LENGTH,
        case.production_dilation,
        case.production_tile,
    )
    .expect("current production geometry must remain valid");
    let production_workgroups = production_geometry
        .workgroups()
        .expect("current production workgroups must fit usize");
    let production_barriers =
        production_workgroups * 2 * (CHANNELS / case.production_tile.input_channel_tile());
    let residue_geometry = ResidueLaunchGeometry::new(case.residue_dilation);
    println!(
        "C={CHANNELS} L={LENGTH} d={} model_MAC={:.6}G",
        case.production_dilation.value(),
        model_macs() as f64 / 1.0e9,
    );
    println!(
        "  prior tile={} dispatch=1 workgroups={} barriers={} shared={}B",
        case.production_tile.label(),
        production_workgroups,
        production_barriers,
        production_geometry.shared_bytes,
    );
    println!(
        "  residue-production {} dispatch=2 pack_workgroups={} core_workgroups={} core_barriers={} core_shared={}B",
        case.residue_dilation.label(),
        residue_geometry.pack_workgroups,
        residue_geometry.core_workgroups,
        residue_geometry.core_barriers,
        residue_geometry.core_shared_bytes,
    );
    println!(
        "  pack elements={} temp={}B ({:.3}MiB) read_write={}B ({:.3}MiB) output_unpack=0",
        residue_geometry.packed_elements,
        residue_geometry.temporary_bytes,
        residue_geometry.temporary_bytes as f64 / 1_048_576.0,
        residue_geometry.pack_read_write_bytes,
        residue_geometry.pack_read_write_bytes as f64 / 1_048_576.0,
    );
    println!(
        "  prior current-fused      median={:10.3}us range=[{:10.3},{:10.3}] TMAC/s={:.3}",
        timings[0].median_us,
        timings[0].min_us,
        timings[0].max_us,
        tmac_per_second(timings[0].median_us),
    );
    println!(
        "  residue production        median={:10.3}us range=[{:10.3},{:10.3}] TMAC/s={:.3} speedup={:.3}x",
        timings[1].median_us,
        timings[1].min_us,
        timings[1].max_us,
        tmac_per_second(timings[1].median_us),
        timings[0].median_us / timings[1].median_us,
    );
    println!(
        "  diagnostic pack-only      median={:10.3}us range=[{:10.3},{:10.3}]",
        pack_timing.median_us, pack_timing.min_us, pack_timing.max_us,
    );
    println!(
        "  correctness shape=[1,{CHANNELS},{LENGTH}] elements={} finite=true bit_mismatch={} max_abs={:.9e} hard_gate=pass",
        comparison.elements, comparison.mismatched_bits, comparison.max_abs,
    );

    Ok(CaseResult {
        case,
        timings,
        pack_timing,
        comparison,
    })
}

fn print_static_accounting() {
    let production_workgroups = CASES
        .into_iter()
        .map(|case| {
            LaunchGeometry::new(
                CHANNELS,
                LENGTH,
                case.production_dilation,
                case.production_tile,
            )
            .and_then(LaunchGeometry::workgroups)
            .expect("production geometry")
        })
        .sum::<usize>();
    let production_barriers = CASES
        .into_iter()
        .map(|case| {
            let workgroups = LaunchGeometry::new(
                CHANNELS,
                LENGTH,
                case.production_dilation,
                case.production_tile,
            )
            .and_then(LaunchGeometry::workgroups)
            .expect("production geometry");
            workgroups * 2 * (CHANNELS / case.production_tile.input_channel_tile())
        })
        .sum::<usize>();
    let residue = CASES.map(|case| ResidueLaunchGeometry::new(case.residue_dilation));
    let residue_pack_workgroups = residue
        .iter()
        .map(|geometry| geometry.pack_workgroups as usize)
        .sum::<usize>();
    let residue_core_workgroups = residue
        .iter()
        .map(|geometry| geometry.core_workgroups)
        .sum::<usize>();
    let residue_core_barriers = residue
        .iter()
        .map(|geometry| geometry.core_barriers)
        .sum::<usize>();
    println!("static exact-two accounting:");
    println!(
        "  prior dispatches=2 workgroups={production_workgroups} barriers={production_barriers}"
    );
    println!(
        "  residue-production dispatches=4 pack_workgroups={residue_pack_workgroups} core_workgroups={residue_core_workgroups} core_barriers={residue_core_barriers}"
    );
    println!(
        "  per-call pack elements=9216000 temp=36864000B (35.156MiB) read=36864000B write=36864000B read_write=73728000B (70.312MiB)"
    );
    println!(
        "  sequential peak residue temporary=36864000B; exact-two pack traffic=147456000B (140.625MiB); persistent_bytes=0; output_unpack_dispatches=0"
    );
    println!(
        "  residue core mapping: compact [residue][channel][q], d1 pad=3, ci->tap0..6 FMA, direct NCL t=residue+q*d, scalar Snake/store"
    );
}

fn print_summary(results: &[CaseResult]) {
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
    let pack_sum = Timing {
        median_us: results
            .iter()
            .map(|result| result.pack_timing.median_us)
            .sum(),
        min_us: results.iter().map(|result| result.pack_timing.min_us).sum(),
        max_us: results.iter().map(|result| result.pack_timing.max_us).sum(),
    };
    let elements = results
        .iter()
        .map(|result| result.comparison.elements)
        .sum::<usize>();
    let mismatched_bits = results
        .iter()
        .map(|result| result.comparison.mismatched_bits)
        .sum::<usize>();
    let max_abs = results
        .iter()
        .map(|result| result.comparison.max_abs)
        .fold(0.0_f32, f32::max);
    let adoption_pass = sums[1].median_us <= ADOPTION_GATE_US;
    println!("exact-two sums of independently measured per-shape statistics:");
    println!(
        "  prior current-fused       median-sum={:.3}us range-sum=[{:.3},{:.3}]",
        sums[0].median_us, sums[0].min_us, sums[0].max_us,
    );
    println!(
        "  residue production        median-sum={:.3}us range-sum=[{:.3},{:.3}] measured_speedup={:.3}x measured_saving={:.3}us",
        sums[1].median_us,
        sums[1].min_us,
        sums[1].max_us,
        sums[0].median_us / sums[1].median_us,
        sums[0].median_us - sums[1].median_us,
    );
    println!(
        "  diagnostic pack-only      median-sum={:.3}us range-sum=[{:.3},{:.3}]",
        pack_sum.median_us, pack_sum.min_us, pack_sum.max_us,
    );
    println!(
        "  correctness shapes=2/2 elements={elements} finite=true bit_mismatch={mismatched_bits} max_abs={max_abs:.9e} hard_gate=pass"
    );
    println!(
        "  adoption authoritative_current={AUDITED_CURRENT_COMBINED_US:.3}us required_saving={REQUIRED_SAVING_US:.3}us residue_limit={ADOPTION_GATE_US:.3}us residue={:.3}us verdict={}",
        sums[1].median_us,
        if adoption_pass { "ACCEPT" } else { "REJECT" },
    );
    for result in results {
        println!(
            "  route d={} prior={:.3}us residue={:.3}us pack={:.3}us",
            result.case.production_dilation.value(),
            result.timings[0].median_us,
            result.timings[1].median_us,
            result.pack_timing.median_us,
        );
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let (device, monitor) = initialize_wgpu(args.adapter_index);
    B::seed(&device, 0);
    println!(
        "isolated residue-d1 k7 A/B: warmup={} iterations={} trials={} variants=2 cases=2 seed=0",
        args.warmup, args.iterations, args.trials,
    );
    println!(
        "fairness: direct prior fused T256+Snake vec4 baseline; identical input/weight/bias/alpha; rotating full-path order; residue production timing includes compact pack+core+direct scatter; full-output shape/finite/bit0/maxabs0 hard gate"
    );
    print_static_accounting();

    let mut results = Vec::with_capacity(CASES.len());
    for case in CASES {
        results.push(benchmark_case(&device, case, &args)?);
        synchronize_and_check_wgpu(
            &device,
            &monitor,
            &format!(
                "C={CHANNELS} L={LENGTH} d={} residue-d1 A/B",
                case.production_dilation.value()
            ),
        )?;
    }
    print_summary(&results);
    synchronize_and_check_wgpu(&device, &monitor, "residue-d1 A/B completion")?;
    println!("wgpu_uncaptured_errors=0");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_two_cases_match_prior_routes_and_residue_geometry() {
        assert_eq!(CASES.len(), 2);
        for case in CASES {
            assert_eq!(
                production_tile_for_shape(CHANNELS, LENGTH, case.production_dilation),
                Some(case.production_tile),
            );
            let geometry = ResidueLaunchGeometry::new(case.residue_dilation);
            assert_eq!(geometry.dispatches, 2);
            assert_eq!(geometry.pack_workgroups, 36_000);
            assert_eq!(geometry.core_workgroups, 1_134);
            assert_eq!(geometry.core_barriers, 27_216);
        }
    }

    #[test]
    fn defaults_rotation_and_pack_diagnostic_order_are_fixed() {
        assert_eq!(
            (DEFAULT_WARMUP, DEFAULT_ITERATIONS, DEFAULT_TRIALS),
            (10, 50, 5),
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
            vec![vec![0, 1], vec![1, 0], vec![0, 1], vec![1, 0], vec![0, 1]],
        );
    }

    #[test]
    fn adoption_gate_is_exactly_two_milliseconds_below_audited_current() {
        let derived = AUDITED_CURRENT_COMBINED_US - REQUIRED_SAVING_US;
        assert!((derived - ADOPTION_GATE_US).abs() < 1.0e-9);
        assert_eq!(ADOPTION_GATE_US, 9_705.138);
    }

    #[test]
    fn exact_two_static_accounting_is_fixed() {
        let residue = CASES.map(|case| ResidueLaunchGeometry::new(case.residue_dilation));
        assert_eq!(residue.iter().map(|item| item.dispatches).sum::<usize>(), 4);
        assert_eq!(
            residue
                .iter()
                .map(|item| item.pack_workgroups as usize)
                .sum::<usize>(),
            72_000,
        );
        assert_eq!(
            residue
                .iter()
                .map(|item| item.core_workgroups)
                .sum::<usize>(),
            2_268,
        );
        assert_eq!(
            residue.iter().map(|item| item.core_barriers).sum::<usize>(),
            54_432,
        );
        assert_eq!(
            residue
                .iter()
                .map(|item| item.pack_read_write_bytes)
                .sum::<usize>(),
            147_456_000,
        );
    }
}
