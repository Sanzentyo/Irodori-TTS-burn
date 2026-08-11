//! Isolated cross-layer batching experiment for all v4 LowRankAdaLN modules.
//!
//! The released v4 DiT has 12 blocks with attention and MLP AdaLN modules,
//! hence 24 modules. Production currently executes one branch-batched down
//! GEMM and one branch-batched up GEMM per module. This benchmark compares that
//! exact 24-module sequence with a replacement cache that batches both the
//! module and shift/scale/gate dimensions, reducing 48 GEMMs to two.
//!
//! The candidate deliberately includes the cost of materializing the shared
//! activated condition across all 72 module/branch slots. It preserves the
//! final f32 addition order `(up + bias) + raw`. This file is measurement-only:
//! it is not registered in Cargo and does not alter production routing.
//!
//! After explicit registration and GPU authorization, run:
//! `cargo run --release --bin bench_adaln_cross_layer -- 0`

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
    tensor::{DType, Distribution, Tensor, activation::silu, backend::Backend},
};
use cubecl::prelude::Runtime;
use irodori_tts_wgpu::WgpuRaw;

type B = WgpuRaw;

const MODEL_DIM: usize = 1_280;
const RANK: usize = 192;
const BRANCHES: usize = 3;
const DIT_LAYERS: usize = 12;
const MODULES_PER_LAYER: usize = 2;
const MODULES: usize = DIT_LAYERS * MODULES_PER_LAYER;
const PACKED_BATCHES: usize = MODULES * BRANCHES;
const DEFAULT_WARMUP: usize = 10;
const DEFAULT_ITERATIONS: usize = 100;
const DEFAULT_TRIALS: usize = 5;
const SEED: u64 = 0;
const F32_BYTES: usize = core::mem::size_of::<f32>();
const MAX_ABS: f32 = 1.0e-5;

// Accepted same-card isolated results for the current per-module rank-4 path
// and the per-module wide-column candidate. These make the cross-layer result
// immediately actionable without conflating measurements from separate runs.
const MEASURED_RANK4_B1_MODULE_US: f64 = 318.983;
const MEASURED_RANK4_B2_MODULE_US: f64 = 280.594;
const MEASURED_WIDE_COL_B1_MODULE_US: f64 = 99.727;
const MEASURED_WIDE_COL_B2_MODULE_US: f64 = 132.511;
const PYTORCH_STRICT_B1_MODULE_US: f64 = 37.052_159_309_387_21;
const PYTORCH_STRICT_B2_MODULE_US: f64 = 96.921_281_814_575_2;
const B1_EVALUATIONS_PER_SYNTHESIS: usize = 2;
const B2_EVALUATIONS_PER_SYNTHESIS: usize = 2;

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

#[derive(Clone)]
struct ModuleWeights {
    down: [Tensor<B, 2>; BRANCHES],
    up: [Tensor<B, 2>; BRANCHES],
    bias: [Tensor<B, 1>; BRANCHES],
}

#[derive(Clone)]
struct Rank4Weights {
    down: Tensor<B, 4>,
    up: Tensor<B, 4>,
    bias: Tensor<B, 4>,
}

#[derive(Clone)]
struct CrossLayerWeights {
    down: Tensor<B, 4>,
    up: Tensor<B, 4>,
    bias: Tensor<B, 4>,
}

enum PackedSet {
    Sequential(Vec<Rank4Weights>),
    CrossLayer(Box<CrossLayerWeights>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Path {
    SequentialRank4,
    CrossLayer,
}

impl Path {
    const fn label(self) -> &'static str {
        match self {
            Self::SequentialRank4 => "sequential-rank4",
            Self::CrossLayer => "cross-layer",
        }
    }
}

const PATHS: [Path; 2] = [Path::SequentialRank4, Path::CrossLayer];

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
    "usage: bench_adaln_cross_layer <adapter-index> [--warmup N] \
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

fn original_weights(device: &<B as Backend>::Device) -> Vec<ModuleWeights> {
    (0..MODULES)
        .map(|_| {
            let down = std::array::from_fn(|_| {
                Tensor::random(
                    [MODEL_DIM, RANK],
                    Distribution::Uniform(-0.05, 0.05),
                    device,
                )
            });
            let up = std::array::from_fn(|_| {
                Tensor::random(
                    [RANK, MODEL_DIM],
                    Distribution::Uniform(-0.05, 0.05),
                    device,
                )
            });
            let bias = std::array::from_fn(|_| {
                Tensor::random([MODEL_DIM], Distribution::Uniform(-0.05, 0.05), device)
            });
            ModuleWeights { down, up, bias }
        })
        .collect()
}

fn pack_module(weights: &ModuleWeights) -> Rank4Weights {
    let down = Tensor::<B, 2>::stack::<3>(weights.down.to_vec(), 0).unsqueeze_dim::<4>(0);
    let up = Tensor::<B, 2>::stack::<3>(weights.up.to_vec(), 0).unsqueeze_dim::<4>(0);
    let bias =
        Tensor::<B, 1>::stack::<2>(weights.bias.to_vec(), 0).reshape([1, BRANCHES, 1, MODEL_DIM]);
    Rank4Weights { down, up, bias }
}

fn pack_sequential(weights: &[ModuleWeights]) -> Vec<Rank4Weights> {
    weights.iter().map(pack_module).collect()
}

fn pack_cross_layer(weights: &[ModuleWeights]) -> CrossLayerWeights {
    let down = weights
        .iter()
        .flat_map(|module| module.down.iter().cloned())
        .collect::<Vec<_>>();
    let up = weights
        .iter()
        .flat_map(|module| module.up.iter().cloned())
        .collect::<Vec<_>>();
    let bias = weights
        .iter()
        .flat_map(|module| module.bias.iter().cloned())
        .collect::<Vec<_>>();
    CrossLayerWeights {
        down: Tensor::<B, 2>::stack::<3>(down, 0).unsqueeze_dim::<4>(0),
        up: Tensor::<B, 2>::stack::<3>(up, 0).unsqueeze_dim::<4>(0),
        bias: Tensor::<B, 1>::stack::<2>(bias, 0).reshape([1, PACKED_BATCHES, 1, MODEL_DIM]),
    }
}

fn pack_path(weights: &[ModuleWeights], path: Path) -> PackedSet {
    match path {
        Path::SequentialRank4 => PackedSet::Sequential(pack_sequential(weights)),
        Path::CrossLayer => PackedSet::CrossLayer(Box::new(pack_cross_layer(weights))),
    }
}

fn contiguous_strides<const D: usize>(shape: [usize; D]) -> Result<[usize; D], Box<dyn Error>> {
    let mut strides = [0; D];
    let mut stride = 1usize;
    for index in (0..D).rev() {
        strides[index] = stride;
        stride = stride.checked_mul(shape[index]).ok_or_else(|| {
            io::Error::other(format!("contiguous stride overflows for shape {shape:?}"))
        })?;
    }
    Ok(strides)
}

fn validate_tensor<const D: usize>(
    name: &str,
    tensor: &Tensor<B, D>,
    expected_shape: [usize; D],
    device: &WgpuDevice,
) -> Result<(), Box<dyn Error>> {
    let raw = tensor.clone().into_primitive().tensor();
    if raw.dtype != DType::F32 {
        return Err(io::Error::other(format!("{name} must be f32, got {:?}", raw.dtype)).into());
    }
    if &raw.device != device {
        return Err(io::Error::other(format!("{name} is on a different WGPU device")).into());
    }
    if raw.meta.num_dims() != D || raw.meta.shape().dims::<D>() != expected_shape {
        return Err(io::Error::other(format!(
            "{name} shape mismatch: expected {expected_shape:?}, got {:?}",
            raw.meta.shape()
        ))
        .into());
    }
    let expected_strides = contiguous_strides(expected_shape)?;
    if !raw.is_contiguous() || &raw.meta.strides()[..] != expected_strides.as_slice() {
        return Err(io::Error::other(format!(
            "{name} layout mismatch: expected contiguous strides {expected_strides:?}, got {:?}",
            raw.meta.strides()
        ))
        .into());
    }
    Ok(())
}

fn validate_original_weights(
    weights: &[ModuleWeights],
    device: &WgpuDevice,
) -> Result<(), Box<dyn Error>> {
    if weights.len() != MODULES {
        return Err(io::Error::other(format!(
            "expected {MODULES} original modules, got {}",
            weights.len()
        ))
        .into());
    }
    for (module_index, module) in weights.iter().enumerate() {
        for branch in 0..BRANCHES {
            validate_tensor(
                &format!("module[{module_index}].down[{branch}]"),
                &module.down[branch],
                [MODEL_DIM, RANK],
                device,
            )?;
            validate_tensor(
                &format!("module[{module_index}].up[{branch}]"),
                &module.up[branch],
                [RANK, MODEL_DIM],
                device,
            )?;
            validate_tensor(
                &format!("module[{module_index}].bias[{branch}]"),
                &module.bias[branch],
                [MODEL_DIM],
                device,
            )?;
        }
    }
    Ok(())
}

fn validate_sequential_weights(
    weights: &[Rank4Weights],
    device: &WgpuDevice,
) -> Result<(), Box<dyn Error>> {
    if weights.len() != MODULES {
        return Err(io::Error::other(format!(
            "expected {MODULES} rank-4 modules, got {}",
            weights.len()
        ))
        .into());
    }
    for (module_index, module) in weights.iter().enumerate() {
        validate_tensor(
            &format!("rank4[{module_index}].down"),
            &module.down,
            [1, BRANCHES, MODEL_DIM, RANK],
            device,
        )?;
        validate_tensor(
            &format!("rank4[{module_index}].up"),
            &module.up,
            [1, BRANCHES, RANK, MODEL_DIM],
            device,
        )?;
        validate_tensor(
            &format!("rank4[{module_index}].bias"),
            &module.bias,
            [1, BRANCHES, 1, MODEL_DIM],
            device,
        )?;
    }
    Ok(())
}

fn validate_cross_layer_weights(
    weights: &CrossLayerWeights,
    device: &WgpuDevice,
) -> Result<(), Box<dyn Error>> {
    validate_tensor(
        "cross_layer.down",
        &weights.down,
        [1, PACKED_BATCHES, MODEL_DIM, RANK],
        device,
    )?;
    validate_tensor(
        "cross_layer.up",
        &weights.up,
        [1, PACKED_BATCHES, RANK, MODEL_DIM],
        device,
    )?;
    validate_tensor(
        "cross_layer.bias",
        &weights.bias,
        [1, PACKED_BATCHES, 1, MODEL_DIM],
        device,
    )
}

fn rank4_projection(raw: Tensor<B, 4>, weights: &Rank4Weights) -> Tensor<B, 4> {
    let activated = silu(raw.clone());
    let down = activated.matmul(weights.down.clone());
    let up = down.matmul(weights.up.clone());
    let biased = up + weights.bias.clone();
    biased + raw
}

fn sequential_rank4_last(
    raw: &Tensor<B, 4>,
    weights: &[Rank4Weights],
) -> Result<Tensor<B, 4>, Box<dyn Error>> {
    if weights.len() != MODULES {
        return Err(io::Error::other(format!(
            "sequential projection requires {MODULES} modules, got {}",
            weights.len()
        ))
        .into());
    }
    let mut outputs = weights
        .iter()
        .map(|module| rank4_projection(raw.clone(), module));
    let first = outputs
        .next()
        .ok_or_else(|| io::Error::other("sequential projection has no modules"))?;
    Ok(outputs.fold(first, |_, output| output))
}

fn sequential_rank4_all(
    raw: &Tensor<B, 4>,
    weights: &[Rank4Weights],
) -> Result<Tensor<B, 4>, Box<dyn Error>> {
    if weights.len() != MODULES {
        return Err(io::Error::other(format!(
            "correctness projection requires {MODULES} modules, got {}",
            weights.len()
        ))
        .into());
    }
    let outputs = weights
        .iter()
        .map(|module| rank4_projection(raw.clone(), module))
        .collect::<Vec<_>>();
    Ok(Tensor::<B, 4>::cat(outputs, 1))
}

/// Materialize the shared activated condition in module-major branch order.
///
/// `repeat_dim` receives a non-singleton branch dimension (`3`), so WGPU emits
/// an explicit copy rather than a zero-stride view. This cost remains inside
/// every timed candidate call.
fn materialize_cross_layer_input(raw: Tensor<B, 4>) -> Tensor<B, 4> {
    silu(raw).repeat_dim(1, MODULES)
}

fn cross_layer_projection(raw: Tensor<B, 4>, weights: &CrossLayerWeights) -> Tensor<B, 4> {
    let batch = raw.dims()[0];
    let activated = materialize_cross_layer_input(raw.clone());
    let down = activated.matmul(weights.down.clone());
    let up = down.matmul(weights.up.clone());
    let biased = up + weights.bias.clone();

    // Broadcast raw across the module axis without introducing an extra raw
    // materialization. Keeping these as separate statements fixes the f32
    // operation order to `(up + bias) + raw`.
    let biased: Tensor<B, 5> = biased.reshape([batch, MODULES, BRANCHES, 1, MODEL_DIM]);
    let raw: Tensor<B, 5> = raw.reshape([batch, 1, BRANCHES, 1, MODEL_DIM]);
    (biased + raw).reshape([batch, PACKED_BATCHES, 1, MODEL_DIM])
}

fn project_path(
    raw: &Tensor<B, 4>,
    path: Path,
    sequential: &[Rank4Weights],
    cross_layer: &CrossLayerWeights,
) -> Result<Tensor<B, 4>, Box<dyn Error>> {
    match path {
        Path::SequentialRank4 => sequential_rank4_last(raw, sequential),
        Path::CrossLayer => Ok(cross_layer_projection(raw.clone(), cross_layer)),
    }
}

fn compare_tensors(
    expected: Tensor<B, 4>,
    actual: Tensor<B, 4>,
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
                "non-finite output at {index}: expected={expected:?}, actual={actual:?}"
            ))
            .into());
        }
        comparison.bit_mismatches += usize::from(expected.to_bits() != actual.to_bits());
        comparison.max_abs = comparison.max_abs.max((expected - actual).abs());
    }
    if !comparison.max_abs.is_finite() || comparison.max_abs > MAX_ABS {
        return Err(io::Error::other(format!(
            "cross-layer max_abs={:.9e} exceeds fail-closed threshold {:.9e}",
            comparison.max_abs, MAX_ABS
        ))
        .into());
    }
    Ok(comparison)
}

fn sync_output(output: Tensor<B, 4>) -> Result<(), Box<dyn Error>> {
    let [batch, slots, _, width] = output.dims();
    if ![1, 2].contains(&batch)
        || ![BRANCHES, PACKED_BATCHES].contains(&slots)
        || width != MODEL_DIM
    {
        return Err(io::Error::other(format!(
            "synchronization output violates B/slot/D contract: {:?}",
            output.dims()
        ))
        .into());
    }
    let values = output
        .slice([batch - 1..batch, slots - 1..slots, 0..1, width - 1..width])
        .into_data()
        .to_vec::<f32>()?;
    let value = values
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("synchronization readback returned no value"))?;
    if !value.is_finite() {
        return Err(
            io::Error::other(format!("synchronization readback is non-finite: {value:?}")).into(),
        );
    }
    Ok(())
}

fn warm_up_path(
    raw: &Tensor<B, 4>,
    path: Path,
    sequential: &[Rank4Weights],
    cross_layer: &CrossLayerWeights,
    warmup: usize,
) -> Result<(), Box<dyn Error>> {
    let mut output = None;
    for _ in 0..warmup {
        output = Some(project_path(raw, path, sequential, cross_layer)?);
    }
    let output = output.ok_or_else(|| io::Error::other("warmup count must be non-zero"))?;
    sync_output(output)
}

fn measure_projection_once(
    raw: &Tensor<B, 4>,
    path: Path,
    sequential: &[Rank4Weights],
    cross_layer: &CrossLayerWeights,
    iterations: usize,
) -> Result<f64, Box<dyn Error>> {
    let started = Instant::now();
    let mut output = None;
    for _ in 0..iterations {
        output = Some(project_path(raw, path, sequential, cross_layer)?);
    }
    let output = output.ok_or_else(|| io::Error::other("iteration count must be non-zero"))?;
    sync_output(output)?;
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

fn benchmark_projection(
    raw: &Tensor<B, 4>,
    sequential: &[Rank4Weights],
    cross_layer: &CrossLayerWeights,
    args: &Args,
) -> Result<[Timing; PATHS.len()], Box<dyn Error>> {
    for path in PATHS {
        warm_up_path(raw, path, sequential, cross_layer, args.warmup)?;
    }
    let mut samples: [Vec<f64>; PATHS.len()] =
        std::array::from_fn(|_| Vec::with_capacity(args.trials));
    for trial in 0..args.trials {
        for offset in 0..PATHS.len() {
            let index = (trial + offset) % PATHS.len();
            samples[index].push(measure_projection_once(
                raw,
                PATHS[index],
                sequential,
                cross_layer,
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
        .map_err(|_| io::Error::other("both projection timing sets are required").into())
}

fn sync_packed(packed: &PackedSet) -> Result<(), Box<dyn Error>> {
    let bias = match packed {
        PackedSet::Sequential(weights) => weights
            .last()
            .ok_or_else(|| io::Error::other("sequential pack returned no modules"))?
            .bias
            .clone()
            .slice([0..1, BRANCHES - 1..BRANCHES, 0..1, MODEL_DIM - 1..MODEL_DIM]),
        PackedSet::CrossLayer(weights) => weights.bias.clone().slice([
            0..1,
            PACKED_BATCHES - 1..PACKED_BATCHES,
            0..1,
            MODEL_DIM - 1..MODEL_DIM,
        ]),
    };
    let values = bias.into_data().to_vec::<f32>()?;
    let value = values
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("pack synchronization returned no value"))?;
    if !value.is_finite() {
        return Err(io::Error::other(format!(
            "pack synchronization readback is non-finite: {value:?}"
        ))
        .into());
    }
    Ok(())
}

fn warm_up_pack(
    original: &[ModuleWeights],
    path: Path,
    count: usize,
) -> Result<(), Box<dyn Error>> {
    if count == 0 {
        return Err(io::Error::other("pack warmup count must be non-zero").into());
    }
    for _ in 0..count {
        let packed = pack_path(original, path);
        // One cross-layer pack is about 135 MiB. Synchronize each repetition
        // so 100 queued temporary packs cannot exhaust an 8 GiB adapter.
        sync_packed(&packed)?;
    }
    Ok(())
}

fn measure_pack_once(
    original: &[ModuleWeights],
    path: Path,
    iterations: usize,
) -> Result<f64, Box<dyn Error>> {
    if iterations == 0 {
        return Err(io::Error::other("pack iteration count must be non-zero").into());
    }
    let started = Instant::now();
    for _ in 0..iterations {
        let packed = pack_path(original, path);
        sync_packed(&packed)?;
    }
    Ok(started.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64)
}

fn benchmark_pack(
    original: &[ModuleWeights],
    args: &Args,
) -> Result<[Timing; PATHS.len()], Box<dyn Error>> {
    for path in PATHS {
        warm_up_pack(original, path, args.warmup)?;
    }
    let mut samples: [Vec<f64>; PATHS.len()] =
        std::array::from_fn(|_| Vec::with_capacity(args.trials));
    for trial in 0..args.trials {
        for offset in 0..PATHS.len() {
            let index = (trial + offset) % PATHS.len();
            samples[index].push(measure_pack_once(original, PATHS[index], args.iterations)?);
        }
    }
    let timings = samples
        .iter()
        .map(|samples| summarize_samples(samples))
        .collect::<Result<Vec<_>, _>>()?;
    timings
        .try_into()
        .map_err(|_| io::Error::other("both pack timing sets are required").into())
}

const fn packed_weight_bytes() -> usize {
    MODULES * BRANCHES * (MODEL_DIM * RANK + RANK * MODEL_DIM + MODEL_DIM) * F32_BYTES
}

const fn projection_macs(batch: usize) -> usize {
    batch * MODULES * BRANCHES * 2 * MODEL_DIM * RANK
}

const fn baseline_intermediate_writes(batch: usize) -> usize {
    // Per module: SiLU, up, bias-add, raw-add each write 3*B*D; down writes 3*B*R.
    batch * MODULES * (4 * BRANCHES * MODEL_DIM + BRANCHES * RANK) * F32_BYTES
}

const fn cross_layer_intermediate_writes(batch: usize) -> usize {
    // SiLU writes 3BD; materialization and up/bias/raw each write 72BD; down writes 72BR.
    batch
        * (BRANCHES * MODEL_DIM + 4 * PACKED_BATCHES * MODEL_DIM + PACKED_BATCHES * RANK)
        * F32_BYTES
}

const fn baseline_live_intermediate_bytes(batch: usize) -> usize {
    // Raw + activated/up + down for one module; later modules can reuse released buffers.
    batch * BRANCHES * (2 * MODEL_DIM + RANK) * F32_BYTES
}

const fn cross_layer_live_intermediate_bytes(batch: usize) -> usize {
    // Base raw + materialized activated/up + down. The final cache reuses an up-sized buffer.
    batch * (BRANCHES * MODEL_DIM + PACKED_BATCHES * (MODEL_DIM + RANK)) * F32_BYTES
}

const fn cross_layer_output_cache_bytes(batch: usize) -> usize {
    batch * PACKED_BATCHES * MODEL_DIM * F32_BYTES
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn measured_group_us(module_us: f64) -> f64 {
    module_us * MODULES as f64
}

fn measured_four_evaluation_us(b1_module_us: f64, b2_module_us: f64) -> f64 {
    B1_EVALUATIONS_PER_SYNTHESIS as f64 * measured_group_us(b1_module_us)
        + B2_EVALUATIONS_PER_SYNTHESIS as f64 * measured_group_us(b2_module_us)
}

fn report_static_accounting() {
    let benchmark_resident_bytes = 3 * packed_weight_bytes();
    println!(
        "persistent_pack: modules={MODULES} branches/module={BRANCHES} bytes={} ({:.3} MiB); \
         replacement capacity is identical, co-retaining both caches adds the same amount",
        packed_weight_bytes(),
        mib(packed_weight_bytes())
    );
    println!(
        "benchmark_weight_residency: originals + sequential pack + cross-layer pack = {} \
         bytes ({:.3} MiB); timed repacking temporarily adds one further pack",
        benchmark_resident_bytes,
        mib(benchmark_resident_bytes),
    );
    for batch in [1, 2] {
        println!(
            "B={batch} static: MAC={} (same both paths), logical_dispatches=120->6, \
             intermediate_writes={} -> {} bytes, live_intermediate={} -> {} bytes, \
             retained_cross_layer_output={} bytes ({:.3} MiB)",
            projection_macs(batch),
            baseline_intermediate_writes(batch),
            cross_layer_intermediate_writes(batch),
            baseline_live_intermediate_bytes(batch),
            cross_layer_live_intermediate_bytes(batch),
            cross_layer_output_cache_bytes(batch),
            mib(cross_layer_output_cache_bytes(batch)),
        );
    }
}

fn report_measured_reference() {
    let rank4_b1 = measured_group_us(MEASURED_RANK4_B1_MODULE_US);
    let rank4_b2 = measured_group_us(MEASURED_RANK4_B2_MODULE_US);
    let wide_b1 = measured_group_us(MEASURED_WIDE_COL_B1_MODULE_US);
    let wide_b2 = measured_group_us(MEASURED_WIDE_COL_B2_MODULE_US);
    let rank4_four =
        measured_four_evaluation_us(MEASURED_RANK4_B1_MODULE_US, MEASURED_RANK4_B2_MODULE_US);
    let wide_four = measured_four_evaluation_us(
        MEASURED_WIDE_COL_B1_MODULE_US,
        MEASURED_WIDE_COL_B2_MODULE_US,
    );
    let pytorch_b1 = measured_group_us(PYTORCH_STRICT_B1_MODULE_US);
    let pytorch_b2 = measured_group_us(PYTORCH_STRICT_B2_MODULE_US);
    let pytorch_four =
        measured_four_evaluation_us(PYTORCH_STRICT_B1_MODULE_US, PYTORCH_STRICT_B2_MODULE_US);
    println!(
        "accepted_reference: current-rank4 group B1={rank4_b1:.3} us B2={rank4_b2:.3} us; \
         per-module-wide-col group B1={wide_b1:.3} us B2={wide_b2:.3} us; \
         strict-PyTorch rank4 group B1={pytorch_b1:.3} us B2={pytorch_b2:.3} us"
    );
    println!(
        "accepted_four_evaluation_workload: current-rank4={:.3} ms wide-col={:.3} ms \
         strict-PyTorch={:.3} ms; wide-col saving={:.3} ms and leaves {:.3} ms of the \
         AdaLN-only Rust/PyTorch gap; cross-layer's absolute additional ceiling over wide-col \
         is {:.3} ms",
        rank4_four / 1_000.0,
        wide_four / 1_000.0,
        pytorch_four / 1_000.0,
        (rank4_four - wide_four) / 1_000.0,
        (wide_four - pytorch_four) / 1_000.0,
        wide_four / 1_000.0,
    );
}

fn run() -> Result<(), Box<dyn Error>> {
    let ParseOutcome::Run(args) = parse_args()? else {
        println!("{}", usage());
        return Ok(());
    };
    let (device, error_monitor) = initialize_wgpu(args.adapter_index);
    B::seed(&device, SEED);

    println!(
        "LowRankAdaLN cross-layer benchmark: device={device:?}, D={MODEL_DIM}, R={RANK}, \
         layers={DIT_LAYERS}, modules={MODULES}, packed_batches={PACKED_BATCHES}, \
         warmup={}, iterations={} x {} trials, seed={SEED}",
        args.warmup, args.iterations, args.trials
    );
    report_static_accounting();
    report_measured_reference();

    let original = original_weights(&device);
    validate_original_weights(&original, &device)?;
    let sequential = pack_sequential(&original);
    let cross_layer = pack_cross_layer(&original);
    validate_sequential_weights(&sequential, &device)?;
    validate_cross_layer_weights(&cross_layer, &device)?;

    let pack_timings = benchmark_pack(&original, &args)?;
    println!("pack timing for all {MODULES} modules (synchronized wall time per pack):");
    for (index, path) in PATHS.into_iter().enumerate() {
        let timing = pack_timings[index];
        println!(
            "  {:<16} median={:>9.3} us range=[{:>9.3},{:>9.3}] cache_bytes={}",
            path.label(),
            timing.median_us,
            timing.min_us,
            timing.max_us,
            packed_weight_bytes(),
        );
    }
    synchronize_and_check_wgpu(&device, &error_monitor, "pack benchmark")?;

    let mut batch_timings = Vec::with_capacity(2);
    for batch in [1, 2] {
        let raw = Tensor::<B, 4>::random(
            [batch, BRANCHES, 1, MODEL_DIM],
            Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        validate_tensor(
            &format!("B={batch} raw"),
            &raw,
            [batch, BRANCHES, 1, MODEL_DIM],
            &device,
        )?;
        let materialized = materialize_cross_layer_input(raw.clone());
        validate_tensor(
            &format!("B={batch} materialized input"),
            &materialized,
            [batch, PACKED_BATCHES, 1, MODEL_DIM],
            &device,
        )?;
        drop(materialized);

        let expected = sequential_rank4_all(&raw, &sequential)?;
        let actual = cross_layer_projection(raw.clone(), &cross_layer);
        let comparison = compare_tensors(expected, actual)?;
        println!(
            "B={batch} correctness: elements={} bit_mismatch={} max_abs={:.9e} finite=true",
            comparison.elements, comparison.bit_mismatches, comparison.max_abs
        );

        let timings = benchmark_projection(&raw, &sequential, &cross_layer, &args)?;
        let baseline = timings[0].median_us;
        println!("B={batch} projection timing for all {MODULES} modules:");
        for (index, path) in PATHS.into_iter().enumerate() {
            let timing = timings[index];
            println!(
                "  {:<16} median={:>9.3} us range=[{:>9.3},{:>9.3}] speedup={:>7.3}x",
                path.label(),
                timing.median_us,
                timing.min_us,
                timing.max_us,
                baseline / timing.median_us,
            );
        }
        let cross = timings[1].median_us;
        let measured_current = measured_group_us(if batch == 1 {
            MEASURED_RANK4_B1_MODULE_US
        } else {
            MEASURED_RANK4_B2_MODULE_US
        });
        let measured_wide = measured_group_us(if batch == 1 {
            MEASURED_WIDE_COL_B1_MODULE_US
        } else {
            MEASURED_WIDE_COL_B2_MODULE_US
        });
        println!(
            "  cross-layer vs accepted totals: current-rank4={:.3}x ({:+.3} us), \
             wide-col={:.3}x ({:+.3} us; positive means cross-layer saves time)",
            measured_current / cross,
            measured_current - cross,
            measured_wide / cross,
            measured_wide - cross,
        );
        batch_timings.push(timings);
        synchronize_and_check_wgpu(&device, &error_monitor, &format!("B={batch}"))?;
    }

    let [b1, b2]: [[Timing; PATHS.len()]; 2] = batch_timings
        .try_into()
        .map_err(|_| io::Error::other("both B=1 and B=2 timing sets are required"))?;
    for (index, path) in PATHS.into_iter().enumerate() {
        let workload_us = B1_EVALUATIONS_PER_SYNTHESIS as f64 * b1[index].median_us
            + B2_EVALUATIONS_PER_SYNTHESIS as f64 * b2[index].median_us;
        println!(
            "four-evaluation measured {:<16} workload={:.3} ms",
            path.label(),
            workload_us / 1_000.0
        );
    }
    let cross_workload_us = B1_EVALUATIONS_PER_SYNTHESIS as f64 * b1[1].median_us
        + B2_EVALUATIONS_PER_SYNTHESIS as f64 * b2[1].median_us;
    let accepted_rank4_us =
        measured_four_evaluation_us(MEASURED_RANK4_B1_MODULE_US, MEASURED_RANK4_B2_MODULE_US);
    let accepted_wide_us = measured_four_evaluation_us(
        MEASURED_WIDE_COL_B1_MODULE_US,
        MEASURED_WIDE_COL_B2_MODULE_US,
    );
    let pytorch_strict_us =
        measured_four_evaluation_us(PYTORCH_STRICT_B1_MODULE_US, PYTORCH_STRICT_B2_MODULE_US);
    let current_python_gap_us = accepted_rank4_us - pytorch_strict_us;
    let gap_closed = (accepted_rank4_us - cross_workload_us) / current_python_gap_us;
    println!(
        "four-evaluation cross-layer comparison: vs accepted-rank4 speedup={:.3}x \
         saving={:.3} ms; vs accepted-wide-col speedup={:.3}x saving={:.3} ms; \
         strict-PyTorch={:.3} ms cross/PyTorch={:.3}x signed_gap={:+.3} ms \
         current_component_gap_closed={:.1}%",
        accepted_rank4_us / cross_workload_us,
        (accepted_rank4_us - cross_workload_us) / 1_000.0,
        accepted_wide_us / cross_workload_us,
        (accepted_wide_us - cross_workload_us) / 1_000.0,
        pytorch_strict_us / 1_000.0,
        cross_workload_us / pytorch_strict_us,
        (cross_workload_us - pytorch_strict_us) / 1_000.0,
        gap_closed * 100.0,
    );

    synchronize_and_check_wgpu(&device, &error_monitor, "benchmark completion")?;
    println!("wgpu_uncaptured_errors=0");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_shape_and_capacity_accounting_is_exact() {
        assert_eq!(MODULES, 24);
        assert_eq!(PACKED_BATCHES, 72);
        assert_eq!(packed_weight_bytes(), 141_926_400);
        assert_eq!(projection_macs(1), 35_389_440);
        assert_eq!(projection_macs(2), 70_778_880);
        assert_eq!(cross_layer_output_cache_bytes(1), 368_640);
        assert_eq!(cross_layer_output_cache_bytes(2), 737_280);
        assert_eq!(baseline_live_intermediate_bytes(1), 33_024);
        assert_eq!(cross_layer_live_intermediate_bytes(1), 439_296);
    }

    #[test]
    fn measured_reference_reconstructs_accepted_wide_saving() {
        let current =
            measured_four_evaluation_us(MEASURED_RANK4_B1_MODULE_US, MEASURED_RANK4_B2_MODULE_US);
        let wide = measured_four_evaluation_us(
            MEASURED_WIDE_COL_B1_MODULE_US,
            MEASURED_WIDE_COL_B2_MODULE_US,
        );
        assert!(((current - wide) / 1_000.0 - 17.632_272).abs() < 1.0e-9);
        let pytorch =
            measured_four_evaluation_us(PYTORCH_STRICT_B1_MODULE_US, PYTORCH_STRICT_B2_MODULE_US);
        assert!((pytorch / 1_000.0 - 6.430_725_173_950_196).abs() < 1.0e-9);
    }

    #[test]
    fn contiguous_stride_accounting_handles_singleton_dimensions() {
        let strides = contiguous_strides([1, PACKED_BATCHES, 1, MODEL_DIM]);
        assert_eq!(
            strides.ok(),
            Some([PACKED_BATCHES * MODEL_DIM, MODEL_DIM, MODEL_DIM, 1,])
        );
    }
}
