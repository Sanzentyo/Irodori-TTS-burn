//! Exact-shape LowRankAdaLN wide-projection experiment.
//!
//! The production baseline keeps three branches in rank-4 tensors and lets
//! Burn dispatch two branch-batched matmuls. The candidate flattens
//! `[B,3,1,D]` to `[3B,D]`, uses same-capacity wide weights `[D,3R]` and
//! `[R,3D]`, then selects the diagonal branch after each matmul. It performs
//! three times as many MACs in exchange for wider GEMM `N` dimensions.
//!
//! This binary and its selector module are isolated measurement scaffolding;
//! neither candidate is wired into production. Once explicitly registered,
//! run exactly one authorized adapter with:
//! `cargo run --release --bin bench_adaln_wide_projection -- 0`

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
    tensor::{Distribution, Tensor, TensorPrimitive, activation::silu, backend::Backend},
};
use cubecl::prelude::Runtime;
use irodori_tts_wgpu::WgpuRaw;

#[path = "../kernels/adaln_wide_projection_candidate.rs"]
mod adaln_wide_projection_candidate;

use adaln_wide_projection_candidate::{
    adaln_wide_diagonal_finalize_wgsl, adaln_wide_diagonal_select_wgsl,
};

type B = WgpuRaw;

const MODEL_DIM: usize = 1_280;
const RANK: usize = 192;
const BRANCHES: usize = 3;
const DIT_LAYERS: usize = 12;
const ADALN_MODULES_PER_LAYER: usize = 2;
const PINNED_ADALN_MODULES: usize = DIT_LAYERS * ADALN_MODULES_PER_LAYER;
const B1_CALLS_PER_SYNTHESIS: usize = 2 * PINNED_ADALN_MODULES;
const B2_CALLS_PER_SYNTHESIS: usize = 2 * PINNED_ADALN_MODULES;
const DEFAULT_WARMUP: usize = 10;
const DEFAULT_ITERATIONS: usize = 100;
const DEFAULT_TRIALS: usize = 5;
const SEED: u64 = 0;
const F32_BYTES: usize = size_of::<f32>();
const MAX_STAGE_ABS: f32 = 1.0e-3;

// Same-card strict FP32 reference captured on 2026-08-10 in
// /tmp/irodori-python-fp32.json: PyTorch 2.10.0+cu128, RTX 3060 Ti PCI 07,
// 10 warmup, 100 iterations x 5 trials. Upstream uses eager or
// torch.compile(fullgraph), not torch.jit/TorchScript.
const PYTORCH_B1_EAGER_GPU_US: f64 = 37.052_159_309_387_21;
const PYTORCH_B1_EAGER_WALL_US: f64 = 37.231_46;
const PYTORCH_B1_COMPILED_GPU_US: f64 = 101.829_442_977_905_27;
const PYTORCH_B1_COMPILE_FIRST_MS: f64 = 211.345_139;
const PYTORCH_B1_COMPILED_MAX_ABS: f32 = 1.639_127_7e-7;
const PYTORCH_B2_EAGER_GPU_US: f64 = 96.921_281_814_575_2;
const PYTORCH_B2_EAGER_WALL_US: f64 = 97.068_91;
const PYTORCH_B2_COMPILED_GPU_US: f64 = 132.659_196_853_637_7;
const PYTORCH_B2_COMPILE_FIRST_MS: f64 = 409.543_273;
const PYTORCH_B2_COMPILED_MAX_ABS: f32 = 1.490_116_1e-7;

#[derive(Debug)]
struct Args {
    adapter_index: usize,
    warmup: usize,
    iterations: usize,
    trials: usize,
}

#[derive(Clone)]
struct OriginalWeights {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WideLayout {
    RowMajor,
    ColumnMajor,
}

impl WideLayout {
    const fn label(self) -> &'static str {
        match self {
            Self::RowMajor => "wide-row",
            Self::ColumnMajor => "wide-col",
        }
    }
}

#[derive(Clone)]
struct WideWeights {
    down: Tensor<B, 2>,
    up: Tensor<B, 2>,
    bias: Tensor<B, 2>,
    layout: WideLayout,
}

#[derive(Clone)]
enum PackedWeights {
    Rank4(Rank4Weights),
    Wide(WideWeights),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandidatePath {
    Rank4,
    WideRow,
    WideColumn,
}

impl CandidatePath {
    const fn label(self) -> &'static str {
        match self {
            Self::Rank4 => "rank4",
            Self::WideRow => "wide-row",
            Self::WideColumn => "wide-col",
        }
    }
}

const PATHS: [CandidatePath; 3] = [
    CandidatePath::Rank4,
    CandidatePath::WideRow,
    CandidatePath::WideColumn,
];

#[derive(Clone, Copy, Debug)]
struct Timing {
    median_us: f64,
    min_us: f64,
    max_us: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Comparison {
    elements: usize,
    mismatched_bits: usize,
    max_abs: f32,
}

impl Comparison {
    fn merge(&mut self, other: Self) {
        self.elements += other.elements;
        self.mismatched_bits += other.mismatched_bits;
        self.max_abs = self.max_abs.max(other.max_abs);
    }
}

#[derive(Clone)]
struct Rank4Stages {
    activated: Tensor<B, 2>,
    down: Tensor<B, 2>,
    up: Tensor<B, 2>,
    biased: Tensor<B, 2>,
    final_output: Tensor<B, 2>,
}

#[derive(Clone)]
struct WideStages {
    activated: Tensor<B, 2>,
    down: Tensor<B, 2>,
    up: Tensor<B, 2>,
    biased: Tensor<B, 2>,
    unfused_final: Tensor<B, 2>,
    fused_final: Tensor<B, 2>,
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
    "usage: bench_adaln_wide_projection <adapter-index> [--warmup N] \
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

fn original_weights(device: &<B as Backend>::Device) -> OriginalWeights {
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
    OriginalWeights { down, up, bias }
}

fn pack_rank4(weights: &OriginalWeights) -> Rank4Weights {
    let down = Tensor::<B, 2>::stack::<3>(weights.down.to_vec(), 0).unsqueeze_dim::<4>(0);
    let up = Tensor::<B, 2>::stack::<3>(weights.up.to_vec(), 0).unsqueeze_dim::<4>(0);
    let bias =
        Tensor::<B, 1>::stack::<2>(weights.bias.to_vec(), 0).reshape([1, BRANCHES, 1, MODEL_DIM]);
    Rank4Weights { down, up, bias }
}

fn pack_wide_row(weights: &OriginalWeights) -> WideWeights {
    let down = Tensor::<B, 2>::cat(weights.down.to_vec(), 1);
    let up = Tensor::<B, 2>::cat(weights.up.to_vec(), 1);
    let bias = Tensor::<B, 1>::stack::<2>(weights.bias.to_vec(), 0);
    WideWeights {
        down,
        up,
        bias,
        layout: WideLayout::RowMajor,
    }
}

fn pack_wide_column(weights: &OriginalWeights) -> WideWeights {
    let down_physical = Tensor::<B, 2>::cat(
        weights
            .down
            .iter()
            .cloned()
            .map(Tensor::transpose)
            .collect(),
        0,
    );
    let up_physical = Tensor::<B, 2>::cat(
        weights.up.iter().cloned().map(Tensor::transpose).collect(),
        0,
    );
    let bias = Tensor::<B, 1>::stack::<2>(weights.bias.to_vec(), 0);
    WideWeights {
        down: down_physical.transpose(),
        up: up_physical.transpose(),
        bias,
        layout: WideLayout::ColumnMajor,
    }
}

fn pack_path(weights: &OriginalWeights, path: CandidatePath) -> PackedWeights {
    match path {
        CandidatePath::Rank4 => PackedWeights::Rank4(pack_rank4(weights)),
        CandidatePath::WideRow => PackedWeights::Wide(pack_wide_row(weights)),
        CandidatePath::WideColumn => PackedWeights::Wide(pack_wide_column(weights)),
    }
}

fn sync_packed(weights: PackedWeights) {
    match weights {
        PackedWeights::Rank4(weights) => {
            let _ = weights
                .bias
                .slice([0..1, BRANCHES - 1..BRANCHES, 0..1, MODEL_DIM - 1..MODEL_DIM])
                .into_data();
        }
        PackedWeights::Wide(weights) => {
            let _ = weights
                .bias
                .slice([BRANCHES - 1..BRANCHES, MODEL_DIM - 1..MODEL_DIM])
                .into_data();
        }
    }
}

fn assert_weight_layouts(rank4: &Rank4Weights, row: &WideWeights, column: &WideWeights) {
    assert_eq!(row.layout, WideLayout::RowMajor);
    assert_eq!(column.layout, WideLayout::ColumnMajor);

    let rank4_down = rank4.down.clone().into_primitive().tensor();
    let rank4_up = rank4.up.clone().into_primitive().tensor();
    assert!(rank4_down.is_contiguous());
    assert!(rank4_up.is_contiguous());
    assert_eq!(
        &rank4_down.meta.strides()[..],
        &[BRANCHES * MODEL_DIM * RANK, MODEL_DIM * RANK, RANK, 1]
    );
    assert_eq!(
        &rank4_up.meta.strides()[..],
        &[BRANCHES * RANK * MODEL_DIM, RANK * MODEL_DIM, MODEL_DIM, 1]
    );

    let row_down = row.down.clone().into_primitive().tensor();
    let row_up = row.up.clone().into_primitive().tensor();
    assert!(row_down.is_contiguous());
    assert!(row_up.is_contiguous());
    assert_eq!(&row_down.meta.strides()[..], &[BRANCHES * RANK, 1]);
    assert_eq!(&row_up.meta.strides()[..], &[BRANCHES * MODEL_DIM, 1]);

    let column_down = column.down.clone().into_primitive().tensor();
    let column_up = column.up.clone().into_primitive().tensor();
    assert!(!column_down.is_contiguous());
    assert!(!column_up.is_contiguous());
    assert_eq!(&column_down.meta.strides()[..], &[1, MODEL_DIM]);
    assert_eq!(&column_up.meta.strides()[..], &[1, RANK]);

    println!(
        "weight_strides: rank4_down={:?} rank4_up={:?} row_down={:?} row_up={:?} \
         col_down={:?} col_up={:?}",
        rank4_down.meta.strides(),
        rank4_up.meta.strides(),
        row_down.meta.strides(),
        row_up.meta.strides(),
        column_down.meta.strides(),
        column_up.meta.strides(),
    );
}

fn compare_tensors<const D: usize>(
    expected: Tensor<B, D>,
    actual: Tensor<B, D>,
) -> Result<Comparison, Box<dyn Error>> {
    assert_eq!(expected.dims(), actual.dims(), "comparison shape mismatch");
    let expected = expected.into_data().to_vec::<f32>()?;
    let actual = actual.into_data().to_vec::<f32>()?;
    if expected.len() != actual.len() {
        return Err(io::Error::other(format!(
            "comparison length mismatch: expected={}, actual={}",
            expected.len(),
            actual.len()
        ))
        .into());
    }

    let mut comparison = Comparison {
        elements: expected.len(),
        ..Comparison::default()
    };
    for (&expected, &actual) in expected.iter().zip(&actual) {
        if !expected.is_finite() || !actual.is_finite() {
            return Err(io::Error::other(format!(
                "non-finite comparison pair: expected={expected:?}, actual={actual:?}"
            ))
            .into());
        }
        if expected.to_bits() != actual.to_bits() {
            comparison.mismatched_bits += 1;
        }
        comparison.max_abs = comparison.max_abs.max((expected - actual).abs());
    }
    Ok(comparison)
}

fn compare_pack(
    original: &OriginalWeights,
    rank4: &Rank4Weights,
    wide: &WideWeights,
) -> Result<Comparison, Box<dyn Error>> {
    let mut combined = Comparison::default();
    for branch in 0..BRANCHES {
        combined.merge(compare_tensors(
            original.down[branch].clone(),
            rank4
                .down
                .clone()
                .narrow(1, branch, 1)
                .reshape([MODEL_DIM, RANK]),
        )?);
        combined.merge(compare_tensors(
            original.up[branch].clone(),
            rank4
                .up
                .clone()
                .narrow(1, branch, 1)
                .reshape([RANK, MODEL_DIM]),
        )?);
        combined.merge(compare_tensors(
            original.bias[branch].clone(),
            rank4.bias.clone().narrow(1, branch, 1).reshape([MODEL_DIM]),
        )?);
        combined.merge(compare_tensors(
            original.down[branch].clone(),
            wide.down.clone().narrow(1, branch * RANK, RANK),
        )?);
        combined.merge(compare_tensors(
            original.up[branch].clone(),
            wide.up.clone().narrow(1, branch * MODEL_DIM, MODEL_DIM),
        )?);
        combined.merge(compare_tensors(
            original.bias[branch].clone(),
            wide.bias.clone().narrow(0, branch, 1).reshape([MODEL_DIM]),
        )?);
    }
    Ok(combined)
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

fn measure_pack_once(original: &OriginalWeights, path: CandidatePath, iterations: usize) -> f64 {
    let started = Instant::now();
    let packed = (0..iterations)
        .map(|_| pack_path(original, path))
        .reduce(|_, packed| packed)
        .expect("pack iterations must be non-zero");
    sync_packed(packed);
    started.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64
}

fn benchmark_pack(original: &OriginalWeights, args: &Args) -> [Timing; PATHS.len()] {
    for path in PATHS {
        let packed = (0..args.warmup)
            .map(|_| pack_path(original, path))
            .reduce(|_, packed| packed)
            .expect("pack warmup must be non-zero");
        sync_packed(packed);
    }

    let mut samples: [Vec<f64>; PATHS.len()] =
        std::array::from_fn(|_| Vec::with_capacity(args.trials));
    for trial in 0..args.trials {
        for offset in 0..PATHS.len() {
            let index = (trial + offset) % PATHS.len();
            samples[index].push(measure_pack_once(original, PATHS[index], args.iterations));
        }
    }
    std::array::from_fn(|index| summarize_samples(&samples[index]))
}

fn diagonal_select(input: Tensor<B, 2>) -> Tensor<B, 2> {
    let output = adaln_wide_diagonal_select_wgsl(input.into_primitive().tensor(), BRANCHES);
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn diagonal_finalize(wide_up: Tensor<B, 2>, bias: Tensor<B, 2>, raw: Tensor<B, 2>) -> Tensor<B, 2> {
    let output = adaln_wide_diagonal_finalize_wgsl(
        wide_up.into_primitive().tensor(),
        bias.into_primitive().tensor(),
        raw.into_primitive().tensor(),
        BRANCHES,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn rank4_stages(raw: Tensor<B, 4>, weights: &Rank4Weights) -> Rank4Stages {
    let batch = raw.dims()[0];
    let activated_rank4 = silu(raw.clone());
    let down_rank4 = activated_rank4.clone().matmul(weights.down.clone());
    let up_rank4 = down_rank4.clone().matmul(weights.up.clone());
    let biased_rank4 = up_rank4.clone() + weights.bias.clone();
    let final_rank4 = biased_rank4.clone() + raw;
    Rank4Stages {
        activated: activated_rank4.reshape([batch * BRANCHES, MODEL_DIM]),
        down: down_rank4.reshape([batch * BRANCHES, RANK]),
        up: up_rank4.reshape([batch * BRANCHES, MODEL_DIM]),
        biased: biased_rank4.reshape([batch * BRANCHES, MODEL_DIM]),
        final_output: final_rank4.reshape([batch * BRANCHES, MODEL_DIM]),
    }
}

fn wide_stages(raw: Tensor<B, 4>, weights: &WideWeights) -> WideStages {
    let batch = raw.dims()[0];
    let rows = batch * BRANCHES;
    let raw_flat = raw.clone().reshape([rows, MODEL_DIM]);
    let activated = silu(raw).reshape([rows, MODEL_DIM]);
    let full_down = activated.clone().matmul(weights.down.clone());
    let down = diagonal_select(full_down);
    let full_up = down.clone().matmul(weights.up.clone());
    let up = diagonal_select(full_up.clone());
    let biased = (up.clone().reshape([batch, BRANCHES, 1, MODEL_DIM])
        + weights.bias.clone().reshape([1, BRANCHES, 1, MODEL_DIM]))
    .reshape([rows, MODEL_DIM]);
    let unfused_final = biased.clone() + raw_flat.clone();
    let fused_final = diagonal_finalize(full_up, weights.bias.clone(), raw_flat);
    WideStages {
        activated,
        down,
        up,
        biased,
        unfused_final,
        fused_final,
    }
}

fn rank4_projection(raw: Tensor<B, 4>, weights: &Rank4Weights) -> Tensor<B, 4> {
    let refined = silu(raw.clone())
        .matmul(weights.down.clone())
        .matmul(weights.up.clone())
        + weights.bias.clone();
    refined + raw
}

fn wide_projection(raw: Tensor<B, 4>, weights: &WideWeights) -> Tensor<B, 4> {
    let batch = raw.dims()[0];
    let rows = batch * BRANCHES;
    let raw_flat = raw.clone().reshape([rows, MODEL_DIM]);
    let activated = silu(raw).reshape([rows, MODEL_DIM]);
    let full_down = activated.matmul(weights.down.clone());
    let selected_down = diagonal_select(full_down);
    let full_up = selected_down.matmul(weights.up.clone());
    diagonal_finalize(full_up, weights.bias.clone(), raw_flat)
        .reshape([batch, BRANCHES, 1, MODEL_DIM])
}

fn project_path(
    raw: &Tensor<B, 4>,
    path: CandidatePath,
    rank4: &Rank4Weights,
    row: &WideWeights,
    column: &WideWeights,
) -> Tensor<B, 4> {
    match path {
        CandidatePath::Rank4 => rank4_projection(raw.clone(), rank4),
        CandidatePath::WideRow => wide_projection(raw.clone(), row),
        CandidatePath::WideColumn => wide_projection(raw.clone(), column),
    }
}

fn sync_output(output: Tensor<B, 4>, batch: usize) {
    let _ = output
        .slice([
            batch - 1..batch,
            BRANCHES - 1..BRANCHES,
            0..1,
            MODEL_DIM - 1..MODEL_DIM,
        ])
        .into_data();
}

fn measure_projection_once(
    raw: &Tensor<B, 4>,
    path: CandidatePath,
    rank4: &Rank4Weights,
    row: &WideWeights,
    column: &WideWeights,
    iterations: usize,
) -> f64 {
    let batch = raw.dims()[0];
    let started = Instant::now();
    let output = (0..iterations)
        .map(|_| project_path(raw, path, rank4, row, column))
        .reduce(|_, output| output)
        .expect("projection iterations must be non-zero");
    sync_output(output, batch);
    started.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64
}

fn benchmark_projection(
    raw: &Tensor<B, 4>,
    rank4: &Rank4Weights,
    row: &WideWeights,
    column: &WideWeights,
    args: &Args,
) -> [Timing; PATHS.len()] {
    let batch = raw.dims()[0];
    for path in PATHS {
        let output = (0..args.warmup)
            .map(|_| project_path(raw, path, rank4, row, column))
            .reduce(|_, output| output)
            .expect("projection warmup must be non-zero");
        sync_output(output, batch);
    }

    let mut samples: [Vec<f64>; PATHS.len()] =
        std::array::from_fn(|_| Vec::with_capacity(args.trials));
    for trial in 0..args.trials {
        for offset in 0..PATHS.len() {
            let index = (trial + offset) % PATHS.len();
            samples[index].push(measure_projection_once(
                raw,
                PATHS[index],
                rank4,
                row,
                column,
                args.iterations,
            ));
        }
    }
    std::array::from_fn(|index| summarize_samples(&samples[index]))
}

fn compare_stages(
    batch: usize,
    reference: &Rank4Stages,
    candidate: WideStages,
    layout: WideLayout,
) -> Result<Comparison, Box<dyn Error>> {
    let stages = [
        (
            "silu",
            compare_tensors(reference.activated.clone(), candidate.activated)?,
        ),
        (
            "down",
            compare_tensors(reference.down.clone(), candidate.down)?,
        ),
        ("up", compare_tensors(reference.up.clone(), candidate.up)?),
        (
            "bias",
            compare_tensors(reference.biased.clone(), candidate.biased)?,
        ),
        (
            "raw/final",
            compare_tensors(
                reference.final_output.clone(),
                candidate.fused_final.clone(),
            )?,
        ),
        (
            "selector-epilogue",
            compare_tensors(candidate.unfused_final, candidate.fused_final)?,
        ),
    ];
    let mut overall = Comparison::default();
    for (name, comparison) in stages {
        if comparison.max_abs > MAX_STAGE_ABS {
            return Err(io::Error::other(format!(
                "B={batch} {} stage {name} max_abs={:.9e} exceeds {:.9e}",
                layout.label(),
                comparison.max_abs,
                MAX_STAGE_ABS
            ))
            .into());
        }
        println!(
            "    stage={name:<17} elements={:>5} bit_mismatch={:>5} max_abs={:.9e}",
            comparison.elements, comparison.mismatched_bits, comparison.max_abs
        );
        overall.merge(comparison);
    }
    Ok(overall)
}

const fn weight_bytes() -> usize {
    (2 * BRANCHES * MODEL_DIM * RANK + BRANCHES * MODEL_DIM) * F32_BYTES
}

const fn current_macs(batch: usize) -> usize {
    2 * BRANCHES * batch * MODEL_DIM * RANK
}

const fn wide_macs(batch: usize) -> usize {
    3 * current_macs(batch)
}

const fn current_intermediate_write_bytes(batch: usize) -> usize {
    // SiLU + up + bias-add + raw-add each materialize 3*B*D; down writes 3*B*R.
    batch * (12 * MODEL_DIM + 3 * RANK) * F32_BYTES
}

const fn wide_intermediate_write_bytes(batch: usize) -> usize {
    // SiLU 3BD, wide-down 9BR, selected-down 3BR, wide-up 9BD, final 3BD.
    batch * (15 * MODEL_DIM + 12 * RANK) * F32_BYTES
}

const fn selector_logical_bytes(batch: usize) -> usize {
    // Down selector reads+writes 3BR. Finalizer reads wide-up+bias+raw and writes 3BD.
    batch * (6 * RANK + 12 * MODEL_DIM) * F32_BYTES
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn report_static_accounting() {
    let module_bytes = weight_bytes();
    let pinned_bytes = module_bytes * PINNED_ADALN_MODULES;
    println!(
        "weight_capacity: down={} bytes ({:.4} MiB), up={} bytes ({:.4} MiB), \
         bias={} bytes, total/module={} bytes ({:.4} MiB), pinned{}={} bytes ({:.4} MiB)",
        BRANCHES * MODEL_DIM * RANK * F32_BYTES,
        mib(BRANCHES * MODEL_DIM * RANK * F32_BYTES),
        BRANCHES * RANK * MODEL_DIM * F32_BYTES,
        mib(BRANCHES * RANK * MODEL_DIM * F32_BYTES),
        BRANCHES * MODEL_DIM * F32_BYTES,
        module_bytes,
        mib(module_bytes),
        PINNED_ADALN_MODULES,
        pinned_bytes,
        mib(pinned_bytes),
    );
    println!(
        "cache_policy: replacing current #[module(skip)] rank4 cache with wide keeps capacity \
         unchanged; co-retaining both adds {} bytes ({:.4} MiB)",
        pinned_bytes,
        mib(pinned_bytes)
    );
    for batch in [1, 2] {
        println!(
            "B={batch} accounting: current_MAC={} wide_MAC={} ratio={:.1}x, \
             current_intermediate_writes={} bytes, wide_intermediate_writes={} bytes, \
             delta={} bytes, selector_logical_rw={} bytes, logical_dispatches=5 -> 5",
            current_macs(batch),
            wide_macs(batch),
            wide_macs(batch) as f64 / current_macs(batch) as f64,
            current_intermediate_write_bytes(batch),
            wide_intermediate_write_bytes(batch),
            wide_intermediate_write_bytes(batch) - current_intermediate_write_bytes(batch),
            selector_logical_bytes(batch),
        );
    }
}

fn report_python_reference() {
    println!(
        "PyTorch strict FP32 batched reference (same card; not TorchScript):\n  \
         B1 eager_gpu={PYTORCH_B1_EAGER_GPU_US:.3} us eager_wall={PYTORCH_B1_EAGER_WALL_US:.3} us \
         compile_gpu={PYTORCH_B1_COMPILED_GPU_US:.3} us first={PYTORCH_B1_COMPILE_FIRST_MS:.3} ms \
         compiled_max_abs={PYTORCH_B1_COMPILED_MAX_ABS:.9e}\n  \
         B2 eager_gpu={PYTORCH_B2_EAGER_GPU_US:.3} us eager_wall={PYTORCH_B2_EAGER_WALL_US:.3} us \
         compile_gpu={PYTORCH_B2_COMPILED_GPU_US:.3} us first={PYTORCH_B2_COMPILE_FIRST_MS:.3} ms \
         compiled_max_abs={PYTORCH_B2_COMPILED_MAX_ABS:.9e}"
    );
}

fn break_even_requests(incremental_pack_us: f64, savings_us: f64) -> f64 {
    if savings_us <= 0.0 {
        f64::INFINITY
    } else if incremental_pack_us <= 0.0 {
        0.0
    } else {
        (incremental_pack_us / savings_us).ceil()
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let (device, error_monitor) = initialize_wgpu(args.adapter_index);
    B::seed(&device, SEED);

    println!(
        "LowRankAdaLN wide projection: device={device:?}, B=1/2, branches={BRANCHES}, \
         D={MODEL_DIM}, R={RANK}, warmup={}, iterations={} x {} trials, seed={SEED}",
        args.warmup, args.iterations, args.trials
    );
    report_static_accounting();
    report_python_reference();

    let original = original_weights(&device);
    let rank4 = pack_rank4(&original);
    let row = pack_wide_row(&original);
    let column = pack_wide_column(&original);
    assert_weight_layouts(&rank4, &row, &column);

    for wide in [&row, &column] {
        let comparison = compare_pack(&original, &rank4, wide)?;
        println!(
            "pack_correctness {}: elements={} bit_mismatch={} max_abs={:.9e}",
            wide.layout.label(),
            comparison.elements,
            comparison.mismatched_bits,
            comparison.max_abs
        );
    }

    let pack_timings = benchmark_pack(&original, &args);
    println!("pack_timing per module:");
    for (index, path) in PATHS.into_iter().enumerate() {
        let timing = pack_timings[index];
        println!(
            "  {:<9} median={:>9.3} us range=[{:>9.3},{:>9.3}] pinned{}={:.3} ms",
            path.label(),
            timing.median_us,
            timing.min_us,
            timing.max_us,
            PINNED_ADALN_MODULES,
            timing.median_us * PINNED_ADALN_MODULES as f64 / 1_000.0,
        );
    }

    let mut batch_timings = Vec::with_capacity(2);
    for batch in [1, 2] {
        let raw = Tensor::<B, 4>::random(
            [batch, BRANCHES, 1, MODEL_DIM],
            Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let reference = rank4_stages(raw.clone(), &rank4);
        println!("B={batch} staged correctness {}:", row.layout.label());
        let row_comparison = compare_stages(
            batch,
            &reference,
            wide_stages(raw.clone(), &row),
            row.layout,
        )?;
        println!(
            "  {} aggregate: elements={} bit_mismatch={} max_abs={:.9e}",
            row.layout.label(),
            row_comparison.elements,
            row_comparison.mismatched_bits,
            row_comparison.max_abs
        );
        println!("B={batch} staged correctness {}:", column.layout.label());
        let column_comparison = compare_stages(
            batch,
            &reference,
            wide_stages(raw.clone(), &column),
            column.layout,
        )?;
        println!(
            "  {} aggregate: elements={} bit_mismatch={} max_abs={:.9e}",
            column.layout.label(),
            column_comparison.elements,
            column_comparison.mismatched_bits,
            column_comparison.max_abs
        );

        let timings = benchmark_projection(&raw, &rank4, &row, &column, &args);
        println!("B={batch} timing:");
        let baseline = timings[0].median_us;
        for (index, path) in PATHS.into_iter().enumerate() {
            let timing = timings[index];
            println!(
                "  {:<9} median={:>9.3} us range=[{:>9.3},{:>9.3}] speedup={:>7.3}x",
                path.label(),
                timing.median_us,
                timing.min_us,
                timing.max_us,
                baseline / timing.median_us,
            );
        }
        batch_timings.push(timings);
        synchronize_and_check_wgpu(&device, &error_monitor, &format!("B={batch}"))?;
    }

    let [b1, b2]: [[Timing; PATHS.len()]; 2] = batch_timings
        .try_into()
        .map_err(|_| io::Error::other("both B=1 and B=2 timing sets are required"))?;
    let rank4_workload_us = B1_CALLS_PER_SYNTHESIS as f64 * b1[0].median_us
        + B2_CALLS_PER_SYNTHESIS as f64 * b2[0].median_us;
    println!(
        "four-step pinned workload: calls=B1 {} + B2 {}, rank4={:.3} ms",
        B1_CALLS_PER_SYNTHESIS,
        B2_CALLS_PER_SYNTHESIS,
        rank4_workload_us / 1_000.0
    );
    for index in 1..PATHS.len() {
        let workload_us = B1_CALLS_PER_SYNTHESIS as f64 * b1[index].median_us
            + B2_CALLS_PER_SYNTHESIS as f64 * b2[index].median_us;
        let savings_us = rank4_workload_us - workload_us;
        let current_pack_us = pack_timings[0].median_us * PINNED_ADALN_MODULES as f64;
        let candidate_pack_us = pack_timings[index].median_us * PINNED_ADALN_MODULES as f64;
        let incremental_pack_us = candidate_pack_us - current_pack_us;
        println!(
            "  {:<9} workload={:.3} ms speedup={:.3}x saving={:.3} ms, \
             replacement_pack={:.3} ms incremental_vs_rank4={:+.3} ms \
             break_even_requests={:.0}",
            PATHS[index].label(),
            workload_us / 1_000.0,
            rank4_workload_us / workload_us,
            savings_us / 1_000.0,
            candidate_pack_us / 1_000.0,
            incremental_pack_us / 1_000.0,
            break_even_requests(incremental_pack_us, savings_us),
        );
    }

    synchronize_and_check_wgpu(&device, &error_monitor, "benchmark completion")?;
    println!("wgpu_uncaptured_errors=0");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_shape_accounting_matches_wide_design() {
        assert_eq!(weight_bytes(), 5_913_600);
        assert_eq!(weight_bytes() * PINNED_ADALN_MODULES, 141_926_400);
        assert_eq!(current_macs(1), 1_474_560);
        assert_eq!(current_macs(2), 2_949_120);
        assert_eq!(wide_macs(1), 4_423_680);
        assert_eq!(wide_macs(2), 8_847_360);
        assert_eq!(wide_macs(1), 3 * current_macs(1));
        assert_eq!(current_intermediate_write_bytes(1), 63_744);
        assert_eq!(wide_intermediate_write_bytes(1), 86_016);
        assert_eq!(selector_logical_bytes(1), 66_048);
    }

    #[test]
    fn four_step_call_accounting_matches_two_adaln_modules_per_layer() {
        assert_eq!(PINNED_ADALN_MODULES, 24);
        assert_eq!(B1_CALLS_PER_SYNTHESIS, 48);
        assert_eq!(B2_CALLS_PER_SYNTHESIS, 48);
    }

    #[test]
    fn break_even_handles_win_loss_and_free_replacement() {
        assert_eq!(break_even_requests(400.0, 100.0), 4.0);
        assert_eq!(break_even_requests(-1.0, 100.0), 0.0);
        assert!(break_even_requests(100.0, -1.0).is_infinite());
    }
}
