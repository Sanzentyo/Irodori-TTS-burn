//! Isolated layout benchmark for the dominant v4 DiT projection GEMMs.
//!
//! The production model is intentionally not called or modified here.  Each
//! case uses both exact batch shapes from the pinned four-step Euler replay:
//! `B=2, S=50` for steps 0/1 while text CFG is active and `B=1, S=50` for
//! steps 2/3 after CFG is inactive. It compares two independent choices:
//!
//! - rank-3 batched matmul (`M=50`, batch 1 or 2), as used by Burn `Linear`;
//! - rank-2 matmul after flattening `B*S` (`M=50` or `M=100`);
//! - row-major logical `[K, N]` weights;
//! - column-major logical `[K, N]` weights, matching a transposed PyTorch
//!   `[N, K]` checkpoint tensor without materialising it in row-major order.
//!
//! Run with:
//! `cargo run --release --bin bench_dit_matmul_layout -- <wgpu-adapter-index>`

use std::{error::Error, io, time::Instant};

use burn::{
    backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup},
    tensor::{Distribution, Tensor, backend::Backend},
};
use irodori_tts_wgpu::WgpuRaw;

type B = WgpuRaw;

const SEQ_LEN: usize = 50;
const MODEL_DIM: usize = 1_280;
const MLP_HIDDEN: usize = 3_680;
const WARMUP: usize = 10;
const ITERATIONS: usize = 100;
const TRIALS: usize = 5;
const SEED: u64 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeightLayout {
    RowMajor,
    ColMajor,
}

impl WeightLayout {
    const fn label(self) -> &'static str {
        match self {
            Self::RowMajor => "row",
            Self::ColMajor => "col",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputLayout {
    BatchedRank3,
    FlattenedRank2,
}

impl InputLayout {
    const fn label(self) -> &'static str {
        match self {
            Self::BatchedRank3 => "rank3",
            Self::FlattenedRank2 => "flat-rank2",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Variant {
    input: InputLayout,
    weight: WeightLayout,
}

impl Variant {
    const fn label(self) -> &'static str {
        match (self.input, self.weight) {
            (InputLayout::BatchedRank3, WeightLayout::RowMajor) => "rank3-row",
            (InputLayout::BatchedRank3, WeightLayout::ColMajor) => "rank3-col",
            (InputLayout::FlattenedRank2, WeightLayout::RowMajor) => "flat-row",
            (InputLayout::FlattenedRank2, WeightLayout::ColMajor) => "flat-col",
        }
    }
}

const VARIANTS: [Variant; 4] = [
    Variant {
        input: InputLayout::BatchedRank3,
        weight: WeightLayout::RowMajor,
    },
    Variant {
        input: InputLayout::BatchedRank3,
        weight: WeightLayout::ColMajor,
    },
    Variant {
        input: InputLayout::FlattenedRank2,
        weight: WeightLayout::RowMajor,
    },
    Variant {
        input: InputLayout::FlattenedRank2,
        weight: WeightLayout::ColMajor,
    },
];

#[derive(Clone, Copy, Debug)]
struct ShapeCase {
    name: &'static str,
    k: usize,
    n: usize,
    production: Variant,
}

const SHAPES: [ShapeCase; 3] = [
    ShapeCase {
        name: "mlp_expand_w1_w3",
        k: MODEL_DIM,
        n: 2 * MLP_HIDDEN,
        // `Tensor::cat(w1, w3, dim=1)` creates the inference cache.
        production: Variant {
            input: InputLayout::BatchedRank3,
            weight: WeightLayout::RowMajor,
        },
    },
    ShapeCase {
        name: "attention_fused_qkv",
        k: MODEL_DIM,
        n: 3 * MODEL_DIM,
        // `Tensor::cat(wq, wk, wv, dim=1)` creates the inference cache.
        production: Variant {
            input: InputLayout::BatchedRank3,
            weight: WeightLayout::RowMajor,
        },
    },
    ShapeCase {
        name: "mlp_contract_w2",
        k: MLP_HIDDEN,
        n: MODEL_DIM,
        // Checkpoint `[N,K]` is loaded then transposed as metadata, so the
        // logical Burn `[K,N]` matrix is column-major.
        production: Variant {
            input: InputLayout::BatchedRank3,
            weight: WeightLayout::ColMajor,
        },
    },
];

fn parse_adapter_index() -> Result<usize, Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let adapter = args
        .next()
        .ok_or_else(|| io::Error::other("missing required WGPU adapter index"))?
        .parse::<usize>()?;
    if let Some(extra) = args.next() {
        return Err(io::Error::other(format!(
            "unexpected argument {extra:?}; expected one WGPU adapter index"
        ))
        .into());
    }
    Ok(adapter)
}

/// Materialise `[N,K]` in row-major order, then expose its transpose as a
/// logical column-major `[K,N]` matrix.  The final transpose is metadata-only.
fn pack_col_major(weight_row: Tensor<B, 2>) -> Tensor<B, 2> {
    weight_row.transpose().add_scalar(0.0).transpose()
}

fn sync_matrix(matrix: Tensor<B, 2>, rows: usize, cols: usize) {
    let _ = matrix.slice([rows - 1..rows, cols - 1..cols]).into_data();
}

fn sync_output(output: Tensor<B, 3>, batch: usize, n: usize) {
    let _ = output
        .slice([batch - 1..batch, SEQ_LEN - 1..SEQ_LEN, n - 1..n])
        .into_data();
}

fn project(
    input: Tensor<B, 3>,
    weight: Tensor<B, 2>,
    variant: Variant,
    batch: usize,
    k: usize,
    n: usize,
) -> Tensor<B, 3> {
    match variant.input {
        InputLayout::BatchedRank3 => input.matmul(weight.unsqueeze::<3>()),
        InputLayout::FlattenedRank2 => input
            .reshape([batch * SEQ_LEN, k])
            .matmul(weight)
            .reshape([batch, SEQ_LEN, n]),
    }
}

fn max_abs_diff(lhs: Tensor<B, 3>, rhs: Tensor<B, 3>) -> Result<f32, Box<dyn Error>> {
    let values = (lhs - rhs).abs().max().into_data().to_vec::<f32>()?;
    values
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("maximum reduction returned no value").into())
}

fn measure<F>(mut operation: F, batch: usize, n: usize) -> f64
where
    F: FnMut() -> Tensor<B, 3>,
{
    let mut warmup_output = operation();
    for _ in 1..WARMUP {
        warmup_output = operation();
    }
    sync_output(warmup_output, batch, n);

    let started = Instant::now();
    let mut output = operation();
    for _ in 1..ITERATIONS {
        output = operation();
    }
    sync_output(output, batch, n);
    started.elapsed().as_secs_f64() * 1_000_000.0 / ITERATIONS as f64
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn bench_batch(
    case: ShapeCase,
    batch: usize,
    input: Tensor<B, 3>,
    weight_row: &Tensor<B, 2>,
    weight_col: &Tensor<B, 2>,
) -> Result<(), Box<dyn Error>> {
    let select_weight = |layout| match layout {
        WeightLayout::RowMajor => weight_row.clone(),
        WeightLayout::ColMajor => weight_col.clone(),
    };

    // Prime every exact autotune key and verify both layout choices and both
    // input-rank choices against the production variant before timing.
    let reference = project(
        input.clone(),
        select_weight(case.production.weight),
        case.production,
        batch,
        case.k,
        case.n,
    );
    let mut errors = [0.0_f32; VARIANTS.len()];
    for (index, variant) in VARIANTS.iter().copied().enumerate() {
        let output = project(
            input.clone(),
            select_weight(variant.weight),
            variant,
            batch,
            case.k,
            case.n,
        );
        errors[index] = max_abs_diff(reference.clone(), output)?;
        if !errors[index].is_finite() {
            return Err(io::Error::other(format!(
                "{} B={batch} {} produced non-finite error",
                case.name,
                variant.label()
            ))
            .into());
        }
    }

    // Rotate ordering across trials to reduce systematic thermal/order bias.
    let mut timings: [Vec<f64>; VARIANTS.len()] =
        std::array::from_fn(|_| Vec::with_capacity(TRIALS));
    for trial in 0..TRIALS {
        for offset in 0..VARIANTS.len() {
            let index = (trial + offset) % VARIANTS.len();
            let variant = VARIANTS[index];
            let weight = select_weight(variant.weight);
            let us = measure(
                || {
                    project(
                        input.clone(),
                        weight.clone(),
                        variant,
                        batch,
                        case.k,
                        case.n,
                    )
                },
                batch,
                case.n,
            );
            timings[index].push(us);
        }
    }

    let mac_per_call = batch * SEQ_LEN * case.k * case.n;
    println!(
        "  B={batch}: [B,M={SEQ_LEN},K={}] @ [K={},N={}] MAC/call={} ({:.6} G)",
        case.k,
        case.k,
        case.n,
        mac_per_call,
        mac_per_call as f64 / 1.0e9,
    );
    let production_index = VARIANTS
        .iter()
        .position(|variant| *variant == case.production)
        .expect("production variant must be benchmarked");
    let production_us = median(&timings[production_index]);
    for (index, variant) in VARIANTS.iter().copied().enumerate() {
        let measured = median(&timings[index]);
        let min = timings[index].iter().copied().fold(f64::INFINITY, f64::min);
        let max = timings[index]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "    {:<10} input={:<10} weight={:<3} median={:>9.1} us \
             range=[{:>9.1},{:>9.1}] speedup={:>6.3}x max_abs={:.3e}",
            variant.label(),
            variant.input.label(),
            variant.weight.label(),
            measured,
            min,
            max,
            production_us / measured,
            errors[index],
        );
    }
    Ok(())
}

fn bench_shape(case: ShapeCase, device: &<B as Backend>::Device) -> Result<(), Box<dyn Error>> {
    let weight_row =
        Tensor::<B, 2>::random([case.k, case.n], Distribution::Uniform(-0.05, 0.05), device);
    // Exclude random initialisation from the one-time pack measurement.
    sync_matrix(weight_row.clone(), case.k, case.n);
    let pack_started = Instant::now();
    let weight_col = pack_col_major(weight_row.clone());
    sync_matrix(weight_col.clone(), case.k, case.n);
    let pack_us = pack_started.elapsed().as_secs_f64() * 1_000_000.0;

    let weight_bytes = case.k * case.n * core::mem::size_of::<f32>();
    println!(
        "\n{}: K={} N={} weight={:.3} MiB production={} row->col pack={pack_us:.1} us",
        case.name,
        case.k,
        case.n,
        mib(weight_bytes),
        case.production.label()
    );
    for batch in [1, 2] {
        let input = Tensor::<B, 3>::random(
            [batch, SEQ_LEN, case.k],
            Distribution::Uniform(-1.0, 1.0),
            device,
        );
        bench_batch(case, batch, input, &weight_row, &weight_col)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let adapter_index = parse_adapter_index()?;
    let device = WgpuDevice::DiscreteGpu(adapter_index);
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    B::seed(&device, SEED);

    println!(
        "v4 DiT dominant matmul layout benchmark: device={device:?}, B=1/2, S={SEQ_LEN}, \
         {WARMUP} warmup, {ITERATIONS} measured x {TRIALS} trials, seed={SEED}"
    );
    for case in SHAPES {
        bench_shape(case, &device)?;
    }
    Ok(())
}
