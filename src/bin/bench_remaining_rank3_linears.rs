//! Compare the remaining rank-3 DiT linears with a flattened rank-2 GEMM.
//!
//! This benchmark does not call or modify the production model. It reproduces
//! the three remaining `Linear` shapes in the pinned v4-Small four-step Euler
//! replay after QKV+gate and SwiGLU have already adopted rank-2 flattening:
//!
//! - steps 0/1 use `B=2, S=50` while independent text CFG is active;
//! - steps 2/3 use `B=1, S=50` after CFG is inactive;
//! - each of the 12 diffusion layers applies one bias-free attention `wo`;
//! - DiT `in_proj` and `out_proj` are each applied once per backbone call and
//!   both include bias.
//!
//! Released PyTorch weights are stored physically as contiguous `[N,K]` and
//! loaded by Burn as a metadata-transposed logical `[K,N]` view. The benchmark
//! keeps that exact column-major logical layout for both variants. Only the
//! input rank changes:
//!
//! - production: `linear([B,S,K], [K,N], bias)`;
//! - candidate: `linear([B*S,K], [K,N], bias).reshape([B,S,N])`.
//!
//! Run with:
//! `cargo run --release --bin bench_remaining_rank3_linears -- <wgpu-adapter-index>`

use std::{error::Error, io, time::Instant};

use burn::{
    backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup},
    tensor::{Distribution, Tensor, backend::Backend, module::linear},
};
use irodori_tts_wgpu::WgpuRaw;

type B = WgpuRaw;

const SEQ_LEN: usize = 50;
const MODEL_DIM: usize = 1_280;
const PATCHED_LATENT_DIM: usize = 32;
const LAYERS: usize = 12;
const WARMUP: usize = 10;
const ITERATIONS: usize = 100;
const TRIALS: usize = 5;
const SEED: u64 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputRank {
    Rank3,
    FlatRank2,
}

impl InputRank {
    const fn label(self) -> &'static str {
        match self {
            Self::Rank3 => "rank3",
            Self::FlatRank2 => "flat-rank2",
        }
    }
}

const INPUT_RANKS: [InputRank; 2] = [InputRank::Rank3, InputRank::FlatRank2];

#[derive(Clone, Copy, Debug)]
struct ShapeCase {
    name: &'static str,
    k: usize,
    n: usize,
    has_bias: bool,
    b1_calls: usize,
    b2_calls: usize,
}

const SHAPES: [ShapeCase; 3] = [
    ShapeCase {
        name: "joint_attention_wo",
        k: MODEL_DIM,
        n: MODEL_DIM,
        has_bias: false,
        // Two B=1 and two B=2 backbone calls, each with 12 layers.
        b1_calls: 2 * LAYERS,
        b2_calls: 2 * LAYERS,
    },
    ShapeCase {
        name: "dit_in_proj",
        k: PATCHED_LATENT_DIM,
        n: MODEL_DIM,
        has_bias: true,
        b1_calls: 2,
        b2_calls: 2,
    },
    ShapeCase {
        name: "dit_out_proj",
        k: MODEL_DIM,
        n: PATCHED_LATENT_DIM,
        has_bias: true,
        b1_calls: 2,
        b2_calls: 2,
    },
];

#[derive(Clone, Copy, Debug)]
struct BatchTiming {
    rank3_us: f64,
    flat_rank2_us: f64,
    max_abs: f32,
}

impl BatchTiming {
    fn for_rank(self, rank: InputRank) -> f64 {
        match rank {
            InputRank::Rank3 => self.rank3_us,
            InputRank::FlatRank2 => self.flat_rank2_us,
        }
    }
}

fn parse_adapter_index() -> Result<usize, Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let adapter_index = args
        .next()
        .ok_or_else(|| io::Error::other("missing required WGPU adapter index"))?
        .parse::<usize>()?;
    if let Some(extra) = args.next() {
        return Err(io::Error::other(format!(
            "unexpected argument {extra:?}; expected exactly one WGPU adapter index"
        ))
        .into());
    }
    Ok(adapter_index)
}

fn project(
    input: Tensor<B, 3>,
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
    rank: InputRank,
    batch: usize,
    case: ShapeCase,
) -> Tensor<B, 3> {
    match rank {
        InputRank::Rank3 => linear(input, weight, bias),
        InputRank::FlatRank2 => linear(input.reshape([batch * SEQ_LEN, case.k]), weight, bias)
            .reshape([batch, SEQ_LEN, case.n]),
    }
}

fn sync_matrix(matrix: Tensor<B, 2>, rows: usize, cols: usize) {
    let _ = matrix.slice([rows - 1..rows, cols - 1..cols]).into_data();
}

fn sync_output(output: Tensor<B, 3>, batch: usize, n: usize) {
    let _ = output
        .slice([batch - 1..batch, SEQ_LEN - 1..SEQ_LEN, n - 1..n])
        .into_data();
}

fn max_abs_diff(lhs: Tensor<B, 3>, rhs: Tensor<B, 3>) -> Result<f32, Box<dyn Error>> {
    (lhs - rhs)
        .abs()
        .max()
        .into_data()
        .to_vec::<f32>()?
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("maximum reduction returned no value").into())
}

fn measure<F>(mut operation: F, batch: usize, n: usize) -> f64
where
    F: FnMut() -> Tensor<B, 3>,
{
    let warmup_output = (0..WARMUP)
        .map(|_| operation())
        .reduce(|_, output| output)
        .expect("WARMUP must be non-zero");
    sync_output(warmup_output, batch, n);

    let started = Instant::now();
    let output = (0..ITERATIONS)
        .map(|_| operation())
        .reduce(|_, output| output)
        .expect("ITERATIONS must be non-zero");
    sync_output(output, batch, n);
    started.elapsed().as_secs_f64() * 1_000_000.0 / ITERATIONS as f64
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn bench_batch(
    case: ShapeCase,
    batch: usize,
    input: Tensor<B, 3>,
    weight: &Tensor<B, 2>,
    bias: &Option<Tensor<B, 1>>,
) -> Result<BatchTiming, Box<dyn Error>> {
    // Correctness calls also prime both exact autotune keys before timing.
    let expected = project(
        input.clone(),
        weight.clone(),
        bias.clone(),
        InputRank::Rank3,
        batch,
        case,
    );
    let actual = project(
        input.clone(),
        weight.clone(),
        bias.clone(),
        InputRank::FlatRank2,
        batch,
        case,
    );
    let max_abs = max_abs_diff(expected, actual)?;
    if !max_abs.is_finite() {
        return Err(io::Error::other(format!(
            "{} B={batch} produced non-finite max_abs={max_abs}",
            case.name
        ))
        .into());
    }

    // Alternate which input rank is measured first to reduce order bias.
    let mut samples: [Vec<f64>; INPUT_RANKS.len()] =
        std::array::from_fn(|_| Vec::with_capacity(TRIALS));
    for trial in 0..TRIALS {
        for offset in 0..INPUT_RANKS.len() {
            let index = (trial + offset) % INPUT_RANKS.len();
            let rank = INPUT_RANKS[index];
            samples[index].push(measure(
                || {
                    project(
                        input.clone(),
                        weight.clone(),
                        bias.clone(),
                        rank,
                        batch,
                        case,
                    )
                },
                batch,
                case.n,
            ));
        }
    }

    let rank3_us = median(&samples[0]);
    let flat_rank2_us = median(&samples[1]);
    let mac_per_call = batch * SEQ_LEN * case.k * case.n;
    println!(
        "  B={batch}: MAC/call={mac_per_call} ({:.6} G), max_abs={max_abs:.3e}",
        mac_per_call as f64 / 1.0e9
    );
    for (index, rank) in INPUT_RANKS.into_iter().enumerate() {
        let measured = median(&samples[index]);
        let min = samples[index].iter().copied().fold(f64::INFINITY, f64::min);
        let max = samples[index]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "    {:<10} median={:>9.1} us range=[{:>9.1},{:>9.1}] speedup={:>6.3}x",
            rank.label(),
            measured,
            min,
            max,
            rank3_us / measured,
        );
    }

    Ok(BatchTiming {
        rank3_us,
        flat_rank2_us,
        max_abs,
    })
}

fn bench_shape(case: ShapeCase, device: &<B as Backend>::Device) -> Result<(), Box<dyn Error>> {
    // Reproduce checkpoint loading exactly: contiguous physical PyTorch
    // `[N,K]`, metadata-only transpose to Burn's logical `[K,N]`.
    let physical_weight =
        Tensor::<B, 2>::random([case.n, case.k], Distribution::Uniform(-0.05, 0.05), device);
    sync_matrix(physical_weight.clone(), case.n, case.k);
    let weight = physical_weight.transpose();
    let primitive = weight.clone().into_primitive().tensor();
    assert!(!primitive.is_contiguous());
    assert_eq!(&primitive.meta.strides()[..], &[1, case.k]);

    let bias = case
        .has_bias
        .then(|| Tensor::<B, 1>::random([case.n], Distribution::Uniform(-0.05, 0.05), device));
    println!(
        "\n{}: [B,S={SEQ_LEN},K={}] @ [K={},N={}] bias={} logical_weight_strides=[1,{}]",
        case.name, case.k, case.k, case.n, case.has_bias, case.k
    );

    let timings = [1, 2]
        .into_iter()
        .map(|batch| {
            let input = Tensor::<B, 3>::random(
                [batch, SEQ_LEN, case.k],
                Distribution::Uniform(-1.0, 1.0),
                device,
            );
            bench_batch(case, batch, input, &weight, &bias)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let [b1, b2] = timings
        .as_slice()
        .try_into()
        .expect("both B=1 and B=2 timings must exist");
    let workload_us =
        |rank| case.b1_calls as f64 * b1.for_rank(rank) + case.b2_calls as f64 * b2.for_rank(rank);
    let production_us = workload_us(InputRank::Rank3);
    let candidate_us = workload_us(InputRank::FlatRank2);
    let b1_mac = SEQ_LEN * case.k * case.n;
    let b2_mac = 2 * b1_mac;
    let workload_mac = case.b1_calls * b1_mac + case.b2_calls * b2_mac;
    println!(
        "  pinned workload: calls={} (B1={}, B2={}), MAC={} ({:.6} G), \
         rank3={:.3} ms, flat-rank2={:.3} ms, speedup={:.3}x, saving={:.3} ms, \
         max_abs=[B1 {:.3e}, B2 {:.3e}]",
        case.b1_calls + case.b2_calls,
        case.b1_calls,
        case.b2_calls,
        workload_mac,
        workload_mac as f64 / 1.0e9,
        production_us / 1_000.0,
        candidate_us / 1_000.0,
        production_us / candidate_us,
        (production_us - candidate_us) / 1_000.0,
        b1.max_abs,
        b2.max_abs,
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let adapter_index = parse_adapter_index()?;
    let device = WgpuDevice::DiscreteGpu(adapter_index);
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    B::seed(&device, SEED);

    println!(
        "v4 remaining rank-3 Linear benchmark: device={device:?}, B=1/2, S={SEQ_LEN}, \
         {WARMUP} warmup, {ITERATIONS} measured x {TRIALS} trials, seed={SEED}"
    );
    for case in SHAPES {
        bench_shape(case, &device)?;
    }
    Ok(())
}
