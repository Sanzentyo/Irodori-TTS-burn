//! Measure one-time row-major packing for the remaining v4 DiT projections.
//!
//! The current prepared inference path has already flattened `[B,S,K]` to
//! `[B*S,K]`, but `w2` and JointAttention `wo` still use the metadata-
//! transposed column-major views produced by checkpoint loading. This isolated
//! benchmark holds input rank constant and compares only weight layout:
//!
//! - baseline: current column-major logical `[K,N]` view;
//! - candidate: one bit-preserving `into_contiguous` pack to row-major `[K,N]`.
//!
//! Both exact batch shapes from the pinned four-step replay are measured:
//! `B=2, S=50` for steps 0/1 and `B=1, S=50` for steps 2/3. Each projection is
//! called once per layer per step, so the weighted workload contains 24 B=1
//! and 24 B=2 calls across the 12-layer checkpoint.
//!
//! A production cache would initially have to be additive and
//! `#[module(skip)]`. The learned `Linear` parameters remain necessary for
//! training, ordinary forward paths, device/record behavior, and checkpoint
//! compatibility; this benchmark therefore reports the full co-retained cache
//! allocation rather than assuming that the source parameters can be dropped.
//!
//! Run, once registered in `Cargo.toml`, with:
//! `cargo run --release --bin bench_dit_weight_pack -- <wgpu-adapter-index>`

use std::{error::Error, io, time::Instant};

use burn::{
    backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup, into_contiguous},
    tensor::{Distribution, Tensor, TensorPrimitive, backend::Backend, module::linear},
};
use irodori_tts_wgpu::WgpuRaw;

type B = WgpuRaw;

const SEQ_LEN: usize = 50;
const MODEL_DIM: usize = 1_280;
const MLP_HIDDEN: usize = 3_680;
const LAYERS: usize = 12;
const B1_CALLS: usize = 2 * LAYERS;
const B2_CALLS: usize = 2 * LAYERS;
const WARMUP: usize = 10;
const ITERATIONS: usize = 100;
const TRIALS: usize = 5;
const SEED: u64 = 0;
const F32_BYTES: usize = core::mem::size_of::<f32>();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeightLayout {
    ColumnMajor,
    RowMajor,
}

impl WeightLayout {
    const fn label(self) -> &'static str {
        match self {
            Self::ColumnMajor => "flat-col",
            Self::RowMajor => "flat-row",
        }
    }
}

const LAYOUTS: [WeightLayout; 2] = [WeightLayout::ColumnMajor, WeightLayout::RowMajor];

#[derive(Clone, Copy, Debug)]
struct ShapeCase {
    name: &'static str,
    k: usize,
    n: usize,
}

const SHAPES: [ShapeCase; 2] = [
    ShapeCase {
        name: "swiglu_w2",
        k: MLP_HIDDEN,
        n: MODEL_DIM,
    },
    ShapeCase {
        name: "joint_attention_wo",
        k: MODEL_DIM,
        n: MODEL_DIM,
    },
];

#[derive(Clone, Copy, Debug)]
struct BatchTiming {
    column_us: f64,
    row_us: f64,
    max_abs: f32,
}

impl BatchTiming {
    const fn for_layout(self, layout: WeightLayout) -> f64 {
        match layout {
            WeightLayout::ColumnMajor => self.column_us,
            WeightLayout::RowMajor => self.row_us,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ShapeResult {
    weight_bytes: usize,
    pack_us_per_layer: f64,
    column_workload_us: f64,
    row_workload_us: f64,
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

fn pack_row_major(weight: Tensor<B, 2>) -> Tensor<B, 2> {
    let packed = into_contiguous(weight.into_primitive().tensor());
    Tensor::from_primitive(TensorPrimitive::Float(packed))
}

fn project(
    input: Tensor<B, 3>,
    weight: Tensor<B, 2>,
    batch: usize,
    case: ShapeCase,
) -> Tensor<B, 3> {
    linear(input.reshape([batch * SEQ_LEN, case.k]), weight, None).reshape([batch, SEQ_LEN, case.n])
}

fn sync_matrix(matrix: Tensor<B, 2>, rows: usize, cols: usize) {
    let _ = matrix.slice([rows - 1..rows, cols - 1..cols]).into_data();
}

fn sync_output(output: Tensor<B, 3>, batch: usize, n: usize) {
    let _ = output
        .slice([batch - 1..batch, SEQ_LEN - 1..SEQ_LEN, n - 1..n])
        .into_data();
}

fn max_abs_diff_2d(lhs: Tensor<B, 2>, rhs: Tensor<B, 2>) -> Result<f32, Box<dyn Error>> {
    (lhs - rhs)
        .abs()
        .max()
        .into_data()
        .to_vec::<f32>()?
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("weight maximum reduction returned no value").into())
}

fn max_abs_diff_3d(lhs: Tensor<B, 3>, rhs: Tensor<B, 3>) -> Result<f32, Box<dyn Error>> {
    (lhs - rhs)
        .abs()
        .max()
        .into_data()
        .to_vec::<f32>()?
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("output maximum reduction returned no value").into())
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
    weight_column: &Tensor<B, 2>,
    weight_row: &Tensor<B, 2>,
) -> Result<BatchTiming, Box<dyn Error>> {
    let select_weight = |layout| match layout {
        WeightLayout::ColumnMajor => weight_column.clone(),
        WeightLayout::RowMajor => weight_row.clone(),
    };

    // Prime both exact autotune keys and verify the packed result before the
    // explicit warmup loops.
    let expected = project(input.clone(), weight_column.clone(), batch, case);
    let actual = project(input.clone(), weight_row.clone(), batch, case);
    let max_abs = max_abs_diff_3d(expected, actual)?;
    if !max_abs.is_finite() {
        return Err(io::Error::other(format!(
            "{} B={batch} produced non-finite max_abs={max_abs}",
            case.name
        ))
        .into());
    }

    let mut samples: [Vec<f64>; LAYOUTS.len()] =
        std::array::from_fn(|_| Vec::with_capacity(TRIALS));
    for trial in 0..TRIALS {
        for offset in 0..LAYOUTS.len() {
            let index = (trial + offset) % LAYOUTS.len();
            let layout = LAYOUTS[index];
            let weight = select_weight(layout);
            samples[index].push(measure(
                || project(input.clone(), weight.clone(), batch, case),
                batch,
                case.n,
            ));
        }
    }

    let column_us = median(&samples[0]);
    let row_us = median(&samples[1]);
    let mac_per_call = batch * SEQ_LEN * case.k * case.n;
    println!(
        "  B={batch}: MAC/call={mac_per_call} ({:.6} G), max_abs={max_abs:.3e}",
        mac_per_call as f64 / 1.0e9
    );
    for (index, layout) in LAYOUTS.into_iter().enumerate() {
        let measured = median(&samples[index]);
        let min = samples[index].iter().copied().fold(f64::INFINITY, f64::min);
        let max = samples[index]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "    {:<8} median={:>9.1} us range=[{:>9.1},{:>9.1}] speedup={:>6.3}x",
            layout.label(),
            measured,
            min,
            max,
            column_us / measured,
        );
    }

    Ok(BatchTiming {
        column_us,
        row_us,
        max_abs,
    })
}

fn bench_shape(
    case: ShapeCase,
    device: &<B as Backend>::Device,
) -> Result<ShapeResult, Box<dyn Error>> {
    // Match checkpoint loading: physical contiguous PyTorch `[N,K]`, followed
    // by a metadata-only logical transpose to Burn `[K,N]`.
    let physical_weight =
        Tensor::<B, 2>::random([case.n, case.k], Distribution::Uniform(-0.05, 0.05), device);
    sync_matrix(physical_weight.clone(), case.n, case.k);
    let weight_column = physical_weight.transpose();
    let column_primitive = weight_column.clone().into_primitive().tensor();
    assert!(!column_primitive.is_contiguous());
    assert_eq!(&column_primitive.meta.strides()[..], &[1, case.k]);

    let pack_started = Instant::now();
    let weight_row = pack_row_major(weight_column.clone());
    sync_matrix(weight_row.clone(), case.k, case.n);
    let pack_us_per_layer = pack_started.elapsed().as_secs_f64() * 1_000_000.0;
    let row_primitive = weight_row.clone().into_primitive().tensor();
    assert!(row_primitive.is_contiguous());
    assert_eq!(&row_primitive.meta.strides()[..], &[case.n, 1]);
    let pack_max_abs = max_abs_diff_2d(weight_column.clone(), weight_row.clone())?;
    if !pack_max_abs.is_finite() {
        return Err(io::Error::other(format!(
            "{} row-major pack produced non-finite max_abs={pack_max_abs}",
            case.name
        ))
        .into());
    }

    let weight_bytes = case.k * case.n * F32_BYTES;
    println!(
        "\n{}: flat [M=B*{SEQ_LEN},K={}] @ [K={},N={}], cache/layer={} bytes \
         ({:.3} MiB), pinned12={} bytes ({:.3} MiB), pack/layer={pack_us_per_layer:.1} us, \
         pack_max_abs={pack_max_abs:.3e}",
        case.name,
        case.k,
        case.k,
        case.n,
        weight_bytes,
        mib(weight_bytes),
        weight_bytes * LAYERS,
        mib(weight_bytes * LAYERS),
    );

    let timings = [1, 2]
        .into_iter()
        .map(|batch| {
            let input = Tensor::<B, 3>::random(
                [batch, SEQ_LEN, case.k],
                Distribution::Uniform(-1.0, 1.0),
                device,
            );
            bench_batch(case, batch, input, &weight_column, &weight_row)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let [b1, b2] = timings
        .as_slice()
        .try_into()
        .expect("both B=1 and B=2 timings must exist");
    let workload_us =
        |layout| B1_CALLS as f64 * b1.for_layout(layout) + B2_CALLS as f64 * b2.for_layout(layout);
    let column_workload_us = workload_us(WeightLayout::ColumnMajor);
    let row_workload_us = workload_us(WeightLayout::RowMajor);
    let b1_mac = SEQ_LEN * case.k * case.n;
    let b2_mac = 2 * b1_mac;
    let workload_mac = B1_CALLS * b1_mac + B2_CALLS * b2_mac;
    let pinned_pack_us = pack_us_per_layer * LAYERS as f64;
    let saving_us = column_workload_us - row_workload_us;
    let break_even_requests = if saving_us > 0.0 {
        (pinned_pack_us / saving_us).ceil()
    } else {
        f64::INFINITY
    };
    println!(
        "  pinned workload: calls={} (B1={B1_CALLS}, B2={B2_CALLS}), MAC={} ({:.6} G), \
         flat-col={:.3} ms, flat-row={:.3} ms, speedup={:.3}x, saving={:.3} ms, \
         pinned12_pack={:.3} ms, break_even_requests={break_even_requests:.0}, \
         max_abs=[B1 {:.3e}, B2 {:.3e}]",
        B1_CALLS + B2_CALLS,
        workload_mac,
        workload_mac as f64 / 1.0e9,
        column_workload_us / 1_000.0,
        row_workload_us / 1_000.0,
        column_workload_us / row_workload_us,
        saving_us / 1_000.0,
        pinned_pack_us / 1_000.0,
        b1.max_abs,
        b2.max_abs,
    );

    Ok(ShapeResult {
        weight_bytes,
        pack_us_per_layer,
        column_workload_us,
        row_workload_us,
    })
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() -> Result<(), Box<dyn Error>> {
    let adapter_index = parse_adapter_index()?;
    let device = WgpuDevice::DiscreteGpu(adapter_index);
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    B::seed(&device, SEED);

    println!(
        "v4 DiT row-major weight-pack benchmark: device={device:?}, B=1/2, S={SEQ_LEN}, \
         {WARMUP} warmup, {ITERATIONS} measured x {TRIALS} trials, seed={SEED}"
    );
    let results = SHAPES
        .into_iter()
        .map(|case| bench_shape(case, &device))
        .collect::<Result<Vec<_>, _>>()?;

    let cache_bytes_per_layer = results
        .iter()
        .map(|result| result.weight_bytes)
        .sum::<usize>();
    let pinned_pack_us = results
        .iter()
        .map(|result| result.pack_us_per_layer * LAYERS as f64)
        .sum::<f64>();
    let column_workload_us = results
        .iter()
        .map(|result| result.column_workload_us)
        .sum::<f64>();
    let row_workload_us = results
        .iter()
        .map(|result| result.row_workload_us)
        .sum::<f64>();
    let saving_us = column_workload_us - row_workload_us;
    let break_even_requests = if saving_us > 0.0 {
        (pinned_pack_us / saving_us).ceil()
    } else {
        f64::INFINITY
    };
    println!(
        "\ncombined additive-cache candidate: per-layer={} bytes ({:.3} MiB), \
         pinned12={} bytes ({:.3} MiB), pack={:.3} ms, flat-col={:.3} ms, \
         flat-row={:.3} ms, speedup={:.3}x, saving={:.3} ms/request, \
         break_even_requests={break_even_requests:.0}",
        cache_bytes_per_layer,
        mib(cache_bytes_per_layer),
        cache_bytes_per_layer * LAYERS,
        mib(cache_bytes_per_layer * LAYERS),
        pinned_pack_us / 1_000.0,
        column_workload_us / 1_000.0,
        row_workload_us / 1_000.0,
        column_workload_us / row_workload_us,
        saving_us / 1_000.0,
    );
    Ok(())
}
