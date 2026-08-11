//! Compare separate and packed JointAttention QKV/gate projections.
//!
//! Production's gate projection has no bias. The candidate concatenates the
//! existing fused QKV and gate weights once, then replaces two matmuls with one:
//! `[D, 3D] || [D, D] -> [D, 4D]`. Both paths include gate sigmoid.
//!
//! Run with:
//! `cargo run --release --bin bench_qkv_gate_projection -- <wgpu-adapter-index>`

use std::{error::Error, io, time::Instant};

use burn::{
    backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup, into_contiguous},
    tensor::{Distribution, Tensor, TensorPrimitive, activation::sigmoid, backend::Backend},
};
use irodori_tts_wgpu::{
    WgpuRaw,
    kernels::qkv_postprocess::{fused_qkv_gate_postprocess_wgsl, fused_qkv_postprocess_wgsl},
};

type B = WgpuRaw;

const MODEL_DIM: usize = 1_280;
const QKV_DIM: usize = 3 * MODEL_DIM;
const PACKED_DIM: usize = QKV_DIM + MODEL_DIM;
const NUM_HEADS: usize = 20;
const HEAD_DIM: usize = 64;
const SEQ_LEN: usize = 50;
const WARMUP: usize = 10;
const ITERATIONS: usize = 100;
const SEED: u64 = 0;
const EPS: f64 = 1.0e-6;
const F32_BYTES: usize = core::mem::size_of::<f32>();

const CURRENT_QKV_PACK_BYTES: usize = MODEL_DIM * QKV_DIM * F32_BYTES;
const COMBINED_PACK_BYTES: usize = MODEL_DIM * PACKED_DIM * F32_BYTES;
const REPLACEMENT_INCREMENT_BYTES: usize = COMBINED_PACK_BYTES - CURRENT_QKV_PACK_BYTES;

#[derive(Clone)]
struct ProjectionWeights {
    /// Logical `[in,out]` Q/K/V views over contiguous checkpoint-style
    /// physical `[out,in]` allocations.
    qkv_parts: [Tensor<B, 2>; 3],
    /// Current inference pack, materialised as row-major `[D,3D]`.
    qkv: Tensor<B, 2>,
    /// Logical `[in,out]` view over checkpoint-style `[out,in]` storage.
    gate: Tensor<B, 2>,
    q_norm: Tensor<B, 2>,
    k_norm: Tensor<B, 2>,
}

#[derive(Clone)]
struct ProjectionOutputs {
    qkv: Tensor<B, 3>,
    gate: Tensor<B, 3>,
}

#[derive(Clone)]
struct PostprocessOutputs {
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    gate: Tensor<B, 3>,
    gated: Tensor<B, 3>,
}

#[derive(Clone, Copy, Debug)]
struct FullPostTiming {
    separate_us: f64,
    rank3_row_us: f64,
    flat_row_us: f64,
    flat_col_us: f64,
}

fn projection_weights(device: &<B as Backend>::Device) -> ProjectionWeights {
    let qkv_parts = std::array::from_fn(|_| {
        Tensor::random(
            [MODEL_DIM, MODEL_DIM],
            Distribution::Uniform(-0.05, 0.05),
            device,
        )
        .transpose()
    });
    let qkv = Tensor::cat(qkv_parts.to_vec(), 1);
    let gate = Tensor::random(
        [MODEL_DIM, MODEL_DIM],
        Distribution::Uniform(-0.05, 0.05),
        device,
    )
    .transpose();
    let q_norm = Tensor::random(
        [NUM_HEADS, HEAD_DIM],
        Distribution::Uniform(0.5, 1.5),
        device,
    );
    let k_norm = Tensor::random(
        [NUM_HEADS, HEAD_DIM],
        Distribution::Uniform(0.5, 1.5),
        device,
    );
    ProjectionWeights {
        qkv_parts,
        qkv,
        gate,
        q_norm,
        k_norm,
    }
}

/// Current production shape: fused QKV matmul plus an independent gate matmul.
fn separate_projection(input: Tensor<B, 3>, weights: &ProjectionWeights) -> ProjectionOutputs {
    let qkv = input.clone().matmul(weights.qkv.clone().unsqueeze::<3>());
    let gate = sigmoid(input.matmul(weights.gate.clone().unsqueeze::<3>()));
    ProjectionOutputs { qkv, gate }
}

/// Candidate: one wider matmul, metadata-only split, then gate sigmoid.
fn combined_projection(input: Tensor<B, 3>, weight: &Tensor<B, 2>) -> ProjectionOutputs {
    let combined = input.matmul(weight.clone().unsqueeze::<3>());
    let qkv = combined.clone().narrow(2, 0, QKV_DIM);
    let gate = sigmoid(combined.narrow(2, QKV_DIM, MODEL_DIM));
    ProjectionOutputs { qkv, gate }
}

/// Candidate plus the contiguous QKV materialisation required by the current
/// raw SourceKernel launcher. The `[B,S,3D]` prefix view has row stride `4D`.
fn combined_projection_materialized(
    input: Tensor<B, 3>,
    weight: &Tensor<B, 2>,
) -> ProjectionOutputs {
    let combined = input.matmul(weight.clone().unsqueeze::<3>());
    let qkv_view = combined.clone().narrow(2, 0, QKV_DIM);
    let qkv = into_contiguous(qkv_view.into_primitive().tensor());
    let qkv = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(qkv));
    let gate = sigmoid(combined.narrow(2, QKV_DIM, MODEL_DIM));
    ProjectionOutputs { qkv, gate }
}

/// Current WGSL preparation path before SDPA: two projections, gate sigmoid,
/// then the existing QKV norm/half-RoPE/split SourceKernel.
fn separate_postprocess(
    input: Tensor<B, 3>,
    dummy_out: Tensor<B, 3>,
    weights: &ProjectionWeights,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
) -> PostprocessOutputs {
    let qkv = input.clone().matmul(weights.qkv.clone().unsqueeze::<3>());
    let gate = sigmoid(input.matmul(weights.gate.clone().unsqueeze::<3>()));
    let output = fused_qkv_postprocess_wgsl(
        qkv.into_primitive().tensor(),
        weights.q_norm.clone().into_primitive().tensor(),
        weights.k_norm.clone().into_primitive().tensor(),
        cos.into_primitive().tensor(),
        sin.into_primitive().tensor(),
        EPS,
    );
    let gated = gate.clone() * dummy_out;
    PostprocessOutputs {
        q: Tensor::from_primitive(TensorPrimitive::Float(output.q)),
        k: Tensor::from_primitive(TensorPrimitive::Float(output.k)),
        v: Tensor::from_primitive(TensorPrimitive::Float(output.v)),
        gate,
        gated,
    }
}

/// Candidate production path: one combined projection followed by one shader
/// that produces Q/K/V and overwrites the final segment with sigmoid(gate).
fn combined_direct_postprocess_rank3(
    input: Tensor<B, 3>,
    dummy_out: Tensor<B, 3>,
    packed_weight: &Tensor<B, 2>,
    weights: &ProjectionWeights,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
) -> PostprocessOutputs {
    let combined = input.matmul(packed_weight.clone().unsqueeze::<3>());
    direct_postprocess(combined, dummy_out, weights, cos, sin)
}

/// Production candidate using the measured winning rank-2 GEMM layout.
fn combined_direct_postprocess_flat(
    input: Tensor<B, 3>,
    dummy_out: Tensor<B, 3>,
    packed_weight: &Tensor<B, 2>,
    weights: &ProjectionWeights,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
) -> PostprocessOutputs {
    let [batch, seq_len, dim] = input.dims();
    assert_eq!(dim, MODEL_DIM);
    let combined = input
        .reshape([batch * seq_len, dim])
        .matmul(packed_weight.clone())
        .reshape([batch, seq_len, PACKED_DIM]);
    direct_postprocess(combined, dummy_out, weights, cos, sin)
}

fn direct_postprocess(
    combined: Tensor<B, 3>,
    dummy_out: Tensor<B, 3>,
    weights: &ProjectionWeights,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
) -> PostprocessOutputs {
    let output = fused_qkv_gate_postprocess_wgsl(
        combined.into_primitive().tensor(),
        weights.q_norm.clone().into_primitive().tensor(),
        weights.k_norm.clone().into_primitive().tensor(),
        cos.into_primitive().tensor(),
        sin.into_primitive().tensor(),
        EPS,
    );
    let combined = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(output.combined));
    let gate = combined.narrow(2, QKV_DIM, MODEL_DIM);
    let gated = gate.clone() * dummy_out;
    PostprocessOutputs {
        q: Tensor::from_primitive(TensorPrimitive::Float(output.qkv.q)),
        k: Tensor::from_primitive(TensorPrimitive::Float(output.qkv.k)),
        v: Tensor::from_primitive(TensorPrimitive::Float(output.qkv.v)),
        gate,
        gated,
    }
}

fn pack_weight_row(weights: &ProjectionWeights) -> Tensor<B, 2> {
    Tensor::cat(vec![weights.qkv.clone(), weights.gate.clone()], 1)
}

/// Create logical `[D,4D]` with column-major strides in one allocation.
/// Concatenation materialises row-major physical `[4D,D]`; the final
/// transpose is metadata-only.
fn pack_weight_col(weights: &ProjectionWeights) -> Tensor<B, 2> {
    let physical_weights = weights
        .qkv_parts
        .iter()
        .cloned()
        .chain(core::iter::once(weights.gate.clone()))
        .map(Tensor::transpose)
        .collect();
    Tensor::cat(physical_weights, 0).transpose()
}

fn assert_source_weight_layout(weights: &ProjectionWeights) {
    let qkv = weights.qkv.clone().into_primitive().tensor();
    assert!(qkv.is_contiguous(), "current QKV pack must be contiguous");
    assert_eq!(&qkv.meta.strides()[..], &[QKV_DIM, 1]);

    weights
        .qkv_parts
        .iter()
        .chain(core::iter::once(&weights.gate))
        .for_each(|weight| {
            assert_eq!(weight.dims(), [MODEL_DIM, MODEL_DIM]);
            let logical = weight.clone().into_primitive().tensor();
            assert!(
                !logical.is_contiguous(),
                "logical checkpoint weight must be a transposed view"
            );
            assert_eq!(&logical.meta.strides()[..], &[1, MODEL_DIM]);

            let physical = weight.clone().transpose().into_primitive().tensor();
            assert!(
                physical.is_contiguous(),
                "physical checkpoint weight must be row-major contiguous"
            );
            assert_eq!(&physical.meta.strides()[..], &[MODEL_DIM, 1]);
        });
}

fn sync_weight(weight: Tensor<B, 2>) {
    let _ = weight
        .slice([MODEL_DIM - 1..MODEL_DIM, PACKED_DIM - 1..PACKED_DIM])
        .into_data();
}

fn sync_outputs(outputs: ProjectionOutputs) {
    // Submissions are ordered. Gate is submitted after QKV in both paths, so
    // reading it waits for every operation in the measured iteration.
    let _ = outputs.gate.slice([0..1, 0..1, 0..1]).into_data();
}

fn sync_postprocess(outputs: PostprocessOutputs) {
    // Gate application is submitted after QKV post-processing in both paths.
    // Reading it synchronizes the complete compared interval.
    let _ = outputs.gated.slice([0..1, 0..1, 0..1]).into_data();
}

fn max_abs_diff(lhs: Tensor<B, 3>, rhs: Tensor<B, 3>) -> Result<f32, Box<dyn Error>> {
    let values = (lhs - rhs).abs().max().into_data().to_vec::<f32>()?;
    values
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("maximum reduction returned no values").into())
}

fn max_abs_diff_2d(lhs: Tensor<B, 2>, rhs: Tensor<B, 2>) -> Result<f32, Box<dyn Error>> {
    let values = (lhs - rhs).abs().max().into_data().to_vec::<f32>()?;
    values
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("maximum reduction returned no values").into())
}

fn max_abs_diff_4d(lhs: Tensor<B, 4>, rhs: Tensor<B, 4>) -> Result<f32, Box<dyn Error>> {
    let values = (lhs - rhs).abs().max().into_data().to_vec::<f32>()?;
    values
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("maximum reduction returned no values").into())
}

fn measure<F>(mut operation: F) -> f64
where
    F: FnMut() -> ProjectionOutputs,
{
    let warmup_output = (0..WARMUP)
        .map(|_| operation())
        .reduce(|_, output| output)
        .expect("WARMUP must be non-zero");
    sync_outputs(warmup_output);

    let started = Instant::now();
    let output = (0..ITERATIONS)
        .map(|_| operation())
        .reduce(|_, output| output)
        .expect("ITERATIONS must be non-zero");
    sync_outputs(output);
    started.elapsed().as_secs_f64() * 1_000_000.0 / ITERATIONS as f64
}

fn measure_postprocess<F>(mut operation: F) -> f64
where
    F: FnMut() -> PostprocessOutputs,
{
    let warmup_output = (0..WARMUP)
        .map(|_| operation())
        .reduce(|_, output| output)
        .expect("WARMUP must be non-zero");
    sync_postprocess(warmup_output);

    let started = Instant::now();
    let output = (0..ITERATIONS)
        .map(|_| operation())
        .reduce(|_, output| output)
        .expect("ITERATIONS must be non-zero");
    sync_postprocess(output);
    started.elapsed().as_secs_f64() * 1_000_000.0 / ITERATIONS as f64
}

fn measure_pack_once<F>(weights: &ProjectionWeights, pack: F) -> (Tensor<B, 2>, f64)
where
    F: FnOnce(&ProjectionWeights) -> Tensor<B, 2>,
{
    // Weight creation is asynchronous. The gate weight is queued after QKV,
    // so this read excludes both random initialisations from the pack timing.
    let _ = weights
        .gate
        .clone()
        .slice([MODEL_DIM - 1..MODEL_DIM, MODEL_DIM - 1..MODEL_DIM])
        .into_data();

    let started = Instant::now();
    let packed = pack(weights);
    sync_weight(packed.clone());
    (packed, started.elapsed().as_secs_f64() * 1_000_000.0)
}

fn bench_shape(
    device: &<B as Backend>::Device,
    weights: &ProjectionWeights,
    packed_weight_row: &Tensor<B, 2>,
    packed_weight_col: &Tensor<B, 2>,
    batch: usize,
) -> Result<FullPostTiming, Box<dyn Error>> {
    let input = Tensor::<B, 3>::random(
        [batch, SEQ_LEN, MODEL_DIM],
        Distribution::Uniform(-1.0, 1.0),
        device,
    );

    // Correctness calls also prime the three exact GEMM shapes before the
    // explicit ten warmups, keeping autotune/compilation outside measurement.
    let expected = separate_projection(input.clone(), weights);
    let actual = combined_projection(input.clone(), packed_weight_row);
    let materialized = combined_projection_materialized(input.clone(), packed_weight_row);
    let qkv_max_abs = max_abs_diff(expected.qkv.clone(), actual.qkv)?;
    let gate_max_abs = max_abs_diff(expected.gate.clone(), actual.gate)?;
    let materialized_qkv_max_abs = max_abs_diff(expected.qkv, materialized.qkv)?;
    let materialized_gate_max_abs = max_abs_diff(expected.gate, materialized.gate)?;
    if ![
        qkv_max_abs,
        gate_max_abs,
        materialized_qkv_max_abs,
        materialized_gate_max_abs,
    ]
    .into_iter()
    .all(f32::is_finite)
    {
        return Err(io::Error::other(format!(
            "non-finite error: qkv={qkv_max_abs}, gate={gate_max_abs}, \
             materialized_qkv={materialized_qkv_max_abs}, \
             materialized_gate={materialized_gate_max_abs}"
        ))
        .into());
    }

    let separate_us = measure(|| separate_projection(input.clone(), weights));
    let combined_us = measure(|| combined_projection(input.clone(), packed_weight_row));
    let materialized_us =
        measure(|| combined_projection_materialized(input.clone(), packed_weight_row));
    let materialized_bytes = batch * SEQ_LEN * QKV_DIM * F32_BYTES;
    println!(
        "B={batch} S={SEQ_LEN}: separate={separate_us:.1} us, combined_view={combined_us:.1} us \
         ({:.3}x), combined_materialized={materialized_us:.1} us ({:.3}x), \
         qkv_copy={materialized_bytes} bytes ({:.3} MiB), \
         view_max_abs=[qkv {qkv_max_abs:.3e}, gate {gate_max_abs:.3e}], \
         materialized_max_abs=[qkv {materialized_qkv_max_abs:.3e}, \
         gate {materialized_gate_max_abs:.3e}]",
        separate_us / combined_us,
        separate_us / materialized_us,
        mib(materialized_bytes),
    );

    let angles = Tensor::<B, 2>::random(
        [SEQ_LEN, HEAD_DIM / 2],
        Distribution::Uniform(-3.0, 3.0),
        device,
    );
    let cos = angles.clone().cos();
    let sin = angles.sin();
    let dummy_out = Tensor::<B, 3>::random(
        [batch, SEQ_LEN, MODEL_DIM],
        Distribution::Uniform(-1.0, 1.0),
        device,
    );
    let expected = separate_postprocess(
        input.clone(),
        dummy_out.clone(),
        weights,
        cos.clone(),
        sin.clone(),
    );
    let actual = combined_direct_postprocess_flat(
        input.clone(),
        dummy_out.clone(),
        packed_weight_col,
        weights,
        cos.clone(),
        sin.clone(),
    );
    let direct_errors = [
        max_abs_diff_4d(expected.q, actual.q)?,
        max_abs_diff_4d(expected.k, actual.k)?,
        max_abs_diff_4d(expected.v, actual.v)?,
        max_abs_diff(expected.gate, actual.gate)?,
        max_abs_diff(expected.gated, actual.gated)?,
    ];
    if !direct_errors.into_iter().all(f32::is_finite) {
        return Err(io::Error::other(format!(
            "non-finite direct post-process errors: {direct_errors:?}"
        ))
        .into());
    }

    let separate_post_us = measure_postprocess(|| {
        separate_postprocess(
            input.clone(),
            dummy_out.clone(),
            weights,
            cos.clone(),
            sin.clone(),
        )
    });
    let direct_rank3_us = measure_postprocess(|| {
        combined_direct_postprocess_rank3(
            input.clone(),
            dummy_out.clone(),
            packed_weight_row,
            weights,
            cos.clone(),
            sin.clone(),
        )
    });
    let direct_flat_row_us = measure_postprocess(|| {
        combined_direct_postprocess_flat(
            input.clone(),
            dummy_out.clone(),
            packed_weight_row,
            weights,
            cos.clone(),
            sin.clone(),
        )
    });
    let direct_flat_col_us = measure_postprocess(|| {
        combined_direct_postprocess_flat(
            input.clone(),
            dummy_out.clone(),
            packed_weight_col,
            weights,
            cos.clone(),
            sin.clone(),
        )
    });
    println!(
        "B={batch} S={SEQ_LEN} full-post: separate={separate_post_us:.1} us, \
         combined_direct_rank3={direct_rank3_us:.1} us ({:.3}x), \
         combined_direct_flat_row={direct_flat_row_us:.1} us ({:.3}x), \
         combined_direct_flat_col={direct_flat_col_us:.1} us ({:.3}x), \
         max_abs=[q {:.3e}, k {:.3e}, v {:.3e}, gate {:.3e}, gated {:.3e}]",
        separate_post_us / direct_rank3_us,
        separate_post_us / direct_flat_row_us,
        separate_post_us / direct_flat_col_us,
        direct_errors[0],
        direct_errors[1],
        direct_errors[2],
        direct_errors[3],
        direct_errors[4],
    );
    Ok(FullPostTiming {
        separate_us: separate_post_us,
        rank3_row_us: direct_rank3_us,
        flat_row_us: direct_flat_row_us,
        flat_col_us: direct_flat_col_us,
    })
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

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() -> Result<(), Box<dyn Error>> {
    assert_eq!(NUM_HEADS * HEAD_DIM, MODEL_DIM);
    let adapter_index = parse_adapter_index()?;
    let device = WgpuDevice::DiscreteGpu(adapter_index);
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    B::seed(&device, SEED);

    println!(
        "JointAttention QKV+gate projection benchmark device={device:?} D={MODEL_DIM} S={SEQ_LEN} \
         ({WARMUP} warmup, {ITERATIONS} measured, seed={SEED}, gate_bias=false)"
    );
    println!(
        "combined pack allocation: {COMBINED_PACK_BYTES} bytes ({:.3} MiB); \
         incremental while current QKV pack is retained: {COMBINED_PACK_BYTES} bytes ({:.3} MiB); \
         steady increment if it replaces the QKV pack: {REPLACEMENT_INCREMENT_BYTES} bytes ({:.3} MiB)",
        mib(COMBINED_PACK_BYTES),
        mib(COMBINED_PACK_BYTES),
        mib(REPLACEMENT_INCREMENT_BYTES),
    );

    let weights = projection_weights(&device);
    assert_source_weight_layout(&weights);
    let (packed_weight_row, row_pack_us) = measure_pack_once(&weights, pack_weight_row);
    let (packed_weight_col, col_pack_us) = measure_pack_once(&weights, pack_weight_col);
    let pack_max_abs = max_abs_diff_2d(packed_weight_row.clone(), packed_weight_col.clone())?;
    let row_primitive = packed_weight_row.clone().into_primitive().tensor();
    let col_primitive = packed_weight_col.clone().into_primitive().tensor();
    assert!(row_primitive.is_contiguous(), "row pack must be contiguous");
    assert!(
        !col_primitive.is_contiguous(),
        "column-major logical pack must be non-contiguous"
    );
    assert_eq!(&row_primitive.meta.strides()[..], &[PACKED_DIM, 1]);
    assert_eq!(&col_primitive.meta.strides()[..], &[1, MODEL_DIM]);
    println!(
        "one-time weight pack: row={row_pack_us:.1} us, col={col_pack_us:.1} us, \
         row/col max_abs={pack_max_abs:.3e}, steady selected-pack memory={COMBINED_PACK_BYTES} bytes \
         ({:.3} MiB)",
        mib(COMBINED_PACK_BYTES),
    );

    let timings = [1, 2]
        .into_iter()
        .map(|batch| {
            bench_shape(
                &device,
                &weights,
                &packed_weight_row,
                &packed_weight_col,
                batch,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let four_step = |select: fn(&FullPostTiming) -> f64| {
        // Pinned replay: steps 0/1 use B=2 and steps 2/3 use B=1.
        2.0 * (select(&timings[0]) + select(&timings[1]))
    };
    let separate_total = four_step(|timing| timing.separate_us);
    let rank3_total = four_step(|timing| timing.rank3_row_us);
    let flat_row_total = four_step(|timing| timing.flat_row_us);
    let flat_col_total = four_step(|timing| timing.flat_col_us);
    println!(
        "four-step weighted per-layer: separate={separate_total:.1} us, \
         rank3-row={rank3_total:.1} us ({:.3}x), flat-row={flat_row_total:.1} us ({:.3}x), \
         flat-col={flat_col_total:.1} us ({:.3}x); pinned 12-layer candidate totals: \
         row={:.1} ms, col={:.1} ms",
        separate_total / rank3_total,
        separate_total / flat_row_total,
        separate_total / flat_col_total,
        flat_row_total * 12.0 / 1_000.0,
        flat_col_total * 12.0 / 1_000.0,
    );
    Ok(())
}
