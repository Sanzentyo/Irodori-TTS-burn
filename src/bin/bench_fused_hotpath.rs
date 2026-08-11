//! Correctness and latency measurements for the production WGSL fusions.
//!
//! Run with `cargo run --release --bin bench_fused_hotpath -- <wgpu-adapter-index>`.

use std::{error::Error, io, time::Instant};

use burn::{
    backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup},
    tensor::{Tensor, TensorPrimitive, activation::silu, backend::Backend},
};
use irodori_tts_wgpu::{
    WgpuRaw,
    kernels::{
        fused_adaln::fused_adaln_wgsl, fused_residual_gate::fused_residual_gate_wgsl,
        fused_swiglu::fused_swiglu_wgsl, qkv_postprocess::fused_qkv_postprocess_wgsl,
        rms_norm::rms_norm_wgsl, snake::snake_wgsl,
    },
};

type B = WgpuRaw;

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;
const EPS: f64 = 1.0e-6;

fn sync_2d(tensor: Tensor<B, 2>) {
    let _ = tensor.slice([0..1, 0..1]).into_data();
}

fn sync_3d(tensor: Tensor<B, 3>) {
    let _ = tensor.slice([0..1, 0..1, 0..1]).into_data();
}

fn sync_qkv([_q, _k, v]: [Tensor<B, 4>; 3]) {
    // WGPU submissions are ordered. Reading the final output waits for all
    // preceding Q/K work as well as V.
    let _ = v.slice([0..1, 0..1, 0..1, 0..1]).into_data();
}

fn max_abs_diff(lhs: Tensor<B, 2>, rhs: Tensor<B, 2>) -> Result<f32, Box<dyn Error>> {
    let values = (lhs - rhs).abs().max().into_data().to_vec::<f32>()?;
    values
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("maximum reduction returned no values").into())
}

fn check_error(name: &str, error: f32, tolerance: f32) -> Result<(), Box<dyn Error>> {
    if error.is_finite() && error <= tolerance {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "{name} max_abs={error:.3e} exceeds tolerance {tolerance:.3e}"
        ))
        .into())
    }
}

fn measure_2d<F>(mut operation: F) -> f64
where
    F: FnMut() -> Tensor<B, 2>,
{
    for _ in 0..WARMUP {
        let _ = operation();
    }
    sync_2d(operation());

    let started = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = operation();
    }
    sync_2d(operation());
    started.elapsed().as_secs_f64() * 1_000_000.0 / (ITERATIONS + 1) as f64
}

fn measure_3d<F>(mut operation: F) -> f64
where
    F: FnMut() -> Tensor<B, 3>,
{
    for _ in 0..WARMUP {
        let _ = operation();
    }
    sync_3d(operation());

    let started = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = operation();
    }
    sync_3d(operation());
    started.elapsed().as_secs_f64() * 1_000_000.0 / (ITERATIONS + 1) as f64
}

fn measure_qkv<F>(mut operation: F) -> f64
where
    F: FnMut() -> [Tensor<B, 4>; 3],
{
    for _ in 0..WARMUP {
        let _ = operation();
    }
    sync_qkv(operation());

    let started = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = operation();
    }
    sync_qkv(operation());
    started.elapsed().as_secs_f64() * 1_000_000.0 / (ITERATIONS + 1) as f64
}

fn generic_swiglu(input: Tensor<B, 2>, hidden: usize) -> Tensor<B, 2> {
    let gate = silu(input.clone().narrow(1, 0, hidden));
    let value = input.narrow(1, hidden, hidden);
    gate * value
}

fn custom_swiglu(input: Tensor<B, 2>) -> Tensor<B, 2> {
    let output = fused_swiglu_wgsl(input.into_primitive().tensor());
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn bench_swiglu(
    device: &<B as Backend>::Device,
    batch: usize,
    seq_len: usize,
    hidden: usize,
) -> Result<(), Box<dyn Error>> {
    let rows = batch * seq_len;
    let input = Tensor::<B, 2>::random(
        [rows, hidden * 2],
        burn::tensor::Distribution::Uniform(-3.0, 3.0),
        device,
    );
    let expected = generic_swiglu(input.clone(), hidden);
    let actual = custom_swiglu(input.clone());
    let error = max_abs_diff(expected, actual)?;
    check_error("SwiGLU", error, 2.0e-5)?;

    let generic_us = measure_2d(|| generic_swiglu(input.clone(), hidden));
    let fused_us = measure_2d(|| custom_swiglu(input.clone()));
    println!(
        "SwiGLU  B={batch} S={seq_len:3} H={hidden:4}: generic={generic_us:8.1} us, \
         fused={fused_us:8.1} us, speedup={:5.2}x, max_abs={error:.3e}",
        generic_us / fused_us,
    );
    Ok(())
}

fn generic_residual(
    residual: Tensor<B, 3>,
    branch: Tensor<B, 3>,
    gate: Tensor<B, 3>,
) -> Tensor<B, 3> {
    residual + gate * branch
}

fn custom_residual(
    residual: Tensor<B, 3>,
    branch: Tensor<B, 3>,
    gate: Tensor<B, 3>,
    batch: usize,
    seq_len: usize,
    dim: usize,
) -> Tensor<B, 3> {
    let rows = batch * seq_len;
    let output = fused_residual_gate_wgsl(
        residual.reshape([rows, dim]).into_primitive().tensor(),
        branch.reshape([rows, dim]).into_primitive().tensor(),
        gate.reshape([batch, dim]).into_primitive().tensor(),
        batch,
        seq_len,
    );
    Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(output)).reshape([batch, seq_len, dim])
}

fn bench_residual(
    device: &<B as Backend>::Device,
    batch: usize,
    seq_len: usize,
    dim: usize,
) -> Result<(), Box<dyn Error>> {
    let residual = Tensor::<B, 3>::random(
        [batch, seq_len, dim],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        device,
    );
    let branch = Tensor::<B, 3>::random(
        [batch, seq_len, dim],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        device,
    );
    let gate = Tensor::<B, 3>::random(
        [batch, 1, dim],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        device,
    );
    let expected = generic_residual(residual.clone(), branch.clone(), gate.clone());
    let actual = custom_residual(
        residual.clone(),
        branch.clone(),
        gate.clone(),
        batch,
        seq_len,
        dim,
    );
    let error = max_abs_diff(
        expected.reshape([batch * seq_len, dim]),
        actual.reshape([batch * seq_len, dim]),
    )?;
    check_error("gated residual", error, 1.0e-6)?;

    let generic_us =
        measure_3d(|| generic_residual(residual.clone(), branch.clone(), gate.clone()));
    let fused_us = measure_3d(|| {
        custom_residual(
            residual.clone(),
            branch.clone(),
            gate.clone(),
            batch,
            seq_len,
            dim,
        )
    });
    println!(
        "Residual B={batch} S={seq_len:3} D={dim:4}: generic={generic_us:8.1} us, \
         fused={fused_us:8.1} us, speedup={:5.2}x, max_abs={error:.3e}",
        generic_us / fused_us,
    );
    Ok(())
}

fn generic_adaln(input: Tensor<B, 3>, scale: Tensor<B, 3>, shift: Tensor<B, 3>) -> Tensor<B, 3> {
    let rms = input
        .clone()
        .powf_scalar(2.0)
        .mean_dim(2)
        .add_scalar(EPS)
        .sqrt();
    input / rms * (scale + 1.0) + shift
}

fn custom_adaln(
    input: Tensor<B, 3>,
    scale: Tensor<B, 3>,
    shift: Tensor<B, 3>,
    batch: usize,
    seq_len: usize,
    dim: usize,
) -> Tensor<B, 3> {
    let output = fused_adaln_wgsl(
        input
            .reshape([batch * seq_len, dim])
            .into_primitive()
            .tensor(),
        scale.reshape([batch, dim]).into_primitive().tensor(),
        shift.reshape([batch, dim]).into_primitive().tensor(),
        batch,
        seq_len,
        EPS,
    );
    Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(output)).reshape([batch, seq_len, dim])
}

fn bench_adaln(
    device: &<B as Backend>::Device,
    batch: usize,
    seq_len: usize,
    dim: usize,
) -> Result<(), Box<dyn Error>> {
    let input = Tensor::<B, 3>::random(
        [batch, seq_len, dim],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        device,
    );
    let scale = Tensor::<B, 3>::random(
        [batch, 1, dim],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        device,
    );
    let shift = Tensor::<B, 3>::random(
        [batch, 1, dim],
        burn::tensor::Distribution::Uniform(-0.5, 0.5),
        device,
    );
    let expected = generic_adaln(input.clone(), scale.clone(), shift.clone());
    let actual = custom_adaln(
        input.clone(),
        scale.clone(),
        shift.clone(),
        batch,
        seq_len,
        dim,
    );
    let error = max_abs_diff(
        expected.reshape([batch * seq_len, dim]),
        actual.reshape([batch * seq_len, dim]),
    )?;
    check_error("AdaLN", error, 1.0e-3)?;

    let generic_us = measure_3d(|| generic_adaln(input.clone(), scale.clone(), shift.clone()));
    let fused_us = measure_3d(|| {
        custom_adaln(
            input.clone(),
            scale.clone(),
            shift.clone(),
            batch,
            seq_len,
            dim,
        )
    });
    println!(
        "AdaLN    B={batch} S={seq_len:3} D={dim:4}: generic={generic_us:8.1} us, \
         fused={fused_us:8.1} us, speedup={:5.2}x, max_abs={error:.3e}",
        generic_us / fused_us,
    );
    Ok(())
}

fn generic_rms_norm(input: Tensor<B, 3>, weight: Tensor<B, 1>) -> Tensor<B, 3> {
    let rms = input
        .clone()
        .powf_scalar(2.0)
        .mean_dim(2)
        .add_scalar(EPS)
        .sqrt();
    let weight = weight.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
    input / rms * weight
}

fn custom_rms_norm(
    input: Tensor<B, 3>,
    weight: Tensor<B, 1>,
    batch: usize,
    seq_len: usize,
    dim: usize,
) -> Tensor<B, 3> {
    let output = rms_norm_wgsl(
        input
            .reshape([batch * seq_len, dim])
            .into_primitive()
            .tensor(),
        weight.into_primitive().tensor(),
        EPS,
    );
    Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(output)).reshape([batch, seq_len, dim])
}

fn bench_rms_norm(
    device: &<B as Backend>::Device,
    batch: usize,
    seq_len: usize,
    dim: usize,
) -> Result<(), Box<dyn Error>> {
    let input = Tensor::<B, 3>::random(
        [batch, seq_len, dim],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        device,
    );
    let weight =
        Tensor::<B, 1>::random([dim], burn::tensor::Distribution::Uniform(0.5, 1.5), device);
    let expected = generic_rms_norm(input.clone(), weight.clone());
    let actual = custom_rms_norm(input.clone(), weight.clone(), batch, seq_len, dim);
    let error = max_abs_diff(
        expected.reshape([batch * seq_len, dim]),
        actual.reshape([batch * seq_len, dim]),
    )?;
    check_error("RMSNorm", error, 1.0e-3)?;

    let generic_us = measure_3d(|| generic_rms_norm(input.clone(), weight.clone()));
    let fused_us =
        measure_3d(|| custom_rms_norm(input.clone(), weight.clone(), batch, seq_len, dim));
    println!(
        "RMSNorm  B={batch} S={seq_len:3} D={dim:4}: generic={generic_us:8.1} us, \
         fused={fused_us:8.1} us, speedup={:5.2}x, max_abs={error:.3e}",
        generic_us / fused_us,
    );
    Ok(())
}

fn generic_snake(input: Tensor<B, 3>, alpha: Tensor<B, 3>) -> Tensor<B, 3> {
    let sine = (input.clone() * alpha.clone()).sin();
    input + sine.powi_scalar(2) / (alpha + 1.0e-9_f32)
}

fn custom_snake(input: Tensor<B, 3>, alpha: Tensor<B, 3>) -> Tensor<B, 3> {
    let output = snake_wgsl(
        input.into_primitive().tensor(),
        alpha.into_primitive().tensor(),
    );
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn bench_snake(
    device: &<B as Backend>::Device,
    batch: usize,
    channels: usize,
    time: usize,
) -> Result<(), Box<dyn Error>> {
    let input = Tensor::<B, 3>::random(
        [batch, channels, time],
        burn::tensor::Distribution::Uniform(-3.0, 3.0),
        device,
    );
    let alpha = Tensor::<B, 3>::random(
        [1, channels, 1],
        burn::tensor::Distribution::Uniform(0.5, 2.0),
        device,
    );
    let expected = generic_snake(input.clone(), alpha.clone());
    let actual = custom_snake(input.clone(), alpha.clone());
    let error = max_abs_diff(
        expected.reshape([batch * channels, time]),
        actual.reshape([batch * channels, time]),
    )?;
    check_error("Snake1d", error, 2.0e-6)?;

    let generic_us = measure_3d(|| generic_snake(input.clone(), alpha.clone()));
    let fused_us = measure_3d(|| custom_snake(input.clone(), alpha.clone()));
    println!(
        "Snake1d  B={batch} C={channels:3} T={time:5}: generic={generic_us:8.1} us, \
         fused={fused_us:8.1} us, speedup={:5.2}x, max_abs={error:.3e}",
        generic_us / fused_us,
    );
    Ok(())
}

fn generic_head_rms(input: Tensor<B, 4>, weight: Tensor<B, 2>) -> Tensor<B, 4> {
    let rms = input
        .clone()
        .powf_scalar(2.0)
        .mean_dim(3)
        .add_scalar(EPS)
        .sqrt();
    input / rms * weight.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0)
}

fn generic_rotary_half(input: Tensor<B, 4>, cos: Tensor<B, 2>, sin: Tensor<B, 2>) -> Tensor<B, 4> {
    let [batch, seq_len, heads, head_dim] = input.dims();
    let half_heads = heads / 2;
    let half_dim = head_dim / 2;
    let rotating = input
        .clone()
        .narrow(2, 0, half_heads)
        .reshape([batch, seq_len, half_heads, half_dim, 2]);
    let real = rotating
        .clone()
        .slice([0..batch, 0..seq_len, 0..half_heads, 0..half_dim, 0..1])
        .reshape([batch, seq_len, half_heads, half_dim]);
    let imaginary = rotating
        .slice([0..batch, 0..seq_len, 0..half_heads, 0..half_dim, 1..2])
        .reshape([batch, seq_len, half_heads, half_dim]);
    let cos = cos.reshape([1, seq_len, 1, half_dim]);
    let sin = sin.reshape([1, seq_len, 1, half_dim]);
    let rotated_real = real.clone() * cos.clone() - imaginary.clone() * sin.clone();
    let rotated_imaginary = real * sin + imaginary * cos;
    let rotated = Tensor::<B, 4>::stack::<5>(vec![rotated_real, rotated_imaginary], 4)
        .reshape([batch, seq_len, half_heads, head_dim]);
    let passthrough = input.narrow(2, half_heads, heads - half_heads);
    Tensor::cat(vec![rotated, passthrough], 2)
}

fn generic_qkv_postprocess(
    fused: Tensor<B, 3>,
    q_weight: Tensor<B, 2>,
    k_weight: Tensor<B, 2>,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
) -> [Tensor<B, 4>; 3] {
    let [batch, seq_len, width] = fused.dims();
    let [heads, head_dim] = q_weight.dims();
    let kv_dim = width / 3;
    let q = fused
        .clone()
        .narrow(2, 0, kv_dim)
        .reshape([batch, seq_len, heads, head_dim]);
    let k = fused
        .clone()
        .narrow(2, kv_dim, kv_dim)
        .reshape([batch, seq_len, heads, head_dim]);
    let v = fused
        .narrow(2, 2 * kv_dim, kv_dim)
        .reshape([batch, seq_len, heads, head_dim]);
    let q = generic_rotary_half(generic_head_rms(q, q_weight), cos.clone(), sin.clone());
    let k = generic_rotary_half(generic_head_rms(k, k_weight), cos, sin);
    [q, k, v]
}

fn custom_qkv_postprocess(
    fused: Tensor<B, 3>,
    q_weight: Tensor<B, 2>,
    k_weight: Tensor<B, 2>,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
) -> [Tensor<B, 4>; 3] {
    let output = fused_qkv_postprocess_wgsl(
        fused.into_primitive().tensor(),
        q_weight.into_primitive().tensor(),
        k_weight.into_primitive().tensor(),
        cos.into_primitive().tensor(),
        sin.into_primitive().tensor(),
        EPS,
    );
    [output.q, output.k, output.v]
        .map(|tensor| Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(tensor)))
}

fn bench_qkv_postprocess(
    device: &<B as Backend>::Device,
    batch: usize,
    seq_len: usize,
    heads: usize,
    head_dim: usize,
) -> Result<(), Box<dyn Error>> {
    let kv_dim = heads * head_dim;
    let fused = Tensor::<B, 3>::random(
        [batch, seq_len, 3 * kv_dim],
        burn::tensor::Distribution::Uniform(-2.0, 2.0),
        device,
    );
    let q_weight = Tensor::<B, 2>::random(
        [heads, head_dim],
        burn::tensor::Distribution::Uniform(0.5, 1.5),
        device,
    );
    let k_weight = Tensor::<B, 2>::random(
        [heads, head_dim],
        burn::tensor::Distribution::Uniform(0.5, 1.5),
        device,
    );
    let angles = Tensor::<B, 2>::random(
        [seq_len, head_dim / 2],
        burn::tensor::Distribution::Uniform(-3.0, 3.0),
        device,
    );
    let cos = angles.clone().cos();
    let sin = angles.sin();

    let expected = generic_qkv_postprocess(
        fused.clone(),
        q_weight.clone(),
        k_weight.clone(),
        cos.clone(),
        sin.clone(),
    );
    let actual = custom_qkv_postprocess(
        fused.clone(),
        q_weight.clone(),
        k_weight.clone(),
        cos.clone(),
        sin.clone(),
    );
    let mut max_abs = 0.0_f32;
    for (name, expected, actual) in ["Q", "K", "V"]
        .into_iter()
        .zip(expected)
        .zip(actual)
        .map(|((name, expected), actual)| (name, expected, actual))
    {
        let error = max_abs_diff(
            expected.reshape([batch * seq_len * heads, head_dim]),
            actual.reshape([batch * seq_len * heads, head_dim]),
        )?;
        max_abs = max_abs.max(error);
        check_error(&format!("QKV {name}"), error, 5.0e-5)?;
    }

    let generic_us = measure_qkv(|| {
        generic_qkv_postprocess(
            fused.clone(),
            q_weight.clone(),
            k_weight.clone(),
            cos.clone(),
            sin.clone(),
        )
    });
    let fused_us = measure_qkv(|| {
        custom_qkv_postprocess(
            fused.clone(),
            q_weight.clone(),
            k_weight.clone(),
            cos.clone(),
            sin.clone(),
        )
    });
    println!(
        "QKV post B={batch} S={seq_len:3} H={heads:2} Dh={head_dim:2}: \
         generic={generic_us:8.1} us, fused={fused_us:8.1} us, speedup={:5.2}x, \
         max_abs={max_abs:.3e}",
        generic_us / fused_us,
    );
    Ok(())
}

fn bench_production_shape(
    device: &<B as Backend>::Device,
    batch: usize,
    seq_len: usize,
    dim: usize,
    hidden: usize,
) -> Result<(), Box<dyn Error>> {
    println!("-- production B={batch} S={seq_len} D={dim} H={hidden} --");
    bench_swiglu(device, batch, seq_len, hidden)?;
    bench_residual(device, batch, seq_len, dim)?;
    bench_adaln(device, batch, seq_len, dim)?;
    bench_rms_norm(device, batch, seq_len, dim)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let adapter_index = std::env::args()
        .nth(1)
        .map(|value| value.parse::<usize>())
        .transpose()?;
    let device = adapter_index
        .map(WgpuDevice::DiscreteGpu)
        .unwrap_or(WgpuDevice::DefaultDevice);
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    B::seed(&device, 0);

    println!(
        "WGPU fused hot-path benchmark device={device:?} ({WARMUP} warmup, {ITERATIONS} measured)"
    );

    // Exact latent-token shapes exercised by the pinned v4 E2E replay.
    // Text-only independent CFG expands the model batch from one to two.
    for batch in [1, 2] {
        bench_production_shape(&device, batch, 50, 1_280, 3_680)?;
        bench_qkv_postprocess(&device, batch, 50, 20, 64)?;
    }

    // Retain the longer-shape measurements used while selecting the SwiGLU
    // and gated-residual policies, including a three-context CFG batch of four.
    println!("-- extended S=256 hot paths --");
    for batch in [1, 4] {
        bench_swiglu(&device, batch, 256, 3_680)?;
        bench_residual(&device, batch, 256, 1_280)?;
    }
    println!("-- codec decode hot path --");
    bench_snake(&device, 1, 96, 96_000)?;
    Ok(())
}
