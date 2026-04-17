//! Isolated benchmark: Fused AdaLN WGSL kernel vs burn's generic ops on WgpuRaw.
//!
//! Measures the fused RMSNorm+modulate kernel against the equivalent sequence
//! of generic tensor operations (powf, mean, sqrt, div, mul, add) that burn
//! would execute without fusion.
//!
//! Run with: `cargo run --release --bin bench_fused_adaln`

use std::time::Instant;

use burn::{
    backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup},
    tensor::{Tensor, TensorPrimitive, backend::Backend},
};

use irodori_tts_burn::{WgpuRaw, kernels::fused_adaln::fused_adaln_wgsl};

const WARMUP_ITERS: usize = 10;
const BENCH_ITERS: usize = 50;
const TIMEOUT_SECS: u64 = 120;

type B = WgpuRaw;

fn gpu_sync_2d(t: Tensor<B, 2>) {
    let _ = t.slice([0..1, 0..1]).into_data();
}

fn gpu_sync_3d(t: Tensor<B, 3>) {
    let _ = t.slice([0..1, 0..1, 0..1]).into_data();
}

/// Burn's generic RMSNorm + modulate (reproduces LowRankAdaLn lines 159-169).
///
/// Input: x [B, S, D], scale [B, 1, D], shift [B, 1, D]
/// Output: (x / rms) * (1 + scale) + shift  [B, S, D]
fn burn_adaln_generic(
    x: Tensor<B, 3>,
    scale: Tensor<B, 3>,
    shift: Tensor<B, 3>,
    eps: f64,
) -> Tensor<B, 3> {
    let rms = x
        .clone()
        .powf_scalar(2.0)
        .mean_dim(2)
        .add_scalar(eps)
        .sqrt();
    let x_norm = x / rms;
    x_norm * (scale + 1.0) + shift
}

/// Benchmark: burn generic AdaLN (separate ops).
fn bench_burn_adaln(device: &<B as Backend>::Device, batch: usize, seq: usize, dim: usize) -> f64 {
    let x = Tensor::<B, 3>::ones([batch, seq, dim], device);
    let scale = Tensor::<B, 3>::zeros([batch, 1, dim], device);
    let shift = Tensor::<B, 3>::zeros([batch, 1, dim], device);
    let eps = 1e-6_f64;

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let _ = burn_adaln_generic(x.clone(), scale.clone(), shift.clone(), eps);
    }
    gpu_sync_3d(burn_adaln_generic(
        x.clone(),
        scale.clone(),
        shift.clone(),
        eps,
    ));

    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let _ = burn_adaln_generic(x.clone(), scale.clone(), shift.clone(), eps);
    }
    gpu_sync_3d(burn_adaln_generic(
        x.clone(),
        scale.clone(),
        shift.clone(),
        eps,
    ));
    let elapsed = start.elapsed();

    elapsed.as_micros() as f64 / (BENCH_ITERS + 1) as f64
}

/// Benchmark: custom fused AdaLN WGSL kernel.
fn bench_custom_adaln(
    device: &<B as Backend>::Device,
    batch: usize,
    seq: usize,
    dim: usize,
) -> f64 {
    let x = Tensor::<B, 3>::ones([batch, seq, dim], device);
    let scale_2d = Tensor::<B, 2>::zeros([batch, dim], device);
    let shift_2d = Tensor::<B, 2>::zeros([batch, dim], device);
    let eps = 1e-6_f64;
    let rows = batch * seq;

    // Warmup (includes pipeline compilation)
    for _ in 0..WARMUP_ITERS {
        let x_2d: Tensor<B, 2> = x.clone().reshape([rows, dim]);
        let _ = fused_adaln_wgsl(
            x_2d.into_primitive().tensor(),
            scale_2d.clone().into_primitive().tensor(),
            shift_2d.clone().into_primitive().tensor(),
            batch,
            seq,
            eps,
        );
    }
    {
        let x_2d: Tensor<B, 2> = x.clone().reshape([rows, dim]);
        let out = fused_adaln_wgsl(
            x_2d.into_primitive().tensor(),
            scale_2d.clone().into_primitive().tensor(),
            shift_2d.clone().into_primitive().tensor(),
            batch,
            seq,
            eps,
        );
        gpu_sync_2d(Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(out)));
    }

    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let x_2d: Tensor<B, 2> = x.clone().reshape([rows, dim]);
        let _ = fused_adaln_wgsl(
            x_2d.into_primitive().tensor(),
            scale_2d.clone().into_primitive().tensor(),
            shift_2d.clone().into_primitive().tensor(),
            batch,
            seq,
            eps,
        );
    }
    {
        let x_2d: Tensor<B, 2> = x.clone().reshape([rows, dim]);
        let out = fused_adaln_wgsl(
            x_2d.into_primitive().tensor(),
            scale_2d.clone().into_primitive().tensor(),
            shift_2d.clone().into_primitive().tensor(),
            batch,
            seq,
            eps,
        );
        gpu_sync_2d(Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(out)));
    }
    let elapsed = start.elapsed();

    elapsed.as_micros() as f64 / (BENCH_ITERS + 1) as f64
}

fn main() {
    println!("=== Fused AdaLN Micro-Benchmark: Custom WGSL vs burn Generic (WgpuRaw) ===");
    println!("Config: warmup={WARMUP_ITERS}, bench_iters={BENCH_ITERS}, timeout={TIMEOUT_SECS}s\n");

    let deadline = Instant::now() + std::time::Duration::from_secs(TIMEOUT_SECS);

    let device = WgpuDevice::DefaultDevice;
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    println!("Device initialised.\n");

    let scenarios: &[(&str, usize, usize, usize)] = &[
        ("DiT AdaLN (1×750×1024)", 1, 750, 1024),
        ("DiT AdaLN short (1×100×1024)", 1, 100, 1024),
        ("DiT AdaLN batch2 (2×750×1024)", 2, 750, 1024),
        ("Small dim (1×750×256)", 1, 750, 256),
    ];

    for (name, batch, seq, dim) in scenarios {
        if Instant::now() > deadline {
            println!("⏱ Timeout reached, stopping.");
            break;
        }

        println!("--- {name} ---");

        let burn_us = bench_burn_adaln(&device, *batch, *seq, *dim);
        let custom_us = bench_custom_adaln(&device, *batch, *seq, *dim);
        let speedup = burn_us / custom_us;

        println!("  burn generic : {burn_us:>10.1} µs/iter");
        println!("  custom fused : {custom_us:>10.1} µs/iter");
        println!("  speedup      : {speedup:>10.2}×");
        println!();

        // Impact estimate: 24 AdaLN calls per forward × 40 steps = 960 per inference
        let total_savings_ms = (burn_us - custom_us) * 960.0 / 1000.0;
        println!(
            "  estimated 40-step savings: {total_savings_ms:>+.1} ms (960 calls × {:.1} µs delta)",
            burn_us - custom_us
        );
        println!();
    }

    println!("Done.");
}
