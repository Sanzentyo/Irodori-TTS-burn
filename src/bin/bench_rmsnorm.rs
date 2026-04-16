//! Isolated benchmark: Custom WGSL RMSNorm vs burn's generic RMSNorm on WgpuRaw.
//!
//! Measures kernel execution time on realistic model dimensions, including
//! reshape overhead for the custom 2D kernel when called on 3D/4D model tensors.
//!
//! Run with: `cargo run --release --bin bench_rmsnorm`

use std::time::{Duration, Instant};

use burn::{
    backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup},
    tensor::{Tensor, TensorPrimitive, backend::Backend},
};

use irodori_tts_burn::{WgpuRaw, kernels::rms_norm::rms_norm_wgsl};

const WARMUP_ITERS: usize = 10;
const BENCH_ITERS: usize = 50;
const TIMEOUT_SECS: u64 = 120;

type B = WgpuRaw;

/// Sync by reading one element (forces all pending GPU work to complete).
fn gpu_sync(t: Tensor<B, 2>) {
    let _ = t.slice([0..1, 0..1]).into_data();
}

fn gpu_sync_3d(t: Tensor<B, 3>) {
    let _ = t.slice([0..1, 0..1, 0..1]).into_data();
}

// ---------------------------------------------------------------------------
// Benchmark: burn generic RmsNorm<WgpuRaw> on 3D input
// ---------------------------------------------------------------------------
fn bench_burn_rmsnorm_3d(
    device: &<B as Backend>::Device,
    batch: usize,
    seq: usize,
    dim: usize,
) -> f64 {
    let norm = irodori_tts_burn::model::norm::RmsNorm::<B>::new(dim, 1e-6, device);
    let x = Tensor::<B, 3>::ones([batch, seq, dim], device);

    // Warmup (includes JIT compile)
    for _ in 0..WARMUP_ITERS {
        let _ = norm.forward(x.clone());
    }
    gpu_sync_3d(norm.forward(x.clone()));

    // Benchmark
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let _ = norm.forward(x.clone());
    }
    gpu_sync_3d(norm.forward(x.clone()));
    let elapsed = start.elapsed();

    elapsed.as_micros() as f64 / (BENCH_ITERS + 1) as f64
}

// ---------------------------------------------------------------------------
// Benchmark: custom WGSL RmsNorm on 2D input (pre-flattened, no reshape cost)
// ---------------------------------------------------------------------------
fn bench_custom_rmsnorm_2d(device: &<B as Backend>::Device, rows: usize, dim: usize) -> f64 {
    let x = Tensor::<B, 2>::ones([rows, dim], device);
    let weight = Tensor::<B, 1>::ones([dim], device);
    let eps = 1e-6_f64;

    // Warmup (includes pipeline cache warm)
    for _ in 0..WARMUP_ITERS {
        let ip = x.clone().into_primitive().tensor();
        let wp = weight.clone().into_primitive().tensor();
        let _ = rms_norm_wgsl(ip, wp, eps);
    }
    {
        let ip = x.clone().into_primitive().tensor();
        let wp = weight.clone().into_primitive().tensor();
        let out = rms_norm_wgsl(ip, wp, eps);
        gpu_sync(Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(out)));
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let ip = x.clone().into_primitive().tensor();
        let wp = weight.clone().into_primitive().tensor();
        let _ = rms_norm_wgsl(ip, wp, eps);
    }
    {
        let ip = x.clone().into_primitive().tensor();
        let wp = weight.clone().into_primitive().tensor();
        let out = rms_norm_wgsl(ip, wp, eps);
        gpu_sync(Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(out)));
    }
    let elapsed = start.elapsed();

    elapsed.as_micros() as f64 / (BENCH_ITERS + 1) as f64
}

// ---------------------------------------------------------------------------
// Benchmark: custom WGSL RmsNorm on 3D input (includes reshape 3D→2D→3D)
// ---------------------------------------------------------------------------
fn bench_custom_rmsnorm_3d(
    device: &<B as Backend>::Device,
    batch: usize,
    seq: usize,
    dim: usize,
) -> f64 {
    let x = Tensor::<B, 3>::ones([batch, seq, dim], device);
    let weight = Tensor::<B, 1>::ones([dim], device);
    let eps = 1e-6_f64;
    let rows = batch * seq;

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let x_2d: Tensor<B, 2> = x.clone().reshape([rows, dim]);
        let ip = x_2d.into_primitive().tensor();
        let wp = weight.clone().into_primitive().tensor();
        let out = rms_norm_wgsl(ip, wp, eps);
        let _y: Tensor<B, 3> =
            Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(out)).reshape([batch, seq, dim]);
    }
    {
        let x_2d: Tensor<B, 2> = x.clone().reshape([rows, dim]);
        let ip = x_2d.into_primitive().tensor();
        let wp = weight.clone().into_primitive().tensor();
        let out = rms_norm_wgsl(ip, wp, eps);
        let y: Tensor<B, 3> =
            Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(out)).reshape([batch, seq, dim]);
        gpu_sync_3d(y);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let x_2d: Tensor<B, 2> = x.clone().reshape([rows, dim]);
        let ip = x_2d.into_primitive().tensor();
        let wp = weight.clone().into_primitive().tensor();
        let out = rms_norm_wgsl(ip, wp, eps);
        let _y: Tensor<B, 3> =
            Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(out)).reshape([batch, seq, dim]);
    }
    {
        let x_2d: Tensor<B, 2> = x.clone().reshape([rows, dim]);
        let ip = x_2d.into_primitive().tensor();
        let wp = weight.clone().into_primitive().tensor();
        let out = rms_norm_wgsl(ip, wp, eps);
        let y: Tensor<B, 3> =
            Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(out)).reshape([batch, seq, dim]);
        gpu_sync_3d(y);
    }
    let elapsed = start.elapsed();

    elapsed.as_micros() as f64 / (BENCH_ITERS + 1) as f64
}

fn main() {
    println!("=== RMSNorm Micro-Benchmark: Custom WGSL vs burn Generic (WgpuRaw) ===");
    println!("Config: warmup={WARMUP_ITERS}, bench_iters={BENCH_ITERS}, timeout={TIMEOUT_SECS}s\n");

    let device = WgpuDevice::DefaultDevice;
    init_setup::<AutoGraphicsApi>(&device, Default::default());

    let deadline = Instant::now() + Duration::from_secs(TIMEOUT_SECS);

    // Scenario 1: DiT RmsNorm (seq=750, dim=1024)
    println!("--- Scenario 1: DiT RmsNorm (batch=1, seq=750, dim=1024) ---");
    {
        let burn_us = bench_burn_rmsnorm_3d(&device, 1, 750, 1024);
        let custom_2d_us = bench_custom_rmsnorm_2d(&device, 750, 1024);
        let custom_3d_us = bench_custom_rmsnorm_3d(&device, 1, 750, 1024);
        println!("  burn generic (3D):        {:>10.1} µs", burn_us);
        println!("  custom WGSL (2D, no reshape): {:>6.1} µs", custom_2d_us);
        println!("  custom WGSL (3D→2D→3D):  {:>10.1} µs", custom_3d_us);
        println!(
            "  speedup (vs burn, with reshape): {:>5.2}×",
            burn_us / custom_3d_us
        );
        println!();
    }

    if Instant::now() > deadline {
        println!("TIMEOUT");
        std::process::exit(0);
    }

    // Scenario 2: TextEncoder RmsNorm (seq=200, dim=1024)
    println!("--- Scenario 2: TextEncoder RmsNorm (batch=1, seq=200, dim=1024) ---");
    {
        let burn_us = bench_burn_rmsnorm_3d(&device, 1, 200, 1024);
        let custom_3d_us = bench_custom_rmsnorm_3d(&device, 1, 200, 1024);
        println!("  burn generic (3D):        {:>10.1} µs", burn_us);
        println!("  custom WGSL (3D→2D→3D):  {:>10.1} µs", custom_3d_us);
        println!("  speedup: {:>5.2}×", burn_us / custom_3d_us);
        println!();
    }

    if Instant::now() > deadline {
        println!("TIMEOUT");
        std::process::exit(0);
    }

    // Scenario 3: HeadRmsNorm equivalent (flatten [B,S,H,Dh] → [B*S*H, Dh])
    println!("--- Scenario 3: HeadRmsNorm equiv (batch=1, seq=750, heads=16, dim=64) ---");
    {
        let burn_us = bench_burn_rmsnorm_3d(&device, 1, 750 * 16, 64);
        let custom_3d_us = bench_custom_rmsnorm_3d(&device, 1, 750 * 16, 64);
        println!("  burn generic (3D):        {:>10.1} µs", burn_us);
        println!("  custom WGSL (3D→2D→3D):  {:>10.1} µs", custom_3d_us);
        println!("  speedup: {:>5.2}×", burn_us / custom_3d_us);
        println!();
    }

    if Instant::now() > deadline {
        println!("TIMEOUT");
        std::process::exit(0);
    }

    // Scenario 4: Full model RmsNorm call count estimate
    println!("--- Impact Estimate ---");
    println!("  ~30 RmsNorm calls per forward pass × 40 sampling steps = 1,200 kernel launches");
    println!("  If custom kernel saves 50µs per call → 60ms per inference (~1% of 6,720ms total)");
    println!(
        "  If custom kernel saves 500µs per call → 600ms per inference (~9% of 6,720ms total)"
    );

    println!("\nDone.");
    std::process::exit(0);
}
