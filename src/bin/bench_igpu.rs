//! WGPU iGPU (Intel Arc) validation + micro-benchmark.
//!
//! Tests that burn's WGPU backend works on the Intel integrated GPU and measures
//! basic compute performance (matmul, elementwise).
//!
//! # Running
//!
//! ```sh
//! cargo run --release --bin bench_igpu
//! ```

use burn::backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup};
use burn::tensor::Tensor;
use irodori_tts_burn::WgpuRaw;
use std::time::Instant;

type B = WgpuRaw;

fn main() {
    println!("=== WGPU iGPU (Intel Arc) Validation & Benchmark ===\n");

    // Try to initialize on integrated GPU
    let device = WgpuDevice::IntegratedGpu(0);

    println!("Requesting IntegratedGpu(0)...");
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    println!("Device initialized successfully.\n");

    // --- Correctness test ---
    println!("--- Correctness Tests ---");

    // Test 1: Simple matmul
    let a = Tensor::<B, 2>::ones([64, 64], &device);
    let b = Tensor::<B, 2>::ones([64, 64], &device);
    let c = a.matmul(b);
    let c_data = c.into_data().to_vec::<f32>().unwrap();
    let expected = 64.0f32;
    let max_diff = c_data
        .iter()
        .map(|x| (x - expected).abs())
        .fold(0.0f32, f32::max);
    println!("matmul 64x64 ones: max_diff={max_diff:.2e} (expected all {expected})");
    assert!(
        max_diff < 1e-3,
        "matmul correctness failed: max_diff={max_diff}"
    );

    // Test 2: Elementwise ops
    let x = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
    let y = x.clone() * x.clone() + x;
    let y_data = y.into_data().to_vec::<f32>().unwrap();
    let expected_elem = [2.0, 6.0, 12.0, 20.0]; // x^2 + x
    let elem_diff: f32 = y_data
        .iter()
        .zip(expected_elem.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("elementwise (x^2+x): max_diff={elem_diff:.2e}");
    assert!(elem_diff < 1e-5, "elementwise correctness failed");

    // Test 3: Softmax
    let logits = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
    let probs = burn::tensor::activation::softmax(logits, 1);
    let probs_data = probs.into_data().to_vec::<f32>().unwrap();
    let sum: f32 = probs_data.iter().sum();
    println!("softmax sum: {sum:.6} (should be ~1.0)");
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum not 1.0");

    println!("\nAll correctness tests passed ✓\n");

    // --- Performance benchmark ---
    println!("--- Micro-Benchmarks (5 warmup, 20 iters) ---");

    let warmup = 5;
    let iters = 20;

    // Benchmark: matmul 256x256
    {
        let a = Tensor::<B, 2>::random([256, 256], burn::tensor::Distribution::Default, &device);
        let b = Tensor::<B, 2>::random([256, 256], burn::tensor::Distribution::Default, &device);
        for _ in 0..warmup {
            let _ = a.clone().matmul(b.clone());
        }
        // Sync by reading data
        let _ = a.clone().matmul(b.clone()).into_data();

        let start = Instant::now();
        for _ in 0..iters {
            let c = a.clone().matmul(b.clone());
            let _ = c.into_data();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        println!("matmul 256×256:   {per_iter:?}/iter");
    }

    // Benchmark: matmul 1024x1024
    {
        let a = Tensor::<B, 2>::random([1024, 1024], burn::tensor::Distribution::Default, &device);
        let b = Tensor::<B, 2>::random([1024, 1024], burn::tensor::Distribution::Default, &device);
        for _ in 0..warmup {
            let _ = a.clone().matmul(b.clone());
        }
        let _ = a.clone().matmul(b.clone()).into_data();

        let start = Instant::now();
        for _ in 0..iters {
            let c = a.clone().matmul(b.clone());
            let _ = c.into_data();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        println!("matmul 1024×1024: {per_iter:?}/iter");
    }

    // Benchmark: elementwise (exp + add) on large tensor
    {
        let x = Tensor::<B, 1>::random([1_000_000], burn::tensor::Distribution::Default, &device);
        for _ in 0..warmup {
            let _ = x.clone().exp();
        }
        let _ = x.clone().exp().into_data();

        let start = Instant::now();
        for _ in 0..iters {
            let y = x.clone().exp();
            let _ = y.into_data();
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed / iters as u32;
        println!("exp 1M elements:  {per_iter:?}/iter");
    }

    println!("\nDone.");
}
