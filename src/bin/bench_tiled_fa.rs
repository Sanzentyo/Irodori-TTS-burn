//! Isolated benchmark: Tiled FlashAttention WGSL kernel vs burn's generic SDPA on WgpuRaw.
//!
//! Compares three implementations:
//! 1. burn's generic multi-step SDPA (Q@K^T → softmax → @V)
//! 2. Tiled FA (16×8) — fewer Q tiles, less K/V traffic
//! 3. Tiled FA (8×16) — more Q tiles, fewer KV blocks
//!
//! Run with: `cargo run --release --bin bench_tiled_fa`

use std::time::Instant;

use burn::{
    backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup},
    tensor::{Tensor, TensorPrimitive, backend::Backend},
};

use irodori_tts_burn::{
    WgpuRaw,
    kernels::fused_sdpa_tiled::{TiledFaConfig, tiled_fa_sdpa_wgsl},
};

const WARMUP_ITERS: usize = 10;
const BENCH_ITERS: usize = 50;
const TIMEOUT_SECS: u64 = 300;

type B = WgpuRaw;

fn gpu_sync_4d(t: Tensor<B, 4>) {
    let _ = t.slice([0..1, 0..1, 0..1, 0..1]).into_data();
}

/// burn's generic SDPA: Q@K^T * scale + mask → softmax → @V
fn burn_sdpa(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Tensor<B, 2>,
    scale: f64,
) -> Tensor<B, 4> {
    let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale);
    let m = mask.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(1);
    let neg_inf = Tensor::<B, 4>::full(scores.shape(), f32::NEG_INFINITY, &scores.device());
    let scores = scores.mask_where(m.lower_equal_elem(0.5), neg_inf);
    let attn = burn::tensor::activation::softmax(scores, 3);
    attn.matmul(v)
}

fn bench_burn(
    device: &<B as Backend>::Device,
    batch: usize,
    heads: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
) -> f64 {
    let q = Tensor::<B, 4>::ones([batch, heads, seq_q, head_dim], device);
    let k = Tensor::<B, 4>::ones([batch, heads, seq_kv, head_dim], device);
    let v = Tensor::<B, 4>::ones([batch, heads, seq_kv, head_dim], device);
    let mask = Tensor::<B, 2>::ones([batch, seq_kv], device);
    let scale = (head_dim as f64).powf(-0.5);

    for _ in 0..WARMUP_ITERS {
        let _ = burn_sdpa(q.clone(), k.clone(), v.clone(), mask.clone(), scale);
    }
    gpu_sync_4d(burn_sdpa(
        q.clone(),
        k.clone(),
        v.clone(),
        mask.clone(),
        scale,
    ));

    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let _ = burn_sdpa(q.clone(), k.clone(), v.clone(), mask.clone(), scale);
    }
    gpu_sync_4d(burn_sdpa(
        q.clone(),
        k.clone(),
        v.clone(),
        mask.clone(),
        scale,
    ));
    start.elapsed().as_micros() as f64 / (BENCH_ITERS + 1) as f64
}

fn bench_tiled_fa(
    device: &<B as Backend>::Device,
    batch: usize,
    heads: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    config: &TiledFaConfig,
) -> f64 {
    let q = Tensor::<B, 4>::ones([batch, heads, seq_q, head_dim], device);
    let k = Tensor::<B, 4>::ones([batch, heads, seq_kv, head_dim], device);
    let v = Tensor::<B, 4>::ones([batch, heads, seq_kv, head_dim], device);
    let mask = Tensor::<B, 2>::ones([batch, seq_kv], device);
    let scale = (head_dim as f64).powf(-0.5);

    for _ in 0..WARMUP_ITERS {
        let _ = tiled_fa_sdpa_wgsl(
            q.clone().into_primitive().tensor(),
            k.clone().into_primitive().tensor(),
            v.clone().into_primitive().tensor(),
            mask.clone().into_primitive().tensor(),
            scale,
            config,
        );
    }
    {
        let out = tiled_fa_sdpa_wgsl(
            q.clone().into_primitive().tensor(),
            k.clone().into_primitive().tensor(),
            v.clone().into_primitive().tensor(),
            mask.clone().into_primitive().tensor(),
            scale,
            config,
        );
        gpu_sync_4d(Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(out)));
    }

    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let _ = tiled_fa_sdpa_wgsl(
            q.clone().into_primitive().tensor(),
            k.clone().into_primitive().tensor(),
            v.clone().into_primitive().tensor(),
            mask.clone().into_primitive().tensor(),
            scale,
            config,
        );
    }
    {
        let out = tiled_fa_sdpa_wgsl(
            q.clone().into_primitive().tensor(),
            k.clone().into_primitive().tensor(),
            v.clone().into_primitive().tensor(),
            mask.clone().into_primitive().tensor(),
            scale,
            config,
        );
        gpu_sync_4d(Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(out)));
    }
    start.elapsed().as_micros() as f64 / (BENCH_ITERS + 1) as f64
}

fn main() {
    println!("=== Tiled FlashAttention Micro-Benchmark: Custom WGSL vs burn Generic ===");
    println!("Config: warmup={WARMUP_ITERS}, bench_iters={BENCH_ITERS}, timeout={TIMEOUT_SECS}s\n");

    let deadline = Instant::now() + std::time::Duration::from_secs(TIMEOUT_SECS);

    let device = WgpuDevice::DefaultDevice;
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    println!("Device initialised.\n");

    // Tiled FA requires tile_q × tile_kv == head_dim, so only D=128 scenarios
    let scenarios: &[(&str, usize, usize, usize, usize, usize)] = &[
        ("DiT joint attn (1×16×750×850×128)", 1, 16, 750, 850, 128),
        ("Short seq (1×16×100×150×128)", 1, 16, 100, 150, 128),
        ("Square (1×16×256×256×128)", 1, 16, 256, 256, 128),
        ("Large seq (1×16×1024×1200×128)", 1, 16, 1024, 1200, 128),
    ];

    println!(
        "{:<40} {:>12} {:>12} {:>12} {:>8} {:>8}",
        "Scenario", "burn (µs)", "FA 16×8 (µs)", "FA 8×16 (µs)", "16×8 vs", "8×16 vs"
    );
    println!("{}", "-".repeat(100));

    for (name, batch, heads, seq_q, seq_kv, head_dim) in scenarios {
        if Instant::now() > deadline {
            println!("⏱ Timeout reached, stopping.");
            break;
        }

        let burn_us = bench_burn(&device, *batch, *heads, *seq_q, *seq_kv, *head_dim);

        let fa16x8_us = bench_tiled_fa(
            &device,
            *batch,
            *heads,
            *seq_q,
            *seq_kv,
            *head_dim,
            &TiledFaConfig::Q16_KV8,
        );

        let fa8x16_us = bench_tiled_fa(
            &device,
            *batch,
            *heads,
            *seq_q,
            *seq_kv,
            *head_dim,
            &TiledFaConfig::Q8_KV16,
        );

        let ratio_16x8 = fa16x8_us / burn_us;
        let ratio_8x16 = fa8x16_us / burn_us;

        println!(
            "{:<40} {:>12.1} {:>12.1} {:>12.1} {:>7.2}× {:>7.2}×",
            name, burn_us, fa16x8_us, fa8x16_us, ratio_16x8, ratio_8x16
        );
    }

    println!();

    // Impact estimate for best config at DiT dims
    println!("--- Impact Estimate (960 SDPA calls per inference) ---");
    let burn_dit = bench_burn(&device, 1, 16, 750, 850, 128);
    let fa_dit = bench_tiled_fa(&device, 1, 16, 750, 850, 128, &TiledFaConfig::Q16_KV8);
    let savings_ms = (burn_dit - fa_dit) * 960.0 / 1000.0;
    println!(
        "  burn: {burn_dit:.1}µs  FA-16×8: {fa_dit:.1}µs  Δ: {:.1}µs/call  → {savings_ms:+.1}ms total",
        burn_dit - fa_dit
    );

    println!("\nDone.");
}
