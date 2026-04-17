//! Isolated benchmark: Fused SDPA WGSL kernel vs burn's generic attention on WgpuRaw.
//!
//! Measures the fused single-pass online-softmax SDPA kernel against burn's
//! multi-step SDPA (Q@K^T → scale → mask → softmax → @V).
//!
//! Run with: `cargo run --release --bin bench_fused_sdpa`

use std::time::Instant;

use burn::{
    backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup},
    tensor::{Tensor, TensorPrimitive, backend::Backend},
};

use irodori_tts_burn::{WgpuRaw, kernels::fused_sdpa::fused_sdpa_wgsl};

const WARMUP_ITERS: usize = 10;
const BENCH_ITERS: usize = 50;
const TIMEOUT_SECS: u64 = 180;

type B = WgpuRaw;

fn gpu_sync_4d(t: Tensor<B, 4>) {
    let _ = t.slice([0..1, 0..1, 0..1, 0..1]).into_data();
}

/// burn's generic SDPA: Q@K^T * scale + mask → softmax → @V
fn burn_sdpa(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<Tensor<B, 2>>,
    scale: f64,
) -> Tensor<B, 4> {
    // Q [B, H, S_Q, D] @ K^T [B, H, D, S_KV] → [B, H, S_Q, S_KV]
    let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale);

    let scores = if let Some(m) = mask {
        // m [B, S_KV] → [B, 1, 1, S_KV]
        let m = m.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(1);
        let neg_inf = Tensor::<B, 4>::full(scores.shape(), f32::NEG_INFINITY, &scores.device());
        scores.mask_where(m.lower_equal_elem(0.5), neg_inf)
    } else {
        scores
    };

    // softmax along last dim (S_KV)
    let attn = burn::tensor::activation::softmax(scores, 3);
    attn.matmul(v)
}

/// Benchmark: burn generic SDPA (matmul + softmax + matmul).
fn bench_burn_sdpa(
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
        let _ = burn_sdpa(q.clone(), k.clone(), v.clone(), Some(mask.clone()), scale);
    }
    gpu_sync_4d(burn_sdpa(
        q.clone(),
        k.clone(),
        v.clone(),
        Some(mask.clone()),
        scale,
    ));

    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let _ = burn_sdpa(q.clone(), k.clone(), v.clone(), Some(mask.clone()), scale);
    }
    gpu_sync_4d(burn_sdpa(
        q.clone(),
        k.clone(),
        v.clone(),
        Some(mask.clone()),
        scale,
    ));
    let elapsed = start.elapsed();

    elapsed.as_micros() as f64 / (BENCH_ITERS + 1) as f64
}

/// Benchmark: custom fused SDPA WGSL kernel.
fn bench_custom_sdpa(
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

    // Warmup (includes pipeline compilation)
    for _ in 0..WARMUP_ITERS {
        let _ = fused_sdpa_wgsl(
            q.clone().into_primitive().tensor(),
            k.clone().into_primitive().tensor(),
            v.clone().into_primitive().tensor(),
            mask.clone().into_primitive().tensor(),
            scale,
        );
    }
    {
        let out = fused_sdpa_wgsl(
            q.clone().into_primitive().tensor(),
            k.clone().into_primitive().tensor(),
            v.clone().into_primitive().tensor(),
            mask.clone().into_primitive().tensor(),
            scale,
        );
        gpu_sync_4d(Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(out)));
    }

    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let _ = fused_sdpa_wgsl(
            q.clone().into_primitive().tensor(),
            k.clone().into_primitive().tensor(),
            v.clone().into_primitive().tensor(),
            mask.clone().into_primitive().tensor(),
            scale,
        );
    }
    {
        let out = fused_sdpa_wgsl(
            q.clone().into_primitive().tensor(),
            k.clone().into_primitive().tensor(),
            v.clone().into_primitive().tensor(),
            mask.clone().into_primitive().tensor(),
            scale,
        );
        gpu_sync_4d(Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(out)));
    }
    let elapsed = start.elapsed();

    elapsed.as_micros() as f64 / (BENCH_ITERS + 1) as f64
}

fn main() {
    println!("=== Fused SDPA Micro-Benchmark: Custom WGSL vs burn Generic (WgpuRaw) ===");
    println!("Config: warmup={WARMUP_ITERS}, bench_iters={BENCH_ITERS}, timeout={TIMEOUT_SECS}s\n");

    let deadline = Instant::now() + std::time::Duration::from_secs(TIMEOUT_SECS);

    let device = WgpuDevice::DefaultDevice;
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    println!("Device initialised.\n");

    let scenarios: &[(&str, usize, usize, usize, usize, usize)] = &[
        // (name, batch, heads, seq_q, seq_kv, head_dim)
        ("DiT joint attn (1×16×750×850×128)", 1, 16, 750, 850, 128),
        ("Short seq (1×16×100×150×128)", 1, 16, 100, 150, 128),
        ("Square (1×16×256×256×128)", 1, 16, 256, 256, 128),
        ("Small head (1×16×750×850×64)", 1, 16, 750, 850, 64),
    ];

    for (name, batch, heads, seq_q, seq_kv, head_dim) in scenarios {
        if Instant::now() > deadline {
            println!("⏱ Timeout reached, stopping.");
            break;
        }

        println!("--- {name} ---");

        let burn_us = bench_burn_sdpa(&device, *batch, *heads, *seq_q, *seq_kv, *head_dim);
        let custom_us = bench_custom_sdpa(&device, *batch, *heads, *seq_q, *seq_kv, *head_dim);
        let speedup = burn_us / custom_us;

        println!("  burn generic : {burn_us:>10.1} µs/iter");
        println!("  custom fused : {custom_us:>10.1} µs/iter");
        println!("  speedup      : {speedup:>10.2}×");
        println!();

        // Impact estimate: 12 attention layers × 40 steps × 2 (cond+uncond) = 960 calls per inference
        let total_savings_ms = (burn_us - custom_us) * 960.0 / 1000.0;
        println!(
            "  estimated 40-step savings: {total_savings_ms:>+.1} ms (960 calls × {:.1} µs delta)",
            burn_us - custom_us
        );
        println!();
    }

    println!("Done.");
}
