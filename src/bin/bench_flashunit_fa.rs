//! FlashUnit FA feasibility benchmark.
//!
//! Tests `burn_cubecl::kernel::attention::AttentionStrategy::FlashUnit` on production
//! shapes (1 × 20 heads × 750 seq_q × 950 seq_kv × D=64). This strategy uses thread-
//! level tiled FA-2 with `plane_*` (subgroup) ops for online softmax — it is NEVER
//! tried by burn's autotune because `PRIORITY_MIN` causes it to be skipped once the
//! fallback (naive SDPA) wins first.
//!
//! Results here directly answer: "Does FlashUnit compile and run on Metal/WGPU?
//! Is it faster than naive SDPA (6,638 µs baseline)?"
//!
//! Run with: `just bench-flashunit-fa`

use std::time::Instant;

use burn::{
    backend::wgpu::{WgpuDevice, WgpuRuntime, graphics::AutoGraphicsApi, init_setup},
    tensor::{
        Distribution, Tensor, TensorPrimitive, backend::Backend, ops::AttentionModuleOptions,
    },
};
use burn_cubecl::{
    kernel::attention::{AttentionStrategy, attention as cubecl_attention},
    tensor::CubeTensor,
};

use irodori_tts_burn::WgpuRaw;

const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 10;

type B = WgpuRaw;

fn gpu_sync(t: Tensor<B, 4>) {
    let _ = t.slice([0..1, 0..1, 0..1, 0..1]).into_data();
}

/// Wrap a CubeTensor back into a burn Tensor for reading results / GPU sync.
fn from_prim(t: CubeTensor<WgpuRuntime>) -> Tensor<B, 4> {
    Tensor::from_primitive(TensorPrimitive::Float(t))
}

/// Call burn_cubecl attention with an explicit strategy.
///
/// Returns the output CubeTensor or an error string if the strategy fails.
fn cubecl_attn(
    q: &Tensor<B, 4>,
    k: &Tensor<B, 4>,
    v: &Tensor<B, 4>,
    options: AttentionModuleOptions,
    strategy: AttentionStrategy,
) -> Result<Tensor<B, 4>, String> {
    let q_p = q.clone().into_primitive().tensor();
    let k_p = k.clone().into_primitive().tensor();
    let v_p = v.clone().into_primitive().tensor();

    cubecl_attention::<WgpuRuntime>(q_p, k_p, v_p, None, None, options, strategy)
        .map(from_prim)
        .map_err(|e| format!("{e:?}"))
}

/// Benchmark a closure after warmup; returns mean µs.
fn bench<F: Fn() -> Tensor<B, 4>>(f: F) -> f64 {
    for _ in 0..WARMUP_ITERS {
        gpu_sync(f());
    }
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        gpu_sync(f());
    }
    start.elapsed().as_micros() as f64 / BENCH_ITERS as f64
}

fn run_section(
    device: &<B as Backend>::Device,
    label: &str,
    batch: usize,
    heads: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
) {
    println!("\n── {label} ──────────────────────────────");
    println!("  Shape: batch={batch} heads={heads} seq_q={seq_q} seq_kv={seq_kv} D={head_dim}");

    let q = Tensor::<B, 4>::random(
        [batch, heads, seq_q, head_dim],
        Distribution::Uniform(-1.0, 1.0),
        device,
    );
    let k = Tensor::<B, 4>::random(
        [batch, heads, seq_kv, head_dim],
        Distribution::Uniform(-1.0, 1.0),
        device,
    );
    let v = Tensor::<B, 4>::random(
        [batch, heads, seq_kv, head_dim],
        Distribution::Uniform(-1.0, 1.0),
        device,
    );
    let options = AttentionModuleOptions {
        scale: None,
        softcap: None,
        is_causal: false,
    };

    // ── Fallback (naive SDPA baseline) ──────────────────────────────────────
    let fallback_out = match cubecl_attn(&q, &k, &v, options, AttentionStrategy::Fallback) {
        Ok(t) => {
            gpu_sync(t.clone());
            t
        }
        Err(e) => {
            println!("  fallback: FAILED — {e}");
            return;
        }
    };

    let fallback_us = bench(|| {
        cubecl_attn(&q, &k, &v, options, AttentionStrategy::Fallback)
            .expect("fallback bench failed")
    });
    println!("  Fallback (naive SDPA):  {fallback_us:8.0} µs  (baseline)");

    // ── FlashUnit ────────────────────────────────────────────────────────────
    match cubecl_attn(&q, &k, &v, options, AttentionStrategy::FlashUnit) {
        Err(e) => println!("  FlashUnit:              SETUP FAILED — {e}"),
        Ok(flash_out) => {
            // Correctness check
            let diff = (fallback_out - flash_out.clone()).abs().max().into_scalar();
            println!("  FlashUnit vs fallback:  max_abs_diff = {diff:.4e}");

            let flash_us = bench(|| {
                cubecl_attn(&q, &k, &v, options, AttentionStrategy::FlashUnit)
                    .expect("FlashUnit bench failed")
            });
            let ratio = flash_us / fallback_us;
            println!("  FlashUnit:              {flash_us:8.0} µs  ({ratio:.2}× vs fallback)");
        }
    }
}

fn main() {
    let device = WgpuDevice::default();
    init_setup::<AutoGraphicsApi>(&device, Default::default());

    println!("FlashUnit FA feasibility benchmark");
    println!("===================================");

    // § 1 — Production dims: 20 heads, D=64 (model default)
    run_section(&device, "Production (D=64, 20 heads)", 1, 20, 750, 950, 64);

    // § 2 — Larger seq (stress test — still within Metal limits)
    run_section(
        &device,
        "Stress (D=64, 8 heads, seq=1024)",
        1,
        8,
        1024,
        1024,
        64,
    );

    // § 3 — Tiny check (correctness sanity)
    run_section(&device, "Tiny sanity (D=64, 4 heads)", 1, 4, 32, 32, 64);

    println!("\nDone.");
}
