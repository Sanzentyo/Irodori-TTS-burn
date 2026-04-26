//! Isolated benchmark: FlashAttention WGSL kernels vs burn's generic SDPA on WgpuRaw.
//!
//! Two sections:
//!   § D=64  — production model dims (20 heads, head_dim=64): burn vs N32×8 / N16×16 / N8×32 / T8×8 / T4×16
//!   § D=128 — legacy reference dims (16 heads, head_dim=128): burn vs T16×8 / T8×16 / N32×8 / N16×16
//!
//! Tiled-FA constraint: tile_q × tile_kv == head_dim (workgroup = head_dim).
//!   D=64  → Q8_KV8, Q4_KV16  (WG=64)
//!   D=128 → Q16_KV8, Q8_KV16 (WG=128)
//!
//! Native-FA constraint: head_dim % tile_kv == 0, head_dim is power-of-2.
//!   Works at both D=64 and D=128 for all NativeFaConfig variants.
//!
//! Run with: `just bench-tiled-fa`

use std::time::Instant;

use burn::{
    backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup},
    tensor::{
        Bool, Tensor, TensorPrimitive, backend::Backend, module::attention as burn_attention,
        ops::AttentionModuleOptions,
    },
};
use irodori_tts_burn::{
    WgpuRaw,
    kernels::fused_sdpa_native::{NativeFaConfig, native_fa_sdpa_wgsl},
    kernels::fused_sdpa_tiled::{TiledFaConfig, tiled_fa_sdpa_wgsl},
};

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;
const TIMEOUT_SECS: u64 = 600;

type B = WgpuRaw;

fn gpu_sync_4d(t: Tensor<B, 4>) {
    let _ = t.slice([0..1, 0..1, 0..1, 0..1]).into_data();
}

/// burn's `attention()` module: CubeCL flash attention, scale=None (production path).
fn bench_burn_attention(
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
    let mask_bool = Tensor::<B, 2, Bool>::from_data(
        // All-False = "do not mask" = all positions valid.
        // burn_attention uses True=masked-out convention (NdArray/CubeCL), so False here
        // means every key position attends — the realistic full-sequence scenario.
        burn::tensor::TensorData::new(vec![false; batch * seq_kv], [batch, seq_kv]),
        device,
    );
    let mask_4d = Some(mask_bool.unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(2));
    let options = AttentionModuleOptions {
        scale: None,
        softcap: None,
        is_causal: false,
    };

    for _ in 0..WARMUP_ITERS {
        let _ = burn_attention(
            q.clone(),
            k.clone(),
            v.clone(),
            mask_4d.clone(),
            None,
            options,
        );
    }
    gpu_sync_4d(burn_attention(
        q.clone(),
        k.clone(),
        v.clone(),
        mask_4d.clone(),
        None,
        options,
    ));

    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let _ = burn_attention(
            q.clone(),
            k.clone(),
            v.clone(),
            mask_4d.clone(),
            None,
            options,
        );
    }
    gpu_sync_4d(burn_attention(
        q.clone(),
        k.clone(),
        v.clone(),
        mask_4d.clone(),
        None,
        options,
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

fn bench_native_fa(
    device: &<B as Backend>::Device,
    batch: usize,
    heads: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    config: &NativeFaConfig,
) -> f64 {
    let q = Tensor::<B, 4>::ones([batch, heads, seq_q, head_dim], device);
    let k = Tensor::<B, 4>::ones([batch, heads, seq_kv, head_dim], device);
    let v = Tensor::<B, 4>::ones([batch, heads, seq_kv, head_dim], device);
    let mask = Tensor::<B, 2>::ones([batch, seq_kv], device);
    let scale = (head_dim as f64).powf(-0.5);

    for _ in 0..WARMUP_ITERS {
        let _ = native_fa_sdpa_wgsl(
            q.clone().into_primitive().tensor(),
            k.clone().into_primitive().tensor(),
            v.clone().into_primitive().tensor(),
            mask.clone().into_primitive().tensor(),
            scale,
            config,
        );
    }
    {
        let out = native_fa_sdpa_wgsl(
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
        let _ = native_fa_sdpa_wgsl(
            q.clone().into_primitive().tensor(),
            k.clone().into_primitive().tensor(),
            v.clone().into_primitive().tensor(),
            mask.clone().into_primitive().tensor(),
            scale,
            config,
        );
    }
    {
        let out = native_fa_sdpa_wgsl(
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

fn fmt_ratio(custom_us: f64, burn_us: f64) -> String {
    format!("{:.2}×", custom_us / burn_us)
}

fn main() {
    println!("=== FlashAttention Micro-Benchmark: Custom WGSL vs burn Generic ===");
    println!("Config: warmup={WARMUP_ITERS}, bench_iters={BENCH_ITERS}, timeout={TIMEOUT_SECS}s\n");

    let deadline = Instant::now() + std::time::Duration::from_secs(TIMEOUT_SECS);

    let device = WgpuDevice::DefaultDevice;
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    println!("Device initialised.\n");

    // ── § 1: D=64 Production Model (head_dim=64, 20 heads) ────────────────────
    // Tiled: Q8×8 (WG=64), Q4×16 (WG=64)  |  Native: N32×8, N16×16, N8×32
    println!("§ D=64 Production Model (head_dim=64, 20 heads)");
    println!("  Tiled: Q8×8 (WG=64), Q4×16 (WG=64)  |  Native: N32×8, N16×16, N8×32");
    println!(
        "{:<50} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Scenario", "burn", "T8×8", "T4×16", "N32×8", "N16×16", "N8×32"
    );
    println!(
        "{:<50} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "", "(µs)", "(µs)", "(µs)", "(µs)", "(µs)", "(µs)"
    );
    println!("{}", "-".repeat(120));

    let d64_scenarios: &[(&str, usize, usize, usize, usize)] = &[
        ("1×20×750×950 (single, 750-token seq)", 1, 20, 750, 950),
        ("3×20×750×950 (batch=3 CFG hot path)", 3, 20, 750, 950),
        ("1×20×512×512 (square mid-seq)", 1, 20, 512, 512),
        ("1×20×256×256 (short seq)", 1, 20, 256, 256),
    ];

    for &(name, batch, heads, seq_q, seq_kv) in d64_scenarios {
        if Instant::now() > deadline {
            println!("⏱ Timeout reached, stopping.");
            return;
        }
        let head_dim = 64;
        let burn_us = bench_burn_attention(&device, batch, heads, seq_q, seq_kv, head_dim);
        let t8x8_us = bench_tiled_fa(
            &device,
            batch,
            heads,
            seq_q,
            seq_kv,
            head_dim,
            &TiledFaConfig::Q8_KV8,
        );
        let t4x16_us = bench_tiled_fa(
            &device,
            batch,
            heads,
            seq_q,
            seq_kv,
            head_dim,
            &TiledFaConfig::Q4_KV16,
        );
        let n32x8_us = bench_native_fa(
            &device,
            batch,
            heads,
            seq_q,
            seq_kv,
            head_dim,
            &NativeFaConfig::Q32_KV8,
        );
        let n16x16_us = bench_native_fa(
            &device,
            batch,
            heads,
            seq_q,
            seq_kv,
            head_dim,
            &NativeFaConfig::Q16_KV16,
        );
        let n8x32_us = bench_native_fa(
            &device,
            batch,
            heads,
            seq_q,
            seq_kv,
            head_dim,
            &NativeFaConfig::Q8_KV32,
        );

        println!(
            "{:<50} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1}",
            name, burn_us, t8x8_us, t4x16_us, n32x8_us, n16x16_us, n8x32_us
        );
        println!(
            "{:<50} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "",
            "1.00×",
            fmt_ratio(t8x8_us, burn_us),
            fmt_ratio(t4x16_us, burn_us),
            fmt_ratio(n32x8_us, burn_us),
            fmt_ratio(n16x16_us, burn_us),
            fmt_ratio(n8x32_us, burn_us),
        );
    }

    println!();

    // ── § 2: D=128 Legacy Reference (head_dim=128, 16 heads) ──────────────────
    // Tiled: Q16×8 (WG=128), Q8×16 (WG=128)  |  Native: N32×8, N16×16
    println!("§ D=128 Legacy Reference (head_dim=128, 16 heads)");
    println!("  Tiled: Q16×8 (WG=128), Q8×16 (WG=128)  |  Native: N32×8, N16×16");
    println!(
        "{:<50} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Scenario", "burn", "T16×8", "T8×16", "N32×8", "N16×16"
    );
    println!(
        "{:<50} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "", "(µs)", "(µs)", "(µs)", "(µs)", "(µs)"
    );
    println!("{}", "-".repeat(100));

    let d128_scenarios: &[(&str, usize, usize, usize, usize)] = &[
        ("1×16×750×850 (legacy full seq)", 1, 16, 750, 850),
        ("1×16×256×256 (legacy square)", 1, 16, 256, 256),
        ("1×16×100×150 (legacy short)", 1, 16, 100, 150),
    ];

    for &(name, batch, heads, seq_q, seq_kv) in d128_scenarios {
        if Instant::now() > deadline {
            println!("⏱ Timeout reached, stopping.");
            return;
        }
        let head_dim = 128;
        let burn_us = bench_burn_attention(&device, batch, heads, seq_q, seq_kv, head_dim);
        let t16x8_us = bench_tiled_fa(
            &device,
            batch,
            heads,
            seq_q,
            seq_kv,
            head_dim,
            &TiledFaConfig::Q16_KV8,
        );
        let t8x16_us = bench_tiled_fa(
            &device,
            batch,
            heads,
            seq_q,
            seq_kv,
            head_dim,
            &TiledFaConfig::Q8_KV16,
        );
        let n32x8_us = bench_native_fa(
            &device,
            batch,
            heads,
            seq_q,
            seq_kv,
            head_dim,
            &NativeFaConfig::Q32_KV8,
        );
        let n16x16_us = bench_native_fa(
            &device,
            batch,
            heads,
            seq_q,
            seq_kv,
            head_dim,
            &NativeFaConfig::Q16_KV16,
        );

        println!(
            "{:<50} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1}",
            name, burn_us, t16x8_us, t8x16_us, n32x8_us, n16x16_us
        );
        println!(
            "{:<50} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "",
            "1.00×",
            fmt_ratio(t16x8_us, burn_us),
            fmt_ratio(t8x16_us, burn_us),
            fmt_ratio(n32x8_us, burn_us),
            fmt_ratio(n16x16_us, burn_us),
        );
    }

    println!();

    // ── Impact estimate: production dims (D=64, 20 heads, 40 steps) ───────────
    println!("--- Impact Estimate (D=64, 40 diffusion steps, ~1200 SDPA calls/step) ---");
    let burn_model = bench_burn_attention(&device, 1, 20, 750, 950, 64);
    let n32x8_model = bench_native_fa(&device, 1, 20, 750, 950, 64, &NativeFaConfig::Q32_KV8);
    let n16x16_model = bench_native_fa(&device, 1, 20, 750, 950, 64, &NativeFaConfig::Q16_KV16);
    let n8x32_model = bench_native_fa(&device, 1, 20, 750, 950, 64, &NativeFaConfig::Q8_KV32);

    let best = n32x8_model.min(n16x16_model).min(n8x32_model);
    let best_name = if n8x32_model <= n32x8_model && n8x32_model <= n16x16_model {
        "N8×32"
    } else if n32x8_model <= n16x16_model {
        "N32×8"
    } else {
        "N16×16"
    };
    let savings_ms = (burn_model - best) * 1200.0 * 40.0 / 1_000_000.0;
    println!(
        "  burn: {burn_model:.1}µs  N32×8: {n32x8_model:.1}µs  N16×16: {n16x16_model:.1}µs  N8×32: {n8x32_model:.1}µs"
    );
    println!(
        "  Best: {best_name} ({best:.1}µs)  Δ: {:.1}µs/call  → {savings_ms:+.1}ms total",
        burn_model - best
    );

    // ── Scaling diagnostic: D=64 vs D=128 ─────────────────────────────────────
    println!("\n--- Scaling Diagnostic (D=64 vs D=128, 1×16×256×256) ---");
    let burn_d64 = bench_burn_attention(&device, 1, 16, 256, 256, 64);
    let burn_d128 = bench_burn_attention(&device, 1, 16, 256, 256, 128);
    let ratio = burn_d128 / burn_d64;
    println!(
        "  burn D=64: {burn_d64:.1}µs  |  burn D=128: {burn_d128:.1}µs  |  ratio: {ratio:.2}×"
    );
    if ratio > 2.5 {
        println!("  ⚠ Super-linear ({ratio:.2}× vs ~2.0× compute) — barrier-bound.");
    } else if ratio < 1.8 {
        println!("  ⚠ Sub-linear ({ratio:.2}× vs ~2.0× compute) — overhead-dominated.");
    } else {
        println!("  ✓ ~linear ({ratio:.2}× for ~2.0× compute) — likely compute-bound.");
    }

    println!("\nDone.");
}
