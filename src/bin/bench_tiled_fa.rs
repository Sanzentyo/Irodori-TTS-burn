//! Isolated benchmark: FlashAttention WGSL kernels vs burn's generic SDPA on WgpuRaw.
//!
//! Compares five implementations:
//! 1. burn's generic multi-step SDPA (Q@K^T → softmax → @V)
//! 2. Tiled FA 16×8 (WebGPU-portable, WG=128, 13KB shared)
//! 3. Tiled FA 8×16 (WebGPU-portable, WG=128, 13KB shared)
//! 4. Native FA 32×8 (native-only, WG=256, 22KB shared)
//! 5. Native FA 16×16 (native-only, WG=256, 18KB shared)
//!
//! Run with: `cargo run --release --bin bench_tiled_fa`

use std::time::Instant;

use burn::{
    backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup},
    tensor::{Tensor, TensorPrimitive, backend::Backend},
};

use burn::backend::wgpu::CubeTensor;
use burn::tensor::{DType, Shape};
use irodori_tts_burn::{
    WgpuRaw,
    kernels::fused_sdpa_native::{NativeFaConfig, native_fa_sdpa_wgsl},
    kernels::fused_sdpa_native_f16::native_fa_sdpa_f16_in,
    kernels::fused_sdpa_tiled::{TiledFaConfig, tiled_fa_sdpa_wgsl},
};

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 20;
const TIMEOUT_SECS: u64 = 600;

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

fn bench_native_fa_f16(
    device: &<B as Backend>::Device,
    batch: usize,
    heads: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    config: &NativeFaConfig,
) -> f64 {
    // Bootstrap the wgpu client via a standard burn tensor.
    let dummy = Tensor::<B, 2>::ones([batch, seq_kv], device);
    let dummy_prim = dummy.into_primitive().tensor();
    let client = dummy_prim.client.clone();
    let dev = dummy_prim.device.clone();
    drop(dummy_prim);

    let q_bytes = batch * heads * seq_q * head_dim * core::mem::size_of::<half::f16>();
    let kv_bytes = batch * heads * seq_kv * head_dim * core::mem::size_of::<half::f16>();
    let scale = (head_dim as f64).powf(-0.5);

    // Pre-allocate; create fresh CubeTensor handles each iteration (matching
    // the pattern used by bench_native_fa with .clone().into_primitive().tensor()).
    let make_tensors = || {
        let q_t = CubeTensor::new_contiguous(
            client.clone(),
            dev.clone(),
            Shape::from([batch, heads, seq_q, head_dim]),
            client.empty(q_bytes),
            DType::F16,
        );
        let k_t = CubeTensor::new_contiguous(
            client.clone(),
            dev.clone(),
            Shape::from([batch, heads, seq_kv, head_dim]),
            client.empty(kv_bytes),
            DType::F16,
        );
        let v_t = CubeTensor::new_contiguous(
            client.clone(),
            dev.clone(),
            Shape::from([batch, heads, seq_kv, head_dim]),
            client.empty(kv_bytes),
            DType::F16,
        );
        let mask_t = Tensor::<B, 2>::ones([batch, seq_kv], device)
            .into_primitive()
            .tensor();
        (q_t, k_t, v_t, mask_t)
    };

    for _ in 0..WARMUP_ITERS {
        let (q_t, k_t, v_t, mask_t) = make_tensors();
        let _ = native_fa_sdpa_f16_in(q_t, k_t, v_t, mask_t, scale, config);
    }
    {
        let (q_t, k_t, v_t, mask_t) = make_tensors();
        let out = native_fa_sdpa_f16_in(q_t, k_t, v_t, mask_t, scale, config);
        gpu_sync_4d(Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(out)));
    }

    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let (q_t, k_t, v_t, mask_t) = make_tensors();
        let _ = native_fa_sdpa_f16_in(q_t, k_t, v_t, mask_t, scale, config);
    }
    {
        let (q_t, k_t, v_t, mask_t) = make_tensors();
        let out = native_fa_sdpa_f16_in(q_t, k_t, v_t, mask_t, scale, config);
        gpu_sync_4d(Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(out)));
    }
    start.elapsed().as_micros() as f64 / (BENCH_ITERS + 1) as f64
}

fn main() {
    println!("=== FlashAttention Micro-Benchmark: Custom WGSL vs burn Generic ===");
    println!("Config: warmup={WARMUP_ITERS}, bench_iters={BENCH_ITERS}, timeout={TIMEOUT_SECS}s\n");

    let deadline = Instant::now() + std::time::Duration::from_secs(TIMEOUT_SECS);

    let device = WgpuDevice::DefaultDevice;
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    println!("Device initialised.\n");

    // All configs use D=128
    let scenarios: &[(&str, usize, usize, usize, usize, usize)] = &[
        ("DiT joint attn (1×16×750×850×128)", 1, 16, 750, 850, 128),
        ("Short seq (1×16×100×150×128)", 1, 16, 100, 150, 128),
        ("Square (1×16×256×256×128)", 1, 16, 256, 256, 128),
        ("Large seq (1×16×1024×1200×128)", 1, 16, 1024, 1200, 128),
    ];

    println!(
        "{:<40} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Scenario",
        "burn",
        "T16×8",
        "T8×16",
        "N32×8",
        "N16×16",
        "N32×16",
        "Nf16 32×8",
        "Nf16 16×16"
    );
    println!(
        "{:<40} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "", "(µs)", "(µs)", "(µs)", "(µs)", "(µs)", "(µs)", "(µs)", "(µs)"
    );
    println!("{}", "-".repeat(140));

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

        let native32x8_us = bench_native_fa(
            &device,
            *batch,
            *heads,
            *seq_q,
            *seq_kv,
            *head_dim,
            &NativeFaConfig::Q32_KV8,
        );

        let native16x16_us = bench_native_fa(
            &device,
            *batch,
            *heads,
            *seq_q,
            *seq_kv,
            *head_dim,
            &NativeFaConfig::Q16_KV16,
        );

        let native32x16_us = bench_native_fa(
            &device,
            *batch,
            *heads,
            *seq_q,
            *seq_kv,
            *head_dim,
            &NativeFaConfig::Q32_KV16,
        );

        let nf16_32x8_us = bench_native_fa_f16(
            &device,
            *batch,
            *heads,
            *seq_q,
            *seq_kv,
            *head_dim,
            &NativeFaConfig::Q32_KV8,
        );

        let nf16_16x16_us = bench_native_fa_f16(
            &device,
            *batch,
            *heads,
            *seq_q,
            *seq_kv,
            *head_dim,
            &NativeFaConfig::Q16_KV16,
        );

        println!(
            "{:<40} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1}",
            name,
            burn_us,
            fa16x8_us,
            fa8x16_us,
            native32x8_us,
            native16x16_us,
            native32x16_us,
            nf16_32x8_us,
            nf16_16x16_us,
        );
        println!(
            "{:<40} {:>10} {:>10.2}× {:>10.2}× {:>10.2}× {:>10.2}× {:>10.2}× {:>10.2}× {:>10.2}×",
            "",
            "1.00×",
            fa16x8_us / burn_us,
            fa8x16_us / burn_us,
            native32x8_us / burn_us,
            native16x16_us / burn_us,
            native32x16_us / burn_us,
            nf16_32x8_us / burn_us,
            nf16_16x16_us / burn_us,
        );
    }

    println!();

    // Impact estimate for best config at DiT dims
    println!("--- Impact Estimate (960 SDPA calls per inference) ---");
    let burn_dit = bench_burn(&device, 1, 16, 750, 850, 128);
    let fa_dit = bench_tiled_fa(&device, 1, 16, 750, 850, 128, &TiledFaConfig::Q16_KV8);
    let native_dit = bench_native_fa(&device, 1, 16, 750, 850, 128, &NativeFaConfig::Q32_KV8);
    let nf16_dit = bench_native_fa_f16(&device, 1, 16, 750, 850, 128, &NativeFaConfig::Q32_KV8);
    println!(
        "  burn: {burn_dit:.1}µs  T16×8: {fa_dit:.1}µs  N32×8 f32: {native_dit:.1}µs  N32×8 f16: {nf16_dit:.1}µs"
    );
    let best = fa_dit.min(native_dit).min(nf16_dit);
    let savings_ms = (burn_dit - best) * 960.0 / 1000.0;
    let best_name = if nf16_dit <= fa_dit && nf16_dit <= native_dit {
        "N32×8 f16in"
    } else if fa_dit < native_dit {
        "T16×8"
    } else {
        "N32×8"
    };
    println!(
        "  Best: {best_name} ({best:.1}µs)  Δ: {:.1}µs/call  → {savings_ms:+.1}ms total",
        burn_dit - best
    );

    println!("\nDone.");
}
