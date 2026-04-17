//! WGPU operator-level profiling for the DiT model.
//!
//! Times individual operation categories at real-model dimensions to identify
//! bottlenecks specific to the WGPU backend (which may differ from CUDA).
//!
//! Run: `cargo run --release --bin profile_wgpu_ops`

use std::time::{Duration, Instant};

use burn::{
    backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup},
    tensor::{
        Tensor, backend::Backend, module::attention as burn_attention, ops::AttentionModuleOptions,
    },
};

use irodori_tts_burn::WgpuRaw;

type B = WgpuRaw;

const WARMUP: usize = 5;
const ITERS: usize = 30;
const TIMEOUT: Duration = Duration::from_secs(300);

// Model dimensions (Irodori-TTS-500M-v2)
const MODEL_DIM: usize = 1280;
const NUM_HEADS: usize = 20;
const HEAD_DIM: usize = MODEL_DIM / NUM_HEADS; // 64
const SEQ_LEN: usize = 750;
const TEXT_LEN: usize = 200;
const CTX_LEN: usize = TEXT_LEN + SEQ_LEN; // joint attention context
const SWI_HIDDEN: usize = MODEL_DIM * 4; // SwiGLU hidden dim

fn gpu_sync<const D: usize>(t: Tensor<B, D>) {
    let _ = t.into_data();
}

/// Benchmark a closure, returning mean µs per iteration.
fn bench<F>(_device: &<B as Backend>::Device, label: &str, mut f: F) -> f64
where
    F: FnMut() -> Tensor<B, 2>,
{
    // Warmup
    for _ in 0..WARMUP {
        let _ = f();
    }
    gpu_sync(f());

    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = f();
    }
    gpu_sync(f());
    let elapsed = start.elapsed();

    let us = elapsed.as_micros() as f64 / (ITERS + 1) as f64;
    println!("  {label:<45} {us:>10.1} µs/iter");
    us
}

fn bench_4d<F>(_device: &<B as Backend>::Device, label: &str, mut f: F) -> f64
where
    F: FnMut() -> Tensor<B, 4>,
{
    for _ in 0..WARMUP {
        let _ = f();
    }
    // Force GPU sync by reading a single element.
    let r = f();
    let _ = r.slice([0..1, 0..1, 0..1, 0..1]).into_data();

    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = f();
    }
    let r = f();
    let _ = r.slice([0..1, 0..1, 0..1, 0..1]).into_data();
    let elapsed = start.elapsed();

    let us = elapsed.as_micros() as f64 / (ITERS + 1) as f64;
    println!("  {label:<45} {us:>10.1} µs/iter");
    us
}

fn main() {
    let deadline = Instant::now() + TIMEOUT;
    println!("=== WGPU Operator Profile (WgpuRaw f32, DX12) ===");
    println!("Model: dim={MODEL_DIM}, heads={NUM_HEADS}, head_dim={HEAD_DIM}");
    println!("seq_len={SEQ_LEN}, text_len={TEXT_LEN}\n");

    let device = WgpuDevice::DefaultDevice;
    init_setup::<AutoGraphicsApi>(&device, Default::default());

    let mut totals: Vec<(&str, f64)> = Vec::new();

    // ── 1. Linear projections (matmul) ──────────────────────────────────
    println!("1. Linear projections (matmul)");
    {
        let x = Tensor::<B, 2>::ones([SEQ_LEN, MODEL_DIM], &device);
        let w = Tensor::<B, 2>::ones([MODEL_DIM, MODEL_DIM], &device);
        let w3 = Tensor::<B, 2>::ones([MODEL_DIM, MODEL_DIM * 3], &device);
        let w_swi = Tensor::<B, 2>::ones([MODEL_DIM, SWI_HIDDEN * 2], &device);

        let us = bench(&device, "x @ W  [750×1280] × [1280×1280]", || {
            x.clone().matmul(w.clone())
        });
        // 12 blocks × (q_proj + k_proj + v_proj + out_proj + ctx variants) ≈ 12 × 8 = 96
        totals.push(("linear_single", us));

        let us = bench(&device, "fused QKV  [750×1280] × [1280×3840]", || {
            x.clone().matmul(w3.clone())
        });
        totals.push(("linear_qkv", us));

        let us = bench(&device, "SwiGLU proj  [750×1280] × [1280×10240]", || {
            x.clone().matmul(w_swi.clone())
        });
        totals.push(("linear_swiglu", us));
    }
    println!();

    if Instant::now() > deadline {
        println!("⏱ Timeout");
        return;
    }

    // ── 2. Attention (Q@K^T → softmax → @V) ────────────────────────────
    println!("2. Attention operations");
    {
        // Self-attention: latent-only
        let q_self = Tensor::<B, 4>::ones([1, NUM_HEADS, SEQ_LEN, HEAD_DIM], &device);
        let k_self = q_self.clone();
        let v_self = q_self.clone();

        let us = bench_4d(
            &device,
            "self-attn Q@K^T  [20×750×64]×[20×64×750]",
            || q_self.clone().matmul(k_self.clone().swap_dims(2, 3)),
        );
        totals.push(("attn_qk_self", us));

        let us = bench_4d(
            &device,
            "self-attn scores@V  [20×750×750]×[20×750×64]",
            || {
                let scores = q_self.clone().matmul(k_self.clone().swap_dims(2, 3));
                scores.matmul(v_self.clone())
            },
        );
        totals.push(("attn_qkv_self", us));

        // Joint attention: latent queries, [latent+text] KV
        let k_joint = Tensor::<B, 4>::ones([1, NUM_HEADS, CTX_LEN, HEAD_DIM], &device);
        let _v_joint = k_joint.clone();

        let us = bench_4d(
            &device,
            "joint-attn Q@K^T  [20×750×64]×[20×64×950]",
            || q_self.clone().matmul(k_joint.clone().swap_dims(2, 3)),
        );
        totals.push(("attn_qk_joint", us));

        // burn SDPA (what the model actually calls)
        let q_sdpa = Tensor::<B, 4>::ones([1, NUM_HEADS, SEQ_LEN, HEAD_DIM], &device);
        let k_sdpa = Tensor::<B, 4>::ones([1, NUM_HEADS, CTX_LEN, HEAD_DIM], &device);
        let v_sdpa = k_sdpa.clone();

        let us = bench_4d(
            &device,
            "burn SDPA joint  [1,20,750,64]×[1,20,950,64]",
            || {
                let options = AttentionModuleOptions {
                    scale: None,
                    softcap: None,
                    is_causal: false,
                };
                burn_attention(
                    q_sdpa.clone(),
                    k_sdpa.clone(),
                    v_sdpa.clone(),
                    None,
                    None,
                    options,
                )
            },
        );
        totals.push(("sdpa_joint", us));

        let us = bench_4d(
            &device,
            "burn SDPA self  [1,20,750,64]×[1,20,750,64]",
            || {
                let options = AttentionModuleOptions {
                    scale: None,
                    softcap: None,
                    is_causal: false,
                };
                burn_attention(
                    q_self.clone(),
                    k_self.clone(),
                    v_self.clone(),
                    None,
                    None,
                    options,
                )
            },
        );
        totals.push(("sdpa_self", us));
    }
    println!();

    if Instant::now() > deadline {
        println!("⏱ Timeout");
        return;
    }

    // ── 3. Norm operations ──────────────────────────────────────────────
    println!("3. Norm operations (RMSNorm)");
    {
        let x = Tensor::<B, 2>::ones([SEQ_LEN, MODEL_DIM], &device);

        let us = bench(&device, "RMSNorm [750×1280] (powf+mean+sqrt+div)", || {
            let rms = x
                .clone()
                .powf_scalar(2.0)
                .mean_dim(1)
                .add_scalar(1e-6)
                .sqrt();
            (x.clone() / rms).reshape([SEQ_LEN, MODEL_DIM])
        });
        totals.push(("rmsnorm", us));

        // AdaLN = RMSNorm + modulate
        let scale = Tensor::<B, 2>::zeros([1, MODEL_DIM], &device);
        let shift = Tensor::<B, 2>::zeros([1, MODEL_DIM], &device);

        let us = bench(&device, "AdaLN [750×1280] (rms + scale + shift)", || {
            let rms = x
                .clone()
                .powf_scalar(2.0)
                .mean_dim(1)
                .add_scalar(1e-6)
                .sqrt();
            let x_norm = x.clone() / rms;
            let out = x_norm * (scale.clone() + 1.0) + shift.clone();
            out.reshape([SEQ_LEN, MODEL_DIM])
        });
        totals.push(("adaln", us));
    }
    println!();

    if Instant::now() > deadline {
        println!("⏱ Timeout");
        return;
    }

    // ── 4. Elementwise / activation ─────────────────────────────────────
    println!("4. Elementwise & activation");
    {
        let x = Tensor::<B, 2>::ones([SEQ_LEN, SWI_HIDDEN], &device);
        let gate = Tensor::<B, 2>::ones([SEQ_LEN, SWI_HIDDEN], &device);

        let us = bench(&device, "SiLU  [750×5120]", || {
            burn::tensor::activation::silu(x.clone())
        });
        totals.push(("silu", us));

        let us = bench(&device, "SwiGLU gate  silu(x) * gate  [750×5120]", || {
            burn::tensor::activation::silu(x.clone()) * gate.clone()
        });
        totals.push(("swiglu_gate", us));

        let a = Tensor::<B, 2>::ones([SEQ_LEN, MODEL_DIM], &device);
        let b = a.clone();
        let us = bench(&device, "add  [750×1280]", || a.clone() + b.clone());
        totals.push(("add", us));
    }
    println!();

    // ── 5. Softmax ──────────────────────────────────────────────────────
    println!("5. Softmax");
    {
        let scores = Tensor::<B, 2>::ones([SEQ_LEN, CTX_LEN], &device);
        let us = bench(&device, "softmax  [750×950]", || {
            burn::tensor::activation::softmax(scores.clone(), 1)
        });
        totals.push(("softmax", us));
    }
    println!();

    // ── Summary ─────────────────────────────────────────────────────────
    println!("=== Estimated per-forward-pass breakdown (12 DiT blocks) ===");
    println!("(Rough estimates based on call counts per forward pass)\n");

    // Per-forward counts (single-call CFG with joint attention):
    // linear: 12 blocks × (fused_qkv_latent + fused_qkv_ctx + out_proj_lat + out_proj_ctx
    //          + swiglu_proj + swiglu_out) = 12 × 6 = 72 linear ops
    // SDPA: 12 blocks × 1 joint attention = 12
    // RMSNorm/AdaLN: 24 AdaLN + 14 RMSNorm + 24 HeadRmsNorm = 62
    // Softmax: inside SDPA, counted there
    // SwiGLU gate: 12
    // add (residuals): 12 × 2 = 24

    struct OpEstimate {
        name: &'static str,
        key: &'static str,
        count: usize,
    }

    let estimates = [
        OpEstimate {
            name: "Linear (single, 1280→1280)",
            key: "linear_single",
            count: 48,
        },
        OpEstimate {
            name: "Linear (fused QKV, 1280→3840)",
            key: "linear_qkv",
            count: 24,
        },
        OpEstimate {
            name: "Linear (SwiGLU proj, 1280→10240)",
            key: "linear_swiglu",
            count: 12,
        },
        OpEstimate {
            name: "SDPA (joint attention)",
            key: "sdpa_joint",
            count: 12,
        },
        OpEstimate {
            name: "AdaLN (RMSNorm + modulate)",
            key: "adaln",
            count: 24,
        },
        OpEstimate {
            name: "RMSNorm (standalone)",
            key: "rmsnorm",
            count: 14,
        },
        OpEstimate {
            name: "SwiGLU gate (silu*gate)",
            key: "swiglu_gate",
            count: 12,
        },
        OpEstimate {
            name: "Add (residuals)",
            key: "add",
            count: 24,
        },
    ];

    let mut grand_total = 0.0;
    let mut rows: Vec<(String, f64)> = Vec::new();

    for est in &estimates {
        if let Some((_, us)) = totals.iter().find(|(k, _)| *k == est.key) {
            let total_us = us * est.count as f64;
            grand_total += total_us;
            rows.push((format!("{} (×{})", est.name, est.count), total_us));
        }
    }

    for (name, total_us) in &rows {
        let pct = total_us / grand_total * 100.0;
        println!("  {name:<50} {total_us:>10.0} µs  ({pct:>5.1}%)");
    }
    println!("  {:<50} {:>10.0} µs", "TOTAL (estimated)", grand_total);
    println!(
        "  {:<50} {:>10.1} ms",
        "Per forward pass",
        grand_total / 1000.0
    );
    println!(
        "  {:<50} {:>10.1} ms",
        "40-step inference",
        grand_total * 40.0 / 1000.0
    );

    println!("\nDone.");
}
