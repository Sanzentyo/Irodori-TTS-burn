//! Multi-backend real-model benchmark.
//!
//! Times full-model inference over the converted `Aratako/Irodori-TTS-500M-v2`
//! checkpoint across three backends by selecting a Cargo feature at build time:
//!
//! ```sh
//! just bench-cpu    # --features backend_cpu   → NdArray<f32>
//! just bench-wgpu   # --features backend_wgpu  → Wgpu
//! just bench-cuda   # --features backend_cuda  → Cuda
//! just bench-tch    # --features backend_tch   → LibTorch (cuBLAS/FA3)
//! ```
//!
//! Sequence length defaults to the `fixed_target_latent_steps` value in the
//! checkpoint metadata (750 for the 500M model).

// WGPU pulls in very deeply-nested types; raise recursion limit as needed.
#![recursion_limit = "512"]

// ── Backend selection ─────────────────────────────────────────────────────
//
// Exactly one of the backend_* features should be active.
// The fallback (no feature) is NdArray, so a plain `cargo build` still works.

// Guard against invalid multi-feature combinations.
#[cfg(any(
    all(feature = "backend_wgpu", feature = "backend_cuda"),
    all(feature = "backend_wgpu", feature = "backend_cuda_bf16"),
    all(feature = "backend_cuda", feature = "backend_cuda_bf16"),
    all(feature = "backend_cuda", feature = "backend_tch"),
    all(feature = "backend_cuda", feature = "backend_tch_bf16"),
    all(feature = "backend_cuda_bf16", feature = "backend_tch"),
    all(feature = "backend_cuda_bf16", feature = "backend_tch_bf16"),
    all(feature = "backend_wgpu", feature = "backend_tch"),
    all(feature = "backend_wgpu", feature = "backend_tch_bf16"),
    all(feature = "backend_tch", feature = "backend_tch_bf16"),
))]
compile_error!("backend_* features are mutually exclusive — select exactly one");

#[cfg(feature = "backend_wgpu")]
type B = burn::backend::Wgpu;

#[cfg(feature = "backend_cuda")]
type B = burn::backend::Cuda;

#[cfg(feature = "backend_cuda_bf16")]
type B = burn::backend::Cuda<half::bf16>;

#[cfg(feature = "backend_tch")]
type B = burn::backend::LibTorch;

#[cfg(feature = "backend_tch_bf16")]
type B = burn::backend::LibTorch<half::bf16>;

// NdArray is the fallback: active when no explicit backend feature is set.
#[cfg(not(any(
    feature = "backend_wgpu",
    feature = "backend_cuda",
    feature = "backend_cuda_bf16",
    feature = "backend_tch",
    feature = "backend_tch_bf16",
)))]
type B = burn::backend::NdArray<f32>;

// ── Imports ───────────────────────────────────────────────────────────────

use std::{path::Path, time::Instant};

use anyhow::{Context, Result, bail};
use burn::tensor::{Bool, Int, Tensor};
use clap::Parser;

use irodori_tts_burn::{
    CfgGuidanceMode, GuidanceConfig, SamplerParams, SamplingRequest, backend_config::BackendConfig,
    sample_euler_rf_cfg, weights::load_model,
};

// ── CLI ───────────────────────────────────────────────────────────────────

#[derive(Debug, Parser)]
#[command(
    name = "bench_realmodel",
    about = "Time inference with the real converted checkpoint (multi-backend)"
)]
struct Args {
    /// Path to the converted Burn safetensors checkpoint.
    #[arg(long, default_value = "target/model_converted.safetensors")]
    checkpoint: String,

    /// Output sequence length (latent frames).
    ///
    /// Defaults to `fixed_target_latent_steps` from the checkpoint metadata.
    #[arg(long)]
    seq_len: Option<usize>,

    /// Number of diffusion sampling steps.
    #[arg(long, default_value_t = 40)]
    num_steps: usize,

    /// Number of warm-up runs (not timed, GPU pipeline warm-up).
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Number of timed benchmark runs.
    #[arg(long, default_value_t = 3)]
    runs: usize,

    /// CFG speaker scale.
    #[arg(long, default_value_t = 5.0)]
    cfg_speaker: f32,

    /// Minimum timestep for CFG application.
    #[arg(long, default_value_t = 0.5)]
    cfg_min_t: f32,

    /// GPU device index (0-based).
    ///
    /// For CUDA/LibTorch backends selects the CUDA device.
    /// For WGPU selects the Nth discrete GPU.
    /// For CPU backends this is ignored.
    #[arg(long, default_value_t = 0)]
    gpu_id: u32,
}

// ── Main ──────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    let backend_name = B::backend_label();
    eprintln!("Backend    : {backend_name}");
    eprintln!("Checkpoint : {}", args.checkpoint);

    // ── Load model ────────────────────────────────────────────────────────
    eprintln!("Loading model …");
    let device = B::device_from_id(args.gpu_id);
    let t_load = Instant::now();
    let (model, cfg) = load_model::<B>(Path::new(&args.checkpoint), &device)
        .context("failed to load model — run `just convert-model` first")?;
    let load_ms = t_load.elapsed().as_millis();
    eprintln!("Model loaded in {load_ms} ms");
    eprintln!(
        "Config     : model_dim={}, layers={}, heads={}",
        cfg.model_dim, cfg.num_layers, cfg.num_heads
    );

    let seq_len = args
        .seq_len
        .or(cfg.fixed_target_latent_steps)
        .unwrap_or(256);
    eprintln!("seq_len    : {seq_len}");
    eprintln!("num_steps  : {}", args.num_steps);

    // ── Sampler params ────────────────────────────────────────────────────
    let params = SamplerParams {
        num_steps: args.num_steps,
        guidance: GuidanceConfig {
            mode: CfgGuidanceMode::Independent,
            scale_text: 3.0,
            scale_speaker: args.cfg_speaker,
            scale_caption: 0.0,
            min_t: args.cfg_min_t,
            max_t: 1.0,
        },
        truncation_factor: None,
        temporal_rescale: None,
        speaker_kv: None,
        use_context_kv_cache: true,
    };

    // ── Synthetic inputs (batch=1, short reference) ───────────────────────
    let ref_frames = 8_usize;
    let text_len = 4_usize;
    let text_ids = Tensor::<B, 2, Int>::zeros([1, text_len], &device);
    let text_mask = Tensor::<B, 2, Bool>::from_data(
        burn::tensor::TensorData::new(vec![true; text_len], [1, text_len]),
        &device,
    );
    let ref_latent = Tensor::<B, 3>::zeros([1, ref_frames, cfg.latent_dim], &device);
    let ref_mask = Tensor::<B, 2, Bool>::from_data(
        burn::tensor::TensorData::new(vec![true; ref_frames], [1, ref_frames]),
        &device,
    );

    // One closure per run to avoid borrow issues with clones.
    let run_once = || -> Result<()> {
        let req = SamplingRequest {
            text_ids: text_ids.clone(),
            text_mask: text_mask.clone(),
            ref_latent: Some(ref_latent.clone()),
            ref_mask: Some(ref_mask.clone()),
            sequence_length: seq_len,
            caption_ids: None,
            caption_mask: None,
            initial_noise: None,
        };
        let output =
            sample_euler_rf_cfg(&model, req, &params, &device).context("sampler failed")?;
        // Force full execution (GPU synchronisation).
        let _ = output.into_data();
        Ok(())
    };

    // ── Warm-up ───────────────────────────────────────────────────────────
    if args.warmup > 0 {
        eprintln!("Warm-up ({} run(s)) …", args.warmup);
        for i in 0..args.warmup {
            run_once().with_context(|| format!("warm-up run {i} failed"))?;
        }
    }

    // ── Timed runs ────────────────────────────────────────────────────────
    if args.runs == 0 {
        bail!("--runs must be > 0");
    }
    eprintln!("Benchmarking ({} run(s)) …", args.runs);

    let mut times_ms: Vec<f64> = Vec::with_capacity(args.runs);
    for i in 0..args.runs {
        let t = Instant::now();
        run_once().with_context(|| format!("bench run {i} failed"))?;
        times_ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }

    // ── Statistics ────────────────────────────────────────────────────────
    let mean = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
    let mut sorted = times_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min_t = sorted[0];
    let max_t = sorted[sorted.len() - 1];
    let p50 = sorted[sorted.len() / 2];
    let p95_idx = ((sorted.len() as f64) * 0.95) as usize;
    let p95 = sorted[p95_idx.min(sorted.len() - 1)];

    println!();
    println!("=== Benchmark results ===");
    println!("Backend    : {backend_name}");
    println!("gpu_id     : {}", args.gpu_id);
    println!("seq_len    : {seq_len}");
    println!("num_steps  : {}", args.num_steps);
    println!("runs       : {}", args.runs);
    println!("mean       : {mean:.1} ms");
    println!("min        : {min_t:.1} ms");
    println!("p50        : {p50:.1} ms");
    println!("p95        : {p95:.1} ms");
    println!("max        : {max_t:.1} ms");
    for (i, t) in times_ms.iter().enumerate() {
        println!("  run[{i}]   : {t:.1} ms");
    }

    Ok(())
}
