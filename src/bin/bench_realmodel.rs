//! Multi-backend real-model benchmark.
//!
//! Times full-model inference over the converted `Aratako/Irodori-TTS-500M-v2`
//! checkpoint. Backend is selected at runtime via `--backend`:
//!
//! ```sh
//! just bench-tch          # --backend libtorch
//! just bench-tch-bf16     # --backend libtorch-bf16
//! just bench-cuda         # --backend cuda
//! just bench-wgpu         # --backend wgpu
//! ```
//!
//! Sequence length defaults to the `fixed_target_latent_steps` value in the
//! checkpoint metadata (750 for the 500M model).

// WGPU pulls in very deeply-nested types; raise recursion limit as needed.
#![recursion_limit = "512"]

// ── Imports ───────────────────────────────────────────────────────────────

use std::{path::Path, time::Instant};

use anyhow::{Context, Result, bail};
use burn::tensor::{Bool, Int, Tensor};
use clap::Parser;

use irodori_tts_burn::{
    CfgGuidanceMode, GuidanceConfig, InferenceBackendKind, InferenceOptimizedModel, SamplerMethod,
    SamplerParams, SamplingRequest, backend_config::BackendConfig, dispatch_inference, load_model,
    sample_euler_rf_cfg,
};

// ── CLI ───────────────────────────────────────────────────────────────────

#[derive(Debug, Parser)]
#[command(
    name = "bench_realmodel",
    about = "Time inference with the real converted checkpoint (multi-backend)"
)]
struct Args {
    /// Inference backend to use.
    #[arg(long)]
    backend: InferenceBackendKind,

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

    /// CFG text scale.
    #[arg(long, default_value_t = 3.0)]
    cfg_text: f32,

    /// Minimum timestep for CFG application.
    #[arg(long, default_value_t = 0.5)]
    cfg_min_t: f32,

    /// CFG guidance mode: independent (default), joint, or alternating.
    ///
    /// - independent: batched forward with all conditioned + unconditioned variants
    ///   concatenated along batch dim (batch = 1 + #active_signals).
    /// - joint: two separate passes (cond + shared-uncond KV cache); requires
    ///   all active CFG scales to be equal.
    /// - alternating: alternates which signal is unconditioned each step.
    #[arg(long, default_value = "independent", value_name = "MODE")]
    cfg_mode: String,

    /// Disable context KV cache (re-encodes conditions every step).
    ///
    /// By default the conditioned K/V pairs for text and speaker tokens are
    /// pre-computed once and reused across all diffusion steps.
    #[arg(long)]
    no_kv_cache: bool,

    /// ODE integration method: euler (default) or heun.
    ///
    /// With heun, use half the steps (e.g. 20) for the same NFE budget as euler (40).
    #[arg(long, default_value = "euler", value_name = "METHOD")]
    sampler: String,

    /// GPU device index (0-based).
    ///
    /// For CUDA/LibTorch backends selects the CUDA device.
    /// For WGPU selects the Nth discrete GPU.
    #[arg(long, default_value_t = 0)]
    gpu_id: u32,
}

// ── Main ──────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();
    let label = args.backend.label();

    #[cfg(feature = "cli")]
    if matches!(
        args.backend,
        InferenceBackendKind::LibTorchMps | InferenceBackendKind::LibTorchMpsF16
    ) {
        anyhow::ensure!(
            tch::utils::has_mps(),
            "MPS device is not available on this machine"
        );
    }

    dispatch_inference!(args.backend, args.gpu_id, |B, device| {
        run::<B>(args, device, label)
    })
}

fn run<B: BackendConfig>(args: Args, device: B::Device, backend_name: &str) -> Result<()> {
    // Disable LibTorch autograd globally — mirrors Python's `torch.no_grad()`.
    // Even though burn tensors don't set requires_grad, the global GradMode flag
    // still causes measurable C++ dispatch overhead (~1.5% for f32 inference).
    let _no_grad = tch::no_grad_guard();

    B::check_requirements(&device).map_err(|e| anyhow::anyhow!("{e}"))?;
    eprintln!("Backend    : {backend_name}");
    eprintln!("Checkpoint : {}", args.checkpoint);

    // ── Load model ────────────────────────────────────────────────────────
    eprintln!("Loading model …");
    let t_load = Instant::now();
    let (model, cfg) = load_model::<B>(Path::new(&args.checkpoint), &device)
        .context("failed to load model — run `just convert-model` first")?;
    let load_ms = t_load.elapsed().as_millis();
    eprintln!("Model loaded in {load_ms} ms");

    // Fuse QKV + SwiGLU weights via type-safe InferenceOptimizedModel
    let model = InferenceOptimizedModel::new(model);

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
    let cfg_mode = match args.cfg_mode.to_ascii_lowercase().as_str() {
        "independent" => CfgGuidanceMode::Independent,
        "joint" => CfgGuidanceMode::Joint,
        "alternating" => CfgGuidanceMode::Alternating,
        other => bail!("unknown --cfg-mode '{other}'; expected independent, joint, or alternating"),
    };
    let sampler_method = args
        .sampler
        .parse::<SamplerMethod>()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let params = SamplerParams {
        num_steps: args.num_steps,
        method: sampler_method,
        guidance: GuidanceConfig {
            mode: cfg_mode,
            scale_text: args.cfg_text,
            scale_speaker: args.cfg_speaker,
            scale_caption: 0.0,
            min_t: args.cfg_min_t,
            max_t: 1.0,
        },
        truncation_factor: None,
        temporal_rescale: None,
        speaker_kv: None,
        use_context_kv_cache: !args.no_kv_cache,
    };
    eprintln!("cfg_mode   : {}", args.cfg_mode);
    eprintln!("sampler    : {sampler_method}");
    let active_signals = {
        let mut v = Vec::new();
        if args.cfg_text != 0.0 {
            v.push(format!("text={}", args.cfg_text));
        }
        if args.cfg_speaker != 0.0 {
            v.push(format!("speaker={}", args.cfg_speaker));
        }
        v
    };
    eprintln!(
        "cfg_batch  : 1+{} = {} (active signals: {})",
        active_signals.len(),
        1 + active_signals.len(),
        if active_signals.is_empty() {
            "none".to_owned()
        } else {
            active_signals.join(", ")
        }
    );
    eprintln!("kv_cache   : {}", !args.no_kv_cache);

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

    // ── Throughput metrics ────────────────────────────────────────────────
    const LATENT_FRAME_RATE_HZ: f64 = 25.0; // 48000 / 1920
    let audio_duration_s = seq_len as f64 / LATENT_FRAME_RATE_HZ;
    let rtf_mean = mean / 1000.0 / audio_duration_s;
    let rtf_min = min_t / 1000.0 / audio_duration_s;
    let xrt_mean = 1.0 / rtf_mean;
    let evals_per_step = match sampler_method {
        SamplerMethod::Euler => 1,
        SamplerMethod::Heun => 2,
    };
    let nfe = args.num_steps * evals_per_step;
    let tokens_per_sec = (seq_len * nfe) as f64 / mean * 1000.0;

    println!();
    println!("=== Benchmark results ===");
    println!("Backend    : {backend_name}");
    println!("gpu_id     : {}", args.gpu_id);
    println!("seq_len    : {seq_len}");
    println!("num_steps  : {}", args.num_steps);
    println!("sampler    : {sampler_method}  (NFE={nfe})");
    println!("runs       : {}", args.runs);
    println!("mean       : {mean:.1} ms");
    println!("min        : {min_t:.1} ms");
    println!("p50        : {p50:.1} ms");
    println!("p95        : {p95:.1} ms");
    println!("max        : {max_t:.1} ms");
    for (i, t) in times_ms.iter().enumerate() {
        println!("  run[{i}]   : {t:.1} ms");
    }
    println!();
    println!("=== Throughput ===");
    println!("audio_dur  : {audio_duration_s:.1} s  (seq={seq_len}, frame_rate=25 Hz)");
    println!("RTF (mean) : {rtf_mean:.3}  (<1 = faster than real-time)");
    println!("RTF (min)  : {rtf_min:.3}");
    println!("xRT (mean) : {xrt_mean:.2}×  (>1 = faster than real-time)");
    println!("tokens/s   : {tokens_per_sec:.0}  (NFE × seq_len / s)");

    // Structured JSON line for automated parsing by comparison scripts.
    println!(
        "{{\"bench_result\":{{\"backend\":{backend_name:?},\
        \"seq_len\":{seq_len},\"num_steps\":{},\"sampler\":\"{sampler_method}\",\"nfe\":{nfe},\
        \"mean_ms\":{mean:.3},\"min_ms\":{min_t:.3},\
        \"p50_ms\":{p50:.3},\"p95_ms\":{p95:.3},\
        \"audio_duration_s\":{audio_duration_s:.3},\
        \"rtf_mean\":{rtf_mean:.6},\"xrt_mean\":{xrt_mean:.6},\
        \"tokens_per_sec\":{tokens_per_sec:.1}}}}}",
        args.num_steps,
    );

    Ok(())
}
