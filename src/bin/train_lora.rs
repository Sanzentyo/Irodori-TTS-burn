//! LoRA fine-tuning CLI.
//!
//! Uses runtime backend dispatch via `--backend`.
//!
//! ```sh
//! just train-lora --backend libtorch --model model.safetensors \
//!                 --manifest train.jsonl --tokenizer tokenizer.json
//! ```

use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use irodori_tts_burn::{
    LoraTrainConfig, TrainingBackendKind, backend_config::BackendConfig, dispatch_training,
    train::train_lora,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "LoRA fine-tuning for Irodori-TTS")]
struct Cli {
    /// Training backend to use.
    #[arg(long)]
    backend: TrainingBackendKind,

    /// TOML config file.  When set, all other flags become optional overrides.
    #[arg(short = 'c', long)]
    config: Option<PathBuf>,

    /// Base model checkpoint (safetensors).
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// Training manifest JSONL.
    #[arg(short = 'M', long)]
    manifest: Option<PathBuf>,

    /// Hugging Face tokenizer JSON.
    #[arg(short = 't', long)]
    tokenizer: Option<PathBuf>,

    /// Output directory for adapter checkpoints.
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

    /// LoRA rank.
    #[arg(long)]
    lora_r: Option<usize>,

    /// LoRA alpha.
    #[arg(long)]
    lora_alpha: Option<f32>,

    /// Batch size.
    #[arg(short, long)]
    batch_size: Option<usize>,

    /// Peak learning rate.
    #[arg(long)]
    lr: Option<f64>,

    /// AdamW weight decay.
    #[arg(long)]
    weight_decay: Option<f64>,

    /// Warm-up steps.
    #[arg(long)]
    warmup_steps: Option<usize>,

    /// Total training steps.
    #[arg(long)]
    max_steps: Option<usize>,

    /// Log every N steps.
    #[arg(long)]
    log_every: Option<usize>,

    /// Save adapter checkpoint every N steps.
    #[arg(long)]
    save_every: Option<usize>,

    /// GPU device index (0-based).  Ignored for NdArray (CPU-only) backend.
    #[arg(long, default_value_t = 0)]
    gpu_id: u32,

    // ── New options ──────────────────────────────────────────────────────────
    /// Optional validation manifest JSONL.  When set, validation loss is
    /// computed every `--val-every` steps.
    #[arg(long)]
    val_manifest: Option<PathBuf>,

    /// Run validation every N optimiser steps.
    #[arg(long)]
    val_every: Option<usize>,

    /// Number of validation batches per eval (0 = full validation set).
    #[arg(long)]
    val_batches: Option<usize>,

    /// Disable epoch-level dataset shuffling.
    #[arg(long, default_value_t = false)]
    no_shuffle: bool,

    /// Seed for the shuffle RNG (for reproducibility).
    #[arg(long)]
    shuffle_seed: Option<u64>,

    /// Accumulate gradients over N micro-batches before an optimiser step.
    /// Effective batch size = batch_size × grad_accum_steps.
    #[arg(long)]
    grad_accum_steps: Option<usize>,

    /// Resume from an existing checkpoint directory (`output_dir/step-NNNNNNN/`).
    /// Only LoRA weights are restored; optimizer state resets (warm restart).
    #[arg(long)]
    resume_from: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    // Build config: load from TOML file if provided, else from CLI flags
    let mut cfg = if let Some(ref config_path) = cli.config {
        let content = std::fs::read_to_string(config_path)
            .with_context(|| format!("reading config file: {}", config_path.display()))?;
        toml::from_str::<LoraTrainConfig>(&content)
            .with_context(|| format!("parsing config file: {}", config_path.display()))?
    } else {
        // Without a config file, model/manifest/tokenizer are required
        let model = cli
            .model
            .clone()
            .context("--model is required when --config is not set")?;
        let manifest = cli
            .manifest
            .clone()
            .context("--manifest is required when --config is not set")?;
        let tokenizer = cli
            .tokenizer
            .clone()
            .context("--tokenizer is required when --config is not set")?;
        LoraTrainConfig {
            base_model_path: model,
            manifest_path: manifest,
            tokenizer_path: tokenizer,
            ..LoraTrainConfig::default()
        }
    };

    // CLI flags override config file values
    if let Some(ref m) = cli.model {
        cfg.base_model_path = m.clone();
    }
    if let Some(ref m) = cli.manifest {
        cfg.manifest_path = m.clone();
    }
    if let Some(ref t) = cli.tokenizer {
        cfg.tokenizer_path = t.clone();
    }
    if let Some(d) = cli.output_dir {
        cfg.output_dir = d;
    }
    if let Some(r) = cli.lora_r {
        cfg.lora.r = r;
    }
    if let Some(a) = cli.lora_alpha {
        cfg.lora.alpha = a;
    }
    if let Some(b) = cli.batch_size {
        cfg.batch_size = b;
    }
    if let Some(l) = cli.lr {
        cfg.lr = l;
    }
    if let Some(w) = cli.weight_decay {
        cfg.weight_decay = w;
    }
    if let Some(w) = cli.warmup_steps {
        cfg.warmup_steps = w;
    }
    if let Some(m) = cli.max_steps {
        cfg.max_steps = m;
    }
    if let Some(l) = cli.log_every {
        cfg.log_every = l;
    }
    if let Some(s) = cli.save_every {
        cfg.save_every = s;
    }
    if let Some(v) = cli.val_manifest {
        cfg.val_manifest = Some(v);
    }
    if let Some(v) = cli.val_every {
        cfg.val_every = v;
    }
    if let Some(v) = cli.val_batches {
        cfg.val_batches = v;
    }
    if cli.no_shuffle {
        cfg.shuffle = false;
    }
    if let Some(s) = cli.shuffle_seed {
        cfg.shuffle_seed = s;
    }
    if let Some(g) = cli.grad_accum_steps {
        cfg.grad_accum_steps = g;
    }
    if let Some(r) = cli.resume_from {
        cfg.resume_from = Some(r);
    }

    let backend = cli.backend;
    let gpu_id = cli.gpu_id;
    dispatch_training!(backend, gpu_id, |B, device| {
        tracing::info!(
            backend = <B as BackendConfig>::backend_label(),
            "training starting"
        );
        train_lora::<B>(&cfg, &device).context("training failed")
    })
}
