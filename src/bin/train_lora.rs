//! LoRA fine-tuning CLI.
//!
//! Uses the same backend-feature mechanism as `infer.rs`.  The default is
//! NdArray (CPU).  Pass `--features backend_tch` for LibTorch (GPU via cuBLAS),
//! `--features backend_cuda` for CubeCL CUDA, etc.
//!
//! ```sh
//! just train-lora --model model.safetensors --manifest train.jsonl \
//!                 --tokenizer tokenizer.json
//! ```

// ── Mutually-exclusive backend guards (mirrors infer.rs) ────────────────────
#[cfg(any(
    all(feature = "backend_cuda", feature = "backend_cuda_bf16"),
    all(feature = "backend_cuda", feature = "backend_tch"),
    all(feature = "backend_cuda", feature = "backend_tch_bf16"),
    all(feature = "backend_cuda_bf16", feature = "backend_tch"),
    all(feature = "backend_cuda_bf16", feature = "backend_tch_bf16"),
    all(feature = "backend_tch", feature = "backend_tch_bf16"),
))]
compile_error!("only one backend feature may be selected at a time");

// ── Backend / device type aliases ────────────────────────────────────────────
#[cfg(feature = "backend_cuda")]
type BaseB = burn::backend::Cuda;
#[cfg(feature = "backend_cuda_bf16")]
type BaseB = burn::backend::Cuda<half::bf16>;
#[cfg(feature = "backend_tch")]
type BaseB = burn::backend::LibTorch;
#[cfg(feature = "backend_tch_bf16")]
type BaseB = burn::backend::LibTorch<half::bf16>;
#[cfg(not(any(
    feature = "backend_cuda",
    feature = "backend_cuda_bf16",
    feature = "backend_tch",
    feature = "backend_tch_bf16",
)))]
type BaseB = burn::backend::NdArray;

// Training requires AutodiffBackend
type B = burn::backend::Autodiff<BaseB>;

use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use irodori_tts_burn::{
    LoraConfig, LoraTrainConfig, backend_config::BackendConfig, train::train_lora,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "LoRA fine-tuning for Irodori-TTS")]
struct Cli {
    /// Base model checkpoint (safetensors).
    #[arg(short, long)]
    model: PathBuf,

    /// Training manifest JSONL.
    #[arg(short = 'M', long)]
    manifest: PathBuf,

    /// Hugging Face tokenizer JSON.
    #[arg(short = 't', long)]
    tokenizer: PathBuf,

    /// Output directory for adapter checkpoints.
    #[arg(short, long, default_value = "output")]
    output_dir: PathBuf,

    /// LoRA rank.
    #[arg(long, default_value_t = 8)]
    lora_r: usize,

    /// LoRA alpha.
    #[arg(long, default_value_t = 16.0)]
    lora_alpha: f32,

    /// Batch size.
    #[arg(short, long, default_value_t = 4)]
    batch_size: usize,

    /// Peak learning rate.
    #[arg(long, default_value_t = 1e-4)]
    lr: f64,

    /// AdamW weight decay.
    #[arg(long, default_value_t = 0.01)]
    weight_decay: f64,

    /// Warm-up steps.
    #[arg(long, default_value_t = 100)]
    warmup_steps: usize,

    /// Total training steps.
    #[arg(long, default_value_t = 5000)]
    max_steps: usize,

    /// Log every N steps.
    #[arg(long, default_value_t = 10)]
    log_every: usize,

    /// Save adapter checkpoint every N steps.
    #[arg(long, default_value_t = 500)]
    save_every: usize,

    /// GPU device index (0-based).  Ignored for NdArray (CPU-only) backend.
    #[arg(long, default_value_t = 0)]
    gpu_id: u32,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();

    let cfg = LoraTrainConfig {
        manifest_path: cli.manifest,
        output_dir: cli.output_dir,
        base_model_path: cli.model,
        tokenizer_path: cli.tokenizer,
        lora: LoraConfig {
            r: cli.lora_r,
            alpha: cli.lora_alpha,
            target_modules: vec![
                "wq".to_owned(),
                "wk".to_owned(),
                "wv".to_owned(),
                "wo".to_owned(),
                "gate".to_owned(),
            ],
        },
        batch_size: cli.batch_size,
        lr: cli.lr,
        weight_decay: cli.weight_decay,
        warmup_steps: cli.warmup_steps,
        max_steps: cli.max_steps,
        log_every: cli.log_every,
        save_every: cli.save_every,
        t_mean: 0.0,
        t_std: 1.0,
    };

    let device = <BaseB as BackendConfig>::device_from_id(cli.gpu_id);
    tracing::info!(
        backend = <BaseB as BackendConfig>::backend_label(),
        "training starting"
    );
    train_lora::<B>(&cfg, &device).context("training failed")
}
