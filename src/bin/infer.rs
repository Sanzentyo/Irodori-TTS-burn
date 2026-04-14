//! Inference CLI for Irodori-TTS-burn.
//!
//! Takes a text prompt and an optional reference latent, runs the
//! RF diffusion model, and saves the resulting latent tensor as a
//! safetensors file (DACVAE decoding is not yet ported).
//!
//! # Minimal example
//! ```sh
//! just infer --backend libtorch-bf16 --checkpoint model.safetensors \
//!     --text "こんにちは" --output out.safetensors
//! ```

use std::{path::PathBuf, process};

use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};
use clap::Parser;
use half::f16;
use hf_hub::api::sync::Api;
use safetensors::{Dtype, SafeTensors};
use tokenizers::Tokenizer;
use tracing_subscriber::{EnvFilter, fmt};

use anyhow::{Context, Result, bail};
use irodori_tts_burn::{
    GuidanceConfig, InferenceBackendKind, InferenceBuilder, SamplerParams, SamplingRequest,
    backend_config::BackendConfig, dispatch_inference,
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "infer",
    about = "Run Irodori-TTS-burn inference",
    long_about = "Runs the RF diffusion model and saves the output latent as a safetensors file."
)]
struct Args {
    /// Inference backend to use.
    #[arg(long)]
    backend: InferenceBackendKind,

    /// Path to a Burn-converted safetensors checkpoint.
    ///
    /// Run `just convert <src> <dst>` first if your checkpoint still has the
    /// Python-style `cond_module.{0,2,4}` key names.
    #[arg(short, long)]
    checkpoint: PathBuf,

    /// Text to synthesise.
    #[arg(short, long)]
    text: String,

    /// Optional reference audio latent (safetensors file with a tensor named "latent").
    ///
    /// Shape must be `[1, T, latent_dim]` — the **unpatched** latent dimension
    /// from the model config.  Patching is applied internally by the speaker encoder.
    #[arg(long)]
    ref_latent: Option<PathBuf>,

    /// Path to write the output latent safetensors file.
    #[arg(short, long, default_value = "output.safetensors")]
    output: PathBuf,

    /// Number of diffusion steps.
    #[arg(long, default_value_t = 32)]
    num_steps: usize,

    /// CFG scale for text conditioning.
    #[arg(long, default_value_t = 3.0)]
    cfg_text: f32,

    /// CFG scale for speaker conditioning.
    #[arg(long, default_value_t = 5.0)]
    cfg_speaker: f32,

    /// CFG scale for caption conditioning.
    #[arg(long, default_value_t = 1.0)]
    cfg_caption: f32,

    /// CFG guidance mode: independent | joint | alternating.
    #[arg(long, default_value = "independent")]
    cfg_mode: String,

    /// Output sequence length in frames.
    ///
    /// Defaults to the `fixed_target_latent_steps` value in the checkpoint
    /// metadata (750 for `Aratako/Irodori-TTS-500M-v2`), or 256 if absent.
    #[arg(long)]
    seq_len: Option<usize>,

    /// Minimum timestep for CFG (0.0–1.0).
    #[arg(long, default_value_t = 0.5)]
    cfg_min_t: f32,

    /// GPU device index (0-based).
    ///
    /// For CUDA/LibTorch backends selects the CUDA device.
    /// For WGPU selects the Nth discrete GPU.
    /// For CPU backends this is ignored.
    #[arg(long, default_value_t = 0)]
    gpu_id: u32,

    /// Maximum timestep for CFG (0.0–1.0).
    #[arg(long, default_value_t = 1.0)]
    cfg_max_t: f32,

    /// Optional directory containing a PEFT LoRA adapter
    /// (`adapter_config.json` + `adapter_model.safetensors`).
    ///
    /// When provided the adapter is merged into the base weights at load time.
    /// Requires the `lora` feature.
    #[arg(long)]
    #[cfg(feature = "lora")]
    adapter: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Download the tokenizer for `repo_id` via hf-hub and load it.
fn load_tokenizer(repo_id: &str) -> Result<Tokenizer> {
    let api = Api::new().context("failed to initialise HF Hub API")?;
    let repo = api.model(repo_id.to_string());
    let path = repo
        .get("tokenizer.json")
        .context("failed to fetch tokenizer.json from HF Hub")?;
    Tokenizer::from_file(path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer from file: {e}"))
}

/// Tokenise `text` and return `(input_ids, mask)` as 2-D tensors `[1, T]`.
fn tokenize<B: Backend>(
    tokenizer: &Tokenizer,
    text: &str,
    add_bos: bool,
    device: &B::Device,
) -> Result<(Tensor<B, 2, Int>, Tensor<B, 2, Bool>)> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("failed to tokenise text: {e}"))?;

    let mut ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    if add_bos && let Some(bos_id) = tokenizer.token_to_id("<s>") {
        ids.insert(0, bos_id as i32);
    }
    let seq_len = ids.len();

    let input_ids = Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [1, seq_len]), device);

    // All positions are valid (padding not applied here).
    let mask: Tensor<B, 2, Bool> = Tensor::<B, 2>::ones([1, seq_len], device).greater_elem(0.0f32);

    Ok((input_ids, mask))
}

/// Load a reference latent from a safetensors file.
///
/// Expects a tensor named `"latent"` with shape `[1, T, D]`.
fn load_ref_latent<B: Backend>(
    path: &std::path::Path,
    device: &B::Device,
) -> Result<(Tensor<B, 3>, Tensor<B, 2, Bool>)> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read ref_latent file {:?}", path))?;
    let st = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("malformed safetensors file {:?}", path))?;
    let view = st
        .tensor("latent")
        .with_context(|| "missing 'latent' tensor in ref_latent safetensors file")?;

    let shape = view.shape();
    if shape.len() != 3 {
        bail!(
            "expected 3-D latent tensor [batch, seq, dim], got {} dimensions",
            shape.len()
        );
    }
    let [batch, seq, dim] = [shape[0], shape[1], shape[2]];
    let floats: Vec<f32> = match view.dtype() {
        Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|b| f32::from_le_bytes([0, 0, b[0], b[1]]))
            .collect(),
        Dtype::F16 => view
            .data()
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        dtype => bail!("unsupported latent dtype: {dtype:?}"),
    };

    let tensor = Tensor::<B, 3>::from_data(TensorData::new(floats, [batch, seq, dim]), device);
    let mask: Tensor<B, 2, Bool> = Tensor::<B, 2>::ones([batch, seq], device).greater_elem(0.0f32);

    Ok((tensor, mask))
}

/// Parse the CFG mode string, delegating to the [`FromStr`] impl on `CfgGuidanceMode`.
fn parse_cfg_mode(s: &str) -> Result<irodori_tts_burn::CfgGuidanceMode> {
    s.parse()
        .with_context(|| format!("invalid CFG guidance mode '{s}'"))
}

/// Serialise a `[batch, seq, dim]` tensor to a safetensors file.
///
/// Always writes as f32 regardless of the backend float type, so consumers
/// get a consistent, portable output file.
fn save_output_safetensors<B: burn::tensor::backend::Backend>(
    path: &std::path::Path,
    output: burn::tensor::Tensor<B, 3>,
    batch: usize,
    seq: usize,
    dim: usize,
) -> Result<()> {
    use safetensors::tensor::{Dtype, TensorView};

    // Cast to f32 before extracting bytes; this is a no-op when B::FloatElem
    // is already f32 and an explicit narrowing conversion for bf16 / f16.
    let data = output.into_data().convert::<f32>();

    let view = TensorView::new(Dtype::F32, vec![batch, seq, dim], data.as_bytes())
        .context("failed to create safetensors TensorView for output")?;

    let serialised = safetensors::tensor::serialize([("latent", view)], None)
        .context("failed to serialize output as safetensors")?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, &serialised)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn run<B: BackendConfig>(args: Args, device: B::Device) -> Result<()> {
    tracing::info!("Loading model from {:?}", args.checkpoint);
    #[cfg(feature = "lora")]
    let loaded = match args.adapter {
        Some(ref adapter_dir) => {
            tracing::info!("Merging LoRA adapter from {:?}", adapter_dir);
            InferenceBuilder::<B, _>::new(device.clone())
                .load_weights_with_adapter(&args.checkpoint, adapter_dir)?
        }
        None => InferenceBuilder::<B, _>::new(device.clone()).load_weights(&args.checkpoint)?,
    };
    #[cfg(not(feature = "lora"))]
    let loaded = InferenceBuilder::<B, _>::new(device.clone()).load_weights(&args.checkpoint)?;
    let cfg = loaded.model_config().clone();
    tracing::info!("Model loaded. Config: {:?}", cfg);

    tracing::info!("Loading tokenizer from HF Hub: {}", cfg.text_tokenizer_repo);
    let tokenizer = load_tokenizer(&cfg.text_tokenizer_repo)?;
    let (text_ids, text_mask) = tokenize::<B>(&tokenizer, &args.text, cfg.text_add_bos, &device)?;
    tracing::info!("Text tokenised: {} tokens", text_ids.dims()[1]);

    let (ref_latent, ref_mask) = match args.ref_latent {
        Some(ref path) => {
            let (t, m) = load_ref_latent::<B>(path, &device)?;
            let dims = t.dims();
            tracing::info!("Reference latent loaded: {:?}", dims);
            // Validate the latent dimension matches the model config.
            if dims[2] != cfg.latent_dim {
                bail!(
                    "ref_latent last dim {} != model latent_dim {}",
                    dims[2],
                    cfg.latent_dim
                );
            }
            (Some(t), Some(m))
        }
        None => {
            tracing::info!("No reference latent provided — unconditional speaker mode.");
            (None, None)
        }
    };

    let seq_len = args
        .seq_len
        .or(cfg.fixed_target_latent_steps)
        .unwrap_or(256);
    let cfg_mode = parse_cfg_mode(&args.cfg_mode)?;
    let params = SamplerParams {
        num_steps: args.num_steps,
        guidance: GuidanceConfig {
            mode: cfg_mode,
            scale_text: args.cfg_text,
            scale_caption: args.cfg_caption,
            scale_speaker: args.cfg_speaker,
            min_t: args.cfg_min_t,
            max_t: args.cfg_max_t,
        },
        ..SamplerParams::default()
    };

    let engine = loaded.with_sampling(params).build();

    tracing::info!(
        "Running sampler: {} steps, seq_len={seq_len}, cfg_mode={}",
        engine.sampling_params().num_steps,
        args.cfg_mode
    );

    let output = engine.sample(SamplingRequest {
        text_ids,
        text_mask,
        ref_latent,
        ref_mask,
        sequence_length: seq_len,
        caption_ids: None,
        caption_mask: None,
        initial_noise: None,
    })?;

    let [batch, seq, dim] = output.dims();
    tracing::info!("Sampler complete. Output shape: [{batch}, {seq}, {dim}]");

    save_output_safetensors::<B>(&args.output, output, batch, seq, dim)?;
    tracing::info!("Output written to {:?}", args.output);
    // WGPU/Vulkan atexit handlers segfault during normal process exit; use _exit
    // to bypass all atexit handlers and let the OS reclaim resources.
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    unsafe extern "C" {
        fn _exit(status: i32) -> !;
    }
    unsafe { _exit(0) }
}

fn main() {
    fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

    let args = Args::parse();
    let backend = args.backend;
    let gpu_id = args.gpu_id;
    let result = dispatch_inference!(backend, gpu_id, |B, device| run::<B>(args, device));
    if let Err(e) = result {
        tracing::error!("Fatal: {e}");
        process::exit(1);
    }
    // Note: run() calls _exit(0) on success, so we typically never reach here.
}
