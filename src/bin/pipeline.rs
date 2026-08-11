//! End-to-end Irodori-TTS pipeline: text → WAV.
//!
//! Chains the RF diffusion model with the DACVAE codec to produce an output
//! waveform from a text prompt.  Reference audio (for speaker conditioning)
//! is optional; when omitted, the model operates in unconditional speaker mode.
//!
//! # Example
//! ```sh
//! just pipeline \
//!     --backend wgpu-wgsl \
//!     --checkpoint model.safetensors \
//!     --codec-weights target/dacvae_weights.safetensors \
//!     --text "こんにちは" \
//!     --output output.wav
//! ```

use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process,
    time::Instant,
};

use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};
use clap::Parser;
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;
use tracing_subscriber::{EnvFilter, fmt};

use anyhow::{Context, Result, bail};
use irodori_tts_wgpu::codec::{
    DACVAE_HOP_LENGTH, DACVAE_LATENT_DIM, DACVAE_SAMPLE_RATE, DacVaeCodec,
};
use irodori_tts_wgpu::{
    AuxConditionInput, EncodedCondition, GuidanceConfig, InferenceBackendKind, InferenceBuilder,
    SamplerMethod, SamplerParams, SamplerWorkReport, SamplingRequest, WgpuRaw, WgslInferenceEngine,
    load_codec, unpatchify_latent,
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "pipeline",
    about = "End-to-end Irodori-TTS: text → WAV",
    long_about = "Runs the RF diffusion model then decodes the latent with the DACVAE \
                  codec to produce a WAV file."
)]
struct Args {
    /// Production execution policy. Only fused FP32 WGSL is available.
    #[arg(long, default_value = "wgpu-wgsl")]
    backend: InferenceBackendKind,

    /// Path to the RF model safetensors checkpoint.
    #[arg(short, long)]
    checkpoint: PathBuf,

    /// Path to the DACVAE codec safetensors weights.
    #[arg(long)]
    codec_weights: PathBuf,

    /// Text to synthesise.
    #[arg(short, long)]
    text: String,

    /// Optional voice/style caption. Empty or omitted captions are encoded
    /// with an all-false mask for v4 text-only inference.
    #[arg(long)]
    caption: Option<String>,

    /// Optional reference audio WAV for speaker conditioning.
    ///
    /// Multi-channel input is mixed to mono and input at another sample rate is
    /// linearly resampled. This compatibility path is not parity-equivalent to
    /// official v4 reference preprocessing, which uses audiotools loudness
    /// normalization and torchaudio's band-limited resampler. When omitted,
    /// speaker conditioning is disabled and this limitation does not apply.
    #[arg(long)]
    ref_audio: Option<PathBuf>,

    /// Output WAV file path.
    #[arg(short, long, default_value = "output.wav")]
    output: PathBuf,

    /// Number of RF diffusion steps.
    #[arg(long, default_value_t = 40)]
    num_steps: usize,

    /// ODE solver: euler (1st-order) or heun (2nd-order trapezoidal).
    ///
    /// Heun with N steps performs 2N forward passes (NFE=2N), giving higher
    /// quality than Euler with N steps at the same wall-clock cost as Euler
    /// with 2N steps.  For NFE parity set `--num-steps 20 --sampler heun`.
    #[arg(long, default_value = "euler")]
    sampler: SamplerMethod,

    /// CFG scale for text conditioning.
    #[arg(long, default_value_t = 3.0)]
    cfg_text: f32,

    /// CFG scale for speaker conditioning.
    #[arg(long, default_value_t = 5.0)]
    cfg_speaker: f32,

    /// CFG scale for caption conditioning.
    #[arg(long, default_value_t = 3.0)]
    cfg_caption: f32,

    /// CFG guidance mode: independent | joint | alternating.
    #[arg(long, default_value = "independent")]
    cfg_mode: String,

    /// Output sequence length in latent frames (patched).
    ///
    /// Defaults to `fixed_target_latent_steps` from the checkpoint metadata,
    /// or 256 if absent.
    #[arg(long)]
    seq_len: Option<usize>,

    /// Requested output duration in seconds.
    ///
    /// Mutually exclusive with `--seq-len`. When omitted, v4 checkpoints use
    /// their learned duration predictor.
    #[arg(long)]
    seconds: Option<f64>,

    /// Scale applied to learned duration predictions.
    #[arg(long, default_value_t = 1.0)]
    duration_scale: f64,

    /// Minimum manual or predicted output duration.
    #[arg(long, default_value_t = 0.5)]
    min_seconds: f64,

    /// Maximum manual or predicted output duration.
    #[arg(long, default_value_t = 30.0)]
    max_seconds: f64,

    /// Minimum timestep for CFG (0.0–1.0).
    #[arg(long, default_value_t = 0.5)]
    cfg_min_t: f32,

    /// Maximum timestep for CFG (0.0–1.0).
    #[arg(long, default_value_t = 1.0)]
    cfg_max_t: f32,

    /// Optional LoRA adapter directory (must contain `adapter_config.json`
    /// and `adapter_model.safetensors`).
    /// Requires the `lora` feature.
    #[arg(long)]
    #[cfg(feature = "lora")]
    adapter: Option<PathBuf>,

    /// GPU device index (0-based).
    #[arg(long, default_value_t = 0)]
    gpu_id: u32,

    /// Explicit WGPU discrete-adapter index.
    ///
    /// Unlike legacy `--gpu-id 0`, which means WGPU's default adapter, this
    /// can select `DiscreteGpu(0)`. Adapter ordering is WGPU-specific and may
    /// differ from NVML/CUDA ordering.
    #[arg(long)]
    wgpu_adapter_index: Option<usize>,

    /// New JSON path for the host-visible RF work report from this synthesis.
    ///
    /// This validation-only output is accepted only with `--backend wgpu-wgsl`.
    /// The parent directory must already exist and the path must not exist.
    /// When omitted, the production sampler hot path remains unchanged.
    #[arg(long, value_name = "PATH")]
    rf_work_manifest_out: Option<PathBuf>,

    /// Trim trailing silence using the find-flattening-point heuristic.
    ///
    /// Enabled by default (mirrors Python infer.py). Pass
    /// `--trim-tail false` to disable.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    trim_tail: bool,

    /// Sliding window size (in latent frames) used for tail trimming.
    #[arg(long, default_value_t = 20)]
    trim_window: usize,

    /// Std threshold below which a window is considered flat (tail trimming).
    #[arg(long, default_value_t = 0.05)]
    trim_std: f32,

    /// Mean absolute threshold below which a window is considered near-zero (tail trimming).
    #[arg(long, default_value_t = 0.1)]
    trim_mean: f32,

    /// Random seed for backend-generated initial noise.
    ///
    /// This is reproducible within a fixed backend/version, but Burn and
    /// PyTorch use different RNG implementations. Use `--noise-file` for an
    /// exact cross-runtime comparison.
    #[arg(long)]
    seed: Option<u64>,

    /// Pre-computed initial noise tensor file (safetensors, key "initial_noise").
    ///
    /// When supplied, overrides `--seed`; the tensor is used directly as the
    /// starting latent x_T. The pinned v4 oracle exporter writes this tensor.
    #[arg(long)]
    noise_file: Option<PathBuf>,

    /// Local path to a `tokenizer.json` file.
    ///
    /// When supplied, the tokenizer is loaded from disk and no network access
    /// is needed.  When omitted, the tokenizer is fetched from Hugging Face Hub
    /// using the repo ID and optional text-encoder revision stored in the
    /// checkpoint metadata.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Load a pre-computed initial noise tensor from a safetensors file.
///
/// Returns `None` when `path` is `None` so the RF sampler generates noise internally.
fn load_initial_noise<B: Backend>(
    path: Option<&std::path::Path>,
    seq_len: usize,
    patched_latent_dim: usize,
    device: &B::Device,
) -> Result<Option<Tensor<B, 3>>> {
    let Some(p) = path else {
        return Ok(None);
    };
    let bytes = std::fs::read(p).with_context(|| format!("failed to read noise file {p:?}"))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .with_context(|| format!("invalid safetensors file {p:?}"))?;
    let tv = st
        .tensor("initial_noise")
        .with_context(|| "key 'initial_noise' not found in noise file")?;
    let shape = tv.shape().to_vec();
    anyhow::ensure!(
        shape == [1, seq_len, patched_latent_dim],
        "initial_noise must have shape [1, {seq_len}, {patched_latent_dim}], got {shape:?}"
    );
    anyhow::ensure!(
        tv.dtype() == safetensors::Dtype::F32,
        "initial_noise must use f32 storage, got {:?}",
        tv.dtype()
    );
    let data: Vec<f32> = tv
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(data, [shape[0], shape[1], shape[2]]),
        device,
    );
    tracing::info!("Loaded initial noise from {p:?}: shape={shape:?}");
    Ok(Some(tensor))
}

/// Locate the tokenizer shipped alongside a v4 checkpoint snapshot.
///
/// The released Hugging Face snapshot stores it at `tokenizer/tokenizer.json`.
/// Directly adjacent files are accepted as well for manually bundled models.
fn bundled_tokenizer_path(checkpoint: &std::path::Path) -> Option<PathBuf> {
    let parent = checkpoint.parent()?;
    let mut candidates = vec![
        parent.join("tokenizer").join("tokenizer.json"),
        parent.join("tokenizer.json"),
    ];
    if let Some(grandparent) = parent.parent() {
        candidates.push(grandparent.join("tokenizer").join("tokenizer.json"));
    }
    candidates.into_iter().find(|path| path.is_file())
}

/// Load a tokenizer from a local path or, as fallback, from Hugging Face Hub.
///
/// When `local_path` is `Some`, the file is read directly — no network
/// access required.  When `None`, `repo_id` is used to fetch
/// `tokenizer.json` via hf-hub (which caches the file locally on first use).
/// When the checkpoint pins a text-encoder revision, the tokenizer fetch uses
/// that same immutable revision rather than the repository's mutable default
/// branch.
fn load_tokenizer(
    local_path: Option<&std::path::Path>,
    repo_id: &str,
    revision: Option<&str>,
) -> Result<Tokenizer> {
    if let Some(path) = local_path {
        tracing::info!("Loading tokenizer from local path: {path:?}");
        Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer from {path:?}: {e}"))
    } else {
        tracing::info!(
            "Fetching tokenizer from HF Hub: {repo_id} (revision={})",
            revision.unwrap_or("main")
        );
        let api = Api::new().context("failed to initialise HF Hub API")?;
        let repo = match revision {
            Some(revision) => api.repo(Repo::with_revision(
                repo_id.to_string(),
                RepoType::Model,
                revision.to_string(),
            )),
            None => api.model(repo_id.to_string()),
        };
        let cached = repo
            .get("tokenizer.json")
            .context("failed to fetch tokenizer.json from HF Hub")?;
        Tokenizer::from_file(cached)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer from HF Hub cache: {e}"))
    }
}

struct Tokenized<B: Backend> {
    ids: Tensor<B, 2, Int>,
    mask: Tensor<B, 2, Bool>,
    valid_tokens: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct TokenRow {
    ids: Vec<i32>,
    mask: Vec<bool>,
}

fn prepare_token_row(
    body_ids: &[u32],
    bos_id: Option<u32>,
    pad_id: u32,
    max_length: Option<usize>,
    pad_to_max: bool,
    force_all_false: bool,
) -> Result<TokenRow> {
    if pad_to_max {
        anyhow::ensure!(max_length.is_some(), "fixed padding requires max_length");
    }
    if let Some(max_length) = max_length {
        anyhow::ensure!(max_length > 0, "token max_length must be > 0");
    }

    let bos_slots = usize::from(bos_id.is_some());
    let body_limit = max_length
        .map(|limit| limit.saturating_sub(bos_slots))
        .unwrap_or(body_ids.len());
    let mut ids = Vec::with_capacity(max_length.unwrap_or(body_ids.len() + bos_slots));
    if let Some(bos_id) = bos_id {
        ids.push(i32::try_from(bos_id).context("BOS token id exceeds i32")?);
    }
    ids.extend(
        body_ids
            .iter()
            .take(body_limit)
            .copied()
            .map(i32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("token id exceeds i32")?,
    );

    let mut mask = vec![true; ids.len()];
    if ids.is_empty() {
        ids.push(i32::try_from(pad_id).context("pad token id exceeds i32")?);
        mask.push(false);
    }
    if pad_to_max {
        let max_length = max_length.context("fixed padding requires max_length")?;
        ids.resize(
            max_length,
            i32::try_from(pad_id).context("pad token id exceeds i32")?,
        );
        mask.resize(max_length, false);
    }
    if force_all_false {
        mask.fill(false);
    }
    Ok(TokenRow { ids, mask })
}

fn tokenize<B: Backend>(
    tokenizer: &Tokenizer,
    text: &str,
    add_bos: bool,
    max_length: Option<usize>,
    pad_to_max: bool,
    force_all_false: bool,
    device: &B::Device,
) -> Result<Tokenized<B>> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("failed to tokenise: {e}"))?;

    let bos_id = if add_bos {
        Some(
            tokenizer
                .token_to_id("<s>")
                .or_else(|| tokenizer.token_to_id("<bos>"))
                .or_else(|| tokenizer.token_to_id("[CLS]"))
                .context("tokenizer has no BOS token but checkpoint requires BOS prepend")?,
        )
    } else {
        None
    };
    let pad_id = tokenizer
        .get_padding()
        .map(|padding| padding.pad_id)
        .or_else(|| tokenizer.token_to_id("<pad>"))
        .or_else(|| tokenizer.token_to_id("[PAD]"))
        .or_else(|| tokenizer.token_to_id("</s>"))
        .context("tokenizer has no pad token")?;
    let row = prepare_token_row(
        encoding.get_ids(),
        bos_id,
        pad_id,
        max_length,
        pad_to_max,
        force_all_false,
    )?;
    let seq_len = row.ids.len();
    let valid_tokens = row.mask.iter().filter(|&&value| value).count();
    let ids = Tensor::<B, 2, Int>::from_data(TensorData::new(row.ids, [1, seq_len]), device);
    let mask = Tensor::<B, 2, Bool>::from_data(TensorData::new(row.mask, [1, seq_len]), device);
    Ok(Tokenized {
        ids,
        mask,
        valid_tokens,
    })
}

/// Load a mono WAV file and return samples as f32 in `[-1, 1]`.
///
/// Returns `(samples, sample_rate)`.  Multi-channel files are mixed down by
/// averaging across channels.
fn load_wav_as_f32(path: &std::path::Path) -> Result<(Vec<f32>, u32)> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("cannot open WAV {:?}", path))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    anyhow::ensure!(channels > 0, "reference WAV has zero channels");
    anyhow::ensure!(spec.sample_rate > 0, "reference WAV has zero sample rate");
    if spec.sample_format == hound::SampleFormat::Int {
        anyhow::ensure!(
            (1..=32).contains(&spec.bits_per_sample),
            "unsupported integer WAV bit depth: {}",
            spec.bits_per_sample
        );
    }

    let samples_raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.context("read error"))
            .collect::<Result<Vec<_>>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| Ok(s.context("read error")? as f32 * scale))
                .collect::<Result<Vec<_>>>()?
        }
    };
    anyhow::ensure!(
        samples_raw.len().is_multiple_of(channels),
        "reference WAV contains an incomplete interleaved frame: {} samples for {channels} channels",
        samples_raw.len()
    );

    // Mix down to mono if needed
    let mono: Vec<f32> = if channels == 1 {
        samples_raw
    } else {
        let inv = 1.0 / channels as f32;
        samples_raw
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() * inv)
            .collect()
    };

    Ok((mono, spec.sample_rate))
}

/// Resample `samples` from `src_rate` to `dst_rate` using linear interpolation.
///
/// This is a simple, dependency-free resampler sufficient for the reference
/// audio path (quality is not critical there; accuracy matters more for the
/// codec's own encoding).
fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = src_rate as f64 / dst_rate as f64;
    let out_len = ((samples.len() as f64 / ratio).ceil() as usize).max(1);
    (0..out_len)
        .map(|i| {
            let src_pos = i as f64 * ratio;
            let lo = src_pos.floor() as usize;
            let hi = (lo + 1).min(samples.len() - 1);
            let frac = (src_pos - lo as f64) as f32;
            samples[lo] * (1.0 - frac) + samples[hi] * frac
        })
        .collect()
}

fn trim_reference_samples(
    samples: &mut Vec<f32>,
    sample_rate: u32,
    max_seconds: Option<f64>,
) -> Result<()> {
    let Some(max_seconds) = max_seconds else {
        return Ok(());
    };
    anyhow::ensure!(
        max_seconds.is_finite() && max_seconds > 0.0,
        "reference max seconds must be finite and > 0"
    );
    let max_samples = ((max_seconds * f64::from(sample_rate)) as usize).max(1);
    if samples.len() > max_samples {
        tracing::warn!(
            "Reference audio exceeds ref_max_seconds ({max_seconds}s); trimming from {:.2}s to {:.2}s",
            samples.len() as f64 / f64::from(sample_rate),
            max_samples as f64 / f64::from(sample_rate),
        );
        samples.truncate(max_samples);
    }
    Ok(())
}

/// Load a WAV, resample to `target_rate` if needed, and return a
/// `[1, 1, samples]` Burn tensor.
fn load_and_prepare_audio<B: Backend>(
    path: &std::path::Path,
    target_rate: u32,
    max_seconds: Option<f64>,
    device: &B::Device,
) -> Result<Tensor<B, 3>> {
    let (mut samples, sr) = load_wav_as_f32(path)?;
    anyhow::ensure!(
        !samples.is_empty(),
        "reference WAV contains no audio samples"
    );
    anyhow::ensure!(target_rate > 0, "codec sample rate must be > 0");
    trim_reference_samples(&mut samples, sr, max_seconds)?;
    if sr != target_rate {
        tracing::info!("Resampling ref audio from {} Hz → {} Hz", sr, target_rate);
        samples = resample_linear(&samples, sr, target_rate);
    }
    let n = samples.len();
    Ok(Tensor::<B, 3>::from_data(
        TensorData::new(samples, [1, 1, n]),
        device,
    ))
}

/// Convert codec latents `[B, T, D]` into the model's latent-patched space.
fn patchify_reference_latent<B: Backend>(
    latent: Tensor<B, 3>,
    patch_size: usize,
) -> Result<Tensor<B, 3>> {
    if patch_size <= 1 {
        return Ok(latent);
    }
    let [batch, seq_len, dim] = latent.dims();
    let usable = seq_len / patch_size * patch_size;
    anyhow::ensure!(
        usable > 0,
        "reference latent is too short for latent_patch_size={patch_size}: {seq_len} frames"
    );
    Ok(latent.slice([0..batch, 0..usable, 0..dim]).reshape([
        batch,
        usable / patch_size,
        dim * patch_size,
    ]))
}

/// Convert normalized float audio to the signed PCM16 mapping used by the
/// official torchaudio/torchcodec and soundfile output paths.
fn f32_to_pcm16(sample: f32) -> i16 {
    let scaled = (sample.clamp(-1.0, 1.0) * 32768.0).round();
    scaled.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
}

/// Write a `[1, 1, S]` f32 tensor as a 16-bit PCM WAV file.
fn save_wav<B: Backend>(
    path: &std::path::Path,
    audio: Tensor<B, 3>,
    sample_rate: u32,
) -> Result<()> {
    let [batch, channels, n_samples] = audio.dims();
    anyhow::ensure!(
        batch == 1 && channels == 1,
        "WAV output requires shape [1, 1, samples], got [{batch}, {channels}, {n_samples}]"
    );
    // Clamp to [-1, 1] before converting to i16
    let data = audio.clamp(-1.0f32, 1.0f32).into_data().convert::<f32>();
    let samples: Vec<f32> = data
        .to_vec()
        .context("failed to read decoded audio tensor as f32")?;
    anyhow::ensure!(
        samples.len() == n_samples,
        "decoded audio data length {} does not match tensor length {n_samples}",
        samples.len()
    );

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("cannot create WAV {:?}", path))?;
    for s in &samples {
        let pcm = f32_to_pcm16(*s);
        writer.write_sample(pcm).context("write WAV sample")?;
    }
    writer.finalize().context("finalise WAV")?;
    Ok(())
}

fn parse_cfg_mode(s: &str) -> Result<irodori_tts_wgpu::CfgGuidanceMode> {
    s.parse()
        .with_context(|| format!("invalid CFG guidance mode '{s}'"))
}

/// Echo-style heuristic: find first latent frame where a trailing window
/// becomes near-flat and near-zero (ported from Python `find_flattening_point`).
///
/// `latent_data` contains the raw f32 values in row-major order for a
/// `[T, D]` slice (batch dimension already dropped).  Returns a frame index
/// in `[0, total_t]`.
fn find_flattening_point(
    latent_data: &[f32],
    total_t: usize,
    latent_dim: usize,
    window_size: usize,
    std_threshold: f32,
    mean_threshold: f32,
) -> usize {
    if total_t == 0 || window_size == 0 {
        return total_t;
    }
    // Pad with zeros so the window always has `window_size` frames.
    let padded_t = total_t + window_size;

    for i in 0..(padded_t - window_size) {
        let w_start = i * latent_dim;
        let w_end = (i + window_size) * latent_dim;

        let window: &[f32] = if w_end <= latent_data.len() {
            &latent_data[w_start..w_end]
        } else {
            // Window extends into the zero-padding — compute manually.
            let avail = latent_data.len().saturating_sub(w_start);
            // Mean and std over the available elements + zeros for the rest.
            let n = (window_size * latent_dim) as f32;
            let sum: f32 = latent_data[w_start..w_start + avail].iter().sum();
            let mean = sum / n;
            let sq_sum: f32 = latent_data[w_start..w_start + avail]
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>()
                + (n as usize - avail) as f32 * mean.powi(2); // zero elements
            let std = (sq_sum / n).sqrt();
            if std < std_threshold && mean.abs() < mean_threshold {
                return i;
            }
            continue;
        };

        let n = window.len() as f32;
        let mean: f32 = window.iter().sum::<f32>() / n;
        let std = (window.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
        if std < std_threshold && mean.abs() < mean_threshold {
            return i;
        }
    }
    total_t
}

/// Resolve the final PCM sample limit after duration and tail trimming.
///
/// The official runtime treats a flattening point of zero as "no tail trim";
/// it never asks the codec to decode a zero-length latent.
fn output_sample_limit(
    target_samples: Option<usize>,
    flattening_point: Option<usize>,
    hop_length: usize,
) -> Result<Option<usize>> {
    let flattening_samples = flattening_point
        .filter(|&point| point > 0)
        .map(|point| {
            point
                .checked_mul(hop_length)
                .context("tail-trim sample count overflow")
        })
        .transpose()?;
    Ok(match (target_samples, flattening_samples) {
        (Some(target), Some(flattening)) => Some(target.min(flattening)),
        (Some(target), None) => Some(target),
        (None, Some(flattening)) => Some(flattening),
        (None, None) => None,
    })
}

fn validate_rf_work_manifest_request(manifest_path: Option<&Path>) -> Result<()> {
    let Some(manifest_path) = manifest_path else {
        return Ok(());
    };
    anyhow::ensure!(
        !manifest_path.as_os_str().is_empty(),
        "--rf-work-manifest-out must not be empty"
    );
    match fs::symlink_metadata(manifest_path) {
        Ok(_) => bail!(
            "--rf-work-manifest-out refuses to overwrite an existing path: {}",
            manifest_path.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to inspect --rf-work-manifest-out path {}",
                    manifest_path.display()
                )
            });
        }
    }
    let parent = manifest_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent_metadata = fs::symlink_metadata(parent).with_context(|| {
        format!(
            "--rf-work-manifest-out parent does not exist: {}",
            parent.display()
        )
    })?;
    anyhow::ensure!(
        parent_metadata.file_type().is_dir(),
        "--rf-work-manifest-out parent must be a real directory: {}",
        parent.display()
    );
    Ok(())
}

fn write_new_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut payload = serde_json::to_vec_pretty(value).context("serialize RF work manifest")?;
    payload.push(b'\n');
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create new RF work manifest {}", path.display()))?;
    file.write_all(&payload)
        .with_context(|| format!("write RF work manifest {}", path.display()))?;
    file.flush()
        .with_context(|| format!("flush RF work manifest {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("sync RF work manifest {}", path.display()))?;
    Ok(())
}

trait PipelineEngine<B: Backend> {
    fn sample(&self, request: SamplingRequest<B>) -> irodori_tts_wgpu::Result<Tensor<B, 3>>;

    fn sample_with_work_report(
        &self,
        request: SamplingRequest<B>,
    ) -> irodori_tts_wgpu::Result<(Tensor<B, 3>, SamplerWorkReport)>;

    fn encode_conditions(
        &self,
        text_ids: Tensor<B, 2, Int>,
        text_mask: Tensor<B, 2, Bool>,
        aux_input: AuxConditionInput<B>,
    ) -> irodori_tts_wgpu::Result<EncodedCondition<B>>;

    fn predict_duration_log_frames(
        &self,
        cond: &EncodedCondition<B>,
        duration_features: Tensor<B, 2>,
        has_speaker: Tensor<B, 1, Bool>,
        has_caption: Tensor<B, 1, Bool>,
    ) -> irodori_tts_wgpu::Result<Tensor<B, 1>>;

    fn predict_duration_compact_no_aux(
        &self,
        cond: &EncodedCondition<B>,
        duration_features: Tensor<B, 2>,
        has_speaker: Tensor<B, 1, Bool>,
        has_caption: Tensor<B, 1, Bool>,
    ) -> irodori_tts_wgpu::Result<Tensor<B, 1>>;

    fn has_duration_predictor(&self) -> bool;
    fn sampling_params(&self) -> &SamplerParams;

    fn prepare_codec_for_encode(_codec: &mut DacVaeCodec<B>) {}
    fn prepare_codec_for_decode(_codec: &mut DacVaeCodec<B>) {}

    fn encode_codec(codec: &DacVaeCodec<B>, waveform: Tensor<B, 3>) -> Tensor<B, 3>;
    fn decode_codec(codec: &DacVaeCodec<B>, latent: Tensor<B, 3>) -> Tensor<B, 3>;
}

impl PipelineEngine<WgpuRaw> for WgslInferenceEngine {
    fn sample(
        &self,
        request: SamplingRequest<WgpuRaw>,
    ) -> irodori_tts_wgpu::Result<Tensor<WgpuRaw, 3>> {
        WgslInferenceEngine::sample(self, request)
    }

    fn sample_with_work_report(
        &self,
        request: SamplingRequest<WgpuRaw>,
    ) -> irodori_tts_wgpu::Result<(Tensor<WgpuRaw, 3>, SamplerWorkReport)> {
        WgslInferenceEngine::sample_with_work_report(self, request)
    }

    fn encode_conditions(
        &self,
        text_ids: Tensor<WgpuRaw, 2, Int>,
        text_mask: Tensor<WgpuRaw, 2, Bool>,
        aux_input: AuxConditionInput<WgpuRaw>,
    ) -> irodori_tts_wgpu::Result<EncodedCondition<WgpuRaw>> {
        self.model()
            .encode_conditions(text_ids, text_mask, aux_input)
    }

    fn predict_duration_log_frames(
        &self,
        cond: &EncodedCondition<WgpuRaw>,
        duration_features: Tensor<WgpuRaw, 2>,
        has_speaker: Tensor<WgpuRaw, 1, Bool>,
        has_caption: Tensor<WgpuRaw, 1, Bool>,
    ) -> irodori_tts_wgpu::Result<Tensor<WgpuRaw, 1>> {
        self.model()
            .predict_duration_log_frames(cond, duration_features, has_speaker, has_caption)
    }

    fn predict_duration_compact_no_aux(
        &self,
        cond: &EncodedCondition<WgpuRaw>,
        duration_features: Tensor<WgpuRaw, 2>,
        has_speaker: Tensor<WgpuRaw, 1, Bool>,
        has_caption: Tensor<WgpuRaw, 1, Bool>,
    ) -> irodori_tts_wgpu::Result<Tensor<WgpuRaw, 1>> {
        self.model().predict_duration_compact_no_aux_wgsl(
            cond,
            duration_features,
            has_speaker,
            has_caption,
        )
    }

    fn has_duration_predictor(&self) -> bool {
        self.model().has_duration_predictor()
    }

    fn sampling_params(&self) -> &SamplerParams {
        WgslInferenceEngine::sampling_params(self)
    }

    fn prepare_codec_for_encode(codec: &mut DacVaeCodec<WgpuRaw>) {
        codec.prepare_encoder_for_wgsl();
    }

    fn prepare_codec_for_decode(codec: &mut DacVaeCodec<WgpuRaw>) {
        codec.prepare_decoder_for_wgsl();
    }

    fn encode_codec(
        codec: &DacVaeCodec<WgpuRaw>,
        waveform: Tensor<WgpuRaw, 3>,
    ) -> Tensor<WgpuRaw, 3> {
        codec.encode_wgsl(waveform)
    }

    fn decode_codec(
        codec: &DacVaeCodec<WgpuRaw>,
        latent: Tensor<WgpuRaw, 3>,
    ) -> Tensor<WgpuRaw, 3> {
        codec.decode_wgsl(latent)
    }
}

fn ensure_supported_codec<B: Backend>(codec: &DacVaeCodec<B>) -> Result<()> {
    anyhow::ensure!(
        codec.sample_rate() == DACVAE_SAMPLE_RATE,
        "loaded codec sample rate {} does not match supported {} Hz",
        codec.sample_rate(),
        DACVAE_SAMPLE_RATE
    );
    anyhow::ensure!(
        codec.hop_length() == DACVAE_HOP_LENGTH,
        "loaded codec hop length {} does not match supported {}",
        codec.hop_length(),
        DACVAE_HOP_LENGTH
    );
    Ok(())
}

const ANNOTATION_EMOJIS: &[&str] = &[
    "😮\u{200d}💨",
    "⏱️",
    "⏩",
    "⏸️",
    "🌬️",
    "🍭",
    "🎛️",
    "🎭",
    "🎵",
    "🐢",
    "🐱",
    "👂",
    "👃",
    "👅",
    "👌",
    "👏",
    "💋",
    "💥",
    "💦",
    "💪",
    "📄",
    "📞",
    "📢",
    "📣",
    "😆",
    "😊",
    "😌",
    "😎",
    "😏",
    "😒",
    "😖",
    "😟",
    "😠",
    "😪",
    "😭",
    "😮",
    "😰",
    "😱",
    "😲",
    "😴",
    "🙄",
    "🙏",
    "🤐",
    "🤔",
    "🤢",
    "🤧",
    "🤭",
    "🥤",
    "🥱",
    "🥴",
    "🥵",
    "🥹",
    "🥺",
    "🫣",
    "🫶",
    "📖",
];

fn count_annotation_emojis(text: &str) -> usize {
    let mut count = 0;
    let mut offset = 0;
    while offset < text.len() {
        let rest = &text[offset..];
        if let Some(matched) = ANNOTATION_EMOJIS
            .iter()
            .filter(|emoji| rest.starts_with(**emoji))
            .max_by_key(|emoji| emoji.chars().count())
        {
            count += 1;
            offset += matched.len();
        } else {
            let Some(character) = rest.chars().next() else {
                break;
            };
            offset += character.len_utf8();
        }
    }
    count
}

fn log1p_cap(value: usize, cap: usize) -> f32 {
    (value.min(cap) as f32).ln_1p() / (cap as f32).ln_1p()
}

fn build_duration_features(
    text: &str,
    token_count: usize,
    max_text_len: usize,
    has_speaker: bool,
    aux_dim: usize,
) -> Result<Vec<f32>> {
    anyhow::ensure!(
        aux_dim == 14,
        "released v4 duration features require duration_aux_dim=14, got {aux_dim}"
    );
    anyhow::ensure!(max_text_len > 0, "max_text_len must be > 0");
    let char_count = text.chars().count().max(1);
    let kana_count = text
        .chars()
        .filter(|character| matches!(*character as u32, 0x3040..=0x30ff))
        .count();
    let kanji_count = text
        .chars()
        .filter(|character| {
            matches!(
                *character as u32,
                0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xf900..=0xfaff | 0x20000..=0x2fa1f
            )
        })
        .count();
    let alnum_count = text
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .count();
    let count = |needle: &str| text.matches(needle).count();

    Ok(vec![
        token_count.min(max_text_len) as f32 / max_text_len as f32,
        (char_count.min(512) as f32).ln_1p() / 512.0_f32.ln_1p(),
        token_count as f32 / char_count as f32,
        log1p_cap(count("。") + count("."), 8),
        log1p_cap(count("、") + count(","), 16),
        log1p_cap(count("ー"), 8),
        log1p_cap(count("…"), 8),
        log1p_cap(count("！") + count("!"), 8),
        log1p_cap(count("？") + count("?"), 8),
        log1p_cap(count_annotation_emojis(text), 8),
        kana_count as f32 / char_count as f32,
        kanji_count as f32 / char_count as f32,
        alnum_count as f32 / char_count as f32,
        if has_speaker { 1.0 } else { 0.0 },
    ])
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct OutputLength {
    latent_frames: usize,
    patched_frames: usize,
    target_samples: Option<usize>,
}

fn cleanup_unused_backend_memory<B: Backend>(device: &B::Device, stage: &str) -> Result<()> {
    B::memory_cleanup(device);
    B::sync(device).with_context(|| format!("backend memory cleanup failed after {stage}"))?;
    tracing::info!("Released unused backend allocations after {stage}");
    Ok(())
}

fn manual_output_length(
    seconds: f64,
    min_seconds: f64,
    max_seconds: f64,
    sample_rate: usize,
    hop_length: usize,
    latent_patch_size: usize,
) -> OutputLength {
    let clamped = seconds.clamp(min_seconds, max_seconds);
    let target_samples = ((clamped * sample_rate as f64) as usize).max(1);
    let latent_frames = target_samples.div_ceil(hop_length);
    OutputLength {
        latent_frames,
        patched_frames: latent_frames.div_ceil(latent_patch_size),
        target_samples: Some(target_samples),
    }
}

fn predicted_output_length(
    predicted_frames: f64,
    duration_scale: f64,
    min_seconds: f64,
    max_seconds: f64,
    sample_rate: usize,
    hop_length: usize,
    latent_patch_size: usize,
) -> OutputLength {
    let min_frames = ((min_seconds * sample_rate as f64) / hop_length as f64)
        .ceil()
        .max(1.0) as usize;
    let max_frames = ((max_seconds * sample_rate as f64) / hop_length as f64)
        .floor()
        .max(1.0) as usize;
    let scaled = (predicted_frames * duration_scale)
        .round_ties_even()
        .max(0.0) as usize;
    let latent_frames = scaled.min(max_frames).max(min_frames);
    OutputLength {
        latent_frames,
        patched_frames: latent_frames.div_ceil(latent_patch_size),
        target_samples: Some(latent_frames * hop_length),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn run<B, E, F>(args: Args, device: B::Device, build_engine: F) -> Result<()>
where
    B: Backend,
    E: PipelineEngine<B>,
    F: FnOnce(
        irodori_tts_wgpu::inference::InferenceBuilder<B, irodori_tts_wgpu::inference::Ready>,
    ) -> E,
{
    anyhow::ensure!(
        args.seconds.is_none() || args.seq_len.is_none(),
        "--seconds and --seq-len are mutually exclusive"
    );
    if let Some(seconds) = args.seconds {
        anyhow::ensure!(
            seconds.is_finite() && seconds > 0.0,
            "--seconds must be finite and > 0"
        );
    }
    anyhow::ensure!(
        args.duration_scale.is_finite() && args.duration_scale > 0.0,
        "--duration-scale must be finite and > 0"
    );
    anyhow::ensure!(
        args.min_seconds.is_finite() && args.min_seconds > 0.0,
        "--min-seconds must be finite and > 0"
    );
    anyhow::ensure!(
        args.max_seconds.is_finite() && args.max_seconds >= args.min_seconds,
        "--max-seconds must be finite and >= --min-seconds"
    );
    if let Some(seq_len) = args.seq_len {
        anyhow::ensure!(seq_len > 0, "--seq-len must be > 0");
    }

    if let Some(adapter_index) = args.wgpu_adapter_index {
        tracing::info!(
            "Backend: {} using explicit WGPU adapter index {} (WGPU enumeration; not CUDA/NVML ordinal)",
            args.backend.label(),
            adapter_index
        );
    } else {
        tracing::info!(
            "Backend: {} using legacy gpu-id={} selection",
            args.backend.label(),
            args.gpu_id
        );
    }

    // ── TTS model ────────────────────────────────────────────────────────────
    tracing::info!("Loading TTS model from {:?}", args.checkpoint);
    #[cfg(feature = "lora")]
    let loaded = match args.adapter.as_deref() {
        Some(dir) => {
            tracing::info!("Merging LoRA adapter from {:?}", dir);
            InferenceBuilder::<B, _>::new(device.clone())
                .load_weights_with_adapter(&args.checkpoint, dir)?
        }
        None => InferenceBuilder::<B, _>::new(device.clone()).load_weights(&args.checkpoint)?,
    };
    #[cfg(not(feature = "lora"))]
    let loaded = InferenceBuilder::<B, _>::new(device.clone()).load_weights(&args.checkpoint)?;
    let cfg = loaded.model_config().clone();
    tracing::info!(
        "TTS model loaded (latent_dim={}, patch_size={})",
        cfg.latent_dim,
        cfg.latent_patch_size,
    );

    // Validate latent_dim compatibility
    if cfg.latent_dim != DACVAE_LATENT_DIM {
        bail!(
            "TTS latent_dim={} but DACVAE codec expects {}",
            cfg.latent_dim,
            DACVAE_LATENT_DIM
        );
    }
    anyhow::ensure!(
        args.codec_weights.is_file(),
        "codec weights do not exist or are not a file: {:?}",
        args.codec_weights
    );

    // ── Tokenise ─────────────────────────────────────────────────────────────
    tracing::info!("Loading tokenizer …");
    let bundled_tokenizer = cfg
        .use_pretrained_text_encoder()
        .then(|| bundled_tokenizer_path(&args.checkpoint))
        .flatten();
    let tokenizer_path = args.tokenizer.as_deref().or(bundled_tokenizer.as_deref());
    if args.tokenizer.is_none()
        && let Some(path) = bundled_tokenizer.as_deref()
    {
        tracing::info!("Using checkpoint-bundled v4 tokenizer: {path:?}");
    }
    let tokenizer = load_tokenizer(
        tokenizer_path,
        &cfg.text_tokenizer_repo,
        cfg.text_encoder_revision.as_deref(),
    )?;
    let normalized = irodori_tts_wgpu::normalize_text(&args.text);
    let normalized = normalized.trim();
    anyhow::ensure!(
        !normalized.is_empty(),
        "text became empty after normalization"
    );
    tracing::info!("Text (normalized): {normalized:?}");
    let v4_frontend = cfg.use_pretrained_text_encoder();
    let text_max_len = cfg.max_text_len.unwrap_or(256);
    let text_tokens = tokenize::<B>(
        &tokenizer,
        normalized,
        cfg.text_add_bos,
        if v4_frontend {
            Some(text_max_len)
        } else {
            cfg.max_text_len
        },
        v4_frontend,
        false,
        &device,
    )?;
    tracing::info!(
        "Tokenised text: valid_tokens={}, sequence_length={}",
        text_tokens.valid_tokens,
        text_tokens.ids.dims()[1]
    );

    let caption_text = args.caption.as_deref().unwrap_or("").trim();
    let has_caption_text = cfg.use_caption_condition && !caption_text.is_empty();
    let (caption_ids, caption_mask, caption_valid_tokens) = if cfg.use_caption_condition {
        let caption_tokenizer_owned =
            if !v4_frontend && cfg.caption_tokenizer_repo() != cfg.text_tokenizer_repo {
                Some(load_tokenizer(None, cfg.caption_tokenizer_repo(), None)?)
            } else {
                None
            };
        let caption_tokenizer = caption_tokenizer_owned.as_ref().unwrap_or(&tokenizer);
        let caption_max_len = cfg.max_caption_len.unwrap_or(text_max_len);
        let tokens = tokenize::<B>(
            caption_tokenizer,
            caption_text,
            cfg.caption_add_bos(),
            if v4_frontend {
                Some(caption_max_len)
            } else {
                cfg.max_caption_len
            },
            v4_frontend,
            !has_caption_text,
            &device,
        )?;
        let valid_tokens = tokens.valid_tokens;
        (Some(tokens.ids), Some(tokens.mask), valid_tokens)
    } else {
        if args.caption.is_some() {
            tracing::warn!("Ignoring --caption because this checkpoint has no caption condition");
        }
        (None, None, 0)
    };
    tracing::info!(
        "Caption: enabled={}, present={}, valid_tokens={caption_valid_tokens}",
        cfg.use_caption_condition,
        has_caption_text
    );

    let text_ids = text_tokens.ids;
    let text_mask = text_tokens.mask;
    let text_valid_tokens = text_tokens.valid_tokens;

    // ── Reference audio (optional) ───────────────────────────────────────────
    let has_real_speaker = cfg.use_speaker_condition() && args.ref_audio.is_some();
    let mut reference_codec = if has_real_speaker {
        tracing::info!(
            "Loading DACVAE codec for reference encoding from {:?}",
            args.codec_weights
        );
        let mut codec = load_codec::<B>(&args.codec_weights, &device)?;
        E::prepare_codec_for_encode(&mut codec);
        ensure_supported_codec(&codec)?;
        Some(codec)
    } else {
        None
    };
    let (ref_latent, ref_mask) = if !cfg.use_speaker_condition() {
        if args.ref_audio.is_some() {
            tracing::warn!("Ignoring --ref-audio because this checkpoint has no speaker condition");
        }
        (None, None)
    } else if let Some(ref ref_path) = args.ref_audio {
        tracing::warn!(
            "--ref-audio compatibility preprocessing is not v4-parity-equivalent: \
             official -16 dB audiotools normalization and torchaudio band-limited \
             resampling are not implemented"
        );
        tracing::info!("Encoding reference audio {:?}", ref_path);
        let codec = reference_codec
            .as_ref()
            .context("reference codec was not loaded for active speaker conditioning")?;
        let wav = load_and_prepare_audio::<B>(
            ref_path,
            codec.sample_rate() as u32,
            cfg.ref_max_seconds,
            &device,
        )?;
        let latent = patchify_reference_latent(E::encode_codec(codec, wav), cfg.latent_patch_size)?;
        let [b, t, _d] = latent.dims();
        let sync_value = latent
            .clone()
            .slice([0..1, 0..1, 0..1])
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()?
            .first()
            .copied()
            .context("reference encoder returned an empty latent")?;
        anyhow::ensure!(
            sync_value.is_finite(),
            "reference encoder returned a non-finite latent"
        );
        tracing::info!(
            "Reference latent (latent-patched): [{b}, {t}, {}]",
            cfg.latent_dim * cfg.latent_patch_size
        );
        let mask: Tensor<B, 2, Bool> = Tensor::<B, 2>::ones([b, t], &device).greater_elem(0.0f32);
        (Some(latent), Some(mask))
    } else {
        // Official v4 no-ref sentinel: latent-patched zeros and an all-false
        // mask with exactly speaker_patch_size time steps. The sampler receives
        // a zero speaker CFG scale below, so this is not treated as a real ref.
        let speaker_patch_size = cfg.speaker_patch_size.unwrap_or(1);
        let ref_len = speaker_patch_size.max(1);
        let ref_latent: Tensor<B, 3> = Tensor::zeros(
            [1, ref_len, cfg.latent_dim * cfg.latent_patch_size],
            &device,
        );
        let ref_mask: Tensor<B, 2, Bool> =
            Tensor::<B, 2>::zeros([1, ref_len], &device).greater_elem(0.0f32);
        tracing::info!(
            "No reference audio — all-false dummy ref (speaker_patch_size={ref_len}, latent_patched_dim={})",
            cfg.latent_dim * cfg.latent_patch_size
        );
        (Some(ref_latent), Some(ref_mask))
    };
    // Codec weights are cold during RF denoising. Releasing them here avoids
    // retaining the complete DACVAE alongside the TTS model on 8 GiB GPUs.
    let released_reference_codec = reference_codec.take().is_some();
    if released_reference_codec {
        cleanup_unused_backend_memory::<B>(&device, "reference encoding")?;
    }

    // ── RF sampling ──────────────────────────────────────────────────────────
    let cfg_mode = parse_cfg_mode(&args.cfg_mode)?;
    let effective_cfg_speaker = if has_real_speaker {
        args.cfg_speaker
    } else {
        0.0
    };
    let effective_cfg_caption = if has_caption_text {
        args.cfg_caption
    } else {
        0.0
    };
    let params = SamplerParams {
        num_steps: args.num_steps,
        method: args.sampler,
        guidance: GuidanceConfig {
            mode: cfg_mode,
            scale_text: args.cfg_text,
            scale_caption: effective_cfg_caption,
            scale_speaker: effective_cfg_speaker,
            min_t: args.cfg_min_t,
            max_t: args.cfg_max_t,
        },
        ..SamplerParams::default()
    };
    let engine = build_engine(loaded.with_sampling(params));

    let sample_rate = DACVAE_SAMPLE_RATE;
    let hop_length = DACVAE_HOP_LENGTH;
    let length = if let Some(seconds) = args.seconds {
        let clamped = seconds.clamp(args.min_seconds, args.max_seconds);
        if clamped != seconds {
            tracing::warn!(
                "Manual duration {seconds:.3}s clamped to {clamped:.3}s (bounds {:.3}..{:.3}s)",
                args.min_seconds,
                args.max_seconds
            );
        }
        let length = manual_output_length(
            seconds,
            args.min_seconds,
            args.max_seconds,
            sample_rate,
            hop_length,
            cfg.latent_patch_size,
        );
        tracing::info!(
            "Duration source=manual seconds={clamped:.3} latent_frames={} patched_frames={}",
            length.latent_frames,
            length.patched_frames
        );
        length
    } else if let Some(patched_frames) = args.seq_len {
        let latent_frames = patched_frames * cfg.latent_patch_size;
        tracing::info!(
            "Duration source=explicit-seq-len latent_frames={latent_frames} patched_frames={patched_frames}"
        );
        OutputLength {
            latent_frames,
            patched_frames,
            target_samples: None,
        }
    } else if cfg.use_duration_predictor {
        anyhow::ensure!(
            engine.has_duration_predictor(),
            "checkpoint config enables duration prediction but the loaded model has no duration head"
        );
        let t_duration = Instant::now();
        let duration_text_tokens = text_valid_tokens.max(1);
        anyhow::ensure!(
            duration_text_tokens <= text_ids.dims()[1],
            "duration text extent {duration_text_tokens} exceeds token tensor width {}",
            text_ids.dims()[1]
        );
        let duration_text_ids = text_ids.clone().narrow(1, 0, duration_text_tokens);
        let duration_text_mask = text_mask.clone().narrow(1, 0, duration_text_tokens);
        let aux_input = AuxConditionInput::try_from_request(
            if has_real_speaker {
                ref_latent.clone()
            } else {
                None
            },
            if has_real_speaker {
                ref_mask.clone()
            } else {
                None
            },
            if has_caption_text {
                caption_ids.clone()
            } else {
                None
            },
            if has_caption_text {
                caption_mask.clone()
            } else {
                None
            },
        )?;
        tracing::info!(
            "Duration conditioning: text {}->{duration_text_tokens}, speaker={}, caption={}",
            text_ids.dims()[1],
            has_real_speaker,
            has_caption_text
        );
        let condition =
            engine.encode_conditions(duration_text_ids, duration_text_mask, aux_input)?;
        let duration_features = build_duration_features(
            normalized,
            text_valid_tokens,
            text_max_len,
            has_real_speaker,
            cfg.duration_aux_dim,
        )?;
        let duration_features = Tensor::<B, 2>::from_data(
            TensorData::new(duration_features, [1, cfg.duration_aux_dim]),
            &device,
        );
        let has_speaker =
            Tensor::<B, 1, Bool>::from_data(TensorData::new(vec![has_real_speaker], [1]), &device);
        let has_caption =
            Tensor::<B, 1, Bool>::from_data(TensorData::new(vec![has_caption_text], [1]), &device);
        let predicted_log_frames = if !has_real_speaker && !has_caption_text {
            engine.predict_duration_compact_no_aux(
                &condition,
                duration_features,
                has_speaker,
                has_caption,
            )?
        } else {
            engine.predict_duration_log_frames(
                &condition,
                duration_features,
                has_speaker,
                has_caption,
            )?
        };
        anyhow::ensure!(
            predicted_log_frames.dims() == [1],
            "duration predictor must return shape [1], got {:?}",
            predicted_log_frames.dims()
        );
        let predicted_log_frames = predicted_log_frames
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()?
            .first()
            .copied()
            .context("duration predictor returned an empty tensor")?;
        let predicted_frames = f64::from(predicted_log_frames.exp_m1());
        anyhow::ensure!(
            predicted_frames.is_finite() && predicted_frames >= 0.0,
            "duration predictor returned invalid frames: {predicted_frames}"
        );
        let length = predicted_output_length(
            predicted_frames,
            args.duration_scale,
            args.min_seconds,
            args.max_seconds,
            sample_rate,
            hop_length,
            cfg.latent_patch_size,
        );
        tracing::info!(
            "Duration source=predictor predicted_frames={predicted_frames:.3} scale={:.3} latent_frames={} patched_frames={} duration_time={:.1}ms",
            args.duration_scale,
            length.latent_frames,
            length.patched_frames,
            t_duration.elapsed().as_secs_f64() * 1000.0
        );
        length
    } else if let Some(latent_frames) = cfg.fixed_target_latent_steps {
        let patched_frames = latent_frames.div_ceil(cfg.latent_patch_size);
        tracing::info!(
            "Duration source=checkpoint-fixed latent_frames={latent_frames} patched_frames={patched_frames}"
        );
        OutputLength {
            latent_frames,
            patched_frames,
            target_samples: Some(latent_frames * hop_length),
        }
    } else {
        let patched_frames = 256;
        let latent_frames = patched_frames * cfg.latent_patch_size;
        tracing::info!(
            "Duration source=legacy-fallback latent_frames={latent_frames} patched_frames={patched_frames}"
        );
        OutputLength {
            latent_frames,
            patched_frames,
            target_samples: None,
        }
    };

    tracing::info!(
        "[parity] backend={} text={normalized:?} text_tokens={} text_sequence={} caption_present={} caption_tokens={} speaker_present={} latent_frames={} patched_frames={} target_samples={:?} noise_file={:?}",
        args.backend.label(),
        text_valid_tokens,
        text_ids.dims()[1],
        has_caption_text,
        caption_valid_tokens,
        has_real_speaker,
        length.latent_frames,
        length.patched_frames,
        length.target_samples,
        args.noise_file,
    );

    tracing::info!(
        "Running RF sampler: {} steps ({:?}), seq_len={seq_len}",
        engine.sampling_params().num_steps,
        engine.sampling_params().method,
        seq_len = length.patched_frames,
    );
    if let Some(seed) = args.seed {
        if args.noise_file.is_some() {
            tracing::warn!("Ignoring --seed because --noise-file supplies the initial latent");
        } else {
            // Seed immediately before noise creation so model/codec
            // construction cannot consume and shift the sampler RNG stream.
            tracing::info!("Seeding backend sampler RNG with seed={seed}");
            B::seed(&device, seed);
        }
    }
    let t_sample = Instant::now();
    let sampling_request = SamplingRequest {
        text_ids,
        text_mask,
        ref_latent,
        ref_mask,
        sequence_length: length.patched_frames,
        caption_ids,
        caption_mask,
        initial_noise: load_initial_noise::<B>(
            args.noise_file.as_deref(),
            length.patched_frames,
            cfg.patched_latent_dim(),
            &device,
        )?,
    };
    let (z_patched, rf_work_report) = if args.rf_work_manifest_out.is_some() {
        let (output, report) = engine.sample_with_work_report(sampling_request)?;
        (output, Some(report))
    } else {
        (engine.sample(sampling_request)?, None)
    };
    let [b, s_pat, _] = z_patched.dims();
    let rf_sync_value = z_patched
        .clone()
        .slice([0..1, 0..1, 0..1])
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()?
        .first()
        .copied()
        .context("RF sampler returned an empty latent")?;
    anyhow::ensure!(
        rf_sync_value.is_finite(),
        "RF sampler returned a non-finite latent"
    );
    let rf_elapsed_ms = t_sample.elapsed().as_secs_f64() * 1000.0;
    tracing::info!("Sampler done: [{b}, {s_pat}, patched_dim]  rf_time={rf_elapsed_ms:.0}ms");
    drop(engine);
    cleanup_unused_backend_memory::<B>(&device, "RF sampling")?;

    // ── Unpatchify ───────────────────────────────────────────────────────────
    let z = unpatchify_latent(z_patched, cfg.latent_patch_size, cfg.latent_dim);
    let [_, unpatched_frames, _] = z.dims();
    anyhow::ensure!(
        unpatched_frames >= length.latent_frames,
        "sampler produced {unpatched_frames} latent frames, expected at least {}",
        length.latent_frames
    );
    let z = if unpatched_frames > length.latent_frames {
        z.slice([0..1, 0..length.latent_frames, 0..cfg.latent_dim])
    } else {
        z
    };
    let [_, s, _] = z.dims();
    tracing::info!("Unpatchified latent: [{b}, {s}, {}]", cfg.latent_dim);

    // ── Tail trimming (find_flattening_point) ─────────────────────────────────
    // Match the official runtime: decode the complete latent first, then trim
    // PCM samples. Decoding a sliced latent changes convolution boundaries.
    let flattening_point = if args.trim_tail {
        let z_data: Vec<f32> = z
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .context("failed to read sampled latent tensor as f32")?;
        let point = find_flattening_point(
            &z_data,
            s,
            cfg.latent_dim,
            args.trim_window,
            args.trim_std,
            args.trim_mean,
        );
        if point > 0 && point < s {
            tracing::info!("Tail flattening point: {point}/{s} latent frames");
        } else if point == 0 {
            tracing::info!("Tail flattening point is zero — official no-trim semantics apply");
        } else {
            tracing::info!("No tail trimming applied (full {s} frames)");
        }
        Some(point)
    } else {
        None
    };
    let sample_limit = output_sample_limit(length.target_samples, flattening_point, hop_length)?;

    // ── DACVAE decode ────────────────────────────────────────────────────────
    tracing::info!("Loading DACVAE codec from {:?}", args.codec_weights);
    let mut codec = load_codec::<B>(&args.codec_weights, &device)?;
    E::prepare_codec_for_decode(&mut codec);
    ensure_supported_codec(&codec)?;
    tracing::info!(
        "Codec loaded (sample_rate={} Hz, hop_length={})",
        codec.sample_rate(),
        codec.hop_length()
    );
    tracing::info!("Decoding latent → waveform");
    let t_decode = Instant::now();
    let audio = E::decode_codec(&codec, z); // [B, 1, samples]
    let [_, _, decoded_samples] = audio.dims();
    let audio = if let Some(sample_limit) = sample_limit
        && decoded_samples > sample_limit
    {
        audio.slice([0..1, 0..1, 0..sample_limit])
    } else {
        audio
    };
    let [_, _, n_samples] = audio.dims();
    let codec_elapsed_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    let duration_s = n_samples as f64 / codec.sample_rate() as f64;
    tracing::info!(
        "Audio decoded: {} samples ({:.2}s @ {} Hz), latent frames={s}",
        n_samples,
        duration_s,
        codec.sample_rate()
    );
    tracing::info!(
        "[timing] rf={rf_elapsed_ms:.0}ms  codec={codec_elapsed_ms:.0}ms  audio_duration={duration_s:.3}s"
    );
    // Also print to stdout for reliable programmatic parsing:
    println!(
        "[timing] rf={rf_elapsed_ms:.0}ms  codec={codec_elapsed_ms:.0}ms  audio_duration={duration_s:.3}s"
    );

    // ── Write WAV ────────────────────────────────────────────────────────────
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    save_wav::<B>(&args.output, audio, codec.sample_rate() as u32)?;
    anyhow::ensure!(
        args.rf_work_manifest_out.is_some() == rf_work_report.is_some(),
        "internal RF work-manifest state mismatch"
    );
    if let (Some(path), Some(report)) = (
        args.rf_work_manifest_out.as_deref(),
        rf_work_report.as_ref(),
    ) {
        write_new_json(path, report)?;
        tracing::info!("Wrote RF work manifest to {:?}", path);
    }
    tracing::info!("Wrote output WAV to {:?}", args.output);

    Ok(())
}

fn main() -> process::ExitCode {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(env_filter).init();

    let args = Args::parse();
    if args.rf_work_manifest_out.as_deref() == Some(args.output.as_path()) {
        tracing::error!("Fatal: --rf-work-manifest-out must differ from --output");
        return process::ExitCode::FAILURE;
    }
    if let Err(error) = validate_rf_work_manifest_request(args.rf_work_manifest_out.as_deref()) {
        tracing::error!("Fatal: {error:#}");
        return process::ExitCode::FAILURE;
    }
    let gpu_id = args.gpu_id;
    let explicit_wgpu_adapter = args.wgpu_adapter_index;
    let device = explicit_wgpu_adapter.map_or_else(
        || irodori_tts_wgpu::backend_config::wgpu_device(gpu_id),
        irodori_tts_wgpu::backend_config::wgpu_device_from_adapter_index,
    );
    let result = run::<WgpuRaw, _, _>(args, device, |ready| ready.build_wgsl());
    match result {
        Ok(()) => process::ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!("Fatal: {error:#}");
            process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v4_token_row_prepends_bos_truncates_and_right_pads() {
        let truncated = prepare_token_row(&[10, 11, 12], Some(1), 3, Some(3), true, false)
            .expect("valid token row");
        assert_eq!(truncated.ids, vec![1, 10, 11]);
        assert_eq!(truncated.mask, vec![true, true, true]);

        let empty_caption = prepare_token_row(&[], Some(1), 3, Some(4), true, true)
            .expect("valid empty caption row");
        assert_eq!(empty_caption.ids, vec![1, 3, 3, 3]);
        assert_eq!(empty_caption.mask, vec![false; 4]);
    }

    #[test]
    fn manual_two_seconds_matches_released_v4_codec_geometry() {
        let length = manual_output_length(2.0, 0.5, 30.0, DACVAE_SAMPLE_RATE, DACVAE_HOP_LENGTH, 1);
        assert_eq!(
            length,
            OutputLength {
                latent_frames: 50,
                patched_frames: 50,
                target_samples: Some(96_000),
            }
        );
    }

    #[test]
    fn generic_output_length_patch_ceil_rounds_up() {
        let manual = manual_output_length(1.0, 0.1, 10.0, 10, 4, 2);
        assert_eq!(manual.latent_frames, 3);
        assert_eq!(manual.patched_frames, 2);

        let predicted = predicted_output_length(3.0, 1.0, 0.1, 10.0, 10, 4, 2);
        assert_eq!(predicted.latent_frames, 3);
        assert_eq!(predicted.patched_frames, 2);
    }

    #[test]
    fn predicted_duration_matches_released_v4_rounding_and_bounds() {
        let length = predicted_output_length(
            187.5,
            1.0,
            0.5,
            30.0,
            DACVAE_SAMPLE_RATE,
            DACVAE_HOP_LENGTH,
            1,
        );
        assert_eq!(length.latent_frames, 188);
        assert_eq!(length.patched_frames, 188);
        assert_eq!(length.target_samples, Some(188 * DACVAE_HOP_LENGTH));

        let clamped = predicted_output_length(
            0.0,
            1.0,
            0.5,
            30.0,
            DACVAE_SAMPLE_RATE,
            DACVAE_HOP_LENGTH,
            1,
        );
        assert_eq!(clamped.latent_frames, 13);
        assert_eq!(clamped.patched_frames, 13);
    }

    #[test]
    fn duration_features_match_released_v4_width_and_presence_slot() {
        let features = build_duration_features("こんにちは。", 8, 256, false, 14).unwrap();
        assert_eq!(features.len(), 14);
        assert_eq!(features[13], 0.0);
        assert!(features.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn pcm_limit_matches_official_post_decode_tail_trim() {
        assert_eq!(
            output_sample_limit(Some(96_000), Some(25), DACVAE_HOP_LENGTH).unwrap(),
            Some(48_000)
        );
        assert_eq!(
            output_sample_limit(Some(96_000), Some(0), DACVAE_HOP_LENGTH).unwrap(),
            Some(96_000),
            "flattening point zero must not request a zero-length decode"
        );
        assert_eq!(
            output_sample_limit(None, Some(25), DACVAE_HOP_LENGTH).unwrap(),
            Some(48_000)
        );
        assert_eq!(
            output_sample_limit(None, None, DACVAE_HOP_LENGTH).unwrap(),
            None
        );
        assert!(output_sample_limit(None, Some(usize::MAX), 2).is_err());

        let all_zero_latent = vec![0.0; 4 * 2];
        let point = find_flattening_point(&all_zero_latent, 4, 2, 2, 0.05, 0.1);
        assert_eq!(point, 0);
        assert_eq!(
            output_sample_limit(Some(2_048), Some(point), DACVAE_HOP_LENGTH).unwrap(),
            Some(2_048)
        );
    }

    #[test]
    fn pcm16_quantization_matches_measured_official_writers() {
        let samples = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let pcm = samples.map(f32_to_pcm16);
        assert_eq!(pcm, [i16::MIN, -16_384, 0, 16_384, i16::MAX]);
        assert_eq!(f32_to_pcm16(-2.0), i16::MIN);
        assert_eq!(f32_to_pcm16(2.0), i16::MAX);
    }

    #[test]
    fn reference_audio_is_trimmed_at_checkpoint_limit_before_resampling() {
        let mut samples = vec![0.0; 10];
        trim_reference_samples(&mut samples, 4, Some(1.5)).unwrap();
        assert_eq!(samples.len(), 6);

        trim_reference_samples(&mut samples, 4, None).unwrap();
        assert_eq!(samples.len(), 6);
        assert!(trim_reference_samples(&mut samples, 4, Some(0.0)).is_err());
    }

    #[test]
    fn parity_cli_accepts_wgsl_explicit_adapter_caption_seconds_and_noise() {
        let args = Args::try_parse_from([
            "pipeline",
            "--backend",
            "wgpu-wgsl",
            "--checkpoint",
            "model.safetensors",
            "--codec-weights",
            "codec.safetensors",
            "--text",
            "こんにちは。",
            "--caption",
            "",
            "--seconds",
            "2",
            "--noise-file",
            "oracle.safetensors",
            "--wgpu-adapter-index",
            "0",
        ])
        .unwrap();
        assert_eq!(args.backend, InferenceBackendKind::WgpuWgsl);
        assert_eq!(args.seconds, Some(2.0));
        assert_eq!(args.caption.as_deref(), Some(""));
        assert_eq!(args.wgpu_adapter_index, Some(0));
        assert_eq!(args.noise_file, Some(PathBuf::from("oracle.safetensors")));
        assert!(args.rf_work_manifest_out.is_none());
    }

    #[test]
    fn rf_work_manifest_is_wgsl_only_create_new_and_synced_json() {
        let directory = tempfile::tempdir().expect("temporary work-manifest directory");
        let manifest = directory.path().join("rf-work.json");
        let args = Args::try_parse_from([
            "pipeline",
            "--backend",
            "wgpu-wgsl",
            "--checkpoint",
            "model.safetensors",
            "--codec-weights",
            "codec.safetensors",
            "--text",
            "こんにちは。",
            "--rf-work-manifest-out",
            manifest.to_str().expect("UTF-8 temporary path"),
        ])
        .expect("work-manifest CLI");
        assert_eq!(
            args.rf_work_manifest_out.as_deref(),
            Some(manifest.as_path())
        );
        validate_rf_work_manifest_request(args.rf_work_manifest_out.as_deref())
            .expect("new WGPU-WGSL manifest path");

        let payload = serde_json::json!({"schema_version": 1, "num_steps": 4});
        write_new_json(&manifest, &payload).expect("create and sync work manifest");
        let serialized = fs::read_to_string(&manifest).expect("read work manifest");
        assert!(serialized.ends_with('\n'));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&serialized)
                .expect("parse written work manifest"),
            payload
        );
        assert!(write_new_json(&manifest, &payload).is_err());
        assert!(validate_rf_work_manifest_request(Some(&manifest)).is_err());
        assert!(validate_rf_work_manifest_request(None).is_ok());
        let missing_parent = directory.path().join("missing").join("rf-work.json");
        assert!(validate_rf_work_manifest_request(Some(&missing_parent)).is_err());
    }
}
