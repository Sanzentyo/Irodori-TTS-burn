//! End-to-end Irodori-TTS pipeline: text → WAV.
//!
//! Chains the RF diffusion model with the DACVAE codec to produce an output
//! waveform from a text prompt.  Reference audio (for speaker conditioning)
//! is optional; when omitted, the model operates in unconditional speaker mode.
//!
//! # Example
//! ```sh
//! just pipeline \
//!     --checkpoint model.safetensors \
//!     --codec-weights target/dacvae_weights.safetensors \
//!     --text "こんにちは" \
//!     --output output.wav
//! ```

#[cfg(any(
    all(feature = "backend_wgpu", feature = "backend_wgpu_f16"),
    all(feature = "backend_wgpu", feature = "backend_wgpu_bf16"),
    all(feature = "backend_wgpu_f16", feature = "backend_wgpu_bf16"),
    all(feature = "backend_wgpu", feature = "backend_cuda"),
    all(feature = "backend_wgpu", feature = "backend_cuda_bf16"),
    all(feature = "backend_wgpu", feature = "backend_tch"),
    all(feature = "backend_wgpu", feature = "backend_tch_bf16"),
    all(feature = "backend_wgpu_f16", feature = "backend_cuda"),
    all(feature = "backend_wgpu_f16", feature = "backend_cuda_bf16"),
    all(feature = "backend_wgpu_f16", feature = "backend_tch"),
    all(feature = "backend_wgpu_f16", feature = "backend_tch_bf16"),
    all(feature = "backend_wgpu_bf16", feature = "backend_cuda"),
    all(feature = "backend_wgpu_bf16", feature = "backend_cuda_bf16"),
    all(feature = "backend_wgpu_bf16", feature = "backend_tch"),
    all(feature = "backend_wgpu_bf16", feature = "backend_tch_bf16"),
    all(feature = "backend_cuda", feature = "backend_cuda_bf16"),
    all(feature = "backend_cuda", feature = "backend_tch"),
    all(feature = "backend_cuda", feature = "backend_tch_bf16"),
    all(feature = "backend_cuda_bf16", feature = "backend_tch"),
    all(feature = "backend_cuda_bf16", feature = "backend_tch_bf16"),
    all(feature = "backend_tch", feature = "backend_tch_bf16"),
))]
compile_error!("backend_* features are mutually exclusive — select exactly one");

#[cfg(feature = "backend_wgpu")]
type B = burn::backend::Wgpu;

#[cfg(feature = "backend_wgpu_f16")]
type B = burn::backend::Wgpu<half::f16>;

#[cfg(feature = "backend_wgpu_bf16")]
type B = burn::backend::Wgpu<half::bf16>;

#[cfg(feature = "backend_cuda")]
type B = burn::backend::Cuda;

#[cfg(feature = "backend_cuda_bf16")]
type B = burn::backend::Cuda<half::bf16>;

#[cfg(feature = "backend_tch")]
type B = burn::backend::LibTorch;

#[cfg(feature = "backend_tch_bf16")]
type B = burn::backend::LibTorch<half::bf16>;

#[cfg(not(any(
    feature = "backend_wgpu",
    feature = "backend_wgpu_f16",
    feature = "backend_wgpu_bf16",
    feature = "backend_cuda",
    feature = "backend_cuda_bf16",
    feature = "backend_tch",
    feature = "backend_tch_bf16",
)))]
type B = burn::backend::NdArray;

use std::{path::PathBuf, process, time::Instant};

use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};
use clap::Parser;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use tracing_subscriber::{EnvFilter, fmt};

use anyhow::{Context, Result, bail};
use irodori_tts_burn::{
    backend_config::BackendConfig,
    codec::load_codec,
    inference::InferenceBuilder,
    model::unpatchify_latent,
    rf::{GuidanceConfig, SamplerParams, SamplingRequest},
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
    /// Path to the RF model safetensors checkpoint.
    #[arg(short, long)]
    checkpoint: PathBuf,

    /// Path to the DACVAE codec safetensors weights.
    #[arg(long)]
    codec_weights: PathBuf,

    /// Text to synthesise.
    #[arg(short, long)]
    text: String,

    /// Optional reference audio WAV for speaker conditioning.
    ///
    /// Must be mono; resampled to the codec sample rate if needed.
    /// When omitted, speaker conditioning is disabled.
    #[arg(long)]
    ref_audio: Option<PathBuf>,

    /// Output WAV file path.
    #[arg(short, long, default_value = "output.wav")]
    output: PathBuf,

    /// Number of RF diffusion steps.
    #[arg(long, default_value_t = 40)]
    num_steps: usize,

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

    /// Minimum timestep for CFG (0.0–1.0).
    #[arg(long, default_value_t = 0.5)]
    cfg_min_t: f32,

    /// Maximum timestep for CFG (0.0–1.0).
    #[arg(long, default_value_t = 1.0)]
    cfg_max_t: f32,

    /// Optional LoRA adapter directory (must contain `adapter_config.json`
    /// and `adapter_model.safetensors`).
    #[arg(long)]
    adapter: Option<PathBuf>,

    /// GPU device index (0-based).
    #[arg(long, default_value_t = 0)]
    gpu_id: u32,

    /// Trim trailing silence using the find-flattening-point heuristic.
    ///
    /// Enabled by default (mirrors Python infer.py).  Pass `--no-trim-tail`
    /// to disable.
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

    /// Random seed for the initial noise tensor (sets the backend RNG deterministically).
    ///
    /// When omitted the backend uses its default (non-deterministic) RNG state.
    /// Pass the same value used in `infer.py --seed <N>` to obtain bit-for-bit
    /// identical initial noise and a directly comparable output.
    #[arg(long)]
    seed: Option<u64>,

    /// Pre-computed initial noise tensor file (safetensors, key "initial_noise").
    ///
    /// When supplied, overrides `--seed`; the tensor is used directly as the
    /// starting latent x_T.  Generate with `scripts/export_initial_noise.py`.
    #[arg(long)]
    noise_file: Option<PathBuf>,
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
        shape.len() == 3,
        "initial_noise must be rank-3, got shape {shape:?}"
    );
    anyhow::ensure!(
        shape[1] == seq_len,
        "initial_noise seq_len={} but pipeline seq_len={}; re-export with --seq-len {}",
        shape[1],
        seq_len,
        seq_len,
    );
    // Always stored as f32 by the export script
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

fn load_tokenizer(repo_id: &str) -> Result<Tokenizer> {
    let api = Api::new().context("failed to initialise HF Hub API")?;
    let repo = api.model(repo_id.to_string());
    let path = repo
        .get("tokenizer.json")
        .context("failed to fetch tokenizer.json from HF Hub")?;
    Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))
}

fn tokenize<B: Backend>(
    tokenizer: &Tokenizer,
    text: &str,
    add_bos: bool,
    device: &B::Device,
) -> Result<(Tensor<B, 2, Int>, Tensor<B, 2, Bool>)> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("failed to tokenise: {e}"))?;

    let mut ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    if add_bos && let Some(bos_id) = tokenizer.token_to_id("<s>") {
        ids.insert(0, bos_id as i32);
    }
    let seq_len = ids.len();
    let input_ids = Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [1, seq_len]), device);
    let mask: Tensor<B, 2, Bool> = Tensor::<B, 2>::ones([1, seq_len], device).greater_elem(0.0f32);
    Ok((input_ids, mask))
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

/// Load a WAV, resample to `target_rate` if needed, and return a
/// `[1, 1, samples]` Burn tensor.
fn load_and_prepare_audio<B: Backend>(
    path: &std::path::Path,
    target_rate: u32,
    device: &B::Device,
) -> Result<Tensor<B, 3>> {
    let (mut samples, sr) = load_wav_as_f32(path)?;
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

/// Write a `[1, S]` or `[S]` f32 tensor as a 16-bit PCM WAV file.
fn save_wav<B: Backend>(
    path: &std::path::Path,
    audio: Tensor<B, 3>,
    sample_rate: u32,
) -> Result<()> {
    let [_batch, _ch, n_samples] = audio.dims();
    // Clamp to [-1, 1] before converting to i16
    let data = audio.clamp(-1.0f32, 1.0f32).into_data().convert::<f32>();
    let samples: Vec<f32> = data.to_vec().unwrap();

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("cannot create WAV {:?}", path))?;
    for s in &samples[..n_samples] {
        let pcm = (s * 32767.0).round() as i16;
        writer.write_sample(pcm).context("write WAV sample")?;
    }
    writer.finalize().context("finalise WAV")?;
    Ok(())
}

fn parse_cfg_mode(s: &str) -> Result<irodori_tts_burn::CfgGuidanceMode> {
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn run(args: Args) -> Result<()> {
    let device = B::device_from_id(args.gpu_id);

    if let Some(seed) = args.seed {
        tracing::info!("Seeding backend RNG with seed={seed}");
        B::seed(&device, seed);
    }

    // ── TTS model ────────────────────────────────────────────────────────────
    tracing::info!("Loading TTS model from {:?}", args.checkpoint);
    let loaded = match args.adapter {
        Some(ref dir) => {
            tracing::info!("Merging LoRA adapter from {:?}", dir);
            #[allow(clippy::clone_on_copy)]
            InferenceBuilder::<B, _>::new(device.clone())
                .load_weights_with_adapter(&args.checkpoint, dir)?
        }
        #[allow(clippy::clone_on_copy)]
        None => InferenceBuilder::<B, _>::new(device.clone()).load_weights(&args.checkpoint)?,
    };
    let cfg = loaded.model_config().clone();
    tracing::info!(
        "TTS model loaded (latent_dim={}, patch_size={})",
        cfg.latent_dim,
        cfg.latent_patch_size,
    );

    // ── Codec ────────────────────────────────────────────────────────────────
    tracing::info!("Loading DACVAE codec from {:?}", args.codec_weights);
    let codec = load_codec::<B>(&args.codec_weights, &device)?;
    tracing::info!(
        "Codec loaded (sample_rate={} Hz, hop_length={})",
        codec.sample_rate(),
        codec.hop_length()
    );

    // Validate latent_dim compatibility
    if cfg.latent_dim != 32 {
        bail!(
            "TTS latent_dim={} but DACVAE codec expects 32",
            cfg.latent_dim
        );
    }

    // ── Tokenise ─────────────────────────────────────────────────────────────
    tracing::info!("Loading tokenizer from HF Hub: {}", cfg.text_tokenizer_repo);
    let tokenizer = load_tokenizer(&cfg.text_tokenizer_repo)?;
    let normalized = irodori_tts_burn::text_normalization::normalize_text(&args.text);
    tracing::info!("Text (normalized): {normalized:?}");
    let (text_ids, text_mask) = tokenize::<B>(&tokenizer, &normalized, cfg.text_add_bos, &device)?;
    tracing::info!("Tokenised: {} tokens", text_ids.dims()[1]);

    // ── Reference audio (optional) ───────────────────────────────────────────
    let (ref_latent, ref_mask) = if let Some(ref ref_path) = args.ref_audio {
        tracing::info!("Encoding reference audio {:?}", ref_path);
        let wav = load_and_prepare_audio::<B>(ref_path, codec.sample_rate() as u32, &device)?;
        let latent = codec.encode(wav); // [1, T, 32]
        let [b, t, _d] = latent.dims();
        tracing::info!("Reference latent: [{b}, {t}, 32]");
        let mask: Tensor<B, 2, Bool> = Tensor::<B, 2>::ones([b, t], &device).greater_elem(0.0f32);
        (Some(latent), Some(mask))
    } else if cfg.use_speaker_condition() {
        // Match Python `--no-ref`: zeros ref + all-False mask so the speaker
        // encoder path is active but contributes nothing (masked-out padding).
        let speaker_patch_size = cfg.speaker_patch_size.unwrap_or(1);
        let ref_len = speaker_patch_size.max(1);
        let patched_dim = cfg.speaker_patched_latent_dim();
        let ref_latent: Tensor<B, 3> = Tensor::zeros([1, ref_len, patched_dim], &device);
        let ref_mask: Tensor<B, 2, Bool> =
            Tensor::<B, 2>::zeros([1, ref_len], &device).greater_elem(0.0f32);
        tracing::info!(
            "No reference audio — zeros dummy ref (speaker_patch_size={ref_len}, patched_dim={patched_dim})"
        );
        (Some(ref_latent), Some(ref_mask))
    } else {
        tracing::info!("No reference audio — unconditional caption mode (no speaker encoder)");
        (None, None)
    };

    // ── RF sampling ──────────────────────────────────────────────────────────
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
        "Running RF sampler: {} steps, seq_len={seq_len}",
        engine.sampling_params().num_steps
    );
    let t_sample = Instant::now();
    let z_patched = engine.sample(SamplingRequest {
        text_ids,
        text_mask,
        ref_latent,
        ref_mask,
        sequence_length: seq_len,
        caption_ids: None,
        caption_mask: None,
        initial_noise: load_initial_noise::<B>(args.noise_file.as_deref(), seq_len, &device)?,
    })?;
    let [b, s_pat, _] = z_patched.dims();
    let rf_elapsed_ms = t_sample.elapsed().as_secs_f64() * 1000.0;
    tracing::info!("Sampler done: [{b}, {s_pat}, patched_dim]  rf_time={rf_elapsed_ms:.0}ms");

    // ── Unpatchify ───────────────────────────────────────────────────────────
    let z = unpatchify_latent(z_patched, cfg.latent_patch_size, cfg.latent_dim);
    let [_, s, _] = z.dims();
    tracing::info!("Unpatchified latent: [{b}, {s}, {}]", cfg.latent_dim);

    // ── Tail trimming (find_flattening_point) ─────────────────────────────────
    let z_trimmed = if args.trim_tail {
        let z_data: Vec<f32> = z.clone().into_data().convert::<f32>().to_vec().unwrap();
        let trim_t = find_flattening_point(
            &z_data,
            s,
            cfg.latent_dim,
            args.trim_window,
            args.trim_std,
            args.trim_mean,
        );
        if trim_t < s {
            tracing::info!("Tail trimmed: {s} → {trim_t} frames");
            z.slice([0..1, 0..trim_t, 0..cfg.latent_dim])
        } else {
            tracing::info!("No tail trimming applied (full {s} frames)");
            z
        }
    } else {
        z
    };
    let [_, s_trimmed, _] = z_trimmed.dims();

    // ── DACVAE decode ────────────────────────────────────────────────────────
    tracing::info!("Decoding latent → waveform");
    let t_decode = Instant::now();
    let audio = codec.decode(z_trimmed); // [B, 1, samples]
    let [_, _, n_samples] = audio.dims();
    let codec_elapsed_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
    let duration_s = n_samples as f64 / codec.sample_rate() as f64;
    tracing::info!(
        "Audio decoded: {} samples ({:.2}s @ {} Hz), latent frames={s_trimmed}",
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
    tracing::info!("Wrote output WAV to {:?}", args.output);

    // WGPU/Vulkan atexit handlers registered by the GPU driver segfault during
    // normal process exit.  Use POSIX `_exit` to skip all atexit handlers so
    // that the OS reclaims resources cleanly.  The WAV is already flushed.
    //
    // Safety: we have no resources that need explicit cleanup beyond the WAV
    // already written, and `_exit` is always safe to call from any context.
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
    if let Err(e) = run(args) {
        tracing::error!("Fatal: {e:#}");
        process::exit(1);
    }
    unreachable!("run() always calls process::exit");
}
