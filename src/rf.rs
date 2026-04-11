//! Rectified-flow Euler sampler with classifier-free guidance (CFG).
//!
//! Ports `irodori_tts/rf.py` to Rust + burn.
//!
//! # Usage
//!
//! ```rust,ignore
//! let params = SamplerParams::default();
//! let latent = sample_euler_rf_cfg(&model, text_ids, text_mask, None, None, 100, None, None, &params, None, &device);
//! ```

use burn::tensor::{Bool, Int, Tensor, backend::Backend};

use crate::{
    config::CfgGuidanceMode,
    model::{EncodedCondition, TextToLatentRfDiT},
};

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

/// CFG guidance strength and scheduling parameters.
#[derive(Debug, Clone)]
pub struct GuidanceConfig {
    /// How to combine multiple guidance signals.
    pub mode: CfgGuidanceMode,
    /// CFG scale for text conditioning.
    pub scale_text: f32,
    /// CFG scale for caption conditioning.
    pub scale_caption: f32,
    /// CFG scale for speaker conditioning.
    pub scale_speaker: f32,
    /// Minimum timestep at which to apply CFG (inclusive).
    pub min_t: f32,
    /// Maximum timestep at which to apply CFG (inclusive).
    pub max_t: f32,
}

impl Default for GuidanceConfig {
    fn default() -> Self {
        Self {
            mode: CfgGuidanceMode::Independent,
            scale_text: 3.0,
            scale_caption: 3.0,
            scale_speaker: 5.0,
            min_t: 0.5,
            max_t: 1.0,
        }
    }
}

/// Parameters for temporal score rescaling (arxiv:2510.01184).
///
/// Both `k` and `sigma` must be set together — they form a single coupled pair.
#[derive(Debug, Clone, Copy)]
pub struct TemporalRescaleConfig {
    pub k: f32,
    pub sigma: f32,
}

/// Force-speaker KV scaling configuration.
///
/// Scales the speaker K/V projections to amplify speaker conditioning.
#[derive(Debug, Clone)]
pub struct SpeakerKvConfig {
    /// Scale factor applied to speaker K/V tensors.
    pub scale: f32,
    /// Limit scaling to the first `N` layers; `None` = all layers.
    pub max_layers: Option<usize>,
    /// Revert scaling once `t` drops below this threshold; `None` = never revert.
    pub min_t: Option<f32>,
}

/// Parameters for Euler sampling with CFG.
#[derive(Debug, Clone)]
pub struct SamplerParams {
    /// Number of denoising steps.
    pub num_steps: usize,
    /// CFG guidance configuration.
    pub guidance: GuidanceConfig,
    /// If `Some(k)`, multiply initial Gaussian noise by `k < 1` to truncate tails.
    pub truncation_factor: Option<f32>,
    /// Temporal score rescaling (arxiv:2510.01184).  `None` = disabled.
    pub temporal_rescale: Option<TemporalRescaleConfig>,
    /// Force-speaker KV scaling.  `None` = disabled.
    pub speaker_kv: Option<SpeakerKvConfig>,
    /// Cache projected context K/V tensors across denoising steps for speed.
    pub use_context_kv_cache: bool,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            num_steps: 40,
            guidance: GuidanceConfig::default(),
            truncation_factor: None,
            temporal_rescale: None,
            speaker_kv: None,
            use_context_kv_cache: true,
        }
    }
}

impl SamplerParams {
    /// Validate the parameters, returning a typed error on invalid combinations.
    ///
    /// Call this (or use [`InferenceEngine::sample`] which calls it automatically)
    /// before running the sampler.
    pub fn validate(&self) -> crate::error::Result<()> {
        use crate::error::IrodoriError;

        if self.num_steps == 0 {
            return Err(IrodoriError::Config("num_steps must be > 0".to_string()));
        }
        if self
            .truncation_factor
            .is_some_and(|k| k <= 0.0 || !k.is_finite())
        {
            return Err(IrodoriError::Config(
                "truncation_factor must be finite and > 0".to_string(),
            ));
        }
        if let Some(trc) = self.temporal_rescale
            && trc.sigma == 0.0 {
                return Err(IrodoriError::Config(
                    "temporal_rescale.sigma must not be zero".to_string(),
                ));
            }        Ok(())
    }
}

impl From<crate::config::SamplingConfig> for SamplerParams {
    fn from(cfg: crate::config::SamplingConfig) -> Self {
        // A legacy `cfg_scale` overrides all three per-signal scales.
        let (scale_text, scale_speaker, scale_caption) = if let Some(s) = cfg.cfg_scale {
            let s = s as f32;
            (s, s, s)
        } else {
            (
                cfg.cfg_scale_text as f32,
                cfg.cfg_scale_speaker as f32,
                cfg.cfg_scale_caption as f32,
            )
        };

        Self {
            num_steps: cfg.num_steps,
            guidance: GuidanceConfig {
                mode: cfg.cfg_guidance_mode,
                scale_text,
                scale_caption,
                scale_speaker,
                min_t: cfg.cfg_min_t as f32,
                max_t: cfg.cfg_max_t as f32,
            },
            truncation_factor: cfg.truncation_factor.map(|v| v as f32),
            temporal_rescale: match (cfg.rescale_k, cfg.rescale_sigma) {
                (Some(k), Some(sigma)) => Some(TemporalRescaleConfig {
                    k: k as f32,
                    sigma: sigma as f32,
                }),
                _ => None,
            },
            speaker_kv: cfg.speaker_kv_scale.map(|scale| SpeakerKvConfig {
                scale: scale as f32,
                min_t: cfg.speaker_kv_min_t.map(|v| v as f32),
                max_layers: cfg.speaker_kv_max_layers,
            }),
            use_context_kv_cache: cfg.context_kv_cache,
        }
    }
}

/// All per-call inputs to [`sample_euler_rf_cfg`].
///
/// Groups the per-request tensors that change between calls so they don't
/// pollute the function signature.
#[derive(Debug)]
pub struct SamplingRequest<B: Backend> {
    pub text_ids: Tensor<B, 2, Int>,
    pub text_mask: Tensor<B, 2, Bool>,
    /// Optional reference audio latent `[1, T, D]`.
    pub ref_latent: Option<Tensor<B, 3>>,
    pub ref_mask: Option<Tensor<B, 2, Bool>>,
    /// Number of output latent frames to generate.
    pub sequence_length: usize,
    /// Optional caption token ids for caption conditioning.
    pub caption_ids: Option<Tensor<B, 2, Int>>,
    pub caption_mask: Option<Tensor<B, 2, Bool>>,
    /// Pre-generated initial noise for reproducibility; `None` = sample fresh.
    pub initial_noise: Option<Tensor<B, 3>>,
}

// ---------------------------------------------------------------------------
// Core RF math
// ---------------------------------------------------------------------------

/// Straight-line interpolation: `x_t = (1 - t) * x0 + t * noise`.
///
/// `t: [B]`, `x0 / noise: [B, S, D]`.
pub fn rf_interpolate<B: Backend>(
    x0: Tensor<B, 3>,
    noise: Tensor<B, 3>,
    t: Tensor<B, 1>,
) -> Tensor<B, 3> {
    let t3 = t.unsqueeze_dim::<2>(1).unsqueeze_dim::<3>(2); // [B, 1, 1]
    (Tensor::ones_like(&t3) - t3.clone()) * x0 + t3 * noise
}

/// RF velocity target: `v = noise - x0`.
pub fn rf_velocity_target<B: Backend>(x0: Tensor<B, 3>, noise: Tensor<B, 3>) -> Tensor<B, 3> {
    noise - x0
}

/// Recover clean sample from noisy + predicted velocity:
/// `x0 = x_t - t * v_pred`.
pub fn rf_predict_x0<B: Backend>(
    x_t: Tensor<B, 3>,
    v_pred: Tensor<B, 3>,
    t: Tensor<B, 1>,
) -> Tensor<B, 3> {
    let t3 = t.unsqueeze_dim::<2>(1).unsqueeze_dim::<3>(2); // [B, 1, 1]
    x_t - t3 * v_pred
}

/// Temporal score rescaling (arxiv:2510.01184).
///
/// No-op when `t >= 1`.
pub fn temporal_score_rescale<B: Backend>(
    v_pred: Tensor<B, 3>,
    x_t: Tensor<B, 3>,
    t: f32,
    rescale_k: f32,
    rescale_sigma: f32,
) -> Tensor<B, 3> {
    if t >= 1.0 {
        return v_pred;
    }
    let one_minus_t = 1.0 - t;
    let snr = (one_minus_t * one_minus_t) / (t * t);
    let sigma_sq = rescale_sigma * rescale_sigma;
    let ratio = (snr * sigma_sq + 1.0) / (snr * sigma_sq / rescale_k + 1.0);
    (v_pred * one_minus_t + x_t.clone()) * ratio / one_minus_t - x_t / one_minus_t
}

// ---------------------------------------------------------------------------
// KV-cache helpers
// ---------------------------------------------------------------------------

use crate::model::attention::CondKvCache;

/// Scale the speaker (aux) K/V tensors in a precomputed KV cache.
///
/// Returns a new `Vec` — burn tensors are immutable.
pub fn scale_speaker_kv_cache<B: Backend>(
    caches: Vec<CondKvCache<B>>,
    scale: f32,
    max_layers: Option<usize>,
) -> Vec<CondKvCache<B>> {
    let n = max_layers.map_or(caches.len(), |m| m.min(caches.len()));
    caches
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            if i < n {
                CondKvCache {
                    aux_k: c.aux_k.map(|k| k * scale),
                    aux_v: c.aux_v.map(|v| v * scale),
                    ..c
                }
            } else {
                c
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Main sampler
// ---------------------------------------------------------------------------

/// Euler sampler over the RF ODE with classifier-free guidance.
///
/// Returns the denoised latent: `[batch, sequence_length, patched_latent_dim]`.
///
/// # Parameters
/// - `request.initial_noise`: supply pre-generated noise for reproducibility; if `None`
///   a standard Gaussian is sampled from burn's RNG.
///
/// # Errors
///
/// Returns [`IrodoriError::Config`] if `params` fails validation (e.g. `num_steps == 0`,
/// Joint mode with mismatched CFG scales).
pub fn sample_euler_rf_cfg<B: Backend>(
    model: &TextToLatentRfDiT<B>,
    request: SamplingRequest<B>,
    params: &SamplerParams,
    device: &B::Device,
) -> crate::error::Result<Tensor<B, 3>> {
    use crate::error::IrodoriError;

    params.validate()?;

    let batch_size = request.text_ids.dims()[0];
    let latent_dim = model.patched_latent_dim();

    // --- Initial noise ---
    let mut x_t = request.initial_noise.unwrap_or_else(|| {
        Tensor::random(
            [batch_size, request.sequence_length, latent_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        )
    });
    if let Some(k) = params.truncation_factor {
        x_t = x_t * k;
    }

    let g = &params.guidance;

    // Resolve effective CFG scales
    let cfg_scale_text = g.scale_text;
    let cfg_scale_caption = g.scale_caption;
    let cfg_scale_speaker = if model.use_speaker_condition() {
        g.scale_speaker
    } else {
        0.0
    };

    // --- Encode conditioned state once ---
    let cond = model.encode_conditions(
        request.text_ids,
        request.text_mask,
        request.ref_latent,
        request.ref_mask,
        request.caption_ids,
        request.caption_mask,
    );
    let uncond = cond.zeros_like(device);

    // Which CFG signals are active?
    let has_text_cfg = cfg_scale_text > 0.0;
    // Speaker CFG is only meaningful when a reference audio was actually provided.
    let has_speaker_cfg =
        cfg_scale_speaker > 0.0 && model.use_speaker_condition() && cond.speaker_state.is_some();
    let has_caption_cfg = cfg_scale_caption > 0.0
        && cond.caption_mask.as_ref().is_some_and(|m| {
            let flat: Vec<i32> = m.clone().int().to_data().to_vec().unwrap_or_default();
            flat.iter().any(|&v| v > 0)
        });

    // Build list of active CFG names (determines alternating order)
    let mut enabled_cfg: Vec<CfgName> = Vec::new();
    if has_text_cfg {
        enabled_cfg.push(CfgName::Text);
    }
    if has_speaker_cfg {
        enabled_cfg.push(CfgName::Speaker);
    }
    if has_caption_cfg {
        enabled_cfg.push(CfgName::Caption);
    }

    // Joint CFG requires all active guidance scales to be equal.
    if matches!(g.mode, CfgGuidanceMode::Joint) && !enabled_cfg.is_empty() {
        let active_scales: Vec<f32> = enabled_cfg
            .iter()
            .map(|n| cfg_scale_for(n, cfg_scale_text, cfg_scale_speaker, cfg_scale_caption))
            .collect();
        let min_s = active_scales.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_s = active_scales
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        if max_s - min_s >= 1e-6 {
            return Err(IrodoriError::Config(format!(
                "cfg_guidance_mode=Joint requires all active cfg scales to be equal, \
                 but got text={}, speaker={}, caption={}. \
                 Pass a single equal value for all active signals.",
                g.scale_text, g.scale_speaker, g.scale_caption
            )));
        }
    }

    // --- Precompute KV caches ---
    let effective_kv_cache = params.use_context_kv_cache || params.speaker_kv.is_some();

    let mut kv_cond: Option<Vec<CondKvCache<B>>> =
        effective_kv_cache.then(|| model.build_kv_caches(&cond));

    // Scale speaker K/V if requested — only valid in speaker mode (not caption mode).
    if let Some(ref skv) = params.speaker_kv
        && model.use_speaker_condition() {
            kv_cond = kv_cond.map(|cache| scale_speaker_kv_cache(cache, skv.scale, skv.max_layers));
        }

    // Pre-build uncond/alternating KV caches for non-independent CFG modes
    let kv_uncond: Option<Vec<CondKvCache<B>>> = if effective_kv_cache
        && !matches!(g.mode, CfgGuidanceMode::Independent)
        && !enabled_cfg.is_empty()
    {
        Some(model.build_kv_caches(&uncond))
    } else {
        None
    };

    // Alternating per-signal uncond caches
    let kv_alt_text: Option<Vec<CondKvCache<B>>> =
        if effective_kv_cache && matches!(g.mode, CfgGuidanceMode::Alternating) {
            has_text_cfg.then(|| {
                let uncond_text = make_text_uncond(&cond, &uncond, device);
                model.build_kv_caches(&uncond_text)
            })
        } else {
            None
        };
    let kv_alt_speaker: Option<Vec<CondKvCache<B>>> =
        if effective_kv_cache && matches!(g.mode, CfgGuidanceMode::Alternating) {
            has_speaker_cfg.then(|| {
                let uncond_spk = make_speaker_uncond(&cond, &uncond, device);
                model.build_kv_caches(&uncond_spk)
            })
        } else {
            None
        };
    let kv_alt_caption: Option<Vec<CondKvCache<B>>> =
        if effective_kv_cache && matches!(g.mode, CfgGuidanceMode::Alternating) {
            has_caption_cfg.then(|| {
                let uncond_cap = make_caption_uncond(&cond, &uncond, device);
                model.build_kv_caches(&uncond_cap)
            })
        } else {
            None
        };

    // --- Timestep schedule: linearly spaced [0.999, 0] ---
    let init_scale = 0.999_f32;
    let t_schedule: Vec<f32> = (0..=params.num_steps)
        .map(|i| init_scale * (1.0 - i as f32 / params.num_steps as f32))
        .collect();

    let mut speaker_kv_active = params.speaker_kv.is_some();

    // --- Euler ODE loop ---
    for i in 0..params.num_steps {
        let t = t_schedule[i];
        let t_next = t_schedule[i + 1];
        let tt = Tensor::from_floats([t].repeat(batch_size).as_slice(), device); // [B]

        let use_cfg = !enabled_cfg.is_empty() && g.min_t <= t && t <= g.max_t;

        let kv_cond_ref = kv_cond.as_deref();

        let v = if use_cfg {
            match g.mode {
                CfgGuidanceMode::Independent => {
                    let v_cond =
                        model.forward_with_cond(x_t.clone(), tt.clone(), &cond, None, kv_cond_ref);
                    let mut v_out = v_cond.clone();
                    for name in &enabled_cfg {
                        let alt = make_single_uncond(name, &cond, &uncond, device);
                        let kv_alt_ref: Option<&[CondKvCache<B>]> = match name {
                            CfgName::Text => kv_alt_text.as_deref(),
                            CfgName::Speaker => kv_alt_speaker.as_deref(),
                            CfgName::Caption => kv_alt_caption.as_deref(),
                        };
                        let v_alt = model.forward_with_cond(
                            x_t.clone(),
                            tt.clone(),
                            &alt,
                            None,
                            kv_alt_ref,
                        );
                        let scale = cfg_scale_for(
                            name,
                            cfg_scale_text,
                            cfg_scale_speaker,
                            cfg_scale_caption,
                        );
                        v_out = v_out + (v_cond.clone() - v_alt) * scale;
                    }
                    v_out
                }
                CfgGuidanceMode::Joint => {
                    let v_cond =
                        model.forward_with_cond(x_t.clone(), tt.clone(), &cond, None, kv_cond_ref);
                    if enabled_cfg.is_empty() {
                        v_cond
                    } else {
                        let joint_scale = cfg_scale_for(
                            &enabled_cfg[0],
                            cfg_scale_text,
                            cfg_scale_speaker,
                            cfg_scale_caption,
                        );
                        let v_uncond = model.forward_with_cond(
                            x_t.clone(),
                            tt.clone(),
                            &uncond,
                            None,
                            kv_uncond.as_deref(),
                        );
                        v_cond.clone() + (v_cond - v_uncond) * joint_scale
                    }
                }
                CfgGuidanceMode::Alternating => {
                    let v_cond =
                        model.forward_with_cond(x_t.clone(), tt.clone(), &cond, None, kv_cond_ref);
                    if enabled_cfg.is_empty() {
                        v_cond
                    } else {
                        let alt_name = &enabled_cfg[i % enabled_cfg.len()];
                        let alt_cond = make_single_uncond(alt_name, &cond, &uncond, device);
                        let kv_alt_ref: Option<&[CondKvCache<B>]> = match alt_name {
                            CfgName::Text => kv_alt_text.as_deref(),
                            CfgName::Speaker => kv_alt_speaker.as_deref(),
                            CfgName::Caption => kv_alt_caption.as_deref(),
                        };
                        let v_alt = model.forward_with_cond(
                            x_t.clone(),
                            tt.clone(),
                            &alt_cond,
                            None,
                            kv_alt_ref,
                        );
                        let scale = cfg_scale_for(
                            alt_name,
                            cfg_scale_text,
                            cfg_scale_speaker,
                            cfg_scale_caption,
                        );
                        v_cond.clone() + (v_cond - v_alt) * scale
                    }
                }
            }
        } else {
            model.forward_with_cond(x_t.clone(), tt.clone(), &cond, None, kv_cond_ref)
        };

        // Temporal score rescaling
        let v = if let Some(trc) = params.temporal_rescale {
            temporal_score_rescale(v, x_t.clone(), t, trc.k, trc.sigma)
        } else {
            v
        };

        // Disable force-speaker scaling once t crosses the threshold
        if speaker_kv_active
            && model.use_speaker_condition()
            && let Some(ref skv) = params.speaker_kv
            && let Some(_) = skv.min_t.filter(|&min_t| t_next < min_t && t >= min_t)
        {
            let inv_scale = 1.0 / skv.scale;
            kv_cond = kv_cond.map(|cache| scale_speaker_kv_cache(cache, inv_scale, skv.max_layers));
            speaker_kv_active = false;
        }

        // Euler step: x_{t+dt} = x_t + v * dt   (dt = t_next - t, which is negative)
        let dt = t_next - t;
        x_t = x_t + v * dt;
    }

    Ok(x_t)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Which conditioning signal to drop for a particular CFG bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
enum CfgName {
    Text,
    Speaker,
    Caption,
}

fn cfg_scale_for(name: &CfgName, text: f32, speaker: f32, caption: f32) -> f32 {
    match name {
        CfgName::Text => text,
        CfgName::Speaker => speaker,
        CfgName::Caption => caption,
    }
}

/// Build an `EncodedCondition` that nullifies only the text signal.
fn make_text_uncond<B: Backend>(
    cond: &EncodedCondition<B>,
    uncond: &EncodedCondition<B>,
    _device: &B::Device,
) -> EncodedCondition<B> {
    EncodedCondition {
        text_state: uncond.text_state.clone(),
        text_mask: uncond.text_mask.clone(),
        speaker_state: cond.speaker_state.clone(),
        speaker_mask: cond.speaker_mask.clone(),
        caption_state: cond.caption_state.clone(),
        caption_mask: cond.caption_mask.clone(),
    }
}

/// Build an `EncodedCondition` that nullifies only the speaker signal.
fn make_speaker_uncond<B: Backend>(
    cond: &EncodedCondition<B>,
    uncond: &EncodedCondition<B>,
    _device: &B::Device,
) -> EncodedCondition<B> {
    EncodedCondition {
        text_state: cond.text_state.clone(),
        text_mask: cond.text_mask.clone(),
        speaker_state: uncond.speaker_state.clone(),
        speaker_mask: uncond.speaker_mask.clone(),
        caption_state: cond.caption_state.clone(),
        caption_mask: cond.caption_mask.clone(),
    }
}

/// Build an `EncodedCondition` that nullifies only the caption signal.
fn make_caption_uncond<B: Backend>(
    cond: &EncodedCondition<B>,
    uncond: &EncodedCondition<B>,
    _device: &B::Device,
) -> EncodedCondition<B> {
    EncodedCondition {
        text_state: cond.text_state.clone(),
        text_mask: cond.text_mask.clone(),
        speaker_state: cond.speaker_state.clone(),
        speaker_mask: cond.speaker_mask.clone(),
        caption_state: uncond.caption_state.clone(),
        caption_mask: uncond.caption_mask.clone(),
    }
}

/// Build an `EncodedCondition` that nullifies a single named signal.
fn make_single_uncond<'a, B: Backend>(
    name: &CfgName,
    cond: &'a EncodedCondition<B>,
    uncond: &'a EncodedCondition<B>,
    device: &B::Device,
) -> EncodedCondition<B> {
    match name {
        CfgName::Text => make_text_uncond(cond, uncond, device),
        CfgName::Speaker => make_speaker_uncond(cond, uncond, device),
        CfgName::Caption => make_caption_uncond(cond, uncond, device),
    }
}
