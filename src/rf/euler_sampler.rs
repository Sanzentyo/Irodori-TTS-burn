//! Euler sampler over the RF ODE with classifier-free guidance (CFG).

use burn::tensor::{Tensor, backend::Backend};

use crate::{
    config::CfgGuidanceMode,
    model::{
        EncodedCondition, TextToLatentRfDiT,
        condition::{AuxConditionInput, AuxConditionState},
    },
    nvtx_range,
};

use super::kv_scaling::scale_speaker_kv_cache;
use super::math::temporal_score_rescale;
use super::params::{SamplerParams, SamplingRequest};

use crate::model::attention::CondKvCache;

// ---------------------------------------------------------------------------
// Private CFG helpers
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
) -> EncodedCondition<B> {
    EncodedCondition {
        text_state: uncond.text_state.clone(),
        text_mask: uncond.text_mask.clone(),
        aux: cond.aux.clone(),
    }
}

/// Build an `EncodedCondition` that nullifies the auxiliary conditioning signal.
///
/// Since speaker and caption are mutually exclusive, both `Speaker` and
/// `Caption` CFG nullification use the same structure: keep real text, swap
/// to zeroed auxiliary state.
fn make_aux_uncond<B: Backend>(
    cond: &EncodedCondition<B>,
    uncond: &EncodedCondition<B>,
) -> EncodedCondition<B> {
    EncodedCondition {
        text_state: cond.text_state.clone(),
        text_mask: cond.text_mask.clone(),
        aux: uncond.aux.clone(),
    }
}

/// Build an `EncodedCondition` that nullifies a single named signal.
fn make_single_uncond<B: Backend>(
    name: &CfgName,
    cond: &EncodedCondition<B>,
    uncond: &EncodedCondition<B>,
    _device: &B::Device,
) -> EncodedCondition<B> {
    match name {
        CfgName::Text => make_text_uncond(cond, uncond),
        // Speaker and caption are mutually exclusive; both nullify `aux`.
        CfgName::Speaker | CfgName::Caption => make_aux_uncond(cond, uncond),
    }
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
    let aux_input = AuxConditionInput::from_request(
        request.ref_latent,
        request.ref_mask,
        request.caption_ids,
        request.caption_mask,
    );
    let cond = model.encode_conditions(request.text_ids, request.text_mask, aux_input)?;
    let uncond = cond.zeros_like(device);

    // Precompute RoPE tables for the latent sequence once — reused across all 40 × 3 forward passes.
    let lat_rope = model.precompute_latent_rope(request.sequence_length, device);

    // Which CFG signals are active?
    let has_text_cfg = cfg_scale_text > 0.0;
    // Speaker CFG is only meaningful when a reference audio was actually provided.
    let has_speaker_cfg =
        cfg_scale_speaker > 0.0 && matches!(&cond.aux, Some(AuxConditionState::Speaker { .. }));
    let has_caption_cfg = cfg_scale_caption > 0.0 && {
        if let Some(AuxConditionState::Caption { mask, .. }) = &cond.aux {
            let flat: Vec<i32> = mask.clone().int().to_data().to_vec().unwrap_or_default();
            flat.iter().any(|&v| v > 0)
        } else {
            false
        }
    };

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
        && model.use_speaker_condition()
    {
        kv_cond = kv_cond.map(|cache| scale_speaker_kv_cache(cache, skv.scale, skv.max_layers));
    }

    // Joint mode: one shared fully-unconditioned pass per step.
    let kv_uncond: Option<Vec<CondKvCache<B>>> = if effective_kv_cache
        && matches!(g.mode, CfgGuidanceMode::Joint)
        && !enabled_cfg.is_empty()
    {
        Some(model.build_kv_caches(&uncond))
    } else {
        None
    };

    // Independent / Alternating: per-signal unconditioned caches.
    let use_alt_caches =
        effective_kv_cache && !matches!(g.mode, CfgGuidanceMode::Joint) && !enabled_cfg.is_empty();
    let kv_alt_text: Option<Vec<CondKvCache<B>>> = use_alt_caches
        .then(|| {
            has_text_cfg.then(|| {
                let uncond_text = make_text_uncond(&cond, &uncond);
                model.build_kv_caches(&uncond_text)
            })
        })
        .flatten();
    let kv_alt_speaker: Option<Vec<CondKvCache<B>>> = use_alt_caches
        .then(|| {
            has_speaker_cfg.then(|| {
                let uncond_spk = make_aux_uncond(&cond, &uncond);
                model.build_kv_caches(&uncond_spk)
            })
        })
        .flatten();
    let kv_alt_caption: Option<Vec<CondKvCache<B>>> = use_alt_caches
        .then(|| {
            has_caption_cfg.then(|| {
                let uncond_cap = make_aux_uncond(&cond, &uncond);
                model.build_kv_caches(&uncond_cap)
            })
        })
        .flatten();

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

        {
            if tracing::enabled!(tracing::Level::DEBUG) {
                let x_data: Vec<f32> = x_t.clone().into_data().convert::<f32>().to_vec().unwrap();
                let mean = x_data.iter().sum::<f32>() / x_data.len() as f32;
                let std = (x_data.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                    / x_data.len() as f32)
                    .sqrt();
                let min = x_data.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = x_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                tracing::debug!(
                    "RF step {i}: t={t:.4} x_t min={min:.4} max={max:.4} mean={mean:.4} std={std:.4}"
                );
            }
        }

        let use_cfg = !enabled_cfg.is_empty() && g.min_t <= t && t <= g.max_t;

        let kv_cond_ref = kv_cond.as_deref();

        let _step_label = format!("euler_step_{i}");
        let v = nvtx_range!(_step_label.as_str(), {
            if use_cfg {
                match g.mode {
                    CfgGuidanceMode::Independent => {
                        let v_cond = nvtx_range!(
                            "forward_cond",
                            model.forward_with_cond_cached(
                                x_t.clone(),
                                tt.clone(),
                                &cond,
                                None,
                                kv_cond_ref,
                                &lat_rope,
                            )
                        );
                        let mut v_out = v_cond.clone();
                        for name in &enabled_cfg {
                            let alt = make_single_uncond(name, &cond, &uncond, device);
                            let kv_alt_ref: Option<&[CondKvCache<B>]> = match name {
                                CfgName::Text => kv_alt_text.as_deref(),
                                CfgName::Speaker => kv_alt_speaker.as_deref(),
                                CfgName::Caption => kv_alt_caption.as_deref(),
                            };
                            let v_alt = nvtx_range!(
                                "forward_uncond",
                                model.forward_with_cond_cached(
                                    x_t.clone(),
                                    tt.clone(),
                                    &alt,
                                    None,
                                    kv_alt_ref,
                                    &lat_rope,
                                )
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
                        let v_cond = nvtx_range!(
                            "forward_cond",
                            model.forward_with_cond_cached(
                                x_t.clone(),
                                tt.clone(),
                                &cond,
                                None,
                                kv_cond_ref,
                                &lat_rope,
                            )
                        );
                        if enabled_cfg.is_empty() {
                            v_cond
                        } else {
                            let joint_scale = cfg_scale_for(
                                &enabled_cfg[0],
                                cfg_scale_text,
                                cfg_scale_speaker,
                                cfg_scale_caption,
                            );
                            let v_uncond = nvtx_range!(
                                "forward_uncond",
                                model.forward_with_cond_cached(
                                    x_t.clone(),
                                    tt.clone(),
                                    &uncond,
                                    None,
                                    kv_uncond.as_deref(),
                                    &lat_rope,
                                )
                            );
                            v_cond.clone() + (v_cond - v_uncond) * joint_scale
                        }
                    }
                    CfgGuidanceMode::Alternating => {
                        let v_cond = nvtx_range!(
                            "forward_cond",
                            model.forward_with_cond_cached(
                                x_t.clone(),
                                tt.clone(),
                                &cond,
                                None,
                                kv_cond_ref,
                                &lat_rope,
                            )
                        );
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
                            let v_alt = nvtx_range!(
                                "forward_uncond",
                                model.forward_with_cond_cached(
                                    x_t.clone(),
                                    tt.clone(),
                                    &alt_cond,
                                    None,
                                    kv_alt_ref,
                                    &lat_rope,
                                )
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
                nvtx_range!(
                    "forward_uncfg",
                    model.forward_with_cond_cached(
                        x_t.clone(),
                        tt.clone(),
                        &cond,
                        None,
                        kv_cond_ref,
                        &lat_rope,
                    )
                )
            }
        });

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

        {
            if tracing::enabled!(tracing::Level::DEBUG) {
                let v_data: Vec<f32> = v.clone().into_data().convert::<f32>().to_vec().unwrap();
                let mean = v_data.iter().sum::<f32>() / v_data.len() as f32;
                let std = (v_data.iter().map(|a| (a - mean).powi(2)).sum::<f32>()
                    / v_data.len() as f32)
                    .sqrt();
                let min = v_data.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = v_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                tracing::debug!(
                    "RF step {i}: v min={min:.4} max={max:.4} mean={mean:.4} std={std:.4}"
                );
            }
        }

        // Euler step: x_{t+dt} = x_t + v * dt   (dt = t_next - t, which is negative)
        let dt = t_next - t;
        x_t = x_t + v * dt;
    }

    Ok(x_t)
}
