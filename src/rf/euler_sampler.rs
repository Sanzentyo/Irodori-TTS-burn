//! Euler sampler over the RF ODE with classifier-free guidance (CFG).

use burn::tensor::{Tensor, backend::Backend};

use crate::{
    config::CfgGuidanceMode,
    model::{
        EncodedCondition, InferenceOptimizedModel,
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
    model: &InferenceOptimizedModel<B>,
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
            // Use backend-agnostic conversion: bool tensor → data → Vec<bool>.
            // Previous i32 approach silently failed on LibTorch (IntElem = i64).
            let flat: Vec<bool> = mask.clone().into_data().to_vec().unwrap_or_default();
            flat.iter().any(|&v| v)
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
    let seq_lat = request.sequence_length;

    let mut kv_cond: Option<Vec<CondKvCache<B>>> =
        effective_kv_cache.then(|| model.build_kv_caches(&cond, Some(seq_lat)));

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
        Some(model.build_kv_caches(&uncond, Some(seq_lat)))
    } else {
        None
    };

    // --- Batched Independent CFG ---
    //
    // Instead of N sequential forward passes (1 conditioned + per-signal unconditioned),
    // concatenate all conditioning variants along the batch dimension and run a single
    // forward pass with batch=cfg_batch_mult.  This matches the Python implementation
    // and significantly reduces GPU kernel launch overhead (~20% per step).
    let use_batched_independent =
        matches!(g.mode, CfgGuidanceMode::Independent) && !enabled_cfg.is_empty();
    let cfg_batch_mult = if use_batched_independent {
        1 + enabled_cfg.len() // conditioned + one per enabled signal
    } else {
        1
    };

    // Pre-concatenated condition for batched Independent CFG.
    let batched_cfg_cond: Option<EncodedCondition<B>> = use_batched_independent.then(|| {
        let uncond_bundles: Vec<EncodedCondition<B>> = enabled_cfg
            .iter()
            .map(|name| make_single_uncond(name, &cond, &uncond, device))
            .collect();
        let mut refs: Vec<&EncodedCondition<B>> = Vec::with_capacity(cfg_batch_mult);
        refs.push(&cond);
        refs.extend(uncond_bundles.iter());
        EncodedCondition::cat_batch(&refs)
    });

    // Batched KV cache for Independent CFG (includes speaker scaling).
    let mut kv_batched_cfg: Option<Vec<CondKvCache<B>>> = batched_cfg_cond
        .as_ref()
        .filter(|_| effective_kv_cache)
        .map(|bc| {
            let mut cache = model.build_kv_caches(bc, Some(seq_lat));
            if let Some(ref skv) = params.speaker_kv
                && model.use_speaker_condition()
            {
                cache = scale_speaker_kv_cache(cache, skv.scale, skv.max_layers);
            }
            cache
        });

    // Alternating mode: per-signal unconditioned caches (not used by Independent).
    let use_alt_caches = effective_kv_cache
        && matches!(g.mode, CfgGuidanceMode::Alternating)
        && !enabled_cfg.is_empty();
    let kv_alt_text: Option<Vec<CondKvCache<B>>> = use_alt_caches
        .then(|| {
            has_text_cfg.then(|| {
                let uncond_text = make_text_uncond(&cond, &uncond);
                model.build_kv_caches(&uncond_text, Some(seq_lat))
            })
        })
        .flatten();
    let kv_alt_speaker: Option<Vec<CondKvCache<B>>> = use_alt_caches
        .then(|| {
            has_speaker_cfg.then(|| {
                let uncond_spk = make_aux_uncond(&cond, &uncond);
                model.build_kv_caches(&uncond_spk, Some(seq_lat))
            })
        })
        .flatten();
    let kv_alt_caption: Option<Vec<CondKvCache<B>>> = use_alt_caches
        .then(|| {
            has_caption_cfg.then(|| {
                let uncond_cap = make_aux_uncond(&cond, &uncond);
                model.build_kv_caches(&uncond_cap, Some(seq_lat))
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

        #[cfg(feature = "profile")]
        let _step_label = format!("euler_step_{i}");
        #[cfg(not(feature = "profile"))]
        let _step_label = "";
        let v = nvtx_range!(&_step_label, {
            if use_cfg {
                match g.mode {
                    CfgGuidanceMode::Independent => {
                        // Batched forward: one pass with batch=cfg_batch_mult
                        // instead of cfg_batch_mult sequential passes.
                        let batched_cond =
                            batched_cfg_cond.as_ref().expect("batched cond must exist");
                        let x_t_cfg = Tensor::cat(vec![x_t.clone(); cfg_batch_mult], 0);
                        let tt_cfg = tt.clone().repeat(&[cfg_batch_mult]);

                        let kv_ref = kv_batched_cfg.as_deref();
                        let v_out = nvtx_range!(
                            "forward_batched_cfg",
                            model.forward_with_cond_cached(
                                x_t_cfg,
                                tt_cfg,
                                batched_cond,
                                None,
                                kv_ref,
                                &lat_rope,
                            )
                        );

                        // Split output: chunks[0] = conditioned, chunks[1..] = unconditioned
                        let chunks = v_out.chunk(cfg_batch_mult, 0);
                        let v_cond = &chunks[0];
                        let mut v = v_cond.clone();
                        for (idx, name) in enabled_cfg.iter().enumerate() {
                            let scale = cfg_scale_for(
                                name,
                                cfg_scale_text,
                                cfg_scale_speaker,
                                cfg_scale_caption,
                            );
                            v = v + (v_cond.clone() - chunks[idx + 1].clone()) * scale;
                        }
                        v
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
            kv_batched_cfg = kv_batched_cfg
                .map(|cache| scale_speaker_kv_cache(cache, inv_scale, skv.max_layers));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfg_scale_for_returns_correct_scale() {
        assert_eq!(cfg_scale_for(&CfgName::Text, 3.0, 5.0, 2.0), 3.0);
        assert_eq!(cfg_scale_for(&CfgName::Speaker, 3.0, 5.0, 2.0), 5.0);
        assert_eq!(cfg_scale_for(&CfgName::Caption, 3.0, 5.0, 2.0), 2.0);
    }

    #[test]
    fn timestep_schedule_shape_and_endpoints() {
        let num_steps = 40;
        let init_scale = 0.999_f32;
        let t_schedule: Vec<f32> = (0..=num_steps)
            .map(|i| init_scale * (1.0 - i as f32 / num_steps as f32))
            .collect();

        assert_eq!(t_schedule.len(), num_steps + 1);
        assert!(
            (t_schedule[0] - 0.999).abs() < 1e-6,
            "first step should be 0.999"
        );
        assert!(
            (t_schedule[num_steps]).abs() < 1e-6,
            "last step should be ~0"
        );
        // Monotonically decreasing
        for w in t_schedule.windows(2) {
            assert!(w[0] > w[1], "schedule must be strictly decreasing");
        }
    }

    #[test]
    fn timestep_schedule_uniform_spacing() {
        let num_steps = 10;
        let init_scale = 0.999_f32;
        let t_schedule: Vec<f32> = (0..=num_steps)
            .map(|i| init_scale * (1.0 - i as f32 / num_steps as f32))
            .collect();

        let dt = t_schedule[0] - t_schedule[1];
        for w in t_schedule.windows(2) {
            assert!((w[0] - w[1] - dt).abs() < 1e-6, "spacing should be uniform");
        }
    }

    #[test]
    fn alternating_cfg_selection_cycles() {
        let enabled = [CfgName::Text, CfgName::Speaker, CfgName::Caption];
        let selected: Vec<&CfgName> = (0..9).map(|i| &enabled[i % enabled.len()]).collect();
        assert_eq!(
            selected,
            [
                &CfgName::Text,
                &CfgName::Speaker,
                &CfgName::Caption,
                &CfgName::Text,
                &CfgName::Speaker,
                &CfgName::Caption,
                &CfgName::Text,
                &CfgName::Speaker,
                &CfgName::Caption,
            ]
        );
    }

    #[test]
    fn alternating_single_signal_always_same() {
        let enabled = [CfgName::Text];
        let selected: Vec<&CfgName> = (0..5).map(|i| &enabled[i % enabled.len()]).collect();
        assert!(
            selected.iter().all(|n| **n == CfgName::Text),
            "single-signal alternating should always pick the same signal"
        );
    }

    #[test]
    fn use_cfg_check_respects_t_range() {
        let enabled_cfg = [CfgName::Text];
        let min_t = 0.1_f32;
        let max_t = 0.9_f32;

        // In range
        let t = 0.5;
        assert!(
            !enabled_cfg.is_empty() && min_t <= t && t <= max_t,
            "should use cfg"
        );
        // Below min_t
        let t = 0.05;
        assert!(
            !(min_t <= t && t <= max_t),
            "below min_t should not use cfg"
        );
        // Above max_t
        let t = 0.95;
        assert!(
            !(min_t <= t && t <= max_t),
            "above max_t should not use cfg"
        );
        // Empty cfg
        let empty: Vec<CfgName> = vec![];
        assert!(!(!empty.is_empty() && min_t <= 0.5 && 0.5 <= max_t));
    }

    // -----------------------------------------------------------------------
    // Integration tests: run `sample_euler_rf_cfg` with a tiny model
    // -----------------------------------------------------------------------
    use burn::backend::NdArray;

    use super::super::params::{GuidanceConfig, SamplerParams, SamplingRequest, SpeakerKvConfig};
    use crate::model::{InferenceOptimizedModel, TextToLatentRfDiT};

    type B = NdArray<f32>;

    fn tiny_model_and_request() -> (
        InferenceOptimizedModel<B>,
        SamplingRequest<B>,
        <B as Backend>::Device,
    ) {
        use crate::config::tiny_model_config;

        let device = <B as Backend>::Device::default();
        let cfg = tiny_model_config();
        let model = TextToLatentRfDiT::<B>::new(&cfg, &device);

        let (batch, seq_txt, seq_lat) = (1, 4, 6);
        let text_ids = Tensor::<B, 2, burn::tensor::Int>::zeros([batch, seq_txt], &device);
        let text_mask: Tensor<B, 2, burn::tensor::Bool> =
            Tensor::<B, 2>::ones([batch, seq_txt], &device).greater_elem(0.0);

        let speaker_dim = cfg.speaker_patched_latent_dim();
        let ref_lat = Tensor::<B, 3>::ones([batch, 3, speaker_dim], &device);
        let ref_mask: Tensor<B, 2, burn::tensor::Bool> =
            Tensor::<B, 2>::ones([batch, 3], &device).greater_elem(0.0);

        let noise = Tensor::<B, 3>::ones([batch, seq_lat, cfg.patched_latent_dim()], &device);

        let request = SamplingRequest {
            text_ids,
            text_mask,
            ref_latent: Some(ref_lat),
            ref_mask: Some(ref_mask),
            sequence_length: seq_lat,
            caption_ids: None,
            caption_mask: None,
            initial_noise: Some(noise),
        };

        (InferenceOptimizedModel::new(model), request, device)
    }

    fn tiny_caption_model_and_request() -> (
        InferenceOptimizedModel<B>,
        SamplingRequest<B>,
        <B as Backend>::Device,
    ) {
        use crate::config::tiny_caption_config;

        let device = <B as Backend>::Device::default();
        let cfg = tiny_caption_config();
        let model = TextToLatentRfDiT::<B>::new(&cfg, &device);

        let (batch, seq_txt, seq_cap, seq_lat) = (1, 4, 3, 6);
        let text_ids = Tensor::<B, 2, burn::tensor::Int>::zeros([batch, seq_txt], &device);
        let text_mask: Tensor<B, 2, burn::tensor::Bool> =
            Tensor::<B, 2>::ones([batch, seq_txt], &device).greater_elem(0.0);

        // Caption tokens (vocab indices 1..=seq_cap to avoid pad=0)
        let cap_ids = Tensor::<B, 2, burn::tensor::Int>::ones([batch, seq_cap], &device);
        let cap_mask: Tensor<B, 2, burn::tensor::Bool> =
            Tensor::<B, 2>::ones([batch, seq_cap], &device).greater_elem(0.0);

        let noise = Tensor::<B, 3>::ones([batch, seq_lat, cfg.patched_latent_dim()], &device);

        let request = SamplingRequest {
            text_ids,
            text_mask,
            ref_latent: None,
            ref_mask: None,
            sequence_length: seq_lat,
            caption_ids: Some(cap_ids),
            caption_mask: Some(cap_mask),
            initial_noise: Some(noise),
        };

        (InferenceOptimizedModel::new(model), request, device)
    }

    #[test]
    fn sampler_no_cfg_produces_finite_output() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 2,
            guidance: GuidanceConfig {
                scale_text: 0.0,
                scale_speaker: 0.0,
                scale_caption: 0.0,
                ..Default::default()
            },
            use_context_kv_cache: false,
            ..Default::default()
        };

        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        let [b, s, _d] = out.dims();
        assert_eq!(b, 1);
        assert_eq!(s, 6);
        let vals: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "output must be all finite"
        );
    }

    #[test]
    fn sampler_independent_cfg_runs_without_error() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 2,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Independent,
                scale_text: 3.0,
                scale_speaker: 5.0,
                scale_caption: 0.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: false,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        let [b, s, _d] = out.dims();
        assert_eq!(b, 1);
        assert_eq!(s, 6);
        let vals: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sampler_alternating_cfg_runs_without_error() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 4, // enough steps to cycle through text + speaker
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Alternating,
                scale_text: 2.0,
                scale_speaker: 3.0,
                scale_caption: 0.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: false,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        let [b, s, _d] = out.dims();
        assert_eq!(b, 1);
        assert_eq!(s, 6);
        let vals: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sampler_independent_cfg_cached_runs() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 2,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Independent,
                scale_text: 3.0,
                scale_speaker: 5.0,
                scale_caption: 0.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: true,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        assert!(
            out.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn sampler_speaker_kv_deactivation() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 4,
            guidance: GuidanceConfig {
                scale_text: 0.0,
                scale_speaker: 0.0,
                scale_caption: 0.0,
                ..Default::default()
            },
            speaker_kv: Some(SpeakerKvConfig {
                scale: 2.0,
                max_layers: None,
                min_t: Some(0.5), // should deactivate mid-schedule
            }),
            use_context_kv_cache: true,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        assert!(
            out.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn sampler_joint_unequal_scales_errors() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 2,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Joint,
                scale_text: 3.0,
                scale_speaker: 5.0, // unequal → should error
                scale_caption: 0.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: false,
            ..Default::default()
        };

        let result = sample_euler_rf_cfg(&model, request, &params, &device);
        assert!(
            result.is_err(),
            "Joint mode with unequal text/speaker scales must error"
        );
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("Joint"),
            "error should mention Joint mode: {msg}"
        );
    }

    #[test]
    fn sampler_cached_matches_uncached() {
        let (model, request, device) = tiny_model_and_request();

        let base_params = SamplerParams {
            num_steps: 2,
            guidance: GuidanceConfig {
                scale_text: 0.0,
                scale_speaker: 0.0,
                scale_caption: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        // Uncached
        let request_uncached = SamplingRequest {
            text_ids: request.text_ids.clone(),
            text_mask: request.text_mask.clone(),
            ref_latent: request.ref_latent.clone(),
            ref_mask: request.ref_mask.clone(),
            sequence_length: request.sequence_length,
            caption_ids: None,
            caption_mask: None,
            initial_noise: request.initial_noise.clone(),
        };
        let params_uncached = SamplerParams {
            use_context_kv_cache: false,
            ..base_params.clone()
        };
        let out_uncached =
            sample_euler_rf_cfg(&model, request_uncached, &params_uncached, &device).unwrap();

        // Cached
        let params_cached = SamplerParams {
            use_context_kv_cache: true,
            ..base_params
        };
        let out_cached = sample_euler_rf_cfg(&model, request, &params_cached, &device).unwrap();

        let diff: f32 = (out_uncached - out_cached).abs().max().into_scalar();
        assert_eq!(
            diff, 0.0,
            "cached and uncached should produce identical output on NdArray"
        );
    }

    // --- Rubber duck finding #1: Joint happy-path ---
    #[test]
    fn sampler_joint_cfg_happy_path() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 2,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Joint,
                scale_text: 3.0,
                scale_speaker: 3.0, // must equal text for Joint
                scale_caption: 0.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: true,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        let [b, s, _d] = out.dims();
        assert_eq!((b, s), (1, 6));
        assert!(
            out.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    // --- Rubber duck finding #2: Caption CFG ---
    #[test]
    fn sampler_caption_independent_cfg_runs() {
        let (model, request, device) = tiny_caption_model_and_request();
        let params = SamplerParams {
            num_steps: 2,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Independent,
                scale_text: 3.0,
                scale_speaker: 0.0,
                scale_caption: 2.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: false,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        let [b, s, _d] = out.dims();
        assert_eq!((b, s), (1, 6));
        assert!(
            out.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn sampler_caption_alternating_cfg_runs() {
        let (model, request, device) = tiny_caption_model_and_request();
        let params = SamplerParams {
            num_steps: 4,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Alternating,
                scale_text: 2.0,
                scale_speaker: 0.0,
                scale_caption: 3.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: true,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        assert!(
            out.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    // --- Rubber duck finding #3: Cached-vs-uncached under CFG ---
    #[test]
    fn sampler_independent_cfg_cached_matches_uncached() {
        let (model, request, device) = tiny_model_and_request();
        let guidance = GuidanceConfig {
            mode: CfgGuidanceMode::Independent,
            scale_text: 3.0,
            scale_speaker: 5.0,
            scale_caption: 0.0,
            min_t: 0.0,
            max_t: 1.0,
        };

        let request2 = SamplingRequest {
            text_ids: request.text_ids.clone(),
            text_mask: request.text_mask.clone(),
            ref_latent: request.ref_latent.clone(),
            ref_mask: request.ref_mask.clone(),
            sequence_length: request.sequence_length,
            caption_ids: None,
            caption_mask: None,
            initial_noise: request.initial_noise.clone(),
        };

        let out_uncached = sample_euler_rf_cfg(
            &model,
            request2,
            &SamplerParams {
                num_steps: 2,
                guidance: guidance.clone(),
                use_context_kv_cache: false,
                ..Default::default()
            },
            &device,
        )
        .unwrap();

        let out_cached = sample_euler_rf_cfg(
            &model,
            request,
            &SamplerParams {
                num_steps: 2,
                guidance,
                use_context_kv_cache: true,
                ..Default::default()
            },
            &device,
        )
        .unwrap();

        let diff: f32 = (out_uncached - out_cached).abs().max().into_scalar();
        assert_eq!(
            diff, 0.0,
            "Independent CFG cached and uncached must match on NdArray"
        );
    }

    #[test]
    fn sampler_alternating_cfg_cached_matches_uncached() {
        let (model, request, device) = tiny_model_and_request();
        let guidance = GuidanceConfig {
            mode: CfgGuidanceMode::Alternating,
            scale_text: 2.0,
            scale_speaker: 3.0,
            scale_caption: 0.0,
            min_t: 0.0,
            max_t: 1.0,
        };

        let request2 = SamplingRequest {
            text_ids: request.text_ids.clone(),
            text_mask: request.text_mask.clone(),
            ref_latent: request.ref_latent.clone(),
            ref_mask: request.ref_mask.clone(),
            sequence_length: request.sequence_length,
            caption_ids: None,
            caption_mask: None,
            initial_noise: request.initial_noise.clone(),
        };

        let out_uncached = sample_euler_rf_cfg(
            &model,
            request2,
            &SamplerParams {
                num_steps: 4,
                guidance: guidance.clone(),
                use_context_kv_cache: false,
                ..Default::default()
            },
            &device,
        )
        .unwrap();

        let out_cached = sample_euler_rf_cfg(
            &model,
            request,
            &SamplerParams {
                num_steps: 4,
                guidance,
                use_context_kv_cache: true,
                ..Default::default()
            },
            &device,
        )
        .unwrap();

        let diff: f32 = (out_uncached - out_cached).abs().max().into_scalar();
        assert_eq!(
            diff, 0.0,
            "Alternating CFG cached and uncached must match on NdArray"
        );
    }

    // --- Rubber duck finding #4: Speaker KV deactivation with both paths ---
    #[test]
    fn sampler_speaker_kv_with_and_without_min_t_both_succeed() {
        let (model, request, device) = tiny_model_and_request();

        let request2 = SamplingRequest {
            text_ids: request.text_ids.clone(),
            text_mask: request.text_mask.clone(),
            ref_latent: request.ref_latent.clone(),
            ref_mask: request.ref_mask.clone(),
            sequence_length: request.sequence_length,
            caption_ids: None,
            caption_mask: None,
            initial_noise: request.initial_noise.clone(),
        };

        // scale=2.0, min_t=None → scaling stays for all steps
        let out_always = sample_euler_rf_cfg(
            &model,
            request2,
            &SamplerParams {
                num_steps: 4,
                guidance: GuidanceConfig {
                    scale_text: 0.0,
                    scale_speaker: 0.0,
                    scale_caption: 0.0,
                    ..Default::default()
                },
                speaker_kv: Some(SpeakerKvConfig {
                    scale: 2.0,
                    max_layers: None,
                    min_t: None,
                }),
                use_context_kv_cache: true,
                ..Default::default()
            },
            &device,
        )
        .unwrap();
        assert!(
            out_always
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );

        // scale=2.0, min_t=Some(0.5) → deactivation branch fires at step 2
        let out_reverted = sample_euler_rf_cfg(
            &model,
            request,
            &SamplerParams {
                num_steps: 4,
                guidance: GuidanceConfig {
                    scale_text: 0.0,
                    scale_speaker: 0.0,
                    scale_caption: 0.0,
                    ..Default::default()
                },
                speaker_kv: Some(SpeakerKvConfig {
                    scale: 2.0,
                    max_layers: None,
                    min_t: Some(0.5),
                }),
                use_context_kv_cache: true,
                ..Default::default()
            },
            &device,
        )
        .unwrap();
        assert!(
            out_reverted
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }
}
