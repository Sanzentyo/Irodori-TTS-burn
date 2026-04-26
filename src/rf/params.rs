//! Sampler hyperparameter types for the RF Euler sampler.

use burn::tensor::{Bool, Int, Tensor, backend::Backend};

use crate::config::CfgGuidanceMode;

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

        // Guidance scale finiteness
        for (name, scale) in [
            ("guidance.scale_text", self.guidance.scale_text),
            ("guidance.scale_caption", self.guidance.scale_caption),
            ("guidance.scale_speaker", self.guidance.scale_speaker),
        ] {
            if !scale.is_finite() {
                return Err(IrodoriError::Config(format!(
                    "{name} must be finite, got {scale}"
                )));
            }
        }

        // Guidance timestep window
        let (min_t, max_t) = (self.guidance.min_t, self.guidance.max_t);
        if !min_t.is_finite() || !(0.0..=1.0).contains(&min_t) {
            return Err(IrodoriError::Config(format!(
                "guidance.min_t must be finite and in [0, 1], got {min_t}"
            )));
        }
        if !max_t.is_finite() || !(0.0..=1.0).contains(&max_t) {
            return Err(IrodoriError::Config(format!(
                "guidance.max_t must be finite and in [0, 1], got {max_t}"
            )));
        }
        if min_t > max_t {
            return Err(IrodoriError::Config(format!(
                "guidance.min_t ({min_t}) must not exceed guidance.max_t ({max_t})"
            )));
        }

        if self
            .truncation_factor
            .is_some_and(|k| k <= 0.0 || !k.is_finite())
        {
            return Err(IrodoriError::Config(
                "truncation_factor must be finite and > 0".to_string(),
            ));
        }
        if let Some(trc) = self.temporal_rescale {
            if !trc.k.is_finite() {
                return Err(IrodoriError::Config(
                    "temporal_rescale.k must be finite".to_string(),
                ));
            }
            if trc.sigma == 0.0 {
                return Err(IrodoriError::Config(
                    "temporal_rescale.sigma must not be zero".to_string(),
                ));
            }
        }
        if let Some(ref skv) = self.speaker_kv {
            if !skv.scale.is_finite() || skv.scale <= 0.0 {
                return Err(IrodoriError::Config(
                    "speaker_kv.scale must be finite and > 0".to_string(),
                ));
            }
            if let Some(min_t) = skv.min_t
                && (!min_t.is_finite() || !(0.0..=1.0).contains(&min_t))
            {
                return Err(IrodoriError::Config(
                    "speaker_kv.min_t must be finite and in [0, 1]".to_string(),
                ));
            }
        }
        Ok(())
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

/// All per-call inputs to [`sample_euler_rf_cfg`](super::sample_euler_rf_cfg).
///
/// Groups the per-request tensors that change between calls so they don't
/// pollute the function signature.
#[derive(Debug, Clone)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_zero_steps_fails() {
        let p = SamplerParams {
            num_steps: 0,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn validate_speaker_kv_zero_scale_fails() {
        let p = SamplerParams {
            speaker_kv: Some(SpeakerKvConfig {
                scale: 0.0,
                max_layers: None,
                min_t: None,
            }),
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn validate_speaker_kv_negative_scale_fails() {
        let p = SamplerParams {
            speaker_kv: Some(SpeakerKvConfig {
                scale: -1.0,
                max_layers: None,
                min_t: None,
            }),
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn validate_speaker_kv_inf_scale_fails() {
        let p = SamplerParams {
            speaker_kv: Some(SpeakerKvConfig {
                scale: f32::INFINITY,
                max_layers: None,
                min_t: None,
            }),
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn validate_speaker_kv_out_of_range_min_t_fails() {
        let p = SamplerParams {
            speaker_kv: Some(SpeakerKvConfig {
                scale: 2.0,
                max_layers: None,
                min_t: Some(1.5),
            }),
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn validate_speaker_kv_valid_passes() {
        let p = SamplerParams {
            speaker_kv: Some(SpeakerKvConfig {
                scale: 2.0,
                max_layers: Some(6),
                min_t: Some(0.5),
            }),
            ..Default::default()
        };
        assert!(p.validate().is_ok());
    }

    #[test]
    fn validate_guidance_nan_scale_fails() {
        let mut p = SamplerParams::default();
        p.guidance.scale_text = f32::NAN;
        assert!(p.validate().is_err());
    }

    #[test]
    fn validate_guidance_inf_scale_fails() {
        let mut p = SamplerParams::default();
        p.guidance.scale_speaker = f32::INFINITY;
        assert!(p.validate().is_err());
    }

    #[test]
    fn validate_guidance_min_t_out_of_range_fails() {
        let mut p = SamplerParams::default();
        p.guidance.min_t = -0.1;
        assert!(p.validate().is_err());
    }

    #[test]
    fn validate_guidance_max_t_out_of_range_fails() {
        let mut p = SamplerParams::default();
        p.guidance.max_t = 1.1;
        assert!(p.validate().is_err());
    }

    #[test]
    fn validate_guidance_min_t_gt_max_t_fails() {
        let mut p = SamplerParams::default();
        p.guidance.min_t = 0.8;
        p.guidance.max_t = 0.2;
        assert!(p.validate().is_err());
    }

    #[test]
    fn validate_temporal_rescale_nan_k_fails() {
        let p = SamplerParams {
            temporal_rescale: Some(crate::rf::params::TemporalRescaleConfig {
                k: f32::NAN,
                sigma: 1.0,
            }),
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn validate_default_passes() {
        assert!(SamplerParams::default().validate().is_ok());
    }
}
