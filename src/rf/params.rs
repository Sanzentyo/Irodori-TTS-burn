//! Sampler hyperparameter types for the RF Euler sampler.

use burn::tensor::{Bool, Int, Tensor};

use crate::{
    config::{CfgGuidanceMode, SamplerMethod},
    error::IrodoriError,
};

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

/// Parameters for Euler / Heun RF sampling with CFG.
#[derive(Debug, Clone)]
pub struct SamplerParams {
    /// Number of denoising steps.
    pub num_steps: usize,
    /// ODE integration method (`Euler` or `Heun`).  Default: `Euler`.
    ///
    /// With `Heun`, use `num_steps / 2` steps for the same NFE budget as Euler.
    pub method: SamplerMethod,
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
            method: SamplerMethod::Euler,
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
            if !trc.k.is_finite() || trc.k <= 0.0 {
                return Err(IrodoriError::Config(
                    "temporal_rescale.k must be finite and > 0".to_string(),
                ));
            }
            if !trc.sigma.is_finite() || trc.sigma <= 0.0 {
                return Err(IrodoriError::Config(
                    "temporal_rescale.sigma must be finite and > 0".to_string(),
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
        // PLMS-4 requires a consistent ODE RHS across history steps; Alternating CFG changes
        // the dropped condition each step, producing mismatched velocity fields in the history.
        if matches!(self.method, crate::config::SamplerMethod::PLMS4)
            && matches!(self.guidance.mode, CfgGuidanceMode::Alternating)
        {
            return Err(IrodoriError::Config(
                "sampler PLMS4 is not compatible with cfg_guidance_mode=Alternating; \
                 use Independent or Joint instead"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

impl TryFrom<crate::config::SamplingConfig> for SamplerParams {
    type Error = IrodoriError;

    fn try_from(cfg: crate::config::SamplingConfig) -> Result<Self, Self::Error> {
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

        let temporal_rescale = match (cfg.rescale_k, cfg.rescale_sigma) {
            (Some(k), Some(sigma)) => Some(TemporalRescaleConfig {
                k: k as f32,
                sigma: sigma as f32,
            }),
            (None, None) => None,
            _ => {
                return Err(IrodoriError::Config(
                    "rescale_k and rescale_sigma must be supplied together".to_string(),
                ));
            }
        };

        let params = Self {
            num_steps: cfg.num_steps,
            method: cfg.sampler_method,
            guidance: GuidanceConfig {
                mode: cfg.cfg_guidance_mode,
                scale_text,
                scale_caption,
                scale_speaker,
                min_t: cfg.cfg_min_t as f32,
                max_t: cfg.cfg_max_t as f32,
            },
            truncation_factor: cfg.truncation_factor.map(|v| v as f32),
            temporal_rescale,
            speaker_kv: cfg.speaker_kv_scale.map(|scale| SpeakerKvConfig {
                scale: scale as f32,
                min_t: cfg.speaker_kv_min_t.map(|v| v as f32),
                max_layers: cfg.speaker_kv_max_layers,
            }),
            use_context_kv_cache: cfg.context_kv_cache,
        };
        params.validate()?;
        Ok(params)
    }
}

/// All per-call inputs to [`sample_euler_rf_cfg`](super::sample_euler_rf_cfg).
///
/// Groups the per-request tensors that change between calls so they don't
/// pollute the function signature.
#[derive(Debug, Clone)]
pub struct SamplingRequest {
    pub text_ids: Tensor<2, Int>,
    pub text_mask: Tensor<2, Bool>,
    /// Optional reference audio latent `[1, T, D]`.
    pub ref_latent: Option<Tensor<3>>,
    pub ref_mask: Option<Tensor<2, Bool>>,
    /// Number of output latent frames to generate.
    pub sequence_length: usize,
    /// Optional caption token ids for caption conditioning.
    pub caption_ids: Option<Tensor<2, Int>>,
    pub caption_mask: Option<Tensor<2, Bool>>,
    /// Pre-generated initial noise for reproducibility; `None` = sample fresh.
    pub initial_noise: Option<Tensor<3>>,
}

/// Sampling inputs after all data-dependent host preparation has completed.
///
/// This is the only request form accepted by compile-only warmup. In
/// particular, masked suffix compaction and all-masked auxiliary removal run
/// before a CubeCL [`DryRun`](cubecl::dry_run::DryRun) guard is opened, so the
/// guarded graph is driven exclusively by shapes and host metadata.
#[derive(Debug, Clone)]
pub struct PreparedSamplingRequest {
    pub(crate) request: SamplingRequest,
    pub(crate) requested_text_tokens: usize,
    pub(crate) requested_speaker_tokens: Option<usize>,
    pub(crate) requested_caption_tokens: Option<usize>,
    pub(crate) conditioned_text_mask_all_valid: bool,
    pub(crate) has_speaker_context: bool,
    pub(crate) has_caption_context: bool,
}

impl PreparedSamplingRequest {
    pub fn sequence_length(&self) -> usize {
        self.request.sequence_length
    }

    pub fn has_speaker_context(&self) -> bool {
        self.has_speaker_context
    }

    pub fn has_caption_context(&self) -> bool {
        self.has_caption_context
    }
}

impl SamplingRequest {
    /// Validate tensor compatibility before any condition encoder or sampler
    /// allocation runs.
    pub fn validate(&self, expected_latent_dim: usize) -> crate::error::Result<()> {
        use crate::error::IrodoriError;

        let [batch, text_sequence] = self.text_ids.dims();
        if batch == 0 || text_sequence == 0 {
            return Err(IrodoriError::Shape(format!(
                "text_input_ids must have non-zero batch and sequence dimensions, got [{batch}, {text_sequence}]"
            )));
        }
        let text_mask_shape = self.text_mask.dims();
        if text_mask_shape != [batch, text_sequence] {
            return Err(IrodoriError::Shape(format!(
                "text_mask shape {text_mask_shape:?} must match text_input_ids shape [{batch}, {text_sequence}]"
            )));
        }
        if self.sequence_length == 0 {
            return Err(IrodoriError::Shape(
                "sequence_length must be greater than zero".to_string(),
            ));
        }

        match (&self.ref_latent, &self.ref_mask) {
            (Some(ref_latent), Some(ref_mask)) => {
                let [ref_batch, ref_sequence, ref_dim] = ref_latent.dims();
                let ref_mask_shape = ref_mask.dims();
                if ref_batch != batch
                    || ref_sequence == 0
                    || ref_dim != expected_latent_dim
                    || ref_mask_shape != [batch, ref_sequence]
                {
                    return Err(IrodoriError::Shape(format!(
                        "reference inputs must have ref_latent=[{batch}, T>0, {expected_latent_dim}] and ref_mask=[{batch}, T] with the same T; got ref_latent=[{ref_batch}, {ref_sequence}, {ref_dim}], ref_mask={ref_mask_shape:?}"
                    )));
                }
            }
            (Some(_), None) => {
                return Err(IrodoriError::MissingInput(
                    "ref_mask must be supplied together with ref_latent".to_string(),
                ));
            }
            (None, Some(_)) => {
                return Err(IrodoriError::MissingInput(
                    "ref_latent must be supplied together with ref_mask".to_string(),
                ));
            }
            (None, None) => {}
        }

        match (&self.caption_ids, &self.caption_mask) {
            (Some(caption_ids), Some(caption_mask)) => {
                let [caption_batch, caption_sequence] = caption_ids.dims();
                let caption_mask_shape = caption_mask.dims();
                if caption_batch != batch
                    || caption_sequence == 0
                    || caption_mask_shape != [batch, caption_sequence]
                {
                    return Err(IrodoriError::Shape(format!(
                        "caption inputs must have caption_ids=[{batch}, T>0] and caption_mask=[{batch}, T] with the same T; got caption_ids=[{caption_batch}, {caption_sequence}], caption_mask={caption_mask_shape:?}"
                    )));
                }
            }
            (Some(_), None) => {
                return Err(IrodoriError::MissingInput(
                    "caption_mask must be supplied together with caption_ids".to_string(),
                ));
            }
            (None, Some(_)) => {
                return Err(IrodoriError::MissingInput(
                    "caption_ids must be supplied together with caption_mask".to_string(),
                ));
            }
            (None, None) => {}
        }

        if let Some(initial_noise) = &self.initial_noise {
            let shape = initial_noise.dims();
            let expected = [batch, self.sequence_length, expected_latent_dim];
            if shape != expected {
                return Err(IrodoriError::Shape(format!(
                    "initial_noise shape {shape:?} must be exactly {expected:?}"
                )));
            }
        }

        Ok(())
    }

    /// Complete every CPU-readback-dependent preparation step.
    pub fn prepare(
        self,
        expected_latent_dim: usize,
    ) -> crate::error::Result<PreparedSamplingRequest> {
        self.validate(expected_latent_dim)?;
        let requested_text_tokens = self.text_ids.dims()[1];
        let requested_speaker_tokens = self.ref_latent.as_ref().map(|state| state.dims()[1]);
        let requested_caption_tokens = self.caption_ids.as_ref().map(|ids| ids.dims()[1]);
        let (request, conditioned_text_mask_all_valid) = self.compact_conditioning()?;
        Ok(PreparedSamplingRequest {
            has_speaker_context: request.ref_latent.is_some(),
            has_caption_context: request.caption_ids.is_some(),
            request,
            requested_text_tokens,
            requested_speaker_tokens,
            requested_caption_tokens,
            conditioned_text_mask_all_valid,
        })
    }

    /// Remove conditioning work that is provably masked out.
    ///
    /// Token sequences are right-trimmed to the last column that is valid in
    /// any batch row.  Completely masked caption and reference pairs are
    /// removed, allowing the condition frontend and joint attention to skip
    /// them entirely.  At least one text column is retained because tensor
    /// backends do not uniformly support zero-length sequence dimensions.
    ///
    /// This transformation preserves every valid token and its original
    /// position; only trailing columns that no batch row can attend to are
    /// discarded.  Call [`Self::validate`] first so tensor pairs and batch
    /// dimensions are known to agree. The returned boolean proves that every
    /// retained text-mask element is valid, using the same host readback.
    pub(crate) fn compact_conditioning(mut self) -> crate::error::Result<(Self, bool)> {
        let original_text_sequence = self.text_ids.dims()[1];
        let text_extent = mask_extent("text mask", &self.text_mask)?;
        self.text_ids = narrow_token_sequence(self.text_ids, text_extent.used_columns);
        self.text_mask = narrow_mask_sequence(self.text_mask, text_extent.used_columns);

        let original_caption_sequence = self.caption_ids.as_ref().map(|ids| ids.dims()[1]);
        let (caption_ids, caption_mask) = match (self.caption_ids, self.caption_mask) {
            (Some(ids), Some(mask)) => {
                let extent = mask_extent("caption mask", &mask)?;
                if extent.any_valid {
                    (
                        Some(narrow_token_sequence(ids, extent.used_columns)),
                        Some(narrow_mask_sequence(mask, extent.used_columns)),
                    )
                } else {
                    (None, None)
                }
            }
            (None, None) => (None, None),
            _ => {
                return Err(IrodoriError::MissingInput(
                    "caption_ids and caption_mask must be supplied together".to_string(),
                ));
            }
        };
        self.caption_ids = caption_ids;
        self.caption_mask = caption_mask;

        let had_reference = self.ref_latent.is_some();
        let (ref_latent, ref_mask) = match (self.ref_latent, self.ref_mask) {
            (Some(latent), Some(mask)) => {
                if mask_extent("reference mask", &mask)?.any_valid {
                    (Some(latent), Some(mask))
                } else {
                    (None, None)
                }
            }
            (None, None) => (None, None),
            _ => {
                return Err(IrodoriError::MissingInput(
                    "ref_latent and ref_mask must be supplied together".to_string(),
                ));
            }
        };
        self.ref_latent = ref_latent;
        self.ref_mask = ref_mask;

        tracing::debug!(
            original_text_sequence,
            compacted_text_sequence = text_extent.used_columns,
            original_caption_sequence,
            compacted_caption_sequence = self.caption_ids.as_ref().map(|ids| ids.dims()[1]),
            reference_removed = had_reference && self.ref_latent.is_none(),
            "compacted masked conditioning inputs"
        );

        // This fact is derived from the same mandatory host readback used to
        // compact the mask. The exact WGPU text-CFG cache path consumes it to
        // prove that omitting the conditioned attention mask is sound.
        Ok((self, text_extent.all_used_valid))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MaskExtent {
    any_valid: bool,
    used_columns: usize,
    all_used_valid: bool,
}

fn mask_extent(name: &str, mask: &Tensor<2, Bool>) -> crate::error::Result<MaskExtent> {
    let [batch, sequence] = mask.dims();
    let values = mask
        .clone()
        .into_data()
        // WGPU stores booleans as u32.  Normalize before converting to the
        // host representation so this optimization stays backend-independent.
        .convert::<bool>()
        .to_vec::<bool>()
        .map_err(|error| IrodoriError::Dtype(name.to_string(), error.to_string()))?;

    let last_valid =
        (0..sequence).rfind(|&column| (0..batch).any(|row| values[row * sequence + column]));
    let used_columns = last_valid.map_or(1, |column| column + 1);
    let all_used_valid = (0..batch).all(|row| {
        values[row * sequence..row * sequence + used_columns]
            .iter()
            .all(|&value| value)
    });
    Ok(MaskExtent {
        any_valid: last_valid.is_some(),
        used_columns,
        all_used_valid,
    })
}

fn narrow_token_sequence(tokens: Tensor<2, Int>, used_columns: usize) -> Tensor<2, Int> {
    let sequence = tokens.dims()[1];
    if used_columns < sequence {
        tokens.narrow(1, 0, used_columns)
    } else {
        tokens
    }
}

fn narrow_mask_sequence(mask: Tensor<2, Bool>, used_columns: usize) -> Tensor<2, Bool> {
    let sequence = mask.dims()[1];
    if used_columns < sequence {
        mask.narrow(1, 0, used_columns)
    } else {
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn valid_sampling_request() -> SamplingRequest {
        let device = Default::default();
        SamplingRequest {
            text_ids: Tensor::zeros([2, 4], &device),
            text_mask: Tensor::<2>::ones([2, 4], &device).greater_elem(0.0),
            ref_latent: Some(Tensor::zeros([2, 3, 8], &device)),
            ref_mask: Some(Tensor::<2>::ones([2, 3], &device).greater_elem(0.0)),
            sequence_length: 6,
            caption_ids: Some(Tensor::zeros([2, 5], &device)),
            caption_mask: Some(Tensor::<2>::ones([2, 5], &device).greater_elem(0.0)),
            initial_noise: Some(Tensor::zeros([2, 6, 8], &device)),
        }
    }

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
    fn validate_temporal_rescale_requires_positive_finite_pair() {
        for (k, sigma) in [
            (0.0, 1.0),
            (-1.0, 1.0),
            (f32::INFINITY, 1.0),
            (1.0, 0.0),
            (1.0, -1.0),
            (1.0, f32::NAN),
            (1.0, f32::INFINITY),
        ] {
            let params = SamplerParams {
                temporal_rescale: Some(TemporalRescaleConfig { k, sigma }),
                ..Default::default()
            };
            assert!(
                params.validate().is_err(),
                "k={k}, sigma={sigma} must be rejected"
            );
        }
    }

    #[test]
    fn sampling_config_conversion_requires_complete_rescale_pair() {
        let only_k = crate::config::SamplingConfig {
            rescale_k: Some(2.0),
            ..Default::default()
        };
        assert!(matches!(
            SamplerParams::try_from(only_k),
            Err(IrodoriError::Config(_))
        ));

        let only_sigma = crate::config::SamplingConfig {
            rescale_sigma: Some(1.0),
            ..Default::default()
        };
        assert!(matches!(
            SamplerParams::try_from(only_sigma),
            Err(IrodoriError::Config(_))
        ));

        let complete = crate::config::SamplingConfig {
            rescale_k: Some(2.0),
            rescale_sigma: Some(1.0),
            ..Default::default()
        };
        assert!(SamplerParams::try_from(complete).is_ok());
    }

    #[test]
    fn sampling_request_valid_shapes_pass() {
        assert!(valid_sampling_request().validate(8).is_ok());
    }

    #[test]
    fn compact_conditioning_trims_text_and_removes_masked_aux_inputs() {
        let device = Default::default();
        let mut request = valid_sampling_request();
        request.text_mask = Tensor::<2, Bool>::from_data(
            [[true, true, false, false], [true, false, true, false]],
            &device,
        );
        request.caption_mask = Some(Tensor::<2, Bool>::from_data(
            [[false; 5], [false; 5]],
            &device,
        ));
        request.ref_mask = Some(Tensor::<2, Bool>::from_data(
            [[false; 3], [false; 3]],
            &device,
        ));

        let (compacted, text_mask_all_valid) = request.compact_conditioning().unwrap();
        assert!(!text_mask_all_valid);
        assert_eq!(compacted.text_ids.dims(), [2, 3]);
        assert_eq!(compacted.text_mask.dims(), [2, 3]);
        assert!(compacted.caption_ids.is_none());
        assert!(compacted.caption_mask.is_none());
        assert!(compacted.ref_latent.is_none());
        assert!(compacted.ref_mask.is_none());
    }

    #[test]
    fn compact_conditioning_keeps_active_aux_and_trims_caption_suffix() {
        let device = Default::default();
        let mut request = valid_sampling_request();
        request.caption_mask = Some(Tensor::<2, Bool>::from_data(
            [
                [true, false, false, false, false],
                [true, true, false, true, false],
            ],
            &device,
        ));
        request.ref_mask = Some(Tensor::<2, Bool>::from_data(
            [[true, false, false], [false, false, false]],
            &device,
        ));

        let (compacted, text_mask_all_valid) = request.compact_conditioning().unwrap();
        assert!(text_mask_all_valid);
        assert_eq!(compacted.caption_ids.unwrap().dims(), [2, 4]);
        assert_eq!(compacted.caption_mask.unwrap().dims(), [2, 4]);
        assert_eq!(compacted.ref_latent.unwrap().dims(), [2, 3, 8]);
        assert_eq!(compacted.ref_mask.unwrap().dims(), [2, 3]);
    }

    #[test]
    fn compact_conditioning_retains_one_all_masked_text_column() {
        let device = Default::default();
        let mut request = valid_sampling_request();
        request.text_mask = Tensor::<2, Bool>::from_data([[false; 4], [false; 4]], &device);

        let (compacted, text_mask_all_valid) = request.compact_conditioning().unwrap();
        assert!(!text_mask_all_valid);
        assert_eq!(compacted.text_ids.dims(), [2, 1]);
        assert_eq!(compacted.text_mask.dims(), [2, 1]);
    }

    #[test]
    fn sampling_request_rejects_unpaired_optional_inputs() {
        let mut missing_ref_mask = valid_sampling_request();
        missing_ref_mask.ref_mask = None;
        assert!(matches!(
            missing_ref_mask.validate(8),
            Err(crate::error::IrodoriError::MissingInput(_))
        ));

        let mut missing_ref_latent = valid_sampling_request();
        missing_ref_latent.ref_latent = None;
        assert!(matches!(
            missing_ref_latent.validate(8),
            Err(crate::error::IrodoriError::MissingInput(_))
        ));

        let mut missing_caption_ids = valid_sampling_request();
        missing_caption_ids.caption_ids = None;
        assert!(matches!(
            missing_caption_ids.validate(8),
            Err(crate::error::IrodoriError::MissingInput(_))
        ));

        let mut missing_caption_mask = valid_sampling_request();
        missing_caption_mask.caption_mask = None;
        assert!(matches!(
            missing_caption_mask.validate(8),
            Err(crate::error::IrodoriError::MissingInput(_))
        ));
    }

    #[test]
    fn sampling_request_rejects_text_mask_shape_mismatch() {
        let device = Default::default();
        let mut request = valid_sampling_request();
        request.text_mask = Tensor::<2>::ones([2, 3], &device).greater_elem(0.0);
        assert!(matches!(
            request.validate(8),
            Err(crate::error::IrodoriError::Shape(_))
        ));
    }

    #[test]
    fn sampling_request_rejects_reference_shape_mismatch() {
        let device = Default::default();

        let mut wrong_batch = valid_sampling_request();
        wrong_batch.ref_latent = Some(Tensor::zeros([1, 3, 8], &device));
        assert!(wrong_batch.validate(8).is_err());

        let mut wrong_sequence = valid_sampling_request();
        wrong_sequence.ref_mask = Some(Tensor::<2>::ones([2, 2], &device).greater_elem(0.0));
        assert!(wrong_sequence.validate(8).is_err());

        let mut wrong_dim = valid_sampling_request();
        wrong_dim.ref_latent = Some(Tensor::zeros([2, 3, 7], &device));
        assert!(wrong_dim.validate(8).is_err());
    }

    #[test]
    fn sampling_request_rejects_caption_shape_mismatch() {
        let device = Default::default();
        let mut request = valid_sampling_request();
        request.caption_mask = Some(Tensor::<2>::ones([2, 4], &device).greater_elem(0.0));
        assert!(matches!(
            request.validate(8),
            Err(crate::error::IrodoriError::Shape(_))
        ));
    }

    #[test]
    fn sampling_request_rejects_initial_noise_shape_mismatch() {
        let device = Default::default();
        for shape in [[1, 6, 8], [2, 5, 8], [2, 6, 7]] {
            let mut request = valid_sampling_request();
            request.initial_noise = Some(Tensor::zeros(shape, &device));
            assert!(
                request.validate(8).is_err(),
                "initial noise shape {shape:?} must be rejected"
            );
        }
    }

    #[test]
    fn validate_default_passes() {
        assert!(SamplerParams::default().validate().is_ok());
    }
}
