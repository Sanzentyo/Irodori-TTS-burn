//! Diffusion model architecture configuration.

use serde::{Deserialize, Serialize};

/// Configuration for the diffusion model architecture.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelConfig {
    pub latent_dim: usize,
    pub latent_patch_size: usize,
    pub model_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub mlp_ratio: f64,
    pub text_mlp_ratio: Option<f64>,
    pub speaker_mlp_ratio: Option<f64>,
    pub dropout: f64,
    pub text_vocab_size: usize,
    pub text_tokenizer_repo: String,
    pub text_encoder_revision: Option<String>,
    pub text_add_bos: bool,
    pub text_encoder_type: String,
    pub pretrained_projector_type: String,
    pub pretrained_projector_hidden_ratio: f64,
    pub pretrained_projector_dropout: f64,
    pub text_dim: usize,
    pub text_layers: usize,
    pub text_heads: usize,
    pub use_caption_condition: bool,
    pub caption_vocab_size: Option<usize>,
    pub caption_tokenizer_repo: Option<String>,
    pub caption_add_bos: Option<bool>,
    pub caption_dim: Option<usize>,
    pub caption_layers: Option<usize>,
    pub caption_heads: Option<usize>,
    pub caption_mlp_ratio: Option<f64>,
    /// Explicit speaker-conditioning flag used by v4 checkpoints.
    ///
    /// Older checkpoints omit this field. In that case speaker conditioning
    /// retains the legacy behavior of being enabled when caption conditioning
    /// is disabled.
    pub use_speaker_condition: Option<bool>,
    /// Speaker fields used only when [`Self::use_speaker_condition`] resolves to true.
    pub speaker_dim: Option<usize>,
    pub speaker_layers: Option<usize>,
    pub speaker_heads: Option<usize>,
    pub speaker_patch_size: Option<usize>,
    pub timestep_embed_dim: usize,
    pub adaln_rank: usize,
    pub norm_eps: f64,
    pub use_duration_predictor: bool,
    pub duration_aux_dim: usize,
    pub duration_hidden_dim: usize,
    pub duration_layers: usize,
    pub duration_dropout: f64,
    pub duration_attention_heads: usize,
    pub duration_architecture: String,
    pub duration_token_init_frames: f64,
    pub duration_speaker_fusion: String,
    pub duration_caption_fusion: String,
    pub duration_caption_pooling: String,
    /// Runtime limits embedded in released inference checkpoints.
    pub max_text_len: Option<usize>,
    pub max_caption_len: Option<usize>,
    pub ref_max_seconds: Option<f64>,
    /// Fixed output length in latent frames (from checkpoint metadata).
    ///
    /// When present, `just infer` uses this as the default `--seq-len` value.
    pub fixed_target_latent_steps: Option<usize>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            latent_dim: 128,
            latent_patch_size: 1,
            model_dim: 2048,
            num_layers: 24,
            num_heads: 16,
            mlp_ratio: 2.875,
            text_mlp_ratio: Some(2.6),
            speaker_mlp_ratio: Some(2.6),
            dropout: 0.0,
            text_vocab_size: 102400,
            text_tokenizer_repo: "sbintuitions/sarashina2.2-0.5b".to_string(),
            text_encoder_revision: None,
            text_add_bos: true,
            text_encoder_type: "scratch".to_string(),
            pretrained_projector_type: "linear".to_string(),
            pretrained_projector_hidden_ratio: 2.0,
            pretrained_projector_dropout: 0.0,
            text_dim: 1280,
            text_layers: 14,
            text_heads: 10,
            use_caption_condition: false,
            caption_vocab_size: None,
            caption_tokenizer_repo: None,
            caption_add_bos: None,
            caption_dim: None,
            caption_layers: None,
            caption_heads: None,
            caption_mlp_ratio: None,
            use_speaker_condition: None,
            speaker_dim: Some(1280),
            speaker_layers: Some(14),
            speaker_heads: Some(10),
            speaker_patch_size: Some(1),
            timestep_embed_dim: 512,
            adaln_rank: 256,
            norm_eps: 1e-5,
            use_duration_predictor: false,
            duration_aux_dim: 14,
            duration_hidden_dim: 1024,
            duration_layers: 3,
            duration_dropout: 0.1,
            duration_attention_heads: 8,
            duration_architecture: "token_sum_adarn_zero_no_aux".to_string(),
            duration_token_init_frames: 9.0,
            duration_speaker_fusion: "adarn_zero".to_string(),
            duration_caption_fusion: "adarn_zero".to_string(),
            duration_caption_pooling: "masked_mean".to_string(),
            max_text_len: None,
            max_caption_len: None,
            ref_max_seconds: None,
            fixed_target_latent_steps: None,
        }
    }
}

impl ModelConfig {
    fn validate_finite_positive(field: &str, value: f64) -> crate::error::Result<()> {
        if value.is_finite() && value > 0.0 {
            Ok(())
        } else {
            Err(crate::error::IrodoriError::Config(format!(
                "{field} must be finite and > 0"
            )))
        }
    }

    fn validate_dropout(field: &str, value: f64) -> crate::error::Result<()> {
        if value.is_finite() && (0.0..1.0).contains(&value) {
            Ok(())
        } else {
            Err(crate::error::IrodoriError::Config(format!(
                "{field} must be finite and in [0, 1)"
            )))
        }
    }

    /// Resolve whether speaker conditioning is active.
    ///
    /// v4 checkpoints store an explicit value and may enable speaker and
    /// caption conditioning together. Older checkpoints omit it and keep the
    /// legacy mutually-exclusive behavior.
    pub fn use_speaker_condition(&self) -> bool {
        self.use_speaker_condition
            .unwrap_or(!self.use_caption_condition)
    }

    /// Whether the checkpoint uses the shared pretrained text backbone.
    pub fn use_pretrained_text_encoder(&self) -> bool {
        self.text_encoder_type
            .trim()
            .eq_ignore_ascii_case("pretrained")
    }

    /// Validate the configuration.
    ///
    /// Returns an error if any combination of fields would cause incorrect
    /// or undefined behaviour (e.g. non-divisible head dimensions, missing
    /// required conditional fields).
    pub fn validate(&self) -> crate::error::Result<()> {
        use crate::error::IrodoriError;

        // ── Main diffusion head ────────────────────────────────────────────
        if self.latent_dim == 0 {
            return Err(IrodoriError::Config("latent_dim must be > 0".to_string()));
        }
        if self.model_dim == 0 {
            return Err(IrodoriError::Config("model_dim must be > 0".to_string()));
        }
        if self.num_layers == 0 {
            return Err(IrodoriError::Config("num_layers must be > 0".to_string()));
        }
        if self.num_heads == 0 {
            return Err(IrodoriError::Config("num_heads must be > 0".to_string()));
        }
        if !self.model_dim.is_multiple_of(self.num_heads) {
            return Err(IrodoriError::Config(format!(
                "model_dim ({}) must be divisible by num_heads ({})",
                self.model_dim, self.num_heads
            )));
        }
        let hd = self.head_dim();
        if !hd.is_multiple_of(2) {
            return Err(IrodoriError::Config(format!(
                "head_dim ({hd}) must be even for RoPE"
            )));
        }

        // ── Misc positivity ────────────────────────────────────────────────
        if self.latent_patch_size == 0 {
            return Err(IrodoriError::Config(
                "latent_patch_size must be > 0".to_string(),
            ));
        }
        if self.timestep_embed_dim == 0 {
            return Err(IrodoriError::Config(
                "timestep_embed_dim must be > 0".to_string(),
            ));
        }
        if self.adaln_rank == 0 {
            return Err(IrodoriError::Config("adaln_rank must be > 0".to_string()));
        }
        Self::validate_finite_positive("mlp_ratio", self.mlp_ratio)?;
        Self::validate_finite_positive("norm_eps", self.norm_eps)?;
        Self::validate_dropout("dropout", self.dropout)?;

        // ── Text encoder ──────────────────────────────────────────────────
        if self.text_dim == 0 {
            return Err(IrodoriError::Config("text_dim must be > 0".to_string()));
        }
        if self.use_pretrained_text_encoder() {
            Self::validate_finite_positive(
                "pretrained_projector_hidden_ratio",
                self.pretrained_projector_hidden_ratio,
            )?;
            Self::validate_dropout(
                "pretrained_projector_dropout",
                self.pretrained_projector_dropout,
            )?;
        } else {
            if self.text_layers == 0 {
                return Err(IrodoriError::Config("text_layers must be > 0".to_string()));
            }
            Self::validate_finite_positive("text_mlp_ratio", self.text_mlp_ratio())?;
            if self.text_heads == 0 {
                return Err(IrodoriError::Config("text_heads must be > 0".to_string()));
            }
            if !self.text_dim.is_multiple_of(self.text_heads) {
                return Err(IrodoriError::Config(format!(
                    "text_dim ({}) must be divisible by text_heads ({})",
                    self.text_dim, self.text_heads
                )));
            }
            let text_hd = self.text_dim / self.text_heads;
            if !text_hd.is_multiple_of(2) {
                return Err(IrodoriError::Config(format!(
                    "text head_dim ({text_hd}) must be even for RoPE"
                )));
            }
        }

        // ── Speaker encoder ───────────────────────────────────────────────
        if self.use_speaker_condition() {
            let spk_dim = self.speaker_dim.ok_or_else(|| {
                IrodoriError::Config(
                    "speaker_dim must be set when speaker conditioning is enabled".to_string(),
                )
            })?;
            let spk_heads = self.speaker_heads.ok_or_else(|| {
                IrodoriError::Config(
                    "speaker_heads must be set when speaker conditioning is enabled".to_string(),
                )
            })?;
            let spk_layers = self.speaker_layers.ok_or_else(|| {
                IrodoriError::Config(
                    "speaker_layers must be set when speaker conditioning is enabled".to_string(),
                )
            })?;
            let spk_patch_size = self.speaker_patch_size.ok_or_else(|| {
                IrodoriError::Config(
                    "speaker_patch_size must be set when speaker conditioning is enabled"
                        .to_string(),
                )
            })?;
            if spk_dim == 0 {
                return Err(IrodoriError::Config("speaker_dim must be > 0".to_string()));
            }
            if spk_layers == 0 {
                return Err(IrodoriError::Config(
                    "speaker_layers must be > 0".to_string(),
                ));
            }
            if spk_patch_size == 0 {
                return Err(IrodoriError::Config(
                    "speaker_patch_size must be > 0".to_string(),
                ));
            }
            if spk_heads == 0 {
                return Err(IrodoriError::Config(
                    "speaker_heads must be > 0".to_string(),
                ));
            }
            Self::validate_finite_positive("speaker_mlp_ratio", self.speaker_mlp_ratio())?;
            if !spk_dim.is_multiple_of(spk_heads) {
                return Err(IrodoriError::Config(format!(
                    "speaker_dim ({spk_dim}) must be divisible by speaker_heads ({spk_heads})"
                )));
            }
            let spk_hd = spk_dim / spk_heads;
            if !spk_hd.is_multiple_of(2) {
                return Err(IrodoriError::Config(format!(
                    "speaker head_dim ({spk_hd}) must be even for RoPE"
                )));
            }
        }

        // ── Caption encoder ────────────────────────────────────────────────
        if self.use_caption_condition {
            let cap_dim = self.caption_dim();
            if cap_dim == 0 {
                return Err(IrodoriError::Config("caption_dim must be > 0".to_string()));
            }
            if !self.use_pretrained_text_encoder() {
                if self.caption_layers() == 0 {
                    return Err(IrodoriError::Config(
                        "caption_layers must be > 0".to_string(),
                    ));
                }
                Self::validate_finite_positive("caption_mlp_ratio", self.caption_mlp_ratio())?;
                let cap_heads = self.caption_heads();
                if cap_heads == 0 {
                    return Err(IrodoriError::Config(
                        "caption_heads must be > 0".to_string(),
                    ));
                }
                if !cap_dim.is_multiple_of(cap_heads) {
                    return Err(IrodoriError::Config(format!(
                        "caption_dim ({cap_dim}) must be divisible by caption_heads ({cap_heads})"
                    )));
                }
                let cap_hd = cap_dim / cap_heads;
                if !cap_hd.is_multiple_of(2) {
                    return Err(IrodoriError::Config(format!(
                        "caption head_dim ({cap_hd}) must be even for RoPE"
                    )));
                }
            }
        }

        if self.use_duration_predictor {
            if self.duration_aux_dim == 0
                || self.duration_hidden_dim == 0
                || self.duration_layers == 0
                || self.duration_attention_heads == 0
            {
                return Err(IrodoriError::Config(
                    "duration predictor dimensions and layer counts must be > 0".to_string(),
                ));
            }
            Self::validate_dropout("duration_dropout", self.duration_dropout)?;
            Self::validate_finite_positive(
                "duration_token_init_frames",
                self.duration_token_init_frames,
            )?;
        }

        if self
            .ref_max_seconds
            .is_some_and(|v| v <= 0.0 || !v.is_finite())
        {
            return Err(IrodoriError::Config(
                "ref_max_seconds must be finite and > 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Dimension of each attention head.
    pub fn head_dim(&self) -> usize {
        self.model_dim / self.num_heads
    }

    /// Latent dimension after patching: `latent_dim * latent_patch_size`.
    pub fn patched_latent_dim(&self) -> usize {
        self.latent_dim * self.latent_patch_size
    }

    /// Speaker latent dimension after full patching (latent + speaker patches combined).
    pub fn speaker_patched_latent_dim(&self) -> usize {
        self.patched_latent_dim() * self.speaker_patch_size.unwrap_or(1)
    }

    pub fn text_mlp_ratio(&self) -> f64 {
        self.text_mlp_ratio.unwrap_or(self.mlp_ratio)
    }

    pub fn speaker_mlp_ratio(&self) -> f64 {
        self.speaker_mlp_ratio.unwrap_or(self.mlp_ratio)
    }

    pub fn caption_vocab_size(&self) -> usize {
        self.caption_vocab_size.unwrap_or(self.text_vocab_size)
    }

    pub fn caption_tokenizer_repo(&self) -> &str {
        self.caption_tokenizer_repo
            .as_deref()
            .unwrap_or(&self.text_tokenizer_repo)
    }

    pub fn caption_add_bos(&self) -> bool {
        self.caption_add_bos.unwrap_or(self.text_add_bos)
    }

    pub fn caption_dim(&self) -> usize {
        self.caption_dim.unwrap_or(self.text_dim)
    }

    pub fn caption_layers(&self) -> usize {
        self.caption_layers.unwrap_or(self.text_layers)
    }

    pub fn caption_heads(&self) -> usize {
        self.caption_heads.unwrap_or(self.text_heads)
    }

    pub fn caption_mlp_ratio(&self) -> f64 {
        self.caption_mlp_ratio
            .unwrap_or_else(|| self.text_mlp_ratio())
    }
}

/// Tiny model configuration for unit tests.
///
/// Produces a valid `ModelConfig` with minimal dimensions to keep test
/// memory and runtime negligible while exercising the full model structure.
#[cfg(test)]
pub(crate) fn tiny_model_config() -> ModelConfig {
    let cfg = ModelConfig {
        latent_dim: 8,
        latent_patch_size: 1,
        model_dim: 32,
        num_heads: 4,
        num_layers: 1,
        mlp_ratio: 2.0,
        text_mlp_ratio: Some(2.0),
        speaker_mlp_ratio: Some(2.0),
        dropout: 0.0,
        text_dim: 16,
        text_heads: 2,
        text_layers: 1,
        text_vocab_size: 64,
        speaker_dim: Some(16),
        speaker_layers: Some(1),
        speaker_heads: Some(2),
        speaker_patch_size: Some(1),
        timestep_embed_dim: 32,
        adaln_rank: 16,
        norm_eps: 1e-5,
        use_caption_condition: false,
        ..Default::default()
    };
    cfg.validate().expect("tiny_model_config must be valid");
    cfg
}

/// Tiny model config with caption conditioning instead of speaker.
#[cfg(test)]
pub(crate) fn tiny_caption_config() -> ModelConfig {
    let mut cfg = tiny_model_config();
    cfg.use_caption_condition = true;
    cfg.caption_vocab_size = Some(32);
    cfg.caption_dim = Some(16);
    cfg.caption_layers = Some(1);
    cfg.caption_heads = Some(2);
    cfg.caption_mlp_ratio = Some(2.0);
    cfg.speaker_dim = None;
    cfg.speaker_layers = None;
    cfg.speaker_heads = None;
    cfg.speaker_patch_size = None;
    cfg.validate().expect("tiny_caption_config must be valid");
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_speaker_config() -> ModelConfig {
        ModelConfig {
            model_dim: 16,
            num_heads: 2,
            latent_dim: 4,
            latent_patch_size: 1,
            num_layers: 1,
            text_dim: 8,
            text_heads: 2,
            text_layers: 1,
            text_vocab_size: 32,
            timestep_embed_dim: 16,
            adaln_rank: 4,
            norm_eps: 1e-5,
            speaker_dim: Some(8),
            speaker_heads: Some(2),
            speaker_layers: Some(1),
            speaker_patch_size: Some(1),
            ..Default::default()
        }
    }

    #[test]
    fn valid_speaker_config_passes() {
        assert!(valid_speaker_config().validate().is_ok());
    }

    #[test]
    fn zero_num_heads_fails() {
        let mut cfg = valid_speaker_config();
        cfg.num_heads = 0;
        assert!(cfg.validate().is_err(), "num_heads=0 must fail");
    }

    #[test]
    fn zero_latent_dim_fails() {
        let mut cfg = valid_speaker_config();
        cfg.latent_dim = 0;
        assert!(cfg.validate().is_err(), "latent_dim=0 must fail");
    }

    #[test]
    fn zero_num_layers_fails() {
        let mut cfg = valid_speaker_config();
        cfg.num_layers = 0;
        assert!(cfg.validate().is_err(), "num_layers=0 must fail");
    }

    #[test]
    fn non_divisible_model_dim_fails() {
        let mut cfg = valid_speaker_config();
        cfg.model_dim = 15;
        cfg.num_heads = 4; // 15 / 4 is not divisible
        assert!(cfg.validate().is_err(), "15/4 head_dim must fail");
    }

    #[test]
    fn odd_head_dim_fails_rope() {
        let mut cfg = valid_speaker_config();
        // model_dim=18, num_heads=2 → head_dim=9 (odd, invalid for RoPE)
        cfg.model_dim = 18;
        cfg.num_heads = 2;
        assert!(cfg.validate().is_err(), "odd head_dim must fail for RoPE");
    }

    #[test]
    fn missing_speaker_dim_in_speaker_mode_fails() {
        let mut cfg = valid_speaker_config();
        cfg.speaker_dim = None;
        assert!(cfg.validate().is_err(), "missing speaker_dim must fail");
    }

    #[test]
    fn missing_speaker_heads_in_speaker_mode_fails() {
        let mut cfg = valid_speaker_config();
        cfg.speaker_heads = None;
        assert!(cfg.validate().is_err(), "missing speaker_heads must fail");
    }

    #[test]
    fn missing_speaker_layers_in_speaker_mode_fails() {
        let mut cfg = valid_speaker_config();
        cfg.speaker_layers = None;
        assert!(cfg.validate().is_err(), "missing speaker_layers must fail");
    }

    #[test]
    fn zero_speaker_layers_in_speaker_mode_fails() {
        let mut cfg = valid_speaker_config();
        cfg.speaker_layers = Some(0);
        assert!(cfg.validate().is_err(), "speaker_layers=0 must fail");
    }

    #[test]
    fn zero_speaker_patch_size_in_speaker_mode_fails() {
        let mut cfg = valid_speaker_config();
        cfg.speaker_patch_size = Some(0);
        assert!(cfg.validate().is_err(), "speaker_patch_size=0 must fail");
    }

    #[test]
    fn zero_adaln_rank_fails() {
        let mut cfg = valid_speaker_config();
        cfg.adaln_rank = 0;
        assert!(cfg.validate().is_err(), "adaln_rank=0 must fail");
    }

    #[test]
    fn zero_latent_patch_size_fails() {
        let mut cfg = valid_speaker_config();
        cfg.latent_patch_size = 0;
        assert!(cfg.validate().is_err(), "latent_patch_size=0 must fail");
    }

    #[test]
    fn zero_timestep_embed_dim_fails() {
        let mut cfg = valid_speaker_config();
        cfg.timestep_embed_dim = 0;
        assert!(cfg.validate().is_err(), "timestep_embed_dim=0 must fail");
    }

    #[test]
    fn non_finite_or_non_positive_core_floats_fail() {
        type ConfigMutation = fn(&mut ModelConfig);
        let cases: [(&str, ConfigMutation); 6] = [
            ("mlp_ratio=0", |cfg| cfg.mlp_ratio = 0.0),
            ("mlp_ratio=NaN", |cfg| cfg.mlp_ratio = f64::NAN),
            ("norm_eps=0", |cfg| cfg.norm_eps = 0.0),
            ("norm_eps=inf", |cfg| cfg.norm_eps = f64::INFINITY),
            ("text_mlp_ratio=0", |cfg| cfg.text_mlp_ratio = Some(0.0)),
            ("speaker_mlp_ratio=NaN", |cfg| {
                cfg.speaker_mlp_ratio = Some(f64::NAN)
            }),
        ];

        for (name, mutate) in cases {
            let mut cfg = valid_speaker_config();
            mutate(&mut cfg);
            assert!(cfg.validate().is_err(), "{name} must fail");
        }
    }

    #[test]
    fn invalid_main_dropout_values_fail() {
        for dropout in [-0.1, 1.0, f64::NAN, f64::INFINITY] {
            let mut cfg = valid_speaker_config();
            cfg.dropout = dropout;
            assert!(cfg.validate().is_err(), "dropout={dropout} must fail");
        }
    }

    #[test]
    fn invalid_relevant_conditional_floats_fail() {
        let mut pretrained = valid_speaker_config();
        pretrained.text_encoder_type = "pretrained".to_string();
        pretrained.pretrained_projector_hidden_ratio = f64::NAN;
        assert!(pretrained.validate().is_err());

        let mut pretrained_dropout = valid_speaker_config();
        pretrained_dropout.text_encoder_type = "pretrained".to_string();
        pretrained_dropout.pretrained_projector_dropout = f64::INFINITY;
        assert!(pretrained_dropout.validate().is_err());

        let mut duration = valid_speaker_config();
        duration.use_duration_predictor = true;
        duration.duration_dropout = f64::NAN;
        assert!(duration.validate().is_err());

        let mut caption = valid_speaker_config();
        caption.use_speaker_condition = Some(false);
        caption.use_caption_condition = true;
        caption.caption_mlp_ratio = Some(0.0);
        assert!(caption.validate().is_err());
    }

    #[test]
    fn legacy_json_resolves_speaker_from_caption_flag() {
        let mut value = serde_json::to_value(ModelConfig::default()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("use_speaker_condition");

        let speaker: ModelConfig = serde_json::from_value(value.clone()).unwrap();
        assert!(speaker.use_speaker_condition());
        assert!(speaker.validate().is_ok());

        value["use_caption_condition"] = serde_json::Value::Bool(true);
        let caption: ModelConfig = serde_json::from_value(value).unwrap();
        assert!(!caption.use_speaker_condition());
        assert!(caption.validate().is_ok());
    }

    #[test]
    fn explicit_speaker_and_caption_flags_can_coexist() {
        let mut cfg = valid_speaker_config();
        cfg.use_speaker_condition = Some(true);
        cfg.use_caption_condition = true;
        cfg.caption_dim = Some(8);
        cfg.caption_heads = Some(2);
        cfg.caption_layers = Some(1);

        assert!(cfg.validate().is_ok());
        assert!(cfg.use_speaker_condition());
        assert!(cfg.use_caption_condition);
    }

    #[test]
    fn pretrained_v4_dimensions_skip_unused_scratch_head_validation() {
        let mut cfg = valid_speaker_config();
        cfg.text_encoder_type = " PRETRAINED ".to_string();
        cfg.text_dim = 512;
        cfg.text_heads = 10;
        cfg.use_speaker_condition = Some(true);
        cfg.use_caption_condition = true;
        cfg.caption_dim = Some(512);
        cfg.caption_heads = Some(10);
        cfg.caption_layers = Some(1);

        assert!(cfg.use_pretrained_text_encoder());
        assert!(cfg.validate().is_ok());
    }
}
