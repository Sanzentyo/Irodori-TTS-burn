use std::path::PathBuf;

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
    pub text_add_bos: bool,
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
    /// Always-present speaker fields — used only when `use_speaker_condition() == true`.
    pub speaker_dim: Option<usize>,
    pub speaker_layers: Option<usize>,
    pub speaker_heads: Option<usize>,
    pub speaker_patch_size: Option<usize>,
    pub timestep_embed_dim: usize,
    pub adaln_rank: usize,
    pub norm_eps: f64,
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
            text_add_bos: true,
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
            speaker_dim: Some(1280),
            speaker_layers: Some(14),
            speaker_heads: Some(10),
            speaker_patch_size: Some(1),
            timestep_embed_dim: 512,
            adaln_rank: 256,
            norm_eps: 1e-5,
            fixed_target_latent_steps: None,
        }
    }
}

impl ModelConfig {
    /// Speaker conditioning is active when caption conditioning is *not* used.
    pub fn use_speaker_condition(&self) -> bool {
        !self.use_caption_condition
    }

    /// Validate the configuration.
    ///
    /// Returns an error if any combination of fields would cause incorrect
    /// or undefined behaviour (e.g. non-divisible head dimensions, missing
    /// required conditional fields).
    pub fn validate(&self) -> crate::error::Result<()> {
        use crate::error::IrodoriError;

        // ── Main diffusion head ────────────────────────────────────────────
        if self.model_dim == 0 {
            return Err(IrodoriError::Config("model_dim must be > 0".to_string()));
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

        // ── Text encoder ──────────────────────────────────────────────────
        if self.text_dim == 0 {
            return Err(IrodoriError::Config("text_dim must be > 0".to_string()));
        }
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

        // ── Speaker encoder (active when caption is disabled) ─────────────
        if self.use_speaker_condition() {
            let spk_dim = self.speaker_dim.ok_or_else(|| {
                IrodoriError::Config(
                    "speaker_dim must be set when use_caption_condition is false".to_string(),
                )
            })?;
            let spk_heads = self.speaker_heads.ok_or_else(|| {
                IrodoriError::Config(
                    "speaker_heads must be set when use_caption_condition is false".to_string(),
                )
            })?;
            if self.speaker_layers.is_none() {
                return Err(IrodoriError::Config(
                    "speaker_layers must be set when use_caption_condition is false".to_string(),
                ));
            }
            if self.speaker_patch_size.is_none() {
                return Err(IrodoriError::Config(
                    "speaker_patch_size must be set when use_caption_condition is false"
                        .to_string(),
                ));
            }
            if spk_dim == 0 {
                return Err(IrodoriError::Config("speaker_dim must be > 0".to_string()));
            }
            if spk_heads == 0 {
                return Err(IrodoriError::Config(
                    "speaker_heads must be > 0".to_string(),
                ));
            }
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
            let cap_heads = self.caption_heads();
            if cap_dim == 0 {
                return Err(IrodoriError::Config("caption_dim must be > 0".to_string()));
            }
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

/// Hyperparameters for a single LoRA adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct LoraConfig {
    /// Low-rank dimension.
    pub r: usize,
    /// LoRA scaling factor (scale = alpha / r).
    pub alpha: f32,
    /// Attention projection names to adapt.
    ///
    /// When empty, ALL projections receive LoRA adapters.
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            r: 8,
            alpha: 16.0,
            target_modules: vec![
                "wq".to_owned(),
                "wk".to_owned(),
                "wv".to_owned(),
                "wo".to_owned(),
                "gate".to_owned(),
            ],
        }
    }
}

/// Full configuration for a LoRA fine-tuning run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraTrainConfig {
    /// Path to the JSONL training manifest.
    pub manifest_path: PathBuf,
    /// Directory where adapter checkpoints are saved.
    pub output_dir: PathBuf,
    /// Path to the base model safetensors checkpoint.
    pub base_model_path: PathBuf,
    /// Path to the Hugging Face tokenizer JSON.
    pub tokenizer_path: PathBuf,
    /// LoRA adapter hyperparameters.
    pub lora: LoraConfig,
    /// Samples per batch.
    pub batch_size: usize,
    /// Peak learning rate.
    pub lr: f64,
    /// AdamW weight decay.
    pub weight_decay: f64,
    /// Linear warm-up steps.
    pub warmup_steps: usize,
    /// Total training steps.
    pub max_steps: usize,
    /// Log every N steps.
    pub log_every: usize,
    /// Save adapter checkpoint every N steps.
    pub save_every: usize,
    /// Logit-normal timestep distribution mean.
    pub t_mean: f32,
    /// Logit-normal timestep distribution std.
    pub t_std: f32,

    // ── New in v2 ────────────────────────────────────────────────────────────
    /// Optional validation manifest JSONL.  When set, validation is run every
    /// `val_every` steps and the mean loss is logged.
    pub val_manifest: Option<PathBuf>,
    /// Run validation every N steps (0 = disabled).
    pub val_every: usize,
    /// Number of validation batches per eval (0 = full validation set).
    pub val_batches: usize,
    /// Shuffle training samples at the start of each epoch.
    pub shuffle: bool,
    /// Seed for the shuffle RNG (deterministic across runs with same seed).
    pub shuffle_seed: u64,
    /// Accumulate gradients over this many micro-batches before an optimiser
    /// step.  Effective batch size = `batch_size × grad_accum_steps`.
    pub grad_accum_steps: usize,
    /// Resume training from an existing checkpoint directory
    /// (`output_dir/step-NNNNNNN/`).  Only adapter weights are restored;
    /// optimizer state resets (warm restart).
    pub resume_from: Option<PathBuf>,

    // ── New in v3: regularisation & sampling ─────────────────────────────────
    /// Per-sample probability of dropping text conditioning (zeroing text mask)
    /// during training. Enables classifier-free guidance at inference time.
    /// Must be in `[0, 1]`. Default 0.1 (matching Python reference).
    pub text_condition_dropout: f64,
    /// Per-sample probability of dropping speaker conditioning (zeroing speaker
    /// mask and latent) during training. Must be in `[0, 1]`. Default 0.1.
    pub speaker_condition_dropout: f64,
    /// Global gradient norm clipping threshold. Gradients are scaled so their
    /// combined L2 norm does not exceed this value. `None` = disabled.
    /// Default `Some(1.0)` (matching Python reference).
    pub grad_clip_norm: Option<f64>,
    /// Use stratified logit-normal timestep sampling for variance reduction.
    /// Default `true` (matching Python reference).
    pub timestep_stratified: bool,
}

impl Default for LoraTrainConfig {
    fn default() -> Self {
        Self {
            manifest_path: PathBuf::from("train.jsonl"),
            output_dir: PathBuf::from("output"),
            base_model_path: PathBuf::from("model.safetensors"),
            tokenizer_path: PathBuf::from("tokenizer.json"),
            lora: LoraConfig::default(),
            batch_size: 4,
            lr: 1e-4,
            weight_decay: 0.01,
            warmup_steps: 100,
            max_steps: 5000,
            log_every: 10,
            save_every: 500,
            t_mean: 0.0,
            t_std: 1.0,
            val_manifest: None,
            val_every: 500,
            val_batches: 50,
            shuffle: true,
            shuffle_seed: 42,
            grad_accum_steps: 1,
            resume_from: None,
            text_condition_dropout: 0.1,
            speaker_condition_dropout: 0.1,
            grad_clip_norm: Some(1.0),
            timestep_stratified: true,
        }
    }
}

impl LoraTrainConfig {
    /// Validate training configuration, catching common mistakes early.
    pub fn validate(&self) -> crate::error::Result<()> {
        use crate::error::IrodoriError;

        if self.batch_size == 0 {
            return Err(IrodoriError::Config("batch_size must be > 0".to_string()));
        }
        if self.max_steps == 0 {
            return Err(IrodoriError::Config("max_steps must be > 0".to_string()));
        }
        if self.grad_accum_steps == 0 {
            return Err(IrodoriError::Config(
                "grad_accum_steps must be > 0".to_string(),
            ));
        }
        if self.log_every == 0 {
            return Err(IrodoriError::Config("log_every must be > 0".to_string()));
        }
        if self.save_every == 0 {
            return Err(IrodoriError::Config("save_every must be > 0".to_string()));
        }
        if self.lr < 0.0 {
            return Err(IrodoriError::Config("lr must be >= 0".to_string()));
        }
        if self.weight_decay < 0.0 {
            return Err(IrodoriError::Config(
                "weight_decay must be >= 0".to_string(),
            ));
        }
        if self.lora.r == 0 {
            return Err(IrodoriError::Config("lora.r must be > 0".to_string()));
        }
        if self.warmup_steps >= self.max_steps {
            return Err(IrodoriError::Config(format!(
                "warmup_steps ({}) must be < max_steps ({})",
                self.warmup_steps, self.max_steps,
            )));
        }
        if !(0.0..=1.0).contains(&self.text_condition_dropout) {
            return Err(IrodoriError::Config(format!(
                "text_condition_dropout must be in [0, 1], got {}",
                self.text_condition_dropout,
            )));
        }
        if !(0.0..=1.0).contains(&self.speaker_condition_dropout) {
            return Err(IrodoriError::Config(format!(
                "speaker_condition_dropout must be in [0, 1], got {}",
                self.speaker_condition_dropout,
            )));
        }
        if let Some(clip) = self.grad_clip_norm
            && clip <= 0.0
        {
            return Err(IrodoriError::Config(format!(
                "grad_clip_norm must be > 0, got {clip}",
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SamplingConfig {
    pub num_steps: usize,
    pub cfg_scale_text: f64,
    pub cfg_scale_caption: f64,
    pub cfg_scale_speaker: f64,
    pub cfg_guidance_mode: CfgGuidanceMode,
    /// Deprecated: single-scale override (sets all scales).
    pub cfg_scale: Option<f64>,
    pub cfg_min_t: f64,
    pub cfg_max_t: f64,
    pub truncation_factor: Option<f64>,
    pub rescale_k: Option<f64>,
    pub rescale_sigma: Option<f64>,
    pub context_kv_cache: bool,
    pub speaker_kv_scale: Option<f64>,
    pub speaker_kv_min_t: Option<f64>,
    pub speaker_kv_max_layers: Option<usize>,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            num_steps: 40,
            cfg_scale_text: 3.0,
            cfg_scale_caption: 3.0,
            cfg_scale_speaker: 5.0,
            cfg_guidance_mode: CfgGuidanceMode::Independent,
            cfg_scale: None,
            cfg_min_t: 0.5,
            cfg_max_t: 1.0,
            truncation_factor: None,
            rescale_k: None,
            rescale_sigma: None,
            context_kv_cache: true,
            speaker_kv_scale: None,
            speaker_kv_min_t: Some(0.9),
            speaker_kv_max_layers: None,
            seed: 0,
        }
    }
}

/// CFG guidance application strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CfgGuidanceMode {
    /// Separate forward pass per dropped condition (most common).
    #[default]
    Independent,
    /// Single fully-unconditional forward pass.
    Joint,
    /// Alternate which condition is dropped each step.
    Alternating,
}

impl std::fmt::Display for CfgGuidanceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Independent => write!(f, "independent"),
            Self::Joint => write!(f, "joint"),
            Self::Alternating => write!(f, "alternating"),
        }
    }
}

impl std::str::FromStr for CfgGuidanceMode {
    type Err = crate::error::IrodoriError;

    fn from_str(s: &str) -> crate::error::Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "independent" => Ok(Self::Independent),
            "joint" => Ok(Self::Joint),
            "alternating" => Ok(Self::Alternating),
            other => Err(crate::error::IrodoriError::UnsupportedMode(format!(
                "unknown cfg_guidance_mode: {other:?}"
            ))),
        }
    }
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

    // ── LoraTrainConfig validation ──────────────────────────────────────────

    #[test]
    fn valid_train_config_passes() {
        assert!(LoraTrainConfig::default().validate().is_ok());
    }

    #[test]
    fn train_config_zero_batch_size_fails() {
        let cfg = LoraTrainConfig {
            batch_size: 0,
            ..LoraTrainConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn train_config_zero_max_steps_fails() {
        let cfg = LoraTrainConfig {
            max_steps: 0,
            ..LoraTrainConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn train_config_zero_grad_accum_fails() {
        let cfg = LoraTrainConfig {
            grad_accum_steps: 0,
            ..LoraTrainConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn train_config_zero_lora_rank_fails() {
        let cfg = LoraTrainConfig {
            lora: LoraConfig {
                r: 0,
                ..LoraConfig::default()
            },
            ..LoraTrainConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn train_config_warmup_ge_max_steps_fails() {
        let cfg = LoraTrainConfig {
            warmup_steps: 5000,
            max_steps: 5000,
            ..LoraTrainConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn train_config_negative_lr_fails() {
        let cfg = LoraTrainConfig {
            lr: -1.0,
            ..LoraTrainConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn train_config_text_dropout_out_of_range_fails() {
        let cfg = LoraTrainConfig {
            text_condition_dropout: 1.5,
            ..LoraTrainConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn train_config_speaker_dropout_out_of_range_fails() {
        let cfg = LoraTrainConfig {
            speaker_condition_dropout: -0.1,
            ..LoraTrainConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn train_config_grad_clip_zero_fails() {
        let cfg = LoraTrainConfig {
            grad_clip_norm: Some(0.0),
            ..LoraTrainConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn train_config_grad_clip_none_passes() {
        let cfg = LoraTrainConfig {
            grad_clip_norm: None,
            ..LoraTrainConfig::default()
        };
        assert!(cfg.validate().is_ok());
    }
}
