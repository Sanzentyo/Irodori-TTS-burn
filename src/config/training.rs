//! LoRA fine-tuning configuration.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

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
#[serde(default)]
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
    /// Optional path to a separate tokenizer for caption text. When `None`,
    /// falls back to `tokenizer_path` (i.e. captions share the text tokenizer).
    pub caption_tokenizer_path: Option<PathBuf>,
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
    /// Per-sample probability of dropping caption conditioning (zeroing caption
    /// mask post-encoding) during training. Only used when
    /// `ModelConfig::use_caption_condition` is true. Must be in `[0, 1]`.
    /// Default 0.1 (matching Python reference).
    pub caption_condition_dropout: f64,
    /// Global gradient norm clipping threshold. Gradients are scaled so their
    /// combined L2 norm does not exceed this value. `None` or `0.0` = disabled.
    /// Default `Some(1.0)` (matching Python reference).
    #[serde(
        default = "default_grad_clip_norm",
        deserialize_with = "deserialize_grad_clip_norm"
    )]
    pub grad_clip_norm: Option<f64>,
    /// Use stratified logit-normal timestep sampling for variance reduction.
    /// Default `true` (matching Python reference).
    pub timestep_stratified: bool,
    /// Seed for the training RNG.
    ///
    /// Controls:
    /// - Backend tensor RNG (noise sampling, LoRA weight initialization)
    /// - Host-side `StdRng` (timestep sampling, condition dropout)
    ///
    /// Given the same seed, data order, backend, and hardware, training is
    /// deterministic.  Note: resume from checkpoint resets the RNG state
    /// (warm restart), so the resumed sequence may differ from an
    /// uninterrupted run.  Default `42`.
    pub training_seed: u64,
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
            caption_condition_dropout: 0.1,
            caption_tokenizer_path: None,
            grad_clip_norm: Some(1.0),
            timestep_stratified: true,
            training_seed: 42,
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
        if !(0.0..=1.0).contains(&self.caption_condition_dropout) {
            return Err(IrodoriError::Config(format!(
                "caption_condition_dropout must be in [0, 1], got {}",
                self.caption_condition_dropout,
            )));
        }
        if let Some(clip) = self.grad_clip_norm
            && clip <= 0.0
        {
            return Err(IrodoriError::Config(format!(
                "grad_clip_norm must be > 0, got {clip}",
            )));
        }
        if self.t_std <= 0.0 || !self.t_std.is_finite() {
            return Err(IrodoriError::Config(format!(
                "t_std must be finite and > 0, got {}",
                self.t_std,
            )));
        }
        if !self.t_mean.is_finite() {
            return Err(IrodoriError::Config(format!(
                "t_mean must be finite, got {}",
                self.t_mean,
            )));
        }
        Ok(())
    }
}

// ── Serde helpers for grad_clip_norm (Option<f64> from TOML) ──────────────────

fn default_grad_clip_norm() -> Option<f64> {
    Some(1.0)
}

/// Deserialize `grad_clip_norm`: treat `0.0` (or absent) as `None` (disabled).
fn deserialize_grad_clip_norm<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val: Option<f64> = Option::deserialize(deserializer)?;
    Ok(val.filter(|&v| v != 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn train_config_caption_dropout_out_of_range_fails() {
        let cfg = LoraTrainConfig {
            caption_condition_dropout: 1.5,
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

    #[test]
    fn train_config_grad_clip_zero_toml_becomes_none() {
        let toml_str = r#"
            grad_clip_norm = 0.0
        "#;
        let cfg: LoraTrainConfig = toml::from_str(toml_str).unwrap();
        assert!(
            cfg.grad_clip_norm.is_none(),
            "0.0 should deserialize to None (disabled)"
        );
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn train_config_zero_t_std_fails() {
        let cfg = LoraTrainConfig {
            t_std: 0.0,
            ..LoraTrainConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn train_config_negative_t_std_fails() {
        let cfg = LoraTrainConfig {
            t_std: -1.0,
            ..LoraTrainConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn train_config_nan_t_mean_fails() {
        let cfg = LoraTrainConfig {
            t_mean: f32::NAN,
            ..LoraTrainConfig::default()
        };
        assert!(cfg.validate().is_err());
    }
}
