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
    /// or undefined behaviour (e.g. non-divisible head dimensions).
    pub fn validate(&self) -> crate::error::Result<()> {
        use crate::error::IrodoriError;

        if self.num_heads == 0 {
            return Err(IrodoriError::Config(
                "num_heads must be greater than 0".to_string(),
            ));
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

/// Inference/sampling configuration.
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
