//! LoRA fine-tuning infrastructure.
//!
//! Provides low-rank adapter layers, a training model, dataset utilities,
//! the RF flow-matching loss, an LR schedule, and a training loop.

pub mod checkpoint;
pub mod dataset;
pub mod lora_layer;
pub mod lora_model;
pub mod lora_weights;
pub mod loss;
pub mod lr_schedule;
pub mod trainer;

pub use crate::config::{LoraConfig, LoraTrainConfig};
pub use lora_layer::{LoraLinear, LoraLinearConfig};
pub use lora_model::{LoraDiffusionBlock, LoraJointAttention, LoraTextToLatentRfDiT};
pub use trainer::train_lora;

/// Tiny model configuration for unit tests.
///
/// Produces a valid `ModelConfig` with minimal dimensions to keep test
/// memory and runtime negligible while exercising the full model structure.
#[cfg(test)]
pub(crate) fn tiny_model_config() -> crate::config::ModelConfig {
    let cfg = crate::config::ModelConfig {
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
pub(crate) fn tiny_caption_config() -> crate::config::ModelConfig {
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
