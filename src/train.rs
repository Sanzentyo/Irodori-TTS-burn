//! LoRA fine-tuning infrastructure.
//!
//! Provides low-rank adapter layers, a training model, dataset utilities,
//! the RF flow-matching loss, an LR schedule, and a training loop.

pub mod checkpoint;
pub(crate) mod dataset;
pub(crate) mod lora_layer;
pub(crate) mod lora_model;
pub(crate) mod lora_weights;
pub(crate) mod loss;
pub(crate) mod lr_schedule;
pub(crate) mod trainer;

pub use crate::config::{LoraConfig, LoraTrainConfig};
pub use lora_layer::{LoraLinear, LoraLinearConfig};
pub use lora_model::{LoraDiffusionBlock, LoraJointAttention, LoraTextToLatentRfDiT};
pub use trainer::train_lora;
