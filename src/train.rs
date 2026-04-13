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
