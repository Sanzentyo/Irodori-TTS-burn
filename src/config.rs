//! Model, training, and sampling configuration.

mod model;
pub use model::ModelConfig;
#[cfg(test)]
pub(crate) use model::{tiny_caption_config, tiny_model_config};

#[cfg(any(feature = "lora", feature = "train"))]
mod training;
#[cfg(any(feature = "lora", feature = "train"))]
pub use training::{LoraConfig, LoraTrainConfig};

mod sampling;
pub use sampling::{CfgGuidanceMode, SamplerMethod, SamplingConfig};
