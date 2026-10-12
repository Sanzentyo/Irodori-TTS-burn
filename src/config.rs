//! Model, training, and sampling configuration.

mod model;
pub use model::ModelConfig;
#[cfg(test)]
pub(crate) use model::{tiny_caption_config, tiny_model_config};

mod sampling;
pub use sampling::{CfgGuidanceMode, SamplerMethod, SamplingConfig};
