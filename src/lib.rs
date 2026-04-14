pub mod backend_config;
#[cfg(feature = "codec")]
pub mod codec;
pub mod config;
pub mod error;
#[cfg(feature = "inference")]
pub mod inference;
#[cfg(feature = "lora")]
pub mod lora;
pub mod model;
pub mod profiling;
pub mod rf;
#[cfg(feature = "text-normalization")]
pub mod text_normalization;
#[cfg(feature = "train")]
pub mod train;
pub mod weights;

pub use backend_config::{BackendConfig, InferenceBackendKind, TrainingBackendKind};
pub use config::{CfgGuidanceMode, ModelConfig, SamplingConfig};
#[cfg(any(feature = "lora", feature = "train"))]
pub use config::{LoraConfig, LoraTrainConfig};
pub use error::{IrodoriError, Result};
#[cfg(feature = "inference")]
pub use inference::{InferenceBuilder, InferenceEngine};
pub use model::{
    BlockDebugOutputs, CondKvCache, EncodedCondition, TextToLatentRfDiT, unpatchify_latent,
};
pub use rf::{
    GuidanceConfig, SamplerParams, SamplingRequest, SpeakerKvConfig, TemporalRescaleConfig,
    sample_euler_rf_cfg,
};
