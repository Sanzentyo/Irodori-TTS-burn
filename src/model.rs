pub(crate) mod adaln_cross_layer;
pub(crate) mod attention;
pub(crate) mod condition;
pub(crate) mod diffusion;
pub(crate) mod dit;
pub(crate) mod duration;
pub(crate) mod feed_forward;
mod linear_ops;
pub mod modern_bert;
pub mod norm;
pub(crate) mod optimized;
pub(crate) mod rope;
pub(crate) mod speaker_encoder;
pub(crate) mod text_encoder;
pub(crate) mod timestep_condition;
pub(crate) mod wgsl;

// Re-export the primary types for convenient use
pub use attention::CondKvCache;
pub use condition::{AuxConditionInput, AuxConditionState, EncodedCondition};
pub use dit::{
    AuxConditioner, BlockDebugOutputs, BothConditioner, CaptionConditioner, SpeakerConditioner,
    TextToLatentRfDiT,
};
pub use duration::{
    DurationPredictor, DurationPredictorConfig, DurationPredictorInput, V4_DURATION_ARCHITECTURE,
};
pub use optimized::{InferenceOptimizedModel, WgslInferenceOptimizedModel};
pub use speaker_encoder::unpatchify_latent;
