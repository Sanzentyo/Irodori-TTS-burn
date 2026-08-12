pub mod backend_config;
#[cfg(feature = "codec")]
pub mod codec;
pub mod config;
pub mod error;
#[cfg(feature = "inference")]
pub mod inference;
pub mod kernels;
#[cfg(feature = "lora")]
pub mod lora;
pub mod model;
#[cfg(all(feature = "inference", feature = "codec"))]
pub mod phase_batch;
pub(crate) mod profiling;
pub mod rf;
#[cfg(feature = "text-normalization")]
pub mod text_normalization;
pub mod validation;
pub mod weights;

pub use backend_config::{InferenceBackendKind, WgpuRaw};
#[cfg(feature = "codec")]
pub use codec::{load_codec, load_decoder};
pub use config::{CfgGuidanceMode, ModelConfig, SamplerMethod, SamplingConfig};
pub use error::{IrodoriError, Result};
#[cfg(feature = "inference")]
pub use inference::{InferenceBuilder, InferenceEngine, WgslInferenceEngine, WgslWeightProfile};
pub use model::{
    AuxConditionInput, AuxConditionState, BlockDebugOutputs, BothConditioner, CondKvCache,
    EncodedCondition, InferenceOptimizedModel, TextToLatentRfDiT, WgslInferenceOptimizedModel,
    unpatchify_latent,
};
#[cfg(all(feature = "inference", feature = "codec"))]
pub use phase_batch::{
    BatchAudio, BatchItemId, CodecItemTiming, Complete as PhaseBatchComplete, LatentsResident,
    OutputGeometry, PhaseBatch, PhaseBatchMetrics, PlannedSynthesis, RfItemTiming, RfResident,
    SpeakerKey, VoiceIdentity,
};
pub use rf::{
    ConditioningGeometry, ConditioningSignal, ContextKvWorkReport,
    FixedTimestepConditionWorkReport, GuidanceConfig, SamplerForwardEvaluation, SamplerForwardLane,
    SamplerForwardWork, SamplerParams, SamplerWorkReport, SamplingRequest, SpeakerKvConfig,
    TemporalRescaleConfig, sample_euler_rf_cfg, sample_euler_rf_cfg_wgsl,
};
#[cfg(feature = "text-normalization")]
pub use text_normalization::normalize_text;
pub use weights::{load_model, load_model_exact_only};
