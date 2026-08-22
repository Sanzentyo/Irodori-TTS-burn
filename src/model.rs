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

/// Read the CubeCL allocator state without making production builds depend on
/// profiling concerns. The caller synchronizes the device before sampling.
#[cfg(feature = "profile")]
pub(crate) fn wgpu_memory_usage<const D: usize>(
    reference: &burn::tensor::Tensor<D>,
) -> cubecl::MemoryUsage {
    let primitive = reference
        .clone()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("RF live-range profiling requires the WGPU production backend");
    primitive
        .client
        .memory_usage()
        .unwrap_or_else(|error| panic!("RF live-range memory query failed: {error}"))
}

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
#[cfg(all(feature = "inference", feature = "codec"))]
pub use optimized::{LayoutsSelected, PreparedModel, ProfileLocked};
pub use speaker_encoder::unpatchify_latent;
