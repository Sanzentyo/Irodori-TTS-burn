pub(crate) mod algorithm;
#[cfg(feature = "profile")]
pub(crate) mod autotune;
pub(crate) mod bottleneck;
pub(crate) mod decoder;
pub(crate) mod encoder;
mod graph;
pub(crate) mod layers;
pub(crate) mod model;
#[cfg(feature = "profile")]
pub(crate) mod profiling;
mod streaming;
pub(crate) mod weights;

pub use algorithm::CodecK7Algorithm;
#[cfg(feature = "profile")]
pub use algorithm::{
    C768CrossBlockRows, CodecAlgorithmPlan, CodecConvTransposeSnakeFusion, CodecCrossBlockFusion,
    CodecPointwiseAlgorithm, CodecResidualStateLayout, CodecStemAlgorithm, K7SelectorChoice,
    K7SelectorManifest, K7SelectorProblem,
};
#[cfg(feature = "profile")]
pub use algorithm::{K7WeightRepackReceipt, PreparedK7WeightPolicy};
#[cfg(feature = "profile")]
pub use autotune::{
    ApprovedK7SelectorManifestSet, K7SelectorAccuracyGate, K7SelectorCaseReceipt,
    K7SelectorPerformanceReceipt, K7SelectorSelection, K7SelectorShapeApproval,
    K7SelectorVerificationReceipt,
};
pub use graph::{CapturedCodecDecode, CapturedCodecOutput, CapturedDacVaeDecoder};
pub use model::{
    DACVAE_HOP_LENGTH, DACVAE_LATENT_DIM, DACVAE_SAMPLE_RATE, DacVaeCodec, DacVaeDecoder,
    Fixed112DacVaeDecoder,
};
#[cfg(feature = "profile")]
pub use profiling::{CodecStageTiming, CodecTimingSource};
pub use streaming::{DecodeStreamReport, DecodeWindow};
pub use weights::{
    load_codec, load_codec_with_float_dtype, load_decoder, load_decoder_with_float_dtype,
};
