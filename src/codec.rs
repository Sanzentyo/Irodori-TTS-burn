pub(crate) mod algorithm;
pub(crate) mod bottleneck;
pub(crate) mod decoder;
pub(crate) mod encoder;
pub(crate) mod layers;
pub(crate) mod model;
#[cfg(feature = "profile")]
pub(crate) mod profiling;
pub(crate) mod weights;

pub use algorithm::CodecK7Algorithm;
#[cfg(feature = "profile")]
pub use algorithm::{
    CodecAlgorithmPlan, CodecConvTransposeSnakeFusion, CodecCrossBlockFusion,
    CodecPointwiseAlgorithm, CodecResidualStateLayout, CodecStemAlgorithm,
};
#[cfg(feature = "profile")]
pub use algorithm::{K7WeightRepackReceipt, PreparedK7WeightPolicy};
pub use model::{
    DACVAE_HOP_LENGTH, DACVAE_LATENT_DIM, DACVAE_SAMPLE_RATE, DacVaeCodec, DacVaeDecoder,
    Fixed112DacVaeDecoder,
};
#[cfg(feature = "profile")]
pub use profiling::{CodecStageTiming, CodecTimingSource};
pub use weights::{load_codec, load_decoder};
