pub(crate) mod bottleneck;
pub(crate) mod decoder;
pub(crate) mod encoder;
pub(crate) mod layers;
pub(crate) mod model;
pub(crate) mod weights;

pub use model::{
    DACVAE_HOP_LENGTH, DACVAE_LATENT_DIM, DACVAE_SAMPLE_RATE, DacVaeCodec, DacVaeDecoder,
    Fixed112DacVaeDecoder,
};
pub use weights::{load_codec, load_decoder};
