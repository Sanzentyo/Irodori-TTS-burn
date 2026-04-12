//! Top-level DACVAE codec: encode waveform → latent, decode latent → waveform.

use burn::{prelude::*, tensor::ops::PadMode};

use super::{bottleneck::VaeBottleneck, decoder::Decoder, encoder::Encoder};

/// Combined DACVAE encode/decode model.
///
/// **Encode** (`[B,1,S]` → `[B,T,32]`):
/// 1. Reflect-pad input to the nearest multiple of `hop_length`.
/// 2. Run encoder to produce `z [B, 1024, T]`.
/// 3. Project with `bottleneck.in_proj` and take the mean half → `[B, 32, T]`.
/// 4. Transpose to channel-last: `[B, T, 32]`.
///
/// **Decode** (`[B,T,32]` → `[B,1,S]`):
/// 1. Transpose to channel-first: `[B, 32, T]`.
/// 2. Project with `bottleneck.out_proj` → `[B, 1024, T]`.
/// 3. Run decoder (no-watermark path) → `[B, 1, S]`.
#[derive(Module, Debug)]
pub struct DacVaeCodec<B: Backend> {
    pub(crate) encoder: Encoder<B>,
    pub(crate) bottleneck: VaeBottleneck<B>,
    pub(crate) decoder: Decoder<B>,
    pub(crate) hop_length: usize,
    pub(crate) sample_rate: usize,
}

impl<B: Backend> DacVaeCodec<B> {
    /// The model sample rate in Hz (48 kHz).
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// The hop length (number of audio samples per latent frame).
    pub fn hop_length(&self) -> usize {
        self.hop_length
    }
    /// Encode a mono waveform to a channel-last latent tensor.
    ///
    /// # Arguments
    /// * `wav` — `[B, 1, samples]` float32 audio at the model sample rate (48 kHz).
    ///
    /// # Returns
    /// Latent tensor `[B, T, 32]`.
    pub fn encode(&self, wav: Tensor<B, 3>) -> Tensor<B, 3> {
        let wav = self.pad_to_hop_length(wav);
        let z = self.encoder.forward(wav);
        let code = self.bottleneck.encode(z);
        // [B, 32, T] → [B, T, 32]
        code.swap_dims(1, 2)
    }

    /// Decode a channel-last latent tensor to a mono waveform.
    ///
    /// # Arguments
    /// * `latent` — `[B, T, 32]`.
    ///
    /// # Returns
    /// Audio tensor `[B, 1, samples]`.
    pub fn decode(&self, latent: Tensor<B, 3>) -> Tensor<B, 3> {
        // [B, T, 32] → [B, 32, T]
        let code = latent.swap_dims(1, 2);
        let emb = self.bottleneck.decode(code);
        self.decoder.forward(emb)
    }

    /// Zero-pad the last dimension to the nearest multiple of `hop_length` (right-side only).
    ///
    /// Matches Python `DACVAE._pad`: `F.pad(audio, (0, right_pad), "reflect")`.
    fn pad_to_hop_length(&self, wav: Tensor<B, 3>) -> Tensor<B, 3> {
        let len = wav.dims()[2];
        let rem = len % self.hop_length;
        if rem == 0 {
            return wav;
        }
        let pad = self.hop_length - rem;
        wav.pad([(0, 0), (0, 0), (0, pad)], PadMode::Reflect)
    }
}
