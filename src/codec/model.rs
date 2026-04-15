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

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::prelude::*;
    use burn::tensor::ops::PadMode;

    type B = NdArray<f32>;

    /// Standalone reflect-pad helper matching `DacVaeCodec::pad_to_hop_length`.
    fn pad_to_hop(wav: Tensor<B, 3>, hop_length: usize) -> Tensor<B, 3> {
        let len = wav.dims()[2];
        let rem = len % hop_length;
        if rem == 0 {
            return wav;
        }
        let pad = hop_length - rem;
        wav.pad([(0, 0), (0, 0), (0, pad)], PadMode::Reflect)
    }

    #[test]
    fn pad_already_aligned_is_identity() {
        let device = <B as Backend>::Device::default();
        let hop = 1920;
        let wav = Tensor::<B, 3>::ones([1, 1, hop], &device);
        let padded = pad_to_hop(wav, hop);
        assert_eq!(padded.dims(), [1, 1, hop]);
    }

    #[test]
    fn pad_one_less_than_hop_pads_by_one() {
        let device = <B as Backend>::Device::default();
        let hop = 1920;
        let wav = Tensor::<B, 3>::ones([1, 1, hop - 1], &device);
        let padded = pad_to_hop(wav, hop);
        assert_eq!(
            padded.dims(),
            [1, 1, hop],
            "should pad to nearest hop multiple"
        );
    }

    #[test]
    fn pad_one_more_than_hop_pads_to_two_hops() {
        let device = <B as Backend>::Device::default();
        let hop = 1920;
        let wav = Tensor::<B, 3>::ones([1, 1, hop + 1], &device);
        let padded = pad_to_hop(wav, hop);
        assert_eq!(padded.dims(), [1, 1, hop * 2], "should pad to 2× hop");
    }

    #[test]
    fn pad_preserves_original_content() {
        let device = <B as Backend>::Device::default();
        let hop = 8;
        let data = Tensor::<B, 3>::from_floats([[[1.0, 2.0, 3.0, 4.0, 5.0]]], &device);
        let padded = pad_to_hop(data, hop);
        assert_eq!(padded.dims(), [1, 1, 8]);
        let vals: Vec<f32> = padded.into_data().to_vec().unwrap();
        assert_eq!(&vals[..5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn pad_uses_reflect_mode() {
        let device = <B as Backend>::Device::default();
        let hop = 8;
        // Input: [1, 2, 3, 4, 5] → reflect pad 3 → [1, 2, 3, 4, 5, 4, 3, 2]
        let data = Tensor::<B, 3>::from_floats([[[1.0, 2.0, 3.0, 4.0, 5.0]]], &device);
        let padded = pad_to_hop(data, hop);
        let vals: Vec<f32> = padded.into_data().to_vec().unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn pad_batched_preserves_batch_dim() {
        let device = <B as Backend>::Device::default();
        let hop = 8;
        let wav = Tensor::<B, 3>::ones([3, 1, 5], &device);
        let padded = pad_to_hop(wav, hop);
        assert_eq!(padded.dims(), [3, 1, 8]);
    }
}
