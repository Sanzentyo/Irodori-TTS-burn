//! Top-level DACVAE codec: encode waveform → latent, decode latent → waveform.

use burn::{prelude::*, tensor::ops::PadMode};

use super::{bottleneck::VaeBottleneck, decoder::Decoder, encoder::Encoder};

#[cfg(feature = "profile")]
use super::algorithm::{
    CodecAlgorithmPlan, CodecK7Algorithm, CodecPointwiseAlgorithm, CodecStemAlgorithm,
};
#[cfg(feature = "profile")]
use super::profiling::{
    CodecStageProfiler, CodecStageTiming, DeviceCodecStageProfiler, NoopCodecStageProfiler,
    SynchronizedCodecStageProfiler,
};

/// Sample rate of the only Semantic-DACVAE topology accepted by the loader.
pub const DACVAE_SAMPLE_RATE: usize = 48_000;

/// Audio samples represented by one unpatched Semantic-DACVAE latent frame.
pub const DACVAE_HOP_LENGTH: usize = 2 * 8 * 10 * 12;

/// Channel width of one unpatched Semantic-DACVAE latent frame.
pub const DACVAE_LATENT_DIM: usize = 32;

/// Per-stage timings emitted only by diagnostic profiling builds.
#[cfg(feature = "profile")]
pub type CodecStageTimings = Vec<CodecStageTiming>;

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
pub struct DacVaeCodec {
    pub(crate) encoder: Encoder,
    pub(crate) bottleneck: VaeBottleneck,
    pub(crate) decoder: Decoder,
    pub(crate) hop_length: usize,
    pub(crate) sample_rate: usize,
}

/// Decode-only DACVAE model for synthesis workloads.
///
/// Unlike [`DacVaeCodec`], this type cannot encode reference audio and therefore
/// never allocates the encoder or the encode-side bottleneck projection.  The
/// decoder graph and weights are identical to [`DacVaeCodec::decode`].
#[derive(Module, Debug)]
pub struct DacVaeDecoder {
    pub(crate) out_proj: burn::nn::conv::Conv1d,
    pub(crate) decoder: Decoder,
    pub(crate) hop_length: usize,
    pub(crate) sample_rate: usize,
}

/// WGPU decoder committed to the exact 112-frame polyphase route.
///
/// The consuming transition releases the first upsampler source weight, so
/// this newtype exposes only fallible fixed-geometry decoding and cannot call
/// the portable fallback path.
#[derive(Debug)]
pub struct Fixed112DacVaeDecoder {
    inner: DacVaeDecoder,
}

impl DacVaeCodec {
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
    pub fn encode(&self, wav: Tensor<3>) -> Tensor<3> {
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
    pub fn decode(&self, latent: Tensor<3>) -> Tensor<3> {
        // [B, T, 32] → [B, 32, T]
        let code = latent.swap_dims(1, 2);
        let emb = self.bottleneck.decode(code);
        self.decoder.forward(emb)
    }

    /// Zero-pad the last dimension to the nearest multiple of `hop_length` (right-side only).
    ///
    /// Matches Python `DACVAE._pad`: `F.pad(audio, (0, right_pad), "reflect")`.
    fn pad_to_hop_length(&self, wav: Tensor<3>) -> Tensor<3> {
        let len = wav.dims()[2];
        let rem = len % self.hop_length;
        if rem == 0 {
            return wav;
        }
        let pad = self.hop_length - rem;
        wav.pad([(0, 0), (0, 0), (0, pad)], PadMode::Reflect)
    }

    /// Consume the full codec and release its encode-only state.
    ///
    /// Prefer [`crate::codec::load_decoder`] when decode-only residency is known
    /// before loading, since conversion still incurs the encoder's transient
    /// allocation peak.
    pub fn into_decoder(self) -> DacVaeDecoder {
        DacVaeDecoder {
            out_proj: self.bottleneck.out_proj,
            decoder: self.decoder,
            hop_length: self.hop_length,
            sample_rate: self.sample_rate,
        }
    }
}

impl DacVaeDecoder {
    /// The model sample rate in Hz (48 kHz).
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// The number of output audio samples represented by one latent frame.
    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// Decode a channel-last latent tensor to a mono waveform.
    pub fn decode(&self, latent: Tensor<3>) -> Tensor<3> {
        let code = latent.swap_dims(1, 2);
        let emb = self.out_proj.forward(code);
        self.decoder.forward(emb)
    }
}

impl DacVaeCodec {
    /// Materialize encoder pointwise weights in the channel-last GEMM layout.
    pub fn prepare_encoder_for_wgsl(&mut self) {
        self.encoder.prepare_for_inference();
    }

    /// Materialize decoder pointwise weights and the first polyphase upsampler.
    pub fn prepare_decoder_for_wgsl(&mut self) {
        self.decoder.prepare_for_wgsl();
    }

    /// Prepare diagnostic caches for one explicitly selected k=7 policy.
    #[cfg(feature = "profile")]
    pub fn prepare_decoder_for_wgsl_with_k7_algorithm(&mut self, k7_algorithm: CodecK7Algorithm) {
        self.decoder
            .prepare_for_wgsl_with_k7_algorithm(k7_algorithm);
    }

    /// Profile only the twelve request-time k=7 weight materializations.
    #[cfg(feature = "profile")]
    pub fn profile_k7_weight_repacks(
        &self,
    ) -> crate::error::Result<Vec<super::algorithm::K7WeightRepackReceipt>> {
        self.decoder.profile_k7_weight_repacks()
    }

    /// Materialize all pointwise codec weights for workloads using both sides.
    pub fn prepare_for_wgsl(&mut self) {
        self.prepare_encoder_for_wgsl();
        self.prepare_decoder_for_wgsl();
    }

    /// Encode with the production fused WGSL Snake activations.
    pub fn encode_wgsl(&self, wav: Tensor<3>) -> Tensor<3> {
        let wav = self.pad_to_hop_length(wav);
        let z = self.encoder.forward_wgsl(wav);
        self.bottleneck.encode_wgsl(z).swap_dims(1, 2)
    }

    /// Decode with the production fused WGSL Snake activations.
    pub fn decode_wgsl(&self, latent: Tensor<3>) -> Tensor<3> {
        let code = latent.swap_dims(1, 2);
        let emb = self.bottleneck.decode_wgsl(code);
        self.decoder.forward_wgsl(emb)
    }

    /// Profile the exact production decoder operators with an explicit synchronization
    /// after each stage. This is available only in profiling builds and is not used by
    /// the production decode path.
    #[cfg(feature = "profile")]
    pub fn decode_wgsl_profiled<E, S>(
        &self,
        latent: Tensor<3>,
        mut synchronize: S,
    ) -> Result<(Tensor<3>, CodecStageTimings), E>
    where
        S: FnMut(&'static str) -> Result<(), E>,
    {
        let mut profiler = SynchronizedCodecStageProfiler::new(&mut synchronize);
        let code = latent.swap_dims(1, 2);
        let emb = profiler.profile("codec_bottleneck", || self.bottleneck.decode_wgsl(code))?;
        let waveform = self.decoder.forward_wgsl_profiled(
            emb,
            CodecStemAlgorithm::AccuracyApproved,
            CodecK7Algorithm::AccuracyApproved,
            CodecPointwiseAlgorithm::AccuracyApproved,
            &mut profiler,
        )?;
        Ok((waveform, profiler.finish()))
    }

    /// Profile the production decoder with CubeCL stream timestamps.
    ///
    /// Unlike [`Self::decode_wgsl_profiled`], this does not synchronize after
    /// every stage. All timestamp futures are resolved only after the complete
    /// decoder graph has been enqueued. Adapters without device timestamps use
    /// CubeCL's explicit synchronized system-clock fallback and report that
    /// source in every [`CodecStageTiming`].
    #[cfg(feature = "profile")]
    pub fn decode_wgsl_device_profiled(
        &self,
        latent: Tensor<3>,
    ) -> crate::error::Result<(Tensor<3>, CodecStageTimings)> {
        self.decode_wgsl_device_profiled_with_k7_algorithm(
            latent,
            CodecK7Algorithm::AccuracyApproved,
        )
    }

    /// Profile one explicitly selected k=7 candidate on the production graph.
    ///
    /// This diagnostic API never changes [`Self::decode_wgsl`]; promotion to
    /// production requires an external accuracy and fresh-session receipt.
    #[cfg(feature = "profile")]
    pub fn decode_wgsl_device_profiled_with_k7_algorithm(
        &self,
        latent: Tensor<3>,
        k7_algorithm: CodecK7Algorithm,
    ) -> crate::error::Result<(Tensor<3>, CodecStageTimings)> {
        self.decode_wgsl_device_profiled_with_plan(
            latent,
            CodecAlgorithmPlan::new(k7_algorithm, CodecPointwiseAlgorithm::AccuracyApproved),
        )
    }

    /// Profile one complete differential algorithm plan.
    #[cfg(feature = "profile")]
    pub fn decode_wgsl_device_profiled_with_plan(
        &self,
        latent: Tensor<3>,
        plan: CodecAlgorithmPlan,
    ) -> crate::error::Result<(Tensor<3>, CodecStageTimings)> {
        let mut profiler = DeviceCodecStageProfiler::from_tensor(&latent)?;
        let code = latent.swap_dims(1, 2);
        let emb = profiler.profile("codec_bottleneck", || self.bottleneck.decode_wgsl(code))?;
        let waveform = self.decoder.forward_wgsl_profiled(
            emb,
            plan.stem,
            plan.k7,
            plan.pointwise,
            &mut profiler,
        )?;
        Ok((waveform, profiler.finish()?))
    }

    /// Run a selected k=7 policy without timestamp instrumentation.
    ///
    /// This is a profiling-only differential hook. Ordinary callers should
    /// use [`Self::decode_wgsl`], which owns the accuracy-approved policy.
    #[cfg(feature = "profile")]
    pub fn decode_wgsl_with_k7_algorithm(
        &self,
        latent: Tensor<3>,
        k7_algorithm: CodecK7Algorithm,
    ) -> Tensor<3> {
        self.decode_wgsl_with_plan(
            latent,
            CodecAlgorithmPlan::new(k7_algorithm, CodecPointwiseAlgorithm::AccuracyApproved),
        )
    }

    /// Run one complete differential algorithm plan without timestamps.
    #[cfg(feature = "profile")]
    pub fn decode_wgsl_with_plan(&self, latent: Tensor<3>, plan: CodecAlgorithmPlan) -> Tensor<3> {
        let mut profiler = NoopCodecStageProfiler;
        let code = latent.swap_dims(1, 2);
        let emb = self.bottleneck.decode_wgsl(code);
        match self.decoder.forward_wgsl_profiled(
            emb,
            plan.stem,
            plan.k7,
            plan.pointwise,
            &mut profiler,
        ) {
            Ok(waveform) => waveform,
            Err(never) => match never {},
        }
    }

    /// Materialize the exact decoder-stem input for an isolated profiling A/B.
    #[cfg(feature = "profile")]
    pub fn decoder_stem_input_wgsl(&self, latent: Tensor<3>) -> Tensor<3> {
        let code = latent.swap_dims(1, 2);
        self.bottleneck.decode_wgsl(code)
    }

    /// Run the current production WGPU decoder-stem route for profiling.
    #[cfg(feature = "profile")]
    pub fn decoder_stem_current_wgsl(&self, input: Tensor<3>) -> Tensor<3> {
        self.decoder.stem_wgsl_or_fallback(input)
    }

    /// Run the unchanged Burn stem as the profiling reference only.
    #[cfg(feature = "profile")]
    pub fn decoder_stem_burn_reference_wgsl(&self, input: Tensor<3>) -> Tensor<3> {
        self.decoder.stem.forward(input)
    }
}

impl DacVaeDecoder {
    /// Materialize the same decoder caches as [`DacVaeCodec::prepare_decoder_for_wgsl`].
    pub fn prepare_for_wgsl(&mut self) {
        self.decoder.prepare_for_wgsl();
    }

    /// Consume this decoder, validate its prepared polyphase cache, and commit
    /// it to 112-frame inputs.
    pub fn into_fixed_112_for_wgsl(mut self) -> crate::error::Result<Fixed112DacVaeDecoder> {
        self.decoder.lock_fixed_112_wgsl()?;
        Ok(Fixed112DacVaeDecoder { inner: self })
    }

    /// Decode with the same production WGSL path as [`DacVaeCodec::decode_wgsl`].
    pub fn decode_wgsl(&self, latent: Tensor<3>) -> Tensor<3> {
        let code = latent.swap_dims(1, 2);
        let emb = super::layers::pointwise_conv1d(&self.out_proj, code);
        self.decoder.forward_wgsl(emb)
    }
}

impl Fixed112DacVaeDecoder {
    /// Decode an exact 112-frame latent or return a configuration error before
    /// submitting codec work.
    pub fn decode_wgsl(&self, latent: Tensor<3>) -> crate::error::Result<Tensor<3>> {
        let frames = latent.dims()[1];
        if frames != 112 {
            return Err(crate::error::IrodoriError::Config(format!(
                "fixed-112 codec rejects {frames} latent frames"
            )));
        }
        let code = latent.swap_dims(1, 2);
        let emb = super::layers::pointwise_conv1d(&self.inner.out_proj, code);
        self.inner.decoder.forward_fixed_112_wgsl(emb)
    }
}

#[cfg(test)]
mod tests {
    use burn::prelude::*;
    use burn::tensor::ops::PadMode;

    /// Standalone reflect-pad helper matching `DacVaeCodec::pad_to_hop_length`.
    fn pad_to_hop(wav: Tensor<3>, hop_length: usize) -> Tensor<3> {
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
        let device = Device::default();
        let hop = 1920;
        let wav = Tensor::<3>::ones([1, 1, hop], &device);
        let padded = pad_to_hop(wav, hop);
        assert_eq!(padded.dims(), [1, 1, hop]);
    }

    #[test]
    fn pad_one_less_than_hop_pads_by_one() {
        let device = Device::default();
        let hop = 1920;
        let wav = Tensor::<3>::ones([1, 1, hop - 1], &device);
        let padded = pad_to_hop(wav, hop);
        assert_eq!(
            padded.dims(),
            [1, 1, hop],
            "should pad to nearest hop multiple"
        );
    }

    #[test]
    fn pad_one_more_than_hop_pads_to_two_hops() {
        let device = Device::default();
        let hop = 1920;
        let wav = Tensor::<3>::ones([1, 1, hop + 1], &device);
        let padded = pad_to_hop(wav, hop);
        assert_eq!(padded.dims(), [1, 1, hop * 2], "should pad to 2× hop");
    }

    #[test]
    fn pad_preserves_original_content() {
        let device = Device::default();
        let hop = 8;
        let data = Tensor::<3>::from_floats([[[1.0, 2.0, 3.0, 4.0, 5.0]]], &device);
        let padded = pad_to_hop(data, hop);
        assert_eq!(padded.dims(), [1, 1, 8]);
        let vals: Vec<f32> = padded.into_data().to_vec().unwrap();
        assert_eq!(&vals[..5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn pad_uses_reflect_mode() {
        let device = Device::default();
        let hop = 8;
        // Input: [1, 2, 3, 4, 5] → reflect pad 3 → [1, 2, 3, 4, 5, 4, 3, 2]
        let data = Tensor::<3>::from_floats([[[1.0, 2.0, 3.0, 4.0, 5.0]]], &device);
        let padded = pad_to_hop(data, hop);
        let vals: Vec<f32> = padded.into_data().to_vec().unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0]);
    }

    #[test]
    fn pad_batched_preserves_batch_dim() {
        let device = Device::default();
        let hop = 8;
        let wav = Tensor::<3>::ones([3, 1, 5], &device);
        let padded = pad_to_hop(wav, hop);
        assert_eq!(padded.dims(), [3, 1, 8]);
    }
}
