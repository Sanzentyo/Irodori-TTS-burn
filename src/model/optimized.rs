//! Type-safe wrapper for inference-optimized models.
//!
//! [`InferenceOptimizedModel`] guarantees at the type level that weight matrices
//! have been fused for inference. It exposes only read-only inference methods
//! and cannot be moved to a different device (fused tensors are `#[module(skip)]`
//! and would become stale).
//!
//! # Construction
//!
//! ```rust,ignore
//! use irodori_tts_burn::model::TextToLatentRfDiT;
//! use irodori_tts_burn::model::InferenceOptimizedModel;
//!
//! let model: TextToLatentRfDiT<B> = load_model(...)?;
//! let optimized = InferenceOptimizedModel::from(model);
//! ```

use burn::tensor::{Bool, Int, Tensor, backend::Backend};

use super::TextToLatentRfDiT;
use super::attention::CondKvCache;
use super::condition::{AuxConditionInput, EncodedCondition};
use super::rope::RopeFreqs;

/// A [`TextToLatentRfDiT`] with all weight matrices fused for inference.
///
/// This newtype enforces at the type level that:
/// - QKV projections in every attention block are fused (3→1 matmul)
/// - SwiGLU w1/w3 projections in every MLP block are fused (2→1 matmul)
///
/// The wrapper exposes only the read-only methods needed for sampling.
/// It does **not** implement `Deref`, derive `Module`, or expose `&mut` access
/// to the inner model — preventing accidental `to_device()` / `fork()` calls
/// that would invalidate the fused `#[module(skip)]` tensors.
///
/// Created via [`From<TextToLatentRfDiT<B>>`] or [`InferenceOptimizedModel::new`].
#[derive(Debug)]
pub struct InferenceOptimizedModel<B: Backend> {
    inner: TextToLatentRfDiT<B>,
}

impl<B: Backend> From<TextToLatentRfDiT<B>> for InferenceOptimizedModel<B> {
    /// Consume an unfused model, fuse all weight matrices, and return
    /// an inference-optimized wrapper.
    ///
    /// This is a **one-way transition**: the original model is consumed
    /// and cannot be recovered.
    fn from(mut model: TextToLatentRfDiT<B>) -> Self {
        model.prepare_for_inference();
        Self { inner: model }
    }
}

impl<B: Backend> InferenceOptimizedModel<B> {
    /// Consume an unfused model, fuse all weight matrices, and return
    /// an inference-optimized wrapper.
    ///
    /// Equivalent to `InferenceOptimizedModel::from(model)`.
    pub fn new(model: TextToLatentRfDiT<B>) -> Self {
        Self::from(model)
    }

    // -----------------------------------------------------------------------
    // Delegated read-only inference methods
    // -----------------------------------------------------------------------

    /// Encode all conditioning inputs (text + optional speaker/caption).
    ///
    /// See [`TextToLatentRfDiT::encode_conditions`] for details.
    pub fn encode_conditions(
        &self,
        text_input_ids: Tensor<B, 2, Int>,
        text_mask: Tensor<B, 2, Bool>,
        aux_input: AuxConditionInput<B>,
    ) -> crate::error::Result<EncodedCondition<B>> {
        self.inner
            .encode_conditions(text_input_ids, text_mask, aux_input)
    }

    /// Run the diffusion backbone with pre-encoded conditions and KV caches,
    /// using fused weight matrices (branch-free hot path).
    ///
    /// See [`TextToLatentRfDiT::forward_with_cond_cached`] for details.
    pub fn forward_with_cond_cached(
        &self,
        x_t: Tensor<B, 3>,
        t: Tensor<B, 1>,
        cond: &EncodedCondition<B>,
        latent_mask: Option<Tensor<B, 2, Bool>>,
        kv_caches: Option<&[CondKvCache<B>]>,
        lat_rope: &RopeFreqs<B>,
    ) -> Tensor<B, 3> {
        self.inner
            .forward_with_cond_cached_fused(x_t, t, cond, latent_mask, kv_caches, lat_rope)
    }

    /// Precompute RoPE frequency tables for the latent sequence.
    ///
    /// See [`TextToLatentRfDiT::precompute_latent_rope`] for details.
    pub fn precompute_latent_rope(&self, seq_lat: usize, device: &B::Device) -> RopeFreqs<B> {
        self.inner.precompute_latent_rope(seq_lat, device)
    }

    /// Pre-project all context K/V for each block.
    ///
    /// See [`TextToLatentRfDiT::build_kv_caches`] for details.
    pub fn build_kv_caches(
        &self,
        cond: &EncodedCondition<B>,
        seq_lat: Option<usize>,
    ) -> Vec<CondKvCache<B>> {
        self.inner.build_kv_caches(cond, seq_lat)
    }

    /// Whether the model uses speaker (reference audio) conditioning.
    pub fn use_speaker_condition(&self) -> bool {
        self.inner.use_speaker_condition()
    }

    /// Dimension of the patched latent space (input/output channels per token).
    pub fn patched_latent_dim(&self) -> usize {
        self.inner.patched_latent_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type B = NdArray<f32>;

    fn tiny_cfg() -> crate::config::ModelConfig {
        crate::config::tiny_model_config()
    }

    fn device() -> <B as Backend>::Device {
        Default::default()
    }

    #[test]
    fn from_model_produces_fused_output() {
        let cfg = tiny_cfg();
        let dev = device();
        let model = TextToLatentRfDiT::<B>::new(&cfg, &dev);

        let x_t = Tensor::random(
            [1, 4, cfg.patched_latent_dim()],
            Distribution::Default,
            &dev,
        );
        let t = Tensor::from_data([0.5_f32], &dev);
        let text_ids = Tensor::<B, 2, Int>::zeros([1, 2], &dev);
        let text_mask = Tensor::<B, 2, Bool>::from_data(
            burn::tensor::TensorData::new(vec![true, true], [1, 2]),
            &dev,
        );

        // Compute reference output with the unfused model
        let cond = model
            .encode_conditions(text_ids.clone(), text_mask.clone(), AuxConditionInput::None)
            .unwrap();
        let lat_rope = model.precompute_latent_rope(4, &dev);
        let out_unfused =
            model.forward_with_cond_cached(x_t.clone(), t.clone(), &cond, None, None, &lat_rope);

        // Convert to optimized and compute
        let optimized = InferenceOptimizedModel::from(model);
        let cond = optimized
            .encode_conditions(text_ids, text_mask, AuxConditionInput::None)
            .unwrap();
        let lat_rope = optimized.precompute_latent_rope(4, &dev);
        let out_fused = optimized.forward_with_cond_cached(x_t, t, &cond, None, None, &lat_rope);

        let diff: f32 = (out_unfused - out_fused)
            .abs()
            .max()
            .to_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(
            diff < 1e-5,
            "optimized model should match unfused: max_diff={diff}"
        );
    }

    #[test]
    fn use_speaker_condition_delegates() {
        let cfg = tiny_cfg();
        let dev = device();
        let model = TextToLatentRfDiT::<B>::new(&cfg, &dev);
        let expected = model.use_speaker_condition();
        let optimized = InferenceOptimizedModel::from(model);
        assert_eq!(optimized.use_speaker_condition(), expected);
    }

    #[test]
    fn patched_latent_dim_delegates() {
        let cfg = tiny_cfg();
        let dev = device();
        let model = TextToLatentRfDiT::<B>::new(&cfg, &dev);
        let expected = model.patched_latent_dim();
        let optimized = InferenceOptimizedModel::from(model);
        assert_eq!(optimized.patched_latent_dim(), expected);
    }
}
