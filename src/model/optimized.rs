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
//! use irodori_tts_wgpu::model::TextToLatentRfDiT;
//! use irodori_tts_wgpu::model::InferenceOptimizedModel;
//!
//! let model: TextToLatentRfDiT<B> = load_model(...)?;
//! let optimized = InferenceOptimizedModel::from(model);
//! ```

use burn::tensor::{Bool, Int, Tensor, backend::Backend};

use super::TextToLatentRfDiT;
use super::adaln_cross_layer::CrossLayerAdaLnCache;
use super::attention::{CondKvCache, TextCfgKvCachePair};
use super::condition::{AuxConditionInput, EncodedCondition};
use super::rope::RopeFreqs;
use super::timestep_condition::{FixedEulerCondCache, ModelGeneration};
use super::wgsl::TextOnlyCfgCacheProof;

/// A [`TextToLatentRfDiT`] with all weight matrices fused for inference.
///
/// This newtype enforces at the type level that:
/// - QKV+gate projections in every attention block are fused (4→1 matmul)
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
    device: B::Device,
}

impl<B: Backend> From<TextToLatentRfDiT<B>> for InferenceOptimizedModel<B> {
    /// Consume an unfused model, fuse all weight matrices, and return
    /// an inference-optimized wrapper.
    ///
    /// This is a **one-way transition**: the original model is consumed
    /// and cannot be recovered.
    fn from(mut model: TextToLatentRfDiT<B>) -> Self {
        use burn::module::Module;
        let device = model
            .devices()
            .into_iter()
            .next()
            .expect("model must reside on at least one device");
        model.prepare_for_inference();
        Self {
            inner: model,
            device,
        }
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

    /// The device this model resides on.
    ///
    /// Captured at construction time and guaranteed not to change (the wrapper
    /// prevents `to_device()` calls).
    pub fn device(&self) -> &B::Device {
        &self.device
    }

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

    /// Whether the model uses caption conditioning.
    pub fn use_caption_condition(&self) -> bool {
        self.inner.use_caption_condition()
    }

    /// Predict `log1p(latent_frames)` from already encoded conditions.
    pub fn predict_duration_log_frames(
        &self,
        cond: &EncodedCondition<B>,
        duration_features: Tensor<B, 2>,
        has_speaker: Tensor<B, 1, Bool>,
        has_caption: Tensor<B, 1, Bool>,
    ) -> crate::error::Result<Tensor<B, 1>> {
        self.inner
            .predict_duration_log_frames(cond, duration_features, has_speaker, has_caption)
    }

    /// Whether automatic duration weights are present.
    pub fn has_duration_predictor(&self) -> bool {
        self.inner.has_duration_predictor()
    }

    /// Dimension of the patched latent space (input/output channels per token).
    pub fn patched_latent_dim(&self) -> usize {
        self.inner.patched_latent_dim()
    }
}

/// Inference-optimized WGPU model whose hot path is dispatched through the
/// production WGSL fusion policy.
///
/// This separate newtype makes the backend policy explicit: constructing the
/// portable [`InferenceOptimizedModel`] never silently changes numerical
/// kernels, while this type guarantees that the measured AdaLN, SwiGLU,
/// JointAttention materialization, gated-residual, and final RMSNorm shaders
/// are used.
#[derive(Debug)]
pub struct WgslInferenceOptimizedModel {
    inner: InferenceOptimizedModel<crate::WgpuRaw>,
    cross_layer_adaln: Option<Box<CrossLayerAdaLnCache<crate::WgpuRaw>>>,
    generation: ModelGeneration,
}

impl From<TextToLatentRfDiT<crate::WgpuRaw>> for WgslInferenceOptimizedModel {
    fn from(mut model: TextToLatentRfDiT<crate::WgpuRaw>) -> Self {
        let modules = model
            .blocks
            .iter()
            .flat_map(|block| [&block.attention_adaln, &block.mlp_adaln])
            .collect::<Vec<_>>();
        let mut cross_layer_adaln = None;
        CrossLayerAdaLnCache::prepare_v4_wgsl(&mut cross_layer_adaln, &modules);
        model.prepare_attention_materialization_wgsl();
        model.prepare_swiglu_w2_row_major_wgsl();
        Self {
            inner: InferenceOptimizedModel::from(model),
            cross_layer_adaln: cross_layer_adaln.map(Box::new),
            generation: ModelGeneration::fresh(),
        }
    }
}

impl WgslInferenceOptimizedModel {
    /// Consume a loaded WGPU model, fuse inference weights, and select WGSL
    /// hot-path execution.
    pub fn new(model: TextToLatentRfDiT<crate::WgpuRaw>) -> Self {
        Self::from(model)
    }

    pub fn device(&self) -> &<crate::WgpuRaw as Backend>::Device {
        self.inner.device()
    }

    pub(crate) const fn model_generation(&self) -> ModelGeneration {
        self.generation
    }

    pub(crate) fn try_build_fixed_euler_cond_cache(&self) -> Option<FixedEulerCondCache> {
        FixedEulerCondCache::try_build(&self.inner.inner, self.generation, self.inner.device())
    }

    pub(crate) fn fixed_euler_cond_cache_matches(&self, cache: &FixedEulerCondCache) -> bool {
        cache.matches_model(self.generation, self.inner.device())
    }

    pub fn encode_conditions(
        &self,
        text_input_ids: Tensor<crate::WgpuRaw, 2, Int>,
        text_mask: Tensor<crate::WgpuRaw, 2, Bool>,
        aux_input: AuxConditionInput<crate::WgpuRaw>,
    ) -> crate::error::Result<EncodedCondition<crate::WgpuRaw>> {
        self.inner
            .inner
            .encode_conditions_wgsl(text_input_ids, text_mask, aux_input)
    }

    pub fn forward_with_cond_cached(
        &self,
        x_t: Tensor<crate::WgpuRaw, 3>,
        t: Tensor<crate::WgpuRaw, 1>,
        cond: &EncodedCondition<crate::WgpuRaw>,
        latent_mask: Option<Tensor<crate::WgpuRaw, 2, Bool>>,
        kv_caches: Option<&[CondKvCache<crate::WgpuRaw>]>,
        lat_rope: &RopeFreqs<crate::WgpuRaw>,
    ) -> Tensor<crate::WgpuRaw, 3> {
        self.inner.inner.forward_with_cond_cached_wgsl(
            self.cross_layer_adaln.as_deref(),
            x_t,
            t,
            cond,
            latent_mask,
            kv_caches,
            lat_rope,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_forward_with_precomputed_cond_cached(
        &self,
        x_t: Tensor<crate::WgpuRaw, 3>,
        cond_embed: Tensor<crate::WgpuRaw, 3>,
        cond: &EncodedCondition<crate::WgpuRaw>,
        latent_mask: Option<Tensor<crate::WgpuRaw, 2, Bool>>,
        kv_caches: Option<&[CondKvCache<crate::WgpuRaw>]>,
        lat_rope: &RopeFreqs<crate::WgpuRaw>,
    ) -> Option<Tensor<crate::WgpuRaw, 3>> {
        self.inner.inner.try_forward_with_precomputed_cond_wgsl(
            self.cross_layer_adaln.as_deref(),
            x_t,
            cond_embed,
            cond,
            latent_mask,
            kv_caches,
            lat_rope,
        )
    }

    pub fn precompute_latent_rope(
        &self,
        seq_lat: usize,
        device: &<crate::WgpuRaw as Backend>::Device,
    ) -> RopeFreqs<crate::WgpuRaw> {
        self.inner.precompute_latent_rope(seq_lat, device)
    }

    pub fn build_kv_caches(
        &self,
        cond: &EncodedCondition<crate::WgpuRaw>,
        seq_lat: Option<usize>,
    ) -> Vec<CondKvCache<crate::WgpuRaw>> {
        self.inner.inner.build_kv_caches_wgsl(cond, seq_lat)
    }

    pub(crate) fn try_build_text_cfg_kv_caches(
        &self,
        cond: &EncodedCondition<crate::WgpuRaw>,
        batched_cfg: &EncodedCondition<crate::WgpuRaw>,
        seq_lat: usize,
        proof: Option<&TextOnlyCfgCacheProof>,
    ) -> Option<TextCfgKvCachePair<crate::WgpuRaw>> {
        self.inner.inner.try_build_text_cfg_kv_caches_wgsl(
            cond,
            batched_cfg,
            seq_lat,
            self.inner.device(),
            proof,
        )
    }

    pub fn use_speaker_condition(&self) -> bool {
        self.inner.use_speaker_condition()
    }

    pub fn use_caption_condition(&self) -> bool {
        self.inner.use_caption_condition()
    }

    /// Predict `log1p(latent_frames)` through the loaded v4 duration head.
    pub fn predict_duration_log_frames(
        &self,
        cond: &EncodedCondition<crate::WgpuRaw>,
        duration_features: Tensor<crate::WgpuRaw, 2>,
        has_speaker: Tensor<crate::WgpuRaw, 1, Bool>,
        has_caption: Tensor<crate::WgpuRaw, 1, Bool>,
    ) -> crate::error::Result<Tensor<crate::WgpuRaw, 1>> {
        self.inner
            .predict_duration_log_frames(cond, duration_features, has_speaker, has_caption)
    }

    pub fn has_duration_predictor(&self) -> bool {
        self.inner.has_duration_predictor()
    }

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

    #[test]
    fn device_returns_construction_device() {
        let cfg = tiny_cfg();
        let dev = device();
        let model = TextToLatentRfDiT::<B>::new(&cfg, &dev);
        let optimized = InferenceOptimizedModel::from(model);
        assert_eq!(*optimized.device(), dev);
    }

    #[test]
    fn fused_parity_with_speaker_conditioning() {
        let cfg = tiny_cfg();
        let dev = device();
        let model = TextToLatentRfDiT::<B>::new(&cfg, &dev);

        let (batch, seq_txt, seq_ref, seq_lat) = (1, 2, 3, 4);
        let x_t = Tensor::random(
            [batch, seq_lat, cfg.patched_latent_dim()],
            Distribution::Default,
            &dev,
        );
        let t = Tensor::from_data([0.5_f32], &dev);
        let text_ids = Tensor::<B, 2, Int>::zeros([batch, seq_txt], &dev);
        let text_mask = Tensor::<B, 2, Bool>::from_data(
            burn::tensor::TensorData::new(vec![true; batch * seq_txt], [batch, seq_txt]),
            &dev,
        );
        let speaker_dim = cfg.speaker_patched_latent_dim();
        let ref_latent = Tensor::<B, 3>::ones([batch, seq_ref, speaker_dim], &dev);
        let ref_mask = Tensor::<B, 2, Bool>::from_data(
            burn::tensor::TensorData::new(vec![true; batch * seq_ref], [batch, seq_ref]),
            &dev,
        );

        let aux = AuxConditionInput::Speaker {
            ref_latent: ref_latent.clone(),
            ref_mask: ref_mask.clone(),
        };

        // Unfused forward
        let cond = model
            .encode_conditions(text_ids.clone(), text_mask.clone(), aux)
            .unwrap();
        let lat_rope = model.precompute_latent_rope(seq_lat, &dev);
        let out_unfused =
            model.forward_with_cond_cached(x_t.clone(), t.clone(), &cond, None, None, &lat_rope);

        // Fused forward
        let optimized = InferenceOptimizedModel::from(model);
        let aux = AuxConditionInput::Speaker {
            ref_latent,
            ref_mask,
        };
        let cond = optimized
            .encode_conditions(text_ids, text_mask, aux)
            .unwrap();
        let lat_rope = optimized.precompute_latent_rope(seq_lat, &dev);
        let out_fused = optimized.forward_with_cond_cached(x_t, t, &cond, None, None, &lat_rope);

        let diff: f32 = (out_unfused - out_fused)
            .abs()
            .max()
            .to_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(
            diff < 1e-5,
            "speaker-conditioned fused should match unfused: max_diff={diff}"
        );
    }

    #[test]
    fn fused_parity_with_kv_caches() {
        let cfg = tiny_cfg();
        let dev = device();
        let model = TextToLatentRfDiT::<B>::new(&cfg, &dev);

        let (batch, seq_txt, seq_ref, seq_lat) = (1, 2, 3, 4);
        let x_t = Tensor::random(
            [batch, seq_lat, cfg.patched_latent_dim()],
            Distribution::Default,
            &dev,
        );
        let t = Tensor::from_data([0.5_f32], &dev);
        let text_ids = Tensor::<B, 2, Int>::zeros([batch, seq_txt], &dev);
        let text_mask = Tensor::<B, 2, Bool>::from_data(
            burn::tensor::TensorData::new(vec![true; batch * seq_txt], [batch, seq_txt]),
            &dev,
        );
        let speaker_dim = cfg.speaker_patched_latent_dim();
        let ref_latent = Tensor::<B, 3>::ones([batch, seq_ref, speaker_dim], &dev);
        let ref_mask = Tensor::<B, 2, Bool>::from_data(
            burn::tensor::TensorData::new(vec![true; batch * seq_ref], [batch, seq_ref]),
            &dev,
        );

        let aux = AuxConditionInput::Speaker {
            ref_latent: ref_latent.clone(),
            ref_mask: ref_mask.clone(),
        };

        // Unfused forward with KV caches
        let cond = model
            .encode_conditions(text_ids.clone(), text_mask.clone(), aux)
            .unwrap();
        let kv_caches = model.build_kv_caches(&cond, Some(seq_lat));
        let lat_rope = model.precompute_latent_rope(seq_lat, &dev);
        let out_unfused = model.forward_with_cond_cached(
            x_t.clone(),
            t.clone(),
            &cond,
            None,
            Some(&kv_caches),
            &lat_rope,
        );

        // Fused forward with KV caches
        let optimized = InferenceOptimizedModel::from(model);
        let aux = AuxConditionInput::Speaker {
            ref_latent,
            ref_mask,
        };
        let cond = optimized
            .encode_conditions(text_ids, text_mask, aux)
            .unwrap();
        let kv_caches = optimized.build_kv_caches(&cond, Some(seq_lat));
        let lat_rope = optimized.precompute_latent_rope(seq_lat, &dev);
        let out_fused =
            optimized.forward_with_cond_cached(x_t, t, &cond, None, Some(&kv_caches), &lat_rope);

        let diff: f32 = (out_unfused - out_fused)
            .abs()
            .max()
            .to_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(
            diff < 1e-5,
            "KV-cached fused should match unfused: max_diff={diff}"
        );
    }
}
