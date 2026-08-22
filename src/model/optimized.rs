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
//! let model: TextToLatentRfDiT = load_model(...)?;
//! let optimized = InferenceOptimizedModel::from(model);
//! ```

use burn::tensor::Device;
use burn::tensor::{Bool, Int, Tensor};

use super::adaln_cross_layer::{CrossLayerAdaLnCache, CrossLayerAdaLnModulations};
use super::attention::{CondKvCache, TextCfgKvCachePair};
use super::condition::{AuxConditionInput, EncodedCondition};
use super::rope::RopeFreqs;
use super::timestep_condition::{FixedEulerCondCache, ModelGeneration};
use super::wgsl::TextOnlyCfgCacheProof;
use super::{BlockDebugOutputs, TextToLatentRfDiT};

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
/// Created via [`From<TextToLatentRfDiT>`] or [`InferenceOptimizedModel::new`].
#[derive(Debug)]
pub struct InferenceOptimizedModel {
    inner: TextToLatentRfDiT,
    device: Device,
}

impl From<TextToLatentRfDiT> for InferenceOptimizedModel {
    /// Consume an unfused model, fuse all weight matrices, and return
    /// an inference-optimized wrapper.
    ///
    /// This is a **one-way transition**: the original model is consumed
    /// and cannot be recovered.
    fn from(mut model: TextToLatentRfDiT) -> Self {
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

impl InferenceOptimizedModel {
    fn from_prepared(model: TextToLatentRfDiT) -> Self {
        use burn::module::Module;
        let device = model
            .devices()
            .into_iter()
            .next()
            .expect("model must reside on at least one device");
        Self {
            inner: model,
            device,
        }
    }

    /// Consume an unfused model, fuse all weight matrices, and return
    /// an inference-optimized wrapper.
    ///
    /// Equivalent to `InferenceOptimizedModel::from(model)`.
    pub fn new(model: TextToLatentRfDiT) -> Self {
        Self::from(model)
    }

    // -----------------------------------------------------------------------
    // Delegated read-only inference methods
    // -----------------------------------------------------------------------

    /// The device this model resides on.
    ///
    /// Captured at construction time and guaranteed not to change (the wrapper
    /// prevents `to_device()` calls).
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Encode all conditioning inputs (text + optional speaker/caption).
    ///
    /// See [`TextToLatentRfDiT::encode_conditions`] for details.
    pub fn encode_conditions(
        &self,
        text_input_ids: Tensor<2, Int>,
        text_mask: Tensor<2, Bool>,
        aux_input: AuxConditionInput,
    ) -> crate::error::Result<EncodedCondition> {
        self.inner
            .encode_conditions(text_input_ids, text_mask, aux_input)
    }

    /// Run the diffusion backbone with pre-encoded conditions and KV caches,
    /// using fused weight matrices (branch-free hot path).
    ///
    /// See [`TextToLatentRfDiT::forward_with_cond_cached`] for details.
    pub fn forward_with_cond_cached(
        &self,
        x_t: Tensor<3>,
        t: Tensor<1>,
        cond: &EncodedCondition,
        latent_mask: Option<Tensor<2, Bool>>,
        kv_caches: Option<&[CondKvCache]>,
        lat_rope: &RopeFreqs,
    ) -> Tensor<3> {
        self.inner
            .forward_with_cond_cached_fused(x_t, t, cond, latent_mask, kv_caches, lat_rope)
    }

    /// Precompute RoPE frequency tables for the latent sequence.
    ///
    /// See [`TextToLatentRfDiT::precompute_latent_rope`] for details.
    pub fn precompute_latent_rope(&self, seq_lat: usize, device: &Device) -> RopeFreqs {
        self.inner.precompute_latent_rope(seq_lat, device)
    }

    /// Pre-project all context K/V for each block.
    ///
    /// See [`TextToLatentRfDiT::build_kv_caches`] for details.
    pub fn build_kv_caches(
        &self,
        cond: &EncodedCondition,
        seq_lat: Option<usize>,
    ) -> Vec<CondKvCache> {
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
        cond: &EncodedCondition,
        duration_features: Tensor<2>,
        has_speaker: Tensor<1, Bool>,
        has_caption: Tensor<1, Bool>,
    ) -> crate::error::Result<Tensor<1>> {
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

/// Layouts have been selected, but the irreversible preparation has not run.
#[cfg(all(feature = "inference", feature = "codec"))]
#[derive(Debug)]
pub struct LayoutsSelected {
    model: TextToLatentRfDiT,
}

/// Proof that only the selected physical weight layouts remain reachable.
#[cfg(all(feature = "inference", feature = "codec"))]
#[derive(Debug)]
pub struct ProfileLocked {
    model: WgslInferenceOptimizedModel,
}

/// Type-state transition from a loaded model plus layout set to a profile-
/// locked WGPU model. Invalid intermediate states are never exposed.
#[cfg(all(feature = "inference", feature = "codec"))]
#[derive(Debug)]
pub struct PreparedModel<S> {
    state: S,
    layouts: crate::runtime::WeightLayoutSet,
}

#[cfg(all(feature = "inference", feature = "codec"))]
impl PreparedModel<LayoutsSelected> {
    pub fn new(model: TextToLatentRfDiT, layouts: crate::runtime::WeightLayoutSet) -> Self {
        Self {
            state: LayoutsSelected { model },
            layouts,
        }
    }

    pub fn lock(self) -> crate::error::Result<PreparedModel<ProfileLocked>> {
        let locked = WgslInferenceOptimizedModel::from_layout_set(self.state.model, &self.layouts)?;
        Ok(PreparedModel {
            state: ProfileLocked { model: locked },
            layouts: self.layouts,
        })
    }
}

#[cfg(all(feature = "inference", feature = "codec"))]
impl PreparedModel<ProfileLocked> {
    pub fn layouts(&self) -> &crate::runtime::WeightLayoutSet {
        &self.layouts
    }

    pub(crate) fn into_inner(self) -> WgslInferenceOptimizedModel {
        self.state.model
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
    inner: InferenceOptimizedModel,
    cross_layer_adaln: Option<Box<CrossLayerAdaLnCache>>,
    generation: ModelGeneration,
}

impl From<TextToLatentRfDiT> for WgslInferenceOptimizedModel {
    fn from(mut model: TextToLatentRfDiT) -> Self {
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
    #[cfg(all(feature = "inference", feature = "codec"))]
    fn from_layout_set(
        mut model: TextToLatentRfDiT,
        layouts: &crate::runtime::WeightLayoutSet,
    ) -> crate::error::Result<Self> {
        let admit_long_b3 = layouts.contains(crate::runtime::WeightLayout::SwiGluInterleaved);
        let modules = model
            .blocks
            .iter()
            .flat_map(|block| [&block.attention_adaln, &block.mlp_adaln])
            .collect::<Vec<_>>();
        let mut cross_layer_adaln = None;
        CrossLayerAdaLnCache::prepare_v4_wgsl(&mut cross_layer_adaln, &modules);

        for block in &mut model.blocks {
            block
                .attention
                .retain_weight_layouts_wgsl(layouts, admit_long_b3)?;
            block
                .mlp
                .retain_weight_layouts_wgsl(layouts, admit_long_b3)?;
        }
        // Duration has its own fixed topology and is deliberately outside the
        // RF residency set. Preserve its established prepared WGSL path.
        if let Some(predictor) = model.duration_predictor.as_mut() {
            predictor.prepare_for_inference();
            for block in &mut predictor.token_blocks {
                block.mlp.prepare_w2_row_major_wgsl();
            }
        }

        Ok(Self {
            inner: InferenceOptimizedModel::from_prepared(model),
            cross_layer_adaln: cross_layer_adaln.map(Box::new),
            generation: ModelGeneration::fresh(),
        })
    }

    /// Commit to the production WGSL graph for arbitrary supported lengths and
    /// release only logical projection sources that graph cannot reach.
    pub(crate) fn release_production_sources(mut self) -> crate::error::Result<Self> {
        for block in &mut self.inner.inner.blocks {
            block.attention.release_production_qkv_sources_wgsl()?;
            block.mlp.release_production_expand_sources_wgsl()?;
        }
        Ok(self)
    }

    /// Commit to the batch-one, text-only, 100+-frame serving topology.
    ///
    /// Independent text CFG evaluates B2 while active and B1 afterwards. For
    /// this admitted length range both output projections select their
    /// prepared row-major layouts, allowing the remaining `wo` and `w2`
    /// learned source storage to be released without changing the route.
    pub(crate) fn lock_long_text_prepared_only(mut self) -> crate::error::Result<Self> {
        for block in &mut self.inner.inner.blocks {
            block.attention.release_production_qkv_sources_wgsl()?;
            block.attention.release_prepared_wo_source_wgsl()?;
            block.mlp.release_production_expand_sources_wgsl()?;
            block.mlp.release_prepared_w2_source_wgsl()?;
        }
        Ok(self)
    }

    /// Commit long text/design/clone B1--B3 calls to prepared output weights.
    pub(crate) fn lock_long_all_voice_prepared_only(mut self) -> crate::error::Result<Self> {
        for block in &mut self.inner.inner.blocks {
            block.attention.release_production_qkv_sources_wgsl()?;
            block.attention.enable_long_b3_prepared_wo_wgsl()?;
            block.attention.release_prepared_wo_source_wgsl()?;
            block.mlp.release_production_expand_sources_wgsl()?;
            block.mlp.enable_long_b3_prepared_w2_wgsl()?;
            block.mlp.release_prepared_w2_source_wgsl()?;
        }
        Ok(self)
    }

    /// Consume a loaded WGPU model, fuse inference weights, and select WGSL
    /// hot-path execution.
    pub fn new(model: TextToLatentRfDiT) -> Self {
        Self::from(model)
    }

    /// Consume the portable-fallback WGSL wrapper and commit it to the exact
    /// 112-frame v4-Small route.
    ///
    /// Unsupported sequence lengths are rejected by the owning inference
    /// engine before sampling. This permits source projections and the unused
    /// long-sequence layout to be released while retaining the measured
    /// packed/fused kernels. `release_source_weights == false` isolates the
    /// benefit of dropping only the unused long-sequence QKV layout.
    pub(crate) fn lock_fixed_112_profile(
        mut self,
        release_source_weights: bool,
    ) -> crate::error::Result<Self> {
        if !self
            .cross_layer_adaln
            .as_deref()
            .is_some_and(CrossLayerAdaLnCache::supports_profile_lock)
            || self.inner.inner.blocks.len() != 12
        {
            return Err(crate::error::IrodoriError::Config(
                "fixed-112 profile requires the exact 12-layer v4 AdaLN cache".to_owned(),
            ));
        }
        for block in &mut self.inner.inner.blocks {
            block
                .attention
                .lock_fixed_112_wgsl(release_source_weights)?;
            block.mlp.lock_fixed_112_wgsl(release_source_weights)?;
        }
        Ok(self)
    }

    pub fn device(&self) -> &Device {
        self.inner.device()
    }

    pub(crate) const fn model_generation(&self) -> ModelGeneration {
        self.generation
    }

    pub(crate) fn try_build_fixed_euler_cond_cache(&self) -> Option<FixedEulerCondCache> {
        FixedEulerCondCache::try_build(
            &self.inner.inner,
            self.cross_layer_adaln.as_deref(),
            self.generation,
            self.inner.device(),
        )
    }

    pub(crate) fn fixed_euler_cond_cache_matches(&self, cache: &FixedEulerCondCache) -> bool {
        cache.matches_model(self.generation, self.inner.device())
    }

    pub fn encode_conditions(
        &self,
        text_input_ids: Tensor<2, Int>,
        text_mask: Tensor<2, Bool>,
        aux_input: AuxConditionInput,
    ) -> crate::error::Result<EncodedCondition> {
        self.inner
            .inner
            .encode_conditions_wgsl(text_input_ids, text_mask, aux_input)
    }

    pub fn forward_with_cond_cached(
        &self,
        x_t: Tensor<3>,
        t: Tensor<1>,
        cond: &EncodedCondition,
        latent_mask: Option<Tensor<2, Bool>>,
        kv_caches: Option<&[CondKvCache]>,
        lat_rope: &RopeFreqs,
    ) -> Tensor<3> {
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

    /// Diagnostic-only production forward with retained per-block outputs.
    pub fn forward_with_cond_cached_debug(
        &self,
        x_t: Tensor<3>,
        t: Tensor<1>,
        cond: &EncodedCondition,
        latent_mask: Option<Tensor<2, Bool>>,
        kv_caches: Option<&[CondKvCache]>,
        lat_rope: &RopeFreqs,
    ) -> (Tensor<3>, BlockDebugOutputs) {
        self.inner.inner.forward_with_cond_cached_wgsl_debug(
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
        x_t: Tensor<3>,
        cond_embed: Tensor<3>,
        precomputed_adaln: Option<CrossLayerAdaLnModulations>,
        cond: &EncodedCondition,
        latent_mask: Option<Tensor<2, Bool>>,
        kv_caches: Option<&[CondKvCache]>,
        lat_rope: &RopeFreqs,
    ) -> Option<Tensor<3>> {
        self.inner.inner.try_forward_with_precomputed_cond_wgsl(
            self.cross_layer_adaln.as_deref(),
            x_t,
            cond_embed,
            precomputed_adaln,
            cond,
            latent_mask,
            kv_caches,
            lat_rope,
        )
    }

    pub fn precompute_latent_rope(&self, seq_lat: usize, device: &Device) -> RopeFreqs {
        self.inner.precompute_latent_rope(seq_lat, device)
    }

    pub fn build_kv_caches(
        &self,
        cond: &EncodedCondition,
        seq_lat: Option<usize>,
    ) -> Vec<CondKvCache> {
        self.inner.inner.build_kv_caches_wgsl(cond, seq_lat)
    }

    pub(crate) fn try_build_text_cfg_kv_caches(
        &self,
        cond: &EncodedCondition,
        batched_cfg: &EncodedCondition,
        seq_lat: usize,
        proof: Option<&TextOnlyCfgCacheProof>,
    ) -> Option<TextCfgKvCachePair> {
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
        cond: &EncodedCondition,
        duration_features: Tensor<2>,
        has_speaker: Tensor<1, Bool>,
        has_caption: Tensor<1, Bool>,
    ) -> crate::error::Result<Tensor<1>> {
        self.inner
            .predict_duration_log_frames(cond, duration_features, has_speaker, has_caption)
    }

    pub fn predict_duration_compact_no_aux_wgsl(
        &self,
        cond: &EncodedCondition,
        duration_features: Tensor<2>,
        has_speaker: Tensor<1, Bool>,
        has_caption: Tensor<1, Bool>,
    ) -> crate::error::Result<Tensor<1>> {
        self.inner.inner.predict_duration_compact_no_aux_wgsl(
            cond,
            duration_features,
            has_speaker,
            has_caption,
        )
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
    use burn::tensor::Distribution;
    fn tiny_cfg() -> crate::config::ModelConfig {
        crate::config::tiny_model_config()
    }

    fn device() -> Device {
        Default::default()
    }

    #[test]
    fn from_model_produces_fused_output() {
        let cfg = tiny_cfg();
        let dev = device();
        let model = TextToLatentRfDiT::new(&cfg, &dev);

        let x_t = Tensor::random(
            [1, 4, cfg.patched_latent_dim()],
            Distribution::Default,
            &dev,
        );
        let t = Tensor::from_data([0.5_f32], &dev);
        let text_ids = Tensor::<2, Int>::zeros([1, 2], &dev);
        let text_mask = Tensor::<2, Bool>::from_data(
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
        let model = TextToLatentRfDiT::new(&cfg, &dev);
        let expected = model.use_speaker_condition();
        let optimized = InferenceOptimizedModel::from(model);
        assert_eq!(optimized.use_speaker_condition(), expected);
    }

    #[test]
    fn patched_latent_dim_delegates() {
        let cfg = tiny_cfg();
        let dev = device();
        let model = TextToLatentRfDiT::new(&cfg, &dev);
        let expected = model.patched_latent_dim();
        let optimized = InferenceOptimizedModel::from(model);
        assert_eq!(optimized.patched_latent_dim(), expected);
    }

    #[test]
    fn device_returns_construction_device() {
        let cfg = tiny_cfg();
        let dev = device();
        let model = TextToLatentRfDiT::new(&cfg, &dev);
        let optimized = InferenceOptimizedModel::from(model);
        assert_eq!(*optimized.device(), dev);
    }

    #[test]
    fn fused_parity_with_speaker_conditioning() {
        let cfg = tiny_cfg();
        let dev = device();
        let model = TextToLatentRfDiT::new(&cfg, &dev);

        let (batch, seq_txt, seq_ref, seq_lat) = (1, 2, 3, 4);
        let x_t = Tensor::random(
            [batch, seq_lat, cfg.patched_latent_dim()],
            Distribution::Default,
            &dev,
        );
        let t = Tensor::from_data([0.5_f32], &dev);
        let text_ids = Tensor::<2, Int>::zeros([batch, seq_txt], &dev);
        let text_mask = Tensor::<2, Bool>::from_data(
            burn::tensor::TensorData::new(vec![true; batch * seq_txt], [batch, seq_txt]),
            &dev,
        );
        let speaker_dim = cfg.speaker_patched_latent_dim();
        let ref_latent = Tensor::<3>::ones([batch, seq_ref, speaker_dim], &dev);
        let ref_mask = Tensor::<2, Bool>::from_data(
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
        let model = TextToLatentRfDiT::new(&cfg, &dev);

        let (batch, seq_txt, seq_ref, seq_lat) = (1, 2, 3, 4);
        let x_t = Tensor::random(
            [batch, seq_lat, cfg.patched_latent_dim()],
            Distribution::Default,
            &dev,
        );
        let t = Tensor::from_data([0.5_f32], &dev);
        let text_ids = Tensor::<2, Int>::zeros([batch, seq_txt], &dev);
        let text_mask = Tensor::<2, Bool>::from_data(
            burn::tensor::TensorData::new(vec![true; batch * seq_txt], [batch, seq_txt]),
            &dev,
        );
        let speaker_dim = cfg.speaker_patched_latent_dim();
        let ref_latent = Tensor::<3>::ones([batch, seq_ref, speaker_dim], &dev);
        let ref_mask = Tensor::<2, Bool>::from_data(
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
