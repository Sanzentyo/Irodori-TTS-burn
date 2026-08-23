//! Backend-specific production execution for the fused WGSL kernels.

use burn::tensor::Device;
mod condition;

use burn::tensor::{Bool, Tensor, TensorData};

use crate::nvtx_range;

use super::{
    adaln_cross_layer::{CrossLayerAdaLnCache, CrossLayerAdaLnModulations},
    attention::{CondKvCache, TextCfgKvCachePair, WgslJointMask},
    condition::EncodedCondition,
    dit::{BlockDebugOutputs, TextToLatentRfDiT},
    duration::DurationPredictorInput,
    rope::{RopeFreqs, get_timestep_embedding},
    timestep_condition::has_v4_cond_embed_layout,
};

const TEXT_CFG_LAYERS: usize = 12;
const TEXT_CFG_TEXT_DIM: usize = 512;
const TEXT_CFG_MODEL_DIM: usize = 1_280;

/// Host-side semantic proof required before deriving a B2 text CFG cache.
///
/// Physical tensor checks alone cannot establish that the second CFG row was
/// constructed from `EncodedCondition::zeros_like`, that every conditioned
/// text token is valid, or that auxiliary masks have no active context. The
/// sampler creates this opaque token only on its exact, conditioned-first
/// text-only path; absence rejects the optimization.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TextOnlyCfgCacheProof {
    _private: (),
}

impl TextOnlyCfgCacheProof {
    pub(crate) const fn try_new(
        exact_fixed_request: bool,
        text_only: bool,
        conditioned_first: bool,
        conditioned_mask_all_true: bool,
        unconditional_from_zeros_like: bool,
        no_active_aux: bool,
    ) -> Option<Self> {
        if exact_fixed_request
            && text_only
            && conditioned_first
            && conditioned_mask_all_true
            && unconditional_from_zeros_like
            && no_active_aux
        {
            Some(Self { _private: () })
        } else {
            None
        }
    }
}

impl TextToLatentRfDiT {
    pub(crate) fn predict_duration_compact_no_aux_wgsl(
        &self,
        cond: &EncodedCondition,
        duration_features: Tensor<2>,
        has_speaker: Tensor<1, Bool>,
        has_caption: Tensor<1, Bool>,
    ) -> crate::error::Result<Tensor<1>> {
        if cond.aux.is_some() {
            return Err(crate::error::IrodoriError::Config(
                "compact duration WGSL path requires no auxiliary condition".to_string(),
            ));
        }
        let predictor = self.duration_predictor.as_ref().ok_or_else(|| {
            crate::error::IrodoriError::Config(
                "duration prediction requested, but this model has no duration predictor"
                    .to_string(),
            )
        })?;
        predictor.forward_compact_no_aux_wgsl(DurationPredictorInput {
            text_state: cond.text_state.clone(),
            text_mask: cond.text_mask.clone(),
            aux_features: duration_features,
            speaker_state: None,
            has_speaker,
            caption_state: None,
            caption_mask: None,
            has_caption,
        })
    }

    /// Prepare the additional record-skipped bindings used only by the WGSL
    /// JointAttention materialization policy.
    pub(crate) fn prepare_attention_materialization_wgsl(&mut self) {
        for block in &mut self.blocks {
            block.attention.prepare_qk_norm_weight_wgsl();
            block.attention.prepare_long_projection_wgsl();
        }
    }

    /// Prepare the measured row-major SwiGLU output weights used exclusively
    /// by the production WGSL execution policy.
    pub(crate) fn prepare_swiglu_w2_row_major_wgsl(&mut self) {
        for block in &mut self.blocks {
            block.mlp.prepare_w2_row_major_wgsl();
        }
        if let Some(predictor) = self.duration_predictor.as_mut() {
            for block in &mut predictor.token_blocks {
                block.mlp.prepare_w2_row_major_wgsl();
            }
        }
    }

    /// Build ordinary conditional K/V caches, then add the exact-shape packed
    /// K/V binding once before the denoising loop.
    pub(crate) fn build_kv_caches_wgsl(
        &self,
        cond: &EncodedCondition,
        seq_lat: Option<usize>,
    ) -> Vec<CondKvCache> {
        let mut caches = self.build_kv_caches(cond, seq_lat);
        for cache in &mut caches {
            cache.prepare_packed_ctx_kv_wgsl();
        }
        caches
    }

    /// Build the exact text-only B1 cache once and derive the B2 Independent
    /// CFG cache without repeating the 24 text K/V projections.
    ///
    /// Every semantic, model, tensor, layout, device, and resource mismatch
    /// returns `None`; the sampler then executes its unchanged pair of ordinary
    /// cache builds. The B1 all-valid mask is omitted. All 12 B2 layers share
    /// one CubeCL-native masked-out mask.
    pub(crate) fn try_build_text_cfg_kv_caches_wgsl(
        &self,
        cond: &EncodedCondition,
        batched_cfg: &EncodedCondition,
        seq_lat: usize,
        device: &Device,
        proof: Option<&TextOnlyCfgCacheProof>,
    ) -> Option<TextCfgKvCachePair> {
        use crate::kernels::text_cfg_kv_derive::{
            CFG_BATCH, CONTEXT_LEN, HEAD_DIM, NUM_HEADS, PLANES, supports_text_cfg_kv_derive,
            try_derive_text_cfg_kv_wgsl,
        };

        let _proof = proof?;
        if self.model_dim != TEXT_CFG_MODEL_DIM
            || self.num_heads != NUM_HEADS
            || self.head_dim != HEAD_DIM
            || self.blocks.len() != TEXT_CFG_LAYERS
            || cond.text_state.dims() != [1, CONTEXT_LEN, TEXT_CFG_TEXT_DIM]
            || cond.text_mask.dims() != [1, CONTEXT_LEN]
            || batched_cfg.text_state.dims() != [CFG_BATCH, CONTEXT_LEN, TEXT_CFG_TEXT_DIM]
            || batched_cfg.text_mask.dims() != [CFG_BATCH, CONTEXT_LEN]
            || cond.aux.is_some()
            || batched_cfg.aux.is_some()
            || cond.text_state.device() != *device
            || cond.text_mask.device() != *device
            || batched_cfg.text_state.device() != *device
            || batched_cfg.text_mask.device() != *device
        {
            return None;
        }
        for block in &self.blocks {
            for projection in [&block.attention.wk_text, &block.attention.wv_text] {
                let weight = projection.weight.val();
                if projection.bias.is_some()
                    || weight.dims() != [TEXT_CFG_TEXT_DIM, TEXT_CFG_MODEL_DIM]
                    || weight.device() != *device
                {
                    return None;
                }
            }
        }

        // `None` deliberately suppresses ordinary joint-mask construction. The
        // exact WGPU policies are attached only after every layer passes its
        // physical selector and every B2 cache has been derived successfully.
        let mut conditional = self.build_kv_caches_wgsl(cond, None);
        if conditional.len() != TEXT_CFG_LAYERS
            || conditional.iter().any(|cache| {
                let Some(packed) = cache.packed_ctx_kv_wgsl.as_ref() else {
                    return true;
                };
                let primitive = packed
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend");
                !supports_text_cfg_kv_derive(&primitive, &primitive.device)
            })
        {
            return None;
        }

        let total_kv_len = seq_lat + CONTEXT_LEN;
        let native_mask_values = [
            vec![false; total_kv_len],
            [vec![false; seq_lat], vec![true; CONTEXT_LEN]].concat(),
        ]
        .concat();
        let native_mask = Tensor::from_data(
            TensorData::new(native_mask_values, [CFG_BATCH, total_kv_len]),
            device,
        );
        let conditional_attend_mask = Tensor::ones([1, total_kv_len], device);
        let derived_attend_mask = native_mask.clone().bool_not().float();

        let derived = conditional
            .iter()
            .map(|cache| {
                let source = cache.packed_ctx_kv_wgsl.as_ref()?;
                let source = source
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend");
                let raw_device = source.device.clone();
                let output = try_derive_text_cfg_kv_wgsl(source, &raw_device)?;
                let packed = Tensor::<5>::from_primitive::<crate::WgpuRaw>(output);
                let ctx_k = packed
                    .clone()
                    .slice([
                        0..1,
                        0..CFG_BATCH,
                        0..CONTEXT_LEN,
                        0..NUM_HEADS,
                        0..HEAD_DIM,
                    ])
                    .reshape([CFG_BATCH, CONTEXT_LEN, NUM_HEADS, HEAD_DIM]);
                let ctx_v = packed
                    .clone()
                    .slice([
                        1..PLANES,
                        0..CFG_BATCH,
                        0..CONTEXT_LEN,
                        0..NUM_HEADS,
                        0..HEAD_DIM,
                    ])
                    .reshape([CFG_BATCH, CONTEXT_LEN, NUM_HEADS, HEAD_DIM]);
                Some(CondKvCache {
                    ctx_k,
                    ctx_v,
                    ctx_mask: batched_cfg.text_mask.clone(),
                    joint_mask: None,
                    speaker_range: None,
                    packed_ctx_kv_wgsl: Some(packed),
                    joint_mask_wgsl: Some(WgslJointMask::MaskedOut(native_mask.clone())),
                    joint_attend_mask_wgsl: Some(derived_attend_mask.clone()),
                })
            })
            .collect::<Option<Vec<_>>>()?;

        for cache in &mut conditional {
            cache.joint_mask = None;
            cache.joint_mask_wgsl = Some(WgslJointMask::AllValid);
            cache.joint_attend_mask_wgsl = Some(conditional_attend_mask.clone());
        }
        Some((conditional, derived))
    }

    /// Run the DiT hot path with the production-connected WGSL fusions.
    ///
    /// The execution policy is explicit at the type boundary: callers must
    /// opt into the WGPU-only wrapper, while portable backends keep the Burn
    /// implementation. Tuned Burn/CubeCL matmuls remain the default; the
    /// measured native SDPA selector is limited to S13/S25/S50, while longer
    /// sequences retain Burn attention.
    #[allow(clippy::too_many_arguments)] // Mirrors the typed DiT forward contract plus its cache.
    pub(crate) fn forward_with_cond_cached_wgsl(
        &self,
        adaln_cache: Option<&CrossLayerAdaLnCache>,
        x_t: Tensor<3>,
        t: Tensor<1>,
        cond: &EncodedCondition,
        latent_mask: Option<Tensor<2, Bool>>,
        kv_caches: Option<&[CondKvCache]>,
        lat_rope: &RopeFreqs,
    ) -> Tensor<3> {
        nvtx_range!("dit_forward_wgsl", {
            let device = x_t.device();
            let t_embed = nvtx_range!(
                "timestep_embed",
                get_timestep_embedding(t, self.timestep_embed_dim, &device)
            );
            let cond_embed = nvtx_range!("cond_module", self.cond_module.forward(t_embed));
            self.forward_with_cond_embed_wgsl(
                adaln_cache,
                x_t,
                cond_embed,
                None,
                cond,
                latent_mask,
                kv_caches,
                lat_rope,
            )
        })
    }

    /// Diagnostic-only production forward retaining the input projection and
    /// every DiT block output. The arithmetic and route selection match
    /// [`Self::forward_with_cond_cached_wgsl`]; retained handles make the call
    /// unsuitable for latency measurement.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_with_cond_cached_wgsl_debug(
        &self,
        adaln_cache: Option<&CrossLayerAdaLnCache>,
        x_t: Tensor<3>,
        t: Tensor<1>,
        cond: &EncodedCondition,
        latent_mask: Option<Tensor<2, Bool>>,
        kv_caches: Option<&[CondKvCache]>,
        lat_rope: &RopeFreqs,
    ) -> (Tensor<3>, BlockDebugOutputs) {
        let device = x_t.device();
        let t_embed = get_timestep_embedding(t, self.timestep_embed_dim, &device);
        let cond_embed = self.cond_module.forward(t_embed);
        let cross_layer_adaln =
            adaln_cache.and_then(|cache| cache.precompute_v4_wgsl(cond_embed.clone()));
        let after_in_proj = self.in_proj.forward(x_t);
        let mut x = after_in_proj.clone();
        let mut block_outputs = Vec::with_capacity(self.blocks.len());
        for (index, block) in self.blocks.iter().enumerate() {
            x = block.forward_fused_wgsl(
                index,
                x,
                cond_embed.clone(),
                cross_layer_adaln
                    .as_ref()
                    .and_then(|modulations| modulations.block(index)),
                cond,
                lat_rope.cos.clone(),
                lat_rope.sin.clone(),
                kv_caches.map(|caches| &caches[index]),
                latent_mask.clone(),
            );
            block_outputs.push(x.clone());
        }
        let x = self.out_norm.forward_wgsl(x);
        let output = self.out_proj.forward(x);
        (
            output,
            BlockDebugOutputs {
                after_in_proj,
                block_outputs,
            },
        )
    }

    /// Consume an engine-owned timestep condition only when its exact physical
    /// contract still matches this forward. A rejected value submits no work;
    /// the sampler then recomputes from the raw timestep through the method
    /// above.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_forward_with_precomputed_cond_wgsl(
        &self,
        adaln_cache: Option<&CrossLayerAdaLnCache>,
        x_t: Tensor<3>,
        cond_embed: Tensor<3>,
        precomputed_adaln: Option<CrossLayerAdaLnModulations>,
        cond: &EncodedCondition,
        latent_mask: Option<Tensor<2, Bool>>,
        kv_caches: Option<&[CondKvCache]>,
        lat_rope: &RopeFreqs,
    ) -> Option<Tensor<3>> {
        let batch = x_t.dims()[0];
        let device = x_t.device();
        if !has_v4_cond_embed_layout(&cond_embed, batch, &device) {
            return None;
        }
        Some(nvtx_range!("dit_forward_wgsl", {
            self.forward_with_cond_embed_wgsl(
                adaln_cache,
                x_t,
                cond_embed,
                precomputed_adaln,
                cond,
                latent_mask,
                kv_caches,
                lat_rope,
            )
        }))
    }

    /// Profile candidate for Independent CFG: project one physical latent row
    /// once, then broadcast the projected activation to the B2/B3 condition
    /// topology. The ordinary path first duplicates the 32-channel latent and
    /// redundantly evaluates the condition-independent input projection.
    #[cfg(feature = "profile")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_forward_with_broadcast_precomputed_cond_wgsl(
        &self,
        adaln_cache: Option<&CrossLayerAdaLnCache>,
        x_t: Tensor<3>,
        broadcast_batch: usize,
        cond_embed: Tensor<3>,
        precomputed_adaln: Option<CrossLayerAdaLnModulations>,
        cond: &EncodedCondition,
        latent_mask: Option<Tensor<2, Bool>>,
        kv_caches: Option<&[CondKvCache]>,
        lat_rope: &RopeFreqs,
    ) -> Option<Tensor<3>> {
        let [source_batch, sequence, _] = x_t.dims();
        let device = x_t.device();
        if source_batch != 1
            || !(2..=3).contains(&broadcast_batch)
            || latent_mask.is_some()
            || !has_v4_cond_embed_layout(&cond_embed, broadcast_batch, &device)
        {
            return None;
        }
        let cross_layer_adaln = precomputed_adaln
            .or_else(|| adaln_cache.and_then(|cache| cache.precompute_v4_wgsl(cond_embed.clone())));
        let input = x_t.try_into_primitive::<crate::WgpuRaw>().ok()?;
        let weight = self
            .in_proj
            .weight
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .ok()?;
        let bias = self
            .in_proj
            .bias
            .as_ref()?
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .ok()?;
        let projected =
            crate::kernels::dit_projection_t64::try_dit_input_projection_broadcast_wgsl(
                input,
                weight,
                bias,
                broadcast_batch,
            )
            .map(Tensor::from_primitive::<crate::WgpuRaw>)?;
        if projected.dims() != [broadcast_batch, sequence, self.model_dim] {
            return None;
        }
        Some(nvtx_range!(
            "dit_forward_wgsl_broadcast_in_proj",
            self.forward_projected_with_cond_embed_wgsl(
                projected,
                cond_embed,
                cross_layer_adaln,
                cond,
                None,
                kv_caches,
                lat_rope,
            )
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_cond_embed_wgsl(
        &self,
        adaln_cache: Option<&CrossLayerAdaLnCache>,
        x_t: Tensor<3>,
        cond_embed: Tensor<3>,
        precomputed_adaln: Option<CrossLayerAdaLnModulations>,
        cond: &EncodedCondition,
        latent_mask: Option<Tensor<2, Bool>>,
        kv_caches: Option<&[CondKvCache]>,
        lat_rope: &RopeFreqs,
    ) -> Tensor<3> {
        let cross_layer_adaln = precomputed_adaln.or_else(|| {
            nvtx_range!(
                "adaln_cross_layer_precompute",
                adaln_cache.and_then(|cache| cache.precompute_v4_wgsl(cond_embed.clone()))
            )
        });
        let x = nvtx_range!("in_proj", self.in_proj.forward(x_t));
        self.forward_projected_with_cond_embed_wgsl(
            x,
            cond_embed,
            cross_layer_adaln,
            cond,
            latent_mask,
            kv_caches,
            lat_rope,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_projected_with_cond_embed_wgsl(
        &self,
        mut x: Tensor<3>,
        cond_embed: Tensor<3>,
        cross_layer_adaln: Option<CrossLayerAdaLnModulations>,
        cond: &EncodedCondition,
        latent_mask: Option<Tensor<2, Bool>>,
        kv_caches: Option<&[CondKvCache]>,
        lat_rope: &RopeFreqs,
    ) -> Tensor<3> {
        for (index, block) in self.blocks.iter().enumerate() {
            #[cfg(feature = "profile")]
            let _label = format!("dit_block_wgsl_{index}");
            #[cfg(not(feature = "profile"))]
            let _label = "";
            x = nvtx_range!(
                &_label,
                block.forward_fused_wgsl(
                    index,
                    x,
                    cond_embed.clone(),
                    cross_layer_adaln
                        .as_ref()
                        .and_then(|modulations| modulations.block(index)),
                    cond,
                    lat_rope.cos.clone(),
                    lat_rope.sin.clone(),
                    kv_caches.map(|caches| &caches[index]),
                    latent_mask.clone(),
                )
            );
        }

        let x = nvtx_range!("out_norm_wgsl", self.out_norm.forward_wgsl(x));
        nvtx_range!("out_proj", self.out_proj.forward(x))
    }
}

#[cfg(test)]
mod tests {
    use super::TextOnlyCfgCacheProof;

    #[test]
    fn text_cfg_host_proof_is_fail_closed() {
        assert!(TextOnlyCfgCacheProof::try_new(true, true, true, true, true, true).is_some());
        for rejected in [
            (false, true, true, true, true, true),
            (true, false, true, true, true, true),
            (true, true, false, true, true, true),
            (true, true, true, false, true, true),
            (true, true, true, true, false, true),
            (true, true, true, true, true, false),
        ] {
            assert!(
                TextOnlyCfgCacheProof::try_new(
                    rejected.0, rejected.1, rejected.2, rejected.3, rejected.4, rejected.5,
                )
                .is_none()
            );
        }
    }
}
