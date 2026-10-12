//! Core DiT model: [`TextToLatentRfDiT`], [`CondModule`], and shared
//! construction helpers.

use burn::tensor::Device;
use burn::{
    module::{Module, Param, ParamId},
    nn::{Linear, LinearConfig},
    tensor::{Bool, Int, Tensor, activation::silu},
};

use crate::{config::ModelConfig, nvtx_range};

use super::super::{
    attention::CondKvCache,
    condition::{AuxConditionInput, EncodedCondition},
    diffusion::DiffusionBlock,
    duration::{DurationPredictor, DurationPredictorInput},
    modern_bert::ModernBertConfig,
    norm::RmsNorm,
    rope::{RopeFreqs, get_timestep_embedding, precompute_rope_freqs_typed},
};

use super::aux_conditioner::ConditionFrontend;

// ---------------------------------------------------------------------------
// Zero-initialized output projection
// ---------------------------------------------------------------------------

/// Build a zero-initialized output projection for stable early training.
pub(crate) fn init_zero_out_proj(cfg: &ModelConfig, device: &Device) -> Linear {
    let mut out_proj = LinearConfig::new(cfg.model_dim, cfg.patched_latent_dim())
        .with_bias(true)
        .init(device);
    out_proj.weight = Param::initialized(
        ParamId::new(),
        Tensor::zeros([cfg.model_dim, cfg.patched_latent_dim()], device),
    );
    if let Some(ref mut b) = out_proj.bias {
        *b = Param::initialized(
            ParamId::new(),
            Tensor::zeros([cfg.patched_latent_dim()], device),
        );
    }
    out_proj
}

// ---------------------------------------------------------------------------
// Timestep conditioning module  (cond_module)
// ---------------------------------------------------------------------------
//
// Python: nn.Sequential(Linear(t_dim, D), SiLU, Linear(D, D), SiLU, Linear(D, D*3))
// State-dict keys: cond_module.0.weight, cond_module.2.weight, cond_module.4.weight
//
// We use descriptive Rust names and handle key-mapping at weight load time.

#[derive(Module, Debug)]
pub struct CondModule {
    /// Python state_dict: `cond_module.0`
    pub(crate) linear0: Linear,
    /// Python state_dict: `cond_module.2`
    pub(crate) linear1: Linear,
    /// Python state_dict: `cond_module.4`
    pub(crate) linear2: Linear,
}

impl CondModule {
    pub fn new(timestep_embed_dim: usize, model_dim: usize, device: &Device) -> Self {
        Self {
            linear0: LinearConfig::new(timestep_embed_dim, model_dim)
                .with_bias(false)
                .init(device),
            linear1: LinearConfig::new(model_dim, model_dim)
                .with_bias(false)
                .init(device),
            linear2: LinearConfig::new(model_dim, model_dim * 3)
                .with_bias(false)
                .init(device),
        }
    }

    /// `t_embed: [B, timestep_embed_dim]` → `[B, 1, model_dim * 3]`
    pub fn forward(&self, t_embed: Tensor<2>) -> Tensor<3> {
        let x = silu(self.linear0.forward(t_embed)); // [B, D]
        let x = silu(self.linear1.forward(x)); // [B, D]
        let x = self.linear2.forward(x); // [B, D*3]
        x.unsqueeze_dim::<3>(1) // [B, 1, D*3]
    }
}

// ---------------------------------------------------------------------------
// Main model
// ---------------------------------------------------------------------------

/// Per-layer intermediate outputs captured during a debug forward pass.
///
/// Produced by [`TextToLatentRfDiT::forward_with_cond_debug`].
/// Not used in production inference paths.
#[derive(Debug)]
pub struct BlockDebugOutputs {
    /// Output of `in_proj` before the first DiT block. Shape: `[B, S, D]`.
    pub after_in_proj: Tensor<3>,
    /// Output of each DiT block in order. `block_outputs[i]` has shape `[B, S, D]`.
    pub block_outputs: Vec<Tensor<3>>,
}

/// Input `x_t: [B, S, latent_dim * latent_patch_size]`.
/// Output `v_pred: same shape`.
///
/// Field names match the Python state_dict exactly for weight loading.
#[derive(Module, Debug)]
pub struct TextToLatentRfDiT {
    pub(crate) condition_frontend: ConditionFrontend,
    pub(crate) duration_predictor: Option<DurationPredictor>,
    // Diffusion backbone
    pub(crate) cond_module: CondModule,
    pub(crate) in_proj: Linear,
    pub(crate) blocks: Vec<DiffusionBlock>,
    pub(crate) out_norm: RmsNorm,
    pub(crate) out_proj: Linear,
    // Non-learnable config: stored so we don't need cfg at inference time
    pub(crate) model_dim: usize,
    pub(crate) num_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) timestep_embed_dim: usize,
    pub(crate) speaker_patch_size: usize,
    pub(crate) patched_latent_dim: usize,
}

impl TextToLatentRfDiT {
    pub fn new(cfg: &ModelConfig, device: &Device) -> Self {
        Self::try_new(cfg, device).expect("invalid TextToLatentRfDiT configuration")
    }

    /// Construct a model after validating frontend and duration configuration.
    pub fn try_new(cfg: &ModelConfig, device: &Device) -> crate::error::Result<Self> {
        Self::try_new_inner(cfg, None, device)
    }

    fn try_new_inner(
        cfg: &ModelConfig,
        pretrained_config: Option<&ModernBertConfig>,
        device: &Device,
    ) -> crate::error::Result<Self> {
        cfg.validate()?;
        let condition_frontend = match pretrained_config {
            Some(backbone_cfg) => ConditionFrontend::pretrained(cfg, backbone_cfg, device)?,
            None => ConditionFrontend::from_model_config(cfg, device)?,
        };
        let duration_predictor = cfg
            .use_duration_predictor
            .then(|| DurationPredictor::from_model_config(cfg, device))
            .transpose()?;
        let out_proj = init_zero_out_proj(cfg, device);

        let blocks = (0..cfg.num_layers)
            .map(|_| DiffusionBlock::new(cfg, device))
            .collect();

        let head_dim = cfg.head_dim();
        assert!(
            head_dim.is_multiple_of(2),
            "model head_dim must be even for RoPE, got {head_dim}"
        );

        Ok(Self {
            condition_frontend,
            duration_predictor,
            cond_module: CondModule::new(cfg.timestep_embed_dim, cfg.model_dim, device),
            in_proj: LinearConfig::new(cfg.patched_latent_dim(), cfg.model_dim)
                .with_bias(true)
                .init(device),
            blocks,
            out_norm: RmsNorm::new(cfg.model_dim, cfg.norm_eps, device),
            out_proj,
            model_dim: cfg.model_dim,
            num_heads: cfg.num_heads,
            head_dim,
            timestep_embed_dim: cfg.timestep_embed_dim,
            speaker_patch_size: cfg.speaker_patch_size.unwrap_or(1),
            patched_latent_dim: cfg.patched_latent_dim(),
        })
    }

    #[cfg(test)]
    fn try_new_with_pretrained_config(
        cfg: &ModelConfig,
        pretrained_config: &ModernBertConfig,
        device: &Device,
    ) -> crate::error::Result<Self> {
        Self::try_new_inner(cfg, Some(pretrained_config), device)
    }

    // -----------------------------------------------------------------------
    // Condition encoding
    // -----------------------------------------------------------------------

    /// Encode all conditioning inputs.
    ///
    /// **CFG dropout is applied by zeroing the relevant masks before calling this.**
    ///
    /// All-`false` text masks are safe: burn's attention backend handles them without
    /// producing NaN (verified by test). Training text-condition dropout zeroes the mask
    /// pre-encode following the Python reference. Caption dropout, by contrast, is applied
    /// *post-encode* as an extra conservative measure because the caption TextEncoder
    /// sees shorter, variable-length sequences. Speaker and caption conditioning can
    /// be omitted by passing `AuxConditionInput::None`.
    pub fn encode_conditions(
        &self,
        text_input_ids: Tensor<2, Int>,
        text_mask: Tensor<2, Bool>,
        aux_input: AuxConditionInput,
    ) -> crate::error::Result<EncodedCondition> {
        self.condition_frontend.encode(
            text_input_ids,
            text_mask,
            aux_input,
            self.speaker_patch_size,
        )
    }

    /// Predict `log1p(total_frames)` from an already encoded condition.
    ///
    /// Presence flags are explicit per batch item so callers can select the
    /// learned null speaker/caption vectors without rebuilding conditions.
    pub fn predict_duration_log_frames(
        &self,
        cond: &EncodedCondition,
        duration_features: Tensor<2>,
        has_speaker: Tensor<1, Bool>,
        has_caption: Tensor<1, Bool>,
    ) -> crate::error::Result<Tensor<1>> {
        let predictor = self.duration_predictor.as_ref().ok_or_else(|| {
            crate::error::IrodoriError::Config(
                "duration prediction requested, but this model has no duration predictor"
                    .to_string(),
            )
        })?;
        let speaker_state = cond
            .aux
            .as_ref()
            .and_then(|aux| aux.speaker())
            .map(|(state, _mask)| state.clone());
        let (caption_state, caption_mask) = cond
            .aux
            .as_ref()
            .and_then(|aux| aux.caption())
            .map(|(state, mask)| (Some(state.clone()), Some(mask.clone())))
            .unwrap_or((None, None));

        predictor.forward(DurationPredictorInput {
            text_state: cond.text_state.clone(),
            text_mask: cond.text_mask.clone(),
            aux_features: duration_features,
            speaker_state,
            has_speaker,
            caption_state,
            caption_mask,
            has_caption,
        })
    }

    /// Whether the released duration predictor is present.
    pub fn has_duration_predictor(&self) -> bool {
        self.duration_predictor.is_some()
    }

    // -----------------------------------------------------------------------
    // Forward with pre-encoded conditions
    // -----------------------------------------------------------------------

    /// Precompute RoPE frequency tables for the latent sequence.
    ///
    /// Returns a [`RopeFreqs`] that can be reused across all denoising steps
    /// within a single sampling trajectory, avoiding redundant trig recomputation.
    pub fn precompute_latent_rope(&self, seq_lat: usize, device: &Device) -> RopeFreqs {
        precompute_rope_freqs_typed(self.head_dim, seq_lat, 10000.0, device)
    }

    /// Run the diffusion backbone given pre-encoded conditions and pre-cached RoPE tables.
    ///
    /// This is the hot path: call `encode_conditions` once, build KV caches via
    /// [`Self::build_kv_caches`] and precompute the latent RoPE via
    /// [`Self::precompute_latent_rope`] once, then call this for each denoising step.
    ///
    /// - `x_t: [B, S, D_patch]` — noisy latent
    /// - `t: [B]` — timesteps in [0, 1]
    /// - `kv_caches: Option<&[CondKvCache]>` — per-layer precomputed context KVs
    /// - `lat_rope` — RoPE tables precomputed for the latent sequence length
    pub fn forward_with_cond_cached(
        &self,
        x_t: Tensor<3>,
        t: Tensor<1>,
        cond: &EncodedCondition,
        latent_mask: Option<Tensor<2, Bool>>,
        kv_caches: Option<&[CondKvCache]>,
        lat_rope: &RopeFreqs,
    ) -> Tensor<3> {
        nvtx_range!("dit_forward_with_cond", {
            let (x, _) =
                self.forward_backbone(x_t, t, cond, kv_caches, lat_rope, latent_mask, false);
            x
        })
    }

    /// Branch-free variant of [`Self::forward_with_cond_cached`].
    ///
    /// Uses pre-fused weight matrices in all blocks; panics if fusion has not
    /// been applied.
    pub(crate) fn forward_with_cond_cached_fused(
        &self,
        x_t: Tensor<3>,
        t: Tensor<1>,
        cond: &EncodedCondition,
        latent_mask: Option<Tensor<2, Bool>>,
        kv_caches: Option<&[CondKvCache]>,
        lat_rope: &RopeFreqs,
    ) -> Tensor<3> {
        nvtx_range!(
            "dit_forward_with_cond_fused",
            self.forward_backbone_fused(x_t, t, cond, kv_caches, lat_rope, latent_mask)
        )
    }

    /// Debug forward: same as [`Self::forward_with_cond_cached`] but additionally
    /// returns per-layer intermediate activations.
    ///
    /// Intended for use in validation/debugging only — not used in production paths.
    /// The per-block `.clone()` calls add non-trivial overhead on large models.
    pub fn forward_with_cond_debug(
        &self,
        x_t: Tensor<3>,
        t: Tensor<1>,
        cond: &EncodedCondition,
        lat_rope: &RopeFreqs,
    ) -> (Tensor<3>, BlockDebugOutputs) {
        self.forward_backbone(x_t, t, cond, None, lat_rope, None, true)
    }

    /// Shared implementation for production and debug forward passes.
    ///
    /// When `capture == true` each block output is cloned into the returned
    /// [`BlockDebugOutputs`].  When `capture == false` the Vec is empty and no
    /// `.clone()` is performed, so there is zero runtime overhead compared to
    /// the old inlined loop.
    #[allow(clippy::too_many_arguments)]
    fn forward_backbone(
        &self,
        x_t: Tensor<3>,
        t: Tensor<1>,
        cond: &EncodedCondition,
        kv_caches: Option<&[CondKvCache]>,
        lat_rope: &RopeFreqs,
        latent_mask: Option<Tensor<2, Bool>>,
        capture: bool,
    ) -> (Tensor<3>, BlockDebugOutputs) {
        let device = x_t.device();

        let t_embed = nvtx_range!(
            "timestep_embed",
            get_timestep_embedding(t, self.timestep_embed_dim, &device)
        );
        let cond_embed = nvtx_range!("cond_module", self.cond_module.forward(t_embed));
        let after_in_proj = nvtx_range!("in_proj", self.in_proj.forward(x_t));

        let mut x = after_in_proj.clone();
        let mut block_outputs = if capture {
            Vec::with_capacity(self.blocks.len())
        } else {
            Vec::new()
        };

        for (i, block) in self.blocks.iter().enumerate() {
            #[cfg(feature = "profile")]
            let _label = format!("dit_block_{i}");
            #[cfg(not(feature = "profile"))]
            let _label = "";
            x = nvtx_range!(
                &_label,
                block.forward(
                    x,
                    cond_embed.clone(),
                    cond,
                    lat_rope.cos.clone(),
                    lat_rope.sin.clone(),
                    kv_caches.map(|c| &c[i]),
                    latent_mask.clone(),
                )
            );
            if capture {
                block_outputs.push(x.clone());
            }
        }

        let x = nvtx_range!("out_norm", self.out_norm.forward(x));
        let v_pred = nvtx_range!("out_proj", self.out_proj.forward(x));

        let debug = BlockDebugOutputs {
            after_in_proj,
            block_outputs,
        };
        (v_pred, debug)
    }

    /// Branch-free backbone using pre-fused weight matrices.
    ///
    /// Identical to [`Self::forward_backbone`] but calls `block.forward_fused()`
    /// instead of `block.forward()`, eliminating the `if let Some(fused_w)` branch
    /// on every block at every denoising step.
    ///
    /// # Panics
    ///
    /// Panics if any block has not been prepared via
    /// [`prepare_for_inference`](Self::prepare_for_inference).
    pub(crate) fn forward_backbone_fused(
        &self,
        x_t: Tensor<3>,
        t: Tensor<1>,
        cond: &EncodedCondition,
        kv_caches: Option<&[CondKvCache]>,
        lat_rope: &RopeFreqs,
        latent_mask: Option<Tensor<2, Bool>>,
    ) -> Tensor<3> {
        let device = x_t.device();

        let t_embed = nvtx_range!(
            "timestep_embed",
            get_timestep_embedding(t, self.timestep_embed_dim, &device)
        );
        let cond_embed = nvtx_range!("cond_module", self.cond_module.forward(t_embed));
        let mut x = nvtx_range!("in_proj", self.in_proj.forward(x_t));

        for (i, block) in self.blocks.iter().enumerate() {
            #[cfg(feature = "profile")]
            let _label = format!("dit_block_{i}");
            #[cfg(not(feature = "profile"))]
            let _label = "";
            x = nvtx_range!(
                &_label,
                block.forward_fused(
                    x,
                    cond_embed.clone(),
                    cond,
                    lat_rope.cos.clone(),
                    lat_rope.sin.clone(),
                    kv_caches.map(|c| &c[i]),
                    latent_mask.clone(),
                )
            );
        }

        let x = nvtx_range!("out_norm", self.out_norm.forward(x));
        nvtx_range!("out_proj", self.out_proj.forward(x))
    }

    /// Run the diffusion backbone given pre-encoded conditions.
    ///
    /// Precomputes RoPE tables internally. For repeated calls with the same
    /// `x_t` shape (as in a sampling loop), prefer [`Self::precompute_latent_rope`]
    /// + [`Self::forward_with_cond_cached`] to avoid redundant recomputation.
    pub fn forward_with_cond(
        &self,
        x_t: Tensor<3>,
        t: Tensor<1>,
        cond: &EncodedCondition,
        latent_mask: Option<Tensor<2, Bool>>,
        kv_caches: Option<&[CondKvCache]>,
    ) -> Tensor<3> {
        let [_batch, seq_lat, _] = x_t.dims();
        let device = x_t.device();
        let lat_rope = self.precompute_latent_rope(seq_lat, &device);
        self.forward_with_cond_cached(x_t, t, cond, latent_mask, kv_caches, &lat_rope)
    }

    // -----------------------------------------------------------------------
    // Full forward (encode + diffuse in one call)
    // -----------------------------------------------------------------------

    /// Combined encode + forward.  Useful for training.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x_t: Tensor<3>,
        t: Tensor<1>,
        text_input_ids: Tensor<2, Int>,
        text_mask: Tensor<2, Bool>,
        aux_input: AuxConditionInput,
        latent_mask: Option<Tensor<2, Bool>>,
    ) -> crate::error::Result<Tensor<3>> {
        let cond = self.encode_conditions(text_input_ids, text_mask, aux_input)?;
        Ok(self.forward_with_cond(x_t, t, &cond, latent_mask, None))
    }

    // -----------------------------------------------------------------------
    // KV cache construction for fast sampling
    // -----------------------------------------------------------------------

    /// Pre-project all context K/V for each block.
    ///
    /// Call once per trajectory; reuse across denoising steps via
    /// [`Self::forward_with_cond_cached`].
    ///
    /// When `seq_lat` is provided, the joint mask `[ones(B, seq_lat) | ctx_mask]`
    /// is pre-built once per layer — avoiding repeated allocation in the hot loop.
    pub fn build_kv_caches(
        &self,
        cond: &EncodedCondition,
        seq_lat: Option<usize>,
    ) -> Vec<CondKvCache> {
        let (speaker_state, speaker_mask) = cond
            .aux
            .as_ref()
            .and_then(|aux| aux.speaker())
            .map(|(state, mask)| (Some(state.clone()), Some(mask.clone())))
            .unwrap_or((None, None));
        let (caption_state, caption_mask) = cond
            .aux
            .as_ref()
            .and_then(|aux| aux.caption())
            .map(|(state, mask)| (Some(state.clone()), Some(mask.clone())))
            .unwrap_or((None, None));
        self.blocks
            .iter()
            .map(|block| {
                let mut cache = block.attention.build_kv_cache(
                    cond.text_state.clone(),
                    cond.text_mask.clone(),
                    speaker_state.clone(),
                    speaker_mask.clone(),
                    caption_state.clone(),
                    caption_mask.clone(),
                );
                if let Some(sl) = seq_lat {
                    cache.precompute_joint_mask(sl);
                }
                cache
            })
            .collect()
    }

    /// Whether the model uses speaker (reference audio) conditioning.
    pub fn use_speaker_condition(&self) -> bool {
        self.condition_frontend.use_speaker_condition()
    }

    /// Whether the model uses caption conditioning.
    pub fn use_caption_condition(&self) -> bool {
        self.condition_frontend.use_caption_condition()
    }

    /// Dimension of the patched latent space (input/output channels per token).
    pub fn patched_latent_dim(&self) -> usize {
        self.patched_latent_dim
    }

    /// Pre-fuse weight matrices across all diffusion blocks for faster inference.
    ///
    /// Combines QKV projections (3→1 matmul) and SwiGLU w1/w3 (2→1 matmul)
    /// in each block, reducing total kernel launches by ~1440 per inference run
    /// (12 blocks × 40 steps × 3 saved launches).
    ///
    /// Must be called after final weights are loaded and device placement is
    /// complete. Fused tensors are `#[module(skip)]` and will not follow
    /// subsequent `to_device()` calls.
    ///
    /// Prefer [`InferenceOptimizedModel::from`] which enforces the fusion
    /// invariant at the type level.
    pub(crate) fn prepare_for_inference(&mut self) {
        for block in &mut self.blocks {
            block.prepare_for_inference();
        }
        if let Some(duration_predictor) = self.duration_predictor.as_mut() {
            duration_predictor.prepare_for_inference();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::aux_conditioner::{AuxConditioner, ScratchConditionFrontend};
    use super::*;
    fn tiny_cfg() -> ModelConfig {
        crate::config::tiny_model_config()
    }

    fn device() -> Device {
        Default::default()
    }

    // --- CondModule tests ---

    #[test]
    fn cond_module_output_shape() {
        let cfg = tiny_cfg();
        let dev = device();
        let cm = CondModule::new(cfg.timestep_embed_dim, cfg.model_dim, &dev);

        let t_embed = Tensor::<2>::zeros([2, cfg.timestep_embed_dim], &dev);
        let out = cm.forward(t_embed);
        // Expected: [batch=2, 1, model_dim*3]
        assert_eq!(out.dims(), [2, 1, cfg.model_dim * 3]);
    }

    #[test]
    fn cond_module_silu_activation_applied() {
        let cfg = tiny_cfg();
        let dev = device();
        let cm = CondModule::new(cfg.timestep_embed_dim, cfg.model_dim, &dev);

        // Non-zero input should produce non-zero output
        let t_embed = Tensor::<2>::ones([1, cfg.timestep_embed_dim], &dev);
        let out = cm.forward(t_embed);
        let sum: f32 = out.abs().sum().to_data().to_vec::<f32>().unwrap()[0];
        assert!(
            sum > 0.0,
            "CondModule output should be non-zero for non-zero input"
        );
    }

    // --- TextToLatentRfDiT construction ---

    #[test]
    fn model_new_speaker_mode() {
        let cfg = tiny_cfg();
        let dev = device();
        let model = TextToLatentRfDiT::new(&cfg, &dev);

        assert!(model.use_speaker_condition());
        assert!(!model.condition_frontend.is_pretrained());
        assert_eq!(model.blocks.len(), cfg.num_layers);
        assert_eq!(model.patched_latent_dim(), cfg.patched_latent_dim());
    }

    #[test]
    fn model_new_caption_mode() {
        let cfg = crate::config::tiny_caption_config();
        let dev = device();
        let model = TextToLatentRfDiT::new(&cfg, &dev);

        assert!(!model.use_speaker_condition());
        assert!(matches!(
            model.condition_frontend,
            ConditionFrontend::Scratch(ScratchConditionFrontend {
                aux_conditioner: Some(AuxConditioner::Caption(_)),
                ..
            })
        ));
    }

    #[test]
    fn model_new_speaker_and_caption_mode() {
        let mut cfg = tiny_cfg();
        cfg.use_speaker_condition = Some(true);
        cfg.use_caption_condition = true;
        cfg.caption_vocab_size = Some(32);
        cfg.caption_dim = Some(16);
        cfg.caption_layers = Some(1);
        cfg.caption_heads = Some(2);
        let dev = device();
        let model = TextToLatentRfDiT::new(&cfg, &dev);

        assert!(model.use_speaker_condition());
        assert!(model.use_caption_condition());
        assert!(matches!(
            model.condition_frontend,
            ConditionFrontend::Scratch(ScratchConditionFrontend {
                aux_conditioner: Some(AuxConditioner::Both(_)),
                ..
            })
        ));
    }

    fn tiny_pretrained_config(with_duration: bool) -> (ModelConfig, ModernBertConfig) {
        use super::super::super::modern_bert::ModernBertLayerType;

        let mut cfg = tiny_cfg();
        cfg.text_encoder_type = "pretrained".to_string();
        cfg.pretrained_projector_type = "residual_mlp".to_string();
        cfg.text_vocab_size = 32;
        cfg.text_dim = 8;
        cfg.use_speaker_condition = Some(true);
        cfg.use_caption_condition = true;
        cfg.caption_vocab_size = Some(32);
        cfg.caption_dim = Some(8);
        cfg.caption_layers = Some(1);
        cfg.caption_heads = Some(2);
        cfg.use_duration_predictor = with_duration;
        cfg.duration_aux_dim = 4;
        cfg.duration_hidden_dim = 16;
        cfg.duration_layers = 1;
        cfg.duration_attention_heads = 2;
        cfg.duration_architecture =
            super::super::super::duration::V4_DURATION_ARCHITECTURE.to_string();
        cfg.duration_token_init_frames = 2.0;

        let backbone_cfg = ModernBertConfig {
            vocab_size: 32,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            max_position_embeddings: 32,
            norm_eps: 1e-5,
            pad_token_id: 3,
            local_attention: 4,
            full_rope_theta: 160_000.0,
            sliding_rope_theta: 10_000.0,
            layer_types: vec![ModernBertLayerType::Full, ModernBertLayerType::Sliding],
        };
        (cfg, backbone_cfg)
    }

    #[test]
    fn pretrained_frontend_allocates_shared_backbone_without_scratch_encoders() {
        let (cfg, backbone_cfg) = tiny_pretrained_config(false);
        let dev = device();
        let model =
            TextToLatentRfDiT::try_new_with_pretrained_config(&cfg, &backbone_cfg, &dev).unwrap();

        assert!(model.condition_frontend.is_pretrained());
        assert!(model.use_speaker_condition());
        assert!(model.use_caption_condition());
        match &model.condition_frontend {
            ConditionFrontend::Pretrained(frontend) => {
                assert!(frontend.speaker.is_some());
                assert!(frontend.caption_norm.is_some());
            }
            ConditionFrontend::Scratch(_) => panic!("expected pretrained frontend"),
        }
    }

    #[test]
    fn pretrained_frontend_encodes_text_speaker_and_caption_together() {
        let (cfg, backbone_cfg) = tiny_pretrained_config(false);
        let dev = device();
        let model =
            TextToLatentRfDiT::try_new_with_pretrained_config(&cfg, &backbone_cfg, &dev).unwrap();
        let encoded = model
            .encode_conditions(
                Tensor::zeros([1, 4], &dev),
                Tensor::ones([1, 4], &dev),
                AuxConditionInput::Both {
                    ref_latent: Tensor::zeros([1, 3, cfg.speaker_patched_latent_dim()], &dev),
                    ref_mask: Tensor::ones([1, 3], &dev),
                    caption_ids: Tensor::ones([1, 2], &dev),
                    caption_mask: Tensor::ones([1, 2], &dev),
                },
            )
            .unwrap();

        assert_eq!(encoded.text_state.dims(), [1, 4, cfg.text_dim]);
        let aux = encoded.aux.unwrap();
        assert_eq!(
            aux.speaker().unwrap().0.dims(),
            [1, 4, cfg.speaker_dim.unwrap()]
        );
        assert_eq!(aux.caption().unwrap().0.dims(), [1, 2, cfg.caption_dim()]);

        let speaker_only = model
            .encode_conditions(
                Tensor::zeros([1, 4], &dev),
                Tensor::ones([1, 4], &dev),
                AuxConditionInput::Speaker {
                    ref_latent: Tensor::zeros([1, 3, cfg.speaker_patched_latent_dim()], &dev),
                    ref_mask: Tensor::ones([1, 3], &dev),
                },
            )
            .unwrap()
            .aux
            .unwrap();
        assert!(speaker_only.speaker().is_some());
        assert!(speaker_only.caption().is_none());

        let caption_only = model
            .encode_conditions(
                Tensor::zeros([1, 4], &dev),
                Tensor::ones([1, 4], &dev),
                AuxConditionInput::Caption {
                    ids: Tensor::ones([1, 2], &dev),
                    mask: Tensor::ones([1, 2], &dev),
                },
            )
            .unwrap()
            .aux
            .unwrap();
        assert!(caption_only.speaker().is_none());
        assert!(caption_only.caption().is_some());

        let unconditional = model
            .encode_conditions(
                Tensor::zeros([1, 4], &dev),
                Tensor::ones([1, 4], &dev),
                AuxConditionInput::None,
            )
            .unwrap();
        assert!(unconditional.aux.is_none());

        let output = model
            .forward(
                Tensor::zeros([1, 3, cfg.patched_latent_dim()], &dev),
                Tensor::zeros([1], &dev),
                Tensor::zeros([1, 4], &dev),
                Tensor::ones([1, 4], &dev),
                AuxConditionInput::Both {
                    ref_latent: Tensor::zeros([1, 3, cfg.speaker_patched_latent_dim()], &dev),
                    ref_mask: Tensor::ones([1, 3], &dev),
                    caption_ids: Tensor::ones([1, 2], &dev),
                    caption_mask: Tensor::ones([1, 2], &dev),
                },
                None,
            )
            .unwrap();
        assert_eq!(output.dims(), [1, 3, cfg.patched_latent_dim()]);
    }

    #[test]
    fn duration_api_consumes_encoded_both_context_and_presence_flags() {
        let (cfg, backbone_cfg) = tiny_pretrained_config(true);
        let dev = device();
        let model =
            TextToLatentRfDiT::try_new_with_pretrained_config(&cfg, &backbone_cfg, &dev).unwrap();
        let encoded = model
            .encode_conditions(
                Tensor::zeros([1, 4], &dev),
                Tensor::ones([1, 4], &dev),
                AuxConditionInput::Both {
                    ref_latent: Tensor::zeros([1, 3, cfg.speaker_patched_latent_dim()], &dev),
                    ref_mask: Tensor::ones([1, 3], &dev),
                    caption_ids: Tensor::ones([1, 2], &dev),
                    caption_mask: Tensor::ones([1, 2], &dev),
                },
            )
            .unwrap();

        assert!(model.has_duration_predictor());
        let prediction = model
            .predict_duration_log_frames(
                &encoded,
                Tensor::zeros([1, cfg.duration_aux_dim], &dev),
                Tensor::ones([1], &dev),
                Tensor::ones([1], &dev),
            )
            .unwrap();
        assert_eq!(prediction.dims(), [1]);
        let actual: f32 = prediction.into_scalar();
        let expected = (1.0_f32 + 4.0 * cfg.duration_token_init_frames as f32).ln();
        assert!((actual - expected).abs() < 1e-5);
    }

    #[test]
    fn out_proj_zero_initialized() {
        let cfg = tiny_cfg();
        let dev = device();
        let model = TextToLatentRfDiT::new(&cfg, &dev);

        let w_sum: f32 = model
            .out_proj
            .weight
            .val()
            .abs()
            .sum()
            .to_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert_eq!(w_sum, 0.0, "out_proj weight should be zero-initialized");

        if let Some(ref b) = model.out_proj.bias {
            let b_sum: f32 = b.val().abs().sum().to_data().to_vec::<f32>().unwrap()[0];
            assert_eq!(b_sum, 0.0, "out_proj bias should be zero-initialized");
        }
    }

    #[test]
    fn out_proj_weight_shape_row_layout() {
        let cfg = tiny_cfg();
        let dev = device();
        let model = TextToLatentRfDiT::new(&cfg, &dev);

        // Row layout: [d_input=model_dim, d_output=patched_latent_dim]
        let dims = model.out_proj.weight.val().dims();
        assert_eq!(
            dims,
            [cfg.model_dim, cfg.patched_latent_dim()],
            "out_proj weight should be [model_dim, patched_latent_dim] in Row layout"
        );
    }

    // --- Full forward pass shape test ---

    #[test]
    fn forward_output_shape() {
        let cfg = tiny_cfg();
        let dev = device();
        let model = TextToLatentRfDiT::new(&cfg, &dev);

        let batch = 2;
        let seq_lat = 4;
        let seq_txt = 6;
        let latent_dim = cfg.patched_latent_dim();

        let x_t = Tensor::<3>::zeros([batch, seq_lat, latent_dim], &dev);
        let t = Tensor::<1>::zeros([batch], &dev);
        let text_ids = Tensor::<2, Int>::zeros([batch, seq_txt], &dev);
        let text_mask = Tensor::<2, Bool>::ones([batch, seq_txt], &dev);

        let speaker_len = 3;
        let speaker_input_dim = cfg.speaker_patched_latent_dim();
        let speaker_latent = Tensor::<3>::zeros([batch, speaker_len, speaker_input_dim], &dev);
        let speaker_mask = Tensor::<2, Bool>::ones([batch, speaker_len], &dev);
        let aux = AuxConditionInput::Speaker {
            ref_latent: speaker_latent,
            ref_mask: speaker_mask,
        };

        let v_pred = model
            .forward(x_t, t, text_ids, text_mask, aux, None)
            .unwrap();
        assert_eq!(
            v_pred.dims(),
            [batch, seq_lat, latent_dim],
            "forward should preserve input shape"
        );
    }

    #[test]
    fn forward_with_cond_cached_matches_forward() {
        let cfg = tiny_cfg();
        let dev = device();
        let model = TextToLatentRfDiT::new(&cfg, &dev);

        let batch = 1;
        let seq_lat = 3;
        let seq_txt = 4;
        let latent_dim = cfg.patched_latent_dim();
        let speaker_input_dim = cfg.speaker_patched_latent_dim();

        let x_t = Tensor::<3>::ones([batch, seq_lat, latent_dim], &dev) * 0.1;
        let t = Tensor::<1>::from_data([0.5f32], &dev);
        let text_ids = Tensor::<2, Int>::zeros([batch, seq_txt], &dev);
        let text_mask = Tensor::<2, Bool>::ones([batch, seq_txt], &dev);
        let speaker_latent = Tensor::<3>::ones([batch, 2, speaker_input_dim], &dev) * 0.3;
        let speaker_mask = Tensor::<2, Bool>::ones([batch, 2], &dev);

        // Full forward
        let aux1 = AuxConditionInput::Speaker {
            ref_latent: speaker_latent.clone(),
            ref_mask: speaker_mask.clone(),
        };
        let out_full = model
            .forward(
                x_t.clone(),
                t.clone(),
                text_ids.clone(),
                text_mask.clone(),
                aux1,
                None,
            )
            .unwrap();

        // Decomposed: encode + cached forward
        let aux2 = AuxConditionInput::Speaker {
            ref_latent: speaker_latent,
            ref_mask: speaker_mask,
        };
        let cond = model.encode_conditions(text_ids, text_mask, aux2).unwrap();
        let out_cached = model.forward_with_cond(x_t, t, &cond, None, None);

        let diff: f32 = (out_full - out_cached)
            .abs()
            .sum()
            .to_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(
            diff < 1e-6,
            "forward and forward_with_cond should produce identical output, got diff={diff}"
        );
    }

    // --- init_zero_out_proj ---

    #[test]
    fn init_zero_out_proj_is_zero() {
        let cfg = tiny_cfg();
        let dev = device();
        let proj = init_zero_out_proj(&cfg, &dev);

        let w_sum: f32 = proj
            .weight
            .val()
            .abs()
            .sum()
            .to_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert_eq!(w_sum, 0.0, "out_proj weight should be zero");

        if let Some(ref b) = proj.bias {
            let b_sum: f32 = b.val().abs().sum().to_data().to_vec::<f32>().unwrap()[0];
            assert_eq!(b_sum, 0.0, "out_proj bias should be zero");
        }
    }

    #[test]
    fn init_zero_out_proj_row_layout_shape() {
        let cfg = tiny_cfg();
        let dev = device();
        let proj = init_zero_out_proj(&cfg, &dev);

        assert_eq!(
            proj.weight.val().dims(),
            [cfg.model_dim, cfg.patched_latent_dim()],
            "Row layout: [d_input, d_output]"
        );
    }
}
