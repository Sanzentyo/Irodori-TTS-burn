use burn::{
    module::{Module, Param, ParamId},
    nn::{Linear, LinearConfig},
    tensor::{Bool, Int, Tensor, activation::silu, backend::Backend},
};

use crate::{config::ModelConfig, nvtx_range};

use super::{
    attention::CondKvCache,
    condition::{AuxConditionInput, AuxConditionState, EncodedCondition},
    diffusion::DiffusionBlock,
    norm::RmsNorm,
    rope::{RopeFreqs, get_timestep_embedding, precompute_rope_freqs_typed},
    speaker_encoder::{ReferenceLatentEncoder, patch_sequence_with_mask},
    text_encoder::{TextEncoder, TextEncoderSpec},
};

// ---------------------------------------------------------------------------
// Auxiliary conditioner — speaker XOR caption, weight-bearing module
// ---------------------------------------------------------------------------

/// Encoder + normalization for reference-audio (speaker) conditioning.
#[derive(Module, Debug)]
pub struct SpeakerConditioner<B: Backend> {
    pub(crate) encoder: ReferenceLatentEncoder<B>,
    pub(crate) norm: RmsNorm<B>,
}

/// Encoder + normalization for text-caption conditioning.
#[derive(Module, Debug)]
pub struct CaptionConditioner<B: Backend> {
    pub(crate) encoder: TextEncoder<B>,
    pub(crate) norm: RmsNorm<B>,
}

/// Auxiliary conditioning module: exactly one of speaker or caption.
///
/// Wrapped in `Option` in `TextToLatentRfDiT` so models without any auxiliary
/// conditioning are represented as `None` rather than a phantom unit variant.
#[derive(Module, Debug)]
pub enum AuxConditioner<B: Backend> {
    /// Reference-audio (speaker) conditioning path.
    Speaker(SpeakerConditioner<B>),
    /// Text-caption conditioning path.
    Caption(CaptionConditioner<B>),
}

impl<B: Backend> AuxConditioner<B> {
    /// Encode auxiliary input tensors into a runtime `AuxConditionState`.
    ///
    /// The `speaker_patch_size` argument is only used for the `Speaker` variant;
    /// it is ignored when `self` is `Caption`.
    pub(crate) fn encode(
        &self,
        input: AuxConditionInput<B>,
        speaker_patch_size: usize,
    ) -> Option<AuxConditionState<B>> {
        match (self, input) {
            (
                Self::Speaker(sp),
                AuxConditionInput::Speaker {
                    ref_latent,
                    ref_mask,
                },
            ) => {
                let (patched_latent, patched_mask) =
                    patch_sequence_with_mask(ref_latent, ref_mask, speaker_patch_size);
                let sp_state = sp.encoder.forward(patched_latent, patched_mask.clone());
                let sp_state = sp.norm.forward(sp_state);
                let (sp_state, sp_mask) = prepend_masked_mean_token(sp_state, patched_mask);
                Some(AuxConditionState::Speaker {
                    state: sp_state,
                    mask: sp_mask,
                })
            }
            (Self::Caption(cap), AuxConditionInput::Caption { ids, mask }) => {
                let cap_state = cap.encoder.forward(ids, mask.clone());
                let cap_state = cap.norm.forward(cap_state);
                Some(AuxConditionState::Caption {
                    state: cap_state,
                    mask,
                })
            }
            // Mismatched mode or no input → no aux conditioning for this pass.
            _ => None,
        }
    }
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
pub struct CondModule<B: Backend> {
    /// Python state_dict: `cond_module.0`
    pub(crate) linear0: Linear<B>,
    /// Python state_dict: `cond_module.2`
    pub(crate) linear1: Linear<B>,
    /// Python state_dict: `cond_module.4`
    pub(crate) linear2: Linear<B>,
}

impl<B: Backend> CondModule<B> {
    pub fn new(timestep_embed_dim: usize, model_dim: usize, device: &B::Device) -> Self {
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
    pub fn forward(&self, t_embed: Tensor<B, 2>) -> Tensor<B, 3> {
        let x = silu(self.linear0.forward(t_embed)); // [B, D]
        let x = silu(self.linear1.forward(x)); // [B, D]
        let x = self.linear2.forward(x); // [B, D*3]
        x.unsqueeze_dim::<3>(1) // [B, 1, D*3]
    }
}

// ---------------------------------------------------------------------------
// Main model
// ---------------------------------------------------------------------------

/// Text + reference-latent conditioned RF diffusion model.
///
/// Per-layer intermediate outputs captured during a debug forward pass.
///
/// Produced by [`TextToLatentRfDiT::forward_with_cond_debug`].
/// Not used in production inference paths.
#[derive(Debug)]
pub struct BlockDebugOutputs<B: Backend> {
    /// Output of `in_proj` before the first DiT block. Shape: `[B, S, D]`.
    pub after_in_proj: Tensor<B, 3>,
    /// Output of each DiT block in order. `block_outputs[i]` has shape `[B, S, D]`.
    pub block_outputs: Vec<Tensor<B, 3>>,
}

/// Input `x_t: [B, S, latent_dim * latent_patch_size]`.
/// Output `v_pred: same shape`.
///
/// Field names match the Python state_dict exactly for weight loading.
#[derive(Module, Debug)]
pub struct TextToLatentRfDiT<B: Backend> {
    pub(crate) text_encoder: TextEncoder<B>,
    pub(crate) text_norm: RmsNorm<B>,
    /// Speaker XOR caption conditioning module; `None` for unconditioned models.
    pub(crate) aux_conditioner: Option<AuxConditioner<B>>,
    // Diffusion backbone
    pub(crate) cond_module: CondModule<B>,
    pub(crate) in_proj: Linear<B>,
    pub(crate) blocks: Vec<DiffusionBlock<B>>,
    pub(crate) out_norm: RmsNorm<B>,
    pub(crate) out_proj: Linear<B>,
    // Non-learnable config: stored so we don't need cfg at inference time
    model_dim: usize,
    num_heads: usize,
    head_dim: usize,
    timestep_embed_dim: usize,
    speaker_patch_size: usize,
    patched_latent_dim: usize,
}

impl<B: Backend> TextToLatentRfDiT<B> {
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> Self {
        // Text encoder
        let text_encoder = TextEncoder::from_cfg(cfg, device);
        let text_norm = RmsNorm::new(cfg.text_dim, cfg.norm_eps, device);

        // Speaker or caption encoder — mutually exclusive; `None` when neither is used.
        let aux_conditioner = if cfg.use_speaker_condition() {
            let sp_dim = cfg
                .speaker_dim
                .expect("speaker_dim required for speaker mode");
            Some(AuxConditioner::Speaker(SpeakerConditioner {
                encoder: ReferenceLatentEncoder::from_cfg(cfg, device),
                norm: RmsNorm::new(sp_dim, cfg.norm_eps, device),
            }))
        } else if cfg.use_caption_condition {
            Some(AuxConditioner::Caption(CaptionConditioner {
                encoder: TextEncoder::new(
                    &TextEncoderSpec {
                        vocab_size: cfg.caption_vocab_size(),
                        dim: cfg.caption_dim(),
                        num_layers: cfg.caption_layers(),
                        num_heads: cfg.caption_heads(),
                        mlp_ratio: cfg.caption_mlp_ratio(),
                        norm_eps: cfg.norm_eps,
                        dropout: cfg.dropout,
                    },
                    device,
                ),
                norm: RmsNorm::new(cfg.caption_dim(), cfg.norm_eps, device),
            }))
        } else {
            None
        };

        // Output projection — zero-initialized for stable early training
        let mut out_proj = LinearConfig::new(cfg.model_dim, cfg.patched_latent_dim())
            .with_bias(true)
            .init::<B>(device);
        out_proj.weight = Param::initialized(
            ParamId::new(),
            Tensor::zeros([cfg.patched_latent_dim(), cfg.model_dim], device),
        );
        if let Some(ref mut b) = out_proj.bias {
            *b = Param::initialized(
                ParamId::new(),
                Tensor::zeros([cfg.patched_latent_dim()], device),
            );
        }

        let blocks = (0..cfg.num_layers)
            .map(|_| DiffusionBlock::new(cfg, device))
            .collect();

        let head_dim = cfg.head_dim();
        assert!(
            head_dim.is_multiple_of(2),
            "model head_dim must be even for RoPE, got {head_dim}"
        );

        Self {
            text_encoder,
            text_norm,
            aux_conditioner,
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
        }
    }

    // -----------------------------------------------------------------------
    // Condition encoding
    // -----------------------------------------------------------------------

    /// Encode all conditioning inputs.
    ///
    /// **CFG dropout is applied by zeroing the relevant masks before calling this.**
    ///
    /// > **Warning — NaN risk**: Do NOT pass all-`false` masks for the text
    /// > encoder input. An all-masked row causes softmax over all-`-inf` logits,
    /// > which produces `NaN`. The sampler handles unconditional text by zeroing
    /// > the *output* of this function via [`EncodedCondition::zeros_like`], not
    /// > by zeroing the input mask. Speaker and caption conditioning can safely
    /// > be omitted by passing `AuxConditionInput::None`.
    pub fn encode_conditions(
        &self,
        text_input_ids: Tensor<B, 2, Int>,
        text_mask: Tensor<B, 2, Bool>,
        aux_input: AuxConditionInput<B>,
    ) -> EncodedCondition<B> {
        // Text
        let text_state = self.text_encoder.forward(text_input_ids, text_mask.clone());
        let text_state = self.text_norm.forward(text_state);

        // Aux (speaker XOR caption) — delegate to the module enum
        let aux = self
            .aux_conditioner
            .as_ref()
            .and_then(|cond| cond.encode(aux_input, self.speaker_patch_size));

        EncodedCondition {
            text_state,
            text_mask,
            aux,
        }
    }

    // -----------------------------------------------------------------------
    // Forward with pre-encoded conditions
    // -----------------------------------------------------------------------

    /// Precompute RoPE frequency tables for the latent sequence.
    ///
    /// Returns a [`RopeFreqs`] that can be reused across all denoising steps
    /// within a single sampling trajectory, avoiding redundant trig recomputation.
    pub fn precompute_latent_rope(&self, seq_lat: usize, device: &B::Device) -> RopeFreqs<B> {
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
        x_t: Tensor<B, 3>,
        t: Tensor<B, 1>,
        cond: &EncodedCondition<B>,
        latent_mask: Option<Tensor<B, 2, Bool>>,
        kv_caches: Option<&[CondKvCache<B>]>,
        lat_rope: &RopeFreqs<B>,
    ) -> Tensor<B, 3> {
        nvtx_range!("dit_forward_with_cond", {
            let (x, _) =
                self.forward_backbone(x_t, t, cond, kv_caches, lat_rope, latent_mask, false);
            x
        })
    }

    /// Debug forward: same as [`Self::forward_with_cond_cached`] but additionally
    /// returns per-layer intermediate activations.
    ///
    /// Intended for use in validation/debugging only — not used in production paths.
    /// The per-block `.clone()` calls add non-trivial overhead on large models.
    pub fn forward_with_cond_debug(
        &self,
        x_t: Tensor<B, 3>,
        t: Tensor<B, 1>,
        cond: &EncodedCondition<B>,
        lat_rope: &RopeFreqs<B>,
    ) -> (Tensor<B, 3>, BlockDebugOutputs<B>) {
        self.forward_backbone(x_t, t, cond, None, lat_rope, None, true)
    }

    /// Shared implementation for production and debug forward passes.
    ///
    /// When `capture == true` each block output is cloned into the returned
    /// [`BlockDebugOutputs`].  When `capture == false` the Vec is empty and no
    /// `.clone()` is performed, so there is zero runtime overhead compared to
    /// the old inlined loop.
    fn forward_backbone(
        &self,
        x_t: Tensor<B, 3>,
        t: Tensor<B, 1>,
        cond: &EncodedCondition<B>,
        kv_caches: Option<&[CondKvCache<B>]>,
        lat_rope: &RopeFreqs<B>,
        latent_mask: Option<Tensor<B, 2, Bool>>,
        capture: bool,
    ) -> (Tensor<B, 3>, BlockDebugOutputs<B>) {
        let device = x_t.device();

        let t_embed = nvtx_range!(
            "timestep_embed",
            get_timestep_embedding::<B>(t, self.timestep_embed_dim, &device)
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
            let _label = format!("dit_block_{i}");
            x = nvtx_range!(
                _label.as_str(),
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

    /// Run the diffusion backbone given pre-encoded conditions.
    ///
    /// Precomputes RoPE tables internally. For repeated calls with the same
    /// `x_t` shape (as in a sampling loop), prefer [`Self::precompute_latent_rope`]
    /// + [`Self::forward_with_cond_cached`] to avoid redundant recomputation.
    pub fn forward_with_cond(
        &self,
        x_t: Tensor<B, 3>,
        t: Tensor<B, 1>,
        cond: &EncodedCondition<B>,
        latent_mask: Option<Tensor<B, 2, Bool>>,
        kv_caches: Option<&[CondKvCache<B>]>,
    ) -> Tensor<B, 3> {
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
        x_t: Tensor<B, 3>,
        t: Tensor<B, 1>,
        text_input_ids: Tensor<B, 2, Int>,
        text_mask: Tensor<B, 2, Bool>,
        aux_input: AuxConditionInput<B>,
        latent_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let cond = self.encode_conditions(text_input_ids, text_mask, aux_input);
        self.forward_with_cond(x_t, t, &cond, latent_mask, None)
    }

    // -----------------------------------------------------------------------
    // KV cache construction for fast sampling
    // -----------------------------------------------------------------------

    /// Pre-project all context K/V for each block.
    ///
    /// Call once per trajectory; reuse across denoising steps via
    /// [`Self::forward_with_cond_cached`].
    pub fn build_kv_caches(&self, cond: &EncodedCondition<B>) -> Vec<CondKvCache<B>> {
        let aux_state = cond.aux.as_ref().map(|a| a.state_and_mask().0.clone());
        self.blocks
            .iter()
            .map(|block| {
                block
                    .attention
                    .build_kv_cache(cond.text_state.clone(), aux_state.clone())
            })
            .collect()
    }

    /// Whether the model uses speaker (reference audio) conditioning.
    pub fn use_speaker_condition(&self) -> bool {
        matches!(self.aux_conditioner, Some(AuxConditioner::Speaker(_)))
    }

    /// Dimension of the patched latent space (input/output channels per token).
    pub fn patched_latent_dim(&self) -> usize {
        self.patched_latent_dim
    }
}

// ---------------------------------------------------------------------------
// Helper: prepend masked-mean summary token
// ---------------------------------------------------------------------------

/// Prepend one global summary token = masked mean over the time axis.
///
/// This is applied to speaker context after encoding so the model has a
/// summary token that attends cleanly even when `mask` is all-False.
///
/// - `state: [B, S, D]`, `mask: [B, S]`
/// - Returns `(state': [B, S+1, D], mask': [B, S+1])`
fn prepend_masked_mean_token<B: Backend>(
    state: Tensor<B, 3>,
    mask: Tensor<B, 2, Bool>,
) -> (Tensor<B, 3>, Tensor<B, 2, Bool>) {
    let [batch, seq, _dim] = state.dims();
    let device = state.device();

    // Float mask: [B, S, 1]
    let mask_f: Tensor<B, 3> = {
        let ones: Tensor<B, 2> = Tensor::ones([batch, seq], &device);
        let zeros: Tensor<B, 2> = Tensor::zeros([batch, seq], &device);
        ones.mask_where(mask.clone().bool_not(), zeros)
            .unsqueeze_dim::<3>(2) // [B, S, 1]
    };

    // Masked sum / count: [B, 1, D]
    let sum = (state.clone() * mask_f.clone()).sum_dim(1);
    let count = mask_f.clone().sum_dim(1).clamp_min(1.0_f32); // [B, 1, 1]
    let mean_token = sum / count; // [B, 1, D]

    // Prepend mean token
    let state_out = Tensor::cat(vec![mean_token, state], 1); // [B, S+1, D]

    // has_any: True if at least one valid frame; reshape [B,1,1] → [B,1]
    let count2: Tensor<B, 2> = mask_f.sum_dim(1).reshape([batch, 1]); // [B, 1]
    let has_any: Tensor<B, 2, Bool> = count2.greater_elem(0.0); // [B, 1]
    let mask_out = Tensor::cat(vec![has_any, mask], 1); // [B, S+1]

    (state_out, mask_out)
}
