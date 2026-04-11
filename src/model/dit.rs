use burn::{
    module::{Module, Param, ParamId},
    nn::{Linear, LinearConfig},
    tensor::{Bool, Int, Tensor, activation::silu, backend::Backend},
};

use crate::config::ModelConfig;

use super::{
    attention::CondKvCache,
    condition::{AuxConditionInput, AuxConditionState, EncodedCondition},
    diffusion::DiffusionBlock,
    norm::RmsNorm,
    rope::{get_timestep_embedding, precompute_rope_freqs},
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
            (Self::Speaker(sp), AuxConditionInput::Speaker { ref_latent, ref_mask }) => {
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

    /// Run the diffusion backbone given pre-encoded conditions.
    ///
    /// This is the hot path: call `encode_conditions` once per sampling trajectory,
    /// then call this for each denoising step.
    ///
    /// - `x_t: [B, S, D_patch]` — noisy latent
    /// - `t: [B]` — timesteps in [0, 1]
    /// - `kv_caches: Option<&[CondKvCache]>` — per-layer precomputed context KVs
    pub fn forward_with_cond(
        &self,
        x_t: Tensor<B, 3>,
        t: Tensor<B, 1>,
        cond: &EncodedCondition<B>,
        _latent_mask: Option<Tensor<B, 2, Bool>>,
        kv_caches: Option<&[CondKvCache<B>]>,
    ) -> Tensor<B, 3> {
        let [_batch, seq_lat, _] = x_t.dims();
        let device = x_t.device();

        // Timestep embedding
        let t_embed = get_timestep_embedding::<B>(t, self.timestep_embed_dim, &device); // [B, t_dim]
        let cond_embed = self.cond_module.forward(t_embed); // [B, 1, D*3]

        let mut x = self.in_proj.forward(x_t); // [B, S, D]

        // Precompute RoPE for latent sequence
        let (cos, sin) = precompute_rope_freqs::<B>(self.head_dim, seq_lat, 10000.0, &device);

        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(
                x,
                cond_embed.clone(),
                cond,
                cos.clone(),
                sin.clone(),
                kv_caches.map(|c| &c[i]),
            );
        }

        let x = self.out_norm.forward(x);
        self.out_proj.forward(x)
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
    /// Call once per trajectory; reuse across denoising steps.
    pub(crate) fn build_kv_caches(&self, cond: &EncodedCondition<B>) -> Vec<CondKvCache<B>> {
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
    let count = mask_f.clone().sum_dim(1).clamp(1.0, f32::MAX); // [B, 1, 1]
    let mean_token = sum / count; // [B, 1, D]

    // Prepend mean token
    let state_out = Tensor::cat(vec![mean_token, state], 1); // [B, S+1, D]

    // has_any: True if at least one valid frame; reshape [B,1,1] → [B,1]
    let count2: Tensor<B, 2> = mask_f.sum_dim(1).reshape([batch, 1]); // [B, 1]
    let has_any: Tensor<B, 2, Bool> = count2.greater_elem(0.0); // [B, 1]
    let mask_out = Tensor::cat(vec![has_any, mask], 1); // [B, S+1]

    (state_out, mask_out)
}
