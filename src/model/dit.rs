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
    ) -> crate::error::Result<Option<AuxConditionState<B>>> {
        match (self, input) {
            (
                Self::Speaker(sp),
                AuxConditionInput::Speaker {
                    ref_latent,
                    ref_mask,
                },
            ) => {
                let (patched_latent, patched_mask) =
                    patch_sequence_with_mask(ref_latent, ref_mask, speaker_patch_size)?;
                let sp_state = sp.encoder.forward(patched_latent, patched_mask.clone());
                let sp_state = sp.norm.forward(sp_state);
                let (sp_state, sp_mask) = prepend_masked_mean_token(sp_state, patched_mask);
                Ok(Some(AuxConditionState::Speaker {
                    state: sp_state,
                    mask: sp_mask,
                }))
            }
            (Self::Caption(cap), AuxConditionInput::Caption { ids, mask }) => {
                let cap_state = cap.encoder.forward(ids, mask.clone());
                let cap_state = cap.norm.forward(cap_state);
                Ok(Some(AuxConditionState::Caption {
                    state: cap_state,
                    mask,
                }))
            }
            // Mismatched mode or no input → no aux conditioning for this pass.
            _ => Ok(None),
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
        // Row layout: weight shape is [d_input=model_dim, d_output=patched_latent_dim]
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
    ) -> crate::error::Result<EncodedCondition<B>> {
        // Text
        let text_state = self.text_encoder.forward(text_input_ids, text_mask.clone());
        let text_state = self.text_norm.forward(text_state);

        // Aux (speaker XOR caption) — delegate to the module enum
        let aux = self
            .aux_conditioner
            .as_ref()
            .map(|cond| cond.encode(aux_input, self.speaker_patch_size))
            .transpose()?
            .flatten();

        Ok(EncodedCondition {
            text_state,
            text_mask,
            aux,
        })
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
    #[allow(clippy::too_many_arguments)]
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
    ) -> crate::error::Result<Tensor<B, 3>> {
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
    pub fn build_kv_caches(&self, cond: &EncodedCondition<B>) -> Vec<CondKvCache<B>> {
        let (aux_state, aux_mask) = cond
            .aux
            .as_ref()
            .map(|a| {
                let (s, m) = a.state_and_mask();
                (Some(s.clone()), Some(m.clone()))
            })
            .unwrap_or((None, None));
        self.blocks
            .iter()
            .map(|block| {
                block.attention.build_kv_cache(
                    cond.text_state.clone(),
                    cond.text_mask.clone(),
                    aux_state.clone(),
                    aux_mask.clone(),
                )
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray<f32>;

    fn tiny_cfg() -> ModelConfig {
        crate::train::tiny_model_config()
    }

    fn device() -> <B as Backend>::Device {
        Default::default()
    }

    // --- CondModule tests ---

    #[test]
    fn cond_module_output_shape() {
        let cfg = tiny_cfg();
        let dev = device();
        let cm = CondModule::<B>::new(cfg.timestep_embed_dim, cfg.model_dim, &dev);

        let t_embed = Tensor::<B, 2>::zeros([2, cfg.timestep_embed_dim], &dev);
        let out = cm.forward(t_embed);
        // Expected: [batch=2, 1, model_dim*3]
        assert_eq!(out.dims(), [2, 1, cfg.model_dim * 3]);
    }

    #[test]
    fn cond_module_silu_activation_applied() {
        let cfg = tiny_cfg();
        let dev = device();
        let cm = CondModule::<B>::new(cfg.timestep_embed_dim, cfg.model_dim, &dev);

        // Non-zero input should produce non-zero output
        let t_embed = Tensor::<B, 2>::ones([1, cfg.timestep_embed_dim], &dev);
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
        let model = TextToLatentRfDiT::<B>::new(&cfg, &dev);

        assert!(model.use_speaker_condition());
        assert!(model.aux_conditioner.is_some());
        assert_eq!(model.blocks.len(), cfg.num_layers);
        assert_eq!(model.patched_latent_dim(), cfg.patched_latent_dim());
    }

    #[test]
    fn model_new_caption_mode() {
        let cfg = crate::train::tiny_caption_config();
        let dev = device();
        let model = TextToLatentRfDiT::<B>::new(&cfg, &dev);

        assert!(!model.use_speaker_condition());
        assert!(matches!(
            model.aux_conditioner,
            Some(AuxConditioner::Caption(_))
        ));
    }

    #[test]
    fn out_proj_zero_initialized() {
        let cfg = tiny_cfg();
        let dev = device();
        let model = TextToLatentRfDiT::<B>::new(&cfg, &dev);

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
        let model = TextToLatentRfDiT::<B>::new(&cfg, &dev);

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
        let model = TextToLatentRfDiT::<B>::new(&cfg, &dev);

        let batch = 2;
        let seq_lat = 4;
        let seq_txt = 6;
        let latent_dim = cfg.patched_latent_dim();

        let x_t = Tensor::<B, 3>::zeros([batch, seq_lat, latent_dim], &dev);
        let t = Tensor::<B, 1>::zeros([batch], &dev);
        let text_ids = Tensor::<B, 2, Int>::zeros([batch, seq_txt], &dev);
        let text_mask = Tensor::<B, 2, Bool>::ones([batch, seq_txt], &dev);

        let speaker_len = 3;
        let speaker_input_dim = cfg.speaker_patched_latent_dim();
        let speaker_latent = Tensor::<B, 3>::zeros([batch, speaker_len, speaker_input_dim], &dev);
        let speaker_mask = Tensor::<B, 2, Bool>::ones([batch, speaker_len], &dev);
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
        let model = TextToLatentRfDiT::<B>::new(&cfg, &dev);

        let batch = 1;
        let seq_lat = 3;
        let seq_txt = 4;
        let latent_dim = cfg.patched_latent_dim();
        let speaker_input_dim = cfg.speaker_patched_latent_dim();

        let x_t = Tensor::<B, 3>::ones([batch, seq_lat, latent_dim], &dev) * 0.1;
        let t = Tensor::<B, 1>::from_data([0.5f32], &dev);
        let text_ids = Tensor::<B, 2, Int>::zeros([batch, seq_txt], &dev);
        let text_mask = Tensor::<B, 2, Bool>::ones([batch, seq_txt], &dev);
        let speaker_latent = Tensor::<B, 3>::ones([batch, 2, speaker_input_dim], &dev) * 0.3;
        let speaker_mask = Tensor::<B, 2, Bool>::ones([batch, 2], &dev);

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

    // --- prepend_masked_mean_token ---

    #[test]
    fn prepend_masked_mean_token_shape() {
        let dev = device();
        let batch = 2;
        let seq = 4;
        let dim = 8;

        let state = Tensor::<B, 3>::ones([batch, seq, dim], &dev);
        let mask = Tensor::<B, 2, Bool>::ones([batch, seq], &dev);
        let (out_state, out_mask) = prepend_masked_mean_token(state, mask);

        assert_eq!(out_state.dims(), [batch, seq + 1, dim]);
        assert_eq!(out_mask.dims(), [batch, seq + 1]);
    }

    #[test]
    fn prepend_masked_mean_token_value() {
        let dev = device();
        let state = Tensor::<B, 3>::ones([1, 3, 2], &dev);
        let mask = Tensor::<B, 2, Bool>::ones([1, 3], &dev);
        let (out_state, _) = prepend_masked_mean_token(state, mask);

        let first_token: Vec<f32> = out_state
            .slice([0..1, 0..1, 0..2])
            .flatten::<1>(0, 2)
            .to_data()
            .to_vec()
            .unwrap();
        assert!((first_token[0] - 1.0).abs() < 1e-5);
        assert!((first_token[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn prepend_masked_mean_token_all_masked_out() {
        let dev = device();
        let state = Tensor::<B, 3>::ones([1, 3, 2], &dev);
        let mask = Tensor::<B, 2, Bool>::from_data(TensorData::from([[false, false, false]]), &dev);
        let (out_state, out_mask) = prepend_masked_mean_token(state, mask);

        let mean: Vec<f32> = out_state
            .slice([0..1, 0..1, 0..2])
            .flatten::<1>(0, 2)
            .to_data()
            .to_vec()
            .unwrap();
        assert!(mean[0].abs() < 1e-5, "masked-out mean should be 0");

        let mask_data: Vec<bool> = out_mask.to_data().to_vec().unwrap();
        assert!(
            !mask_data[0],
            "mean token mask should be false when all inputs masked"
        );
    }
}
