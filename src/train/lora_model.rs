//! LoRA-augmented diffusion transformer model.
//!
//! Mirrors the base model hierarchy (`JointAttention` → `DiffusionBlock` →
//! `TextToLatentRfDiT`) but replaces all attention projections with
//! [`LoraLinear`] adapters.  Everything else (text encoder, aux conditioner,
//! cond_module, in_proj, out_norm, out_proj) remains frozen.

use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    tensor::{Bool, Int, Tensor, activation::sigmoid, backend::Backend},
};

use crate::{
    config::ModelConfig,
    model::{
        attention::{
            CondKvCache, JointAttnCtx, build_joint_mask, concat_ctx_kv,
            scaled_dot_product_attention,
        },
        condition::{AuxConditionInput, EncodedCondition},
        dit::{AuxConditioner, CondModule, build_aux_conditioner, init_zero_out_proj},
        feed_forward::SwiGlu,
        norm::{HeadRmsNorm, LowRankAdaLn, RmsNorm},
        rope::{RopeFreqs, apply_rotary_half, get_timestep_embedding, precompute_rope_freqs_typed},
        text_encoder::TextEncoder,
    },
};

use super::lora_layer::{LoraLinear, LoraLinearConfig};

// ---------------------------------------------------------------------------
// LoraJointAttention
// ---------------------------------------------------------------------------

/// Joint multi-head attention with all projections replaced by LoRA adapters.
///
/// Field names mirror [`JointAttention`] exactly for compatible weight loading.
/// The forward pass is identical; only the linear computations pass through
/// the LoRA delta path.
#[derive(Module, Debug)]
pub struct LoraJointAttention<B: Backend> {
    pub(crate) wq: LoraLinear<B>,
    pub(crate) wk: LoraLinear<B>,
    pub(crate) wv: LoraLinear<B>,
    pub(crate) wk_text: LoraLinear<B>,
    pub(crate) wv_text: LoraLinear<B>,
    pub(crate) wk_speaker: Option<LoraLinear<B>>,
    pub(crate) wv_speaker: Option<LoraLinear<B>>,
    pub(crate) wk_caption: Option<LoraLinear<B>>,
    pub(crate) wv_caption: Option<LoraLinear<B>>,
    pub(crate) gate: LoraLinear<B>,
    pub(crate) wo: LoraLinear<B>,
    pub(crate) q_norm: HeadRmsNorm<B>,
    pub(crate) k_norm: HeadRmsNorm<B>,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl<B: Backend> LoraJointAttention<B> {
    pub fn new(cfg: &ModelConfig, r: usize, alpha: f32, device: &B::Device) -> Self {
        let dim = cfg.model_dim;
        let text_dim = cfg.text_dim;
        let num_heads = cfg.num_heads;
        let head_dim = cfg.head_dim();
        let kv_dim = num_heads * head_dim;
        let scale = (head_dim as f64).powf(-0.5);

        let make_lora = |in_d: usize, out_d: usize| {
            let base = LinearConfig::new(in_d, out_d)
                .with_bias(false)
                .init::<B>(device);
            LoraLinear::from_base(
                base,
                &LoraLinearConfig {
                    in_features: in_d,
                    out_features: out_d,
                    r,
                    alpha,
                },
                device,
            )
        };

        let (wk_speaker, wv_speaker, wk_caption, wv_caption) = if cfg.use_speaker_condition() {
            let sp_dim = cfg.speaker_dim.unwrap_or(dim);
            (
                Some(make_lora(sp_dim, kv_dim)),
                Some(make_lora(sp_dim, kv_dim)),
                None,
                None,
            )
        } else if cfg.use_caption_condition {
            let cap_dim = cfg.caption_dim();
            (
                None,
                None,
                Some(make_lora(cap_dim, kv_dim)),
                Some(make_lora(cap_dim, kv_dim)),
            )
        } else {
            (None, None, None, None)
        };

        Self {
            wq: make_lora(dim, kv_dim),
            wk: make_lora(dim, kv_dim),
            wv: make_lora(dim, kv_dim),
            wk_text: make_lora(text_dim, kv_dim),
            wv_text: make_lora(text_dim, kv_dim),
            wk_speaker,
            wv_speaker,
            wk_caption,
            wv_caption,
            gate: make_lora(dim, dim),
            wo: make_lora(kv_dim, dim),
            q_norm: HeadRmsNorm::new(num_heads, head_dim, cfg.norm_eps, device),
            k_norm: HeadRmsNorm::new(num_heads, head_dim, cfg.norm_eps, device),
            num_heads,
            head_dim,
            scale,
        }
    }

    /// Freeze all base weights in every LoRA projection, plus `q_norm` and `k_norm`.
    ///
    /// Call this **before** `load_record` so that pretrained weights load with
    /// `require_grad = false` and LoRA params remain trainable.
    pub fn freeze_base(self) -> Self {
        let freeze_opt = |opt: Option<LoraLinear<B>>| opt.map(LoraLinear::freeze_base);
        Self {
            wq: self.wq.freeze_base(),
            wk: self.wk.freeze_base(),
            wv: self.wv.freeze_base(),
            wk_text: self.wk_text.freeze_base(),
            wv_text: self.wv_text.freeze_base(),
            wk_speaker: freeze_opt(self.wk_speaker),
            wv_speaker: freeze_opt(self.wv_speaker),
            wk_caption: freeze_opt(self.wk_caption),
            wv_caption: freeze_opt(self.wv_caption),
            gate: self.gate.freeze_base(),
            wo: self.wo.freeze_base(),
            q_norm: self.q_norm.no_grad(),
            k_norm: self.k_norm.no_grad(),
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            scale: self.scale,
        }
    }

    /// Forward pass — identical logic to [`JointAttention::forward`].
    ///
    /// All `Linear` calls are replaced by `LoraLinear` but the computation
    /// graph and shapes are unchanged.
    pub(crate) fn forward(
        &self,
        x: Tensor<B, 3>,
        ctx: JointAttnCtx<'_, B>,
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
        latent_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_lat, _dim] = x.dims();
        let device = x.device();

        let gate_input = x.clone();

        // Q: project + reshape + normalise + half-RoPE
        let q = self
            .wq
            .forward(x.clone())
            .reshape([batch, seq_lat, self.num_heads, self.head_dim]);
        let q = self.q_norm.forward(q);
        let q = apply_rotary_half(q, cos.clone(), sin.clone());

        // Self K/V: project + reshape + normalise K + half-RoPE on K
        let k_self =
            self.wk
                .forward(x.clone())
                .reshape([batch, seq_lat, self.num_heads, self.head_dim]);
        let v_self = self
            .wv
            .forward(x)
            .reshape([batch, seq_lat, self.num_heads, self.head_dim]);
        let k_self = self.k_norm.forward(k_self);
        let k_self = apply_rotary_half(k_self, cos, sin);

        // Context K/V: use pre-concatenated cache in the hot-path; project from scratch
        // (training path) otherwise.
        let (k_ctx, v_ctx, ctx_mask) = if let Some(cache) = ctx.kv_cache {
            (
                cache.ctx_k.clone(),
                cache.ctx_v.clone(),
                Some(cache.ctx_mask.clone()),
            )
        } else {
            let [_, seq_txt, _] = ctx.text_state.dims();
            let k_text = self.wk_text.forward(ctx.text_state.clone()).reshape([
                batch,
                seq_txt,
                self.num_heads,
                self.head_dim,
            ]);
            let k_text = self.k_norm.forward(k_text);
            let v_text = self.wv_text.forward(ctx.text_state).reshape([
                batch,
                seq_txt,
                self.num_heads,
                self.head_dim,
            ]);

            let (k_aux, v_aux) = if let (Some(wk), Some(wv)) = (&self.wk_speaker, &self.wv_speaker)
            {
                if let Some(ref aux) = ctx.aux_state {
                    let [_, seq_aux, _] = aux.dims();
                    let k = wk.forward(aux.clone()).reshape([
                        batch,
                        seq_aux,
                        self.num_heads,
                        self.head_dim,
                    ]);
                    let k = self.k_norm.forward(k);
                    let v = wv.forward(aux.clone()).reshape([
                        batch,
                        seq_aux,
                        self.num_heads,
                        self.head_dim,
                    ]);
                    (Some(k), Some(v))
                } else {
                    (None, None)
                }
            } else if let (Some(wk), Some(wv)) = (&self.wk_caption, &self.wv_caption) {
                if let Some(ref aux) = ctx.aux_state {
                    let [_, seq_aux, _] = aux.dims();
                    let k = wk.forward(aux.clone()).reshape([
                        batch,
                        seq_aux,
                        self.num_heads,
                        self.head_dim,
                    ]);
                    let k = self.k_norm.forward(k);
                    let v = wv.forward(aux.clone()).reshape([
                        batch,
                        seq_aux,
                        self.num_heads,
                        self.head_dim,
                    ]);
                    (Some(k), Some(v))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

            // Concatenate context K/V: [text | aux?]
            concat_ctx_kv(k_text, v_text, k_aux, v_aux, ctx.text_mask, ctx.aux_mask)
        };

        // Full K/V: [self | context]
        let k_all = Tensor::cat(vec![k_self, k_ctx], 1);
        let v_all = Tensor::cat(vec![v_self, v_ctx], 1);

        let mask = build_joint_mask(seq_lat, latent_mask, ctx_mask, batch, &device);
        let out = scaled_dot_product_attention(q, k_all, v_all, mask, self.scale, false);
        let out = out.reshape([batch, seq_lat, self.num_heads * self.head_dim]);

        let gated = sigmoid(self.gate.forward(gate_input)) * out;
        self.wo.forward(gated)
    }
}

// ---------------------------------------------------------------------------
// LoraDiffusionBlock
// ---------------------------------------------------------------------------

/// Single diffusion transformer block with a LoRA-augmented attention layer.
///
/// The MLP and AdaLN projections remain frozen; only the attention projections
/// adapt via LoRA.  Field names mirror [`DiffusionBlock`].
#[derive(Module, Debug)]
pub struct LoraDiffusionBlock<B: Backend> {
    pub(crate) attention: LoraJointAttention<B>,
    pub(crate) mlp: SwiGlu<B>,
    pub(crate) attention_adaln: LowRankAdaLn<B>,
    pub(crate) mlp_adaln: LowRankAdaLn<B>,
    pub(crate) dropout: Dropout,
}

impl<B: Backend> LoraDiffusionBlock<B> {
    pub fn new(cfg: &ModelConfig, r: usize, alpha: f32, device: &B::Device) -> Self {
        let hidden_dim = ((cfg.model_dim as f64 * cfg.mlp_ratio) as usize).max(1);
        let adaln_rank = cfg.adaln_rank.max(1).min(cfg.model_dim);
        Self {
            attention: LoraJointAttention::new(cfg, r, alpha, device),
            mlp: SwiGlu::new(cfg.model_dim, Some(hidden_dim), device),
            attention_adaln: LowRankAdaLn::new(cfg.model_dim, adaln_rank, cfg.norm_eps, device),
            mlp_adaln: LowRankAdaLn::new(cfg.model_dim, adaln_rank, cfg.norm_eps, device),
            dropout: DropoutConfig::new(cfg.dropout).init(),
        }
    }

    /// Freeze the MLP, AdaLN layers and all LoRA base weights.
    ///
    /// Call this before `load_record`.
    pub fn freeze_base(self) -> Self {
        Self {
            attention: self.attention.freeze_base(),
            mlp: self.mlp.no_grad(),
            attention_adaln: self.attention_adaln.no_grad(),
            mlp_adaln: self.mlp_adaln.no_grad(),
            dropout: self.dropout,
        }
    }

    /// Forward — identical to [`DiffusionBlock::forward`].
    #[allow(clippy::too_many_arguments)] // ML forward passes naturally have many inputs
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cond_embed: Tensor<B, 3>,
        cond: &EncodedCondition<B>,
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
        kv_cache: Option<&CondKvCache<B>>,
        latent_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let (aux_state, aux_mask) = match &cond.aux {
            Some(aux) => {
                let (s, m) = aux.state_and_mask();
                (Some(s.clone()), Some(m.clone()))
            }
            None => (None, None),
        };

        let ctx = JointAttnCtx {
            text_state: cond.text_state.clone(),
            text_mask: cond.text_mask.clone(),
            aux_state,
            aux_mask,
            kv_cache,
        };

        let (h_attn, attn_gate) = self.attention_adaln.forward(x.clone(), cond_embed.clone());
        let attn_out = self.attention.forward(h_attn, ctx, cos, sin, latent_mask);
        let x = x + self.dropout.forward(attn_gate * attn_out);

        let (h_mlp, mlp_gate) = self.mlp_adaln.forward(x.clone(), cond_embed);
        let mlp_out = self.mlp.forward(h_mlp);
        x + self.dropout.forward(mlp_gate * mlp_out)
    }
}

// ---------------------------------------------------------------------------
// LoraTextToLatentRfDiT
// ---------------------------------------------------------------------------

/// Full LoRA-adapted diffusion model for training.
///
/// All sub-modules except the attention projections inside each
/// `LoraDiffusionBlock` are frozen after [`Self::freeze_base_weights`].
///
/// Field names match the base model for compatible weight loading.
#[derive(Module, Debug)]
pub struct LoraTextToLatentRfDiT<B: Backend> {
    pub(crate) text_encoder: TextEncoder<B>,
    pub(crate) text_norm: RmsNorm<B>,
    pub(crate) aux_conditioner: Option<AuxConditioner<B>>,
    pub(crate) cond_module: CondModule<B>,
    pub(crate) in_proj: Linear<B>,
    pub(crate) blocks: Vec<LoraDiffusionBlock<B>>,
    pub(crate) out_norm: RmsNorm<B>,
    pub(crate) out_proj: Linear<B>,
    // Non-serialised config
    model_dim: usize,
    num_heads: usize,
    head_dim: usize,
    timestep_embed_dim: usize,
    speaker_patch_size: usize,
    patched_latent_dim: usize,
}

impl<B: Backend> LoraTextToLatentRfDiT<B> {
    /// Construct a fresh model with random base weights and zero-initialised LoRA params.
    ///
    /// Intended use: call [`Self::freeze_base_weights`] immediately, then
    /// `load_record` to replace the base weights from a checkpoint.
    pub fn new(cfg: &ModelConfig, r: usize, alpha: f32, device: &B::Device) -> Self {
        let text_encoder = TextEncoder::from_cfg(cfg, device);
        let text_norm = RmsNorm::new(cfg.text_dim, cfg.norm_eps, device);

        let aux_conditioner = build_aux_conditioner(cfg, device);
        let out_proj = init_zero_out_proj(cfg, device);

        let blocks = (0..cfg.num_layers)
            .map(|_| LoraDiffusionBlock::new(cfg, r, alpha, device))
            .collect();

        let head_dim = cfg.head_dim();

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

    /// Freeze all base (non-LoRA) weights.
    ///
    /// **Call this before `load_record`** so that pretrained weights load with
    /// `require_grad = false`, while LoRA params retain `require_grad = true`.
    pub fn freeze_base_weights(self) -> Self {
        Self {
            text_encoder: self.text_encoder.no_grad(),
            text_norm: self.text_norm.no_grad(),
            aux_conditioner: self.aux_conditioner.map(|c| c.no_grad()),
            cond_module: self.cond_module.no_grad(),
            in_proj: self.in_proj.no_grad(),
            blocks: self
                .blocks
                .into_iter()
                .map(LoraDiffusionBlock::freeze_base)
                .collect(),
            out_norm: self.out_norm.no_grad(),
            out_proj: self.out_proj.no_grad(),
            model_dim: self.model_dim,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            timestep_embed_dim: self.timestep_embed_dim,
            speaker_patch_size: self.speaker_patch_size,
            patched_latent_dim: self.patched_latent_dim,
        }
    }

    /// Precompute RoPE frequency tables for the latent sequence length.
    pub fn precompute_latent_rope(&self, seq_lat: usize, device: &B::Device) -> RopeFreqs<B> {
        precompute_rope_freqs_typed(self.head_dim, seq_lat, 10000.0, device)
    }

    // -----------------------------------------------------------------------
    // Conditioning
    // -----------------------------------------------------------------------

    /// Encode all conditioning inputs into a runtime [`EncodedCondition`].
    pub fn encode_conditions(
        &self,
        text_input_ids: Tensor<B, 2, Int>,
        text_mask: Tensor<B, 2, Bool>,
        aux_input: AuxConditionInput<B>,
    ) -> crate::error::Result<EncodedCondition<B>> {
        let text_state = self.text_encoder.forward(text_input_ids, text_mask.clone());
        let text_state = self.text_norm.forward(text_state);

        let aux = self
            .aux_conditioner
            .as_ref()
            .map(|c| c.encode(aux_input, self.speaker_patch_size))
            .transpose()?
            .flatten();

        Ok(EncodedCondition {
            text_state,
            text_mask,
            aux,
        })
    }

    // -----------------------------------------------------------------------
    // Forward
    // -----------------------------------------------------------------------

    /// Training forward: encode conditions + run diffusion backbone.
    ///
    /// No KV-cache is used — all context projections are recomputed each call.
    pub fn forward_train(
        &self,
        x_t: Tensor<B, 3>,
        t: Tensor<B, 1>,
        text_input_ids: Tensor<B, 2, Int>,
        text_mask: Tensor<B, 2, Bool>,
        aux_input: AuxConditionInput<B>,
        latent_mask: Option<Tensor<B, 2, Bool>>,
    ) -> crate::error::Result<Tensor<B, 3>> {
        let cond = self.encode_conditions(text_input_ids, text_mask, aux_input)?;
        Ok(self.forward_backbone(x_t, t, &cond, latent_mask))
    }

    /// Diffusion backbone: timestep embedding → in_proj → DiT blocks → out_proj.
    ///
    /// Separated from [`forward_train`] to allow the caller to pre-encode
    /// conditions on a non-AD backend for better training throughput.
    pub fn forward_backbone(
        &self,
        x_t: Tensor<B, 3>,
        t: Tensor<B, 1>,
        cond: &EncodedCondition<B>,
        latent_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [_batch, seq_lat, _] = x_t.dims();
        let device = x_t.device();

        let t_embed = get_timestep_embedding::<B>(t, self.timestep_embed_dim, &device);
        let cond_embed = self.cond_module.forward(t_embed);
        let lat_rope = self.precompute_latent_rope(seq_lat, &device);
        let mut x = self.in_proj.forward(x_t);

        for block in &self.blocks {
            x = block.forward(
                x,
                cond_embed.clone(),
                cond,
                lat_rope.cos.clone(),
                lat_rope.sin.clone(),
                None,
                latent_mask.clone(),
            );
        }

        let x = self.out_norm.forward(x);
        self.out_proj.forward(x)
    }
}

// ---------------------------------------------------------------------------
// Helper — prepend mean token for speaker conditioning
// (mirrors dit.rs helper, copied here to avoid visibility issues)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn attention_speaker_mode_has_speaker_projections() {
        let cfg = crate::config::tiny_model_config();
        assert!(cfg.use_speaker_condition());
        let device = Default::default();
        let attn = LoraJointAttention::<TestBackend>::new(&cfg, 2, 4.0, &device);

        assert!(attn.wk_speaker.is_some());
        assert!(attn.wv_speaker.is_some());
        assert!(attn.wk_caption.is_none());
        assert!(attn.wv_caption.is_none());
    }

    #[test]
    fn attention_caption_mode_has_caption_projections() {
        let cfg = crate::config::tiny_caption_config();
        assert!(!cfg.use_speaker_condition());
        let device = Default::default();
        let attn = LoraJointAttention::<TestBackend>::new(&cfg, 2, 4.0, &device);

        assert!(attn.wk_speaker.is_none());
        assert!(attn.wv_speaker.is_none());
        assert!(attn.wk_caption.is_some());
        assert!(attn.wv_caption.is_some());
    }

    #[test]
    fn forward_backbone_output_shape() {
        let cfg = crate::config::tiny_model_config();
        let device = Default::default();
        let model = LoraTextToLatentRfDiT::<TestBackend>::new(&cfg, 2, 4.0, &device);

        let batch = 2;
        let seq_lat = 4;
        let patched = cfg.patched_latent_dim(); // 8

        // Build conditioning manually
        let text_ids = Tensor::<TestBackend, 2, Int>::ones([batch, 3], &device);
        let text_mask = Tensor::<TestBackend, 2, Bool>::from_data(
            burn::tensor::TensorData::from([[true, true, true], [true, true, false]]),
            &device,
        );
        let ref_latent =
            Tensor::<TestBackend, 3>::zeros([batch, 2, cfg.speaker_patched_latent_dim()], &device);
        let ref_mask = Tensor::<TestBackend, 2, Bool>::from_data(
            burn::tensor::TensorData::from([[true, true], [true, false]]),
            &device,
        );

        let cond = model
            .encode_conditions(
                text_ids,
                text_mask,
                AuxConditionInput::Speaker {
                    ref_latent,
                    ref_mask,
                },
            )
            .unwrap();

        let x_t = Tensor::<TestBackend, 3>::zeros([batch, seq_lat, patched], &device);
        let t = Tensor::<TestBackend, 1>::from_data(
            burn::tensor::TensorData::from([0.5f32, 0.3]),
            &device,
        );

        let out = model.forward_backbone(x_t, t, &cond, None);
        assert_eq!(out.dims(), [batch, seq_lat, patched]);
    }

    #[test]
    fn forward_train_matches_manual_encode_plus_backbone() {
        let cfg = crate::config::tiny_model_config();
        let device = Default::default();
        let model = LoraTextToLatentRfDiT::<TestBackend>::new(&cfg, 2, 4.0, &device);

        let batch = 1;
        let seq_lat = 3;
        let patched = cfg.patched_latent_dim();

        let text_ids = Tensor::<TestBackend, 2, Int>::ones([batch, 2], &device);
        let text_mask = Tensor::<TestBackend, 2, Bool>::from_data(
            burn::tensor::TensorData::from([[true, true]]),
            &device,
        );
        let aux = AuxConditionInput::<TestBackend>::None;

        let x_t = Tensor::<TestBackend, 3>::zeros([batch, seq_lat, patched], &device);
        let t =
            Tensor::<TestBackend, 1>::from_data(burn::tensor::TensorData::from([0.5f32]), &device);

        // forward_train encodes conditions internally then calls backbone
        let out_train = model
            .forward_train(
                x_t.clone(),
                t.clone(),
                text_ids.clone(),
                text_mask.clone(),
                aux,
                None,
            )
            .unwrap();

        // Manual: encode conditions then call backbone
        let cond = model
            .encode_conditions(text_ids, text_mask, AuxConditionInput::None)
            .unwrap();
        let out_manual = model.forward_backbone(x_t, t, &cond, None);

        let a: Vec<f32> = out_train.into_data().to_vec().unwrap();
        let b: Vec<f32> = out_manual.into_data().to_vec().unwrap();
        for (va, vb) in a.iter().zip(b.iter()) {
            assert!(
                (va - vb).abs() < 1e-5,
                "forward_train must match encode_conditions + forward_backbone"
            );
        }
    }
}
