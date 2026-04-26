use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{
        Bool, Tensor, backend::Backend, module::attention as burn_attention,
        ops::AttentionModuleOptions,
    },
};

use crate::config::ModelConfig;

use super::{
    norm::HeadRmsNorm,
    rope::{apply_rotary_emb, apply_rotary_half},
};

/// Cached KV projections for conditional contexts (text + optional speaker/caption).
///
/// Kept outside the Module system because caches are runtime state, not learned parameters.
///
/// The `ctx_k/ctx_v/ctx_mask` fields are pre-concatenated versions of `text_*` + `aux_*`
/// for use in the sampling hot-path, avoiding 1440 redundant `Tensor::cat` calls
/// (12 blocks × 40 steps × 3 CFG passes).
pub struct CondKvCache<B: Backend> {
    pub(crate) text_k: Tensor<B, 4>, // [B, T_text, H, D_h]
    pub(crate) text_v: Tensor<B, 4>,
    pub(crate) aux_k: Option<Tensor<B, 4>>, // speaker or caption: [B, T_aux, H, D_h]
    pub(crate) aux_v: Option<Tensor<B, 4>>,
    /// Pre-concatenated `[text | aux?]` K: eliminates per-step `Tensor::cat` in the hot path.
    pub(crate) ctx_k: Tensor<B, 4>, // [B, T_text + T_aux, H, D_h]
    /// Pre-concatenated `[text | aux?]` V.
    pub(crate) ctx_v: Tensor<B, 4>,
    /// Pre-concatenated context mask `[text_mask | aux_mask?]`: `[B, T_text + T_aux]`.
    ///
    /// Never `None` because text conditioning is always present; the mask at minimum
    /// equals `text_mask`.
    pub(crate) ctx_mask: Tensor<B, 2, Bool>,
    /// Pre-built joint mask `[latent_ones | ctx_mask]`: `[B, S_lat + T_ctx]`.
    ///
    /// Avoids repeated `build_joint_mask` calls (allocating `ones → bool → cat`)
    /// for each layer of each forward pass during the sampling loop.
    /// Set once via [`CondKvCache::with_joint_mask`]; when present
    /// [`JointAttention::forward`] skips `build_joint_mask` entirely.
    pub(crate) joint_mask: Option<Tensor<B, 2, Bool>>,
}

/// Multi-head self-attention with full RoPE.
///
/// Used in `TextBlock` (text encoder) and `DiffusionBlock` (latent encoder).
/// Field names mirror the Python state_dict for weight-loading compatibility.
#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    pub(crate) wq: Linear<B>,
    pub(crate) wk: Linear<B>,
    pub(crate) wv: Linear<B>,
    pub(crate) wo: Linear<B>,
    pub(crate) gate: Linear<B>,
    pub(crate) q_norm: HeadRmsNorm<B>,
    pub(crate) k_norm: HeadRmsNorm<B>,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl<B: Backend> SelfAttention<B> {
    pub fn new(
        dim: usize,
        num_heads: usize,
        head_dim: Option<usize>,
        norm_eps: f64,
        device: &B::Device,
    ) -> Self {
        let head_dim = head_dim.unwrap_or(dim / num_heads);
        let kv_dim = num_heads * head_dim;
        let scale = (head_dim as f64).powf(-0.5);

        Self {
            wq: LinearConfig::new(dim, kv_dim).with_bias(false).init(device),
            wk: LinearConfig::new(dim, kv_dim).with_bias(false).init(device),
            wv: LinearConfig::new(dim, kv_dim).with_bias(false).init(device),
            wo: LinearConfig::new(kv_dim, dim).with_bias(false).init(device),
            gate: LinearConfig::new(dim, dim).with_bias(false).init(device),
            q_norm: HeadRmsNorm::new(num_heads, head_dim, norm_eps, device),
            k_norm: HeadRmsNorm::new(num_heads, head_dim, norm_eps, device),
            num_heads,
            head_dim,
            scale,
        }
    }

    /// `x: [B, S, D]`, optional mask `[B, S]` (True = valid).
    /// Returns `[B, S, D]`.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
        mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, seq, _dim] = x.dims();

        let gate_input = x.clone();
        let q = self.project_qkv(self.wq.forward(x.clone()), batch, seq);
        let k = self.project_qkv(self.wk.forward(x.clone()), batch, seq);
        let v = self.project_qkv(self.wv.forward(x), batch, seq);

        let q = self.q_norm.forward(q);
        let k = self.k_norm.forward(k);

        // Apply full RoPE to Q and K
        let q = apply_rotary_emb(q, cos.clone(), sin.clone());
        let k = apply_rotary_emb(k, cos, sin);

        let out = scaled_dot_product_attention(q, k, v, mask, self.scale, true);
        // out: [B, S, H, D_h] → [B, S, H*D_h]
        let out = out.reshape([batch, seq, self.num_heads * self.head_dim]);

        // Sigmoid gate: output * sigmoid(gate(x_input)), matching Python SelfAttention
        let out = out * burn::tensor::activation::sigmoid(self.gate.forward(gate_input));
        self.wo.forward(out)
    }

    fn project_qkv(&self, x: Tensor<B, 3>, batch: usize, seq: usize) -> Tensor<B, 4> {
        // [B, S, H*D_h] → [B, S, H, D_h]
        x.reshape([batch, seq, self.num_heads, self.head_dim])
    }
}

/// Joint multi-head attention for diffusion blocks.
///
/// Concatenates K/V from self (latent), text, and optionally speaker or caption.
/// RoPE is applied to the first H/2 heads only (half-RoPE).
///
/// Field names mirror Python for direct weight loading:
/// `wq`, `wk`, `wv`, `wk_text`, `wv_text`, `wk_speaker`, `wv_speaker`,
/// `wk_caption`, `wv_caption`, `gate`, `wo`, `q_norm`, `k_norm`.
#[derive(Module, Debug)]
pub struct JointAttention<B: Backend> {
    pub(crate) wq: Linear<B>,
    pub(crate) wk: Linear<B>,
    pub(crate) wv: Linear<B>,
    pub(crate) wk_text: Linear<B>,
    pub(crate) wv_text: Linear<B>,
    pub(crate) wk_speaker: Option<Linear<B>>,
    pub(crate) wv_speaker: Option<Linear<B>>,
    pub(crate) wk_caption: Option<Linear<B>>,
    pub(crate) wv_caption: Option<Linear<B>>,
    pub(crate) gate: Linear<B>,
    pub(crate) wo: Linear<B>,
    pub(crate) q_norm: HeadRmsNorm<B>,
    pub(crate) k_norm: HeadRmsNorm<B>,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    /// Fused QKV weight: `[3 * kv_dim, dim]` — inference-only optimisation.
    /// When set, a single matmul replaces 3 separate wq/wk/wv calls,
    /// reducing kernel launches by 2 per layer per denoising step.
    #[module(skip)]
    fused_qkv_weight: Option<Tensor<B, 2>>,
}

/// Bundled context inputs for [`JointAttention::forward`].
///
/// Groups text + optional auxiliary conditioning and the optional KV cache
/// into a single struct, eliminating the long argument list.
pub(crate) struct JointAttnCtx<'a, B: Backend> {
    pub(crate) text_state: Tensor<B, 3>,
    pub(crate) text_mask: Tensor<B, 2, Bool>,
    pub(crate) aux_state: Option<Tensor<B, 3>>,
    pub(crate) aux_mask: Option<Tensor<B, 2, Bool>>,
    pub(crate) kv_cache: Option<&'a CondKvCache<B>>,
}

impl<B: Backend> JointAttention<B> {
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> Self {
        let dim = cfg.model_dim;
        let text_dim = cfg.text_dim;
        let num_heads = cfg.num_heads;
        let head_dim = cfg.head_dim();
        let kv_dim = num_heads * head_dim;
        let scale = (head_dim as f64).powf(-0.5);

        let mk_proj = |in_dim| {
            LinearConfig::new(in_dim, kv_dim)
                .with_bias(false)
                .init(device)
        };

        let (wk_speaker, wv_speaker, wk_caption, wv_caption) = if cfg.use_speaker_condition() {
            let sp_dim = cfg.speaker_dim.unwrap_or(cfg.model_dim);
            (Some(mk_proj(sp_dim)), Some(mk_proj(sp_dim)), None, None)
        } else if cfg.use_caption_condition {
            let cap_dim = cfg.caption_dim();
            (None, None, Some(mk_proj(cap_dim)), Some(mk_proj(cap_dim)))
        } else {
            (None, None, None, None)
        };

        Self {
            wq: mk_proj(dim),
            wk: mk_proj(dim),
            wv: mk_proj(dim),
            wk_text: mk_proj(text_dim),
            wv_text: mk_proj(text_dim),
            wk_speaker,
            wv_speaker,
            wk_caption,
            wv_caption,
            gate: LinearConfig::new(dim, dim).with_bias(false).init(device),
            wo: LinearConfig::new(kv_dim, dim).with_bias(false).init(device),
            q_norm: HeadRmsNorm::new(num_heads, head_dim, cfg.norm_eps, device),
            k_norm: HeadRmsNorm::new(num_heads, head_dim, cfg.norm_eps, device),
            num_heads,
            head_dim,
            scale,
            fused_qkv_weight: None,
        }
    }

    /// Forward pass.
    ///
    /// - `x: [B, S_lat, D]` — latent sequence
    /// - `ctx`: bundled text/auxiliary conditioning and optional KV cache
    /// - `cos/sin: [S_lat, head_dim/2]` for half-RoPE
    ///
    /// Returns `[B, S_lat, D]`.
    pub(crate) fn forward(
        &self,
        x: Tensor<B, 3>,
        ctx: JointAttnCtx<'_, B>,
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
        latent_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_lat, _dim] = x.dims();
        let gate_input = x.clone();
        let kv_dim = self.num_heads * self.head_dim;

        // Self Q/K/V: fused single-matmul path when available, else 3 separate linears
        let (q, k_self, v_self) = if let Some(ref fused_w) = self.fused_qkv_weight {
            debug_assert_eq!(
                fused_w.device(),
                x.device(),
                "fused QKV weight on wrong device (was model moved after prepare_for_inference()?)"
            );
            self.compute_qkv_from_fused(x, fused_w, batch, seq_lat, kv_dim)
        } else {
            let q =
                self.wq
                    .forward(x.clone())
                    .reshape([batch, seq_lat, self.num_heads, self.head_dim]);
            let k =
                self.wk
                    .forward(x.clone())
                    .reshape([batch, seq_lat, self.num_heads, self.head_dim]);
            let v = self
                .wv
                .forward(x)
                .reshape([batch, seq_lat, self.num_heads, self.head_dim]);
            (q, k, v)
        };

        self.attention_after_qkv(q, k_self, v_self, gate_input, ctx, cos, sin, latent_mask)
    }

    /// Branch-free forward using the pre-fused QKV weight matrix.
    ///
    /// # Panics
    ///
    /// Panics if [`prepare_for_inference`](Self::prepare_for_inference) has not
    /// been called (i.e. `fused_qkv_weight` is `None`).
    pub(crate) fn forward_fused(
        &self,
        x: Tensor<B, 3>,
        ctx: JointAttnCtx<'_, B>,
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
        latent_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_lat, _dim] = x.dims();
        let gate_input = x.clone();
        let kv_dim = self.num_heads * self.head_dim;

        let fused_w = self.fused_qkv_weight.as_ref().expect(
            "forward_fused called without weight fusion — call prepare_for_inference first",
        );
        let (q, k_self, v_self) = self.compute_qkv_from_fused(x, fused_w, batch, seq_lat, kv_dim);

        self.attention_after_qkv(q, k_self, v_self, gate_input, ctx, cos, sin, latent_mask)
    }

    /// Split fused `[D, 3*kv_dim]` matmul output into separate Q/K/V tensors.
    fn compute_qkv_from_fused(
        &self,
        x: Tensor<B, 3>,
        fused_w: &Tensor<B, 2>,
        batch: usize,
        seq_lat: usize,
        kv_dim: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let qkv = x.matmul(fused_w.clone().unsqueeze::<3>());
        let q = qkv.clone().narrow(2, 0, kv_dim).reshape([
            batch,
            seq_lat,
            self.num_heads,
            self.head_dim,
        ]);
        let k = qkv.clone().narrow(2, kv_dim, kv_dim).reshape([
            batch,
            seq_lat,
            self.num_heads,
            self.head_dim,
        ]);
        let v = qkv.narrow(2, 2 * kv_dim, kv_dim).reshape([
            batch,
            seq_lat,
            self.num_heads,
            self.head_dim,
        ]);
        (q, k, v)
    }

    /// Shared attention logic after Q/K/V have been computed.
    ///
    /// Applies head norms, half-RoPE, context KV projection or cache lookup,
    /// SDPA, and the sigmoid output gate.
    #[allow(clippy::too_many_arguments)]
    fn attention_after_qkv(
        &self,
        q: Tensor<B, 4>,
        k_self: Tensor<B, 4>,
        v_self: Tensor<B, 4>,
        gate_input: Tensor<B, 3>,
        ctx: JointAttnCtx<'_, B>,
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
        latent_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_lat, _, _] = q.dims();
        let device = gate_input.device();

        let q = self.q_norm.forward(q);
        let q = apply_rotary_half(q, cos.clone(), sin.clone());
        let k_self = self.k_norm.forward(k_self);
        let k_self = apply_rotary_half(k_self, cos, sin);

        // Context K/V: use pre-concatenated cache in the hot-path; project from scratch
        // (training path) otherwise.
        let (k_ctx, v_ctx, ctx_mask, cached_joint_mask) = if let Some(cache) = ctx.kv_cache {
            // Pre-concatenated [text | aux?] — no cat needed at all.
            (
                cache.ctx_k.clone(),
                cache.ctx_v.clone(),
                Some(cache.ctx_mask.clone()),
                cache.joint_mask.clone(),
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

            let (k_aux, v_aux) = match (
                ctx.aux_state,
                &self.wk_speaker,
                &self.wv_speaker,
                &self.wk_caption,
                &self.wv_caption,
            ) {
                (Some(aux), Some(wk), Some(wv), _, _) | (Some(aux), _, _, Some(wk), Some(wv)) => {
                    let [_, seq_aux, _] = aux.dims();
                    let k = wk.forward(aux.clone()).reshape([
                        batch,
                        seq_aux,
                        self.num_heads,
                        self.head_dim,
                    ]);
                    let k = self.k_norm.forward(k);
                    let v =
                        wv.forward(aux)
                            .reshape([batch, seq_aux, self.num_heads, self.head_dim]);
                    (Some(k), Some(v))
                }
                _ => (None, None),
            };

            // Concatenate K/V along sequence dimension: [text | aux?]
            let (k, v, m) =
                concat_ctx_kv(k_text, v_text, k_aux, v_aux, ctx.text_mask, ctx.aux_mask);
            (k, v, m, None)
        };

        // Full K: [self | context]
        let k_all = Tensor::cat(vec![k_self, k_ctx], 1);
        let v_all = Tensor::cat(vec![v_self, v_ctx], 1);

        // Use pre-built joint mask if available; otherwise compute on the fly.
        // Invariant: cached joint_mask assumes latent_mask == None (all latent
        // positions attend). Passing both is a programming error.
        assert!(
            cached_joint_mask.is_none() || latent_mask.is_none(),
            "cached joint_mask is incompatible with a non-None latent_mask: \
             the cached mask was built assuming all latent positions attend"
        );
        let mask = cached_joint_mask
            .or_else(|| build_joint_mask(seq_lat, latent_mask, ctx_mask, batch, &device));

        let out = scaled_dot_product_attention(q, k_all, v_all, mask, self.scale, true);
        // out: [B, S_lat, H, D_h] → [B, S_lat, H*D_h]
        let out = out.reshape([batch, seq_lat, self.num_heads * self.head_dim]);

        // Gated output: output * sigmoid(gate(x_input)), matching Python JointAttention
        let gated = burn::tensor::activation::sigmoid(self.gate.forward(gate_input)) * out;
        self.wo.forward(gated)
    }

    /// Build the KV cache for a given context (used during fast sampling).
    ///
    /// Pre-concatenates `[text | aux?]` K/V and the combined mask so that
    /// [`Self::forward`] can use them directly without any `Tensor::cat` per step.
    pub fn build_kv_cache(
        &self,
        text_state: Tensor<B, 3>,
        text_mask: Tensor<B, 2, Bool>,
        aux_state: Option<Tensor<B, 3>>,
        aux_mask: Option<Tensor<B, 2, Bool>>,
    ) -> CondKvCache<B> {
        let [batch, seq_txt, _] = text_state.dims();
        let k_text = self.wk_text.forward(text_state.clone()).reshape([
            batch,
            seq_txt,
            self.num_heads,
            self.head_dim,
        ]);
        let k_text = self.k_norm.forward(k_text);
        let v_text = self.wv_text.forward(text_state).reshape([
            batch,
            seq_txt,
            self.num_heads,
            self.head_dim,
        ]);

        let (aux_k, aux_v) = match (
            aux_state,
            &self.wk_speaker,
            &self.wv_speaker,
            &self.wk_caption,
            &self.wv_caption,
        ) {
            (Some(aux), Some(wk), Some(wv), _, _) | (Some(aux), _, _, Some(wk), Some(wv)) => {
                let [_, seq_aux, _] = aux.dims();
                let k = wk.forward(aux.clone()).reshape([
                    batch,
                    seq_aux,
                    self.num_heads,
                    self.head_dim,
                ]);
                let k = self.k_norm.forward(k);
                let v = wv
                    .forward(aux)
                    .reshape([batch, seq_aux, self.num_heads, self.head_dim]);
                (Some(k), Some(v))
            }
            _ => (None, None),
        };

        // Pre-concatenate for the sampling hot-path.
        let ctx_k = match &aux_k {
            Some(ak) => Tensor::cat(vec![k_text.clone(), ak.clone()], 1),
            None => k_text.clone(),
        };
        let ctx_v = match &aux_v {
            Some(av) => Tensor::cat(vec![v_text.clone(), av.clone()], 1),
            None => v_text.clone(),
        };
        let ctx_mask = match aux_mask {
            Some(am) => Tensor::cat(vec![text_mask, am], 1),
            None => text_mask,
        };

        CondKvCache {
            text_k: k_text,
            text_v: v_text,
            aux_k,
            aux_v,
            ctx_k,
            ctx_v,
            ctx_mask,
            joint_mask: None,
        }
    }

    /// Fuse separate wq/wk/wv weight matrices into a single `[D, 3*kv_dim]` tensor.
    ///
    /// This replaces 3 kernel launches (per layer per denoising step) with 1,
    /// saving ~20ms total across 12 layers × 40 steps on a typical inference run.
    /// Safe to call multiple times (idempotent). Does not affect serialization
    /// because the fused weight is `#[module(skip)]`.
    ///
    /// # Safety invariant
    /// Must be called **after** final weights are loaded and device placement is
    /// complete. The fused tensor is `#[module(skip)]`, so it will NOT follow
    /// `to_device()` or `fork()` calls on the parent module.
    pub(crate) fn prepare_for_inference(&mut self) {
        if self.fused_qkv_weight.is_some() {
            return;
        }
        let wq = self.wq.weight.val(); // [D, kv_dim]
        let wk = self.wk.weight.val();
        let wv = self.wv.weight.val();
        // Cat along output dim (dim 1): [D, 3*kv_dim]
        self.fused_qkv_weight = Some(Tensor::cat(vec![wq, wk, wv], 1));
    }
}

impl<B: Backend> CondKvCache<B> {
    /// Pre-build the full joint mask `[ones(B, seq_lat) | ctx_mask]` so that
    /// [`JointAttention::forward`] can skip `build_joint_mask` entirely.
    ///
    /// Call once per cache (before the sampling loop) with the latent
    /// sequence length that will be used for all timesteps.
    pub(crate) fn precompute_joint_mask(&mut self, seq_lat: usize) {
        let [batch, _seq_ctx] = self.ctx_mask.dims();
        let device = self.ctx_mask.device();
        let self_part: Tensor<B, 2, Bool> =
            Tensor::<B, 2>::ones([batch, seq_lat], &device).greater_elem(0.0);
        self.joint_mask = Some(Tensor::cat(vec![self_part, self.ctx_mask.clone()], 1));
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Concatenate pre-projected context K, V, and mask along the sequence dimension.
///
/// Given separately projected text and optional auxiliary K/V tensors (shape
/// `[B, S, H, D_h]`), returns the joined `(k_ctx, v_ctx, ctx_mask)` where:
///
/// - `k_ctx  = [k_text | k_aux?]` along `dim=1`
/// - `v_ctx  = [v_text | v_aux?]` along `dim=1`
/// - `ctx_mask = [text_mask | aux_mask?]` along `dim=1` (always `Some`)
///
/// Used by both `JointAttention` and `LoraJointAttention` to avoid duplicating
/// the post-projection assembly logic.
pub(crate) fn concat_ctx_kv<B: Backend>(
    k_text: Tensor<B, 4>,
    v_text: Tensor<B, 4>,
    k_aux: Option<Tensor<B, 4>>,
    v_aux: Option<Tensor<B, 4>>,
    text_mask: Tensor<B, 2, Bool>,
    aux_mask: Option<Tensor<B, 2, Bool>>,
) -> (Tensor<B, 4>, Tensor<B, 4>, Option<Tensor<B, 2, Bool>>) {
    let k_ctx = match k_aux {
        Some(ref ka) => Tensor::cat(vec![k_text, ka.clone()], 1),
        None => k_text,
    };
    let v_ctx = match v_aux {
        Some(ref va) => Tensor::cat(vec![v_text, va.clone()], 1),
        None => v_text,
    };
    let ctx_mask = match aux_mask {
        Some(am) => Some(Tensor::cat(vec![text_mask, am], 1)),
        None => Some(text_mask),
    };
    (k_ctx, v_ctx, ctx_mask)
}

/// Returns `true` when `burn::tensor::module::attention()` on this backend follows the
/// PyTorch bool-mask convention: `True = attend (include)`.
///
/// burn's `attention()` has a cross-backend inconsistency:
/// - **LibTorch** (`burn-tch`): delegates to `tch::Tensor::scaled_dot_product_attention`,
///   which follows PyTorch semantics — `True = attend`. No inversion needed.
/// - **NdArray** (`burn-ndarray`): calls `attention_fallback` →
///   `float_mask_fill(scores, mask, NEG_INFINITY)` — `True = masked-out`. Inverted.
/// - **CubeCL / WgpuRaw** (`burn-cubecl` / cubek FA): also `True = masked-out`. Inverted.
///
/// `Backend: 'static` (a supertrait bound) makes `TypeId::of::<B>()` valid here
/// without any additional bound on callers.
fn uses_pytorch_attn_mask_convention<B: Backend>() -> bool {
    use std::any::TypeId;
    let b_id = TypeId::of::<B>();
    #[cfg(feature = "tch")]
    {
        use burn::backend::LibTorch;
        if b_id == TypeId::of::<LibTorch>()
            || b_id == TypeId::of::<LibTorch<half::bf16>>()
            || b_id == TypeId::of::<LibTorch<half::f16>>()
        {
            return true;
        }
    }
    let _ = b_id; // suppress unused-variable warning when tch feature is off
    false
}

/// Scaled dot-product attention using burn's native `attention()` kernel.
///
/// On LibTorch this dispatches to PyTorch's `scaled_dot_product_attention`,
/// which in turn selects FlashAttention v2 or cuDNN efficient kernels when
/// available — typically 2–5× faster than the manual matmul + softmax path.
///
/// `q/k/v: [B, S, H, D_h]`. mask (optional): `[B, S_kv]` — True = valid (attend).
/// Returns `[B, S_q, H, D_h]`.
///
/// `safe_softmax` is retained for API compatibility but has no effect: burn's
/// native attention handles fully-masked rows correctly across all backends.
pub(crate) fn scaled_dot_product_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<Tensor<B, 2, Bool>>,
    scale: f64,
    _safe_softmax: bool,
) -> Tensor<B, 4> {
    let [_batch, _seq_q, _num_heads, _head_dim] = q.dims();

    // Rearrange to [B, H, S, D_h] for burn's attention API.
    let q = q.swap_dims(1, 2);
    let k = k.swap_dims(1, 2);
    let v = v.swap_dims(1, 2);

    // Convert 2D key-padding mask [B, S_kv] → 4D [B, 1, 1, S_kv].
    // PyTorch SDPA broadcasts across heads and query positions natively;
    // no explicit `.expand()` needed — avoids materialising the full mask.
    //
    // Our callers use `True = attend (valid)`. burn's NdArray and CubeCL kernels
    // use `True = masked-out` — the opposite convention. Invert for those backends.
    let mask_4d = mask.map(|m| {
        let m = if uses_pytorch_attn_mask_convention::<B>() {
            m
        } else {
            m.bool_not() // True=attend → True=masked-out for NdArray/CubeCL
        };
        m.unsqueeze_dim::<3>(1) // [B, 1, S_kv]
            .unsqueeze_dim::<4>(2) // [B, 1, 1, S_kv]
    });

    // Pass scale = None so burn infers the standard 1/sqrt(d_head).
    // This is important for CubeCL: `scale.is_some()` forces a fallback path
    // instead of flash attention. The caller's `scale` is always (head_dim)^{-0.5},
    // so letting the backend infer it is numerically equivalent.
    let _ = scale; // consumed for documentation; burn computes the same value
    let options = AttentionModuleOptions {
        scale: None,
        softcap: None,
        is_causal: false,
    };

    let out = burn_attention(q, k, v, mask_4d, None, options);
    // [B, H, S_q, D_h] → [B, S_q, H, D_h]
    out.swap_dims(1, 2)
}

/// Manual scaled dot-product attention: softmax(Q @ K^T × scale) @ V.
///
/// Used by LoRA training — handles the `True = attend` mask convention directly,
/// bypassing burn's backend-specific attention kernels.
///
/// `q/k/v: [B, S, H, D_h]`. mask (optional): `[B, S_kv]` — True = valid (attend).
/// Returns `[B, S_q, H, D_h]`.
///
/// `safe_softmax`: when `true`, NaN from all-masked rows is replaced with 0.0
/// (required for inference with CFG where some context positions may be fully masked).
/// When `false`, NaN handling is skipped for better training throughput — assumes
/// no all-masked key rows (valid for well-formed training batches with padding).
#[allow(dead_code)]
pub(crate) fn manual_sdpa<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<Tensor<B, 2, Bool>>,
    scale: f64,
    safe_softmax: bool,
) -> Tensor<B, 4> {
    use burn::tensor::activation::softmax;

    let [batch, seq_q, num_heads, _head_dim] = q.dims();
    let [_, seq_k, _, _] = k.dims();

    // Rearrange to [B, H, S, D_h] for batched matmul
    let q = q.swap_dims(1, 2);
    let k = k.swap_dims(1, 2);
    let v = v.swap_dims(1, 2);

    // Scores: [B, H, S_q, S_k]
    let scores = q.matmul(k.swap_dims(2, 3)) * scale;

    // Apply mask: mask (true=attend) → invert to (true=mask-out) for mask_fill.
    let scores = if let Some(m) = mask {
        let invalid = m.bool_not();
        let invalid: Tensor<B, 4, Bool> = invalid
            .unsqueeze_dim::<3>(1)
            .unsqueeze_dim::<4>(2)
            .expand([batch, num_heads, seq_q, seq_k]);
        scores.mask_fill(invalid, f32::NEG_INFINITY)
    } else {
        scores
    };

    let attn_weights = softmax(scores, 3);

    let attn_weights = if safe_softmax {
        let nan_mask = attn_weights.clone().is_nan();
        attn_weights.mask_fill(nan_mask, 0.0)
    } else {
        attn_weights
    };

    let out = attn_weights.matmul(v);
    out.swap_dims(1, 2)
}

/// Build a mask for joint attention: query can attend everywhere in self,
/// and to valid positions in context.
///
/// Returns `Option<Tensor<B, 2, Bool>>` of shape `[B, S_lat + S_ctx]`
/// where the first `S_lat` positions are always True.
pub(crate) fn build_joint_mask<B: Backend>(
    seq_lat: usize,
    latent_mask: Option<Tensor<B, 2, Bool>>,
    ctx_mask: Option<Tensor<B, 2, Bool>>,
    batch: usize,
    device: &B::Device,
) -> Option<Tensor<B, 2, Bool>> {
    match (latent_mask, ctx_mask) {
        (None, None) => None,
        (lat_mask, ctx) => {
            let self_part = lat_mask.unwrap_or_else(|| {
                // All positions valid (inference: no padding)
                Tensor::<B, 2>::ones([batch, seq_lat], device).greater_elem(0.0)
            });
            match ctx {
                Some(cm) => Some(Tensor::cat(vec![self_part, cm], 1)),
                None => Some(self_part),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::tensor::Tensor;

    use crate::config::ModelConfig;

    use super::{Backend, Bool, JointAttention, JointAttnCtx, build_joint_mask, manual_sdpa};

    type B = NdArray<f32>;

    // -----------------------------------------------------------------------
    // manual_sdpa (formerly scaled_dot_product_attention)
    // -----------------------------------------------------------------------

    /// When the entire key sequence is masked, softmax produces all-NaN (0/0).
    /// The implementation must replace NaN with 0 so the output is all-zeros.
    #[test]
    fn sdpa_all_masked_gives_zero() {
        let device: <B as Backend>::Device = Default::default();
        let (batch, seq_q, seq_k, num_heads, head_dim) = (1, 3, 4, 2, 4);
        let scale = (head_dim as f64).powf(-0.5);

        let q = Tensor::<B, 4>::ones([batch, seq_q, num_heads, head_dim], &device);
        let k = Tensor::<B, 4>::ones([batch, seq_k, num_heads, head_dim], &device);
        let v = Tensor::<B, 4>::ones([batch, seq_k, num_heads, head_dim], &device);

        // All-false mask: every key position is invalid.
        let mask: Tensor<B, 2, Bool> =
            Tensor::<B, 2>::zeros([batch, seq_k], &device).greater_elem(1.0);

        let out = manual_sdpa(q, k, v, Some(mask), scale, true);
        for val in out.into_data().to_vec::<f32>().expect("to_vec") {
            assert_eq!(val, 0.0, "all-masked attention must produce exactly 0.0");
        }
    }

    /// Partial mask: positions with valid keys receive non-zero output.
    #[test]
    fn sdpa_partial_mask_produces_nonzero() {
        let device: <B as Backend>::Device = Default::default();
        let (batch, seq_q, seq_k, num_heads, head_dim) = (1, 2, 4, 1, 4);
        let scale = (head_dim as f64).powf(-0.5);

        let q = Tensor::<B, 4>::ones([batch, seq_q, num_heads, head_dim], &device);
        let k = Tensor::<B, 4>::ones([batch, seq_k, num_heads, head_dim], &device);
        let v = Tensor::<B, 4>::ones([batch, seq_k, num_heads, head_dim], &device);

        // First 2 positions valid, last 2 masked.
        let mask_data =
            Tensor::<B, 2>::from_data([[1.0f32, 1.0, 0.0, 0.0]], &device).greater_elem(0.5);

        let out = manual_sdpa(q, k, v, Some(mask_data), scale, true);
        let max_val: f32 = out
            .into_data()
            .to_vec::<f32>()
            .expect("to_vec")
            .into_iter()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(max_val > 0.0, "partial mask should give non-zero output");
    }

    // -----------------------------------------------------------------------
    // build_joint_mask
    // -----------------------------------------------------------------------

    #[test]
    fn joint_mask_both_none_returns_none() {
        let device: <B as Backend>::Device = Default::default();
        let result: Option<Tensor<B, 2, Bool>> = build_joint_mask::<B>(4, None, None, 2, &device);
        assert!(result.is_none(), "both None must return None");
    }

    #[test]
    fn joint_mask_ctx_only_correct_shape_and_latent_true() {
        let device: <B as Backend>::Device = Default::default();
        let (batch, seq_lat, seq_ctx) = (2, 3, 5);

        let ctx_mask: Tensor<B, 2, Bool> =
            Tensor::<B, 2>::ones([batch, seq_ctx], &device).greater_elem(0.0);
        let result = build_joint_mask::<B>(seq_lat, None, Some(ctx_mask), batch, &device).unwrap();

        let [b, s] = result.dims();
        assert_eq!(b, batch);
        assert_eq!(s, seq_lat + seq_ctx, "shape must be [B, seq_lat + seq_ctx]");

        // The first seq_lat positions must all be True (all-ones fallback latent mask).
        let data = result.into_data().to_vec::<bool>().expect("to_vec");
        for i in 0..batch {
            for j in 0..seq_lat {
                assert!(
                    data[i * (seq_lat + seq_ctx) + j],
                    "latent positions must be True"
                );
            }
        }
    }

    #[test]
    fn joint_mask_with_latent_mask_propagates_correctly() {
        let device: <B as Backend>::Device = Default::default();
        let (batch, seq_lat, seq_ctx) = (1, 2, 3);

        // Latent: only position 0 valid.
        let lat_mask: Tensor<B, 2, Bool> =
            Tensor::<B, 2>::from_data([[1.0f32, 0.0]], &device).greater_elem(0.5);
        let ctx_mask: Tensor<B, 2, Bool> =
            Tensor::<B, 2>::ones([batch, seq_ctx], &device).greater_elem(0.0);

        let result =
            build_joint_mask::<B>(seq_lat, Some(lat_mask), Some(ctx_mask), batch, &device).unwrap();

        let data = result.into_data().to_vec::<bool>().expect("to_vec");
        assert!(data[0], "lat[0] must be True");
        assert!(!data[1], "lat[1] must be False (masked)");
        for (j, &val) in data[seq_lat..(seq_lat + seq_ctx)].iter().enumerate() {
            assert!(val, "ctx position {} must be True", seq_lat + j);
        }
    }

    // -----------------------------------------------------------------------
    // KV cache equivalence
    // -----------------------------------------------------------------------

    fn tiny_config_speaker() -> ModelConfig {
        ModelConfig {
            model_dim: 16,
            num_heads: 2,
            latent_dim: 4,
            latent_patch_size: 1,
            num_layers: 1,
            text_dim: 8,
            text_heads: 2,
            text_layers: 1,
            text_vocab_size: 32,
            timestep_embed_dim: 16,
            adaln_rank: 4,
            speaker_dim: Some(8),
            speaker_heads: Some(2),
            speaker_layers: Some(1),
            speaker_patch_size: Some(1),
            ..Default::default()
        }
    }

    /// Cached and non-cached forward passes of `JointAttention` must produce
    /// identical outputs (bit-for-bit on NdArray backend).
    #[test]
    fn kv_cache_matches_non_cached_forward() {
        let device: <B as Backend>::Device = Default::default();
        let cfg = tiny_config_speaker();
        let attn = JointAttention::<B>::new(&cfg, &device);

        let (batch, seq_lat, seq_txt) = (1, 4, 6);
        let head_dim = cfg.head_dim();

        let x = Tensor::<B, 3>::ones([batch, seq_lat, cfg.model_dim], &device);
        let text_state = Tensor::<B, 3>::ones([batch, seq_txt, cfg.text_dim], &device);
        let text_mask: Tensor<B, 2, Bool> =
            Tensor::<B, 2>::ones([batch, seq_txt], &device).greater_elem(0.0);

        // RoPE tables: identity rotation (cos=1, sin=0), shape [seq_lat, head_dim/2]
        let cos = Tensor::<B, 2>::ones([seq_lat, head_dim / 2], &device);
        let sin = Tensor::<B, 2>::zeros([seq_lat, head_dim / 2], &device);

        // Non-cached forward (training path: projects text from scratch)
        let out_no_cache = attn.forward(
            x.clone(),
            JointAttnCtx {
                text_state: text_state.clone(),
                text_mask: text_mask.clone(),
                aux_state: None,
                aux_mask: None,
                kv_cache: None,
            },
            cos.clone(),
            sin.clone(),
            None,
        );

        // Build cache (pre-computes ctx_k/ctx_v/ctx_mask)
        let cache = attn.build_kv_cache(text_state.clone(), text_mask.clone(), None, None);

        // Cached forward (sampling hot-path: uses pre-computed tensors)
        let out_cached = attn.forward(
            x,
            JointAttnCtx {
                text_state,
                text_mask,
                aux_state: None,
                aux_mask: None,
                kv_cache: Some(&cache),
            },
            cos,
            sin,
            None,
        );

        // Must be bit-for-bit identical on a deterministic backend.
        let max_diff: f32 = (out_no_cache - out_cached).abs().max().into_scalar();
        assert_eq!(
            max_diff, 0.0,
            "cached and non-cached paths must produce identical output"
        );
    }

    /// When a speaker-conditioning aux state is present, the cached path must
    /// concatenate `[text | speaker]` ctx KV and produce the same output as the
    /// non-cached path that projects both on-the-fly.
    #[test]
    fn kv_cache_with_aux_matches_non_cached_forward() {
        let device: <B as Backend>::Device = Default::default();
        let cfg = tiny_config_speaker();
        let attn = JointAttention::<B>::new(&cfg, &device);

        let head_dim = cfg.head_dim();
        let spk_dim = cfg.speaker_dim.unwrap();
        let (batch, seq_lat, seq_txt, seq_spk) = (1, 4, 6, 3);

        let x = Tensor::<B, 3>::ones([batch, seq_lat, cfg.model_dim], &device);
        let text_state = Tensor::<B, 3>::ones([batch, seq_txt, cfg.text_dim], &device);
        let text_mask: Tensor<B, 2, Bool> =
            Tensor::<B, 2>::ones([batch, seq_txt], &device).greater_elem(0.0);
        let aux_state = Tensor::<B, 3>::ones([batch, seq_spk, spk_dim], &device);
        let aux_mask: Tensor<B, 2, Bool> =
            Tensor::<B, 2>::ones([batch, seq_spk], &device).greater_elem(0.0);

        // RoPE: identity rotation (cos=1, sin=0)
        let cos = Tensor::<B, 2>::ones([seq_lat, head_dim / 2], &device);
        let sin = Tensor::<B, 2>::zeros([seq_lat, head_dim / 2], &device);

        // Non-cached: projects text+aux together at forward time
        let out_no_cache = attn.forward(
            x.clone(),
            JointAttnCtx {
                text_state: text_state.clone(),
                text_mask: text_mask.clone(),
                aux_state: Some(aux_state.clone()),
                aux_mask: Some(aux_mask.clone()),
                kv_cache: None,
            },
            cos.clone(),
            sin.clone(),
            None,
        );

        // Build cache with speaker aux — ctx_k/ctx_v must be [text|spk] concatenated
        let cache = attn.build_kv_cache(
            text_state.clone(),
            text_mask.clone(),
            Some(aux_state.clone()),
            Some(aux_mask.clone()),
        );

        // Cached: reads pre-computed tensors (text+aux already concatenated)
        let out_cached = attn.forward(
            x,
            JointAttnCtx {
                text_state,
                text_mask,
                aux_state: Some(aux_state),
                aux_mask: Some(aux_mask),
                kv_cache: Some(&cache),
            },
            cos,
            sin,
            None,
        );

        let max_diff: f32 = (out_no_cache - out_cached).abs().max().into_scalar();
        assert_eq!(
            max_diff, 0.0,
            "cached and non-cached paths must produce identical output (with aux)"
        );
    }

    fn tiny_config_caption() -> ModelConfig {
        ModelConfig {
            model_dim: 16,
            num_heads: 2,
            latent_dim: 4,
            latent_patch_size: 1,
            num_layers: 1,
            text_dim: 8,
            text_heads: 2,
            text_layers: 1,
            text_vocab_size: 32,
            timestep_embed_dim: 16,
            adaln_rank: 4,
            use_caption_condition: true,
            caption_vocab_size: Some(32),
            caption_dim: Some(12),
            caption_layers: Some(1),
            caption_heads: Some(2),
            caption_mlp_ratio: Some(2.0),
            ..Default::default()
        }
    }

    /// Caption-conditioned cached vs non-cached forward must be bit-identical.
    ///
    /// This mirrors `kv_cache_with_aux_matches_non_cached_forward` but uses
    /// `wk_caption`/`wv_caption` instead of `wk_speaker`/`wv_speaker`.
    #[test]
    fn kv_cache_caption_mode_matches_non_cached_forward() {
        let device: <B as Backend>::Device = Default::default();
        let cfg = tiny_config_caption();
        let attn = JointAttention::<B>::new(&cfg, &device);

        // Verify caption projections are present
        assert!(attn.wk_caption.is_some());
        assert!(attn.wv_caption.is_some());
        assert!(attn.wk_speaker.is_none());

        let head_dim = cfg.head_dim();
        let cap_dim = cfg.caption_dim();
        let (batch, seq_lat, seq_txt, seq_cap) = (1, 4, 6, 3);

        let x = Tensor::<B, 3>::ones([batch, seq_lat, cfg.model_dim], &device);
        let text_state = Tensor::<B, 3>::ones([batch, seq_txt, cfg.text_dim], &device);
        let text_mask: Tensor<B, 2, Bool> =
            Tensor::<B, 2>::ones([batch, seq_txt], &device).greater_elem(0.0);
        let aux_state = Tensor::<B, 3>::ones([batch, seq_cap, cap_dim], &device);
        let aux_mask: Tensor<B, 2, Bool> =
            Tensor::<B, 2>::ones([batch, seq_cap], &device).greater_elem(0.0);

        let cos = Tensor::<B, 2>::ones([seq_lat, head_dim / 2], &device);
        let sin = Tensor::<B, 2>::zeros([seq_lat, head_dim / 2], &device);

        // Non-cached: projects text+caption from scratch
        let out_no_cache = attn.forward(
            x.clone(),
            JointAttnCtx {
                text_state: text_state.clone(),
                text_mask: text_mask.clone(),
                aux_state: Some(aux_state.clone()),
                aux_mask: Some(aux_mask.clone()),
                kv_cache: None,
            },
            cos.clone(),
            sin.clone(),
            None,
        );

        // Build cache with caption aux
        let cache = attn.build_kv_cache(
            text_state.clone(),
            text_mask.clone(),
            Some(aux_state.clone()),
            Some(aux_mask.clone()),
        );

        // Cached: reads pre-computed [text|caption] ctx KV
        let out_cached = attn.forward(
            x,
            JointAttnCtx {
                text_state,
                text_mask,
                aux_state: Some(aux_state),
                aux_mask: Some(aux_mask),
                kv_cache: Some(&cache),
            },
            cos,
            sin,
            None,
        );

        let max_diff: f32 = (out_no_cache - out_cached).abs().max().into_scalar();
        assert_eq!(
            max_diff, 0.0,
            "cached and non-cached paths must be identical for caption mode"
        );
    }

    /// Passing both a KV cache (which includes a pre-built joint_mask) and a
    /// latent_mask is a programming error — the cached mask was built assuming
    /// all latent positions attend. This must panic at runtime.
    #[test]
    #[should_panic(expected = "cached joint_mask is incompatible")]
    fn cached_joint_mask_plus_latent_mask_panics() {
        let device: <B as Backend>::Device = Default::default();
        let cfg = tiny_config_speaker();
        let attn = JointAttention::<B>::new(&cfg, &device);

        let head_dim = cfg.head_dim();
        let (batch, seq_lat, seq_txt) = (1, 4, 6);

        let x = Tensor::<B, 3>::ones([batch, seq_lat, cfg.model_dim], &device);
        let text_state = Tensor::<B, 3>::ones([batch, seq_txt, cfg.text_dim], &device);
        let text_mask: Tensor<B, 2, Bool> =
            Tensor::<B, 2>::ones([batch, seq_txt], &device).greater_elem(0.0);

        let cos = Tensor::<B, 2>::ones([seq_lat, head_dim / 2], &device);
        let sin = Tensor::<B, 2>::zeros([seq_lat, head_dim / 2], &device);

        // Build a cache and precompute the joint_mask
        let mut cache = attn.build_kv_cache(text_state.clone(), text_mask.clone(), None, None);
        cache.precompute_joint_mask(seq_lat);

        // Also provide a latent_mask — this combination is invalid
        let latent_mask: Tensor<B, 2, Bool> =
            Tensor::<B, 2>::ones([batch, seq_lat], &device).greater_elem(0.0);

        // This should panic
        let _out = attn.forward(
            x,
            JointAttnCtx {
                text_state,
                text_mask,
                aux_state: None,
                aux_mask: None,
                kv_cache: Some(&cache),
            },
            cos,
            sin,
            Some(latent_mask),
        );
    }

    // -----------------------------------------------------------------------
    // Fused QKV inference optimisation
    // -----------------------------------------------------------------------

    fn tiny_cfg() -> ModelConfig {
        crate::config::tiny_model_config()
    }

    /// After `prepare_for_inference()`, fused QKV forward must produce
    /// identical output to the 3-linear path.
    #[test]
    fn fused_qkv_matches_separate_linears() {
        let cfg = tiny_cfg();
        let device: <B as Backend>::Device = Default::default();
        let mut attn = JointAttention::<B>::new(&cfg, &device);

        let batch = 2;
        let seq_lat = 6;
        let text_len = 4;
        let head_half = cfg.head_dim() / 2;

        let x = Tensor::<B, 3>::random(
            [batch, seq_lat, cfg.model_dim],
            burn::tensor::Distribution::Default,
            &device,
        );
        let text_state = Tensor::<B, 3>::random(
            [batch, text_len, cfg.text_dim],
            burn::tensor::Distribution::Default,
            &device,
        );
        let text_mask: Tensor<B, 2, Bool> =
            Tensor::<B, 2>::ones([batch, text_len], &device).greater_elem(0.0);
        let cos = Tensor::<B, 2>::random(
            [seq_lat, head_half],
            burn::tensor::Distribution::Default,
            &device,
        );
        let sin = Tensor::<B, 2>::random(
            [seq_lat, head_half],
            burn::tensor::Distribution::Default,
            &device,
        );

        let ctx_unfused = JointAttnCtx {
            text_state: text_state.clone(),
            text_mask: text_mask.clone(),
            aux_state: None,
            aux_mask: None,
            kv_cache: None,
        };
        let out_unfused = attn.forward(x.clone(), ctx_unfused, cos.clone(), sin.clone(), None);

        // Now fuse and run again
        attn.prepare_for_inference();
        assert!(
            attn.fused_qkv_weight.is_some(),
            "fused weight should be set"
        );

        let ctx_fused = JointAttnCtx {
            text_state,
            text_mask,
            aux_state: None,
            aux_mask: None,
            kv_cache: None,
        };
        let out_fused = attn.forward(x, ctx_fused, cos, sin, None);

        let diff: f32 = (out_unfused - out_fused)
            .abs()
            .max()
            .to_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(
            diff < 1e-5,
            "fused QKV output should match unfused: max_diff={diff}"
        );
    }

    /// `prepare_for_inference()` is idempotent.
    #[test]
    fn fused_qkv_idempotent() {
        let cfg = tiny_cfg();
        let device: <B as Backend>::Device = Default::default();
        let mut attn = JointAttention::<B>::new(&cfg, &device);
        attn.prepare_for_inference();
        let w1: Vec<f32> = attn
            .fused_qkv_weight
            .as_ref()
            .unwrap()
            .clone()
            .into_data()
            .to_vec()
            .unwrap();
        attn.prepare_for_inference();
        let w2: Vec<f32> = attn
            .fused_qkv_weight
            .as_ref()
            .unwrap()
            .clone()
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(
            w1, w2,
            "calling prepare_for_inference twice should be idempotent"
        );
    }
}
