use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Bool, Tensor, activation::softmax, backend::Backend},
};

use crate::config::ModelConfig;

use super::{
    norm::HeadRmsNorm,
    rope::{apply_rotary_emb, apply_rotary_half},
};

/// K/V projections for auxiliary conditioning (speaker or caption).
/// Cached KV projections for conditional contexts (text + optional speaker/caption).
///
/// Kept outside the Module system because caches are runtime state, not learned parameters.
pub struct CondKvCache<B: Backend> {
    pub(crate) text_k: Tensor<B, 4>, // [B, T, H, D_h]
    pub(crate) text_v: Tensor<B, 4>,
    pub(crate) aux_k: Option<Tensor<B, 4>>, // speaker or caption
    pub(crate) aux_v: Option<Tensor<B, 4>>,
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
    pub(crate) q_norm: HeadRmsNorm<B>,
    pub(crate) k_norm: HeadRmsNorm<B>,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl<B: Backend> SelfAttention<B> {
    pub fn new(dim: usize, num_heads: usize, head_dim: Option<usize>, device: &B::Device) -> Self {
        let head_dim = head_dim.unwrap_or(dim / num_heads);
        let kv_dim = num_heads * head_dim;
        let scale = (head_dim as f64).powf(-0.5);

        Self {
            wq: LinearConfig::new(dim, kv_dim).with_bias(false).init(device),
            wk: LinearConfig::new(dim, kv_dim).with_bias(false).init(device),
            wv: LinearConfig::new(dim, kv_dim).with_bias(false).init(device),
            wo: LinearConfig::new(kv_dim, dim).with_bias(false).init(device),
            q_norm: HeadRmsNorm::new(num_heads, head_dim, 1e-6, device),
            k_norm: HeadRmsNorm::new(num_heads, head_dim, 1e-6, device),
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

        let q = self.project_qkv(self.wq.forward(x.clone()), batch, seq);
        let k = self.project_qkv(self.wk.forward(x.clone()), batch, seq);
        let v = self.project_qkv(self.wv.forward(x), batch, seq);

        let q = self.q_norm.forward(q);
        let k = self.k_norm.forward(k);

        // Apply full RoPE to Q and K
        let q = apply_rotary_emb(q, cos.clone(), sin.clone());
        let k = apply_rotary_emb(k, cos, sin);

        let out = scaled_dot_product_attention(q, k, v, mask, self.scale);
        // [B, S, H, D_h] → [B, S, H*D_h]
        let out = out
            .swap_dims(1, 2) // [B, H, S, D_h]
            .reshape([batch, seq, self.num_heads * self.head_dim]);
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
            gate: LinearConfig::new(kv_dim, kv_dim)
                .with_bias(true)
                .init(device),
            wo: LinearConfig::new(kv_dim, dim).with_bias(false).init(device),
            q_norm: HeadRmsNorm::new(num_heads, head_dim, 1e-6, device),
            k_norm: HeadRmsNorm::new(num_heads, head_dim, 1e-6, device),
            num_heads,
            head_dim,
            scale,
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
    ) -> Tensor<B, 3> {
        let [batch, seq_lat, _dim] = x.dims();
        let device = x.device();

        let q = self
            .wq
            .forward(x.clone())
            .reshape([batch, seq_lat, self.num_heads, self.head_dim]);
        let q = self.q_norm.forward(q);
        let q = apply_rotary_half(q, cos, sin); // half-RoPE on first H/2 heads

        // Self K/V
        let k_self =
            self.wk
                .forward(x.clone())
                .reshape([batch, seq_lat, self.num_heads, self.head_dim]);
        let v_self = self
            .wv
            .forward(x)
            .reshape([batch, seq_lat, self.num_heads, self.head_dim]);
        let k_self = self.k_norm.forward(k_self);

        // Context K/V from cache or projection
        let (k_text, v_text, k_aux, v_aux, text_mask_full, aux_mask_full) = if let Some(cache) =
            ctx.kv_cache
        {
            (
                cache.text_k.clone(),
                cache.text_v.clone(),
                cache.aux_k.clone(),
                cache.aux_v.clone(),
                ctx.text_mask,
                ctx.aux_mask,
            )
        } else {
            let [_, seq_txt, _] = ctx.text_state.dims();
            let k_text = self.wk_text.forward(ctx.text_state.clone()).reshape([
                batch,
                seq_txt,
                self.num_heads,
                self.head_dim,
            ]);
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
                    let v =
                        wv.forward(aux)
                            .reshape([batch, seq_aux, self.num_heads, self.head_dim]);
                    (Some(k), Some(v))
                }
                _ => (None, None),
            };

            (k_text, v_text, k_aux, v_aux, ctx.text_mask, ctx.aux_mask)
        };

        // Concatenate K/V along sequence dimension: [self | text | aux?]
        let k_ctx = match k_aux {
            Some(ref ka) => Tensor::cat(vec![k_text, ka.clone()], 1),
            None => k_text,
        };
        let v_ctx = match v_aux {
            Some(ref va) => Tensor::cat(vec![v_text, va.clone()], 1),
            None => v_text,
        };
        let ctx_mask = match aux_mask_full {
            Some(am) => Some(Tensor::cat(vec![text_mask_full, am], 1)),
            None => Some(text_mask_full),
        };

        // Full K: [self | context]
        let k_all = Tensor::cat(vec![k_self, k_ctx], 1);
        let v_all = Tensor::cat(vec![v_self, v_ctx], 1);

        // Build query-context mask: query positions always attend to self (no self-mask)
        // then restricted by ctx_mask for context positions.
        let mask = build_joint_mask(seq_lat, ctx_mask, batch, &device);

        let out = scaled_dot_product_attention(q, k_all, v_all, mask, self.scale);
        // [B, S_lat, H, D_h] → [B, S_lat, H*D_h]
        let out = out
            .swap_dims(1, 2)
            .reshape([batch, seq_lat, self.num_heads * self.head_dim]);

        // Gated output
        let gated = burn::tensor::activation::sigmoid(self.gate.forward(out.clone())) * out;
        self.wo.forward(gated)
    }

    /// Build the KV cache for a given context (used during fast sampling).
    pub fn build_kv_cache(
        &self,
        text_state: Tensor<B, 3>,
        aux_state: Option<Tensor<B, 3>>,
    ) -> CondKvCache<B> {
        let [batch, seq_txt, _] = text_state.dims();
        let k_text = self.wk_text.forward(text_state.clone()).reshape([
            batch,
            seq_txt,
            self.num_heads,
            self.head_dim,
        ]);
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
                let v = wv
                    .forward(aux)
                    .reshape([batch, seq_aux, self.num_heads, self.head_dim]);
                (Some(k), Some(v))
            }
            _ => (None, None),
        };

        CondKvCache {
            text_k: k_text,
            text_v: v_text,
            aux_k,
            aux_v,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Scaled dot-product attention. `q/k/v: [B, S, H, D_h]`.
///
/// mask (optional): `[B, S_kv]` — True = valid (attended), False = masked out.
/// Returns `[B, S_q, H, D_h]`.
fn scaled_dot_product_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<Tensor<B, 2, Bool>>,
    scale: f64,
) -> Tensor<B, 4> {
    let [_batch, _seq_q, _num_heads, _head_dim] = q.dims();
    let [_, _seq_k, _, _] = k.dims();

    // Rearrange to [B, H, S, D_h] for batched matmul
    let q = q.swap_dims(1, 2); // [B, H, S_q, D_h]
    let k = k.swap_dims(1, 2); // [B, H, S_k, D_h]
    let v = v.swap_dims(1, 2); // [B, H, S_k, D_h]

    // attn = Q K^T / sqrt(d_k): [B, H, S_q, S_k]
    let attn = q.matmul(k.swap_dims(2, 3)) * scale as f32;

    // Apply mask: positions where mask=False get -inf
    let attn = if let Some(m) = mask {
        // m: [B, S_k] → [B, 1, 1, S_k]
        let m3: Tensor<B, 3, Bool> = m.unsqueeze_dim::<3>(1); // [B, 1, S_k]
        let m4: Tensor<B, 4, Bool> = m3.unsqueeze_dim::<4>(2); // [B, 1, 1, S_k]
        attn.mask_fill(m4.bool_not(), f32::NEG_INFINITY)
    } else {
        attn
    };

    let attn = softmax(attn, 3); // [B, H, S_q, S_k]

    // Output: [B, H, S_q, D_h]
    let out = attn.matmul(v);
    // Rearrange back to [B, S_q, H, D_h]
    out.swap_dims(1, 2)
}

/// Build a mask for joint attention: query can attend everywhere in self,
/// and to valid positions in context.
///
/// Returns `Option<Tensor<B, 2, Bool>>` of shape `[B, S_lat + S_ctx]`
/// where the first `S_lat` positions are always True.
fn build_joint_mask<B: Backend>(
    seq_lat: usize,
    ctx_mask: Option<Tensor<B, 2, Bool>>,
    batch: usize,
    device: &B::Device,
) -> Option<Tensor<B, 2, Bool>> {
    ctx_mask.map(|cm| {
        // All-true mask for self (latent) positions
        let self_mask: Tensor<B, 2, Bool> =
            Tensor::<B, 2>::ones([batch, seq_lat], device).greater_elem(0.0);
        Tensor::cat(vec![self_mask, cm], 1)
    })
}
