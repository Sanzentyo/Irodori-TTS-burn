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
    linear_ops::linear_rank3_flattened,
    norm::HeadRmsNorm,
    rope::{apply_rotary_emb, apply_rotary_half},
};

/// Location of speaker tokens inside packed context K/V tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SpeakerKvRange {
    start: usize,
    end: usize,
}

impl SpeakerKvRange {
    pub(crate) fn from_start_len(start: usize, len: usize) -> Self {
        let end = start
            .checked_add(len)
            .expect("speaker KV range must fit in usize");
        Self { start, end }
    }

    pub(crate) const fn start(self) -> usize {
        self.start
    }

    pub(crate) const fn end(self) -> usize {
        self.end
    }

    pub(crate) const fn len(self) -> usize {
        self.end - self.start
    }
}

/// Packed KV projections for conditional contexts (text + optional speaker/caption).
///
/// Kept outside the Module system because caches are runtime state, not learned parameters.
/// `ctx_k/ctx_v/ctx_mask` use the canonical `[text | speaker? | caption?]` order.
/// Only the packed tensors are retained, avoiding both hot-path concatenation and a
/// second long-lived copy of every split context projection.
pub struct CondKvCache<B: Backend> {
    /// Pre-concatenated `[text | speaker? | caption?]` K.
    pub(crate) ctx_k: Tensor<B, 4>,
    /// Pre-concatenated `[text | speaker? | caption?]` V.
    pub(crate) ctx_v: Tensor<B, 4>,
    /// Pre-concatenated `[text_mask | speaker_mask? | caption_mask?]`.
    ///
    /// Never `None` because text conditioning is always present; the mask at minimum
    /// equals `text_mask`.
    pub(crate) ctx_mask: Tensor<B, 2, Bool>,
    /// Pre-built joint mask `[latent_ones | ctx_mask]`: `[B, S_lat + T_ctx]`.
    ///
    /// Avoids repeated `build_joint_mask` calls (allocating `ones → bool → cat`)
    /// for each layer of each forward pass during the sampling loop.
    /// Set once via [`CondKvCache::precompute_joint_mask`]; when present
    /// [`JointAttention::forward`] skips `build_joint_mask` entirely.
    pub(crate) joint_mask: Option<Tensor<B, 2, Bool>>,
    /// Speaker token range within `ctx_k` and `ctx_v`, if speaker conditioning exists.
    pub(crate) speaker_range: Option<SpeakerKvRange>,
    /// WGPU-only contiguous `[K | V]` view used by the direct packed-K/V
    /// materialization shader.
    ///
    /// This runtime allocation is populated only by
    /// [`WgslInferenceOptimizedModel`](super::WgslInferenceOptimizedModel).
    /// Portable cache construction and every ordinary/training forward leave it
    /// absent and keep the existing concatenation path unchanged.
    pub(crate) packed_ctx_kv_wgsl: Option<Tensor<B, 5>>,
    /// WGPU-only mask policy proven while constructing an exact text CFG cache.
    ///
    /// `None` preserves the ordinary valid-mask convention and all existing
    /// fallback behavior. Exact derived caches can either omit an all-valid B1
    /// mask or provide one shared B2 mask already expressed in CubeCL's
    /// `true = masked out` convention, avoiding a per-forward `bool_not`.
    pub(crate) joint_mask_wgsl: Option<WgslJointMask<B>>,
    /// WGPU-only persistent f32 mask (`1 = attend`) for the native SDPA path.
    ///
    /// Stored beside the boolean Burn mask so the denoising loop never casts,
    /// concatenates, or reads a mask through the CPU for each layer/step.
    pub(crate) joint_attend_mask_wgsl: Option<Tensor<B, 2>>,
}

pub(crate) enum WgslJointMask<B: Backend> {
    AllValid,
    MaskedOut(Tensor<B, 2, Bool>),
}

/// Conditional and batched-CFG cache sets produced as one atomic operation.
pub(crate) type TextCfgKvCachePair<B> = (Vec<CondKvCache<B>>, Vec<CondKvCache<B>>);

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
    /// Combined row-major `[dim, 4 * dim]` QKV+gate inference cache.
    ///
    /// This replaces the former `[dim, 3 * dim]` QKV cache; both are never
    /// retained. At the f32 v4-Small shape (`dim=1280`) it occupies 25 MiB,
    /// a steady increase of exactly 6.25 MiB per layer over that old cache,
    /// or 75 MiB for the pinned 12-layer checkpoint. Runtime layer count is
    /// still taken from the loaded config; 12 is not hard-coded here.
    #[module(skip)]
    combined_qkv_gate_weight: Option<Tensor<B, 2>>,
    /// Row-major `wo` inference cache selected only for `B=1`.
    ///
    /// The v4 checkpoint loader exposes `wo` as a column-major logical view.
    /// On RTX 3060 Ti, a one-time contiguous pack made the exact B1/S50 GEMM
    /// 1.195x faster, while B2 regressed slightly and therefore keeps using
    /// the source weight. Co-retaining this cache costs 6.25 MiB per layer
    /// (75 MiB for 12 layers); the measured 12-layer pack was 3.959 ms and
    /// amortised after four fixed-replay requests. It remains skipped from
    /// records and device moves like the combined QKV+gate cache.
    #[module(skip)]
    packed_wo_weight: Option<Tensor<B, 2>>,
    /// Contiguous `[Q norm | K norm]` WGPU inference cache.
    ///
    /// The direct K/V shader is limited to WebGPU's guaranteed eight storage
    /// bindings, so these two tiny learned tensors share one binding. This is
    /// prepared only for the explicit WGSL wrapper and skipped from records and
    /// device moves like the other inference caches.
    #[module(skip)]
    packed_qk_norm_weight_wgsl: Option<Tensor<B, 3>>,
}

/// Gate source carried through the shared attention tail.
///
/// The ordinary path defers its existing gate linear until after SDPA. Fused
/// inference passes the sigmoid-transformed segment of the combined projection
/// and therefore does not dispatch the gate linear again.
enum JointAttentionGate<B: Backend> {
    Unprojected(Tensor<B, 3>),
    Projected(Tensor<B, 3>),
}

/// Tensor-level result of the exact WGPU direct packed-K/V selector.
struct WgslDirectMaterialization {
    q: Tensor<crate::WgpuRaw, 4>,
    k_all: Tensor<crate::WgpuRaw, 4>,
    v_all: Tensor<crate::WgpuRaw, 4>,
    combined: Tensor<crate::WgpuRaw, 3>,
}

fn native_sdpa_config_for_sequence(
    sequence: usize,
) -> Option<crate::kernels::fused_sdpa_native::NativeFaConfig> {
    use crate::kernels::fused_sdpa_native::NativeFaConfig;
    match sequence {
        13 => Some(NativeFaConfig::Q16_KV16),
        25 | 50 => Some(NativeFaConfig::Q8_KV32),
        _ => None,
    }
}

/// Bundled context inputs for [`JointAttention::forward`].
///
/// Groups text + optional auxiliary conditioning and the optional KV cache
/// into a single struct, eliminating the long argument list.
pub(crate) struct JointAttnCtx<'a, B: Backend> {
    pub(crate) text_state: Tensor<B, 3>,
    pub(crate) text_mask: Tensor<B, 2, Bool>,
    pub(crate) speaker_state: Option<Tensor<B, 3>>,
    pub(crate) speaker_mask: Option<Tensor<B, 2, Bool>>,
    pub(crate) caption_state: Option<Tensor<B, 3>>,
    pub(crate) caption_mask: Option<Tensor<B, 2, Bool>>,
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

        let (wk_speaker, wv_speaker) = if cfg.use_speaker_condition() {
            let sp_dim = cfg.speaker_dim.unwrap_or(cfg.model_dim);
            (Some(mk_proj(sp_dim)), Some(mk_proj(sp_dim)))
        } else {
            (None, None)
        };
        let (wk_caption, wv_caption) = if cfg.use_caption_condition {
            let cap_dim = cfg.caption_dim();
            (Some(mk_proj(cap_dim)), Some(mk_proj(cap_dim)))
        } else {
            (None, None)
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
            combined_qkv_gate_weight: None,
            packed_wo_weight: None,
            packed_qk_norm_weight_wgsl: None,
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
        let kv_dim = self.num_heads * self.head_dim;

        // Inference uses one combined QKV+gate projection. The ordinary path
        // remains the three learned linears plus the deferred learned gate.
        let (q, k_self, v_self, gate) = if self.combined_qkv_gate_weight.is_some() {
            let combined_w = self.validated_combined_weight(&x, "JointAttention::forward");
            self.compute_qkv_gate_from_combined(x, combined_w, batch, seq_lat, kv_dim)
        } else {
            let gate_input = x.clone();
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
            (q, k, v, JointAttentionGate::Unprojected(gate_input))
        };

        let gated =
            self.attention_after_qkv_gated(q, k_self, v_self, gate, ctx, cos, sin, latent_mask);
        self.wo.forward(gated)
    }

    /// Branch-free forward using the combined QKV+gate weight matrix.
    ///
    /// # Panics
    ///
    /// Panics if [`prepare_for_inference`](Self::prepare_for_inference) has not
    /// been called (i.e. `combined_qkv_gate_weight` is `None`).
    pub(crate) fn forward_fused(
        &self,
        x: Tensor<B, 3>,
        ctx: JointAttnCtx<'_, B>,
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
        latent_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_lat, _dim] = x.dims();
        let kv_dim = self.num_heads * self.head_dim;

        let combined_w = self.validated_combined_weight(&x, "JointAttention::forward_fused");
        let (q, k_self, v_self, gate) =
            self.compute_qkv_gate_from_combined(x, combined_w, batch, seq_lat, kv_dim);

        let gated =
            self.attention_after_qkv_gated(q, k_self, v_self, gate, ctx, cos, sin, latent_mask);
        self.project_wo_flattened(gated)
    }

    /// Project and split `[D, 4*kv_dim]` into Q/K/V plus sigmoid(gate).
    fn compute_qkv_gate_from_combined(
        &self,
        x: Tensor<B, 3>,
        combined_w: &Tensor<B, 2>,
        batch: usize,
        seq_lat: usize,
        kv_dim: usize,
    ) -> (
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        JointAttentionGate<B>,
    ) {
        let combined = linear_rank3_flattened(x, combined_w.clone(), None);
        let q = combined.clone().narrow(2, 0, kv_dim).reshape([
            batch,
            seq_lat,
            self.num_heads,
            self.head_dim,
        ]);
        let k = combined.clone().narrow(2, kv_dim, kv_dim).reshape([
            batch,
            seq_lat,
            self.num_heads,
            self.head_dim,
        ]);
        let v = combined.clone().narrow(2, 2 * kv_dim, kv_dim).reshape([
            batch,
            seq_lat,
            self.num_heads,
            self.head_dim,
        ]);
        let gate = burn::tensor::activation::sigmoid(combined.narrow(2, 3 * kv_dim, kv_dim));
        (q, k, v, JointAttentionGate::Projected(gate))
    }

    /// Return the combined cache only after validating its hot-path contract.
    #[track_caller]
    fn validated_combined_weight(&self, x: &Tensor<B, 3>, caller: &str) -> &Tensor<B, 2> {
        let weight = self
            .combined_qkv_gate_weight
            .as_ref()
            .unwrap_or_else(|| {
                panic!(
                    "{caller} called without combined QKV+gate weight — call prepare_for_inference first"
                )
            });
        let [batch, seq_len, input_dim] = x.dims();
        let kv_dim = self
            .num_heads
            .checked_mul(self.head_dim)
            .expect("joint-attention H * Dh overflow");
        let combined_dim = kv_dim
            .checked_mul(4)
            .expect("joint-attention combined projection width overflow");
        assert!(
            batch > 0 && seq_len > 0 && input_dim > 0,
            "{caller} requires non-empty [B,S,D], got {:?}",
            [batch, seq_len, input_dim]
        );
        assert_eq!(
            weight.dims(),
            [input_dim, combined_dim],
            "{caller} combined QKV+gate cache shape mismatch"
        );
        assert_eq!(
            weight.device(),
            x.device(),
            "{caller} combined QKV+gate cache is on the wrong device (was the model moved after prepare_for_inference()?)"
        );
        weight
    }

    /// Apply the prepared inference output projection as one rank-2 GEMM.
    ///
    /// Callers are restricted to branch-free prepared inference paths. The
    /// ordinary forward path deliberately retains [`Linear::forward`] so
    /// training and unprepared execution preserve their existing behavior.
    fn project_wo_flattened(&self, gated: Tensor<B, 3>) -> Tensor<B, 3> {
        let batch = gated.dims()[0];
        let weight = if batch == 1 {
            self.validated_packed_wo_weight(&gated).clone()
        } else {
            self.wo.weight.val()
        };
        linear_rank3_flattened(gated, weight, self.wo.bias.as_ref().map(|bias| bias.val()))
    }

    /// Return the B1 row-major cache only after validating its inference
    /// contract against the current input and learned source weight.
    #[track_caller]
    fn validated_packed_wo_weight(&self, input: &Tensor<B, 3>) -> &Tensor<B, 2> {
        let packed = self.packed_wo_weight.as_ref().unwrap_or_else(|| {
            panic!("B1 fused wo called without row-major cache — call prepare_for_inference first")
        });
        let [batch, sequence, input_dim] = input.dims();
        assert_eq!(batch, 1, "row-major wo cache is specialised for B=1");
        assert!(
            sequence > 0 && input_dim > 0,
            "row-major wo cache requires non-empty [B,S,D], got {:?}",
            [batch, sequence, input_dim]
        );
        let source_shape = self.wo.weight.dims();
        assert_eq!(
            packed.dims(),
            source_shape,
            "packed wo cache shape mismatch"
        );
        assert_eq!(
            source_shape[0], input_dim,
            "packed wo cache input width mismatch"
        );
        assert_eq!(
            packed.device(),
            input.device(),
            "packed wo cache is on the wrong device (was the model moved after prepare_for_inference()?)"
        );
        packed
    }

    /// Shared attention logic after Q/K/V have been computed, up to the
    /// gated tensor immediately before the output projection.
    ///
    /// Applies head norms, half-RoPE, context KV projection or cache lookup,
    /// SDPA, and the sigmoid output gate.
    #[allow(clippy::too_many_arguments)]
    fn attention_after_qkv_gated(
        &self,
        q: Tensor<B, 4>,
        k_self: Tensor<B, 4>,
        v_self: Tensor<B, 4>,
        gate: JointAttentionGate<B>,
        ctx: JointAttnCtx<'_, B>,
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
        latent_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let q = self.q_norm.forward(q);
        let q = apply_rotary_half(q, cos.clone(), sin.clone());
        let k_self = self.k_norm.forward(k_self);
        let k_self = apply_rotary_half(k_self, cos, sin);

        self.attention_after_processed_qkv_gated(q, k_self, v_self, gate, ctx, latent_mask)
    }

    /// Shared attention logic after head normalisation and half-RoPE, up to
    /// the gated tensor immediately before the output projection.
    fn attention_after_processed_qkv_gated(
        &self,
        q: Tensor<B, 4>,
        k_self: Tensor<B, 4>,
        v_self: Tensor<B, 4>,
        gate: JointAttentionGate<B>,
        ctx: JointAttnCtx<'_, B>,
        latent_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_lat, _, _] = q.dims();
        let device = q.device();

        let (k_all, v_all, mask) =
            self.assemble_kv_and_mask(k_self, v_self, ctx, latent_mask, batch, seq_lat, &device);
        let out = scaled_dot_product_attention(q, k_all, v_all, mask, self.scale, true);
        // out: [B, S_lat, H, D_h] → [B, S_lat, H*D_h]
        let out = out.reshape([batch, seq_lat, self.num_heads * self.head_dim]);

        self.apply_attention_gate(out, gate, batch, seq_lat)
    }

    /// Assemble context K/V and the joint mask without changing the existing
    /// portable/training execution policy.
    #[allow(clippy::too_many_arguments)]
    fn assemble_kv_and_mask(
        &self,
        k_self: Tensor<B, 4>,
        v_self: Tensor<B, 4>,
        ctx: JointAttnCtx<'_, B>,
        latent_mask: Option<Tensor<B, 2, Bool>>,
        batch: usize,
        seq_lat: usize,
        device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Option<Tensor<B, 2, Bool>>) {
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

            let (speaker_k, speaker_v) = self.project_optional_context(
                ctx.speaker_state,
                self.wk_speaker.as_ref(),
                self.wv_speaker.as_ref(),
                batch,
            );
            let (caption_k, caption_v) = self.project_optional_context(
                ctx.caption_state,
                self.wk_caption.as_ref(),
                self.wv_caption.as_ref(),
                batch,
            );

            let (k, v, m) = concat_all_ctx_kv(
                k_text,
                v_text,
                speaker_k,
                speaker_v,
                caption_k,
                caption_v,
                ctx.text_mask,
                ctx.speaker_mask,
                ctx.caption_mask,
            );
            (k, v, Some(m), None)
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
            .or_else(|| build_joint_mask(seq_lat, latent_mask, ctx_mask, batch, device));
        (k_all, v_all, mask)
    }

    fn apply_attention_gate(
        &self,
        out: Tensor<B, 3>,
        gate: JointAttentionGate<B>,
        batch: usize,
        seq_lat: usize,
    ) -> Tensor<B, 3> {
        let gate = match gate {
            JointAttentionGate::Unprojected(input) => {
                burn::tensor::activation::sigmoid(self.gate.forward(input))
            }
            JointAttentionGate::Projected(gate) => gate,
        };
        assert_eq!(
            gate.dims(),
            [batch, seq_lat, self.num_heads * self.head_dim],
            "joint-attention gate shape mismatch"
        );
        // Gated output: output * sigmoid(gate(x_input)), matching Python JointAttention.
        gate * out
    }

    /// Build the KV cache for a given context (used during fast sampling).
    ///
    /// Pre-concatenates `[text | aux?]` K/V and the combined mask so that
    /// [`Self::forward`] can use them directly without any `Tensor::cat` per step.
    pub fn build_kv_cache(
        &self,
        text_state: Tensor<B, 3>,
        text_mask: Tensor<B, 2, Bool>,
        speaker_state: Option<Tensor<B, 3>>,
        speaker_mask: Option<Tensor<B, 2, Bool>>,
        caption_state: Option<Tensor<B, 3>>,
        caption_mask: Option<Tensor<B, 2, Bool>>,
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

        let (speaker_k, speaker_v) = self.project_optional_context(
            speaker_state,
            self.wk_speaker.as_ref(),
            self.wv_speaker.as_ref(),
            batch,
        );
        let (caption_k, caption_v) = self.project_optional_context(
            caption_state,
            self.wk_caption.as_ref(),
            self.wv_caption.as_ref(),
            batch,
        );

        let speaker_range = speaker_k.as_ref().map(|k| {
            let seq_speaker = k.dims()[1];
            debug_assert_eq!(
                speaker_v.as_ref().map(|v| v.dims()[1]),
                Some(seq_speaker),
                "speaker K/V sequence lengths must match"
            );
            SpeakerKvRange::from_start_len(seq_txt, seq_speaker)
        });
        let (ctx_k, ctx_v, ctx_mask) = concat_all_ctx_kv(
            k_text,
            v_text,
            speaker_k,
            speaker_v,
            caption_k,
            caption_v,
            text_mask,
            speaker_mask,
            caption_mask,
        );

        CondKvCache {
            ctx_k,
            ctx_v,
            ctx_mask,
            joint_mask: None,
            speaker_range,
            packed_ctx_kv_wgsl: None,
            joint_mask_wgsl: None,
            joint_attend_mask_wgsl: None,
        }
    }

    fn project_optional_context(
        &self,
        state: Option<Tensor<B, 3>>,
        wk: Option<&Linear<B>>,
        wv: Option<&Linear<B>>,
        batch: usize,
    ) -> (Option<Tensor<B, 4>>, Option<Tensor<B, 4>>) {
        match (state, wk, wv) {
            (Some(state), Some(wk), Some(wv)) => {
                let [_, seq, _] = state.dims();
                let k =
                    wk.forward(state.clone())
                        .reshape([batch, seq, self.num_heads, self.head_dim]);
                let k = self.k_norm.forward(k);
                let v = wv
                    .forward(state)
                    .reshape([batch, seq, self.num_heads, self.head_dim]);
                (Some(k), Some(v))
            }
            (None, _, _) => (None, None),
            (Some(_), _, _) => {
                panic!("conditioning state supplied without matching projection weights")
            }
        }
    }

    /// Materialise the row-major QKV+gate and B1 `wo` inference caches.
    ///
    /// The four logical source weights are concatenated exactly once. This
    /// cache replaces (and does not retain) the former QKV-only packed tensor.
    /// `wo` is copied exactly once into a separate contiguous allocation while
    /// its learned source parameter remains available to B2+, ordinary
    /// forward, and training. Safe to call multiple times: repeated calls
    /// validate and reuse both allocations. Serialization is unchanged because
    /// both caches are skipped.
    ///
    /// # Safety invariant
    /// Must be called **after** final weights are loaded and device placement is
    /// complete. The cached tensors are `#[module(skip)]`, so they will NOT
    /// follow `to_device()` or `fork()` calls on the parent module.
    pub(crate) fn prepare_for_inference(&mut self) {
        let (expected_shape, expected_device) = self.validate_combined_source_weights();
        let (wo_shape, wo_device) = self.validate_wo_source_weight();
        assert_eq!(
            wo_device, expected_device,
            "JointAttention wo and QKV+gate weights must share one device"
        );

        if let Some(combined) = self.combined_qkv_gate_weight.as_ref() {
            assert_eq!(
                combined.dims(),
                expected_shape,
                "existing combined QKV+gate cache shape mismatch"
            );
            assert_eq!(
                combined.device(),
                expected_device,
                "existing combined QKV+gate cache is on the wrong device"
            );
        } else {
            // One allocation/materialisation: `[Wq | Wk | Wv | Wgate]`.
            let combined = Tensor::cat(
                vec![
                    self.wq.weight.val(),
                    self.wk.weight.val(),
                    self.wv.weight.val(),
                    self.gate.weight.val(),
                ],
                1,
            );
            assert_eq!(
                combined.dims(),
                expected_shape,
                "new combined QKV+gate cache shape mismatch"
            );
            assert_eq!(
                combined.device(),
                expected_device,
                "new combined QKV+gate cache is on the wrong device"
            );
            self.combined_qkv_gate_weight = Some(combined);
        }

        if let Some(packed) = self.packed_wo_weight.as_ref() {
            assert_eq!(
                packed.dims(),
                wo_shape,
                "existing packed wo cache shape mismatch"
            );
            assert_eq!(
                packed.device(),
                wo_device,
                "existing packed wo cache is on the wrong device"
            );
        } else {
            // Even a single-input cat allocates a new tensor and assigns the
            // logical source view into canonical row-major output storage.
            let packed = Tensor::cat(vec![self.wo.weight.val()], 0);
            assert_eq!(
                packed.dims(),
                wo_shape,
                "new packed wo cache shape mismatch"
            );
            assert_eq!(
                packed.device(),
                wo_device,
                "new packed wo cache is on the wrong device"
            );
            self.packed_wo_weight = Some(packed);
        }
    }

    /// Materialise the tiny Q/K RMSNorm binding required by the exact-shape
    /// WGPU direct-K/V kernel.
    ///
    /// Ordinary inference preparation deliberately does not call this method;
    /// only [`WgslInferenceOptimizedModel`](super::WgslInferenceOptimizedModel)
    /// opts into the additional allocation. Repeated calls validate and reuse
    /// the existing `#[module(skip)]` tensor. A stale shape or device therefore
    /// fails during preparation rather than reaching a raw shader launch.
    pub(crate) fn prepare_qk_norm_weight_wgsl(&mut self) {
        use crate::kernels::joint_attention_materialization::{HEAD_DIM, NUM_HEADS};

        let q_weight = self.q_norm.weight.val();
        let k_weight = self.k_norm.weight.val();
        let expected_shape = [2, self.num_heads, self.head_dim];
        let source_device = q_weight.device();
        assert_eq!(
            q_weight.dims(),
            [self.num_heads, self.head_dim],
            "Q norm source weight shape mismatch before WGPU packing"
        );
        assert_eq!(
            k_weight.dims(),
            [self.num_heads, self.head_dim],
            "K norm source weight shape mismatch before WGPU packing"
        );
        assert_eq!(
            k_weight.device(),
            source_device,
            "Q/K norm source weights must share one device before WGPU packing"
        );

        if let Some(packed) = self.packed_qk_norm_weight_wgsl.as_ref() {
            assert_eq!(
                packed.dims(),
                expected_shape,
                "existing packed Q/K norm WGPU cache shape mismatch"
            );
            assert_eq!(
                packed.device(),
                source_device,
                "existing packed Q/K norm WGPU cache is on the wrong device"
            );
            return;
        }

        if self.num_heads != NUM_HEADS || self.head_dim != HEAD_DIM {
            return;
        }
        let packed = Tensor::<B, 2>::stack::<3>(vec![q_weight, k_weight], 0);
        assert_eq!(
            packed.dims(),
            expected_shape,
            "new packed Q/K norm WGPU cache shape mismatch"
        );
        assert_eq!(
            packed.device(),
            source_device,
            "new packed Q/K norm WGPU cache is on the wrong device"
        );
        self.packed_qk_norm_weight_wgsl = Some(packed);
    }

    /// Validate the learned `wo` parameter before creating or reusing its
    /// row-major inference cache.
    fn validate_wo_source_weight(&self) -> ([usize; 2], B::Device) {
        assert!(
            self.wo.bias.is_none(),
            "JointAttention wo bias must be absent before row-major packing"
        );
        let weight = self.wo.weight.val();
        let kv_dim = self
            .num_heads
            .checked_mul(self.head_dim)
            .expect("joint-attention H * Dh overflow");
        let expected_shape = [kv_dim, kv_dim];
        assert_eq!(
            weight.dims(),
            expected_shape,
            "JointAttention wo weight shape mismatch"
        );
        (expected_shape, weight.device())
    }

    /// Validate every source tensor before a bias-free fusion is materialised.
    fn validate_combined_source_weights(&self) -> ([usize; 2], B::Device) {
        for (name, bias) in [
            ("wq", self.wq.bias.as_ref()),
            ("wk", self.wk.bias.as_ref()),
            ("wv", self.wv.bias.as_ref()),
            ("gate", self.gate.bias.as_ref()),
        ] {
            assert!(
                bias.is_none(),
                "JointAttention {name} bias must be absent before combined QKV+gate fusion"
            );
        }

        let weights = [
            ("wq", self.wq.weight.val()),
            ("wk", self.wk.weight.val()),
            ("wv", self.wv.weight.val()),
            ("gate", self.gate.weight.val()),
        ];
        let [input_dim, output_dim] = weights[0].1.dims();
        let kv_dim = self
            .num_heads
            .checked_mul(self.head_dim)
            .expect("joint-attention H * Dh overflow");
        assert!(
            input_dim > 0 && kv_dim > 0,
            "combined QKV+gate fusion requires non-zero dimensions"
        );
        assert_eq!(
            output_dim, kv_dim,
            "wq output width must equal num_heads * head_dim"
        );
        assert_eq!(
            input_dim, kv_dim,
            "combined QKV+gate fusion requires D == num_heads * head_dim"
        );
        let expected_weight_shape = [input_dim, kv_dim];
        let expected_device = weights[0].1.device();
        for (name, weight) in &weights {
            assert_eq!(
                weight.dims(),
                expected_weight_shape,
                "JointAttention {name} weight shape mismatch"
            );
            assert_eq!(
                weight.device(),
                expected_device,
                "JointAttention {name} weight is on the wrong device"
            );
        }
        assert_eq!(
            self.q_norm.weight.dims(),
            [self.num_heads, self.head_dim],
            "q_norm weight shape mismatch"
        );
        assert_eq!(
            self.k_norm.weight.dims(),
            [self.num_heads, self.head_dim],
            "k_norm weight shape mismatch"
        );
        assert_eq!(
            self.q_norm.epsilon(),
            self.k_norm.epsilon(),
            "Q/K head RMSNorm epsilons must match for fused post-processing"
        );
        let combined_dim = kv_dim
            .checked_mul(4)
            .expect("joint-attention combined projection width overflow");
        ([input_dim, combined_dim], expected_device)
    }
}

impl JointAttention<crate::WgpuRaw> {
    /// Combined QKV+gate path with shape-checked WGSL materialization.
    ///
    /// The tuned rank-2 projection and CubeCL SDPA remain unchanged. Exact
    /// B1/B2, positive S, H20, Dh64, ctx3 inputs use direct packed K/V
    /// construction and the post-SDPA layout+gate epilogue. Every selector is
    /// fail-closed: any
    /// unsupported shape, layout, device, binding count, or hardware limit
    /// continues through the previously accepted QKV-postprocess, K/V-cat, and
    /// reshape+gate operations.
    pub(crate) fn forward_fused_wgsl(
        &self,
        x: Tensor<crate::WgpuRaw, 3>,
        ctx: JointAttnCtx<'_, crate::WgpuRaw>,
        cos: Tensor<crate::WgpuRaw, 2>,
        sin: Tensor<crate::WgpuRaw, 2>,
        latent_mask: Option<Tensor<crate::WgpuRaw, 2, Bool>>,
    ) -> Tensor<crate::WgpuRaw, 3> {
        use burn::tensor::TensorPrimitive;

        let kv_dim = self
            .num_heads
            .checked_mul(self.head_dim)
            .expect("joint-attention H * Dh overflow");
        let combined_w = self.validated_combined_weight(&x, "JointAttention::forward_fused_wgsl");
        let packed = combined_w.clone().into_primitive().tensor();
        assert!(
            packed.is_contiguous(),
            "WGSL combined QKV+gate cache must be row-major contiguous"
        );
        assert_eq!(
            &packed.meta.strides()[..],
            &[4 * kv_dim, 1],
            "WGSL combined QKV+gate cache must have row-major strides"
        );

        let [batch, seq_lat, _] = x.dims();
        let combined = linear_rank3_flattened(x, combined_w.clone(), None);
        assert_eq!(self.q_norm.epsilon(), self.k_norm.epsilon());
        let direct = self.try_direct_packed_kv(&combined, &ctx, &cos, &sin);
        let (
            q_head_major,
            k_head_major,
            v_head_major,
            combined,
            mask,
            mask_is_backend_native,
            attend_mask_wgsl,
        ) = if let Some(direct) = direct {
            let cache = ctx
                .kv_cache
                .expect("direct packed K/V selection requires a conditional cache");
            assert!(
                cache.joint_mask.is_none() || latent_mask.is_none(),
                "cached joint_mask is incompatible with a non-None latent_mask: \
                 the cached mask was built assuming all latent positions attend"
            );
            let device = direct.q.device();
            let (mask, mask_is_backend_native) = if latent_mask.is_none() {
                match cache.joint_mask_wgsl.as_ref() {
                    Some(WgslJointMask::AllValid) => (None, true),
                    Some(WgslJointMask::MaskedOut(mask)) => (Some(mask.clone()), true),
                    None => (
                        cache.joint_mask.clone().or_else(|| {
                            build_joint_mask(
                                seq_lat,
                                None,
                                Some(cache.ctx_mask.clone()),
                                batch,
                                &device,
                            )
                        }),
                        false,
                    ),
                }
            } else {
                (
                    build_joint_mask(
                        seq_lat,
                        latent_mask,
                        Some(cache.ctx_mask.clone()),
                        batch,
                        &device,
                    ),
                    false,
                )
            };
            (
                direct.q,
                direct.k_all,
                direct.v_all,
                direct.combined,
                mask,
                mask_is_backend_native,
                cache.joint_attend_mask_wgsl.clone(),
            )
        } else {
            let output = crate::kernels::qkv_postprocess::fused_qkv_gate_postprocess_wgsl(
                combined.into_primitive().tensor(),
                self.q_norm.weight.val().into_primitive().tensor(),
                self.k_norm.weight.val().into_primitive().tensor(),
                cos.into_primitive().tensor(),
                sin.into_primitive().tensor(),
                self.q_norm.epsilon(),
            );
            let q =
                Tensor::<crate::WgpuRaw, 4>::from_primitive(TensorPrimitive::Float(output.qkv.q));
            let k_self =
                Tensor::<crate::WgpuRaw, 4>::from_primitive(TensorPrimitive::Float(output.qkv.k));
            let v_self =
                Tensor::<crate::WgpuRaw, 4>::from_primitive(TensorPrimitive::Float(output.qkv.v));
            let combined = Tensor::<crate::WgpuRaw, 3>::from_primitive(TensorPrimitive::Float(
                output.combined,
            ));
            let device = q.device();
            let (k_all, v_all, mask) = self.assemble_kv_and_mask(
                k_self,
                v_self,
                ctx,
                latent_mask,
                batch,
                seq_lat,
                &device,
            );
            (
                q.swap_dims(1, 2),
                k_all.swap_dims(1, 2),
                v_all.swap_dims(1, 2),
                combined,
                mask,
                false,
                None,
            )
        };

        let attention = self
            .try_native_sdpa_wgsl(
                &q_head_major,
                &k_head_major,
                &v_head_major,
                attend_mask_wgsl,
                seq_lat,
            )
            .unwrap_or_else(|| {
                scaled_dot_product_attention_prepared_head_major_with_mask_convention(
                    q_head_major,
                    k_head_major,
                    v_head_major,
                    mask,
                    self.scale,
                    mask_is_backend_native,
                )
            });
        let gated = self
            .try_post_sdpa_layout_gate(&attention, &combined)
            .unwrap_or_else(|| {
                let out = attention.swap_dims(1, 2).reshape([batch, seq_lat, kv_dim]);
                let gate = combined.narrow(2, 3 * kv_dim, kv_dim);
                self.apply_attention_gate(out, JointAttentionGate::Projected(gate), batch, seq_lat)
            });
        self.assert_b1_packed_wo_row_major(&gated);
        self.project_wo_flattened(gated)
    }

    fn try_native_sdpa_wgsl(
        &self,
        q: &Tensor<crate::WgpuRaw, 4>,
        k: &Tensor<crate::WgpuRaw, 4>,
        v: &Tensor<crate::WgpuRaw, 4>,
        attend_mask: Option<Tensor<crate::WgpuRaw, 2>>,
        sequence: usize,
    ) -> Option<Tensor<crate::WgpuRaw, 4>> {
        use crate::kernels::fused_sdpa_native::{
            native_fa_sdpa_wgsl, supports_native_fa_sdpa_wgsl,
        };
        use burn::tensor::TensorPrimitive;

        let config = native_sdpa_config_for_sequence(sequence)?;
        let attend_mask = attend_mask?;
        let q_primitive = q.clone().into_primitive().tensor();
        let k_primitive = k.clone().into_primitive().tensor();
        let v_primitive = v.clone().into_primitive().tensor();
        let mask_primitive = attend_mask.into_primitive().tensor();
        if !supports_native_fa_sdpa_wgsl(
            &q_primitive,
            &k_primitive,
            &v_primitive,
            &mask_primitive,
            self.scale,
            &config,
        ) {
            return None;
        }
        let output = native_fa_sdpa_wgsl(
            q_primitive,
            k_primitive,
            v_primitive,
            mask_primitive,
            self.scale,
            &config,
        );
        Some(Tensor::from_primitive(TensorPrimitive::Float(output)))
    }

    /// Select the direct K/V kernel without consuming the fallback inputs.
    #[allow(clippy::too_many_arguments)]
    fn try_direct_packed_kv(
        &self,
        combined: &Tensor<crate::WgpuRaw, 3>,
        ctx: &JointAttnCtx<'_, crate::WgpuRaw>,
        cos: &Tensor<crate::WgpuRaw, 2>,
        sin: &Tensor<crate::WgpuRaw, 2>,
    ) -> Option<WgslDirectMaterialization> {
        use crate::kernels::joint_attention_materialization::{
            CONTEXT_LEN, HEAD_DIM, NUM_HEADS, direct_packed_kv_wgsl, supports_direct_packed_kv,
        };
        use burn::tensor::TensorPrimitive;

        let packed_qk = self.packed_qk_norm_weight_wgsl.as_ref()?;
        let cache = ctx.kv_cache?;
        let packed_ctx = cache.packed_ctx_kv_wgsl.as_ref()?;
        let [batch, seq_lat, _] = combined.dims();
        let total_kv_len = seq_lat.checked_add(CONTEXT_LEN)?;
        let device = combined.device();
        let joint_mask_valid = cache
            .joint_mask
            .as_ref()
            .is_none_or(|mask| mask.dims() == [batch, total_kv_len] && mask.device() == device);
        let wgsl_mask_valid = match cache.joint_mask_wgsl.as_ref() {
            None => true,
            Some(WgslJointMask::AllValid) => batch == 1 && cache.joint_mask.is_none(),
            Some(WgslJointMask::MaskedOut(mask)) => {
                batch == 2
                    && cache.joint_mask.is_none()
                    && mask.dims() == [batch, total_kv_len]
                    && mask.device() == device
            }
        };
        if !matches!(batch, 1 | 2)
            || seq_lat < CONTEXT_LEN
            || self.num_heads != NUM_HEADS
            || self.head_dim != HEAD_DIM
            || cache.ctx_k.dims() != [batch, CONTEXT_LEN, NUM_HEADS, HEAD_DIM]
            || cache.ctx_v.dims() != [batch, CONTEXT_LEN, NUM_HEADS, HEAD_DIM]
            || cache.ctx_mask.dims() != [batch, CONTEXT_LEN]
            || cache.ctx_k.device() != device
            || cache.ctx_v.device() != device
            || cache.ctx_mask.device() != device
            || !joint_mask_valid
            || !wgsl_mask_valid
        {
            return None;
        }

        let combined_primitive = combined.clone().into_primitive().tensor();
        let packed_qk_primitive = packed_qk.clone().into_primitive().tensor();
        let cos_primitive = cos.clone().into_primitive().tensor();
        let sin_primitive = sin.clone().into_primitive().tensor();
        let packed_ctx_primitive = packed_ctx.clone().into_primitive().tensor();
        if !supports_direct_packed_kv(
            &combined_primitive,
            &packed_qk_primitive,
            &cos_primitive,
            &sin_primitive,
            &packed_ctx_primitive,
            self.q_norm.epsilon(),
        ) {
            return None;
        }

        let output = direct_packed_kv_wgsl(
            combined_primitive,
            packed_qk_primitive,
            cos_primitive,
            sin_primitive,
            packed_ctx_primitive,
            self.q_norm.epsilon(),
        );
        Some(WgslDirectMaterialization {
            q: Tensor::from_primitive(TensorPrimitive::Float(output.q)),
            k_all: Tensor::from_primitive(TensorPrimitive::Float(output.k_all)),
            v_all: Tensor::from_primitive(TensorPrimitive::Float(output.v_all)),
            combined: Tensor::from_primitive(TensorPrimitive::Float(output.combined)),
        })
    }

    /// Select the layout+gate epilogue while retaining both fallback tensors.
    fn try_post_sdpa_layout_gate(
        &self,
        attention: &Tensor<crate::WgpuRaw, 4>,
        combined: &Tensor<crate::WgpuRaw, 3>,
    ) -> Option<Tensor<crate::WgpuRaw, 3>> {
        use crate::kernels::joint_attention_materialization::{
            post_sdpa_layout_gate_wgsl, supports_post_sdpa_layout_gate,
        };
        use burn::tensor::TensorPrimitive;

        let attention_primitive = attention.clone().into_primitive().tensor();
        let combined_primitive = combined.clone().into_primitive().tensor();
        if !supports_post_sdpa_layout_gate(&attention_primitive, &combined_primitive) {
            return None;
        }
        let output = post_sdpa_layout_gate_wgsl(attention_primitive, combined_primitive);
        Some(Tensor::from_primitive(TensorPrimitive::Float(output)))
    }

    /// Enforce the measured WGPU cache layout at the backend boundary.
    fn assert_b1_packed_wo_row_major(&self, input: &Tensor<crate::WgpuRaw, 3>) {
        if input.dims()[0] != 1 {
            return;
        }
        let packed = self
            .validated_packed_wo_weight(input)
            .clone()
            .into_primitive()
            .tensor();
        let [rows, columns] = packed.meta.shape().dims::<2>();
        assert!(
            packed.is_contiguous(),
            "B1 packed wo cache must be row-major contiguous"
        );
        assert_eq!(
            &packed.meta.strides()[..],
            &[columns, 1],
            "B1 packed wo cache must have row-major strides"
        );
        assert_eq!(
            [rows, columns],
            self.wo.weight.dims(),
            "B1 packed wo cache shape mismatch at WGPU boundary"
        );
    }
}

impl<B: Backend> CondKvCache<B> {
    /// Pack context K/V into one exact-shape WGPU binding once per trajectory.
    ///
    /// Unsupported batches or context/head shapes leave the optional cache
    /// absent, selecting the existing K/V concatenation path. If a packed view
    /// already exists, its source-relative shape and device are always checked;
    /// this makes repeated preparation idempotent and rejects stale caches after
    /// source replacement or device movement.
    pub(crate) fn prepare_packed_ctx_kv_wgsl(&mut self) {
        use crate::kernels::joint_attention_materialization::{CONTEXT_LEN, HEAD_DIM, NUM_HEADS};

        let [batch, context_len, num_heads, head_dim] = self.ctx_k.dims();
        let source_shape = [batch, context_len, num_heads, head_dim];
        let source_device = self.ctx_k.device();
        assert_eq!(
            self.ctx_v.dims(),
            source_shape,
            "context K/V source shapes differ before WGPU packing"
        );
        assert_eq!(
            self.ctx_v.device(),
            source_device,
            "context K/V sources must share one device before WGPU packing"
        );
        assert_eq!(
            self.ctx_mask.dims(),
            [batch, context_len],
            "context mask shape differs from context K/V before WGPU packing"
        );
        assert_eq!(
            self.ctx_mask.device(),
            source_device,
            "context mask must share the context K/V device before WGPU packing"
        );
        let expected_shape = [2, batch, context_len, num_heads, head_dim];

        if let Some(packed) = self.packed_ctx_kv_wgsl.as_ref() {
            assert_eq!(
                packed.dims(),
                expected_shape,
                "existing packed context K/V WGPU cache shape mismatch"
            );
            assert_eq!(
                packed.device(),
                source_device,
                "existing packed context K/V WGPU cache is on the wrong device"
            );
            return;
        }

        if !matches!(batch, 1 | 2)
            || context_len != CONTEXT_LEN
            || num_heads != NUM_HEADS
            || head_dim != HEAD_DIM
        {
            return;
        }
        let packed = Tensor::<B, 4>::stack::<5>(vec![self.ctx_k.clone(), self.ctx_v.clone()], 0);
        assert_eq!(
            packed.dims(),
            expected_shape,
            "new packed context K/V WGPU cache shape mismatch"
        );
        assert_eq!(
            packed.device(),
            source_device,
            "new packed context K/V WGPU cache is on the wrong device"
        );
        self.packed_ctx_kv_wgsl = Some(packed);
    }

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

#[allow(clippy::too_many_arguments)]
fn concat_all_ctx_kv<B: Backend>(
    text_k: Tensor<B, 4>,
    text_v: Tensor<B, 4>,
    speaker_k: Option<Tensor<B, 4>>,
    speaker_v: Option<Tensor<B, 4>>,
    caption_k: Option<Tensor<B, 4>>,
    caption_v: Option<Tensor<B, 4>>,
    text_mask: Tensor<B, 2, Bool>,
    speaker_mask: Option<Tensor<B, 2, Bool>>,
    caption_mask: Option<Tensor<B, 2, Bool>>,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 2, Bool>) {
    (
        concat_optional_context(text_k, speaker_k, caption_k),
        concat_optional_context(text_v, speaker_v, caption_v),
        concat_optional_context_mask(text_mask, speaker_mask, caption_mask),
    )
}

fn concat_optional_context<B: Backend>(
    text: Tensor<B, 4>,
    speaker: Option<Tensor<B, 4>>,
    caption: Option<Tensor<B, 4>>,
) -> Tensor<B, 4> {
    match (speaker, caption) {
        (None, None) => text,
        (speaker, caption) => {
            let mut tensors = Vec::with_capacity(3);
            tensors.push(text);
            tensors.extend(speaker);
            tensors.extend(caption);
            Tensor::cat(tensors, 1)
        }
    }
}

fn concat_optional_context_mask<B: Backend>(
    text: Tensor<B, 2, Bool>,
    speaker: Option<Tensor<B, 2, Bool>>,
    caption: Option<Tensor<B, 2, Bool>>,
) -> Tensor<B, 2, Bool> {
    match (speaker, caption) {
        (None, None) => text,
        (speaker, caption) => {
            let mut tensors = Vec::with_capacity(3);
            tensors.push(text);
            tensors.extend(speaker);
            tensors.extend(caption);
            Tensor::cat(tensors, 1)
        }
    }
}

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
#[cfg(feature = "train")]
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
    safe_softmax: bool,
) -> Tensor<B, 4> {
    scaled_dot_product_attention_head_major(q, k, v, mask, scale, safe_softmax).swap_dims(1, 2)
}

/// Same tuned SDPA call as [`scaled_dot_product_attention`], retaining the
/// backend's contiguous `[B,H,S,Dh]` output for the WGPU layout+gate epilogue.
fn scaled_dot_product_attention_head_major<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<Tensor<B, 2, Bool>>,
    scale: f64,
    safe_softmax: bool,
) -> Tensor<B, 4> {
    scaled_dot_product_attention_head_major_with_mask_convention(
        q,
        k,
        v,
        mask,
        scale,
        safe_softmax,
        false,
    )
}

fn scaled_dot_product_attention_head_major_with_mask_convention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<Tensor<B, 2, Bool>>,
    scale: f64,
    _safe_softmax: bool,
    mask_is_backend_native: bool,
) -> Tensor<B, 4> {
    // Rearrange to [B, H, S, D_h] for burn's attention API.
    let q = q.swap_dims(1, 2);
    let k = k.swap_dims(1, 2);
    let v = v.swap_dims(1, 2);

    scaled_dot_product_attention_prepared_head_major_with_mask_convention(
        q,
        k,
        v,
        mask,
        scale,
        mask_is_backend_native,
    )
}

/// Execute the tuned backend attention when Q/K/V are already physically or
/// logically head-major `[B,H,S,Dh]`.
fn scaled_dot_product_attention_prepared_head_major_with_mask_convention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    mask: Option<Tensor<B, 2, Bool>>,
    scale: f64,
    mask_is_backend_native: bool,
) -> Tensor<B, 4> {
    // Convert 2D key-padding mask [B, S_kv] → 4D [B, 1, 1, S_kv].
    // PyTorch SDPA broadcasts across heads and query positions natively;
    // no explicit `.expand()` needed — avoids materialising the full mask.
    //
    // Ordinary callers use `True = attend (valid)`. burn's NdArray and CubeCL
    // kernels use `True = masked-out` — the opposite convention. Invert for
    // those backends unless an exact WGPU cache supplied a native mask.
    let mask_4d = mask.map(|m| {
        let m = if mask_is_backend_native || uses_pytorch_attn_mask_convention::<B>() {
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

    burn_attention(q, k, v, mask_4d, None, options)
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
    use burn::module::Module;
    use burn::tensor::Tensor;

    use crate::config::ModelConfig;

    use super::{
        Backend, Bool, JointAttention, JointAttentionGate, JointAttnCtx, SpeakerKvRange,
        build_joint_mask, manual_sdpa,
    };

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
                speaker_state: None,
                speaker_mask: None,
                caption_state: None,
                caption_mask: None,
                kv_cache: None,
            },
            cos.clone(),
            sin.clone(),
            None,
        );

        // Build cache (pre-computes ctx_k/ctx_v/ctx_mask)
        let cache = attn.build_kv_cache(
            text_state.clone(),
            text_mask.clone(),
            None,
            None,
            None,
            None,
        );
        assert_eq!(cache.speaker_range, None);
        assert_eq!(cache.ctx_k.dims()[1], seq_txt);

        // Cached forward (sampling hot-path: uses pre-computed tensors)
        let out_cached = attn.forward(
            x,
            JointAttnCtx {
                text_state,
                text_mask,
                speaker_state: None,
                speaker_mask: None,
                caption_state: None,
                caption_mask: None,
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
                speaker_state: Some(aux_state.clone()),
                speaker_mask: Some(aux_mask.clone()),
                caption_state: None,
                caption_mask: None,
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
            None,
            None,
        );
        assert_eq!(
            cache.speaker_range,
            Some(SpeakerKvRange::from_start_len(seq_txt, seq_spk))
        );

        // Cached: reads pre-computed tensors (text+aux already concatenated)
        let out_cached = attn.forward(
            x,
            JointAttnCtx {
                text_state,
                text_mask,
                speaker_state: Some(aux_state),
                speaker_mask: Some(aux_mask),
                caption_state: None,
                caption_mask: None,
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
                speaker_state: None,
                speaker_mask: None,
                caption_state: Some(aux_state.clone()),
                caption_mask: Some(aux_mask.clone()),
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
            None,
            None,
            Some(aux_state.clone()),
            Some(aux_mask.clone()),
        );

        // Cached: reads pre-computed [text|caption] ctx KV
        let out_cached = attn.forward(
            x,
            JointAttnCtx {
                text_state,
                text_mask,
                speaker_state: None,
                speaker_mask: None,
                caption_state: Some(aux_state),
                caption_mask: Some(aux_mask),
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

    #[test]
    fn kv_cache_with_speaker_and_caption_matches_non_cached_forward() {
        let device: <B as Backend>::Device = Default::default();
        let mut cfg = tiny_config_speaker();
        cfg.use_speaker_condition = Some(true);
        cfg.use_caption_condition = true;
        cfg.caption_dim = Some(12);
        cfg.caption_heads = Some(2);
        cfg.caption_layers = Some(1);
        let attn = JointAttention::<B>::new(&cfg, &device);

        assert!(attn.wk_speaker.is_some());
        assert!(attn.wk_caption.is_some());

        let (batch, seq_lat, seq_txt, seq_spk, seq_cap) = (1, 4, 6, 3, 5);
        let x = Tensor::<B, 3>::ones([batch, seq_lat, cfg.model_dim], &device);
        let text_state = Tensor::<B, 3>::ones([batch, seq_txt, cfg.text_dim], &device);
        let text_mask = Tensor::<B, 2, Bool>::ones([batch, seq_txt], &device);
        let speaker_state =
            Tensor::<B, 3>::ones([batch, seq_spk, cfg.speaker_dim.unwrap()], &device);
        let speaker_mask = Tensor::<B, 2, Bool>::ones([batch, seq_spk], &device);
        let caption_state = Tensor::<B, 3>::ones([batch, seq_cap, cfg.caption_dim()], &device);
        let caption_mask = Tensor::<B, 2, Bool>::ones([batch, seq_cap], &device);
        let cos = Tensor::<B, 2>::ones([seq_lat, cfg.head_dim() / 2], &device);
        let sin = Tensor::<B, 2>::zeros([seq_lat, cfg.head_dim() / 2], &device);

        let out_no_cache = attn.forward(
            x.clone(),
            JointAttnCtx {
                text_state: text_state.clone(),
                text_mask: text_mask.clone(),
                speaker_state: Some(speaker_state.clone()),
                speaker_mask: Some(speaker_mask.clone()),
                caption_state: Some(caption_state.clone()),
                caption_mask: Some(caption_mask.clone()),
                kv_cache: None,
            },
            cos.clone(),
            sin.clone(),
            None,
        );

        let cache = attn.build_kv_cache(
            text_state.clone(),
            text_mask.clone(),
            Some(speaker_state.clone()),
            Some(speaker_mask.clone()),
            Some(caption_state.clone()),
            Some(caption_mask.clone()),
        );
        assert_eq!(cache.ctx_k.dims()[1], seq_txt + seq_spk + seq_cap);
        assert_eq!(cache.ctx_mask.dims()[1], seq_txt + seq_spk + seq_cap);
        assert_eq!(
            cache.speaker_range,
            Some(SpeakerKvRange::from_start_len(seq_txt, seq_spk))
        );

        let out_cached = attn.forward(
            x,
            JointAttnCtx {
                text_state,
                text_mask,
                speaker_state: Some(speaker_state),
                speaker_mask: Some(speaker_mask),
                caption_state: Some(caption_state),
                caption_mask: Some(caption_mask),
                kv_cache: Some(&cache),
            },
            cos,
            sin,
            None,
        );

        let max_diff: f32 = (out_no_cache - out_cached).abs().max().into_scalar();
        assert_eq!(max_diff, 0.0);
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
        let mut cache = attn.build_kv_cache(
            text_state.clone(),
            text_mask.clone(),
            None,
            None,
            None,
            None,
        );
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
                speaker_state: None,
                speaker_mask: None,
                caption_state: None,
                caption_mask: None,
                kv_cache: Some(&cache),
            },
            cos,
            sin,
            Some(latent_mask),
        );
    }

    // -----------------------------------------------------------------------
    // Combined QKV+gate inference optimisation
    // -----------------------------------------------------------------------

    fn tiny_cfg() -> ModelConfig {
        crate::config::tiny_model_config()
    }

    fn max_abs<const D: usize>(left: Tensor<B, D>, right: Tensor<B, D>) -> f32 {
        (left - right).abs().max().into_scalar()
    }

    #[test]
    fn combined_projection_matches_qkv_and_gate_for_b1_b2() {
        let cfg = tiny_cfg();
        let device: <B as Backend>::Device = Default::default();
        let mut attn = JointAttention::<B>::new(&cfg, &device);
        attn.prepare_for_inference();
        let combined_weight = attn
            .combined_qkv_gate_weight
            .as_ref()
            .expect("combined cache")
            .clone();
        assert_eq!(combined_weight.dims(), [cfg.model_dim, 4 * cfg.model_dim]);

        for batch in [1, 2] {
            let seq_len = 3;
            let x = Tensor::<B, 3>::random(
                [batch, seq_len, cfg.model_dim],
                burn::tensor::Distribution::Default,
                &device,
            );
            let expected_q =
                attn.wq
                    .forward(x.clone())
                    .reshape([batch, seq_len, cfg.num_heads, cfg.head_dim()]);
            let expected_k =
                attn.wk
                    .forward(x.clone())
                    .reshape([batch, seq_len, cfg.num_heads, cfg.head_dim()]);
            let expected_v =
                attn.wv
                    .forward(x.clone())
                    .reshape([batch, seq_len, cfg.num_heads, cfg.head_dim()]);
            let expected_gate = burn::tensor::activation::sigmoid(attn.gate.forward(x.clone()));
            let (actual_q, actual_k, actual_v, actual_gate) = attn.compute_qkv_gate_from_combined(
                x,
                &combined_weight,
                batch,
                seq_len,
                cfg.model_dim,
            );
            let JointAttentionGate::Projected(actual_gate) = actual_gate else {
                panic!("combined projection must return a projected gate");
            };

            for (name, error) in [
                ("q", max_abs(expected_q, actual_q)),
                ("k", max_abs(expected_k, actual_k)),
                ("v", max_abs(expected_v, actual_v)),
            ] {
                assert!(error <= 1.0e-6, "batch={batch} {name} max_abs={error}");
            }
            let gate_error = max_abs(expected_gate, actual_gate);
            assert!(
                gate_error <= 1.0e-6,
                "batch={batch} gate max_abs={gate_error}"
            );
        }
    }

    #[test]
    fn prepared_fused_wo_flattening_matches_standard_for_b1_b2() {
        let cfg = tiny_cfg();
        let device: <B as Backend>::Device = Default::default();
        let mut attn = JointAttention::<B>::new(&cfg, &device);
        attn.prepare_for_inference();
        assert!(attn.wo.bias.is_none(), "v4 JointAttention wo is bias-free");
        let packed_wo = attn.packed_wo_weight.as_ref().expect("packed wo cache");
        assert_eq!(packed_wo.dims(), [cfg.model_dim, cfg.model_dim]);
        assert_eq!(packed_wo.device(), device);
        assert_eq!(
            max_abs(attn.wo.weight.val(), packed_wo.clone()),
            0.0,
            "row-major wo pack must preserve every value"
        );

        let seq_lat = 3;
        let text_len = 2;
        let head_half = cfg.head_dim() / 2;
        for batch in [1, 2] {
            let x = Tensor::<B, 3>::ones([batch, seq_lat, cfg.model_dim], &device)
                * (0.125 * batch as f32);
            let text_state = Tensor::<B, 3>::ones([batch, text_len, cfg.text_dim], &device) * 0.25;
            let text_mask: Tensor<B, 2, Bool> =
                Tensor::<B, 2>::ones([batch, text_len], &device).greater_elem(0.0);
            let cos = Tensor::<B, 2>::ones([seq_lat, head_half], &device);
            let sin = Tensor::<B, 2>::zeros([seq_lat, head_half], &device);

            let expected = attn.forward(
                x.clone(),
                JointAttnCtx {
                    text_state: text_state.clone(),
                    text_mask: text_mask.clone(),
                    speaker_state: None,
                    speaker_mask: None,
                    caption_state: None,
                    caption_mask: None,
                    kv_cache: None,
                },
                cos.clone(),
                sin.clone(),
                None,
            );
            let actual = attn.forward_fused(
                x,
                JointAttnCtx {
                    text_state,
                    text_mask,
                    speaker_state: None,
                    speaker_mask: None,
                    caption_state: None,
                    caption_mask: None,
                    kv_cache: None,
                },
                cos,
                sin,
                None,
            );
            let error = max_abs(expected, actual);
            assert!(
                error <= 1.0e-6,
                "batch={batch} flattened wo max_abs={error}"
            );
        }
    }

    /// After `prepare_for_inference()`, combined QKV+gate forward must produce
    /// identical output to the separate learned projections.
    #[test]
    fn combined_qkv_gate_matches_separate_linears() {
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
            speaker_state: None,
            speaker_mask: None,
            caption_state: None,
            caption_mask: None,
            kv_cache: None,
        };
        let out_unfused = attn.forward(x.clone(), ctx_unfused, cos.clone(), sin.clone(), None);

        // Now fuse and run again
        attn.prepare_for_inference();
        assert!(
            attn.combined_qkv_gate_weight.is_some(),
            "combined weight should be set"
        );

        let ctx_fused = JointAttnCtx {
            text_state,
            text_mask,
            speaker_state: None,
            speaker_mask: None,
            caption_state: None,
            caption_mask: None,
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
            "combined QKV+gate output should match unfused: max_diff={diff}"
        );
    }

    /// `prepare_for_inference()` is idempotent for both attention caches.
    #[test]
    fn combined_qkv_gate_idempotent() {
        let cfg = tiny_cfg();
        let device: <B as Backend>::Device = Default::default();
        let mut attn = JointAttention::<B>::new(&cfg, &device);
        attn.prepare_for_inference();
        assert_eq!(
            attn.combined_qkv_gate_weight
                .as_ref()
                .expect("combined cache")
                .dims(),
            [cfg.model_dim, 4 * cfg.model_dim]
        );
        let w1: Vec<f32> = attn
            .combined_qkv_gate_weight
            .as_ref()
            .unwrap()
            .clone()
            .into_data()
            .to_vec()
            .unwrap();
        let packed_wo_1: Vec<f32> = attn
            .packed_wo_weight
            .as_ref()
            .expect("packed wo cache")
            .clone()
            .into_data()
            .to_vec()
            .unwrap();
        attn.prepare_for_inference();
        let w2: Vec<f32> = attn
            .combined_qkv_gate_weight
            .as_ref()
            .unwrap()
            .clone()
            .into_data()
            .to_vec()
            .unwrap();
        let packed_wo_2: Vec<f32> = attn
            .packed_wo_weight
            .as_ref()
            .expect("packed wo cache")
            .clone()
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(
            w1, w2,
            "calling prepare_for_inference twice should be idempotent"
        );
        assert_eq!(
            packed_wo_1, packed_wo_2,
            "calling prepare_for_inference twice must reuse the packed wo cache"
        );
    }

    #[test]
    #[should_panic(expected = "gate bias must be absent")]
    fn combined_qkv_gate_rejects_gate_bias() {
        let cfg = tiny_cfg();
        let device: <B as Backend>::Device = Default::default();
        let mut attn = JointAttention::<B>::new(&cfg, &device);
        attn.gate = burn::nn::LinearConfig::new(cfg.model_dim, cfg.model_dim)
            .with_bias(true)
            .init(&device);
        attn.prepare_for_inference();
    }

    #[test]
    #[should_panic(expected = "existing combined QKV+gate cache shape mismatch")]
    fn combined_qkv_gate_rejects_stale_cache_shape() {
        let cfg = tiny_cfg();
        let device: <B as Backend>::Device = Default::default();
        let mut attn = JointAttention::<B>::new(&cfg, &device);
        attn.combined_qkv_gate_weight = Some(Tensor::zeros(
            [cfg.model_dim, 4 * cfg.model_dim + 1],
            &device,
        ));
        attn.prepare_for_inference();
    }

    #[test]
    #[should_panic(expected = "existing packed wo cache shape mismatch")]
    fn packed_wo_rejects_stale_cache_shape() {
        let cfg = tiny_cfg();
        let device: <B as Backend>::Device = Default::default();
        let mut attn = JointAttention::<B>::new(&cfg, &device);
        attn.packed_wo_weight = Some(Tensor::zeros([cfg.model_dim, cfg.model_dim + 1], &device));
        attn.prepare_for_inference();
    }

    #[test]
    #[should_panic(expected = "wo bias must be absent")]
    fn packed_wo_rejects_bias() {
        let cfg = tiny_cfg();
        let device: <B as Backend>::Device = Default::default();
        let mut attn = JointAttention::<B>::new(&cfg, &device);
        attn.wo = burn::nn::LinearConfig::new(cfg.model_dim, cfg.model_dim)
            .with_bias(true)
            .init(&device);
        attn.prepare_for_inference();
    }

    #[test]
    #[should_panic(expected = "without combined QKV+gate weight")]
    fn fused_forward_requires_combined_cache_presence() {
        let cfg = tiny_cfg();
        let device: <B as Backend>::Device = Default::default();
        let attn = JointAttention::<B>::new(&cfg, &device);
        let x = Tensor::<B, 3>::zeros([1, 2, cfg.model_dim], &device);
        let _ = attn.validated_combined_weight(&x, "test");
    }

    fn exact_wgsl_context_cache(batch: usize) -> super::CondKvCache<B> {
        use crate::kernels::joint_attention_materialization::{CONTEXT_LEN, HEAD_DIM, NUM_HEADS};

        let device: <B as Backend>::Device = Default::default();
        super::CondKvCache {
            ctx_k: Tensor::ones([batch, CONTEXT_LEN, NUM_HEADS, HEAD_DIM], &device),
            ctx_v: Tensor::ones([batch, CONTEXT_LEN, NUM_HEADS, HEAD_DIM], &device) * 2.0,
            ctx_mask: Tensor::<B, 2>::ones([batch, CONTEXT_LEN], &device).greater_elem(0.0),
            joint_mask: None,
            speaker_range: None,
            packed_ctx_kv_wgsl: None,
            joint_mask_wgsl: None,
            joint_attend_mask_wgsl: None,
        }
    }

    #[test]
    fn exact_wgsl_context_pack_is_bit_exact_and_idempotent_for_b1_b2() {
        use crate::kernels::joint_attention_materialization::{CONTEXT_LEN, HEAD_DIM, NUM_HEADS};

        for batch in [1, 2] {
            let mut cache = exact_wgsl_context_cache(batch);
            cache.prepare_packed_ctx_kv_wgsl();
            let first = cache
                .packed_ctx_kv_wgsl
                .as_ref()
                .expect("exact WGPU cache must be packed")
                .clone();
            assert_eq!(first.dims(), [2, batch, CONTEXT_LEN, NUM_HEADS, HEAD_DIM]);
            let packed_k =
                first
                    .clone()
                    .narrow(0, 0, 1)
                    .reshape([batch, CONTEXT_LEN, NUM_HEADS, HEAD_DIM]);
            let packed_v =
                first
                    .clone()
                    .narrow(0, 1, 1)
                    .reshape([batch, CONTEXT_LEN, NUM_HEADS, HEAD_DIM]);
            assert_eq!(max_abs(cache.ctx_k.clone(), packed_k), 0.0);
            assert_eq!(max_abs(cache.ctx_v.clone(), packed_v), 0.0);

            cache.prepare_packed_ctx_kv_wgsl();
            let second = cache
                .packed_ctx_kv_wgsl
                .as_ref()
                .expect("idempotent WGPU cache pack")
                .clone();
            assert_eq!(max_abs(first, second), 0.0);
        }
    }

    #[test]
    fn unsupported_wgsl_context_shape_falls_back_without_packing() {
        let mut cache = exact_wgsl_context_cache(3);
        cache.prepare_packed_ctx_kv_wgsl();
        assert!(cache.packed_ctx_kv_wgsl.is_none());
    }

    #[test]
    #[should_panic(expected = "existing packed context K/V WGPU cache shape mismatch")]
    fn stale_wgsl_context_cache_is_rejected_during_preparation() {
        use crate::kernels::joint_attention_materialization::{CONTEXT_LEN, HEAD_DIM, NUM_HEADS};

        let device: <B as Backend>::Device = Default::default();
        let mut cache = exact_wgsl_context_cache(1);
        cache.packed_ctx_kv_wgsl = Some(Tensor::zeros(
            [2, 1, CONTEXT_LEN, NUM_HEADS, HEAD_DIM - 1],
            &device,
        ));
        cache.prepare_packed_ctx_kv_wgsl();
    }

    #[test]
    fn exact_wgsl_qk_norm_pack_is_bit_exact_and_idempotent() {
        use crate::kernels::joint_attention_materialization::{HEAD_DIM, NUM_HEADS};
        use crate::model::norm::HeadRmsNorm;

        let cfg = tiny_cfg();
        let device: <B as Backend>::Device = Default::default();
        let mut attn = JointAttention::<B>::new(&cfg, &device);
        attn.num_heads = NUM_HEADS;
        attn.head_dim = HEAD_DIM;
        attn.q_norm = HeadRmsNorm::new(NUM_HEADS, HEAD_DIM, cfg.norm_eps, &device);
        attn.k_norm = HeadRmsNorm::new(NUM_HEADS, HEAD_DIM, cfg.norm_eps, &device);
        attn.prepare_qk_norm_weight_wgsl();
        let first = attn
            .packed_qk_norm_weight_wgsl
            .as_ref()
            .expect("exact Q/K norm weights must be packed")
            .clone();
        assert_eq!(first.dims(), [2, NUM_HEADS, HEAD_DIM]);
        assert_eq!(
            max_abs(
                attn.q_norm.weight.val(),
                first.clone().narrow(0, 0, 1).reshape([NUM_HEADS, HEAD_DIM]),
            ),
            0.0
        );
        assert_eq!(
            max_abs(
                attn.k_norm.weight.val(),
                first.clone().narrow(0, 1, 1).reshape([NUM_HEADS, HEAD_DIM]),
            ),
            0.0
        );
        attn.prepare_qk_norm_weight_wgsl();
        assert_eq!(
            max_abs(
                first,
                attn.packed_qk_norm_weight_wgsl
                    .as_ref()
                    .expect("idempotent Q/K norm pack")
                    .clone(),
            ),
            0.0
        );
    }

    #[test]
    #[should_panic(expected = "existing packed Q/K norm WGPU cache shape mismatch")]
    fn stale_wgsl_qk_norm_cache_is_rejected_during_preparation() {
        let cfg = tiny_cfg();
        let device: <B as Backend>::Device = Default::default();
        let mut attn = JointAttention::<B>::new(&cfg, &device);
        attn.packed_qk_norm_weight_wgsl = Some(Tensor::zeros(
            [2, cfg.num_heads, cfg.head_dim() + 1],
            &device,
        ));
        attn.prepare_qk_norm_weight_wgsl();
    }

    #[test]
    fn wgsl_qk_norm_cache_is_skipped_from_records() {
        let cfg = tiny_cfg();
        let device: <B as Backend>::Device = Default::default();
        let mut attn = JointAttention::<B>::new(&cfg, &device);
        attn.packed_qk_norm_weight_wgsl =
            Some(Tensor::ones([2, cfg.num_heads, cfg.head_dim()], &device));
        let record = attn.into_record();
        let restored = JointAttention::<B>::new(&cfg, &device).load_record(record);
        assert!(
            restored.packed_qk_norm_weight_wgsl.is_none(),
            "record loading must not restore the WGPU-only packed Q/K cache"
        );
    }

    #[test]
    fn native_sdpa_selector_is_limited_to_measured_short_lengths() {
        let s13 = super::native_sdpa_config_for_sequence(13).expect("S13 native SDPA");
        assert_eq!([s13.tile_q, s13.tile_kv], [16, 16]);
        for sequence in [25, 50] {
            let config = super::native_sdpa_config_for_sequence(sequence)
                .expect("measured native SDPA length");
            assert_eq!([config.tile_q, config.tile_kv], [8, 32]);
        }
        for sequence in [0, 12, 14, 49, 51, 100, 200] {
            assert!(super::native_sdpa_config_for_sequence(sequence).is_none());
        }
    }
}
