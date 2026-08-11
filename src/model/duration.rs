//! Automatic duration prediction for the released Irodori-TTS v4 architecture.
//!
//! v4-Small predicts a positive frame contribution for every valid text token,
//! sums those contributions, and returns `log1p(total_frames)`. Speaker and
//! caption conditions modulate every SwiGLU block through independent
//! AdaRN-Zero projections. The auxiliary feature tensor remains part of the
//! checkpoint interface, but this `*_no_aux` architecture deliberately does
//! not consume its values.

use burn::{
    module::{Module, Param, ParamId},
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    tensor::{Bool, FloatDType, Tensor, activation::silu, activation::softplus, backend::Backend},
};

use crate::{
    config::ModelConfig,
    error::{IrodoriError, Result},
};

use super::{feed_forward::SwiGlu, norm::RmsNorm};

/// The sole duration architecture present in the released v4-Small weights.
pub const V4_DURATION_ARCHITECTURE: &str = "token_sum_dual_adarn_zero_no_aux";

const V4_DURATION_FUSION: &str = "adarn_zero";
const V4_CAPTION_POOLING: &str = "masked_mean";

/// Validated construction parameters for the released v4 duration predictor.
#[derive(Debug, Clone, PartialEq)]
pub struct DurationPredictorConfig {
    pub text_dim: usize,
    pub aux_dim: usize,
    pub hidden_dim: usize,
    pub layers: usize,
    pub dropout: f64,
    pub speaker_dim: usize,
    pub caption_dim: usize,
    pub attention_heads: usize,
    pub norm_eps: f64,
    pub token_init_frames: f64,
}

impl TryFrom<&ModelConfig> for DurationPredictorConfig {
    type Error = IrodoriError;

    fn try_from(cfg: &ModelConfig) -> Result<Self> {
        if !cfg.use_duration_predictor {
            return Err(IrodoriError::Config(
                "duration predictor construction requested, but use_duration_predictor=false"
                    .to_string(),
            ));
        }

        let architecture = cfg.duration_architecture.trim().to_ascii_lowercase();
        if architecture != V4_DURATION_ARCHITECTURE {
            return Err(IrodoriError::Config(format!(
                "unsupported duration_architecture {architecture:?}; only released v4 architecture {V4_DURATION_ARCHITECTURE:?} is supported"
            )));
        }

        let speaker_fusion = cfg.duration_speaker_fusion.trim().to_ascii_lowercase();
        if speaker_fusion != V4_DURATION_FUSION {
            return Err(IrodoriError::Config(format!(
                "unsupported duration_speaker_fusion {speaker_fusion:?} for {V4_DURATION_ARCHITECTURE}; expected {V4_DURATION_FUSION:?}"
            )));
        }

        let caption_fusion = cfg.duration_caption_fusion.trim().to_ascii_lowercase();
        if caption_fusion != V4_DURATION_FUSION {
            return Err(IrodoriError::Config(format!(
                "unsupported duration_caption_fusion {caption_fusion:?} for {V4_DURATION_ARCHITECTURE}; expected {V4_DURATION_FUSION:?}"
            )));
        }

        let caption_pooling = cfg.duration_caption_pooling.trim().to_ascii_lowercase();
        if caption_pooling != V4_CAPTION_POOLING {
            return Err(IrodoriError::Config(format!(
                "unsupported duration_caption_pooling {caption_pooling:?} for {V4_DURATION_ARCHITECTURE}; expected {V4_CAPTION_POOLING:?}"
            )));
        }

        if !cfg.use_speaker_condition() {
            return Err(IrodoriError::Config(format!(
                "{V4_DURATION_ARCHITECTURE} requires speaker conditioning"
            )));
        }
        if !cfg.use_caption_condition {
            return Err(IrodoriError::Config(format!(
                "{V4_DURATION_ARCHITECTURE} requires caption conditioning"
            )));
        }

        let speaker_dim = cfg.speaker_dim.ok_or_else(|| {
            IrodoriError::Config(format!("{V4_DURATION_ARCHITECTURE} requires speaker_dim"))
        })?;
        let caption_dim = cfg.caption_dim();

        let duration_cfg = Self {
            text_dim: cfg.text_dim,
            aux_dim: cfg.duration_aux_dim,
            hidden_dim: cfg.duration_hidden_dim,
            layers: cfg.duration_layers,
            dropout: cfg.duration_dropout,
            speaker_dim,
            caption_dim,
            attention_heads: cfg.duration_attention_heads,
            norm_eps: cfg.norm_eps,
            token_init_frames: cfg.duration_token_init_frames,
        };
        duration_cfg.validate()?;
        Ok(duration_cfg)
    }
}

impl DurationPredictorConfig {
    fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("text_dim", self.text_dim),
            ("aux_dim", self.aux_dim),
            ("hidden_dim", self.hidden_dim),
            ("layers", self.layers),
            ("speaker_dim", self.speaker_dim),
            ("caption_dim", self.caption_dim),
            ("attention_heads", self.attention_heads),
        ] {
            if value == 0 {
                return Err(IrodoriError::Config(format!(
                    "duration predictor {name} must be > 0"
                )));
            }
        }
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(IrodoriError::Config(
                "duration predictor dropout must be finite and in [0, 1)".to_string(),
            ));
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err(IrodoriError::Config(
                "duration predictor norm_eps must be finite and > 0".to_string(),
            ));
        }
        if !self.token_init_frames.is_finite() || self.token_init_frames <= 0.0 {
            return Err(IrodoriError::Config(
                "duration predictor token_init_frames must be finite and > 0".to_string(),
            ));
        }
        let initial_bias = self.token_init_frames.exp_m1().ln();
        if !initial_bias.is_finite() {
            return Err(IrodoriError::Config(
                "duration predictor token_init_frames is too large for inverse softplus"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Inputs consumed by the v4 token-sum duration path.
///
/// `has_speaker` and `has_caption` are explicit `[batch]` presence vectors.
/// A false entry selects the corresponding learned null vector. A caption row
/// whose effective mask is entirely false also selects the null caption.
pub struct DurationPredictorInput<B: Backend> {
    pub text_state: Tensor<B, 3>,
    pub text_mask: Tensor<B, 2, Bool>,
    pub aux_features: Tensor<B, 2>,
    pub speaker_state: Option<Tensor<B, 3>>,
    pub has_speaker: Tensor<B, 1, Bool>,
    pub caption_state: Option<Tensor<B, 3>>,
    pub caption_mask: Option<Tensor<B, 2, Bool>>,
    pub has_caption: Tensor<B, 1, Bool>,
}

/// A v4 duration block with additive speaker and caption AdaRN-Zero controls.
///
/// Field names intentionally mirror the Python checkpoint hierarchy.
#[derive(Module, Debug)]
pub struct DurationSwiGluBlock<B: Backend> {
    pub(crate) norm: RmsNorm<B>,
    pub(crate) mlp: SwiGlu<B>,
    pub(crate) dropout: Dropout,
    pub(crate) modulation: Linear<B>,
    pub(crate) caption_modulation: Linear<B>,
    /// Precomputed AdaRN-Zero modulation for the common no-speaker/no-caption
    /// inference path. These tensors are derived after checkpoint loading and
    /// deliberately do not participate in serialization or device moves.
    #[module(skip)]
    cached_null_shift: Option<Tensor<B, 2>>,
    #[module(skip)]
    cached_null_scale_plus_one: Option<Tensor<B, 2>>,
    #[module(skip)]
    cached_null_gate_tanh: Option<Tensor<B, 2>>,
}

impl<B: Backend> DurationSwiGluBlock<B> {
    fn new(
        dim: usize,
        hidden_dim: usize,
        speaker_dim: usize,
        caption_dim: usize,
        dropout: f64,
        norm_eps: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            norm: RmsNorm::new(dim, norm_eps, device),
            mlp: SwiGlu::new(dim, Some(hidden_dim), device),
            dropout: DropoutConfig::new(dropout).init(),
            modulation: zero_linear(speaker_dim, dim * 3, device),
            caption_modulation: zero_linear(caption_dim, dim * 3, device),
            cached_null_shift: None,
            cached_null_scale_plus_one: None,
            cached_null_gate_tanh: None,
        }
    }

    /// Apply speaker and caption AdaRN-Zero, then the gated SwiGLU residual.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        speaker: Tensor<B, 2>,
        caption: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let h = self.norm.forward(x.clone());

        let speaker_parts = self.modulation.forward(silu(speaker)).chunk(3, 1);
        let caption_parts = self.caption_modulation.forward(silu(caption)).chunk(3, 1);

        let shift = (speaker_parts[0].clone() + caption_parts[0].clone()).unsqueeze_dim::<3>(1);
        let scale = (speaker_parts[1].clone() + caption_parts[1].clone()).unsqueeze_dim::<3>(1);
        let gate = (speaker_parts[2].clone() + caption_parts[2].clone()).unsqueeze_dim::<3>(1);

        let h = h * (scale + 1.0) + shift;
        x + self.dropout.forward(gate.tanh() * self.mlp.forward(h))
    }

    /// Materialize the inference-only fused SwiGLU projection and the fixed
    /// no-aux modulation values.
    fn prepare_for_inference(&mut self, null_speaker: Tensor<B, 2>, null_caption: Tensor<B, 2>) {
        self.mlp.prepare_for_inference();
        if self.cached_null_shift.is_some() {
            assert!(
                self.cached_null_scale_plus_one.is_some() && self.cached_null_gate_tanh.is_some(),
                "duration null-modulation cache is partial"
            );
            return;
        }
        let speaker_parts = self.modulation.forward(silu(null_speaker)).chunk(3, 1);
        let caption_parts = self
            .caption_modulation
            .forward(silu(null_caption))
            .chunk(3, 1);
        self.cached_null_shift = Some(speaker_parts[0].clone() + caption_parts[0].clone());
        self.cached_null_scale_plus_one =
            Some(speaker_parts[1].clone() + caption_parts[1].clone() + 1.0);
        self.cached_null_gate_tanh =
            Some((speaker_parts[2].clone() + caption_parts[2].clone()).tanh());
    }

    /// Fast path for a condition bundle with no encoded speaker or caption.
    fn forward_cached_null(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let shift = self
            .cached_null_shift
            .as_ref()
            .expect("duration null shift cache was not prepared")
            .clone()
            .unsqueeze_dim::<3>(1);
        let scale = self
            .cached_null_scale_plus_one
            .as_ref()
            .expect("duration null scale-plus-one cache was not prepared")
            .clone()
            .unsqueeze_dim::<3>(1);
        let gate = self
            .cached_null_gate_tanh
            .as_ref()
            .expect("duration null tanh-gate cache was not prepared")
            .clone()
            .unsqueeze_dim::<3>(1);
        let h = self.norm.forward(x.clone()) * scale + shift;
        x + self.dropout.forward(gate * self.mlp.forward_fused(h))
    }

    fn has_cached_null(&self) -> bool {
        self.cached_null_shift.is_some()
            && self.cached_null_scale_plus_one.is_some()
            && self.cached_null_gate_tanh.is_some()
    }
}

impl DurationSwiGluBlock<crate::WgpuRaw> {
    /// No-aux production route using the WGSL SwiGLU epilogue and prepared
    /// row-major `w2` cache. The modulation tensors remain GPU-resident.
    fn forward_cached_null_wgsl(&self, x: Tensor<crate::WgpuRaw, 3>) -> Tensor<crate::WgpuRaw, 3> {
        use burn::tensor::TensorPrimitive;

        let shift = self
            .cached_null_shift
            .as_ref()
            .expect("duration null shift cache was not prepared")
            .clone()
            .unsqueeze_dim::<3>(1);
        let scale = self
            .cached_null_scale_plus_one
            .as_ref()
            .expect("duration null scale-plus-one cache was not prepared")
            .clone()
            .unsqueeze_dim::<3>(1);
        let gate = self
            .cached_null_gate_tanh
            .as_ref()
            .expect("duration null tanh-gate cache was not prepared")
            .clone()
            .unsqueeze_dim::<3>(1);
        let [_, _, dim] = x.dims();
        let h = crate::kernels::duration_block_preprocess::try_duration_block_preprocess_wgsl(
            x.clone().into_primitive().tensor(),
            self.norm.weight.val().into_primitive().tensor(),
            scale.clone().reshape([1, dim]).into_primitive().tensor(),
            shift.clone().reshape([1, dim]).into_primitive().tensor(),
            self.norm.epsilon(),
        )
        .map(|output| Tensor::<crate::WgpuRaw, 3>::from_primitive(TensorPrimitive::Float(output)))
        .unwrap_or_else(|| self.norm.forward(x.clone()) * scale + shift);
        let branch = self.mlp.forward_duration_fused_wgsl(h);
        let branch = self.dropout.forward(branch);
        crate::kernels::duration_residual_finalize::try_duration_residual_finalize_wgsl(
            x.clone().into_primitive().tensor(),
            branch.clone().into_primitive().tensor(),
            gate.clone().into_primitive().tensor(),
        )
        .map(|output| Tensor::<crate::WgpuRaw, 3>::from_primitive(TensorPrimitive::Float(output)))
        .unwrap_or_else(|| x + gate * branch)
    }
}

/// Automatic duration predictor matching v4-Small's released state dictionary.
#[derive(Module, Debug)]
pub struct DurationPredictor<B: Backend> {
    pub(crate) null_speaker: Param<Tensor<B, 1>>,
    pub(crate) null_caption: Param<Tensor<B, 1>>,
    pub(crate) token_input_proj: Linear<B>,
    pub(crate) token_blocks: Vec<DurationSwiGluBlock<B>>,
    pub(crate) token_out_norm: RmsNorm<B>,
    pub(crate) token_out_proj: Linear<B>,
    text_dim: usize,
    aux_dim: usize,
    speaker_dim: usize,
    caption_dim: usize,
}

impl<B: Backend> DurationPredictor<B> {
    /// Build from the full model configuration after rejecting non-v4 variants.
    pub fn from_model_config(cfg: &ModelConfig, device: &B::Device) -> Result<Self> {
        Self::new(DurationPredictorConfig::try_from(cfg)?, device)
    }

    /// Build the released v4 token-sum architecture.
    pub fn new(cfg: DurationPredictorConfig, device: &B::Device) -> Result<Self> {
        cfg.validate()?;

        let token_blocks = (0..cfg.layers)
            .map(|_| {
                DurationSwiGluBlock::new(
                    cfg.hidden_dim,
                    cfg.hidden_dim,
                    cfg.speaker_dim,
                    cfg.caption_dim,
                    cfg.dropout,
                    cfg.norm_eps,
                    device,
                )
            })
            .collect();

        let mut token_out_proj = LinearConfig::new(cfg.hidden_dim, 1)
            .with_bias(true)
            .init::<B>(device);
        token_out_proj.weight =
            Param::initialized(ParamId::new(), Tensor::zeros([cfg.hidden_dim, 1], device));
        let initial_bias = cfg.token_init_frames.exp_m1().ln();
        token_out_proj.bias = Some(Param::initialized(
            ParamId::new(),
            Tensor::zeros([1], device).add_scalar(initial_bias),
        ));

        Ok(Self {
            null_speaker: Param::initialized(
                ParamId::new(),
                Tensor::zeros([cfg.speaker_dim], device),
            ),
            null_caption: Param::initialized(
                ParamId::new(),
                Tensor::zeros([cfg.caption_dim], device),
            ),
            token_input_proj: LinearConfig::new(cfg.text_dim, cfg.hidden_dim)
                .with_bias(true)
                .init(device),
            token_blocks,
            token_out_norm: RmsNorm::new(cfg.hidden_dim, cfg.norm_eps, device),
            token_out_proj,
            text_dim: cfg.text_dim,
            aux_dim: cfg.aux_dim,
            speaker_dim: cfg.speaker_dim,
            caption_dim: cfg.caption_dim,
        })
    }

    /// Prepare every duration block for the same fused inference policy used
    /// by the diffusion backbone. This is idempotent and must run only after
    /// checkpoint loading and final device placement.
    pub(crate) fn prepare_for_inference(&mut self) {
        let null_speaker = self.null_speaker.val().unsqueeze_dim::<2>(0);
        let null_caption = self.null_caption.val().unsqueeze_dim::<2>(0);
        for block in &mut self.token_blocks {
            block.prepare_for_inference(null_speaker.clone(), null_caption.clone());
        }
    }

    /// Predict `log1p(total_frames)` for each batch item.
    pub fn forward(&self, input: DurationPredictorInput<B>) -> Result<Tensor<B, 1>> {
        let (hidden, text_mask_f) = self.forward_hidden_with_cached(
            input,
            false,
            |projection, text| projection.forward(text),
            |block, hidden| block.forward_cached_null(hidden),
        )?;
        Ok(self.finalize_hidden(
            hidden,
            text_mask_f.expect("generic duration path must retain its text mask"),
        ))
    }

    fn finalize_hidden(&self, hidden: Tensor<B, 3>, text_mask_f: Tensor<B, 2>) -> Tensor<B, 1> {
        let [batch, seq_len, _] = hidden.dims();
        let token_logits = self
            .token_out_proj
            .forward(self.token_out_norm.forward(hidden))
            .reshape([batch, seq_len])
            // Python calls `.float()` before Softplus and accumulation.
            .cast(FloatDType::F32);
        let token_frames = pytorch_softplus(token_logits);
        let total_frames = (token_frames * text_mask_f)
            .sum_dim(1)
            .reshape([batch])
            .clamp_min(0.0);
        total_frames.log1p()
    }

    fn forward_hidden_with_cached<F, P>(
        &self,
        input: DurationPredictorInput<B>,
        compact_all_valid: bool,
        project_input: P,
        mut cached_forward: F,
    ) -> Result<(Tensor<B, 3>, Option<Tensor<B, 2>>)>
    where
        F: FnMut(&DurationSwiGluBlock<B>, Tensor<B, 3>) -> Tensor<B, 3>,
        P: FnOnce(&Linear<B>, Tensor<B, 3>) -> Tensor<B, 3>,
    {
        let DurationPredictorInput {
            text_state,
            text_mask,
            aux_features,
            speaker_state,
            has_speaker,
            caption_state,
            caption_mask,
            has_caption,
        } = input;

        let [batch, seq_len, text_dim] = text_state.dims();
        if text_dim != self.text_dim {
            return Err(IrodoriError::Shape(format!(
                "duration text_state must have shape (B, S, {}), got ({batch}, {seq_len}, {text_dim})",
                self.text_dim
            )));
        }
        let [mask_batch, mask_seq_len] = text_mask.dims();
        if [mask_batch, mask_seq_len] != [batch, seq_len] {
            return Err(IrodoriError::Shape(format!(
                "duration text_mask must have shape ({batch}, {seq_len}), got ({mask_batch}, {mask_seq_len})"
            )));
        }
        if seq_len == 0 {
            return Err(IrodoriError::Shape(
                "duration text sequence must contain at least one token".to_string(),
            ));
        }

        let [aux_batch, aux_dim] = aux_features.dims();
        if [aux_batch, aux_dim] != [batch, self.aux_dim] {
            return Err(IrodoriError::Shape(format!(
                "duration aux_features must have shape ({batch}, {}), got ({aux_batch}, {aux_dim})",
                self.aux_dim
            )));
        }
        let [has_speaker_batch] = has_speaker.dims();
        if has_speaker_batch != batch {
            return Err(IrodoriError::Shape(format!(
                "duration has_speaker must have shape ({batch},), got ({has_speaker_batch},)"
            )));
        }
        let [has_caption_batch] = has_caption.dims();
        if has_caption_batch != batch {
            return Err(IrodoriError::Shape(format!(
                "duration has_caption must have shape ({batch},), got ({has_caption_batch},)"
            )));
        }

        // This validates the checkpoint-compatible interface. Values are
        // intentionally unused by `token_sum_dual_adarn_zero_no_aux`.
        let _ = aux_features;

        let (text_state, text_mask_f) = if compact_all_valid {
            (text_state, None)
        } else {
            let (state, mask) = safe_text_state_and_mask(text_state, text_mask, batch, seq_len);
            (state, Some(mask))
        };
        let use_cached_null = speaker_state.is_none()
            && caption_state.is_none()
            && self
                .token_blocks
                .iter()
                .all(DurationSwiGluBlock::has_cached_null);
        let mut hidden = project_input(&self.token_input_proj, text_state);
        if use_cached_null {
            // The prepared values already include both learned null vectors.
            // Avoid constructing, selecting, and broadcasting speaker/caption
            // tensors that no block will consume on this production path.
            for block in &self.token_blocks {
                hidden = cached_forward(block, hidden);
            }
        } else {
            let speaker = self.speaker_vec(batch, speaker_state, has_speaker)?;
            let caption = self.caption_vec(batch, caption_state, caption_mask, has_caption)?;
            for block in &self.token_blocks {
                hidden = block.forward(hidden, speaker.clone(), caption.clone());
            }
        }

        Ok((hidden, text_mask_f))
    }

    fn speaker_vec(
        &self,
        batch: usize,
        speaker_state: Option<Tensor<B, 3>>,
        has_speaker: Tensor<B, 1, Bool>,
    ) -> Result<Tensor<B, 2>> {
        let null = self
            .null_speaker
            .val()
            .unsqueeze_dim::<2>(0)
            .expand([batch, self.speaker_dim]);
        let Some(speaker_state) = speaker_state else {
            return Ok(null);
        };

        let [speaker_batch, speaker_seq_len, speaker_dim] = speaker_state.dims();
        if speaker_batch != batch || speaker_dim != self.speaker_dim {
            return Err(IrodoriError::Shape(format!(
                "duration speaker_state must have shape ({batch}, S, {}), got ({speaker_batch}, {speaker_seq_len}, {speaker_dim})",
                self.speaker_dim
            )));
        }
        if speaker_seq_len == 0 {
            return Err(IrodoriError::Shape(
                "duration speaker_state must contain at least one token".to_string(),
            ));
        }

        let real = speaker_state
            .narrow(1, 0, 1)
            .reshape([batch, self.speaker_dim]);
        let use_real = has_speaker
            .unsqueeze_dim::<2>(1)
            .expand([batch, self.speaker_dim]);
        Ok(null.mask_where(use_real, real))
    }

    fn caption_vec(
        &self,
        batch: usize,
        caption_state: Option<Tensor<B, 3>>,
        caption_mask: Option<Tensor<B, 2, Bool>>,
        has_caption: Tensor<B, 1, Bool>,
    ) -> Result<Tensor<B, 2>> {
        let null = self
            .null_caption
            .val()
            .unsqueeze_dim::<2>(0)
            .expand([batch, self.caption_dim]);
        let Some(caption_state) = caption_state else {
            return Ok(null);
        };

        let [caption_batch, caption_seq_len, caption_dim] = caption_state.dims();
        if caption_batch != batch || caption_dim != self.caption_dim {
            return Err(IrodoriError::Shape(format!(
                "duration caption_state must have shape ({batch}, S, {}), got ({caption_batch}, {caption_seq_len}, {caption_dim})",
                self.caption_dim
            )));
        }

        let caption_mask = match caption_mask {
            Some(mask) => {
                let [mask_batch, mask_seq_len] = mask.dims();
                if [mask_batch, mask_seq_len] != [batch, caption_seq_len] {
                    return Err(IrodoriError::Shape(format!(
                        "duration caption_mask must have shape ({batch}, {caption_seq_len}), got ({mask_batch}, {mask_seq_len})"
                    )));
                }
                mask
            }
            None => Tensor::<B, 2>::ones([batch, caption_seq_len], &caption_state.device())
                .greater_elem(0.0),
        };

        let effective_mask = caption_mask.float() * has_caption.float().unsqueeze_dim::<2>(1);
        let denom = effective_mask.clone().sum_dim(1);
        let pooled = (caption_state * effective_mask.unsqueeze_dim::<3>(2))
            .sum_dim(1)
            .reshape([batch, self.caption_dim])
            / denom.clone().clamp_min(1.0);
        let use_pooled = denom.greater_elem(0.0).expand([batch, self.caption_dim]);
        Ok(null.mask_where(use_pooled, pooled))
    }
}

impl DurationPredictor<crate::WgpuRaw> {
    /// Production fast path for a batch-one condition compacted to an entirely
    /// valid text prefix with no speaker or caption state.
    pub(crate) fn forward_compact_no_aux_wgsl(
        &self,
        input: DurationPredictorInput<crate::WgpuRaw>,
    ) -> Result<Tensor<crate::WgpuRaw, 1>> {
        if input.speaker_state.is_some() || input.caption_state.is_some() {
            return Err(IrodoriError::Config(
                "compact duration WGSL path requires no speaker/caption state".to_string(),
            ));
        }
        let (hidden, mask) = self.forward_hidden_with_cached(
            input,
            true,
            |projection, text| {
                use burn::tensor::TensorPrimitive;

                let [batch, sequence, input_dim] = text.dims();
                let [weight_input, output_dim] = projection.weight.dims();
                let candidate = (batch == 1
                    && (1..=64).contains(&sequence)
                    && input_dim == 512
                    && weight_input == 512
                    && output_dim == 1_024)
                    .then(|| {
                        let bias = projection.bias.as_ref()?;
                        crate::kernels::dit_projection_t64::try_duration_input_projection_t64_wgsl(
                            text.clone()
                                .reshape([batch * sequence, input_dim])
                                .into_primitive()
                                .tensor(),
                            projection.weight.val().into_primitive().tensor(),
                            bias.val().into_primitive().tensor(),
                        )
                    })
                    .flatten();
                candidate
                    .map(|output| {
                        Tensor::<crate::WgpuRaw, 2>::from_primitive(TensorPrimitive::Float(output))
                            .reshape([batch, sequence, output_dim])
                    })
                    .unwrap_or_else(|| projection.forward(text))
            },
            |block, hidden| block.forward_cached_null_wgsl(hidden),
        )?;
        debug_assert!(mask.is_none());
        self.finalize_compact_no_aux_wgsl(hidden)
    }

    fn finalize_compact_no_aux_wgsl(
        &self,
        hidden: Tensor<crate::WgpuRaw, 3>,
    ) -> Result<Tensor<crate::WgpuRaw, 1>> {
        use burn::tensor::TensorPrimitive;

        let [batch, sequence, _] = hidden.dims();
        if batch == 1 {
            let bias = self.token_out_proj.bias.as_ref().ok_or_else(|| {
                IrodoriError::Config("duration output projection bias is missing".to_string())
            })?;
            if let Some(output) =
                crate::kernels::duration_output_finalize::try_duration_output_finalize_wgsl(
                    hidden.clone().into_primitive().tensor(),
                    self.token_out_norm.weight.val().into_primitive().tensor(),
                    self.token_out_proj.weight.val().into_primitive().tensor(),
                    bias.val().into_primitive().tensor(),
                    self.token_out_norm.epsilon(),
                )
            {
                return Ok(Tensor::<crate::WgpuRaw, 1>::from_primitive(
                    TensorPrimitive::Float(output),
                ));
            }
        }
        let mask = Tensor::<crate::WgpuRaw, 2>::ones([batch, sequence], &hidden.device())
            .cast(FloatDType::F32);
        Ok(self.finalize_hidden(hidden, mask))
    }
}

fn zero_linear<B: Backend>(input_dim: usize, output_dim: usize, device: &B::Device) -> Linear<B> {
    let mut linear = LinearConfig::new(input_dim, output_dim)
        .with_bias(true)
        .init::<B>(device);
    linear.weight = Param::initialized(
        ParamId::new(),
        Tensor::zeros([input_dim, output_dim], device),
    );
    linear.bias = Some(Param::initialized(
        ParamId::new(),
        Tensor::zeros([output_dim], device),
    ));
    linear
}

/// Match Python's `_safe_attention_mask`: fully masked rows become a zero text
/// row with token zero marked valid. This preserves the trained token-sum
/// fallback instead of silently predicting zero total frames.
fn safe_text_state_and_mask<B: Backend>(
    text_state: Tensor<B, 3>,
    text_mask: Tensor<B, 2, Bool>,
    batch: usize,
    seq_len: usize,
) -> (Tensor<B, 3>, Tensor<B, 2>) {
    let device = text_state.device();
    let [_, _, text_dim] = text_state.dims();
    let mask_f = text_mask.float().cast(FloatDType::F32);
    let has_any = mask_f.clone().sum_dim(1).greater_elem(0.0);
    let has_any_f = has_any.clone().float().cast(FloatDType::F32);

    // Use selection rather than multiplication so even non-finite values in a
    // fully masked input row are overwritten exactly as in Python's assignment.
    let invalid_rows = has_any
        .bool_not()
        .unsqueeze_dim::<3>(2)
        .expand([batch, seq_len, text_dim]);
    let safe_state = text_state.mask_where(
        invalid_rows,
        Tensor::zeros([batch, seq_len, text_dim], &device),
    );
    let fallback = if seq_len == 1 {
        Tensor::<B, 2>::ones([batch, 1], &device).cast(FloatDType::F32)
    } else {
        Tensor::cat(
            vec![
                Tensor::<B, 2>::ones([batch, 1], &device).cast(FloatDType::F32),
                Tensor::<B, 2>::zeros([batch, seq_len - 1], &device).cast(FloatDType::F32),
            ],
            1,
        )
    };
    let missing = Tensor::<B, 2>::ones([batch, 1], &device).cast(FloatDType::F32) - has_any_f;
    let safe_mask = mask_f + fallback * missing;
    (safe_state, safe_mask)
}

/// PyTorch's default Softplus is linear above a threshold of 20. Clamping the
/// nonlinear branch prevents overflow while preserving the exact threshold
/// behaviour used by `torch.nn.functional.softplus`.
fn pytorch_softplus<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    let linear = tensor.clone().greater_elem(20.0);
    let nonlinear = softplus(tensor.clone().clamp_max(20.0), 1.0);
    nonlinear.mask_where(linear, tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::NdArray,
        tensor::{TensorData, backend::Backend},
    };

    type B = NdArray<f32>;

    fn device() -> <B as Backend>::Device {
        Default::default()
    }

    fn tiny_config() -> DurationPredictorConfig {
        DurationPredictorConfig {
            text_dim: 4,
            aux_dim: 2,
            hidden_dim: 8,
            layers: 2,
            dropout: 0.0,
            speaker_dim: 3,
            caption_dim: 2,
            attention_heads: 1,
            norm_eps: 1e-5,
            token_init_frames: 9.0,
        }
    }

    fn released_model_config() -> ModelConfig {
        ModelConfig {
            use_duration_predictor: true,
            use_speaker_condition: Some(true),
            speaker_dim: Some(3),
            use_caption_condition: true,
            caption_dim: Some(2),
            duration_architecture: V4_DURATION_ARCHITECTURE.to_string(),
            duration_speaker_fusion: V4_DURATION_FUSION.to_string(),
            duration_caption_fusion: V4_DURATION_FUSION.to_string(),
            duration_caption_pooling: V4_CAPTION_POOLING.to_string(),
            ..Default::default()
        }
    }

    fn bool_1(data: Vec<bool>, len: usize) -> Tensor<B, 1, Bool> {
        Tensor::from_data(TensorData::new(data, [len]), &device())
    }

    fn bool_2(data: Vec<bool>, rows: usize, cols: usize) -> Tensor<B, 2, Bool> {
        Tensor::from_data(TensorData::new(data, [rows, cols]), &device())
    }

    #[test]
    fn rejects_unreleased_duration_variants_with_clear_config_errors() {
        let cases = [
            ("architecture", "pooled"),
            ("speaker fusion", "concat"),
            ("caption fusion", "concat"),
            ("caption pooling", "attention"),
        ];

        for (kind, value) in cases {
            let mut cfg = released_model_config();
            match kind {
                "architecture" => cfg.duration_architecture = value.to_string(),
                "speaker fusion" => cfg.duration_speaker_fusion = value.to_string(),
                "caption fusion" => cfg.duration_caption_fusion = value.to_string(),
                "caption pooling" => cfg.duration_caption_pooling = value.to_string(),
                _ => unreachable!(),
            }
            let error = DurationPredictorConfig::try_from(&cfg).expect_err("must reject variant");
            assert!(
                matches!(error, IrodoriError::Config(_)),
                "{kind} returned the wrong error: {error}"
            );
            assert!(
                error.to_string().contains("unsupported"),
                "{kind} error was not clear: {error}"
            );
        }
    }

    #[test]
    fn zero_initialization_sums_frames_per_valid_token_and_handles_all_masked_text() {
        let predictor = DurationPredictor::<B>::new(tiny_config(), &device()).unwrap();
        let text_state = Tensor::cat(
            vec![
                Tensor::zeros([1, 3, 4], &device()),
                Tensor::zeros([1, 3, 4], &device()).add_scalar(f32::NAN),
            ],
            0,
        );
        let output = predictor
            .forward(DurationPredictorInput {
                // Python overwrites a fully masked row. NaNs therefore must
                // not survive from this second input row into the prediction.
                text_state,
                text_mask: bool_2(vec![true, true, false, false, false, false], 2, 3),
                aux_features: Tensor::zeros([2, 2], &device()),
                speaker_state: None,
                has_speaker: bool_1(vec![false, false], 2),
                caption_state: None,
                caption_mask: None,
                has_caption: bool_1(vec![false, false], 2),
            })
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap();

        let expected_two_tokens = 19.0_f32.ln();
        let expected_safe_fallback = 10.0_f32.ln();
        assert!((output[0] - expected_two_tokens).abs() < 1e-5);
        assert!((output[1] - expected_safe_fallback).abs() < 1e-5);
    }

    #[test]
    fn caption_pooling_is_masked_mean_and_all_masked_uses_null_caption() {
        let predictor = DurationPredictor::<B>::new(tiny_config(), &device()).unwrap();
        let caption_state = Tensor::from_data(
            TensorData::new(
                vec![1.0_f32, 3.0, 100.0, 200.0, 7.0, 9.0, 11.0, 13.0],
                [2, 2, 2],
            ),
            &device(),
        );
        let pooled = predictor
            .caption_vec(
                2,
                Some(caption_state),
                Some(bool_2(vec![true, false, false, false], 2, 2)),
                bool_1(vec![true, true], 2),
            )
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap();

        assert_eq!(pooled, vec![1.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn presence_flags_select_null_vectors() {
        let predictor = DurationPredictor::<B>::new(tiny_config(), &device()).unwrap();
        let speaker_state = Tensor::ones([1, 1, 3], &device()) * 7.0;
        let caption_state = Tensor::ones([1, 1, 2], &device()) * 5.0;

        let speaker = predictor
            .speaker_vec(1, Some(speaker_state), bool_1(vec![false], 1))
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let caption = predictor
            .caption_vec(1, Some(caption_state), None, bool_1(vec![false], 1))
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap();

        assert_eq!(speaker, vec![0.0, 0.0, 0.0]);
        assert_eq!(caption, vec![0.0, 0.0]);
    }

    #[test]
    fn prepared_null_modulation_cache_preserves_no_aux_prediction() {
        let mut predictor = DurationPredictor::<B>::new(tiny_config(), &device()).unwrap();
        let text_state = Tensor::zeros([1, 3, 4], &device());
        let text_mask = bool_2(vec![true, true, false], 1, 3);
        let predict = |predictor: &DurationPredictor<B>| {
            predictor
                .forward(DurationPredictorInput {
                    text_state: text_state.clone(),
                    text_mask: text_mask.clone(),
                    aux_features: Tensor::zeros([1, 2], &device()),
                    speaker_state: None,
                    has_speaker: bool_1(vec![false], 1),
                    caption_state: None,
                    caption_mask: None,
                    has_caption: bool_1(vec![false], 1),
                })
                .unwrap()
                .to_data()
                .to_vec::<f32>()
                .unwrap()[0]
        };

        let ordinary = predict(&predictor);
        predictor.prepare_for_inference();
        assert!(
            predictor
                .token_blocks
                .iter()
                .all(DurationSwiGluBlock::has_cached_null)
        );
        let cached = predict(&predictor);
        assert!(ordinary.is_finite() && cached.is_finite());
        assert!((ordinary - cached).abs() <= 1.0e-5);
    }
}
