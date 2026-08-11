//! Auxiliary conditioning modules for the DiT model.
//!
//! The model supports two auxiliary conditioning paths:
//! - **Speaker**: reference-audio latent encoder + RMS norm
//! - **Caption**: text-caption encoder + RMS norm
//!
//! v4 checkpoints can enable both paths simultaneously. The [`AuxConditioner`]
//! enum wraps the configured path(s). Construction is handled by
//! [`build_aux_conditioner`], which reads the [`ModelConfig`] to decide which
//! variant (if any) to instantiate.

use burn::{
    module::Module,
    tensor::{Bool, Tensor, backend::Backend},
};

use crate::config::ModelConfig;

use super::super::{
    condition::{AuxConditionInput, AuxConditionState, EncodedCondition},
    modern_bert::{ModernBertConfig, SharedModernBertConditioner},
    norm::RmsNorm,
    speaker_encoder::{ReferenceLatentEncoder, patch_sequence_with_mask},
    text_encoder::{TextEncoder, TextEncoderSpec},
};

mod wgsl;

// ---------------------------------------------------------------------------
// Full conditioning frontends
// ---------------------------------------------------------------------------

/// Legacy scratch text frontend and its optional scratch auxiliary encoder(s).
#[derive(Module, Debug)]
pub(crate) struct ScratchConditionFrontend<B: Backend> {
    pub(crate) text_encoder: TextEncoder<B>,
    pub(crate) text_norm: RmsNorm<B>,
    pub(crate) aux_conditioner: Option<AuxConditioner<B>>,
}

/// v4 shared-ModernBERT frontend.
///
/// Caption projection lives in `shared`; `caption_norm` records whether the
/// model configuration exposes that path. Speaker conditioning remains the
/// released reference-latent encoder and never allocates a scratch caption
/// encoder.
#[derive(Module, Debug)]
pub(crate) struct PretrainedConditionFrontend<B: Backend> {
    pub(crate) shared: SharedModernBertConditioner<B>,
    pub(crate) text_norm: RmsNorm<B>,
    pub(crate) speaker: Option<SpeakerConditioner<B>>,
    pub(crate) caption_norm: Option<RmsNorm<B>>,
}

/// Mutually exclusive conditioning frontend allocation.
// Burn's `Module` derive requires module fields by value; `Box<T>` doesn't
// implement `Module`, so indirection cannot be used to shrink this enum.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub(crate) enum ConditionFrontend<B: Backend> {
    Scratch(ScratchConditionFrontend<B>),
    Pretrained(PretrainedConditionFrontend<B>),
}

impl<B: Backend> ConditionFrontend<B> {
    pub(crate) fn from_model_config(
        cfg: &ModelConfig,
        device: &B::Device,
    ) -> crate::error::Result<Self> {
        if cfg.use_pretrained_text_encoder() {
            Self::pretrained(cfg, &ModernBertConfig::v4_small(), device)
        } else {
            Ok(Self::scratch(cfg, device))
        }
    }

    fn scratch(cfg: &ModelConfig, device: &B::Device) -> Self {
        Self::Scratch(ScratchConditionFrontend {
            text_encoder: TextEncoder::from_cfg(cfg, device),
            text_norm: RmsNorm::new(cfg.text_dim, cfg.norm_eps, device),
            aux_conditioner: build_aux_conditioner(cfg, device),
        })
    }

    pub(crate) fn pretrained(
        cfg: &ModelConfig,
        backbone_cfg: &ModernBertConfig,
        device: &B::Device,
    ) -> crate::error::Result<Self> {
        validate_pretrained_frontend_config(cfg, backbone_cfg)?;
        Ok(Self::Pretrained(PretrainedConditionFrontend {
            shared: SharedModernBertConditioner::new(
                backbone_cfg,
                cfg.text_dim,
                cfg.pretrained_projector_hidden_ratio,
                device,
            ),
            text_norm: RmsNorm::new(cfg.text_dim, cfg.norm_eps, device),
            speaker: cfg
                .use_speaker_condition()
                .then(|| build_speaker_conditioner(cfg, device)),
            caption_norm: cfg
                .use_caption_condition
                .then(|| RmsNorm::new(cfg.caption_dim(), cfg.norm_eps, device)),
        }))
    }

    /// Construct a frontend from a loaded record without allocating a second
    /// randomly initialized v4 ModernBERT backbone.
    pub(crate) fn from_record(
        cfg: &ModelConfig,
        record: ConditionFrontendRecord<B>,
        device: &B::Device,
    ) -> crate::error::Result<Self> {
        use crate::error::IrodoriError;

        cfg.validate()?;
        match (cfg.use_pretrained_text_encoder(), record) {
            (false, ConditionFrontendRecord::Scratch(record)) => {
                let Self::Scratch(frontend) = Self::scratch(cfg, device) else {
                    unreachable!("scratch constructor returned a pretrained frontend")
                };
                Ok(Self::Scratch(frontend.load_record(record)))
            }
            (true, ConditionFrontendRecord::Pretrained(record)) => {
                let backbone_cfg = ModernBertConfig::v4_small();
                validate_pretrained_frontend_config(cfg, &backbone_cfg)?;
                if cfg.text_dim != 512 {
                    return Err(IrodoriError::Config(format!(
                        "the v4-Small conditioner record projects to 512 dimensions, got text_dim={}",
                        cfg.text_dim
                    )));
                }
                if cfg.pretrained_projector_hidden_ratio != 2.0 {
                    return Err(IrodoriError::Config(format!(
                        "the v4-Small conditioner record requires pretrained_projector_hidden_ratio=2, got {}",
                        cfg.pretrained_projector_hidden_ratio
                    )));
                }

                let PretrainedConditionFrontendRecord {
                    shared,
                    text_norm,
                    speaker,
                    caption_norm,
                } = record;
                let speaker = match (cfg.use_speaker_condition(), speaker) {
                    (true, Some(record)) => {
                        Some(build_speaker_conditioner(cfg, device).load_record(record))
                    }
                    (false, None) => None,
                    (expected, actual) => {
                        return Err(IrodoriError::Config(format!(
                            "speaker conditioner record presence ({}) does not match configuration ({expected})",
                            actual.is_some()
                        )));
                    }
                };
                let caption_norm = match (cfg.use_caption_condition, caption_norm) {
                    (true, Some(record)) => Some(
                        RmsNorm::new(cfg.caption_dim(), cfg.norm_eps, device).load_record(record),
                    ),
                    (false, None) => None,
                    (expected, actual) => {
                        return Err(IrodoriError::Config(format!(
                            "caption norm record presence ({}) does not match configuration ({expected})",
                            actual.is_some()
                        )));
                    }
                };

                Ok(Self::Pretrained(PretrainedConditionFrontend {
                    shared: SharedModernBertConditioner::v4_small_from_record(shared, device),
                    text_norm: RmsNorm::new(cfg.text_dim, cfg.norm_eps, device)
                        .load_record(text_norm),
                    speaker,
                    caption_norm,
                }))
            }
            (true, ConditionFrontendRecord::Scratch(_)) => Err(IrodoriError::Config(
                "pretrained text configuration cannot load a scratch frontend record".to_string(),
            )),
            (false, ConditionFrontendRecord::Pretrained(_)) => Err(IrodoriError::Config(
                "scratch text configuration cannot load a pretrained frontend record".to_string(),
            )),
        }
    }

    pub(crate) fn encode(
        &self,
        text_input_ids: burn::tensor::Tensor<B, 2, burn::tensor::Int>,
        text_mask: Tensor<B, 2, Bool>,
        aux_input: AuxConditionInput<B>,
        speaker_patch_size: usize,
    ) -> crate::error::Result<EncodedCondition<B>> {
        match self {
            Self::Scratch(frontend) => {
                let text_state = frontend.text_norm.forward(
                    frontend
                        .text_encoder
                        .forward(text_input_ids, text_mask.clone()),
                );
                let aux = frontend
                    .aux_conditioner
                    .as_ref()
                    .map(|conditioner| conditioner.encode(aux_input, speaker_patch_size))
                    .transpose()?
                    .flatten();
                Ok(EncodedCondition {
                    text_state,
                    text_mask,
                    aux,
                })
            }
            Self::Pretrained(frontend) => {
                let text_state = frontend.text_norm.forward(
                    frontend
                        .shared
                        .encode_text(text_input_ids, text_mask.clone()),
                );
                let aux = frontend.encode_aux(aux_input, speaker_patch_size)?;
                Ok(EncodedCondition {
                    text_state,
                    text_mask,
                    aux,
                })
            }
        }
    }

    pub(crate) fn use_speaker_condition(&self) -> bool {
        match self {
            Self::Scratch(frontend) => frontend
                .aux_conditioner
                .as_ref()
                .is_some_and(AuxConditioner::is_speaker),
            Self::Pretrained(frontend) => frontend.speaker.is_some(),
        }
    }

    pub(crate) fn use_caption_condition(&self) -> bool {
        match self {
            Self::Scratch(frontend) => frontend
                .aux_conditioner
                .as_ref()
                .is_some_and(AuxConditioner::is_caption),
            Self::Pretrained(frontend) => frontend.caption_norm.is_some(),
        }
    }

    #[cfg(test)]
    pub(crate) fn is_pretrained(&self) -> bool {
        matches!(self, Self::Pretrained(_))
    }
}

impl<B: Backend> PretrainedConditionFrontend<B> {
    fn encode_aux(
        &self,
        input: AuxConditionInput<B>,
        speaker_patch_size: usize,
    ) -> crate::error::Result<Option<AuxConditionState<B>>> {
        match input {
            AuxConditionInput::Speaker {
                ref_latent,
                ref_mask,
            } => {
                let speaker = self.speaker.as_ref().ok_or_else(|| {
                    crate::error::IrodoriError::Config(
                        "speaker input supplied to a pretrained frontend without speaker conditioning"
                            .to_string(),
                    )
                })?;
                let (state, mask) =
                    encode_speaker(speaker, ref_latent, ref_mask, speaker_patch_size)?;
                Ok(Some(AuxConditionState::Speaker { state, mask }))
            }
            AuxConditionInput::Caption { ids, mask } => {
                let (state, mask) = self.encode_caption(ids, mask)?;
                Ok(Some(AuxConditionState::Caption { state, mask }))
            }
            AuxConditionInput::Both {
                ref_latent,
                ref_mask,
                caption_ids,
                caption_mask,
            } => {
                let speaker = self.speaker.as_ref().ok_or_else(|| {
                    crate::error::IrodoriError::Config(
                        "speaker+caption input requires pretrained speaker conditioning"
                            .to_string(),
                    )
                })?;
                let (speaker_state, speaker_mask) =
                    encode_speaker(speaker, ref_latent, ref_mask, speaker_patch_size)?;
                let (caption_state, caption_mask) =
                    self.encode_caption(caption_ids, caption_mask)?;
                Ok(Some(AuxConditionState::Both {
                    speaker_state,
                    speaker_mask,
                    caption_state,
                    caption_mask,
                }))
            }
            AuxConditionInput::None => Ok(None),
        }
    }

    fn encode_caption(
        &self,
        ids: burn::tensor::Tensor<B, 2, burn::tensor::Int>,
        mask: Tensor<B, 2, Bool>,
    ) -> crate::error::Result<(Tensor<B, 3>, Tensor<B, 2, Bool>)> {
        let norm = self.caption_norm.as_ref().ok_or_else(|| {
            crate::error::IrodoriError::Config(
                "caption input supplied to a pretrained frontend without caption conditioning"
                    .to_string(),
            )
        })?;
        let state = norm.forward(self.shared.encode_caption(ids, mask.clone()));
        Ok((state, mask))
    }
}

fn validate_pretrained_frontend_config(
    cfg: &ModelConfig,
    backbone_cfg: &ModernBertConfig,
) -> crate::error::Result<()> {
    use crate::error::IrodoriError;

    if !cfg.use_pretrained_text_encoder() {
        return Err(IrodoriError::Config(
            "a pretrained conditioning frontend requires text_encoder_type=\"pretrained\""
                .to_string(),
        ));
    }
    if !cfg
        .pretrained_projector_type
        .trim()
        .eq_ignore_ascii_case("residual_mlp")
    {
        return Err(IrodoriError::Config(
            "the embedded v4 frontend requires pretrained_projector_type=\"residual_mlp\""
                .to_string(),
        ));
    }
    if cfg.pretrained_projector_dropout != 0.0 {
        return Err(IrodoriError::Config(
            "the embedded v4 frontend currently requires pretrained_projector_dropout=0"
                .to_string(),
        ));
    }
    if cfg.text_vocab_size != backbone_cfg.vocab_size {
        return Err(IrodoriError::Config(format!(
            "pretrained text_vocab_size {} does not match embedded backbone vocab_size {}",
            cfg.text_vocab_size, backbone_cfg.vocab_size
        )));
    }
    if cfg.use_caption_condition && cfg.caption_vocab_size() != backbone_cfg.vocab_size {
        return Err(IrodoriError::Config(format!(
            "pretrained caption_vocab_size {} does not match embedded backbone vocab_size {}",
            cfg.caption_vocab_size(),
            backbone_cfg.vocab_size
        )));
    }
    if cfg.use_caption_condition && cfg.caption_dim() != cfg.text_dim {
        return Err(IrodoriError::Config(format!(
            "shared pretrained frontend requires caption_dim ({}) to equal text_dim ({})",
            cfg.caption_dim(),
            cfg.text_dim
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Conditioner structs
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

/// Speaker and caption conditioning modules enabled together (v4).
#[derive(Module, Debug)]
pub struct BothConditioner<B: Backend> {
    pub(crate) speaker: SpeakerConditioner<B>,
    pub(crate) caption: CaptionConditioner<B>,
}

/// Auxiliary conditioning module for the configured path(s).
///
/// Wrapped in `Option` in `TextToLatentRfDiT` so models without any auxiliary
/// conditioning are represented as `None` rather than a phantom unit variant.
// Burn's `Module` derive needs module fields by value; `Box<T>` doesn't implement
// `Module`, so the otherwise-preferred enum indirection isn't available here.
#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum AuxConditioner<B: Backend> {
    /// Reference-audio (speaker) conditioning path.
    Speaker(SpeakerConditioner<B>),
    /// Text-caption conditioning path.
    Caption(CaptionConditioner<B>),
    /// Concurrent reference-audio and caption conditioning paths.
    Both(BothConditioner<B>),
}

impl<B: Backend> AuxConditioner<B> {
    pub(crate) fn is_speaker(&self) -> bool {
        matches!(self, Self::Speaker(_) | Self::Both(_))
    }

    pub(crate) fn is_caption(&self) -> bool {
        matches!(self, Self::Caption(_) | Self::Both(_))
    }

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
                let (sp_state, sp_mask) =
                    encode_speaker(sp, ref_latent, ref_mask, speaker_patch_size)?;
                Ok(Some(AuxConditionState::Speaker {
                    state: sp_state,
                    mask: sp_mask,
                }))
            }
            (Self::Caption(cap), AuxConditionInput::Caption { ids, mask }) => {
                let (cap_state, mask) = encode_caption(cap, ids, mask);
                Ok(Some(AuxConditionState::Caption {
                    state: cap_state,
                    mask,
                }))
            }
            (
                Self::Both(both),
                AuxConditionInput::Both {
                    ref_latent,
                    ref_mask,
                    caption_ids,
                    caption_mask,
                },
            ) => {
                let (speaker_state, speaker_mask) =
                    encode_speaker(&both.speaker, ref_latent, ref_mask, speaker_patch_size)?;
                let (caption_state, caption_mask) =
                    encode_caption(&both.caption, caption_ids, caption_mask);
                Ok(Some(AuxConditionState::Both {
                    speaker_state,
                    speaker_mask,
                    caption_state,
                    caption_mask,
                }))
            }
            (
                Self::Both(both),
                AuxConditionInput::Speaker {
                    ref_latent,
                    ref_mask,
                },
            ) => {
                let (state, mask) =
                    encode_speaker(&both.speaker, ref_latent, ref_mask, speaker_patch_size)?;
                Ok(Some(AuxConditionState::Speaker { state, mask }))
            }
            (Self::Both(both), AuxConditionInput::Caption { ids, mask }) => {
                let (state, mask) = encode_caption(&both.caption, ids, mask);
                Ok(Some(AuxConditionState::Caption { state, mask }))
            }
            // Mismatched conditioning type — surface an actionable error rather
            // than silently discarding the aux input and producing wrong output.
            (
                Self::Speaker(_),
                AuxConditionInput::Caption { .. } | AuxConditionInput::Both { .. },
            ) => Err(crate::error::IrodoriError::Config(
                "speaker conditioner received caption input; \
                     check that the checkpoint and inference request use the same conditioning mode"
                    .into(),
            )),
            (
                Self::Caption(_),
                AuxConditionInput::Speaker { .. } | AuxConditionInput::Both { .. },
            ) => Err(crate::error::IrodoriError::Config(
                "caption conditioner received speaker input; \
                     check that the checkpoint and inference request use the same conditioning mode"
                    .into(),
            )),
            // No aux input — valid for any conditioner (fully unconditional pass).
            _ => Ok(None),
        }
    }
}

fn encode_speaker<B: Backend>(
    conditioner: &SpeakerConditioner<B>,
    ref_latent: Tensor<B, 3>,
    ref_mask: Tensor<B, 2, Bool>,
    speaker_patch_size: usize,
) -> crate::error::Result<(Tensor<B, 3>, Tensor<B, 2, Bool>)> {
    let (patched_latent, patched_mask) =
        patch_sequence_with_mask(ref_latent, ref_mask, speaker_patch_size)?;
    let state = conditioner.norm.forward(
        conditioner
            .encoder
            .forward(patched_latent, patched_mask.clone()),
    );
    Ok(prepend_masked_mean_token(state, patched_mask))
}

fn encode_caption<B: Backend>(
    conditioner: &CaptionConditioner<B>,
    ids: burn::tensor::Tensor<B, 2, burn::tensor::Int>,
    mask: Tensor<B, 2, Bool>,
) -> (Tensor<B, 3>, Tensor<B, 2, Bool>) {
    let state = conditioner
        .norm
        .forward(conditioner.encoder.forward(ids, mask.clone()));
    (state, mask)
}

// ---------------------------------------------------------------------------
// Construction helper
// ---------------------------------------------------------------------------

/// Build the optional auxiliary conditioner.
///
/// Returns `None` when neither speaker nor caption conditioning is configured.
pub(crate) fn build_aux_conditioner<B: Backend>(
    cfg: &ModelConfig,
    device: &B::Device,
) -> Option<AuxConditioner<B>> {
    debug_assert!(
        !cfg.use_pretrained_text_encoder(),
        "scratch auxiliary conditioner must not be built for a pretrained frontend"
    );
    let speaker = cfg
        .use_speaker_condition()
        .then(|| build_speaker_conditioner(cfg, device));
    let caption = cfg.use_caption_condition.then(|| CaptionConditioner {
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
    });

    match (speaker, caption) {
        (Some(speaker), Some(caption)) => {
            Some(AuxConditioner::Both(BothConditioner { speaker, caption }))
        }
        (Some(speaker), None) => Some(AuxConditioner::Speaker(speaker)),
        (None, Some(caption)) => Some(AuxConditioner::Caption(caption)),
        (None, None) => None,
    }
}

fn build_speaker_conditioner<B: Backend>(
    cfg: &ModelConfig,
    device: &B::Device,
) -> SpeakerConditioner<B> {
    let speaker_dim = cfg
        .speaker_dim
        .expect("speaker_dim required for speaker conditioning");
    SpeakerConditioner {
        encoder: ReferenceLatentEncoder::from_cfg(cfg, device),
        norm: RmsNorm::new(speaker_dim, cfg.norm_eps, device),
    }
}

// ---------------------------------------------------------------------------
// Speaker mean-token prepend
// ---------------------------------------------------------------------------

/// Prepend a masked-mean summary token to a speaker-encoded sequence.
///
/// - `state: [B, S, D]`, `mask: [B, S]`
/// - Returns `(state': [B, S+1, D], mask': [B, S+1])`
pub(super) fn prepend_masked_mean_token<B: Backend>(
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
    use burn::tensor::{Int, TensorData};

    type B = NdArray<f32>;

    fn device() -> <B as Backend>::Device {
        Default::default()
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

    // --- build_aux_conditioner ---

    #[test]
    fn build_aux_conditioner_speaker_mode() {
        let cfg = crate::config::tiny_model_config();
        let dev = device();
        let aux = build_aux_conditioner::<B>(&cfg, &dev);
        assert!(
            matches!(aux, Some(AuxConditioner::Speaker(_))),
            "speaker config should produce Speaker variant"
        );
    }

    #[test]
    fn build_aux_conditioner_caption_mode() {
        let cfg = crate::config::tiny_caption_config();
        let dev = device();
        let aux = build_aux_conditioner::<B>(&cfg, &dev);
        assert!(
            matches!(aux, Some(AuxConditioner::Caption(_))),
            "caption config should produce Caption variant"
        );
    }

    #[test]
    fn build_aux_conditioner_no_caption_defaults_to_speaker() {
        let cfg = crate::config::tiny_model_config();
        assert!(!cfg.use_caption_condition);
        let dev = device();
        let aux = build_aux_conditioner::<B>(&cfg, &dev);
        assert!(
            matches!(aux, Some(AuxConditioner::Speaker(_))),
            "non-caption config defaults to speaker"
        );
    }

    #[test]
    fn build_and_encode_both_conditioners() {
        let mut cfg = crate::config::tiny_model_config();
        cfg.use_speaker_condition = Some(true);
        cfg.use_caption_condition = true;
        cfg.caption_vocab_size = Some(32);
        cfg.caption_dim = Some(16);
        cfg.caption_layers = Some(1);
        cfg.caption_heads = Some(2);
        cfg.caption_mlp_ratio = Some(2.0);
        cfg.validate().unwrap();

        let dev = device();
        let aux = build_aux_conditioner::<B>(&cfg, &dev).unwrap();
        assert!(matches!(aux, AuxConditioner::Both(_)));

        let encoded = aux
            .encode(
                AuxConditionInput::Both {
                    ref_latent: Tensor::zeros([1, 4, cfg.patched_latent_dim()], &dev),
                    ref_mask: Tensor::ones([1, 4], &dev),
                    caption_ids: Tensor::zeros([1, 3], &dev),
                    caption_mask: Tensor::ones([1, 3], &dev),
                },
                cfg.speaker_patch_size.unwrap(),
            )
            .unwrap()
            .unwrap();

        let (speaker_state, speaker_mask) = encoded.speaker().unwrap();
        let (caption_state, caption_mask) = encoded.caption().unwrap();
        assert_eq!(speaker_state.dims(), [1, 5, cfg.speaker_dim.unwrap()]);
        assert_eq!(speaker_mask.dims(), [1, 5]);
        assert_eq!(caption_state.dims(), [1, 3, cfg.caption_dim()]);
        assert_eq!(caption_mask.dims(), [1, 3]);
    }

    // --- encode mismatch tests ---

    #[test]
    fn encode_speaker_model_with_caption_input_returns_error() {
        let cfg = crate::config::tiny_model_config();
        let dev = device();
        let aux = build_aux_conditioner::<B>(&cfg, &dev).unwrap();
        assert!(matches!(aux, AuxConditioner::Speaker(_)));

        let ids = Tensor::<B, 2, Int>::zeros([1, 4], &dev);
        let mask = Tensor::<B, 2, Bool>::ones([1, 4], &dev);
        let result = aux.encode(AuxConditionInput::Caption { ids, mask }, 2);
        assert!(
            result.is_err(),
            "speaker model + caption input must return an error, not silently discard conditioning"
        );
    }

    #[test]
    fn encode_caption_model_with_speaker_input_returns_error() {
        let cfg = crate::config::tiny_caption_config();
        let dev = device();
        let aux = build_aux_conditioner::<B>(&cfg, &dev).unwrap();
        assert!(matches!(aux, AuxConditioner::Caption(_)));

        let ref_latent = Tensor::<B, 3>::zeros([1, 4, cfg.model_dim], &dev);
        let ref_mask = Tensor::<B, 2, Bool>::ones([1, 4], &dev);
        let result = aux.encode(
            AuxConditionInput::Speaker {
                ref_latent,
                ref_mask,
            },
            2,
        );
        assert!(
            result.is_err(),
            "caption model + speaker input must return an error, not silently discard conditioning"
        );
    }

    #[test]
    fn encode_with_none_input_returns_none() {
        let cfg = crate::config::tiny_model_config();
        let dev = device();
        let aux = build_aux_conditioner::<B>(&cfg, &dev).unwrap();
        let result = aux.encode(AuxConditionInput::None, 2).unwrap();
        assert!(result.is_none(), "any model + None input → None");
    }
}
