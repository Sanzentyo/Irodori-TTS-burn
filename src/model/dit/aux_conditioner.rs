//! Auxiliary conditioning modules for the DiT model.
//!
//! The model supports two mutually exclusive auxiliary conditioning modes:
//! - **Speaker**: reference-audio latent encoder + RMS norm
//! - **Caption**: text-caption encoder + RMS norm
//!
//! The [`AuxConditioner`] enum wraps both variants. Construction is handled by
//! [`build_aux_conditioner`], which reads the [`ModelConfig`] to decide which
//! variant (if any) to instantiate.

use burn::{
    module::Module,
    tensor::{Bool, Tensor, backend::Backend},
};

use crate::config::ModelConfig;

use super::super::{
    condition::{AuxConditionInput, AuxConditionState},
    norm::RmsNorm,
    speaker_encoder::{ReferenceLatentEncoder, patch_sequence_with_mask},
    text_encoder::{TextEncoder, TextEncoderSpec},
};

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
// Construction helper
// ---------------------------------------------------------------------------

/// Build the optional auxiliary conditioner (speaker XOR caption).
///
/// Returns `None` when neither speaker nor caption conditioning is configured.
pub(crate) fn build_aux_conditioner<B: Backend>(
    cfg: &ModelConfig,
    device: &B::Device,
) -> Option<AuxConditioner<B>> {
    if cfg.use_speaker_condition() {
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

    // --- encode mismatch tests ---

    #[test]
    fn encode_speaker_model_with_caption_input_returns_none() {
        let cfg = crate::config::tiny_model_config();
        let dev = device();
        let aux = build_aux_conditioner::<B>(&cfg, &dev).unwrap();
        assert!(matches!(aux, AuxConditioner::Speaker(_)));

        let ids = Tensor::<B, 2, Int>::zeros([1, 4], &dev);
        let mask = Tensor::<B, 2, Bool>::ones([1, 4], &dev);
        let result = aux
            .encode(AuxConditionInput::Caption { ids, mask }, 2)
            .unwrap();
        assert!(result.is_none(), "speaker model + caption input → None");
    }

    #[test]
    fn encode_caption_model_with_speaker_input_returns_none() {
        let cfg = crate::config::tiny_caption_config();
        let dev = device();
        let aux = build_aux_conditioner::<B>(&cfg, &dev).unwrap();
        assert!(matches!(aux, AuxConditioner::Caption(_)));

        let ref_latent = Tensor::<B, 3>::zeros([1, 4, cfg.model_dim], &dev);
        let ref_mask = Tensor::<B, 2, Bool>::ones([1, 4], &dev);
        let result = aux
            .encode(
                AuxConditionInput::Speaker {
                    ref_latent,
                    ref_mask,
                },
                2,
            )
            .unwrap();
        assert!(result.is_none(), "caption model + speaker input → None");
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
