//! [`EncodedCondition`] — the runtime bundle of all conditioning tensors.

use burn::tensor::Device;
use burn::tensor::{Bool, Int, Tensor};

// ---------------------------------------------------------------------------
// AuxConditionState — runtime tensor bundle for aux conditioning
// ---------------------------------------------------------------------------

/// Runtime tensor bundle for auxiliary conditioning.
///
/// v2/v3 checkpoints use a single speaker or caption context. v4 checkpoints
/// can carry both contexts at once. Use [`Option<AuxConditionState>`] to
/// represent the "no aux conditioning" case.
// Tensor handles are intentionally inline: these ADTs replace invalid paired
// Options at a per-request boundary, where boxing would add avoidable heap work.
#[allow(clippy::large_enum_variant)]
pub enum AuxConditionState {
    /// Speaker (reference audio) conditioning is active.
    Speaker {
        state: Tensor<3>,
        mask: Tensor<2, Bool>,
    },
    /// Caption conditioning is active.
    Caption {
        state: Tensor<3>,
        mask: Tensor<2, Bool>,
    },
    /// Speaker and caption conditioning are both active (v4).
    Both {
        speaker_state: Tensor<3>,
        speaker_mask: Tensor<2, Bool>,
        caption_state: Tensor<3>,
        caption_mask: Tensor<2, Bool>,
    },
}

impl AuxConditionState {
    /// Whether this is the Speaker variant.
    pub fn is_speaker(&self) -> bool {
        matches!(self, Self::Speaker { .. } | Self::Both { .. })
    }

    /// Whether this is the Caption variant.
    pub fn is_caption(&self) -> bool {
        matches!(self, Self::Caption { .. } | Self::Both { .. })
    }

    /// Return the speaker context when present.
    pub fn speaker(&self) -> Option<(&Tensor<3>, &Tensor<2, Bool>)> {
        match self {
            Self::Speaker { state, mask } => Some((state, mask)),
            Self::Both {
                speaker_state,
                speaker_mask,
                ..
            } => Some((speaker_state, speaker_mask)),
            Self::Caption { .. } => None,
        }
    }

    /// Return the caption context when present.
    pub fn caption(&self) -> Option<(&Tensor<3>, &Tensor<2, Bool>)> {
        match self {
            Self::Caption { state, mask } => Some((state, mask)),
            Self::Both {
                caption_state,
                caption_mask,
                ..
            } => Some((caption_state, caption_mask)),
            Self::Speaker { .. } => None,
        }
    }

    /// Returns the sole `(&state, &mask)` context for legacy single-context variants.
    ///
    /// # Panics
    ///
    /// Panics for [`Self::Both`]. Call [`Self::speaker`] and [`Self::caption`]
    /// when both contexts may be enabled.
    pub fn state_and_mask(&self) -> (&Tensor<3>, &Tensor<2, Bool>) {
        match self {
            Self::Speaker { state, mask } | Self::Caption { state, mask } => (state, mask),
            Self::Both { .. } => {
                panic!("state_and_mask is ambiguous for speaker+caption conditioning")
            }
        }
    }

    /// Produce an all-zero version with the **same variant**.
    ///
    /// Preserving the variant is critical so CFG can nullify the correct signal
    /// without collapsing to "no aux conditioning at all".
    pub fn zeros_like(&self, device: &Device) -> Self {
        match self {
            Self::Speaker { state, mask } => Self::Speaker {
                state: Tensor::zeros(state.dims(), device),
                mask: Tensor::<2>::zeros(mask.dims(), device).greater_elem(0.0),
            },
            Self::Caption { state, mask } => Self::Caption {
                state: Tensor::zeros(state.dims(), device),
                mask: Tensor::<2>::zeros(mask.dims(), device).greater_elem(0.0),
            },
            Self::Both {
                speaker_state,
                speaker_mask,
                caption_state,
                caption_mask,
            } => Self::Both {
                speaker_state: Tensor::zeros(speaker_state.dims(), device),
                speaker_mask: Tensor::<2>::zeros(speaker_mask.dims(), device).greater_elem(0.0),
                caption_state: Tensor::zeros(caption_state.dims(), device),
                caption_mask: Tensor::<2>::zeros(caption_mask.dims(), device).greater_elem(0.0),
            },
        }
    }

    /// Zero only the speaker context, retaining text-independent caption state.
    pub fn speaker_unconditional(&self, device: &Device) -> Self {
        match self {
            Self::Speaker { state, mask } => Self::Speaker {
                state: Tensor::zeros(state.dims(), device),
                mask: Tensor::<2>::zeros(mask.dims(), device).greater_elem(0.0),
            },
            Self::Caption { .. } => self.clone(),
            Self::Both {
                speaker_state,
                speaker_mask,
                caption_state,
                caption_mask,
            } => Self::Both {
                speaker_state: Tensor::zeros(speaker_state.dims(), device),
                speaker_mask: Tensor::<2>::zeros(speaker_mask.dims(), device).greater_elem(0.0),
                caption_state: caption_state.clone(),
                caption_mask: caption_mask.clone(),
            },
        }
    }

    /// Zero only the caption context, retaining speaker state.
    pub fn caption_unconditional(&self, device: &Device) -> Self {
        match self {
            Self::Speaker { .. } => self.clone(),
            Self::Caption { state, mask } => Self::Caption {
                state: Tensor::zeros(state.dims(), device),
                mask: Tensor::<2>::zeros(mask.dims(), device).greater_elem(0.0),
            },
            Self::Both {
                speaker_state,
                speaker_mask,
                caption_state,
                caption_mask,
            } => Self::Both {
                speaker_state: speaker_state.clone(),
                speaker_mask: speaker_mask.clone(),
                caption_state: Tensor::zeros(caption_state.dims(), device),
                caption_mask: Tensor::<2>::zeros(caption_mask.dims(), device).greater_elem(0.0),
            },
        }
    }
}

impl Clone for AuxConditionState {
    fn clone(&self) -> Self {
        match self {
            Self::Speaker { state, mask } => Self::Speaker {
                state: state.clone(),
                mask: mask.clone(),
            },
            Self::Caption { state, mask } => Self::Caption {
                state: state.clone(),
                mask: mask.clone(),
            },
            Self::Both {
                speaker_state,
                speaker_mask,
                caption_state,
                caption_mask,
            } => Self::Both {
                speaker_state: speaker_state.clone(),
                speaker_mask: speaker_mask.clone(),
                caption_state: caption_state.clone(),
                caption_mask: caption_mask.clone(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// AuxConditionInput — typed input for aux encoder dispatch
// ---------------------------------------------------------------------------

/// Typed input bundle passed to `AuxConditioner::encode`.
///
/// Makes each supported input combination and the no-conditioning case explicit.
#[allow(clippy::large_enum_variant)]
pub enum AuxConditionInput {
    /// Reference audio latent + mask for speaker conditioning.
    Speaker {
        ref_latent: Tensor<3>,
        ref_mask: Tensor<2, Bool>,
    },
    /// Token IDs + mask for caption conditioning.
    Caption {
        ids: Tensor<2, Int>,
        mask: Tensor<2, Bool>,
    },
    /// Reference audio and caption inputs supplied together (v4).
    Both {
        ref_latent: Tensor<3>,
        ref_mask: Tensor<2, Bool>,
        caption_ids: Tensor<2, Int>,
        caption_mask: Tensor<2, Bool>,
    },
    /// No auxiliary input supplied.
    None,
}

impl AuxConditionInput {
    /// Construct from raw optional fields (e.g., from `SamplingRequest`).
    ///
    /// Each tensor/mask pair is atomic. Supplying only one half is an error
    /// rather than silently dropping that conditioning signal.
    pub fn try_from_request(
        ref_latent: Option<Tensor<3>>,
        ref_mask: Option<Tensor<2, Bool>>,
        caption_ids: Option<Tensor<2, Int>>,
        caption_mask: Option<Tensor<2, Bool>>,
    ) -> crate::error::Result<Self> {
        use crate::error::IrodoriError;

        let speaker = match (ref_latent, ref_mask) {
            (Some(ref_latent), Some(ref_mask)) => Some((ref_latent, ref_mask)),
            (Some(_), None) => {
                return Err(IrodoriError::MissingInput(
                    "ref_mask must be supplied together with ref_latent".to_string(),
                ));
            }
            (None, Some(_)) => {
                return Err(IrodoriError::MissingInput(
                    "ref_latent must be supplied together with ref_mask".to_string(),
                ));
            }
            (None, None) => None,
        };
        let caption = match (caption_ids, caption_mask) {
            (Some(ids), Some(mask)) => Some((ids, mask)),
            (Some(_), None) => {
                return Err(IrodoriError::MissingInput(
                    "caption_mask must be supplied together with caption_ids".to_string(),
                ));
            }
            (None, Some(_)) => {
                return Err(IrodoriError::MissingInput(
                    "caption_ids must be supplied together with caption_mask".to_string(),
                ));
            }
            (None, None) => None,
        };

        Ok(match (speaker, caption) {
            (Some((ref_latent, ref_mask)), Some((caption_ids, caption_mask))) => Self::Both {
                ref_latent,
                ref_mask,
                caption_ids,
                caption_mask,
            },
            (Some((ref_latent, ref_mask)), None) => Self::Speaker {
                ref_latent,
                ref_mask,
            },
            (None, Some((ids, mask))) => Self::Caption { ids, mask },
            (None, None) => Self::None,
        })
    }
}

// ---------------------------------------------------------------------------
// EncodedCondition — full runtime bundle
// ---------------------------------------------------------------------------

/// All encoded conditioning tensors for one forward pass.
///
/// `aux` is `None` when the model uses no auxiliary conditioning.  For
/// CFG-unconditional passes it is `Some(zeroed)` — the variant is preserved
/// so the sampler can still nullify the correct signal.
pub struct EncodedCondition {
    pub text_state: Tensor<3>,
    pub text_mask: Tensor<2, Bool>,
    /// Speaker and/or caption encoded state; `None` when not used by this model.
    pub aux: Option<AuxConditionState>,
}

impl Clone for EncodedCondition {
    fn clone(&self) -> Self {
        Self {
            text_state: self.text_state.clone(),
            text_mask: self.text_mask.clone(),
            aux: self.aux.clone(),
        }
    }
}

impl EncodedCondition {
    /// Create an all-zero unconditional version of this condition.
    ///
    /// State tensors are zeroed; Bool masks are all-False.
    /// The `aux` variant is preserved so CFG can still nullify the correct signal.
    pub fn zeros_like(&self, device: &Device) -> Self {
        let zero_text = Tensor::zeros(self.text_state.dims(), device);
        let zero_text_mask: Tensor<2, Bool> =
            Tensor::<2>::zeros(self.text_mask.dims(), device).greater_elem(0.0);

        Self {
            text_state: zero_text,
            text_mask: zero_text_mask,
            aux: self.aux.as_ref().map(|a| a.zeros_like(device)),
        }
    }

    /// Concatenate multiple conditions along the batch dimension.
    ///
    /// Used for batched Independent CFG: instead of N sequential forward
    /// passes with batch=1, concatenate all conditioning variants into a
    /// single `EncodedCondition` with batch=N and run one forward pass.
    ///
    /// All conditions must have the same `aux` variant (all `Some(Speaker)`,
    /// all `Some(Caption)`, all `Some(Both)`, or all `None`).
    ///
    /// # Panics
    ///
    /// Panics if `conditions` is empty or if `aux` variants are inconsistent.
    pub fn cat_batch(conditions: &[&Self]) -> Self {
        assert!(
            !conditions.is_empty(),
            "cat_batch requires at least one condition"
        );

        let text_states: Vec<_> = conditions.iter().map(|c| c.text_state.clone()).collect();
        let text_masks: Vec<_> = conditions.iter().map(|c| c.text_mask.clone()).collect();

        let text_state = Tensor::cat(text_states, 0);
        let text_mask = Tensor::cat(text_masks, 0);

        let aux = match &conditions[0].aux {
            None => {
                debug_assert!(
                    conditions.iter().all(|c| c.aux.is_none()),
                    "cat_batch: mixed aux variants (first is None, others have Some)"
                );
                None
            }
            Some(AuxConditionState::Speaker { .. }) => Some(AuxConditionState::Speaker {
                state: Tensor::cat(
                    conditions
                        .iter()
                        .map(|c| match c.aux.as_ref() {
                            Some(AuxConditionState::Speaker { state, .. }) => state.clone(),
                            _ => panic!("cat_batch: mixed aux variants"),
                        })
                        .collect(),
                    0,
                ),
                mask: Tensor::cat(
                    conditions
                        .iter()
                        .map(|c| match c.aux.as_ref() {
                            Some(AuxConditionState::Speaker { mask, .. }) => mask.clone(),
                            _ => panic!("cat_batch: mixed aux variants"),
                        })
                        .collect(),
                    0,
                ),
            }),
            Some(AuxConditionState::Caption { .. }) => Some(AuxConditionState::Caption {
                state: Tensor::cat(
                    conditions
                        .iter()
                        .map(|c| match c.aux.as_ref() {
                            Some(AuxConditionState::Caption { state, .. }) => state.clone(),
                            _ => panic!("cat_batch: mixed aux variants"),
                        })
                        .collect(),
                    0,
                ),
                mask: Tensor::cat(
                    conditions
                        .iter()
                        .map(|c| match c.aux.as_ref() {
                            Some(AuxConditionState::Caption { mask, .. }) => mask.clone(),
                            _ => panic!("cat_batch: mixed aux variants"),
                        })
                        .collect(),
                    0,
                ),
            }),
            Some(AuxConditionState::Both { .. }) => Some(AuxConditionState::Both {
                speaker_state: Tensor::cat(
                    conditions
                        .iter()
                        .map(|c| match c.aux.as_ref() {
                            Some(AuxConditionState::Both { speaker_state, .. }) => {
                                speaker_state.clone()
                            }
                            _ => panic!("cat_batch: mixed aux variants"),
                        })
                        .collect(),
                    0,
                ),
                speaker_mask: Tensor::cat(
                    conditions
                        .iter()
                        .map(|c| match c.aux.as_ref() {
                            Some(AuxConditionState::Both { speaker_mask, .. }) => {
                                speaker_mask.clone()
                            }
                            _ => panic!("cat_batch: mixed aux variants"),
                        })
                        .collect(),
                    0,
                ),
                caption_state: Tensor::cat(
                    conditions
                        .iter()
                        .map(|c| match c.aux.as_ref() {
                            Some(AuxConditionState::Both { caption_state, .. }) => {
                                caption_state.clone()
                            }
                            _ => panic!("cat_batch: mixed aux variants"),
                        })
                        .collect(),
                    0,
                ),
                caption_mask: Tensor::cat(
                    conditions
                        .iter()
                        .map(|c| match c.aux.as_ref() {
                            Some(AuxConditionState::Both { caption_mask, .. }) => {
                                caption_mask.clone()
                            }
                            _ => panic!("cat_batch: mixed aux variants"),
                        })
                        .collect(),
                    0,
                ),
            }),
        };

        Self {
            text_state,
            text_mask,
            aux,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn dev() -> Device {
        Default::default()
    }

    // --- AuxConditionState ---

    #[test]
    fn aux_state_speaker_variant_identification() {
        let d = dev();
        let state = AuxConditionState::Speaker {
            state: Tensor::zeros([1, 4, 8], &d),
            mask: Tensor::ones([1, 4], &d),
        };
        assert!(state.is_speaker());
        assert!(!state.is_caption());
    }

    #[test]
    fn aux_state_caption_variant_identification() {
        let d = dev();
        let state = AuxConditionState::Caption {
            state: Tensor::zeros([1, 4, 8], &d),
            mask: Tensor::ones([1, 4], &d),
        };
        assert!(state.is_caption());
        assert!(!state.is_speaker());
    }

    #[test]
    fn aux_state_state_and_mask_shapes() {
        let d = dev();
        let state = AuxConditionState::Speaker {
            state: Tensor::zeros([2, 5, 16], &d),
            mask: Tensor::ones([2, 5], &d),
        };
        let (s, m) = state.state_and_mask();
        assert_eq!(s.dims(), [2, 5, 16]);
        assert_eq!(m.dims(), [2, 5]);
    }

    #[test]
    fn aux_state_zeros_like_preserves_variant_and_shape() {
        let d = dev();
        let original = AuxConditionState::Speaker {
            state: Tensor::ones([1, 3, 8], &d),
            mask: Tensor::ones([1, 3], &d),
        };
        let zeroed = original.zeros_like(&d);

        assert!(zeroed.is_speaker());
        let (s, m) = zeroed.state_and_mask();
        assert_eq!(s.dims(), [1, 3, 8]);
        assert_eq!(m.dims(), [1, 3]);

        let sum: f32 = s.clone().abs().sum().to_data().to_vec::<f32>().unwrap()[0];
        assert_eq!(sum, 0.0);

        let mask_sum: i32 = m.clone().int().sum().to_data().to_vec::<i32>().unwrap()[0];
        assert_eq!(mask_sum, 0);
    }

    #[test]
    fn aux_state_clone_preserves_values() {
        let d = dev();
        let original = AuxConditionState::Caption {
            state: Tensor::ones([1, 2, 4], &d) * 3.0,
            mask: Tensor::ones([1, 2], &d),
        };
        let cloned = original.clone();
        assert!(cloned.is_caption());
        let (s, _) = cloned.state_and_mask();
        let vals: Vec<f32> = s.clone().to_data().to_vec().unwrap();
        assert!(vals.iter().all(|v| (*v - 3.0).abs() < 1e-6));
    }

    // --- AuxConditionInput ---

    #[test]
    fn input_from_request_preserves_speaker_and_caption() {
        let d = dev();
        let lat = Some(Tensor::<3>::zeros([1, 2, 8], &d));
        let mask = Some(Tensor::<2, Bool>::ones([1, 2], &d));
        let cap_ids = Some(Tensor::<2, Int>::zeros([1, 4], &d));
        let cap_mask = Some(Tensor::<2, Bool>::ones([1, 4], &d));

        let input = AuxConditionInput::try_from_request(lat, mask, cap_ids, cap_mask).unwrap();
        assert!(matches!(input, AuxConditionInput::Both { .. }));
    }

    #[test]
    fn input_from_request_caption_fallback() {
        let d = dev();
        let cap_ids = Some(Tensor::<2, Int>::zeros([1, 4], &d));
        let cap_mask = Some(Tensor::<2, Bool>::ones([1, 4], &d));

        let input = AuxConditionInput::try_from_request(None, None, cap_ids, cap_mask).unwrap();
        assert!(matches!(input, AuxConditionInput::Caption { .. }));
    }

    #[test]
    fn input_from_request_none() {
        let input = AuxConditionInput::try_from_request(None, None, None, None).unwrap();
        assert!(matches!(input, AuxConditionInput::None));
    }

    #[test]
    fn input_from_request_rejects_half_pairs() {
        let d = dev();
        let latent = Tensor::<3>::zeros([1, 2, 8], &d);
        let ref_mask = Tensor::<2, Bool>::ones([1, 2], &d);
        let caption_ids = Tensor::<2, Int>::zeros([1, 4], &d);
        let caption_mask = Tensor::<2, Bool>::ones([1, 4], &d);

        assert!(matches!(
            AuxConditionInput::try_from_request(Some(latent), None, None, None),
            Err(crate::error::IrodoriError::MissingInput(_))
        ));
        assert!(matches!(
            AuxConditionInput::try_from_request(None, Some(ref_mask), None, None),
            Err(crate::error::IrodoriError::MissingInput(_))
        ));
        assert!(matches!(
            AuxConditionInput::try_from_request(None, None, Some(caption_ids), None),
            Err(crate::error::IrodoriError::MissingInput(_))
        ));
        assert!(matches!(
            AuxConditionInput::try_from_request(None, None, None, Some(caption_mask)),
            Err(crate::error::IrodoriError::MissingInput(_))
        ));
    }

    // --- EncodedCondition ---

    #[test]
    fn encoded_condition_zeros_like_shapes_and_values() {
        let d = dev();
        let cond = EncodedCondition {
            text_state: Tensor::ones([2, 6, 16], &d),
            text_mask: Tensor::ones([2, 6], &d),
            aux: Some(AuxConditionState::Speaker {
                state: Tensor::ones([2, 3, 8], &d),
                mask: Tensor::ones([2, 3], &d),
            }),
        };
        let zeroed = cond.zeros_like(&d);

        assert_eq!(zeroed.text_state.dims(), [2, 6, 16]);
        assert_eq!(zeroed.text_mask.dims(), [2, 6]);

        let txt_sum: f32 = zeroed
            .text_state
            .abs()
            .sum()
            .to_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert_eq!(txt_sum, 0.0);

        let aux = zeroed.aux.unwrap();
        assert!(aux.is_speaker());
        let (s, _) = aux.state_and_mask();
        let aux_sum: f32 = s.clone().abs().sum().to_data().to_vec::<f32>().unwrap()[0];
        assert_eq!(aux_sum, 0.0);
    }

    #[test]
    fn encoded_condition_zeros_like_no_aux() {
        let d = dev();
        let cond = EncodedCondition {
            text_state: Tensor::ones([1, 4, 8], &d),
            text_mask: Tensor::ones([1, 4], &d),
            aux: None,
        };
        let zeroed = cond.zeros_like(&d);
        assert!(zeroed.aux.is_none());
    }

    #[test]
    fn both_state_selective_unconditioning_preserves_other_context() {
        let d = dev();
        let both = AuxConditionState::Both {
            speaker_state: Tensor::ones([1, 2, 4], &d) * 2.0,
            speaker_mask: Tensor::ones([1, 2], &d),
            caption_state: Tensor::ones([1, 3, 4], &d) * 3.0,
            caption_mask: Tensor::ones([1, 3], &d),
        };

        let speaker_uncond = both.speaker_unconditional(&d);
        let (speaker, speaker_mask) = speaker_uncond.speaker().unwrap();
        let (caption, caption_mask) = speaker_uncond.caption().unwrap();
        assert_eq!(speaker.clone().abs().sum().into_scalar::<f32>(), 0.0);
        assert_eq!(speaker_mask.clone().int().sum().into_scalar::<i32>(), 0);
        assert_eq!(caption.clone().min().into_scalar::<f32>(), 3.0);
        assert_eq!(caption_mask.clone().int().sum().into_scalar::<i32>(), 3);

        let caption_uncond = both.caption_unconditional(&d);
        let (speaker, speaker_mask) = caption_uncond.speaker().unwrap();
        let (caption, caption_mask) = caption_uncond.caption().unwrap();
        assert_eq!(speaker.clone().min().into_scalar::<f32>(), 2.0);
        assert_eq!(speaker_mask.clone().int().sum().into_scalar::<i32>(), 2);
        assert_eq!(caption.clone().abs().sum().into_scalar::<f32>(), 0.0);
        assert_eq!(caption_mask.clone().int().sum().into_scalar::<i32>(), 0);
    }

    #[test]
    fn cat_batch_supports_both_contexts() {
        let d = dev();
        let make = |value: f32| EncodedCondition {
            text_state: Tensor::ones([1, 2, 4], &d) * value,
            text_mask: Tensor::ones([1, 2], &d),
            aux: Some(AuxConditionState::Both {
                speaker_state: Tensor::ones([1, 3, 4], &d) * value,
                speaker_mask: Tensor::ones([1, 3], &d),
                caption_state: Tensor::ones([1, 5, 4], &d) * value,
                caption_mask: Tensor::ones([1, 5], &d),
            }),
        };
        let first = make(1.0);
        let second = make(2.0);
        let combined = EncodedCondition::cat_batch(&[&first, &second]);

        assert_eq!(combined.text_state.dims(), [2, 2, 4]);
        let aux = combined.aux.unwrap();
        assert_eq!(aux.speaker().unwrap().0.dims(), [2, 3, 4]);
        assert_eq!(aux.caption().unwrap().0.dims(), [2, 5, 4]);
    }
}
