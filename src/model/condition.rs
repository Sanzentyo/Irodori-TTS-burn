//! [`EncodedCondition`] — the runtime bundle of all conditioning tensors.

use burn::tensor::{Bool, Int, Tensor, backend::Backend};

// ---------------------------------------------------------------------------
// AuxConditionState — runtime tensor bundle for aux conditioning
// ---------------------------------------------------------------------------

/// Runtime tensor bundle for auxiliary conditioning (speaker or caption).
///
/// Only one variant is active at a time.  Use [`Option<AuxConditionState>`] to
/// represent the "no aux conditioning" case.
pub enum AuxConditionState<B: Backend> {
    /// Speaker (reference audio) conditioning is active.
    Speaker {
        state: Tensor<B, 3>,
        mask: Tensor<B, 2, Bool>,
    },
    /// Caption conditioning is active.
    Caption {
        state: Tensor<B, 3>,
        mask: Tensor<B, 2, Bool>,
    },
}

impl<B: Backend> AuxConditionState<B> {
    /// Whether this is the Speaker variant.
    pub fn is_speaker(&self) -> bool {
        matches!(self, Self::Speaker { .. })
    }

    /// Whether this is the Caption variant.
    pub fn is_caption(&self) -> bool {
        matches!(self, Self::Caption { .. })
    }

    /// Returns `(&state, &mask)` regardless of variant.
    pub fn state_and_mask(&self) -> (&Tensor<B, 3>, &Tensor<B, 2, Bool>) {
        match self {
            Self::Speaker { state, mask } | Self::Caption { state, mask } => (state, mask),
        }
    }

    /// Produce an all-zero version with the **same variant**.
    ///
    /// Preserving the variant is critical so CFG can nullify the correct signal
    /// without collapsing to "no aux conditioning at all".
    pub fn zeros_like(&self, device: &B::Device) -> Self {
        match self {
            Self::Speaker { state, mask } => Self::Speaker {
                state: Tensor::zeros(state.dims(), device),
                mask: Tensor::<B, 2>::zeros(mask.dims(), device).greater_elem(0.0),
            },
            Self::Caption { state, mask } => Self::Caption {
                state: Tensor::zeros(state.dims(), device),
                mask: Tensor::<B, 2>::zeros(mask.dims(), device).greater_elem(0.0),
            },
        }
    }
}

impl<B: Backend> Clone for AuxConditionState<B> {
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
        }
    }
}

// ---------------------------------------------------------------------------
// AuxConditionInput — typed input for aux encoder dispatch
// ---------------------------------------------------------------------------

/// Typed input bundle passed to `AuxConditioner::encode`.
///
/// Prevents mixing speaker and caption inputs at the call site, and makes
/// the "no aux conditioning requested" case explicit.
pub enum AuxConditionInput<B: Backend> {
    /// Reference audio latent + mask for speaker conditioning.
    Speaker {
        ref_latent: Tensor<B, 3>,
        ref_mask: Tensor<B, 2, Bool>,
    },
    /// Token IDs + mask for caption conditioning.
    Caption {
        ids: Tensor<B, 2, Int>,
        mask: Tensor<B, 2, Bool>,
    },
    /// No auxiliary input supplied.
    None,
}

impl<B: Backend> AuxConditionInput<B> {
    /// Construct from raw optional fields (e.g., from `SamplingRequest`).
    ///
    /// Speaker takes precedence if both speaker and caption inputs are present.
    pub fn from_request(
        ref_latent: Option<Tensor<B, 3>>,
        ref_mask: Option<Tensor<B, 2, Bool>>,
        caption_ids: Option<Tensor<B, 2, Int>>,
        caption_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Self {
        match (ref_latent, ref_mask) {
            (Some(ref_latent), Some(ref_mask)) => Self::Speaker {
                ref_latent,
                ref_mask,
            },
            _ => match (caption_ids, caption_mask) {
                (Some(ids), Some(mask)) => Self::Caption { ids, mask },
                _ => Self::None,
            },
        }
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
pub struct EncodedCondition<B: Backend> {
    pub text_state: Tensor<B, 3>,
    pub text_mask: Tensor<B, 2, Bool>,
    /// Speaker or caption encoded state; `None` when not used by this model.
    pub aux: Option<AuxConditionState<B>>,
}

impl<B: Backend> EncodedCondition<B> {
    /// Create an all-zero unconditional version of this condition.
    ///
    /// State tensors are zeroed; Bool masks are all-False.
    /// The `aux` variant is preserved so CFG can still nullify the correct signal.
    pub fn zeros_like(&self, device: &B::Device) -> Self {
        let zero_text = Tensor::zeros(self.text_state.dims(), device);
        let zero_text_mask: Tensor<B, 2, Bool> =
            Tensor::<B, 2>::zeros(self.text_mask.dims(), device).greater_elem(0.0);

        Self {
            text_state: zero_text,
            text_mask: zero_text_mask,
            aux: self.aux.as_ref().map(|a| a.zeros_like(device)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn dev() -> <B as Backend>::Device {
        Default::default()
    }

    // --- AuxConditionState ---

    #[test]
    fn aux_state_speaker_variant_identification() {
        let d = dev();
        let state = AuxConditionState::<B>::Speaker {
            state: Tensor::zeros([1, 4, 8], &d),
            mask: Tensor::ones([1, 4], &d),
        };
        assert!(state.is_speaker());
        assert!(!state.is_caption());
    }

    #[test]
    fn aux_state_caption_variant_identification() {
        let d = dev();
        let state = AuxConditionState::<B>::Caption {
            state: Tensor::zeros([1, 4, 8], &d),
            mask: Tensor::ones([1, 4], &d),
        };
        assert!(state.is_caption());
        assert!(!state.is_speaker());
    }

    #[test]
    fn aux_state_state_and_mask_shapes() {
        let d = dev();
        let state = AuxConditionState::<B>::Speaker {
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
        let original = AuxConditionState::<B>::Speaker {
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

        let mask_sum: i64 = m.clone().int().sum().to_data().to_vec::<i64>().unwrap()[0];
        assert_eq!(mask_sum, 0);
    }

    #[test]
    fn aux_state_clone_preserves_values() {
        let d = dev();
        let original = AuxConditionState::<B>::Caption {
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
    fn input_from_request_speaker_priority() {
        let d = dev();
        let lat = Some(Tensor::<B, 3>::zeros([1, 2, 8], &d));
        let mask = Some(Tensor::<B, 2, Bool>::ones([1, 2], &d));
        let cap_ids = Some(Tensor::<B, 2, Int>::zeros([1, 4], &d));
        let cap_mask = Some(Tensor::<B, 2, Bool>::ones([1, 4], &d));

        let input = AuxConditionInput::from_request(lat, mask, cap_ids, cap_mask);
        assert!(matches!(input, AuxConditionInput::Speaker { .. }));
    }

    #[test]
    fn input_from_request_caption_fallback() {
        let d = dev();
        let cap_ids = Some(Tensor::<B, 2, Int>::zeros([1, 4], &d));
        let cap_mask = Some(Tensor::<B, 2, Bool>::ones([1, 4], &d));

        let input = AuxConditionInput::from_request(None, None, cap_ids, cap_mask);
        assert!(matches!(input, AuxConditionInput::Caption { .. }));
    }

    #[test]
    fn input_from_request_none() {
        let input = AuxConditionInput::<B>::from_request(None, None, None, None);
        assert!(matches!(input, AuxConditionInput::None));
    }

    // --- EncodedCondition ---

    #[test]
    fn encoded_condition_zeros_like_shapes_and_values() {
        let d = dev();
        let cond = EncodedCondition::<B> {
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
        let cond = EncodedCondition::<B> {
            text_state: Tensor::ones([1, 4, 8], &d),
            text_mask: Tensor::ones([1, 4], &d),
            aux: None,
        };
        let zeroed = cond.zeros_like(&d);
        assert!(zeroed.aux.is_none());
    }
}
