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
