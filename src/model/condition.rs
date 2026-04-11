//! [`EncodedCondition`] — the runtime bundle of all conditioning tensors.

use burn::tensor::{Bool, Tensor, backend::Backend};

/// All encoded conditioning tensors for one forward pass.
///
/// Speaker or caption will be `None` if the architecture doesn't use that mode.
/// For CFG-unconditional passes, these tensors are present but their masks
/// are all-False (zero-state semantics — never `None`).
pub struct EncodedCondition<B: Backend> {
    pub text_state: Tensor<B, 3>,
    pub text_mask: Tensor<B, 2, Bool>,
    /// Present when `use_speaker_condition = true` in model config.
    pub speaker_state: Option<Tensor<B, 3>>,
    pub speaker_mask: Option<Tensor<B, 2, Bool>>,
    /// Present when `use_caption_condition = true` in model config.
    pub caption_state: Option<Tensor<B, 3>>,
    pub caption_mask: Option<Tensor<B, 2, Bool>>,
}

impl<B: Backend> EncodedCondition<B> {
    /// Create an all-zero unconditional version of this condition.
    ///
    /// State tensors are zeroed; Bool masks are all-False.
    /// This is the CFG "null" condition (never `None`).
    pub fn zeros_like(&self, device: &B::Device) -> Self {
        let text_shape = self.text_state.dims();
        let mask_shape = self.text_mask.dims();

        let zero_text = Tensor::zeros(text_shape, device);
        let zero_text_mask: Tensor<B, 2, Bool> =
            Tensor::<B, 2>::zeros(mask_shape, device).greater_elem(0.0);

        let (speaker_state, speaker_mask) = match &self.speaker_state {
            Some(s) => {
                let sp_shape = s.dims();
                let sp_m_shape = self.speaker_mask.as_ref().unwrap().dims();
                let zs = Tensor::zeros(sp_shape, device);
                let zm: Tensor<B, 2, Bool> =
                    Tensor::<B, 2>::zeros(sp_m_shape, device).greater_elem(0.0);
                (Some(zs), Some(zm))
            }
            None => (None, None),
        };

        let (caption_state, caption_mask) = match &self.caption_state {
            Some(s) => {
                let cap_shape = s.dims();
                let cap_m_shape = self.caption_mask.as_ref().unwrap().dims();
                let zs = Tensor::zeros(cap_shape, device);
                let zm: Tensor<B, 2, Bool> =
                    Tensor::<B, 2>::zeros(cap_m_shape, device).greater_elem(0.0);
                (Some(zs), Some(zm))
            }
            None => (None, None),
        };

        Self {
            text_state: zero_text,
            text_mask: zero_text_mask,
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
        }
    }
}
