//! WGPU-only routing for the pretrained condition frontend.

use burn::tensor::{Bool, Int, Tensor};

use super::{AuxConditionInput, ConditionFrontend, EncodedCondition};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WgslConditionRoute {
    Generic,
    PretrainedTextOnly,
}

const fn select_route(is_pretrained: bool, aux_is_empty: bool) -> WgslConditionRoute {
    if is_pretrained && aux_is_empty {
        WgslConditionRoute::PretrainedTextOnly
    } else {
        WgslConditionRoute::Generic
    }
}

impl ConditionFrontend {
    /// Route only pretrained text-only inference through ModernBERT WGSL.
    ///
    /// Scratch frontends and every nonempty speaker/caption input retain the
    /// existing generic frontend. The ModernBERT method performs the exact
    /// B1/S3/dtype/layout/device/resource selection and itself falls back to
    /// the generic backbone if any part of that contract is unsupported.
    pub(crate) fn encode_wgsl(
        &self,
        text_input_ids: Tensor<2, Int>,
        text_mask: Tensor<2, Bool>,
        aux_input: AuxConditionInput,
        speaker_patch_size: usize,
    ) -> crate::error::Result<EncodedCondition> {
        let route = select_route(
            matches!(self, Self::Pretrained(_)),
            matches!(&aux_input, AuxConditionInput::None),
        );
        if route == WgslConditionRoute::PretrainedTextOnly
            && let Self::Pretrained(frontend) = self
        {
            let text_state = frontend.text_norm.forward(
                frontend
                    .shared
                    .encode_text_wgsl(text_input_ids, text_mask.clone()),
            );
            return Ok(EncodedCondition {
                text_state,
                text_mask,
                aux: None,
            });
        }

        self.encode(text_input_ids, text_mask, aux_input, speaker_patch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_only_admits_pretrained_text_without_aux() {
        assert_eq!(
            select_route(true, true),
            WgslConditionRoute::PretrainedTextOnly
        );
        assert_eq!(select_route(false, true), WgslConditionRoute::Generic);
        assert_eq!(select_route(true, false), WgslConditionRoute::Generic);
        assert_eq!(select_route(false, false), WgslConditionRoute::Generic);
    }
}
