//! WGPU-only routing for the pretrained condition frontend.

use crate::kernels::modern_bert_residual_layer_norm::ReductionOrder;
use burn::tensor::{Bool, Int, Tensor};

use super::{AuxConditionInput, ConditionFrontend, EncodedCondition};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WgslConditionRoute {
    Generic,
    Pretrained(ReductionOrder),
}

const fn select_route(is_pretrained: bool, aux_is_empty: bool) -> WgslConditionRoute {
    if is_pretrained {
        WgslConditionRoute::Pretrained(if aux_is_empty {
            ReductionOrder::LegacyShortText
        } else {
            ReductionOrder::Backend
        })
    } else {
        WgslConditionRoute::Generic
    }
}

impl ConditionFrontend {
    /// Route the pretrained text encoder through ModernBERT WGSL.
    ///
    /// Scratch frontends and auxiliary encoding retain their existing graph.
    /// The ModernBERT method performs topology/dtype/layout/resource selection and falls back to
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
        if let WgslConditionRoute::Pretrained(reduction_order) = route
            && let Self::Pretrained(frontend) = self
        {
            let text_state = frontend.text_norm.forward(frontend.shared.encode_text_wgsl(
                text_input_ids,
                text_mask.clone(),
                reduction_order,
            ));
            return Ok(EncodedCondition {
                text_state,
                text_mask,
                aux: frontend.encode_aux(aux_input, speaker_patch_size)?,
            });
        }

        self.encode(text_input_ids, text_mask, aux_input, speaker_patch_size)
    }

    #[cfg(all(feature = "profile", feature = "inference"))]
    pub(crate) fn trace_text_encoder_wgsl(
        &self,
        ids: Tensor<2, Int>,
        mask: Tensor<2, Bool>,
        aux: AuxConditionInput,
    ) -> crate::error::Result<crate::model::modern_bert::TextEncoderTrace> {
        let route = select_route(
            matches!(self, Self::Pretrained(_)),
            matches!(&aux, AuxConditionInput::None),
        );
        if let WgslConditionRoute::Pretrained(reduction_order) = route
            && let Self::Pretrained(frontend) = self
        {
            return frontend.shared.trace_text_wgsl(
                ids,
                mask,
                &frontend.text_norm,
                reduction_order,
            );
        }
        Err(crate::error::IrodoriError::UnsupportedMode(
            "layer tracing requires the ModernBERT frontend".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_only_admits_pretrained_text() {
        assert_eq!(
            select_route(true, true),
            WgslConditionRoute::Pretrained(ReductionOrder::LegacyShortText)
        );
        assert_eq!(
            select_route(true, false),
            WgslConditionRoute::Pretrained(ReductionOrder::Backend)
        );
        assert_eq!(select_route(false, true), WgslConditionRoute::Generic);
        assert_eq!(select_route(false, false), WgslConditionRoute::Generic);
    }
}
