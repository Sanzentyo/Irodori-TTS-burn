//! WGPU-specific condition-encoding entrypoint.

use burn::tensor::{Bool, Int, Tensor};

use super::super::{
    condition::{AuxConditionInput, EncodedCondition},
    dit::TextToLatentRfDiT,
};

impl TextToLatentRfDiT {
    pub(crate) fn encode_conditions_wgsl(
        &self,
        text_input_ids: Tensor<2, Int>,
        text_mask: Tensor<2, Bool>,
        aux_input: AuxConditionInput,
    ) -> crate::error::Result<EncodedCondition> {
        self.condition_frontend.encode_wgsl(
            text_input_ids,
            text_mask,
            aux_input,
            self.speaker_patch_size,
        )
    }
}
