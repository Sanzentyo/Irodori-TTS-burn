//! WGPU-specific condition-encoding entrypoint.

use burn::tensor::{Bool, Int, Tensor};

use crate::WgpuRaw;

use super::super::{
    condition::{AuxConditionInput, EncodedCondition},
    dit::TextToLatentRfDiT,
};

impl TextToLatentRfDiT<WgpuRaw> {
    pub(crate) fn encode_conditions_wgsl(
        &self,
        text_input_ids: Tensor<WgpuRaw, 2, Int>,
        text_mask: Tensor<WgpuRaw, 2, Bool>,
        aux_input: AuxConditionInput<WgpuRaw>,
    ) -> crate::error::Result<EncodedCondition<WgpuRaw>> {
        self.condition_frontend.encode_wgsl(
            text_input_ids,
            text_mask,
            aux_input,
            self.speaker_patch_size,
        )
    }
}
