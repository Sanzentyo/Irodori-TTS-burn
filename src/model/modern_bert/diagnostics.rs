//! Explicit, synchronized observations. No tensor handles survive a record,
//! so diagnostics do not pin intermediate GPU allocations across layers.

use super::{Observer, SharedModernBertConditioner};
use crate::error::{IrodoriError, Result};
use crate::kernels::modern_bert_residual_layer_norm::ReductionOrder;
use burn::tensor::{Bool, Int, Tensor};

/// A CPU copy of one text-encoder boundary, in logical tensor order.
#[derive(Debug)]
pub struct TextEncoderTraceTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

/// Diagnostic input and layer values; this intentionally synchronizes each
/// boundary and must not be used for performance measurement.
#[derive(Debug)]
pub struct TextEncoderTrace {
    pub input_shape: [usize; 2],
    pub input_ids: Vec<i32>,
    pub attention_mask: Vec<bool>,
    pub tensors: Vec<TextEncoderTraceTensor>,
}

pub(super) struct Recorder {
    pub(super) tensors: Vec<TextEncoderTraceTensor>,
    error: Option<IrodoriError>,
}

impl Observer for Recorder {
    fn record<const D: usize>(
        &mut self,
        layer: Option<usize>,
        stage: &'static str,
        tensor: &Tensor<D>,
    ) {
        if self.error.is_some() {
            return;
        }
        let name = layer.map_or_else(|| stage.into(), |layer| format!("layer_{layer:02}_{stage}"));
        match tensor
            .clone()
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
        {
            Ok(values) => self.tensors.push(TextEncoderTraceTensor {
                name,
                shape: tensor.dims().to_vec(),
                values,
            }),
            Err(error) => self.error = Some(IrodoriError::Profile(format!("{name}: {error:?}"))),
        }
    }
}

impl SharedModernBertConditioner {
    pub(crate) fn trace_text_wgsl(
        &self,
        ids: Tensor<2, Int>,
        mask: Tensor<2, Bool>,
        norm: &super::RmsNorm,
        reduction_order: ReductionOrder,
    ) -> Result<TextEncoderTrace> {
        let input_shape = ids.dims();
        let input_ids = ids
            .clone()
            .into_data()
            .try_to_vec::<i32>()
            .map_err(|error| IrodoriError::Profile(format!("text IDs: {error:?}")))?;
        let attention_mask = mask
            .clone()
            .into_data()
            .convert::<bool>()
            .try_to_vec::<bool>()
            .map_err(|error| IrodoriError::Profile(format!("text mask: {error:?}")))?;
        let mut recorder = Recorder {
            tensors: Vec::new(),
            error: None,
        };
        let model = &self.pretrained_text_backbone.backbone;
        let backbone = if model.supports_forward_wgsl(&ids, &mask) {
            model
                .try_forward_wgsl_observed(ids, mask.clone(), reduction_order, &mut recorder)
                .ok_or_else(|| {
                    IrodoriError::Profile("text encoder's selected WGSL route failed".into())
                })?
        } else {
            model.forward_observed(ids, mask.clone(), &mut recorder)
        };
        let projected = self.text_encoder.forward(backbone, mask);
        recorder.record(None, "projected", &projected);
        let text_state = norm.forward(projected);
        recorder.record(None, "text_state", &text_state);
        if let Some(error) = recorder.error {
            return Err(error);
        }
        Ok(TextEncoderTrace {
            input_shape,
            input_ids,
            attention_mask,
            tensors: recorder.tensors,
        })
    }
}
