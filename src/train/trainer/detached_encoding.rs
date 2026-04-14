//! Detached conditioning encoding (inner-backend optimization).

use burn::{
    module::AutodiffModule,
    tensor::{Tensor, TensorPrimitive, backend::AutodiffBackend},
};

use crate::{
    model::condition::{AuxConditionInput, AuxConditionState, EncodedCondition},
    train::LoraTextToLatentRfDiT,
};

type IB<B> = <B as AutodiffBackend>::InnerBackend;

/// Encode conditions on the inner (non-AD) backend to avoid autodiff dispatch
/// overhead for the frozen text encoder and aux conditioner.
///
/// Since these components are entirely frozen (no trainable parameters), running
/// them through the AD graph adds ~270 unnecessary dispatch calls.  This helper
/// strips the AD wrapper, runs encode on the raw backend, and wraps the results
/// back as AD leaf tensors (no gradient flows through conditioning).
///
/// # Safety invariant
///
/// This is correct **only while** `text_encoder`, `text_norm`, and
/// `aux_conditioner` remain fully frozen (no trainable parameters).
/// If any of these modules are later LoRA-wrapped or unfrozen, this
/// function will silently cut their gradients — replace with
/// `model.encode_conditions(...)` in that case.
pub(super) fn encode_conditions_detached<B: AutodiffBackend>(
    model: &LoraTextToLatentRfDiT<B>,
    text_input_ids: Tensor<B, 2, burn::tensor::Int>,
    text_mask: Tensor<B, 2, burn::tensor::Bool>,
    aux_input: AuxConditionInput<B>,
) -> crate::error::Result<EncodedCondition<B>> {
    // Float tensor conversion helpers that handle TensorPrimitive::Float wrapping.
    let to_inner_float = |t: Tensor<B, 3>| -> Tensor<IB<B>, 3> {
        let prim = match t.into_primitive() {
            TensorPrimitive::Float(p) => p,
            _ => unreachable!("expected Float primitive"),
        };
        Tensor::from_primitive(TensorPrimitive::Float(B::inner(prim)))
    };
    let from_inner_float = |t: Tensor<IB<B>, 3>| -> Tensor<B, 3> {
        let prim = match t.into_primitive() {
            TensorPrimitive::Float(p) => p,
            _ => unreachable!("expected Float primitive"),
        };
        Tensor::from_primitive(TensorPrimitive::Float(B::from_inner(prim)))
    };

    let inner_model = model.valid();

    // Int and Bool tensors are pass-through for the Autodiff backend.
    let inner_text_ids = Tensor::<IB<B>, 2, burn::tensor::Int>::from_primitive(B::int_inner(
        text_input_ids.into_primitive(),
    ));
    let inner_text_mask = Tensor::<IB<B>, 2, burn::tensor::Bool>::from_primitive(B::bool_inner(
        text_mask.into_primitive(),
    ));

    let inner_aux = match aux_input {
        AuxConditionInput::Speaker {
            ref_latent,
            ref_mask,
        } => AuxConditionInput::Speaker {
            ref_latent: to_inner_float(ref_latent),
            ref_mask: Tensor::<IB<B>, 2, burn::tensor::Bool>::from_primitive(B::bool_inner(
                ref_mask.into_primitive(),
            )),
        },
        AuxConditionInput::Caption { ids, mask } => AuxConditionInput::Caption {
            ids: Tensor::<IB<B>, 2, burn::tensor::Int>::from_primitive(B::int_inner(
                ids.into_primitive(),
            )),
            mask: Tensor::<IB<B>, 2, burn::tensor::Bool>::from_primitive(B::bool_inner(
                mask.into_primitive(),
            )),
        },
        AuxConditionInput::None => AuxConditionInput::None,
    };

    let inner_cond = inner_model.encode_conditions(inner_text_ids, inner_text_mask, inner_aux)?;

    // Wrap results back as AD leaf tensors (no gradient).
    let text_state = from_inner_float(inner_cond.text_state);
    let text_mask = Tensor::<B, 2, burn::tensor::Bool>::from_primitive(B::bool_from_inner(
        inner_cond.text_mask.into_primitive(),
    ));
    let aux = inner_cond.aux.map(|a| match a {
        AuxConditionState::Speaker { state, mask } => AuxConditionState::Speaker {
            state: from_inner_float(state),
            mask: Tensor::<B, 2, burn::tensor::Bool>::from_primitive(B::bool_from_inner(
                mask.into_primitive(),
            )),
        },
        AuxConditionState::Caption { state, mask } => AuxConditionState::Caption {
            state: from_inner_float(state),
            mask: Tensor::<B, 2, burn::tensor::Bool>::from_primitive(B::bool_from_inner(
                mask.into_primitive(),
            )),
        },
    });

    Ok(EncodedCondition {
        text_state,
        text_mask,
        aux,
    })
}
