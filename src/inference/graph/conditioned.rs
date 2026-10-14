//! Reuse denoising dispatches while refreshing encoded text/voice values.
//! Geometry and masks define the captured control flow and remain immutable.

use super::*;
use crate::{AuxConditionState, EncodedCondition, graph::copy_into_stable_input};
use burn::tensor::{Bool, Tensor};

type ConditionPart<'a> = (&'a Tensor<3>, &'a Tensor<2, Bool>);

fn parts(condition: &EncodedCondition) -> [Option<ConditionPart<'_>>; 3] {
    [
        Some((&condition.text_state, &condition.text_mask)),
        condition.aux.as_ref().and_then(AuxConditionState::speaker),
        condition.aux.as_ref().and_then(AuxConditionState::caption),
    ]
}

fn mask_values(mask: &Tensor<2, Bool>) -> crate::Result<Vec<bool>> {
    mask.clone()
        .into_data()
        .convert::<bool>()
        .try_to_vec::<bool>()
        .map_err(|error| IrodoriError::Shape(format!("graph condition mask: {error}")))
}

fn persistent_copy(source: &Tensor<3>) -> crate::Result<Tensor<3>> {
    let client = source
        .clone()
        .try_into_primitive::<crate::WgpuRaw>()
        .map_err(|_| IrodoriError::UnsupportedMode("condition graph requires WGPU".into()))?
        .device
        .client();
    let destination = client.memory_persistent_allocation((), |()| {
        Tensor::zeros(source.dims(), (&source.device(), source.dtype()))
    });
    copy_into_stable_input(source, &destination)?;
    Ok(destination)
}

#[derive(Debug)]
struct StableCondition {
    value: EncodedCondition,
    masks: [Option<Vec<bool>>; 3],
}

impl StableCondition {
    fn new(source: EncodedCondition) -> crate::Result<Self> {
        let mut masks = [None, None, None];
        for (index, part) in parts(&source).into_iter().enumerate() {
            if let Some((_, mask)) = part {
                masks[index] = Some(mask_values(mask)?);
            }
        }
        let aux = match source.aux {
            None => None,
            Some(AuxConditionState::Speaker { state, mask }) => Some(AuxConditionState::Speaker {
                state: persistent_copy(&state)?,
                mask,
            }),
            Some(AuxConditionState::Caption { state, mask }) => Some(AuxConditionState::Caption {
                state: persistent_copy(&state)?,
                mask,
            }),
            Some(AuxConditionState::Both {
                speaker_state,
                speaker_mask,
                caption_state,
                caption_mask,
            }) => Some(AuxConditionState::Both {
                speaker_state: persistent_copy(&speaker_state)?,
                speaker_mask,
                caption_state: persistent_copy(&caption_state)?,
                caption_mask,
            }),
        };
        Ok(Self {
            value: EncodedCondition {
                text_state: persistent_copy(&source.text_state)?,
                text_mask: source.text_mask,
                aux,
            },
            masks,
        })
    }

    fn validate(&self, source: &EncodedCondition) -> crate::Result<()> {
        for (index, (actual, expected)) in parts(source)
            .into_iter()
            .zip(parts(&self.value))
            .enumerate()
        {
            match (actual, expected) {
                (None, None) => {}
                (Some((state, mask)), Some((stable, stable_mask))) => {
                    if state.dims() != stable.dims()
                        || mask.dims() != stable_mask.dims()
                        || state.dtype() != stable.dtype()
                        || state.device() != stable.device()
                        || mask.device() != stable_mask.device()
                    {
                        return Err(IrodoriError::Shape(format!(
                            "graph condition part {index} changed shape, dtype or device"
                        )));
                    }
                    let raw = state
                        .clone()
                        .try_into_primitive::<crate::WgpuRaw>()
                        .map_err(|_| {
                            IrodoriError::UnsupportedMode("condition graph requires WGPU".into())
                        })?;
                    if !raw.is_contiguous() {
                        return Err(IrodoriError::Shape(format!(
                            "graph condition part {index} must be contiguous"
                        )));
                    }
                    if Some(mask_values(mask)?) != self.masks[index] {
                        return Err(IrodoriError::Config(format!(
                            "graph condition part {index} changed its attention mask"
                        )));
                    }
                }
                _ => {
                    return Err(IrodoriError::Config(
                        "graph condition changed its auxiliary variant".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn refresh(&self, source: &EncodedCondition) -> crate::Result<()> {
        for (source, destination) in parts(source)
            .into_iter()
            .flatten()
            .zip(parts(&self.value).into_iter().flatten())
        {
            copy_into_stable_input(source.0, destination.0)?;
        }
        Ok(())
    }
}

/// A fixed-geometry sampler graph with mutable encoded conditioning values.
/// Text, speaker and caption encoders run outside the captured trajectory.
/// Every replay checks exact masks, shapes, precision and device before copying
/// values; changed topology requires a new capture. No implicit recapture occurs.
#[derive(Debug)]
pub struct CapturedConditionedSampling<'engine> {
    // Graph is released before stable inputs and model bindings.
    graph: CapturedTensorGraph,
    condition: StableCondition,
    _request: PreparedSamplingRequest,
    _engine: PhantomData<&'engine WgslInferenceEngine>,
}

impl CapturedConditionedSampling<'_> {
    pub fn input_dims(&self) -> [usize; 3] {
        self.graph.input_dims
    }

    pub fn enqueue(
        &mut self,
        noise: Tensor<3>,
        condition: &EncodedCondition,
    ) -> crate::Result<CapturedSamplingOutput<'_>> {
        self.condition.validate(condition)?;
        self.graph
            .enqueue_with(noise, || self.condition.refresh(condition))
    }

    pub fn sample_to_cpu_f32(
        &mut self,
        noise: Tensor<3>,
        condition: &EncodedCondition,
    ) -> crate::Result<Vec<f32>> {
        self.enqueue(noise, condition)?.to_cpu_f32()
    }
}

impl WgslInferenceEngine {
    /// Capture a trajectory whose encoded conditions can be replaced between
    /// requests with identical geometry and masks. Parameters and weights stay
    /// borrowed, and all speaker/caption KV projections remain inside capture.
    pub fn capture_conditioned_sampling(
        &self,
        request: SamplingRequest,
    ) -> crate::Result<CapturedConditionedSampling<'_>> {
        let (mut request, example) = self.prepare_capture(request)?;
        let condition = StableCondition::new(
            request
                .encoded_condition
                .take()
                .ok_or_else(|| IrodoriError::MissingInput("prepared capture condition".into()))?,
        )?;
        request.encoded_condition = Some(condition.value.clone());
        let graph = capture_tensor_graph(&example, |noise| {
            let mut input = request.clone();
            input.request.initial_noise = Some(noise);
            self.sample_prepared(input)
        })?;
        Ok(CapturedConditionedSampling {
            graph,
            condition,
            _request: request,
            _engine: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_environment::stream::StreamId;

    #[test]
    fn refreshes_all_conditions_and_rejects_mask_changes_before_replay() {
        let device =
            crate::backend_config::strict_fp32_device(&burn::backend::wgpu::WgpuDevice::default())
                .unwrap();
        let condition = |value: f32| EncodedCondition {
            text_state: Tensor::full([1, 2, 8], value, &device),
            text_mask: Tensor::<2>::ones([1, 2], &device).greater_elem(0.0),
            aux: Some(AuxConditionState::Both {
                speaker_state: Tensor::full([1, 2, 8], value * 2.0, &device),
                speaker_mask: Tensor::<2>::ones([1, 2], &device).greater_elem(0.0),
                caption_state: Tensor::full([1, 2, 8], value * 3.0, &device),
                caption_mask: Tensor::<2>::zeros([1, 2], &device).greater_elem(0.0),
            }),
        };
        let stable = StableCondition::new(condition(1.0)).unwrap();
        let noise = Tensor::full([1, 2, 8], 7.0, &device);
        let mut graph = StreamId::allocate()
            .executes(|| {
                capture_tensor_graph(&noise, |input| {
                    Ok(parts(&stable.value)
                        .into_iter()
                        .flatten()
                        .fold(input, |sum, (state, _)| sum + state.clone()))
                })
            })
            .unwrap();
        for value in [1.0, 5.0, -2.0, 1.0] {
            let fresh = StreamId::allocate().executes(|| condition(value));
            stable.validate(&fresh).unwrap();
            let result = StreamId::allocate()
                .executes(|| {
                    graph
                        .enqueue_with(noise.clone(), || stable.refresh(&fresh))
                        .unwrap()
                        .to_cpu_f32()
                })
                .unwrap();
            assert_eq!(result, vec![7.0 + value * 6.0; 16]);
        }
        let mut bad = condition(12.0);
        bad.text_mask = bad.text_mask.bool_not();
        assert!(matches!(
            stable.validate(&bad),
            Err(IrodoriError::Config(_))
        ));
        let mut missing = condition(12.0);
        missing.aux = None;
        assert!(matches!(
            stable.validate(&missing),
            Err(IrodoriError::Config(_))
        ));
        // Rejection did not partially change the graph's stable condition.
        assert_eq!(
            graph.enqueue(noise).unwrap().to_cpu_f32().unwrap(),
            vec![13.0; 16]
        );
    }
}
