//! Inference-only cache for a prepared v4 Euler timestep schedule.
//!
//! The cache is owned by [`crate::inference::WgslInferenceEngine`], not by a
//! Burn module, so it is deliberately absent from model records.  It is built
//! only for the measured raw WGPU F32/F16 policies and rejects every schedule,
//! sampler configuration, model generation, shape, layout, or device outside
//! its exact contract. F32 and F16 use distinct typed cache entries.

use burn::tensor::Device;
use std::{
    num::NonZeroU64,
    sync::atomic::{AtomicU64, Ordering},
};

use burn::tensor::{DType, Tensor};

use crate::{
    config::{CfgGuidanceMode, SamplerMethod},
    rf::SamplerParams,
};

use super::{
    adaln_cross_layer::{
        CrossLayerAdaLnBatchSchedule, CrossLayerAdaLnCache, CrossLayerAdaLnModulations,
    },
    dit::TextToLatentRfDiT,
    rope::get_timestep_embedding,
};

pub(crate) const MAX_PREPARED_EULER_STEPS: usize = 40;
pub(crate) const V4_TIMESTEP_EMBED_DIM: usize = 512;
pub(crate) const V4_COND_MODEL_DIM: usize = 1_280;
pub(crate) const V4_COND_WIDTH: usize = V4_COND_MODEL_DIM * 3;
const V4_BLOCKS: usize = 12;
static NEXT_MODEL_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Opaque identity assigned after a model has loaded and before any
/// inference-only cache is constructed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ModelGeneration(NonZeroU64);

impl ModelGeneration {
    pub(crate) fn fresh() -> Self {
        let value = NEXT_MODEL_GENERATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .expect("model generation counter exhausted");
        Self(NonZeroU64::new(value).expect("model generation starts at one"))
    }
}

/// Return whether engine construction may prepare the exact Euler schedule.
/// Request-dependent B1/B2/B3 topology checks are repeated inside the sampler
/// after the active conditioning signals have been resolved.
pub(crate) fn supports_prepared_euler_params(params: &SamplerParams) -> bool {
    (1..=MAX_PREPARED_EULER_STEPS).contains(&params.num_steps)
        && matches!(params.method, SamplerMethod::Euler)
        && matches!(params.guidance.mode, CfgGuidanceMode::Independent)
        && params.guidance.min_t.is_finite()
        && params.guidance.max_t.is_finite()
        && params.guidance.min_t <= params.guidance.max_t
}

/// A contiguous CFG-active interval materialized for one batch topology.
struct PreparedEulerBatchInterval {
    first_step: usize,
    step_count: usize,
    batch: usize,
    cond_embed: Tensor<4>,
    adaln: Option<CrossLayerAdaLnBatchSchedule>,
}

/// Exact `[B, 1, 3D]` timestep conditions for a prepared Euler schedule.
///
/// All unique timestep rows and all AdaLN projections are evaluated together
/// during session construction. B2/B3 copies are retained only for the
/// contiguous CFG-active interval, not for the entire trajectory.
pub(crate) struct PreparedEulerCondCache {
    generation: ModelGeneration,
    schedule_bits: Vec<u32>,
    guidance_window_bits: [u32; 2],
    b1_cond_embed: Tensor<3>,
    b1_adaln: Option<CrossLayerAdaLnBatchSchedule>,
    b2: Option<PreparedEulerBatchInterval>,
    b3: Option<PreparedEulerBatchInterval>,
    dtype: DType,
    device: Device,
}

/// Engine-owned prepared condition and its optional cross-layer AdaLN result.
#[derive(Clone, Debug)]
pub(crate) struct PreparedEulerCondition {
    pub(crate) cond_embed: Tensor<3>,
    pub(crate) adaln: Option<CrossLayerAdaLnModulations>,
}

impl PreparedEulerBatchInterval {
    fn step(&self, index: usize) -> Option<PreparedEulerCondition> {
        let local = index.checked_sub(self.first_step)?;
        if local >= self.step_count
            || self.cond_embed.dims() != [self.step_count, self.batch, 1, V4_COND_WIDTH]
        {
            return None;
        }
        Some(PreparedEulerCondition {
            cond_embed: self.cond_embed.clone().narrow(0, local, 1).reshape([
                self.batch,
                1,
                V4_COND_WIDTH,
            ]),
            adaln: self.adaln.as_ref().and_then(|value| value.step(local)),
        })
    }
}

impl PreparedEulerCondCache {
    /// Build all unique rows together, then materialize the active B2/B3
    /// intervals. Any partial or mismatching result rejects the cache atomically.
    pub(crate) fn try_build(
        model: &TextToLatentRfDiT,
        adaln_cache: Option<&CrossLayerAdaLnCache>,
        generation: ModelGeneration,
        params: &SamplerParams,
        include_adaln: bool,
        device: &Device,
    ) -> Option<Self> {
        if !supports_prepared_euler_params(params) || !model_has_v4_contract(model, device) {
            return None;
        }
        let dtype = model.cond_module.linear0.weight.dtype();
        let schedule = reference_linear_schedule(params.num_steps);
        let evaluated = &schedule[..params.num_steps];
        let b1_cond_embed = prepare_condition_schedule(model, evaluated, dtype, device)?;
        if !wgpu_tensor_has_layout(
            &b1_cond_embed,
            [params.num_steps, 1, V4_COND_WIDTH],
            dtype,
            device,
        ) {
            return None;
        }

        let active = evaluated
            .iter()
            .enumerate()
            .filter(|(_, timestep)| {
                params.guidance.min_t <= **timestep && **timestep <= params.guidance.max_t
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let active_range = match (active.first(), active.last()) {
            (Some(first), Some(last)) if active.len() == last - first + 1 => {
                Some((*first, last - first + 1))
            }
            (None, None) => None,
            _ => return None,
        };
        let materialize = |batch| {
            let (first_step, step_count) = active_range?;
            let cond_embed = b1_cond_embed
                .clone()
                .narrow(0, first_step, step_count)
                .unsqueeze_dim::<4>(1)
                .repeat_dim(1, batch);
            let cond_embed = Tensor::<4>::from_primitive::<crate::WgpuRaw>(
                burn::backend::wgpu::into_contiguous(
                    cond_embed
                        .try_into_primitive::<crate::WgpuRaw>()
                        .expect("prepared condition must use WGPU raw backend"),
                ),
            );
            Some(PreparedEulerBatchInterval {
                first_step,
                step_count,
                batch,
                cond_embed,
                adaln: None,
            })
        };
        let mut b2 = materialize(2);
        let mut b3 = materialize(3);
        let all_adaln = (if include_adaln { adaln_cache } else { None }).and_then(|adaln_cache| {
            let b1 = (0..params.num_steps)
                .map(|index| {
                    adaln_cache.precompute_v4_wgsl(b1_cond_embed.clone().narrow(0, index, 1))
                })
                .collect::<Option<Vec<_>>>()?;
            let interval = |value: &PreparedEulerBatchInterval| {
                (0..value.step_count)
                    .map(|local| {
                        adaln_cache.precompute_v4_wgsl(
                            value.cond_embed.clone().narrow(0, local, 1).reshape([
                                value.batch,
                                1,
                                V4_COND_WIDTH,
                            ]),
                        )
                    })
                    .collect::<Option<Vec<_>>>()
            };
            let b2 = match &b2 {
                Some(value) => Some(interval(value)?),
                None => None,
            };
            let b3 = match &b3 {
                Some(value) => Some(interval(value)?),
                None => None,
            };
            let b2 = match b2 {
                Some(values) => Some(CrossLayerAdaLnModulations::pack_schedule(values)?),
                None => None,
            };
            let b3 = match b3 {
                Some(values) => Some(CrossLayerAdaLnModulations::pack_schedule(values)?),
                None => None,
            };
            Some((CrossLayerAdaLnModulations::pack_schedule(b1)?, b2, b3))
        });
        let (b1_adaln, b2_adaln, b3_adaln) = all_adaln
            .map(|(b1, b2, b3)| (Some(b1), b2, b3))
            .unwrap_or((None, None, None));
        if let Some(value) = &mut b2 {
            value.adaln = b2_adaln;
        }
        if let Some(value) = &mut b3 {
            value.adaln = b3_adaln;
        }
        let cache = Self {
            generation,
            schedule_bits: evaluated.iter().copied().map(f32::to_bits).collect(),
            guidance_window_bits: [
                params.guidance.min_t.to_bits(),
                params.guidance.max_t.to_bits(),
            ],
            b1_cond_embed,
            b1_adaln,
            b2,
            b3,
            dtype,
            device: device.clone(),
        };
        cache.has_storage_contract().then_some(cache)
    }

    /// Return a step only when every semantic and physical cache key matches.
    pub(crate) fn step(
        &self,
        generation: ModelGeneration,
        index: usize,
        timestep_bits: u32,
        batch: usize,
        device: &Device,
    ) -> Option<PreparedEulerCondition> {
        if self.generation != generation
            || &self.device != device
            || self.schedule_bits.get(index).copied()? != timestep_bits
            || !self.has_storage_contract()
        {
            return None;
        }
        match batch {
            1 => Some(PreparedEulerCondition {
                cond_embed: self.b1_cond_embed.clone().narrow(0, index, 1),
                adaln: self.b1_adaln.as_ref().and_then(|value| value.step(index)),
            }),
            2 => self.b2.as_ref()?.step(index),
            3 => self.b3.as_ref()?.step(index),
            _ => None,
        }
    }

    pub(crate) fn matches_model(
        &self,
        generation: ModelGeneration,
        params: &SamplerParams,
        device: &Device,
    ) -> bool {
        self.generation == generation
            && &self.device == device
            && supports_prepared_euler_params(params)
            && self.schedule_bits.len() == params.num_steps
            && self.schedule_bits
                == reference_linear_schedule(params.num_steps)[..params.num_steps]
                    .iter()
                    .copied()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>()
            && self.guidance_window_bits
                == [
                    params.guidance.min_t.to_bits(),
                    params.guidance.max_t.to_bits(),
                ]
            && self.has_storage_contract()
    }

    fn has_storage_contract(&self) -> bool {
        !self.schedule_bits.is_empty()
            && self
                .b1_adaln
                .as_ref()
                .is_none_or(|value| value.step_count() == self.schedule_bits.len())
            && wgpu_tensor_has_layout(
                &self.b1_cond_embed,
                [self.schedule_bits.len(), 1, V4_COND_WIDTH],
                self.dtype,
                &self.device,
            )
            && [self.b2.as_ref(), self.b3.as_ref()]
                .into_iter()
                .flatten()
                .all(|interval| {
                    interval.adaln.is_some() == self.b1_adaln.is_some()
                        && interval
                            .adaln
                            .as_ref()
                            .is_none_or(|value| value.step_count() == interval.step_count)
                        && wgpu_tensor_has_layout(
                            &interval.cond_embed,
                            [interval.step_count, interval.batch, 1, V4_COND_WIDTH],
                            self.dtype,
                            &self.device,
                        )
                })
    }
}

/// Match the official PyTorch CUDA `linspace(0, 1, steps + 1)` rounding before
/// applying Irodori's `(1 - u) * 0.999` transform.
pub(crate) fn reference_linear_schedule(num_steps: usize) -> Vec<f32> {
    assert!(num_steps > 0, "RF sampling requires at least one step");
    let steps = num_steps + 1;
    let halfway = steps / 2;
    let step = 1.0_f32 / num_steps as f32;
    (0..steps)
        .map(|index| {
            let u = if index < halfway {
                step.mul_add(index as f32, 0.0)
            } else {
                (-step).mul_add((steps - index - 1) as f32, 1.0)
            };
            (1.0_f32 - u) * 0.999_f32
        })
        .collect()
}

/// Use only the B1/B2/B3 condition-module shapes that production inference
/// already needs. A single B40 matmul creates a distinct autotune/compile class
/// and made schedule preparation dominate startup despite doing little work.
fn prepare_condition_schedule(
    model: &TextToLatentRfDiT,
    evaluated: &[f32],
    dtype: DType,
    device: &Device,
) -> Option<Tensor<3>> {
    let mut rows = Vec::with_capacity(evaluated.len());
    for timesteps in evaluated.chunks(3) {
        let timestep = Tensor::<1>::from_floats(timesteps, device);
        let embedding = get_timestep_embedding(timestep, V4_TIMESTEP_EMBED_DIM, device);
        let output = model.cond_module.forward(embedding);
        if !wgpu_tensor_has_layout(&output, [timesteps.len(), 1, V4_COND_WIDTH], dtype, device) {
            return None;
        }
        rows.extend((0..timesteps.len()).map(|index| output.clone().narrow(0, index, 1)));
    }
    let output = Tensor::cat(rows, 0);
    wgpu_tensor_has_layout(&output, [evaluated.len(), 1, V4_COND_WIDTH], dtype, device)
        .then_some(output)
}

fn model_has_v4_contract(model: &TextToLatentRfDiT, device: &Device) -> bool {
    if model.model_dim != V4_COND_MODEL_DIM
        || model.timestep_embed_dim != V4_TIMESTEP_EMBED_DIM
        || model.blocks.len() != V4_BLOCKS
    {
        return false;
    }
    let linear0 = model.cond_module.linear0.weight.val();
    let linear1 = model.cond_module.linear1.weight.val();
    let linear2 = model.cond_module.linear2.weight.val();
    let dtype = linear0.dtype();
    float_element_bytes(dtype).is_some()
        && tensor_has_semantic_contract(
            &linear0,
            [V4_TIMESTEP_EMBED_DIM, V4_COND_MODEL_DIM],
            dtype,
            device,
        )
        && tensor_has_semantic_contract(
            &linear1,
            [V4_COND_MODEL_DIM, V4_COND_MODEL_DIM],
            dtype,
            device,
        )
        && tensor_has_semantic_contract(&linear2, [V4_COND_MODEL_DIM, V4_COND_WIDTH], dtype, device)
        && model.cond_module.linear0.bias.is_none()
        && model.cond_module.linear1.bias.is_none()
        && model.cond_module.linear2.bias.is_none()
}

fn tensor_has_semantic_contract<const D: usize>(
    tensor: &Tensor<D>,
    shape: [usize; D],
    dtype: DType,
    device: &Device,
) -> bool {
    tensor.dims() == shape && tensor.dtype() == dtype && tensor.device() == device.clone()
}

pub(crate) fn has_v4_cond_embed_layout(tensor: &Tensor<3>, batch: usize, device: &Device) -> bool {
    matches!(batch, 1..=3)
        && float_element_bytes(tensor.dtype()).is_some()
        && wgpu_tensor_has_layout(tensor, [batch, 1, V4_COND_WIDTH], tensor.dtype(), device)
}

const fn float_element_bytes(dtype: DType) -> Option<usize> {
    match dtype {
        DType::F32 => Some(size_of::<f32>()),
        DType::F16 => Some(size_of::<half::f16>()),
        _ => None,
    }
}

fn contiguous_strides<const D: usize>(shape: [usize; D]) -> Option<[usize; D]> {
    let mut strides = [0; D];
    let mut stride = 1usize;
    for index in (0..D).rev() {
        strides[index] = stride;
        stride = stride.checked_mul(shape[index])?;
    }
    Some(strides)
}

fn wgpu_tensor_has_layout<const D: usize>(
    tensor: &Tensor<D>,
    shape: [usize; D],
    dtype: DType,
    device: &Device,
) -> bool {
    let primitive = tensor
        .clone()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    let Some(strides) = contiguous_strides(shape) else {
        return false;
    };
    primitive.dtype == dtype
        && Device::from(primitive.device.clone()) == device.clone()
        && primitive.meta.num_dims() == D
        && primitive.meta.shape().dims::<D>() == shape
        && primitive.is_contiguous()
        && &primitive.meta.strides()[..] == strides.as_slice()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rf::GuidanceConfig;

    fn pinned_params() -> SamplerParams {
        SamplerParams {
            num_steps: 4,
            method: SamplerMethod::Euler,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Independent,
                scale_text: 3.0,
                scale_caption: 3.0,
                scale_speaker: 5.0,
                min_t: 0.5,
                max_t: 1.0,
            },
            truncation_factor: None,
            temporal_rescale: None,
            speaker_kv: None,
            use_context_kv_cache: true,
        }
    }

    #[test]
    fn pinned_and_product_schedule_bits_are_exact() {
        let schedule = reference_linear_schedule(4);
        assert_eq!(&schedule[..4], &[0.999, 0.74925, 0.4995, 0.24975]);
        assert_eq!(reference_linear_schedule(40).len(), 41);
        assert_eq!(reference_linear_schedule(40)[40].to_bits(), 0);
    }

    #[test]
    fn model_generations_are_nonzero_and_distinct() {
        let first = ModelGeneration::fresh();
        let second = ModelGeneration::fresh();
        assert_ne!(first, second);
    }

    #[test]
    fn prepared_sampler_policy_is_flexible_but_bounded_and_fail_closed() {
        let mut params = pinned_params();
        assert!(supports_prepared_euler_params(&params));

        params.num_steps = MAX_PREPARED_EULER_STEPS;
        assert!(supports_prepared_euler_params(&params));

        let mut changed = params.clone();
        changed.num_steps = MAX_PREPARED_EULER_STEPS + 1;
        assert!(!supports_prepared_euler_params(&changed));

        let mut changed = params.clone();
        changed.method = SamplerMethod::Heun;
        assert!(!supports_prepared_euler_params(&changed));

        let mut changed = params.clone();
        changed.guidance.mode = CfgGuidanceMode::Joint;
        assert!(!supports_prepared_euler_params(&changed));

        let mut changed = params.clone();
        changed.guidance.min_t = 1.0;
        changed.guidance.max_t = 0.5;
        assert!(!supports_prepared_euler_params(&changed));

        let mut changed = params.clone();
        changed.guidance.min_t = f32::NAN;
        assert!(!supports_prepared_euler_params(&changed));

        // These policies do not alter timestep embeddings and therefore do
        // not invalidate a cache with the same schedule and guidance window.
        let mut changed = params.clone();
        changed.guidance.scale_text = 7.0;
        changed.truncation_factor = Some(0.9);
        changed.temporal_rescale = Some(crate::rf::TemporalRescaleConfig { k: 1.0, sigma: 1.0 });
        changed.speaker_kv = Some(crate::rf::SpeakerKvConfig {
            scale: 2.0,
            max_layers: None,
            min_t: None,
        });
        changed.use_context_kv_cache = false;
        assert!(supports_prepared_euler_params(&changed));
    }
}
