//! Inference-only cache for the pinned v4 four-step timestep condition.
//!
//! The cache is owned by [`crate::inference::WgslInferenceEngine`], not by a
//! Burn module, so it is deliberately absent from model records.  It is built
//! only for the measured raw-f32 WGPU policy and rejects every schedule,
//! sampler configuration, model generation, shape, layout, or device outside
//! the pinned contract.

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

use super::{dit::TextToLatentRfDiT, rope::get_timestep_embedding};

pub(crate) const FIXED_EULER_STEPS: usize = 4;
pub(crate) const FIXED_EULER_BATCHES: [usize; FIXED_EULER_STEPS] = [2, 2, 1, 1];
pub(crate) const V4_TIMESTEP_EMBED_DIM: usize = 512;
pub(crate) const V4_COND_MODEL_DIM: usize = 1_280;
pub(crate) const V4_COND_WIDTH: usize = V4_COND_MODEL_DIM * 3;
pub(crate) const MATERIALIZED_CACHE_BYTES: usize = 122_880;

const V4_BLOCKS: usize = 12;
const INIT_SCALE: f32 = 0.999;
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

/// Exact four evaluations used by the pinned Euler request.  The endpoint at
/// `t=0` is not evaluated by Euler and is intentionally absent.
pub(crate) fn fixed_euler_schedule() -> [f32; FIXED_EULER_STEPS] {
    std::array::from_fn(|index| INIT_SCALE * (1.0 - index as f32 / FIXED_EULER_STEPS as f32))
}

/// Return whether engine construction may prepare the measured cache.
///
/// Request-dependent text-only/B=1 checks are performed again inside the
/// sampler after condition masks have been resolved.
pub(crate) fn supports_fixed_euler_params(params: &SamplerParams) -> bool {
    params.num_steps == FIXED_EULER_STEPS
        && matches!(params.method, SamplerMethod::Euler)
        && matches!(params.guidance.mode, CfgGuidanceMode::Independent)
        && params.guidance.scale_text.to_bits() == 3.0_f32.to_bits()
        && params.guidance.scale_caption.to_bits() == 3.0_f32.to_bits()
        && params.guidance.scale_speaker.to_bits() == 5.0_f32.to_bits()
        && params.guidance.min_t.to_bits() == 0.5_f32.to_bits()
        && params.guidance.max_t.to_bits() == 1.0_f32.to_bits()
        && params.truncation_factor.is_none()
        && params.temporal_rescale.is_none()
        && params.speaker_kv.is_none()
        && params.use_context_kv_cache
}

/// Materialized `[B, 1, 3D]` timestep conditions for all four Euler calls.
///
/// The B=1 outputs retain the original B=4 allocation while the first two
/// entries own explicit B=2 `Tensor::cat` buffers.  Physical retained storage
/// is therefore 120 KiB, not the 90 KiB logical output size.
pub(crate) struct FixedEulerCondCache {
    generation: ModelGeneration,
    schedule_bits: [u32; FIXED_EULER_STEPS],
    batches: [usize; FIXED_EULER_STEPS],
    outputs: [Tensor<3>; FIXED_EULER_STEPS],
    device: Device,
}

impl FixedEulerCondCache {
    /// Build all four unique rows together, then materialize the two B=2 rows.
    /// Any partial or mismatching result rejects the cache atomically.
    pub(crate) fn try_build(
        model: &TextToLatentRfDiT,
        generation: ModelGeneration,
        device: &Device,
    ) -> Option<Self> {
        debug_assert_eq!(
            MATERIALIZED_CACHE_BYTES,
            (FIXED_EULER_STEPS + FIXED_EULER_BATCHES[0] + FIXED_EULER_BATCHES[1])
                * V4_COND_WIDTH
                * size_of::<f32>()
        );
        if !model_has_v4_contract(model, device) {
            return None;
        }

        let schedule = fixed_euler_schedule();
        let timesteps = Tensor::<1>::from_floats(schedule, device);
        let timestep_embed = get_timestep_embedding(timesteps, V4_TIMESTEP_EMBED_DIM, device);
        let unique = model.cond_module.forward(timestep_embed);
        if !wgpu_tensor_has_layout(&unique, [FIXED_EULER_STEPS, 1, V4_COND_WIDTH], device) {
            return None;
        }

        let row = |index| {
            unique
                .clone()
                .slice([index..index + 1, 0..1, 0..V4_COND_WIDTH])
        };
        let first = row(0);
        let second = row(1);
        let outputs = [
            Tensor::cat(vec![first.clone(), first], 0),
            Tensor::cat(vec![second.clone(), second], 0),
            row(2),
            row(3),
        ];
        let cache = Self {
            generation,
            schedule_bits: schedule.map(f32::to_bits),
            batches: FIXED_EULER_BATCHES,
            outputs,
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
    ) -> Option<Tensor<3>> {
        if self.generation != generation
            || &self.device != device
            || self.schedule_bits != fixed_euler_schedule().map(f32::to_bits)
            || self.batches != FIXED_EULER_BATCHES
            || self.schedule_bits.get(index).copied()? != timestep_bits
            || self.batches.get(index).copied()? != batch
            || !self.has_storage_contract()
        {
            return None;
        }
        self.outputs.get(index).cloned()
    }

    pub(crate) fn matches_model(&self, generation: ModelGeneration, device: &Device) -> bool {
        self.generation == generation && &self.device == device && self.has_storage_contract()
    }

    fn has_storage_contract(&self) -> bool {
        self.outputs.iter().enumerate().all(|(index, output)| {
            wgpu_tensor_has_layout(
                output,
                [FIXED_EULER_BATCHES[index], 1, V4_COND_WIDTH],
                &self.device,
            )
        })
    }
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
    tensor_has_semantic_contract(&linear0, [V4_TIMESTEP_EMBED_DIM, V4_COND_MODEL_DIM], device)
        && tensor_has_semantic_contract(&linear1, [V4_COND_MODEL_DIM, V4_COND_MODEL_DIM], device)
        && tensor_has_semantic_contract(&linear2, [V4_COND_MODEL_DIM, V4_COND_WIDTH], device)
        && model.cond_module.linear0.bias.is_none()
        && model.cond_module.linear1.bias.is_none()
        && model.cond_module.linear2.bias.is_none()
}

fn tensor_has_semantic_contract<const D: usize>(
    tensor: &Tensor<D>,
    shape: [usize; D],
    device: &Device,
) -> bool {
    tensor.dims() == shape && tensor.dtype() == DType::F32 && tensor.device() == device.clone()
}

pub(crate) fn has_v4_cond_embed_layout(tensor: &Tensor<3>, batch: usize, device: &Device) -> bool {
    matches!(batch, 1 | 2) && wgpu_tensor_has_layout(tensor, [batch, 1, V4_COND_WIDTH], device)
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
    device: &Device,
) -> bool {
    let primitive = tensor
        .clone()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    let Some(strides) = contiguous_strides(shape) else {
        return false;
    };
    primitive.dtype == DType::F32
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
            num_steps: FIXED_EULER_STEPS,
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
    fn pinned_schedule_and_storage_accounting_are_exact() {
        let schedule = fixed_euler_schedule();
        assert_eq!(schedule, [0.999, 0.74925, 0.4995, 0.24975]);
        assert_eq!(FIXED_EULER_BATCHES, [2, 2, 1, 1]);
        assert_eq!(MATERIALIZED_CACHE_BYTES, 120 * 1024);
    }

    #[test]
    fn model_generations_are_nonzero_and_distinct() {
        let first = ModelGeneration::fresh();
        let second = ModelGeneration::fresh();
        assert_ne!(first, second);
    }

    #[test]
    fn pinned_sampler_params_are_the_only_supported_policy() {
        let params = pinned_params();
        assert!(supports_fixed_euler_params(&params));

        let mut changed = params.clone();
        changed.num_steps = 5;
        assert!(!supports_fixed_euler_params(&changed));

        let mut changed = params.clone();
        changed.method = SamplerMethod::Heun;
        assert!(!supports_fixed_euler_params(&changed));

        let mut changed = params.clone();
        changed.guidance.mode = CfgGuidanceMode::Joint;
        assert!(!supports_fixed_euler_params(&changed));

        let mut changed = params.clone();
        changed.guidance.min_t = f32::from_bits(0.5_f32.to_bits() + 1);
        assert!(!supports_fixed_euler_params(&changed));

        let mut changed = params.clone();
        changed.guidance.scale_text = f32::from_bits(3.0_f32.to_bits() + 1);
        assert!(!supports_fixed_euler_params(&changed));

        let mut changed = params.clone();
        changed.guidance.scale_caption = f32::from_bits(3.0_f32.to_bits() + 1);
        assert!(!supports_fixed_euler_params(&changed));

        let mut changed = params.clone();
        changed.guidance.scale_speaker = f32::from_bits(5.0_f32.to_bits() + 1);
        assert!(!supports_fixed_euler_params(&changed));

        let mut changed = params.clone();
        changed.guidance.max_t = f32::from_bits(1.0_f32.to_bits() - 1);
        assert!(!supports_fixed_euler_params(&changed));

        let mut changed = params.clone();
        changed.truncation_factor = Some(0.9);
        assert!(!supports_fixed_euler_params(&changed));

        let mut changed = params.clone();
        changed.temporal_rescale = Some(crate::rf::TemporalRescaleConfig { k: 1.0, sigma: 1.0 });
        assert!(!supports_fixed_euler_params(&changed));

        let mut changed = params.clone();
        changed.speaker_kv = Some(crate::rf::SpeakerKvConfig {
            scale: 2.0,
            max_layers: None,
            min_t: None,
        });
        assert!(!supports_fixed_euler_params(&changed));

        let mut changed = params;
        changed.use_context_kv_cache = false;
        assert!(!supports_fixed_euler_params(&changed));
    }
}
