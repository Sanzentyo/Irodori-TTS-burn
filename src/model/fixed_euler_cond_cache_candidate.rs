//! Isolated fixed-schedule condition-cache candidate for the pinned v4 replay.
//!
//! This module is included only by `bench_fixed_euler_cond_cache`; it is not
//! registered by the production model.  The specialization is deliberately
//! narrow: four Euler evaluations, the exact production `f32` schedule, and
//! effective batches `[2, 2, 1, 1]`.  Every public entry point checks the
//! schedule key, tensor shape, dtype, physical layout, and WGPU device before
//! reusing cached values.

use std::{error::Error, io, path::Path};

use burn::{
    backend::wgpu::WgpuDevice,
    tensor::{DType, Distribution, Tensor, TensorData, activation::silu, module::linear},
};
use irodori_tts_wgpu::{ModelConfig, WgpuRaw, weights::TensorStore};

type B = WgpuRaw;

pub const EULER_STEPS: usize = 4;
pub const TIMESTEP_EMBED_DIM: usize = 512;
pub const MODEL_DIM: usize = 1_280;
pub const COND_WIDTH: usize = MODEL_DIM * 3;
pub const EFFECTIVE_BATCHES: [usize; EULER_STEPS] = [2, 2, 1, 1];
pub const UNIQUE_CACHE_BYTES: usize = EULER_STEPS * COND_WIDTH * size_of::<f32>();
pub const LOGICAL_MATERIALIZED_BYTES: usize =
    (EFFECTIVE_BATCHES[0] + EFFECTIVE_BATCHES[1] + EFFECTIVE_BATCHES[2] + EFFECTIVE_BATCHES[3])
        * COND_WIDTH
        * size_of::<f32>();

const INIT_SCALE: f32 = 0.999;
const HALF_TIMESTEP_EMBED_DIM: usize = TIMESTEP_EMBED_DIM / 2;
const V4_NUM_LAYERS: usize = 12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CondWeightLayout {
    RandomContiguous,
    ProductionLoaderTransposed,
}

/// Reproduce `rf/euler_sampler.rs` in the same `f32` operation order.
pub fn pinned_schedule() -> [f32; EULER_STEPS] {
    std::array::from_fn(|index| INIT_SCALE * (1.0 - index as f32 / EULER_STEPS as f32))
}

/// Cache identity for one immutable model instance and the exact v4 request.
///
/// Production integration would make the cache a child of the immutable model;
/// `model_generation` makes that ownership requirement explicit in this
/// isolated experiment and prevents accidental reuse after replacing weights.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedEulerCondCacheKey {
    schedule_bits: [u32; EULER_STEPS],
    effective_batches: [usize; EULER_STEPS],
    model_generation: u64,
}

impl FixedEulerCondCacheKey {
    pub fn pinned(model_generation: u64) -> Result<Self, Box<dyn Error>> {
        if model_generation == 0 {
            return Err(io::Error::other("model generation must be non-zero").into());
        }
        Ok(Self {
            schedule_bits: pinned_schedule().map(f32::to_bits),
            effective_batches: EFFECTIVE_BATCHES,
            model_generation,
        })
    }

    fn validate(self) -> Result<(), Box<dyn Error>> {
        let expected_bits = pinned_schedule().map(f32::to_bits);
        if self.schedule_bits != expected_bits {
            return Err(io::Error::other(format!(
                "fixed Euler cache schedule mismatch: expected {expected_bits:x?}, got {:x?}",
                self.schedule_bits
            ))
            .into());
        }
        if self.effective_batches != EFFECTIVE_BATCHES {
            return Err(io::Error::other(format!(
                "fixed Euler cache batch plan mismatch: expected {EFFECTIVE_BATCHES:?}, got {:?}",
                self.effective_batches
            ))
            .into());
        }
        if self.model_generation == 0 {
            return Err(io::Error::other("model generation must be non-zero").into());
        }
        Ok(())
    }

    fn values(self) -> [f32; EULER_STEPS] {
        self.schedule_bits.map(f32::from_bits)
    }
}

#[derive(Clone)]
pub struct CondWeights {
    linear0: Tensor<B, 2>,
    linear1: Tensor<B, 2>,
    linear2: Tensor<B, 2>,
    layout: CondWeightLayout,
    checkpoint_keys: Option<[String; 3]>,
}

impl CondWeights {
    pub fn random(device: &WgpuDevice) -> Self {
        Self {
            linear0: Tensor::random(
                [TIMESTEP_EMBED_DIM, MODEL_DIM],
                Distribution::Uniform(-0.05, 0.05),
                device,
            ),
            linear1: Tensor::random(
                [MODEL_DIM, MODEL_DIM],
                Distribution::Uniform(-0.05, 0.05),
                device,
            ),
            linear2: Tensor::random(
                [MODEL_DIM, COND_WIDTH],
                Distribution::Uniform(-0.05, 0.05),
                device,
            ),
            layout: CondWeightLayout::RandomContiguous,
            checkpoint_keys: None,
        }
    }

    /// Load only the official checkpoint's `CondModule` weights using the
    /// exact key preference and PyTorch-to-Burn transpose used by production.
    ///
    /// The caller must verify the pinned checkpoint SHA-256 before entering
    /// this function. `TensorStore` necessarily reads the full checkpoint, but
    /// it is dropped as soon as these three tensors have been transferred.
    pub fn from_verified_official_checkpoint(
        path: &Path,
        device: &WgpuDevice,
    ) -> Result<Self, Box<dyn Error>> {
        let store = TensorStore::load(path)?;
        let config: ModelConfig = serde_json::from_str(&store.config_json)?;
        config.validate()?;
        if config.model_dim != MODEL_DIM
            || config.timestep_embed_dim != TIMESTEP_EMBED_DIM
            || config.num_layers != V4_NUM_LAYERS
        {
            return Err(io::Error::other(format!(
                "official v4 geometry mismatch: expected model_dim={MODEL_DIM}, \
                 timestep_embed_dim={TIMESTEP_EMBED_DIM}, num_layers={V4_NUM_LAYERS}; got \
                 model_dim={}, timestep_embed_dim={}, num_layers={}",
                config.model_dim, config.timestep_embed_dim, config.num_layers
            ))
            .into());
        }

        let (linear0, linear0_key) = load_checkpoint_linear(
            &store,
            "linear0",
            0,
            [MODEL_DIM, TIMESTEP_EMBED_DIM],
            [TIMESTEP_EMBED_DIM, MODEL_DIM],
            device,
        )?;
        let (linear1, linear1_key) = load_checkpoint_linear(
            &store,
            "linear1",
            2,
            [MODEL_DIM, MODEL_DIM],
            [MODEL_DIM, MODEL_DIM],
            device,
        )?;
        let (linear2, linear2_key) = load_checkpoint_linear(
            &store,
            "linear2",
            4,
            [COND_WIDTH, MODEL_DIM],
            [MODEL_DIM, COND_WIDTH],
            device,
        )?;
        drop(store);

        let weights = Self {
            linear0,
            linear1,
            linear2,
            layout: CondWeightLayout::ProductionLoaderTransposed,
            checkpoint_keys: Some([linear0_key, linear1_key, linear2_key]),
        };
        weights.validate(device)?;
        Ok(weights)
    }

    pub const fn source_label(&self) -> &'static str {
        match self.layout {
            CondWeightLayout::RandomContiguous => "seeded-random-contiguous",
            CondWeightLayout::ProductionLoaderTransposed => {
                "official-v4-checkpoint-production-transposed"
            }
        }
    }

    pub fn checkpoint_keys(&self) -> Option<[&str; 3]> {
        self.checkpoint_keys
            .as_ref()
            .map(|keys| [keys[0].as_str(), keys[1].as_str(), keys[2].as_str()])
    }

    pub fn validate(&self, device: &WgpuDevice) -> Result<(), Box<dyn Error>> {
        let expected_strides = match self.layout {
            CondWeightLayout::RandomContiguous => [MODEL_DIM, 1],
            CondWeightLayout::ProductionLoaderTransposed => [1, TIMESTEP_EMBED_DIM],
        };
        validate_weight_tensor(
            "cond.linear0",
            &self.linear0,
            [TIMESTEP_EMBED_DIM, MODEL_DIM],
            expected_strides,
            device,
        )?;
        let expected_strides = match self.layout {
            CondWeightLayout::RandomContiguous => [MODEL_DIM, 1],
            CondWeightLayout::ProductionLoaderTransposed => [1, MODEL_DIM],
        };
        validate_weight_tensor(
            "cond.linear1",
            &self.linear1,
            [MODEL_DIM, MODEL_DIM],
            expected_strides,
            device,
        )?;
        let expected_strides = match self.layout {
            CondWeightLayout::RandomContiguous => [COND_WIDTH, 1],
            CondWeightLayout::ProductionLoaderTransposed => [1, MODEL_DIM],
        };
        validate_weight_tensor(
            "cond.linear2",
            &self.linear2,
            [MODEL_DIM, COND_WIDTH],
            expected_strides,
            device,
        )?;

        match (self.layout, &self.checkpoint_keys) {
            (CondWeightLayout::RandomContiguous, None)
            | (CondWeightLayout::ProductionLoaderTransposed, Some(_)) => Ok(()),
            _ => Err(io::Error::other(
                "condition weight source metadata does not match its physical layout",
            )
            .into()),
        }
    }

    pub const fn parameter_macs_per_row() -> usize {
        TIMESTEP_EMBED_DIM * MODEL_DIM + MODEL_DIM * MODEL_DIM + MODEL_DIM * COND_WIDTH
    }
}

fn load_checkpoint_linear(
    store: &TensorStore,
    burn_name: &str,
    torch_index: usize,
    pytorch_shape: [usize; 2],
    burn_shape: [usize; 2],
    device: &WgpuDevice,
) -> Result<(Tensor<B, 2>, String), Box<dyn Error>> {
    let converted_prefix = format!("cond_module.{burn_name}");
    let official_prefix = format!("cond_module.{torch_index}");
    let converted_weight = format!("{converted_prefix}.weight");
    let official_weight = format!("{official_prefix}.weight");
    let selected_prefix = if store.has(&converted_weight) {
        converted_prefix
    } else if store.has(&official_weight) {
        official_prefix
    } else {
        return Err(io::Error::other(format!(
            "missing CondModule weight: expected production key {converted_weight:?} \
             or official key {official_weight:?}"
        ))
        .into());
    };
    let weight_key = format!("{selected_prefix}.weight");
    let bias_key = format!("{selected_prefix}.bias");
    if store.has(&bias_key) {
        return Err(io::Error::other(format!(
            "unsupported CondModule bias {bias_key:?}; the pinned v4 module is bias-free"
        ))
        .into());
    }

    let pytorch_weight: Tensor<B, 2> = store.tensor(&weight_key, device)?;
    validate_tensor(
        &format!("checkpoint {weight_key}"),
        &pytorch_weight,
        pytorch_shape,
        device,
    )?;
    let burn_weight = pytorch_weight.transpose();
    validate_weight_tensor(
        &format!("transposed {weight_key}"),
        &burn_weight,
        burn_shape,
        [1, pytorch_shape[1]],
        device,
    )?;
    Ok((burn_weight, weight_key))
}

/// Device tensors are built outside the timed paths, matching the sampler's
/// pre-allocation of `tt_base` and `tt_cfg` before its Euler loop.
#[derive(Clone)]
pub struct FixedEulerTimestepInputs {
    per_step: [Tensor<B, 1>; EULER_STEPS],
    unique: Tensor<B, 1>,
}

impl FixedEulerTimestepInputs {
    pub fn new(key: FixedEulerCondCacheKey, device: &WgpuDevice) -> Result<Self, Box<dyn Error>> {
        key.validate()?;
        let values = key.values();
        let per_step = std::array::from_fn(|index| {
            let values = vec![values[index]; EFFECTIVE_BATCHES[index]];
            Tensor::<B, 1>::from_floats(values.as_slice(), device)
        });
        let unique = Tensor::<B, 1>::from_floats(values, device);
        let inputs = Self { per_step, unique };
        inputs.validate(device)?;
        Ok(inputs)
    }

    fn validate(&self, device: &WgpuDevice) -> Result<(), Box<dyn Error>> {
        for (index, tensor) in self.per_step.iter().enumerate() {
            validate_tensor(
                &format!("timestep.per_step[{index}]"),
                tensor,
                [EFFECTIVE_BATCHES[index]],
                device,
            )?;
        }
        validate_tensor("timestep.unique", &self.unique, [EULER_STEPS], device)
    }
}

#[derive(Clone)]
pub struct FixedEulerCondOutputs {
    steps: [Tensor<B, 3>; EULER_STEPS],
}

impl FixedEulerCondOutputs {
    pub fn step(&self, index: usize) -> Result<Tensor<B, 3>, Box<dyn Error>> {
        self.steps.get(index).cloned().ok_or_else(|| {
            io::Error::other(format!("Euler step index {index} is out of range")).into()
        })
    }

    pub fn validate(&self, device: &WgpuDevice) -> Result<(), Box<dyn Error>> {
        for (index, output) in self.steps.iter().enumerate() {
            validate_tensor(
                &format!("cond_output[{index}]"),
                output,
                [EFFECTIVE_BATCHES[index], 1, COND_WIDTH],
                device,
            )?;
        }
        Ok(())
    }

    pub fn last(&self) -> Tensor<B, 3> {
        self.steps[EULER_STEPS - 1].clone()
    }
}

/// Current sampler behavior: execute timestep embedding and `CondModule` once
/// for every model evaluation, including the duplicate CFG rows at steps 0/1.
pub fn baseline_fixed_request(
    key: FixedEulerCondCacheKey,
    inputs: &FixedEulerTimestepInputs,
    weights: &CondWeights,
    device: &WgpuDevice,
) -> Result<FixedEulerCondOutputs, Box<dyn Error>> {
    key.validate()?;
    inputs.validate(device)?;
    weights.validate(device)?;
    let steps = std::array::from_fn(|index| {
        condition_from_timestep(inputs.per_step[index].clone(), weights, device)
    });
    Ok(FixedEulerCondOutputs { steps })
}

/// Cache only the four unique condition rows.  A hit still materializes the
/// two duplicated B=2 outputs required by the first CFG steps.
#[derive(Clone)]
pub struct UniqueFixedEulerCondCache {
    key: FixedEulerCondCacheKey,
    unique: Tensor<B, 3>,
    device: WgpuDevice,
}

impl UniqueFixedEulerCondCache {
    pub fn build(
        key: FixedEulerCondCacheKey,
        inputs: &FixedEulerTimestepInputs,
        weights: &CondWeights,
        device: &WgpuDevice,
    ) -> Result<Self, Box<dyn Error>> {
        key.validate()?;
        inputs.validate(device)?;
        weights.validate(device)?;
        let unique = condition_from_timestep(inputs.unique.clone(), weights, device);
        let cache = Self {
            key,
            unique,
            device: device.clone(),
        };
        cache.validate(key, device)?;
        Ok(cache)
    }

    fn validate(
        &self,
        requested_key: FixedEulerCondCacheKey,
        device: &WgpuDevice,
    ) -> Result<(), Box<dyn Error>> {
        requested_key.validate()?;
        if self.key != requested_key {
            return Err(io::Error::other(format!(
                "fixed Euler cache key mismatch: built={:?}, requested={requested_key:?}",
                self.key
            ))
            .into());
        }
        if &self.device != device {
            return Err(io::Error::other("fixed Euler cache WGPU device mismatch").into());
        }
        validate_tensor(
            "fixed_euler.unique_cache",
            &self.unique,
            [EULER_STEPS, 1, COND_WIDTH],
            device,
        )
    }

    pub fn materialize(
        &self,
        requested_key: FixedEulerCondCacheKey,
        device: &WgpuDevice,
    ) -> Result<FixedEulerCondOutputs, Box<dyn Error>> {
        self.validate(requested_key, device)?;
        let steps = std::array::from_fn(|index| {
            let unique_row = self
                .unique
                .clone()
                .slice([index..index + 1, 0..1, 0..COND_WIDTH]);
            if EFFECTIVE_BATCHES[index] == 1 {
                unique_row
            } else {
                Tensor::cat(vec![unique_row.clone(), unique_row], 0)
            }
        });
        let outputs = FixedEulerCondOutputs { steps };
        outputs.validate(device)?;
        Ok(outputs)
    }

    pub fn last(&self) -> Tensor<B, 3> {
        self.unique
            .clone()
            .slice([EULER_STEPS - 1..EULER_STEPS, 0..1, 0..COND_WIDTH])
    }
}

/// Build B=4 once and include the mandatory B=2 materializations in the same
/// per-request measurement.
pub fn batched_fixed_request(
    key: FixedEulerCondCacheKey,
    inputs: &FixedEulerTimestepInputs,
    weights: &CondWeights,
    device: &WgpuDevice,
) -> Result<FixedEulerCondOutputs, Box<dyn Error>> {
    UniqueFixedEulerCondCache::build(key, inputs, weights, device)?.materialize(key, device)
}

/// Fully reusable fixed-schedule outputs.  The two repeated B=2 allocations
/// and the B=4 backing allocation retained by the B=1 views are constructed
/// once, so a cache hit submits no GPU work.
#[derive(Clone)]
pub struct MaterializedFixedEulerCondCache {
    key: FixedEulerCondCacheKey,
    outputs: FixedEulerCondOutputs,
    device: WgpuDevice,
}

impl MaterializedFixedEulerCondCache {
    pub fn build(
        key: FixedEulerCondCacheKey,
        inputs: &FixedEulerTimestepInputs,
        weights: &CondWeights,
        device: &WgpuDevice,
    ) -> Result<Self, Box<dyn Error>> {
        let outputs = batched_fixed_request(key, inputs, weights, device)?;
        let cache = Self {
            key,
            outputs,
            device: device.clone(),
        };
        cache.validate(key, device)?;
        Ok(cache)
    }

    fn validate(
        &self,
        requested_key: FixedEulerCondCacheKey,
        device: &WgpuDevice,
    ) -> Result<(), Box<dyn Error>> {
        requested_key.validate()?;
        if self.key != requested_key {
            return Err(io::Error::other(format!(
                "materialized fixed Euler cache key mismatch: built={:?}, requested={requested_key:?}",
                self.key
            ))
            .into());
        }
        if &self.device != device {
            return Err(
                io::Error::other("materialized fixed Euler cache WGPU device mismatch").into(),
            );
        }
        self.outputs.validate(device)
    }

    pub fn get(
        &self,
        requested_key: FixedEulerCondCacheKey,
        device: &WgpuDevice,
    ) -> Result<FixedEulerCondOutputs, Box<dyn Error>> {
        self.validate(requested_key, device)?;
        Ok(self.outputs.clone())
    }

    pub fn last(&self) -> Tensor<B, 3> {
        self.outputs.last()
    }
}

fn exact_timestep_embedding(timestep: Tensor<B, 1>, device: &WgpuDevice) -> Tensor<B, 2> {
    // This intentionally mirrors model/rope.rs rather than algebraically
    // rearranging it.  In particular, all frequency values are generated by
    // scalar f32 operations in index order on the host.
    let log_10000 = (10000.0_f32).ln();
    let freqs_data = (0..HALF_TIMESTEP_EMBED_DIM)
        .map(|index| 1000.0 * ((-log_10000 * index as f32) / HALF_TIMESTEP_EMBED_DIM as f32).exp())
        .collect::<Vec<_>>();
    let freqs: Tensor<B, 2> = Tensor::from_floats(
        TensorData::new(freqs_data, [1, HALF_TIMESTEP_EMBED_DIM]),
        device,
    );
    let timestep: Tensor<B, 2> = timestep.unsqueeze_dim::<2>(1);
    let args = timestep * freqs;
    Tensor::cat(vec![args.clone().cos(), args.sin()], 1)
}

fn condition_from_timestep(
    timestep: Tensor<B, 1>,
    weights: &CondWeights,
    device: &WgpuDevice,
) -> Tensor<B, 3> {
    // Keep the exact CondModule statement order:
    // Linear(512,D) -> SiLU -> Linear(D,D) -> SiLU -> Linear(D,3D).
    let embedded = exact_timestep_embedding(timestep, device);
    let hidden = silu(linear(embedded, weights.linear0.clone(), None));
    let hidden = silu(linear(hidden, weights.linear1.clone(), None));
    linear(hidden, weights.linear2.clone(), None).unsqueeze_dim::<3>(1)
}

fn contiguous_strides<const D: usize>(shape: [usize; D]) -> Result<[usize; D], Box<dyn Error>> {
    let mut strides = [0; D];
    let mut stride = 1usize;
    for index in (0..D).rev() {
        strides[index] = stride;
        stride = stride.checked_mul(shape[index]).ok_or_else(|| {
            io::Error::other(format!("contiguous stride overflows for shape {shape:?}"))
        })?;
    }
    Ok(strides)
}

fn validate_tensor<const D: usize>(
    name: &str,
    tensor: &Tensor<B, D>,
    expected_shape: [usize; D],
    device: &WgpuDevice,
) -> Result<(), Box<dyn Error>> {
    let raw = tensor.clone().into_primitive().tensor();
    if raw.dtype != DType::F32 {
        return Err(io::Error::other(format!("{name} must be f32, got {:?}", raw.dtype)).into());
    }
    if &raw.device != device {
        return Err(io::Error::other(format!("{name} is on a different WGPU device")).into());
    }
    if raw.meta.num_dims() != D || raw.meta.shape().dims::<D>() != expected_shape {
        return Err(io::Error::other(format!(
            "{name} shape mismatch: expected {expected_shape:?}, got {:?}",
            raw.meta.shape()
        ))
        .into());
    }
    let expected_strides = contiguous_strides(expected_shape)?;
    if !raw.is_contiguous() || &raw.meta.strides()[..] != expected_strides.as_slice() {
        return Err(io::Error::other(format!(
            "{name} layout mismatch: expected contiguous strides {expected_strides:?}, got {:?}",
            raw.meta.strides()
        ))
        .into());
    }
    Ok(())
}

fn validate_weight_tensor(
    name: &str,
    tensor: &Tensor<B, 2>,
    expected_shape: [usize; 2],
    expected_strides: [usize; 2],
    device: &WgpuDevice,
) -> Result<(), Box<dyn Error>> {
    let raw = tensor.clone().into_primitive().tensor();
    if raw.dtype != DType::F32 {
        return Err(io::Error::other(format!("{name} must be f32, got {:?}", raw.dtype)).into());
    }
    if &raw.device != device {
        return Err(io::Error::other(format!("{name} is on a different WGPU device")).into());
    }
    if raw.meta.num_dims() != 2 || raw.meta.shape().dims::<2>() != expected_shape {
        return Err(io::Error::other(format!(
            "{name} shape mismatch: expected {expected_shape:?}, got {:?}",
            raw.meta.shape()
        ))
        .into());
    }
    if &raw.meta.strides()[..] != expected_strides.as_slice() {
        return Err(io::Error::other(format!(
            "{name} layout mismatch: expected strides {expected_strides:?}, got {:?}",
            raw.meta.strides()
        ))
        .into());
    }
    let expected_contiguous = expected_strides == contiguous_strides(expected_shape)?;
    if raw.is_contiguous() != expected_contiguous {
        return Err(io::Error::other(format!(
            "{name} contiguity mismatch: expected {expected_contiguous}, got {}",
            raw.is_contiguous()
        ))
        .into());
    }
    Ok(())
}
