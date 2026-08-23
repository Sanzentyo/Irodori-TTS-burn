//! Inference-only cross-layer batching for v4 LowRankAdaLN projections.
//!
//! All diffusion blocks consume the same timestep condition. Packing the
//! attention/MLP AdaLN weights in block-major order lets the WGSL execution
//! policy compute every modulation with one down and one up batched matmul.
//! The cache is owned by the inference wrapper and is deliberately absent from
//! Burn records and portable execution paths.

use burn::backend::wgpu::CubeTensor;
use burn::module::{Param, ParamId};
use burn::tensor::Device;
use burn::tensor::{DType, Shape, Tensor, activation::silu};

use super::norm::{AdaLnModulation, LowRankAdaLn};

pub(crate) const ADALN_BRANCHES: usize = 3;
pub(crate) const V4_ADALN_BLOCKS: usize = 12;
pub(crate) const V4_ADALN_MODULES: usize = V4_ADALN_BLOCKS * 2;
pub(crate) const V4_ADALN_MODEL_DIM: usize = 1_280;
pub(crate) const V4_ADALN_RANK: usize = 192;
// Independent CFG uses B2 for text-only and B3 when one auxiliary signal is
// active.  The timestep condition is shared by every block in both cases, so
// excluding B3 would silently re-run all 24 low-rank AdaLN modules for the
// first half of every design/clone trajectory.
const V4_MAX_BATCH: usize = 3;

#[derive(Clone, Debug)]
struct ModuleSources {
    down: [Tensor<2>; ADALN_BRANCHES],
    up: [Tensor<2>; ADALN_BRANCHES],
    bias: [Tensor<1>; ADALN_BRANCHES],
}

impl ModuleSources {
    fn from_module(module: &LowRankAdaLn) -> Option<Self> {
        Some(Self {
            down: [
                module.shift_down.weight.val(),
                module.scale_down.weight.val(),
                module.gate_down.weight.val(),
            ],
            up: [
                module.shift_up.weight.val(),
                module.scale_up.weight.val(),
                module.gate_up.weight.val(),
            ],
            bias: [
                module.shift_up.bias.as_ref()?.val(),
                module.scale_up.bias.as_ref()?.val(),
                module.gate_up.bias.as_ref()?.val(),
            ],
        })
    }

    fn has_contract(&self, model_dim: usize, rank: usize, dtype: DType, device: &Device) -> bool {
        self.down.iter().all(|tensor| {
            tensor.dims() == [model_dim, rank]
                && tensor.dtype() == dtype
                && tensor.device() == device.clone()
        }) && self.up.iter().all(|tensor| {
            tensor.dims() == [rank, model_dim]
                && tensor.dtype() == dtype
                && tensor.device() == device.clone()
        }) && self.bias.iter().all(|tensor| {
            tensor.dims() == [model_dim]
                && tensor.dtype() == dtype
                && tensor.device() == device.clone()
        })
    }
}

/// Atomically prepared module-major cache.
///
/// Slot order is `[block0.attn, block0.mlp, block1.attn, block1.mlp, ...]`,
/// with shift/scale/gate contiguous inside every module slot.
#[derive(Debug)]
pub(crate) struct CrossLayerAdaLnCache {
    down: Tensor<4>,
    up: Tensor<4>,
    bias: Tensor<4>,
    module_count: usize,
    model_dim: usize,
    rank: usize,
    dtype: DType,
    device: Device,
}

/// All precomputed module modulations for one DiT evaluation.
#[derive(Clone, Debug)]
pub(crate) struct CrossLayerAdaLnModulations {
    values: Tensor<4>,
    batch: usize,
    module_count: usize,
    model_dim: usize,
}

/// One packed allocation containing equal-batch modulations for a schedule.
///
/// Keeping each step as an independent tensor is disproportionately expensive
/// with page-isolated allocators. Packing after the ordinary B1/B2/B3 matmul
/// routes preserves those proven kernels while reducing retained allocations
/// from O(steps) to one per batch topology.
#[derive(Clone, Debug)]
pub(crate) struct CrossLayerAdaLnBatchSchedule {
    values: Tensor<5>,
    step_count: usize,
    batch: usize,
    module_count: usize,
    model_dim: usize,
}

/// Attention and MLP modulation slices for one diffusion block.
#[derive(Debug)]
pub(crate) struct BlockAdaLnModulations {
    pub(crate) attention: AdaLnModulation,
    pub(crate) mlp: AdaLnModulation,
}

impl CrossLayerAdaLnCache {
    /// Build one cache from modules already ordered block-major attn/MLP.
    ///
    /// Validation happens before any stack operation, so missing biases or a
    /// partial/mixed-shape module set rejects atomically.
    pub(crate) fn try_from_modules(modules: &[&LowRankAdaLn]) -> Option<Self> {
        let first = modules.first()?;
        if modules
            .iter()
            .any(|module| module.has_per_module_inference_cache())
        {
            return None;
        }
        let [model_dim, rank] = first.shift_down.weight.dims();
        if model_dim == 0 || rank == 0 {
            return None;
        }
        let device = first.shift_down.weight.device();
        let dtype = first.shift_down.weight.dtype();
        if !matches!(dtype, DType::F32 | DType::F16) {
            return None;
        }
        let sources = modules
            .iter()
            .map(|module| ModuleSources::from_module(module))
            .collect::<Option<Vec<_>>>()?;
        if sources.is_empty()
            || sources
                .iter()
                .any(|source| !source.has_contract(model_dim, rank, dtype, &device))
        {
            return None;
        }
        let slot_count = sources.len().checked_mul(ADALN_BRANCHES)?;
        let down = sources
            .iter()
            .flat_map(|source| source.down.iter().cloned())
            .collect::<Vec<_>>();
        let up = sources
            .iter()
            .flat_map(|source| source.up.iter().cloned())
            .collect::<Vec<_>>();
        let bias = sources
            .iter()
            .flat_map(|source| source.bias.iter().cloned())
            .collect::<Vec<_>>();
        if down.len() != slot_count || up.len() != slot_count || bias.len() != slot_count {
            return None;
        }

        Some(Self {
            down: Tensor::<2>::stack::<3>(down, 0).unsqueeze_dim::<4>(0),
            up: Tensor::<2>::stack::<3>(up, 0).unsqueeze_dim::<4>(0),
            bias: Tensor::<1>::stack::<2>(bias, 0).reshape([1, slot_count, 1, model_dim]),
            module_count: sources.len(),
            model_dim,
            rank,
            dtype,
            device,
        })
    }

    /// Fill an empty slot exactly once. A complete existing cache is reused.
    pub(crate) fn prepare(slot: &mut Option<Self>, modules: &[&LowRankAdaLn]) -> bool {
        if slot.is_some() {
            return false;
        }
        let Some(cache) = Self::try_from_modules(modules) else {
            return false;
        };
        *slot = Some(cache);
        true
    }

    fn slot_count(&self) -> Option<usize> {
        self.module_count.checked_mul(ADALN_BRANCHES)
    }

    fn has_semantic_contract(&self) -> bool {
        let Some(slot_count) = self.slot_count() else {
            return false;
        };
        self.module_count > 0
            && self.model_dim > 0
            && self.rank > 0
            && self.down.dims() == [1, slot_count, self.model_dim, self.rank]
            && self.up.dims() == [1, slot_count, self.rank, self.model_dim]
            && self.bias.dims() == [1, slot_count, 1, self.model_dim]
            && matches!(self.dtype, DType::F32 | DType::F16)
            && self.down.dtype() == self.dtype
            && self.up.dtype() == self.dtype
            && self.bias.dtype() == self.dtype
            && self.down.device() == self.device
            && self.up.device() == self.device
            && self.bias.device() == self.device
    }

    pub(crate) fn packed_bytes(&self) -> Option<usize> {
        packed_bytes_for(self.module_count, self.model_dim, self.rank, self.dtype)
    }

    fn module_views(&self, module_index: usize) -> Option<ModuleSources> {
        if !self.has_semantic_contract() || module_index >= self.module_count {
            return None;
        }
        let first_slot = module_index.checked_mul(ADALN_BRANCHES)?;
        let down_elements = self.model_dim.checked_mul(self.rank)?;
        let up_elements = self.rank.checked_mul(self.model_dim)?;
        let down_view = |branch: usize| {
            contiguous_wgpu_view(
                &self.down,
                (first_slot + branch).checked_mul(down_elements)?,
                [self.model_dim, self.rank],
            )
        };
        let up_view = |branch: usize| {
            contiguous_wgpu_view(
                &self.up,
                (first_slot + branch).checked_mul(up_elements)?,
                [self.rank, self.model_dim],
            )
        };
        let bias_view = |branch: usize| {
            contiguous_wgpu_view(
                &self.bias,
                (first_slot + branch).checked_mul(self.model_dim)?,
                [self.model_dim],
            )
        };
        let down = [down_view(0)?, down_view(1)?, down_view(2)?];
        let up = [up_view(0)?, up_view(1)?, up_view(2)?];
        let bias = [bias_view(0)?, bias_view(1)?, bias_view(2)?];
        let views = ModuleSources { down, up, bias };
        views
            .has_contract(self.model_dim, self.rank, self.dtype, &self.device)
            .then_some(views)
    }

    pub(crate) fn can_rebind_module_sources(
        &self,
        module_index: usize,
        module: &LowRankAdaLn,
    ) -> bool {
        ModuleSources::from_module(module).is_some_and(|sources| {
            sources.has_contract(self.model_dim, self.rank, self.dtype, &self.device)
                && self.module_views(module_index).is_some()
        })
    }

    /// Rebind one logical module to zero-copy views of the canonical cache.
    ///
    /// The cache remains the fast cross-layer representation. The six Linears
    /// retain a complete portable fallback without retaining duplicate storage.
    pub(crate) fn rebind_module_sources(
        &self,
        module_index: usize,
        module: &mut LowRankAdaLn,
    ) -> bool {
        if ModuleSources::from_module(module).is_none_or(|sources| {
            !sources.has_contract(self.model_dim, self.rank, self.dtype, &self.device)
        }) {
            return false;
        }
        let Some(ModuleSources {
            down: [shift_down, scale_down, gate_down],
            up: [shift_up, scale_up, gate_up],
            bias: [shift_bias, scale_bias, gate_bias],
        }) = self.module_views(module_index)
        else {
            return false;
        };

        module.shift_down.weight = Param::initialized(ParamId::new(), shift_down);
        module.scale_down.weight = Param::initialized(ParamId::new(), scale_down);
        module.gate_down.weight = Param::initialized(ParamId::new(), gate_down);
        module.shift_up.weight = Param::initialized(ParamId::new(), shift_up);
        module.scale_up.weight = Param::initialized(ParamId::new(), scale_up);
        module.gate_up.weight = Param::initialized(ParamId::new(), gate_up);
        module.shift_up.bias = Some(Param::initialized(ParamId::new(), shift_bias));
        module.scale_up.bias = Some(Param::initialized(ParamId::new(), scale_bias));
        module.gate_up.bias = Some(Param::initialized(ParamId::new(), gate_bias));
        true
    }

    /// Generic implementation used by CPU parity tests after semantic checks.
    fn precompute_with_max_batch(
        &self,
        cond_embed: Tensor<3>,
        max_batch: usize,
    ) -> Option<CrossLayerAdaLnModulations> {
        if !self.has_semantic_contract() {
            return None;
        }
        let [batch, sequence, width] = cond_embed.dims();
        let expected_width = self.model_dim.checked_mul(ADALN_BRANCHES)?;
        if batch == 0
            || batch > max_batch
            || sequence != 1
            || width != expected_width
            || cond_embed.dtype() != self.dtype
            || cond_embed.device() != self.device
        {
            return None;
        }
        let slots = self.slot_count()?;
        let raw = cond_embed.reshape([batch, ADALN_BRANCHES, 1, self.model_dim]);
        let activated = silu(raw.clone()).repeat_dim(1, self.module_count);
        let up = activated.matmul(self.down.clone()).matmul(self.up.clone());
        let biased = up + self.bias.clone();
        let biased: Tensor<5> =
            biased.reshape([batch, self.module_count, ADALN_BRANCHES, 1, self.model_dim]);
        let raw: Tensor<5> = raw.reshape([batch, 1, ADALN_BRANCHES, 1, self.model_dim]);
        let values = (biased + raw).reshape([batch, slots, 1, self.model_dim]);
        if values.dims() != [batch, slots, 1, self.model_dim]
            || values.dtype() != self.dtype
            || values.device() != self.device
        {
            return None;
        }
        Some(CrossLayerAdaLnModulations {
            values,
            batch,
            module_count: self.module_count,
            model_dim: self.model_dim,
        })
    }
}

fn contiguous_wgpu_view<const SOURCE_D: usize, const VIEW_D: usize>(
    source: &Tensor<SOURCE_D>,
    element_offset: usize,
    shape: [usize; VIEW_D],
) -> Option<Tensor<VIEW_D>> {
    let primitive = source.clone().try_into_primitive::<crate::WgpuRaw>().ok()?;
    if !primitive.is_contiguous() || primitive.qparams.is_some() {
        return None;
    }
    let element_bytes = primitive.dtype.size();
    let offset_bytes = element_offset.checked_mul(element_bytes)?;
    let view_bytes = shape
        .into_iter()
        .try_fold(element_bytes, |bytes, dim| bytes.checked_mul(dim))?;
    let available_bytes = usize::try_from(primitive.handle.size_in_used()).ok()?;
    available_bytes.checked_sub(offset_bytes.checked_add(view_bytes)?)?;
    let alignment = usize::try_from(primitive.client.properties().memory.alignment).ok()?;
    if alignment == 0 || !offset_bytes.is_multiple_of(alignment) {
        return None;
    }
    let handle = primitive
        .handle
        .clone()
        .offset_start(u64::try_from(offset_bytes).ok()?);
    let view = CubeTensor::new_contiguous(
        primitive.client,
        primitive.device,
        Shape::new(shape),
        handle,
        primitive.dtype,
    );
    Some(Tensor::from_primitive::<crate::WgpuRaw>(view))
}

fn packed_bytes_for(
    module_count: usize,
    model_dim: usize,
    rank: usize,
    dtype: DType,
) -> Option<usize> {
    let slots = module_count.checked_mul(ADALN_BRANCHES)?;
    let elements = model_dim
        .checked_mul(rank)?
        .checked_add(rank.checked_mul(model_dim)?)?
        .checked_add(model_dim)?
        .checked_mul(slots)?;
    elements.checked_mul(match dtype {
        DType::F32 => core::mem::size_of::<f32>(),
        DType::F16 => core::mem::size_of::<half::f16>(),
        _ => return None,
    })
}

impl CrossLayerAdaLnModulations {
    fn module(&self, module_index: usize) -> Option<AdaLnModulation> {
        let slots = self.module_count.checked_mul(ADALN_BRANCHES)?;
        if self.batch == 0
            || self.model_dim == 0
            || self.values.dims() != [self.batch, slots, 1, self.model_dim]
            || module_index >= self.module_count
        {
            return None;
        }
        let start = module_index.checked_mul(ADALN_BRANCHES)?;
        let module = self.values.clone().narrow(1, start, ADALN_BRANCHES);
        let branch = |index| {
            module
                .clone()
                .narrow(1, index, 1)
                .reshape([self.batch, 1, self.model_dim])
        };
        Some(AdaLnModulation {
            shift: branch(0),
            scale: branch(1),
            gate: branch(2),
        })
    }

    pub(crate) fn block(&self, block_index: usize) -> Option<BlockAdaLnModulations> {
        let attention_index = block_index.checked_mul(2)?;
        let mlp_index = attention_index.checked_add(1)?;
        Some(BlockAdaLnModulations {
            attention: self.module(attention_index)?,
            mlp: self.module(mlp_index)?,
        })
    }

    pub(crate) fn pack_schedule(values: Vec<Self>) -> Option<CrossLayerAdaLnBatchSchedule> {
        let first = values.first()?;
        let batch = first.batch;
        let module_count = first.module_count;
        let model_dim = first.model_dim;
        let slots = module_count.checked_mul(ADALN_BRANCHES)?;
        if batch == 0
            || model_dim == 0
            || values.iter().any(|value| {
                value.batch != batch
                    || value.module_count != module_count
                    || value.model_dim != model_dim
                    || value.values.dims() != [batch, slots, 1, model_dim]
            })
        {
            return None;
        }
        let step_count = values.len();
        let values =
            Tensor::<4>::stack::<5>(values.into_iter().map(|value| value.values).collect(), 0);
        (values.dims() == [step_count, batch, slots, 1, model_dim]).then_some(
            CrossLayerAdaLnBatchSchedule {
                values,
                step_count,
                batch,
                module_count,
                model_dim,
            },
        )
    }
}

impl CrossLayerAdaLnBatchSchedule {
    pub(crate) fn step(&self, index: usize) -> Option<CrossLayerAdaLnModulations> {
        let slots = self.module_count.checked_mul(ADALN_BRANCHES)?;
        if index >= self.step_count
            || self.values.dims() != [self.step_count, self.batch, slots, 1, self.model_dim]
        {
            return None;
        }
        Some(CrossLayerAdaLnModulations {
            values: self.values.clone().narrow(0, index, 1).reshape([
                self.batch,
                slots,
                1,
                self.model_dim,
            ]),
            batch: self.batch,
            module_count: self.module_count,
            model_dim: self.model_dim,
        })
    }

    pub(crate) const fn step_count(&self) -> usize {
        self.step_count
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

impl CrossLayerAdaLnCache {
    /// Prepare the released v4 cache once; unsupported models stay uncached.
    pub(crate) fn prepare_v4_wgsl(slot: &mut Option<Self>, modules: &[&LowRankAdaLn]) -> bool {
        if slot.is_some() || modules.len() != V4_ADALN_MODULES {
            return false;
        }
        let Some(first) = modules.first() else {
            return false;
        };
        if first.shift_down.weight.dims() != [V4_ADALN_MODEL_DIM, V4_ADALN_RANK]
            || !Self::prepare(slot, modules)
        {
            return false;
        }
        if slot.as_ref().is_some_and(Self::has_exact_v4_contract) {
            true
        } else {
            *slot = None;
            false
        }
    }

    fn has_exact_v4_contract(&self) -> bool {
        self.module_count == V4_ADALN_MODULES
            && self.model_dim == V4_ADALN_MODEL_DIM
            && self.rank == V4_ADALN_RANK
            && self.packed_bytes()
                == match self.dtype {
                    DType::F32 => Some(141_926_400),
                    DType::F16 => Some(70_963_200),
                    _ => None,
                }
            && self.has_semantic_contract()
            && wgpu_tensor_has_layout(
                &self.down,
                [
                    1,
                    V4_ADALN_MODULES * ADALN_BRANCHES,
                    V4_ADALN_MODEL_DIM,
                    V4_ADALN_RANK,
                ],
                self.dtype,
                &self.device,
            )
            && wgpu_tensor_has_layout(
                &self.up,
                [
                    1,
                    V4_ADALN_MODULES * ADALN_BRANCHES,
                    V4_ADALN_RANK,
                    V4_ADALN_MODEL_DIM,
                ],
                self.dtype,
                &self.device,
            )
            && wgpu_tensor_has_layout(
                &self.bias,
                [1, V4_ADALN_MODULES * ADALN_BRANCHES, 1, V4_ADALN_MODEL_DIM],
                self.dtype,
                &self.device,
            )
    }

    pub(crate) fn supports_profile_lock(&self) -> bool {
        self.has_exact_v4_contract()
    }

    /// Return all v4 modulations, or reject the entire fast path before use.
    pub(crate) fn precompute_v4_wgsl(
        &self,
        cond_embed: Tensor<3>,
    ) -> Option<CrossLayerAdaLnModulations> {
        if !self.has_exact_v4_contract() {
            return None;
        }
        let [batch, _, _] = cond_embed.dims();
        if !wgpu_tensor_has_layout(
            &cond_embed,
            [batch, 1, V4_ADALN_MODEL_DIM * ADALN_BRANCHES],
            self.dtype,
            &self.device,
        ) {
            return None;
        }
        let modulations = self.precompute_with_max_batch(cond_embed, V4_MAX_BATCH)?;
        if !wgpu_tensor_has_layout(
            &modulations.values,
            [
                batch,
                V4_ADALN_MODULES * ADALN_BRANCHES,
                1,
                V4_ADALN_MODEL_DIM,
            ],
            self.dtype,
            &self.device,
        ) {
            return None;
        }
        Some(modulations)
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, io};

    use super::*;

    fn set_linear_weight(
        linear: &mut burn::nn::Linear,
        shape: [usize; 2],
        value: f32,
        device: &Device,
    ) {
        linear.weight = Param::initialized(
            ParamId::new(),
            Tensor::ones(shape, device).mul_scalar(value),
        );
    }

    fn set_linear_bias(linear: &mut burn::nn::Linear, width: usize, value: f32, device: &Device) {
        linear.bias = Some(Param::initialized(
            ParamId::new(),
            Tensor::ones([width], device).mul_scalar(value),
        ));
    }

    fn module(module_index: usize, model_dim: usize, rank: usize, device: &Device) -> LowRankAdaLn {
        let mut module = LowRankAdaLn::new(model_dim, rank, 1.0e-6, device);
        for (branch, linear) in [
            &mut module.shift_down,
            &mut module.scale_down,
            &mut module.gate_down,
        ]
        .into_iter()
        .enumerate()
        {
            set_linear_weight(
                linear,
                [model_dim, rank],
                0.001 * (1 + module_index * ADALN_BRANCHES + branch) as f32,
                device,
            );
        }
        for (branch, linear) in [
            &mut module.shift_up,
            &mut module.scale_up,
            &mut module.gate_up,
        ]
        .into_iter()
        .enumerate()
        {
            set_linear_weight(
                linear,
                [rank, model_dim],
                0.01 * (1 + module_index * ADALN_BRANCHES + branch) as f32,
                device,
            );
            set_linear_bias(
                linear,
                model_dim,
                0.1 * (1 + module_index * ADALN_BRANCHES + branch) as f32,
                device,
            );
        }
        module
    }

    fn require<T>(value: Option<T>, message: &str) -> Result<T, Box<dyn Error>> {
        value.ok_or_else(|| io::Error::other(message).into())
    }

    fn managed_memory_id<const D: usize>(tensor: &Tensor<D>) -> String {
        let primitive = tensor
            .clone()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("test tensor must use WGPU raw backend");
        let binding = format!("{:?}", primitive.handle.binding().memory);
        binding
            .split(", location:")
            .next()
            .expect("managed memory debug form carries an id")
            .to_owned()
    }

    #[test]
    fn pack_is_module_major_attention_then_mlp() -> Result<(), Box<dyn Error>> {
        let device = Default::default();
        let modules = (0..4)
            .map(|index| module(index, 4, 2, &device))
            .collect::<Vec<_>>();
        let references = modules.iter().collect::<Vec<_>>();
        let cache = require(
            CrossLayerAdaLnCache::try_from_modules(&references),
            "valid modules should pack",
        )?;

        let bias = cache
            .bias
            .clone()
            .reshape([4 * ADALN_BRANCHES * 4])
            .into_data()
            .to_vec::<f32>()?;
        let expected = (0..4)
            .flat_map(|module_index| {
                (0..ADALN_BRANCHES).flat_map(move |branch| {
                    std::iter::repeat_n(
                        0.1 * (1 + module_index * ADALN_BRANCHES + branch) as f32,
                        4,
                    )
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(bias, expected);
        Ok(())
    }

    #[test]
    fn b1_b2_b3_slices_match_original_modules() -> Result<(), Box<dyn Error>> {
        let device = Default::default();
        let modules = (0..4)
            .map(|index| module(index, 8, 4, &device))
            .collect::<Vec<_>>();
        let references = modules.iter().collect::<Vec<_>>();
        let cache = require(
            CrossLayerAdaLnCache::try_from_modules(&references),
            "valid modules should pack",
        )?;

        for batch in [1, 2, 3] {
            let cond = Tensor::<3>::ones([batch, 1, 24], &device).mul_scalar(0.25);
            let all = require(
                cache.precompute_with_max_batch(cond.clone(), 3),
                "B1/B2/B3 should use cross-layer precompute",
            )?;
            for (block_index, pair) in modules.chunks_exact(2).enumerate() {
                let slices = require(all.block(block_index), "block slice should exist")?;
                for (expected_module, actual) in
                    [(&pair[0], slices.attention), (&pair[1], slices.mlp)]
                {
                    let expected = expected_module.modulation(cond.clone());
                    for (expected, actual) in [
                        (expected.0, actual.shift),
                        (expected.1, actual.scale),
                        (expected.2, actual.gate),
                    ] {
                        let max_abs = (expected - actual).abs().max().into_scalar::<f32>();
                        assert!(max_abs < 1.0e-6, "B={batch} max_abs={max_abs}");
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn canonical_views_preserve_portable_module_fallback() -> Result<(), Box<dyn Error>> {
        let device = Default::default();
        let mut modules = (0..4)
            .map(|index| module(index, 64, 4, &device))
            .collect::<Vec<_>>();
        let cond = Tensor::<3>::ones([3, 1, 192], &device).mul_scalar(0.25);
        let expected = modules
            .iter()
            .map(|module| module.modulation(cond.clone()))
            .collect::<Vec<_>>();
        let references = modules.iter().collect::<Vec<_>>();
        let cache = require(
            CrossLayerAdaLnCache::try_from_modules(&references),
            "valid modules should pack",
        )?;
        drop(references);

        for (module_index, module) in modules.iter_mut().enumerate() {
            assert!(cache.rebind_module_sources(module_index, module));
        }
        let down_memory = managed_memory_id(&cache.down);
        let up_memory = managed_memory_id(&cache.up);
        let bias_memory = managed_memory_id(&cache.bias);
        for (module, expected) in modules.iter().zip(expected) {
            for tensor in [
                module.shift_down.weight.val(),
                module.scale_down.weight.val(),
                module.gate_down.weight.val(),
            ] {
                assert_eq!(managed_memory_id(&tensor), down_memory);
            }
            for tensor in [
                module.shift_up.weight.val(),
                module.scale_up.weight.val(),
                module.gate_up.weight.val(),
            ] {
                assert_eq!(managed_memory_id(&tensor), up_memory);
            }
            for tensor in [
                module.shift_up.bias.as_ref().expect("shift bias").val(),
                module.scale_up.bias.as_ref().expect("scale bias").val(),
                module.gate_up.bias.as_ref().expect("gate bias").val(),
            ] {
                assert_eq!(managed_memory_id(&tensor), bias_memory);
            }
            let actual = module.modulation(cond.clone());
            for (expected, actual) in [
                (expected.0, actual.0),
                (expected.1, actual.1),
                (expected.2, actual.2),
            ] {
                let max_abs = (expected - actual).abs().max().into_scalar::<f32>();
                assert!(max_abs < 1.0e-6, "canonical view max_abs={max_abs}");
            }
        }
        Ok(())
    }

    #[test]
    fn schedule_pack_preserves_equal_batch_modulations() -> Result<(), Box<dyn Error>> {
        let device = Default::default();
        let modules = (0..4)
            .map(|index| module(index, 8, 4, &device))
            .collect::<Vec<_>>();
        let references = modules.iter().collect::<Vec<_>>();
        let cache = require(
            CrossLayerAdaLnCache::try_from_modules(&references),
            "valid modules should pack",
        )?;
        let mut expected = Vec::new();
        let mut values = Vec::new();
        for step in 0..3 {
            let cond = Tensor::<3>::ones([2, 1, 24], &device).mul_scalar(0.125 * (step + 1) as f32);
            let value = require(
                cache.precompute_with_max_batch(cond, 3),
                "schedule row should precompute",
            )?;
            expected.push(value.values.clone().into_data().to_vec::<f32>()?);
            values.push(value);
        }
        let schedule = require(
            CrossLayerAdaLnModulations::pack_schedule(values),
            "equal-batch rows should pack",
        )?;
        assert_eq!(schedule.step_count(), 3);
        for (step, expected) in expected.iter().enumerate() {
            let actual = require(schedule.step(step), "packed step should exist")?
                .values
                .into_data()
                .to_vec::<f32>()?;
            assert_eq!(&actual, expected);
        }
        assert!(schedule.step(3).is_none());
        Ok(())
    }

    #[test]
    fn prepare_is_idempotent_and_missing_bias_is_rejected() {
        let device = Default::default();
        let modules = (0..2)
            .map(|index| module(index, 8, 4, &device))
            .collect::<Vec<_>>();
        let references = modules.iter().collect::<Vec<_>>();
        let mut slot = None;
        assert!(CrossLayerAdaLnCache::prepare(&mut slot, &references));
        assert!(!CrossLayerAdaLnCache::prepare(&mut slot, &references));

        let mut missing = modules;
        missing[1].gate_up.bias = None;
        let references = missing.iter().collect::<Vec<_>>();
        assert!(CrossLayerAdaLnCache::try_from_modules(&references).is_none());
    }

    #[test]
    fn stale_and_b4_contracts_fail_closed() -> Result<(), Box<dyn Error>> {
        let device = Default::default();
        let modules = (0..2)
            .map(|index| module(index, 8, 4, &device))
            .collect::<Vec<_>>();
        let references = modules.iter().collect::<Vec<_>>();
        let mut cache = require(
            CrossLayerAdaLnCache::try_from_modules(&references),
            "valid modules should pack",
        )?;
        let b4 = Tensor::<3>::zeros([4, 1, 24], &device);
        assert!(cache.precompute_with_max_batch(b4, 3).is_none());

        cache.module_count += 1;
        let b1 = Tensor::<3>::zeros([1, 1, 24], &device);
        assert!(cache.precompute_with_max_batch(b1, 3).is_none());
        Ok(())
    }

    #[test]
    fn missing_and_stale_slices_fall_back_without_mutating_normal_path()
    -> Result<(), Box<dyn Error>> {
        let device = Default::default();
        let modules = (0..2)
            .map(|index| module(index, 8, 4, &device))
            .collect::<Vec<_>>();
        let cond = Tensor::<3>::ones([2, 1, 24], &device).mul_scalar(0.25);
        let expected = modules[0].modulation(cond.clone());

        let missing = modules[0].resolve_modulation(cond.clone(), None);
        let stale = AdaLnModulation {
            shift: Tensor::zeros([1, 1, 8], &device),
            scale: Tensor::zeros([1, 1, 8], &device),
            gate: Tensor::zeros([1, 1, 8], &device),
        };
        let rejected = modules[0].resolve_modulation(cond.clone(), Some(stale));
        for (expected, missing, rejected) in [
            (expected.0, missing.shift, rejected.shift),
            (expected.1, missing.scale, rejected.scale),
            (expected.2, missing.gate, rejected.gate),
        ] {
            assert_eq!(
                (expected.clone() - missing)
                    .abs()
                    .max()
                    .into_scalar::<f32>(),
                0.0
            );
            assert_eq!((expected - rejected).abs().max().into_scalar::<f32>(), 0.0);
        }

        let references = modules.iter().collect::<Vec<_>>();
        let mut cache = None;
        assert!(CrossLayerAdaLnCache::prepare(&mut cache, &references));
        let x = Tensor::<3>::ones([2, 3, 8], &device);
        let before = modules[0].forward(x.clone(), cond.clone());
        let after = modules[0].forward(x, cond);
        assert_eq!((before.0 - after.0).abs().max().into_scalar::<f32>(), 0.0);
        assert_eq!((before.1 - after.1).abs().max().into_scalar::<f32>(), 0.0);
        Ok(())
    }

    #[test]
    fn exact_v4_capacity_is_135_mib() {
        assert_eq!(
            packed_bytes_for(
                V4_ADALN_MODULES,
                V4_ADALN_MODEL_DIM,
                V4_ADALN_RANK,
                DType::F32,
            ),
            Some(141_926_400)
        );
        assert_eq!(
            packed_bytes_for(
                V4_ADALN_MODULES,
                V4_ADALN_MODEL_DIM,
                V4_ADALN_RANK,
                DType::F16,
            ),
            Some(70_963_200)
        );
    }
}
