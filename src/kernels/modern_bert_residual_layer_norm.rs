//! Exact-shape residual-add + LayerNorm for the v4-Small ModernBERT encoder.
//!
//! The production selector only admits contiguous f32 `[1, 3, 768]` residual
//! and branch tensors plus contiguous f32 `[768]` gamma on one WGPU device.
//! Unsupported shapes, layouts, dtypes, devices, or hardware limits fail
//! closed before allocation or dispatch so callers can use the Burn graph.

use std::{error::Error, fmt};

use burn::{
    backend::wgpu::{CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime},
    tensor::{DType, Shape},
};
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

pub(crate) const BATCH: usize = 1;
pub(crate) const SEQUENCE: usize = 3;
pub(crate) const WIDTH: usize = 768;
pub(crate) const ROWS: usize = BATCH * SEQUENCE;
pub(crate) const ELEMENTS: usize = ROWS * WIDTH;
pub(crate) const V4_BOUNDARIES: usize = 50;

const EPSILON: f64 = 1.0e-5;
const WORKGROUP_SIZE: u32 = 256;
const REQUIRED_BINDINGS: u32 = 5;
const SHARED_BYTES: usize = WORKGROUP_SIZE as usize * size_of::<f32>();
const F32_BYTES: usize = size_of::<f32>();

/// Physical-tensor traffic lower bounds used by the accepted isolated A/B.
#[cfg(test)]
pub(crate) const CURRENT_BYTES_PER_BOUNDARY: usize = 141_444;
#[cfg(test)]
pub(crate) const FUSED_BYTES_PER_BOUNDARY: usize = 39_936;
#[cfg(test)]
pub(crate) const CURRENT_DISPATCHES_PER_BOUNDARY: usize = 11;
#[cfg(test)]
pub(crate) const FUSED_DISPATCHES_PER_BOUNDARY: usize = 1;

#[derive(Debug)]
pub(crate) struct ModernBertResidualLayerNormError {
    message: String,
}

impl ModernBertResidualLayerNormError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ModernBertResidualLayerNormError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for ModernBertResidualLayerNormError {}

#[derive(Clone, Copy, Debug)]
struct HardwareContract {
    max_page_size: u64,
    max_bindings: u32,
    max_shared_memory_size: usize,
    max_units_per_cube: u32,
    max_cube_dim: (u32, u32, u32),
    max_cube_count: (u32, u32, u32),
}

impl HardwareContract {
    fn supports_exact_launch(self) -> bool {
        let Some(output_bytes) = ELEMENTS
            .checked_mul(F32_BYTES)
            .and_then(|bytes| u64::try_from(bytes).ok())
        else {
            return false;
        };
        let Ok(rows) = u32::try_from(ROWS) else {
            return false;
        };
        self.max_page_size >= output_bytes
            && self.max_bindings >= REQUIRED_BINDINGS
            && self.max_shared_memory_size >= SHARED_BYTES
            && self.max_units_per_cube >= WORKGROUP_SIZE
            && self.max_cube_dim.0 >= WORKGROUP_SIZE
            && self.max_cube_dim.1 >= 1
            && self.max_cube_dim.2 >= 1
            && self.max_cube_count.0 >= rows
            && self.max_cube_count.1 >= 1
            && self.max_cube_count.2 >= 1
    }
}

#[derive(Debug)]
struct ModernBertResidualLayerNormKernel;

impl KernelSource for ModernBertResidualLayerNormKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("modern_bert_residual_layer_norm.wgsl"))
            .register("width", WIDTH.to_string())
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
            .register("epsilon", format!("{EPSILON:e}"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

fn hardware_contract(reference: &CubeTensor<WgpuRuntime>) -> HardwareContract {
    let properties = reference.client.properties();
    let hardware = &properties.hardware;
    HardwareContract {
        max_page_size: properties.memory.max_page_size,
        max_bindings: hardware.max_bindings,
        max_shared_memory_size: hardware.max_shared_memory_size,
        max_units_per_cube: hardware.max_units_per_cube,
        max_cube_dim: hardware.max_cube_dim,
        max_cube_count: hardware.max_cube_count,
    }
}

fn has_rank3_layout(tensor: &CubeTensor<WgpuRuntime>) -> bool {
    tensor.dtype == DType::F32
        && tensor.meta.num_dims() == 3
        && tensor.meta.shape().dims::<3>() == [BATCH, SEQUENCE, WIDTH]
        && tensor.is_contiguous()
        && &tensor.meta.strides()[..] == [ELEMENTS, WIDTH, 1].as_slice()
}

fn has_gamma_layout(gamma: &CubeTensor<WgpuRuntime>) -> bool {
    gamma.dtype == DType::F32
        && gamma.meta.num_dims() == 1
        && gamma.meta.shape().dims::<1>() == [WIDTH]
        && gamma.is_contiguous()
        && &gamma.meta.strides()[..] == [1].as_slice()
}

/// Allocation-free preflight for one exact v4 LayerNorm gamma and its device.
///
/// The full launcher repeats all checks for residual, branch, and gamma. This
/// narrower predicate lets the ModernBERT frontend reject an unsupported WGPU
/// before doing any encoder work and route to the existing generic forward.
pub(crate) fn supports_modern_bert_residual_layer_norm_device(
    gamma: &CubeTensor<WgpuRuntime>,
) -> bool {
    has_gamma_layout(gamma) && hardware_contract(gamma).supports_exact_launch()
}

fn validate_contract(
    residual: &CubeTensor<WgpuRuntime>,
    branch: &CubeTensor<WgpuRuntime>,
    gamma: &CubeTensor<WgpuRuntime>,
) -> Result<usize, ModernBertResidualLayerNormError> {
    if !has_rank3_layout(residual) {
        return Err(ModernBertResidualLayerNormError::new(format!(
            "residual must be contiguous f32 [1,3,768] with strides [{ELEMENTS},{WIDTH},1], got dtype={:?} shape={:?} strides={:?}",
            residual.dtype,
            residual.meta.shape(),
            residual.meta.strides()
        )));
    }
    if !has_rank3_layout(branch) {
        return Err(ModernBertResidualLayerNormError::new(format!(
            "branch must be contiguous f32 [1,3,768] with strides [{ELEMENTS},{WIDTH},1], got dtype={:?} shape={:?} strides={:?}",
            branch.dtype,
            branch.meta.shape(),
            branch.meta.strides()
        )));
    }
    if !has_gamma_layout(gamma) {
        return Err(ModernBertResidualLayerNormError::new(format!(
            "gamma must be contiguous f32 [768] with stride [1], got dtype={:?} shape={:?} strides={:?}",
            gamma.dtype,
            gamma.meta.shape(),
            gamma.meta.strides()
        )));
    }
    if branch.device != residual.device || gamma.device != residual.device {
        return Err(ModernBertResidualLayerNormError::new(
            "residual, branch, and gamma must use the same WGPU device",
        ));
    }
    if !hardware_contract(residual).supports_exact_launch() {
        return Err(ModernBertResidualLayerNormError::new(
            "WGPU device does not satisfy the exact ModernBERT kernel resource contract",
        ));
    }

    ELEMENTS
        .checked_mul(F32_BYTES)
        .ok_or_else(|| ModernBertResidualLayerNormError::new("output byte count overflow"))
}

/// Emit `(updated_residual, normalized)` for one exact v4 encoder boundary.
///
/// The updated residual is computed and written before normalization. The
/// normalization uses biased variance, epsilon `1e-5`, gamma, and no beta,
/// matching Burn's v4 LayerNorm graph.
pub(crate) fn modern_bert_residual_layer_norm_wgsl(
    residual: CubeTensor<WgpuRuntime>,
    branch: CubeTensor<WgpuRuntime>,
    gamma: CubeTensor<WgpuRuntime>,
) -> Result<(CubeTensor<WgpuRuntime>, CubeTensor<WgpuRuntime>), ModernBertResidualLayerNormError> {
    let output_bytes = validate_contract(&residual, &branch, &gamma)?;
    let client = residual.client.clone();
    let device = residual.device.clone();
    let shape = Shape::from([BATCH, SEQUENCE, WIDTH]);
    let updated = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        shape.clone(),
        client.empty(output_bytes),
        DType::F32,
    );
    let normalized = CubeTensor::new_contiguous(
        client.clone(),
        device,
        shape,
        client.empty(output_bytes),
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            ModernBertResidualLayerNormKernel,
            CubeDim::new_1d(WORKGROUP_SIZE),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(residual.handle.binding())
        .with_buffer(branch.handle.binding())
        .with_buffer(gamma.handle.binding())
        .with_buffer(updated.handle.clone().binding())
        .with_buffer(normalized.handle.clone().binding());
    client.launch(task, CubeCount::new_1d(ROWS as u32), bindings);
    Ok((updated, normalized))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn supported_hardware() -> HardwareContract {
        HardwareContract {
            max_page_size: (ELEMENTS * F32_BYTES) as u64,
            max_bindings: REQUIRED_BINDINGS,
            max_shared_memory_size: SHARED_BYTES,
            max_units_per_cube: WORKGROUP_SIZE,
            max_cube_dim: (WORKGROUP_SIZE, 1, 1),
            max_cube_count: (ROWS as u32, 1, 1),
        }
    }

    #[test]
    fn exact_hardware_contract_fails_closed() {
        let supported = supported_hardware();
        assert!(supported.supports_exact_launch());
        assert!(
            !HardwareContract {
                max_bindings: REQUIRED_BINDINGS - 1,
                ..supported
            }
            .supports_exact_launch()
        );
        assert!(
            !HardwareContract {
                max_page_size: (ELEMENTS * F32_BYTES - 1) as u64,
                ..supported
            }
            .supports_exact_launch()
        );
        assert!(
            !HardwareContract {
                max_shared_memory_size: SHARED_BYTES - 1,
                ..supported
            }
            .supports_exact_launch()
        );
        assert!(
            !HardwareContract {
                max_units_per_cube: WORKGROUP_SIZE - 1,
                ..supported
            }
            .supports_exact_launch()
        );
        assert!(
            !HardwareContract {
                max_cube_dim: (WORKGROUP_SIZE - 1, 1, 1),
                ..supported
            }
            .supports_exact_launch()
        );
        assert!(
            !HardwareContract {
                max_cube_count: (ROWS as u32 - 1, 1, 1),
                ..supported
            }
            .supports_exact_launch()
        );
    }

    #[test]
    fn single_boundary_static_contract_is_exact() {
        assert_eq!([BATCH, SEQUENCE, WIDTH], [1, 3, 768]);
        assert_eq!(ROWS, 3);
        assert_eq!(ELEMENTS, 2_304);
        assert_eq!(WORKGROUP_SIZE, 256);
        assert_eq!(SHARED_BYTES, 1_024);
        assert_eq!(REQUIRED_BINDINGS, 5);
        assert_eq!(CURRENT_BYTES_PER_BOUNDARY, 141_444);
        assert_eq!(FUSED_BYTES_PER_BOUNDARY, 39_936);
        assert_eq!(CURRENT_DISPATCHES_PER_BOUNDARY, 11);
        assert_eq!(FUSED_DISPATCHES_PER_BOUNDARY, 1);
    }

    #[test]
    fn fifty_boundary_static_contract_is_exact() {
        assert_eq!(V4_BOUNDARIES, 25 + 24 + 1);
        assert_eq!(CURRENT_DISPATCHES_PER_BOUNDARY * V4_BOUNDARIES, 550);
        assert_eq!(FUSED_DISPATCHES_PER_BOUNDARY * V4_BOUNDARIES, 50);
        assert_eq!(
            (CURRENT_BYTES_PER_BOUNDARY - FUSED_BYTES_PER_BOUNDARY) * V4_BOUNDARIES,
            5_075_400
        );
    }

    #[test]
    fn shader_preserves_biased_no_beta_two_output_contract() {
        let shader = include_str!("modern_bert_residual_layer_norm.wgsl");
        assert_eq!(shader.matches("var<storage, read_write>").count(), 5);
        assert!(shader.contains("updated_residual[index_0] = value_0;"));
        assert!(shader.contains("let variance = partial[0] / f32(WIDTH);"));
        assert!(shader.contains("sqrt(variance + EPSILON)"));
        assert!(!shader.contains("beta["));
    }
}
