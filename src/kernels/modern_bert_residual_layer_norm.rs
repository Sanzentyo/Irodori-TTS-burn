//! Residual-add + LayerNorm for the v4-Small ModernBERT encoder.
//!
//! Contiguous F32/F16 `[batch, sequence, 768]` residual and branch tensors
//! share a dtype and WGPU device with the contiguous `[768]` gamma.
//! Unsupported shapes, layouts, dtypes, devices, or hardware limits fail
//! closed before allocation or dispatch so callers can use the Burn graph.

use std::{error::Error, fmt};

use burn::{
    backend::wgpu::{CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate},
    tensor::{Shape, Tensor},
};
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

use super::precision::{KernelFloatPrecision, common_float_precision};

#[cfg(test)]
pub(crate) const BATCH: usize = 1;
#[cfg(test)]
pub(crate) const SEQUENCE: usize = 3;
pub(crate) const WIDTH: usize = 768;
#[cfg(test)]
pub(crate) const ROWS: usize = BATCH * SEQUENCE;
#[cfg(test)]
pub(crate) const ELEMENTS: usize = ROWS * WIDTH;
pub(crate) const V4_BOUNDARIES: usize = 50;

const EPSILON: f64 = 1.0e-5;
const WORKGROUP_SIZE: u32 = 256;
const REQUIRED_BINDINGS: u32 = 5;
const SHARED_BYTES: usize = WORKGROUP_SIZE as usize * size_of::<f32>();
#[cfg(test)]
const F32_BYTES: usize = size_of::<f32>();

/// Preserve the reduction selected before the frontend's WGSL route expanded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReductionOrder {
    /// The original text-only route already used the fixed three-token tree.
    LegacyShortText,
    /// Auxiliary-conditioned requests previously used Burn at every length.
    Backend,
}

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
    fn supports_launch(self, rows: usize) -> bool {
        let Some(output_bytes) = rows.checked_mul(WIDTH).and_then(|n| {
            n.checked_mul(size_of::<f32>())
                .and_then(|bytes| u64::try_from(bytes).ok())
        }) else {
            return false;
        };
        let Ok(rows) = u32::try_from(rows) else {
            return false;
        };
        let Some(elements) = rows.checked_mul(WIDTH as u32) else {
            return false;
        };
        // The epilogue dispatches one workgroup per 256 elements, rather than
        // one per row. Validate its complete 2D grid before selecting the route.
        let groups = elements.div_ceil(WORKGROUP_SIZE);
        let groups_x = groups.min(self.max_cube_count.0);
        if groups_x == 0 {
            return false;
        }
        let groups_y = groups.div_ceil(groups_x);
        rows > 0
            && groups_x
                .checked_mul(groups_y)
                .and_then(|n| n.checked_mul(WORKGROUP_SIZE))
                .is_some()
            && self.max_page_size >= output_bytes
            && self.max_bindings >= REQUIRED_BINDINGS
            && self.max_shared_memory_size >= SHARED_BYTES * 2
            && self.max_units_per_cube >= WORKGROUP_SIZE
            && self.max_cube_dim.0 >= WORKGROUP_SIZE
            && self.max_cube_dim.1 >= 1
            && self.max_cube_dim.2 >= 1
            && self.max_cube_count.0 >= rows
            && self.max_cube_count.1 >= groups_y
            && self.max_cube_count.2 >= 1
    }
}

#[derive(Debug)]
struct ModernBertResidualLayerNormKernel {
    precision: KernelFloatPrecision,
}

impl KernelSource for ModernBertResidualLayerNormKernel {
    fn lang(&self) -> &'static str {
        "wgsl"
    }

    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("modern_bert_residual_layer_norm.wgsl"),
                include_str!("modern_bert_residual_layer_norm_f16.wgsl"),
            )
            .register("width", WIDTH.to_string())
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
            .register("epsilon", format!("{EPSILON:e}"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.precision)
    }
}

fn hardware_contract(reference: &CubeTensor) -> HardwareContract {
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

fn has_rank3_layout(tensor: &CubeTensor) -> bool {
    KernelFloatPrecision::from_dtype(tensor.dtype).is_some()
        && tensor.meta.num_dims() == 3
        && tensor.meta.shape()[0] > 0
        && tensor.meta.shape()[1] > 0
        && tensor.meta.shape()[2] == WIDTH
        && tensor.is_contiguous()
}

fn has_gamma_layout(gamma: &CubeTensor) -> bool {
    KernelFloatPrecision::from_dtype(gamma.dtype).is_some()
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
    gamma: &CubeTensor,
    rows: usize,
) -> bool {
    has_gamma_layout(gamma) && hardware_contract(gamma).supports_launch(rows)
}

fn validate_contract(
    residual: &CubeTensor,
    branch: &CubeTensor,
    gamma: &CubeTensor,
) -> Result<(KernelFloatPrecision, usize), ModernBertResidualLayerNormError> {
    let precision = common_float_precision([residual.dtype, branch.dtype, gamma.dtype])
        .ok_or_else(|| {
            ModernBertResidualLayerNormError::new(
                "residual, branch, and gamma must share f32 or f16 dtype",
            )
        })?;
    if !has_rank3_layout(residual) {
        return Err(ModernBertResidualLayerNormError::new(format!(
            "residual must be contiguous [batch>0,sequence>0,768], got dtype={:?} shape={:?} strides={:?}",
            residual.dtype,
            residual.meta.shape(),
            residual.meta.strides()
        )));
    }
    if !has_rank3_layout(branch) || branch.meta.shape() != residual.meta.shape() {
        return Err(ModernBertResidualLayerNormError::new(format!(
            "branch must match the contiguous residual shape, got dtype={:?} shape={:?} strides={:?}",
            branch.dtype,
            branch.meta.shape(),
            branch.meta.strides()
        )));
    }
    if !has_gamma_layout(gamma) {
        return Err(ModernBertResidualLayerNormError::new(format!(
            "gamma must be contiguous [768] with stride [1], got dtype={:?} shape={:?} strides={:?}",
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
    if !hardware_contract(residual).supports_launch(residual.meta.num_elements() / WIDTH) {
        return Err(ModernBertResidualLayerNormError::new(
            "WGPU device does not satisfy the exact ModernBERT kernel resource contract",
        ));
    }

    let output_bytes = residual
        .meta
        .num_elements()
        .checked_mul(precision.element_bytes())
        .ok_or_else(|| ModernBertResidualLayerNormError::new("output byte count overflow"))?;
    if [&residual.handle, &branch.handle]
        .iter()
        .any(|h| h.size_in_used() < output_bytes as u64)
        || gamma.handle.size_in_used() < (WIDTH * precision.element_bytes()) as u64
    {
        return Err(ModernBertResidualLayerNormError::new(
            "normalization buffer is smaller than its metadata",
        ));
    }
    Ok((precision, output_bytes))
}

/// Emit `(updated_residual, normalized)` for one exact v4 encoder boundary.
///
/// The updated residual is computed and written before normalization. The
/// normalization uses biased variance, epsilon `1e-5`, gamma, and no beta,
/// matching Burn's v4 LayerNorm graph.
pub(crate) fn modern_bert_residual_layer_norm_wgsl(
    residual: CubeTensor,
    branch: CubeTensor,
    gamma: CubeTensor,
    reduction_order: ReductionOrder,
) -> Result<(CubeTensor, CubeTensor), ModernBertResidualLayerNormError> {
    let (precision, output_bytes) = validate_contract(&residual, &branch, &gamma)?;
    if reduction_order == ReductionOrder::Backend
        || precision != KernelFloatPrecision::F32
        || residual.meta.shape().dims::<3>() != [1, 3, WIDTH]
    {
        // Keep Burn's shape/device-tuned reduction order. A single fixed
        // reduction tree changes long-sequence and F16 model outputs.
        let updated = Tensor::<3>::from_primitive::<crate::WgpuRaw>(residual)
            + Tensor::<3>::from_primitive::<crate::WgpuRaw>(branch);
        let normalized = super::modern_bert_norm_epilogue::normalize(
            updated.clone(),
            Tensor::<1>::from_primitive::<crate::WgpuRaw>(gamma),
        );
        return Ok((
            updated
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU residual"),
            normalized
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU normalization"),
        ));
    }
    let client = residual.client.clone();
    let device = residual.device.clone();
    let shape = Shape::from(residual.meta.shape().dims::<3>());
    let rows = u32::try_from(residual.meta.num_elements() / WIDTH)
        .expect("validated normalization row count");
    let updated = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        shape.clone(),
        client.empty(output_bytes),
        precision.dtype(),
    );
    let normalized = CubeTensor::new_contiguous(
        client.clone(),
        device,
        shape,
        client.empty(output_bytes),
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeKernel> = Box::new(SourceKernel::new(
        ModernBertResidualLayerNormKernel { precision },
        CubeDim::new_1d(WORKGROUP_SIZE),
    ));
    let bindings = KernelArguments::new()
        .with_buffer(residual.handle.binding())
        .with_buffer(branch.handle.binding())
        .with_buffer(gamma.handle.binding())
        .with_buffer(updated.handle.clone().binding())
        .with_buffer(normalized.handle.clone().binding());
    client.launch(task, CubeCount::new_1d(rows), bindings);
    Ok((updated, normalized))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varied_rows_match_burn_fp32_normalization() {
        compare_rows(
            crate::WgpuFloatPrecision::Fp32,
            ReductionOrder::LegacyShortText,
        );
    }

    #[test]
    fn varied_rows_match_burn_fp16_normalization() {
        compare_rows(
            crate::WgpuFloatPrecision::Fp16,
            ReductionOrder::LegacyShortText,
        );
    }

    #[test]
    fn auxiliary_fp32_preserves_backend_reduction_at_every_length() {
        compare_rows(crate::WgpuFloatPrecision::Fp32, ReductionOrder::Backend);
    }

    fn compare_rows(precision: crate::WgpuFloatPrecision, reduction_order: ReductionOrder) {
        use burn::{
            nn::LayerNormConfig,
            tensor::{FloatDType, Tensor, TensorData},
        };
        let kind = if std::env::var_os("IRODORI_FUSION_TEST_INTEGRATED").is_some() {
            cubecl::wgpu::WgpuDeviceKind::IntegratedGpu(0)
        } else {
            cubecl::wgpu::WgpuDeviceKind::DiscreteGpu(0)
        };
        let device = crate::backend_config::wgpu_device_with_precision(
            &burn::backend::wgpu::WgpuDevice::new(kind),
            precision,
        )
        .unwrap();
        let dtype = match precision {
            crate::WgpuFloatPrecision::Fp32 => FloatDType::F32,
            crate::WgpuFloatPrecision::Fp16 => FloatDType::F16,
        };
        let norm = LayerNormConfig::new(WIDTH).with_bias(false).init(&device);
        for [batch, sequence] in [[1, 1], [1, 3], [1, 25], [2, 17], [1, 129]] {
            let shape = [batch, sequence, WIDTH];
            let residual = Tensor::<3>::from_data(
                TensorData::new(
                    (0..batch * sequence * WIDTH)
                        .map(|i| ((i * 31 % 997) as f32 - 498.0) * 0.013)
                        .collect::<Vec<_>>(),
                    shape,
                ),
                &device,
            )
            .cast(dtype);
            let branch = Tensor::<3>::from_data(
                TensorData::new(
                    (0..batch * sequence * WIDTH)
                        .map(|i| ((i * 23 % 1013) as f32 - 506.0) * 0.0013)
                        .collect::<Vec<_>>(),
                    shape,
                ),
                &device,
            )
            .cast(dtype);
            let updated = residual.clone() + branch.clone();
            let expected = norm
                .forward(updated.clone())
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            let (sum, actual) = modern_bert_residual_layer_norm_wgsl(
                residual.try_into_primitive::<crate::WgpuRaw>().unwrap(),
                branch.try_into_primitive::<crate::WgpuRaw>().unwrap(),
                norm.gamma
                    .val()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .unwrap(),
                reduction_order,
            )
            .unwrap();
            let expected_sum = updated
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            let sum = Tensor::<3>::from_primitive::<crate::WgpuRaw>(sum)
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            assert_eq!(
                sum.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected_sum.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            let actual = Tensor::<3>::from_primitive::<crate::WgpuRaw>(actual)
                .into_data()
                .convert::<f32>()
                .try_to_vec::<f32>()
                .unwrap();
            let mismatch = actual
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let max_abs = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            eprintln!("norm {precision:?} {shape:?} mismatches={mismatch} max_abs={max_abs:e}");
            if reduction_order == ReductionOrder::LegacyShortText
                && dtype == FloatDType::F32
                && shape == [1, 3, WIDTH]
            {
                // The previously accepted single-dispatch path retains its
                // original tree. New shapes use the backend's reduction order.
                assert!(max_abs <= 1.0e-5, "normalization changed: {max_abs}");
            } else {
                assert_eq!(mismatch, 0, "normalization rounding changed");
            }
        }
    }

    fn supported_hardware() -> HardwareContract {
        HardwareContract {
            max_page_size: (ELEMENTS * F32_BYTES) as u64,
            max_bindings: REQUIRED_BINDINGS,
            max_shared_memory_size: SHARED_BYTES * 2,
            max_units_per_cube: WORKGROUP_SIZE,
            max_cube_dim: (WORKGROUP_SIZE, 1, 1),
            max_cube_count: ((ELEMENTS as u32).div_ceil(WORKGROUP_SIZE), 1, 1),
        }
    }

    #[test]
    fn exact_hardware_contract_fails_closed() {
        let supported = supported_hardware();
        assert!(supported.supports_launch(ROWS));
        assert!(!supported.supports_launch(0));
        assert!(!supported.supports_launch(usize::MAX));
        assert!(
            !HardwareContract {
                max_bindings: REQUIRED_BINDINGS - 1,
                ..supported
            }
            .supports_launch(ROWS)
        );
        assert!(
            !HardwareContract {
                max_page_size: (ELEMENTS * F32_BYTES - 1) as u64,
                ..supported
            }
            .supports_launch(ROWS)
        );
        assert!(
            !HardwareContract {
                max_shared_memory_size: SHARED_BYTES * 2 - 1,
                ..supported
            }
            .supports_launch(ROWS)
        );
        assert!(
            !HardwareContract {
                max_units_per_cube: WORKGROUP_SIZE - 1,
                ..supported
            }
            .supports_launch(ROWS)
        );
        assert!(
            !HardwareContract {
                max_cube_dim: (WORKGROUP_SIZE - 1, 1, 1),
                ..supported
            }
            .supports_launch(ROWS)
        );
        assert!(
            !HardwareContract {
                max_cube_count: (ROWS as u32 - 1, 1, 1),
                ..supported
            }
            .supports_launch(ROWS)
        );
        assert!(
            !HardwareContract {
                max_cube_count: (ROWS as u32, 2, 1),
                ..supported
            }
            .supports_launch(ROWS)
        );
        assert!(
            HardwareContract {
                max_cube_count: (ROWS as u32, 3, 1),
                ..supported
            }
            .supports_launch(ROWS)
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
