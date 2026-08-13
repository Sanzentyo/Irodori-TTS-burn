//! Fused pointwise bias and residual finalizer for the production WGPU codec.

use std::{error::Error, fmt};

use burn::{
    backend::wgpu::{CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime},
    tensor::Shape,
};
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

use super::precision::{KernelFloatPrecision, common_float_precision};

const BATCH: usize = 1;
const WORKGROUP_SIZE: u32 = 256;
const REQUIRED_BINDINGS: u32 = 4;
fn supported_decoder_shape(channels: usize, length: usize) -> bool {
    matches!(channels, 768 | 384 | 192 | 96) && length > 0
}

#[derive(Debug)]
pub struct FinalizerError {
    message: String,
}

impl FinalizerError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for FinalizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for FinalizerError {}

#[derive(Debug)]
struct PointwiseResidualFinalizerKernel {
    precision: KernelFloatPrecision,
    channels: u32,
    length: u32,
    elements: u32,
}

impl KernelSource for PointwiseResidualFinalizerKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("pointwise_residual_finalizer.wgsl"),
                include_str!("pointwise_residual_finalizer_f16.wgsl"),
            )
            .register("channels", self.channels.to_string())
            .register("length", self.length.to_string())
            .register("elements", self.elements.to_string())
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.channels, self.length, self.elements))
    }
}

/// Check the output allocation and dispatch limits before materializing the
/// packed pointwise GEMM result.
///
/// `reference_ncl` must be the contiguous residual tensor that will be passed
/// to [`pointwise_residual_finalizer_wgsl`]. The launcher validates all four
/// tensors again before dispatch, so a caller must still handle its `Result`.
pub fn device_supports_pointwise_residual_finalizer(
    reference_ncl: &CubeTensor<WgpuRuntime>,
    channels: usize,
    length: usize,
) -> bool {
    let Some(precision) = KernelFloatPrecision::from_dtype(reference_ncl.dtype) else {
        return false;
    };
    if reference_ncl.meta.num_dims() != 3 {
        return false;
    }
    if !supported_decoder_shape(channels, length) {
        return false;
    }
    let Some(elements) = channels.checked_mul(length) else {
        return false;
    };
    let shape = reference_ncl.meta.shape();
    if [shape[0], shape[1], shape[2]] != [BATCH, channels, length]
        || !reference_ncl.is_contiguous()
        || rank3_strides(reference_ncl) != [elements, length, 1]
    {
        return false;
    }
    let Some(output_bytes) = elements.checked_mul(precision.element_bytes()) else {
        return false;
    };
    let Ok(output_bytes) = u64::try_from(output_bytes) else {
        return false;
    };
    let Ok(elements) = u32::try_from(elements) else {
        return false;
    };
    let workgroups = elements.div_ceil(WORKGROUP_SIZE);
    let properties = reference_ncl.client.properties();
    let hardware = &properties.hardware;

    output_bytes <= properties.memory.max_page_size
        && hardware.max_bindings >= REQUIRED_BINDINGS
        && hardware.max_units_per_cube >= WORKGROUP_SIZE
        && hardware.max_cube_dim.0 >= WORKGROUP_SIZE
        && hardware.max_cube_dim.1 >= 1
        && hardware.max_cube_dim.2 >= 1
        && hardware.max_cube_count.0 >= workgroups
        && hardware.max_cube_count.1 >= 1
        && hardware.max_cube_count.2 >= 1
}

fn rank3_strides(tensor: &CubeTensor<WgpuRuntime>) -> [usize; 3] {
    let strides = tensor.meta.strides();
    [strides[0], strides[1], strides[2]]
}

fn validate_contract(
    branch_nlc: &CubeTensor<WgpuRuntime>,
    bias: &CubeTensor<WgpuRuntime>,
    residual_ncl: &CubeTensor<WgpuRuntime>,
) -> Result<(KernelFloatPrecision, usize, usize, usize, u32), FinalizerError> {
    let precision = common_float_precision([branch_nlc.dtype, bias.dtype, residual_ncl.dtype])
        .ok_or_else(|| FinalizerError::new("all bindings must share f32 or f16 dtype"))?;
    for (name, tensor) in [
        ("branch_nlc", branch_nlc),
        ("bias", bias),
        ("residual_ncl", residual_ncl),
    ] {
        if tensor.device != branch_nlc.device {
            return Err(FinalizerError::new(format!(
                "{name} is on a different WGPU device"
            )));
        }
    }

    if branch_nlc.meta.num_dims() != 3 {
        return Err(FinalizerError::new(format!(
            "branch_nlc must be rank 3 [1,L,C], got rank {}",
            branch_nlc.meta.num_dims()
        )));
    }
    if bias.meta.num_dims() != 1 {
        return Err(FinalizerError::new(format!(
            "bias must be rank 1 [C], got rank {}",
            bias.meta.num_dims()
        )));
    }
    if residual_ncl.meta.num_dims() != 3 {
        return Err(FinalizerError::new(format!(
            "residual_ncl must be rank 3 [1,C,L], got rank {}",
            residual_ncl.meta.num_dims()
        )));
    }

    let branch_shape = branch_nlc.meta.shape();
    let [batch, length, channels] = [branch_shape[0], branch_shape[1], branch_shape[2]];
    if batch != BATCH {
        return Err(FinalizerError::new(format!(
            "branch_nlc is specialized for B=1, got B={batch}"
        )));
    }
    if !supported_decoder_shape(channels, length) {
        return Err(FinalizerError::new(format!(
            "unsupported codec pointwise shape C={channels} L={length}; expected positive length and C in [768,384,192,96]"
        )));
    }
    if bias.meta.shape()[0] != channels {
        return Err(FinalizerError::new(format!(
            "bias shape mismatch: expected [{channels}], got {:?}",
            bias.meta.shape()
        )));
    }
    let residual_shape = residual_ncl.meta.shape();
    if [residual_shape[0], residual_shape[1], residual_shape[2]] != [BATCH, channels, length] {
        return Err(FinalizerError::new(format!(
            "residual_ncl shape mismatch: expected [1,{channels},{length}], got {:?}",
            residual_shape
        )));
    }

    let elements = channels
        .checked_mul(length)
        .ok_or_else(|| FinalizerError::new("C*L overflows usize"))?;
    let branch_expected_strides = [elements, channels, 1];
    let residual_expected_strides = [elements, length, 1];
    if !branch_nlc.is_contiguous() || rank3_strides(branch_nlc) != branch_expected_strides {
        return Err(FinalizerError::new(format!(
            "branch_nlc must have exact contiguous [1,L,C] strides {branch_expected_strides:?}, got {:?}",
            rank3_strides(branch_nlc)
        )));
    }
    if !bias.is_contiguous() || bias.meta.strides()[0] != 1 {
        return Err(FinalizerError::new(format!(
            "bias must have exact contiguous [C] stride [1], got {:?}",
            bias.meta.strides()
        )));
    }
    if !residual_ncl.is_contiguous() || rank3_strides(residual_ncl) != residual_expected_strides {
        return Err(FinalizerError::new(format!(
            "residual_ncl must have exact contiguous [1,C,L] strides {residual_expected_strides:?}, got {:?}",
            rank3_strides(residual_ncl)
        )));
    }

    let elements_u32 = u32::try_from(elements)
        .map_err(|_| FinalizerError::new(format!("element count {elements} exceeds u32")))?;
    let channels_u32 = u32::try_from(channels)
        .map_err(|_| FinalizerError::new(format!("channel count {channels} exceeds u32")))?;
    let length_u32 = u32::try_from(length)
        .map_err(|_| FinalizerError::new(format!("length {length} exceeds u32")))?;
    let final_branch_index = length_u32
        .checked_sub(1)
        .and_then(|time| time.checked_mul(channels_u32))
        .and_then(|base| base.checked_add(channels_u32.saturating_sub(1)))
        .ok_or_else(|| FinalizerError::new("NLC branch index calculation overflows u32"))?;
    if final_branch_index != elements_u32 - 1 {
        return Err(FinalizerError::new(format!(
            "NLC branch index contract mismatch: last={final_branch_index}, elements={elements_u32}"
        )));
    }

    let output_bytes = elements
        .checked_mul(precision.element_bytes())
        .ok_or_else(|| FinalizerError::new("output byte count overflows usize"))?;
    let output_bytes_u64 = u64::try_from(output_bytes)
        .map_err(|_| FinalizerError::new("output byte count exceeds u64"))?;
    let properties = branch_nlc.client.properties();
    if output_bytes_u64 > properties.memory.max_page_size {
        return Err(FinalizerError::new(format!(
            "output requires {output_bytes} bytes, device max page is {} bytes",
            properties.memory.max_page_size
        )));
    }
    let hardware = &properties.hardware;
    let workgroups = elements_u32.div_ceil(WORKGROUP_SIZE);
    if hardware.max_bindings < REQUIRED_BINDINGS {
        return Err(FinalizerError::new(format!(
            "candidate requires {REQUIRED_BINDINGS} storage bindings, device exposes {}",
            hardware.max_bindings
        )));
    }
    if hardware.max_units_per_cube < WORKGROUP_SIZE
        || hardware.max_cube_dim.0 < WORKGROUP_SIZE
        || hardware.max_cube_dim.1 < 1
        || hardware.max_cube_dim.2 < 1
    {
        return Err(FinalizerError::new(format!(
            "candidate requires workgroup [{WORKGROUP_SIZE},1,1], device limits units={} dim={:?}",
            hardware.max_units_per_cube, hardware.max_cube_dim
        )));
    }
    if hardware.max_cube_count.0 < workgroups
        || hardware.max_cube_count.1 < 1
        || hardware.max_cube_count.2 < 1
    {
        return Err(FinalizerError::new(format!(
            "candidate requires dispatch [{workgroups},1,1], device limit is {:?}",
            hardware.max_cube_count
        )));
    }

    Ok((precision, channels, length, output_bytes, workgroups))
}

/// Compute `(matmul_nlc + bias) + residual_ncl` and emit contiguous NCL.
///
/// The two additions deliberately remain separate and ordered in WGSL. The
/// caller supplies the already materialized pointwise GEMM result in physical
/// NLC order, avoiding an intermediate bias kernel and a later NLC-to-NCL
/// materialization.
pub fn pointwise_residual_finalizer_wgsl(
    branch_nlc: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    residual_ncl: CubeTensor<WgpuRuntime>,
) -> Result<CubeTensor<WgpuRuntime>, FinalizerError> {
    let (precision, channels, length, output_bytes, workgroups) =
        validate_contract(&branch_nlc, &bias, &residual_ncl)?;
    let elements = channels
        .checked_mul(length)
        .ok_or_else(|| FinalizerError::new("validated C*L unexpectedly overflowed usize"))?;
    let client = branch_nlc.client.clone();
    let output = CubeTensor::new_contiguous(
        client.clone(),
        branch_nlc.device.clone(),
        Shape::from([BATCH, channels, length]),
        client.empty(output_bytes),
        precision.dtype(),
    );
    let kernel = PointwiseResidualFinalizerKernel {
        precision,
        channels: u32::try_from(channels)
            .map_err(|_| FinalizerError::new("validated channel count exceeds u32"))?,
        length: u32::try_from(length)
            .map_err(|_| FinalizerError::new("validated length exceeds u32"))?,
        elements: u32::try_from(elements)
            .map_err(|_| FinalizerError::new("validated element count exceeds u32"))?,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(kernel, CubeDim::new_1d(WORKGROUP_SIZE)));
    let bindings = KernelArguments::new()
        .with_buffer(branch_nlc.handle.binding())
        .with_buffer(bias.handle.binding())
        .with_buffer(residual_ncl.handle.binding())
        .with_buffer(output.handle.clone().binding());
    client.launch(task, CubeCount::new_1d(workgroups), bindings);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_sweep_stage_shapes_are_supported() {
        for latent_steps in [13, 25, 50, 100, 200] {
            for (channels, length) in [
                (768, latent_steps * 12),
                (384, latent_steps * 120),
                (192, latent_steps * 960),
                (96, latent_steps * 1_920),
            ] {
                assert!(supported_decoder_shape(channels, length));
            }
        }
        assert!(!supported_decoder_shape(96, 0));
        assert!(!supported_decoder_shape(95, 96_000));
    }
}
