//! Prepared pointwise-residual and next-Snake pair for the production codec.
//!
//! This combines the accepted pointwise residual finalizer with the immediately
//! following `Snake1d`, returning both the raw residual tensor and the
//! activated tensor needed by the next `ResidualUnit`. It is restricted to the
//! eight measured intra-block boundaries in the released decoder.

use core::fmt;

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::{DType, Shape};
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

const BATCH: usize = 1;
const WORKGROUP_SIZE: u32 = 256;
const REQUIRED_BINDINGS: u32 = 6;
const F32_BYTES: usize = size_of::<f32>();
#[cfg(test)]
const ELIGIBLE_SHAPES: [(usize, usize); 4] =
    [(768, 600), (384, 6_000), (192, 48_000), (96, 96_000)];

fn supported_decoder_shape(channels: usize, length: usize) -> bool {
    matches!(channels, 768 | 384 | 192 | 96) && length > 0
}

/// Validation failure that prevents launching the prepared-pair kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PairFinalizerError {
    message: String,
}

impl PairFinalizerError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for PairFinalizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for PairFinalizerError {}

/// Raw residual and Snake-activated NCL outputs from one dispatch.
#[derive(Debug)]
pub struct PointwiseResidualSnakePair {
    raw_ncl: CubeTensor<WgpuRuntime>,
    activated_ncl: CubeTensor<WgpuRuntime>,
}

impl PointwiseResidualSnakePair {
    /// Consume the pair while preserving both reference-counted WGPU tensors.
    pub fn into_tensors(self) -> (CubeTensor<WgpuRuntime>, CubeTensor<WgpuRuntime>) {
        (self.raw_ncl, self.activated_ncl)
    }
}

#[derive(Debug)]
struct PointwiseResidualSnakePairKernel {
    channels: u32,
    length: u32,
    elements: u32,
}

impl KernelSource for PointwiseResidualSnakePairKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("pointwise_residual_snake_pair.wgsl"))
            .register("channels", self.channels.to_string())
            .register("length", self.length.to_string())
            .register("elements", self.elements.to_string())
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.channels, self.length, self.elements))
    }
}

fn exact_strides<const D: usize>(tensor: &CubeTensor<WgpuRuntime>) -> [usize; D] {
    let strides = tensor.meta.strides();
    std::array::from_fn(|index| strides[index])
}

fn validate_tensor(
    name: &str,
    tensor: &CubeTensor<WgpuRuntime>,
    rank: usize,
    reference: &CubeTensor<WgpuRuntime>,
) -> Result<(), PairFinalizerError> {
    if tensor.dtype != DType::F32 {
        return Err(PairFinalizerError::new(format!(
            "{name} must be f32, got {:?}",
            tensor.dtype
        )));
    }
    if tensor.meta.num_dims() != rank {
        return Err(PairFinalizerError::new(format!(
            "{name} must have rank {rank}, got {}",
            tensor.meta.num_dims()
        )));
    }
    if tensor.device != reference.device {
        return Err(PairFinalizerError::new(format!(
            "{name} is on a different WGPU device"
        )));
    }
    if !tensor.is_contiguous() {
        return Err(PairFinalizerError::new(format!(
            "{name} must be contiguous, got strides {:?}",
            tensor.meta.strides()
        )));
    }
    Ok(())
}

fn checked_bytes(elements: usize, label: &str) -> Result<usize, PairFinalizerError> {
    elements
        .checked_mul(F32_BYTES)
        .ok_or_else(|| PairFinalizerError::new(format!("{label} byte count overflows usize")))
}

/// Check the pair outputs and next-Snake inputs before materializing the
/// packed pointwise GEMM result.
///
/// The caller must separately validate its pointwise input, weight, bias, and
/// cache. The launcher repeats the complete six-tensor validation before its
/// dispatch, so callers must still handle its `Result`.
pub fn device_supports_pointwise_residual_snake_pair(
    residual_ncl: &CubeTensor<WgpuRuntime>,
    alpha: &CubeTensor<WgpuRuntime>,
    channels: usize,
    length: usize,
) -> bool {
    if !supported_decoder_shape(channels, length)
        || residual_ncl.dtype != DType::F32
        || alpha.dtype != DType::F32
        || residual_ncl.device != alpha.device
        || residual_ncl.meta.num_dims() != 3
        || alpha.meta.num_dims() != 3
    {
        return false;
    }
    let Some(elements) = channels.checked_mul(length) else {
        return false;
    };
    if residual_ncl.meta.shape().dims::<3>() != [BATCH, channels, length]
        || exact_strides::<3>(residual_ncl) != [elements, length, 1]
        || !residual_ncl.is_contiguous()
        || alpha.meta.shape().dims::<3>() != [1, channels, 1]
        || exact_strides::<3>(alpha) != [channels, 1, 1]
        || !alpha.is_contiguous()
    {
        return false;
    }
    let Some(output_bytes) = elements.checked_mul(F32_BYTES) else {
        return false;
    };
    let Ok(output_bytes) = u64::try_from(output_bytes) else {
        return false;
    };
    let Ok(elements) = u32::try_from(elements) else {
        return false;
    };
    let workgroups = elements.div_ceil(WORKGROUP_SIZE);
    let properties = residual_ncl.client.properties();
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

fn validate_contract(
    branch_nlc: &CubeTensor<WgpuRuntime>,
    bias: &CubeTensor<WgpuRuntime>,
    residual_ncl: &CubeTensor<WgpuRuntime>,
    alpha: &CubeTensor<WgpuRuntime>,
) -> Result<(usize, usize, usize, u32), PairFinalizerError> {
    validate_tensor("branch_nlc", branch_nlc, 3, branch_nlc)?;
    validate_tensor("bias", bias, 1, branch_nlc)?;
    validate_tensor("residual_ncl", residual_ncl, 3, branch_nlc)?;
    validate_tensor("alpha", alpha, 3, branch_nlc)?;

    let branch_shape = branch_nlc.meta.shape().dims::<3>();
    let [batch, length, channels] = branch_shape;
    if batch != BATCH || !supported_decoder_shape(channels, length) {
        return Err(PairFinalizerError::new(format!(
            "unsupported branch shape {branch_shape:?}; expected B=1, positive L, and C in [768,384,192,96]"
        )));
    }
    let elements = channels
        .checked_mul(length)
        .ok_or_else(|| PairFinalizerError::new("C*L overflows usize"))?;
    let expected_branch_strides = [elements, channels, 1];
    if exact_strides::<3>(branch_nlc) != expected_branch_strides {
        return Err(PairFinalizerError::new(format!(
            "branch_nlc strides must be {expected_branch_strides:?}, got {:?}",
            exact_strides::<3>(branch_nlc)
        )));
    }

    let expected_ncl_shape = [BATCH, channels, length];
    let actual_residual_shape = residual_ncl.meta.shape().dims::<3>();
    if actual_residual_shape != expected_ncl_shape {
        return Err(PairFinalizerError::new(format!(
            "residual_ncl shape must be {expected_ncl_shape:?}, got {actual_residual_shape:?}"
        )));
    }
    let expected_ncl_strides = [elements, length, 1];
    if exact_strides::<3>(residual_ncl) != expected_ncl_strides {
        return Err(PairFinalizerError::new(format!(
            "residual_ncl strides must be {expected_ncl_strides:?}, got {:?}",
            exact_strides::<3>(residual_ncl)
        )));
    }

    let actual_bias_shape = bias.meta.shape().dims::<1>();
    if actual_bias_shape != [channels] || exact_strides::<1>(bias) != [1] {
        return Err(PairFinalizerError::new(format!(
            "bias must be contiguous [{channels}] with stride [1], got shape={actual_bias_shape:?} strides={:?}",
            exact_strides::<1>(bias)
        )));
    }
    let actual_alpha_shape = alpha.meta.shape().dims::<3>();
    let expected_alpha_shape = [1, channels, 1];
    let expected_alpha_strides = [channels, 1, 1];
    if actual_alpha_shape != expected_alpha_shape
        || exact_strides::<3>(alpha) != expected_alpha_strides
    {
        return Err(PairFinalizerError::new(format!(
            "alpha must be contiguous {expected_alpha_shape:?} with strides {expected_alpha_strides:?}, got shape={actual_alpha_shape:?} strides={:?}",
            exact_strides::<3>(alpha)
        )));
    }

    let elements_u32 = u32::try_from(elements)
        .map_err(|_| PairFinalizerError::new(format!("element count {elements} exceeds u32")))?;
    let channels_u32 = u32::try_from(channels)
        .map_err(|_| PairFinalizerError::new(format!("channel count {channels} exceeds u32")))?;
    let length_u32 = u32::try_from(length)
        .map_err(|_| PairFinalizerError::new(format!("length {length} exceeds u32")))?;
    let final_branch_index = length_u32
        .checked_sub(1)
        .and_then(|time| time.checked_mul(channels_u32))
        .and_then(|base| base.checked_add(channels_u32.saturating_sub(1)))
        .ok_or_else(|| PairFinalizerError::new("NLC index calculation overflows u32"))?;
    if final_branch_index != elements_u32 - 1 {
        return Err(PairFinalizerError::new(format!(
            "NLC index contract mismatch: last={final_branch_index}, elements={elements_u32}"
        )));
    }

    let output_bytes = checked_bytes(elements, "output")?;
    let workgroups = elements_u32.div_ceil(WORKGROUP_SIZE);
    let properties = branch_nlc.client.properties();
    let page_limit = properties.memory.max_page_size;
    for (name, tensor_elements) in [
        ("branch_nlc", branch_nlc.meta.num_elements()),
        ("bias", bias.meta.num_elements()),
        ("residual_ncl", residual_ncl.meta.num_elements()),
        ("alpha", alpha.meta.num_elements()),
        ("raw_ncl", elements),
        ("activated_ncl", elements),
    ] {
        let bytes = checked_bytes(tensor_elements, name)?;
        let bytes_u64 = u64::try_from(bytes).map_err(|_| {
            PairFinalizerError::new(format!("{name} byte count {bytes} exceeds u64"))
        })?;
        if bytes_u64 > page_limit {
            return Err(PairFinalizerError::new(format!(
                "{name} requires {bytes} bytes, device page limit is {page_limit}"
            )));
        }
    }

    let hardware = &properties.hardware;
    if hardware.max_bindings < REQUIRED_BINDINGS {
        return Err(PairFinalizerError::new(format!(
            "prepared pair requires {REQUIRED_BINDINGS} storage bindings, device exposes {}",
            hardware.max_bindings
        )));
    }
    if hardware.max_units_per_cube < WORKGROUP_SIZE
        || hardware.max_cube_dim.0 < WORKGROUP_SIZE
        || hardware.max_cube_dim.1 < 1
        || hardware.max_cube_dim.2 < 1
    {
        return Err(PairFinalizerError::new(format!(
            "prepared pair requires workgroup [{WORKGROUP_SIZE},1,1], device limits units={} dim={:?}",
            hardware.max_units_per_cube, hardware.max_cube_dim
        )));
    }
    if hardware.max_cube_count.0 < workgroups
        || hardware.max_cube_count.1 < 1
        || hardware.max_cube_count.2 < 1
    {
        return Err(PairFinalizerError::new(format!(
            "prepared pair requires dispatch [{workgroups},1,1], device limit is {:?}",
            hardware.max_cube_count
        )));
    }

    Ok((channels, length, output_bytes, workgroups))
}

/// Emit raw `(branch + bias) + residual` and its exact Snake activation.
///
/// Both outputs are contiguous `[1,C,L]`. The accepted shapes are limited to
/// the four released decoder shapes used by the eight measured
/// `res0 -> res1` and `res1 -> res2` boundaries.
///
/// # Errors
///
/// Returns an error before allocation or launch when dtype, shape, strides,
/// device, integer range, binding size, workgroup, or dispatch contracts do
/// not match the production contract.
pub fn pointwise_residual_snake_pair_wgsl(
    branch_nlc: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    residual_ncl: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
) -> Result<PointwiseResidualSnakePair, PairFinalizerError> {
    let (channels, length, output_bytes, workgroups) =
        validate_contract(&branch_nlc, &bias, &residual_ncl, &alpha)?;
    let elements = channels
        .checked_mul(length)
        .ok_or_else(|| PairFinalizerError::new("validated C*L unexpectedly overflowed usize"))?;
    let client = branch_nlc.client.clone();
    let device = branch_nlc.device.clone();
    let shape = Shape::from([BATCH, channels, length]);
    let raw_ncl = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        shape.clone(),
        client.empty(output_bytes),
        DType::F32,
    );
    let activated_ncl = CubeTensor::new_contiguous(
        client.clone(),
        device,
        shape,
        client.empty(output_bytes),
        DType::F32,
    );
    let kernel = PointwiseResidualSnakePairKernel {
        channels: u32::try_from(channels)
            .map_err(|_| PairFinalizerError::new("validated channel count exceeds u32"))?,
        length: u32::try_from(length)
            .map_err(|_| PairFinalizerError::new("validated length exceeds u32"))?,
        elements: u32::try_from(elements)
            .map_err(|_| PairFinalizerError::new("validated element count exceeds u32"))?,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(kernel, CubeDim::new_1d(WORKGROUP_SIZE)));
    let bindings = KernelArguments::new()
        .with_buffer(branch_nlc.handle.binding())
        .with_buffer(bias.handle.binding())
        .with_buffer(residual_ncl.handle.binding())
        .with_buffer(alpha.handle.binding())
        .with_buffer(raw_ncl.handle.clone().binding())
        .with_buffer(activated_ncl.handle.clone().binding());
    client.launch(task, CubeCount::new_1d(workgroups), bindings);
    Ok(PointwiseResidualSnakePair {
        raw_ncl,
        activated_ncl,
    })
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

    #[test]
    fn released_pair_shapes_fit_wgsl_indexing_and_dispatch() {
        for (channels, length) in ELIGIBLE_SHAPES {
            let elements = channels * length;
            assert!(u32::try_from(elements).is_ok());
            assert!(u32::try_from(channels).is_ok());
            assert!(u32::try_from(length).is_ok());
            assert!(u32::try_from(elements.div_ceil(WORKGROUP_SIZE as usize)).is_ok());
        }
    }

    #[test]
    fn shader_preserves_finalizer_then_snake_source_order() {
        let shader = include_str!("pointwise_residual_snake_pair.wgsl");
        let production_finalizer = include_str!("pointwise_residual_finalizer.wgsl");
        let production_snake = include_str!("snake.wgsl");
        assert_eq!(shader.matches("var<storage, read_write>").count(), 6);
        assert!(!shader.contains("var<storage, read>"));

        let branch_expression = "let biased = branch_nlc[branch_index] + bias[channel];";
        let residual_expression = "biased + residual_ncl[output_index]";
        let snake_expression = "raw + (sine * sine) / (a + 1e-9)";
        assert!(production_finalizer.contains(branch_expression));
        assert!(production_finalizer.contains(residual_expression));
        assert!(production_snake.contains("x + (sine * sine) / (a + 1e-9)"));

        let branch = shader
            .find(branch_expression)
            .expect("ordered branch plus bias");
        let residual = shader
            .find("let raw = biased + residual_ncl[output_index];")
            .expect("ordered residual addition");
        let raw_write = shader
            .find("raw_ncl[output_index] = raw;")
            .expect("raw output write");
        let alpha = shader.find("let a = alpha[channel];").expect("alpha load");
        let sine = shader.find("let sine = sin(a * raw);").expect("Snake sine");
        let activated = shader
            .find("activated_ncl[output_index] = raw + (sine * sine) / (a + 1e-9);")
            .expect("Snake output formula");
        assert!(shader.contains(snake_expression));
        assert!(branch < residual && residual < raw_write && raw_write < alpha);
        assert!(alpha < sine && sine < activated);
    }
}
