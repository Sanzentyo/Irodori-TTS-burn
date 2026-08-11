//! Production WmHead Snake/layout kernel.
//!
//! This accepts only the released WmHead input and writes the exact production
//! Snake values into a physically contiguous NLC allocation. Calling
//! `swap_dims(1, 2)` on the returned tensor creates the logical NCL view
//! expected by the unchanged Burn `Conv1d` module while its internal NHWC view
//! remains contiguous.
//!
//! On the RTX 3060 Ti with production weights, the isolated full-head A/B was
//! bit-exact at Snake, pre-tanh, and final tanh boundaries and reduced median
//! latency from 1,730.926 us to 1,375.178 us (1.259x; 355.748 us saved). It
//! removes one 36,864,000-byte transient and 73,728,000 logical boundary bytes.

use core::fmt;

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::{DType, Shape};
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

pub const BATCH: usize = 1;
pub const CHANNELS: usize = 96;
pub const TIME: usize = 96_000;
pub const TILE: usize = 32;
const LOCAL_TIME_ROWS: usize = 8;
const WORKGROUP_SIZE: u32 = (TILE * LOCAL_TIME_ROWS) as u32;
const TILE_STRIDE: usize = TILE + 1;
const SHARED_ELEMENTS: usize = TILE * TILE_STRIDE;
const SHARED_BYTES: usize = SHARED_ELEMENTS * size_of::<f32>();
const REQUIRED_BINDINGS: u32 = 3;
const F32_BYTES: usize = size_of::<f32>();
const OUTPUT_ELEMENTS: usize = BATCH * CHANNELS * TIME;
const OUTPUT_BYTES: usize = OUTPUT_ELEMENTS * F32_BYTES;
const DISPATCH_X: u32 = (TIME / TILE) as u32;
const DISPATCH_Y: u32 = (CHANNELS / TILE) as u32;

const _: () = assert!(TIME.is_multiple_of(TILE));
const _: () = assert!(CHANNELS.is_multiple_of(TILE));
const _: () = assert!(SHARED_ELEMENTS == 1_056);
const _: () = assert!(SHARED_BYTES == 4_224);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WmHeadSnakeNlcError {
    message: String,
}

impl WmHeadSnakeNlcError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for WmHeadSnakeNlcError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for WmHeadSnakeNlcError {}

#[derive(Debug)]
struct WmHeadSnakeNlcKernel;

impl KernelSource for WmHeadSnakeNlcKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wm_head_snake_nlc.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

fn tensor_bytes(
    tensor: &CubeTensor<WgpuRuntime>,
    label: &str,
) -> Result<usize, WmHeadSnakeNlcError> {
    tensor
        .meta
        .num_elements()
        .checked_mul(F32_BYTES)
        .ok_or_else(|| WmHeadSnakeNlcError::new(format!("{label} byte count overflow")))
}

fn validate_tensor(
    tensor: &CubeTensor<WgpuRuntime>,
    expected_shape: [usize; 3],
    label: &str,
) -> Result<(), WmHeadSnakeNlcError> {
    if tensor.dtype != DType::F32 {
        return Err(WmHeadSnakeNlcError::new(format!(
            "WmHead Snake/NLC {label} must be f32, got {}",
            tensor.dtype.name()
        )));
    }
    if tensor.meta.num_dims() != 3 {
        return Err(WmHeadSnakeNlcError::new(format!(
            "WmHead Snake/NLC {label} must have rank 3, got {}",
            tensor.meta.num_dims()
        )));
    }
    let actual_shape = tensor.meta.shape().dims::<3>();
    if actual_shape != expected_shape {
        return Err(WmHeadSnakeNlcError::new(format!(
            "WmHead Snake/NLC {label} shape mismatch: expected {expected_shape:?}, got {actual_shape:?}"
        )));
    }
    if !tensor.is_contiguous() {
        return Err(WmHeadSnakeNlcError::new(format!(
            "WmHead Snake/NLC {label} must be contiguous, strides={:?}",
            tensor.meta.strides()
        )));
    }
    Ok(())
}

fn validate_resources(
    input: &CubeTensor<WgpuRuntime>,
    alpha: &CubeTensor<WgpuRuntime>,
) -> Result<(), WmHeadSnakeNlcError> {
    let properties = input.client.properties();
    let hardware = &properties.hardware;
    if hardware.max_bindings < REQUIRED_BINDINGS {
        return Err(WmHeadSnakeNlcError::new(format!(
            "WmHead Snake/NLC requires {REQUIRED_BINDINGS} storage bindings, device supports {}",
            hardware.max_bindings
        )));
    }
    if hardware.max_units_per_cube < WORKGROUP_SIZE
        || hardware.max_cube_dim.0 < TILE as u32
        || hardware.max_cube_dim.1 < LOCAL_TIME_ROWS as u32
    {
        return Err(WmHeadSnakeNlcError::new(format!(
            "WmHead Snake/NLC requires workgroup [{TILE},{LOCAL_TIME_ROWS},1] ({WORKGROUP_SIZE} invocations), device supports units={} dims={:?}",
            hardware.max_units_per_cube, hardware.max_cube_dim
        )));
    }
    if hardware.max_shared_memory_size < SHARED_BYTES {
        return Err(WmHeadSnakeNlcError::new(format!(
            "WmHead Snake/NLC requires {SHARED_BYTES} shared bytes, device supports {}",
            hardware.max_shared_memory_size
        )));
    }
    if hardware.max_cube_count.0 < DISPATCH_X || hardware.max_cube_count.1 < DISPATCH_Y {
        return Err(WmHeadSnakeNlcError::new(format!(
            "WmHead Snake/NLC dispatch [{DISPATCH_X},{DISPATCH_Y},1] exceeds device limit {:?}",
            hardware.max_cube_count
        )));
    }

    let page_limit = properties.memory.max_page_size;
    for (label, bytes) in [
        ("input", tensor_bytes(input, "input")?),
        ("alpha", tensor_bytes(alpha, "alpha")?),
        ("output", OUTPUT_BYTES),
    ] {
        let bytes = u64::try_from(bytes).map_err(|_| {
            WmHeadSnakeNlcError::new(format!("WmHead Snake/NLC {label} byte count exceeds u64"))
        })?;
        if bytes > page_limit {
            return Err(WmHeadSnakeNlcError::new(format!(
                "WmHead Snake/NLC {label} requires {bytes} bytes, device page limit is {page_limit}"
            )));
        }
    }
    Ok(())
}

/// Apply the exact production Snake expression and write contiguous NLC.
///
/// The returned primitive has shape `[1, 96000, 96]`. The caller converts it
/// to the logical `[1, 96, 96000]` NCL view with `swap_dims(1, 2)` before
/// passing it to the unchanged Burn convolution.
pub fn wm_head_snake_ncl_to_nlc_wgsl(
    input_ncl: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
) -> Result<CubeTensor<WgpuRuntime>, WmHeadSnakeNlcError> {
    validate_tensor(&input_ncl, [BATCH, CHANNELS, TIME], "input")?;
    validate_tensor(&alpha, [BATCH, CHANNELS, 1], "alpha")?;
    if input_ncl.device != alpha.device {
        return Err(WmHeadSnakeNlcError::new(format!(
            "WmHead Snake/NLC input and alpha must be on one device, got {:?} and {:?}",
            input_ncl.device, alpha.device
        )));
    }
    validate_resources(&input_ncl, &alpha)?;

    let client = input_ncl.client.clone();
    let output_nlc = CubeTensor::new_contiguous(
        client.clone(),
        input_ncl.device.clone(),
        Shape::from([BATCH, TIME, CHANNELS]),
        client.empty(OUTPUT_BYTES),
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            WmHeadSnakeNlcKernel,
            CubeDim::new_2d(TILE as u32, LOCAL_TIME_ROWS as u32),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(input_ncl.handle.binding())
        .with_buffer(alpha.handle.binding())
        .with_buffer(output_nlc.handle.clone().binding());
    client.launch(task, CubeCount::new_2d(DISPATCH_X, DISPATCH_Y), bindings);
    Ok(output_nlc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_released_shape_tiles_without_guards() {
        assert_eq!(TIME / TILE, 3_000);
        assert_eq!(CHANNELS / TILE, 3);
        assert_eq!(DISPATCH_X, 3_000);
        assert_eq!(DISPATCH_Y, 3);
        assert_eq!(WORKGROUP_SIZE, 256);
        assert_eq!(SHARED_BYTES, 4_224);
        assert_eq!(OUTPUT_BYTES, 36_864_000);
    }

    #[test]
    fn shader_preserves_exact_snake_and_uniform_storage_contract() {
        let shader = include_str!("wm_head_snake_nlc.wgsl");
        let production = include_str!("snake.wgsl");
        for expression in [
            "let sine = sin(a * x);",
            "x + (sine * sine) / (a + 1e-9);",
            "workgroupBarrier();",
            "@compute @workgroup_size(32, 8, 1)",
        ] {
            assert!(shader.contains(expression), "missing {expression:?}");
        }
        for expression in [
            "let sine = sin(a * x);",
            "output[index] = x + (sine * sine) / (a + 1e-9);",
        ] {
            assert!(
                production.contains(expression),
                "production Snake source drifted at {expression:?}"
            );
        }
        assert_eq!(shader.matches("var<storage, read_write>").count(), 3);
        assert!(!shader.contains("var<storage, read>"));
    }
}
