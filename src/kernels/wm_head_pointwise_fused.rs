//! Profile-only fusion of the final decoder pointwise residual with WmHead.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::{DType, Shape};
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

const CHANNELS: usize = 96;
const KERNEL_SIZE: usize = 7;
const TIME_TILE: usize = 64;
const WORKGROUP_SIZE: u32 = 256;
const REQUIRED_BINDINGS: u32 = 8;
const SHARED_F16_ELEMENTS: usize =
    2 * 70 * CHANNELS + CHANNELS * CHANNELS + CHANNELS + CHANNELS + CHANNELS * KERNEL_SIZE;
const SHARED_BYTES: usize = SHARED_F16_ELEMENTS * size_of::<half::f16>();

#[derive(Debug)]
struct WmHeadPointwiseFusedKernel {
    time: usize,
}

impl KernelSource for WmHeadPointwiseFusedKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("wm_head_pointwise_fused_f16.wgsl"))
            .register("time", self.time.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.time)
    }
}

fn exact_contiguous(tensor: &CubeTensor<WgpuRuntime>, shape: &[usize]) -> bool {
    tensor.dtype == DType::F16 && tensor.meta.shape().as_slice() == shape && tensor.is_contiguous()
}

#[allow(clippy::too_many_arguments)]
pub fn try_wm_head_pointwise_fused_f16(
    input_nhwc: CubeTensor<WgpuRuntime>,
    pointwise_weight_oik: CubeTensor<WgpuRuntime>,
    pointwise_bias: CubeTensor<WgpuRuntime>,
    residual_ncl: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
    head_weight_oik: CubeTensor<WgpuRuntime>,
    head_bias: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    let [batch, time, channels] = input_nhwc.meta.shape().dims::<3>();
    if batch != 1
        || channels != CHANNELS
        || time == 0
        || !time.is_multiple_of(TIME_TILE)
        || !exact_contiguous(&input_nhwc, &[1, time, CHANNELS])
        || !exact_contiguous(&pointwise_weight_oik, &[CHANNELS, CHANNELS, 1])
        || !exact_contiguous(&pointwise_bias, &[CHANNELS])
        || !exact_contiguous(&residual_ncl, &[1, CHANNELS, time])
        || !exact_contiguous(&alpha, &[1, CHANNELS, 1])
        || !exact_contiguous(&head_weight_oik, &[1, CHANNELS, KERNEL_SIZE])
        || !exact_contiguous(&head_bias, &[1])
    {
        return None;
    }
    let tensors = [
        &pointwise_weight_oik,
        &pointwise_bias,
        &residual_ncl,
        &alpha,
        &head_weight_oik,
        &head_bias,
    ];
    if tensors
        .into_iter()
        .any(|tensor| tensor.device != input_nhwc.device)
    {
        return None;
    }
    let properties = input_nhwc.client.properties();
    let hardware = &properties.hardware;
    let dispatch_x = u32::try_from(time / TIME_TILE).ok()?;
    if hardware.max_bindings < REQUIRED_BINDINGS
        || hardware.max_shared_memory_size < SHARED_BYTES
        || hardware.max_units_per_cube < WORKGROUP_SIZE
        || hardware.max_cube_dim.0 < WORKGROUP_SIZE
        || hardware.max_cube_count.0 < dispatch_x
    {
        return None;
    }

    use burn_cubecl::ops::numeric::empty_device_dtype;
    let output: CubeTensor<WgpuRuntime> = empty_device_dtype(
        input_nhwc.client.clone(),
        input_nhwc.device.clone(),
        Shape::new([1, 1, time]),
        DType::F16,
    );
    let kernel = WmHeadPointwiseFusedKernel { time };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(kernel, CubeDim::new_1d(WORKGROUP_SIZE)));
    input_nhwc.client.launch(
        task,
        CubeCount::new_1d(dispatch_x),
        KernelArguments::new()
            .with_buffer(input_nhwc.handle.binding())
            .with_buffer(pointwise_weight_oik.handle.binding())
            .with_buffer(pointwise_bias.handle.binding())
            .with_buffer(residual_ncl.handle.binding())
            .with_buffer(alpha.handle.binding())
            .with_buffer(head_weight_oik.handle.binding())
            .with_buffer(head_bias.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_memory_contract_fits_webgpu_floor() {
        assert_eq!(SHARED_F16_ELEMENTS, 23_520);
        assert_eq!(SHARED_BYTES, 47_040);
        assert!(SHARED_BYTES <= 48 * 1_024);
    }
}
