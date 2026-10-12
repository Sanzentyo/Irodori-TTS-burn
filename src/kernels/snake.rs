//! Fused Snake1d activation for the production WGPU codec path.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime, into_contiguous,
};
use burn::tensor::Shape;
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

use super::precision::{KernelFloatPrecision, common_float_precision};

const WORKGROUP_SIZE: u32 = 256;

#[derive(Debug)]
struct SnakeKernel {
    precision: KernelFloatPrecision,
    channels: u32,
    time: u32,
    elements: u32,
    dispatch_x: u32,
}

#[derive(Debug)]
struct SnakeNhwcToNchwKernel {
    precision: KernelFloatPrecision,
    batch: u32,
    channels: u32,
    time: u32,
}

#[derive(Debug)]
struct SnakeNhwcKernel {
    precision: KernelFloatPrecision,
    channels: u32,
    time: u32,
    elements: u32,
    dispatch_x: u32,
}

#[derive(Debug)]
struct SnakeNchwToNhwcKernel {
    precision: KernelFloatPrecision,
    batch: u32,
    channels: u32,
    time: u32,
}

impl KernelSource for SnakeKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(include_str!("snake.wgsl"), include_str!("snake_f16.wgsl"))
            .register("channels", self.channels.to_string())
            .register("time", self.time.to_string())
            .register("elements", self.elements.to_string())
            .register("dispatch_x", self.dispatch_x.to_string())
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.precision,
            self.channels,
            self.time,
            self.elements,
            self.dispatch_x,
        ))
    }
}

impl KernelSource for SnakeNhwcToNchwKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("snake_nhwc_to_nchw.wgsl"),
                include_str!("snake_nhwc_to_nchw_f16.wgsl"),
            )
            .register("batch", self.batch.to_string())
            .register("channels", self.channels.to_string())
            .register("time", self.time.to_string())
            .register("tile", "32")
            .register("tile_stride", "33")
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.batch, self.channels, self.time))
    }
}

impl KernelSource for SnakeNhwcKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("snake_nhwc.wgsl"),
                include_str!("snake_nhwc_f16.wgsl"),
            )
            .register("channels", self.channels.to_string())
            .register("time", self.time.to_string())
            .register("elements", self.elements.to_string())
            .register("dispatch_x", self.dispatch_x.to_string())
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.precision,
            self.channels,
            self.time,
            self.elements,
            self.dispatch_x,
        ))
    }
}

impl KernelSource for SnakeNchwToNhwcKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("snake_nchw_to_nhwc.wgsl"),
                include_str!("snake_nchw_to_nhwc_f16.wgsl"),
            )
            .register("batch", self.batch.to_string())
            .register("channels", self.channels.to_string())
            .register("time", self.time.to_string())
            .register("tile", "32")
            .register("tile_stride", "33")
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.batch, self.channels, self.time))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Dispatch2d {
    x: u32,
    y: u32,
}

fn linear_workgroups_to_2d(workgroups: u32, max_cube_count: (u32, u32, u32)) -> Option<Dispatch2d> {
    if workgroups == 0 || max_cube_count.0 == 0 || max_cube_count.1 == 0 {
        return None;
    }
    let x = workgroups.min(max_cube_count.0);
    let y = workgroups.div_ceil(x);
    (y <= max_cube_count.1).then_some(Dispatch2d { x, y })
}

/// Apply `x + sin(alpha * x)^2 / (alpha + 1e-9)` in one dispatch.
///
/// `input` and `alpha` must use the same f32 or f16 storage precision.
pub fn snake_wgsl(
    input: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
) -> CubeTensor<WgpuRuntime> {
    let precision = common_float_precision([input.dtype, alpha.dtype])
        .expect("Snake input and alpha must share f32 or f16 dtype");
    assert_eq!(input.meta.num_dims(), 3, "Snake input must be rank 3");
    assert_eq!(alpha.meta.num_dims(), 3, "Snake alpha must be rank 3");

    let input = into_contiguous(input);
    let alpha = into_contiguous(alpha);
    let input_shape = input.meta.shape();
    let [batch, channels, time] = [input_shape[0], input_shape[1], input_shape[2]];
    let alpha_shape = alpha.meta.shape();
    assert_eq!(
        [alpha_shape[0], alpha_shape[1], alpha_shape[2]],
        [1, channels, 1],
        "Snake alpha shape must be [1, channels, 1]"
    );

    let elements = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(time))
        .expect("Snake output size overflow");
    let client = input.client.clone();
    let output_handle = client.empty(elements * precision.element_bytes());
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([batch, channels, time]),
        output_handle,
        precision.dtype(),
    );

    let channels = u32::try_from(channels).expect("Snake channels exceed u32");
    let time = u32::try_from(time).expect("Snake time exceeds u32");
    let elements = u32::try_from(elements).expect("Snake element count exceeds u32");
    let workgroups = elements.div_ceil(WORKGROUP_SIZE);
    let max_cube_count = client.properties().hardware.max_cube_count;
    let dispatch = linear_workgroups_to_2d(workgroups, max_cube_count).unwrap_or_else(|| {
        panic!(
            "Snake requires {workgroups} linear workgroups, device dispatch limit is {max_cube_count:?}"
        )
    });
    let kernel = SnakeKernel {
        precision,
        channels,
        time,
        elements,
        dispatch_x: dispatch.x,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(kernel, CubeDim::new_1d(WORKGROUP_SIZE)));
    let bindings = KernelArguments::new()
        .with_buffer(input.handle.binding())
        .with_buffer(alpha.handle.binding())
        .with_buffer(output.handle.clone().binding());
    client.launch(task, CubeCount::new_2d(dispatch.x, dispatch.y), bindings);
    output
}

/// Apply Snake while materializing a contiguous NHWC input as contiguous NCHW.
///
/// This is the layout boundary produced by CubeCL implicit-GEMM convolution.
/// Folding the transpose into the elementwise activation avoids one complete
/// intermediate allocation and one dispatch without changing accumulation or
/// activation precision.
pub fn snake_nhwc_to_nchw_wgsl(
    input_nhwc: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    let precision = common_float_precision([input_nhwc.dtype, alpha.dtype])?;
    if input_nhwc.meta.num_dims() != 3 || alpha.meta.num_dims() != 3 || !input_nhwc.is_contiguous()
    {
        return None;
    }
    let input_shape = input_nhwc.meta.shape();
    let [batch, time, channels] = [input_shape[0], input_shape[1], input_shape[2]];
    let alpha = into_contiguous(alpha);
    let alpha_shape = alpha.meta.shape();
    if [alpha_shape[0], alpha_shape[1], alpha_shape[2]] != [1, channels, 1] {
        return None;
    }

    let elements = batch.checked_mul(channels)?.checked_mul(time)?;
    let client = input_nhwc.client.clone();
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input_nhwc.device.clone(),
        Shape::from([batch, channels, time]),
        client.empty(elements.checked_mul(precision.element_bytes())?),
        precision.dtype(),
    );

    let batch = u32::try_from(batch).ok()?;
    let channels = u32::try_from(channels).ok()?;
    let time = u32::try_from(time).ok()?;
    let dispatch_time = time.div_ceil(32);
    let dispatch_channels = channels.div_ceil(32);
    let limits = client.properties().hardware.max_cube_count;
    if dispatch_time > limits.0 || dispatch_channels > limits.1 || batch > limits.2 {
        return None;
    }
    let kernel = SnakeNhwcToNchwKernel {
        precision,
        batch,
        channels,
        time,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(kernel, CubeDim::new_2d(32, 8)));
    client.launch(
        task,
        CubeCount::new_3d(dispatch_time, dispatch_channels, batch),
        KernelArguments::new()
            .with_buffer(input_nhwc.handle.binding())
            .with_buffer(alpha.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

/// Apply Snake to contiguous NHWC storage without changing its layout.
pub fn snake_nhwc_wgsl(
    input_nhwc: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    let precision = common_float_precision([input_nhwc.dtype, alpha.dtype])?;
    if input_nhwc.meta.num_dims() != 3 || alpha.meta.num_dims() != 3 || !input_nhwc.is_contiguous()
    {
        return None;
    }
    let [batch, time, channels] = input_nhwc.meta.shape().dims::<3>();
    let alpha = into_contiguous(alpha);
    if alpha.meta.shape().dims::<3>() != [1, channels, 1] {
        return None;
    }
    let elements = batch.checked_mul(time)?.checked_mul(channels)?;
    let client = input_nhwc.client.clone();
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input_nhwc.device.clone(),
        Shape::from([batch, time, channels]),
        client.empty(elements.checked_mul(precision.element_bytes())?),
        precision.dtype(),
    );
    let elements = u32::try_from(elements).ok()?;
    let workgroups = elements.div_ceil(WORKGROUP_SIZE);
    let dispatch =
        linear_workgroups_to_2d(workgroups, client.properties().hardware.max_cube_count)?;
    let kernel = SnakeNhwcKernel {
        precision,
        channels: u32::try_from(channels).ok()?,
        time: u32::try_from(time).ok()?,
        elements,
        dispatch_x: dispatch.x,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(kernel, CubeDim::new_1d(WORKGROUP_SIZE)));
    client.launch(
        task,
        CubeCount::new_2d(dispatch.x, dispatch.y),
        KernelArguments::new()
            .with_buffer(input_nhwc.handle.binding())
            .with_buffer(alpha.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

/// Apply Snake while materializing contiguous NCHW as contiguous NHWC.
///
/// The result can be consumed directly by CubeCL implicit-GEMM convolution,
/// avoiding its otherwise mandatory input-layout materialization.
pub fn snake_nchw_to_nhwc_wgsl(
    input_nchw: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    let precision = common_float_precision([input_nchw.dtype, alpha.dtype])?;
    if input_nchw.meta.num_dims() != 3 || alpha.meta.num_dims() != 3 || !input_nchw.is_contiguous()
    {
        return None;
    }
    let input_shape = input_nchw.meta.shape();
    let [batch, channels, time] = [input_shape[0], input_shape[1], input_shape[2]];
    let alpha = into_contiguous(alpha);
    let alpha_shape = alpha.meta.shape();
    if [alpha_shape[0], alpha_shape[1], alpha_shape[2]] != [1, channels, 1] {
        return None;
    }

    let elements = batch.checked_mul(channels)?.checked_mul(time)?;
    let client = input_nchw.client.clone();
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input_nchw.device.clone(),
        Shape::from([batch, time, channels]),
        client.empty(elements.checked_mul(precision.element_bytes())?),
        precision.dtype(),
    );
    let batch = u32::try_from(batch).ok()?;
    let channels = u32::try_from(channels).ok()?;
    let time = u32::try_from(time).ok()?;
    let dispatch_time = time.div_ceil(32);
    let dispatch_channels = channels.div_ceil(32);
    let limits = client.properties().hardware.max_cube_count;
    if dispatch_time > limits.0 || dispatch_channels > limits.1 || batch > limits.2 {
        return None;
    }
    let kernel = SnakeNchwToNhwcKernel {
        precision,
        batch,
        channels,
        time,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(kernel, CubeDim::new_2d(32, 8)));
    client.launch(
        task,
        CubeCount::new_3d(dispatch_time, dispatch_channels, batch),
        KernelArguments::new()
            .with_buffer(input_nchw.handle.binding())
            .with_buffer(alpha.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::{Dispatch2d, linear_workgroups_to_2d};

    const VULKAN_MIN_COUNTS: (u32, u32, u32) = (65_535, 65_535, 65_535);

    #[test]
    fn keeps_short_dispatch_on_one_row() {
        assert_eq!(
            linear_workgroups_to_2d(36_000, VULKAN_MIN_COUNTS),
            Some(Dispatch2d { x: 36_000, y: 1 })
        );
    }

    #[test]
    fn folds_four_and_eight_second_codec_shapes_across_y() {
        assert_eq!(
            linear_workgroups_to_2d(72_000, VULKAN_MIN_COUNTS),
            Some(Dispatch2d { x: 65_535, y: 2 })
        );
        assert_eq!(
            linear_workgroups_to_2d(144_000, VULKAN_MIN_COUNTS),
            Some(Dispatch2d { x: 65_535, y: 3 })
        );
    }

    #[test]
    fn rejects_unsupported_or_empty_dispatches() {
        assert_eq!(linear_workgroups_to_2d(0, VULKAN_MIN_COUNTS), None);
        assert_eq!(linear_workgroups_to_2d(65_536, (65_535, 1, 1)), None);
        assert_eq!(linear_workgroups_to_2d(1, (0, 65_535, 1)), None);
    }
}
