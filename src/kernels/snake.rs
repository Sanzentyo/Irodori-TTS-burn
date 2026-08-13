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
