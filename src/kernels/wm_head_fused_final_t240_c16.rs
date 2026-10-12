//! Production one-dispatch f32 fast path for the released DACVAE WmHead.
//!
//! The exact logical graph is `Snake1d -> Conv1d(96 -> 1, k=7, pad=3) ->
//! tanh` for `[1, 96, T]`, where `T` is a non-zero multiple of the 240-sample
//! tile. Every other logical shape, physical layout, dtype, device, binding
//! range, or resource limit is rejected before dispatch so the decoder can
//! retain its established WmHead fallbacks.
//!
//! On the RTX 3060 Ti with released weights and the strict f32 fixture, the
//! frozen isolated A/B measured 284.770 us median (278.751--318.691 us) versus
//! 1,376.077 us (1,373.967--1,382.923 us) for the prior production head, with
//! max absolute error 2.98e-7 and zero uncaptured WGPU errors.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::Shape;
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

use super::precision::{KernelFloatPrecision, common_float_precision};

pub const BATCH: usize = 1;
pub const INPUT_CHANNELS: usize = 96;
pub const OUTPUT_CHANNELS: usize = 1;
pub const TIME: usize = 96_000;
pub const KERNEL_SIZE: usize = 7;
pub const PADDING: usize = 3;
pub const TIME_TILE: usize = 240;
pub const INPUT_CHANNEL_TILE: usize = 16;
pub const INPUT_SPAN: usize = TIME_TILE + 2 * PADDING;
pub const INPUT_TILE_ELEMENTS: usize = INPUT_CHANNEL_TILE * INPUT_SPAN;
pub const WEIGHT_CACHE_ELEMENTS: usize = WEIGHT_ELEMENTS;
pub const ALPHA_CACHE_ELEMENTS: usize = INPUT_CHANNELS;
pub const SHARED_MEMORY_BYTES: usize =
    (INPUT_TILE_ELEMENTS + WEIGHT_CACHE_ELEMENTS + ALPHA_CACHE_ELEMENTS) * size_of::<f32>();
pub const WORKGROUP_SIZE: u32 = TIME_TILE as u32;
pub const DISPATCH_X: u32 = (TIME / TIME_TILE) as u32;
pub const REQUIRED_BINDINGS: u32 = 5;
pub const INPUT_ELEMENTS: usize = BATCH * INPUT_CHANNELS * TIME;
#[cfg(test)]
const F32_BYTES: usize = size_of::<f32>();
pub const ALPHA_ELEMENTS: usize = INPUT_CHANNELS;
pub const WEIGHT_ELEMENTS: usize = OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_SIZE;
pub const BIAS_ELEMENTS: usize = OUTPUT_CHANNELS;
pub const OUTPUT_ELEMENTS: usize = BATCH * OUTPUT_CHANNELS * TIME;
pub const INPUT_STRIDES: [usize; 3] = [INPUT_CHANNELS * TIME, TIME, 1];
pub const ALPHA_STRIDES: [usize; 3] = [INPUT_CHANNELS, 1, 1];
pub const WEIGHT_STRIDES: [usize; 3] = [INPUT_CHANNELS * KERNEL_SIZE, KERNEL_SIZE, 1];
pub const BIAS_STRIDES: [usize; 1] = [1];
pub const CONVENTIONAL_MACS: usize = OUTPUT_ELEMENTS * INPUT_CHANNELS * KERNEL_SIZE;
pub const SNAKE_EVALUATIONS: usize =
    DISPATCH_X as usize * (INPUT_CHANNELS / INPUT_CHANNEL_TILE) * INPUT_TILE_ELEMENTS;
pub const BARRIERS_PER_WORKGROUP: usize = 1 + 2 * (INPUT_CHANNELS / INPUT_CHANNEL_TILE);

const _: () = assert!(TIME.is_multiple_of(TIME_TILE));
const _: () = assert!(INPUT_CHANNELS.is_multiple_of(INPUT_CHANNEL_TILE));
const _: () = assert!(INPUT_SPAN == 246);
const _: () = assert!(INPUT_TILE_ELEMENTS == 3_936);
const _: () = assert!(SHARED_MEMORY_BYTES == 18_816);
const _: () = assert!(DISPATCH_X == 400);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DeviceLimits {
    max_bindings: u32,
    max_shared_memory_size: usize,
    max_cube_count: (u32, u32, u32),
    max_units_per_cube: u32,
    max_cube_dim: (u32, u32, u32),
    max_page_size: u64,
    memory_alignment: u64,
}

impl DeviceLimits {
    fn supports_released_head(self, time: usize, precision: KernelFloatPrecision) -> bool {
        let Some(input_elements) = BATCH
            .checked_mul(INPUT_CHANNELS)
            .and_then(|value| value.checked_mul(time))
        else {
            return false;
        };
        let Some(output_elements) = BATCH
            .checked_mul(OUTPUT_CHANNELS)
            .and_then(|value| value.checked_mul(time))
        else {
            return false;
        };
        let Ok(dispatch_x) = u32::try_from(time / TIME_TILE) else {
            return false;
        };
        let buffers_fit = [
            input_elements,
            ALPHA_ELEMENTS,
            WEIGHT_ELEMENTS,
            BIAS_ELEMENTS,
            output_elements,
        ]
        .into_iter()
        .all(|elements| {
            elements
                .checked_mul(precision.element_bytes())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .is_some_and(|bytes| bytes <= self.max_page_size)
        });

        self.memory_alignment >= precision.element_bytes() as u64
            && self
                .memory_alignment
                .is_multiple_of(precision.element_bytes() as u64)
            && self.max_bindings >= REQUIRED_BINDINGS
            && self.max_shared_memory_size >= SHARED_MEMORY_BYTES
            && self.max_units_per_cube >= WORKGROUP_SIZE
            && self.max_cube_dim.0 >= WORKGROUP_SIZE
            && self.max_cube_dim.1 >= 1
            && self.max_cube_dim.2 >= 1
            && self.max_cube_count.0 >= dispatch_x
            && self.max_cube_count.1 >= 1
            && self.max_cube_count.2 >= 1
            && buffers_fit
    }
}

#[derive(Debug)]
struct WmHeadFusedFinalT240C16Kernel {
    precision: KernelFloatPrecision,
    time: usize,
}

impl KernelSource for WmHeadFusedFinalT240C16Kernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("wm_head_fused_final_t240_c16.wgsl"),
                include_str!("wm_head_fused_final_t240_c16_f16.wgsl"),
            )
            .register("time", self.time.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.time))
    }
}

fn exact_shape(tensor: &CubeTensor<WgpuRuntime>, shape: &[usize]) -> bool {
    tensor.meta.shape().as_slice() == shape
}

fn exact_strides(tensor: &CubeTensor<WgpuRuntime>, strides: &[usize]) -> bool {
    &tensor.meta.strides()[..] == strides
}

fn binding_range_is_compatible(
    precision: KernelFloatPrecision,
    size_in_used: u64,
    offset_start: u64,
    elements: usize,
) -> bool {
    elements
        .checked_mul(precision.element_bytes())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .is_some_and(|bytes| {
            size_in_used >= bytes && offset_start.is_multiple_of(precision.element_bytes() as u64)
        })
}

fn tensor_binding_is_compatible(
    precision: KernelFloatPrecision,
    tensor: &CubeTensor<WgpuRuntime>,
    elements: usize,
) -> bool {
    let binding = tensor.handle.clone().binding();
    binding_range_is_compatible(
        precision,
        binding.size_in_used(),
        binding.offset_start.unwrap_or(0),
        elements,
    )
}

/// Complete fail-closed contract for the released production fast path.
pub fn contract_is_compatible(
    input: &CubeTensor<WgpuRuntime>,
    alpha: &CubeTensor<WgpuRuntime>,
    weight: &CubeTensor<WgpuRuntime>,
    bias: &CubeTensor<WgpuRuntime>,
) -> bool {
    let input_shape = input.meta.shape().as_slice();
    let time = input_shape.get(2).copied().unwrap_or(0);
    let Some(input_elements) = BATCH
        .checked_mul(INPUT_CHANNELS)
        .and_then(|v| v.checked_mul(time))
    else {
        return false;
    };
    let input_strides = [INPUT_CHANNELS * time, time, 1];
    let precision = common_float_precision([input.dtype, alpha.dtype, weight.dtype, bias.dtype]);
    let logical = precision.is_some()
        && input.meta.num_dims() == 3
        && alpha.meta.num_dims() == 3
        && weight.meta.num_dims() == 3
        && bias.meta.num_dims() == 1
        && time > 0
        && time.is_multiple_of(TIME_TILE)
        && exact_shape(input, &[BATCH, INPUT_CHANNELS, time])
        && exact_shape(alpha, &[BATCH, INPUT_CHANNELS, 1])
        && exact_shape(weight, &[OUTPUT_CHANNELS, INPUT_CHANNELS, KERNEL_SIZE])
        && exact_shape(bias, &[OUTPUT_CHANNELS])
        && exact_strides(input, &input_strides)
        && exact_strides(alpha, &ALPHA_STRIDES)
        && exact_strides(weight, &WEIGHT_STRIDES)
        && exact_strides(bias, &BIAS_STRIDES)
        && tensor_binding_is_compatible(precision.expect("dtype checked"), input, input_elements)
        && tensor_binding_is_compatible(precision.expect("dtype checked"), alpha, ALPHA_ELEMENTS)
        && tensor_binding_is_compatible(precision.expect("dtype checked"), weight, WEIGHT_ELEMENTS)
        && tensor_binding_is_compatible(precision.expect("dtype checked"), bias, BIAS_ELEMENTS)
        && input.is_contiguous()
        && alpha.is_contiguous()
        && weight.is_contiguous()
        && bias.is_contiguous()
        && input.device == alpha.device
        && input.device == weight.device
        && input.device == bias.device;
    if !logical {
        return false;
    }

    let properties = input.client.properties();
    let hardware = &properties.hardware;
    DeviceLimits {
        max_bindings: hardware.max_bindings,
        max_shared_memory_size: hardware.max_shared_memory_size,
        max_cube_count: hardware.max_cube_count,
        max_units_per_cube: hardware.max_units_per_cube,
        max_cube_dim: hardware.max_cube_dim,
        max_page_size: properties.memory.max_page_size,
        memory_alignment: properties.memory.alignment,
    }
    .supports_released_head(time, precision.expect("logical dtype was checked above"))
}

/// Run the exact released-head graph in one f32 dispatch.
///
/// `None` preserves a non-panicking fallback boundary for every unsupported
/// logical, physical, device, or resource condition.
pub fn try_wm_head_fused_final_t240_c16_wgsl(
    input: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    if !contract_is_compatible(&input, &alpha, &weight, &bias) {
        return None;
    }

    let time = input.meta.shape().dims::<3>()[2];
    let precision = KernelFloatPrecision::from_dtype(input.dtype)?;
    let output_elements = BATCH * OUTPUT_CHANNELS * time;
    let dispatch_x = u32::try_from(time / TIME_TILE).ok()?;
    let client = input.client.clone();
    let output_handle = client.empty(output_elements * precision.element_bytes());
    if !binding_range_is_compatible(
        precision,
        output_handle.size_in_used(),
        output_handle.offset_start.unwrap_or(0),
        output_elements,
    ) {
        return None;
    }
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([BATCH, OUTPUT_CHANNELS, time]),
        output_handle,
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            WmHeadFusedFinalT240C16Kernel { precision, time },
            CubeDim::new_1d(WORKGROUP_SIZE),
        ));
    client.launch(
        task,
        CubeCount::new_3d(dispatch_x, 1, 1),
        KernelArguments::new()
            .with_buffer(input.handle.binding())
            .with_buffer(alpha.handle.binding())
            .with_buffer(weight.handle.binding())
            .with_buffer(bias.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sufficient_limits() -> DeviceLimits {
        DeviceLimits {
            max_bindings: REQUIRED_BINDINGS,
            max_shared_memory_size: SHARED_MEMORY_BYTES,
            max_cube_count: (DISPATCH_X, 1, 1),
            max_units_per_cube: WORKGROUP_SIZE,
            max_cube_dim: (WORKGROUP_SIZE, 1, 1),
            max_page_size: (INPUT_ELEMENTS * size_of::<f32>()) as u64,
            memory_alignment: 256,
        }
    }

    #[test]
    fn exact_released_accounting() {
        assert_eq!(TIME_TILE, 240);
        assert_eq!(INPUT_CHANNEL_TILE, 16);
        assert_eq!(INPUT_SPAN, 246);
        assert_eq!(INPUT_TILE_ELEMENTS, 3_936);
        assert_eq!(WEIGHT_CACHE_ELEMENTS, 672);
        assert_eq!(SHARED_MEMORY_BYTES, 18_816);
        assert_eq!(DISPATCH_X, 400);
        assert_eq!(CONVENTIONAL_MACS, 64_512_000);
        assert_eq!(SNAKE_EVALUATIONS, 9_446_400);
        assert_eq!(BARRIERS_PER_WORKGROUP, 13);
        assert_eq!(INPUT_STRIDES, [9_216_000, 96_000, 1]);
        assert_eq!(ALPHA_STRIDES, [96, 1, 1]);
        assert_eq!(WEIGHT_STRIDES, [672, 7, 1]);
        assert_eq!(BIAS_STRIDES, [1]);
        assert!(sufficient_limits().supports_released_head(TIME, KernelFloatPrecision::F32));
    }

    #[test]
    fn every_resource_shortfall_rejects() {
        let sufficient = sufficient_limits();
        let unsupported = [
            DeviceLimits {
                max_bindings: REQUIRED_BINDINGS - 1,
                ..sufficient
            },
            DeviceLimits {
                max_shared_memory_size: SHARED_MEMORY_BYTES - 1,
                ..sufficient
            },
            DeviceLimits {
                max_cube_count: (DISPATCH_X - 1, 1, 1),
                ..sufficient
            },
            DeviceLimits {
                max_cube_count: (DISPATCH_X, 0, 1),
                ..sufficient
            },
            DeviceLimits {
                max_cube_count: (DISPATCH_X, 1, 0),
                ..sufficient
            },
            DeviceLimits {
                max_units_per_cube: WORKGROUP_SIZE - 1,
                ..sufficient
            },
            DeviceLimits {
                max_cube_dim: (WORKGROUP_SIZE - 1, 1, 1),
                ..sufficient
            },
            DeviceLimits {
                max_cube_dim: (WORKGROUP_SIZE, 0, 1),
                ..sufficient
            },
            DeviceLimits {
                max_cube_dim: (WORKGROUP_SIZE, 1, 0),
                ..sufficient
            },
            DeviceLimits {
                max_page_size: (INPUT_ELEMENTS * size_of::<f32>() - 1) as u64,
                ..sufficient
            },
            DeviceLimits {
                memory_alignment: 2,
                ..sufficient
            },
        ];
        assert!(
            unsupported
                .into_iter()
                .all(|limits| { !limits.supports_released_head(TIME, KernelFloatPrecision::F32) })
        );
    }

    #[test]
    fn physical_binding_range_is_fail_closed() {
        let exact_input_bytes = (INPUT_ELEMENTS * F32_BYTES) as u64;
        assert!(binding_range_is_compatible(
            KernelFloatPrecision::F32,
            exact_input_bytes,
            0,
            INPUT_ELEMENTS
        ));
        assert!(binding_range_is_compatible(
            KernelFloatPrecision::F32,
            exact_input_bytes + 256,
            256,
            INPUT_ELEMENTS
        ));
        assert!(!binding_range_is_compatible(
            KernelFloatPrecision::F32,
            exact_input_bytes - 1,
            0,
            INPUT_ELEMENTS
        ));
        assert!(!binding_range_is_compatible(
            KernelFloatPrecision::F32,
            exact_input_bytes,
            2,
            INPUT_ELEMENTS
        ));
    }

    #[test]
    fn t240_cin16_indices_equal_same_padding_convolution() {
        for output_time in [0, 1, 3, 239, 240, TIME - 2, TIME - 1] {
            let group = output_time / TIME_TILE;
            let local_time = output_time % TIME_TILE;
            for input_channel in [0, 15, 16, INPUT_CHANNELS - 1] {
                let channel_tile = input_channel / INPUT_CHANNEL_TILE;
                let tile_channel = input_channel % INPUT_CHANNEL_TILE;
                assert_eq!(
                    channel_tile * INPUT_CHANNEL_TILE + tile_channel,
                    input_channel
                );
                for kernel_index in 0..KERNEL_SIZE {
                    let staged_time = group * TIME_TILE + local_time + kernel_index;
                    let shader_source_time = staged_time as isize - PADDING as isize;
                    let reference_source_time =
                        output_time as isize + kernel_index as isize - PADDING as isize;
                    assert_eq!(shader_source_time, reference_source_time);
                    assert!(local_time + kernel_index < INPUT_SPAN);
                }
            }
        }
    }

    #[test]
    fn shader_keeps_exact_operation_and_padding_order() {
        let shader = include_str!("wm_head_fused_final_t240_c16.wgsl");
        let production_snake = include_str!("snake.wgsl");
        for expression in [
            "const TIME: u32 = {{ time }}u;",
            "let sine = sin(a * x);",
            "activated = x + (sine * sine) / (a + 1e-9);",
            "var accumulator = bias[0u];",
            "accumulator = fma(",
            "output_ncl[output_time] = tanh(accumulator);",
            "@compute @workgroup_size(240, 1, 1)",
        ] {
            assert!(shader.contains(expression), "missing {expression:?}");
        }
        assert!(production_snake.contains("let sine = sin(a * x);"));
        assert!(production_snake.contains("output[index] = x + (sine * sine) / (a + 1e-9);"));
        assert_eq!(shader.matches("workgroupBarrier();").count(), 3);
        assert_eq!(shader.matches("var<storage, read_write>").count(), 5);
        assert!(!shader.contains("var<storage, read>"));
        let bias = shader.find("var accumulator = bias[0u]").unwrap();
        let channel = shader.find("var input_channel_base = 0u").unwrap();
        let kernel = shader.find("var kernel_index = 0u").unwrap();
        let fma = shader.find("accumulator = fma(").unwrap();
        let tanh = shader.find("tanh(accumulator)").unwrap();
        assert!(bias < channel && channel < kernel && kernel < fma && fma < tanh);
    }
}
