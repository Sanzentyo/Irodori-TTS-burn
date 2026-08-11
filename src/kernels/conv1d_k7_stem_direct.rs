//! Exact released-decoder stem T64/O32/Cin16 f32 convolution.
//!
//! # Selection evidence
//!
//! A production-weight isolated A/B on the released `[1, 1024, 50]` stem
//! (10 warmups, 100 executions per trial, five rotating trials) measured the
//! unchanged Burn path at 7,139.149 us median `[7,135.982, 7,197.141]`, an
//! im2col plus tuned-GEMM candidate at 1,335.017 us
//! `[1,334.308, 1,350.447]`, and this direct candidate at 1,036.779 us
//! `[1,024.130, 1,040.514]`. Both candidates were finite and had the same full
//! 76,800-element output hash; versus Burn they had max/mean absolute error
//! `1.907348633e-5` / `3.144332065e-7`. The im2col candidate was rejected as
//! strictly slower and its transient pack/finalizer implementation was removed.
//! These isolated accuracy numbers are screening evidence only; production
//! adoption additionally requires the full decoder waveform gates.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime, into_contiguous,
};
use burn::tensor::{DType, Shape};
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

const BATCH: usize = 1;
const INPUT_CHANNELS: usize = 1_024;
const OUTPUT_CHANNELS: usize = 1_536;
const LENGTH: usize = 50;
const KERNEL_SIZE: usize = 7;
const WORKGROUP_SIZE: u32 = 256;
const LOCAL_TIME_LANES: u32 = 16;
const LOCAL_CHANNEL_LANES: u32 = 16;
const OUTPUT_CHANNEL_TILE: u32 = 32;
const OUTPUT_CHANNEL_WORKGROUPS: u32 = OUTPUT_CHANNELS as u32 / OUTPUT_CHANNEL_TILE;
const REQUIRED_BINDINGS: u32 = 4;
const INPUT_ELEMENTS: usize = BATCH * INPUT_CHANNELS * LENGTH;
const WEIGHT_ELEMENTS: usize = OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_SIZE;
const BIAS_ELEMENTS: usize = OUTPUT_CHANNELS;
const OUTPUT_ELEMENTS: usize = BATCH * OUTPUT_CHANNELS * LENGTH;
pub const SHARED_MEMORY_BYTES: usize = (1_120 + 3_584) * size_of::<f32>();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DeviceLimits {
    max_bindings: u32,
    max_shared_memory_size: usize,
    max_cube_count: (u32, u32, u32),
    max_units_per_cube: u32,
    max_cube_dim: (u32, u32, u32),
    max_page_size: u64,
}

impl DeviceLimits {
    fn supports_released_stem(self) -> bool {
        let buffers_fit = [
            INPUT_ELEMENTS,
            WEIGHT_ELEMENTS,
            BIAS_ELEMENTS,
            OUTPUT_ELEMENTS,
        ]
        .into_iter()
        .all(|elements| {
            elements
                .checked_mul(size_of::<f32>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .is_some_and(|bytes| bytes <= self.max_page_size)
        });

        self.max_bindings >= REQUIRED_BINDINGS
            && self.max_shared_memory_size >= SHARED_MEMORY_BYTES
            && self.max_units_per_cube >= WORKGROUP_SIZE
            && self.max_cube_dim.0 >= LOCAL_TIME_LANES
            && self.max_cube_dim.1 >= LOCAL_CHANNEL_LANES
            && self.max_cube_dim.2 >= 1
            && self.max_cube_count.0 >= 1
            && self.max_cube_count.1 >= OUTPUT_CHANNEL_WORKGROUPS
            && self.max_cube_count.2 >= BATCH as u32
            && buffers_fit
    }
}

#[derive(Debug)]
struct StemDirectKernel;

impl KernelSource for StemDirectKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("conv1d_k7_stem_direct.wgsl"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

/// Check the complete non-panicking production launch contract.
///
/// The input may be a logical view because the launcher materializes it before
/// dispatch. Checkpoint-native OIK weight and bias storage must already be
/// contiguous, f32, and on the same device as the exact released input.
pub fn stem_direct_contract_is_compatible(
    input: &CubeTensor<WgpuRuntime>,
    weight: &CubeTensor<WgpuRuntime>,
    bias: &CubeTensor<WgpuRuntime>,
) -> bool {
    if input.meta.num_dims() != 3 || weight.meta.num_dims() != 3 || bias.meta.num_dims() != 1 {
        return false;
    }
    let logical = input.dtype == DType::F32
        && weight.dtype == DType::F32
        && bias.dtype == DType::F32
        && input.meta.shape().as_slice() == [BATCH, INPUT_CHANNELS, LENGTH]
        && weight.meta.shape().as_slice() == [OUTPUT_CHANNELS, INPUT_CHANNELS, KERNEL_SIZE]
        && bias.meta.shape().as_slice() == [OUTPUT_CHANNELS]
        && weight.is_contiguous()
        && bias.is_contiguous()
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
    }
    .supports_released_stem()
}

/// Try the released direct stem, returning `None` for every unsupported
/// shape/layout/device/resource condition so the caller can use Burn fallback.
pub fn try_conv1d_k7_stem_direct_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    if !stem_direct_contract_is_compatible(&input, &weight, &bias) {
        return None;
    }

    let input = into_contiguous(input);
    let client = input.client.clone();
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([BATCH, OUTPUT_CHANNELS, LENGTH]),
        client.empty(OUTPUT_ELEMENTS * size_of::<f32>()),
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            StemDirectKernel,
            CubeDim::new_2d(LOCAL_TIME_LANES, LOCAL_CHANNEL_LANES),
        ));
    client.launch(
        task,
        CubeCount::new_3d(1, OUTPUT_CHANNEL_WORKGROUPS, BATCH as u32),
        KernelArguments::new()
            .with_buffer(input.handle.binding())
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
            max_cube_count: (1, OUTPUT_CHANNEL_WORKGROUPS, BATCH as u32),
            max_units_per_cube: WORKGROUP_SIZE,
            max_cube_dim: (LOCAL_TIME_LANES, LOCAL_CHANNEL_LANES, 1),
            max_page_size: (WEIGHT_ELEMENTS * size_of::<f32>()) as u64,
        }
    }

    #[test]
    fn released_stem_static_accounting_is_exact() {
        assert_eq!(SHARED_MEMORY_BYTES, 18_816);
        assert_eq!(OUTPUT_CHANNEL_WORKGROUPS, 48);
        assert_eq!(64 - LENGTH, 14, "T64 has fourteen guarded tails");
        assert_eq!(WEIGHT_ELEMENTS, 11_010_048);
        assert_eq!(OUTPUT_ELEMENTS, 76_800);
        assert!(sufficient_limits().supports_released_stem());
    }

    #[test]
    fn released_stem_rejects_each_device_or_resource_shortfall() {
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
                max_cube_count: (0, OUTPUT_CHANNEL_WORKGROUPS, 1),
                ..sufficient
            },
            DeviceLimits {
                max_cube_count: (1, OUTPUT_CHANNEL_WORKGROUPS - 1, 1),
                ..sufficient
            },
            DeviceLimits {
                max_cube_count: (1, OUTPUT_CHANNEL_WORKGROUPS, 0),
                ..sufficient
            },
            DeviceLimits {
                max_units_per_cube: WORKGROUP_SIZE - 1,
                ..sufficient
            },
            DeviceLimits {
                max_cube_dim: (LOCAL_TIME_LANES - 1, LOCAL_CHANNEL_LANES, 1),
                ..sufficient
            },
            DeviceLimits {
                max_cube_dim: (LOCAL_TIME_LANES, LOCAL_CHANNEL_LANES - 1, 1),
                ..sufficient
            },
            DeviceLimits {
                max_cube_dim: (LOCAL_TIME_LANES, LOCAL_CHANNEL_LANES, 0),
                ..sufficient
            },
            DeviceLimits {
                max_page_size: (WEIGHT_ELEMENTS * size_of::<f32>() - 1) as u64,
                ..sufficient
            },
        ];
        assert!(
            unsupported
                .into_iter()
                .all(|limits| !limits.supports_released_stem())
        );
    }

    #[test]
    fn shader_preserves_checkpoint_oik_and_bias_last_order() {
        let source = include_str!("conv1d_k7_stem_direct.wgsl");
        assert!(source.contains("(output_channel * INPUT_CHANNELS + input_channel)"));
        assert_eq!(source.matches("var accumulator_").count(), 8);
        for suffix in ["00", "01", "10", "11", "20", "21", "30", "31"] {
            assert!(source.contains(&format!("var accumulator_{suffix} = 0.0;")));
        }
        assert_eq!(source.matches("+ bias_buf[output_channel_").count(), 8);
        assert!(source.contains("if (time_3 < LENGTH)"));
    }
}
