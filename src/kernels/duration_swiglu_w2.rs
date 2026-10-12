//! Fused SwiGLU activation and `w2` projection for released duration shapes.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::Shape;
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

use super::precision::{KernelFloatPrecision, common_float_precision};

const DIM: usize = 1024;
const MAX_SEQUENCE: usize = 64;
const TILE_ROWS: usize = 16;
const WORKGROUP_Y: u32 = 8;
const REQUIRED_BINDINGS: u32 = 3;
const LONG_SEQUENCE_MIN: usize = 48;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DurationSwiGluW2Variant {
    O32Scalar,
    O64Vec4,
}

impl DurationSwiGluW2Variant {
    const fn tile_outputs(self) -> usize {
        match self {
            Self::O32Scalar => 32,
            Self::O64Vec4 => 64,
        }
    }

    const fn workgroup_x(self) -> u32 {
        match self {
            Self::O32Scalar => 32,
            Self::O64Vec4 => 16,
        }
    }

    const fn tile_k(self) -> usize {
        match self {
            Self::O32Scalar => 32,
            Self::O64Vec4 => 128,
        }
    }

    const fn shared_bytes(self) -> usize {
        (TILE_ROWS * self.tile_k() + self.tile_k() * self.tile_outputs()) * size_of::<f32>()
    }
}

const fn variant_for_sequence(sequence: usize) -> DurationSwiGluW2Variant {
    if sequence >= LONG_SEQUENCE_MIN {
        DurationSwiGluW2Variant::O64Vec4
    } else {
        DurationSwiGluW2Variant::O32Scalar
    }
}

const fn row_workgroups(sequence: usize) -> u32 {
    (sequence as u32).div_ceil(TILE_ROWS as u32)
}

#[derive(Debug)]
struct DurationSwiGluW2Kernel {
    precision: KernelFloatPrecision,
    sequence: u32,
    variant: DurationSwiGluW2Variant,
}

#[derive(Debug)]
struct DurationSwiGluW2ResidualKernel {
    precision: KernelFloatPrecision,
    sequence: u32,
}

impl KernelSource for DurationSwiGluW2ResidualKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("duration_swiglu_w2_o64_vec4_residual.wgsl"),
                include_str!("duration_swiglu_w2_o64_vec4_residual_f16.wgsl"),
            )
            .register("sequence", self.sequence.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.sequence))
    }
}

impl KernelSource for DurationSwiGluW2Kernel {
    fn source(&self) -> SourceTemplate {
        let (f32_source, f16_source) = match self.variant {
            DurationSwiGluW2Variant::O32Scalar => (
                include_str!("duration_swiglu_w2.wgsl"),
                include_str!("duration_swiglu_w2_f16.wgsl"),
            ),
            DurationSwiGluW2Variant::O64Vec4 => (
                include_str!("duration_swiglu_w2_o64_vec4.wgsl"),
                include_str!("duration_swiglu_w2_o64_vec4_f16.wgsl"),
            ),
        };
        self.precision
            .source(f32_source, f16_source)
            .register("sequence", self.sequence.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.sequence, self.variant))
    }
}

/// Consume contiguous `[S, 2048]` `w1||w3` output and row-major `[1024,
/// 1024]` `w2` without materializing the `[S, 1024]` activation tensor.
pub fn try_duration_swiglu_w2_wgsl(
    projected: CubeTensor<WgpuRuntime>,
    w2: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    if projected.meta.num_dims() != 2 {
        return None;
    }
    let sequence = projected.meta.shape()[0];
    let precision = common_float_precision([projected.dtype, w2.dtype])?;
    let compatible = (1..=MAX_SEQUENCE).contains(&sequence)
        && projected.meta.shape().as_slice() == [sequence, DIM * 2]
        && w2.meta.shape().as_slice() == [DIM, DIM]
        && projected.meta.strides()[..] == [DIM * 2, 1]
        && w2.meta.strides()[..] == [DIM, 1]
        && projected.is_contiguous()
        && w2.is_contiguous()
        && projected.device == w2.device;
    if !compatible {
        return None;
    }
    let variant = variant_for_sequence(sequence);
    let workgroup_x = variant.workgroup_x();
    let hardware = &projected.client.properties().hardware;
    if hardware.max_bindings < REQUIRED_BINDINGS
        || hardware.max_shared_memory_size < variant.shared_bytes()
        || hardware.max_units_per_cube < workgroup_x * WORKGROUP_Y
        || hardware.max_cube_dim.0 < workgroup_x
        || hardware.max_cube_dim.1 < WORKGROUP_Y
    {
        return None;
    }

    let client = projected.client.clone();
    let output_handle = client.empty(sequence * DIM * precision.element_bytes());
    let output = CubeTensor::new_contiguous(
        client.clone(),
        projected.device.clone(),
        Shape::from([1, sequence, DIM]),
        output_handle,
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DurationSwiGluW2Kernel {
                precision,
                sequence: sequence as u32,
                variant,
            },
            CubeDim::new_2d(workgroup_x, WORKGROUP_Y),
        ));
    client.launch(
        task,
        CubeCount::new_2d(
            (DIM as u32).div_ceil(variant.tile_outputs() as u32),
            row_workgroups(sequence),
        ),
        KernelArguments::new()
            .with_buffer(projected.handle.binding())
            .with_buffer(w2.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

/// Long-text contraction with `residual + gate * branch` in the final store.
pub fn try_duration_swiglu_w2_residual_wgsl(
    projected: CubeTensor<WgpuRuntime>,
    w2: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    if projected.meta.num_dims() != 2 || residual.meta.num_dims() != 3 {
        return None;
    }
    let sequence = projected.meta.shape()[0];
    let precision =
        common_float_precision([projected.dtype, w2.dtype, residual.dtype, gate.dtype])?;
    let compatible = (LONG_SEQUENCE_MIN..=MAX_SEQUENCE).contains(&sequence)
        && projected.meta.shape().as_slice() == [sequence, DIM * 2]
        && w2.meta.shape().as_slice() == [DIM, DIM]
        && residual.meta.shape().as_slice() == [1, sequence, DIM]
        && gate.meta.shape().as_slice() == [1, 1, DIM]
        && projected.meta.strides()[..] == [DIM * 2, 1]
        && w2.meta.strides()[..] == [DIM, 1]
        && residual.meta.strides()[..] == [sequence * DIM, DIM, 1]
        && gate.meta.strides()[..] == [DIM, DIM, 1]
        && projected.is_contiguous()
        && w2.is_contiguous()
        && residual.is_contiguous()
        && gate.is_contiguous()
        && projected.device == w2.device
        && projected.device == residual.device
        && projected.device == gate.device;
    if !compatible {
        return None;
    }
    let variant = DurationSwiGluW2Variant::O64Vec4;
    let hardware = &projected.client.properties().hardware;
    if hardware.max_bindings < 5
        || hardware.max_shared_memory_size < variant.shared_bytes()
        || hardware.max_units_per_cube < variant.workgroup_x() * WORKGROUP_Y
        || hardware.max_cube_dim.0 < variant.workgroup_x()
        || hardware.max_cube_dim.1 < WORKGROUP_Y
    {
        return None;
    }

    let client = projected.client.clone();
    let output_handle = client.empty(sequence * DIM * precision.element_bytes());
    let output = CubeTensor::new_contiguous(
        client.clone(),
        projected.device.clone(),
        Shape::from([1, sequence, DIM]),
        output_handle,
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DurationSwiGluW2ResidualKernel {
                precision,
                sequence: sequence as u32,
            },
            CubeDim::new_2d(variant.workgroup_x(), WORKGROUP_Y),
        ));
    client.launch(
        task,
        CubeCount::new_2d(
            (DIM as u32).div_ceil(variant.tile_outputs() as u32),
            row_workgroups(sequence),
        ),
        KernelArguments::new()
            .with_buffer(projected.handle.binding())
            .with_buffer(w2.handle.binding())
            .with_buffer(residual.handle.binding())
            .with_buffer(gate.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_tile_accounting_and_selection_are_bounded() {
        assert_eq!(DurationSwiGluW2Variant::O32Scalar.workgroup_x(), 32);
        assert_eq!(DurationSwiGluW2Variant::O32Scalar.shared_bytes(), 6144);
        assert_eq!(DurationSwiGluW2Variant::O64Vec4.workgroup_x(), 16);
        assert_eq!(DurationSwiGluW2Variant::O64Vec4.tile_outputs(), 64);
        assert_eq!(DurationSwiGluW2Variant::O64Vec4.tile_k(), 128);
        assert_eq!(DurationSwiGluW2Variant::O64Vec4.shared_bytes(), 40960);
        assert_eq!(REQUIRED_BINDINGS, 3);
        assert_eq!(DurationSwiGluW2Variant::O32Scalar.tile_k(), 32);
        assert_eq!(DIM / DurationSwiGluW2Variant::O32Scalar.tile_k(), 32);
        assert_eq!(variant_for_sequence(3), DurationSwiGluW2Variant::O32Scalar);
        assert_eq!(variant_for_sequence(28), DurationSwiGluW2Variant::O32Scalar);
        assert_eq!(variant_for_sequence(47), DurationSwiGluW2Variant::O32Scalar);
        assert_eq!(variant_for_sequence(48), DurationSwiGluW2Variant::O64Vec4);
        assert_eq!(variant_for_sequence(61), DurationSwiGluW2Variant::O64Vec4);
        assert_eq!(row_workgroups(3), 1);
        assert_eq!(row_workgroups(12), 1);
        assert_eq!(row_workgroups(28), 2);
        assert_eq!(row_workgroups(61), 4);
        assert_eq!(
            include_str!("duration_swiglu_w2_o64_vec4_residual.wgsl")
                .matches("var<storage, read_write>")
                .count(),
            5
        );
    }
}
