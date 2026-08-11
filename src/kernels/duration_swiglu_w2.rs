//! Fused SwiGLU activation and `w2` projection for released duration shapes.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::{DType, Shape};
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

const DIM: usize = 1024;
const MAX_SEQUENCE: usize = 64;
const TILE_ROWS: usize = 16;
const TILE_OUTPUTS: usize = 32;
const TILE_K: usize = 32;
const WORKGROUP_X: u32 = TILE_OUTPUTS as u32;
const WORKGROUP_Y: u32 = 8;
const REQUIRED_BINDINGS: u32 = 3;
const SHARED_BYTES: usize = (TILE_ROWS * TILE_K + TILE_K * TILE_OUTPUTS) * size_of::<f32>();

const fn row_workgroups(sequence: usize) -> u32 {
    (sequence as u32).div_ceil(TILE_ROWS as u32)
}

#[derive(Debug)]
struct DurationSwiGluW2Kernel {
    sequence: u32,
}

impl KernelSource for DurationSwiGluW2Kernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("duration_swiglu_w2.wgsl"))
            .register("sequence", self.sequence.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.sequence)
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
    let compatible = (1..=MAX_SEQUENCE).contains(&sequence)
        && projected.dtype == DType::F32
        && w2.dtype == DType::F32
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
    let hardware = &projected.client.properties().hardware;
    if hardware.max_bindings < REQUIRED_BINDINGS
        || hardware.max_shared_memory_size < SHARED_BYTES
        || hardware.max_units_per_cube < WORKGROUP_X * WORKGROUP_Y
        || hardware.max_cube_dim.0 < WORKGROUP_X
        || hardware.max_cube_dim.1 < WORKGROUP_Y
    {
        return None;
    }

    let client = projected.client.clone();
    let output_handle = client.empty(sequence * DIM * size_of::<f32>());
    let output = CubeTensor::new_contiguous(
        client.clone(),
        projected.device.clone(),
        Shape::from([1, sequence, DIM]),
        output_handle,
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DurationSwiGluW2Kernel {
                sequence: sequence as u32,
            },
            CubeDim::new_2d(WORKGROUP_X, WORKGROUP_Y),
        ));
    client.launch(
        task,
        CubeCount::new_2d((DIM as u32).div_ceil(WORKGROUP_X), row_workgroups(sequence)),
        KernelArguments::new()
            .with_buffer(projected.handle.binding())
            .with_buffer(w2.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_tile_accounting_is_bounded() {
        assert_eq!(WORKGROUP_X * WORKGROUP_Y, 256);
        assert_eq!(SHARED_BYTES, 6144);
        assert_eq!(TILE_K, 32);
        assert_eq!(DIM / TILE_K, 32);
        assert_eq!(row_workgroups(3), 1);
        assert_eq!(row_workgroups(12), 1);
        assert_eq!(row_workgroups(28), 2);
        assert_eq!(row_workgroups(61), 4);
    }
}
