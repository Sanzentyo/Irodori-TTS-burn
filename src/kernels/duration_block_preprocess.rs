//! Fused RMSNorm plus fixed scale/shift for released duration blocks.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::{DType, Shape};
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

const DIM: usize = 1024;
const MAX_SEQUENCE: usize = 64;
const WORKGROUP_SIZE: u32 = 256;
const REQUIRED_BINDINGS: u32 = 5;
const SHARED_BYTES: usize = WORKGROUP_SIZE as usize * size_of::<f32>();

#[derive(Debug)]
struct DurationBlockPreprocessKernel {
    sequence: u32,
    eps: f64,
}

impl KernelSource for DurationBlockPreprocessKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("duration_block_preprocess.wgsl"))
            .register("eps", format!("{:e}", self.eps))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.sequence, self.eps.to_bits()))
    }
}

fn shape(tensor: &CubeTensor<WgpuRuntime>, expected: &[usize]) -> bool {
    tensor.meta.shape().as_slice() == expected
}

fn strides(tensor: &CubeTensor<WgpuRuntime>, expected: &[usize]) -> bool {
    &tensor.meta.strides()[..] == expected
}

/// Fuse the no-aux block input transform without materializing intermediate
/// norm, multiply, or add tensors. A mismatch preserves the Burn fallback.
pub fn try_duration_block_preprocess_wgsl(
    input: CubeTensor<WgpuRuntime>,
    norm_weight: CubeTensor<WgpuRuntime>,
    scale: CubeTensor<WgpuRuntime>,
    shift: CubeTensor<WgpuRuntime>,
    eps: f64,
) -> Option<CubeTensor<WgpuRuntime>> {
    if input.meta.num_dims() != 3 {
        return None;
    }
    let sequence = input.meta.shape()[1];
    let compatible = eps.is_finite()
        && eps > 0.0
        && (1..=MAX_SEQUENCE).contains(&sequence)
        && [input.dtype, norm_weight.dtype, scale.dtype, shift.dtype]
            .into_iter()
            .all(|dtype| dtype == DType::F32)
        && shape(&input, &[1, sequence, DIM])
        && shape(&norm_weight, &[DIM])
        && shape(&scale, &[1, DIM])
        && shape(&shift, &[1, DIM])
        && strides(&input, &[sequence * DIM, DIM, 1])
        && strides(&norm_weight, &[1])
        && strides(&scale, &[DIM, 1])
        && strides(&shift, &[DIM, 1])
        && input.is_contiguous()
        && norm_weight.is_contiguous()
        && scale.is_contiguous()
        && shift.is_contiguous()
        && input.device == norm_weight.device
        && input.device == scale.device
        && input.device == shift.device;
    if !compatible {
        return None;
    }
    let hardware = &input.client.properties().hardware;
    if hardware.max_bindings < REQUIRED_BINDINGS
        || hardware.max_shared_memory_size < SHARED_BYTES
        || hardware.max_units_per_cube < WORKGROUP_SIZE
        || hardware.max_cube_dim.0 < WORKGROUP_SIZE
    {
        return None;
    }

    let client = input.client.clone();
    let output_handle = client.empty(sequence * DIM * size_of::<f32>());
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([1, sequence, DIM]),
        output_handle,
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DurationBlockPreprocessKernel {
                sequence: sequence as u32,
                eps,
            },
            CubeDim::new_1d(WORKGROUP_SIZE),
        ));
    client.launch(
        task,
        CubeCount::new_1d(sequence as u32),
        KernelArguments::new()
            .with_buffer(input.handle.binding())
            .with_buffer(norm_weight.handle.binding())
            .with_buffer(scale.handle.binding())
            .with_buffer(shift.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_contract_accounting_is_bounded() {
        assert_eq!(DIM, 1024);
        assert_eq!(MAX_SEQUENCE, 64);
        assert_eq!(WORKGROUP_SIZE, 256);
        assert_eq!(SHARED_BYTES, 1024);
        assert_eq!(REQUIRED_BINDINGS, 5);
    }
}
