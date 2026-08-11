//! One-dispatch final reduction for the released compact duration head.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::{DType, Shape};
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

const DIM: usize = 1024;
const MAX_SEQUENCE: usize = 64;
const LANES_PER_TOKEN: usize = 4;
const WORKGROUP_SIZE: u32 = (MAX_SEQUENCE * LANES_PER_TOKEN) as u32;
const REQUIRED_BINDINGS: u32 = 5;
const SHARED_BYTES: usize = (WORKGROUP_SIZE as usize * 2 + MAX_SEQUENCE) * size_of::<f32>();

#[derive(Debug)]
struct DurationOutputFinalizeKernel {
    sequence: u32,
    eps: f64,
}

impl KernelSource for DurationOutputFinalizeKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("duration_output_finalize.wgsl"))
            .register("sequence", self.sequence.to_string())
            .register("eps", format!("{:e}", self.eps))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.sequence, self.eps.to_bits()))
    }
}

fn exact_shape(tensor: &CubeTensor<WgpuRuntime>, shape: &[usize]) -> bool {
    tensor.meta.shape().as_slice() == shape
}

fn exact_strides(tensor: &CubeTensor<WgpuRuntime>, strides: &[usize]) -> bool {
    &tensor.meta.strides()[..] == strides
}

/// Complete no-copy selector for the released f32, batch-one duration head.
pub fn contract_is_compatible(
    hidden: &CubeTensor<WgpuRuntime>,
    norm_weight: &CubeTensor<WgpuRuntime>,
    output_weight: &CubeTensor<WgpuRuntime>,
    output_bias: &CubeTensor<WgpuRuntime>,
) -> bool {
    if hidden.meta.num_dims() != 3 {
        return false;
    }
    let sequence = hidden.meta.shape()[1];
    let logical = hidden.dtype == DType::F32
        && norm_weight.dtype == DType::F32
        && output_weight.dtype == DType::F32
        && output_bias.dtype == DType::F32
        && (1..=MAX_SEQUENCE).contains(&sequence)
        && exact_shape(hidden, &[1, sequence, DIM])
        && exact_shape(norm_weight, &[DIM])
        && exact_shape(output_weight, &[DIM, 1])
        && exact_shape(output_bias, &[1])
        && exact_strides(hidden, &[sequence * DIM, DIM, 1])
        && exact_strides(norm_weight, &[1])
        && exact_strides(output_weight, &[1, 1])
        && exact_strides(output_bias, &[1])
        && hidden.is_contiguous()
        && norm_weight.is_contiguous()
        && output_weight.is_contiguous()
        && output_bias.is_contiguous()
        && hidden.device == norm_weight.device
        && hidden.device == output_weight.device
        && hidden.device == output_bias.device;
    if !logical {
        return false;
    }
    let hardware = &hidden.client.properties().hardware;
    hardware.max_bindings >= REQUIRED_BINDINGS
        && hardware.max_shared_memory_size >= SHARED_BYTES
        && hardware.max_units_per_cube >= WORKGROUP_SIZE
        && hardware.max_cube_dim.0 >= WORKGROUP_SIZE
}

/// Fuse final RMSNorm, scalar projection, Softplus, token sum, and log1p.
///
/// The compact caller guarantees every one of the `sequence <= 64` tokens is
/// valid. `None` preserves the generic masked Burn fallback without copying.
pub fn try_duration_output_finalize_wgsl(
    hidden: CubeTensor<WgpuRuntime>,
    norm_weight: CubeTensor<WgpuRuntime>,
    output_weight: CubeTensor<WgpuRuntime>,
    output_bias: CubeTensor<WgpuRuntime>,
    eps: f64,
) -> Option<CubeTensor<WgpuRuntime>> {
    if !eps.is_finite()
        || eps <= 0.0
        || !contract_is_compatible(&hidden, &norm_weight, &output_weight, &output_bias)
    {
        return None;
    }
    let sequence = hidden.meta.shape()[1];
    let client = hidden.client.clone();
    let output_handle = client.empty(size_of::<f32>());
    let output = CubeTensor::new_contiguous(
        client.clone(),
        hidden.device.clone(),
        Shape::from([1]),
        output_handle,
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DurationOutputFinalizeKernel {
                sequence: sequence as u32,
                eps,
            },
            CubeDim::new_1d(WORKGROUP_SIZE),
        ));
    client.launch(
        task,
        CubeCount::new_1d(1),
        KernelArguments::new()
            .with_buffer(hidden.handle.binding())
            .with_buffer(norm_weight.handle.binding())
            .with_buffer(output_weight.handle.binding())
            .with_buffer(output_bias.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_launch_accounting_is_bounded() {
        assert_eq!(WORKGROUP_SIZE, 256);
        assert_eq!(MAX_SEQUENCE, 64);
        assert_eq!(LANES_PER_TOKEN, 4);
        assert_eq!(SHARED_BYTES, 2304);
        assert_eq!(REQUIRED_BINDINGS, 5);
    }

    #[test]
    fn shader_keeps_pytorch_softplus_threshold_and_uniform_bindings() {
        let source = include_str!("duration_output_finalize.wgsl");
        assert_eq!(source.matches("var<storage, read_write>").count(), 5);
        assert!(source.contains("logit <= 20.0"));
        assert!(source.contains("log(1.0 + total_frames)"));
    }
}
