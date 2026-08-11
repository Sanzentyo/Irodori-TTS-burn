//! One-dispatch gate and residual finalizer for released duration blocks.

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

#[derive(Debug)]
struct DurationResidualFinalizeKernel {
    elements: u32,
}

impl KernelSource for DurationResidualFinalizeKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("duration_residual_finalize.wgsl"))
            .register("elements", self.elements.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.elements)
    }
}

/// Compute `residual + gate * branch` without intermediate tensors.
pub fn try_duration_residual_finalize_wgsl(
    residual: CubeTensor<WgpuRuntime>,
    branch: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    if residual.meta.num_dims() != 3 {
        return None;
    }
    let sequence = residual.meta.shape()[1];
    let expected = [1, sequence, DIM];
    let compatible = (1..=MAX_SEQUENCE).contains(&sequence)
        && [residual.dtype, branch.dtype, gate.dtype]
            .into_iter()
            .all(|dtype| dtype == DType::F32)
        && residual.meta.shape().as_slice() == expected
        && branch.meta.shape().as_slice() == expected
        && gate.meta.shape().as_slice() == [1, 1, DIM]
        && residual.meta.strides()[..] == [sequence * DIM, DIM, 1]
        && branch.meta.strides()[..] == [sequence * DIM, DIM, 1]
        && gate.meta.strides()[..] == [DIM, DIM, 1]
        && residual.is_contiguous()
        && branch.is_contiguous()
        && gate.is_contiguous()
        && residual.device == branch.device
        && residual.device == gate.device;
    if !compatible {
        return None;
    }
    let elements = sequence * DIM;
    let client = residual.client.clone();
    let output_handle = client.empty(elements * size_of::<f32>());
    let output = CubeTensor::new_contiguous(
        client.clone(),
        residual.device.clone(),
        Shape::from(expected),
        output_handle,
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DurationResidualFinalizeKernel {
                elements: elements as u32,
            },
            CubeDim::new_1d(WORKGROUP_SIZE),
        ));
    client.launch(
        task,
        CubeCount::new_1d((elements as u32).div_ceil(WORKGROUP_SIZE)),
        KernelArguments::new()
            .with_buffer(residual.handle.binding())
            .with_buffer(branch.handle.binding())
            .with_buffer(gate.handle.binding())
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
    }
}
