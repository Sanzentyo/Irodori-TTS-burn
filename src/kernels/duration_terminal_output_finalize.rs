//! Final duration block residual plus output reduction in one dispatch.

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
const LANES_PER_TOKEN: usize = 4;
const WORKGROUP_SIZE: u32 = (MAX_SEQUENCE * LANES_PER_TOKEN) as u32;
const REQUIRED_BINDINGS: u32 = 7;
const SHARED_BYTES: usize = (WORKGROUP_SIZE as usize * 2 + MAX_SEQUENCE) * size_of::<f32>();

#[derive(Debug)]
struct DurationTerminalOutputFinalizeKernel {
    precision: KernelFloatPrecision,
    sequence: u32,
    eps: f64,
}

impl KernelSource for DurationTerminalOutputFinalizeKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("duration_terminal_output_finalize.wgsl"),
                include_str!("duration_terminal_output_finalize_f16.wgsl"),
            )
            .register("sequence", self.sequence.to_string())
            .register("eps", format!("{:e}", self.eps))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.sequence, self.eps.to_bits()))
    }
}

fn shape(tensor: &CubeTensor<WgpuRuntime>, expected: &[usize]) -> bool {
    tensor.meta.shape().as_slice() == expected
}

fn strides(tensor: &CubeTensor<WgpuRuntime>, expected: &[usize]) -> bool {
    &tensor.meta.strides()[..] == expected
}

/// Fuse the final `residual + gate * branch` with the compact output head.
/// A rejected contract returns `None` before allocation or dispatch.
#[allow(clippy::too_many_arguments)]
pub fn try_duration_terminal_output_finalize_wgsl(
    residual: CubeTensor<WgpuRuntime>,
    branch: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    norm_weight: CubeTensor<WgpuRuntime>,
    output_weight: CubeTensor<WgpuRuntime>,
    output_bias: CubeTensor<WgpuRuntime>,
    eps: f64,
) -> Option<CubeTensor<WgpuRuntime>> {
    let precision = common_float_precision([
        residual.dtype,
        branch.dtype,
        gate.dtype,
        norm_weight.dtype,
        output_weight.dtype,
        output_bias.dtype,
    ])?;
    if residual.meta.num_dims() != 3 {
        return None;
    }
    let sequence = residual.meta.shape()[1];
    let compatible = eps.is_finite()
        && eps > 0.0
        && (1..=MAX_SEQUENCE).contains(&sequence)
        && shape(&residual, &[1, sequence, DIM])
        && shape(&branch, &[1, sequence, DIM])
        && shape(&gate, &[1, 1, DIM])
        && shape(&norm_weight, &[DIM])
        && shape(&output_weight, &[DIM, 1])
        && shape(&output_bias, &[1])
        && strides(&residual, &[sequence * DIM, DIM, 1])
        && strides(&branch, &[sequence * DIM, DIM, 1])
        && strides(&gate, &[DIM, DIM, 1])
        && strides(&norm_weight, &[1])
        && strides(&output_weight, &[1, 1])
        && strides(&output_bias, &[1])
        && residual.is_contiguous()
        && branch.is_contiguous()
        && gate.is_contiguous()
        && norm_weight.is_contiguous()
        && output_weight.is_contiguous()
        && output_bias.is_contiguous()
        && residual.device == branch.device
        && residual.device == gate.device
        && residual.device == norm_weight.device
        && residual.device == output_weight.device
        && residual.device == output_bias.device;
    if !compatible {
        return None;
    }
    let hardware = &residual.client.properties().hardware;
    if hardware.max_bindings < REQUIRED_BINDINGS
        || hardware.max_shared_memory_size < SHARED_BYTES
        || hardware.max_units_per_cube < WORKGROUP_SIZE
        || hardware.max_cube_dim.0 < WORKGROUP_SIZE
    {
        return None;
    }

    let client = residual.client.clone();
    let output_handle = client.empty(precision.element_bytes());
    let output = CubeTensor::new_contiguous(
        client.clone(),
        residual.device.clone(),
        Shape::from([1]),
        output_handle,
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DurationTerminalOutputFinalizeKernel {
                precision,
                sequence: sequence as u32,
                eps,
            },
            CubeDim::new_1d(WORKGROUP_SIZE),
        ));
    client.launch(
        task,
        CubeCount::new_1d(1),
        KernelArguments::new()
            .with_buffer(residual.handle.binding())
            .with_buffer(branch.handle.binding())
            .with_buffer(gate.handle.binding())
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
        assert_eq!(REQUIRED_BINDINGS, 7);
    }

    #[test]
    fn shader_preserves_residual_and_output_orders() {
        let source = include_str!("duration_terminal_output_finalize.wgsl");
        assert_eq!(source.matches("var<storage, read_write>").count(), 7);
        assert!(source.contains("fma(gate[feature], branch[index], residual[index])"));
        assert!(source.contains("logit <= 20.0"));
        assert!(source.contains("log(1.0 + max(square_parts[0], 0.0))"));
    }
}
