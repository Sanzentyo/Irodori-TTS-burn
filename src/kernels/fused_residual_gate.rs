//! Fused gated residual update for inference-only DiT branches.

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
struct FusedResidualGateKernel {
    precision: KernelFloatPrecision,
    dim: u32,
    seq_len: u32,
    elements: u32,
}

impl KernelSource for FusedResidualGateKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("fused_residual_gate.wgsl"),
                include_str!("fused_residual_gate_f16.wgsl"),
            )
            .register("dim", self.dim.to_string())
            .register("seq_len", self.seq_len.to_string())
            .register("elements", self.elements.to_string())
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.dim, self.seq_len, self.elements))
    }
}

/// Compute `residual + gate * branch` in one dispatch.
///
/// `residual` and `branch` have shape `[batch * seq_len, dim]`; `gate` has
/// shape `[batch, dim]`. Inputs must share contiguous f32 or f16 storage.
pub fn fused_residual_gate_wgsl(
    residual: CubeTensor<WgpuRuntime>,
    branch: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    seq_len: usize,
) -> CubeTensor<WgpuRuntime> {
    for (name, tensor) in [
        ("residual", &residual),
        ("branch", &branch),
        ("gate", &gate),
    ] {
        assert_eq!(tensor.meta.num_dims(), 2, "{name} must be rank 2");
    }
    let precision = common_float_precision([residual.dtype, branch.dtype, gate.dtype])
        .expect("residual, branch and gate must share f32 or f16 dtype");

    let residual = into_contiguous(residual);
    let branch = into_contiguous(branch);
    let gate = into_contiguous(gate);
    let rows = residual.meta.shape()[0];
    let dim = residual.meta.shape()[1];
    assert_eq!(rows, batch * seq_len, "residual row count mismatch");
    assert_eq!(
        branch.meta.shape(),
        residual.meta.shape(),
        "branch shape mismatch"
    );
    assert_eq!(gate.meta.shape()[0], batch, "gate batch mismatch");
    assert_eq!(gate.meta.shape()[1], dim, "gate dimension mismatch");

    let elements = rows
        .checked_mul(dim)
        .expect("fused residual output size overflow");
    let client = residual.client.clone();
    let output_handle = client.empty(elements * precision.element_bytes());
    let output = CubeTensor::new_contiguous(
        client.clone(),
        residual.device.clone(),
        Shape::from([rows, dim]),
        output_handle,
        precision.dtype(),
    );
    let kernel = FusedResidualGateKernel {
        precision,
        dim: dim as u32,
        seq_len: seq_len as u32,
        elements: elements as u32,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(kernel, CubeDim::new_1d(WORKGROUP_SIZE)));
    let bindings = KernelArguments::new()
        .with_buffer(residual.handle.binding())
        .with_buffer(branch.handle.binding())
        .with_buffer(gate.handle.binding())
        .with_buffer(output.handle.clone().binding());
    client.launch(
        task,
        CubeCount::new_1d((elements as u32).div_ceil(WORKGROUP_SIZE)),
        bindings,
    );
    output
}
