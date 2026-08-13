//! Fused WGSL activation for the inference-time `w1 || w3` projection.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime, into_contiguous,
};
use burn::tensor::Shape;
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

use super::precision::KernelFloatPrecision;

const WORKGROUP_SIZE: u32 = 256;

#[derive(Debug)]
struct FusedSwiGluKernel {
    precision: KernelFloatPrecision,
    hidden: u32,
    elements: u32,
}

impl KernelSource for FusedSwiGluKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("fused_swiglu.wgsl"),
                include_str!("fused_swiglu_f16.wgsl"),
            )
            .register("hidden", self.hidden.to_string())
            .register("elements", self.elements.to_string())
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.hidden, self.elements))
    }
}

/// Apply `silu(gate) * value` to a fused projection.
///
/// `input` must be contiguous f32 or f16 with shape `[rows, 2 * hidden]`. The returned
/// tensor has shape `[rows, hidden]`.
pub fn fused_swiglu_wgsl(input: CubeTensor<WgpuRuntime>) -> CubeTensor<WgpuRuntime> {
    let precision =
        KernelFloatPrecision::from_dtype(input.dtype).expect("fused SwiGLU requires f32 or f16");
    let input = into_contiguous(input);
    assert_eq!(
        input.meta.num_dims(),
        2,
        "fused SwiGLU input must be rank 2"
    );

    let rows = input.meta.shape()[0];
    let doubled_hidden = input.meta.shape()[1];
    assert!(
        doubled_hidden > 0 && doubled_hidden.is_multiple_of(2),
        "last dimension must be a positive even number"
    );
    let hidden = doubled_hidden / 2;
    let elements = rows
        .checked_mul(hidden)
        .expect("fused SwiGLU output size overflow");

    let client = input.client.clone();
    let output_handle = client.empty(elements * precision.element_bytes());
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([rows, hidden]),
        output_handle,
        precision.dtype(),
    );

    let kernel = FusedSwiGluKernel {
        precision,
        hidden: hidden as u32,
        elements: elements as u32,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(kernel, CubeDim::new_1d(WORKGROUP_SIZE)));
    let bindings = KernelArguments::new()
        .with_buffer(input.handle.binding())
        .with_buffer(output.handle.clone().binding());
    client.launch(
        task,
        CubeCount::new_1d((elements as u32).div_ceil(WORKGROUP_SIZE)),
        bindings,
    );
    output
}
