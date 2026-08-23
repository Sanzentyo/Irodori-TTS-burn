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

#[derive(Debug)]
struct FusedSwiGluInPlaceKernel {
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

impl KernelSource for FusedSwiGluInPlaceKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("fused_swiglu_in_place.wgsl"),
                include_str!("fused_swiglu_in_place_f16.wgsl"),
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

/// Apply SwiGLU in-place to the gate half of a contiguous fused projection.
///
/// The returned storage keeps its original `[rows, 2 * hidden]` metadata. The
/// caller must expose `[.., 0..hidden]` as a pitched `[rows, hidden]` view and
/// pass that row stride to a compatible consumer. This avoids allocating the
/// ordinary compressed activation without pretending the view is contiguous.
pub fn try_fused_swiglu_pitched_in_place_wgsl(
    input: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    let precision = KernelFloatPrecision::from_dtype(input.dtype)?;
    if input.meta.num_dims() != 2 || !input.is_contiguous() {
        return None;
    }
    let rows = input.meta.shape()[0];
    let doubled_hidden = input.meta.shape()[1];
    if rows == 0 || doubled_hidden == 0 || !doubled_hidden.is_multiple_of(2) {
        return None;
    }
    let hidden = doubled_hidden / 2;
    let elements = rows.checked_mul(hidden)?;
    let required_bytes = rows
        .checked_mul(doubled_hidden)?
        .checked_mul(precision.element_bytes())?;
    if input.handle.size_in_used() < u64::try_from(required_bytes).ok()? {
        return None;
    }
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            FusedSwiGluInPlaceKernel {
                precision,
                hidden: u32::try_from(hidden).ok()?,
                elements: u32::try_from(elements).ok()?,
            },
            CubeDim::new_1d(WORKGROUP_SIZE),
        ));
    input.client.launch(
        task,
        CubeCount::new_1d(u32::try_from(elements).ok()?.div_ceil(WORKGROUP_SIZE)),
        KernelArguments::new().with_buffer(input.handle.clone().binding()),
    );
    Some(input)
}
