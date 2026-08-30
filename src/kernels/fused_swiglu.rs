//! Fused WGSL activation for the inference-time `w1 || w3` projection.

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

#[derive(Debug)]
struct FusedSwiGluPairKernel {
    precision: KernelFloatPrecision,
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

impl KernelSource for FusedSwiGluPairKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("fused_swiglu_pair.wgsl"),
                include_str!("fused_swiglu_pair_f16.wgsl"),
            )
            .register("elements", self.elements.to_string())
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.elements))
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

/// Apply `silu(gate) * value` to two independently projected tensors.
///
/// Both inputs must be equal-shaped, contiguous rank-2 f32 or f16 tensors on
/// the same WGPU device. Returning `None` before allocation makes an invalid
/// split-projection route fail closed at its typed caller.
pub fn try_fused_swiglu_pair_wgsl(
    gate: CubeTensor<WgpuRuntime>,
    value: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    let precision = common_float_precision([gate.dtype, value.dtype])?;
    if gate.meta.num_dims() != 2
        || value.meta.num_dims() != 2
        || gate.meta.shape() != value.meta.shape()
        || gate.device != value.device
        || !gate.is_contiguous()
        || !value.is_contiguous()
    {
        return None;
    }
    let [rows, hidden] = gate.meta.shape().dims::<2>();
    let elements = rows.checked_mul(hidden)?;
    let elements_u32 = u32::try_from(elements).ok()?;
    if elements_u32 == 0 {
        return None;
    }
    let required_bytes = u64::try_from(elements.checked_mul(precision.element_bytes())?).ok()?;
    if gate.handle.size_in_used() < required_bytes || value.handle.size_in_used() < required_bytes {
        return None;
    }
    let workgroups = elements_u32.div_ceil(WORKGROUP_SIZE);
    let properties = gate.client.properties();
    if properties.hardware.max_bindings < 3
        || properties.hardware.max_units_per_cube < WORKGROUP_SIZE
        || properties.hardware.max_cube_dim.0 < WORKGROUP_SIZE
        || properties.hardware.max_cube_count.0 < workgroups
    {
        return None;
    }

    let client = gate.client.clone();
    let output_handle = client.empty(elements * precision.element_bytes());
    if output_handle.size_in_used() < required_bytes {
        return None;
    }
    let output = CubeTensor::new_contiguous(
        client.clone(),
        gate.device.clone(),
        Shape::from([rows, hidden]),
        output_handle,
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            FusedSwiGluPairKernel {
                precision,
                elements: elements_u32,
            },
            CubeDim::new_1d(WORKGROUP_SIZE),
        ));
    client.launch(
        task,
        CubeCount::new_1d(workgroups),
        KernelArguments::new()
            .with_buffer(gate.handle.binding())
            .with_buffer(value.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup};
    use burn::tensor::{Tensor, TensorData};

    #[test]
    fn split_projection_epilogue_matches_the_portable_graph() {
        let raw_device = WgpuDevice::DefaultDevice;
        init_setup::<AutoGraphicsApi>(&raw_device, Default::default());
        let device = crate::backend_config::strict_fp32_device(&raw_device)
            .expect("test WGPU device must support strict F32");
        let gate_values = [-3.0_f32, -0.5, 0.0, 0.25, 1.0, 4.0];
        let value_values = [0.5_f32, -2.0, 3.0, -4.0, 0.125, 2.0];
        let gate = Tensor::<2>::from_data(TensorData::new(gate_values.to_vec(), [2, 3]), &device);
        let value = Tensor::<2>::from_data(TensorData::new(value_values.to_vec(), [2, 3]), &device);
        let expected = burn::tensor::activation::silu(gate.clone()) * value.clone();
        let output = try_fused_swiglu_pair_wgsl(
            gate.try_into_primitive::<crate::WgpuRaw>()
                .expect("gate must use raw WGPU"),
            value
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("value must use raw WGPU"),
        )
        .expect("valid pair must launch");
        let actual = Tensor::<2>::from_primitive::<crate::WgpuRaw>(output);
        let max_abs: f32 = (actual - expected).abs().max().into_scalar();
        assert!(max_abs <= 1.0e-6, "split SwiGLU max_abs={max_abs}");
    }
}
