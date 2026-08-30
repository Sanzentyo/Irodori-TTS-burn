//! Global split-K candidates for under-occupied MLP contract problems.
//!
//! The first dispatch assigns each half of K to independent workgroups and
//! writes two F32 partials. A compact second dispatch combines those partials
//! with the existing gated-residual epilogue. This is deliberately a candidate
//! route: profile selection must prove that the additional write and dispatch
//! are cheaper than leaving the GPU under-occupied.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::{DType, Shape};
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

use super::precision::{KernelFloatPrecision, common_float_precision};

const INPUT_DIM: usize = 3_680;
const OUTPUT_DIM: usize = 1_280;
const TILE_ROWS: usize = 64;
const TILE_COLUMNS: usize = 128;
const TILE_K: usize = 16;
const SPLITS: usize = 2;
const WORKGROUP_X: u32 = 16;
const WORKGROUP_Y: u32 = 16;
const FINAL_WORKGROUP_X: u32 = 256;
const PARTIAL_BINDINGS: u32 = 3;
const FINAL_BINDINGS: u32 = 4;
const SHARED_BYTES: usize = (TILE_ROWS * TILE_K + TILE_K * TILE_COLUMNS) * size_of::<f32>();

#[derive(Debug)]
struct SplitK2PartialKernel {
    precision: KernelFloatPrecision,
    rows: u32,
    input_row_stride: u32,
}

impl KernelSource for SplitK2PartialKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("dit_mlp_contract_split_k2_partial.wgsl"))
            .register(
                "enable_f16",
                if self.precision == KernelFloatPrecision::F16 {
                    "enable f16;"
                } else {
                    ""
                },
            )
            .register(
                "storage",
                if self.precision == KernelFloatPrecision::F16 {
                    "f16"
                } else {
                    "f32"
                },
            )
            .register("rows", self.rows.to_string())
            .register("input_row_stride", self.input_row_stride.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.rows, self.input_row_stride))
    }
}

#[derive(Debug)]
struct SplitK2FinalizeKernel {
    precision: KernelFloatPrecision,
    rows: u32,
    sequence: u32,
}

impl KernelSource for SplitK2FinalizeKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("dit_mlp_contract_split_k2_finalize.wgsl"))
            .register(
                "enable_f16",
                if self.precision == KernelFloatPrecision::F16 {
                    "enable f16;"
                } else {
                    ""
                },
            )
            .register(
                "storage",
                if self.precision == KernelFloatPrecision::F16 {
                    "f16"
                } else {
                    "f32"
                },
            )
            .register("rows", self.rows.to_string())
            .register("sequence", self.sequence.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.rows, self.sequence))
    }
}

fn binding_is_compatible(
    tensor: &CubeTensor<WgpuRuntime>,
    required_elements: usize,
    precision: KernelFloatPrecision,
    alignment: u64,
) -> bool {
    let Some(required_bytes) = required_elements
        .checked_mul(precision.element_bytes())
        .and_then(|bytes| u64::try_from(bytes).ok())
    else {
        return false;
    };
    let binding = tensor.handle.clone().binding();
    tensor.client.properties().memory.alignment >= alignment
        && tensor
            .client
            .properties()
            .memory
            .alignment
            .is_multiple_of(alignment)
        && binding.size_in_used() >= required_bytes
        && binding.offset_start.unwrap_or(0).is_multiple_of(alignment)
}

/// Try a global split-K2 MLP contract and gated-residual update.
///
/// Unsupported layouts, devices, mixed precisions, and dtypes other than F32
/// or F16 fail closed before allocation or dispatch. Reductions and partials
/// remain F32 in either storage mode.
#[allow(clippy::too_many_arguments)]
pub fn try_dit_mlp_contract_residual_split_k2_wgsl(
    activated: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    if activated.meta.num_dims() != 2
        || weight.meta.num_dims() != 2
        || residual.meta.num_dims() != 2
        || gate.meta.num_dims() != 2
    {
        return None;
    }
    let rows = batch.checked_mul(sequence)?;
    let output_elements = rows.checked_mul(OUTPUT_DIM)?;
    let partial_elements = output_elements.checked_mul(SPLITS)?;
    let input_row_stride = *activated.meta.strides().first()?;
    let required_input_elements = rows
        .checked_sub(1)?
        .checked_mul(input_row_stride)?
        .checked_add(INPUT_DIM)?;
    let precision =
        common_float_precision([activated.dtype, weight.dtype, residual.dtype, gate.dtype])?;
    let vec4_bytes = u64::try_from(4 * precision.element_bytes()).ok()?;
    let partial_vec4_bytes = u64::try_from(4 * size_of::<f32>()).ok()?;
    let same_device = activated.device == weight.device
        && activated.device == residual.device
        && activated.device == gate.device;
    let compatible = matches!(batch, 1..=3)
        && (13..=685).contains(&sequence)
        && activated.meta.shape().as_slice() == [rows, INPUT_DIM]
        && weight.meta.shape().as_slice() == [INPUT_DIM, OUTPUT_DIM]
        && residual.meta.shape().as_slice() == [rows, OUTPUT_DIM]
        && gate.meta.shape().as_slice() == [batch, OUTPUT_DIM]
        && activated.meta.strides()[1] == 1
        && (input_row_stride == INPUT_DIM || input_row_stride == 2 * INPUT_DIM)
        && input_row_stride.is_multiple_of(4)
        && weight.meta.strides()[..] == [OUTPUT_DIM, 1]
        && residual.meta.strides()[..] == [OUTPUT_DIM, 1]
        && gate.meta.strides()[..] == [OUTPUT_DIM, 1]
        && weight.is_contiguous()
        && residual.is_contiguous()
        && gate.is_contiguous()
        && same_device
        && binding_is_compatible(&activated, required_input_elements, precision, vec4_bytes)
        && binding_is_compatible(&weight, INPUT_DIM * OUTPUT_DIM, precision, vec4_bytes)
        && binding_is_compatible(&residual, output_elements, precision, vec4_bytes)
        && binding_is_compatible(&gate, batch * OUTPUT_DIM, precision, vec4_bytes);
    if !compatible {
        return None;
    }

    let output_workgroups = u32::try_from(OUTPUT_DIM / TILE_COLUMNS).ok()?;
    let row_workgroups = u32::try_from(rows.div_ceil(TILE_ROWS)).ok()?;
    let hardware = &activated.client.properties().hardware;
    if hardware.max_bindings < PARTIAL_BINDINGS.max(FINAL_BINDINGS)
        || hardware.max_shared_memory_size < SHARED_BYTES
        || hardware.max_units_per_cube < WORKGROUP_X * WORKGROUP_Y
        || hardware.max_units_per_cube < FINAL_WORKGROUP_X
        || hardware.max_cube_dim.0 < FINAL_WORKGROUP_X
        || hardware.max_cube_dim.1 < WORKGROUP_Y
        || hardware.max_cube_dim.2 < 1
        || hardware.max_cube_count.0 < output_workgroups
        || hardware.max_cube_count.1 < row_workgroups
        || hardware.max_cube_count.2 < u32::try_from(SPLITS).ok()?
    {
        return None;
    }

    let partial_bytes = partial_elements.checked_mul(size_of::<f32>())?;
    let output_bytes = output_elements.checked_mul(precision.element_bytes())?;
    let client = activated.client.clone();
    let partial_handle = client.empty(partial_bytes);
    let output_handle = client.empty(output_bytes);
    if partial_handle.size_in_used() < u64::try_from(partial_bytes).ok()?
        || output_handle.size_in_used() < u64::try_from(output_bytes).ok()?
        || !partial_handle
            .offset_start
            .unwrap_or(0)
            .is_multiple_of(partial_vec4_bytes)
        || !output_handle
            .offset_start
            .unwrap_or(0)
            .is_multiple_of(vec4_bytes)
    {
        return None;
    }
    let partial = CubeTensor::new_contiguous(
        client.clone(),
        activated.device.clone(),
        Shape::from([SPLITS, rows, OUTPUT_DIM]),
        partial_handle,
        DType::F32,
    );
    let output = CubeTensor::new_contiguous(
        client.clone(),
        activated.device.clone(),
        Shape::from([rows, OUTPUT_DIM]),
        output_handle,
        precision.dtype(),
    );
    let partial_task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            SplitK2PartialKernel {
                precision,
                rows: u32::try_from(rows).ok()?,
                input_row_stride: u32::try_from(input_row_stride).ok()?,
            },
            CubeDim::new_2d(WORKGROUP_X, WORKGROUP_Y),
        ));
    client.launch(
        partial_task,
        CubeCount::new_3d(
            output_workgroups,
            row_workgroups,
            u32::try_from(SPLITS).ok()?,
        ),
        KernelArguments::new()
            .with_buffer(activated.handle.binding())
            .with_buffer(weight.handle.binding())
            .with_buffer(partial.handle.clone().binding()),
    );

    let final_task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            SplitK2FinalizeKernel {
                precision,
                rows: u32::try_from(rows).ok()?,
                sequence: u32::try_from(sequence).ok()?,
            },
            CubeDim::new_1d(FINAL_WORKGROUP_X),
        ));
    let output_vecs = u32::try_from(output_elements / 4).ok()?;
    client.launch(
        final_task,
        CubeCount::new_1d(output_vecs.div_ceil(FINAL_WORKGROUP_X)),
        KernelArguments::new()
            .with_buffer(partial.handle.binding())
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
    fn split_k2_geometry_is_integral_and_bounded() {
        assert_eq!(INPUT_DIM % SPLITS, 0);
        assert_eq!((INPUT_DIM / SPLITS) % TILE_K, 0);
        assert_eq!(OUTPUT_DIM % TILE_COLUMNS, 0);
        assert_eq!(SHARED_BYTES, 12_288);
        assert_eq!(OUTPUT_DIM / TILE_COLUMNS, 10);
    }

    #[test]
    fn partial_and_finalize_bindings_are_explicit() {
        let partial = include_str!("dit_mlp_contract_split_k2_partial.wgsl");
        let finalize = include_str!("dit_mlp_contract_split_k2_finalize.wgsl");
        assert_eq!(
            partial.matches("@binding(").count(),
            PARTIAL_BINDINGS as usize
        );
        assert_eq!(
            finalize.matches("@binding(").count(),
            FINAL_BINDINGS as usize
        );
        assert!(partial.contains("let split = group_id.z"));
        assert!(finalize.contains("partial[index] + partial[OUTPUT_VECS + index]"));
    }

    /// Compile and execute the mixed-storage contract on a shader-f16 adapter.
    /// Ignored by default because WGPU teardown is unreliable in the ordinary
    /// test harness; CI or a profiling campaign may invoke it explicitly.
    #[test]
    #[ignore = "requires a shader-f16 WGPU adapter"]
    fn f16_storage_keeps_f32_partials_and_returns_finite_output() {
        use burn::backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup};
        use burn::tensor::{FloatDType, Tensor};

        let raw_device = WgpuDevice::DefaultDevice;
        init_setup::<AutoGraphicsApi>(&raw_device, Default::default());
        let device = crate::backend_config::wgpu_device_with_precision(
            &raw_device,
            crate::backend_config::WgpuFloatPrecision::Fp16,
        )
        .expect("test adapter must support F16");
        let batch = 1;
        let sequence = 13;
        let rows = batch * sequence;
        let activated = (Tensor::<2>::ones([rows, 2 * INPUT_DIM], &device) * 0.003_f32)
            .cast(FloatDType::F16)
            .slice([0..rows, 0..INPUT_DIM]);
        let weight =
            (Tensor::<2>::ones([INPUT_DIM, OUTPUT_DIM], &device) * 0.001_f32).cast(FloatDType::F16);
        let residual =
            (Tensor::<2>::ones([rows, OUTPUT_DIM], &device) * 0.25_f32).cast(FloatDType::F16);
        let gate =
            (Tensor::<2>::ones([batch, OUTPUT_DIM], &device) * 0.125_f32).cast(FloatDType::F16);
        assert_eq!(activated.dtype(), DType::F16);
        assert_eq!(weight.dtype(), DType::F16);
        assert_eq!(residual.dtype(), DType::F16);
        assert_eq!(gate.dtype(), DType::F16);
        let output = try_dit_mlp_contract_residual_split_k2_wgsl(
            activated
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("F16 activation must use raw WGPU"),
            weight
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("F16 weight must use raw WGPU"),
            residual
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("F16 residual must use raw WGPU"),
            gate.try_into_primitive::<crate::WgpuRaw>()
                .expect("F16 gate must use raw WGPU"),
            batch,
            sequence,
        )
        .expect("split-K2 must admit the exact F16 profile");
        assert_eq!(output.dtype, DType::F16);
        let values = Tensor::<2>::from_primitive::<crate::WgpuRaw>(output)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("F16 output readback");
        assert!(values.iter().all(|value| value.is_finite()));
    }
}
