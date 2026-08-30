//! Fused long-sequence DiT MLP contract and gated residual update.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::Shape;
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

use super::precision::{KernelFloatPrecision, common_float_precision};

const INPUT_DIM: usize = 3_680;
const OUTPUT_DIM: usize = 1_280;
// Physical shader capability only. The model-side route selector owns the
// device/profile-specific production envelope.
const MIN_SEQUENCE: usize = 13;
const MAX_SEQUENCE: usize = 685;
const TILE_ROWS: usize = 64;
const TILE_COLUMNS: usize = 128;
const TILE_K: usize = 32;
const WORKGROUP_X: u32 = 16;
const WORKGROUP_Y: u32 = 16;
const REQUIRED_BINDINGS: u32 = 5;
const SHARED_BYTES: usize = (TILE_ROWS * TILE_K + TILE_K * TILE_COLUMNS) * size_of::<f32>();

#[derive(Debug)]
struct DitMlpContractResidualKernel {
    precision: KernelFloatPrecision,
    rows: u32,
    sequence: u32,
    inner: u32,
    input_row_stride: u32,
    vectorized_input: bool,
}

#[derive(Debug)]
struct DitAttentionOutputDirectResidualKernel {
    rows: u32,
    sequence: u32,
    gate_row_stride: u32,
    gate_offset: u32,
}

impl KernelSource for DitMlpContractResidualKernel {
    fn source(&self) -> SourceTemplate {
        let source = if self.vectorized_input {
            self.precision.source(
                include_str!("dit_mlp_contract_residual_vec4.wgsl"),
                include_str!("dit_mlp_contract_residual_vec4_f16.wgsl"),
            )
        } else {
            self.precision.source(
                include_str!("dit_mlp_contract_residual.wgsl"),
                include_str!("dit_mlp_contract_residual_f16.wgsl"),
            )
        };
        source
            .register("rows", self.rows.to_string())
            .register("sequence", self.sequence.to_string())
            .register("inner", self.inner.to_string())
            .register("input_row_stride", self.input_row_stride.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.precision,
            self.rows,
            self.sequence,
            self.inner,
            self.input_row_stride,
            self.vectorized_input,
        ))
    }
}

impl KernelSource for DitAttentionOutputDirectResidualKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("dit_attention_output_direct_residual.wgsl"))
            .register("rows", self.rows.to_string())
            .register("sequence", self.sequence.to_string())
            .register("gate_row_stride", self.gate_row_stride.to_string())
            .register("gate_offset", self.gate_offset.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.rows,
            self.sequence,
            self.gate_row_stride,
            self.gate_offset,
        ))
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

/// Compute `residual + gate * (activated @ weight)` without materialising the
/// projected branch. The released inference graph supplies an already-tanh'd
/// gate and identity dropout.
pub fn try_dit_mlp_contract_residual_wgsl(
    activated: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        activated, weight, residual, gate, batch, sequence, INPUT_DIM, false,
    )
}

/// Vectorize the K-contiguous activation load and shared-memory staging while
/// preserving the established scalar reduction order and fused residual
/// epilogue. Both contiguous and explicitly pitched SwiGLU views are admitted.
pub fn try_dit_mlp_contract_residual_vec4_wgsl(
    activated: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        activated, weight, residual, gate, batch, sequence, INPUT_DIM, true,
    )
}

/// Compute the released attention output projection and its gated residual in
/// one dispatch.
pub fn try_dit_attention_output_residual_wgsl(
    attention: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        attention, weight, residual, gate, batch, sequence, OUTPUT_DIM, false,
    )
}

/// Vector-staged form of the released attention output projection and fused
/// gated residual. The public route remains distinct from the scalar launcher
/// so exact-device tuning owns its admission.
pub fn try_dit_attention_output_residual_vec4_wgsl(
    attention: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        attention, weight, residual, gate, batch, sequence, OUTPUT_DIM, true,
    )
}

/// Consume head-major SDPA output and its compact learned gate directly in the
/// released output projection, then apply the block gate/residual at store.
///
/// A successful call is one dispatch and never materializes token-major gated
/// attention. The exact layout contract is validated before allocation; all
/// other inputs return `None` to preserve the established two-stage route.
#[allow(clippy::too_many_arguments)]
pub fn try_dit_attention_output_direct_residual_wgsl(
    attention: CubeTensor<WgpuRuntime>,
    attention_gate: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    block_gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    const HEADS: usize = 20;
    const HEAD_DIM: usize = 64;
    const BINDINGS: u32 = 6;

    if attention.meta.num_dims() != 4
        || attention_gate.meta.num_dims() != 3
        || weight.meta.num_dims() != 2
        || residual.meta.num_dims() != 2
        || block_gate.meta.num_dims() != 2
    {
        return None;
    }
    let rows = batch.checked_mul(sequence)?;
    let output_elements = rows.checked_mul(OUTPUT_DIM)?;
    let gate_width = attention_gate.meta.shape()[2];
    let (gate_row_stride, gate_offset) = match gate_width {
        OUTPUT_DIM => (OUTPUT_DIM, 0),
        width if width == OUTPUT_DIM * 4 => (OUTPUT_DIM * 4, OUTPUT_DIM * 3),
        _ => return None,
    };
    let gate_elements = rows.checked_mul(gate_width)?;
    let precision = common_float_precision([
        attention.dtype,
        attention_gate.dtype,
        weight.dtype,
        residual.dtype,
        block_gate.dtype,
    ])?;
    if precision != KernelFloatPrecision::F32 {
        return None;
    }
    let vec4_bytes = u64::try_from(4 * precision.element_bytes()).ok()?;
    let same_device = attention.device == attention_gate.device
        && attention.device == weight.device
        && attention.device == residual.device
        && attention.device == block_gate.device;
    let compatible = matches!(batch, 1..=3)
        && (MIN_SEQUENCE..=MAX_SEQUENCE).contains(&sequence)
        && attention.meta.shape().as_slice() == [batch, HEADS, sequence, HEAD_DIM]
        && attention_gate.meta.shape().as_slice() == [batch, sequence, gate_width]
        && weight.meta.shape().as_slice() == [OUTPUT_DIM, OUTPUT_DIM]
        && residual.meta.shape().as_slice() == [rows, OUTPUT_DIM]
        && block_gate.meta.shape().as_slice() == [batch, OUTPUT_DIM]
        && attention.meta.strides()[..]
            == [
                HEADS * sequence * HEAD_DIM,
                sequence * HEAD_DIM,
                HEAD_DIM,
                1,
            ]
        && attention_gate.meta.strides()[..] == [sequence * gate_width, gate_width, 1]
        && weight.meta.strides()[..] == [OUTPUT_DIM, 1]
        && residual.meta.strides()[..] == [OUTPUT_DIM, 1]
        && block_gate.meta.strides()[..] == [OUTPUT_DIM, 1]
        && attention.is_contiguous()
        && attention_gate.is_contiguous()
        && weight.is_contiguous()
        && residual.is_contiguous()
        && block_gate.is_contiguous()
        && same_device
        && binding_is_compatible(&attention, output_elements, precision, vec4_bytes)
        && binding_is_compatible(&attention_gate, gate_elements, precision, vec4_bytes)
        && binding_is_compatible(&weight, OUTPUT_DIM * OUTPUT_DIM, precision, vec4_bytes)
        && binding_is_compatible(&residual, output_elements, precision, vec4_bytes)
        && binding_is_compatible(&block_gate, batch * OUTPUT_DIM, precision, vec4_bytes);
    if !compatible {
        return None;
    }

    let hardware = &attention.client.properties().hardware;
    if hardware.max_bindings < BINDINGS
        || hardware.max_shared_memory_size < SHARED_BYTES
        || hardware.max_units_per_cube < WORKGROUP_X * WORKGROUP_Y
        || hardware.max_cube_dim.0 < WORKGROUP_X
        || hardware.max_cube_dim.1 < WORKGROUP_Y
        || hardware.max_cube_count.0 < u32::try_from(OUTPUT_DIM / TILE_COLUMNS).ok()?
        || hardware.max_cube_count.1 < u32::try_from(rows.div_ceil(TILE_ROWS)).ok()?
    {
        return None;
    }

    let output_bytes = output_elements.checked_mul(precision.element_bytes())?;
    let client = attention.client.clone();
    let output_handle = client.empty(output_bytes);
    if output_handle.size_in_used() < u64::try_from(output_bytes).ok()?
        || !output_handle
            .offset_start
            .unwrap_or(0)
            .is_multiple_of(vec4_bytes)
    {
        return None;
    }
    let output = CubeTensor::new_contiguous(
        client.clone(),
        attention.device.clone(),
        Shape::from([rows, OUTPUT_DIM]),
        output_handle,
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DitAttentionOutputDirectResidualKernel {
                rows: u32::try_from(rows).ok()?,
                sequence: u32::try_from(sequence).ok()?,
                gate_row_stride: u32::try_from(gate_row_stride).ok()?,
                gate_offset: u32::try_from(gate_offset).ok()?,
            },
            CubeDim::new_2d(WORKGROUP_X, WORKGROUP_Y),
        ));
    client.launch(
        task,
        CubeCount::new_2d(
            u32::try_from(OUTPUT_DIM / TILE_COLUMNS).ok()?,
            u32::try_from(rows.div_ceil(TILE_ROWS)).ok()?,
        ),
        KernelArguments::new()
            .with_buffer(attention.handle.binding())
            .with_buffer(attention_gate.handle.binding())
            .with_buffer(weight.handle.binding())
            .with_buffer(residual.handle.binding())
            .with_buffer(block_gate.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

#[allow(clippy::too_many_arguments)]
fn try_dit_projection_residual_wgsl(
    activated: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
    inner: usize,
    vectorized_input: bool,
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
    let precision =
        common_float_precision([activated.dtype, weight.dtype, residual.dtype, gate.dtype])?;
    let vec4_bytes = u64::try_from(4 * precision.element_bytes()).ok()?;
    let same_device = activated.device == weight.device
        && activated.device == residual.device
        && activated.device == gate.device;
    let input_row_stride = *activated.meta.strides().first()?;
    let required_input_elements = rows
        .checked_sub(1)?
        .checked_mul(input_row_stride)?
        .checked_add(inner)?;
    let supported_input_pitch =
        input_row_stride == inner || (inner == INPUT_DIM && input_row_stride == 2 * INPUT_DIM);
    let compatible = matches!(batch, 1..=3)
        && (MIN_SEQUENCE..=MAX_SEQUENCE).contains(&sequence)
        && matches!(inner, INPUT_DIM | OUTPUT_DIM)
        && inner.is_multiple_of(TILE_K)
        && activated.meta.shape().as_slice() == [rows, inner]
        && weight.meta.shape().as_slice() == [inner, OUTPUT_DIM]
        && residual.meta.shape().as_slice() == [rows, OUTPUT_DIM]
        && gate.meta.shape().as_slice() == [batch, OUTPUT_DIM]
        && activated.meta.strides()[1] == 1
        && (!vectorized_input || input_row_stride.is_multiple_of(4))
        && supported_input_pitch
        && weight.meta.strides()[..] == [OUTPUT_DIM, 1]
        && residual.meta.strides()[..] == [OUTPUT_DIM, 1]
        && gate.meta.strides()[..] == [OUTPUT_DIM, 1]
        && weight.is_contiguous()
        && residual.is_contiguous()
        && gate.is_contiguous()
        && same_device
        && binding_is_compatible(
            &activated,
            required_input_elements,
            precision,
            if vectorized_input {
                vec4_bytes
            } else {
                precision.element_bytes() as u64
            },
        )
        && binding_is_compatible(&weight, inner * OUTPUT_DIM, precision, vec4_bytes)
        && binding_is_compatible(&residual, output_elements, precision, vec4_bytes)
        && binding_is_compatible(&gate, batch * OUTPUT_DIM, precision, vec4_bytes);
    if !compatible {
        return None;
    }

    let hardware = &activated.client.properties().hardware;
    if hardware.max_bindings < REQUIRED_BINDINGS
        || hardware.max_shared_memory_size < SHARED_BYTES
        || hardware.max_units_per_cube < WORKGROUP_X * WORKGROUP_Y
        || hardware.max_cube_dim.0 < WORKGROUP_X
        || hardware.max_cube_dim.1 < WORKGROUP_Y
        || hardware.max_cube_count.0 < u32::try_from(OUTPUT_DIM / TILE_COLUMNS).ok()?
        || hardware.max_cube_count.1 < u32::try_from(rows.div_ceil(TILE_ROWS)).ok()?
    {
        return None;
    }

    let output_bytes = output_elements.checked_mul(precision.element_bytes())?;
    let client = activated.client.clone();
    let output_handle = client.empty(output_bytes);
    if output_handle.size_in_used() < u64::try_from(output_bytes).ok()?
        || !output_handle
            .offset_start
            .unwrap_or(0)
            .is_multiple_of(vec4_bytes)
    {
        return None;
    }
    let output = CubeTensor::new_contiguous(
        client.clone(),
        activated.device.clone(),
        Shape::from([rows, OUTPUT_DIM]),
        output_handle,
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DitMlpContractResidualKernel {
                precision,
                rows: u32::try_from(rows).ok()?,
                sequence: u32::try_from(sequence).ok()?,
                inner: u32::try_from(inner).ok()?,
                input_row_stride: u32::try_from(input_row_stride).ok()?,
                vectorized_input,
            },
            CubeDim::new_2d(WORKGROUP_X, WORKGROUP_Y),
        ));
    client.launch(
        task,
        CubeCount::new_2d(
            u32::try_from(OUTPUT_DIM / TILE_COLUMNS).ok()?,
            u32::try_from(rows.div_ceil(TILE_ROWS)).ok()?,
        ),
        KernelArguments::new()
            .with_buffer(activated.handle.binding())
            .with_buffer(weight.handle.binding())
            .with_buffer(residual.handle.binding())
            .with_buffer(gate.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::wgpu::WgpuDevice,
        tensor::{FloatDType, Tensor},
    };

    #[test]
    fn released_geometry_and_source_contract_are_fixed() {
        assert_eq!(SHARED_BYTES, 24_576);
        assert_eq!(OUTPUT_DIM / TILE_COLUMNS, 10);
        assert_eq!(WORKGROUP_X * WORKGROUP_Y, 256);
        let shader = include_str!("dit_mlp_contract_residual.wgsl");
        assert_eq!(shader.matches("@binding(").count(), 5);
        assert!(shader.contains("residual_value + gate_value * branch"));
        for accumulator in 0..8 {
            assert_eq!(
                shader.matches(&format!("acc_{accumulator} = fma")).count(),
                1
            );
        }

        let vectorized = include_str!("dit_mlp_contract_residual_vec4.wgsl");
        assert_eq!(vectorized.matches("@binding(").count(), 5);
        assert!(vectorized.contains("input: array<vec4<f32>>"));
        assert!(vectorized.contains("input_tile: array<vec4<f32>, 512>"));
        for accumulator in 0..8 {
            assert_eq!(
                vectorized
                    .matches(&format!("acc_{accumulator} = fma"))
                    .count(),
                4
            );
        }
    }

    #[test]
    fn vectorized_input_matches_scalar_for_contiguous_and_pitched_views() {
        #[cfg(feature = "cli")]
        let _ = crate::backend_config::initialize_cli_tracing("warn");
        let device: burn::tensor::Device = WgpuDevice::DefaultDevice.into();
        assert_eq!(device.settings().float_dtype, FloatDType::F32);
        let batch = 1;
        let sequence = MIN_SEQUENCE;
        let rows = batch * sequence;
        let weight = Tensor::<2>::ones([INPUT_DIM, OUTPUT_DIM], &device);
        let residual = Tensor::<2>::ones([rows, OUTPUT_DIM], &device);
        let gate = Tensor::<2>::ones([batch, OUTPUT_DIM], &device);

        for activated in [
            Tensor::<2>::ones([rows, INPUT_DIM], &device),
            Tensor::<2>::ones([rows, INPUT_DIM * 2], &device).slice([0..rows, 0..INPUT_DIM]),
        ] {
            let scalar = try_dit_mlp_contract_residual_wgsl(
                activated
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU activation"),
                weight
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU weight"),
                residual
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU residual"),
                gate.clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU gate"),
                batch,
                sequence,
            )
            .expect("scalar-input contract route");
            let vectorized = try_dit_mlp_contract_residual_vec4_wgsl(
                activated
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU activation"),
                weight
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU weight"),
                residual
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU residual"),
                gate.clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU gate"),
                batch,
                sequence,
            )
            .expect("vector-input contract route");
            let scalar = Tensor::<2>::from_primitive::<crate::WgpuRaw>(scalar)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let vectorized = Tensor::<2>::from_primitive::<crate::WgpuRaw>(vectorized)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(scalar, vectorized);
        }
    }
}
