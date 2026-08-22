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
}

impl KernelSource for DitMlpContractResidualKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("dit_mlp_contract_residual.wgsl"),
                include_str!("dit_mlp_contract_residual_f16.wgsl"),
            )
            .register("rows", self.rows.to_string())
            .register("sequence", self.sequence.to_string())
            .register("inner", self.inner.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.rows, self.sequence, self.inner))
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
        activated, weight, residual, gate, batch, sequence, INPUT_DIM,
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
        attention, weight, residual, gate, batch, sequence, OUTPUT_DIM,
    )
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
    let compatible = matches!(batch, 1..=3)
        && (MIN_SEQUENCE..=MAX_SEQUENCE).contains(&sequence)
        && matches!(inner, INPUT_DIM | OUTPUT_DIM)
        && inner.is_multiple_of(TILE_K)
        && activated.meta.shape().as_slice() == [rows, inner]
        && weight.meta.shape().as_slice() == [inner, OUTPUT_DIM]
        && residual.meta.shape().as_slice() == [rows, OUTPUT_DIM]
        && gate.meta.shape().as_slice() == [batch, OUTPUT_DIM]
        && activated.meta.strides()[..] == [inner, 1]
        && weight.meta.strides()[..] == [OUTPUT_DIM, 1]
        && residual.meta.strides()[..] == [OUTPUT_DIM, 1]
        && gate.meta.strides()[..] == [OUTPUT_DIM, 1]
        && activated.is_contiguous()
        && weight.is_contiguous()
        && residual.is_contiguous()
        && gate.is_contiguous()
        && same_device
        && binding_is_compatible(
            &activated,
            rows * inner,
            precision,
            precision.element_bytes() as u64,
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
    }
}
