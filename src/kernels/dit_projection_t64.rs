//! Exact-shape long-sequence DiT projection GEMMs.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::{DType, Shape};
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

pub const EXPAND_K: usize = 1_280;
pub const EXPAND_N: usize = 7_360;
pub const CONTRACT_K: usize = 3_680;
pub const CONTRACT_N: usize = 1_280;
pub const ATTENTION_QKV_GATE_K: usize = 1_280;
pub const ATTENTION_QKV_GATE_N: usize = 5_120;
pub const ATTENTION_OUTPUT_K: usize = 1_280;
pub const ATTENTION_OUTPUT_N: usize = 1_280;
pub const DURATION_EXPAND_K: usize = 1_024;
pub const DURATION_EXPAND_N: usize = 2_048;
pub const DURATION_INPUT_K: usize = 512;
pub const DURATION_INPUT_N: usize = 1_024;
const DIT_ADMITTED_ROWS: [usize; 7] = [100, 200, 333, 400, 666, 685, 1_370];
const DURATION_MAX_ROWS: usize = 64;
const TILE_ROWS: usize = 64;
const LONG_TILE_ROWS: usize = 128;
const TILE_COLUMNS: usize = 64;
const TILE_K: usize = 16;
const WORKGROUP_X: u32 = 16;
const WORKGROUP_Y: u32 = 16;
const REQUIRED_BINDINGS: u32 = 3;
const VEC4_BYTES: u64 = 16;
const SHARED_BYTES: usize = (TILE_ROWS * TILE_K + TILE_K * TILE_COLUMNS) * size_of::<f32>();
const LONG_SHARED_BYTES: usize =
    (LONG_TILE_ROWS * TILE_K + TILE_K * TILE_COLUMNS) * size_of::<f32>();

#[derive(Debug)]
struct DitProjectionT64Kernel {
    rows: u32,
    inner: u32,
    columns: u32,
}

#[derive(Debug)]
struct DitProjectionT128Kernel {
    rows: u32,
    inner: u32,
    columns: u32,
}

#[derive(Debug)]
struct DurationInputProjectionT64Kernel {
    rows: u32,
}

impl KernelSource for DurationInputProjectionT64Kernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("duration_input_projection_t64.wgsl"))
            .register("rows", self.rows.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.rows)
    }
}

impl KernelSource for DitProjectionT64Kernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("dit_projection_t64.wgsl"))
            .register("rows", self.rows.to_string())
            .register("inner", self.inner.to_string())
            .register("columns", self.columns.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.rows, self.inner, self.columns))
    }
}

impl KernelSource for DitProjectionT128Kernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("dit_projection_t128.wgsl"))
            .register("rows", self.rows.to_string())
            .register("inner", self.inner.to_string())
            .register("columns", self.columns.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.rows, self.inner, self.columns))
    }
}

fn binding_is_compatible(
    tensor: &CubeTensor<WgpuRuntime>,
    required_elements: usize,
    alignment: u64,
) -> bool {
    let Some(required_bytes) = required_elements
        .checked_mul(size_of::<f32>())
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

/// Launch only for dense released B1/B2 measured-length rows and packed
/// row-major weight.
/// Every contract mismatch returns `None` to preserve the tuned Burn fallback.
fn try_dit_projection_t64_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    inner: usize,
    columns: usize,
    rows_are_admitted: fn(usize) -> bool,
    use_long_tile: bool,
) -> Option<CubeTensor<WgpuRuntime>> {
    if input.meta.num_dims() != 2 || weight.meta.num_dims() != 2 {
        return None;
    }
    let rows = input.meta.shape()[0];
    let output_elements = rows.checked_mul(columns)?;
    let compatible = rows_are_admitted(rows)
        && inner.is_multiple_of(TILE_K)
        && columns.is_multiple_of(TILE_COLUMNS)
        && input.dtype == DType::F32
        && weight.dtype == DType::F32
        && input.meta.shape().as_slice() == [rows, inner]
        && weight.meta.shape().as_slice() == [inner, columns]
        && input.meta.strides()[..] == [inner, 1]
        && weight.meta.strides()[..] == [columns, 1]
        && input.is_contiguous()
        && weight.is_contiguous()
        && input.device == weight.device
        && binding_is_compatible(&input, rows * inner, size_of::<f32>() as u64)
        && binding_is_compatible(&weight, inner * columns, VEC4_BYTES);
    if !compatible {
        return None;
    }
    let hardware = &input.client.properties().hardware;
    let tile_rows = if use_long_tile {
        LONG_TILE_ROWS
    } else {
        TILE_ROWS
    };
    let shared_bytes = if use_long_tile {
        LONG_SHARED_BYTES
    } else {
        SHARED_BYTES
    };
    if hardware.max_bindings < REQUIRED_BINDINGS
        || hardware.max_shared_memory_size < shared_bytes
        || hardware.max_units_per_cube < WORKGROUP_X * WORKGROUP_Y
        || hardware.max_cube_dim.0 < WORKGROUP_X
        || hardware.max_cube_dim.1 < WORKGROUP_Y
        || hardware.max_cube_count.0 < u32::try_from(columns / TILE_COLUMNS).ok()?
        || hardware.max_cube_count.1 < u32::try_from(rows.div_ceil(tile_rows)).ok()?
    {
        return None;
    }

    let output_bytes = output_elements.checked_mul(size_of::<f32>())?;
    let client = input.client.clone();
    let output_handle = client.empty(output_bytes);
    if output_handle.size_in_used() < u64::try_from(output_bytes).ok()?
        || !output_handle
            .offset_start
            .unwrap_or(0)
            .is_multiple_of(VEC4_BYTES)
    {
        return None;
    }
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([rows, columns]),
        output_handle,
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> = if use_long_tile {
        Box::new(SourceKernel::new(
            DitProjectionT128Kernel {
                rows: u32::try_from(rows).ok()?,
                inner: u32::try_from(inner).ok()?,
                columns: u32::try_from(columns).ok()?,
            },
            CubeDim::new_2d(WORKGROUP_X, WORKGROUP_Y),
        ))
    } else {
        Box::new(SourceKernel::new(
            DitProjectionT64Kernel {
                rows: u32::try_from(rows).ok()?,
                inner: u32::try_from(inner).ok()?,
                columns: u32::try_from(columns).ok()?,
            },
            CubeDim::new_2d(WORKGROUP_X, WORKGROUP_Y),
        ))
    };
    client.launch(
        task,
        CubeCount::new_2d(
            u32::try_from(columns / TILE_COLUMNS).ok()?,
            u32::try_from(rows.div_ceil(tile_rows)).ok()?,
        ),
        KernelArguments::new()
            .with_buffer(input.handle.binding())
            .with_buffer(weight.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

fn dit_rows_are_admitted(rows: usize) -> bool {
    DIT_ADMITTED_ROWS.contains(&rows)
}

/// Exact latent lengths admitted by the measured production projection route.
pub const fn dit_sequence_is_admitted(sequence: usize) -> bool {
    matches!(sequence, 100 | 200 | 333 | 685)
}

const fn duration_rows_are_admitted(rows: usize) -> bool {
    rows > 0 && rows <= DURATION_MAX_ROWS
}

/// Launch the exact released `w1 || w3` projection.
pub fn try_dit_mlp_expand_t64_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_t64_wgsl(
        input,
        weight,
        EXPAND_K,
        EXPAND_N,
        dit_rows_are_admitted,
        true,
    )
}

/// Launch the exact released `w2` projection.
pub fn try_dit_mlp_contract_t64_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_t64_wgsl(
        input,
        weight,
        CONTRACT_K,
        CONTRACT_N,
        dit_rows_are_admitted,
        true,
    )
}

/// Launch the exact released long-sequence `QKV || gate` projection.
pub fn try_dit_attention_qkv_gate_t64_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_t64_wgsl(
        input,
        weight,
        ATTENTION_QKV_GATE_K,
        ATTENTION_QKV_GATE_N,
        dit_rows_are_admitted,
        true,
    )
}

/// Launch the exact released long-sequence attention output projection.
pub fn try_dit_attention_output_t64_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_t64_wgsl(
        input,
        weight,
        ATTENTION_OUTPUT_K,
        ATTENTION_OUTPUT_N,
        dit_rows_are_admitted,
        true,
    )
}

/// Launch the exact released duration-block `w1 || w3` projection.
///
/// This candidate consumes the already fused, contiguous GPU-resident weight;
/// it performs no packing, host transfer, or readback.
pub fn try_duration_mlp_expand_t64_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_t64_wgsl(
        input,
        weight,
        DURATION_EXPAND_K,
        DURATION_EXPAND_N,
        duration_rows_are_admitted,
        false,
    )
}

/// Launch the released duration input projection with bias in one dispatch.
///
/// The text embedding, checkpoint-native output-major weight view, and bias remain on
/// the same WGPU device. No packing or host transfer is introduced.
pub fn try_duration_input_projection_t64_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    if input.meta.num_dims() != 2 || weight.meta.num_dims() != 2 || bias.meta.num_dims() != 1 {
        return None;
    }
    let rows = input.meta.shape()[0];
    let output_elements = rows.checked_mul(DURATION_INPUT_N)?;
    let compatible = duration_rows_are_admitted(rows)
        && input.dtype == DType::F32
        && weight.dtype == DType::F32
        && bias.dtype == DType::F32
        && input.meta.shape().as_slice() == [rows, DURATION_INPUT_K]
        && weight.meta.shape().as_slice() == [DURATION_INPUT_K, DURATION_INPUT_N]
        && bias.meta.shape().as_slice() == [DURATION_INPUT_N]
        && input.meta.strides()[..] == [DURATION_INPUT_K, 1]
        && weight.meta.strides()[..] == [1, DURATION_INPUT_K]
        && bias.meta.strides()[..] == [1]
        && input.is_contiguous()
        && !weight.is_contiguous()
        && bias.is_contiguous()
        && input.device == weight.device
        && input.device == bias.device
        && binding_is_compatible(&input, rows * DURATION_INPUT_K, size_of::<f32>() as u64)
        && binding_is_compatible(&weight, DURATION_INPUT_K * DURATION_INPUT_N, VEC4_BYTES)
        && binding_is_compatible(&bias, DURATION_INPUT_N, VEC4_BYTES);
    if !compatible {
        return None;
    }
    let hardware = &input.client.properties().hardware;
    if hardware.max_bindings < 4
        || hardware.max_shared_memory_size < SHARED_BYTES
        || hardware.max_units_per_cube < WORKGROUP_X * WORKGROUP_Y
        || hardware.max_cube_dim.0 < WORKGROUP_X
        || hardware.max_cube_dim.1 < WORKGROUP_Y
        || hardware.max_cube_count.0 < u32::try_from(DURATION_INPUT_N / TILE_COLUMNS).ok()?
        || hardware.max_cube_count.1 < u32::try_from(rows.div_ceil(TILE_ROWS)).ok()?
    {
        return None;
    }

    let output_bytes = output_elements.checked_mul(size_of::<f32>())?;
    let client = input.client.clone();
    let output_handle = client.empty(output_bytes);
    if output_handle.size_in_used() < u64::try_from(output_bytes).ok()?
        || !output_handle
            .offset_start
            .unwrap_or(0)
            .is_multiple_of(VEC4_BYTES)
    {
        return None;
    }
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([rows, DURATION_INPUT_N]),
        output_handle,
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DurationInputProjectionT64Kernel {
                rows: u32::try_from(rows).ok()?,
            },
            CubeDim::new_2d(WORKGROUP_X, WORKGROUP_Y),
        ));
    client.launch(
        task,
        CubeCount::new_2d(
            u32::try_from(DURATION_INPUT_N / TILE_COLUMNS).ok()?,
            u32::try_from(rows.div_ceil(TILE_ROWS)).ok()?,
        ),
        KernelArguments::new()
            .with_buffer(input.handle.binding())
            .with_buffer(weight.handle.binding())
            .with_buffer(bias.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_geometry_and_accounting_are_stable() {
        assert_eq!(EXPAND_N % TILE_COLUMNS, 0);
        assert_eq!(EXPAND_K % TILE_K, 0);
        assert_eq!(CONTRACT_N % TILE_COLUMNS, 0);
        assert_eq!(CONTRACT_K % TILE_K, 0);
        assert_eq!(ATTENTION_QKV_GATE_N % TILE_COLUMNS, 0);
        assert_eq!(ATTENTION_QKV_GATE_K % TILE_K, 0);
        assert_eq!(ATTENTION_OUTPUT_N % TILE_COLUMNS, 0);
        assert_eq!(ATTENTION_OUTPUT_K % TILE_K, 0);
        assert_eq!(DURATION_EXPAND_N % TILE_COLUMNS, 0);
        assert_eq!(DURATION_EXPAND_K % TILE_K, 0);
        assert_eq!(DURATION_INPUT_N % TILE_COLUMNS, 0);
        assert_eq!(DURATION_INPUT_K % TILE_K, 0);
        assert_eq!(SHARED_BYTES, 8_192);
        assert_eq!(LONG_SHARED_BYTES, 12_288);
        assert_eq!(WORKGROUP_X * WORKGROUP_Y, 256);
        assert_eq!(EXPAND_N / TILE_COLUMNS, 115);
        assert_eq!(CONTRACT_N / TILE_COLUMNS, 20);
        assert_eq!(ATTENTION_QKV_GATE_N / TILE_COLUMNS, 80);
        assert_eq!(ATTENTION_OUTPUT_N / TILE_COLUMNS, 20);
        assert_eq!(DURATION_EXPAND_N / TILE_COLUMNS, 32);
        assert_eq!(DURATION_INPUT_N / TILE_COLUMNS, 16);
        assert_eq!(100_usize.div_ceil(LONG_TILE_ROWS), 1);
        assert_eq!(200_usize.div_ceil(LONG_TILE_ROWS), 2);
        assert_eq!(400_usize.div_ceil(LONG_TILE_ROWS), 4);
        assert_eq!(333_usize.div_ceil(LONG_TILE_ROWS), 3);
        assert_eq!(666_usize.div_ceil(LONG_TILE_ROWS), 6);
        assert_eq!(685_usize.div_ceil(LONG_TILE_ROWS), 6);
        assert_eq!(1_370_usize.div_ceil(LONG_TILE_ROWS), 11);
        for sequence in [100, 200, 333, 685] {
            assert!(dit_sequence_is_admitted(sequence));
            assert!(dit_rows_are_admitted(sequence));
            assert!(dit_rows_are_admitted(sequence * 2));
        }
        assert!(!dit_sequence_is_admitted(50));
        assert!(!dit_sequence_is_admitted(334));
        assert_eq!(3_usize.div_ceil(TILE_ROWS), 1);
        assert_eq!(12_usize.div_ceil(TILE_ROWS), 1);
        assert_eq!(28_usize.div_ceil(TILE_ROWS), 1);
        assert_eq!(61_usize.div_ceil(TILE_ROWS), 1);
        assert!(duration_rows_are_admitted(1));
        assert!(duration_rows_are_admitted(64));
        assert!(!duration_rows_are_admitted(0));
        assert!(!duration_rows_are_admitted(65));
    }

    #[test]
    fn shader_keeps_k_ascending_and_vec4_weight_output() {
        let shader = include_str!("dit_projection_t64.wgsl");
        assert_eq!(shader.matches("array<vec4<f32>>").count(), 2);
        assert_eq!(shader.matches("var<storage, read_write>").count(), 3);
        assert!(shader.contains("k_base = k_base + TILE_K"));
        assert!(shader.contains("tile_k_index = tile_k_index + 1u"));
        assert_eq!(shader.matches("acc_0 = fma").count(), 1);
        assert_eq!(shader.matches("acc_1 = fma").count(), 1);
        assert_eq!(shader.matches("acc_2 = fma").count(), 1);
        assert_eq!(shader.matches("acc_3 = fma").count(), 1);
        assert_eq!(shader.matches("acc_4 = fma").count(), 0);

        let long_shader = include_str!("dit_projection_t128.wgsl");
        assert_eq!(long_shader.matches("array<vec4<f32>>").count(), 2);
        assert_eq!(long_shader.matches("var<storage, read_write>").count(), 3);
        for accumulator in 0..8 {
            assert_eq!(
                long_shader
                    .matches(&format!("acc_{accumulator} = fma"))
                    .count(),
                1
            );
        }

        let duration_input = include_str!("duration_input_projection_t64.wgsl");
        assert_eq!(duration_input.matches("@binding(").count(), 4);
        assert_eq!(duration_input.matches(" = fma(").count(), 4);
        assert_eq!(duration_input.matches(" + bias_value;").count(), 4);
        assert!(duration_input.contains("k_base = k_base + TILE_K"));
    }
}
