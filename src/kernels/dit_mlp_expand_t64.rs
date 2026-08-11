//! Exact-shape long-sequence DiT MLP expand GEMM.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::{DType, Shape};
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

pub const K: usize = 1_280;
pub const N: usize = 7_360;
const ADMITTED_ROWS: [usize; 2] = [200, 400];
const TILE_ROWS: usize = 64;
const TILE_COLUMNS: usize = 64;
const TILE_K: usize = 16;
const WORKGROUP_X: u32 = 16;
const WORKGROUP_Y: u32 = 16;
const REQUIRED_BINDINGS: u32 = 3;
const VEC4_BYTES: u64 = 16;
const SHARED_BYTES: usize = (TILE_ROWS * TILE_K + TILE_K * TILE_COLUMNS) * size_of::<f32>();

#[derive(Debug)]
struct DitMlpExpandT64Kernel {
    rows: u32,
}

impl KernelSource for DitMlpExpandT64Kernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("dit_mlp_expand_t64.wgsl"))
            .register("rows", self.rows.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.rows)
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

/// Launch only for dense released B1/B2 S200 rows and packed row-major weight.
/// Every contract mismatch returns `None` to preserve the tuned Burn fallback.
pub fn try_dit_mlp_expand_t64_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    if input.meta.num_dims() != 2 || weight.meta.num_dims() != 2 {
        return None;
    }
    let rows = input.meta.shape()[0];
    let output_elements = rows.checked_mul(N)?;
    let compatible = ADMITTED_ROWS.contains(&rows)
        && input.dtype == DType::F32
        && weight.dtype == DType::F32
        && input.meta.shape().as_slice() == [rows, K]
        && weight.meta.shape().as_slice() == [K, N]
        && input.meta.strides()[..] == [K, 1]
        && weight.meta.strides()[..] == [N, 1]
        && input.is_contiguous()
        && weight.is_contiguous()
        && input.device == weight.device
        && binding_is_compatible(&input, rows * K, size_of::<f32>() as u64)
        && binding_is_compatible(&weight, K * N, VEC4_BYTES);
    if !compatible {
        return None;
    }
    let hardware = &input.client.properties().hardware;
    if hardware.max_bindings < REQUIRED_BINDINGS
        || hardware.max_shared_memory_size < SHARED_BYTES
        || hardware.max_units_per_cube < WORKGROUP_X * WORKGROUP_Y
        || hardware.max_cube_dim.0 < WORKGROUP_X
        || hardware.max_cube_dim.1 < WORKGROUP_Y
        || hardware.max_cube_count.0 < u32::try_from(N / TILE_COLUMNS).ok()?
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
        Shape::from([rows, N]),
        output_handle,
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DitMlpExpandT64Kernel {
                rows: u32::try_from(rows).ok()?,
            },
            CubeDim::new_2d(WORKGROUP_X, WORKGROUP_Y),
        ));
    client.launch(
        task,
        CubeCount::new_2d(
            u32::try_from(N / TILE_COLUMNS).ok()?,
            u32::try_from(rows.div_ceil(TILE_ROWS)).ok()?,
        ),
        KernelArguments::new()
            .with_buffer(input.handle.binding())
            .with_buffer(weight.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_geometry_and_accounting_are_stable() {
        assert_eq!(N % TILE_COLUMNS, 0);
        assert_eq!(K % TILE_K, 0);
        assert_eq!(SHARED_BYTES, 8_192);
        assert_eq!(WORKGROUP_X * WORKGROUP_Y, 256);
        assert_eq!(N / TILE_COLUMNS, 115);
        assert_eq!(200_usize.div_ceil(TILE_ROWS), 4);
        assert_eq!(400_usize.div_ceil(TILE_ROWS), 7);
    }

    #[test]
    fn shader_keeps_k_ascending_and_vec4_weight_output() {
        let shader = include_str!("dit_mlp_expand_t64.wgsl");
        assert_eq!(shader.matches("array<vec4<f32>>").count(), 2);
        assert_eq!(shader.matches("var<storage, read_write>").count(), 3);
        assert!(shader.contains("k_base = k_base + TILE_K"));
        assert!(shader.contains("tile_k_index = tile_k_index + 1u"));
        assert_eq!(shader.matches("acc_0 = fma").count(), 1);
        assert_eq!(shader.matches("acc_1 = fma").count(), 1);
        assert_eq!(shader.matches("acc_2 = fma").count(), 1);
        assert_eq!(shader.matches("acc_3 = fma").count(), 1);
    }
}
