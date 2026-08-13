//! Exact-v4 text-only CFG packed K/V derivation for raw f32 WGPU inference.

use burn::{
    backend::wgpu::{
        CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuDevice, WgpuRuntime,
    },
    tensor::Shape,
};
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

use super::precision::KernelFloatPrecision;

pub const PLANES: usize = 2;
pub const CONDITIONAL_BATCH: usize = 1;
pub const CFG_BATCH: usize = 2;
pub const CONTEXT_LEN: usize = 3;
pub const NUM_HEADS: usize = 20;
pub const HEAD_DIM: usize = 64;
pub const MODEL_DIM: usize = NUM_HEADS * HEAD_DIM;
pub const WORKGROUP_SIZE: u32 = 256;

const REQUIRED_BINDINGS: u32 = 2;
const CONDITIONAL_ELEMENTS: usize = PLANES * CONDITIONAL_BATCH * CONTEXT_LEN * NUM_HEADS * HEAD_DIM;
const OUTPUT_ELEMENTS: usize = PLANES * CFG_BATCH * CONTEXT_LEN * NUM_HEADS * HEAD_DIM;
#[cfg(test)]
const F32_OUTPUT_BYTES: usize = OUTPUT_ELEMENTS * size_of::<f32>();
const WORKGROUPS: u32 = OUTPUT_ELEMENTS.div_ceil(WORKGROUP_SIZE as usize) as u32;

#[derive(Debug)]
struct TextCfgKvDeriveKernel {
    precision: KernelFloatPrecision,
}

impl KernelSource for TextCfgKvDeriveKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("text_cfg_kv_derive.wgsl"),
                include_str!("text_cfg_kv_derive_f16.wgsl"),
            )
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.precision)
    }
}

fn rank5_strides(tensor: &CubeTensor<WgpuRuntime>) -> [usize; 5] {
    let strides = tensor.meta.strides();
    [strides[0], strides[1], strides[2], strides[3], strides[4]]
}

/// Return whether the already-proven conditional cache satisfies the physical
/// kernel contract. This selector allocates and dispatches nothing.
pub fn supports_text_cfg_kv_derive(
    conditional: &CubeTensor<WgpuRuntime>,
    expected_device: &WgpuDevice,
) -> bool {
    if conditional.device != *expected_device
        || KernelFloatPrecision::from_dtype(conditional.dtype).is_none()
        || conditional.meta.num_dims() != 5
        || conditional.meta.shape().dims::<5>()
            != [PLANES, CONDITIONAL_BATCH, CONTEXT_LEN, NUM_HEADS, HEAD_DIM]
        || !conditional.is_contiguous()
        || rank5_strides(conditional)
            != [
                CONDITIONAL_ELEMENTS / PLANES,
                CONDITIONAL_ELEMENTS / PLANES,
                MODEL_DIM,
                HEAD_DIM,
                1,
            ]
    {
        return false;
    }

    let properties = conditional.client.properties();
    let hardware = &properties.hardware;
    let output_bytes = OUTPUT_ELEMENTS
        * KernelFloatPrecision::from_dtype(conditional.dtype)
            .expect("dtype was checked above")
            .element_bytes();
    u64::try_from(output_bytes).is_ok_and(|bytes| bytes <= properties.memory.max_page_size)
        && hardware.max_bindings >= REQUIRED_BINDINGS
        && hardware.max_units_per_cube >= WORKGROUP_SIZE
        && hardware.max_cube_dim.0 >= WORKGROUP_SIZE
        && hardware.max_cube_dim.1 >= 1
        && hardware.max_cube_dim.2 >= 1
        && hardware.max_cube_count.0 >= WORKGROUPS
        && hardware.max_cube_count.1 >= 1
        && hardware.max_cube_count.2 >= 1
}

/// Derive `[K/V,B=2,T=3,H=20,Dh=64]` from the canonical B=1 packed cache.
///
/// The sampler must separately hold a host proof that row zero is conditioned,
/// row one is a bias-free all-zero text condition, and no auxiliary context is
/// active. Any physical mismatch returns `None` without allocation or dispatch.
pub fn try_derive_text_cfg_kv_wgsl(
    conditional: CubeTensor<WgpuRuntime>,
    expected_device: &WgpuDevice,
) -> Option<CubeTensor<WgpuRuntime>> {
    if !supports_text_cfg_kv_derive(&conditional, expected_device) {
        return None;
    }

    let client = conditional.client.clone();
    let precision = KernelFloatPrecision::from_dtype(conditional.dtype)
        .expect("support check accepted only f32 or f16");
    let output = CubeTensor::new_contiguous(
        client.clone(),
        conditional.device.clone(),
        Shape::from([PLANES, CFG_BATCH, CONTEXT_LEN, NUM_HEADS, HEAD_DIM]),
        client.empty(OUTPUT_ELEMENTS * precision.element_bytes()),
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            TextCfgKvDeriveKernel { precision },
            CubeDim::new_1d(WORKGROUP_SIZE),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(conditional.handle.binding())
        .with_buffer(output.handle.clone().binding());
    client.launch(task, CubeCount::new_1d(WORKGROUPS), bindings);
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_dispatch_accounting_is_stable() {
        assert_eq!(CONDITIONAL_ELEMENTS, 7_680);
        assert_eq!(OUTPUT_ELEMENTS, 15_360);
        assert_eq!(F32_OUTPUT_BYTES, 61_440);
        assert_eq!(OUTPUT_ELEMENTS * size_of::<half::f16>(), 30_720);
        assert_eq!(WORKGROUPS, 60);
    }
}
