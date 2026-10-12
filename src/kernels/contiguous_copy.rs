//! Bit-preserving GPU-to-GPU copy into a caller-owned contiguous buffer.

use burn::backend::wgpu::{CubeDim, CubeTensor, KernelSource, SourceKernel, WgpuRuntime};
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

const WORKGROUP_SIZE: u32 = 256;

#[derive(Debug)]
struct ContiguousCopyKernel {
    vectors: u32,
}

impl KernelSource for ContiguousCopyKernel {
    fn source(&self) -> burn::backend::wgpu::SourceTemplate {
        burn::backend::wgpu::SourceTemplate::new(include_str!("contiguous_copy.wgsl"))
            .register("vectors", self.vectors.to_string())
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.vectors)
    }
}

/// Copy equal-shaped contiguous storage without changing its bit pattern.
///
/// The destination is caller-owned so a long-lived graph can keep a stable
/// input address while each request supplies a different GPU-resident latent.
/// The byte length must be a multiple of 16; supported codec latents satisfy
/// this for both F16 and F32.
pub fn copy_contiguous_into_wgsl(
    source: CubeTensor<WgpuRuntime>,
    destination: CubeTensor<WgpuRuntime>,
) -> Result<(), &'static str> {
    if source.dtype != destination.dtype {
        return Err("source and destination dtypes differ");
    }
    if source.meta.shape() != destination.meta.shape() {
        return Err("source and destination shapes differ");
    }
    if !source.is_contiguous() || !destination.is_contiguous() {
        return Err("source and destination must be contiguous");
    }
    if source.device != destination.device {
        return Err("source and destination devices differ");
    }
    let bytes = source.handle.size();
    if bytes != destination.handle.size() || bytes % 16 != 0 {
        return Err("source and destination byte lengths must match and be divisible by 16");
    }
    let vectors = u32::try_from(bytes / 16).map_err(|_| "copy length exceeds u32")?;
    if vectors == 0 {
        return Ok(());
    }
    let workgroups = vectors.div_ceil(WORKGROUP_SIZE);
    let limits = source.client.properties().hardware.max_cube_count;
    if workgroups > limits.0 {
        return Err("copy dispatch exceeds device x workgroup limit");
    }
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            ContiguousCopyKernel { vectors },
            CubeDim::new_1d(WORKGROUP_SIZE),
        ));
    source.client.launch(
        task,
        CubeCount::new_1d(workgroups),
        KernelArguments::new()
            .with_buffer(source.handle.binding())
            .with_buffer(destination.handle.binding()),
    );
    Ok(())
}
