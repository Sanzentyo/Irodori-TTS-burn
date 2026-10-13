//! Bit-preserving GPU-to-GPU copy into a caller-owned contiguous buffer.

use burn::backend::wgpu::{CubeDim, CubeTensor, KernelSource, SourceKernel};
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

const WORKGROUP_SIZE: u32 = 256;

#[derive(Debug)]
struct ContiguousCopyKernel {
    vectors: u32,
}

impl KernelSource for ContiguousCopyKernel {
    fn lang(&self) -> &'static str {
        "wgsl"
    }

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
    source: CubeTensor,
    destination: CubeTensor,
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
    // Allocator handles can include padding which is outside the logical
    // shader binding. Counting that padding admits an incomplete vec4 and
    // silently drops its tail through WebGPU's robust bounds handling.
    let bytes = source
        .meta
        .num_elements()
        .checked_mul(source.dtype.size())
        .ok_or("copy byte length overflow")?;
    if !bytes.is_multiple_of(16)
        || bytes as u64 > source.handle.size()
        || bytes as u64 > destination.handle.size()
    {
        return Err("logical copy byte length must fit both buffers and be divisible by 16");
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
    let task: Box<dyn cubecl::CubeKernel> = Box::new(SourceKernel::new(
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
