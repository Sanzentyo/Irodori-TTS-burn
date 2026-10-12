//! FP16 SiLU with FP32 arithmetic and a single final rounding.
use burn::{
    backend::wgpu::{
        CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
        into_contiguous,
    },
    tensor::{DType, Shape},
};
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

#[derive(Debug)]
struct SiluF16 {
    elements: u32,
}
impl KernelSource for SiluF16 {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("silu_f16.wgsl"))
            .register("elements", self.elements.to_string())
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(self.elements)
    }
}
pub(crate) fn silu_f16(input: CubeTensor<WgpuRuntime>) -> CubeTensor<WgpuRuntime> {
    assert_eq!(input.dtype, DType::F16);
    let input = into_contiguous(input);
    let count = input.meta.shape().iter().product::<usize>();
    let elements = u32::try_from(count).expect("SiLU tensor exceeds WGSL indexing");
    if count == 0 {
        return input;
    }
    let output = CubeTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        Shape::from(input.meta.shape()),
        input.client.empty(count * 2),
        DType::F16,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> = Box::new(
        SourceKernel::new(SiluF16 { elements }, CubeDim::new_1d(256)),
    );
    input.client.launch(
        task,
        CubeCount::new_1d(elements.div_ceil(256)),
        KernelArguments::new()
            .with_buffer(input.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    output
}
