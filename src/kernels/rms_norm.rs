//! Custom RMSNorm WGSL kernel for the WGPU backend.
//!
//! Single-pass RMSNorm: `output = (x / sqrt(mean(x²) + eps)) * weight`.
//! One workgroup per row (token position); shared-memory parallel reduction.
//!
//! Platform compatibility:
//! - Uses workgroup shared memory only (WGSL core).
//! - No subgroup operations — WebGPU forward-compatible.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime, into_contiguous,
};
use burn::tensor::Shape;
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

/// RMSNorm kernel with baked-in dimension, workgroup size, and epsilon.
///
/// Each unique `(dim, workgroup_size, eps)` compiles to a distinct pipeline,
/// cached by `KernelId`.
#[derive(Debug)]
pub(crate) struct RmsNormKernel {
    workgroup_size: u32,
    dim: u32,
    eps: f64,
}

impl RmsNormKernel {
    pub(crate) fn new(dim: u32, workgroup_size: u32, eps: f64) -> Self {
        Self {
            workgroup_size,
            dim,
            eps,
        }
    }
}

impl KernelSource for RmsNormKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("rms_norm.wgsl"))
            .register("workgroup_size_x", self.workgroup_size.to_string())
            .register("dim", self.dim.to_string())
            .register("elem", "f32")
            .register("eps", format!("{:e}", self.eps))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(self.dim)
            .info(self.workgroup_size)
            .info(self.eps.to_bits())
    }
}

/// Launch the custom RMSNorm kernel on the WGPU backend.
///
/// # Arguments
/// - `input`: `[num_rows, dim]` tensor (will be made contiguous if needed)
/// - `weight`: `[dim]` scale parameter
/// - `eps`: epsilon for numerical stability
///
/// # Returns
/// Output tensor with same shape as input.
pub(crate) fn rms_norm_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    eps: f64,
) -> CubeTensor<WgpuRuntime> {
    let input = into_contiguous(input);

    let ndims = input.meta.num_dims();
    assert_eq!(ndims, 2, "RMSNorm kernel expects 2D input [rows, dim]");

    let num_rows = input.meta.shape()[0];
    let dim = input.meta.shape()[1];

    // Workgroup size: min(256, next_power_of_2(dim)) for efficient reduction.
    let workgroup_size = (dim as u32).next_power_of_two().min(256);

    let client = input.client.clone();
    let device = input.device.clone();

    // All WGSL bindings are declared read_write (same usage), so the WGPU
    // suballocator can safely pack input/output into the same physical buffer.
    let output_handle = client.empty(num_rows * dim * core::mem::size_of::<f32>());
    let output = CubeTensor::new_contiguous(
        client.clone(),
        device,
        Shape::from([num_rows, dim]),
        output_handle,
        input.dtype,
    );

    // One workgroup per row.
    let cube_count = CubeCount::new_1d(num_rows as u32);
    let cube_dim = CubeDim::new_1d(workgroup_size);

    let kernel = RmsNormKernel::new(dim as u32, workgroup_size, eps);
    let kernel_box: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(kernel, cube_dim));

    let bindings = KernelArguments::new()
        .with_buffer(input.handle.binding())
        .with_buffer(weight.handle.binding())
        .with_buffer(output.handle.clone().binding());

    client.launch(kernel_box, cube_count, bindings);

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::graphics::AutoGraphicsApi;
    use burn::backend::wgpu::{WgpuDevice, init_setup};
    use burn::tensor::{Tensor, backend::Backend};

    /// Non-fusion WGPU backend for direct CubeTensor access.
    type WgpuRaw = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;

    fn setup_device() -> <WgpuRaw as Backend>::Device {
        let device = WgpuDevice::DefaultDevice;
        init_setup::<AutoGraphicsApi>(&device, Default::default());
        device
    }

    /// Reference RMSNorm on CPU for comparison.
    fn reference_rms_norm(input: &[f32], weight: &[f32], dim: usize, eps: f64) -> Vec<f32> {
        let num_rows = input.len() / dim;
        let mut output = vec![0.0f32; input.len()];
        for row in 0..num_rows {
            let start = row * dim;
            let end = start + dim;
            let row_data = &input[start..end];
            let mean_sq: f32 = row_data.iter().map(|x| x * x).sum::<f32>() / dim as f32;
            let rms = (mean_sq + eps as f32).sqrt();
            for (j, &x) in row_data.iter().enumerate() {
                output[start + j] = (x / rms) * weight[j];
            }
        }
        output
    }

    /// Test with dim=256 (1KB per row) to avoid WGPU suballocator page sharing.
    #[test]
    fn rms_norm_kernel_matches_reference() {
        let device = setup_device();

        const DIM: usize = 256;
        const ROWS: usize = 4;

        // Deterministic input pattern: row_i col_j = (i * DIM + j + 1) / (DIM * ROWS)
        let input_data: Vec<f32> = (0..ROWS * DIM)
            .map(|i| (i as f32 + 1.0) / (ROWS * DIM) as f32)
            .collect();
        let weight_data: Vec<f32> = (0..DIM).map(|j| 1.0 + 0.01 * j as f32).collect();
        let eps = 1e-6;

        let expected = reference_rms_norm(&input_data, &weight_data, DIM, eps);

        let input_tensor =
            Tensor::<WgpuRaw, 1>::from_floats(input_data.as_slice(), &device).reshape([ROWS, DIM]);

        let weight_tensor = Tensor::<WgpuRaw, 1>::from_floats(weight_data.as_slice(), &device);

        let input_prim = input_tensor.into_primitive().tensor();
        let weight_prim = weight_tensor.into_primitive().tensor();
        let output_prim = rms_norm_wgsl(input_prim, weight_prim, eps);

        let output_tensor =
            Tensor::<WgpuRaw, 2>::from_primitive(burn::tensor::TensorPrimitive::Float(output_prim));
        let output_data = output_tensor.into_data().to_vec::<f32>().unwrap();

        for (i, (got, want)) in output_data.iter().zip(expected.iter()).enumerate() {
            let diff = (got - want).abs();
            assert!(
                diff < 1e-4,
                "index {i}: got {got}, want {want}, diff {diff}"
            );
        }
    }
}
