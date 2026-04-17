//! Fused AdaLN WGSL kernel for the WGPU backend.
//!
//! Single-pass RMSNorm + modulation:
//! `output = (x / sqrt(mean(x²) + eps)) * (1 + scale) + shift`
//!
//! Fuses the RMSNorm and modulation steps of [`LowRankAdaLn::forward`]
//! into a single kernel launch, eliminating intermediate tensors.
//!
//! Platform compatibility:
//! - Uses workgroup shared memory only (WGSL core).
//! - No subgroup operations — WebGPU forward-compatible.
//! - 1D dispatch: one workgroup per row (B*S total workgroups).
//!   `seq_len` is baked as a template parameter (constant per inference run).

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime, into_contiguous,
};
use burn::tensor::Shape;
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

/// Fused AdaLN kernel with baked-in dimension, seq_len, and epsilon.
///
/// Template parameters: `dim`, `seq_len`, `workgroup_size`, `shared_mem_size`, `eps`.
/// `seq_len` is templated because it determines the batch-index derivation
/// inside the shader. For inference, seq_len is constant across all 40 diffusion
/// steps, so the pipeline is compiled once and reused.
#[derive(Debug)]
struct FusedAdaLnKernel {
    workgroup_size: u32,
    dim: u32,
    seq_len: u32,
    shared_mem_size: u32,
    eps: f64,
}

impl FusedAdaLnKernel {
    fn new(dim: u32, seq_len: u32, workgroup_size: u32, eps: f64) -> Self {
        let shared_mem_size = dim.max(workgroup_size);
        Self {
            workgroup_size,
            dim,
            seq_len,
            shared_mem_size,
            eps,
        }
    }
}

impl KernelSource for FusedAdaLnKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("fused_adaln.wgsl"))
            .register("workgroup_size_x", self.workgroup_size.to_string())
            .register("dim", self.dim.to_string())
            .register("seq_len", self.seq_len.to_string())
            .register("shared_mem_size", self.shared_mem_size.to_string())
            .register("elem", "f32")
            .register("eps", format!("{:e}", self.eps))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(self.dim)
            .info(self.seq_len)
            .info(self.workgroup_size)
            .info(self.eps.to_bits())
    }
}

/// Launch the fused AdaLN kernel on the WGPU backend.
///
/// Computes `(x / rms) * (1 + scale) + shift` in a single pass.
///
/// # Arguments
/// - `input`: `[batch * seq_len, dim]` f32 tensor (flattened from `[B, S, D]`)
/// - `scale`: `[batch, dim]` f32 tensor (from `[B, 1, D]` squeezed)
/// - `shift`: `[batch, dim]` f32 tensor (from `[B, 1, D]` squeezed)
/// - `batch_size`: B dimension
/// - `seq_len`: S dimension (baked into kernel as template parameter)
/// - `eps`: RMSNorm epsilon
///
/// # Panics
/// - If shapes don't match expectations
/// - If any tensor is not f32
///
/// # Returns
/// Output tensor with same shape as input: `[batch * seq_len, dim]`.
pub fn fused_adaln_wgsl(
    input: CubeTensor<WgpuRuntime>,
    scale: CubeTensor<WgpuRuntime>,
    shift: CubeTensor<WgpuRuntime>,
    batch_size: usize,
    seq_len: usize,
    eps: f64,
) -> CubeTensor<WgpuRuntime> {
    // Validate dtypes.
    assert_eq!(
        input.dtype,
        burn::tensor::DType::F32,
        "fused AdaLN kernel only supports f32 input"
    );
    assert_eq!(
        scale.dtype,
        burn::tensor::DType::F32,
        "fused AdaLN kernel only supports f32 scale"
    );
    assert_eq!(
        shift.dtype,
        burn::tensor::DType::F32,
        "fused AdaLN kernel only supports f32 shift"
    );

    // Ensure contiguity — raw buffer indexing requires dense layout.
    let input = into_contiguous(input);
    let scale = into_contiguous(scale);
    let shift = into_contiguous(shift);

    // Validate shapes.
    let input_ndims = input.meta.num_dims();
    assert_eq!(
        input_ndims, 2,
        "fused AdaLN expects 2D input [B*S, D], got {input_ndims}D"
    );
    let total_rows = input.meta.shape()[0];
    let dim = input.meta.shape()[1];
    assert!(dim > 0, "fused AdaLN requires dim > 0");
    assert_eq!(
        total_rows,
        batch_size * seq_len,
        "input rows {} != batch_size {} * seq_len {}",
        total_rows,
        batch_size,
        seq_len
    );

    assert_eq!(scale.meta.num_dims(), 2, "scale must be 2D [B, D]");
    assert_eq!(
        scale.meta.shape()[0],
        batch_size,
        "scale batch {} != {}",
        scale.meta.shape()[0],
        batch_size
    );
    assert_eq!(
        scale.meta.shape()[1],
        dim,
        "scale dim {} != {}",
        scale.meta.shape()[1],
        dim
    );

    assert_eq!(shift.meta.num_dims(), 2, "shift must be 2D [B, D]");
    assert_eq!(
        shift.meta.shape()[0],
        batch_size,
        "shift batch {} != {}",
        shift.meta.shape()[0],
        batch_size
    );
    assert_eq!(
        shift.meta.shape()[1],
        dim,
        "shift dim {} != {}",
        shift.meta.shape()[1],
        dim
    );

    // Workgroup size: min(256, next_power_of_2(dim)) for efficient reduction.
    let workgroup_size = (dim as u32).next_power_of_two().min(256);

    let client = input.client.clone();
    let device = input.device.clone();

    let output_handle = client.empty(total_rows * dim * core::mem::size_of::<f32>());
    let output = CubeTensor::new_contiguous(
        client.clone(),
        device,
        Shape::from([total_rows, dim]),
        output_handle,
        input.dtype,
    );

    // 1D dispatch: one workgroup per row. Kernel derives batch_idx from
    // group_id.x and the baked-in SEQ_LEN constant.
    let cube_count = CubeCount::new_1d(total_rows as u32);
    let cube_dim = CubeDim::new_1d(workgroup_size);

    let kernel = FusedAdaLnKernel::new(dim as u32, seq_len as u32, workgroup_size, eps);
    let kernel_box: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(kernel, cube_dim));

    let bindings = KernelArguments::new()
        .with_buffer(input.handle.binding())
        .with_buffer(scale.handle.binding())
        .with_buffer(shift.handle.binding())
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

    type WgpuRaw = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;

    fn setup_device() -> <WgpuRaw as Backend>::Device {
        let device = WgpuDevice::DefaultDevice;
        init_setup::<AutoGraphicsApi>(&device, Default::default());
        device
    }

    /// Reference fused AdaLN on CPU.
    fn reference_fused_adaln(
        input: &[f32],
        scale: &[f32],
        shift: &[f32],
        batch_size: usize,
        seq_len: usize,
        dim: usize,
        eps: f64,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; input.len()];
        for b in 0..batch_size {
            for s in 0..seq_len {
                let row = b * seq_len + s;
                let row_offset = row * dim;
                let batch_offset = b * dim;

                let row_data = &input[row_offset..row_offset + dim];
                let mean_sq: f32 = row_data.iter().map(|x| x * x).sum::<f32>() / dim as f32;
                let rms = (mean_sq + eps as f32).sqrt();

                for j in 0..dim {
                    let x_norm = row_data[j] / rms;
                    output[row_offset + j] =
                        x_norm * (1.0 + scale[batch_offset + j]) + shift[batch_offset + j];
                }
            }
        }
        output
    }

    /// Parity test: fused AdaLN kernel vs CPU reference.
    ///
    /// Ignored by default: WGPU device teardown triggers SIGSEGV in the test
    /// harness. Run manually:
    /// `cargo test --all-features fused_adaln -- --ignored --nocapture`
    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually, not in CI"]
    fn fused_adaln_matches_reference() {
        let device = setup_device();

        const DIM: usize = 256;
        const BATCH: usize = 2;
        const SEQ: usize = 8;
        let eps = 1e-6;

        let input_data: Vec<f32> = (0..BATCH * SEQ * DIM)
            .map(|i| ((i as f32 + 1.0) / (BATCH * SEQ * DIM) as f32) * 2.0 - 1.0)
            .collect();
        let scale_data: Vec<f32> = (0..BATCH * DIM)
            .map(|i| 0.1 * (i as f32 / (BATCH * DIM) as f32))
            .collect();
        let shift_data: Vec<f32> = (0..BATCH * DIM)
            .map(|i| -0.05 + 0.1 * (i as f32 / (BATCH * DIM) as f32))
            .collect();

        let expected =
            reference_fused_adaln(&input_data, &scale_data, &shift_data, BATCH, SEQ, DIM, eps);

        let input_tensor = Tensor::<WgpuRaw, 1>::from_floats(input_data.as_slice(), &device)
            .reshape([BATCH * SEQ, DIM]);
        let scale_tensor =
            Tensor::<WgpuRaw, 1>::from_floats(scale_data.as_slice(), &device).reshape([BATCH, DIM]);
        let shift_tensor =
            Tensor::<WgpuRaw, 1>::from_floats(shift_data.as_slice(), &device).reshape([BATCH, DIM]);

        let output_prim = fused_adaln_wgsl(
            input_tensor.into_primitive().tensor(),
            scale_tensor.into_primitive().tensor(),
            shift_tensor.into_primitive().tensor(),
            BATCH,
            SEQ,
            eps,
        );

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

    /// Batch=1 edge case.
    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually, not in CI"]
    fn fused_adaln_batch_one() {
        let device = setup_device();

        const DIM: usize = 64;
        const SEQ: usize = 16;
        let eps = 1e-8;

        let input_data: Vec<f32> = (0..SEQ * DIM).map(|i| (i as f32).sin()).collect();
        let scale_data: Vec<f32> = (0..DIM).map(|i| 0.5 * (i as f32).cos()).collect();
        let shift_data: Vec<f32> = (0..DIM).map(|i| 0.1 * (i as f32).sin()).collect();

        let expected =
            reference_fused_adaln(&input_data, &scale_data, &shift_data, 1, SEQ, DIM, eps);

        let input_tensor =
            Tensor::<WgpuRaw, 1>::from_floats(input_data.as_slice(), &device).reshape([SEQ, DIM]);
        let scale_tensor =
            Tensor::<WgpuRaw, 1>::from_floats(scale_data.as_slice(), &device).reshape([1, DIM]);
        let shift_tensor =
            Tensor::<WgpuRaw, 1>::from_floats(shift_data.as_slice(), &device).reshape([1, DIM]);

        let output_prim = fused_adaln_wgsl(
            input_tensor.into_primitive().tensor(),
            scale_tensor.into_primitive().tensor(),
            shift_tensor.into_primitive().tensor(),
            1,
            SEQ,
            eps,
        );

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

    /// Large dim (1024, the actual model dim) with non-power-of-2 seq_len.
    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually, not in CI"]
    fn fused_adaln_model_dim() {
        let device = setup_device();

        const DIM: usize = 1024;
        const BATCH: usize = 1;
        const SEQ: usize = 750; // typical inference seq_len
        let eps = 1e-6;

        let input_data: Vec<f32> = (0..BATCH * SEQ * DIM)
            .map(|i| (i % 1000) as f32 * 0.001 - 0.5)
            .collect();
        let scale_data: Vec<f32> = (0..BATCH * DIM).map(|i| 0.01 * (i as f32)).collect();
        let shift_data: Vec<f32> = (0..BATCH * DIM).map(|_| 0.0).collect();

        let expected =
            reference_fused_adaln(&input_data, &scale_data, &shift_data, BATCH, SEQ, DIM, eps);

        let input_tensor = Tensor::<WgpuRaw, 1>::from_floats(input_data.as_slice(), &device)
            .reshape([BATCH * SEQ, DIM]);
        let scale_tensor =
            Tensor::<WgpuRaw, 1>::from_floats(scale_data.as_slice(), &device).reshape([BATCH, DIM]);
        let shift_tensor =
            Tensor::<WgpuRaw, 1>::from_floats(shift_data.as_slice(), &device).reshape([BATCH, DIM]);

        let output_prim = fused_adaln_wgsl(
            input_tensor.into_primitive().tensor(),
            scale_tensor.into_primitive().tensor(),
            shift_tensor.into_primitive().tensor(),
            BATCH,
            SEQ,
            eps,
        );

        let output_tensor =
            Tensor::<WgpuRaw, 2>::from_primitive(burn::tensor::TensorPrimitive::Float(output_prim));
        let output_data = output_tensor.into_data().to_vec::<f32>().unwrap();

        let max_diff = output_data
            .iter()
            .zip(expected.iter())
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-3,
            "max diff {max_diff} exceeds tolerance 1e-3"
        );
    }
}
