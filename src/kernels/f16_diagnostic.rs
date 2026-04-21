//! Diagnostic test: checks whether `enable f16;` causes silent kernel failure
//! on the current wgpu backend/driver.
//!
//! Unlike [`subgroup_diagnostic`], this test uses actual f16 buffer storage
//! (not just the directive alone) to verify end-to-end f16 I/O.
//!
//! The kernel reads `array<f16>` input, multiplies each element by 2.0,
//! and writes `array<f32>` output. If `enable f16;` triggers a naga codegen
//! failure similar to `enable subgroups;`, the output will be all zeros.
//!
//! # Running
//!
//! ```sh
//! cargo test --lib kernels::f16_diagnostic -- --ignored --nocapture
//! ```
//!
//! # Expected outcomes
//!
//! - **PASS** (no bug): output[i] ≈ 2 × input[i] (with f16 quantization error < 0.001)
//! - **FAIL** (naga bug): output is all zeros or garbage

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::Shape;
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;
use half::f16;

/// Minimal `enable f16;` kernel: reads f16 input, writes f32 output.
#[derive(Debug)]
struct F16DiagnosticKernel {
    num_elements: u32,
    workgroup_size: u32,
}

impl KernelSource for F16DiagnosticKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("f16_diagnostic.wgsl"))
            .register("num_elements", self.num_elements.to_string())
            .register("workgroup_size", self.workgroup_size.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.num_elements, self.workgroup_size))
    }
}

type WgpuRaw = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;

/// Launch the f16 diagnostic kernel.
///
/// Creates a f16 input buffer from `input_data`, runs the kernel,
/// and returns the f32 output (expected: 2 × input).
fn run_f16_diagnostic(
    input_data: &[f32],
    device: &<WgpuRaw as burn::tensor::backend::Backend>::Device,
) -> Vec<f32> {
    use burn::tensor::Tensor;

    let n = input_data.len();
    let workgroup_size = 64u32;

    // Bootstrap client from a dummy f32 tensor.
    let dummy = Tensor::<WgpuRaw, 1>::from_floats(input_data, device);
    let dummy_prim = dummy.into_primitive().tensor();
    let client = dummy_prim.client.clone();
    let dev = dummy_prim.device.clone();
    drop(dummy_prim);

    // Create f16 input buffer from host data.
    let input_f16: Vec<f16> = input_data.iter().map(|&x| f16::from_f32(x)).collect();
    let input_bytes: &[u8] = bytemuck::cast_slice(&input_f16);
    let input_handle = client.create_from_slice(input_bytes);
    let input_cube = CubeTensor::new_contiguous(
        client.clone(),
        dev.clone(),
        Shape::from([n]),
        input_handle,
        burn::tensor::DType::F16,
    );

    // Create f32 output buffer.
    let output_handle = client.empty(std::mem::size_of_val(input_data));
    let output_cube = CubeTensor::new_contiguous(
        client.clone(),
        dev,
        Shape::from([n]),
        output_handle,
        burn::tensor::DType::F32,
    );

    let kernel = F16DiagnosticKernel {
        num_elements: n as u32,
        workgroup_size,
    };

    let num_workgroups = (n as u32).div_ceil(workgroup_size);
    let cube_count = CubeCount::new_1d(num_workgroups);
    let cube_dim = CubeDim::new_1d(workgroup_size);

    let kernel_box: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(kernel, cube_dim));

    let bindings = KernelArguments::new()
        .with_buffer(input_cube.handle.binding())
        .with_buffer(output_cube.handle.clone().binding());

    client.launch(kernel_box, cube_count, bindings);

    // Read back f32 output via burn tensor.
    let output_tensor = burn::tensor::Tensor::<WgpuRaw, 1>::from_primitive(
        burn::tensor::TensorPrimitive::Float(output_cube),
    );
    output_tensor.into_data().to_vec::<f32>().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup};

    fn setup_device() -> <WgpuRaw as burn::tensor::backend::Backend>::Device {
        let device = WgpuDevice::DefaultDevice;
        init_setup::<AutoGraphicsApi>(&device, Default::default());
        device
    }

    /// Test that `enable f16;` with f16 buffer storage does NOT cause all-zero output.
    ///
    /// Expected: output[i] ≈ 2 × input[i] (f16 quantization error < 0.01).
    /// If `enable f16;` has the same naga bug as `enable subgroups;`, this fails.
    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn f16_diagnostic_enable_f16_works() {
        let device = setup_device();
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected: Vec<f32> = input.iter().map(|&x| x * 2.0).collect();

        let output = run_f16_diagnostic(&input, &device);

        assert_eq!(output.len(), input.len(), "output length mismatch");

        let all_zero = output.iter().all(|&x| x == 0.0);
        let max_diff = output
            .iter()
            .zip(expected.iter())
            .map(|(&got, &want)| (got - want).abs())
            .fold(0.0f32, f32::max);

        eprintln!("f16 diagnostic: all_zero={all_zero}, max_diff={max_diff:.2e}");
        eprintln!("  input:    {input:?}");
        eprintln!("  expected: {expected:?}");
        eprintln!("  got:      {output:?}");

        if all_zero {
            panic!(
                ">>> BUG CONFIRMED: `enable f16;` causes all-zero output on this backend <<<\n\
                 Same naga issue as `enable subgroups;`. Defer f16 kernel work."
            );
        }

        assert!(
            max_diff < 0.01,
            "max_diff={max_diff:.2e} exceeds f16 quantization tolerance (0.01)"
        );

        eprintln!(">>> PASS: `enable f16;` works correctly on this backend <<<");
        eprintln!(">>> f16 storage optimization is viable! <<<");
    }

    /// Larger test with representative DiT-scale sizes to rule out edge cases.
    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn f16_diagnostic_large() {
        let device = setup_device();
        // 256 elements — covers multiple workgroups
        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();

        let output = run_f16_diagnostic(&input, &device);

        let all_zero = output.iter().all(|&x| x == 0.0);
        let max_diff = output
            .iter()
            .zip(input.iter())
            .map(|(&got, &want)| (got - want * 2.0).abs())
            .fold(0.0f32, f32::max);

        eprintln!("f16 diagnostic large: all_zero={all_zero}, max_diff={max_diff:.2e}");
        assert!(!all_zero, "all-zero output: `enable f16;` has naga bug");
        assert!(max_diff < 0.01, "max_diff={max_diff:.2e} exceeds tolerance");
    }
}
