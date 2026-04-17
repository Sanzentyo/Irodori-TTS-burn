//! Diagnostic test: proves `enable subgroups;` causes silent kernel failure
//! on wgpu 29 + DX12 (NVIDIA).
//!
//! The test creates two variants of a trivial copy kernel (output = input × 2):
//! 1. **Without** `enable subgroups;` → produces correct output
//! 2. **With** `enable subgroups;` → produces all zeros (bug)
//!
//! No subgroup operations are used in either variant — the directive alone
//! is sufficient to trigger the failure.
//!
//! # Running
//!
//! ```sh
//! cargo test --lib kernels::subgroup_diagnostic -- --ignored --nocapture
//! ```
//!
//! # Bug Details
//!
//! - **Affected**: wgpu 29.0.1, naga (WGSL → DXIL/HLSL), DX12 backend, NVIDIA
//! - **Symptom**: compute shader compiles and dispatches without error, but
//!   output buffer contains all zeros
//! - **Workaround**: omit `enable subgroups;` directive entirely
//! - **Tracking**: <https://github.com/gfx-rs/wgpu/issues/5555>

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime, into_contiguous,
};
use burn::tensor::Shape;
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

/// Minimal kernel: output[i] = input[i] * 2.0
#[derive(Debug)]
struct SubgroupDiagnosticKernel {
    num_elements: u32,
    workgroup_size: u32,
    enable_subgroups: bool,
}

impl KernelSource for SubgroupDiagnosticKernel {
    fn source(&self) -> SourceTemplate {
        let subgroup_line = if self.enable_subgroups {
            "enable subgroups;"
        } else {
            "// subgroups disabled"
        };
        SourceTemplate::new(include_str!("subgroup_diagnostic.wgsl"))
            .register("subgroup_enable", subgroup_line)
            .register("num_elements", self.num_elements.to_string())
            .register("workgroup_size", self.workgroup_size.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.num_elements,
            self.workgroup_size,
            self.enable_subgroups,
        ))
    }
}

/// Launch the diagnostic kernel and return the output data.
fn run_diagnostic(
    input_data: &[f32],
    enable_subgroups: bool,
    device: &<WgpuRaw as burn::tensor::backend::Backend>::Device,
) -> Vec<f32> {
    use burn::tensor::Tensor;

    let n = input_data.len();
    let workgroup_size = 256u32;

    let input_t = Tensor::<WgpuRaw, 1>::from_floats(input_data, device);
    let input_prim = input_t.into_primitive().tensor();
    let input_prim = into_contiguous(input_prim);

    let client = input_prim.client.clone();
    let dev = input_prim.device.clone();

    let kernel = SubgroupDiagnosticKernel {
        num_elements: n as u32,
        workgroup_size,
        enable_subgroups,
    };

    let num_workgroups = (n as u32).div_ceil(workgroup_size);
    let cube_count = CubeCount::new_1d(num_workgroups);
    let cube_dim = CubeDim::new_1d(workgroup_size);

    let output_handle = client.empty(std::mem::size_of_val(input_data));
    let output = CubeTensor::new_contiguous(
        client.clone(),
        dev,
        Shape::from([n]),
        output_handle,
        input_prim.dtype,
    );

    let kernel_box: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(kernel, cube_dim));

    let bindings = KernelArguments::new()
        .with_buffer(input_prim.handle.binding())
        .with_buffer(output.handle.clone().binding());

    client.launch(kernel_box, cube_count, bindings);

    let output_tensor =
        Tensor::<WgpuRaw, 1>::from_primitive(burn::tensor::TensorPrimitive::Float(output));
    output_tensor.into_data().to_vec::<f32>().unwrap()
}

type WgpuRaw = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup};

    fn setup_device() -> <WgpuRaw as burn::tensor::backend::Backend>::Device {
        let device = WgpuDevice::DefaultDevice;
        init_setup::<AutoGraphicsApi>(&device, Default::default());
        device
    }

    /// Baseline: trivial copy kernel WITHOUT `enable subgroups;` works correctly.
    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn subgroup_diagnostic_without_enable() {
        let device = setup_device();
        let input: Vec<f32> = (0..1024).map(|i| i as f32 * 0.1).collect();
        let output = run_diagnostic(&input, false, &device);

        let expected: Vec<f32> = input.iter().map(|x| x * 2.0).collect();
        let max_diff = output
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        eprintln!(
            "WITHOUT enable subgroups: max_diff={max_diff:.2e}, output[0..4]={:?}",
            &output[0..4]
        );
        assert!(
            max_diff < 1e-6,
            "Without subgroups: kernel should produce correct output, max_diff={max_diff}"
        );
    }

    /// BUG PROOF: trivial copy kernel WITH `enable subgroups;` produces all zeros.
    ///
    /// This test documents the wgpu 29 + DX12 bug where merely adding the
    /// `enable subgroups;` directive (without using ANY subgroup operations)
    /// causes the compute shader to produce incorrect output (all zeros).
    ///
    /// Expected behavior: output should equal input * 2.0 (same as without_enable).
    /// Actual behavior on wgpu 29 + DX12: output is all zeros.
    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn subgroup_diagnostic_with_enable() {
        let device = setup_device();
        let input: Vec<f32> = (0..1024).map(|i| i as f32 * 0.1).collect();
        let output = run_diagnostic(&input, true, &device);

        let expected: Vec<f32> = input.iter().map(|x| x * 2.0).collect();

        let max_diff = output
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let all_zero = output.iter().all(|&x| x == 0.0);
        let first_nonzero = output.iter().position(|&x| x != 0.0);

        eprintln!("WITH enable subgroups:");
        eprintln!("  max_diff={max_diff:.2e}");
        eprintln!("  output[0..4]={:?}", &output[0..4]);
        eprintln!("  expected[0..4]={:?}", &expected[0..4]);
        eprintln!("  all_zero={all_zero}");
        eprintln!("  first_nonzero_idx={first_nonzero:?}");

        if all_zero {
            eprintln!(
                "\n  *** BUG CONFIRMED: `enable subgroups;` causes all-zero output ***\n\
                   *** on this backend (wgpu 29 + DX12/NVIDIA).                   ***\n\
                   *** No subgroup ops are used — the directive alone is enough.   ***"
            );
        }

        // This assertion documents the EXPECTED CORRECT behavior.
        // On backends where subgroups work correctly, this should pass.
        // On wgpu 29 + DX12, this WILL FAIL — proving the bug.
        assert!(
            max_diff < 1e-6,
            "BUG: `enable subgroups;` directive alone causes incorrect output.\n\
             max_diff={max_diff:.2e}, all_zero={all_zero}\n\
             This is a wgpu/naga codegen bug — see docs/wgsl/extensions-status.md"
        );
    }

    /// Run both variants back-to-back for comparison.
    /// Runs the subgroup variant FIRST to avoid false positives from
    /// GPU buffer recycling (the allocator may reuse a buffer still
    /// containing correct data from a previous kernel's output).
    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn subgroup_diagnostic_comparison() {
        let device = setup_device();
        let input: Vec<f32> = (0..1024).map(|i| i as f32 * 0.1).collect();

        // Run WITH subgroup FIRST to avoid buffer-reuse false positive
        let output_with_subgroup = run_diagnostic(&input, true, &device);
        let output_no_subgroup = run_diagnostic(&input, false, &device);

        let expected: Vec<f32> = input.iter().map(|x| x * 2.0).collect();

        let diff_no = output_no_subgroup
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let diff_with = output_with_subgroup
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let all_zero_with = output_with_subgroup.iter().all(|&x| x == 0.0);

        eprintln!("=== Subgroup Diagnostic Comparison ===");
        eprintln!(
            "WITH    `enable subgroups;` (ran first): max_diff={diff_with:.2e} (all_zero={all_zero_with})"
        );
        eprintln!(
            "WITHOUT `enable subgroups;` (ran second): max_diff={diff_no:.2e} (should be ~0)"
        );
        eprintln!(
            "output_with_subgroup[0..4] = {:?}",
            &output_with_subgroup[0..4]
        );
        eprintln!("output_no_subgroup[0..4] = {:?}", &output_no_subgroup[0..4]);

        // Assert the non-subgroup variant works
        assert!(
            diff_no < 1e-6,
            "Baseline (no subgroups) should work: max_diff={diff_no}"
        );

        // Document whether the subgroup variant fails
        if diff_with > 1e-3 {
            eprintln!(
                "\n>>> BUG CONFIRMED: `enable subgroups;` breaks kernel output on this backend <<<"
            );
        } else {
            eprintln!("\n>>> `enable subgroups;` works correctly on this backend <<<");
        }
    }
}
