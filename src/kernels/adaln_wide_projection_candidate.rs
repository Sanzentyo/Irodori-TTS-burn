//! Isolated selectors for the LowRankAdaLN wide-projection experiment.
//!
//! The candidate deliberately computes all three branch projections with a
//! single wide matrix, then retains only the diagonal branch for each flattened
//! `[batch, branch]` row. These launchers are benchmark-only and are not
//! registered in the production kernel module.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::{DType, Shape};
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

const WORKGROUP_SIZE: u32 = 256;
const SELECT_BINDINGS: u32 = 2;
const FINALIZE_BINDINGS: u32 = 4;

#[derive(Debug)]
struct DiagonalSelectKernel {
    rows: u32,
    branches: u32,
    branch_width: u32,
    elements: u32,
}

impl KernelSource for DiagonalSelectKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("adaln_wide_projection_select.wgsl"))
            .register("rows", self.rows.to_string())
            .register("branches", self.branches.to_string())
            .register("branch_width", self.branch_width.to_string())
            .register("elements", self.elements.to_string())
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.rows, self.branches, self.branch_width, self.elements))
    }
}

#[derive(Debug)]
struct DiagonalFinalizeKernel {
    rows: u32,
    branches: u32,
    branch_width: u32,
    elements: u32,
}

impl KernelSource for DiagonalFinalizeKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("adaln_wide_projection_finalize.wgsl"))
            .register("rows", self.rows.to_string())
            .register("branches", self.branches.to_string())
            .register("branch_width", self.branch_width.to_string())
            .register("elements", self.elements.to_string())
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.rows, self.branches, self.branch_width, self.elements))
    }
}

fn checked_u32(value: usize, name: &str) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| panic!("{name}={value} exceeds WGSL u32 indexing"))
}

fn assert_contiguous_rank2(tensor: &CubeTensor<WgpuRuntime>, name: &str) {
    assert_eq!(tensor.dtype, DType::F32, "{name} must be f32");
    assert_eq!(tensor.meta.num_dims(), 2, "{name} must be rank 2");
    assert!(tensor.is_contiguous(), "{name} must be contiguous");
}

fn assert_dispatch_support(
    tensor: &CubeTensor<WgpuRuntime>,
    bindings: u32,
    workgroups: u32,
    name: &str,
) {
    let hardware = &tensor.client.properties().hardware;
    assert!(
        hardware.max_bindings >= bindings,
        "{name} requires {bindings} bindings, device supports {}",
        hardware.max_bindings
    );
    assert!(
        hardware.max_units_per_cube >= WORKGROUP_SIZE,
        "{name} requires {WORKGROUP_SIZE} invocations, device supports {}",
        hardware.max_units_per_cube
    );
    assert!(
        hardware.max_cube_dim.0 >= WORKGROUP_SIZE,
        "{name} requires workgroup x={WORKGROUP_SIZE}, device supports {:?}",
        hardware.max_cube_dim
    );
    assert!(
        hardware.max_cube_count.0 >= workgroups,
        "{name} dispatch x={workgroups} exceeds device limit {:?}",
        hardware.max_cube_count
    );
}

/// Select the diagonal branch from `[rows, branches * branch_width]`.
///
/// Flattened row `r` belongs to branch `r % branches`; the result therefore
/// contains `input[r, (r % branches) * branch_width ..]`. The returned tensor
/// is contiguous `[rows, branch_width]`.
///
/// # Panics
///
/// Panics for a non-contiguous/non-f32/rank-mismatched input, fewer than two
/// branches, an incompatible width, integer overflow, or insufficient device
/// limits.
pub fn adaln_wide_diagonal_select_wgsl(
    input: CubeTensor<WgpuRuntime>,
    branches: usize,
) -> CubeTensor<WgpuRuntime> {
    assert_contiguous_rank2(&input, "wide input");
    assert!(
        branches >= 2,
        "wide selector requires at least two branches"
    );

    let rows = input.meta.shape()[0];
    let wide_width = input.meta.shape()[1];
    assert!(rows > 0, "wide selector requires at least one row");
    assert!(
        wide_width > 0 && wide_width.is_multiple_of(branches),
        "wide input width must be a positive multiple of branches"
    );
    let branch_width = wide_width / branches;
    let elements = rows
        .checked_mul(branch_width)
        .expect("wide selector output element count overflow");
    let rows_u32 = checked_u32(rows, "rows");
    let branches_u32 = checked_u32(branches, "branches");
    let branch_width_u32 = checked_u32(branch_width, "branch width");
    let elements_u32 = checked_u32(elements, "output elements");
    let workgroups = elements_u32.div_ceil(WORKGROUP_SIZE);
    assert_dispatch_support(&input, SELECT_BINDINGS, workgroups, "wide selector");

    let client = input.client.clone();
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([rows, branch_width]),
        client.empty(
            elements
                .checked_mul(size_of::<f32>())
                .expect("wide selector output byte count overflow"),
        ),
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DiagonalSelectKernel {
                rows: rows_u32,
                branches: branches_u32,
                branch_width: branch_width_u32,
                elements: elements_u32,
            },
            CubeDim::new_1d(WORKGROUP_SIZE),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(input.handle.binding())
        .with_buffer(output.handle.clone().binding());
    client.launch(task, CubeCount::new_1d(workgroups), bindings);
    output
}

/// Select the diagonal wide-up branch, then add bias and raw residual.
///
/// `wide_up` is contiguous `[rows, branches * branch_width]`, `bias` is
/// contiguous `[branches, branch_width]`, and `raw` is contiguous
/// `[rows, branch_width]`. Each output is evaluated in the exact order
/// `(selected + bias) + raw` and returned as contiguous
/// `[rows, branch_width]`.
///
/// # Panics
///
/// Panics for dtype/device/shape/contiguity mismatches, fewer than two
/// branches, integer overflow, or insufficient device limits.
pub fn adaln_wide_diagonal_finalize_wgsl(
    wide_up: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    raw: CubeTensor<WgpuRuntime>,
    branches: usize,
) -> CubeTensor<WgpuRuntime> {
    assert_contiguous_rank2(&wide_up, "wide up output");
    assert_contiguous_rank2(&bias, "wide bias");
    assert_contiguous_rank2(&raw, "raw modulation");
    assert!(
        branches >= 2,
        "wide finalizer requires at least two branches"
    );
    wide_up.assert_is_on_same_device(&bias);
    wide_up.assert_is_on_same_device(&raw);

    let rows = wide_up.meta.shape()[0];
    let wide_width = wide_up.meta.shape()[1];
    assert!(rows > 0, "wide finalizer requires at least one row");
    assert!(
        wide_width > 0 && wide_width.is_multiple_of(branches),
        "wide up width must be a positive multiple of branches"
    );
    let branch_width = wide_width / branches;
    assert_eq!(
        bias.meta.shape().dims::<2>(),
        [branches, branch_width],
        "wide bias shape mismatch"
    );
    assert_eq!(
        raw.meta.shape().dims::<2>(),
        [rows, branch_width],
        "raw modulation shape mismatch"
    );

    let elements = rows
        .checked_mul(branch_width)
        .expect("wide finalizer output element count overflow");
    let rows_u32 = checked_u32(rows, "rows");
    let branches_u32 = checked_u32(branches, "branches");
    let branch_width_u32 = checked_u32(branch_width, "branch width");
    let elements_u32 = checked_u32(elements, "output elements");
    let workgroups = elements_u32.div_ceil(WORKGROUP_SIZE);
    assert_dispatch_support(&wide_up, FINALIZE_BINDINGS, workgroups, "wide finalizer");

    let client = wide_up.client.clone();
    let output = CubeTensor::new_contiguous(
        client.clone(),
        wide_up.device.clone(),
        Shape::from([rows, branch_width]),
        client.empty(
            elements
                .checked_mul(size_of::<f32>())
                .expect("wide finalizer output byte count overflow"),
        ),
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DiagonalFinalizeKernel {
                rows: rows_u32,
                branches: branches_u32,
                branch_width: branch_width_u32,
                elements: elements_u32,
            },
            CubeDim::new_1d(WORKGROUP_SIZE),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(wide_up.handle.binding())
        .with_buffer(bias.handle.binding())
        .with_buffer(raw.handle.binding())
        .with_buffer(output.handle.clone().binding());
    client.launch(task, CubeCount::new_1d(workgroups), bindings);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_mapping_selects_the_matching_branch_for_b1_b2() {
        const BRANCHES: usize = 3;
        for batch in [1, 2] {
            for branch_width in [7, 11] {
                let rows = batch * BRANCHES;
                let wide_width = BRANCHES * branch_width;
                for output_index in 0..rows * branch_width {
                    let row = output_index / branch_width;
                    let column = output_index % branch_width;
                    let branch = row % BRANCHES;
                    let source = row * wide_width + branch * branch_width + column;
                    assert_eq!(source / wide_width, row);
                    assert_eq!((source % wide_width) / branch_width, branch);
                    assert_eq!(source % branch_width, column);
                }
            }
        }
    }

    #[test]
    fn final_mapping_keeps_bias_branch_and_left_associative_add_order() {
        const BRANCHES: usize = 3;
        const WIDTH: usize = 13;
        for batch in [1, 2] {
            for output_index in 0..batch * BRANCHES * WIDTH {
                let row = output_index / WIDTH;
                let column = output_index % WIDTH;
                let branch = row % BRANCHES;
                let selected_index = row * BRANCHES * WIDTH + branch * WIDTH + column;
                let bias_index = branch * WIDTH + column;

                let selected = selected_index as f32 * 0.000_976_562_5 - 0.25;
                let bias = bias_index as f32 * 0.001_953_125 - 0.125;
                let raw = output_index as f32 * -0.000_488_281_25 + 0.5;
                let expected = (selected + bias) + raw;
                let biased: f32 = selected + bias;
                let candidate: f32 = biased + raw;
                assert_eq!(candidate.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn shaders_use_uniform_read_write_storage_and_complete_templates() {
        let shaders = [
            (
                "select",
                include_str!("adaln_wide_projection_select.wgsl"),
                SELECT_BINDINGS as usize,
            ),
            (
                "finalize",
                include_str!("adaln_wide_projection_finalize.wgsl"),
                FINALIZE_BINDINGS as usize,
            ),
        ];
        for (name, shader, binding_count) in shaders {
            let bindings = shader
                .lines()
                .map(str::trim)
                .filter(|line| line.starts_with("@group(0)") && line.contains("var<storage"))
                .collect::<Vec<_>>();
            assert_eq!(bindings.len(), binding_count, "{name} binding count");
            assert!(
                bindings
                    .iter()
                    .all(|line| line.contains("var<storage, read_write>")),
                "{name} mixes storage access: {bindings:?}"
            );
            for placeholder in [
                "rows",
                "branches",
                "branch_width",
                "elements",
                "workgroup_size",
            ] {
                assert!(
                    shader.contains(&format!("{{{{ {placeholder} }}}}")),
                    "{name} omits {placeholder}"
                );
            }
        }
    }
}
