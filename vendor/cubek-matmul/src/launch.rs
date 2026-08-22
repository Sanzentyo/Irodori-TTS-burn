use cubecl::{
    Runtime,
    client::ComputeClient,
    prelude::TensorBinding,
    zspace::{Shape, Strides},
};
use cubek_std::{InputBinding, MatrixLayout};

use crate::{
    args::TensorArgs,
    components::global::PairwiseAccumulatorGlobalEpilogue,
    definition::{AvailableVectorSizes, MatmulElems, MatmulProblem, MatmulSetupError},
    routines::{
        BlueprintStrategy,
        batch::simple_unit::SimpleUnitPairwiseCompressedAlgorithm,
    },
    strategy::{Strategy, launch_kernel_concrete},
};

#[allow(clippy::result_large_err)]
/// Launches a matrix multiplication kernel..
///
/// # Notes
///
/// The matmul elements may get changed during selection for improved performance when
/// the hardware supports it.
/// Only the inner element types may change such as the stage or register element types.
pub fn launch_ref<R: Runtime>(
    strategy: &Strategy,
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    strategy.launch_ref(client, lhs, rhs, out, dtypes)
}

/// Launch a dense matmul whose adjacent output columns are reduced into a
/// physically half-width output by `E`.
///
/// The RHS must already use the interleaved pair layout
/// `[a0, b0, a1, b1, ...]`; this function never repacks either operand. The
/// physical output binding is `[M, N / 2]`. Internally its logical view remains
/// `[M, N]`, with a compressed row stride, so CubeK keeps the original matmul
/// tiling while the pairwise writer stores only valid compressed coordinates.
#[allow(clippy::result_large_err)]
pub fn launch_pairwise_compressed_ref<R: Runtime, E: PairwiseAccumulatorGlobalEpilogue<()>>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    strategy: &BlueprintStrategy<(), SimpleUnitPairwiseCompressedAlgorithm<E>>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    type Algorithm<E> = SimpleUnitPairwiseCompressedAlgorithm<E>;

    if lhs.scheme().is_some() || rhs.scheme().is_some() {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "pairwise compressed matmul does not support quantized operands".to_owned(),
        )));
    }
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    if lhs_shape.len() != 2 || rhs_shape.len() != 2 || out.shape.len() != 2 {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "pairwise compressed matmul currently requires rank-2 operands".to_owned(),
        )));
    }
    let m = lhs_shape[0];
    let k = lhs_shape[1];
    let n = rhs_shape[1];
    let compressed_n = n / 2;
    if n == 0
        || !n.is_multiple_of(2)
        || rhs_shape[0] != k
        || out.shape.as_slice() != [m, compressed_n]
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "invalid pairwise compressed shapes lhs={:?}, rhs={:?}, out={:?}",
            lhs_shape, rhs_shape, out.shape
        ))));
    }
    let lhs_layout = MatrixLayout::from_shape_and_strides(
        lhs_shape,
        &lhs.data().strides,
        lhs.scheme(),
    )?;
    let rhs_layout = MatrixLayout::from_shape_and_strides(
        rhs_shape,
        &rhs.data().strides,
        rhs.scheme(),
    )?;
    if lhs_layout != MatrixLayout::RowMajor
        || rhs_layout != MatrixLayout::ColMajor
        || &out.strides[..] != [compressed_n, 1]
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "pairwise compressed matmul requires row-major LHS/output and column-major RHS"
                .to_owned(),
        )));
    }

    let address_type = lhs
        .required_address_type()
        .max(rhs.required_address_type())
        .max(out.required_address_type(dtypes.acc_global.size()));
    let logical_out_shape: Shape = [m, n].into();
    // Selection and tiling describe the logical MxN matmul. The writer alone
    // owns the compressed physical stride installed on the binding below.
    let problem_out_strides: Strides = [n, 1].into();
    let problem = MatmulProblem::from_shapes_and_strides(
        lhs_shape.into(),
        rhs_shape.into(),
        logical_out_shape.clone(),
        lhs.data().strides.clone(),
        rhs.data().strides.clone(),
        problem_out_strides,
        dtypes.as_global_elems(),
        address_type,
        lhs.scheme(),
        rhs.scheme(),
    )?;
    let vector_sizes = AvailableVectorSizes::from_type_sizes(
        client,
        lhs.data_elem_size(),
        rhs.data_elem_size(),
        dtypes.acc_global.size(),
    )
    .filter_lhs_with_tensor(&problem.lhs_strides, &problem.lhs_shape, problem.lhs_layout)
    .filter_rhs_with_tensor(&problem.rhs_strides, &problem.rhs_shape, problem.rhs_layout)
    .filter_out(|size| *size == 1)
    .pick_max()?;

    let mut logical_out = out;
    logical_out.shape = logical_out_shape;
    logical_out.strides = [compressed_n, 1].into();
    launch_kernel_concrete::<TensorArgs, R, Algorithm<E>>(
        client,
        lhs,
        rhs,
        logical_out,
        problem,
        vector_sizes,
        strategy,
        dtypes,
    )
}
