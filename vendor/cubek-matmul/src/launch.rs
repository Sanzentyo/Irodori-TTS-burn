use cubecl::{
    Runtime,
    client::ComputeClient,
    prelude::TensorBinding,
    zspace::{Shape, Strides},
};
use cubek_std::{InputBinding, MatrixLayout};

use crate::{
    args::{
        ConcreteInputsFactory, ConcreteOutputFactory, ConfigRuntimeArg, InputArg, OutputArg,
        RuntimeConfig, TensorArgs,
    },
    components::global::{
        AccumulatorGlobalScatter, AccumulatorGlobalStoreTransform,
        PairwiseAccumulatorGlobalEpilogue,
    },
    definition::{AvailableVectorSizes, MatmulElems, MatmulProblem, MatmulSetupError},
    routines::{
        BatchMatmulRoutine, BlueprintStrategy,
        batch::{
            double_unit::DoubleUnitPairwiseCompressedAlgorithm,
            simple::{
                SimpleCyclicAccumulatorTransformAlgorithm,
                SimpleCyclicPairwiseCompressedAlgorithm,
            },
            simple_unit::{
                SimpleUnitAccumulatorScatterAlgorithm, SimpleUnitAccumulatorTransformAlgorithm,
                SimpleUnitPairwiseCompressedAlgorithm,
            },
        },
    },
    strategy::{Strategy, launch_kernel_concrete, launch_kernel_concrete_configured},
};

/// Launch a plane-tiled dense matmul whose typed accumulator-domain transform
/// owns the final value written to the ordinary output matrix.
///
/// `runtime_address_type` must include every auxiliary binding reachable from
/// `runtime_config`; this keeps writer-owned views in the same address-width
/// decision as the primary operands and output.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_accumulator_transform_plane_ref<
    R: Runtime,
    RC: RuntimeConfig,
    T: AccumulatorGlobalStoreTransform<RC>,
>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    runtime_config: ConfigRuntimeArg<TensorArgs<RC>, R>,
    runtime_address_type: cubecl::ir::AddressType,
    strategy: &BlueprintStrategy<RC, SimpleCyclicAccumulatorTransformAlgorithm<T>>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError>
where
    InputArg<TensorArgs<RC>>:
        ConcreteInputsFactory<SimpleCyclicAccumulatorTransformAlgorithm<T>, RC>,
    OutputArg<TensorArgs<RC>>:
        ConcreteOutputFactory<SimpleCyclicAccumulatorTransformAlgorithm<T>, RC>,
{
    if lhs.scheme().is_some() || rhs.scheme().is_some() {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "accumulator transform matmul does not support quantized operands".to_owned(),
        )));
    }
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    if lhs_shape.len() != 2 || rhs_shape.len() != 2 || out.shape.len() != 2 {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "accumulator transform matmul requires rank-2 operands/output".to_owned(),
        )));
    }
    let m = lhs_shape[0];
    let k = lhs_shape[1];
    let n = rhs_shape[1];
    if m == 0
        || k == 0
        || n == 0
        || rhs_shape[0] != k
        || out.shape.as_slice() != [m, n]
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "invalid accumulator transform shapes lhs={lhs_shape:?}, rhs={rhs_shape:?}, out={:?}",
            out.shape
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
        || !matches!(rhs_layout, MatrixLayout::RowMajor | MatrixLayout::ColMajor)
        || &out.strides[..] != [n, 1]
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "accumulator transform requires row-major LHS/output and a dense RHS".to_owned(),
        )));
    }
    let address_type = lhs
        .required_address_type()
        .max(rhs.required_address_type())
        .max(out.required_address_type(dtypes.acc_global.size()))
        .max(runtime_address_type);
    let problem = MatmulProblem::from_shapes_and_strides(
        lhs_shape.into(),
        rhs_shape.into(),
        out.shape.clone(),
        lhs.data().strides.clone(),
        rhs.data().strides.clone(),
        out.strides.clone(),
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
    .filter_out_with_tensor(&problem.out_strides, &problem.out_shape)
    .pick_max()?;
    launch_kernel_concrete_configured::<
        RC,
        TensorArgs<RC>,
        R,
        SimpleCyclicAccumulatorTransformAlgorithm<T>,
    >(
        client,
        lhs,
        rhs,
        out,
        runtime_config,
        problem,
        vector_sizes,
        strategy,
        dtypes,
    )
}

/// Launch a strict-F32 unit-tiled dense matmul with the same typed
/// accumulator-domain transform contract as the plane launcher.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_accumulator_transform_unit_ref<
    R: Runtime,
    RC: RuntimeConfig,
    T: AccumulatorGlobalStoreTransform<RC>,
>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    runtime_config: ConfigRuntimeArg<TensorArgs<RC>, R>,
    runtime_address_type: cubecl::ir::AddressType,
    strategy: &BlueprintStrategy<RC, SimpleUnitAccumulatorTransformAlgorithm<T>>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError>
where
    InputArg<TensorArgs<RC>>:
        ConcreteInputsFactory<SimpleUnitAccumulatorTransformAlgorithm<T>, RC>,
    OutputArg<TensorArgs<RC>>:
        ConcreteOutputFactory<SimpleUnitAccumulatorTransformAlgorithm<T>, RC>,
{
    if lhs.scheme().is_some() || rhs.scheme().is_some() {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "accumulator transform matmul does not support quantized operands".to_owned(),
        )));
    }
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    if lhs_shape.len() != 2 || rhs_shape.len() != 2 || out.shape.len() != 2 {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "accumulator transform matmul requires rank-2 operands/output".to_owned(),
        )));
    }
    let m = lhs_shape[0];
    let k = lhs_shape[1];
    let n = rhs_shape[1];
    if m == 0
        || k == 0
        || n == 0
        || rhs_shape[0] != k
        || out.shape.as_slice() != [m, n]
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "invalid accumulator transform shapes lhs={lhs_shape:?}, rhs={rhs_shape:?}, out={:?}",
            out.shape
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
        || &out.strides[..] != [n, 1]
    {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "unit accumulator transform requires row-major LHS/output and column-major RHS"
                .to_owned(),
        )));
    }
    let address_type = lhs
        .required_address_type()
        .max(rhs.required_address_type())
        .max(out.required_address_type(dtypes.acc_global.size()))
        .max(runtime_address_type);
    let problem = MatmulProblem::from_shapes_and_strides(
        lhs_shape.into(),
        rhs_shape.into(),
        out.shape.clone(),
        lhs.data().strides.clone(),
        rhs.data().strides.clone(),
        out.strides.clone(),
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
    launch_kernel_concrete_configured::<
        RC,
        TensorArgs<RC>,
        R,
        SimpleUnitAccumulatorTransformAlgorithm<T>,
    >(
        client,
        lhs,
        rhs,
        out,
        runtime_config,
        problem,
        vector_sizes,
        strategy,
        dtypes,
    )
}

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
    launch_pairwise_compressed_with::<R, SimpleUnitPairwiseCompressedAlgorithm<E>>(
        client, lhs, rhs, out, strategy, dtypes,
    )
}

/// Launch a pairwise-compressed dense matmul through CubeK's plane-tiled
/// cyclic routine.
///
/// This has the same shape and storage contract as
/// [`launch_pairwise_compressed_ref`], but it uses cooperative plane tiling
/// rather than the conservative unit routine. Keeping the writer generic lets
/// applications reuse the same compressed-output epilogue with the regular
/// high-throughput matmul architecture.
#[allow(clippy::result_large_err)]
pub fn launch_pairwise_compressed_plane_ref<
    R: Runtime,
    E: PairwiseAccumulatorGlobalEpilogue<()>,
>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    strategy: &BlueprintStrategy<(), SimpleCyclicPairwiseCompressedAlgorithm<E>>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    launch_pairwise_compressed_with::<R, SimpleCyclicPairwiseCompressedAlgorithm<E>>(
        client, lhs, rhs, out, strategy, dtypes,
    )
}

/// Launch a pairwise-compressed dense matmul through CubeK's strict-f32
/// double-buffered unit routine.
///
/// Unlike cooperative-matrix plane algorithms this route never changes the
/// input precision (including TF32), while retaining double-buffered global
/// loads and the one-dispatch compressed output contract.
#[allow(clippy::result_large_err)]
pub fn launch_pairwise_compressed_double_unit_ref<
    R: Runtime,
    E: PairwiseAccumulatorGlobalEpilogue<()>,
>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    strategy: &BlueprintStrategy<(), DoubleUnitPairwiseCompressedAlgorithm<E>>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    launch_pairwise_compressed_with::<R, DoubleUnitPairwiseCompressedAlgorithm<E>>(
        client, lhs, rhs, out, strategy, dtypes,
    )
}

/// Launch a dense matmul whose typed accumulator scatter owns every physical
/// destination. The conventional output binding is a one-scalar placeholder;
/// it is never read or written by the scatter family.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub fn launch_accumulator_scatter_ref<
    R: Runtime,
    RC: RuntimeConfig,
    T: AccumulatorGlobalScatter<RC>,
>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    placeholder: TensorBinding<R>,
    runtime_config: ConfigRuntimeArg<TensorArgs<RC>, R>,
    strategy: &BlueprintStrategy<RC, SimpleUnitAccumulatorScatterAlgorithm<T>>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError>
where
    InputArg<TensorArgs<RC>>:
        ConcreteInputsFactory<SimpleUnitAccumulatorScatterAlgorithm<T>, RC>,
    OutputArg<TensorArgs<RC>>:
        ConcreteOutputFactory<SimpleUnitAccumulatorScatterAlgorithm<T>, RC>,
{
    if lhs.scheme().is_some() || rhs.scheme().is_some() {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "accumulator scatter matmul does not support quantized operands".to_owned(),
        )));
    }
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "accumulator scatter matmul requires rank-2 operands".to_owned(),
        )));
    }
    let m = lhs_shape[0];
    let k = lhs_shape[1];
    let n = rhs_shape[1];
    if m == 0 || k == 0 || n == 0 || rhs_shape[0] != k {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "invalid accumulator scatter shapes lhs={lhs_shape:?}, rhs={rhs_shape:?}"
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
    if lhs_layout != MatrixLayout::RowMajor || rhs_layout != MatrixLayout::ColMajor {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "accumulator scatter requires row-major LHS and column-major RHS".to_owned(),
        )));
    }
    let address_type = lhs
        .required_address_type()
        .max(rhs.required_address_type());
    let output_shape: Shape = [m, n].into();
    let output_strides: Strides = [n, 1].into();
    let problem = MatmulProblem::from_shapes_and_strides(
        lhs_shape.into(),
        rhs_shape.into(),
        output_shape.clone(),
        lhs.data().strides.clone(),
        rhs.data().strides.clone(),
        output_strides.clone(),
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
    let mut logical_placeholder = placeholder;
    logical_placeholder.shape = output_shape;
    logical_placeholder.strides = output_strides;
    launch_kernel_concrete_configured::<
        RC,
        TensorArgs<RC>,
        R,
        SimpleUnitAccumulatorScatterAlgorithm<T>,
    >(
        client,
        lhs,
        rhs,
        logical_placeholder,
        runtime_config,
        problem,
        vector_sizes,
        strategy,
        dtypes,
    )
}

#[allow(clippy::result_large_err)]
fn launch_pairwise_compressed_with<R: Runtime, A: BatchMatmulRoutine<()>>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    strategy: &BlueprintStrategy<(), A>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError>
where
    InputArg<TensorArgs>: ConcreteInputsFactory<A>,
    OutputArg<TensorArgs>: ConcreteOutputFactory<A>,
{

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
    launch_kernel_concrete::<TensorArgs, R, A>(
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
