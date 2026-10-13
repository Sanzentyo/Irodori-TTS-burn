use crate::components::ConvolutionOperation;
use crate::components::global::args::RuntimeArgs;
use crate::components::global::epilogue::{NoPostCastEpilogue, PostCastEpilogueSpec};
use crate::components::global::read::strategy::async_full_cyclic::AsyncFullCyclicLoading;
use crate::components::global::read::strategy::async_full_strided::AsyncFullStridedLoading;
use crate::components::global::read::strategy::sync_bias::SyncBiasLoading;
use crate::components::global::read::strategy::sync_k7_fanout::SyncK7FanoutLoading;
use crate::components::global::read::strategy::sync_k7_halo::SyncK7HaloLoading;
use crate::routines::{Routine, contiguous_pitched_layout, into_tensor_handle_tma};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::TensorBinding;
use cubecl::server::LaunchError;
use cubek_matmul::definition::AvailableVectorSizes;
use cubek_matmul::multi_level::args::{TensorArgs, TensorMapArgs};
use cubek_matmul::multi_level::components::global::read::FullLoadingStrategy;
use cubek_matmul::multi_level::components::global::read::async_full_tma::AsyncFullTmaLoading;
use cubek_matmul::multi_level::components::global::read::sync_full_cyclic::SyncFullCyclicLoading;
use cubek_matmul::multi_level::components::global::read::sync_full_strided::SyncFullStridedLoading;
use cubek_matmul::multi_level::components::global::read::sync_full_tilewise::SyncFullTilewiseLoading;
use cubek_matmul::multi_level::components::global::single_stage::simple::DirectStagePartition;
use cubek_matmul::multi_level::components::global::{
    AccumulatorGlobalStoreTransform, AccumulatorTransformPlaneWriterFamily, PlaneWriterFamily,
    PostCastEpiloguePlaneWriterFamily,
};
use cubek_matmul::multi_level::definition::BatchMatmulBlueprint;
use cubek_matmul::multi_level::routines::batch::simple::{
    SimpleAlgorithm, SimpleArgs, TunableSimpleAlgorithm, TunableSimpleArgs,
};
use cubek_matmul::multi_level::tile::{ColMajorTilingOrder, RowMajorTilingOrder};
use std::marker::PhantomData;

/// Cmma convolution
pub struct SimpleConv<LL: FullLoadingStrategy<RuntimeArgs>, LR: FullLoadingStrategy<RuntimeArgs>> {
    _loader: PhantomData<(LL, LR)>,
}

/// A convolution routine whose post-cast epilogue and launch arguments are
/// part of the type, so it cannot be launched through the standard API.
pub struct SimplePostCastEpilogueConv<
    LL: FullLoadingStrategy<RuntimeArgs>,
    LR: FullLoadingStrategy<RuntimeArgs>,
    E: PostCastEpilogueSpec,
> {
    _loader: PhantomData<(LL, LR, E)>,
}

/// Post-cast convolution whose LHS reader exposes a concrete non-affine stage
/// directly to the partition MMA instead of erasing it into `StageTile`.
pub struct SimpleDirectStagePostCastEpilogueConv<
    LL: FullLoadingStrategy<RuntimeArgs>,
    LR: FullLoadingStrategy<RuntimeArgs>,
    E: PostCastEpilogueSpec,
> {
    _loader: PhantomData<(LL, LR, E)>,
}

/// Post-cast epilogue convolution with explicit generic matmul selector knobs
/// for application-level autotuning.
pub struct TunableSimplePostCastEpilogueConv<
    LL: FullLoadingStrategy<RuntimeArgs>,
    LR: FullLoadingStrategy<RuntimeArgs>,
    E: PostCastEpilogueSpec,
> {
    _loader: PhantomData<(LL, LR, E)>,
}

/// A convolution routine whose accumulator-domain store transform can write
/// auxiliary outputs while producing the primary convolution output.
pub struct SimpleAccumulatorTransformConv<
    LL: FullLoadingStrategy<RuntimeArgs>,
    LR: FullLoadingStrategy<RuntimeArgs>,
    E: PostCastEpilogueSpec + AccumulatorGlobalStoreTransform<RuntimeArgs>,
> {
    _loader: PhantomData<(LL, LR, E)>,
}

/// Accumulator-domain store transform with explicit, cache-key-visible matmul
/// selector knobs for application-level autotuning.
pub struct TunableSimpleAccumulatorTransformConv<
    LL: FullLoadingStrategy<RuntimeArgs>,
    LR: FullLoadingStrategy<RuntimeArgs>,
    E: PostCastEpilogueSpec + AccumulatorGlobalStoreTransform<RuntimeArgs>,
> {
    _loader: PhantomData<(LL, LR, E)>,
}

/// Diagnostic post-cast routine that preserves caller-provided strides. This
/// permits layout-aware kernels to consume a logical weight view directly.
pub struct SimpleStridedPostCastEpilogueConv<
    LL: FullLoadingStrategy<RuntimeArgs>,
    LR: FullLoadingStrategy<RuntimeArgs>,
    E: PostCastEpilogueSpec,
> {
    _loader: PhantomData<(LL, LR, E)>,
}

pub type SimpleSyncCyclicConv = SimpleConv<
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    SyncFullCyclicLoading<ColMajorTilingOrder>,
>;
pub type SimpleSyncStridedConv = SimpleConv<SyncFullStridedLoading, SyncFullStridedLoading>;
pub type SimpleSyncTilewiseConv = SimpleConv<
    SyncFullTilewiseLoading<RowMajorTilingOrder>,
    SyncFullTilewiseLoading<ColMajorTilingOrder>,
>;
pub type SimpleAsyncCyclicConv = SimpleConv<
    AsyncFullCyclicLoading<RowMajorTilingOrder>,
    AsyncFullCyclicLoading<ColMajorTilingOrder>,
>;
pub type SimpleAsyncStridedConv = SimpleConv<AsyncFullStridedLoading, AsyncFullStridedLoading>;
pub type SimpleSyncCyclicPostCastEpilogueConv<E> = SimplePostCastEpilogueConv<
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    SyncFullCyclicLoading<ColMajorTilingOrder>,
    E,
>;
pub type TunableSimpleSyncCyclicPostCastEpilogueConv<E> = TunableSimplePostCastEpilogueConv<
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    SyncFullCyclicLoading<ColMajorTilingOrder>,
    E,
>;
pub type SimpleSyncCyclicAccumulatorTransformConv<E> = SimpleAccumulatorTransformConv<
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    SyncFullCyclicLoading<ColMajorTilingOrder>,
    E,
>;
pub type TunableSimpleSyncCyclicAccumulatorTransformConv<E> = TunableSimpleAccumulatorTransformConv<
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    SyncFullCyclicLoading<ColMajorTilingOrder>,
    E,
>;
pub type SimpleSyncCyclicStridedPostCastEpilogueConv<E> = SimpleStridedPostCastEpilogueConv<
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    SyncFullCyclicLoading<ColMajorTilingOrder>,
    E,
>;
pub type SimpleSyncK7HaloPostCastEpilogueConv<E> =
    SimplePostCastEpilogueConv<SyncK7HaloLoading, SyncFullCyclicLoading<ColMajorTilingOrder>, E>;
pub type SimpleSyncK7FanoutPostCastEpilogueConv<E> = SimpleDirectStagePostCastEpilogueConv<
    SyncK7FanoutLoading,
    SyncFullCyclicLoading<ColMajorTilingOrder>,
    E,
>;

pub struct SimpleAsyncTmaConv;

impl<
    LL: FullLoadingStrategy<RuntimeArgs>,
    LR: FullLoadingStrategy<RuntimeArgs, SyncStrategy = LL::SyncStrategy>,
> Routine for SimpleConv<LL, LR>
{
    type Blueprint = BatchMatmulBlueprint;
    type Strategy = SimpleArgs;
    type MatmulRoutine = SimpleAlgorithm<LL, LR, SyncBiasLoading, PlaneWriterFamily>;
    type Args = TensorArgs<RuntimeArgs>;
    type PostCastEpilogue = NoPostCastEpilogue;

    fn correct_layout(
        client: &Client,
        handle: TensorBinding,
        dtype: ElemType,
        _operation: ConvolutionOperation,
    ) -> Result<TensorBinding, LaunchError> {
        contiguous_pitched_layout(client, handle, dtype)
    }
}

impl<
    LL: FullLoadingStrategy<RuntimeArgs>,
    LR: FullLoadingStrategy<RuntimeArgs, SyncStrategy = LL::SyncStrategy>,
    E: PostCastEpilogueSpec + AccumulatorGlobalStoreTransform<RuntimeArgs>,
> Routine for TunableSimpleAccumulatorTransformConv<LL, LR, E>
where
    AccumulatorTransformPlaneWriterFamily<E>:
        cubek_matmul::multi_level::components::global::GlobalWriterFamily<RuntimeArgs>,
{
    type Blueprint = BatchMatmulBlueprint;
    type Strategy = TunableSimpleArgs;
    type MatmulRoutine =
        TunableSimpleAlgorithm<LL, LR, SyncBiasLoading, AccumulatorTransformPlaneWriterFamily<E>>;
    type Args = TensorArgs<RuntimeArgs>;
    type PostCastEpilogue = E;

    fn correct_layout(
        client: &Client,
        handle: TensorBinding,
        dtype: ElemType,
        _operation: ConvolutionOperation,
    ) -> Result<TensorBinding, LaunchError> {
        contiguous_pitched_layout(client, handle, dtype)
    }
}

impl<
    LL: FullLoadingStrategy<RuntimeArgs>,
    LR: FullLoadingStrategy<RuntimeArgs, SyncStrategy = LL::SyncStrategy>,
    E: PostCastEpilogueSpec,
> Routine for TunableSimplePostCastEpilogueConv<LL, LR, E>
where
    PostCastEpiloguePlaneWriterFamily<E>:
        cubek_matmul::multi_level::components::global::GlobalWriterFamily<RuntimeArgs>,
{
    type Blueprint = BatchMatmulBlueprint;
    type Strategy = TunableSimpleArgs;
    type MatmulRoutine =
        TunableSimpleAlgorithm<LL, LR, SyncBiasLoading, PostCastEpiloguePlaneWriterFamily<E>>;
    type Args = TensorArgs<RuntimeArgs>;
    type PostCastEpilogue = E;

    fn correct_layout(
        client: &Client,
        handle: TensorBinding,
        dtype: ElemType,
        _operation: ConvolutionOperation,
    ) -> Result<TensorBinding, LaunchError> {
        contiguous_pitched_layout(client, handle, dtype)
    }
}

impl<
    LL: FullLoadingStrategy<RuntimeArgs>,
    LR: FullLoadingStrategy<RuntimeArgs, SyncStrategy = LL::SyncStrategy>,
    E: PostCastEpilogueSpec + AccumulatorGlobalStoreTransform<RuntimeArgs>,
> Routine for SimpleAccumulatorTransformConv<LL, LR, E>
where
    AccumulatorTransformPlaneWriterFamily<E>:
        cubek_matmul::multi_level::components::global::GlobalWriterFamily<RuntimeArgs>,
{
    type Blueprint = BatchMatmulBlueprint;
    type Strategy = SimpleArgs;
    type MatmulRoutine =
        SimpleAlgorithm<LL, LR, SyncBiasLoading, AccumulatorTransformPlaneWriterFamily<E>>;
    type Args = TensorArgs<RuntimeArgs>;
    type PostCastEpilogue = E;

    fn correct_layout(
        client: &Client,
        handle: TensorBinding,
        dtype: ElemType,
        _operation: ConvolutionOperation,
    ) -> Result<TensorBinding, LaunchError> {
        contiguous_pitched_layout(client, handle, dtype)
    }
}

impl<
    LL: FullLoadingStrategy<RuntimeArgs>,
    LR: FullLoadingStrategy<RuntimeArgs, SyncStrategy = LL::SyncStrategy>,
    E: PostCastEpilogueSpec,
> Routine for SimplePostCastEpilogueConv<LL, LR, E>
where
    PostCastEpiloguePlaneWriterFamily<E>:
        cubek_matmul::multi_level::components::global::GlobalWriterFamily<RuntimeArgs>,
{
    type Blueprint = BatchMatmulBlueprint;
    type Strategy = SimpleArgs;
    type MatmulRoutine =
        SimpleAlgorithm<LL, LR, SyncBiasLoading, PostCastEpiloguePlaneWriterFamily<E>>;
    type Args = TensorArgs<RuntimeArgs>;
    type PostCastEpilogue = E;

    fn correct_layout(
        client: &Client,
        handle: TensorBinding,
        dtype: ElemType,
        _operation: ConvolutionOperation,
    ) -> Result<TensorBinding, LaunchError> {
        contiguous_pitched_layout(client, handle, dtype)
    }
}

impl<
    LL: FullLoadingStrategy<RuntimeArgs>,
    LR: FullLoadingStrategy<RuntimeArgs, SyncStrategy = LL::SyncStrategy>,
    E: PostCastEpilogueSpec,
> Routine for SimpleDirectStagePostCastEpilogueConv<LL, LR, E>
where
    PostCastEpiloguePlaneWriterFamily<E>:
        cubek_matmul::multi_level::components::global::GlobalWriterFamily<RuntimeArgs>,
{
    type Blueprint = BatchMatmulBlueprint;
    type Strategy = SimpleArgs;
    type MatmulRoutine = SimpleAlgorithm<
        LL,
        LR,
        SyncBiasLoading,
        PostCastEpiloguePlaneWriterFamily<E>,
        DirectStagePartition,
    >;
    type Args = TensorArgs<RuntimeArgs>;
    type PostCastEpilogue = E;

    fn correct_layout(
        client: &Client,
        handle: TensorBinding,
        dtype: ElemType,
        _operation: ConvolutionOperation,
    ) -> Result<TensorBinding, LaunchError> {
        contiguous_pitched_layout(client, handle, dtype)
    }
}

impl<
    LL: FullLoadingStrategy<RuntimeArgs>,
    LR: FullLoadingStrategy<RuntimeArgs, SyncStrategy = LL::SyncStrategy>,
    E: PostCastEpilogueSpec,
> Routine for SimpleStridedPostCastEpilogueConv<LL, LR, E>
where
    PostCastEpiloguePlaneWriterFamily<E>:
        cubek_matmul::multi_level::components::global::GlobalWriterFamily<RuntimeArgs>,
{
    type Blueprint = BatchMatmulBlueprint;
    type Strategy = SimpleArgs;
    type MatmulRoutine =
        SimpleAlgorithm<LL, LR, SyncBiasLoading, PostCastEpiloguePlaneWriterFamily<E>>;
    type Args = TensorArgs<RuntimeArgs>;
    type PostCastEpilogue = E;

    fn correct_layout(
        _client: &Client,
        handle: TensorBinding,
        _dtype: ElemType,
        _operation: ConvolutionOperation,
    ) -> Result<TensorBinding, LaunchError> {
        Ok(handle)
    }
}

impl Routine for SimpleAsyncTmaConv {
    type Blueprint = BatchMatmulBlueprint;
    type Strategy = SimpleArgs;
    type MatmulRoutine = SimpleAlgorithm<AsyncFullTmaLoading, AsyncFullTmaLoading, SyncBiasLoading>;
    type Args = TensorMapArgs<RuntimeArgs>;
    type PostCastEpilogue = NoPostCastEpilogue;

    fn correct_layout(
        client: &Client,
        handle: TensorBinding,
        dtype: ElemType,
        operation: ConvolutionOperation,
    ) -> Result<TensorBinding, LaunchError> {
        into_tensor_handle_tma(client, handle, dtype, operation)
    }

    fn filter_vector_sizes(vector_sizes: AvailableVectorSizes) -> AvailableVectorSizes {
        AvailableVectorSizes {
            lhs: vec![1],
            rhs: vec![1],
            out: vector_sizes.out,
        }
    }
}
