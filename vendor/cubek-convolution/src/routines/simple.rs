use cubecl::{
    server::LaunchError,
    {Runtime, client::ComputeClient, ir::StorageType, prelude::TensorBinding},
};
use cubek_matmul::components::global::read::FullLoadingStrategy;
use cubek_matmul::components::global::read::sync_full_cyclic::SyncFullCyclicLoading;
use cubek_matmul::{
    args::{TensorArgs, TensorMapArgs},
    definition::{AvailableVectorSizes, BatchMatmulBlueprint},
};
use cubek_matmul::{
    components::global::read::{
        async_full_tma::AsyncFullTmaLoading, sync_full_strided::SyncFullStridedLoading,
        sync_full_tilewise::SyncFullTilewiseLoading,
    },
    components::global::{
        AccumulatorGlobalStoreTransform, AccumulatorTransformPlaneWriterFamily,
        PlaneWriterFamily, PostCastEpiloguePlaneWriterFamily,
    },
    routines::batch::simple::{SimpleAlgorithm, SimpleArgs},
};
use cubek_std::tile::{ColMajorTilingOrder, RowMajorTilingOrder};
use std::marker::PhantomData;

use crate::{
    components::{
        ConvolutionOperation,
        global::{
            args::RuntimeArgs,
            epilogue::{NoPostCastEpilogue, PostCastEpilogueSpec},
            read::strategy::{
                async_full_cyclic::AsyncFullCyclicLoading,
                async_full_strided::AsyncFullStridedLoading, sync_bias::SyncBiasLoading,
                sync_k7_halo::SyncK7HaloLoading,
            },
        },
    },
    routines::{Routine, contiguous_pitched_layout, into_tensor_handle_tma},
};

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

/// A convolution routine whose accumulator-domain store transform can write
/// auxiliary outputs while producing the primary convolution output.
pub struct SimpleAccumulatorTransformConv<
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
pub type SimpleSyncCyclicAccumulatorTransformConv<E> = SimpleAccumulatorTransformConv<
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

    fn correct_layout<R: Runtime>(
        client: &ComputeClient<R>,
        handle: TensorBinding<R>,
        dtype: StorageType,
        _operation: ConvolutionOperation,
    ) -> Result<TensorBinding<R>, LaunchError> {
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
        cubek_matmul::components::global::GlobalWriterFamily<RuntimeArgs>,
{
    type Blueprint = BatchMatmulBlueprint;
    type Strategy = SimpleArgs;
    type MatmulRoutine =
        SimpleAlgorithm<LL, LR, SyncBiasLoading, AccumulatorTransformPlaneWriterFamily<E>>;
    type Args = TensorArgs<RuntimeArgs>;
    type PostCastEpilogue = E;

    fn correct_layout<R: Runtime>(
        client: &ComputeClient<R>,
        handle: TensorBinding<R>,
        dtype: StorageType,
        _operation: ConvolutionOperation,
    ) -> Result<TensorBinding<R>, LaunchError> {
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
        cubek_matmul::components::global::GlobalWriterFamily<RuntimeArgs>,
{
    type Blueprint = BatchMatmulBlueprint;
    type Strategy = SimpleArgs;
    type MatmulRoutine =
        SimpleAlgorithm<LL, LR, SyncBiasLoading, PostCastEpiloguePlaneWriterFamily<E>>;
    type Args = TensorArgs<RuntimeArgs>;
    type PostCastEpilogue = E;

    fn correct_layout<R: Runtime>(
        client: &ComputeClient<R>,
        handle: TensorBinding<R>,
        dtype: StorageType,
        _operation: ConvolutionOperation,
    ) -> Result<TensorBinding<R>, LaunchError> {
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
        cubek_matmul::components::global::GlobalWriterFamily<RuntimeArgs>,
{
    type Blueprint = BatchMatmulBlueprint;
    type Strategy = SimpleArgs;
    type MatmulRoutine =
        SimpleAlgorithm<LL, LR, SyncBiasLoading, PostCastEpiloguePlaneWriterFamily<E>>;
    type Args = TensorArgs<RuntimeArgs>;
    type PostCastEpilogue = E;

    fn correct_layout<R: Runtime>(
        _client: &ComputeClient<R>,
        handle: TensorBinding<R>,
        _dtype: StorageType,
        _operation: ConvolutionOperation,
    ) -> Result<TensorBinding<R>, LaunchError> {
        Ok(handle)
    }
}

impl Routine for SimpleAsyncTmaConv {
    type Blueprint = BatchMatmulBlueprint;
    type Strategy = SimpleArgs;
    type MatmulRoutine = SimpleAlgorithm<AsyncFullTmaLoading, AsyncFullTmaLoading, SyncBiasLoading>;
    type Args = TensorMapArgs<RuntimeArgs>;
    type PostCastEpilogue = NoPostCastEpilogue;

    fn correct_layout<R: Runtime>(
        client: &ComputeClient<R>,
        handle: TensorBinding<R>,
        dtype: StorageType,
        operation: ConvolutionOperation,
    ) -> Result<TensorBinding<R>, LaunchError> {
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
