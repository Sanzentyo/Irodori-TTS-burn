use crate::components::ConvolutionOperation;
use crate::components::global::args::RuntimeArgs;
use crate::components::global::epilogue::NoPostCastEpilogue;
use crate::components::global::read::strategy::sync_bias::SyncBiasLoading;
use crate::routines::{Routine, contiguous_pitched_layout, into_tensor_handle_tma};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::TensorBinding;
use cubecl::server::LaunchError;
use cubek_matmul::definition::AvailableVectorSizes;
use cubek_matmul::multi_level::args::{TensorArgs, TensorMapArgs};
use cubek_matmul::multi_level::components::global::read::AsyncPartialLoadingStrategy;
use cubek_matmul::multi_level::components::global::read::async_partial_cyclic::AsyncPartialCyclicLoading;
use cubek_matmul::multi_level::components::global::read::async_partial_strided::AsyncPartialStridedLoading;
use cubek_matmul::multi_level::components::global::read::async_partial_tma::AsyncPartialTmaLoading;
use cubek_matmul::multi_level::definition::BatchMatmulBlueprint;
use cubek_matmul::multi_level::routines::batch::specialized::{
    SpecializedAlgorithm, SpecializedStrategy,
};
use cubek_matmul::multi_level::tile::ColMajorTilingOrder;
use std::marker::PhantomData;

/// Cmma convolution with a partial async loading strategy.
pub struct SpecializedConv<L: AsyncPartialLoadingStrategy<RuntimeArgs>> {
    _loader: PhantomData<L>,
}

pub type SpecializedAsyncCyclicConv =
    SpecializedConv<AsyncPartialCyclicLoading<ColMajorTilingOrder>>;
pub type SpecializedAsyncStridedConv = SpecializedConv<AsyncPartialStridedLoading>;

pub struct SpecializedTmaConv;

impl<L: AsyncPartialLoadingStrategy<RuntimeArgs>> Routine for SpecializedConv<L> {
    type Blueprint = BatchMatmulBlueprint;
    type Strategy = SpecializedStrategy;
    type MatmulRoutine = SpecializedAlgorithm<L, SyncBiasLoading>;
    type Args = TensorArgs<RuntimeArgs>;
    type PostCastEpilogue = NoPostCastEpilogue;
    const IS_SPECIALIZED: bool = true;

    fn correct_layout(
        client: &Client,
        handle: TensorBinding,
        dtype: ElemType,
        _operation: ConvolutionOperation,
    ) -> Result<TensorBinding, LaunchError> {
        contiguous_pitched_layout(client, handle, dtype)
    }
}

impl Routine for SpecializedTmaConv {
    type Blueprint = BatchMatmulBlueprint;
    type Strategy = SpecializedStrategy;
    type MatmulRoutine = SpecializedAlgorithm<AsyncPartialTmaLoading, SyncBiasLoading>;
    type Args = TensorMapArgs<RuntimeArgs>;
    type PostCastEpilogue = NoPostCastEpilogue;
    const IS_SPECIALIZED: bool = true;

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
