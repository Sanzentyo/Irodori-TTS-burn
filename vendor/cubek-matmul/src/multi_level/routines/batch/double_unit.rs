use crate::definition::{MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes};
use crate::multi_level::args::{
    ConfigRuntimeArg, InputRuntimeArg, MatmulArgs, OutputRuntimeArg, RuntimeConfig,
};
use crate::multi_level::components::batch::{
    BatchMatmulFamily, PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul,
};
use crate::multi_level::components::global::multi_stage::double_buffering::DoubleBufferingMatmulFamily;
use crate::multi_level::components::global::read::sync_full_cyclic::SyncFullCyclicLoading;
use crate::multi_level::components::global::read::sync_partial_cyclic::SyncPartialCyclicLoading;
use crate::multi_level::components::global::{
    AccumulatorTransformUnitWriterFamily, GlobalWriterFamily, PairwiseCompressedUnitWriterFamily,
    UnitWriterFamily,
};
use crate::multi_level::components::stage::{NumStages, UnitPartitioner};
use crate::multi_level::components::tile::TileMatmulKind;
use crate::multi_level::definition::{BatchMatmulBlueprint, CubeMappingLaunch};
use crate::multi_level::routines::selector::{
    TileSizeSelection, UnitTilingBlueprintOptions, infer_blueprint_unit,
};
use crate::multi_level::tile::RowMajorTilingOrder;
use crate::multi_level::{BatchMatmulRoutine, ExpandInfo, LaunchInfo, batch_validate_blueprint};
use crate::routine::{BlueprintStrategy, DeviceSettings, Routine};
use cubecl::client::Client;
use cubecl::ir::AddressType;
use cubecl::{CubeCount, CubeDim};
use std::fmt::Display;
use std::marker::PhantomData;

/// The batch-matmul family powering [`DoubleUnitAlgorithm`].
type DoubleUnitBatch<RC, GW = UnitWriterFamily> = PartitionedBatchMatmulFamily<
    RC,
    DoubleBufferingMatmulFamily<
        UnitPartitioner,
        RC,
        SyncPartialCyclicLoading<RowMajorTilingOrder>,
        SyncPartialCyclicLoading<RowMajorTilingOrder>,
        SyncFullCyclicLoading<RowMajorTilingOrder>,
        GW,
    >,
    RowMajorGlobalPartitionMatmul,
>;

/// Unit double buffered matmul with cyclic readers
pub struct DoubleUnitAlgorithm<GW = UnitWriterFamily> {
    pub _writer: PhantomData<GW>,
}

/// Strict-f32 double-buffered matmul whose writer combines adjacent
/// accumulator columns into one physical output column.
pub type DoubleUnitPairwiseCompressedAlgorithm<E> =
    DoubleUnitAlgorithm<PairwiseCompressedUnitWriterFamily<E>>;

/// Strict-f32 double-buffered matmul whose typed writer transforms each
/// accumulator before its primary store.
pub type DoubleUnitAccumulatorTransformAlgorithm<T> =
    DoubleUnitAlgorithm<AccumulatorTransformUnitWriterFamily<T>>;

#[derive(Default, Clone, Debug)]
pub struct DoubleUnitSelectionArgs {
    pub tile_size: TileSizeSelection,
}

impl Display for DoubleUnitSelectionArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "_{}", self.tile_size)
    }
}

impl<RC, GW> Routine<RC> for DoubleUnitAlgorithm<GW>
where
    RC: RuntimeConfig,
    GW: GlobalWriterFamily<RC>,
{
    type Strategy = DoubleUnitSelectionArgs;
    type Blueprint = BatchMatmulBlueprint;
}

impl<RC, GW> BatchMatmulRoutine<RC> for DoubleUnitAlgorithm<GW>
where
    RC: RuntimeConfig,
    GW: GlobalWriterFamily<RC>,
{
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    fn launch<MA: MatmulArgs<Config = RC>>(
        client: &Client,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        address_type: AddressType,
        input: InputRuntimeArg<MA>,
        output: OutputRuntimeArg<MA>,
        config: ConfigRuntimeArg<MA>,
        cube_count_input: CubeMappingLaunch,
        blueprint: Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        {
            unsafe {
                <DoubleUnitBatch<RC, GW>>::launch_unchecked::<MA>(
                    client,
                    cube_dim,
                    cube_count,
                    address_type,
                    input,
                    output,
                    config,
                    cube_count_input,
                    blueprint,
                    dtypes,
                    vector_sizes,
                )?
            }
            Ok(())
        }
    }

    #[allow(clippy::result_large_err)]
    fn validate_blueprint(
        client: &Client,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        batch_validate_blueprint::<DoubleUnitBatch<RC, GW>, RC>(
            client,
            blueprint,
            problem,
            dtypes,
            vector_sizes,
        )
    }

    fn num_stages() -> NumStages {
        DoubleUnitBatch::<RC, GW>::num_stages()
    }

    fn expand_blueprint(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);

        if TileMatmulKind::Register.can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(strategy) => infer_blueprint_unit(
                &device_settings.client,
                problem,
                device_settings.plane_dim,
                true,
                &device_settings.vector_sizes,
                UnitTilingBlueprintOptions {
                    tile: strategy.tile_size,
                    ..Default::default()
                },
                &problem.global_dtypes,
            ),
        };
        Ok(ExpandInfo { blueprint, dtypes })
    }

    fn prepare(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<BatchMatmulBlueprint>, MatmulSetupError> {
        let ExpandInfo { blueprint, dtypes } = expand_info;

        <Self as BatchMatmulRoutine<RC>>::validate_blueprint(
            &device_settings.client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.vector_sizes,
        )?;

        let cubedim_resource = DoubleUnitBatch::<RC, GW>::cubedim_resource(
            &blueprint,
            &dtypes,
            &device_settings.vector_sizes,
        )?;

        LaunchInfo::new(
            blueprint,
            dtypes,
            problem,
            cubedim_resource,
            device_settings,
        )
    }

    fn device_settings(client: &Client, vector_sizes: MatmulVectorSizes) -> DeviceSettings {
        let plane_dim = match client.properties().hardware.plane_size_min {
            0 => 32,
            plane_dim => plane_dim,
        };

        DeviceSettings {
            client: client.clone(),
            plane_dim,
            vector_sizes,
            max_cube_count: client.properties().hardware.max_cube_count,
        }
    }
}
