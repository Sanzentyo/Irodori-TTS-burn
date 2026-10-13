use crate::definition::{MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes};
use crate::multi_level::args::{
    ConfigRuntimeArg, InputRuntimeArg, MatmulArgs, OutputRuntimeArg, RuntimeConfig,
};
use crate::multi_level::components::batch::{
    BatchMatmulFamily, PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul,
};
use crate::multi_level::components::global::read::FullLoadingStrategy;
use crate::multi_level::components::global::read::sync_full_cyclic::SyncFullCyclicLoading;
use crate::multi_level::components::global::single_stage::simple::SimpleMatmulFamily;
use crate::multi_level::components::global::{
    AccumulatorScatterUnitWriterFamily, AccumulatorTransformUnitWriterFamily, GlobalWriterFamily,
    PairwiseCompressedUnitWriterFamily, UnitWriterFamily,
};
use crate::multi_level::components::stage::{NumStages, UnitPartitioner};
use crate::multi_level::components::tile::TileMatmulKind;
use crate::multi_level::definition::{BatchMatmulBlueprint, CubeMappingLaunch};
use crate::multi_level::routines::selector::{
    PartitionScaling, StageScaling, TileSizeSelection, UnitTilingBlueprintOptions,
    infer_blueprint_unit,
};
use crate::multi_level::tile::{ColMajorTilingOrder, RowMajorTilingOrder};
use crate::multi_level::{BatchMatmulRoutine, ExpandInfo, LaunchInfo, batch_validate_blueprint};
use crate::routine::{BlueprintStrategy, DeviceSettings, Routine};
use cubecl::client::Client;
use cubecl::ir::AddressType;
use cubecl::{CubeCount, CubeDim};
use std::fmt::Display;
use std::marker::PhantomData;

/// The batch-matmul family powering [`SimpleUnitAlgorithm`].
type SimpleUnitBatch<RC, LL, RL, AL, GW> = PartitionedBatchMatmulFamily<
    RC,
    SimpleMatmulFamily<UnitPartitioner, RC, LL, RL, AL, GW>,
    RowMajorGlobalPartitionMatmul,
>;

/// Unit single stage matmul with configurable readers (default to cyclic)
pub struct SimpleUnitAlgorithm<
    LL = SyncFullCyclicLoading<ColMajorTilingOrder>,
    RL = SyncFullCyclicLoading<RowMajorTilingOrder>,
    AL = SyncFullCyclicLoading<RowMajorTilingOrder>,
    GW = UnitWriterFamily,
> {
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
    pub _al: PhantomData<AL>,
    pub _writer: PhantomData<GW>,
}

pub type SimpleUnitPairwiseCompressedAlgorithm<E> = SimpleUnitAlgorithm<
    SyncFullCyclicLoading<ColMajorTilingOrder>,
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    PairwiseCompressedUnitWriterFamily<E>,
>;

pub type SimpleUnitAccumulatorScatterAlgorithm<T> = SimpleUnitAlgorithm<
    SyncFullCyclicLoading<ColMajorTilingOrder>,
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    AccumulatorScatterUnitWriterFamily<T>,
>;

pub type SimpleUnitAccumulatorTransformAlgorithm<T> = SimpleUnitAlgorithm<
    SyncFullCyclicLoading<ColMajorTilingOrder>,
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    AccumulatorTransformUnitWriterFamily<T>,
>;

#[derive(Default, Clone, Debug)]
pub struct SimpleUnitSelectionArgs {
    pub tile_size: TileSizeSelection,
}

impl Display for SimpleUnitSelectionArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "_{}", self.tile_size)
    }
}

impl<RC, LL, RL, AL, GW> Routine<RC> for SimpleUnitAlgorithm<LL, RL, AL, GW>
where
    RC: RuntimeConfig,
    LL: FullLoadingStrategy<RC>,
    RL: FullLoadingStrategy<RC, Stage = LL::Stage, SyncStrategy = LL::SyncStrategy>,
    AL: FullLoadingStrategy<RC, SyncStrategy = LL::SyncStrategy>,
    GW: GlobalWriterFamily<RC>,
{
    type Strategy = SimpleUnitSelectionArgs;
    type Blueprint = BatchMatmulBlueprint;
}

impl<RC, LL, RL, AL, GW> BatchMatmulRoutine<RC> for SimpleUnitAlgorithm<LL, RL, AL, GW>
where
    RC: RuntimeConfig,
    LL: FullLoadingStrategy<RC>,
    RL: FullLoadingStrategy<RC, Stage = LL::Stage, SyncStrategy = LL::SyncStrategy>,
    AL: FullLoadingStrategy<RC, SyncStrategy = LL::SyncStrategy>,
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
                <SimpleUnitBatch<RC, LL, RL, AL, GW>>::launch_unchecked::<MA>(
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
        batch_validate_blueprint::<SimpleUnitBatch<RC, LL, RL, AL, GW>, RC>(
            client,
            blueprint,
            problem,
            dtypes,
            vector_sizes,
        )
    }

    fn num_stages() -> NumStages {
        SimpleUnitBatch::<RC, LL, RL, AL, GW>::num_stages()
    }

    fn expand_blueprint(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);
        let tile_matmul = TileMatmulKind::Register;

        if tile_matmul.can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(strategy) => infer_blueprint_unit(
                &device_settings.client,
                problem,
                device_settings.plane_dim,
                false,
                &device_settings.vector_sizes,
                UnitTilingBlueprintOptions {
                    tile: strategy.tile_size,
                    stage: match strategy.tile_size {
                        TileSizeSelection::MinTileSize => StageScaling::Enabled(2),
                        TileSizeSelection::MaxTileSize => StageScaling::Disabled,
                    },
                    partition: match strategy.tile_size {
                        TileSizeSelection::MinTileSize => PartitionScaling::Disabled,
                        TileSizeSelection::MaxTileSize => PartitionScaling::Enabled,
                    },
                    swizzle: tile_matmul.should_swizzle(&device_settings.client),
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
    ) -> Result<LaunchInfo<Self::Blueprint>, MatmulSetupError> {
        let ExpandInfo { blueprint, dtypes } = expand_info;

        Self::validate_blueprint(
            &device_settings.client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.vector_sizes,
        )?;

        let cubedim_resource = SimpleUnitBatch::<RC, LL, RL, AL, GW>::cubedim_resource(
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
