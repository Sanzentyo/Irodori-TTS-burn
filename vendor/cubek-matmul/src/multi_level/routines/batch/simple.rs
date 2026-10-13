use crate::definition::{MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes};
use crate::multi_level::args::{
    ConfigRuntimeArg, InputRuntimeArg, MatmulArgs, OutputRuntimeArg, RuntimeConfig,
};
use crate::multi_level::components::batch::{
    BatchMatmulFamily, PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul,
};
use crate::multi_level::components::global::read::FullLoadingStrategy;
use crate::multi_level::components::global::read::async_full_tma::AsyncFullTmaLoading;
use crate::multi_level::components::global::read::sync_full_cyclic::SyncFullCyclicLoading;
use crate::multi_level::components::global::single_stage::simple::{
    ErasedStagePartition, SimpleMatmulFamily, StagePartitionMode,
};
use crate::multi_level::components::global::{
    AccumulatorTransformPlaneWriterFamily, GlobalWriterFamily, PairwiseCompressedPlaneWriterFamily,
    PlaneWriterFamily,
};
use crate::multi_level::{BatchMatmulRoutine, ExpandInfo, LaunchInfo, batch_validate_blueprint};
use cubecl::client::Client;
use cubecl::features::MmaConfig;
use cubecl::ir::AddressType;
use cubecl::{CubeCount, CubeDim};
use cubek_std::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint, SmAllocation};
use std::fmt::Display;
use std::marker::PhantomData;

use crate::multi_level::components::stage::{NumStages, PartitionBuffering, PlanePartitioner};
use crate::multi_level::components::tile::TileMatmulKind;
use crate::multi_level::definition::{
    BatchMatmulBlueprint, CubeMappingLaunch, MultiRowStrategy, TilingScheme, adjust_dtypes,
};
use crate::multi_level::routines::TilingArgs;
use crate::multi_level::routines::selector::{PlaneTilingBlueprintOptions, infer_blueprint_plane};
use crate::multi_level::tile::{ColMajorTilingOrder, RowMajorTilingOrder};
use crate::routine::{BlueprintStrategy, DeviceSettings, Routine};

/// Simple cyclic matmul whose writer combines adjacent accumulator columns
/// into one physical output column.
pub type SimpleCyclicPairwiseCompressedAlgorithm<E> = SimpleAlgorithm<
    SyncFullCyclicLoading<ColMajorTilingOrder>,
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    PairwiseCompressedPlaneWriterFamily<E>,
>;

/// Plane-tiled matmul whose accumulator-domain writer applies a typed store
/// transform before the primary output is materialized.
pub type SimpleCyclicAccumulatorTransformAlgorithm<T> = SimpleAlgorithm<
    SyncFullCyclicLoading<ColMajorTilingOrder>,
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    AccumulatorTransformPlaneWriterFamily<T>,
>;

/// Plane accelerated single stage matmul with configurable readers (default to cyclic)
pub struct SimpleAlgorithm<
    LL = SyncFullCyclicLoading<ColMajorTilingOrder>,
    RL = SyncFullCyclicLoading<RowMajorTilingOrder>,
    AL = SyncFullCyclicLoading<RowMajorTilingOrder>,
    GW = PlaneWriterFamily,
    PM = ErasedStagePartition,
> {
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
    pub _al: PhantomData<AL>,
    pub _writer: PhantomData<GW>,
    pub _partition_mode: PhantomData<PM>,
}

pub type SimpleTmaAlgorithm = SimpleAlgorithm<
    AsyncFullTmaLoading,
    AsyncFullTmaLoading,
    SyncFullCyclicLoading<RowMajorTilingOrder>,
>;
pub type SimpleBarrierAlgorithm<L> = SimpleAlgorithm<L, L>;

#[derive(Debug, Clone)]
pub struct SimpleArgs {
    /// Which tile matmul variant to dispatch on. Also stored into the blueprint.
    pub tile_matmul: TileMatmulKind,
    // Uses an optimized multi rows strategy.
    pub multi_rows: bool,
}

impl Default for SimpleArgs {
    fn default() -> Self {
        Self {
            tile_matmul: TileMatmulKind::Cmma,
            multi_rows: false,
        }
    }
}

impl TilingArgs for SimpleArgs {
    fn set_tile_matmul(&mut self, kind: TileMatmulKind) {
        self.tile_matmul = kind;
    }
}

impl Display for SimpleArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.multi_rows { "_multi_rows" } else { "" })
    }
}

/// Explicit selector policy for application-level autotuning of the simple
/// matmul routine. Kept separate from [`SimpleArgs`] so adding tuning knobs
/// does not break downstream struct literals or change released defaults.
#[derive(Debug, Clone)]
pub struct TunableSimpleArgs {
    pub tile_matmul: TileMatmulKind,
    pub multi_rows: bool,
    pub swizzled: Option<bool>,
    pub partition_buffering: Option<PartitionBuffering>,
    pub tiny_selection_enabled: bool,
}

impl Default for TunableSimpleArgs {
    fn default() -> Self {
        Self {
            tile_matmul: TileMatmulKind::Cmma,
            multi_rows: false,
            swizzled: None,
            partition_buffering: Some(PartitionBuffering::Single),
            tiny_selection_enabled: true,
        }
    }
}

impl TilingArgs for TunableSimpleArgs {
    fn set_tile_matmul(&mut self, kind: TileMatmulKind) {
        self.tile_matmul = kind;
    }
}

impl Display for TunableSimpleArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "_tunable_multi{}_swizzle{:?}_partition{:?}_tiny{}",
            self.multi_rows, self.swizzled, self.partition_buffering, self.tiny_selection_enabled,
        )
    }
}

/// Simple algorithm with an explicit, cache-key-visible selector policy.
pub struct TunableSimpleAlgorithm<
    LL = SyncFullCyclicLoading<ColMajorTilingOrder>,
    RL = SyncFullCyclicLoading<RowMajorTilingOrder>,
    AL = SyncFullCyclicLoading<RowMajorTilingOrder>,
    GW = PlaneWriterFamily,
> {
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
    pub _al: PhantomData<AL>,
    pub _writer: PhantomData<GW>,
}

/// The batch-matmul family powering [`SimpleAlgorithm`].
type SimpleBatch<RC, LL, RL, AL, GW, PM = ErasedStagePartition> = PartitionedBatchMatmulFamily<
    RC,
    SimpleMatmulFamily<PlanePartitioner, RC, LL, RL, AL, GW, PM>,
    RowMajorGlobalPartitionMatmul,
>;

impl<RC, LL, RL, AL, GW, PM> Routine<RC> for SimpleAlgorithm<LL, RL, AL, GW, PM>
where
    RC: RuntimeConfig,
    LL: FullLoadingStrategy<RC>,
    RL: FullLoadingStrategy<RC, SyncStrategy = LL::SyncStrategy>,
    AL: FullLoadingStrategy<RC>,
    GW: GlobalWriterFamily<RC>,
    PM: StagePartitionMode,
{
    type Strategy = SimpleArgs;
    type Blueprint = BatchMatmulBlueprint;
}

impl<RC, LL, RL, AL, GW, PM> BatchMatmulRoutine<RC> for SimpleAlgorithm<LL, RL, AL, GW, PM>
where
    RC: RuntimeConfig,
    LL: FullLoadingStrategy<RC>,
    RL: FullLoadingStrategy<RC, SyncStrategy = LL::SyncStrategy>,
    AL: FullLoadingStrategy<RC>,
    GW: GlobalWriterFamily<RC>,
    PM: StagePartitionMode,
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
                <SimpleBatch<RC, LL, RL, AL, GW, PM>>::launch_unchecked::<MA>(
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
        batch_validate_blueprint::<SimpleBatch<RC, LL, RL, AL, GW, PM>, RC>(
            client,
            blueprint,
            problem,
            dtypes,
            vector_sizes,
        )
    }

    fn num_stages() -> NumStages {
        SimpleBatch::<RC, LL, RL, AL, GW, PM>::num_stages()
    }

    fn expand_blueprint(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);
        let client = &device_settings.client;

        let tile_matmul = match strategy {
            BlueprintStrategy::Forced(blueprint) => blueprint.tile_matmul,
            BlueprintStrategy::Inferred(args) => args.tile_matmul,
        };

        if tile_matmul.can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(strategy) => {
                if strategy.multi_rows {
                    infer_blueprint_multi_rows(
                        tile_matmul,
                        client,
                        problem,
                        device_settings.plane_dim,
                        dtypes,
                        &device_settings.vector_sizes,
                    )
                } else {
                    infer_blueprint_plane(
                        tile_matmul,
                        client,
                        problem,
                        device_settings.plane_dim,
                        dtypes,
                        &device_settings.vector_sizes,
                        PlaneTilingBlueprintOptions {
                            partition_buffering: Some(PartitionBuffering::Single),
                            tiny_selection_enabled: true,
                            swizzled: tile_matmul.should_swizzle(client),
                            ..Default::default()
                        },
                    )
                }?
            }
        };
        Ok(ExpandInfo { blueprint, dtypes })
    }

    fn prepare(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<BatchMatmulBlueprint>, MatmulSetupError> {
        let ExpandInfo { blueprint, dtypes } = expand_info;

        let client = &device_settings.client;

        Self::validate_blueprint(
            client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.vector_sizes,
        )?;

        let cubedim_resource = SimpleBatch::<RC, LL, RL, AL, GW, PM>::cubedim_resource(
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
}

impl<RC, LL, RL, AL, GW> Routine<RC> for TunableSimpleAlgorithm<LL, RL, AL, GW>
where
    RC: RuntimeConfig,
    LL: FullLoadingStrategy<RC>,
    RL: FullLoadingStrategy<RC, SyncStrategy = LL::SyncStrategy>,
    AL: FullLoadingStrategy<RC>,
    GW: GlobalWriterFamily<RC>,
{
    type Strategy = TunableSimpleArgs;
    type Blueprint = BatchMatmulBlueprint;
}

impl<RC, LL, RL, AL, GW> BatchMatmulRoutine<RC> for TunableSimpleAlgorithm<LL, RL, AL, GW>
where
    RC: RuntimeConfig,
    LL: FullLoadingStrategy<RC>,
    RL: FullLoadingStrategy<RC, SyncStrategy = LL::SyncStrategy>,
    AL: FullLoadingStrategy<RC>,
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
        <SimpleAlgorithm<LL, RL, AL, GW> as BatchMatmulRoutine<RC>>::launch::<MA>(
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
        )
    }

    #[allow(clippy::result_large_err)]
    fn validate_blueprint(
        client: &Client,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        <SimpleAlgorithm<LL, RL, AL, GW> as BatchMatmulRoutine<RC>>::validate_blueprint(
            client,
            blueprint,
            problem,
            dtypes,
            vector_sizes,
        )
    }

    fn num_stages() -> NumStages {
        <SimpleAlgorithm<LL, RL, AL, GW> as BatchMatmulRoutine<RC>>::num_stages()
    }

    fn expand_blueprint(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);
        let client = &device_settings.client;
        let tile_matmul = match strategy {
            BlueprintStrategy::Forced(blueprint) => blueprint.tile_matmul,
            BlueprintStrategy::Inferred(args) => args.tile_matmul,
        };
        if tile_matmul.can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }
        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(args) if args.multi_rows => infer_blueprint_multi_rows(
                tile_matmul,
                client,
                problem,
                device_settings.plane_dim,
                dtypes,
                &device_settings.vector_sizes,
            )?,
            BlueprintStrategy::Inferred(args) => infer_blueprint_plane(
                tile_matmul,
                client,
                problem,
                device_settings.plane_dim,
                dtypes,
                &device_settings.vector_sizes,
                PlaneTilingBlueprintOptions {
                    partition_buffering: args.partition_buffering,
                    tiny_selection_enabled: args.tiny_selection_enabled,
                    swizzled: args
                        .swizzled
                        .unwrap_or_else(|| tile_matmul.should_swizzle(client)),
                    ..Default::default()
                },
            )?,
        };
        Ok(ExpandInfo { blueprint, dtypes })
    }

    fn prepare(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<BatchMatmulBlueprint>, MatmulSetupError> {
        <SimpleAlgorithm<LL, RL, AL, GW> as BatchMatmulRoutine<RC>>::prepare(
            problem,
            device_settings,
            expand_info,
        )
    }
}

fn infer_blueprint_multi_rows(
    tile_matmul: TileMatmulKind,
    client: &Client,
    problem: &MatmulProblem,
    plane_dim: u32,
    mut dtypes: MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> Result<(BatchMatmulBlueprint, MatmulElems), MatmulSetupError> {
    adjust_dtypes(client, &mut dtypes, tile_matmul.requires_accelerator());

    let supported = |m: u32, n: u32, k: u32| {
        tile_matmul.is_supported(
            client,
            MmaConfig {
                a_type: dtypes.lhs_register,
                b_type: dtypes.rhs_register,
                cd_type: dtypes.acc_register,
                m,
                n,
                k,
            },
        )
    };
    let cube_count_strategy = match client.properties().hardware.num_streaming_multiprocessors {
        Some(num_sms) => CubeCountStrategy::Sm {
            num_sms,
            sm_usage: SmAllocation::Exact,
            cubes_first: true,
        },
        None => CubeCountStrategy::Flattened,
    };

    if supported(8, 32, 16) {
        // A lot of multi-rows balanced with a
        // tile size of (8, 32, 16)
        let tiling_scheme = TilingScheme::builder()
            .with_tile_size((8, 32, 16).into())
            .with_partition_size((4, 4, 2).into())
            .with_stage_size((4, 1, 1).into())
            .build()
            .unwrap();

        let hypercube = HypercubeBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(4))
            .cube_count_strategy(cube_count_strategy)
            .build();

        Ok((
            BatchMatmulBlueprint::builder(tile_matmul, tiling_scheme, plane_dim, problem)
                .partition_buffering(PartitionBuffering::Single)
                .hypercube_blueprint(hypercube)
                .build(),
            dtypes,
        ))
    } else if supported(8, 8, 8) {
        let tiling_scheme = TilingScheme::builder()
            .with_tile_size((8, 8, 8).into())
            .with_partition_size((4, 8, 2).into())
            .with_stage_size((4, 1, 1).into())
            .build()
            .unwrap();
        let hypercube = HypercubeBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(4))
            .cube_count_strategy(cube_count_strategy)
            .build();

        Ok((
            BatchMatmulBlueprint::builder(tile_matmul, tiling_scheme, plane_dim, problem)
                .partition_buffering(PartitionBuffering::Single)
                .hypercube_blueprint(hypercube)
                .build(),
            dtypes,
        ))
    } else {
        infer_blueprint_plane(
            tile_matmul,
            client,
            problem,
            plane_dim,
            dtypes,
            vector_sizes,
            PlaneTilingBlueprintOptions {
                partition_buffering: Some(PartitionBuffering::Single),
                multi_row_strategy: MultiRowStrategy::Always(2),
                partition_k: Some(2),
                ..Default::default()
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tunable_simple_default_matches_released_simple_policy() {
        let args = TunableSimpleArgs::default();
        assert!(!args.multi_rows);
        assert_eq!(args.swizzled, None);
        assert_eq!(args.partition_buffering, Some(PartitionBuffering::Single));
        assert!(args.tiny_selection_enabled);
    }
}
