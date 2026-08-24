//! Accumulator-domain scatter stores without a primary matrix output.
//!
//! This writer preserves the generic CubeK matmul core while allowing a typed
//! runtime configuration to own all physical destinations. It is useful when
//! the logical MxN projection is consumed only as differently laid-out
//! tensors, so materializing a conventional MxN output would be wasteful.

use core::marker::PhantomData;

use cubecl::{
    prelude::*,
    std::tensor::{
        ViewMut,
        layout::{Coords2d, Layout, LayoutExpand},
    },
};
use cubek_std::{stage::StageMemoryConfig, tile::StridedTile};

use crate::{
    args::RuntimeConfig,
    components::{
        global::{
            GlobalWriter, GlobalWriterConfig, GlobalWriterFamily, PartitionedStage,
            PartitionedStageFamily, WriteEvent, WriteEventExpand, WriteEventListener,
            read::tiled::TiledLayout,
        },
        stage::{UnitPartitioner, partition_coordinates},
    },
    definition::{MatrixTypes, StageIdent},
};

/// Store one valid accumulator scalar into destinations owned by `RC`.
///
/// The writer performs logical M/N tail masking before invoking this method.
/// Implementations may therefore index coordinate-dependent auxiliary views
/// without repeating the matmul tile's validity checks.
#[cube]
pub trait AccumulatorGlobalScatter<RC: RuntimeConfig>: Send + Sync + 'static {
    fn store<ES: Numeric>(value: ES, coordinate: Coords2d, runtime_config: &mut RC);
}

pub struct AccumulatorScatterUnitWriterFamily<T> {
    _scatter: PhantomData<T>,
}

#[derive(CubeType)]
pub struct AccumulatorScatterUnitWriter<
    'a,
    IP: MatrixTypes,
    RC: RuntimeConfig,
    T: AccumulatorGlobalScatter<RC>,
> {
    stage: PartitionedStage<IP::Stage, IP::StageSize>,
    runtime_config: RC,
    origin: Coords2d,
    valid_shape: Coords2d,
    #[cube(comptime)]
    smem_config: StageMemoryConfig,
    #[cube(comptime)]
    _lifetime: PhantomData<&'a IP>,
    #[cube(comptime)]
    _scatter: PhantomData<T>,
}

#[cube]
impl<'a, IP, RC, T> AccumulatorScatterUnitWriter<'a, IP, RC, T>
where
    IP: MatrixTypes,
    RC: RuntimeConfig,
    T: AccumulatorGlobalScatter<RC>,
{
    fn write(&mut self, tile_pos: Coords2d) {
        accumulator_scatter_unit_write::<IP::Stage, IP::StageSize, RC, T>(
            &self.stage.unit_tile,
            tile_pos,
            self.origin,
            self.valid_shape,
            &mut self.runtime_config,
            self.smem_config,
        );
    }
}

#[cube]
impl<IP, RC, T> WriteEventListener for AccumulatorScatterUnitWriter<'_, IP, RC, T>
where
    IP: MatrixTypes,
    RC: RuntimeConfig,
    T: AccumulatorGlobalScatter<RC>,
{
    fn on_event(this: &mut Self, event: WriteEvent) {
        #[allow(clippy::single_match)]
        match event {
            WriteEvent::TileStored { tile } => this.write(tile),
            _ => {}
        }
    }
}

#[cube]
impl<'a, IP, RC, T> GlobalWriter<'a, IP, RC>
    for AccumulatorScatterUnitWriter<'a, IP, RC, T>
where
    IP: MatrixTypes,
    RC: RuntimeConfig,
    T: AccumulatorGlobalScatter<RC>,
{
    type Stage = PartitionedStage<IP::Stage, IP::StageSize>;

    fn init(
        _tensor: ViewMut<'a, Vector<IP::Global, IP::GlobalSize>, Coords2d>,
        runtime_config: RC,
        origin: Coords2d,
        valid_shape: Coords2d,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        assert!(config.gmem_config.vector_size == 1);
        let stage = PartitionedStage::new(
            partition_coordinates::<UnitPartitioner>(
                config.plane_flow_partition_rule,
                config.plane_dim,
                config.smem_config.partitions_per_stage_along_col,
            ),
            config.smem_config,
        );
        AccumulatorScatterUnitWriter::<'a, IP, RC, T> {
            stage,
            runtime_config,
            origin,
            valid_shape,
            smem_config: config.smem_config,
            _lifetime: PhantomData,
            _scatter: PhantomData,
        }
    }

    fn stage(this: &Self) -> Self::Stage {
        this.stage.clone()
    }
}

impl<RC, T> GlobalWriterFamily<RC> for AccumulatorScatterUnitWriterFamily<T>
where
    RC: RuntimeConfig,
    T: AccumulatorGlobalScatter<RC>,
{
    type Stage = PartitionedStageFamily;
    type Writer<'a, IP: MatrixTypes> = AccumulatorScatterUnitWriter<'a, IP, RC, T>;
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn accumulator_scatter_unit_write<ES, NS, RC, T>(
    smem_tile: &StridedTile<ES, NS>,
    tile_pos: Coords2d,
    origin: Coords2d,
    valid_shape: Coords2d,
    runtime_config: &mut RC,
    #[comptime] smem_config: StageMemoryConfig,
) where
    ES: Numeric,
    NS: Size,
    RC: RuntimeConfig,
    T: AccumulatorGlobalScatter<RC>,
{
    let elements = smem_config.comptime().elements_per_tile();
    let layout = TiledLayout::new(StageIdent::Out, smem_config);
    let vector_size = smem_tile.container.vector_size().comptime() as u32;
    for linear in 0..elements {
        let local = layout.to_source_pos((tile_pos, linear));
        if local.0 < valid_shape.0 && local.1 < valid_shape.1 {
            let stage_offset = smem_tile.stage_offset(linear / vector_size);
            let value = smem_tile.container[stage_offset as usize]
                .extract((linear % vector_size) as usize);
            T::store::<ES>(
                value,
                (origin.0 + local.0, origin.1 + local.1),
                runtime_config,
            );
        }
    }
}
