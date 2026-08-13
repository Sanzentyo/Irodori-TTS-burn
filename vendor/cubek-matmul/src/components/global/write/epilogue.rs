//! Generic output epilogues for plane-written matrix multiplication kernels.
//!
//! The epilogue runs after accumulation and before the final global-memory
//! store.  It can inspect runtime configuration and the absolute output
//! coordinate, which makes parameterized activations possible without a
//! second dispatch or an intermediate tensor.

use core::marker::PhantomData;

use cubecl::{
    prelude::*,
    std::tensor::{ViewMut, layout::Coords2d},
};
use cubek_std::{stage::StageMemoryConfig, tile::StridedTile};

use crate::{
    args::RuntimeConfig,
    components::{
        global::{
            GlobalWriter, GlobalWriterConfig, GlobalWriterFamily, PartitionedStage,
            PartitionedStageFamily, WriteEvent, WriteEventExpand, WriteEventListener,
            read::tiled::{TiledCoords, TiledLayout},
        },
        stage::{PlanePartitioner, partition_coordinates},
    },
    definition::{MatrixTypes, StageIdent},
};

/// A scalar output transform applied immediately before a global-memory store.
///
/// Implementations must be pure: the writer can invoke the transform in any
/// tile order. `coordinate` is absolute in the logical MxN output matrix.
#[cube]
pub trait GlobalEpilogue<RC: RuntimeConfig>: Send + Sync + 'static {
    fn apply<E: Numeric>(value: E, coordinate: Coords2d, runtime_config: &RC) -> E;
}

/// A plane writer family that applies [`GlobalEpilogue`] to every output scalar.
pub struct EpiloguePlaneWriterFamily<E> {
    _epilogue: PhantomData<E>,
}

#[derive(CubeType)]
pub struct EpiloguePlaneWriter<'a, IP: MatrixTypes, RC: RuntimeConfig, E: GlobalEpilogue<RC>> {
    global: ViewMut<'a, Vector<IP::Global, IP::GlobalSize>, TiledCoords>,
    stage: PartitionedStage<IP::Stage, IP::StageSize>,
    runtime_config: RC,
    origin: Coords2d,

    #[cube(comptime)]
    plane_dim: u32,
    #[cube(comptime)]
    smem_config: StageMemoryConfig,
    #[cube(comptime)]
    _epilogue: PhantomData<E>,
}

#[cube]
impl<'a, IP, RC, E> EpiloguePlaneWriter<'a, IP, RC, E>
where
    IP: MatrixTypes,
    RC: RuntimeConfig,
    E: GlobalEpilogue<RC>,
{
    fn new(
        global: ViewMut<'a, Vector<IP::Global, IP::GlobalSize>, Coords2d>,
        runtime_config: RC,
        origin: Coords2d,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        let stage = PartitionedStage::new(
            partition_coordinates::<PlanePartitioner>(
                config.plane_flow_partition_rule,
                config.plane_dim,
                config.smem_config.partitions_per_stage_along_col,
            ),
            config.smem_config,
        );

        EpiloguePlaneWriter::<'a, IP, RC, E> {
            global: global.view_mut(TiledLayout::new(StageIdent::Out, config.smem_config)),
            stage,
            runtime_config,
            origin,
            plane_dim: config.plane_dim,
            smem_config: config.smem_config,
            _epilogue: PhantomData,
        }
    }

    fn write(&mut self, tile_pos: Coords2d) {
        epilogue_plane_write::<IP::Stage, IP::StageSize, IP::Global, IP::GlobalSize, RC, E>(
            &mut self.global,
            &self.stage.unit_tile,
            tile_pos,
            self.origin,
            &self.runtime_config,
            self.plane_dim,
            self.smem_config,
        );
    }
}

#[cube]
impl<IP, RC, E> WriteEventListener for EpiloguePlaneWriter<'_, IP, RC, E>
where
    IP: MatrixTypes,
    RC: RuntimeConfig,
    E: GlobalEpilogue<RC>,
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
impl<'a, IP, RC, E> GlobalWriter<'a, IP, RC> for EpiloguePlaneWriter<'a, IP, RC, E>
where
    IP: MatrixTypes,
    RC: RuntimeConfig,
    E: GlobalEpilogue<RC>,
{
    type Stage = PartitionedStage<IP::Stage, IP::StageSize>;

    fn init(
        tensor: ViewMut<'a, Vector<IP::Global, IP::GlobalSize>, Coords2d>,
        runtime_config: RC,
        origin: Coords2d,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        Self::new(tensor, runtime_config, origin, config)
    }

    fn stage(this: &Self) -> Self::Stage {
        this.stage.clone()
    }
}

impl<RC, E> GlobalWriterFamily<RC> for EpiloguePlaneWriterFamily<E>
where
    RC: RuntimeConfig,
    E: GlobalEpilogue<RC>,
{
    type Stage = PartitionedStageFamily;
    type Writer<'a, IP: MatrixTypes> = EpiloguePlaneWriter<'a, IP, RC, E>;
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn epilogue_plane_write<ES, NS, EG, NG, RC, E>(
    global: &mut ViewMut<Vector<EG, NG>, TiledCoords>,
    smem_tile: &StridedTile<ES, NS>,
    tile_pos: Coords2d,
    origin: Coords2d,
    runtime_config: &RC,
    #[comptime] plane_dim: u32,
    #[comptime] smem_config: StageMemoryConfig,
) where
    ES: Numeric,
    NS: Size,
    EG: Numeric,
    NG: Size,
    RC: RuntimeConfig,
    E: GlobalEpilogue<RC>,
{
    let output_vector_size = global.vector_size().comptime();
    let elements_in_tile = smem_config.comptime().elements_per_tile();
    let unit_step = plane_dim * output_vector_size as u32;
    let num_unit_writes = elements_in_tile.div_ceil(unit_step);

    #[unroll(num_unit_writes == 1)]
    for i in 0..num_unit_writes {
        let unit_write = UNIT_POS_X * output_vector_size as u32 + i * unit_step;
        if unit_write < elements_in_tile {
            epilogue_write_vector::<ES, NS, EG, NG, RC, E>(
                global,
                smem_tile,
                unit_write,
                tile_pos,
                origin,
                runtime_config,
                smem_config,
            );
        }
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn epilogue_write_vector<ES, NS, EG, NG, RC, E>(
    view: &mut ViewMut<Vector<EG, NG>, TiledCoords>,
    out_smem_tile: &StridedTile<ES, NS>,
    unit_write: u32,
    tile: Coords2d,
    origin: Coords2d,
    runtime_config: &RC,
    #[comptime] smem_config: StageMemoryConfig,
) where
    ES: Numeric,
    NS: Size,
    EG: Numeric,
    NG: Size,
    RC: RuntimeConfig,
    E: GlobalEpilogue<RC>,
{
    let output_vector_size = view.vector_size().comptime();
    let out_smem_vector_size = out_smem_tile.container.vector_size().comptime();

    let staged = if output_vector_size == out_smem_vector_size {
        let offset = out_smem_tile.stage_offset(unit_write / output_vector_size as u32);
        out_smem_tile.container[offset as usize]
    } else if out_smem_vector_size < output_vector_size
        && output_vector_size.is_multiple_of(out_smem_vector_size)
    {
        let mut value = Vector::empty();
        #[unroll]
        for i in 0..output_vector_size / out_smem_vector_size {
            let offset = out_smem_tile.stage_offset(unit_write + i as u32);
            #[unroll]
            for j in 0..out_smem_vector_size {
                value.insert(
                    i * out_smem_vector_size + j,
                    out_smem_tile.container[offset as usize].extract(j),
                );
            }
        }
        value
    } else {
        unimplemented!()
    };

    let mut value: Vector<EG, NG> = Vector::cast_from(staged);
    let tile_rows = smem_config.comptime().elements_per_tile_along_row;
    let tile_cols = smem_config.comptime().elements_per_tile_along_col;
    let tile_base_row = origin.0 + tile.0 * tile_rows;
    let tile_base_col = origin.1 + tile.1 * tile_cols;
    #[unroll]
    for lane in 0..output_vector_size {
        let linear = unit_write + lane as u32;
        let coordinate = (
            tile_base_row + linear / tile_cols,
            tile_base_col + linear % tile_cols,
        );
        value.insert(
            lane,
            E::apply::<EG>(value.extract(lane), coordinate, runtime_config),
        );
    }
    view.write_checked((tile, unit_write), value);
}
