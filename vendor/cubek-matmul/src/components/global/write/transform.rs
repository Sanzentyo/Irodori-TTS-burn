//! Accumulator-domain global store transforms.
//!
//! Unlike a scalar post-cast epilogue, a store transform sees the accumulator
//! value before conversion to the primary output type and may write typed
//! auxiliary outputs owned by its runtime configuration. This supports
//! projection epilogues such as residual addition plus a prepared activation
//! without materializing the projection or launching a second finalizer.

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
            read::tiled::{TiledCoords, TiledLayout},
        },
        stage::{PlanePartitioner, partition_coordinates},
    },
    definition::{MatrixTypes, StageIdent},
};

/// Transform one valid accumulator scalar and perform any auxiliary stores.
///
/// `coordinate` is absolute in the logical MxN output matrix. Implementations
/// own the semantic contract of auxiliary bindings through `RC`; the writer
/// guarantees that this method is never called for an out-of-bounds lane.
#[cube]
pub trait AccumulatorGlobalStoreTransform<RC: RuntimeConfig>: Send + Sync + 'static {
    fn apply<ES: Numeric, EG: Numeric>(
        value: ES,
        coordinate: Coords2d,
        runtime_config: &mut RC,
    ) -> EG;
}

pub struct AccumulatorTransformPlaneWriterFamily<T> {
    _transform: PhantomData<T>,
}

#[derive(CubeType)]
pub struct AccumulatorTransformPlaneWriter<
    'a,
    IP: MatrixTypes,
    RC: RuntimeConfig,
    T: AccumulatorGlobalStoreTransform<RC>,
> {
    global: ViewMut<'a, Vector<IP::Global, IP::GlobalSize>, TiledCoords>,
    stage: PartitionedStage<IP::Stage, IP::StageSize>,
    runtime_config: RC,
    origin: Coords2d,
    valid_shape: Coords2d,

    #[cube(comptime)]
    plane_dim: u32,
    #[cube(comptime)]
    smem_config: StageMemoryConfig,
    #[cube(comptime)]
    _transform: PhantomData<T>,
}

#[cube]
impl<'a, IP, RC, T> AccumulatorTransformPlaneWriter<'a, IP, RC, T>
where
    IP: MatrixTypes,
    RC: RuntimeConfig,
    T: AccumulatorGlobalStoreTransform<RC>,
{
    fn new(
        global: ViewMut<'a, Vector<IP::Global, IP::GlobalSize>, Coords2d>,
        runtime_config: RC,
        origin: Coords2d,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        let valid_shape = global.shape();
        let stage = PartitionedStage::new(
            partition_coordinates::<PlanePartitioner>(
                config.plane_flow_partition_rule,
                config.plane_dim,
                config.smem_config.partitions_per_stage_along_col,
            ),
            config.smem_config,
        );
        AccumulatorTransformPlaneWriter::<'a, IP, RC, T> {
            global: global.view_mut(TiledLayout::new(StageIdent::Out, config.smem_config)),
            stage,
            runtime_config,
            origin,
            valid_shape,
            plane_dim: config.plane_dim,
            smem_config: config.smem_config,
            _transform: PhantomData,
        }
    }

    fn write(&mut self, tile_pos: Coords2d) {
        accumulator_transform_plane_write::<
            IP::Stage,
            IP::StageSize,
            IP::Global,
            IP::GlobalSize,
            RC,
            T,
        >(
            &mut self.global,
            &self.stage.unit_tile,
            tile_pos,
            self.origin,
            self.valid_shape,
            &mut self.runtime_config,
            self.plane_dim,
            self.smem_config,
        );
    }
}

#[cube]
impl<IP, RC, T> WriteEventListener for AccumulatorTransformPlaneWriter<'_, IP, RC, T>
where
    IP: MatrixTypes,
    RC: RuntimeConfig,
    T: AccumulatorGlobalStoreTransform<RC>,
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
    for AccumulatorTransformPlaneWriter<'a, IP, RC, T>
where
    IP: MatrixTypes,
    RC: RuntimeConfig,
    T: AccumulatorGlobalStoreTransform<RC>,
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

impl<RC, T> GlobalWriterFamily<RC> for AccumulatorTransformPlaneWriterFamily<T>
where
    RC: RuntimeConfig,
    T: AccumulatorGlobalStoreTransform<RC>,
{
    type Stage = PartitionedStageFamily;
    type Writer<'a, IP: MatrixTypes> = AccumulatorTransformPlaneWriter<'a, IP, RC, T>;
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn accumulator_transform_plane_write<ES, NS, EG, NG, RC, T>(
    global: &mut ViewMut<Vector<EG, NG>, TiledCoords>,
    smem_tile: &StridedTile<ES, NS>,
    tile_pos: Coords2d,
    origin: Coords2d,
    valid_shape: Coords2d,
    runtime_config: &mut RC,
    #[comptime] plane_dim: u32,
    #[comptime] smem_config: StageMemoryConfig,
) where
    ES: Numeric,
    NS: Size,
    EG: Numeric,
    NG: Size,
    RC: RuntimeConfig,
    T: AccumulatorGlobalStoreTransform<RC>,
{
    let output_vector_size = global.vector_size().comptime();
    let elements_in_tile = smem_config.comptime().elements_per_tile();
    let unit_step = plane_dim * output_vector_size as u32;
    let num_unit_writes = elements_in_tile.div_ceil(unit_step);

    #[unroll(num_unit_writes == 1)]
    for i in 0..num_unit_writes {
        let unit_write = UNIT_POS_X * output_vector_size as u32 + i * unit_step;
        if unit_write < elements_in_tile {
            accumulator_transform_write_vector::<ES, NS, EG, NG, RC, T>(
                global,
                smem_tile,
                unit_write,
                tile_pos,
                origin,
                valid_shape,
                runtime_config,
                smem_config,
            );
        }
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn accumulator_transform_write_vector<ES, NS, EG, NG, RC, T>(
    view: &mut ViewMut<Vector<EG, NG>, TiledCoords>,
    out_smem_tile: &StridedTile<ES, NS>,
    unit_write: u32,
    tile: Coords2d,
    origin: Coords2d,
    valid_shape: Coords2d,
    runtime_config: &mut RC,
    #[comptime] smem_config: StageMemoryConfig,
) where
    ES: Numeric,
    NS: Size,
    EG: Numeric,
    NG: Size,
    RC: RuntimeConfig,
    T: AccumulatorGlobalStoreTransform<RC>,
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

    let mut value: Vector<EG, NG> = Vector::empty();
    let layout = TiledLayout::new(StageIdent::Out, smem_config);
    let last_in_tile = smem_config.comptime().elements_per_tile() - 1;
    let tile_end = layout.to_source_pos((tile, last_in_tile));
    let full_tile = tile_end.0 < valid_shape.0 && tile_end.1 < valid_shape.1;
    #[unroll]
    for lane in 0..output_vector_size {
        let local = layout.to_source_pos((tile, unit_write + lane as u32));
        if full_tile || (local.0 < valid_shape.0 && local.1 < valid_shape.1) {
            let coordinate = (origin.0 + local.0, origin.1 + local.1);
            value.insert(
                lane,
                T::apply::<ES, EG>(staged.extract(lane), coordinate, runtime_config),
            );
        }
    }
    view.write_checked((tile, unit_write), value);
}
