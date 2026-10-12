//! Pairwise compressed accumulator epilogues.
//!
//! The matmul still computes the full logical `M x N` accumulator, but every
//! adjacent pair of columns is consumed by one epilogue invocation and stored
//! into an `M x (N / 2)` physical output. The RHS must therefore use an
//! interleaved pair layout (`[a0, b0, a1, b1, ...]`). This keeps the matmul
//! core generic while allowing GLU-family projections to avoid materialising
//! their full expansion.

use core::marker::PhantomData;

use cubecl::{
    prelude::*,
    std::tensor::{
        ViewMut,
        layout::{Coords2d, Layout, LayoutExpand},
    },
};
use cubek_std::{MatrixLayout, stage::StageMemoryConfig, tile::StridedTile};

use crate::{
    args::RuntimeConfig,
    components::{
        global::{
            GlobalWriter, GlobalWriterConfig, GlobalWriterFamily, PartitionedStage,
            PartitionedStageFamily, WriteEvent, WriteEventExpand, WriteEventListener,
            read::tiled::TiledLayout,
        },
        stage::{PlanePartitioner, UnitPartitioner, partition_coordinates},
    },
    definition::{MatrixTypes, StageIdent},
};

/// Combine one adjacent accumulator-column pair into one output scalar.
///
/// `coordinate` names the physical compressed output coordinate. The writer
/// masks row/column tails before calling this method, so implementations may
/// safely read coordinate-dependent runtime parameters.
#[cube]
pub trait PairwiseAccumulatorGlobalEpilogue<RC: RuntimeConfig>: Send + Sync + 'static {
    fn apply<ES: Numeric, EG: Numeric>(
        first: ES,
        second: ES,
        coordinate: Coords2d,
        runtime_config: &RC,
    ) -> EG;
}

/// Unit-tile counterpart used when the runtime has no strict-f32 cooperative
/// matrix implementation.
pub struct PairwiseCompressedUnitWriterFamily<E> {
    _epilogue: PhantomData<E>,
}

#[derive(CubeType)]
pub struct PairwiseCompressedUnitWriter<
    'a,
    IP: MatrixTypes,
    RC: RuntimeConfig,
    E: PairwiseAccumulatorGlobalEpilogue<RC>,
> {
    global: ViewMut<'a, Vector<IP::Global, IP::GlobalSize>, Coords2d>,
    stage: PartitionedStage<IP::Stage, IP::StageSize>,
    runtime_config: RC,
    origin: Coords2d,
    valid_shape: Coords2d,
    #[cube(comptime)]
    smem_config: StageMemoryConfig,
    #[cube(comptime)]
    _epilogue: PhantomData<E>,
}

#[cube]
impl<'a, IP, RC, E> GlobalWriter<'a, IP, RC>
    for PairwiseCompressedUnitWriter<'a, IP, RC, E>
where
    IP: MatrixTypes,
    RC: RuntimeConfig,
    E: PairwiseAccumulatorGlobalEpilogue<RC>,
{
    type Stage = PartitionedStage<IP::Stage, IP::StageSize>;

    fn init(
        tensor: ViewMut<'a, Vector<IP::Global, IP::GlobalSize>, Coords2d>,
        runtime_config: RC,
        origin: Coords2d,
        valid_shape: Coords2d,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        assert!(config.gmem_config.vector_size == 1);
        assert!(config.smem_config.elements_per_tile_along_col.is_multiple_of(2));
        let stage = PartitionedStage::new(
            partition_coordinates::<UnitPartitioner>(
                config.plane_flow_partition_rule,
                config.plane_dim,
                config.smem_config.partitions_per_stage_along_col,
            ),
            config.smem_config,
        );
        PairwiseCompressedUnitWriter::<'a, IP, RC, E> {
            global: tensor,
            stage,
            runtime_config,
            origin,
            valid_shape,
            smem_config: config.smem_config,
            _epilogue: PhantomData,
        }
    }

    fn stage(this: &Self) -> Self::Stage {
        this.stage.clone()
    }
}

#[cube]
impl<IP, RC, E> WriteEventListener for PairwiseCompressedUnitWriter<'_, IP, RC, E>
where
    IP: MatrixTypes,
    RC: RuntimeConfig,
    E: PairwiseAccumulatorGlobalEpilogue<RC>,
{
    fn on_event(this: &mut Self, event: WriteEvent) {
        #[allow(clippy::single_match)]
        match event {
            WriteEvent::TileStored { tile } => pairwise_compressed_unit_write::<
                IP::Stage,
                IP::StageSize,
                IP::Global,
                IP::GlobalSize,
                RC,
                E,
            >(
                &mut this.global,
                &this.stage.unit_tile,
                tile,
                this.origin,
                this.valid_shape,
                &this.runtime_config,
                this.smem_config,
            ),
            _ => {}
        }
    }
}

impl<RC, E> GlobalWriterFamily<RC> for PairwiseCompressedUnitWriterFamily<E>
where
    RC: RuntimeConfig,
    E: PairwiseAccumulatorGlobalEpilogue<RC>,
{
    type Stage = PartitionedStageFamily;
    type Writer<'a, IP: MatrixTypes> = PairwiseCompressedUnitWriter<'a, IP, RC, E>;
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn pairwise_compressed_unit_write<ES, NS, EG, NG, RC, E>(
    global: &mut ViewMut<Vector<EG, NG>, Coords2d>,
    smem_tile: &StridedTile<ES, NS>,
    tile_pos: Coords2d,
    origin: Coords2d,
    valid_shape: Coords2d,
    runtime_config: &RC,
    #[comptime] smem_config: StageMemoryConfig,
) where
    ES: Numeric,
    NS: Size,
    EG: Numeric,
    NG: Size,
    RC: RuntimeConfig,
    E: PairwiseAccumulatorGlobalEpilogue<RC>,
{
    let elements = smem_config.comptime().elements_per_tile();
    let layout = TiledLayout::new(StageIdent::Out, smem_config);
    let vector_size = smem_tile.container.vector_size().comptime() as u32;
    for pair in 0..elements / 2 {
        let linear = pair * 2;
        let local = layout.to_source_pos((tile_pos, linear));
        let absolute_col = origin.1 + local.1;
        if local.0 < valid_shape.0
            && local.1 + 1 < valid_shape.1
            && absolute_col.is_multiple_of(2)
        {
            let second = linear + 1;
            let first_offset = smem_tile.stage_offset(linear / vector_size);
            let second_offset = smem_tile.stage_offset(second / vector_size);
            let first = smem_tile.container[first_offset as usize]
                .extract((linear % vector_size) as usize);
            let second = smem_tile.container[second_offset as usize]
                .extract((second % vector_size) as usize);
            let coordinate = (origin.0 + local.0, absolute_col / 2);
            let mut output: Vector<EG, NG> = Vector::empty();
            output.insert(0, E::apply::<ES, EG>(first, second, coordinate, runtime_config));
            global.write_checked(coordinate, output);
        }
    }
}

/// Plane writer family for [`PairwiseAccumulatorGlobalEpilogue`].
pub struct PairwiseCompressedPlaneWriterFamily<E> {
    _epilogue: PhantomData<E>,
}

#[derive(CubeType)]
pub struct PairwiseCompressedPlaneWriter<
    'a,
    IP: MatrixTypes,
    RC: RuntimeConfig,
    E: PairwiseAccumulatorGlobalEpilogue<RC>,
> {
    /// Full logical output view. Its row stride addresses the compressed
    /// physical buffer; only columns `0..logical_cols/2` are written.
    global: ViewMut<'a, Vector<IP::Global, IP::GlobalSize>, Coords2d>,
    stage: PartitionedStage<IP::Stage, IP::StageSize>,
    runtime_config: RC,
    origin: Coords2d,
    valid_shape: Coords2d,

    #[cube(comptime)]
    plane_dim: u32,
    #[cube(comptime)]
    smem_config: StageMemoryConfig,
    #[cube(comptime)]
    _epilogue: PhantomData<E>,
}

#[cube]
impl<'a, IP, RC, E> PairwiseCompressedPlaneWriter<'a, IP, RC, E>
where
    IP: MatrixTypes,
    RC: RuntimeConfig,
    E: PairwiseAccumulatorGlobalEpilogue<RC>,
{
    fn new(
        global: ViewMut<'a, Vector<IP::Global, IP::GlobalSize>, Coords2d>,
        runtime_config: RC,
        origin: Coords2d,
        valid_shape: Coords2d,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        assert!(
            config.smem_config.matrix_layout == MatrixLayout::RowMajor,
            "pairwise compressed output requires row-major accumulator staging"
        );
        assert!(
            config.gmem_config.vector_size == 1,
            "pairwise compressed output requires scalar global stores"
        );
        assert!(
            config.smem_config.elements_per_tile_along_col.is_multiple_of(2),
            "pairwise compressed output requires an even tile width"
        );
        let stage = PartitionedStage::new(
            partition_coordinates::<PlanePartitioner>(
                config.plane_flow_partition_rule,
                config.plane_dim,
                config.smem_config.partitions_per_stage_along_col,
            ),
            config.smem_config,
        );
        PairwiseCompressedPlaneWriter::<'a, IP, RC, E> {
            global,
            stage,
            runtime_config,
            origin,
            valid_shape,
            plane_dim: config.plane_dim,
            smem_config: config.smem_config,
            _epilogue: PhantomData,
        }
    }

    fn write(&mut self, tile_pos: Coords2d) {
        pairwise_compressed_plane_write::<
            IP::Stage,
            IP::StageSize,
            IP::Global,
            IP::GlobalSize,
            RC,
            E,
        >(
            &mut self.global,
            &self.stage.unit_tile,
            tile_pos,
            self.origin,
            self.valid_shape,
            &self.runtime_config,
            self.plane_dim,
            self.smem_config,
        );
    }
}

#[cube]
impl<IP, RC, E> WriteEventListener for PairwiseCompressedPlaneWriter<'_, IP, RC, E>
where
    IP: MatrixTypes,
    RC: RuntimeConfig,
    E: PairwiseAccumulatorGlobalEpilogue<RC>,
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
impl<'a, IP, RC, E> GlobalWriter<'a, IP, RC>
    for PairwiseCompressedPlaneWriter<'a, IP, RC, E>
where
    IP: MatrixTypes,
    RC: RuntimeConfig,
    E: PairwiseAccumulatorGlobalEpilogue<RC>,
{
    type Stage = PartitionedStage<IP::Stage, IP::StageSize>;

    fn init(
        tensor: ViewMut<'a, Vector<IP::Global, IP::GlobalSize>, Coords2d>,
        runtime_config: RC,
        origin: Coords2d,
        valid_shape: Coords2d,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        Self::new(tensor, runtime_config, origin, valid_shape, config)
    }

    fn stage(this: &Self) -> Self::Stage {
        this.stage.clone()
    }
}

impl<RC, E> GlobalWriterFamily<RC> for PairwiseCompressedPlaneWriterFamily<E>
where
    RC: RuntimeConfig,
    E: PairwiseAccumulatorGlobalEpilogue<RC>,
{
    type Stage = PartitionedStageFamily;
    type Writer<'a, IP: MatrixTypes> = PairwiseCompressedPlaneWriter<'a, IP, RC, E>;
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn pairwise_compressed_plane_write<ES, NS, EG, NG, RC, E>(
    global: &mut ViewMut<Vector<EG, NG>, Coords2d>,
    smem_tile: &StridedTile<ES, NS>,
    tile_pos: Coords2d,
    origin: Coords2d,
    valid_shape: Coords2d,
    runtime_config: &RC,
    #[comptime] plane_dim: u32,
    #[comptime] smem_config: StageMemoryConfig,
) where
    ES: Numeric,
    NS: Size,
    EG: Numeric,
    NG: Size,
    RC: RuntimeConfig,
    E: PairwiseAccumulatorGlobalEpilogue<RC>,
{
    let elements_in_tile = smem_config.comptime().elements_per_tile();
    let unit_step = plane_dim;
    let num_unit_writes = elements_in_tile.div_ceil(unit_step);
    let layout = TiledLayout::new(StageIdent::Out, smem_config);
    let stage_vector_size = smem_tile.container.vector_size().comptime() as u32;

    #[unroll(num_unit_writes == 1)]
    for i in 0..num_unit_writes {
        let linear = UNIT_POS_X + i * unit_step;
        if linear < elements_in_tile {
            let local = layout.to_source_pos((tile_pos, linear));
            let absolute_col = origin.1 + local.1;
            let valid_pair = local.0 < valid_shape.0
                && local.1 + 1 < valid_shape.1
                && absolute_col.is_multiple_of(2);
            if valid_pair {
                let first_offset = smem_tile.stage_offset(linear / stage_vector_size);
                let second_linear = linear + 1;
                let second_offset = smem_tile.stage_offset(second_linear / stage_vector_size);
                let first = smem_tile.container[first_offset as usize]
                    .extract((linear % stage_vector_size) as usize);
                let second = smem_tile.container[second_offset as usize]
                    .extract((second_linear % stage_vector_size) as usize);
                let coordinate = (origin.0 + local.0, absolute_col / 2);
                let mut output: Vector<EG, NG> = Vector::empty();
                output.insert(
                    0,
                    E::apply::<ES, EG>(first, second, coordinate, runtime_config),
                );
                global.write_checked(coordinate, output);
            }
        }
    }
}
