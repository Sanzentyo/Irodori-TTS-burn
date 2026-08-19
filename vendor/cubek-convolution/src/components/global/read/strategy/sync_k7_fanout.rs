//! One-pass fixed-k=7 halo fan-out for implicit-GEMM convolution.
//!
//! Each workgroup unit owns one contiguous NHWC channel vector at one halo
//! position. It reads that physical input vector once, then writes the values
//! directly to every logical im2col stage position that consumes them. Unlike
//! [`super::sync_k7_halo`], this route has no intermediate halo allocation,
//! internal barrier, or second shared-memory read/scatter phase.

use std::marker::PhantomData;

use cubecl::{ir::DeviceProperties, prelude::*, std::Swizzle};
use cubek_matmul::{
    components::{
        global::{
            GlobalReaderConfig, PlaneFlowPartition,
            memory::GlobalIterator,
            multi_stage::LoadMaxRoundPlaneCount,
            read::{
                FullLoadingStrategy, LoadingJob, LoadingValidation, ReaderMode,
                sync::Synchronous,
            },
        },
        stage::{LoadStageFamily, Stage, StageFamily},
    },
    definition::{MatmulElems, MatmulProblem},
};
use cubek_std::{
    InvalidConfigError, MatrixLayout, StageIdent,
    stage::{StageMemoryConfig, TilingLayout, as_swizzle_object},
    tile::{
        ContiguousTilingLayout, RowMajorTilingOrder, SharedTile, StridedTile, Tile, TileScope,
        TilingValidation,
    },
};

use crate::components::{ConvolutionKOrder, global::args::RuntimeArgs};

/// Scalar-backed stage used only by the fan-out loader. Scalar ownership
/// prevents two workgroup units from racing through vector read-modify-write
/// stores when their k7 destinations share a normal stage vector. MMA reads
/// expose the same allocation with the configured vector width.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct ScalarFanoutStageMemory<ES: Numeric, NS: Size, T: TilingLayout> {
    smem: Shared<[Vector<ES, Const<1>>]>,
    swizzle: Swizzle,
    buffer_index: u32,
    #[cube(comptime)]
    stage_size: u32,
    #[cube(comptime)]
    config: StageMemoryConfig,
    #[cube(comptime)]
    _phantom: PhantomData<(NS, T)>,
}

#[cube]
impl<ES: Numeric, NS: Size, T: TilingLayout> ScalarFanoutStageMemory<ES, NS, T> {
    fn new_aligned(
        #[comptime] alignment: usize,
        #[comptime] config: StageMemoryConfig,
    ) -> Self {
        // Fan-out writes are scalar-owned and therefore do not benefit from
        // the vector-store swizzle selected for the standard stage. Keeping
        // this private stage linear also makes scalar producer and vector
        // consumer agree on one byte addressing contract.
        let swizzle = as_swizzle_object(cubek_std::stage::SwizzleMode::None);
        let swizzle_align = swizzle.repeats_after();
        let align = comptime!(Ord::max(alignment, swizzle_align as usize));
        let type_size = ES::type_size().comptime();
        let stage_bytes = config.elements_per_stage() as usize * type_size;
        let stage_size = stage_bytes.next_multiple_of(align) / type_size;
        let smem = Shared::new_aligned_slice(config.num_stages as usize * stage_size, align);
        ScalarFanoutStageMemory::<ES, NS, T> {
            smem,
            swizzle,
            buffer_index: 0u32,
            stage_size: stage_size as u32,
            config,
            _phantom: PhantomData,
        }
    }

    fn with_buffer_index(&self, buffer_index: u32) -> Self {
        ScalarFanoutStageMemory::<ES, NS, T> {
            smem: self.smem.clone(),
            swizzle: self.swizzle,
            buffer_index,
            stage_size: self.stage_size,
            config: self.config,
            _phantom: PhantomData,
        }
    }

    fn as_slice<N: Size>(&self) -> &[Vector<ES, N>] {
        let start = (self.buffer_index * self.stage_size) as usize;
        self.smem[start..start + self.stage_size as usize].with_vector_size()
    }

    fn as_slice_mut<N: Size>(&mut self) -> &mut [Vector<ES, N>] {
        let start = (self.buffer_index * self.stage_size) as usize;
        self.smem[start..start + self.stage_size as usize].with_vector_size_mut()
    }

    unsafe fn free(&self) {
        unsafe { self.smem.free() };
    }
}

pub struct ScalarFanoutStageFamily;

impl StageFamily for ScalarFanoutStageFamily {
    type Stage<ES: Numeric, NS: Size, T: TilingLayout> =
        ScalarFanoutStageMemory<ES, NS, T>;
}

#[cube]
impl<ES: Numeric, NS: Size, T: TilingLayout> Stage<ES>
    for ScalarFanoutStageMemory<ES, NS, T>
{
    fn tile<Sc: TileScope>(this: &Self, tile: (u32, u32)) -> Tile<ES, Sc> {
        let config = this.config.comptime();
        let (tile_row, tile_col) = tile;
        let stage_cols = config.elements_per_stage_along_col();
        let tile_rows = config.elements_per_tile_along_row;
        let tile_cols = config.elements_per_tile_along_col;
        // `SharedTile` erases the storage vector width before the instruction
        // loader projects it to its own width. Keep metadata in scalar units
        // for this scalar-backed stage.
        let stride = stage_cols;
        let start = tile_row * tile_rows * stride + tile_col * tile_cols;
        let length = (tile_rows - 1) * stride + tile_cols;
        let strided = StridedTile::new_strided(
            this.as_slice::<NS>(),
            start,
            start + length,
            stride,
            this.swizzle,
            MatrixLayout::RowMajor,
        );
        Tile::new_SharedTile(SharedTile::wrap::<NS>(strided))
    }

    fn as_stage_tile<Sc: TileScope>(_this: &Self) -> Tile<ES, Sc> {
        // This stage deliberately has no representation in cubek-std's
        // closed StageTile enum. DirectStagePartition calls `tile` above.
        Tile::new_None()
    }
}

#[cube]
impl LoadStageFamily for ScalarFanoutStageFamily {
    fn create<ES: Numeric, NS: Size, T: TilingLayout>(
        #[comptime] alignment: usize,
        #[comptime] config: StageMemoryConfig,
    ) -> Self::Stage<ES, NS, T> {
        ScalarFanoutStageMemory::new_aligned(alignment, config)
    }

    fn with_buffer_index<ES: Numeric, NS: Size, T: TilingLayout>(
        stage: &Self::Stage<ES, NS, T>,
        buffer_index: u32,
    ) -> Self::Stage<ES, NS, T> {
        stage.with_buffer_index(buffer_index)
    }

    fn free<ES: Numeric, NS: Size, T: TilingLayout>(stage: &Self::Stage<ES, NS, T>) {
        unsafe { stage.free() };
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct SyncK7FanoutLoading;

impl LoadingValidation for SyncK7FanoutLoading {
    fn validate_with_config(
        _device_props: &DeviceProperties,
        config: &GlobalReaderConfig,
    ) -> Result<(), InvalidConfigError> {
        if config.stage_ident != StageIdent::Lhs {
            return Err(Box::new("k7 fan-out loading is only valid for the LHS"));
        }
        if config.input_load_flow.has_specialization() {
            return Err(Box::new(
                "k7 fan-out loading requires all workgroup planes to participate",
            ));
        }
        if config.reader_mode != ReaderMode::Relaxed {
            return Err(Box::new(
                "k7 fan-out loading requires relaxed bounds handling",
            ));
        }
        if config.smem_config.matrix_layout != MatrixLayout::RowMajor {
            return Err(Box::new(
                "k7 fan-out loading requires a row-major LHS stage",
            ));
        }
        ContiguousTilingLayout::<RowMajorTilingOrder>::check(config.smem_config)?;
        Ok(())
    }

    fn validate_with_problem(
        _problem: &MatmulProblem,
        _dtypes: &MatmulElems,
        ident: StageIdent,
    ) -> Result<(), InvalidConfigError> {
        if ident != StageIdent::Lhs {
            return Err(Box::new("k7 fan-out loading is only valid for the LHS"));
        }
        Ok(())
    }
}

impl LoadMaxRoundPlaneCount for SyncK7FanoutLoading {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        vector_size: VectorSize,
        plane_dim: u32,
        _dtype: StorageType,
    ) -> u32 {
        // A conservative upper bound. The exact fan-out job is smaller for
        // every supported k7 profile, but load-flow resource inference runs
        // before runtime convolution parameters are available.
        let stage_vectors =
            (elements_per_tile * tiles_per_stage).div_ceil(vector_size as u32);
        stage_vectors.div_ceil(plane_dim)
    }
}

#[cube]
impl FullLoadingStrategy<RuntimeArgs> for SyncK7FanoutLoading {
    type TilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;
    type SyncStrategy = Synchronous;
    type Job<EG: Numeric, NG: Size, ES: Numeric, NS: Size> = K7FanoutJob<EG, NG>;
    type Stage = ScalarFanoutStageFamily;

    fn new_job<EG: Numeric, NG: Size, ES: Numeric, NS: Size>(
        runtime_args: RuntimeArgs,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, NG, ES, NS> {
        let params = runtime_args.params.comptime();
        if comptime!(
            !matches!(params.k_order, ConvolutionKOrder::ChannelMajorK7)
                || !matches!(
                    params.dimensionality,
                    crate::components::Dimensionality::Dim1
                )
                || params.kernel_size[0] != 7
                || params.stride[0] != 1
                || params.padding[0] != 3 * params.dilation[0] as i32
        ) {
            push_validation_error(
                "k7 fan-out loader requires channel-major k=7, stride=1, same-padding Conv1d"
                    .to_string(),
            );
        }

        let vector_size = NG::value().comptime() as u32;
        let stage_m = config.smem_config.elements_per_stage_along_row();
        let stage_k = config.smem_config.elements_per_stage_along_col();
        let halo_span = stage_m + 6 * params.dilation[0];
        let max_channels = (stage_k + 6).div_ceil(7) + 1;
        let max_channel_groups = max_channels.div_ceil(vector_size) + 1;
        let max_halo_vectors = max_channel_groups * halo_span;
        let unit_count = config.loading_units_count();
        let unit_id = PlaneFlowPartition::new(config.plane_flow_config.partition_rule)
            .load_index(config.input_load_flow)
            * config.plane_dim
            + UNIT_POS_X;

        let fanout_rounds = max_halo_vectors.div_ceil(unit_count);

        K7FanoutJob::<EG, NG> {
            runtime_args,
            unit_id,
            unit_count,
            fanout_rounds,
            halo_span,
            stage_m,
            stage_k,
            _phantom: PhantomData,
        }
    }
}

#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct K7FanoutJob<EG: Numeric, NG: Size> {
    runtime_args: RuntimeArgs,
    unit_id: u32,

    #[cube(comptime)]
    unit_count: u32,
    #[cube(comptime)]
    fanout_rounds: u32,
    #[cube(comptime)]
    halo_span: u32,
    #[cube(comptime)]
    stage_m: u32,
    #[cube(comptime)]
    stage_k: u32,
    #[cube(comptime)]
    _phantom: PhantomData<(EG, NG)>,
}

#[cube]
impl<EG: Numeric, NG: Size, ES: Numeric, NS: Size>
    LoadingJob<
        EG,
        NG,
        ES,
        NS,
        ContiguousTilingLayout<RowMajorTilingOrder>,
        Synchronous,
    > for K7FanoutJob<EG, NG>
{
    type Stage = ScalarFanoutStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Vector<EG, NG>>,
        stage: &mut ScalarFanoutStageMemory<
            ES,
            NS,
            ContiguousTilingLayout<RowMajorTilingOrder>,
        >,
        _barrier: &(),
        #[comptime] config: GlobalReaderConfig,
    ) {
        if comptime!(task_id < this.fanout_rounds) {
            let halo_index = this.unit_id + task_id * this.unit_count;
            fan_out_halo_vector(&*this, halo_index, global_iter, stage, config);
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.fanout_rounds
    }
}

#[cube]
fn fan_out_halo_vector<EG: Numeric, NG: Size, ES: Numeric, NS: Size>(
    job: &K7FanoutJob<EG, NG>,
    halo_index: u32,
    global_iter: &GlobalIterator<Vector<EG, NG>>,
    stage: &mut ScalarFanoutStageMemory<
        ES,
        NS,
        ContiguousTilingLayout<RowMajorTilingOrder>,
    >,
    #[comptime] config: GlobalReaderConfig,
) {
    let stage_start = global_iter.offset();
    let stage_end = u32::min(stage_start + job.stage_k, job.runtime_args.shape_k);
    let first_channel = stage_start / 7u32;
    let last_channel = stage_end.saturating_sub(1u32) / 7u32;
    let vector_size = NG::value().comptime() as u32;
    let first_group = first_channel / vector_size;
    let last_group = last_channel / vector_size;
    let active_groups = last_group - first_group + 1;
    let active_halo_vectors = active_groups * job.halo_span;

    if halo_index < active_halo_vectors {
        let group_local = halo_index / job.halo_span;
        let halo_pos = halo_index % job.halo_span;
        let group_base_channel = (first_group + group_local) * vector_size;
        let dilation = job.runtime_args.params.comptime().dilation[0];

        // Any active lane/kernel pair that maps to this physical input is a
        // valid owner coordinate. Prefer the largest kernel so a partial
        // final M tile never decomposes into a nonexistent following batch.
        let mut owner_found = false;
        let mut owner_m = 0u32;
        let mut owner_k_local = 0u32;
        #[unroll]
        for lane in 0..NG::value() {
            let channel = group_base_channel + lane as u32;
            let channel_start = channel * 7u32;
            let intersect_start = u32::max(stage_start, channel_start);
            let intersect_end = u32::min(stage_end, channel_start + 7u32);
            if !owner_found && intersect_start < intersect_end {
                let available_low = intersect_start - channel_start;
                let available_high = intersect_end - channel_start - 1u32;
                let position_low = if halo_pos < job.stage_m {
                    0u32
                } else {
                    (halo_pos - job.stage_m + 1u32).div_ceil(dilation)
                };
                let position_high = u32::min(6u32, halo_pos / dilation);
                let kernel = u32::min(available_high, position_high);
                if kernel >= u32::max(available_low, position_low) {
                    owner_found = true;
                    owner_m = halo_pos - kernel * dilation;
                    owner_k_local = channel_start + kernel - stage_start;
                }
            }
        }

        if owner_found {
            let value = global_iter
                .view()
                .read_checked((owner_m, owner_k_local));
            #[unroll]
            for lane in 0..NG::value() {
                let channel = group_base_channel + lane as u32;
                if channel < job.runtime_args.channels {
                    #[unroll]
                    for kernel in 0..7u32 {
                        let global_k = channel * 7u32 + kernel;
                        if global_k >= stage_start && global_k < stage_end {
                            let shift = kernel * dilation;
                            if halo_pos >= shift {
                                let m = halo_pos - shift;
                                if m < job.stage_m {
                                    let k_local = global_k - stage_start;
                                    store_stage_scalar::<ES, NS>(
                                        stage,
                                        m,
                                        k_local,
                                        ES::cast_from(value.extract(lane)),
                                        config,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cube]
fn store_stage_scalar<ES: Numeric, NS: Size>(
    stage: &mut ScalarFanoutStageMemory<
        ES,
        NS,
        ContiguousTilingLayout<RowMajorTilingOrder>,
    >,
    row: u32,
    col: u32,
    value: ES,
    #[comptime] config: GlobalReaderConfig,
) {
    let smem = config.smem_config.comptime();
    let scalar_offset = row * smem.elements_per_stage_along_col() + col;
    stage.as_slice_mut::<Const<1>>()[scalar_offset as usize] = Vector::new(value);
}
