//! Fixed-k=7 channel-major halo loading for implicit-GEMM convolution.
//!
//! The generic im2col reader fetches the same NHWC source value once per
//! output/kernel pair. This loader first fetches each contiguous channel
//! vector in the one-dimensional halo into a small shared cache, synchronizes
//! the workgroup, then expands that cache into the unchanged CubeK matmul
//! stage. The MMA core and global writer therefore remain reusable.

use std::marker::PhantomData;

use cubecl::{
    ir::DeviceProperties,
    prelude::*,
    std::tensor::layout::{Layout, LayoutExpand},
};
use cubek_matmul::{
    components::{
        global::{
            GlobalReaderConfig, PlaneFlowPartition,
            memory::GlobalIterator,
            multi_stage::LoadMaxRoundPlaneCount,
            read::{
                FullLoadingStrategy, LoadingJob, LoadingValidation, ReaderMode, sync::Synchronous,
                tiled::TiledLayout,
            },
        },
        stage::{StridedStageFamily, StridedStageMemory},
    },
    definition::{MatmulElems, MatmulProblem},
};
use cubek_std::{
    InvalidConfigError, StageIdent,
    tile::{ContiguousTilingLayout, RowMajorTilingOrder, TilingValidation},
};

use crate::components::{ConvolutionKOrder, global::args::RuntimeArgs};

#[derive(CubeType, Clone, Copy)]
pub struct SyncK7HaloLoading;

impl LoadingValidation for SyncK7HaloLoading {
    fn validate_with_config(
        _device_props: &DeviceProperties,
        config: &GlobalReaderConfig,
    ) -> Result<(), InvalidConfigError> {
        if config.stage_ident != StageIdent::Lhs {
            return Err(Box::new("k7 halo loading is only valid for the LHS"));
        }
        if config.input_load_flow.has_specialization() {
            return Err(Box::new(
                "k7 halo loading requires every workgroup plane to reach its internal barrier",
            ));
        }
        if config.reader_mode != ReaderMode::Relaxed {
            return Err(Box::new("k7 halo loading requires relaxed bounds handling"));
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
            return Err(Box::new("k7 halo loading is only valid for the LHS"));
        }
        Ok(())
    }
}

impl LoadMaxRoundPlaneCount for SyncK7HaloLoading {
    fn max_round_plane_count(
        elements_per_tile: u32,
        tiles_per_stage: u32,
        vector_size: VectorSize,
        plane_dim: u32,
        _dtype: StorageType,
    ) -> u32 {
        let expanded_vectors = (elements_per_tile * tiles_per_stage).div_ceil(vector_size as u32);
        expanded_vectors.div_ceil(plane_dim)
    }
}

#[cube]
impl FullLoadingStrategy<RuntimeArgs> for SyncK7HaloLoading {
    type TilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;
    type SyncStrategy = Synchronous;
    type Job<EG: Numeric, NG: Size, ES: Numeric, NS: Size> = K7HaloJob<EG, NG>;
    type Stage = StridedStageFamily;

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
                "k7 halo loader requires channel-major k=7, stride=1, same-padding Conv1d"
                    .to_string(),
            );
        }

        let vector_size = NG::value().comptime() as u32;
        let stage_m = config.smem_config.elements_per_stage_along_row();
        let stage_k = config.smem_config.elements_per_stage_along_col();
        let dilation = params.dilation[0];
        let halo_span = stage_m + 6 * dilation;

        // Any K slice of `stage_k` scalars can touch at most this many
        // channel vectors, including both partial channel groups at its ends.
        let max_channels = (stage_k + 6).div_ceil(7) + 1;
        let max_channel_groups = max_channels.div_ceil(vector_size) + 1;
        let halo_vectors = max_channel_groups * halo_span;
        let halo = Shared::<[Vector<EG, NG>]>::new_slice(halo_vectors as usize);

        let unit_count = config.loading_units_count();
        let unit_id = PlaneFlowPartition::new(config.plane_flow_config.partition_rule)
            .load_index(config.input_load_flow)
            * config.plane_dim
            + UNIT_POS_X;
        let halo_rounds = halo_vectors.div_ceil(unit_count);
        let expanded_vectors = config
            .smem_config
            .elements_per_stage()
            .div_ceil(config.smem_config.vector_size);
        let expansion_rounds = expanded_vectors.div_ceil(unit_count);

        K7HaloJob::<EG, NG> {
            halo,
            runtime_args,
            unit_id,
            unit_count,
            halo_rounds,
            expansion_rounds,
            halo_span,
            max_channel_groups,
            stage_m,
            stage_k,
            _phantom: PhantomData,
        }
    }
}

#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct K7HaloJob<EG: Numeric, NG: Size> {
    halo: Shared<[Vector<EG, NG>]>,
    runtime_args: RuntimeArgs,
    unit_id: u32,

    #[cube(comptime)]
    unit_count: u32,
    #[cube(comptime)]
    halo_rounds: u32,
    #[cube(comptime)]
    expansion_rounds: u32,
    #[cube(comptime)]
    halo_span: u32,
    #[cube(comptime)]
    max_channel_groups: u32,
    #[cube(comptime)]
    stage_m: u32,
    #[cube(comptime)]
    stage_k: u32,
    #[cube(comptime)]
    _phantom: PhantomData<(EG, NG)>,
}

#[cube]
impl<EG: Numeric, NG: Size, ES: Numeric, NS: Size>
    LoadingJob<EG, NG, ES, NS, ContiguousTilingLayout<RowMajorTilingOrder>, Synchronous>
    for K7HaloJob<EG, NG>
{
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Vector<EG, NG>>,
        stage: &mut StridedStageMemory<ES, NS, ContiguousTilingLayout<RowMajorTilingOrder>>,
        _barrier: &(),
        #[comptime] config: GlobalReaderConfig,
    ) {
        if comptime!(task_id < this.halo_rounds) {
            let halo_index = this.unit_id + task_id * this.unit_count;
            load_halo_vector(this, halo_index, global_iter);
        } else {
            if comptime!(task_id == this.halo_rounds) {
                // Every plane reaches this point because specialized load flows are
                // rejected above. No expanded stage value observes a partial halo.
                sync_cube();
            }

            let expansion_task = comptime!(task_id - this.halo_rounds);
            if comptime!(expansion_task < this.expansion_rounds) {
                let vector_index = this.unit_id + expansion_task * this.unit_count;
                expand_stage_vector(&*this, vector_index, global_iter, stage, config);
            }
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        comptime!(this.halo_rounds + this.expansion_rounds)
    }
}

#[cube]
fn load_halo_vector<EG: Numeric, NG: Size>(
    job: &mut K7HaloJob<EG, NG>,
    halo_index: u32,
    global_iter: &GlobalIterator<Vector<EG, NG>>,
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
                // Prefer the largest usable kernel coordinate. This minimizes
                // the synthetic output row used to address the same physical
                // input value. On a partial final M tile, choosing the lowest
                // kernel can address a row beyond `problem.m`, which then
                // decomposes into a nonexistent batch before the spatial
                // check gets a chance to mask it.
                let kernel = u32::min(available_high, position_high);

                if kernel >= u32::max(available_low, position_low) {
                    owner_found = true;
                    owner_m = halo_pos - kernel * dilation;
                    owner_k_local = channel_start + kernel - stage_start;
                }
            }
        }

        let value = if owner_found {
            global_iter.view().read_checked((owner_m, owner_k_local))
        } else {
            Vector::<EG, NG>::zeroed()
        };
        job.halo[halo_index as usize] = value;
    }
}

#[cube]
fn expand_stage_vector<EG: Numeric, NG: Size, ES: Numeric, NS: Size>(
    job: &K7HaloJob<EG, NG>,
    vector_index: u32,
    global_iter: &GlobalIterator<Vector<EG, NG>>,
    stage: &mut StridedStageMemory<ES, NS, ContiguousTilingLayout<RowMajorTilingOrder>>,
    #[comptime] config: GlobalReaderConfig,
) {
    let scalar_base = vector_index * NS::value().comptime() as u32;
    if scalar_base < config.smem_config.elements_per_stage() {
        let stage_start = global_iter.offset();
        let first_channel = stage_start / 7;
        let vector_size = NG::value().comptime() as u32;
        let first_group = first_channel / vector_size;
        let dilation = job.runtime_args.params.comptime().dilation[0];
        let tile_elements = config.smem_config.elements_per_tile();
        let layout = TiledLayout::new(config.stage_ident, config.smem_config);
        let mut expanded = Vector::<ES, NS>::zeroed();

        #[unroll]
        for lane in 0..NS::value() {
            let physical = scalar_base + lane as u32;
            if physical < config.smem_config.elements_per_stage() {
                let nth_tile = physical / tile_elements;
                let within_tile = physical % tile_elements;
                let tile = ContiguousTilingLayout::<RowMajorTilingOrder>::to_x_y(
                    nth_tile,
                    config.smem_config,
                );
                let (m, k_local) = layout.to_source_pos((tile, within_tile));
                let global_k = stage_start + k_local;

                if global_k < job.runtime_args.shape_k {
                    let channel = global_k / 7;
                    let kernel = global_k % 7;
                    if channel < job.runtime_args.channels {
                        let group_local = channel / vector_size - first_group;
                        let halo_pos = m + kernel * dilation;
                        if group_local < job.max_channel_groups && halo_pos < job.halo_span {
                            let halo_value =
                                job.halo[(group_local * job.halo_span + halo_pos) as usize];
                            expanded.insert(
                                lane,
                                ES::cast_from(halo_value.extract((channel % vector_size) as usize)),
                            );
                        }
                    }
                }
            }
        }

        let stage_offset = stage.swizzle.apply(scalar_base, ES::type_size());
        stage.as_slice_mut::<NS>()[stage_offset as usize / NS::value()] = expanded;
    }
}
