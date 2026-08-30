//! Fused long-sequence DiT MLP contract and gated residual update.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::Shape;
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

use super::precision::{KernelFloatPrecision, common_float_precision};

const INPUT_DIM: usize = 3_680;
const OUTPUT_DIM: usize = 1_280;
// Physical shader capability only. The model-side route selector owns the
// device/profile-specific production envelope.
const MIN_SEQUENCE: usize = 13;
const MAX_SEQUENCE: usize = 685;
const TILE_ROWS: usize = 64;
const TILE_COLUMNS: usize = 128;
const TILE_K: usize = 32;
const WORKGROUP_X: u32 = 16;
const WORKGROUP_Y: u32 = 16;
const REQUIRED_BINDINGS: u32 = 5;
const SHARED_BYTES: usize = (TILE_ROWS * TILE_K + TILE_K * TILE_COLUMNS) * size_of::<f32>();

#[derive(Debug)]
struct DitMlpContractResidualKernel {
    precision: KernelFloatPrecision,
    rows: u32,
    sequence: u32,
    inner: u32,
    input_row_stride: u32,
    layout: ContractTileLayout,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum ContractTileLayout {
    ScalarC128K32,
    VectorC128K32,
    VectorC128K16,
    VectorC128K16DoubleBuffered,
    VectorC128K16Prefetched,
    VectorC128K16Swizzled,
    VectorC64K32,
    VectorRows32C128K32,
    VectorRows48C128K32,
    VectorRows48C128K16,
    Warp32C128K32,
    Warp32C128K16,
    Warp32Rows128C128K32,
}

impl ContractTileLayout {
    const fn vectorizes_input(self) -> bool {
        !matches!(self, Self::ScalarC128K32)
    }

    const fn tile_k(self) -> usize {
        if matches!(
            self,
            Self::VectorC128K16
                | Self::VectorC128K16DoubleBuffered
                | Self::VectorC128K16Prefetched
                | Self::VectorC128K16Swizzled
                | Self::VectorRows48C128K16
                | Self::Warp32C128K16
        ) {
            16
        } else {
            32
        }
    }

    const fn tile_rows(self) -> usize {
        match self {
            Self::VectorRows32C128K32 => 32,
            Self::VectorRows48C128K32 | Self::VectorRows48C128K16 => 48,
            Self::Warp32Rows128C128K32 => 128,
            _ => 64,
        }
    }

    const fn tile_columns(self) -> usize {
        if matches!(self, Self::VectorC64K32) {
            64
        } else {
            128
        }
    }

    const fn workgroup(self) -> (u32, u32) {
        match self {
            Self::Warp32C128K32 | Self::Warp32C128K16 => (32, 8),
            Self::Warp32Rows128C128K32 => (32, 16),
            _ => (16, 16),
        }
    }
}

#[derive(Debug)]
struct DitAttentionOutputDirectResidualKernel {
    rows: u32,
    sequence: u32,
    gate_row_stride: u32,
    gate_offset: u32,
    layout: DirectOutputTileLayout,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum DirectOutputTileLayout {
    ScalarK32,
    VectorK32,
    VectorK16,
    VectorK16Prefetched,
}

impl DirectOutputTileLayout {
    const fn vectorizes_input(self) -> bool {
        !matches!(self, Self::ScalarK32)
    }

    const fn uses_k16(self) -> bool {
        matches!(self, Self::VectorK16 | Self::VectorK16Prefetched)
    }

    const fn uses_prefetch(self) -> bool {
        matches!(self, Self::VectorK16Prefetched)
    }
}

impl KernelSource for DitMlpContractResidualKernel {
    fn source(&self) -> SourceTemplate {
        let source = match self.layout {
            ContractTileLayout::VectorC64K32 => self.precision.source(
                include_str!("dit_mlp_contract_residual_c64_vec4.wgsl"),
                include_str!("dit_mlp_contract_residual_c64_vec4_f16.wgsl"),
            ),
            ContractTileLayout::VectorRows32C128K32 => self.precision.source(
                include_str!("dit_mlp_contract_residual_rows32_vec4.wgsl"),
                include_str!("dit_mlp_contract_residual_rows32_vec4_f16.wgsl"),
            ),
            ContractTileLayout::VectorRows48C128K32 | ContractTileLayout::VectorRows48C128K16 => {
                self.precision.source(
                    include_str!("dit_mlp_contract_residual_rows48_vec4.wgsl"),
                    include_str!("dit_mlp_contract_residual_rows48_vec4_f16.wgsl"),
                )
            }
            ContractTileLayout::VectorC128K16Swizzled => self.precision.source(
                include_str!("dit_mlp_contract_residual_swizzled_vec4.wgsl"),
                include_str!("dit_mlp_contract_residual_swizzled_vec4_f16.wgsl"),
            ),
            ContractTileLayout::VectorC128K16DoubleBuffered => self.precision.source(
                include_str!("dit_mlp_contract_residual_double_buffer_vec4.wgsl"),
                include_str!("dit_mlp_contract_residual_double_buffer_vec4_f16.wgsl"),
            ),
            ContractTileLayout::VectorC128K16Prefetched => self.precision.source(
                include_str!("dit_mlp_contract_residual_prefetch_vec4.wgsl"),
                include_str!("dit_mlp_contract_residual_prefetch_vec4_f16.wgsl"),
            ),
            ContractTileLayout::Warp32C128K32
            | ContractTileLayout::Warp32C128K16
            | ContractTileLayout::Warp32Rows128C128K32 => self.precision.source(
                include_str!("dit_mlp_contract_residual_warp32.wgsl"),
                include_str!("dit_mlp_contract_residual_warp32_f16.wgsl"),
            ),
            ContractTileLayout::VectorC128K32 | ContractTileLayout::VectorC128K16 => {
                self.precision.source(
                    include_str!("dit_mlp_contract_residual_vec4.wgsl"),
                    include_str!("dit_mlp_contract_residual_vec4_f16.wgsl"),
                )
            }
            ContractTileLayout::ScalarC128K32 => self.precision.source(
                include_str!("dit_mlp_contract_residual.wgsl"),
                include_str!("dit_mlp_contract_residual_f16.wgsl"),
            ),
        };
        let (tile_rows, tile_k, local_rows, input_tile_vecs, weight_tile_vecs, workgroup_y) =
            match self.layout {
                ContractTileLayout::VectorRows32C128K32 => (32, 32, 16, 256, 1024, 16),
                ContractTileLayout::VectorRows48C128K32 => (48, 32, 16, 384, 1024, 16),
                ContractTileLayout::VectorRows48C128K16 => (48, 16, 16, 192, 512, 16),
                ContractTileLayout::Warp32Rows128C128K32 => (128, 32, 16, 1024, 1024, 16),
                ContractTileLayout::VectorC128K16 => (64, 16, 16, 256, 512, 16),
                ContractTileLayout::VectorC128K16DoubleBuffered => (64, 16, 16, 256, 512, 16),
                ContractTileLayout::VectorC128K16Prefetched => (64, 16, 16, 256, 512, 16),
                ContractTileLayout::VectorC128K16Swizzled => (64, 16, 16, 256, 512, 16),
                ContractTileLayout::Warp32C128K16 => (64, 16, 8, 256, 512, 8),
                ContractTileLayout::Warp32C128K32 => (64, 32, 8, 512, 1024, 8),
                ContractTileLayout::VectorC64K32 => (64, 32, 16, 512, 512, 16),
                ContractTileLayout::ScalarC128K32 | ContractTileLayout::VectorC128K32 => {
                    (64, 32, 16, 512, 1024, 16)
                }
            };
        let (workgroup_x, _) = self.layout.workgroup();
        source
            .register("rows", self.rows.to_string())
            .register("sequence", self.sequence.to_string())
            .register("inner", self.inner.to_string())
            .register("input_row_stride", self.input_row_stride.to_string())
            .register("tile_rows", tile_rows.to_string())
            .register("tile_k", tile_k.to_string())
            .register("local_rows", local_rows.to_string())
            .register("input_tile_vecs", input_tile_vecs.to_string())
            .register("weight_tile_vecs", weight_tile_vecs.to_string())
            .register("double_input_tile_vecs", (2 * input_tile_vecs).to_string())
            .register(
                "double_weight_tile_vecs",
                (2 * weight_tile_vecs).to_string(),
            )
            .register("weight_tile_scalars", (weight_tile_vecs * 4).to_string())
            .register("workgroup_y", workgroup_y.to_string())
            .register("workgroup_units", (workgroup_x * workgroup_y).to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.precision,
            self.rows,
            self.sequence,
            self.inner,
            self.input_row_stride,
            self.layout,
        ))
    }
}

impl KernelSource for DitAttentionOutputDirectResidualKernel {
    fn source(&self) -> SourceTemplate {
        let source = if self.layout.uses_prefetch() {
            include_str!("dit_attention_output_direct_residual_prefetch_vec4.wgsl")
        } else if self.layout.vectorizes_input() {
            include_str!("dit_attention_output_direct_residual_vec4.wgsl")
        } else {
            include_str!("dit_attention_output_direct_residual.wgsl")
        };
        SourceTemplate::new(source)
            .register("rows", self.rows.to_string())
            .register("sequence", self.sequence.to_string())
            .register("gate_row_stride", self.gate_row_stride.to_string())
            .register("gate_offset", self.gate_offset.to_string())
            .register("tile_k", if self.layout.uses_k16() { "16" } else { "32" })
            .register(
                "input_tile_vecs",
                if self.layout.uses_k16() { "256" } else { "512" },
            )
            .register(
                "weight_tile_vecs",
                if self.layout.uses_k16() {
                    "512"
                } else {
                    "1024"
                },
            )
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.rows,
            self.sequence,
            self.gate_row_stride,
            self.gate_offset,
            self.layout,
        ))
    }
}

fn binding_is_compatible(
    tensor: &CubeTensor<WgpuRuntime>,
    required_elements: usize,
    precision: KernelFloatPrecision,
    alignment: u64,
) -> bool {
    let Some(required_bytes) = required_elements
        .checked_mul(precision.element_bytes())
        .and_then(|bytes| u64::try_from(bytes).ok())
    else {
        return false;
    };
    let binding = tensor.handle.clone().binding();
    tensor.client.properties().memory.alignment >= alignment
        && tensor
            .client
            .properties()
            .memory
            .alignment
            .is_multiple_of(alignment)
        && binding.size_in_used() >= required_bytes
        && binding.offset_start.unwrap_or(0).is_multiple_of(alignment)
}

/// Compute `residual + gate * (activated @ weight)` without materialising the
/// projected branch. The released inference graph supplies an already-tanh'd
/// gate and identity dropout.
pub fn try_dit_mlp_contract_residual_wgsl(
    activated: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        activated,
        weight,
        residual,
        gate,
        batch,
        sequence,
        INPUT_DIM,
        ContractTileLayout::ScalarC128K32,
    )
}

/// Vectorize the K-contiguous activation load and shared-memory staging while
/// preserving the established scalar reduction order and fused residual
/// epilogue. Both contiguous and explicitly pitched SwiGLU views are admitted.
pub fn try_dit_mlp_contract_residual_vec4_wgsl(
    activated: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        activated,
        weight,
        residual,
        gate,
        batch,
        sequence,
        INPUT_DIM,
        ContractTileLayout::VectorC128K32,
    )
}

/// Keep the 64x128 output tile and 256-invocation workgroup while halving the
/// cooperative K tile. This reduces workgroup storage from 24 to 12 KiB and
/// remains an exact-profile candidate until paired measurement approves it.
pub fn try_dit_mlp_contract_residual_vec4_k16_wgsl(
    activated: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        activated,
        weight,
        residual,
        gate,
        batch,
        sequence,
        INPUT_DIM,
        ContractTileLayout::VectorC128K16,
    )
}

/// K16 vector route with alternating workgroup pages. The extra 12 KiB of
/// shared storage removes the overwrite-prevention barrier after every K
/// slice, reducing the inner loop from two barriers per slice to one while
/// preserving the exact scalar FMA order and fused gated-residual store.
pub fn try_dit_mlp_contract_residual_double_buffer_vec4_k16_wgsl(
    activated: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        activated,
        weight,
        residual,
        gate,
        batch,
        sequence,
        INPUT_DIM,
        ContractTileLayout::VectorC128K16DoubleBuffered,
    )
}

/// K16 vector route that keeps the 12-KiB shared footprint and prefetches the
/// next tile's three per-invocation vectors before the overwrite barrier.
pub fn try_dit_mlp_contract_residual_prefetch_vec4_k16_wgsl(
    activated: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        activated,
        weight,
        residual,
        gate,
        batch,
        sequence,
        INPUT_DIM,
        ContractTileLayout::VectorC128K16Prefetched,
    )
}

/// K16 vector route with component-major shared-weight staging. The global
/// layout and arithmetic contract are unchanged; only workgroup-bank mapping
/// differs, so this remains a separately measured route.
pub fn try_dit_mlp_contract_residual_swizzled_vec4_k16_wgsl(
    activated: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        activated,
        weight,
        residual,
        gate,
        batch,
        sequence,
        INPUT_DIM,
        ContractTileLayout::VectorC128K16Swizzled,
    )
}

/// Use a 32-row output tile for small-M workloads. This doubles the number of
/// independent row workgroups and halves accumulator registers per invocation
/// while retaining the K32 reduction order and fused gated-residual store.
pub fn try_dit_mlp_contract_residual_rows32_vec4_wgsl(
    activated: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        activated,
        weight,
        residual,
        gate,
        batch,
        sequence,
        INPUT_DIM,
        ContractTileLayout::VectorRows32C128K32,
    )
}

/// Use a 48-row output tile to balance row reuse, accumulator pressure, and
/// workgroup count on small-M workloads while retaining the K32 reduction.
pub fn try_dit_mlp_contract_residual_rows48_vec4_wgsl(
    activated: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        activated,
        weight,
        residual,
        gate,
        batch,
        sequence,
        INPUT_DIM,
        ContractTileLayout::VectorRows48C128K32,
    )
}

/// The 48-row occupancy route with a 16-wide cooperative K slice. Both K32
/// and K16 remain independently tunable because their barrier/residency
/// trade-off changes with batch size and adapter generation.
pub fn try_dit_mlp_contract_residual_rows48_vec4_k16_wgsl(
    activated: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        activated,
        weight,
        residual,
        gate,
        batch,
        sequence,
        INPUT_DIM,
        ContractTileLayout::VectorRows48C128K16,
    )
}

/// Preserve 64-row reuse while halving the column tile and accumulator set.
/// This exposes twice as many column workgroups and reduces shared memory to
/// 16 KiB at the cost of loading each input row tile twice as often.
pub fn try_dit_mlp_contract_residual_c64_vec4_wgsl(
    activated: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        activated,
        weight,
        residual,
        gate,
        batch,
        sequence,
        INPUT_DIM,
        ContractTileLayout::VectorC64K32,
    )
}

/// Preserve the established 64x128 output tile while mapping each 32-lane
/// subgroup across one contiguous output row. Global traffic and ordered FMA
/// work match the 16x16 vector route; only the thread-to-tile mapping changes.
pub fn try_dit_mlp_contract_residual_warp32_wgsl(
    activated: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        activated,
        weight,
        residual,
        gate,
        batch,
        sequence,
        INPUT_DIM,
        ContractTileLayout::Warp32C128K32,
    )
}

/// Subgroup-aligned mapping with the reduced 16-wide cooperative K tile.
/// This preserves global traffic and ordered FMA work while combining the
/// lower shared-memory residency of K16 with contiguous 32-lane columns.
pub fn try_dit_mlp_contract_residual_warp32_k16_wgsl(
    activated: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        activated,
        weight,
        residual,
        gate,
        batch,
        sequence,
        INPUT_DIM,
        ContractTileLayout::Warp32C128K16,
    )
}

/// A 32x16 workgroup doubles row reuse while retaining eight output vectors
/// per invocation. It computes a 128x128 tile and halves repeated weight loads.
pub fn try_dit_mlp_contract_residual_warp32_rows128_wgsl(
    activated: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        activated,
        weight,
        residual,
        gate,
        batch,
        sequence,
        INPUT_DIM,
        ContractTileLayout::Warp32Rows128C128K32,
    )
}

/// Compute the released attention output projection and its gated residual in
/// one dispatch.
pub fn try_dit_attention_output_residual_wgsl(
    attention: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        attention,
        weight,
        residual,
        gate,
        batch,
        sequence,
        OUTPUT_DIM,
        ContractTileLayout::ScalarC128K32,
    )
}

/// Vector-staged form of the released attention output projection and fused
/// gated residual. The public route remains distinct from the scalar launcher
/// so exact-device tuning owns its admission.
pub fn try_dit_attention_output_residual_vec4_wgsl(
    attention: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_residual_wgsl(
        attention,
        weight,
        residual,
        gate,
        batch,
        sequence,
        OUTPUT_DIM,
        ContractTileLayout::VectorC128K32,
    )
}

/// Consume head-major SDPA output and its compact learned gate directly in the
/// released output projection, then apply the block gate/residual at store.
///
/// A successful call is one dispatch and never materializes token-major gated
/// attention. The exact layout contract is validated before allocation; all
/// other inputs return `None` to preserve the established two-stage route.
#[allow(clippy::too_many_arguments)]
pub fn try_dit_attention_output_direct_residual_wgsl(
    attention: CubeTensor<WgpuRuntime>,
    attention_gate: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    block_gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_attention_output_direct_residual_impl(
        attention,
        attention_gate,
        weight,
        residual,
        block_gate,
        batch,
        sequence,
        DirectOutputTileLayout::ScalarK32,
    )
}

/// Vector-staged form of the direct SDPA-to-output projection. Four adjacent
/// head components and their learned gates share one storage/workgroup
/// transaction while the established scalar FMA order is retained.
#[allow(clippy::too_many_arguments)]
pub fn try_dit_attention_output_direct_residual_vec4_wgsl(
    attention: CubeTensor<WgpuRuntime>,
    attention_gate: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    block_gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_attention_output_direct_residual_impl(
        attention,
        attention_gate,
        weight,
        residual,
        block_gate,
        batch,
        sequence,
        DirectOutputTileLayout::VectorK32,
    )
}

/// Vector-staged direct attention tail with a 16-wide cooperative K tile.
/// Geometry, arithmetic order, and the fused store epilogue are unchanged.
#[allow(clippy::too_many_arguments)]
pub fn try_dit_attention_output_direct_residual_vec4_k16_wgsl(
    attention: CubeTensor<WgpuRuntime>,
    attention_gate: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    block_gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_attention_output_direct_residual_impl(
        attention,
        attention_gate,
        weight,
        residual,
        block_gate,
        batch,
        sequence,
        DirectOutputTileLayout::VectorK16,
    )
}

/// K16 direct attention tail that prefetches the next head-major input/gate
/// product and weight tile without increasing shared-memory residency.
#[allow(clippy::too_many_arguments)]
pub fn try_dit_attention_output_direct_residual_vec4_k16_prefetch_wgsl(
    attention: CubeTensor<WgpuRuntime>,
    attention_gate: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    block_gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_attention_output_direct_residual_impl(
        attention,
        attention_gate,
        weight,
        residual,
        block_gate,
        batch,
        sequence,
        DirectOutputTileLayout::VectorK16Prefetched,
    )
}

#[allow(clippy::too_many_arguments)]
fn try_dit_attention_output_direct_residual_impl(
    attention: CubeTensor<WgpuRuntime>,
    attention_gate: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    block_gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
    layout: DirectOutputTileLayout,
) -> Option<CubeTensor<WgpuRuntime>> {
    const HEADS: usize = 20;
    const HEAD_DIM: usize = 64;
    const BINDINGS: u32 = 6;

    if attention.meta.num_dims() != 4
        || attention_gate.meta.num_dims() != 3
        || weight.meta.num_dims() != 2
        || residual.meta.num_dims() != 2
        || block_gate.meta.num_dims() != 2
    {
        return None;
    }
    let rows = batch.checked_mul(sequence)?;
    let output_elements = rows.checked_mul(OUTPUT_DIM)?;
    let gate_width = attention_gate.meta.shape()[2];
    let (gate_row_stride, gate_offset) = match gate_width {
        OUTPUT_DIM => (OUTPUT_DIM, 0),
        width if width == OUTPUT_DIM * 4 => (OUTPUT_DIM * 4, OUTPUT_DIM * 3),
        _ => return None,
    };
    let gate_elements = rows.checked_mul(gate_width)?;
    let precision = common_float_precision([
        attention.dtype,
        attention_gate.dtype,
        weight.dtype,
        residual.dtype,
        block_gate.dtype,
    ])?;
    if precision != KernelFloatPrecision::F32 {
        return None;
    }
    let vec4_bytes = u64::try_from(4 * precision.element_bytes()).ok()?;
    let same_device = attention.device == attention_gate.device
        && attention.device == weight.device
        && attention.device == residual.device
        && attention.device == block_gate.device;
    let compatible = matches!(batch, 1..=3)
        && (MIN_SEQUENCE..=MAX_SEQUENCE).contains(&sequence)
        && attention.meta.shape().as_slice() == [batch, HEADS, sequence, HEAD_DIM]
        && attention_gate.meta.shape().as_slice() == [batch, sequence, gate_width]
        && weight.meta.shape().as_slice() == [OUTPUT_DIM, OUTPUT_DIM]
        && residual.meta.shape().as_slice() == [rows, OUTPUT_DIM]
        && block_gate.meta.shape().as_slice() == [batch, OUTPUT_DIM]
        && attention.meta.strides()[..]
            == [
                HEADS * sequence * HEAD_DIM,
                sequence * HEAD_DIM,
                HEAD_DIM,
                1,
            ]
        && attention_gate.meta.strides()[..] == [sequence * gate_width, gate_width, 1]
        && weight.meta.strides()[..] == [OUTPUT_DIM, 1]
        && residual.meta.strides()[..] == [OUTPUT_DIM, 1]
        && block_gate.meta.strides()[..] == [OUTPUT_DIM, 1]
        && attention.is_contiguous()
        && attention_gate.is_contiguous()
        && weight.is_contiguous()
        && residual.is_contiguous()
        && block_gate.is_contiguous()
        && same_device
        && binding_is_compatible(&attention, output_elements, precision, vec4_bytes)
        && binding_is_compatible(&attention_gate, gate_elements, precision, vec4_bytes)
        && binding_is_compatible(&weight, OUTPUT_DIM * OUTPUT_DIM, precision, vec4_bytes)
        && binding_is_compatible(&residual, output_elements, precision, vec4_bytes)
        && binding_is_compatible(&block_gate, batch * OUTPUT_DIM, precision, vec4_bytes);
    if !compatible {
        return None;
    }

    let hardware = &attention.client.properties().hardware;
    let shared_bytes = if layout.uses_k16() {
        (TILE_ROWS * 16 + 16 * TILE_COLUMNS) * size_of::<f32>()
    } else {
        SHARED_BYTES
    };
    if hardware.max_bindings < BINDINGS
        || hardware.max_shared_memory_size < shared_bytes
        || hardware.max_units_per_cube < WORKGROUP_X * WORKGROUP_Y
        || hardware.max_cube_dim.0 < WORKGROUP_X
        || hardware.max_cube_dim.1 < WORKGROUP_Y
        || hardware.max_cube_count.0 < u32::try_from(OUTPUT_DIM / TILE_COLUMNS).ok()?
        || hardware.max_cube_count.1 < u32::try_from(rows.div_ceil(TILE_ROWS)).ok()?
    {
        return None;
    }

    let output_bytes = output_elements.checked_mul(precision.element_bytes())?;
    let client = attention.client.clone();
    let output_handle = client.empty(output_bytes);
    if output_handle.size_in_used() < u64::try_from(output_bytes).ok()?
        || !output_handle
            .offset_start
            .unwrap_or(0)
            .is_multiple_of(vec4_bytes)
    {
        return None;
    }
    let output = CubeTensor::new_contiguous(
        client.clone(),
        attention.device.clone(),
        Shape::from([rows, OUTPUT_DIM]),
        output_handle,
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DitAttentionOutputDirectResidualKernel {
                rows: u32::try_from(rows).ok()?,
                sequence: u32::try_from(sequence).ok()?,
                gate_row_stride: u32::try_from(gate_row_stride).ok()?,
                gate_offset: u32::try_from(gate_offset).ok()?,
                layout,
            },
            CubeDim::new_2d(WORKGROUP_X, WORKGROUP_Y),
        ));
    client.launch(
        task,
        CubeCount::new_2d(
            u32::try_from(OUTPUT_DIM / TILE_COLUMNS).ok()?,
            u32::try_from(rows.div_ceil(TILE_ROWS)).ok()?,
        ),
        KernelArguments::new()
            .with_buffer(attention.handle.binding())
            .with_buffer(attention_gate.handle.binding())
            .with_buffer(weight.handle.binding())
            .with_buffer(residual.handle.binding())
            .with_buffer(block_gate.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

#[allow(clippy::too_many_arguments)]
fn try_dit_projection_residual_wgsl(
    activated: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    residual: CubeTensor<WgpuRuntime>,
    gate: CubeTensor<WgpuRuntime>,
    batch: usize,
    sequence: usize,
    inner: usize,
    layout: ContractTileLayout,
) -> Option<CubeTensor<WgpuRuntime>> {
    if activated.meta.num_dims() != 2
        || weight.meta.num_dims() != 2
        || residual.meta.num_dims() != 2
        || gate.meta.num_dims() != 2
    {
        return None;
    }
    let rows = batch.checked_mul(sequence)?;
    let output_elements = rows.checked_mul(OUTPUT_DIM)?;
    let precision =
        common_float_precision([activated.dtype, weight.dtype, residual.dtype, gate.dtype])?;
    let vec4_bytes = u64::try_from(4 * precision.element_bytes()).ok()?;
    let same_device = activated.device == weight.device
        && activated.device == residual.device
        && activated.device == gate.device;
    let input_row_stride = *activated.meta.strides().first()?;
    let required_input_elements = rows
        .checked_sub(1)?
        .checked_mul(input_row_stride)?
        .checked_add(inner)?;
    let supported_input_pitch =
        input_row_stride == inner || (inner == INPUT_DIM && input_row_stride == 2 * INPUT_DIM);
    let tile_k = layout.tile_k();
    let compatible = matches!(batch, 1..=3)
        && (MIN_SEQUENCE..=MAX_SEQUENCE).contains(&sequence)
        && matches!(inner, INPUT_DIM | OUTPUT_DIM)
        && inner.is_multiple_of(tile_k)
        && activated.meta.shape().as_slice() == [rows, inner]
        && weight.meta.shape().as_slice() == [inner, OUTPUT_DIM]
        && residual.meta.shape().as_slice() == [rows, OUTPUT_DIM]
        && gate.meta.shape().as_slice() == [batch, OUTPUT_DIM]
        && activated.meta.strides()[1] == 1
        && (!layout.vectorizes_input() || input_row_stride.is_multiple_of(4))
        && supported_input_pitch
        && weight.meta.strides()[..] == [OUTPUT_DIM, 1]
        && residual.meta.strides()[..] == [OUTPUT_DIM, 1]
        && gate.meta.strides()[..] == [OUTPUT_DIM, 1]
        && weight.is_contiguous()
        && residual.is_contiguous()
        && gate.is_contiguous()
        && same_device
        && binding_is_compatible(
            &activated,
            required_input_elements,
            precision,
            if layout.vectorizes_input() {
                vec4_bytes
            } else {
                precision.element_bytes() as u64
            },
        )
        && binding_is_compatible(&weight, inner * OUTPUT_DIM, precision, vec4_bytes)
        && binding_is_compatible(&residual, output_elements, precision, vec4_bytes)
        && binding_is_compatible(&gate, batch * OUTPUT_DIM, precision, vec4_bytes);
    if !compatible {
        return None;
    }

    let hardware = &activated.client.properties().hardware;
    let tile_rows = layout.tile_rows();
    let tile_columns = layout.tile_columns();
    let shared_pages = if matches!(layout, ContractTileLayout::VectorC128K16DoubleBuffered) {
        2
    } else {
        1
    };
    let shared_bytes =
        shared_pages * (tile_rows * tile_k + tile_k * tile_columns) * size_of::<f32>();
    let (workgroup_x, workgroup_y) = layout.workgroup();
    if hardware.max_bindings < REQUIRED_BINDINGS
        || hardware.max_shared_memory_size < shared_bytes
        || hardware.max_units_per_cube < workgroup_x * workgroup_y
        || hardware.max_cube_dim.0 < workgroup_x
        || hardware.max_cube_dim.1 < workgroup_y
        || hardware.max_cube_count.0 < u32::try_from(OUTPUT_DIM / tile_columns).ok()?
        || hardware.max_cube_count.1 < u32::try_from(rows.div_ceil(tile_rows)).ok()?
    {
        return None;
    }

    let output_bytes = output_elements.checked_mul(precision.element_bytes())?;
    let client = activated.client.clone();
    let output_handle = client.empty(output_bytes);
    if output_handle.size_in_used() < u64::try_from(output_bytes).ok()?
        || !output_handle
            .offset_start
            .unwrap_or(0)
            .is_multiple_of(vec4_bytes)
    {
        return None;
    }
    let output = CubeTensor::new_contiguous(
        client.clone(),
        activated.device.clone(),
        Shape::from([rows, OUTPUT_DIM]),
        output_handle,
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DitMlpContractResidualKernel {
                precision,
                rows: u32::try_from(rows).ok()?,
                sequence: u32::try_from(sequence).ok()?,
                inner: u32::try_from(inner).ok()?,
                input_row_stride: u32::try_from(input_row_stride).ok()?,
                layout,
            },
            CubeDim::new_2d(workgroup_x, workgroup_y),
        ));
    client.launch(
        task,
        CubeCount::new_2d(
            u32::try_from(OUTPUT_DIM / tile_columns).ok()?,
            u32::try_from(rows.div_ceil(tile_rows)).ok()?,
        ),
        KernelArguments::new()
            .with_buffer(activated.handle.binding())
            .with_buffer(weight.handle.binding())
            .with_buffer(residual.handle.binding())
            .with_buffer(gate.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::wgpu::WgpuDevice,
        tensor::{FloatDType, Tensor},
    };

    #[test]
    fn released_geometry_and_source_contract_are_fixed() {
        assert_eq!(SHARED_BYTES, 24_576);
        assert_eq!(OUTPUT_DIM / TILE_COLUMNS, 10);
        assert_eq!(WORKGROUP_X * WORKGROUP_Y, 256);
        let shader = include_str!("dit_mlp_contract_residual.wgsl");
        assert_eq!(shader.matches("@binding(").count(), 5);
        assert!(shader.contains("residual_value + gate_value * branch"));
        for accumulator in 0..8 {
            assert_eq!(
                shader.matches(&format!("acc_{accumulator} = fma")).count(),
                1
            );
        }

        let rows32 = include_str!("dit_mlp_contract_residual_rows32_vec4.wgsl");
        assert_eq!(rows32.matches("@binding(").count(), 5);
        assert!(rows32.contains("const TILE_ROWS: u32 = 32u"));
        assert!(rows32.contains("input_tile: array<vec4<f32>, 256>"));
        assert!(rows32.contains("weight_tile: array<vec4<f32>, 1024>"));
        for accumulator in 0..4 {
            assert_eq!(
                rows32.matches(&format!("acc_{accumulator} = fma")).count(),
                4
            );
        }

        let rows48 = include_str!("dit_mlp_contract_residual_rows48_vec4.wgsl");
        assert_eq!(rows48.matches("@binding(").count(), 5);
        assert!(rows48.contains("const TILE_ROWS: u32 = 48u"));
        assert!(rows48.contains("input_tile: array<vec4<f32>, {{ input_tile_vecs }}>"));
        assert!(rows48.contains("weight_tile: array<vec4<f32>, {{ weight_tile_vecs }}>"));
        for accumulator in 0..6 {
            assert_eq!(
                rows48.matches(&format!("acc_{accumulator} = fma")).count(),
                4
            );
        }

        let c64 = include_str!("dit_mlp_contract_residual_c64_vec4.wgsl");
        assert_eq!(c64.matches("@binding(").count(), 5);
        assert!(c64.contains("const LOCAL_COLUMN_VECS: u32 = 16u"));
        assert!(c64.contains("input_tile: array<vec4<f32>, 512>"));
        assert!(c64.contains("weight_tile: array<vec4<f32>, 512>"));
        for accumulator in 0..4 {
            assert_eq!(c64.matches(&format!("acc_{accumulator} = fma")).count(), 4);
        }

        let warp32 = include_str!("dit_mlp_contract_residual_warp32.wgsl");
        assert_eq!(warp32.matches("@binding(").count(), 5);
        assert!(warp32.contains("@workgroup_size(32, {{ workgroup_y }}, 1)"));
        assert!(warp32.contains("input_tile: array<vec4<f32>, {{ input_tile_vecs }}>"));
        assert!(warp32.contains("weight_tile: array<vec4<f32>, {{ weight_tile_vecs }}>"));
        for accumulator in 0..8 {
            assert_eq!(
                warp32.matches(&format!("acc_{accumulator} = fma")).count(),
                4
            );
        }

        let vectorized = include_str!("dit_mlp_contract_residual_vec4.wgsl");
        assert_eq!(vectorized.matches("@binding(").count(), 5);
        assert!(vectorized.contains("input: array<vec4<f32>>"));
        assert!(vectorized.contains("input_tile: array<vec4<f32>, {{ input_tile_vecs }}>"));
        assert!(vectorized.contains("weight_tile: array<vec4<f32>, {{ weight_tile_vecs }}>"));
        for accumulator in 0..8 {
            assert_eq!(
                vectorized
                    .matches(&format!("acc_{accumulator} = fma"))
                    .count(),
                4
            );
        }

        let double_buffer = include_str!("dit_mlp_contract_residual_double_buffer_vec4.wgsl");
        assert_eq!(double_buffer.matches("@binding(").count(), 5);
        assert!(double_buffer.contains("page = 1u - page"));
        assert!(double_buffer.contains("load_page(page, next_k"));
        assert_eq!(double_buffer.matches("workgroupBarrier()").count(), 2);
        for accumulator in 0..8 {
            assert_eq!(
                double_buffer
                    .matches(&format!("acc_{accumulator} = fma"))
                    .count(),
                4
            );
        }

        let prefetch = include_str!("dit_mlp_contract_residual_prefetch_vec4.wgsl");
        assert_eq!(prefetch.matches("@binding(").count(), 5);
        assert!(prefetch.contains("prefetched_weight_0 = load_weight_value"));
        assert!(prefetch.contains("weight_tile[weight_load_0] = prefetched_weight_0"));
        for accumulator in 0..8 {
            assert_eq!(
                prefetch
                    .matches(&format!("acc_{accumulator} = fma"))
                    .count(),
                4
            );
        }
    }

    #[test]
    fn vectorized_input_matches_scalar_for_contiguous_and_pitched_views() {
        #[cfg(feature = "cli")]
        let _ = crate::backend_config::initialize_cli_tracing("warn");
        let device: burn::tensor::Device = WgpuDevice::DefaultDevice.into();
        assert_eq!(device.settings().float_dtype, FloatDType::F32);
        let batch = 1;
        let sequence = MIN_SEQUENCE;
        let rows = batch * sequence;
        let weight = Tensor::<2>::ones([INPUT_DIM, OUTPUT_DIM], &device);
        let residual = Tensor::<2>::ones([rows, OUTPUT_DIM], &device);
        let gate = Tensor::<2>::ones([batch, OUTPUT_DIM], &device);

        for activated in [
            Tensor::<2>::ones([rows, INPUT_DIM], &device),
            Tensor::<2>::ones([rows, INPUT_DIM * 2], &device).slice([0..rows, 0..INPUT_DIM]),
        ] {
            let scalar = try_dit_mlp_contract_residual_wgsl(
                activated
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU activation"),
                weight
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU weight"),
                residual
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU residual"),
                gate.clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU gate"),
                batch,
                sequence,
            )
            .expect("scalar-input contract route");
            let vectorized = try_dit_mlp_contract_residual_vec4_wgsl(
                activated
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU activation"),
                weight
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU weight"),
                residual
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU residual"),
                gate.clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU gate"),
                batch,
                sequence,
            )
            .expect("vector-input contract route");
            let k16 = try_dit_mlp_contract_residual_vec4_k16_wgsl(
                activated
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU activation"),
                weight
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU weight"),
                residual
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU residual"),
                gate.clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU gate"),
                batch,
                sequence,
            )
            .expect("K16 vector-input contract route");
            let double_buffer = try_dit_mlp_contract_residual_double_buffer_vec4_k16_wgsl(
                activated
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU activation"),
                weight
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU weight"),
                residual
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU residual"),
                gate.clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU gate"),
                batch,
                sequence,
            )
            .expect("double-buffered K16 vector-input contract route");
            let prefetch = try_dit_mlp_contract_residual_prefetch_vec4_k16_wgsl(
                activated
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU activation"),
                weight
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU weight"),
                residual
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU residual"),
                gate.clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU gate"),
                batch,
                sequence,
            )
            .expect("prefetched K16 vector-input contract route");
            let rows32 = try_dit_mlp_contract_residual_rows32_vec4_wgsl(
                activated
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU activation"),
                weight
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU weight"),
                residual
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU residual"),
                gate.clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU gate"),
                batch,
                sequence,
            )
            .expect("rows32 vector-input contract route");
            let rows48 = try_dit_mlp_contract_residual_rows48_vec4_wgsl(
                activated
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU activation"),
                weight
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU weight"),
                residual
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU residual"),
                gate.clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU gate"),
                batch,
                sequence,
            )
            .expect("rows48 vector-input contract route");
            let rows48_k16 = try_dit_mlp_contract_residual_rows48_vec4_k16_wgsl(
                activated
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU activation"),
                weight
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU weight"),
                residual
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU residual"),
                gate.clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU gate"),
                batch,
                sequence,
            )
            .expect("rows48 K16 vector-input contract route");
            let c64 = try_dit_mlp_contract_residual_c64_vec4_wgsl(
                activated
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU activation"),
                weight
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU weight"),
                residual
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU residual"),
                gate.clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU gate"),
                batch,
                sequence,
            )
            .expect("C64 vector-input contract route");
            let swizzled = try_dit_mlp_contract_residual_swizzled_vec4_k16_wgsl(
                activated
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU activation"),
                weight
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU weight"),
                residual
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU residual"),
                gate.clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU gate"),
                batch,
                sequence,
            )
            .expect("swizzled K16 vector-input contract route");
            let warp32_k16 = try_dit_mlp_contract_residual_warp32_k16_wgsl(
                activated
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU activation"),
                weight
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU weight"),
                residual
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU residual"),
                gate.clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU gate"),
                batch,
                sequence,
            )
            .expect("warp32 K16 contract route");
            let warp32 = try_dit_mlp_contract_residual_warp32_wgsl(
                activated
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU activation"),
                weight
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU weight"),
                residual
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU residual"),
                gate.clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU gate"),
                batch,
                sequence,
            )
            .expect("warp32 contract route");
            let warp32_rows128 = try_dit_mlp_contract_residual_warp32_rows128_wgsl(
                Tensor::<2>::ones([rows, INPUT_DIM], &device)
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU activation"),
                weight
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU weight"),
                residual
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU residual"),
                gate.clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU gate"),
                batch,
                sequence,
            )
            .expect("warp32 rows128 contract route");
            let scalar = Tensor::<2>::from_primitive::<crate::WgpuRaw>(scalar)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let vectorized = Tensor::<2>::from_primitive::<crate::WgpuRaw>(vectorized)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(scalar, vectorized);
            let k16 = Tensor::<2>::from_primitive::<crate::WgpuRaw>(k16)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(scalar, k16);
            let double_buffer = Tensor::<2>::from_primitive::<crate::WgpuRaw>(double_buffer)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(scalar, double_buffer);
            let prefetch = Tensor::<2>::from_primitive::<crate::WgpuRaw>(prefetch)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(scalar, prefetch);
            let rows32 = Tensor::<2>::from_primitive::<crate::WgpuRaw>(rows32)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(scalar, rows32);
            let rows48 = Tensor::<2>::from_primitive::<crate::WgpuRaw>(rows48)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(scalar, rows48);
            let rows48_k16 = Tensor::<2>::from_primitive::<crate::WgpuRaw>(rows48_k16)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(scalar, rows48_k16);
            let c64 = Tensor::<2>::from_primitive::<crate::WgpuRaw>(c64)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(scalar, c64);
            let swizzled = Tensor::<2>::from_primitive::<crate::WgpuRaw>(swizzled)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(scalar, swizzled);
            let warp32_k16 = Tensor::<2>::from_primitive::<crate::WgpuRaw>(warp32_k16)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(scalar, warp32_k16);
            let warp32 = Tensor::<2>::from_primitive::<crate::WgpuRaw>(warp32)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(scalar, warp32);
            let warp32_rows128 = Tensor::<2>::from_primitive::<crate::WgpuRaw>(warp32_rows128)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(scalar, warp32_rows128);
        }
    }

    #[test]
    fn direct_attention_k16_matches_vector_control() {
        #[cfg(feature = "cli")]
        let _ = crate::backend_config::initialize_cli_tracing("warn");
        let device: burn::tensor::Device = WgpuDevice::DefaultDevice.into();
        assert_eq!(device.settings().float_dtype, FloatDType::F32);
        let batch = 1;
        let sequence = MIN_SEQUENCE;
        let rows = batch * sequence;
        let attention = Tensor::<4>::ones([batch, 20, sequence, 64], &device);
        let attention_gate = Tensor::<3>::ones([batch, sequence, 4 * OUTPUT_DIM], &device);
        let weight = Tensor::<2>::ones([OUTPUT_DIM, OUTPUT_DIM], &device);
        let residual = Tensor::<2>::ones([rows, OUTPUT_DIM], &device);
        let block_gate = Tensor::<2>::ones([batch, OUTPUT_DIM], &device);
        let launch = |layout| {
            let attention = attention
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU attention");
            let attention_gate = attention_gate
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU attention gate");
            let weight = weight
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU weight");
            let residual = residual
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU residual");
            let block_gate = block_gate
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU block gate");
            match layout {
                DirectOutputTileLayout::VectorK16 => {
                    try_dit_attention_output_direct_residual_vec4_k16_wgsl(
                        attention,
                        attention_gate,
                        weight,
                        residual,
                        block_gate,
                        batch,
                        sequence,
                    )
                }
                DirectOutputTileLayout::VectorK16Prefetched => {
                    try_dit_attention_output_direct_residual_vec4_k16_prefetch_wgsl(
                        attention,
                        attention_gate,
                        weight,
                        residual,
                        block_gate,
                        batch,
                        sequence,
                    )
                }
                DirectOutputTileLayout::VectorK32 => {
                    try_dit_attention_output_direct_residual_vec4_wgsl(
                        attention,
                        attention_gate,
                        weight,
                        residual,
                        block_gate,
                        batch,
                        sequence,
                    )
                }
                _ => panic!("test accepts only vector direct-output layouts"),
            }
            .expect("direct attention output route")
        };
        let control = Tensor::<2>::from_primitive::<crate::WgpuRaw>(launch(
            DirectOutputTileLayout::VectorK32,
        ))
        .into_data()
        .to_vec::<f32>()
        .unwrap();
        let k16 = Tensor::<2>::from_primitive::<crate::WgpuRaw>(launch(
            DirectOutputTileLayout::VectorK16,
        ))
        .into_data()
        .to_vec::<f32>()
        .unwrap();
        assert_eq!(control, k16);
        let prefetched = Tensor::<2>::from_primitive::<crate::WgpuRaw>(launch(
            DirectOutputTileLayout::VectorK16Prefetched,
        ))
        .into_data()
        .to_vec::<f32>()
        .unwrap();
        assert_eq!(control, prefetched);
    }
}
