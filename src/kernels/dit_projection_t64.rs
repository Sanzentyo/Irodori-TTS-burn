//! Exact-shape long-sequence DiT projection GEMMs.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::Shape;
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

use super::precision::{KernelFloatPrecision, common_float_precision};

pub const EXPAND_K: usize = 1_280;
pub const EXPAND_N: usize = 7_360;
pub const CONTRACT_K: usize = 3_680;
pub const CONTRACT_N: usize = 1_280;
pub const ATTENTION_QKV_GATE_K: usize = 1_280;
pub const ATTENTION_QKV_GATE_N: usize = 5_120;
pub const ATTENTION_OUTPUT_K: usize = 1_280;
pub const ATTENTION_OUTPUT_N: usize = 1_280;
pub const DURATION_EXPAND_K: usize = 1_024;
pub const DURATION_EXPAND_N: usize = 2_048;
pub const DURATION_INPUT_K: usize = 512;
pub const DURATION_INPUT_N: usize = 1_024;
pub const DIT_INPUT_K: usize = 32;
pub const DIT_INPUT_N: usize = 1_280;
const DIT_MIN_ROWS: usize = 13;
const DIT_MAX_SEQUENCE: usize = 685;
const DIT_MAX_BATCH: usize = 3;
const DURATION_MAX_ROWS: usize = 64;
const TILE_ROWS: usize = 64;
const TILE_COLUMNS: usize = 64;
const LONG_TILE_COLUMNS: usize = 128;
const TILE_K: usize = 16;
const LONG_TILE_K: usize = 32;
const WORKGROUP_X: u32 = 16;
const WORKGROUP_Y: u32 = 16;
const LONG_WORKGROUP_X: u32 = 32;
const LONG_WORKGROUP_Y: u32 = 8;
const REQUIRED_BINDINGS: u32 = 3;
const SHARED_BYTES: usize = (TILE_ROWS * TILE_K + TILE_K * TILE_COLUMNS) * size_of::<f32>();
const LONG_SHARED_BYTES: usize =
    (TILE_ROWS * LONG_TILE_K + LONG_TILE_K * LONG_TILE_COLUMNS) * size_of::<f32>();

#[derive(Debug)]
struct DitProjectionT64Kernel {
    precision: KernelFloatPrecision,
    rows: u32,
    inner: u32,
    columns: u32,
}

#[derive(Debug)]
struct DitProjectionC128Kernel {
    precision: KernelFloatPrecision,
    rows: u32,
    inner: u32,
    columns: u32,
    layout: ProjectionTileLayout,
}

#[derive(Debug)]
struct DitMlpExpandSwiGluC128Kernel {
    precision: KernelFloatPrecision,
    rows: u32,
    layout: MlpExpandTileLayout,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum MlpExpandTileLayout {
    C128ScalarK32,
    C128VectorK32,
    C128VectorK16,
    C128VectorK16Prefetched,
    Warp32VectorK32,
    Warp32Rows128VectorK32,
}

impl MlpExpandTileLayout {
    const fn vectorizes_input(self) -> bool {
        !matches!(self, Self::C128ScalarK32)
    }

    const fn uses_k16(self) -> bool {
        matches!(self, Self::C128VectorK16 | Self::C128VectorK16Prefetched)
    }

    const fn uses_warp32(self) -> bool {
        matches!(self, Self::Warp32VectorK32 | Self::Warp32Rows128VectorK32)
    }

    const fn uses_rows128(self) -> bool {
        matches!(self, Self::Warp32Rows128VectorK32)
    }
}

#[derive(Debug)]
struct DurationInputProjectionT64Kernel {
    precision: KernelFloatPrecision,
    rows: u32,
}

#[derive(Debug)]
struct DitInputProjectionBroadcastKernel {
    precision: KernelFloatPrecision,
    rows: u32,
    batch: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum ProjectionTileLayout {
    T64,
    C128ScalarK32,
    C128VectorK32,
    C128VectorK16,
    C128VectorK16Prefetched,
    C128VectorRows128K16,
}

impl ProjectionTileLayout {
    const fn uses_c128(self) -> bool {
        !matches!(self, Self::T64)
    }

    const fn vectorizes_input(self) -> bool {
        matches!(
            self,
            Self::C128VectorK32
                | Self::C128VectorK16
                | Self::C128VectorK16Prefetched
                | Self::C128VectorRows128K16
        )
    }

    const fn uses_k16(self) -> bool {
        matches!(
            self,
            Self::C128VectorK16 | Self::C128VectorK16Prefetched | Self::C128VectorRows128K16
        )
    }

    const fn uses_prefetch(self) -> bool {
        matches!(self, Self::C128VectorK16Prefetched)
    }

    const fn uses_rows128(self) -> bool {
        matches!(self, Self::C128VectorRows128K16)
    }
}

#[derive(Clone, Copy)]
struct ProjectionLaunchSpec {
    inner: usize,
    columns: usize,
    rows_are_admitted: fn(usize) -> bool,
    layout: ProjectionTileLayout,
}

impl KernelSource for DurationInputProjectionT64Kernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("duration_input_projection_t64.wgsl"),
                include_str!("duration_input_projection_t64_f16.wgsl"),
            )
            .register("rows", self.rows.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.rows))
    }
}

impl KernelSource for DitInputProjectionBroadcastKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("dit_input_projection_broadcast.wgsl"),
                include_str!("dit_input_projection_broadcast_f16.wgsl"),
            )
            .register("rows", self.rows.to_string())
            .register("batch", self.batch.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.rows, self.batch))
    }
}

impl KernelSource for DitProjectionT64Kernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("dit_projection_t64.wgsl"),
                include_str!("dit_projection_t64_f16.wgsl"),
            )
            .register("rows", self.rows.to_string())
            .register("inner", self.inner.to_string())
            .register("columns", self.columns.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.rows, self.inner, self.columns))
    }
}

impl KernelSource for DitProjectionC128Kernel {
    fn source(&self) -> SourceTemplate {
        let source = if self.layout.uses_prefetch() {
            self.precision.source(
                include_str!("dit_projection_c128_prefetch_vec4.wgsl"),
                include_str!("dit_projection_c128_prefetch_vec4_f16.wgsl"),
            )
        } else if self.layout.vectorizes_input() {
            self.precision.source(
                include_str!("dit_projection_c128_vec4.wgsl"),
                include_str!("dit_projection_c128_vec4_f16.wgsl"),
            )
        } else {
            self.precision.source(
                include_str!("dit_projection_c128.wgsl"),
                include_str!("dit_projection_c128_f16.wgsl"),
            )
        };
        let (tile_rows, tile_k, local_rows, input_tile_vecs, weight_tile_vecs, workgroup_y) =
            if self.layout.uses_rows128() {
                (128, 16, 16, 512, 512, 16)
            } else if self.layout.uses_k16() {
                (64, 16, 8, 256, 512, 8)
            } else {
                (64, 32, 8, 512, 1024, 8)
            };
        source
            .register("rows", self.rows.to_string())
            .register("inner", self.inner.to_string())
            .register("columns", self.columns.to_string())
            .register("tile_rows", tile_rows.to_string())
            .register("tile_k", tile_k.to_string())
            .register("local_rows", local_rows.to_string())
            .register("input_tile_vecs", input_tile_vecs.to_string())
            .register("weight_tile_vecs", weight_tile_vecs.to_string())
            .register("workgroup_y", workgroup_y.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.precision,
            self.rows,
            self.inner,
            self.columns,
            self.layout,
        ))
    }
}

impl KernelSource for DitMlpExpandSwiGluC128Kernel {
    fn source(&self) -> SourceTemplate {
        let source = match self.layout {
            MlpExpandTileLayout::C128VectorK16Prefetched => self.precision.source(
                include_str!("dit_mlp_expand_swiglu_c128_prefetch_vec4.wgsl"),
                include_str!("dit_mlp_expand_swiglu_c128_prefetch_vec4_f16.wgsl"),
            ),
            MlpExpandTileLayout::Warp32VectorK32 | MlpExpandTileLayout::Warp32Rows128VectorK32 => {
                self.precision.source(
                    include_str!("dit_mlp_expand_swiglu_warp32.wgsl"),
                    include_str!("dit_mlp_expand_swiglu_warp32_f16.wgsl"),
                )
            }
            MlpExpandTileLayout::C128VectorK32 | MlpExpandTileLayout::C128VectorK16 => {
                self.precision.source(
                    include_str!("dit_mlp_expand_swiglu_c128_vec4.wgsl"),
                    include_str!("dit_mlp_expand_swiglu_c128_vec4_f16.wgsl"),
                )
            }
            MlpExpandTileLayout::C128ScalarK32 => self.precision.source(
                include_str!("dit_mlp_expand_swiglu_c128.wgsl"),
                include_str!("dit_mlp_expand_swiglu_c128_f16.wgsl"),
            ),
        };
        let tile_rows = if self.layout.uses_rows128() { 128 } else { 64 };
        let tile_k = if self.layout.uses_k16() { 16 } else { 32 };
        let local_rows = if self.layout.uses_rows128() { 16 } else { 8 };
        let input_tile_vecs = tile_rows * tile_k / 4;
        let weight_tile_vecs = tile_k * 16 * 2;
        let workgroup_y = if self.layout.uses_rows128() { 16 } else { 8 };
        source
            .register("rows", self.rows.to_string())
            .register("tile_rows", tile_rows.to_string())
            .register("tile_k", tile_k.to_string())
            .register("local_rows", local_rows.to_string())
            .register("input_tile_vecs", input_tile_vecs.to_string())
            .register("weight_tile_vecs", weight_tile_vecs.to_string())
            .register("workgroup_y", workgroup_y.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.rows, self.layout))
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

/// Launch only for dense released B1/B2/B3 measured-length rows and packed
/// row-major weight.
/// Every contract mismatch returns `None` to preserve the tuned Burn fallback.
fn try_dit_projection_t64_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    spec: ProjectionLaunchSpec,
) -> Option<CubeTensor<WgpuRuntime>> {
    if input.meta.num_dims() != 2 || weight.meta.num_dims() != 2 {
        return None;
    }
    let rows = input.meta.shape()[0];
    let ProjectionLaunchSpec {
        inner,
        columns,
        rows_are_admitted,
        layout,
    } = spec;
    let use_long_tile = layout.uses_c128();
    let vectorized_input = layout.vectorizes_input();
    let k16_tile = layout.uses_k16();
    let rows128_k16 = layout.uses_rows128();
    let output_elements = rows.checked_mul(columns)?;
    let precision = common_float_precision([input.dtype, weight.dtype])?;
    let vec4_bytes = u64::try_from(4 * precision.element_bytes()).ok()?;
    let tile_k = if rows128_k16 || k16_tile {
        16
    } else if use_long_tile {
        LONG_TILE_K
    } else {
        TILE_K
    };
    let compatible = rows_are_admitted(rows)
        && (!vectorized_input || use_long_tile)
        && (!vectorized_input || inner.is_multiple_of(4))
        && inner.is_multiple_of(tile_k)
        && if use_long_tile {
            columns.is_multiple_of(4)
        } else {
            columns.is_multiple_of(TILE_COLUMNS)
        }
        && input.meta.shape().as_slice() == [rows, inner]
        && weight.meta.shape().as_slice() == [inner, columns]
        && input.meta.strides()[..] == [inner, 1]
        && weight.meta.strides()[..] == [columns, 1]
        && input.is_contiguous()
        && weight.is_contiguous()
        && input.device == weight.device
        && binding_is_compatible(
            &input,
            rows * inner,
            precision,
            if vectorized_input {
                vec4_bytes
            } else {
                precision.element_bytes() as u64
            },
        )
        && binding_is_compatible(&weight, inner * columns, precision, vec4_bytes);
    if !compatible {
        return None;
    }
    let tile_columns = if use_long_tile {
        LONG_TILE_COLUMNS
    } else {
        TILE_COLUMNS
    };
    let tile_rows = if rows128_k16 { 128 } else { TILE_ROWS };
    let shared_bytes = if rows128_k16 {
        (128 * 16 + 16 * LONG_TILE_COLUMNS) * size_of::<f32>()
    } else if k16_tile {
        (TILE_ROWS * 16 + 16 * LONG_TILE_COLUMNS) * size_of::<f32>()
    } else if use_long_tile {
        LONG_SHARED_BYTES
    } else {
        SHARED_BYTES
    };
    let hardware = &input.client.properties().hardware;
    let (workgroup_x, workgroup_y) = if rows128_k16 {
        (32, 16)
    } else if use_long_tile {
        (LONG_WORKGROUP_X, LONG_WORKGROUP_Y)
    } else {
        (WORKGROUP_X, WORKGROUP_Y)
    };
    if hardware.max_bindings < REQUIRED_BINDINGS
        || hardware.max_shared_memory_size < shared_bytes
        || hardware.max_units_per_cube < workgroup_x * workgroup_y
        || hardware.max_cube_dim.0 < workgroup_x
        || hardware.max_cube_dim.1 < workgroup_y
        || hardware.max_cube_count.0 < u32::try_from(columns.div_ceil(tile_columns)).ok()?
        || hardware.max_cube_count.1 < u32::try_from(rows.div_ceil(tile_rows)).ok()?
    {
        return None;
    }

    let output_bytes = output_elements.checked_mul(precision.element_bytes())?;
    let client = input.client.clone();
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
        input.device.clone(),
        Shape::from([rows, columns]),
        output_handle,
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> = if use_long_tile {
        Box::new(SourceKernel::new(
            DitProjectionC128Kernel {
                precision,
                rows: u32::try_from(rows).ok()?,
                inner: u32::try_from(inner).ok()?,
                columns: u32::try_from(columns).ok()?,
                layout,
            },
            CubeDim::new_2d(workgroup_x, workgroup_y),
        ))
    } else {
        Box::new(SourceKernel::new(
            DitProjectionT64Kernel {
                precision,
                rows: u32::try_from(rows).ok()?,
                inner: u32::try_from(inner).ok()?,
                columns: u32::try_from(columns).ok()?,
            },
            CubeDim::new_2d(WORKGROUP_X, WORKGROUP_Y),
        ))
    };
    client.launch(
        task,
        CubeCount::new_2d(
            u32::try_from(columns.div_ceil(tile_columns)).ok()?,
            u32::try_from(rows.div_ceil(tile_rows)).ok()?,
        ),
        KernelArguments::new()
            .with_buffer(input.handle.binding())
            .with_buffer(weight.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

fn dit_rows_are_admitted(rows: usize) -> bool {
    (DIT_MIN_ROWS..=DIT_MAX_SEQUENCE * DIT_MAX_BATCH).contains(&rows)
}

/// Physical sequence capability of the handwritten kernel. Device-specific
/// route policy lives in [`crate::route_autotune::ResolvedRouteTable`].
pub const fn dit_sequence_is_admitted(sequence: usize) -> bool {
    sequence >= DIT_MIN_ROWS && sequence <= DIT_MAX_SEQUENCE
}

/// Profile-only route control for comparing the handwritten projection family
/// with Burn/CubeK matmul in the exact same binary. Production builds compile
/// this to `true` and perform no environment lookup.
#[inline]
pub fn dit_projection_route_enabled() -> bool {
    #[cfg(feature = "profile")]
    {
        std::env::var("IRODORI_DISABLE_DIT_PROJECTION").as_deref() != Ok("1")
    }
    #[cfg(not(feature = "profile"))]
    {
        true
    }
}

/// Profile-only component gate for isolating B3 projection accuracy/cost.
#[inline]
pub fn dit_projection_component_enabled(component: &str) -> bool {
    #[cfg(feature = "profile")]
    {
        let variable = format!("IRODORI_DISABLE_DIT_{}", component.to_ascii_uppercase());
        std::env::var(variable).as_deref() != Ok("1")
    }
    #[cfg(not(feature = "profile"))]
    {
        let _ = component;
        true
    }
}

const fn duration_rows_are_admitted(rows: usize) -> bool {
    rows > 0 && rows <= DURATION_MAX_ROWS
}

/// Launch the exact released `w1 || w3` projection.
pub fn try_dit_mlp_expand_t64_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_t64_wgsl(
        input,
        weight,
        ProjectionLaunchSpec {
            inner: EXPAND_K,
            columns: EXPAND_N,
            rows_are_admitted: dit_rows_are_admitted,
            layout: ProjectionTileLayout::C128ScalarK32,
        },
    )
}

/// Launch the released expansion and consume its paired gate/value columns in
/// the same dispatch. The output is the dense `[rows, hidden]` SwiGLU tensor;
/// no `[rows, 2 * hidden]` projection is materialised.
pub fn try_dit_mlp_expand_swiglu_c128_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_mlp_expand_swiglu_c128_wgsl_impl(input, weight, MlpExpandTileLayout::C128ScalarK32)
}

/// Vectorize the K-contiguous input load and shared-memory staging while
/// preserving the scalar FMA order of the C128 projection.
pub fn try_dit_mlp_expand_swiglu_c128_vec4_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_mlp_expand_swiglu_c128_wgsl_impl(input, weight, MlpExpandTileLayout::C128VectorK32)
}

/// Keep the 64x64 compressed output tile and ordered F32 FMAs while reducing
/// cooperative K staging from 32 to 16. The exact route tuner decides whether
/// the doubled barrier count is offset by the 12-KiB workgroup footprint.
pub fn try_dit_mlp_expand_swiglu_c128_vec4_k16_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_mlp_expand_swiglu_c128_wgsl_impl(input, weight, MlpExpandTileLayout::C128VectorK16)
}

/// Overlap K16 input and gate/value weight loads with the preceding reduction
/// slice while retaining the single 12-KiB shared page.
pub fn try_dit_mlp_expand_swiglu_c128_vec4_k16_prefetch_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_mlp_expand_swiglu_c128_wgsl_impl(
        input,
        weight,
        MlpExpandTileLayout::C128VectorK16Prefetched,
    )
}

/// Preserve the established logical 64x64 output tile while mapping a full
/// 32-lane subgroup across each output row. The vec2 column representation
/// keeps the scalar work, shared-memory footprint, and ordered FMA count equal
/// to the 16x16 vec4 route.
pub fn try_dit_mlp_expand_swiglu_warp32_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_mlp_expand_swiglu_c128_wgsl_impl(input, weight, MlpExpandTileLayout::Warp32VectorK32)
}

/// Double the row reuse of the subgroup-aligned tile without changing each
/// invocation's eight-row accumulator footprint. A 32x16 workgroup computes
/// 128x64 outputs and halves repeated expansion-weight loads.
pub fn try_dit_mlp_expand_swiglu_warp32_rows128_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_mlp_expand_swiglu_c128_wgsl_impl(
        input,
        weight,
        MlpExpandTileLayout::Warp32Rows128VectorK32,
    )
}

fn try_dit_mlp_expand_swiglu_c128_wgsl_impl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    layout: MlpExpandTileLayout,
) -> Option<CubeTensor<WgpuRuntime>> {
    if input.meta.num_dims() != 2 || weight.meta.num_dims() != 2 {
        return None;
    }
    let rows = input.meta.shape()[0];
    let hidden = EXPAND_N / 2;
    let output_elements = rows.checked_mul(hidden)?;
    let precision = common_float_precision([input.dtype, weight.dtype])?;
    let vec4_bytes = u64::try_from(4 * precision.element_bytes()).ok()?;
    let tile_k = if layout.uses_k16() { 16 } else { LONG_TILE_K };
    let compatible = dit_rows_are_admitted(rows)
        && EXPAND_K.is_multiple_of(tile_k)
        && hidden.is_multiple_of(4)
        && input.meta.shape().as_slice() == [rows, EXPAND_K]
        && weight.meta.shape().as_slice() == [EXPAND_K, EXPAND_N]
        && input.meta.strides()[..] == [EXPAND_K, 1]
        && weight.meta.strides()[..] == [EXPAND_N, 1]
        && input.is_contiguous()
        && weight.is_contiguous()
        && input.device == weight.device
        && binding_is_compatible(
            &input,
            rows * EXPAND_K,
            precision,
            if layout.vectorizes_input() {
                vec4_bytes
            } else {
                precision.element_bytes() as u64
            },
        )
        && binding_is_compatible(&weight, EXPAND_K * EXPAND_N, precision, vec4_bytes);
    if !compatible {
        return None;
    }

    let output_tile_columns = LONG_TILE_COLUMNS / 2;
    let tile_rows = if layout.uses_rows128() {
        128
    } else {
        TILE_ROWS
    };
    let shared_bytes = (tile_rows * tile_k + tile_k * LONG_TILE_COLUMNS) * size_of::<f32>();
    let hardware = &input.client.properties().hardware;
    let (workgroup_x, workgroup_y) = if layout.uses_rows128() {
        (32, 16)
    } else if layout.uses_warp32() {
        (32, 8)
    } else {
        (WORKGROUP_X, WORKGROUP_Y)
    };
    if hardware.max_bindings < REQUIRED_BINDINGS
        || hardware.max_shared_memory_size < shared_bytes
        || hardware.max_units_per_cube < workgroup_x * workgroup_y
        || hardware.max_cube_dim.0 < workgroup_x
        || hardware.max_cube_dim.1 < workgroup_y
        || hardware.max_cube_count.0 < u32::try_from(hidden.div_ceil(output_tile_columns)).ok()?
        || hardware.max_cube_count.1 < u32::try_from(rows.div_ceil(tile_rows)).ok()?
    {
        return None;
    }

    let output_bytes = output_elements.checked_mul(precision.element_bytes())?;
    let client = input.client.clone();
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
        input.device.clone(),
        Shape::from([rows, hidden]),
        output_handle,
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DitMlpExpandSwiGluC128Kernel {
                precision,
                rows: u32::try_from(rows).ok()?,
                layout,
            },
            CubeDim::new_2d(workgroup_x, workgroup_y),
        ));
    client.launch(
        task,
        CubeCount::new_2d(
            u32::try_from(hidden.div_ceil(output_tile_columns)).ok()?,
            u32::try_from(rows.div_ceil(tile_rows)).ok()?,
        ),
        KernelArguments::new()
            .with_buffer(input.handle.binding())
            .with_buffer(weight.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

/// Launch the exact released `w2` projection.
pub fn try_dit_mlp_contract_t64_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_t64_wgsl(
        input,
        weight,
        ProjectionLaunchSpec {
            inner: CONTRACT_K,
            columns: CONTRACT_N,
            rows_are_admitted: dit_rows_are_admitted,
            layout: ProjectionTileLayout::C128ScalarK32,
        },
    )
}

/// Launch the exact released long-sequence `QKV || gate` projection.
pub fn try_dit_attention_qkv_gate_t64_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_t64_wgsl(
        input,
        weight,
        ProjectionLaunchSpec {
            inner: ATTENTION_QKV_GATE_K,
            columns: ATTENTION_QKV_GATE_N,
            rows_are_admitted: dit_rows_are_admitted,
            layout: ProjectionTileLayout::C128ScalarK32,
        },
    )
}

/// Vector-staged form of the exact released long-sequence `QKV || gate`
/// projection. It is admitted only by an exact route profile.
pub fn try_dit_attention_qkv_gate_c128_vec4_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_t64_wgsl(
        input,
        weight,
        ProjectionLaunchSpec {
            inner: ATTENTION_QKV_GATE_K,
            columns: ATTENTION_QKV_GATE_N,
            rows_are_admitted: dit_rows_are_admitted,
            layout: ProjectionTileLayout::C128VectorK32,
        },
    )
}

/// Keep the 64x128 output tile and 256-invocation workgroup while reducing K
/// staging from 32 to 16. Workgroup storage falls from 24 KiB to 12 KiB; the
/// exact tuner decides whether the added barriers are offset by occupancy.
pub fn try_dit_attention_qkv_gate_c128_k16_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_t64_wgsl(
        input,
        weight,
        ProjectionLaunchSpec {
            inner: ATTENTION_QKV_GATE_K,
            columns: ATTENTION_QKV_GATE_N,
            rows_are_admitted: dit_rows_are_admitted,
            layout: ProjectionTileLayout::C128VectorK16,
        },
    )
}

/// Overlap the next K16 input and weight loads with the current projection
/// slice while retaining the incumbent 12-KiB shared-memory footprint.
pub fn try_dit_attention_qkv_gate_c128_k16_prefetch_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_t64_wgsl(
        input,
        weight,
        ProjectionLaunchSpec {
            inner: ATTENTION_QKV_GATE_K,
            columns: ATTENTION_QKV_GATE_N,
            rows_are_admitted: dit_rows_are_admitted,
            layout: ProjectionTileLayout::C128VectorK16Prefetched,
        },
    )
}

/// Vector-staged QKV projection with a 128-row/K16 cooperative tile. The
/// larger row tile halves repeated weight loads while the smaller K tile keeps
/// workgroup memory at 16 KiB so the 512-invocation group can remain resident.
pub fn try_dit_attention_qkv_gate_c128_rows128_k16_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_t64_wgsl(
        input,
        weight,
        ProjectionLaunchSpec {
            inner: ATTENTION_QKV_GATE_K,
            columns: ATTENTION_QKV_GATE_N,
            rows_are_admitted: dit_rows_are_admitted,
            layout: ProjectionTileLayout::C128VectorRows128K16,
        },
    )
}

/// Launch the exact released long-sequence attention output projection.
pub fn try_dit_attention_output_t64_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_t64_wgsl(
        input,
        weight,
        ProjectionLaunchSpec {
            inner: ATTENTION_OUTPUT_K,
            columns: ATTENTION_OUTPUT_N,
            rows_are_admitted: dit_rows_are_admitted,
            layout: ProjectionTileLayout::C128ScalarK32,
        },
    )
}

/// Vector-staged form of the exact released attention output projection.
pub fn try_dit_attention_output_c128_vec4_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_t64_wgsl(
        input,
        weight,
        ProjectionLaunchSpec {
            inner: ATTENTION_OUTPUT_K,
            columns: ATTENTION_OUTPUT_N,
            rows_are_admitted: dit_rows_are_admitted,
            layout: ProjectionTileLayout::C128VectorK32,
        },
    )
}

/// Launch the exact released duration-block `w1 || w3` projection.
///
/// This candidate consumes the already fused, contiguous GPU-resident weight;
/// it performs no packing, host transfer, or readback.
pub fn try_duration_mlp_expand_t64_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    try_dit_projection_t64_wgsl(
        input,
        weight,
        ProjectionLaunchSpec {
            inner: DURATION_EXPAND_K,
            columns: DURATION_EXPAND_N,
            rows_are_admitted: duration_rows_are_admitted,
            layout: ProjectionTileLayout::T64,
        },
    )
}

/// Launch the released duration input projection with bias in one dispatch.
///
/// The text embedding, checkpoint-native output-major weight view, and bias remain on
/// the same WGPU device. No packing or host transfer is introduced.
pub fn try_duration_input_projection_t64_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    if input.meta.num_dims() != 2 || weight.meta.num_dims() != 2 || bias.meta.num_dims() != 1 {
        return None;
    }
    let rows = input.meta.shape()[0];
    let output_elements = rows.checked_mul(DURATION_INPUT_N)?;
    let precision = common_float_precision([input.dtype, weight.dtype, bias.dtype])?;
    let vec4_bytes = u64::try_from(4 * precision.element_bytes()).ok()?;
    let compatible = duration_rows_are_admitted(rows)
        && input.meta.shape().as_slice() == [rows, DURATION_INPUT_K]
        && weight.meta.shape().as_slice() == [DURATION_INPUT_K, DURATION_INPUT_N]
        && bias.meta.shape().as_slice() == [DURATION_INPUT_N]
        && input.meta.strides()[..] == [DURATION_INPUT_K, 1]
        && weight.meta.strides()[..] == [1, DURATION_INPUT_K]
        && bias.meta.strides()[..] == [1]
        && input.is_contiguous()
        && !weight.is_contiguous()
        && bias.is_contiguous()
        && input.device == weight.device
        && input.device == bias.device
        && binding_is_compatible(
            &input,
            rows * DURATION_INPUT_K,
            precision,
            precision.element_bytes() as u64,
        )
        && binding_is_compatible(
            &weight,
            DURATION_INPUT_K * DURATION_INPUT_N,
            precision,
            vec4_bytes,
        )
        && binding_is_compatible(&bias, DURATION_INPUT_N, precision, vec4_bytes);
    if !compatible {
        return None;
    }
    let hardware = &input.client.properties().hardware;
    if hardware.max_bindings < 4
        || hardware.max_shared_memory_size < SHARED_BYTES
        || hardware.max_units_per_cube < WORKGROUP_X * WORKGROUP_Y
        || hardware.max_cube_dim.0 < WORKGROUP_X
        || hardware.max_cube_dim.1 < WORKGROUP_Y
        || hardware.max_cube_count.0 < u32::try_from(DURATION_INPUT_N / TILE_COLUMNS).ok()?
        || hardware.max_cube_count.1 < u32::try_from(rows.div_ceil(TILE_ROWS)).ok()?
    {
        return None;
    }

    let output_bytes = output_elements.checked_mul(precision.element_bytes())?;
    let client = input.client.clone();
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
        input.device.clone(),
        Shape::from([rows, DURATION_INPUT_N]),
        output_handle,
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DurationInputProjectionT64Kernel {
                precision,
                rows: u32::try_from(rows).ok()?,
            },
            CubeDim::new_2d(WORKGROUP_X, WORKGROUP_Y),
        ));
    client.launch(
        task,
        CubeCount::new_2d(
            u32::try_from(DURATION_INPUT_N / TILE_COLUMNS).ok()?,
            u32::try_from(rows.div_ceil(TILE_ROWS)).ok()?,
        ),
        KernelArguments::new()
            .with_buffer(input.handle.binding())
            .with_buffer(weight.handle.binding())
            .with_buffer(bias.handle.binding())
            .with_buffer(output.handle.clone().binding()),
    );
    Some(output)
}

/// Project one physical latent row and broadcast the result to an Independent
/// CFG B2/B3 topology in the same dispatch. The checkpoint-native contiguous
/// weight is consumed directly; no latent cat or projected repeat is issued.
pub fn try_dit_input_projection_broadcast_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    broadcast_batch: usize,
) -> Option<CubeTensor<WgpuRuntime>> {
    if input.meta.num_dims() != 3 || weight.meta.num_dims() != 2 || bias.meta.num_dims() != 1 {
        return None;
    }
    let rows = input.meta.shape()[1];
    let output_elements = broadcast_batch
        .checked_mul(rows)?
        .checked_mul(DIT_INPUT_N)?;
    let precision = common_float_precision([input.dtype, weight.dtype, bias.dtype])?;
    let vec4_bytes = u64::try_from(4 * precision.element_bytes()).ok()?;
    let compatible = (2..=3).contains(&broadcast_batch)
        && dit_sequence_is_admitted(rows)
        && input.meta.shape().as_slice() == [1, rows, DIT_INPUT_K]
        && weight.meta.shape().as_slice() == [DIT_INPUT_K, DIT_INPUT_N]
        && bias.meta.shape().as_slice() == [DIT_INPUT_N]
        && input.meta.strides()[..] == [rows * DIT_INPUT_K, DIT_INPUT_K, 1]
        && weight.meta.strides()[..] == [DIT_INPUT_N, 1]
        && bias.meta.strides()[..] == [1]
        && input.is_contiguous()
        && weight.is_contiguous()
        && bias.is_contiguous()
        && input.device == weight.device
        && input.device == bias.device
        && binding_is_compatible(
            &input,
            rows * DIT_INPUT_K,
            precision,
            precision.element_bytes() as u64,
        )
        && binding_is_compatible(&weight, DIT_INPUT_K * DIT_INPUT_N, precision, vec4_bytes)
        && binding_is_compatible(&bias, DIT_INPUT_N, precision, vec4_bytes);
    if !compatible {
        return None;
    }
    let hardware = &input.client.properties().hardware;
    if hardware.max_bindings < 4
        || hardware.max_shared_memory_size < SHARED_BYTES
        || hardware.max_units_per_cube < WORKGROUP_X * WORKGROUP_Y
        || hardware.max_cube_dim.0 < WORKGROUP_X
        || hardware.max_cube_dim.1 < WORKGROUP_Y
        || hardware.max_cube_count.0 < u32::try_from(DIT_INPUT_N / TILE_COLUMNS).ok()?
        || hardware.max_cube_count.1 < u32::try_from(rows.div_ceil(TILE_ROWS)).ok()?
    {
        return None;
    }
    let output_bytes = output_elements.checked_mul(precision.element_bytes())?;
    let client = input.client.clone();
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
        input.device.clone(),
        Shape::from([broadcast_batch, rows, DIT_INPUT_N]),
        output_handle,
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DitInputProjectionBroadcastKernel {
                precision,
                rows: u32::try_from(rows).ok()?,
                batch: u32::try_from(broadcast_batch).ok()?,
            },
            CubeDim::new_2d(WORKGROUP_X, WORKGROUP_Y),
        ));
    client.launch(
        task,
        CubeCount::new_2d(
            u32::try_from(DIT_INPUT_N / TILE_COLUMNS).ok()?,
            u32::try_from(rows.div_ceil(TILE_ROWS)).ok()?,
        ),
        KernelArguments::new()
            .with_buffer(input.handle.binding())
            .with_buffer(weight.handle.binding())
            .with_buffer(bias.handle.binding())
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
    fn exact_geometry_and_accounting_are_stable() {
        assert_eq!(EXPAND_N % TILE_COLUMNS, 0);
        assert_eq!(EXPAND_N % LONG_TILE_COLUMNS, 64);
        assert_eq!(EXPAND_K % TILE_K, 0);
        assert_eq!(CONTRACT_N % TILE_COLUMNS, 0);
        assert_eq!(CONTRACT_K % TILE_K, 0);
        assert_eq!(ATTENTION_QKV_GATE_N % TILE_COLUMNS, 0);
        assert_eq!(ATTENTION_QKV_GATE_K % TILE_K, 0);
        assert_eq!(ATTENTION_OUTPUT_N % TILE_COLUMNS, 0);
        assert_eq!(ATTENTION_OUTPUT_K % TILE_K, 0);
        assert_eq!(DURATION_EXPAND_N % TILE_COLUMNS, 0);
        assert_eq!(DURATION_EXPAND_K % TILE_K, 0);
        assert_eq!(DURATION_INPUT_N % TILE_COLUMNS, 0);
        assert_eq!(DURATION_INPUT_K % TILE_K, 0);
        assert_eq!(DIT_INPUT_N % TILE_COLUMNS, 0);
        assert_eq!(DIT_INPUT_K % TILE_K, 0);
        assert_eq!(SHARED_BYTES, 8_192);
        assert_eq!(LONG_SHARED_BYTES, 24_576);
        assert_eq!(WORKGROUP_X * WORKGROUP_Y, 256);
        assert_eq!(LONG_WORKGROUP_X * LONG_WORKGROUP_Y, 256);
        assert_eq!((LONG_WORKGROUP_X, LONG_WORKGROUP_Y), (32, 8));
        assert_eq!(EXPAND_N.div_ceil(LONG_TILE_COLUMNS), 58);
        assert_eq!((EXPAND_N / 2).div_ceil(LONG_TILE_COLUMNS / 2), 58);
        assert_eq!(CONTRACT_N.div_ceil(LONG_TILE_COLUMNS), 10);
        assert_eq!(ATTENTION_QKV_GATE_N.div_ceil(LONG_TILE_COLUMNS), 40);
        assert_eq!(ATTENTION_OUTPUT_N.div_ceil(LONG_TILE_COLUMNS), 10);
        assert_eq!(DURATION_EXPAND_N / TILE_COLUMNS, 32);
        assert_eq!(DURATION_INPUT_N / TILE_COLUMNS, 16);
        assert_eq!(100_usize.div_ceil(TILE_ROWS), 2);
        assert_eq!(200_usize.div_ceil(TILE_ROWS), 4);
        assert_eq!(400_usize.div_ceil(TILE_ROWS), 7);
        assert_eq!(333_usize.div_ceil(TILE_ROWS), 6);
        assert_eq!(666_usize.div_ceil(TILE_ROWS), 11);
        assert_eq!(685_usize.div_ceil(TILE_ROWS), 11);
        assert_eq!(2_055_usize.div_ceil(TILE_ROWS), 33);
        for sequence in [13, 45, 100, 112, 200, 333, 511, 685] {
            assert!(dit_rows_are_admitted(sequence));
            assert!(dit_rows_are_admitted(sequence * 2));
            assert!(dit_rows_are_admitted(sequence * 3));
            assert_eq!(
                dit_sequence_is_admitted(sequence),
                (13..=685).contains(&sequence)
            );
        }
        assert!(!dit_rows_are_admitted(12));
        assert!(!dit_rows_are_admitted(2_056));
        assert!(!dit_sequence_is_admitted(12));
        assert!(dit_sequence_is_admitted(13));
        assert!(dit_sequence_is_admitted(685));
        assert!(!dit_sequence_is_admitted(686));
        assert_eq!(3_usize.div_ceil(TILE_ROWS), 1);
        assert_eq!(12_usize.div_ceil(TILE_ROWS), 1);
        assert_eq!(28_usize.div_ceil(TILE_ROWS), 1);
        assert_eq!(61_usize.div_ceil(TILE_ROWS), 1);
        assert!(duration_rows_are_admitted(1));
        assert!(duration_rows_are_admitted(64));
        assert!(!duration_rows_are_admitted(0));
        assert!(!duration_rows_are_admitted(65));
    }

    #[test]
    fn shader_keeps_k_ascending_and_vec4_weight_output() {
        let shader = include_str!("dit_projection_t64.wgsl");
        assert_eq!(shader.matches("array<vec4<f32>>").count(), 2);
        assert_eq!(shader.matches("var<storage, read_write>").count(), 3);
        assert_eq!(shader.matches("acc_0 = fma").count(), 1);
        assert_eq!(shader.matches("acc_1 = fma").count(), 1);
        assert_eq!(shader.matches("acc_2 = fma").count(), 1);
        assert_eq!(shader.matches("acc_3 = fma").count(), 1);
        assert_eq!(shader.matches("acc_4 = fma").count(), 0);

        let long_shader = include_str!("dit_projection_c128.wgsl");
        assert_eq!(long_shader.matches("array<vec4<f32>>").count(), 2);
        assert_eq!(long_shader.matches("var<storage, read_write>").count(), 3);
        for accumulator in 0..8 {
            assert_eq!(
                long_shader
                    .matches(&format!("acc_{accumulator} = fma"))
                    .count(),
                1
            );
        }

        let fused_expand = include_str!("dit_mlp_expand_swiglu_c128.wgsl");
        assert_eq!(fused_expand.matches("@binding(").count(), 3);
        assert_eq!(fused_expand.matches("var<storage, read_write>").count(), 3);
        assert!(fused_expand.contains("hidden_vec + half * HIDDEN_VECS"));
        assert!(fused_expand.contains("gate / (vec4<f32>(1.0) + exp(-gate)) * value"));
        for accumulator in [
            "gate_0", "gate_1", "gate_2", "gate_3", "value_0", "value_1", "value_2", "value_3",
        ] {
            assert_eq!(
                fused_expand
                    .matches(&format!("{accumulator} = fma"))
                    .count(),
                1
            );
        }

        let warp32 = include_str!("dit_mlp_expand_swiglu_warp32.wgsl");
        assert_eq!(warp32.matches("@binding(").count(), 3);
        assert!(warp32.contains("@workgroup_size(32, {{ workgroup_y }}, 1)"));
        assert!(warp32.contains("weight: array<vec2<f32>>"));
        assert!(warp32.contains("input_tile: array<vec4<f32>, {{ input_tile_vecs }}>"));
        assert!(warp32.contains("weight_tile: array<vec2<f32>, 2048>"));

        let vectorized = include_str!("dit_mlp_expand_swiglu_c128_vec4.wgsl");
        assert_eq!(vectorized.matches("@binding(").count(), 3);
        assert!(vectorized.contains("input: array<vec4<f32>>"));
        assert!(vectorized.contains("const TILE_K: u32 = {{ tile_k }}u"));
        assert!(vectorized.contains("input_tile: array<vec4<f32>, {{ input_tile_vecs }}>"));
        assert!(vectorized.contains("weight_tile: array<vec4<f32>, {{ weight_tile_vecs }}>"));
        for accumulator in [
            "gate_0", "gate_1", "gate_2", "gate_3", "value_0", "value_1", "value_2", "value_3",
        ] {
            assert_eq!(
                vectorized.matches(&format!("{accumulator} = fma")).count(),
                4
            );
        }

        let projection_vectorized = include_str!("dit_projection_c128_vec4.wgsl");
        assert_eq!(projection_vectorized.matches("@binding(").count(), 3);
        assert!(projection_vectorized.contains("input: array<vec4<f32>>"));
        assert!(
            projection_vectorized.contains("input_tile: array<vec4<f32>, {{ input_tile_vecs }}>")
        );
        assert!(
            projection_vectorized.contains("weight_tile: array<vec4<f32>, {{ weight_tile_vecs }}>")
        );
        for accumulator in 0..8 {
            assert_eq!(
                projection_vectorized
                    .matches(&format!("acc_{accumulator} = fma"))
                    .count(),
                4
            );
        }

        let duration_input = include_str!("duration_input_projection_t64.wgsl");
        assert_eq!(duration_input.matches("@binding(").count(), 4);
        assert_eq!(duration_input.matches(" = fma(").count(), 4);
        assert_eq!(duration_input.matches(" + bias_value;").count(), 4);
    }

    #[test]
    fn vectorized_input_shader_matches_scalar_on_exact_shape() {
        #[cfg(feature = "cli")]
        let _ = crate::backend_config::initialize_cli_tracing("warn");
        let device: burn::tensor::Device = WgpuDevice::DefaultDevice.into();
        assert_eq!(device.settings().float_dtype, FloatDType::F32);
        let input = Tensor::<2>::ones([13, EXPAND_K], &device);
        let weight = Tensor::<2>::ones([EXPAND_K, EXPAND_N], &device);
        let scalar = try_dit_mlp_expand_swiglu_c128_wgsl(
            input
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU input"),
            weight
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU weight"),
        )
        .expect("scalar-input C128 route");
        let vectorized = try_dit_mlp_expand_swiglu_c128_vec4_wgsl(
            input
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU input"),
            weight
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU weight"),
        )
        .expect("vector-input C128 route");
        let k16 = try_dit_mlp_expand_swiglu_c128_vec4_k16_wgsl(
            input
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU input"),
            weight
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU weight"),
        )
        .expect("K16 vector-input C128 route");
        let prefetched = try_dit_mlp_expand_swiglu_c128_vec4_k16_prefetch_wgsl(
            input
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU input"),
            weight
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU weight"),
        )
        .expect("prefetched K16 vector-input C128 route");
        let warp32 = try_dit_mlp_expand_swiglu_warp32_wgsl(
            input
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU input"),
            weight
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU weight"),
        )
        .expect("warp32 C128 route");
        let warp32_rows128 = try_dit_mlp_expand_swiglu_warp32_rows128_wgsl(
            Tensor::<2>::ones([13, EXPAND_K], &device)
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU input"),
            Tensor::<2>::ones([EXPAND_K, EXPAND_N], &device)
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU weight"),
        )
        .expect("warp32 rows128 C128 route");
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
        let prefetched = Tensor::<2>::from_primitive::<crate::WgpuRaw>(prefetched)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(scalar, prefetched);
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

    #[test]
    fn vectorized_c128_projection_matches_scalar_on_qkv_shape() {
        #[cfg(feature = "cli")]
        let _ = crate::backend_config::initialize_cli_tracing("warn");
        let device: burn::tensor::Device = WgpuDevice::DefaultDevice.into();
        assert_eq!(device.settings().float_dtype, FloatDType::F32);
        let input = Tensor::<2>::ones([13, ATTENTION_QKV_GATE_K], &device);
        let weight = Tensor::<2>::ones([ATTENTION_QKV_GATE_K, ATTENTION_QKV_GATE_N], &device);
        let scalar = try_dit_attention_qkv_gate_t64_wgsl(
            input
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU input"),
            weight
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU weight"),
        )
        .expect("scalar-input QKV route");
        let vectorized = try_dit_attention_qkv_gate_c128_vec4_wgsl(
            input
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU input"),
            weight
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU weight"),
        )
        .expect("vector-input QKV route");
        let k16 = try_dit_attention_qkv_gate_c128_k16_wgsl(
            input
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU input"),
            weight
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU weight"),
        )
        .expect("K16 QKV route");
        let prefetched = try_dit_attention_qkv_gate_c128_k16_prefetch_wgsl(
            input
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU input"),
            weight
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU weight"),
        )
        .expect("prefetched K16 QKV route");
        let rows128_k16 = try_dit_attention_qkv_gate_c128_rows128_k16_wgsl(
            input
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU input"),
            weight
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU weight"),
        )
        .expect("rows128/K16 QKV route");
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
        let prefetched = Tensor::<2>::from_primitive::<crate::WgpuRaw>(prefetched)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(scalar, prefetched);
        let rows128_k16 = Tensor::<2>::from_primitive::<crate::WgpuRaw>(rows128_k16)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(scalar, rows128_k16);
    }
}
