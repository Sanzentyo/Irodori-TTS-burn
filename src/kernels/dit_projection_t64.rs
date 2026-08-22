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

/// Host-side policy selecting which structurally valid DiT projection shapes
/// may enter the handwritten route.
///
/// The extended envelope is measured on Apple M5/Metal, but is not inferred
/// from an operating system, adapter name, or vendor family: adjacent hardware
/// generations can choose different winners. It is therefore profile-only
/// until an accuracy-approved tuning receipt owns the production selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DitRouteEnvelope {
    ProductionApproved,
    #[cfg(any(feature = "profile", test))]
    ExtendedCandidate,
}

impl DitRouteEnvelope {
    pub(crate) const fn admits_sequence(self, sequence: usize) -> bool {
        let minimum = match self {
            Self::ProductionApproved => 100,
            #[cfg(any(feature = "profile", test))]
            Self::ExtendedCandidate => DIT_MIN_ROWS,
        };
        sequence >= minimum && sequence <= DIT_MAX_SEQUENCE
    }

    pub(crate) const fn admits_full_b3(self) -> bool {
        match self {
            Self::ProductionApproved => false,
            #[cfg(any(feature = "profile", test))]
            Self::ExtendedCandidate => true,
        }
    }

    pub(crate) const fn admits_b3_expand(self) -> bool {
        self.admits_full_b3()
    }

    pub(crate) const fn admits_short_packed_output(self) -> bool {
        self.admits_full_b3()
    }
}

/// Resolve the profile candidate once per process. Production builds never
/// read environment variables and retain the numerically/performance-approved
/// RTX envelope.
pub(crate) fn active_dit_route_envelope() -> DitRouteEnvelope {
    #[cfg(feature = "profile")]
    {
        use std::sync::OnceLock;

        static ENVELOPE: OnceLock<DitRouteEnvelope> = OnceLock::new();
        *ENVELOPE.get_or_init(|| {
            if std::env::var("IRODORI_DIT_ROUTE_ENVELOPE").as_deref() == Ok("extended-candidate") {
                DitRouteEnvelope::ExtendedCandidate
            } else {
                DitRouteEnvelope::ProductionApproved
            }
        })
    }
    #[cfg(not(feature = "profile"))]
    {
        DitRouteEnvelope::ProductionApproved
    }
}

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
}

#[derive(Debug)]
struct DitMlpExpandSwiGluC128Kernel {
    precision: KernelFloatPrecision,
    rows: u32,
}

#[derive(Debug)]
struct DurationInputProjectionT64Kernel {
    precision: KernelFloatPrecision,
    rows: u32,
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
        self.precision
            .source(
                include_str!("dit_projection_c128.wgsl"),
                include_str!("dit_projection_c128_f16.wgsl"),
            )
            .register("rows", self.rows.to_string())
            .register("inner", self.inner.to_string())
            .register("columns", self.columns.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.rows, self.inner, self.columns))
    }
}

impl KernelSource for DitMlpExpandSwiGluC128Kernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("dit_mlp_expand_swiglu_c128.wgsl"),
                include_str!("dit_mlp_expand_swiglu_c128_f16.wgsl"),
            )
            .register("rows", self.rows.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((self.precision, self.rows))
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
    inner: usize,
    columns: usize,
    rows_are_admitted: fn(usize) -> bool,
    use_long_tile: bool,
) -> Option<CubeTensor<WgpuRuntime>> {
    if input.meta.num_dims() != 2 || weight.meta.num_dims() != 2 {
        return None;
    }
    let rows = input.meta.shape()[0];
    let output_elements = rows.checked_mul(columns)?;
    let precision = common_float_precision([input.dtype, weight.dtype])?;
    let vec4_bytes = u64::try_from(4 * precision.element_bytes()).ok()?;
    let tile_k = if use_long_tile { LONG_TILE_K } else { TILE_K };
    let compatible = rows_are_admitted(rows)
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
            precision.element_bytes() as u64,
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
    let shared_bytes = if use_long_tile {
        LONG_SHARED_BYTES
    } else {
        SHARED_BYTES
    };
    let hardware = &input.client.properties().hardware;
    let (workgroup_x, workgroup_y) = if use_long_tile {
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
            },
            CubeDim::new_2d(LONG_WORKGROUP_X, LONG_WORKGROUP_Y),
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
            u32::try_from(rows.div_ceil(TILE_ROWS)).ok()?,
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

/// Latent lengths admitted by the selected host-side projection policy.
///
/// Generated lengths come from the duration predictor and are not restricted to
/// the handful of oracle lengths used during kernel tuning. The shader already
/// guards its final partial row tile, so a profile campaign can exercise the
/// extended candidate without widening the production route on every device.
pub fn dit_sequence_is_admitted(sequence: usize) -> bool {
    active_dit_route_envelope().admits_sequence(sequence)
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
        EXPAND_K,
        EXPAND_N,
        dit_rows_are_admitted,
        true,
    )
}

/// Launch the released expansion and consume its paired gate/value columns in
/// the same dispatch. The output is the dense `[rows, hidden]` SwiGLU tensor;
/// no `[rows, 2 * hidden]` projection is materialised.
pub fn try_dit_mlp_expand_swiglu_c128_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    if input.meta.num_dims() != 2 || weight.meta.num_dims() != 2 {
        return None;
    }
    let rows = input.meta.shape()[0];
    let hidden = EXPAND_N / 2;
    let output_elements = rows.checked_mul(hidden)?;
    let precision = common_float_precision([input.dtype, weight.dtype])?;
    let vec4_bytes = u64::try_from(4 * precision.element_bytes()).ok()?;
    let compatible = dit_rows_are_admitted(rows)
        && EXPAND_K.is_multiple_of(LONG_TILE_K)
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
            precision.element_bytes() as u64,
        )
        && binding_is_compatible(&weight, EXPAND_K * EXPAND_N, precision, vec4_bytes);
    if !compatible {
        return None;
    }

    let output_tile_columns = LONG_TILE_COLUMNS / 2;
    let hardware = &input.client.properties().hardware;
    if hardware.max_bindings < REQUIRED_BINDINGS
        || hardware.max_shared_memory_size < LONG_SHARED_BYTES
        || hardware.max_units_per_cube < WORKGROUP_X * WORKGROUP_Y
        || hardware.max_cube_dim.0 < WORKGROUP_X
        || hardware.max_cube_dim.1 < WORKGROUP_Y
        || hardware.max_cube_count.0 < u32::try_from(hidden.div_ceil(output_tile_columns)).ok()?
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
        Shape::from([rows, hidden]),
        output_handle,
        precision.dtype(),
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            DitMlpExpandSwiGluC128Kernel {
                precision,
                rows: u32::try_from(rows).ok()?,
            },
            CubeDim::new_2d(WORKGROUP_X, WORKGROUP_Y),
        ));
    client.launch(
        task,
        CubeCount::new_2d(
            u32::try_from(hidden.div_ceil(output_tile_columns)).ok()?,
            u32::try_from(rows.div_ceil(TILE_ROWS)).ok()?,
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
        CONTRACT_K,
        CONTRACT_N,
        dit_rows_are_admitted,
        true,
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
        ATTENTION_QKV_GATE_K,
        ATTENTION_QKV_GATE_N,
        dit_rows_are_admitted,
        true,
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
        ATTENTION_OUTPUT_K,
        ATTENTION_OUTPUT_N,
        dit_rows_are_admitted,
        true,
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
        DURATION_EXPAND_K,
        DURATION_EXPAND_N,
        duration_rows_are_admitted,
        false,
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

#[cfg(test)]
mod tests {
    use super::*;

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
                DitRouteEnvelope::ExtendedCandidate.admits_sequence(sequence),
                (13..=685).contains(&sequence)
            );
        }
        assert!(!dit_rows_are_admitted(12));
        assert!(!dit_rows_are_admitted(2_056));
        assert!(!DitRouteEnvelope::ExtendedCandidate.admits_sequence(12));
        assert!(!DitRouteEnvelope::ExtendedCandidate.admits_sequence(686));
        assert!(!DitRouteEnvelope::ProductionApproved.admits_sequence(99));
        assert!(DitRouteEnvelope::ProductionApproved.admits_sequence(100));
        assert!(DitRouteEnvelope::ProductionApproved.admits_sequence(685));
        assert!(!DitRouteEnvelope::ProductionApproved.admits_sequence(686));
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

        let duration_input = include_str!("duration_input_projection_t64.wgsl");
        assert_eq!(duration_input.matches("@binding(").count(), 4);
        assert_eq!(duration_input.matches(" = fma(").count(), 4);
        assert_eq!(duration_input.matches(" + bias_value;").count(), 4);
    }
}
