//! Production T256/O32 Conv1d + Snake kernel with one `vec4<f32>` output store.
//!
//! The rotating isolated A/B was bit-exact on all nine measured T256 routes and
//! reduced their median sum from 28.286 ms to 27.640 ms (0.646 ms, 1.023x).
//! Production conservatively selects the eight winning routes. C768/L600/d9
//! remains on scalar-store T256 because vec4 measured 1924.886 us versus
//! 1922.423 us.

use super::{
    conv1d_k7_t256_snake_epilogue::{
        Conv1dK7T256Tile, LaunchGeometry, conv1d_k7_t256_snake_epilogue_contract_is_compatible,
        production_tile_for_shape as scalar_production_tile_for_shape,
    },
    conv1d_k7_tiled::Conv1dK7Dilation,
};
use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::{DType, Shape};
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

const BATCH: usize = 1;
const LOCAL_TIME_LANES: usize = 16;
const LOCAL_CHANNEL_LANES: usize = 16;
const VEC4_ELEMENTS: usize = 4;
const VEC4_BYTES: u64 = 16;

#[derive(Debug)]
struct Conv1dK7T256SnakeVec4StoreKernel {
    tile: Conv1dK7T256Tile,
    channels: u32,
    length: u32,
    dilation: u32,
    padding: u32,
    input_span: u32,
    input_tile_size: u32,
    weight_tile_size: u32,
}

impl KernelSource for Conv1dK7T256SnakeVec4StoreKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("conv1d_k7_t256_snake_vec4_store.wgsl"))
            .register("channels", self.channels.to_string())
            .register("length", self.length.to_string())
            .register("dilation", self.dilation.to_string())
            .register("padding", self.padding.to_string())
            .register(
                "input_channel_tile",
                self.tile.input_channel_tile().to_string(),
            )
            .register("input_span", self.input_span.to_string())
            .register("input_tile_size", self.input_tile_size.to_string())
            .register("weight_tile_size", self.weight_tile_size.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.tile,
            self.channels,
            self.length,
            self.dilation,
            self.padding,
            self.input_span,
            self.input_tile_size,
            self.weight_tile_size,
        ))
    }
}

fn vec4_output_layout_is_compatible(
    length: usize,
    output_bytes: u64,
    allocator_alignment: u64,
    logical_offset: u64,
) -> bool {
    length.is_multiple_of(VEC4_ELEMENTS)
        && output_bytes.is_multiple_of(VEC4_BYTES)
        && allocator_alignment >= VEC4_BYTES
        && allocator_alignment.is_multiple_of(VEC4_BYTES)
        && logical_offset.is_multiple_of(VEC4_BYTES)
}

/// Select one of the eight measured production wins for vec4 output stores.
///
/// The C768/d9 route intentionally returns `None` at every length because its
/// isolated released-shape median regressed by 2.463 us.
pub const fn production_tile_for_shape(
    channels: usize,
    length: usize,
    dilation: Conv1dK7Dilation,
) -> Option<Conv1dK7T256Tile> {
    if matches!((channels, dilation), (768, Conv1dK7Dilation::Nine)) {
        None
    } else {
        scalar_production_tile_for_shape(channels, length, dilation)
    }
}

/// Validate the production vec4-store kernel without allocating or dispatching.
///
/// The measured scalar-T256 selector is part of the physical contract: no
/// unmeasured T256 shape is admitted. CubeCL aligns every pool slice to
/// `client.properties().memory.alignment`; the launcher separately verifies the
/// logical offset of its newly allocated output handle.
pub fn conv1d_k7_t256_snake_vec4_store_contract_is_compatible(
    input: &CubeTensor<WgpuRuntime>,
    weight: &CubeTensor<WgpuRuntime>,
    bias: &CubeTensor<WgpuRuntime>,
    alpha: &CubeTensor<WgpuRuntime>,
    dilation: Conv1dK7Dilation,
    tile: Conv1dK7T256Tile,
) -> bool {
    if !conv1d_k7_t256_snake_epilogue_contract_is_compatible(
        input, weight, bias, alpha, dilation, tile,
    ) {
        return false;
    }

    let shape = input.meta.shape();
    let [channels, length] = [shape[1], shape[2]];
    let Some(output_elements) = BATCH
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(length))
    else {
        return false;
    };
    let Some(output_bytes) = output_elements.checked_mul(core::mem::size_of::<f32>()) else {
        return false;
    };
    let Ok(output_bytes) = u64::try_from(output_bytes) else {
        return false;
    };

    scalar_production_tile_for_shape(channels, length, dilation) == Some(tile)
        && vec4_output_layout_is_compatible(
            length,
            output_bytes,
            input.client.properties().memory.alignment,
            0,
        )
}

/// Try to launch the vec4-store kernel for a measured T256 route.
///
/// # Returns
///
/// Returns `None` before dispatch when the exact
/// B1/F32/contiguous-NCL+OIK+alpha/device/resource contract or either 16-byte
/// allocator/output-binding check is absent. Callers can then retain the scalar
/// T256 launcher without losing its input handles.
pub fn try_conv1d_k7_same_t256_snake_vec4_store_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
    dilation: Conv1dK7Dilation,
    tile: Conv1dK7T256Tile,
) -> Option<CubeTensor<WgpuRuntime>> {
    if !conv1d_k7_t256_snake_vec4_store_contract_is_compatible(
        &input, &weight, &bias, &alpha, dilation, tile,
    ) {
        return None;
    }

    let input_shape = input.meta.shape();
    let [batch, channels, length] = [input_shape[0], input_shape[1], input_shape[2]];
    let geometry = LaunchGeometry::new(channels, length, dilation, tile)
        .expect("validated T256 vec4-store geometry must remain representable");
    let output_elements = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(length))
        .expect("validated output element count must not overflow");
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .expect("validated output byte count must not overflow");
    let client = input.client.clone();
    let output_handle = client.empty(output_bytes);
    let logical_offset = output_handle.offset_start.unwrap_or(0);
    if !vec4_output_layout_is_compatible(
        length,
        output_handle.size_in_used(),
        client.properties().memory.alignment,
        logical_offset,
    ) {
        return None;
    }
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([batch, channels, length]),
        output_handle,
        DType::F32,
    );

    let dilation_value = dilation.value();
    let kernel = Conv1dK7T256SnakeVec4StoreKernel {
        tile,
        channels: u32::try_from(channels).expect("validated C must fit u32"),
        length: u32::try_from(length).expect("validated L must fit u32"),
        dilation: u32::try_from(dilation_value).expect("validated dilation must fit u32"),
        padding: u32::try_from(3 * dilation_value).expect("validated padding must fit u32"),
        input_span: u32::try_from(geometry.input_span).expect("validated input span must fit u32"),
        input_tile_size: u32::try_from(geometry.input_tile_size)
            .expect("validated input tile size must fit u32"),
        weight_tile_size: u32::try_from(geometry.weight_tile_size)
            .expect("validated weight tile size must fit u32"),
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            kernel,
            CubeDim::new_2d(LOCAL_TIME_LANES as u32, LOCAL_CHANNEL_LANES as u32),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(input.handle.binding())
        .with_buffer(weight.handle.binding())
        .with_buffer(bias.handle.binding())
        .with_buffer(output.handle.clone().binding())
        .with_buffer(alpha.handle.binding());
    client.launch(
        task,
        CubeCount::new_3d(
            geometry.time_tiles,
            geometry.output_channel_tiles,
            BATCH as u32,
        ),
        bindings,
    );
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalized_convolution_body(shader: &str) -> String {
        shader
            .split_once("fn main(")
            .expect("shader must define main")
            .1
            .split_once("    let output_base_0")
            .expect("shader must compute output_base_0")
            .0
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn convolution_body_is_source_identical_to_production() {
        let production = include_str!("conv1d_k7_t256_snake_epilogue.wgsl");
        let vec4_store = include_str!("conv1d_k7_t256_snake_vec4_store.wgsl");
        assert_eq!(
            normalized_convolution_body(vec4_store),
            normalized_convolution_body(production),
        );
    }

    #[test]
    fn shader_changes_only_the_guarded_output_representation() {
        let shader = include_str!("conv1d_k7_t256_snake_vec4_store.wgsl");
        assert_eq!(shader.matches("@group(0) @binding(").count(), 5);
        assert!(shader.contains("output_buf: array<vec4<f32>>;"));
        assert_eq!(shader.matches("output_buf[output_vec_index] =").count(), 1);
        assert_eq!(shader.matches("snake_epilogue(value.").count(), 4);
        assert!(!shader.contains("output_buf[output_base + time"));
        assert!(shader.contains("accumulator_00 = fma("));
    }

    #[test]
    fn vec4_output_layout_rejects_every_misalignment() {
        assert!(vec4_output_layout_is_compatible(600, 9_216, 256, 0));
        assert!(!vec4_output_layout_is_compatible(601, 9_216, 256, 0));
        assert!(!vec4_output_layout_is_compatible(600, 9_220, 256, 0));
        assert!(!vec4_output_layout_is_compatible(600, 9_216, 8, 0));
        assert!(!vec4_output_layout_is_compatible(600, 9_216, 24, 0));
        assert!(!vec4_output_layout_is_compatible(600, 9_216, 256, 4));
    }

    #[test]
    fn all_nine_measured_routes_have_vec4_compatible_lengths() {
        let accepted = [
            (768, 600, Conv1dK7Dilation::One, Conv1dK7T256Tile::Cin16),
            (768, 600, Conv1dK7Dilation::Nine, Conv1dK7T256Tile::Cin16),
            (384, 6_000, Conv1dK7Dilation::One, Conv1dK7T256Tile::Cin16),
            (192, 48_000, Conv1dK7Dilation::One, Conv1dK7T256Tile::Cin16),
            (
                192,
                48_000,
                Conv1dK7Dilation::Three,
                Conv1dK7T256Tile::Cin16,
            ),
            (192, 48_000, Conv1dK7Dilation::Nine, Conv1dK7T256Tile::Cin8),
            (96, 96_000, Conv1dK7Dilation::One, Conv1dK7T256Tile::Cin16),
            (96, 96_000, Conv1dK7Dilation::Three, Conv1dK7T256Tile::Cin16),
            (96, 96_000, Conv1dK7Dilation::Nine, Conv1dK7T256Tile::Cin8),
        ];
        for (channels, length, dilation, tile) in accepted {
            assert_eq!(
                scalar_production_tile_for_shape(channels, length, dilation),
                Some(tile)
            );
            assert!(length.is_multiple_of(VEC4_ELEMENTS));
            assert!(
                (channels * length * core::mem::size_of::<f32>()).is_multiple_of(
                    usize::try_from(VEC4_BYTES).expect("vec4 byte width fits usize")
                )
            );
        }
    }

    #[test]
    fn production_selector_excludes_only_the_measured_c768_d9_loss() {
        let cases = [
            (
                768,
                600,
                Conv1dK7Dilation::One,
                Some(Conv1dK7T256Tile::Cin16),
            ),
            (768, 600, Conv1dK7Dilation::Nine, None),
            (
                384,
                6_000,
                Conv1dK7Dilation::One,
                Some(Conv1dK7T256Tile::Cin16),
            ),
            (
                192,
                48_000,
                Conv1dK7Dilation::One,
                Some(Conv1dK7T256Tile::Cin16),
            ),
            (
                192,
                48_000,
                Conv1dK7Dilation::Three,
                Some(Conv1dK7T256Tile::Cin16),
            ),
            (
                192,
                48_000,
                Conv1dK7Dilation::Nine,
                Some(Conv1dK7T256Tile::Cin8),
            ),
            (
                96,
                96_000,
                Conv1dK7Dilation::One,
                Some(Conv1dK7T256Tile::Cin16),
            ),
            (
                96,
                96_000,
                Conv1dK7Dilation::Three,
                Some(Conv1dK7T256Tile::Cin16),
            ),
            (
                96,
                96_000,
                Conv1dK7Dilation::Nine,
                Some(Conv1dK7T256Tile::Cin8),
            ),
        ];
        for (channels, length, dilation, expected) in cases {
            assert_eq!(
                production_tile_for_shape(channels, length, dilation),
                expected
            );
        }
    }

    #[test]
    fn one_guard_covers_all_four_tail_components() {
        for length in [600usize, 6_000, 48_000, 96_000] {
            let padded_length = length.div_ceil(256) * 256;
            for time_base in (0..padded_length).step_by(256) {
                for local_time in (0..64).step_by(4) {
                    for time_offset in [0usize, 64, 128, 192] {
                        let time = time_base + local_time + time_offset;
                        assert!(time >= length || time + 3 < length);
                    }
                }
            }
        }
    }
}
