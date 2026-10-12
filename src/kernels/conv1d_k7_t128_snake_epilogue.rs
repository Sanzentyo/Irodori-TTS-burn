//! Production T128 Conv1d + Snake epilogue.
//!
//! The exact twelve-shape RTX 3060 Ti run measured 38.707 ms for materialised
//! T128 + Snake versus 37.421 ms fused (1.034x, 1.286 ms saved), with every
//! output bit-exact. The full five-binding contract remains fail-closed so the
//! prior materialised path is always available.

use super::{
    conv1d_k7_t128::{
        Conv1dK7T128Tile, LaunchGeometry, binding_is_compatible,
        conv1d_k7_t128_contract_is_compatible,
    },
    conv1d_k7_tiled::Conv1dK7Dilation,
};
use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::Shape;
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

use super::precision::{KernelFloatPrecision, common_float_precision};

const BATCH: usize = 1;
const REQUIRED_BINDINGS: u32 = 5;
const LOCAL_TIME_LANES: u32 = 16;
const LOCAL_CHANNEL_LANES: u32 = 16;

#[derive(Debug)]
struct Conv1dK7T128SnakeEpilogueKernel {
    precision: KernelFloatPrecision,
    tile: Conv1dK7T128Tile,
    channels: u32,
    length: u32,
    dilation: u32,
    padding: u32,
    input_span: u32,
    input_tile_size: u32,
    weight_tile_size: u32,
}

impl KernelSource for Conv1dK7T128SnakeEpilogueKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("conv1d_k7_t128_snake_epilogue.wgsl"),
                include_str!("conv1d_k7_t128_snake_epilogue_f16.wgsl"),
            )
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
            self.precision,
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

/// Validate the production fused launch without allocating or dispatching.
///
/// The first four buffers must satisfy the complete production T128 contract.
/// Alpha additionally must be contiguous f32 `[1, C, 1]` on the same device,
/// and the device must expose at least five storage bindings.
pub fn conv1d_k7_t128_snake_epilogue_contract_is_compatible(
    input: &CubeTensor<WgpuRuntime>,
    weight: &CubeTensor<WgpuRuntime>,
    bias: &CubeTensor<WgpuRuntime>,
    alpha: &CubeTensor<WgpuRuntime>,
    dilation: Conv1dK7Dilation,
    tile: Conv1dK7T128Tile,
) -> bool {
    if !conv1d_k7_t128_contract_is_compatible(input, weight, bias, dilation, tile)
        || alpha.meta.num_dims() != 3
    {
        return false;
    }

    let input_shape = input.meta.shape();
    let alpha_shape = alpha.meta.shape();
    let Some(precision) =
        common_float_precision([input.dtype, weight.dtype, bias.dtype, alpha.dtype])
    else {
        return false;
    };
    [alpha_shape[0], alpha_shape[1], alpha_shape[2]] == [1, input_shape[1], 1]
        && alpha.device == input.device
        && alpha.is_contiguous()
        && binding_is_compatible(
            alpha,
            input_shape[1],
            precision,
            precision.element_bytes() as u64,
        )
        && input.client.properties().hardware.max_bindings >= REQUIRED_BINDINGS
}

/// Apply the exact standalone Snake scalar expression in the production T128
/// output-store epilogue.
///
/// Call [`conv1d_k7_t128_snake_epilogue_contract_is_compatible`] first. This
/// production launcher asserts its full B1/f32/NCL/OIK/alpha/device contract.
///
/// # Panics
///
/// Panics on any incompatible shape, dtype, layout, device, indexing, shared
/// memory, workgroup, dispatch, or five-binding contract.
pub fn conv1d_k7_same_t128_snake_epilogue_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    alpha: CubeTensor<WgpuRuntime>,
    dilation: Conv1dK7Dilation,
    tile: Conv1dK7T128Tile,
) -> CubeTensor<WgpuRuntime> {
    let precision = common_float_precision([input.dtype, weight.dtype, bias.dtype, alpha.dtype])
        .expect("T128 Conv1d + Snake tensors must share f32 or f16 dtype");
    assert!(
        conv1d_k7_t128_snake_epilogue_contract_is_compatible(
            &input, &weight, &bias, &alpha, dilation, tile,
        ),
        "{} + Snake requires compatible B1/f32/contiguous NCL+OIK+alpha shape, device, and five-binding resources",
        tile.label(),
    );

    let input_shape = input.meta.shape();
    let [batch, channels, length] = [input_shape[0], input_shape[1], input_shape[2]];
    let geometry = LaunchGeometry::new(channels, length, dilation, tile)
        .expect("validated T128 + Snake geometry must remain representable");
    let output_elements = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(length))
        .expect("validated output element count must not overflow");
    let output_bytes = output_elements
        .checked_mul(precision.element_bytes())
        .expect("validated output byte count must not overflow");
    let client = input.client.clone();
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([batch, channels, length]),
        client.empty(output_bytes),
        precision.dtype(),
    );

    let dilation_value = dilation.value();
    let kernel = Conv1dK7T128SnakeEpilogueKernel {
        precision,
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
            CubeDim::new_2d(LOCAL_TIME_LANES, LOCAL_CHANNEL_LANES),
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
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalized_main_prefix(shader: &str) -> String {
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
    fn fused_shader_preserves_the_complete_production_t128_convolution_body() {
        let production = include_str!("conv1d_k7_t128.wgsl");
        let fused = include_str!("conv1d_k7_t128_snake_epilogue.wgsl");
        assert_eq!(
            normalized_main_prefix(fused),
            normalized_main_prefix(production),
            "fused convolution body drifted from production T128"
        );
        assert_eq!(
            normalized_main_prefix(fused).matches(" = fma(").count(),
            normalized_main_prefix(production)
                .matches(" = fma(")
                .count()
        );
        assert_eq!(fused.matches("// tap ").count(), 7);
    }

    #[test]
    fn fused_shader_has_five_uniform_read_write_bindings_and_exact_snake_source() {
        let fused = include_str!("conv1d_k7_t128_snake_epilogue.wgsl");
        let standalone = include_str!("snake.wgsl");
        let storage_bindings = fused
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with("@group(0)") && line.contains("var<storage"))
            .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
            .collect::<Vec<_>>();
        assert_eq!(REQUIRED_BINDINGS, 5);
        assert_eq!(storage_bindings.len(), REQUIRED_BINDINGS as usize);
        assert_eq!(
            storage_bindings,
            [
                "@group(0) @binding(0) var<storage, read_write> input_buf: array<f32>;",
                "@group(0) @binding(1) var<storage, read_write> weight_buf: array<f32>;",
                "@group(0) @binding(2) var<storage, read_write> bias_buf: array<f32>;",
                "@group(0) @binding(3) var<storage, read_write> output_buf: array<f32>;",
                "@group(0) @binding(4) var<storage, read_write> alpha_buf: array<f32>;",
            ]
        );
        let fused_snake_lines = [
            "let a = alpha_buf[output_channel];",
            "let sine = sin(a * x);",
            "return x + (sine * sine) / (a + 1e-9);",
        ];
        let standalone_snake_lines = [
            "let a = alpha[channel];",
            "let sine = sin(a * x);",
            "output[index] = x + (sine * sine) / (a + 1e-9);",
        ];
        for line in fused_snake_lines {
            assert!(fused.contains(line), "missing exact Snake line: {line}");
        }
        for line in standalone_snake_lines {
            assert!(
                standalone.contains(line),
                "standalone Snake operation order drifted: {line}"
            );
        }
        for (shader, lines) in [
            (fused, fused_snake_lines.as_slice()),
            (standalone, standalone_snake_lines.as_slice()),
        ] {
            let positions = lines
                .iter()
                .map(|line| shader.find(line).expect("validated Snake line must exist"))
                .collect::<Vec<_>>();
            assert!(
                positions.windows(2).all(|pair| pair[0] < pair[1]),
                "Snake scalar operation order drifted"
            );
        }
        for component in ['x', 'y', 'z', 'w'] {
            assert!(fused.contains(&format!(
                "snake_epilogue(value.{component}, output_channel)"
            )));
        }
    }

    #[test]
    fn geometry_and_tile_identity_are_reused_from_production() {
        for (channels, length, dilation) in [
            (768, 600, Conv1dK7Dilation::One),
            (384, 6_000, Conv1dK7Dilation::Nine),
            (192, 48_000, Conv1dK7Dilation::Three),
            (96, 96_000, Conv1dK7Dilation::Nine),
        ] {
            let tile = crate::kernels::conv1d_k7_t128::production_tile_for_shape(
                channels, length, dilation,
            )
            .expect("released shape must select a production T128 tile");
            let geometry = LaunchGeometry::new(channels, length, dilation, tile)
                .expect("released production geometry must remain valid");
            assert_eq!(geometry.shared_bytes, tile.shared_memory_bytes(dilation));
            assert_eq!(crate::kernels::conv1d_k7_t128::WORKGROUP_SIZE, 256);
        }
    }
}
