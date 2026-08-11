//! Isolated tile-size candidates for the production DACVAE k=7 Conv1d.
//!
//! This module deliberately does not replace [`super::conv1d_k7_tiled`]. It
//! exists so the exact production shapes can decide tile selection from GPU
//! measurements rather than intuition. All candidates preserve the native
//! NCHW/OIK layouts and the input-channel-then-kernel f32 accumulation order.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime, into_contiguous,
};
use burn::tensor::{DType, Shape};
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

const BATCH: usize = 1;
const KERNEL_SIZE: usize = 7;
const LOCAL_TIME_LANES: usize = 16;
const INPUT_CHANNEL_TILE: usize = 16;

/// A deliberately small tile matrix for the RTX 3060 Ti measurements.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Conv1dK7TileCandidate {
    /// 32 time positions by 16 output channels, 128 invocations, four f32
    /// accumulators per invocation.
    Time32Output16,
    /// 32 time positions by 32 output channels, 256 invocations, four f32
    /// accumulators per invocation.
    Time32Output32,
    /// 64 time positions by 32 output channels, 256 invocations, eight f32
    /// accumulators per invocation.
    Time64Output32,
}

impl Conv1dK7TileCandidate {
    /// Stable short label used by the isolated benchmark.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Time32Output16 => "t32-o16-wg128",
            Self::Time32Output32 => "t32-o32-wg256",
            Self::Time64Output32 => "t64-o32-wg256",
        }
    }

    /// Number of time positions produced by one workgroup.
    pub const fn time_tile(self) -> usize {
        match self {
            Self::Time32Output16 | Self::Time32Output32 => 32,
            Self::Time64Output32 => 64,
        }
    }

    /// Number of output channels produced by one workgroup.
    pub const fn output_channel_tile(self) -> usize {
        match self {
            Self::Time32Output16 => 16,
            Self::Time32Output32 | Self::Time64Output32 => 32,
        }
    }

    /// Workgroup's output-channel dimension.
    pub const fn local_channel_lanes(self) -> usize {
        self.output_channel_tile() / 2
    }

    /// Number of f32 output accumulators owned by one invocation.
    pub const fn accumulators_per_invocation(self) -> usize {
        let time_repeats = self.time_tile() / LOCAL_TIME_LANES;
        time_repeats * 2
    }

    /// Workgroup storage required by this tile for a given supported dilation.
    ///
    /// # Panics
    ///
    /// Panics unless `dilation` is 1, 3, or 9.
    pub fn shared_memory_bytes(self, dilation: usize) -> usize {
        validate_dilation(dilation);
        let input_span = self.time_tile() + 6 * dilation;
        let input_elements = INPUT_CHANNEL_TILE * input_span;
        let weight_elements = self.output_channel_tile() * INPUT_CHANNEL_TILE * KERNEL_SIZE;
        (input_elements + weight_elements) * core::mem::size_of::<f32>()
    }

    const fn shader(self) -> &'static str {
        match self {
            Self::Time32Output16 | Self::Time32Output32 => {
                include_str!("conv1d_k7_tile_candidate_t32.wgsl")
            }
            Self::Time64Output32 => include_str!("conv1d_k7_tile_candidate_t64.wgsl"),
        }
    }
}

#[derive(Debug)]
struct Conv1dK7TileCandidateKernel {
    candidate: Conv1dK7TileCandidate,
    channels: u32,
    length: u32,
    dilation: u32,
    padding: u32,
    input_span: u32,
    input_tile_size: u32,
    weight_tile_size: u32,
    workgroup_size: u32,
}

impl KernelSource for Conv1dK7TileCandidateKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(self.candidate.shader())
            .register("channels", self.channels.to_string())
            .register("length", self.length.to_string())
            .register("dilation", self.dilation.to_string())
            .register("padding", self.padding.to_string())
            .register("input_span", self.input_span.to_string())
            .register("input_tile_size", self.input_tile_size.to_string())
            .register("weight_tile_size", self.weight_tile_size.to_string())
            .register("workgroup_size", self.workgroup_size.to_string())
            .register(
                "local_channel_lanes",
                self.candidate.local_channel_lanes().to_string(),
            )
            .register(
                "output_channel_tile",
                self.candidate.output_channel_tile().to_string(),
            )
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.candidate,
            self.channels,
            self.length,
            self.dilation,
            self.padding,
            self.input_span,
            self.input_tile_size,
            self.weight_tile_size,
            self.workgroup_size,
        ))
    }
}

fn validate_dilation(dilation: usize) {
    assert!(
        matches!(dilation, 1 | 3 | 9),
        "DACVAE k=7 dilation must be one of 1, 3, or 9, got {dilation}"
    );
}

/// Compute an f32 same-length Conv1d using an isolated candidate tile.
///
/// Required shapes and physical layouts are input `[1, C, L]`, weight
/// `[C, C, 7]` in contiguous OIK order, and bias `[C]`. `C` must be divisible
/// by the candidate's output-channel tile. Padding is `3 * dilation`.
///
/// # Panics
///
/// Panics for an incompatible dtype, rank, shape, device, parameter layout,
/// unsupported dilation/channel count, insufficient device limits, or an
/// arithmetic/dispatch/index calculation that cannot be represented safely.
pub fn conv1d_k7_same_tile_candidate_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    dilation: usize,
    candidate: Conv1dK7TileCandidate,
) -> CubeTensor<WgpuRuntime> {
    validate_dilation(dilation);
    for (name, tensor) in [("input", &input), ("weight", &weight), ("bias", &bias)] {
        assert_eq!(
            tensor.dtype,
            DType::F32,
            "candidate k=7 Conv1d only supports f32 {name}"
        );
        input.assert_is_on_same_device(tensor);
    }

    assert_eq!(input.meta.num_dims(), 3, "input must be rank 3 [1, C, L]");
    assert_eq!(weight.meta.num_dims(), 3, "weight must be rank 3 [C, C, 7]");
    assert_eq!(bias.meta.num_dims(), 1, "bias must be rank 1 [C]");

    let input_shape = input.meta.shape();
    let batch = input_shape[0];
    let channels = input_shape[1];
    let length = input_shape[2];
    assert_eq!(batch, BATCH, "candidate k=7 Conv1d is specialised for B=1");
    assert!(channels > 0, "candidate k=7 Conv1d requires C > 0");
    assert!(length > 0, "candidate k=7 Conv1d requires L > 0");
    assert!(
        channels.is_multiple_of(candidate.output_channel_tile()),
        "candidate {} requires C to be a multiple of {}, got {channels}",
        candidate.label(),
        candidate.output_channel_tile(),
    );

    let weight_shape = weight.meta.shape();
    assert_eq!(
        [weight_shape[0], weight_shape[1], weight_shape[2]],
        [channels, channels, KERNEL_SIZE],
        "weight must have shape [C, C, 7]"
    );
    assert_eq!(bias.meta.shape()[0], channels, "bias must have shape [C]");
    assert!(weight.is_contiguous(), "weight must be contiguous OIK");
    assert!(bias.is_contiguous(), "bias must be contiguous");

    let input = into_contiguous(input);
    let padding = 3usize
        .checked_mul(dilation)
        .expect("k=7 Conv1d padding overflow");
    let input_span = candidate
        .time_tile()
        .checked_add(6usize.checked_mul(dilation).expect("input halo overflow"))
        .expect("input tile span overflow");
    let input_tile_size = INPUT_CHANNEL_TILE
        .checked_mul(input_span)
        .expect("input tile element count overflow");
    let weight_tile_size = candidate
        .output_channel_tile()
        .checked_mul(INPUT_CHANNEL_TILE)
        .and_then(|value| value.checked_mul(KERNEL_SIZE))
        .expect("weight tile element count overflow");
    let shared_bytes = input_tile_size
        .checked_add(weight_tile_size)
        .and_then(|value| value.checked_mul(core::mem::size_of::<f32>()))
        .expect("shared-memory byte count overflow");
    assert_eq!(
        shared_bytes,
        candidate.shared_memory_bytes(dilation),
        "candidate shared-memory accounting diverged"
    );

    let input_elements = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(length))
        .expect("input/output element count overflow");
    let weight_elements = channels
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(KERNEL_SIZE))
        .expect("weight element count overflow");
    let time_tiles = length.div_ceil(candidate.time_tile());
    let output_channel_tiles = channels / candidate.output_channel_tile();
    let final_time_base = time_tiles
        .checked_sub(1)
        .and_then(|value| value.checked_mul(candidate.time_tile()))
        .expect("time tile base overflow");
    let largest_staged_time = final_time_base
        .checked_add(input_span - 1)
        .expect("staged time index overflow");

    for (name, value) in [
        ("C", channels),
        ("L", length),
        ("input/output elements", input_elements),
        ("weight elements", weight_elements),
        ("time tiles", time_tiles),
        ("output-channel tiles", output_channel_tiles),
        ("input span", input_span),
        ("input tile elements", input_tile_size),
        ("weight tile elements", weight_tile_size),
    ] {
        assert!(
            u32::try_from(value).is_ok(),
            "{name}={value} exceeds WGSL/WGPU u32 indexing"
        );
    }
    assert!(
        i32::try_from(largest_staged_time).is_ok(),
        "largest staged time index {largest_staged_time} exceeds WGSL i32 indexing"
    );

    let local_channel_lanes = candidate.local_channel_lanes();
    let workgroup_size = LOCAL_TIME_LANES
        .checked_mul(local_channel_lanes)
        .expect("workgroup size overflow");
    let client = input.client.clone();
    let hardware = &client.properties().hardware;
    assert!(
        hardware.max_bindings >= 4,
        "candidate k=7 Conv1d requires four storage bindings, device supports {}",
        hardware.max_bindings
    );
    assert!(
        hardware.max_shared_memory_size >= shared_bytes,
        "candidate {} requires {shared_bytes} shared bytes, device supports {}",
        candidate.label(),
        hardware.max_shared_memory_size
    );
    assert!(
        hardware.max_units_per_cube >= workgroup_size as u32,
        "candidate {} requires {workgroup_size} invocations, device supports {}",
        candidate.label(),
        hardware.max_units_per_cube
    );
    assert!(
        hardware.max_cube_dim.0 >= LOCAL_TIME_LANES as u32
            && hardware.max_cube_dim.1 >= local_channel_lanes as u32,
        "candidate {} requires workgroup dimensions ({LOCAL_TIME_LANES}, {local_channel_lanes}, 1), device supports {:?}",
        candidate.label(),
        hardware.max_cube_dim
    );
    assert!(
        hardware.max_cube_count.0 >= time_tiles as u32
            && hardware.max_cube_count.1 >= output_channel_tiles as u32
            && hardware.max_cube_count.2 >= BATCH as u32,
        "candidate {} dispatch ({time_tiles}, {output_channel_tiles}, {BATCH}) exceeds device limits {:?}",
        candidate.label(),
        hardware.max_cube_count
    );

    let output_bytes = input_elements
        .checked_mul(core::mem::size_of::<f32>())
        .expect("output byte count overflow");
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([batch, channels, length]),
        client.empty(output_bytes),
        DType::F32,
    );

    let kernel = Conv1dK7TileCandidateKernel {
        candidate,
        channels: u32::try_from(channels).expect("validated C must fit u32"),
        length: u32::try_from(length).expect("validated L must fit u32"),
        dilation: u32::try_from(dilation).expect("validated dilation must fit u32"),
        padding: u32::try_from(padding).expect("validated padding must fit u32"),
        input_span: u32::try_from(input_span).expect("validated input span must fit u32"),
        input_tile_size: u32::try_from(input_tile_size)
            .expect("validated input tile size must fit u32"),
        weight_tile_size: u32::try_from(weight_tile_size)
            .expect("validated weight tile size must fit u32"),
        workgroup_size: u32::try_from(workgroup_size)
            .expect("validated workgroup size must fit u32"),
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            kernel,
            CubeDim::new_2d(
                LOCAL_TIME_LANES as u32,
                u32::try_from(local_channel_lanes)
                    .expect("validated channel-lane count must fit u32"),
            ),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(input.handle.binding())
        .with_buffer(weight.handle.binding())
        .with_buffer(bias.handle.binding())
        .with_buffer(output.handle.clone().binding());
    client.launch(
        task,
        CubeCount::new_3d(
            u32::try_from(time_tiles).expect("validated time tile count must fit u32"),
            u32::try_from(output_channel_tiles)
                .expect("validated output-channel tile count must fit u32"),
            BATCH as u32,
        ),
        bindings,
    );
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_accounting_matches_the_bounded_tile_matrix() {
        let expected = [
            (
                Conv1dK7TileCandidate::Time32Output16,
                32,
                16,
                128,
                4,
                [9_600, 10_368, 12_672],
            ),
            (
                Conv1dK7TileCandidate::Time32Output32,
                32,
                32,
                256,
                4,
                [16_768, 17_536, 19_840],
            ),
            (
                Conv1dK7TileCandidate::Time64Output32,
                64,
                32,
                256,
                8,
                [18_816, 19_584, 21_888],
            ),
        ];

        for (candidate, time_tile, output_tile, workgroup_size, accumulators, shared) in expected {
            assert_eq!(candidate.time_tile(), time_tile);
            assert_eq!(candidate.output_channel_tile(), output_tile);
            assert_eq!(
                LOCAL_TIME_LANES * candidate.local_channel_lanes(),
                workgroup_size
            );
            assert_eq!(candidate.accumulators_per_invocation(), accumulators);
            assert_eq!(
                [1, 3, 9].map(|dilation| candidate.shared_memory_bytes(dilation)),
                shared
            );
        }
    }

    #[test]
    fn candidate_shader_contract_is_uniform_and_fully_templated() {
        let placeholders = [
            "channels",
            "length",
            "dilation",
            "padding",
            "input_span",
            "input_tile_size",
            "weight_tile_size",
            "workgroup_size",
            "local_channel_lanes",
            "output_channel_tile",
        ];

        for candidate in [
            Conv1dK7TileCandidate::Time32Output16,
            Conv1dK7TileCandidate::Time32Output32,
            Conv1dK7TileCandidate::Time64Output32,
        ] {
            let shader = candidate.shader();
            let bindings = shader
                .lines()
                .map(str::trim)
                .filter(|line| line.starts_with("@group(0)") && line.contains("var<storage"))
                .collect::<Vec<_>>();
            assert_eq!(bindings.len(), 4, "{} binding count", candidate.label());
            assert!(
                bindings
                    .iter()
                    .all(|line| line.contains("var<storage, read_write>")),
                "{} mixes storage access modes: {bindings:?}",
                candidate.label()
            );
            for placeholder in placeholders {
                assert!(
                    shader.contains(&format!("{{{{ {placeholder} }}}}")),
                    "{} omits template placeholder {placeholder}",
                    candidate.label()
                );
            }
        }
    }
}
