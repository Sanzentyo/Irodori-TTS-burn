//! Production-shape tiled WGSL candidate for DACVAE residual convolutions.
//!
//! This isolated kernel keeps the checkpoint-native NCHW/OIK layouts and
//! specialises the exact common operation: B=1, equal input/output channels,
//! k=7, stride=1, groups=1, f32, and dilation 1, 3, or 9.

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
const LOCAL_CHANNEL_LANES: usize = 8;
const TIME_REPEATS: usize = 4;
const CHANNEL_REPEATS: usize = 2;
const TIME_TILE: usize = LOCAL_TIME_LANES * TIME_REPEATS;
const OUTPUT_CHANNEL_TILE: usize = LOCAL_CHANNEL_LANES * CHANNEL_REPEATS;
const INPUT_CHANNEL_TILE: usize = 16;
const MAX_PORTABLE_WORKGROUP_STORAGE_BYTES: usize = 16 * 1024;

/// Supported DACVAE residual convolution dilations.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Conv1dK7Dilation {
    /// Adjacent seven-tap convolution.
    One,
    /// Seven taps separated by three samples.
    Three,
    /// Seven taps separated by nine samples.
    Nine,
}

impl Conv1dK7Dilation {
    /// Numeric dilation passed to Burn and embedded in WGSL.
    pub const fn value(self) -> usize {
        match self {
            Self::One => 1,
            Self::Three => 3,
            Self::Nine => 9,
        }
    }
}

impl TryFrom<usize> for Conv1dK7Dilation {
    type Error = &'static str;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::One),
            3 => Ok(Self::Three),
            9 => Ok(Self::Nine),
            _ => Err("DACVAE k=7 dilation must be one of 1, 3, or 9"),
        }
    }
}

#[derive(Debug)]
struct Conv1dK7TiledKernel {
    channels: u32,
    length: u32,
    dilation: u32,
    padding: u32,
    input_span: u32,
    input_tile_size: u32,
    weight_tile_size: u32,
}

impl KernelSource for Conv1dK7TiledKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("conv1d_k7_tiled.wgsl"))
            .register("channels", self.channels.to_string())
            .register("length", self.length.to_string())
            .register("dilation", self.dilation.to_string())
            .register("padding", self.padding.to_string())
            .register("input_span", self.input_span.to_string())
            .register("input_tile_size", self.input_tile_size.to_string())
            .register("weight_tile_size", self.weight_tile_size.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
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

/// Compute an f32 same-length Conv1d using a shared-memory tiled kernel.
///
/// Required shapes and layouts:
///
/// - `input`: contiguous or materialisable `[1, C, L]`
/// - `weight`: contiguous `[C, C, 7]`
/// - `bias`: contiguous `[C]`
///
/// `C` must be a non-zero multiple of 16. The output is contiguous
/// `[1, C, L]`. Padding is exactly `3 * dilation`, matching the DACVAE
/// `pad_mode="none"` residual units.
///
/// # Panics
///
/// Panics for an incompatible dtype, rank, shape, device, parameter layout,
/// empty dimension, unsupported channel count, or a storage/dispatch/index
/// calculation that cannot be represented safely by WGPU/WGSL.
pub fn conv1d_k7_same_tiled_wgsl(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    dilation: Conv1dK7Dilation,
) -> CubeTensor<WgpuRuntime> {
    for (name, tensor) in [("input", &input), ("weight", &weight), ("bias", &bias)] {
        assert_eq!(
            tensor.dtype,
            DType::F32,
            "tiled k=7 Conv1d only supports f32 {name}"
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
    assert_eq!(batch, BATCH, "tiled k=7 Conv1d is specialised for B=1");
    assert!(channels > 0, "tiled k=7 Conv1d requires C > 0");
    assert!(length > 0, "tiled k=7 Conv1d requires L > 0");
    assert!(
        channels.is_multiple_of(OUTPUT_CHANNEL_TILE),
        "tiled k=7 Conv1d requires C to be a multiple of {OUTPUT_CHANNEL_TILE}, got {channels}"
    );

    let weight_shape = weight.meta.shape();
    assert_eq!(
        [weight_shape[0], weight_shape[1], weight_shape[2]],
        [channels, channels, KERNEL_SIZE],
        "weight must have shape [C, C, 7]"
    );
    let bias_shape = bias.meta.shape();
    assert_eq!(bias_shape[0], channels, "bias must have shape [C]");
    assert!(weight.is_contiguous(), "weight must be contiguous OIK");
    assert!(bias.is_contiguous(), "bias must be contiguous");

    let input = into_contiguous(input);
    let dilation = dilation.value();
    let padding = 3usize
        .checked_mul(dilation)
        .expect("k=7 Conv1d padding overflow");
    let input_span = TIME_TILE
        .checked_add(6usize.checked_mul(dilation).expect("input halo overflow"))
        .expect("input tile span overflow");
    let input_tile_size = INPUT_CHANNEL_TILE
        .checked_mul(input_span)
        .expect("input tile element count overflow");
    let weight_tile_size = OUTPUT_CHANNEL_TILE
        .checked_mul(INPUT_CHANNEL_TILE)
        .and_then(|value| value.checked_mul(KERNEL_SIZE))
        .expect("weight tile element count overflow");
    let shared_bytes = input_tile_size
        .checked_add(weight_tile_size)
        .and_then(|value| value.checked_mul(core::mem::size_of::<f32>()))
        .expect("shared-memory byte count overflow");
    assert!(
        shared_bytes <= MAX_PORTABLE_WORKGROUP_STORAGE_BYTES,
        "tiled k=7 Conv1d needs {shared_bytes} shared bytes, exceeding the portable 16 KiB limit"
    );

    let input_elements = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(length))
        .expect("input/output element count overflow");
    let weight_elements = channels
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(KERNEL_SIZE))
        .expect("weight element count overflow");
    let time_tiles = length.div_ceil(TIME_TILE);
    let output_channel_tiles = channels / OUTPUT_CHANNEL_TILE;
    let final_time_base = time_tiles
        .checked_sub(1)
        .and_then(|value| value.checked_mul(TIME_TILE))
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

    let client = input.client.clone();
    let hardware = &client.properties().hardware;
    assert!(
        hardware.max_bindings >= 4,
        "tiled k=7 Conv1d requires four storage bindings, device supports {}",
        hardware.max_bindings
    );
    assert!(
        hardware.max_shared_memory_size >= shared_bytes,
        "tiled k=7 Conv1d requires {shared_bytes} shared bytes, device supports {}",
        hardware.max_shared_memory_size
    );
    assert!(
        hardware.max_units_per_cube >= (LOCAL_TIME_LANES * LOCAL_CHANNEL_LANES) as u32,
        "tiled k=7 Conv1d requires 128 invocations per workgroup, device supports {}",
        hardware.max_units_per_cube
    );
    assert!(
        hardware.max_cube_dim.0 >= LOCAL_TIME_LANES as u32
            && hardware.max_cube_dim.1 >= LOCAL_CHANNEL_LANES as u32,
        "tiled k=7 Conv1d requires workgroup dimensions (16, 8, 1), device supports {:?}",
        hardware.max_cube_dim
    );
    assert!(
        hardware.max_cube_count.0 >= time_tiles as u32
            && hardware.max_cube_count.1 >= output_channel_tiles as u32
            && hardware.max_cube_count.2 >= BATCH as u32,
        "tiled k=7 Conv1d dispatch ({time_tiles}, {output_channel_tiles}, {BATCH}) exceeds device limits {:?}",
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

    let kernel = Conv1dK7TiledKernel {
        channels: u32::try_from(channels).expect("validated C must fit u32"),
        length: u32::try_from(length).expect("validated L must fit u32"),
        dilation: u32::try_from(dilation).expect("validated dilation must fit u32"),
        padding: u32::try_from(padding).expect("validated padding must fit u32"),
        input_span: u32::try_from(input_span).expect("validated input span must fit u32"),
        input_tile_size: u32::try_from(input_tile_size)
            .expect("validated input tile size must fit u32"),
        weight_tile_size: u32::try_from(weight_tile_size)
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
