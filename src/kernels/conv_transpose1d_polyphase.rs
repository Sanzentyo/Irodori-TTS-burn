//! Production polyphase WGSL path for the DACVAE decoder upsamplers.
//!
//! The released decoder uses only even strides, `kernel_size = 2 * stride`,
//! `padding = stride / 2`, `output_padding = 0`, `dilation = 1`, and one
//! group. Under that exact contract every output sample depends on two input
//! samples. This module packs the two phase weights once and computes the
//! phase shuffle directly, avoiding Burn's full `Cout * kernel * Lin`
//! columns allocation and the following col2im dispatch. Exact released case
//! 0 uses a measured Cin=32 tile on capable devices; every other shape and
//! unsupported device retains the portable Cin=16 tile.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime, into_contiguous,
};
use burn::tensor::{DType, Shape};
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

const BATCH: usize = 1;
const TIME_REPEATS: usize = 4;
const PACK_WORKGROUP_SIZE: u32 = 256;
const F32_BYTES: usize = size_of::<f32>();
const MAX_PORTABLE_WORKGROUP_STORAGE_BYTES: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum PolyphaseTile {
    PortableCin16,
    Case0Cin32,
}

impl PolyphaseTile {
    const fn code(self) -> u32 {
        match self {
            Self::PortableCin16 => 0,
            Self::Case0Cin32 => 1,
        }
    }

    const fn local_time_lanes(self) -> usize {
        match self {
            Self::PortableCin16 => 8,
            Self::Case0Cin32 => 16,
        }
    }

    const fn output_channel_tile(self) -> usize {
        16
    }

    const fn input_channel_tile(self) -> usize {
        match self {
            Self::PortableCin16 => 16,
            Self::Case0Cin32 => 32,
        }
    }

    const fn time_tile(self) -> usize {
        self.local_time_lanes() * TIME_REPEATS
    }

    const fn input_span(self) -> usize {
        self.time_tile() + 1
    }

    const fn workgroup_size(self) -> usize {
        self.local_time_lanes() * self.output_channel_tile()
    }

    const fn input_tile_elements(self) -> usize {
        self.input_channel_tile() * self.input_span()
    }

    const fn weight_tile_elements(self) -> usize {
        self.output_channel_tile() * self.input_channel_tile() * 2
    }

    const fn shared_bytes(self) -> usize {
        (self.input_tile_elements() + self.weight_tile_elements()) * F32_BYTES
    }
}

const _: () = assert!(PolyphaseTile::PortableCin16.shared_bytes() == 4_160);
const _: () = assert!(PolyphaseTile::Case0Cin32.shared_bytes() == 12_416);
const _: () =
    assert!(PolyphaseTile::Case0Cin32.shared_bytes() <= MAX_PORTABLE_WORKGROUP_STORAGE_BYTES);

fn select_polyphase_tile(
    input_channels: usize,
    output_channels: usize,
    input_length: usize,
    stride: usize,
    case0_cin32_supported: bool,
) -> PolyphaseTile {
    // RTX 3060 Ti isolated 2026-08-10, 10 warmup / 100x5 rotated:
    // portable Cin16 2198.469 us [2185.628,2393.116], Cin32 T64/O16
    // 2150.803 us [2126.272,2155.187] (1.022x), bit-exact and WGPU-error-free.
    // The measured T32/O32 Cin32 branch was slower (2155.087 us) and rejected.
    let exact_case0 =
        (input_channels, output_channels, input_length, stride) == (1_536, 768, 50, 12);
    if exact_case0 && case0_cin32_supported {
        PolyphaseTile::Case0Cin32
    } else {
        PolyphaseTile::PortableCin16
    }
}

/// Even strides present in the released DACVAE decoder.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ConvTranspose1dStride {
    /// Final `192 -> 96` upsampler.
    Two,
    /// `384 -> 192` upsampler.
    Eight,
    /// `768 -> 384` upsampler.
    Ten,
    /// First `1536 -> 768` upsampler.
    Twelve,
}

impl ConvTranspose1dStride {
    /// Numeric stride and number of polyphase components.
    pub const fn value(self) -> usize {
        match self {
            Self::Two => 2,
            Self::Eight => 8,
            Self::Ten => 10,
            Self::Twelve => 12,
        }
    }
}

impl TryFrom<usize> for ConvTranspose1dStride {
    type Error = &'static str;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            2 => Ok(Self::Two),
            8 => Ok(Self::Eight),
            10 => Ok(Self::Ten),
            12 => Ok(Self::Twelve),
            _ => Err("DACVAE ConvTranspose1d stride must be one of 2, 8, 10, or 12"),
        }
    }
}

/// One-time packed `[phase, Cout, Cin, tap]` f32 weights.
///
/// `tap=0` stores original kernel index `phase`; `tap=1` stores
/// `phase + stride`. The packed allocation contains exactly as many elements
/// as the checkpoint-native `[Cin, Cout, 2 * stride]` tensor.
#[derive(Debug)]
pub struct PackedConvTranspose1dWeight {
    tensor: CubeTensor<WgpuRuntime>,
    stride: ConvTranspose1dStride,
    input_channels: usize,
    output_channels: usize,
}

impl PackedConvTranspose1dWeight {
    /// Validate and wrap an already packed `[phase, Cout, Cin, 2]` tensor.
    ///
    /// This is used by model-owned inference caches. It does not allocate or
    /// change the tensor layout.
    pub fn from_tensor(tensor: CubeTensor<WgpuRuntime>, stride: ConvTranspose1dStride) -> Self {
        assert_eq!(
            tensor.dtype,
            DType::F32,
            "packed ConvTranspose1d weight must be f32"
        );
        assert_eq!(
            tensor.meta.num_dims(),
            4,
            "packed ConvTranspose1d weight must be rank 4 [phase, Cout, Cin, 2]"
        );
        let [phases, output_channels, input_channels, taps] = tensor.meta.shape().dims::<4>();
        assert_eq!(phases, stride.value(), "packed phase count mismatch");
        assert_eq!(taps, 2, "packed ConvTranspose1d weight must have two taps");
        assert!(
            input_channels > 0,
            "packed ConvTranspose1d requires Cin > 0"
        );
        assert!(
            output_channels > 0,
            "packed ConvTranspose1d requires Cout > 0"
        );
        assert!(
            tensor.is_contiguous(),
            "packed ConvTranspose1d weight must be contiguous"
        );
        Self {
            tensor,
            stride,
            input_channels,
            output_channels,
        }
    }

    /// Packed weight byte size, useful for reporting persistent VRAM cost.
    pub fn bytes(&self) -> usize {
        self.tensor
            .meta
            .num_elements()
            .checked_mul(core::mem::size_of::<f32>())
            .expect("packed ConvTranspose1d weight byte count overflow")
    }

    /// Clone the reference-counted primitive handle for validation/readback.
    pub fn tensor(&self) -> CubeTensor<WgpuRuntime> {
        self.tensor.clone()
    }

    /// Consume the wrapper and return its reference-counted primitive handle.
    pub fn into_tensor(self) -> CubeTensor<WgpuRuntime> {
        self.tensor
    }
}

#[derive(Debug)]
struct ConvTranspose1dWeightPackKernel {
    input_channels: u32,
    output_channels: u32,
    stride: u32,
    kernel_size: u32,
    elements: u32,
    dispatch_x: u32,
}

impl KernelSource for ConvTranspose1dWeightPackKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("conv_transpose1d_weight_pack.wgsl"))
            .register("input_channels", self.input_channels.to_string())
            .register("output_channels", self.output_channels.to_string())
            .register("stride", self.stride.to_string())
            .register("kernel_size", self.kernel_size.to_string())
            .register("elements", self.elements.to_string())
            .register("dispatch_x", self.dispatch_x.to_string())
            .register("workgroup_size", PACK_WORKGROUP_SIZE.to_string())
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.input_channels,
            self.output_channels,
            self.stride,
            self.kernel_size,
            self.elements,
            self.dispatch_x,
        ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PackDispatch2d {
    x: u32,
    y: u32,
}

impl PackDispatch2d {
    fn for_elements(elements: usize, max_x: u32, max_y: u32) -> Self {
        assert!(elements > 0, "ConvTranspose1d pack requires elements > 0");
        assert!(
            max_x > 0 && max_y > 0,
            "ConvTranspose1d pack requires non-zero 2D dispatch limits, got ({max_x}, {max_y})"
        );

        let total_workgroups = elements.div_ceil(PACK_WORKGROUP_SIZE as usize);
        let groups_x = total_workgroups.min(max_x as usize);
        let groups_y = total_workgroups.div_ceil(groups_x);
        assert!(
            groups_y <= max_y as usize,
            "ConvTranspose1d pack requires ({groups_x}, {groups_y}) workgroups, exceeding device limits ({max_x}, {max_y})"
        );

        let dispatched_workgroups = groups_x
            .checked_mul(groups_y)
            .expect("ConvTranspose1d pack 2D workgroup count overflow");
        let launched_invocations = dispatched_workgroups
            .checked_mul(PACK_WORKGROUP_SIZE as usize)
            .expect("ConvTranspose1d pack invocation count overflow");
        let largest_source_index = launched_invocations
            .checked_sub(1)
            .expect("ConvTranspose1d pack dispatch must launch at least one invocation");
        assert!(
            u32::try_from(largest_source_index).is_ok(),
            "ConvTranspose1d pack largest launched source index {largest_source_index} exceeds WGSL u32 indexing"
        );

        Self {
            x: u32::try_from(groups_x).expect("validated pack dispatch x must fit u32"),
            y: u32::try_from(groups_y).expect("validated pack dispatch y must fit u32"),
        }
    }
}

#[derive(Debug)]
struct ConvTranspose1dPolyphaseKernel {
    tile: PolyphaseTile,
    input_channels: u32,
    output_channels: u32,
    input_length: u32,
    output_length: u32,
    stride: u32,
    padding: u32,
}

impl KernelSource for ConvTranspose1dPolyphaseKernel {
    fn source(&self) -> SourceTemplate {
        let tile = self.tile;
        SourceTemplate::new(include_str!("conv_transpose1d_polyphase.wgsl"))
            .register("input_channels", self.input_channels.to_string())
            .register("output_channels", self.output_channels.to_string())
            .register("input_length", self.input_length.to_string())
            .register("output_length", self.output_length.to_string())
            .register("stride", self.stride.to_string())
            .register("padding", self.padding.to_string())
            .register("local_time_lanes", tile.local_time_lanes().to_string())
            .register(
                "output_channel_tile",
                tile.output_channel_tile().to_string(),
            )
            .register("time_repeats", TIME_REPEATS.to_string())
            .register("time_tile", tile.time_tile().to_string())
            .register("input_channel_tile", tile.input_channel_tile().to_string())
            .register("input_span", tile.input_span().to_string())
            .register("workgroup_size", tile.workgroup_size().to_string())
            .register(
                "input_tile_elements",
                tile.input_tile_elements().to_string(),
            )
            .register(
                "weight_tile_elements",
                tile.weight_tile_elements().to_string(),
            )
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.tile.code(),
            self.input_channels,
            self.output_channels,
            self.input_length,
            self.output_length,
            self.stride,
            self.padding,
        ))
    }
}

/// Pack checkpoint-native `[Cin, Cout, 2 * stride]` weights once.
///
/// The resulting phase-major layout gives each compute workgroup contiguous
/// weight reads. Packing is a one-time inference preparation operation and is
/// intentionally excluded from steady-state timings.
///
/// # Panics
///
/// Panics for a non-f32, non-rank-3, empty, or unsupported weight, or when a
/// storage/dispatch calculation exceeds WGPU/WGSL limits.
pub fn pack_conv_transpose1d_weight_wgsl(
    weight: CubeTensor<WgpuRuntime>,
    stride: ConvTranspose1dStride,
) -> PackedConvTranspose1dWeight {
    assert_eq!(
        weight.dtype,
        DType::F32,
        "ConvTranspose1d weight must be f32"
    );
    assert_eq!(
        weight.meta.num_dims(),
        3,
        "ConvTranspose1d weight must be rank 3 [Cin, Cout, 2*stride]"
    );

    let weight = into_contiguous(weight);
    let shape = weight.meta.shape();
    let input_channels = shape[0];
    let output_channels = shape[1];
    let kernel_size = shape[2];
    let stride_value = stride.value();
    assert!(input_channels > 0, "ConvTranspose1d requires Cin > 0");
    assert!(output_channels > 0, "ConvTranspose1d requires Cout > 0");
    assert_eq!(
        kernel_size,
        2 * stride_value,
        "polyphase packing requires kernel_size = 2 * stride"
    );

    let elements = input_channels
        .checked_mul(output_channels)
        .and_then(|value| value.checked_mul(kernel_size))
        .expect("ConvTranspose1d weight element count overflow");
    for (name, value) in [
        ("Cin", input_channels),
        ("Cout", output_channels),
        ("stride", stride_value),
        ("kernel size", kernel_size),
        ("weight elements", elements),
    ] {
        assert!(
            u32::try_from(value).is_ok(),
            "{name}={value} exceeds WGSL u32 indexing"
        );
    }

    let output_bytes = elements
        .checked_mul(core::mem::size_of::<f32>())
        .expect("packed ConvTranspose1d weight byte count overflow");
    let client = weight.client.clone();
    let packed = CubeTensor::new_contiguous(
        client.clone(),
        weight.device.clone(),
        Shape::from([stride_value, output_channels, input_channels, 2]),
        client.empty(output_bytes),
        DType::F32,
    );
    let hardware = &client.properties().hardware;
    assert!(
        hardware.max_bindings >= 2,
        "ConvTranspose1d pack requires two storage bindings, device supports {}",
        hardware.max_bindings
    );
    assert!(
        hardware.max_units_per_cube >= PACK_WORKGROUP_SIZE,
        "ConvTranspose1d pack requires {PACK_WORKGROUP_SIZE} invocations per workgroup, device supports {}",
        hardware.max_units_per_cube
    );
    let dispatch = PackDispatch2d::for_elements(
        elements,
        hardware.max_cube_count.0,
        hardware.max_cube_count.1,
    );

    let kernel = ConvTranspose1dWeightPackKernel {
        input_channels: u32::try_from(input_channels).expect("validated Cin must fit u32"),
        output_channels: u32::try_from(output_channels).expect("validated Cout must fit u32"),
        stride: u32::try_from(stride_value).expect("validated stride must fit u32"),
        kernel_size: u32::try_from(kernel_size).expect("validated kernel size must fit u32"),
        elements: u32::try_from(elements).expect("validated element count must fit u32"),
        dispatch_x: dispatch.x,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> = Box::new(
        SourceKernel::new(kernel, CubeDim::new_1d(PACK_WORKGROUP_SIZE)),
    );
    let bindings = KernelArguments::new()
        .with_buffer(weight.handle.binding())
        .with_buffer(packed.handle.clone().binding());
    client.launch(task, CubeCount::new_2d(dispatch.x, dispatch.y), bindings);

    PackedConvTranspose1dWeight::from_tensor(packed, stride)
}

/// Compute the released DACVAE ConvTranspose1d contract in one tiled dispatch.
///
/// Required operation and layouts:
///
/// - `input`: contiguous or materialisable f32 `[1, Cin, Lin]`
/// - `packed_weight`: phase-major output from
///   [`pack_conv_transpose1d_weight_wgsl`]
/// - `bias`: contiguous f32 `[Cout]`
/// - `kernel_size = 2 * stride`, `padding = stride / 2`
/// - `output_padding = 0`, `dilation = 1`, `groups = 1`
///
/// The output is contiguous `[1, Cout, Lin * stride]`. The dispatch creates no
/// global scratch allocation besides that output.
///
/// # Panics
///
/// Panics for a mismatching dtype, shape, device, stride, parameter layout,
/// empty dimension, unsupported channel multiple, or a storage/dispatch/index
/// calculation that exceeds WGPU/WGSL limits.
pub fn conv_transpose1d_polyphase_wgsl(
    input: CubeTensor<WgpuRuntime>,
    packed_weight: &PackedConvTranspose1dWeight,
    bias: CubeTensor<WgpuRuntime>,
) -> CubeTensor<WgpuRuntime> {
    for (name, tensor) in [
        ("input", &input),
        ("packed_weight", &packed_weight.tensor),
        ("bias", &bias),
    ] {
        assert_eq!(
            tensor.dtype,
            DType::F32,
            "polyphase ConvTranspose1d only supports f32 {name}"
        );
        input.assert_is_on_same_device(tensor);
    }
    assert_eq!(
        input.meta.num_dims(),
        3,
        "input must be rank 3 [1, Cin, Lin]"
    );
    assert_eq!(bias.meta.num_dims(), 1, "bias must be rank 1 [Cout]");

    let input_shape = input.meta.shape();
    let batch = input_shape[0];
    let input_channels = input_shape[1];
    let input_length = input_shape[2];
    let output_channels = packed_weight.output_channels;
    let stride = packed_weight.stride.value();
    assert_eq!(
        batch, BATCH,
        "polyphase ConvTranspose1d is specialised for B=1"
    );
    assert!(
        input_length > 0,
        "polyphase ConvTranspose1d requires Lin > 0"
    );
    assert_eq!(
        input_channels, packed_weight.input_channels,
        "input Cin does not match packed weight"
    );
    let portable_tile = PolyphaseTile::PortableCin16;
    assert!(
        input_channels.is_multiple_of(portable_tile.input_channel_tile()),
        "polyphase ConvTranspose1d requires Cin to be a multiple of {}, got {input_channels}",
        portable_tile.input_channel_tile()
    );
    assert!(
        output_channels.is_multiple_of(portable_tile.output_channel_tile()),
        "polyphase ConvTranspose1d requires Cout to be a multiple of {}, got {output_channels}",
        portable_tile.output_channel_tile()
    );
    assert_eq!(
        packed_weight.tensor.meta.shape().dims::<4>(),
        [stride, output_channels, input_channels, 2],
        "packed weight shape mismatch"
    );
    assert!(
        packed_weight.tensor.is_contiguous(),
        "packed weight must be contiguous"
    );
    assert!(bias.is_contiguous(), "bias must be contiguous");
    assert_eq!(
        bias.meta.shape()[0],
        output_channels,
        "bias must have shape [Cout]"
    );
    let input = into_contiguous(input);
    let output_length = input_length
        .checked_mul(stride)
        .expect("ConvTranspose1d output length overflow");
    let input_elements = input_channels
        .checked_mul(input_length)
        .expect("ConvTranspose1d input element count overflow");
    let output_elements = output_channels
        .checked_mul(output_length)
        .expect("ConvTranspose1d output element count overflow");
    let packed_elements = stride
        .checked_mul(output_channels)
        .and_then(|value| value.checked_mul(input_channels))
        .and_then(|value| value.checked_mul(2))
        .expect("packed ConvTranspose1d weight element count overflow");
    let client = input.client.clone();
    let hardware = &client.properties().hardware;
    let case0_tile = PolyphaseTile::Case0Cin32;
    let case0_time_tiles = input_length.div_ceil(case0_tile.time_tile());
    let case0_output_channel_tiles = output_channels / case0_tile.output_channel_tile();
    let case0_cin32_supported = hardware.max_shared_memory_size >= case0_tile.shared_bytes()
        && hardware.max_units_per_cube >= case0_tile.workgroup_size() as u32
        && hardware.max_cube_dim.0 >= case0_tile.local_time_lanes() as u32
        && hardware.max_cube_dim.1 >= case0_tile.output_channel_tile() as u32
        && hardware.max_cube_count.0 >= case0_time_tiles as u32
        && hardware.max_cube_count.1 >= case0_output_channel_tiles as u32
        && hardware.max_cube_count.2 >= stride as u32;
    let tile = select_polyphase_tile(
        input_channels,
        output_channels,
        input_length,
        stride,
        case0_cin32_supported,
    );
    assert!(
        input_channels.is_multiple_of(tile.input_channel_tile()),
        "selected polyphase tile requires Cin to be a multiple of {}, got {input_channels}",
        tile.input_channel_tile()
    );
    assert!(
        output_channels.is_multiple_of(tile.output_channel_tile()),
        "selected polyphase tile requires Cout to be a multiple of {}, got {output_channels}",
        tile.output_channel_tile()
    );
    let time_tiles = input_length.div_ceil(tile.time_tile());
    let output_channel_tiles = output_channels / tile.output_channel_tile();
    for (name, value) in [
        ("Cin", input_channels),
        ("Cout", output_channels),
        ("Lin", input_length),
        ("Lout", output_length),
        ("stride", stride),
        ("input elements", input_elements),
        ("output elements", output_elements),
        ("packed weight elements", packed_elements),
        ("time tiles", time_tiles),
        ("output-channel tiles", output_channel_tiles),
    ] {
        assert!(
            u32::try_from(value).is_ok(),
            "{name}={value} exceeds WGSL/WGPU u32 indexing"
        );
    }

    assert!(
        hardware.max_bindings >= 4,
        "polyphase ConvTranspose1d requires four storage bindings, device supports {}",
        hardware.max_bindings
    );
    assert!(
        hardware.max_shared_memory_size >= tile.shared_bytes(),
        "selected polyphase tile requires {} shared bytes, device supports {}",
        tile.shared_bytes(),
        hardware.max_shared_memory_size
    );
    assert!(
        hardware.max_units_per_cube >= tile.workgroup_size() as u32,
        "selected polyphase tile requires {} invocations per workgroup, device supports {}",
        tile.workgroup_size(),
        hardware.max_units_per_cube
    );
    assert!(
        hardware.max_cube_dim.0 >= tile.local_time_lanes() as u32
            && hardware.max_cube_dim.1 >= tile.output_channel_tile() as u32,
        "selected polyphase tile requires workgroup dimensions ({}, {}, 1), device supports {:?}",
        tile.local_time_lanes(),
        tile.output_channel_tile(),
        hardware.max_cube_dim
    );
    assert!(
        hardware.max_cube_count.0 >= time_tiles as u32
            && hardware.max_cube_count.1 >= output_channel_tiles as u32
            && hardware.max_cube_count.2 >= stride as u32,
        "polyphase ConvTranspose1d dispatch ({time_tiles}, {output_channel_tiles}, {stride}) exceeds device limits {:?}",
        hardware.max_cube_count
    );

    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .expect("ConvTranspose1d output byte count overflow");
    let output = CubeTensor::new_contiguous(
        client.clone(),
        input.device.clone(),
        Shape::from([batch, output_channels, output_length]),
        client.empty(output_bytes),
        DType::F32,
    );
    let kernel = ConvTranspose1dPolyphaseKernel {
        tile,
        input_channels: u32::try_from(input_channels).expect("validated Cin must fit u32"),
        output_channels: u32::try_from(output_channels).expect("validated Cout must fit u32"),
        input_length: u32::try_from(input_length).expect("validated Lin must fit u32"),
        output_length: u32::try_from(output_length).expect("validated Lout must fit u32"),
        stride: u32::try_from(stride).expect("validated stride must fit u32"),
        padding: u32::try_from(stride / 2).expect("validated padding must fit u32"),
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            kernel,
            CubeDim::new_2d(
                tile.local_time_lanes() as u32,
                tile.output_channel_tile() as u32,
            ),
        ));
    let bindings = KernelArguments::new()
        .with_buffer(input.handle.binding())
        .with_buffer(packed_weight.tensor.handle.clone().binding())
        .with_buffer(bias.handle.binding())
        .with_buffer(output.handle.clone().binding());
    client.launch(
        task,
        CubeCount::new_3d(
            u32::try_from(time_tiles).expect("validated time tile count must fit u32"),
            u32::try_from(output_channel_tiles)
                .expect("validated output-channel tile count must fit u32"),
            u32::try_from(stride).expect("validated phase count must fit u32"),
        ),
        bindings,
    );
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_shapes_satisfy_exact_polyphase_contract() {
        let portable_tile = PolyphaseTile::PortableCin16;
        for (input_channels, output_channels, input_length, stride) in [
            (1536usize, 768usize, 50usize, 12usize),
            (768, 384, 600, 10),
            (384, 192, 6_000, 8),
            (192, 96, 48_000, 2),
        ] {
            let stride = ConvTranspose1dStride::try_from(stride).unwrap();
            let stride = stride.value();
            let kernel_size = 2 * stride;
            let padding = stride / 2;
            let output_padding = 0;
            let output_length =
                (input_length - 1) * stride - 2 * padding + (kernel_size - 1) + output_padding + 1;
            assert!(input_channels.is_multiple_of(portable_tile.input_channel_tile()));
            assert!(output_channels.is_multiple_of(portable_tile.output_channel_tile()));
            assert_eq!(output_length, input_length * stride);
            assert_eq!(kernel_size, 2 * stride);
            assert_eq!(stride % 2, 0);
        }
    }

    #[test]
    fn case0_cin32_route_is_exact_and_has_portable_fallback() {
        assert_eq!(
            select_polyphase_tile(1_536, 768, 50, 12, true),
            PolyphaseTile::Case0Cin32
        );
        assert_eq!(
            select_polyphase_tile(1_536, 768, 50, 12, false),
            PolyphaseTile::PortableCin16
        );
        for dimensions in [
            (1_535, 768, 50, 12),
            (1_536, 767, 50, 12),
            (1_536, 768, 49, 12),
            (1_536, 768, 50, 10),
            (768, 384, 600, 10),
            (384, 192, 6_000, 8),
            (192, 96, 48_000, 2),
        ] {
            assert_eq!(
                select_polyphase_tile(dimensions.0, dimensions.1, dimensions.2, dimensions.3, true,),
                PolyphaseTile::PortableCin16
            );
        }
    }

    #[test]
    fn case0_cin32_resources_and_barrier_reduction_match_measurement() {
        let portable = PolyphaseTile::PortableCin16;
        let case0 = PolyphaseTile::Case0Cin32;
        assert_eq!(portable.workgroup_size(), 128);
        assert_eq!(portable.shared_bytes(), 4_160);
        assert_eq!(case0.workgroup_size(), 256);
        assert_eq!(case0.shared_bytes(), 12_416);

        let portable_workgroups =
            50usize.div_ceil(portable.time_tile()) * (768 / portable.output_channel_tile()) * 12;
        let case0_workgroups =
            50usize.div_ceil(case0.time_tile()) * (768 / case0.output_channel_tile()) * 12;
        let portable_rounds = 1_536 / portable.input_channel_tile();
        let case0_rounds = 1_536 / case0.input_channel_tile();
        assert_eq!((portable_workgroups, portable_rounds), (1_152, 96));
        assert_eq!((case0_workgroups, case0_rounds), (576, 48));
        assert_eq!(
            case0_workgroups * case0_rounds * 2,
            portable_workgroups * portable_rounds * 2 / 4
        );
    }

    #[test]
    fn polyphase_shader_templates_tile_but_preserves_operation_order() {
        let shader = include_str!("conv_transpose1d_polyphase.wgsl");
        for template in [
            "{{ local_time_lanes }}",
            "{{ output_channel_tile }}",
            "{{ input_channel_tile }}",
            "{{ input_span }}",
            "{{ workgroup_size }}",
        ] {
            assert!(shader.contains(template));
        }
        assert_eq!(shader.matches("workgroupBarrier();").count(), 2);
        assert!(shader.contains("accumulator_main_3"));
        assert!(shader.contains("accumulator_previous_3"));
        assert!(shader.contains("accumulator_main_0 + accumulator_previous_0 + bias"));
    }

    #[test]
    fn phase_mapping_covers_each_output_index_once() {
        for stride in [2usize, 8, 10, 12] {
            let input_length = 7;
            let padding = stride / 2;
            let mut covered = vec![0_u8; input_length * stride];
            for phase in 0..stride {
                let output_phase = (phase + stride - padding) % stride;
                for time in 0..input_length {
                    covered[time * stride + output_phase] += 1;
                }
            }
            assert!(covered.into_iter().all(|count| count == 1));
        }
    }

    #[test]
    fn polyphase_pairs_match_scatter_definition() {
        for stride in [2usize, 8, 10, 12] {
            let input_length = 7usize;
            let padding = stride / 2;
            for phase in 0..stride {
                let shift = usize::from(phase < padding);
                let output_phase = (phase + stride - padding) % stride;
                for time in 0..input_length {
                    let output_time = time * stride + output_phase;
                    let a = time + shift;
                    let mut polyphase = Vec::with_capacity(2);
                    if a < input_length {
                        polyphase.push((a, phase));
                    }
                    if a > 0 && a - 1 < input_length {
                        polyphase.push((a - 1, phase + stride));
                    }
                    polyphase.sort_unstable();

                    let mut scatter = Vec::with_capacity(2);
                    for input_time in 0..input_length {
                        for kernel_index in 0..2 * stride {
                            if input_time * stride + kernel_index == output_time + padding {
                                scatter.push((input_time, kernel_index));
                            }
                        }
                    }
                    scatter.sort_unstable();
                    assert_eq!(polyphase, scatter);
                }
            }
        }
    }

    #[test]
    fn released_pack_dispatches_fit_rtx_3060ti_limits() {
        let max_axis = 65_535;
        for (elements, expected) in [
            (1536usize * 768 * 24, PackDispatch2d { x: 65_535, y: 2 }),
            (768 * 384 * 20, PackDispatch2d { x: 23_040, y: 1 }),
            (384 * 192 * 16, PackDispatch2d { x: 4_608, y: 1 }),
            (192 * 96 * 4, PackDispatch2d { x: 288, y: 1 }),
        ] {
            assert_eq!(
                PackDispatch2d::for_elements(elements, max_axis, max_axis),
                expected
            );
        }
    }

    #[test]
    #[should_panic(expected = "exceeding device limits")]
    fn pack_dispatch_rejects_insufficient_second_axis() {
        let elements = 1536usize * 768 * 24;
        let _ = PackDispatch2d::for_elements(elements, 65_535, 1);
    }

    #[test]
    #[should_panic(expected = "exceeds WGSL u32 indexing")]
    fn pack_dispatch_rejects_padded_grid_index_overflow() {
        let _ = PackDispatch2d::for_elements(u32::MAX as usize, 65_535, 65_535);
    }
}
