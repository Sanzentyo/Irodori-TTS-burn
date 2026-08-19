//! Shared layers for the DACVAE codec: Snake1d activation and Snake ResidualUnit.

use burn::tensor::{Device, FloatDType};
use burn::{
    module::{Param, ParamId},
    nn::{PaddingConfig1d, conv::Conv1d},
    prelude::*,
};

use crate::nvtx_range;

use super::algorithm::CodecK7Algorithm;
#[cfg(feature = "profile")]
use super::algorithm::CodecPointwiseAlgorithm;
#[cfg(feature = "profile")]
use super::profiling::CodecStageProfiler;

// ─── Snake1d ─────────────────────────────────────────────────────────────────

/// `x + sin²(α·x) / (α + ε)` activation, element-wise.
///
/// Alpha has shape `[1, channels, 1]` and is stored as a non-trained
/// inference-time constant.
#[derive(Module, Debug)]
pub(crate) struct Snake1d {
    pub(crate) alpha: Param<Tensor<3>>,
    /// f32 view of the (possibly f16) learned parameter used by CubeK custom
    /// epilogues. It is prepared once with the model rather than converted on
    /// every request.
    #[module(skip)]
    alpha_epilogue_f32: Tensor<3>,
    /// Interleaved `[alpha, 1 / (alpha + eps)]` f32 parameters used only by
    /// the prepared-epilogue differential route.
    #[cfg(feature = "profile")]
    #[module(skip)]
    alpha_recip_epilogue_f32: Option<Tensor<3>>,
}

impl Snake1d {
    pub(crate) fn new(alpha_tensor: Tensor<3>) -> Self {
        let alpha_epilogue_f32 = alpha_tensor.clone().cast(FloatDType::F32);
        Self {
            alpha: Param::initialized(ParamId::new(), alpha_tensor),
            alpha_epilogue_f32,
            #[cfg(feature = "profile")]
            alpha_recip_epilogue_f32: None,
        }
    }

    pub(crate) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let alpha = self.alpha.val();
        let ax = x.clone().mul(alpha.clone());
        let sin_sq = ax.sin().powi_scalar(2);
        let denom = alpha.add_scalar(1e-9_f32);
        x + sin_sq.div(denom)
    }

    fn prepare_post_cast_epilogue(&mut self) {
        // This cache is derived from a learned parameter and is skipped by
        // Module traversal. Rebuild it at every explicit model preparation so
        // record application, device moves, and dtype changes cannot leave a
        // stale epilogue binding behind.
        self.alpha_epilogue_f32 = self.alpha.val().cast(FloatDType::F32);
    }

    #[cfg(feature = "profile")]
    fn prepare_reciprocal_post_cast_epilogue(&mut self) {
        let alpha = self.alpha.val().cast(FloatDType::F32);
        let reciprocal = alpha.clone().add_scalar(1.0e-9).recip();
        self.alpha_recip_epilogue_f32 = Some(Tensor::cat(vec![alpha, reciprocal], 2));
    }
}

impl Snake1d {
    pub(crate) fn forward_wgsl(&self, x: Tensor<3>) -> Tensor<3> {
        let output = crate::kernels::snake::snake_wgsl(
            x.try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            self.alpha
                .val()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
        );
        Tensor::from_primitive::<crate::WgpuRaw>(output)
    }

    fn forward_nchw_to_nhwc_wgsl(&self, x: Tensor<3>) -> Option<Tensor<3>> {
        crate::kernels::snake::snake_nchw_to_nhwc_wgsl(
            x.try_into_primitive::<crate::WgpuRaw>().ok()?,
            self.alpha
                .val()
                .try_into_primitive::<crate::WgpuRaw>()
                .ok()?,
        )
        .map(Tensor::from_primitive::<crate::WgpuRaw>)
    }
}

// ─── ResidualUnit ─────────────────────────────────────────────────────────────

/// Snake → dilated Conv → Snake → 1×1 Conv + identity shortcut.
///
/// Padding for dilated conv: `(kernel - 1) * dil / 2 = 3 * dil` (kernel fixed at 7).
/// All residual units in the main encoder/decoder path use `compress=1` so
/// hidden dimension equals `dim` and the shortcut is a perfect identity.
#[derive(Module, Debug)]
pub(crate) struct ResidualUnit {
    pub(crate) act0: Snake1d,
    pub(crate) conv_dil: Conv1d,
    pub(crate) act1: Snake1d,
    pub(crate) conv_1x1: Conv1d,
    /// Inference-only `[1, channels_in, channels_out]` pointwise weight.
    #[module(skip)]
    pub(crate) packed_conv_1x1_weight: Option<Tensor<3>>,
    /// Inference-only vec4 k=7 weight layout used by the residue d3/d9 core.
    #[module(skip)]
    pub(crate) packed_conv_dil_weight_vectors: Option<Tensor<3>>,
    /// Physical `[O, K, I]` pitched weight consumed directly by CubeK. This
    /// diagnostic cache coexists with source OIK only for same-model A/B.
    #[module(skip)]
    pub(crate) prepared_k7_weight: Option<PreparedK7Weight>,
    /// Request-hot selector resolved once from a persisted tuning manifest.
    #[cfg(feature = "profile")]
    #[module(skip)]
    pub(crate) prepared_k7_selector: Option<super::algorithm::K7SelectorChoice>,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedK7Weight {
    oki: Tensor<3>,
    source_oik_shape: [usize; 3],
    physical_oki_strides: [usize; 3],
    #[cfg(feature = "profile")]
    physical_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Conv1dK7Route {
    TiledO16(crate::kernels::conv1d_k7_tiled::Conv1dK7Dilation),
    TiledO32Preferred(crate::kernels::conv1d_k7_tiled::Conv1dK7Dilation),
    TiledO64Preferred(crate::kernels::conv1d_k7_tiled::Conv1dK7Dilation),
    BurnFallback,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Conv1dK7StandaloneTile {
    T128(crate::kernels::conv1d_k7_t128::Conv1dK7T128Tile),
    Output16,
    Output32,
    Output64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Conv1dK7T128SnakeRoute {
    Fused(crate::kernels::conv1d_k7_t128::Conv1dK7T128Tile),
    Materialized(crate::kernels::conv1d_k7_t128::Conv1dK7T128Tile),
    Legacy,
}

/// Logical properties used by the production k=7 route decision.
///
/// Physical parameter contiguity is WGPU-specific and is checked immediately
/// before launch. Keeping the logical decision backend-independent lets tests
/// prove that every released decoder residual shape selects the tiled route
/// without constructing a GPU device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Conv1dK7Descriptor {
    batch: usize,
    input_channels: usize,
    length: usize,
    output_channels: usize,
    weight_input_channels: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    explicit_padding: Option<(usize, usize)>,
    bias_channels: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PointwiseResidualRoute {
    DirectThenFinalizer,
    FusedFinalizer,
    ExistingFallback,
}

/// Logical module, input, and inference-cache contract for the measured
/// decoder pointwise residual boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PointwiseResidualDescriptor {
    batch: usize,
    input_channels: usize,
    length: usize,
    output_channels: usize,
    weight_input_channels: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    explicit_padding: Option<(usize, usize)>,
    bias_channels: Option<usize>,
    packed_weight: Option<[usize; 3]>,
}

impl PointwiseResidualDescriptor {
    fn from_conv(
        conv: &Conv1d,
        input_shape: [usize; 3],
        packed_weight: Option<[usize; 3]>,
    ) -> Self {
        let [output_channels, weight_input_channels, kernel_size] = conv.weight.dims();
        let explicit_padding = match &conv.padding {
            PaddingConfig1d::Explicit(left, right) => Some((*left, *right)),
            PaddingConfig1d::Same | PaddingConfig1d::Valid => None,
        };
        Self {
            batch: input_shape[0],
            input_channels: input_shape[1],
            length: input_shape[2],
            output_channels,
            weight_input_channels,
            kernel_size,
            stride: conv.stride,
            dilation: conv.dilation,
            groups: conv.groups,
            explicit_padding,
            bias_channels: conv.bias.as_ref().map(|bias| bias.dims()[0]),
            packed_weight,
        }
    }

    fn route(self) -> PointwiseResidualRoute {
        let direct_shape = matches!(self.input_channels, 384 | 192 | 96) && self.length > 0;
        let exact_decoder_shape =
            matches!(self.input_channels, 768 | 384 | 192 | 96) && self.length > 0;
        let supported = self.batch == 1
            && exact_decoder_shape
            && self.input_channels == self.output_channels
            && self.input_channels == self.weight_input_channels
            && self.kernel_size == 1
            && self.stride == 1
            && self.dilation == 1
            && self.groups == 1
            && self.explicit_padding == Some((0, 0))
            && self.bias_channels == Some(self.output_channels)
            && self.packed_weight == Some([1, self.input_channels, self.output_channels]);

        if supported && direct_shape {
            PointwiseResidualRoute::DirectThenFinalizer
        } else if supported {
            PointwiseResidualRoute::FusedFinalizer
        } else {
            PointwiseResidualRoute::ExistingFallback
        }
    }

    fn supports_finalizer(self) -> bool {
        matches!(
            self.route(),
            PointwiseResidualRoute::DirectThenFinalizer | PointwiseResidualRoute::FusedFinalizer
        )
    }
}

impl Conv1dK7Descriptor {
    fn from_conv(conv: &Conv1d, input_shape: [usize; 3]) -> Self {
        let [output_channels, weight_input_channels, kernel_size] = conv.weight.dims();
        let explicit_padding = match &conv.padding {
            PaddingConfig1d::Explicit(left, right) => Some((*left, *right)),
            PaddingConfig1d::Same | PaddingConfig1d::Valid => None,
        };
        Self {
            batch: input_shape[0],
            input_channels: input_shape[1],
            length: input_shape[2],
            output_channels,
            weight_input_channels,
            kernel_size,
            stride: conv.stride,
            dilation: conv.dilation,
            groups: conv.groups,
            explicit_padding,
            bias_channels: conv.bias.as_ref().map(|bias| bias.dims()[0]),
        }
    }

    fn route(self) -> Conv1dK7Route {
        use crate::kernels::conv1d_k7_tiled::Conv1dK7Dilation;

        let Ok(dilation) = Conv1dK7Dilation::try_from(self.dilation) else {
            return Conv1dK7Route::BurnFallback;
        };
        let Some(padding) = 3usize.checked_mul(self.dilation) else {
            return Conv1dK7Route::BurnFallback;
        };
        let supported = self.batch == 1
            && self.input_channels > 0
            && self.length > 0
            && self.input_channels == self.output_channels
            && self.input_channels == self.weight_input_channels
            && self.input_channels.is_multiple_of(16)
            && self.kernel_size == 7
            && self.stride == 1
            && self.groups == 1
            && self.explicit_padding == Some((padding, padding))
            && self.bias_channels == Some(self.output_channels);

        if !supported {
            return Conv1dK7Route::BurnFallback;
        }

        let prefers_o64 = matches!(
            (self.input_channels, self.dilation),
            (768, 9) | (384 | 192 | 96, 1 | 3)
        );
        let decoder_stage_shape = match self.input_channels {
            768 => self.length.is_multiple_of(12),
            384 => self.length.is_multiple_of(120),
            192 => self.length.is_multiple_of(960),
            96 => self.length.is_multiple_of(1_920),
            _ => false,
        };
        if prefers_o64 {
            Conv1dK7Route::TiledO64Preferred(dilation)
        } else if decoder_stage_shape && !(self.input_channels == 768 && self.dilation == 3) {
            Conv1dK7Route::TiledO32Preferred(dilation)
        } else {
            Conv1dK7Route::TiledO16(dilation)
        }
    }

    /// Return the measured T128 reduction tile only after the complete logical
    /// convolution contract passes. Physical/device preflight remains at the
    /// launch site so a mismatch can retain the previous standalone selector.
    fn measured_t128_tile(self) -> Option<crate::kernels::conv1d_k7_t128::Conv1dK7T128Tile> {
        let dilation = match self.route() {
            Conv1dK7Route::TiledO16(dilation)
            | Conv1dK7Route::TiledO32Preferred(dilation)
            | Conv1dK7Route::TiledO64Preferred(dilation) => dilation,
            Conv1dK7Route::BurnFallback => return None,
        };
        crate::kernels::conv1d_k7_t128::production_tile_for_shape(
            self.input_channels,
            self.length,
            dilation,
        )
    }

    /// Return the conservative T256+Snake tile only after the complete logical
    /// convolution contract passes. A physical mismatch must retain the
    /// established T128+Snake chain.
    fn measured_t256_snake_tile(
        self,
    ) -> Option<crate::kernels::conv1d_k7_t256_snake_epilogue::Conv1dK7T256Tile> {
        let dilation = match self.route() {
            Conv1dK7Route::TiledO16(dilation)
            | Conv1dK7Route::TiledO32Preferred(dilation)
            | Conv1dK7Route::TiledO64Preferred(dilation) => dilation,
            Conv1dK7Route::BurnFallback => return None,
        };
        crate::kernels::conv1d_k7_t256_snake_epilogue::production_tile_for_shape(
            self.input_channels,
            self.length,
            dilation,
        )
    }

    /// Return the eight-shape vec4-store policy after the complete logical
    /// convolution contract passes. C768/L600/d9 deliberately retains the
    /// scalar T256 store path because its isolated median was 2.463 us slower.
    fn measured_t256_snake_vec4_store_tile(
        self,
    ) -> Option<crate::kernels::conv1d_k7_t256_snake_epilogue::Conv1dK7T256Tile> {
        let dilation = match self.route() {
            Conv1dK7Route::TiledO16(dilation)
            | Conv1dK7Route::TiledO32Preferred(dilation)
            | Conv1dK7Route::TiledO64Preferred(dilation) => dilation,
            Conv1dK7Route::BurnFallback => return None,
        };
        crate::kernels::conv1d_k7_t256_snake_vec4_store::production_tile_for_shape(
            self.input_channels,
            self.length,
            dilation,
        )
    }

    /// Return the accepted compact-residue d3/d9 routes for decoder-family
    /// C96/C192/C384 lengths.
    ///
    /// Every logical mismatch and all other channel/dilation shapes retain the
    /// current vec4/scalar T256, T128, and legacy selector chain.
    fn measured_residue_d1_dilation(
        self,
    ) -> Option<crate::kernels::conv1d_k7_residue_d1_snake::ResidueDilation> {
        let dilation = match self.route() {
            Conv1dK7Route::TiledO16(dilation)
            | Conv1dK7Route::TiledO32Preferred(dilation)
            | Conv1dK7Route::TiledO64Preferred(dilation) => dilation,
            Conv1dK7Route::BurnFallback => return None,
        };
        crate::kernels::conv1d_k7_residue_d1_snake::production_dilation_for_shape(
            self.input_channels,
            self.length,
            dilation,
        )
    }
}

/// Combine the exact released-shape policy with a caller-provided physical and
/// device preflight. `None` guarantees callers stay on the accepted fallback
/// instead of reaching the asserting T128 launcher.
fn select_compatible_conv1d_k7_t128_tile(
    descriptor: Conv1dK7Descriptor,
    contract_is_compatible: impl FnOnce(crate::kernels::conv1d_k7_t128::Conv1dK7T128Tile) -> bool,
) -> Option<crate::kernels::conv1d_k7_t128::Conv1dK7T128Tile> {
    descriptor
        .measured_t128_tile()
        .filter(|tile| contract_is_compatible(*tile))
}

/// Combine the conservative released-shape policy with the complete physical
/// five-buffer preflight. `None` means the current T128+Snake route must run
/// unchanged.
fn select_compatible_conv1d_k7_t256_snake_tile(
    descriptor: Conv1dK7Descriptor,
    contract_is_compatible: impl FnOnce(
        crate::kernels::conv1d_k7_t256_snake_epilogue::Conv1dK7T256Tile,
    ) -> bool,
) -> Option<crate::kernels::conv1d_k7_t256_snake_epilogue::Conv1dK7T256Tile> {
    descriptor
        .measured_t256_snake_tile()
        .filter(|tile| contract_is_compatible(*tile))
}

/// Combine the exact eight-shape vec4 policy with its complete five-buffer,
/// allocator-alignment, and resource preflight. `None` retains scalar T256.
fn select_compatible_conv1d_k7_t256_snake_vec4_store_tile(
    descriptor: Conv1dK7Descriptor,
    contract_is_compatible: impl FnOnce(
        crate::kernels::conv1d_k7_t256_snake_epilogue::Conv1dK7T256Tile,
    ) -> bool,
) -> Option<crate::kernels::conv1d_k7_t256_snake_epilogue::Conv1dK7T256Tile> {
    descriptor
        .measured_t256_snake_vec4_store_tile()
        .filter(|tile| contract_is_compatible(*tile))
}

/// Combine the measured decoder-family residue policy with its full preflight.
///
/// `None` guarantees that no residue allocation or dispatch occurs and that
/// the established T256/T128/legacy chain remains available unchanged.
fn select_compatible_conv1d_k7_residue_d1_dilation(
    descriptor: Conv1dK7Descriptor,
    contract_is_compatible: impl FnOnce(
        crate::kernels::conv1d_k7_residue_d1_snake::ResidueDilation,
    ) -> bool,
) -> Option<crate::kernels::conv1d_k7_residue_d1_snake::ResidueDilation> {
    descriptor
        .measured_residue_d1_dilation()
        .filter(|dilation| contract_is_compatible(*dilation))
}

/// Select the measured T128 act1 route after both physical preflights.
///
/// The five-binding epilogue wins when available. A failed alpha or resource
/// contract retains the accepted raw T128 launch followed by standalone Snake;
/// only a failed four-binding raw contract reaches the legacy tiled selector.
fn select_conv1d_k7_t128_snake_route(
    measured_tile: Option<crate::kernels::conv1d_k7_t128::Conv1dK7T128Tile>,
    fused_contract_compatible: bool,
    raw_contract_compatible: bool,
) -> Conv1dK7T128SnakeRoute {
    match (
        measured_tile,
        fused_contract_compatible,
        raw_contract_compatible,
    ) {
        (Some(tile), true, true) => Conv1dK7T128SnakeRoute::Fused(tile),
        (Some(tile), false, true) => Conv1dK7T128SnakeRoute::Materialized(tile),
        (None, _, _) | (Some(_), true, false) | (Some(_), false, false) => {
            Conv1dK7T128SnakeRoute::Legacy
        }
    }
}

/// Choose the standalone Conv1d tile without allowing an O64 preflight
/// failure to reach its asserting launcher.
fn select_conv1d_k7_standalone_tile(
    route: Conv1dK7Route,
    compatible_t128_tile: Option<crate::kernels::conv1d_k7_t128::Conv1dK7T128Tile>,
    o64_contract_compatible: bool,
    o32_supported: bool,
) -> Option<Conv1dK7StandaloneTile> {
    if route == Conv1dK7Route::BurnFallback {
        return None;
    }
    if let Some(tile) = compatible_t128_tile {
        return Some(Conv1dK7StandaloneTile::T128(tile));
    }
    match route {
        Conv1dK7Route::TiledO64Preferred(_) if o64_contract_compatible => {
            Some(Conv1dK7StandaloneTile::Output64)
        }
        Conv1dK7Route::TiledO32Preferred(_) | Conv1dK7Route::TiledO64Preferred(_)
            if o32_supported =>
        {
            Some(Conv1dK7StandaloneTile::Output32)
        }
        Conv1dK7Route::TiledO16(_)
        | Conv1dK7Route::TiledO32Preferred(_)
        | Conv1dK7Route::TiledO64Preferred(_) => Some(Conv1dK7StandaloneTile::Output16),
        Conv1dK7Route::BurnFallback => None,
    }
}

/// Choose a fused act1 epilogue without widening the accepted k=7 route.
///
/// O64-preferred shapes use this existing O32/O16 fused selector only when the
/// standalone O64 preflight is unavailable. O32-preferred shapes may fall back
/// to the portable O16 tile when the wider five-binding launch exceeds device
/// limits. An O16 route is never promoted to O32, preserving the measured
/// C=768/dilation=3 exception. `None` means the existing
/// `k7 -> standalone Snake` path must run unchanged.
fn select_conv1d_k7_snake_epilogue_tile(
    route: Conv1dK7Route,
    contract_compatible: bool,
    o16_supported: bool,
    o32_supported: bool,
) -> Option<crate::kernels::conv1d_k7_snake_epilogue::Conv1dK7SnakeTile> {
    use crate::kernels::conv1d_k7_snake_epilogue::Conv1dK7SnakeTile;

    if !contract_compatible {
        return None;
    }
    match route {
        Conv1dK7Route::TiledO32Preferred(_) | Conv1dK7Route::TiledO64Preferred(_)
            if o32_supported =>
        {
            Some(Conv1dK7SnakeTile::Output32)
        }
        Conv1dK7Route::TiledO16(_)
        | Conv1dK7Route::TiledO32Preferred(_)
        | Conv1dK7Route::TiledO64Preferred(_)
            if o16_supported =>
        {
            Some(Conv1dK7SnakeTile::Output16)
        }
        Conv1dK7Route::TiledO16(_)
        | Conv1dK7Route::TiledO32Preferred(_)
        | Conv1dK7Route::TiledO64Preferred(_)
        | Conv1dK7Route::BurnFallback => None,
    }
}

impl ResidualUnit {
    pub(crate) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let residual = x.clone();
        let y = self.act0.forward(x);
        let y = self.conv_dil.forward(y);
        let y = self.act1.forward(y);
        let y = self.conv_1x1.forward(y);
        y + residual
    }

    pub(crate) fn prepare_for_inference(&mut self) {
        if self.packed_conv_1x1_weight.is_none() {
            self.packed_conv_1x1_weight = Some(pack_pointwise_conv1d_weight(&self.conv_1x1));
        }
    }
}

/// Prepared input for the next decoder `ResidualUnit`.
///
/// The raw tensor remains the consumer's identity shortcut while `activated`
/// is the exact `act0` Snake result consumed by its k=7 branch.
#[derive(Debug)]
pub(crate) struct PreparedResidualPair {
    raw: Tensor<3>,
    activated: PreparedActivation,
}

/// Profile-only residual state whose shortcut and activation are both
/// physically contiguous NHWC. Keeping the pair opaque prevents mixing an
/// NHWC shortcut with the NCL-only production state by accident.
#[cfg(feature = "profile")]
#[derive(Debug)]
pub(crate) struct PreparedNhwcResidualPair {
    raw_nhwc: Tensor<3>,
    activated_nhwc: Tensor<3>,
}

impl PreparedResidualPair {
    /// Construct the exact raw-NCL/activated-NHWC contract emitted by a
    /// producer-side Snake fusion. Shape disagreement is rejected so an
    /// invalid shortcut/activation pair cannot enter a residual unit.
    #[cfg(feature = "profile")]
    pub(crate) fn from_ncl_nhwc(raw: Tensor<3>, activated: Tensor<3>) -> Option<Self> {
        let [batch, channels, length] = raw.dims();
        (activated.dims() == [batch, length, channels]).then_some(Self {
            raw,
            activated: PreparedActivation::Nhwc(activated),
        })
    }
}

/// The next residual unit can consume either ordinary NCL activation or the
/// exact compact residue layout required by its measured d3/d9 core.
#[derive(Debug)]
enum PreparedActivation {
    Ncl(Tensor<3>),
    Nhwc(Tensor<3>),
    ResiduePacked {
        tensor: Tensor<1>,
        dilation: crate::kernels::conv1d_k7_residue_d1_snake::ResidueDilation,
    },
}

/// Physical layout of the act1 output consumed by the pointwise projection.
#[derive(Clone, Debug)]
enum PointwiseActivation {
    Ncl(Tensor<3>),
    Nhwc(Tensor<3>),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum PointwiseRowsPolicy {
    #[default]
    Geometry,
    #[cfg(feature = "profile")]
    SingleRow,
}

impl PointwiseRowsPolicy {
    fn enabled(self, length: usize, channels: usize) -> bool {
        match self {
            Self::Geometry => length >= channels,
            #[cfg(feature = "profile")]
            Self::SingleRow => false,
        }
    }
}

impl PointwiseActivation {
    fn dims(&self) -> [usize; 3] {
        match self {
            Self::Ncl(tensor) => tensor.dims(),
            Self::Nhwc(tensor) => {
                let [batch, length, channels] = tensor.dims();
                [batch, channels, length]
            }
        }
    }

    fn into_ncl(self) -> Tensor<3> {
        match self {
            Self::Ncl(tensor) => tensor,
            Self::Nhwc(tensor) => {
                use burn::backend::wgpu::into_contiguous;
                use burn_cubecl::ops::permute_nhwc_to_nchw;
                let raw = tensor
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend");
                Tensor::from_primitive::<crate::WgpuRaw>(into_contiguous(permute_nhwc_to_nchw(raw)))
            }
        }
    }
}

impl ResidualUnit {
    /// Materialize the pointwise OIK parameter as physical KCO exactly once.
    ///
    /// A stale, non-contiguous, wrong-device, or non-f32 cache is replaced.
    /// If the source or resulting allocation misses any physical contract, the
    /// cache remains absent and every forward call retains the generic path.
    pub(crate) fn prepare_for_wgsl(&mut self) {
        self.prepare_for_wgsl_with_algorithm(CodecK7Algorithm::AccuracyApproved);
    }

    pub(crate) fn prepare_for_wgsl_with_algorithm(&mut self, algorithm: CodecK7Algorithm) {
        #[cfg(feature = "profile")]
        if algorithm != CodecK7Algorithm::CubeClImplicitGemmPreparedSelector {
            self.prepared_k7_selector = None;
        }
        self.act0.prepare_post_cast_epilogue();
        self.act1.prepare_post_cast_epilogue();
        #[cfg(feature = "profile")]
        if algorithm == CodecK7Algorithm::CubeClImplicitGemmPreparedEpilogue {
            self.act1.prepare_reciprocal_post_cast_epilogue();
        } else {
            self.act1.alpha_recip_epilogue_f32 = None;
        }
        if !self
            .packed_conv_1x1_weight
            .as_ref()
            .is_some_and(|packed| pointwise_wgpu_pack_is_compatible(&self.conv_1x1, packed))
        {
            self.packed_conv_1x1_weight = try_pack_pointwise_conv1d_weight_wgpu(&self.conv_1x1);
        }
        if use_single_storage_k7(algorithm, &self.conv_dil.weight.val()) {
            canonicalize_k7_weight_for_implicit_gemm(&mut self.conv_dil);
        }
        #[cfg(feature = "profile")]
        if matches!(
            algorithm,
            CodecK7Algorithm::CubeClImplicitGemmPreparedWeight(_)
        ) {
            self.prepared_k7_weight = prepare_k7_weight_for_implicit_gemm(&self.conv_dil);
        } else {
            self.prepared_k7_weight = None;
        }
        if !prepare_residue_layout(algorithm, &self.conv_dil.weight.val()) {
            // The accuracy-approved F16 route consumes the source OIK weight
            // directly through CubeCL implicit-GEMM. Retaining its packed
            // residue duplicate cannot accelerate that route.
            self.packed_conv_dil_weight_vectors = None;
        } else if !self
            .packed_conv_dil_weight_vectors
            .as_ref()
            .is_some_and(|packed| residue_weight_vector_pack_is_compatible(&self.conv_dil, packed))
        {
            self.packed_conv_dil_weight_vectors =
                try_pack_residue_conv1d_weight_vectors_wgpu(&self.conv_dil);
        }
    }

    #[cfg(feature = "profile")]
    pub(crate) fn prepare_for_wgsl_with_k7_selector(
        &mut self,
        manifest: &super::algorithm::K7SelectorManifest,
        output_length: usize,
    ) -> crate::Result<()> {
        use super::algorithm::K7SelectorProblem;

        self.prepared_k7_selector = None;
        self.prepare_for_wgsl_with_algorithm(CodecK7Algorithm::CubeClImplicitGemmPreparedSelector);
        let [output_channels, _, _] = self.conv_dil.weight.dims();
        let problem = K7SelectorProblem {
            output_length,
            output_channels,
            dilation: self.conv_dil.dilation,
        };
        self.prepared_k7_selector = Some(manifest.selection(problem)?);
        Ok(())
    }

    pub(crate) fn forward_wgsl(&self, x: Tensor<3>) -> Tensor<3> {
        let algorithm = CodecK7Algorithm::AccuracyApproved;
        let residual = x.clone();
        let activated = nvtx_range!(
            "codec_residual_snake_0",
            self.prepare_act0_for_algorithm(x, algorithm)
        );
        let y = self.dilated_from_prepared_with_algorithm(&residual, activated, algorithm);
        pointwise_residual_wgsl_or_fallback(
            &self.conv_1x1,
            self.packed_conv_1x1_weight.as_ref(),
            y,
            residual,
        )
    }

    /// Prepare the shortcut and act0 result consumed by this unit.
    pub(crate) fn prepare_input_wgsl(&self, raw: Tensor<3>) -> PreparedResidualPair {
        let activated =
            self.prepare_act0_for_algorithm(raw.clone(), CodecK7Algorithm::AccuracyApproved);
        PreparedResidualPair { raw, activated }
    }

    /// Execute this unit and prepare the next unit's shortcut/Snake pair.
    pub(crate) fn forward_wgsl_prepare_next(
        &self,
        x: Tensor<3>,
        next: &ResidualUnit,
    ) -> PreparedResidualPair {
        let algorithm = CodecK7Algorithm::AccuracyApproved;
        let residual = x.clone();
        let activated = nvtx_range!(
            "codec_residual_snake_0",
            self.prepare_act0_for_algorithm(x, algorithm)
        );
        let y = self.dilated_from_prepared_with_algorithm(&residual, activated, algorithm);
        self.prepare_next_after_pointwise(y, residual, next, algorithm)
    }

    /// Consume a prepared shortcut/Snake pair without recomputing `act0`.
    pub(crate) fn forward_wgsl_from_prepared(&self, pair: PreparedResidualPair) -> Tensor<3> {
        let y = self.dilated_from_prepared_with_algorithm(
            &pair.raw,
            pair.activated,
            CodecK7Algorithm::AccuracyApproved,
        );
        pointwise_residual_wgsl_or_fallback(
            &self.conv_1x1,
            self.packed_conv_1x1_weight.as_ref(),
            y,
            pair.raw,
        )
    }

    /// Profile-only fusion of this final pointwise residual with the complete
    /// watermark-less output head. The caller receives `None` on any dtype,
    /// layout, shape, device, or resource contract miss.
    #[cfg(feature = "profile")]
    pub(crate) fn try_forward_wgsl_from_prepared_fused_wm_head(
        &self,
        pair: PreparedResidualPair,
        head_act: &Snake1d,
        head_conv: &Conv1d,
    ) -> Option<Tensor<3>> {
        let y = self.dilated_from_prepared_with_algorithm(
            &pair.raw,
            pair.activated,
            CodecK7Algorithm::AccuracyApproved,
        );
        let PointwiseActivation::Nhwc(y) = y else {
            return None;
        };
        let pointwise_bias = self.conv_1x1.bias.as_ref()?;
        let head_bias = head_conv.bias.as_ref()?;
        let output = crate::kernels::wm_head_pointwise_fused::try_wm_head_pointwise_fused_f16(
            y.try_into_primitive::<crate::WgpuRaw>().ok()?,
            self.conv_1x1
                .weight
                .val()
                .try_into_primitive::<crate::WgpuRaw>()
                .ok()?,
            pointwise_bias
                .val()
                .try_into_primitive::<crate::WgpuRaw>()
                .ok()?,
            pair.raw.try_into_primitive::<crate::WgpuRaw>().ok()?,
            head_act
                .alpha
                .val()
                .try_into_primitive::<crate::WgpuRaw>()
                .ok()?,
            head_conv
                .weight
                .val()
                .try_into_primitive::<crate::WgpuRaw>()
                .ok()?,
            head_bias
                .val()
                .try_into_primitive::<crate::WgpuRaw>()
                .ok()?,
        )?;
        Some(Tensor::from_primitive::<crate::WgpuRaw>(output))
    }

    /// Consume one prepared pair and produce the following unit's pair.
    pub(crate) fn forward_wgsl_from_prepared_prepare_next(
        &self,
        pair: PreparedResidualPair,
        next: &ResidualUnit,
    ) -> PreparedResidualPair {
        let algorithm = CodecK7Algorithm::AccuracyApproved;
        let y = self.dilated_from_prepared_with_algorithm(&pair.raw, pair.activated, algorithm);
        self.prepare_next_after_pointwise(y, pair.raw, next, algorithm)
    }

    fn prepare_act0_for_algorithm(
        &self,
        input: Tensor<3>,
        algorithm: CodecK7Algorithm,
    ) -> PreparedActivation {
        if use_nhwc_prepared_activation(algorithm, &self.conv_dil.weight.val()) {
            let fallback = input.clone();
            self.act0
                .forward_nchw_to_nhwc_wgsl(input)
                .map(PreparedActivation::Nhwc)
                .unwrap_or_else(|| PreparedActivation::Ncl(self.act0.forward_wgsl(fallback)))
        } else {
            PreparedActivation::Ncl(self.act0.forward_wgsl(input))
        }
    }

    fn prepare_next_after_pointwise(
        &self,
        input: PointwiseActivation,
        residual: Tensor<3>,
        next: &ResidualUnit,
        algorithm: CodecK7Algorithm,
    ) -> PreparedResidualPair {
        if use_nhwc_prepared_activation(algorithm, &next.conv_dil.weight.val()) {
            if let Some(pair) = cubek_pointwise_accumulator_store_pair(
                &self.conv_1x1,
                input.clone(),
                residual.clone(),
                next,
                PointwiseRowsPolicy::Geometry,
            ) {
                return pair;
            }
            pointwise_residual_snake_nhwc_pair_wgsl_or_fallback(
                &self.conv_1x1,
                self.packed_conv_1x1_weight.as_ref(),
                input,
                residual,
                next,
            )
        } else {
            pointwise_residual_snake_pair_wgsl_or_fallback_with_layout(
                &self.conv_1x1,
                self.packed_conv_1x1_weight.as_ref(),
                input.into_ncl(),
                residual,
                &next.act0,
                Some(next),
                prepare_residue_layout(algorithm, &self.conv_1x1.weight.val()),
            )
        }
    }

    /// Consume the final residual pair and produce only the activation needed
    /// by the following decoder block's upsampler.
    pub(crate) fn forward_wgsl_from_prepared_prepare_block(
        &self,
        pair: PreparedResidualPair,
        next_block_act: &Snake1d,
    ) -> Tensor<3> {
        let algorithm = CodecK7Algorithm::AccuracyApproved;
        let y = self.dilated_from_prepared_with_algorithm(&pair.raw, pair.activated, algorithm);
        pointwise_residual_snake_activated_wgsl_or_fallback(
            &self.conv_1x1,
            self.packed_conv_1x1_weight.as_ref(),
            y,
            pair.raw,
            next_block_act,
        )
    }

    /// Profile-only CubeK CMMA replacement for the activated-only block
    /// boundary pointwise producer. Contract misses return `None` so the
    /// measured caller can fail closed instead of silently changing routes.
    #[cfg(feature = "profile")]
    pub(crate) fn try_forward_wgsl_from_prepared_prepare_block_accumulator(
        &self,
        pair: PreparedResidualPair,
        next_block_act: &Snake1d,
    ) -> Option<Tensor<3>> {
        let algorithm = CodecK7Algorithm::AccuracyApproved;
        let y = self.dilated_from_prepared_with_algorithm(&pair.raw, pair.activated, algorithm);
        cubek_pointwise_accumulator_snake_activated(&self.conv_1x1, y, pair.raw, next_block_act)
    }

    /// Enter the profile-only all-NHWC residual state from a block-local NCL
    /// shortcut. Failure is closed before exposing a partially typed state.
    #[cfg(feature = "profile")]
    pub(crate) fn try_forward_wgsl_prepare_nhwc_state(
        &self,
        raw_ncl: Tensor<3>,
        next: &ResidualUnit,
    ) -> Option<PreparedNhwcResidualPair> {
        let activated_nhwc = self.act0.forward_nchw_to_nhwc_wgsl(raw_ncl.clone())?;
        let pointwise_nhwc = self.try_k7_snake_from_nhwc(activated_nhwc)?;
        pointwise_residual_nhwc_outputs(
            &self.conv_1x1,
            self.packed_conv_1x1_weight.as_ref(),
            pointwise_nhwc,
            ProfileResidualLayout::Ncl(raw_ncl),
            next,
        )
    }

    /// Advance one residual unit without materializing NCL at the shortcut or
    /// activation boundary.
    #[cfg(feature = "profile")]
    pub(crate) fn try_forward_wgsl_nhwc_state_prepare_next(
        &self,
        state: PreparedNhwcResidualPair,
        next: &ResidualUnit,
    ) -> Option<PreparedNhwcResidualPair> {
        let pointwise_nhwc = self.try_k7_snake_from_nhwc(state.activated_nhwc)?;
        pointwise_residual_nhwc_outputs(
            &self.conv_1x1,
            self.packed_conv_1x1_weight.as_ref(),
            pointwise_nhwc,
            ProfileResidualLayout::Nhwc(state.raw_nhwc),
            next,
        )
    }

    /// Leave the final unit as the post-storage-cast NCL activation consumed
    /// by the next decoder block.
    #[cfg(feature = "profile")]
    pub(crate) fn try_forward_wgsl_nhwc_state_prepare_block(
        &self,
        state: PreparedNhwcResidualPair,
        next_block_act: &Snake1d,
    ) -> Option<Tensor<3>> {
        let pointwise_nhwc = self.try_k7_snake_from_nhwc(state.activated_nhwc)?;
        let inputs = direct_pointwise_nhwc_inputs(
            &self.conv_1x1,
            self.packed_conv_1x1_weight.as_ref(),
            pointwise_nhwc,
            ProfileResidualLayout::Nhwc(state.raw_nhwc),
        )?;
        let output = crate::kernels::pointwise_residual_direct_tiled::pointwise_residual_direct_snake_activated_wgsl(
            inputs,
            next_block_act.alpha.val().try_into_primitive::<crate::WgpuRaw>().ok()?,
            crate::kernels::pointwise_residual_direct_tiled::PointwiseKTile::PRODUCTION,
        )
        .ok()?;
        Some(Tensor::from_primitive::<crate::WgpuRaw>(output))
    }

    /// Leave the final decoder unit as raw NCL for the watermark head.
    #[cfg(feature = "profile")]
    pub(crate) fn try_forward_wgsl_nhwc_state_raw(
        &self,
        state: PreparedNhwcResidualPair,
    ) -> Option<Tensor<3>> {
        let pointwise_nhwc = self.try_k7_snake_from_nhwc(state.activated_nhwc)?;
        let inputs = direct_pointwise_nhwc_inputs(
            &self.conv_1x1,
            self.packed_conv_1x1_weight.as_ref(),
            pointwise_nhwc,
            ProfileResidualLayout::Nhwc(state.raw_nhwc),
        )?;
        let output =
            crate::kernels::pointwise_residual_direct_tiled::pointwise_residual_direct_raw_wgsl(
                inputs,
                crate::kernels::pointwise_residual_direct_tiled::PointwiseKTile::PRODUCTION,
            )
            .ok()?;
        Some(Tensor::from_primitive::<crate::WgpuRaw>(output))
    }

    #[cfg(feature = "profile")]
    fn try_k7_snake_from_nhwc(&self, activated_nhwc: Tensor<3>) -> Option<Tensor<3>> {
        implicit_gemm_nhwc_dilated_conv1d_then_snake_wgsl(
            &self.conv_dil,
            &self.act1,
            activated_nhwc,
            None,
            false,
            K7MultiRowsSelection::GeometrySelected,
            false,
            false,
        )
    }

    fn dilated_from_prepared_with_algorithm(
        &self,
        residual: &Tensor<3>,
        activated: PreparedActivation,
        algorithm: CodecK7Algorithm,
    ) -> PointwiseActivation {
        if use_implicit_gemm(algorithm, &self.conv_dil.weight.val()) {
            let activated = match activated {
                PreparedActivation::Ncl(activated) => activated,
                PreparedActivation::Nhwc(activated)
                    if use_nhwc_prepared_activation(algorithm, &self.conv_dil.weight.val()) =>
                {
                    let candidate = match algorithm {
                        #[cfg(feature = "profile")]
                        CodecK7Algorithm::CubeClImplicitGemmAsync => {
                            custom_implicit_gemm_dilated_conv1d_then_snake_wgsl(
                                &self.conv_dil,
                                &self.act1,
                                activated,
                                cubek_convolution::ConvAlgorithm::SimpleAsyncCyclic,
                                true,
                            )
                        }
                        #[cfg(feature = "profile")]
                        CodecK7Algorithm::CubeClImplicitGemmSyncStrided => {
                            custom_implicit_gemm_dilated_conv1d_then_snake_wgsl(
                                &self.conv_dil,
                                &self.act1,
                                activated,
                                cubek_convolution::ConvAlgorithm::SimpleSyncStrided,
                                true,
                            )
                        }
                        #[cfg(feature = "profile")]
                        CodecK7Algorithm::CubeClImplicitGemmAsyncStrided => {
                            custom_implicit_gemm_dilated_conv1d_then_snake_wgsl(
                                &self.conv_dil,
                                &self.act1,
                                activated,
                                cubek_convolution::ConvAlgorithm::SimpleAsyncStrided,
                                true,
                            )
                        }
                        _ => {
                            #[cfg(feature = "profile")]
                            let selection = if algorithm
                                == CodecK7Algorithm::CubeClImplicitGemmPreparedSelector
                            {
                                let Some(choice) = self.prepared_k7_selector else {
                                    return PointwiseActivation::Ncl(
                                        dilated_conv1d_act1_with_algorithm(
                                            &self.conv_dil,
                                            &self.act1,
                                            self.packed_conv_dil_weight_vectors.as_ref(),
                                            self.act0.forward_wgsl(residual.clone()),
                                            CodecK7Algorithm::PackedResidue,
                                        ),
                                    );
                                };
                                K7MultiRowsSelection::Prepared(choice)
                            } else {
                                multi_rows_k7_selection(algorithm)
                            };
                            #[cfg(not(feature = "profile"))]
                            let selection = multi_rows_k7_selection(algorithm);
                            #[cfg(feature = "profile")]
                            let halo_loader =
                                algorithm == CodecK7Algorithm::CubeClImplicitGemmK7Halo;
                            #[cfg(not(feature = "profile"))]
                            let halo_loader = false;
                            implicit_gemm_nhwc_dilated_conv1d_then_snake_wgsl(
                                &self.conv_dil,
                                &self.act1,
                                activated,
                                prepared_k7_weight_for_algorithm(
                                    algorithm,
                                    self.prepared_k7_weight.as_ref(),
                                ),
                                use_direct_oik_weight(algorithm),
                                selection,
                                halo_loader,
                                use_prepared_snake_epilogue(algorithm),
                            )
                        }
                    };
                    return candidate.map(PointwiseActivation::Nhwc).unwrap_or_else(|| {
                        PointwiseActivation::Ncl(dilated_conv1d_act1_with_algorithm(
                            &self.conv_dil,
                            &self.act1,
                            self.packed_conv_dil_weight_vectors.as_ref(),
                            self.act0.forward_wgsl(residual.clone()),
                            algorithm,
                        ))
                    });
                }
                PreparedActivation::Nhwc(_) => self.act0.forward_wgsl(residual.clone()),
                PreparedActivation::ResiduePacked { .. } => {
                    self.act0.forward_wgsl(residual.clone())
                }
            };
            return PointwiseActivation::Ncl(dilated_conv1d_act1_with_algorithm(
                &self.conv_dil,
                &self.act1,
                self.packed_conv_dil_weight_vectors.as_ref(),
                activated,
                algorithm,
            ));
        }
        match activated {
            PreparedActivation::Ncl(activated) => {
                PointwiseActivation::Ncl(dilated_conv1d_act1_wgsl_or_fallback(
                    &self.conv_dil,
                    &self.act1,
                    self.packed_conv_dil_weight_vectors.as_ref(),
                    activated,
                ))
            }
            PreparedActivation::Nhwc(_) => {
                let activated = self.act0.forward_wgsl(residual.clone());
                PointwiseActivation::Ncl(dilated_conv1d_act1_wgsl_or_fallback(
                    &self.conv_dil,
                    &self.act1,
                    self.packed_conv_dil_weight_vectors.as_ref(),
                    activated,
                ))
            }
            PreparedActivation::ResiduePacked { tensor, dilation } => {
                let output = self
                    .packed_conv_dil_weight_vectors
                    .as_ref()
                    .zip(self.conv_dil.bias.as_ref())
                    .and_then(|(weight, bias)| {
                        crate::kernels::conv1d_k7_residue_d1_snake::conv1d_k7_residue_d1_snake_from_packed_wgsl(
                            tensor.try_into_primitive::<crate::WgpuRaw>().expect("tensor must use WGPU raw backend"),
                            weight.clone().try_into_primitive::<crate::WgpuRaw>().expect("tensor must use WGPU raw backend"),
                            bias.val().try_into_primitive::<crate::WgpuRaw>().expect("tensor must use WGPU raw backend"),
                            self.act1.alpha.val().try_into_primitive::<crate::WgpuRaw>().expect("tensor must use WGPU raw backend"),
                            dilation,
                            residual.dims()[1],
                            residual.dims()[2],
                        )
                    });
                if let Some(output) = output {
                    return PointwiseActivation::Ncl(Tensor::from_primitive::<crate::WgpuRaw>(
                        output,
                    ));
                }
                let activated = self.act0.forward_wgsl(residual.clone());
                PointwiseActivation::Ncl(dilated_conv1d_act1_wgsl_or_fallback(
                    &self.conv_dil,
                    &self.act1,
                    self.packed_conv_dil_weight_vectors.as_ref(),
                    activated,
                ))
            }
        }
    }

    #[cfg(feature = "profile")]
    fn dilated_from_prepared_profiled(
        &self,
        residual: &Tensor<3>,
        activated: PreparedActivation,
        algorithm: CodecK7Algorithm,
    ) -> PointwiseActivation {
        self.dilated_from_prepared_with_algorithm(residual, activated, algorithm)
    }

    #[cfg(feature = "profile")]
    pub(crate) fn forward_wgsl_profiled_prepare_next<P>(
        &self,
        x: Tensor<3>,
        next: &ResidualUnit,
        labels: [&'static str; 3],
        algorithm: CodecK7Algorithm,
        pointwise_algorithm: CodecPointwiseAlgorithm,
        profiler: &mut P,
    ) -> Result<PreparedResidualPair, P::Error>
    where
        P: CodecStageProfiler,
    {
        let residual = x.clone();
        let activated = profile_residual_stage(
            labels[0],
            || {
                if use_nhwc_prepared_activation(algorithm, &self.conv_dil.weight.val()) {
                    let fallback = x.clone();
                    self.act0
                        .forward_nchw_to_nhwc_wgsl(x)
                        .map(PreparedActivation::Nhwc)
                        .unwrap_or_else(|| {
                            PreparedActivation::Ncl(self.act0.forward_wgsl(fallback))
                        })
                } else {
                    PreparedActivation::Ncl(self.act0.forward_wgsl(x))
                }
            },
            profiler,
        )?;
        let y = profile_residual_stage(
            labels[1],
            || self.dilated_from_prepared_profiled(&residual, activated, algorithm),
            profiler,
        )?;
        profile_residual_stage(
            labels[2],
            || {
                pointwise_residual_snake_pair_with_algorithm(
                    &self.conv_1x1,
                    self.packed_conv_1x1_weight.as_ref(),
                    y,
                    residual,
                    next,
                    algorithm,
                    pointwise_algorithm,
                )
            },
            profiler,
        )
    }

    #[cfg(feature = "profile")]
    pub(crate) fn forward_wgsl_profiled_from_prepared_prepare_next<P>(
        &self,
        pair: PreparedResidualPair,
        next: &ResidualUnit,
        labels: [&'static str; 2],
        algorithm: CodecK7Algorithm,
        pointwise_algorithm: CodecPointwiseAlgorithm,
        profiler: &mut P,
    ) -> Result<PreparedResidualPair, P::Error>
    where
        P: CodecStageProfiler,
    {
        let PreparedResidualPair { raw, activated } = pair;
        let y = profile_residual_stage(
            labels[0],
            || self.dilated_from_prepared_profiled(&raw, activated, algorithm),
            profiler,
        )?;
        profile_residual_stage(
            labels[1],
            || {
                pointwise_residual_snake_pair_with_algorithm(
                    &self.conv_1x1,
                    self.packed_conv_1x1_weight.as_ref(),
                    y,
                    raw,
                    next,
                    algorithm,
                    pointwise_algorithm,
                )
            },
            profiler,
        )
    }

    #[cfg(feature = "profile")]
    pub(crate) fn forward_wgsl_profiled_from_prepared<P>(
        &self,
        pair: PreparedResidualPair,
        labels: [&'static str; 2],
        algorithm: CodecK7Algorithm,
        pointwise_algorithm: CodecPointwiseAlgorithm,
        profiler: &mut P,
    ) -> Result<Tensor<3>, P::Error>
    where
        P: CodecStageProfiler,
    {
        let PreparedResidualPair { raw, activated } = pair;
        let y = profile_residual_stage(
            labels[0],
            || self.dilated_from_prepared_profiled(&raw, activated, algorithm),
            profiler,
        )?;
        profile_residual_stage(
            labels[1],
            || {
                pointwise_residual_with_algorithm(
                    &self.conv_1x1,
                    self.packed_conv_1x1_weight.as_ref(),
                    y,
                    raw,
                    pointwise_algorithm,
                )
            },
            profiler,
        )
    }
}

/// Canonicalize one k=7 weight allocation to physical OKI while retaining the
/// public/logical OIK shape as a zero-copy stride view.
///
/// CubeK consumes the OKI view and therefore no longer materializes a layout
/// copy per request. The fallback Conv1d path sees the OIK view backed by the
/// same storage, so this does not retain a second ~32 MiB model-wide copy.
fn canonicalize_k7_weight_for_implicit_gemm(conv: &mut Conv1d) {
    use burn_backend::cubecl::dtype_to_storage_type;
    use burn_cubecl::ops::permute_nchw_to_nhwc;
    use cubecl::std::tensor::into_contiguous_pitched;

    if conv.kernel_size != 7 || conv.groups != 1 {
        return;
    }
    let Ok(weight) = conv.weight.val().try_into_primitive::<crate::WgpuRaw>() else {
        return;
    };
    let mut oki = permute_nchw_to_nhwc(weight);
    let prepared = into_contiguous_pitched(
        &oki.client,
        oki.clone().binding(),
        dtype_to_storage_type(oki.dtype),
    );
    oki.handle = prepared.handle;
    oki.meta = prepared.metadata;
    let oki = Tensor::from_primitive::<crate::WgpuRaw>(oki);
    let logical_oik = oki.swap_dims(1, 2);
    conv.weight = Param::initialized(ParamId::new(), logical_oik);
}

#[cfg(feature = "profile")]
fn prepare_k7_weight_for_implicit_gemm(conv: &Conv1d) -> Option<PreparedK7Weight> {
    use burn_backend::cubecl::dtype_to_storage_type;
    use burn_cubecl::ops::permute_nchw_to_nhwc;
    use cubecl::std::tensor::into_contiguous_pitched;

    if conv.kernel_size != 7 || conv.groups != 1 {
        return None;
    }
    let source_oik_shape = conv.weight.dims();
    let mut oki = permute_nchw_to_nhwc(
        conv.weight
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .ok()?,
    );
    let prepared = into_contiguous_pitched(
        &oki.client,
        oki.clone().binding(),
        dtype_to_storage_type(oki.dtype),
    );
    oki.handle = prepared.handle;
    oki.meta = prepared.metadata;
    let physical_oki_strides: [usize; 3] = oki.meta.strides()[..].try_into().ok()?;
    let physical_shape = oki.meta.shape().dims::<3>();
    if physical_shape
        != [
            source_oik_shape[0],
            source_oik_shape[2],
            source_oik_shape[1],
        ]
        || physical_oki_strides[2] != 1
    {
        return None;
    }
    let physical_bytes = oki.handle.size_in_used() as usize;
    Some(PreparedK7Weight {
        oki: Tensor::from_primitive::<crate::WgpuRaw>(oki),
        source_oik_shape,
        physical_oki_strides,
        physical_bytes,
    })
}

#[cfg(feature = "profile")]
fn prepared_k7_weight_for_algorithm(
    algorithm: CodecK7Algorithm,
    prepared: Option<&PreparedK7Weight>,
) -> Option<&PreparedK7Weight> {
    let CodecK7Algorithm::CubeClImplicitGemmPreparedWeight(policy) = algorithm else {
        return None;
    };
    prepared.filter(|weight| policy.accepts(weight.physical_bytes))
}

#[cfg(feature = "profile")]
fn use_direct_oik_weight(algorithm: CodecK7Algorithm) -> bool {
    matches!(algorithm, CodecK7Algorithm::CubeClImplicitGemmDirectOik)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum K7MultiRowsSelection {
    #[default]
    Disabled,
    #[cfg(any(feature = "profile", test))]
    Forced,
    GeometrySelected,
    #[cfg(feature = "profile")]
    Autotuned,
    #[cfg(feature = "profile")]
    Prepared(super::algorithm::K7SelectorChoice),
}

impl K7MultiRowsSelection {
    fn enabled(self, output_length: usize, output_channels: usize) -> bool {
        match self {
            Self::Disabled => false,
            #[cfg(any(feature = "profile", test))]
            Self::Forced => true,
            Self::GeometrySelected => output_length >= output_channels && output_channels >= 384,
            #[cfg(feature = "profile")]
            Self::Autotuned => unreachable!("autotuned k7 selection launches through LocalTuner"),
            #[cfg(feature = "profile")]
            Self::Prepared(super::algorithm::K7SelectorChoice::SingleRow) => false,
            #[cfg(feature = "profile")]
            Self::Prepared(super::algorithm::K7SelectorChoice::MultiRow) => true,
            #[cfg(feature = "profile")]
            Self::Prepared(_) => unreachable!(
                "non-geometry prepared k7 selection launches its resolved CubeK policy directly"
            ),
        }
    }
}

#[cfg(feature = "profile")]
fn multi_rows_k7_selection(algorithm: CodecK7Algorithm) -> K7MultiRowsSelection {
    match algorithm {
        CodecK7Algorithm::CubeClImplicitGemmMultiRows => K7MultiRowsSelection::Forced,
        CodecK7Algorithm::CubeClImplicitGemmAutotuned => K7MultiRowsSelection::Autotuned,
        CodecK7Algorithm::CubeClImplicitGemmPreparedSelector => K7MultiRowsSelection::Disabled,
        CodecK7Algorithm::AccuracyApproved
        | CodecK7Algorithm::CubeClImplicitGemmGeometrySelectedMultiRows
        | CodecK7Algorithm::CubeClImplicitGemmPreparedEpilogue => {
            K7MultiRowsSelection::GeometrySelected
        }
        _ => K7MultiRowsSelection::Disabled,
    }
}

#[cfg(not(feature = "profile"))]
fn use_direct_oik_weight(_algorithm: CodecK7Algorithm) -> bool {
    false
}

#[cfg(not(feature = "profile"))]
fn multi_rows_k7_selection(algorithm: CodecK7Algorithm) -> K7MultiRowsSelection {
    match algorithm {
        CodecK7Algorithm::AccuracyApproved => K7MultiRowsSelection::GeometrySelected,
        _ => K7MultiRowsSelection::Disabled,
    }
}

#[cfg(not(feature = "profile"))]
fn prepared_k7_weight_for_algorithm(
    _algorithm: CodecK7Algorithm,
    _prepared: Option<&PreparedK7Weight>,
) -> Option<&PreparedK7Weight> {
    None
}

#[cfg(feature = "profile")]
impl ResidualUnit {
    pub(crate) fn profile_k7_weight_repack(
        &self,
        label: &'static str,
    ) -> crate::error::Result<super::algorithm::K7WeightRepackReceipt> {
        use burn_backend::cubecl::dtype_to_storage_type;
        use burn_cubecl::ops::permute_nchw_to_nhwc;
        use cubecl::{
            future, profile::TimingMethod, std::tensor::into_contiguous_pitched,
            tensor_vector_size_parallel,
        };

        let source = self
            .conv_dil
            .weight
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .map_err(|_| crate::error::IrodoriError::Profile("k7 weight is not WGPU".into()))?;
        let logical = permute_nchw_to_nhwc(source);
        let source_oik_shape = self.conv_dil.weight.dims();
        let logical_oki_strides = logical.meta.strides()[..]
            .try_into()
            .map_err(|_| crate::error::IrodoriError::Profile("k7 weight rank changed".into()))?;
        let dtype = dtype_to_storage_type(logical.dtype);
        let client = logical.client.clone();
        let copy_client = client.clone();
        let logical_binding = logical.clone().binding();
        let (prepared, duration) = client
            .profile(
                move || into_contiguous_pitched(&copy_client, logical_binding, dtype),
                label,
            )
            .map_err(|error| crate::error::IrodoriError::Profile(error.to_string()))?;
        let used_device_timestamps = duration.timing_method() == TimingMethod::Device;
        let device_duration_ms = future::block_on(duration.resolve())
            .duration()
            .as_secs_f64()
            * 1_000.0;
        let materialized_oki_strides = prepared.metadata.strides()[..]
            .try_into()
            .map_err(|_| crate::error::IrodoriError::Profile("prepared k7 rank changed".into()))?;
        let supported_vectors = || logical.client.io_optimized_vector_sizes(dtype.size());
        let logical_rhs_vector_size = tensor_vector_size_parallel(
            supported_vectors(),
            logical.meta.shape(),
            logical.meta.strides(),
            2,
        );
        let materialized_rhs_vector_size = tensor_vector_size_parallel(
            supported_vectors(),
            prepared.metadata.shape(),
            prepared.metadata.strides(),
            2,
        );
        Ok(super::algorithm::K7WeightRepackReceipt {
            label,
            source_oik_shape,
            logical_oki_strides,
            materialized_oki_strides,
            logical_rhs_vector_size,
            materialized_rhs_vector_size,
            materialized_bytes: prepared.handle.size_in_used() as usize,
            device_duration_ms,
            used_device_timestamps,
        })
    }
}

#[cfg(feature = "profile")]
fn profile_residual_stage<T, P>(
    label: &'static str,
    operation: impl FnOnce() -> T + Send,
    profiler: &mut P,
) -> Result<T, P::Error>
where
    T: Send + 'static,
    P: CodecStageProfiler,
{
    profiler.profile(label, operation)
}

fn existing_pointwise_residual_wgsl(
    conv: &Conv1d,
    packed_weight: Option<&Tensor<3>>,
    input: Tensor<3>,
    residual: Tensor<3>,
) -> Tensor<3> {
    let output = nvtx_range!(
        "codec_residual_conv_1x1",
        pointwise_conv1d_with_weight(conv, packed_weight, input)
    );
    nvtx_range!("codec_residual_add", output + residual)
}

fn cube_tensor_has_exact_layout<const D: usize>(
    tensor: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    shape: [usize; D],
    strides: [usize; D],
) -> bool {
    tensor.meta.num_dims() == D
        && tensor.meta.shape().dims::<D>() == shape
        && &tensor.meta.strides()[..] == strides.as_slice()
        && tensor.is_contiguous()
}

fn wgsl_float_dtype_is_supported(dtype: burn::tensor::DType) -> bool {
    matches!(dtype, burn::tensor::DType::F32 | burn::tensor::DType::F16)
}

fn pointwise_wgpu_pack_is_compatible(conv: &Conv1d, packed_weight_kco: &Tensor<3>) -> bool {
    let [output_channels, input_channels, kernel] = conv.weight.dims();
    if output_channels != input_channels || kernel != 1 {
        return false;
    }
    let Some(weight_elements) = input_channels.checked_mul(output_channels) else {
        return false;
    };
    let source_oik = conv
        .weight
        .val()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    let packed_kco = packed_weight_kco
        .clone()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    wgsl_float_dtype_is_supported(source_oik.dtype)
        && packed_kco.dtype == source_oik.dtype
        && source_oik.device == packed_kco.device
        && cube_tensor_has_exact_layout(
            &source_oik,
            [output_channels, input_channels, 1],
            [input_channels, 1, 1],
        )
        && cube_tensor_has_exact_layout(
            &packed_kco,
            [1, input_channels, output_channels],
            [weight_elements, output_channels, 1],
        )
}

fn try_pack_pointwise_conv1d_weight_wgpu(conv: &Conv1d) -> Option<Tensor<3>> {
    use burn::backend::wgpu::into_contiguous;

    if conv.kernel_size != 1 || conv.stride != 1 || conv.dilation != 1 || conv.groups != 1 {
        return None;
    }
    let [output_channels, input_channels, kernel] = conv.weight.dims();
    if output_channels != input_channels || kernel != 1 {
        return None;
    }
    let weight_elements = input_channels.checked_mul(output_channels)?;
    let source = conv.weight.val();
    let source_raw = source
        .clone()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    if !wgsl_float_dtype_is_supported(source_raw.dtype)
        || !cube_tensor_has_exact_layout(
            &source_raw,
            [output_channels, input_channels, 1],
            [input_channels, 1, 1],
        )
    {
        return None;
    }

    // `transpose` is only a logical KCO view. The explicit CubeCL copy is the
    // materialization contract; an elementwise no-op is intentionally not used.
    let logical_kco = source.squeeze_dim::<2>(2).transpose().unsqueeze_dim::<3>(0);
    let logical_raw = logical_kco
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    if logical_raw.dtype != source_raw.dtype
        || logical_raw.device != source_raw.device
        || logical_raw.meta.num_dims() != 3
        || logical_raw.meta.shape().dims::<3>() != [1, input_channels, output_channels]
        || &logical_raw.meta.strides()[..] != [weight_elements, 1, input_channels].as_slice()
        || logical_raw.is_contiguous()
    {
        return None;
    }

    let packed_raw = into_contiguous(logical_raw);
    if packed_raw.dtype != source_raw.dtype
        || packed_raw.device != source_raw.device
        || !cube_tensor_has_exact_layout(
            &packed_raw,
            [1, input_channels, output_channels],
            [weight_elements, output_channels, 1],
        )
    {
        return None;
    }
    let packed = Tensor::from_primitive::<crate::WgpuRaw>(packed_raw);
    pointwise_wgpu_pack_is_compatible(conv, &packed).then_some(packed)
}

fn residue_weight_vector_pack_is_compatible(conv: &Conv1d, packed_weight: &Tensor<3>) -> bool {
    let [output_channels, input_channels, kernel] = conv.weight.dims();
    if output_channels != input_channels
        || !matches!(output_channels, 96 | 192 | 384)
        || kernel != 7
    {
        return false;
    }
    let source = conv
        .weight
        .val()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    let packed = packed_weight
        .clone()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    wgsl_float_dtype_is_supported(source.dtype)
        && packed.dtype == source.dtype
        && source.device == packed.device
        && cube_tensor_has_exact_layout(
            &source,
            [output_channels, input_channels, kernel],
            [input_channels * kernel, kernel, 1],
        )
        && cube_tensor_has_exact_layout(
            &packed,
            [input_channels, kernel, output_channels],
            [kernel * output_channels, output_channels, 1],
        )
        && packed
            .handle
            .clone()
            .binding()
            .offset_start
            .unwrap_or(0)
            .is_multiple_of(8)
}

fn try_pack_residue_conv1d_weight_vectors_wgpu(conv: &Conv1d) -> Option<Tensor<3>> {
    if conv.kernel_size != 7 || conv.stride != 1 || conv.groups != 1 {
        return None;
    }
    let packed =
        crate::kernels::conv1d_k7_residue_d1_snake::try_pack_conv1d_k7_residue_weight_vectors_wgsl(
            conv.weight
                .val()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
        )?;
    let packed = Tensor::from_primitive::<crate::WgpuRaw>(packed);
    residue_weight_vector_pack_is_compatible(conv, &packed).then_some(packed)
}

fn pointwise_direct_source_weight_is_compatible(
    conv: &Conv1d,
    reference: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    channels: usize,
) -> bool {
    let weight = conv
        .weight
        .val()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    wgsl_float_dtype_is_supported(weight.dtype)
        && weight.dtype == reference.dtype
        && weight.device == reference.device
        && cube_tensor_has_exact_layout(&weight, [channels, channels, 1], [channels, 1, 1])
}

fn pointwise_residual_contract_is_compatible(
    input_ncl: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    weight_oik: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    packed_weight_nkk: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    bias: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    residual_ncl: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    channels: usize,
    length: usize,
) -> bool {
    let Some(elements) = channels.checked_mul(length) else {
        return false;
    };
    let Some(weight_elements) = channels.checked_mul(channels) else {
        return false;
    };
    let tensors = [input_ncl, weight_oik, packed_weight_nkk, bias, residual_ncl];
    wgsl_float_dtype_is_supported(input_ncl.dtype)
        && tensors
            .into_iter()
            .all(|tensor| tensor.dtype == input_ncl.dtype && tensor.device == input_ncl.device)
        && cube_tensor_has_exact_layout(
            input_ncl,
            [1, channels, length],
            [elements, length, 1],
        )
        && cube_tensor_has_exact_layout(
            weight_oik,
            [channels, channels, 1],
            [channels, 1, 1],
        )
        && cube_tensor_has_exact_layout(
            packed_weight_nkk,
            [1, channels, channels],
            [weight_elements, channels, 1],
        )
        && cube_tensor_has_exact_layout(bias, [channels], [1])
        && cube_tensor_has_exact_layout(
            residual_ncl,
            [1, channels, length],
            [elements, length, 1],
        )
        && crate::kernels::pointwise_residual_finalizer::device_supports_pointwise_residual_finalizer(
            residual_ncl,
            channels,
            length,
        )
}

/// Fuse the packed pointwise bias and residual boundary only for the twelve
/// released B1 decoder units. Every logical, cache, physical-layout, dtype,
/// device-limit, or launcher mismatch executes the previous path unchanged.
fn pointwise_residual_finalizer_wgsl_or_fallback(
    conv: &Conv1d,
    packed_weight: Option<&Tensor<3>>,
    input: Tensor<3>,
    residual: Tensor<3>,
) -> Tensor<3> {
    let descriptor = PointwiseResidualDescriptor::from_conv(
        conv,
        input.dims(),
        packed_weight.map(|weight| weight.dims()),
    );
    if !descriptor.supports_finalizer() {
        return existing_pointwise_residual_wgsl(conv, packed_weight, input, residual);
    }
    let Some(packed_weight) = packed_weight else {
        return existing_pointwise_residual_wgsl(conv, None, input, residual);
    };
    let Some(bias) = &conv.bias else {
        return existing_pointwise_residual_wgsl(conv, Some(packed_weight), input, residual);
    };

    // Keep preflight aliases in a short scope. If the route falls back, the
    // old elementwise kernels retain their original buffer-reuse conditions.
    let contract_compatible = {
        let input_raw = input
            .clone()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        let weight_raw = conv
            .weight
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        let packed_weight_raw = packed_weight
            .clone()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        let bias_raw = bias
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        let residual_raw = residual
            .clone()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        pointwise_residual_contract_is_compatible(
            &input_raw,
            &weight_raw,
            &packed_weight_raw,
            &bias_raw,
            &residual_raw,
            descriptor.output_channels,
            descriptor.length,
        )
    };
    if !contract_compatible {
        return existing_pointwise_residual_wgsl(conv, None, input, residual);
    }

    let branch_nlc = nvtx_range!(
        "codec_residual_conv_1x1",
        pointwise_conv1d_matmul_nlc_with_weight(conv, Some(packed_weight), input.clone())
    );
    let output = nvtx_range!(
        "codec_residual_pointwise_finalizer",
        crate::kernels::pointwise_residual_finalizer::pointwise_residual_finalizer_wgsl(
            branch_nlc
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            bias.val()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            residual
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
        )
    );
    match output {
        Ok(output) => Tensor::from_primitive::<crate::WgpuRaw>(output),
        Err(_) => existing_pointwise_residual_wgsl(conv, Some(packed_weight), input, residual),
    }
}

/// Prefer the measured direct T64/O96/K32 projection for the released C384/C192/C96
/// decoder shapes. A direct contract or launcher failure continues through
/// the accepted packed-GEMM finalizer and finally the generic pointwise path.
fn pointwise_residual_wgsl_or_fallback(
    conv: &Conv1d,
    packed_weight: Option<&Tensor<3>>,
    input: PointwiseActivation,
    residual: Tensor<3>,
) -> Tensor<3> {
    let descriptor = PointwiseResidualDescriptor::from_conv(
        conv,
        input.dims(),
        packed_weight.map(|weight| weight.dims()),
    );
    if descriptor.route() == PointwiseResidualRoute::DirectThenFinalizer
        && let (Some(packed_weight), Some(bias)) = (packed_weight, &conv.bias)
    {
        let (input_raw, input_is_nhwc) = match &input {
            PointwiseActivation::Ncl(tensor) => (
                tensor
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                false,
            ),
            PointwiseActivation::Nhwc(tensor) => (
                tensor
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                true,
            ),
        };
        if pointwise_direct_source_weight_is_compatible(
            conv,
            &input_raw,
            descriptor.output_channels,
        ) {
            let constructor = if input_is_nhwc {
                crate::kernels::pointwise_residual_direct_tiled::PointwiseResidualDirectInputs::new_nhwc
            } else {
                crate::kernels::pointwise_residual_direct_tiled::PointwiseResidualDirectInputs::new
            };
            let direct_inputs = constructor(
                input_raw,
                packed_weight
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                bias.val()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                residual
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
            );
            let output = nvtx_range!(
                "codec_residual_pointwise_direct",
                crate::kernels::pointwise_residual_direct_tiled::pointwise_residual_direct_raw_wgsl(
                    direct_inputs,
                    crate::kernels::pointwise_residual_direct_tiled::PointwiseKTile::PRODUCTION,
                )
            );
            if let Ok(output) = output {
                return Tensor::from_primitive::<crate::WgpuRaw>(output);
            }
        }
    }
    pointwise_residual_finalizer_wgsl_or_fallback(conv, packed_weight, input.into_ncl(), residual)
}

fn existing_pointwise_residual_snake_pair_wgsl(
    conv: &Conv1d,
    packed_weight: Option<&Tensor<3>>,
    input: Tensor<3>,
    residual: Tensor<3>,
    next_act0: &Snake1d,
) -> PreparedResidualPair {
    let raw = pointwise_residual_finalizer_wgsl_or_fallback(conv, packed_weight, input, residual);
    let activated = nvtx_range!(
        "codec_residual_snake_0",
        next_act0.forward_wgsl(raw.clone())
    );
    PreparedResidualPair {
        raw,
        activated: PreparedActivation::Ncl(activated),
    }
}

/// Produce only the post-storage-cast Snake activation needed by the next
/// decoder block. Unsupported layouts retain the exact two-dispatch boundary.
fn pointwise_residual_snake_activated_wgsl_or_fallback(
    conv: &Conv1d,
    packed_weight: Option<&Tensor<3>>,
    input: PointwiseActivation,
    residual: Tensor<3>,
    next_act: &Snake1d,
) -> Tensor<3> {
    let descriptor = PointwiseResidualDescriptor::from_conv(
        conv,
        input.dims(),
        packed_weight.map(|weight| weight.dims()),
    );
    if descriptor.route() == PointwiseResidualRoute::DirectThenFinalizer
        && let (Some(packed_weight), Some(bias)) = (packed_weight, &conv.bias)
    {
        let (input_raw, input_is_nhwc) = match &input {
            PointwiseActivation::Ncl(tensor) => (
                tensor
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                false,
            ),
            PointwiseActivation::Nhwc(tensor) => (
                tensor
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                true,
            ),
        };
        if pointwise_direct_source_weight_is_compatible(
            conv,
            &input_raw,
            descriptor.output_channels,
        ) {
            let constructor = if input_is_nhwc {
                crate::kernels::pointwise_residual_direct_tiled::PointwiseResidualDirectInputs::new_nhwc
            } else {
                crate::kernels::pointwise_residual_direct_tiled::PointwiseResidualDirectInputs::new
            };
            let direct_inputs = constructor(
                input_raw,
                packed_weight
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                bias.val()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                residual
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
            );
            if let Ok(activated) = crate::kernels::pointwise_residual_direct_tiled::pointwise_residual_direct_snake_activated_wgsl(
                direct_inputs,
                next_act.alpha.val().try_into_primitive::<crate::WgpuRaw>().expect("tensor must use WGPU raw backend"),
                crate::kernels::pointwise_residual_direct_tiled::PointwiseKTile::PRODUCTION,
            ) {
                return Tensor::from_primitive::<crate::WgpuRaw>(activated);
            }
        }
    }
    let raw = pointwise_residual_wgsl_or_fallback(conv, packed_weight, input, residual);
    next_act.forward_wgsl(raw)
}

#[allow(clippy::too_many_arguments)]
fn pointwise_residual_snake_pair_contract_is_compatible(
    input_ncl: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    weight_oik: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    packed_weight_nkk: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    bias: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    residual_ncl: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    alpha: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    channels: usize,
    length: usize,
) -> bool {
    let same_device = alpha.device == input_ncl.device;
    let exact_alpha = cube_tensor_has_exact_layout(alpha, [1, channels, 1], [channels, 1, 1]);
    pointwise_residual_contract_is_compatible(
        input_ncl,
        weight_oik,
        packed_weight_nkk,
        bias,
        residual_ncl,
        channels,
        length,
    ) && same_device
        && alpha.dtype == input_ncl.dtype
        && exact_alpha
        && crate::kernels::pointwise_residual_snake_pair::device_supports_pointwise_residual_snake_pair(
            residual_ncl,
            alpha,
            channels,
            length,
        )
}

/// Produce the prepared input only for the eight measured intra-block
/// `res0 -> res1` and `res1 -> res2` boundaries.
///
/// The isolated one-shot was bit-exact for both outputs and reduced their
/// aggregate full-path time from 10.099 ms to 9.708 ms (1.040x) without a
/// persistent allocation. Any logical, cache, physical-layout, dtype, device,
/// resource, or launcher mismatch executes the accepted finalizer followed by
/// the standalone next-unit Snake.
fn pointwise_residual_snake_pair_wgsl_or_fallback_with_layout(
    conv: &Conv1d,
    packed_weight: Option<&Tensor<3>>,
    input: Tensor<3>,
    residual: Tensor<3>,
    next_act0: &Snake1d,
    next_residual: Option<&ResidualUnit>,
    prepare_residue_layout: bool,
) -> PreparedResidualPair {
    let descriptor = PointwiseResidualDescriptor::from_conv(
        conv,
        input.dims(),
        packed_weight.map(|weight| weight.dims()),
    );
    if descriptor.route() == PointwiseResidualRoute::DirectThenFinalizer
        && let (Some(packed_weight), Some(bias)) = (packed_weight, &conv.bias)
    {
        let input_raw = input
            .clone()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        if pointwise_direct_source_weight_is_compatible(
            conv,
            &input_raw,
            descriptor.output_channels,
        ) {
            let direct_inputs =
                crate::kernels::pointwise_residual_direct_tiled::PointwiseResidualDirectInputs::new(
                    input_raw,
                    packed_weight
                        .clone()
                        .try_into_primitive::<crate::WgpuRaw>()
                        .expect("tensor must use WGPU raw backend"),
                    bias.val()
                        .try_into_primitive::<crate::WgpuRaw>()
                        .expect("tensor must use WGPU raw backend"),
                    residual
                        .clone()
                        .try_into_primitive::<crate::WgpuRaw>()
                        .expect("tensor must use WGPU raw backend"),
                );
            let next_residue_dilation = prepare_residue_layout
                .then(|| {
                    next_residual.and_then(|next| {
                        next.packed_conv_dil_weight_vectors.as_ref().and_then(|_| {
                            Conv1dK7Descriptor::from_conv(
                                &next.conv_dil,
                                [1, descriptor.output_channels, descriptor.length],
                            )
                            .measured_residue_d1_dilation()
                        })
                    })
                })
                .flatten();
            if let Some(dilation) = next_residue_dilation {
                let output = nvtx_range!(
                    "codec_residual_pointwise_direct_snake_residue_pair",
                    crate::kernels::pointwise_residual_direct_tiled::pointwise_residual_direct_snake_residue_pair_wgsl(
                        direct_inputs.clone(),
                        next_act0.alpha.val().try_into_primitive::<crate::WgpuRaw>().expect("tensor must use WGPU raw backend"),
                        dilation,
                        crate::kernels::pointwise_residual_direct_tiled::PointwiseKTile::PRODUCTION,
                    )
                );
                if let Ok(output) = output {
                    let (raw, activated_residue) = output.into_tensors();
                    return PreparedResidualPair {
                        raw: Tensor::from_primitive::<crate::WgpuRaw>(raw),
                        activated: PreparedActivation::ResiduePacked {
                            tensor: Tensor::from_primitive::<crate::WgpuRaw>(activated_residue),
                            dilation,
                        },
                    };
                }
            }
            let output = nvtx_range!(
                "codec_residual_pointwise_direct_snake_pair",
                crate::kernels::pointwise_residual_direct_tiled::pointwise_residual_direct_snake_pair_wgsl(
                    direct_inputs,
                    next_act0.alpha.val().try_into_primitive::<crate::WgpuRaw>().expect("tensor must use WGPU raw backend"),
                    crate::kernels::pointwise_residual_direct_tiled::PointwiseKTile::PRODUCTION,
                )
            );
            if let Ok(output) = output {
                let (raw, activated) = output.into_tensors();
                return PreparedResidualPair {
                    raw: Tensor::from_primitive::<crate::WgpuRaw>(raw),
                    activated: PreparedActivation::Ncl(Tensor::from_primitive::<crate::WgpuRaw>(
                        activated,
                    )),
                };
            }
        }
    }
    if !descriptor.supports_finalizer() {
        return existing_pointwise_residual_snake_pair_wgsl(
            conv,
            packed_weight,
            input,
            residual,
            next_act0,
        );
    }
    let Some(packed_weight) = packed_weight else {
        return existing_pointwise_residual_snake_pair_wgsl(conv, None, input, residual, next_act0);
    };
    let Some(bias) = &conv.bias else {
        return existing_pointwise_residual_snake_pair_wgsl(
            conv,
            Some(packed_weight),
            input,
            residual,
            next_act0,
        );
    };

    let contract_compatible = {
        let input_raw = input
            .clone()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        let weight_raw = conv
            .weight
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        let packed_weight_raw = packed_weight
            .clone()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        let bias_raw = bias
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        let residual_raw = residual
            .clone()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        let alpha_raw = next_act0
            .alpha
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        pointwise_residual_snake_pair_contract_is_compatible(
            &input_raw,
            &weight_raw,
            &packed_weight_raw,
            &bias_raw,
            &residual_raw,
            &alpha_raw,
            descriptor.output_channels,
            descriptor.length,
        )
    };
    if !contract_compatible {
        return existing_pointwise_residual_snake_pair_wgsl(conv, None, input, residual, next_act0);
    }

    let branch_nlc = nvtx_range!(
        "codec_residual_conv_1x1",
        pointwise_conv1d_matmul_nlc_with_weight(conv, Some(packed_weight), input.clone())
    );
    let output = nvtx_range!(
        "codec_residual_pointwise_snake_pair",
        crate::kernels::pointwise_residual_snake_pair::pointwise_residual_snake_pair_wgsl(
            branch_nlc
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            bias.val()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            residual
                .clone()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            next_act0
                .alpha
                .val()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
        )
    );
    match output {
        Ok(output) => {
            let (raw, activated) = output.into_tensors();
            PreparedResidualPair {
                raw: Tensor::from_primitive::<crate::WgpuRaw>(raw),
                activated: PreparedActivation::Ncl(Tensor::from_primitive::<crate::WgpuRaw>(
                    activated,
                )),
            }
        }
        Err(_) => existing_pointwise_residual_snake_pair_wgsl(
            conv,
            Some(packed_weight),
            input,
            residual,
            next_act0,
        ),
    }
}

fn pointwise_residual_snake_nhwc_pair_wgsl_or_fallback(
    conv: &Conv1d,
    packed_weight: Option<&Tensor<3>>,
    input: PointwiseActivation,
    residual: Tensor<3>,
    next: &ResidualUnit,
) -> PreparedResidualPair {
    let descriptor = PointwiseResidualDescriptor::from_conv(
        conv,
        input.dims(),
        packed_weight.map(|weight| weight.dims()),
    );
    if descriptor.route() == PointwiseResidualRoute::DirectThenFinalizer
        && let (Some(packed_weight), Some(bias)) = (packed_weight, &conv.bias)
    {
        let (input_raw, input_is_nhwc) = match &input {
            PointwiseActivation::Ncl(tensor) => (
                tensor
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                false,
            ),
            PointwiseActivation::Nhwc(tensor) => (
                tensor
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                true,
            ),
        };
        if pointwise_direct_source_weight_is_compatible(
            conv,
            &input_raw,
            descriptor.output_channels,
        ) {
            let constructor = if input_is_nhwc {
                crate::kernels::pointwise_residual_direct_tiled::PointwiseResidualDirectInputs::new_nhwc
            } else {
                crate::kernels::pointwise_residual_direct_tiled::PointwiseResidualDirectInputs::new
            };
            let direct_inputs = constructor(
                input_raw,
                packed_weight
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                bias.val()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                residual
                    .clone()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
            );
            if let Ok(output) = crate::kernels::pointwise_residual_direct_tiled::pointwise_residual_direct_snake_nhwc_pair_wgsl(
                direct_inputs,
                next.act0.alpha.val().try_into_primitive::<crate::WgpuRaw>().expect("tensor must use WGPU raw backend"),
                crate::kernels::pointwise_residual_direct_tiled::PointwiseKTile::PRODUCTION,
            ) {
                let (raw, activated_nhwc) = output.into_tensors();
                return PreparedResidualPair {
                    raw: Tensor::from_primitive::<crate::WgpuRaw>(raw),
                    activated: PreparedActivation::Nhwc(Tensor::from_primitive::<crate::WgpuRaw>(
                        activated_nhwc,
                    )),
                };
            }
        }
    }

    let raw = pointwise_residual_wgsl_or_fallback(conv, packed_weight, input, residual);
    if let Some(activated) = next.act0.forward_nchw_to_nhwc_wgsl(raw.clone()) {
        PreparedResidualPair {
            raw,
            activated: PreparedActivation::Nhwc(activated),
        }
    } else {
        PreparedResidualPair {
            activated: PreparedActivation::Ncl(next.act0.forward_wgsl(raw.clone())),
            raw,
        }
    }
}

/// Physical shortcut layout accepted by the profile-only NHWC residual-state
/// experiment. Ownership makes the resulting launcher inputs single-use.
#[cfg(feature = "profile")]
enum ProfileResidualLayout {
    Ncl(Tensor<3>),
    Nhwc(Tensor<3>),
}

#[cfg(feature = "profile")]
fn direct_pointwise_nhwc_inputs(
    conv: &Conv1d,
    packed_weight: Option<&Tensor<3>>,
    input_nhwc: Tensor<3>,
    residual: ProfileResidualLayout,
) -> Option<crate::kernels::pointwise_residual_direct_tiled::PointwiseResidualDirectInputs> {
    let [batch, length, channels] = input_nhwc.dims();
    if batch != 1 || !matches!(channels, 96 | 192 | 384) || length == 0 {
        return None;
    }
    let packed_weight = packed_weight?;
    let bias = conv.bias.as_ref()?;
    let input_raw = input_nhwc.try_into_primitive::<crate::WgpuRaw>().ok()?;
    if !pointwise_direct_source_weight_is_compatible(conv, &input_raw, channels) {
        return None;
    }
    let packed_raw = packed_weight
        .clone()
        .try_into_primitive::<crate::WgpuRaw>()
        .ok()?;
    let bias_raw = bias.val().try_into_primitive::<crate::WgpuRaw>().ok()?;
    Some(match residual {
        ProfileResidualLayout::Ncl(residual_ncl) => {
            crate::kernels::pointwise_residual_direct_tiled::PointwiseResidualDirectInputs::new_nhwc(
                input_raw,
                packed_raw,
                bias_raw,
                residual_ncl.try_into_primitive::<crate::WgpuRaw>().ok()?,
            )
        }
        ProfileResidualLayout::Nhwc(residual_nhwc) => {
            crate::kernels::pointwise_residual_direct_tiled::PointwiseResidualDirectInputs::new_nhwc_state(
                input_raw,
                packed_raw,
                bias_raw,
                residual_nhwc.try_into_primitive::<crate::WgpuRaw>().ok()?,
            )
        }
    })
}

#[cfg(feature = "profile")]
fn pointwise_residual_nhwc_outputs(
    conv: &Conv1d,
    packed_weight: Option<&Tensor<3>>,
    input_nhwc: Tensor<3>,
    residual: ProfileResidualLayout,
    next: &ResidualUnit,
) -> Option<PreparedNhwcResidualPair> {
    let inputs = direct_pointwise_nhwc_inputs(conv, packed_weight, input_nhwc, residual)?;
    let output = crate::kernels::pointwise_residual_direct_tiled::pointwise_residual_direct_snake_nhwc_outputs_pair_wgsl(
        inputs,
        next.act0.alpha.val().try_into_primitive::<crate::WgpuRaw>().ok()?,
        crate::kernels::pointwise_residual_direct_tiled::PointwiseKTile::PRODUCTION,
    )
    .ok()?;
    let (raw_nhwc, activated_nhwc) = output.into_tensors();
    Some(PreparedNhwcResidualPair {
        raw_nhwc: Tensor::from_primitive::<crate::WgpuRaw>(raw_nhwc),
        activated_nhwc: Tensor::from_primitive::<crate::WgpuRaw>(activated_nhwc),
    })
}

#[cfg(feature = "profile")]
fn implicit_gemm_pointwise_branch_nlc(conv: &Conv1d, input: Tensor<3>) -> Option<Tensor<3>> {
    use burn::tensor::ops::ConvOptions;
    use burn_cubecl::kernel::conv::{ConvStrategy, conv_forward};

    if conv.kernel_size != 1 || conv.stride != 1 || conv.dilation != 1 || conv.groups != 1 {
        return None;
    }
    let output = conv_forward::<burn::backend::wgpu::WgpuRuntime, 1>(
        input
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
        conv.weight
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
        None,
        ConvOptions::new([1], [0], [1], 1),
        ConvStrategy::ImplicitGemm,
    )
    .ok()?;
    Some(Tensor::from_primitive::<crate::WgpuRaw>(output).swap_dims(1, 2))
}

#[cfg(feature = "profile")]
fn implicit_gemm_pointwise_residual(
    conv: &Conv1d,
    input: Tensor<3>,
    residual: Tensor<3>,
) -> Option<Tensor<3>> {
    let bias = conv.bias.as_ref()?.val();
    let branch_nlc = implicit_gemm_pointwise_branch_nlc(conv, input)?;
    let output = crate::kernels::pointwise_residual_finalizer::pointwise_residual_finalizer_wgsl(
        branch_nlc
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
        bias.try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
        residual
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
    )
    .ok()?;
    Some(Tensor::from_primitive::<crate::WgpuRaw>(output))
}

#[cfg(feature = "profile")]
fn implicit_gemm_pointwise_residual_snake_pair(
    conv: &Conv1d,
    input: Tensor<3>,
    residual: Tensor<3>,
    next: &ResidualUnit,
) -> Option<PreparedResidualPair> {
    let bias = conv.bias.as_ref()?.val();
    let branch_nlc = implicit_gemm_pointwise_branch_nlc(conv, input)?;
    let output = crate::kernels::pointwise_residual_snake_pair::pointwise_residual_snake_pair_wgsl(
        branch_nlc
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
        bias.try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
        residual
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
        next.act0
            .alpha
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
    )
    .ok()?;
    let (raw, activated) = output.into_tensors();
    Some(PreparedResidualPair {
        raw: Tensor::from_primitive::<crate::WgpuRaw>(raw),
        activated: PreparedActivation::Ncl(Tensor::from_primitive::<crate::WgpuRaw>(activated)),
    })
}

/// Execute a pointwise projection, bias, residual addition, raw-state store,
/// and next-unit Snake preparation in one CubeK dispatch.
///
/// The primary output remains contiguous NHWC for the next implicit-GEMM k=7
/// convolution. The auxiliary output is contiguous NCL because it is the
/// identity shortcut of that next residual unit. Kernel-size one lets the
/// checkpoint-native OIK weight become a contiguous logical OKI view without
/// a weight-layout copy.
fn cubek_pointwise_accumulator_store_pair(
    conv: &Conv1d,
    input: PointwiseActivation,
    residual: Tensor<3>,
    next: &ResidualUnit,
    rows_policy: PointwiseRowsPolicy,
) -> Option<PreparedResidualPair> {
    use burn::tensor::DType;
    use burn_backend::cubecl::dtype_to_storage_type;
    use burn_cubecl::{
        ops::{numeric::empty_device_dtype, permute_nchw_to_nhwc},
        tensor::CubeTensor,
    };
    use cubek_convolution::{
        ConvolutionArgs,
        components::global::epilogue::{F16ResidualSnakeStore, F16ResidualSnakeStoreParameters},
        forward::launch::launch_epilogue,
        routines::simple::SimpleSyncCyclicAccumulatorTransformConv,
    };
    use cubek_matmul::{
        definition::{MatmulElems, MatmulGlobalElems},
        routines::{BlueprintStrategy, batch::simple::SimpleArgs},
    };
    use cubek_std::InputBinding;

    if conv.kernel_size != 1 || conv.stride != 1 || conv.dilation != 1 || conv.groups != 1 {
        return None;
    }
    let PointwiseActivation::Nhwc(input) = input else {
        return None;
    };
    let input = input.try_into_primitive::<crate::WgpuRaw>().ok()?;
    let residual = residual.try_into_primitive::<crate::WgpuRaw>().ok()?;
    let source_weight = conv
        .weight
        .val()
        .try_into_primitive::<crate::WgpuRaw>()
        .ok()?;
    let weight = permute_nchw_to_nhwc(source_weight);
    let bias = conv
        .bias
        .as_ref()?
        .val()
        .try_into_primitive::<crate::WgpuRaw>()
        .ok()?;
    let alpha = next
        .act0
        .alpha
        .val()
        .try_into_primitive::<crate::WgpuRaw>()
        .ok()?;

    let [batch, length, channels] = input.meta.shape().dims::<3>();
    let expected_elements = batch.checked_mul(length)?.checked_mul(channels)?;
    if input.dtype != DType::F16
        || residual.dtype != DType::F16
        || weight.dtype != DType::F16
        || bias.dtype != DType::F16
        || alpha.dtype != DType::F16
        || &input.meta.strides()[..] != [length * channels, channels, 1].as_slice()
        || residual.meta.shape().dims::<3>() != [batch, channels, length]
        || &residual.meta.strides()[..] != [channels * length, length, 1].as_slice()
        || weight.meta.shape().dims::<3>() != [channels, 1, channels]
        || &weight.meta.strides()[..] != [channels, 1, 1].as_slice()
        || bias.meta.shape().dims::<1>() != [channels]
        || alpha.meta.num_elements() < channels
        || input.meta.num_elements() != expected_elements
        || input.device != residual.device
        || input.device != weight.device
        || input.device != bias.device
        || input.device != alpha.device
    {
        return None;
    }

    let activated: CubeTensor<burn::backend::wgpu::WgpuRuntime> = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        [batch, length, channels].into(),
        DType::F16,
    );
    let raw: CubeTensor<burn::backend::wgpu::WgpuRuntime> = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        [batch, channels, length].into(),
        DType::F16,
    );
    let storage = dtype_to_storage_type(DType::F16);
    let transform = F16ResidualSnakeStoreParameters::try_new(
        &input.client,
        InputBinding::new(residual.binding(), storage),
        InputBinding::new(alpha.binding(), storage),
        InputBinding::new(raw.clone().binding(), storage),
    )
    .ok()?;
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: storage,
        rhs: storage,
        out: storage,
    });
    let strategy = BlueprintStrategy::Inferred(SimpleArgs {
        multi_rows: rows_policy.enabled(length, channels),
        ..SimpleArgs::default()
    });
    type TransformConv = SimpleSyncCyclicAccumulatorTransformConv<F16ResidualSnakeStore>;
    let client = input.client.clone();
    launch_epilogue::<burn::backend::wgpu::WgpuRuntime, 1, TransformConv>(
        &client,
        InputBinding::new(input.binding(), storage),
        InputBinding::new(weight.binding(), storage),
        Some(InputBinding::new(bias.binding(), storage)),
        transform,
        activated.clone().binding(),
        ConvolutionArgs {
            stride: [1],
            padding: [0],
            dilation: [1],
        },
        &strategy,
        dtypes,
    )
    .ok()?;

    Some(PreparedResidualPair {
        raw: Tensor::from_primitive::<crate::WgpuRaw>(raw),
        activated: PreparedActivation::Nhwc(Tensor::from_primitive::<crate::WgpuRaw>(activated)),
    })
}

/// Add the shortcut in a CubeK accumulator writer, preserve the former F16
/// pointwise storage boundary, and store only the following block's Snake
/// activation. The physical output remains contiguous NCL; the logical NHWC
/// view is used only to describe the convolution matrix to CubeK.
#[cfg(feature = "profile")]
fn cubek_pointwise_accumulator_snake_activated(
    conv: &Conv1d,
    input: PointwiseActivation,
    residual: Tensor<3>,
    next_act: &Snake1d,
) -> Option<Tensor<3>> {
    use burn::tensor::DType;
    use burn_backend::cubecl::dtype_to_storage_type;
    use burn_cubecl::{
        ops::{numeric::empty_device_dtype, permute_nchw_to_nhwc},
        tensor::CubeTensor,
    };
    use cubek_convolution::{
        ConvolutionArgs,
        components::global::epilogue::{
            F16ResidualPostCastSnakeStore, F16ResidualPostCastSnakeStoreParameters,
        },
        forward::launch::launch_epilogue,
        routines::simple::SimpleSyncCyclicAccumulatorTransformConv,
    };
    use cubek_matmul::{
        definition::{MatmulElems, MatmulGlobalElems},
        routines::{BlueprintStrategy, batch::simple::SimpleArgs},
    };
    use cubek_std::InputBinding;

    if conv.kernel_size != 1 || conv.stride != 1 || conv.dilation != 1 || conv.groups != 1 {
        return None;
    }
    let PointwiseActivation::Nhwc(input) = input else {
        return None;
    };
    let input = input.try_into_primitive::<crate::WgpuRaw>().ok()?;
    let residual = residual.try_into_primitive::<crate::WgpuRaw>().ok()?;
    let weight = permute_nchw_to_nhwc(
        conv.weight
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .ok()?,
    );
    let bias = conv
        .bias
        .as_ref()?
        .val()
        .try_into_primitive::<crate::WgpuRaw>()
        .ok()?;
    let alpha = next_act
        .alpha
        .val()
        .try_into_primitive::<crate::WgpuRaw>()
        .ok()?;
    let [batch, length, channels] = input.meta.shape().dims::<3>();
    if input.dtype != DType::F16
        || residual.dtype != DType::F16
        || weight.dtype != DType::F16
        || bias.dtype != DType::F16
        || alpha.dtype != DType::F16
        || &input.meta.strides()[..] != [length * channels, channels, 1].as_slice()
        || residual.meta.shape().dims::<3>() != [batch, channels, length]
        || &residual.meta.strides()[..] != [channels * length, length, 1].as_slice()
        || weight.meta.shape().dims::<3>() != [channels, 1, channels]
        || &weight.meta.strides()[..] != [channels, 1, 1].as_slice()
        || bias.meta.shape().dims::<1>() != [channels]
        || alpha.meta.num_elements() < channels
        || input.device != residual.device
        || input.device != weight.device
        || input.device != bias.device
        || input.device != alpha.device
    {
        return None;
    }

    let activated: CubeTensor<burn::backend::wgpu::WgpuRuntime> = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        [batch, channels, length].into(),
        DType::F16,
    );
    let activated_nhwc = permute_nchw_to_nhwc(activated.clone());
    let storage = dtype_to_storage_type(DType::F16);
    let transform = F16ResidualPostCastSnakeStoreParameters::try_new(
        &input.client,
        InputBinding::new(residual.binding(), storage),
        InputBinding::new(alpha.binding(), storage),
    )
    .ok()?;
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: storage,
        rhs: storage,
        out: storage,
    });
    let strategy = BlueprintStrategy::Inferred(SimpleArgs {
        multi_rows: length >= channels,
        ..SimpleArgs::default()
    });
    type TransformConv = SimpleSyncCyclicAccumulatorTransformConv<F16ResidualPostCastSnakeStore>;
    let client = input.client.clone();
    launch_epilogue::<burn::backend::wgpu::WgpuRuntime, 1, TransformConv>(
        &client,
        InputBinding::new(input.binding(), storage),
        InputBinding::new(weight.binding(), storage),
        Some(InputBinding::new(bias.binding(), storage)),
        transform,
        activated_nhwc.binding(),
        ConvolutionArgs {
            stride: [1],
            padding: [0],
            dilation: [1],
        },
        &strategy,
        dtypes,
    )
    .ok()?;
    Some(Tensor::from_primitive::<crate::WgpuRaw>(activated))
}

/// Add the shortcut in the CubeK accumulator writer and store directly into
/// physical NCL through a zero-copy logical NHWC output view.
fn cubek_pointwise_accumulator_residual(
    conv: &Conv1d,
    input: PointwiseActivation,
    residual: Tensor<3>,
) -> Option<Tensor<3>> {
    use burn::tensor::DType;
    use burn_backend::cubecl::dtype_to_storage_type;
    use burn_cubecl::{
        ops::{numeric::empty_device_dtype, permute_nchw_to_nhwc},
        tensor::CubeTensor,
    };
    use cubek_convolution::{
        ConvolutionArgs,
        components::global::epilogue::{F16ResidualStore, F16ResidualStoreParameters},
        forward::launch::launch_epilogue,
        routines::simple::SimpleSyncCyclicAccumulatorTransformConv,
    };
    use cubek_matmul::{
        definition::{MatmulElems, MatmulGlobalElems},
        routines::{BlueprintStrategy, batch::simple::SimpleArgs},
    };
    use cubek_std::InputBinding;

    if conv.kernel_size != 1 || conv.stride != 1 || conv.dilation != 1 || conv.groups != 1 {
        return None;
    }
    let PointwiseActivation::Nhwc(input) = input else {
        return None;
    };
    let input = input.try_into_primitive::<crate::WgpuRaw>().ok()?;
    let residual = residual.try_into_primitive::<crate::WgpuRaw>().ok()?;
    let weight = permute_nchw_to_nhwc(
        conv.weight
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .ok()?,
    );
    let bias = conv
        .bias
        .as_ref()?
        .val()
        .try_into_primitive::<crate::WgpuRaw>()
        .ok()?;
    let [batch, length, channels] = input.meta.shape().dims::<3>();
    if input.dtype != DType::F16
        || residual.dtype != DType::F16
        || weight.dtype != DType::F16
        || bias.dtype != DType::F16
        || &input.meta.strides()[..] != [length * channels, channels, 1].as_slice()
        || residual.meta.shape().dims::<3>() != [batch, channels, length]
        || &residual.meta.strides()[..] != [channels * length, length, 1].as_slice()
        || weight.meta.shape().dims::<3>() != [channels, 1, channels]
        || &weight.meta.strides()[..] != [channels, 1, 1].as_slice()
        || bias.meta.shape().dims::<1>() != [channels]
        || input.device != residual.device
        || input.device != weight.device
        || input.device != bias.device
    {
        return None;
    }

    let raw: CubeTensor<burn::backend::wgpu::WgpuRuntime> = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        [batch, channels, length].into(),
        DType::F16,
    );
    // The logical convolution output is NHWC while its physical backing is
    // the NCL tensor returned to the decoder. `permute` changes metadata only.
    let raw_nhwc = permute_nchw_to_nhwc(raw.clone());
    let storage = dtype_to_storage_type(DType::F16);
    let transform = F16ResidualStoreParameters::try_new(
        &input.client,
        InputBinding::new(residual.binding(), storage),
    )
    .ok()?;
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: storage,
        rhs: storage,
        out: storage,
    });
    let strategy = BlueprintStrategy::Inferred(SimpleArgs {
        multi_rows: length >= channels,
        ..SimpleArgs::default()
    });
    type TransformConv = SimpleSyncCyclicAccumulatorTransformConv<F16ResidualStore>;
    let client = input.client.clone();
    launch_epilogue::<burn::backend::wgpu::WgpuRuntime, 1, TransformConv>(
        &client,
        InputBinding::new(input.binding(), storage),
        InputBinding::new(weight.binding(), storage),
        Some(InputBinding::new(bias.binding(), storage)),
        transform,
        raw_nhwc.binding(),
        ConvolutionArgs {
            stride: [1],
            padding: [0],
            dilation: [1],
        },
        &strategy,
        dtypes,
    )
    .ok()?;
    Some(Tensor::from_primitive::<crate::WgpuRaw>(raw))
}

#[cfg(feature = "profile")]
fn pointwise_residual_with_algorithm(
    conv: &Conv1d,
    packed_weight: Option<&Tensor<3>>,
    input: PointwiseActivation,
    residual: Tensor<3>,
    algorithm: CodecPointwiseAlgorithm,
) -> Tensor<3> {
    match algorithm {
        CodecPointwiseAlgorithm::AccuracyApproved | CodecPointwiseAlgorithm::PackedMatmul => {
            pointwise_residual_wgsl_or_fallback(conv, packed_weight, input, residual)
        }
        CodecPointwiseAlgorithm::CubeClImplicitGemm => {
            let input = input.into_ncl();
            implicit_gemm_pointwise_residual(conv, input.clone(), residual.clone()).unwrap_or_else(
                || {
                    pointwise_residual_wgsl_or_fallback(
                        conv,
                        packed_weight,
                        PointwiseActivation::Ncl(input),
                        residual,
                    )
                },
            )
        }
        CodecPointwiseAlgorithm::CubeClAccumulatorStore => {
            cubek_pointwise_accumulator_residual(conv, input.clone(), residual.clone())
                .unwrap_or_else(|| {
                    pointwise_residual_wgsl_or_fallback(conv, packed_weight, input, residual)
                })
        }
        CodecPointwiseAlgorithm::CubeClAccumulatorPairOnly => {
            pointwise_residual_wgsl_or_fallback(conv, packed_weight, input, residual)
        }
        CodecPointwiseAlgorithm::CubeClAccumulatorPairSingleRow => {
            pointwise_residual_wgsl_or_fallback(conv, packed_weight, input, residual)
        }
    }
}

#[cfg(feature = "profile")]
fn pointwise_residual_snake_pair_with_algorithm(
    conv: &Conv1d,
    packed_weight: Option<&Tensor<3>>,
    input: PointwiseActivation,
    residual: Tensor<3>,
    next: &ResidualUnit,
    k7_algorithm: CodecK7Algorithm,
    pointwise_algorithm: CodecPointwiseAlgorithm,
) -> PreparedResidualPair {
    let prepare_residue_layout = prepare_residue_layout(k7_algorithm, &conv.weight.val());
    if matches!(
        pointwise_algorithm,
        CodecPointwiseAlgorithm::AccuracyApproved
            | CodecPointwiseAlgorithm::CubeClAccumulatorStore
            | CodecPointwiseAlgorithm::CubeClAccumulatorPairOnly
            | CodecPointwiseAlgorithm::CubeClAccumulatorPairSingleRow
    ) && !prepare_residue_layout
    {
        let rows_policy =
            if pointwise_algorithm == CodecPointwiseAlgorithm::CubeClAccumulatorPairSingleRow {
                PointwiseRowsPolicy::SingleRow
            } else {
                PointwiseRowsPolicy::Geometry
            };
        return cubek_pointwise_accumulator_store_pair(
            conv,
            input.clone(),
            residual.clone(),
            next,
            rows_policy,
        )
        .unwrap_or_else(|| {
            pointwise_residual_snake_pair_wgsl_or_fallback_with_layout(
                conv,
                packed_weight,
                input.into_ncl(),
                residual,
                &next.act0,
                Some(next),
                prepare_residue_layout,
            )
        });
    }
    if use_nhwc_prepared_activation(k7_algorithm, &next.conv_dil.weight.val()) {
        return pointwise_residual_snake_nhwc_pair_wgsl_or_fallback(
            conv,
            packed_weight,
            input,
            residual,
            next,
        );
    }
    if pointwise_algorithm == CodecPointwiseAlgorithm::CubeClImplicitGemm && !prepare_residue_layout
    {
        let input = input.into_ncl();
        return implicit_gemm_pointwise_residual_snake_pair(
            conv,
            input.clone(),
            residual.clone(),
            next,
        )
        .unwrap_or_else(|| {
            pointwise_residual_snake_pair_wgsl_or_fallback_with_layout(
                conv,
                packed_weight,
                input,
                residual,
                &next.act0,
                Some(next),
                prepare_residue_layout,
            )
        });
    }
    pointwise_residual_snake_pair_wgsl_or_fallback_with_layout(
        conv,
        packed_weight,
        input.into_ncl(),
        residual,
        &next.act0,
        Some(next),
        prepare_residue_layout,
    )
}

fn standalone_dilated_conv1d_then_snake_wgsl(
    conv: &Conv1d,
    act1: &Snake1d,
    input: Tensor<3>,
) -> Tensor<3> {
    let output = nvtx_range!(
        "codec_residual_conv_dilated",
        dilated_conv1d_wgsl_or_fallback(conv, input)
    );
    nvtx_range!("codec_residual_snake_1", act1.forward_wgsl(output))
}

#[cfg(feature = "profile")]
fn implicit_gemm_materialized_dilated_conv1d_then_snake_wgsl(
    conv: &Conv1d,
    act1: &Snake1d,
    input: Tensor<3>,
) -> Option<Tensor<3>> {
    use burn::tensor::ops::ConvOptions;
    use burn_cubecl::kernel::conv::{ConvStrategy, conv_forward};

    if conv.kernel_size != 7 || conv.stride != 1 || conv.groups != 1 {
        return None;
    }
    let bias = conv.bias.as_ref()?.val();
    let output = conv_forward::<burn::backend::wgpu::WgpuRuntime, 1>(
        input
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
        conv.weight
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
        Some(
            bias.try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
        ),
        ConvOptions::new([1], [3 * conv.dilation], [conv.dilation], 1),
        ConvStrategy::ImplicitGemm,
    )
    .ok()?;
    Some(act1.forward_wgsl(Tensor::from_primitive::<crate::WgpuRaw>(output)))
}

fn implicit_gemm_dilated_conv1d_then_snake_wgsl(
    conv: &Conv1d,
    act1: &Snake1d,
    input: Tensor<3>,
) -> Option<Tensor<3>> {
    use burn::tensor::ops::ConvOptions;
    use burn_cubecl::{
        kernel::conv::{ConvStrategy, conv_forward_nhwc},
        ops::permute_nchw_to_nhwc,
    };

    if conv.kernel_size != 7 || conv.stride != 1 || conv.groups != 1 {
        return None;
    }
    let bias = conv.bias.as_ref()?.val();
    let output_nhwc = conv_forward_nhwc::<burn::backend::wgpu::WgpuRuntime, 1>(
        permute_nchw_to_nhwc(
            input
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
        ),
        permute_nchw_to_nhwc(
            conv.weight
                .val()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
        ),
        Some(
            bias.try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
        ),
        ConvOptions::new([1], [3 * conv.dilation], [conv.dilation], 1),
        ConvStrategy::ImplicitGemm,
    )
    .ok()?;
    let output = crate::kernels::snake::snake_nhwc_to_nchw_wgsl(
        output_nhwc,
        act1.alpha
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
    )?;
    Some(Tensor::from_primitive::<crate::WgpuRaw>(output))
}

#[cfg(feature = "profile")]
#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
struct K7AutotuneKey {
    schema: u16,
    dtype: String,
    batch: usize,
    input_length: usize,
    input_channels: usize,
    output_length: usize,
    output_channels: usize,
    dilation: usize,
    input_strides: Vec<usize>,
    weight_strides: Vec<usize>,
}

#[cfg(feature = "profile")]
impl std::fmt::Display for K7AutotuneKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "k7-s{}-{}-b{}-l{}x{}-c{}x{}-d{}-is{:?}-ws{:?}",
            self.schema,
            self.dtype,
            self.batch,
            self.input_length,
            self.output_length,
            self.input_channels,
            self.output_channels,
            self.dilation,
            self.input_strides,
            self.weight_strides,
        )
    }
}

#[cfg(feature = "profile")]
impl cubecl::tune::AutotuneKey for K7AutotuneKey {}

#[cfg(feature = "profile")]
#[derive(Clone)]
struct K7AutotuneInput {
    input: burn_cubecl::tensor::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    weight: burn_cubecl::tensor::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    bias: burn_cubecl::tensor::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    alpha: burn_cubecl::tensor::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    output_shape: [usize; 3],
    args: cubek_convolution::ConvolutionArgs<1>,
}

#[cfg(feature = "profile")]
fn k7_autotune_key(input: &K7AutotuneInput) -> K7AutotuneKey {
    let [batch, input_length, input_channels] = input.input.meta.shape().dims::<3>();
    let [_, output_length, output_channels] = input.output_shape;
    K7AutotuneKey {
        schema: 2,
        dtype: input.input.dtype.name().to_owned(),
        batch,
        input_length,
        input_channels,
        output_length,
        output_channels,
        dilation: input.args.dilation[0],
        input_strides: input.input.meta.strides().to_vec(),
        weight_strides: input.weight.meta.strides().to_vec(),
    }
}

#[cfg(feature = "profile")]
fn launch_k7_autotune_candidate(
    input: K7AutotuneInput,
    strategy_args: cubek_matmul::routines::batch::simple::TunableSimpleArgs,
) -> Result<burn_cubecl::tensor::CubeTensor<burn::backend::wgpu::WgpuRuntime>, String> {
    use burn_backend::cubecl::dtype_to_storage_type;
    use burn_cubecl::ops::numeric::empty_device_dtype;
    use cubek_convolution::{
        components::global::epilogue::{F32EpilogueParameters, SnakeEpilogue},
        forward::launch::launch_epilogue,
        routines::simple::TunableSimpleSyncCyclicPostCastEpilogueConv,
    };
    use cubek_matmul::{
        definition::{MatmulElems, MatmulGlobalElems},
        routines::BlueprintStrategy,
    };
    use cubek_std::InputBinding;

    type SnakeConv = TunableSimpleSyncCyclicPostCastEpilogueConv<SnakeEpilogue>;

    let output = empty_device_dtype(
        input.input.client.clone(),
        input.input.device.clone(),
        input.output_shape.into(),
        input.input.dtype,
    );
    let input_storage = dtype_to_storage_type(input.input.dtype);
    let weight_storage = dtype_to_storage_type(input.weight.dtype);
    let output_storage = dtype_to_storage_type(output.dtype);
    let bias_storage = dtype_to_storage_type(input.bias.dtype);
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: input_storage,
        rhs: weight_storage,
        out: output_storage,
    });
    let alpha_storage = dtype_to_storage_type(input.alpha.dtype);
    let alpha_client = input.alpha.client.clone();
    let alpha = F32EpilogueParameters::try_new(
        &alpha_client,
        InputBinding::new(input.alpha.binding(), alpha_storage),
    )
    .map_err(|err| format!("invalid k7 autotune epilogue: {err:?}"))?;
    let strategy = BlueprintStrategy::Inferred(strategy_args);
    let client = input.input.client.clone();
    launch_epilogue::<burn::backend::wgpu::WgpuRuntime, 1, SnakeConv>(
        &client,
        InputBinding::new(input.input.binding(), input_storage),
        InputBinding::new(input.weight.binding(), weight_storage),
        Some(InputBinding::Normal(input.bias.binding(), bias_storage)),
        alpha,
        output.clone().binding(),
        input.args,
        &strategy,
        dtypes,
    )
    .map_err(|err| format!("k7 autotune launch rejected: {err:?}"))?;
    Ok(output)
}

/// Launch the exact production routine type so tuning never compares a
/// candidate against a nominally equivalent but differently compiled control.
#[cfg(feature = "profile")]
fn launch_k7_production_candidate(
    input: K7AutotuneInput,
    multi_rows: bool,
) -> Result<burn_cubecl::tensor::CubeTensor<burn::backend::wgpu::WgpuRuntime>, String> {
    use burn_backend::cubecl::dtype_to_storage_type;
    use burn_cubecl::ops::numeric::empty_device_dtype;
    use cubek_convolution::{
        components::global::epilogue::{F32EpilogueParameters, SnakeEpilogue},
        forward::launch::launch_epilogue,
        routines::simple::SimpleSyncCyclicPostCastEpilogueConv,
    };
    use cubek_matmul::{
        definition::{MatmulElems, MatmulGlobalElems},
        routines::{BlueprintStrategy, batch::simple::SimpleArgs},
    };
    use cubek_std::InputBinding;

    type SnakeConv = SimpleSyncCyclicPostCastEpilogueConv<SnakeEpilogue>;

    let output = empty_device_dtype(
        input.input.client.clone(),
        input.input.device.clone(),
        input.output_shape.into(),
        input.input.dtype,
    );
    let input_storage = dtype_to_storage_type(input.input.dtype);
    let weight_storage = dtype_to_storage_type(input.weight.dtype);
    let output_storage = dtype_to_storage_type(output.dtype);
    let bias_storage = dtype_to_storage_type(input.bias.dtype);
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: input_storage,
        rhs: weight_storage,
        out: output_storage,
    });
    let alpha_storage = dtype_to_storage_type(input.alpha.dtype);
    let alpha_client = input.alpha.client.clone();
    let alpha = F32EpilogueParameters::try_new(
        &alpha_client,
        InputBinding::new(input.alpha.binding(), alpha_storage),
    )
    .map_err(|err| format!("invalid k7 production-control epilogue: {err:?}"))?;
    let strategy = BlueprintStrategy::Inferred(SimpleArgs {
        multi_rows,
        ..SimpleArgs::default()
    });
    let client = input.input.client.clone();
    launch_epilogue::<burn::backend::wgpu::WgpuRuntime, 1, SnakeConv>(
        &client,
        InputBinding::new(input.input.binding(), input_storage),
        InputBinding::new(input.weight.binding(), weight_storage),
        Some(InputBinding::Normal(input.bias.binding(), bias_storage)),
        alpha,
        output.clone().binding(),
        input.args,
        &strategy,
        dtypes,
    )
    .map_err(|err| format!("k7 production-control launch rejected: {err:?}"))?;
    Ok(output)
}

#[cfg(feature = "profile")]
fn k7_selector_args(
    choice: super::algorithm::K7SelectorChoice,
) -> cubek_matmul::routines::batch::simple::TunableSimpleArgs {
    use cubek_matmul::{
        components::stage::PartitionBuffering, routines::batch::simple::TunableSimpleArgs,
    };

    match choice {
        super::algorithm::K7SelectorChoice::SingleRow => TunableSimpleArgs::default(),
        super::algorithm::K7SelectorChoice::MultiRow => TunableSimpleArgs {
            multi_rows: true,
            ..TunableSimpleArgs::default()
        },
        super::algorithm::K7SelectorChoice::SingleNoSwizzle => TunableSimpleArgs {
            swizzled: Some(false),
            ..TunableSimpleArgs::default()
        },
        super::algorithm::K7SelectorChoice::SingleAutoPartition => TunableSimpleArgs {
            partition_buffering: None,
            ..TunableSimpleArgs::default()
        },
        super::algorithm::K7SelectorChoice::SingleDoublePartition => TunableSimpleArgs {
            partition_buffering: Some(PartitionBuffering::Double),
            ..TunableSimpleArgs::default()
        },
        super::algorithm::K7SelectorChoice::SingleNoSwizzleAutoPartition => TunableSimpleArgs {
            swizzled: Some(false),
            partition_buffering: None,
            ..TunableSimpleArgs::default()
        },
    }
}

#[cfg(feature = "profile")]
fn autotune_k7_snake(
    input: K7AutotuneInput,
) -> burn_cubecl::tensor::CubeTensor<burn::backend::wgpu::WgpuRuntime> {
    use burn_cubecl::CubeTuneId;
    use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
    use cubek_matmul::routines::batch::simple::TunableSimpleArgs;

    type Output = burn_cubecl::tensor::CubeTensor<burn::backend::wgpu::WgpuRuntime>;
    static TUNER: LocalTuner<K7AutotuneKey, CubeTuneId> = local_tuner!("irodori-k7-snake-v2");

    let tunables = TUNER.init(|| {
        TunableSet::<K7AutotuneKey, K7AutotuneInput, Output>::new_cloning_inputs(k7_autotune_key)
            .with(Tunable::new(
                "production-sync-cyclic-single-row-v2",
                |input| launch_k7_production_candidate(input, false),
            ))
            .with(Tunable::new(
                "production-sync-cyclic-multi-row-v2",
                |input| launch_k7_production_candidate(input, true),
            ))
            .with(Tunable::new("sync-cyclic-single-no-swizzle-v1", |input| {
                launch_k7_autotune_candidate(
                    input,
                    TunableSimpleArgs {
                        swizzled: Some(false),
                        ..TunableSimpleArgs::default()
                    },
                )
            }))
            .with(Tunable::new(
                "sync-cyclic-single-auto-partition-v1",
                |input| {
                    launch_k7_autotune_candidate(
                        input,
                        TunableSimpleArgs {
                            partition_buffering: None,
                            ..TunableSimpleArgs::default()
                        },
                    )
                },
            ))
            .with(Tunable::new(
                "sync-cyclic-single-double-partition-v1",
                |input| {
                    launch_k7_autotune_candidate(
                        input,
                        TunableSimpleArgs {
                            partition_buffering: Some(
                                cubek_matmul::components::stage::PartitionBuffering::Double,
                            ),
                            ..TunableSimpleArgs::default()
                        },
                    )
                },
            ))
            .with(Tunable::new(
                "sync-cyclic-single-no-swizzle-auto-partition-v1",
                |input| {
                    launch_k7_autotune_candidate(
                        input,
                        TunableSimpleArgs {
                            swizzled: Some(false),
                            partition_buffering: None,
                            ..TunableSimpleArgs::default()
                        },
                    )
                },
            ))
            .with_short_circuit(false)
    });
    let client = input.input.client.clone();
    TUNER.execute(
        &CubeTuneId::new(&input.input.client, &input.input.device),
        &client,
        tunables,
        input,
    )
}

#[allow(clippy::too_many_arguments)]
fn implicit_gemm_nhwc_dilated_conv1d_then_snake_wgsl(
    conv: &Conv1d,
    act1: &Snake1d,
    input_nhwc: Tensor<3>,
    prepared_weight: Option<&PreparedK7Weight>,
    direct_strided_weight: bool,
    multi_rows: K7MultiRowsSelection,
    halo_loader: bool,
    prepared_epilogue: bool,
) -> Option<Tensor<3>> {
    use burn::tensor::ops::ConvOptions;
    use burn_backend::cubecl::dtype_to_storage_type;
    use burn_cubecl::{
        ops::{numeric::empty_device_dtype, permute_nchw_to_nhwc},
        tensor::CubeTensor,
    };
    use cubek_convolution::{
        ConvolutionArgs,
        components::global::epilogue::{
            F32EpilogueParameters, PreparedSnakeEpilogue, SnakeEpilogue,
        },
        forward::launch::{launch_epilogue, launch_k7_channel_major_epilogue},
        routines::simple::{
            SimpleSyncCyclicPostCastEpilogueConv, SimpleSyncCyclicStridedPostCastEpilogueConv,
            SimpleSyncK7HaloPostCastEpilogueConv,
        },
    };
    use cubek_matmul::{
        definition::{MatmulElems, MatmulGlobalElems},
        routines::{BlueprintStrategy, batch::simple::SimpleArgs},
    };
    use cubek_std::InputBinding;

    if conv.kernel_size != 7 || conv.stride != 1 || conv.groups != 1 {
        return None;
    }
    let options = ConvOptions::new([1], [3 * conv.dilation], [conv.dilation], 1);
    let input = input_nhwc.try_into_primitive::<crate::WgpuRaw>().ok()?;
    let weight = if halo_loader {
        conv.weight
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .ok()?
    } else if let Some(prepared) = prepared_weight {
        if prepared.source_oik_shape != conv.weight.dims() || prepared.physical_oki_strides[2] != 1
        {
            return None;
        }
        prepared
            .oki
            .clone()
            .try_into_primitive::<crate::WgpuRaw>()
            .ok()?
    } else {
        permute_nchw_to_nhwc(
            conv.weight
                .val()
                .try_into_primitive::<crate::WgpuRaw>()
                .ok()?,
        )
    };
    let bias = conv
        .bias
        .as_ref()?
        .val()
        .try_into_primitive::<crate::WgpuRaw>()
        .ok()?;
    #[cfg(feature = "profile")]
    let alpha = if prepared_epilogue {
        act1.alpha_recip_epilogue_f32.clone()?
    } else {
        act1.alpha_epilogue_f32.clone()
    }
    .try_into_primitive::<crate::WgpuRaw>()
    .ok()?;
    #[cfg(not(feature = "profile"))]
    let alpha = {
        let _ = prepared_epilogue;
        act1.alpha_epilogue_f32
            .clone()
            .try_into_primitive::<crate::WgpuRaw>()
            .ok()?
    };

    let [batch, input_length, _] = input.meta.shape().dims::<3>();
    let weight_dims = weight.meta.shape().dims::<3>();
    let (output_channels, kernel_size) = if halo_loader {
        (weight_dims[0], weight_dims[2])
    } else {
        (weight_dims[0], weight_dims[1])
    };
    let effective_kernel = options.dilation[0]
        .checked_mul(kernel_size.checked_sub(1)?)?
        .checked_add(1)?;
    let output_length = input_length
        .checked_add(options.padding[0].checked_mul(2)?)?
        .checked_sub(effective_kernel)?
        .checked_div(options.stride[0])?
        .checked_add(1)?;
    #[cfg(feature = "profile")]
    #[cfg(feature = "profile")]
    let geometry_multi_rows = output_length >= output_channels && output_channels >= 384;
    #[cfg(feature = "profile")]
    let direct_tunable = match multi_rows {
        K7MultiRowsSelection::Autotuned => true,
        K7MultiRowsSelection::Prepared(super::algorithm::K7SelectorChoice::SingleRow) => {
            geometry_multi_rows
        }
        K7MultiRowsSelection::Prepared(super::algorithm::K7SelectorChoice::MultiRow) => {
            !geometry_multi_rows
        }
        K7MultiRowsSelection::Prepared(_) => true,
        _ => false,
    };
    #[cfg(feature = "profile")]
    if direct_tunable {
        if halo_loader || direct_strided_weight || prepared_epilogue || prepared_weight.is_some() {
            return None;
        }
        let tuned_input = K7AutotuneInput {
            input,
            weight,
            bias,
            alpha,
            output_shape: [batch, output_length, output_channels],
            args: cubek_convolution::ConvolutionArgs {
                stride: options.stride,
                padding: options.padding,
                dilation: options.dilation,
            },
        };
        let output = match multi_rows {
            K7MultiRowsSelection::Autotuned => autotune_k7_snake(tuned_input),
            K7MultiRowsSelection::Prepared(super::algorithm::K7SelectorChoice::SingleRow) => {
                launch_k7_production_candidate(tuned_input, false).ok()?
            }
            K7MultiRowsSelection::Prepared(super::algorithm::K7SelectorChoice::MultiRow) => {
                launch_k7_production_candidate(tuned_input, true).ok()?
            }
            K7MultiRowsSelection::Prepared(choice) => {
                launch_k7_autotune_candidate(tuned_input, k7_selector_args(choice)).ok()?
            }
            _ => unreachable!("guarded by prepared/autotuned selection"),
        };
        return Some(Tensor::from_primitive::<crate::WgpuRaw>(output));
    }
    let output: CubeTensor<burn::backend::wgpu::WgpuRuntime> = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        [batch, output_length, output_channels].into(),
        input.dtype,
    );

    let input_storage = dtype_to_storage_type(input.dtype);
    let weight_storage = dtype_to_storage_type(weight.dtype);
    let output_storage = dtype_to_storage_type(output.dtype);
    let bias_storage = dtype_to_storage_type(bias.dtype);
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: input_storage,
        rhs: weight_storage,
        out: output_storage,
    });
    type SnakeConv = SimpleSyncCyclicPostCastEpilogueConv<SnakeEpilogue>;
    type PreparedSnakeConv = SimpleSyncCyclicPostCastEpilogueConv<PreparedSnakeEpilogue>;
    type DirectSnakeConv = SimpleSyncCyclicStridedPostCastEpilogueConv<SnakeEpilogue>;
    type HaloSnakeConv = SimpleSyncK7HaloPostCastEpilogueConv<SnakeEpilogue>;
    let strategy_args = SimpleArgs {
        multi_rows: multi_rows.enabled(output_length, output_channels),
        ..SimpleArgs::default()
    };
    let client = input.client.clone();
    let alpha_storage = dtype_to_storage_type(alpha.dtype);
    let alpha_client = alpha.client.clone();
    let alpha = F32EpilogueParameters::try_new(
        &alpha_client,
        InputBinding::new(alpha.binding(), alpha_storage),
    )
    .ok()?;
    let input = InputBinding::new(input.binding(), input_storage);
    let weight = InputBinding::new(weight.binding(), weight_storage);
    let bias = Some(InputBinding::Normal(bias.binding(), bias_storage));
    let output_binding = output.clone().binding();
    let args = ConvolutionArgs {
        stride: options.stride,
        padding: options.padding,
        dilation: options.dilation,
    };
    if halo_loader {
        let strategy = BlueprintStrategy::Inferred(strategy_args.clone());
        launch_k7_channel_major_epilogue::<burn::backend::wgpu::WgpuRuntime, HaloSnakeConv>(
            &client,
            input,
            weight,
            bias,
            alpha,
            output_binding,
            options.dilation[0],
            &strategy,
            dtypes,
        )
        .ok()?;
    } else if direct_strided_weight {
        let strategy = BlueprintStrategy::Inferred(strategy_args.clone());
        launch_epilogue::<burn::backend::wgpu::WgpuRuntime, 1, DirectSnakeConv>(
            &client,
            input,
            weight,
            bias,
            alpha,
            output_binding,
            args,
            &strategy,
            dtypes,
        )
        .ok()?;
    } else if prepared_epilogue {
        let strategy = BlueprintStrategy::Inferred(strategy_args);
        launch_epilogue::<burn::backend::wgpu::WgpuRuntime, 1, PreparedSnakeConv>(
            &client,
            input,
            weight,
            bias,
            alpha,
            output_binding,
            args,
            &strategy,
            dtypes,
        )
        .ok()?;
    } else {
        let strategy = BlueprintStrategy::Inferred(strategy_args);
        launch_epilogue::<burn::backend::wgpu::WgpuRuntime, 1, SnakeConv>(
            &client,
            input,
            weight,
            bias,
            alpha,
            output_binding,
            args,
            &strategy,
            dtypes,
        )
        .ok()?;
    }

    Some(Tensor::from_primitive::<crate::WgpuRaw>(output))
}

#[cfg(feature = "profile")]
fn custom_implicit_gemm_dilated_conv1d_then_snake_wgsl(
    conv: &Conv1d,
    act1: &Snake1d,
    input: Tensor<3>,
    algorithm: cubek_convolution::ConvAlgorithm,
    input_is_nhwc: bool,
) -> Option<Tensor<3>> {
    use burn::tensor::ops::ConvOptions;
    use burn_backend::cubecl::dtype_to_storage_type;
    use burn_cubecl::{
        ops::{numeric::empty_device_dtype, permute_nchw_to_nhwc, permute_nhwc_to_nchw},
        tensor::CubeTensor,
    };
    use cubek_convolution::{
        AcceleratedTileKind, ConvolutionArgs, ConvolutionInputs, Strategy, launch_ref,
    };
    use cubek_matmul::definition::{MatmulElems, MatmulGlobalElems};
    use cubek_std::InputBinding;

    if conv.kernel_size != 7 || conv.stride != 1 || conv.groups != 1 {
        return None;
    }
    let options = ConvOptions::new([1], [3 * conv.dilation], [conv.dilation], 1);
    let input = input
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    let input = if input_is_nhwc {
        input
    } else {
        permute_nchw_to_nhwc(input)
    };
    let weight = permute_nchw_to_nhwc(
        conv.weight
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
    );
    let bias = conv
        .bias
        .as_ref()?
        .val()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");

    let [batch, input_length, _] = input.meta.shape().dims::<3>();
    let [output_channels, kernel_size, _] = weight.meta.shape().dims::<3>();
    let effective_kernel = options.dilation[0]
        .checked_mul(kernel_size.checked_sub(1)?)?
        .checked_add(1)?;
    let padded_length = input_length.checked_add(options.padding[0].checked_mul(2)?)?;
    let output_length = padded_length
        .checked_sub(effective_kernel)?
        .checked_div(options.stride[0])?
        .checked_add(1)?;
    let output: CubeTensor<burn::backend::wgpu::WgpuRuntime> = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        [batch, output_length, output_channels].into(),
        input.dtype,
    );

    let input_storage = dtype_to_storage_type(input.dtype);
    let weight_storage = dtype_to_storage_type(weight.dtype);
    let output_storage = dtype_to_storage_type(output.dtype);
    let bias_storage = dtype_to_storage_type(bias.dtype);
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: input_storage,
        rhs: weight_storage,
        out: output_storage,
    });
    let client = input.client.clone();
    launch_ref::<burn::backend::wgpu::WgpuRuntime, 1>(
        &Strategy::Inferred {
            algorithm,
            tile_kind: AcceleratedTileKind::Cmma,
        },
        &client,
        ConvolutionInputs::Forward {
            input: InputBinding::new(input.binding(), input_storage),
            weight: InputBinding::new(weight.binding(), weight_storage),
            bias: Some(InputBinding::Normal(bias.binding(), bias_storage)),
            out: output.clone().binding(),
        },
        ConvolutionArgs {
            stride: options.stride,
            padding: options.padding,
            dilation: options.dilation,
        },
        dtypes,
    )
    .ok()?;

    if input_is_nhwc {
        let output = crate::kernels::snake::snake_nhwc_wgsl(
            output,
            act1.alpha
                .val()
                .try_into_primitive::<crate::WgpuRaw>()
                .ok()?,
        )?;
        Some(Tensor::from_primitive::<crate::WgpuRaw>(output))
    } else {
        let output = permute_nhwc_to_nchw(output);
        Some(act1.forward_wgsl(Tensor::from_primitive::<crate::WgpuRaw>(output)))
    }
}

fn tensor_uses_f16(tensor: &Tensor<3>) -> bool {
    tensor
        .clone()
        .try_into_primitive::<crate::WgpuRaw>()
        .is_ok_and(|primitive| primitive.dtype == burn::tensor::DType::F16)
}

fn use_implicit_gemm(algorithm: CodecK7Algorithm, tensor: &Tensor<3>) -> bool {
    match algorithm {
        CodecK7Algorithm::AccuracyApproved => tensor_uses_f16(tensor),
        CodecK7Algorithm::PackedResidue => false,
        CodecK7Algorithm::CubeClImplicitGemm => true,
        #[cfg(feature = "profile")]
        CodecK7Algorithm::CubeClImplicitGemmSingleStorage
        | CodecK7Algorithm::CubeClImplicitGemmPreparedWeight(_)
        | CodecK7Algorithm::CubeClImplicitGemmDirectOik
        | CodecK7Algorithm::CubeClImplicitGemmK7Halo
        | CodecK7Algorithm::CubeClImplicitGemmMultiRows
        | CodecK7Algorithm::CubeClImplicitGemmGeometrySelectedMultiRows
        | CodecK7Algorithm::CubeClImplicitGemmAutotuned
        | CodecK7Algorithm::CubeClImplicitGemmPreparedSelector
        | CodecK7Algorithm::CubeClImplicitGemmPreparedEpilogue
        | CodecK7Algorithm::CubeClImplicitGemmInputLayoutFused
        | CodecK7Algorithm::CubeClImplicitGemmMaterialized
        | CodecK7Algorithm::CubeClImplicitGemmAsync
        | CodecK7Algorithm::CubeClImplicitGemmSyncStrided
        | CodecK7Algorithm::CubeClImplicitGemmAsyncStrided => true,
    }
}

fn use_single_storage_k7(algorithm: CodecK7Algorithm, tensor: &Tensor<3>) -> bool {
    #[cfg(feature = "profile")]
    if algorithm == CodecK7Algorithm::CubeClImplicitGemmSingleStorage {
        return use_implicit_gemm(algorithm, tensor);
    }
    #[cfg(not(feature = "profile"))]
    let _ = (algorithm, tensor);
    false
}

fn use_nhwc_prepared_activation(algorithm: CodecK7Algorithm, tensor: &Tensor<3>) -> bool {
    match algorithm {
        CodecK7Algorithm::AccuracyApproved | CodecK7Algorithm::CubeClImplicitGemm => {
            tensor_uses_f16(tensor)
        }
        #[cfg(feature = "profile")]
        CodecK7Algorithm::CubeClImplicitGemmSingleStorage
        | CodecK7Algorithm::CubeClImplicitGemmPreparedWeight(_)
        | CodecK7Algorithm::CubeClImplicitGemmDirectOik
        | CodecK7Algorithm::CubeClImplicitGemmK7Halo
        | CodecK7Algorithm::CubeClImplicitGemmMultiRows
        | CodecK7Algorithm::CubeClImplicitGemmGeometrySelectedMultiRows
        | CodecK7Algorithm::CubeClImplicitGemmAutotuned
        | CodecK7Algorithm::CubeClImplicitGemmPreparedSelector
        | CodecK7Algorithm::CubeClImplicitGemmPreparedEpilogue
        | CodecK7Algorithm::CubeClImplicitGemmInputLayoutFused => true,
        CodecK7Algorithm::PackedResidue => false,
        #[cfg(feature = "profile")]
        CodecK7Algorithm::CubeClImplicitGemmMaterialized => false,
        #[cfg(feature = "profile")]
        CodecK7Algorithm::CubeClImplicitGemmAsync
        | CodecK7Algorithm::CubeClImplicitGemmSyncStrided
        | CodecK7Algorithm::CubeClImplicitGemmAsyncStrided => true,
    }
}

fn prepare_residue_layout(algorithm: CodecK7Algorithm, tensor: &Tensor<3>) -> bool {
    !use_implicit_gemm(algorithm, tensor)
}

fn use_prepared_snake_epilogue(algorithm: CodecK7Algorithm) -> bool {
    #[cfg(feature = "profile")]
    if algorithm == CodecK7Algorithm::CubeClImplicitGemmPreparedEpilogue {
        return true;
    }
    #[cfg(not(feature = "profile"))]
    let _ = algorithm;
    false
}

fn dilated_conv1d_act1_with_algorithm(
    conv: &Conv1d,
    act1: &Snake1d,
    packed_residue_weight: Option<&Tensor<3>>,
    input: Tensor<3>,
    algorithm: CodecK7Algorithm,
) -> Tensor<3> {
    if use_implicit_gemm(algorithm, &input) {
        let candidate = match algorithm {
            #[cfg(feature = "profile")]
            CodecK7Algorithm::CubeClImplicitGemmMaterialized => {
                implicit_gemm_materialized_dilated_conv1d_then_snake_wgsl(conv, act1, input.clone())
            }
            #[cfg(feature = "profile")]
            CodecK7Algorithm::CubeClImplicitGemmAsync => {
                custom_implicit_gemm_dilated_conv1d_then_snake_wgsl(
                    conv,
                    act1,
                    input.clone(),
                    cubek_convolution::ConvAlgorithm::SimpleAsyncCyclic,
                    false,
                )
            }
            #[cfg(feature = "profile")]
            CodecK7Algorithm::CubeClImplicitGemmSyncStrided => {
                custom_implicit_gemm_dilated_conv1d_then_snake_wgsl(
                    conv,
                    act1,
                    input.clone(),
                    cubek_convolution::ConvAlgorithm::SimpleSyncStrided,
                    false,
                )
            }
            #[cfg(feature = "profile")]
            CodecK7Algorithm::CubeClImplicitGemmAsyncStrided => {
                custom_implicit_gemm_dilated_conv1d_then_snake_wgsl(
                    conv,
                    act1,
                    input.clone(),
                    cubek_convolution::ConvAlgorithm::SimpleAsyncStrided,
                    false,
                )
            }
            _ => implicit_gemm_dilated_conv1d_then_snake_wgsl(conv, act1, input.clone()),
        };
        candidate.unwrap_or_else(|| {
            dilated_conv1d_act1_wgsl_or_fallback(conv, act1, packed_residue_weight, input)
        })
    } else {
        dilated_conv1d_act1_wgsl_or_fallback(conv, act1, packed_residue_weight, input)
    }
}

fn conv1d_k7_snake_epilogue_contract_is_compatible(
    input: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    weight: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    bias: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    alpha: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    channels: usize,
    length: usize,
) -> bool {
    if input.meta.num_dims() != 3
        || weight.meta.num_dims() != 3
        || bias.meta.num_dims() != 1
        || alpha.meta.num_dims() != 3
    {
        return false;
    }
    let input_shape = input.meta.shape();
    let weight_shape = weight.meta.shape();
    let alpha_shape = alpha.meta.shape();
    wgsl_float_dtype_is_supported(input.dtype)
        && [input, weight, bias, alpha]
            .into_iter()
            .all(|tensor| tensor.dtype == input.dtype && tensor.device == input.device)
        && [input_shape[0], input_shape[1], input_shape[2]] == [1, channels, length]
        && [weight_shape[0], weight_shape[1], weight_shape[2]] == [channels, channels, 7]
        && bias.meta.shape()[0] == channels
        && [alpha_shape[0], alpha_shape[1], alpha_shape[2]] == [1, channels, 1]
        && input.is_contiguous()
        && weight.is_contiguous()
        && bias.is_contiguous()
        && alpha.is_contiguous()
}

/// Validate the non-materialisable part of the previous standalone O32/O16
/// launch contract. Input contiguity is intentionally not required here: both
/// prior launchers materialise that view, while the O64-specific contract below
/// remains fail-closed and requires an already contiguous input.
fn conv1d_k7_standalone_base_contract_is_compatible(
    input: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    weight: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    bias: &burn::backend::wgpu::CubeTensor<burn::backend::wgpu::WgpuRuntime>,
    channels: usize,
    length: usize,
) -> bool {
    if input.meta.num_dims() != 3 || weight.meta.num_dims() != 3 || bias.meta.num_dims() != 1 {
        return false;
    }
    let input_shape = input.meta.shape();
    let weight_shape = weight.meta.shape();
    wgsl_float_dtype_is_supported(input.dtype)
        && [input, weight, bias]
            .into_iter()
            .all(|tensor| tensor.dtype == input.dtype && tensor.device == input.device)
        && [input_shape[0], input_shape[1], input_shape[2]] == [1, channels, length]
        && [weight_shape[0], weight_shape[1], weight_shape[2]] == [channels, channels, 7]
        && bias.meta.shape()[0] == channels
        && weight.is_contiguous()
        && bias.is_contiguous()
}

/// Fuse act1 into the measured residue-d1, T256, or T128 tile.
///
/// The original C192/L48000 d3+d9 residue one-shot was bit-identical over
/// 18,432,000 outputs and reduced the median sum from 11.693 ms to 7.914 ms
/// (3.779 ms, 1.477x). A failed residue preflight or try-launch retains the
/// complete prior chain below. The final rotating exact-twelve one-shot reduced the
/// fixed T128+Snake sum from 37.432 ms to a conservative per-shape 36.386 ms.
/// The subsequent vec4-store A/B was bit-identical on all nine T256 routes and
/// reduced their sum from 28.286 ms to 27.640 ms (0.646 ms, 1.023x). Eight
/// winners use vec4 stores; C768/L600/d9 retains scalar T256 because 1924.886 us
/// lost to 1922.423 us. A failed vec4 full-five/alignment contract falls through
/// to scalar T256, whose failure retains the complete accepted T128+Snake chain.
fn dilated_conv1d_act1_wgsl_or_fallback(
    conv: &Conv1d,
    act1: &Snake1d,
    packed_residue_weight: Option<&Tensor<3>>,
    input: Tensor<3>,
) -> Tensor<3> {
    use crate::kernels::conv1d_k7_snake_epilogue::{
        Conv1dK7SnakeTile, conv1d_k7_same_snake_epilogue_wgsl,
        device_supports_conv1d_k7_snake_epilogue,
    };

    let descriptor = Conv1dK7Descriptor::from_conv(conv, input.dims());
    let route = descriptor.route();
    let dilation = match route {
        Conv1dK7Route::TiledO16(dilation)
        | Conv1dK7Route::TiledO32Preferred(dilation)
        | Conv1dK7Route::TiledO64Preferred(dilation) => dilation,
        Conv1dK7Route::BurnFallback => {
            return standalone_dilated_conv1d_then_snake_wgsl(conv, act1, input);
        }
    };
    let Some(bias_param) = &conv.bias else {
        return standalone_dilated_conv1d_then_snake_wgsl(conv, act1, input);
    };

    let input_raw = input
        .clone()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    let weight = conv
        .weight
        .val()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    let bias = bias_param
        .val()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    let alpha = act1
        .alpha
        .val()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    let packed_residue_weight_raw = packed_residue_weight.cloned().map(|packed| {
        packed
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend")
    });
    let compatible_residue_dilation = packed_residue_weight_raw.as_ref().and_then(|packed| {
        select_compatible_conv1d_k7_residue_d1_dilation(descriptor, |residue_dilation| {
            crate::kernels::conv1d_k7_residue_d1_snake::conv1d_k7_residue_d1_snake_contract_is_compatible(
                &input_raw,
                packed,
                &bias,
                &alpha,
                residue_dilation,
            )
        })
    });
    if let Some(residue_dilation) = compatible_residue_dilation {
        let output = nvtx_range!(
            "codec_residual_conv_dilated_residue_d1_snake_1",
            crate::kernels::conv1d_k7_residue_d1_snake::try_conv1d_k7_same_residue_d1_snake_wgsl(
                input_raw.clone(),
                packed_residue_weight_raw
                    .clone()
                    .expect("residue route requires a validated weight-vector cache"),
                bias.clone(),
                alpha.clone(),
                residue_dilation,
            )
        );
        if let Some(output) = output {
            return Tensor::from_primitive::<crate::WgpuRaw>(output);
        }
    }
    let compatible_t256_vec4_tile = select_compatible_conv1d_k7_t256_snake_vec4_store_tile(
        descriptor,
        |tile| {
            packed_residue_weight_raw.as_ref().is_some_and(|packed| {
                crate::kernels::conv1d_k7_t256_snake_vec4_store::conv1d_k7_t256_snake_vec4_store_contract_is_compatible(
                    &input_raw,
                    packed,
                    &bias,
                    &alpha,
                    dilation,
                    tile,
                )
            })
        },
    );
    if let Some(tile) = compatible_t256_vec4_tile {
        let output = nvtx_range!(
            "codec_residual_conv_dilated_t256_snake_vec4_store_1",
                crate::kernels::conv1d_k7_t256_snake_vec4_store::try_conv1d_k7_same_t256_snake_vec4_store_wgsl(
                    input_raw.clone(),
                    packed_residue_weight_raw
                        .clone()
                        .expect("T256 vec4-store route requires a validated weight-vector cache"),
                bias.clone(),
                alpha.clone(),
                dilation,
                tile,
            )
        );
        if let Some(output) = output {
            return Tensor::from_primitive::<crate::WgpuRaw>(output);
        }
    }
    let compatible_t256_tile = select_compatible_conv1d_k7_t256_snake_tile(descriptor, |tile| {
        packed_residue_weight_raw.as_ref().is_some_and(|packed| {
            crate::kernels::conv1d_k7_t256_snake_epilogue::conv1d_k7_t256_snake_epilogue_contract_is_compatible(
                &input_raw,
                packed,
                &bias,
                &alpha,
                dilation,
                tile,
            )
        })
    });
    if let Some(tile) = compatible_t256_tile {
        let output = nvtx_range!(
            "codec_residual_conv_dilated_t256_snake_1",
            crate::kernels::conv1d_k7_t256_snake_epilogue::conv1d_k7_same_t256_snake_epilogue_wgsl(
                input_raw,
                packed_residue_weight_raw
                    .expect("T256 route requires a validated weight-vector cache"),
                bias,
                alpha,
                dilation,
                tile,
            )
        );
        return Tensor::from_primitive::<crate::WgpuRaw>(output);
    }
    let measured_t128_tile = descriptor.measured_t128_tile();
    let fused_t128_compatible = measured_t128_tile.is_some_and(|tile| {
        crate::kernels::conv1d_k7_t128_snake_epilogue::conv1d_k7_t128_snake_epilogue_contract_is_compatible(
            &input_raw,
            &weight,
            &bias,
            &alpha,
            dilation,
            tile,
        )
    });
    let raw_t128_compatible = fused_t128_compatible
        || measured_t128_tile.is_some_and(|tile| {
            crate::kernels::conv1d_k7_t128::conv1d_k7_t128_contract_is_compatible(
                &input_raw, &weight, &bias, dilation, tile,
            )
        });
    match select_conv1d_k7_t128_snake_route(
        measured_t128_tile,
        fused_t128_compatible,
        raw_t128_compatible,
    ) {
        Conv1dK7T128SnakeRoute::Fused(tile) => {
            let output = nvtx_range!(
                "codec_residual_conv_dilated_t128_snake_1",
                crate::kernels::conv1d_k7_t128_snake_epilogue::conv1d_k7_same_t128_snake_epilogue_wgsl(
                    input_raw,
                    weight,
                    bias,
                    alpha,
                    dilation,
                    tile,
                )
            );
            return Tensor::from_primitive::<crate::WgpuRaw>(output);
        }
        Conv1dK7T128SnakeRoute::Materialized(tile) => {
            let output = nvtx_range!(
                "codec_residual_conv_dilated",
                crate::kernels::conv1d_k7_t128::conv1d_k7_same_t128_wgsl(
                    input_raw, weight, bias, dilation, tile,
                )
            );
            let output = Tensor::from_primitive::<crate::WgpuRaw>(output);
            return nvtx_range!("codec_residual_snake_1", act1.forward_wgsl(output));
        }
        Conv1dK7T128SnakeRoute::Legacy => {}
    }
    if matches!(route, Conv1dK7Route::TiledO64Preferred(_))
        && crate::kernels::conv1d_k7_tiled_o64::conv1d_k7_tiled_o64_contract_is_compatible(
            &input_raw, &weight, &bias, dilation,
        )
    {
        return standalone_dilated_conv1d_then_snake_wgsl(conv, act1, input);
    }
    let contract_compatible = conv1d_k7_snake_epilogue_contract_is_compatible(
        &input_raw,
        &weight,
        &bias,
        &alpha,
        descriptor.output_channels,
        descriptor.length,
    );
    if !contract_compatible {
        return standalone_dilated_conv1d_then_snake_wgsl(conv, act1, input);
    }

    let o16_supported =
        device_supports_conv1d_k7_snake_epilogue(&input_raw, dilation, Conv1dK7SnakeTile::Output16);
    let o32_supported = matches!(
        route,
        Conv1dK7Route::TiledO32Preferred(_) | Conv1dK7Route::TiledO64Preferred(_)
    ) && device_supports_conv1d_k7_snake_epilogue(
        &input_raw,
        dilation,
        Conv1dK7SnakeTile::Output32,
    );
    let Some(tile) = select_conv1d_k7_snake_epilogue_tile(
        route,
        contract_compatible,
        o16_supported,
        o32_supported,
    ) else {
        return standalone_dilated_conv1d_then_snake_wgsl(conv, act1, input);
    };

    let output = nvtx_range!(
        "codec_residual_conv_dilated_snake_1",
        conv1d_k7_same_snake_epilogue_wgsl(input_raw, weight, bias, alpha, dilation, tile)
    );
    Tensor::from_primitive::<crate::WgpuRaw>(output)
}

/// Use a measured production tiled k=7 kernel only for its exact contract.
///
/// All twelve released shapes first preflight their measured T128/O32 Cin16 or
/// Cin8 tile. The one-shot was bit-exact and reduced the raw-k7 median sum from
/// 55.478 ms to 37.753 ms (1.470x). Any T128 dtype/layout/device/resource
/// mismatch falls through to the accepted O64, then O32/O16 selector without
/// entering its asserting launcher. The act1 wrapper composes this raw kernel
/// with standalone Snake and retains fused k7+Snake as its fallback. Any
/// structural mismatch, absent bias, or non-contiguous parameter still falls
/// back to Burn.
fn dilated_conv1d_wgsl_or_fallback(conv: &Conv1d, input: Tensor<3>) -> Tensor<3> {
    let descriptor = Conv1dK7Descriptor::from_conv(conv, input.dims());
    let route = descriptor.route();
    let dilation = match route {
        Conv1dK7Route::TiledO16(dilation)
        | Conv1dK7Route::TiledO32Preferred(dilation)
        | Conv1dK7Route::TiledO64Preferred(dilation) => dilation,
        Conv1dK7Route::BurnFallback => return conv.forward(input),
    };
    let Some(bias) = &conv.bias else {
        return conv.forward(input);
    };

    let input_raw = input
        .clone()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    let weight = conv
        .weight
        .val()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    let bias = bias
        .val()
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("tensor must use WGPU raw backend");
    if !conv1d_k7_standalone_base_contract_is_compatible(
        &input_raw,
        &weight,
        &bias,
        descriptor.output_channels,
        descriptor.length,
    ) {
        return conv.forward(input);
    }
    let compatible_t128_tile = select_compatible_conv1d_k7_t128_tile(descriptor, |tile| {
        crate::kernels::conv1d_k7_t128::conv1d_k7_t128_contract_is_compatible(
            &input_raw, &weight, &bias, dilation, tile,
        )
    });
    let o64_contract_compatible = matches!(route, Conv1dK7Route::TiledO64Preferred(_))
        && crate::kernels::conv1d_k7_tiled_o64::conv1d_k7_tiled_o64_contract_is_compatible(
            &input_raw, &weight, &bias, dilation,
        );
    let o32_supported = matches!(
        route,
        Conv1dK7Route::TiledO32Preferred(_) | Conv1dK7Route::TiledO64Preferred(_)
    )
        && crate::kernels::conv1d_k7_tiled_o32::device_supports_conv1d_k7_tiled_o32(
            &input_raw, dilation,
        );
    let Some(tile) = select_conv1d_k7_standalone_tile(
        route,
        compatible_t128_tile,
        o64_contract_compatible,
        o32_supported,
    ) else {
        return conv.forward(input);
    };
    let output = match tile {
        Conv1dK7StandaloneTile::T128(tile) => {
            crate::kernels::conv1d_k7_t128::conv1d_k7_same_t128_wgsl(
                input_raw, weight, bias, dilation, tile,
            )
        }
        Conv1dK7StandaloneTile::Output64 => {
            crate::kernels::conv1d_k7_tiled_o64::conv1d_k7_same_tiled_o64_wgsl(
                input_raw, weight, bias, dilation,
            )
        }
        Conv1dK7StandaloneTile::Output32 => {
            crate::kernels::conv1d_k7_tiled_o32::conv1d_k7_same_tiled_o32_wgsl(
                input_raw, weight, bias, dilation,
            )
        }
        Conv1dK7StandaloneTile::Output16 => {
            crate::kernels::conv1d_k7_tiled::conv1d_k7_same_tiled_wgsl(
                input_raw, weight, bias, dilation,
            )
        }
    };
    Tensor::from_primitive::<crate::WgpuRaw>(output)
}

// ─── Padding helpers ─────────────────────────────────────────────────────────

/// Symmetric explicit padding for a Conv1d with `pad_mode="none"`:
/// `pad = (kernel - stride) * dilation / 2`.
pub(crate) fn conv_pad(kernel: usize, stride: usize, dilation: usize) -> usize {
    (kernel - stride) * dilation / 2
}

/// PyTorch `ConvTranspose1d` padding for integer strides with `pad_mode="none"`:
/// `(padding, output_padding) = ((stride+1)//2, stride%2)`.
/// All decoder strides are even so `output_padding` is always 0.
pub(crate) fn conv_transpose_pad(stride: usize) -> (usize, usize) {
    (stride.div_ceil(2), stride % 2)
}

// ─── Conv1d construction helper ───────────────────────────────────────────────

use burn::nn::conv::Conv1dConfig;

/// Create a `Conv1d` module with pre-loaded weights/bias, using symmetric explicit padding.
///
/// `pad = (kernel - stride) * dilation / 2`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn make_conv1d(
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    weight: Tensor<3>,
    bias: Option<Tensor<1>>,
    device: &Device,
) -> Conv1d {
    let pad = conv_pad(kernel, stride, dilation);
    let mut conv = Conv1dConfig::new(in_ch, out_ch, kernel)
        .with_stride(stride)
        .with_dilation(dilation)
        .with_padding(PaddingConfig1d::Explicit(pad, pad))
        .with_bias(bias.is_some())
        .init(device);
    conv.weight = Param::initialized(ParamId::new(), weight);
    conv.bias = bias.map(|b| Param::initialized(ParamId::new(), b));
    conv
}

/// Apply a stride-one, ungrouped 1x1 convolution as a channel-last GEMM.
///
/// CubeCL's generic convolution path performs layout preparation that is
/// disproportionately expensive for the codec's pointwise residual
/// projections. The equivalent matrix multiplication keeps the temporal axis
/// as the GEMM row dimension and reuses the tuned matmul implementation.
pub(crate) fn pointwise_conv1d(conv: &Conv1d, input: Tensor<3>) -> Tensor<3> {
    pointwise_conv1d_with_weight(conv, None, input)
}

pub(crate) fn pack_pointwise_conv1d_weight(conv: &Conv1d) -> Tensor<3> {
    debug_assert_eq!(conv.kernel_size, 1);
    debug_assert_eq!(conv.groups, 1);
    conv.weight
        .val()
        .squeeze_dim::<2>(2)
        .transpose()
        .add_scalar(0.0)
        .unsqueeze_dim::<3>(0)
}

fn pointwise_conv1d_with_weight(
    conv: &Conv1d,
    packed_weight: Option<&Tensor<3>>,
    input: Tensor<3>,
) -> Tensor<3> {
    let [batch, _, length] = input.dims();
    let [output_channels, _, _] = conv.weight.dims();
    let mut output = pointwise_conv1d_matmul_nlc_with_weight(conv, packed_weight, input);
    if let Some(bias) = &conv.bias {
        output = output + bias.val().reshape([1, 1, output_channels]);
    }
    output
        .swap_dims(1, 2)
        .reshape([batch, output_channels, length])
}

/// Return only the packed GEMM result in contiguous physical NLC order.
///
/// The production pointwise finalizer consumes this before any bias kernel is
/// launched. The existing helper above retains the original bias and NCL-view
/// sequence for every fallback and generic backend.
fn pointwise_conv1d_matmul_nlc_with_weight(
    conv: &Conv1d,
    packed_weight: Option<&Tensor<3>>,
    input: Tensor<3>,
) -> Tensor<3> {
    debug_assert_eq!(conv.kernel_size, 1);
    debug_assert_eq!(conv.stride, 1);
    debug_assert_eq!(conv.dilation, 1);
    debug_assert_eq!(conv.groups, 1);

    let [_, input_channels, _] = input.dims();
    let [_, weight_input_channels, kernel] = conv.weight.dims();
    debug_assert_eq!(input_channels, weight_input_channels);
    debug_assert_eq!(kernel, 1);

    let weight = packed_weight
        .cloned()
        .unwrap_or_else(|| pack_pointwise_conv1d_weight(conv));
    input.swap_dims(1, 2).matmul(weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointwise_row_policy_separates_geometry_from_forced_single() {
        assert!(!PointwiseRowsPolicy::Geometry.enabled(600, 768));
        assert!(PointwiseRowsPolicy::Geometry.enabled(6_000, 384));
        #[cfg(feature = "profile")]
        assert!(!PointwiseRowsPolicy::SingleRow.enabled(96_000, 96));
    }

    #[test]
    fn multi_rows_geometry_selects_wide_non_tall_k7_matrices() {
        let selected = K7MultiRowsSelection::GeometrySelected;
        assert!(!selected.enabled(540, 768));
        assert!(selected.enabled(1_344, 768));
        assert!(selected.enabled(5_400, 384));
        assert!(selected.enabled(82_200, 384));
        assert!(!selected.enabled(43_200, 192));
        assert!(!selected.enabled(86_400, 96));
        assert!(K7MultiRowsSelection::Forced.enabled(1, 1));
        assert!(!K7MultiRowsSelection::Disabled.enabled(1_000_000, 768));
        assert_eq!(
            multi_rows_k7_selection(CodecK7Algorithm::AccuracyApproved),
            K7MultiRowsSelection::GeometrySelected
        );
        assert_eq!(
            multi_rows_k7_selection(CodecK7Algorithm::CubeClImplicitGemm),
            K7MultiRowsSelection::Disabled
        );
        #[cfg(feature = "profile")]
        assert_eq!(
            multi_rows_k7_selection(CodecK7Algorithm::CubeClImplicitGemmAutotuned),
            K7MultiRowsSelection::Autotuned
        );
    }

    #[test]
    #[cfg(feature = "profile")]
    fn k7_autotune_key_is_exact_and_persistable() {
        let key = K7AutotuneKey {
            schema: 2,
            dtype: "f16".to_owned(),
            batch: 1,
            input_length: 1_344,
            input_channels: 384,
            output_length: 1_344,
            output_channels: 384,
            dilation: 3,
            input_strides: vec![516_096, 384, 1],
            weight_strides: vec![2_688, 1, 384],
        };
        let encoded = serde_json::to_string(&key).expect("k7 key must serialize");
        let restored: K7AutotuneKey =
            serde_json::from_str(&encoded).expect("k7 key must deserialize");
        assert_eq!(restored, key);
        assert!(
            key.to_string()
                .contains("k7-s2-f16-b1-l1344x1344-c384x384-d3")
        );

        let mut different_shape = key.clone();
        different_shape.output_length += 1;
        assert_ne!(different_shape, key);

        let mut different_layout = key.clone();
        different_layout.weight_strides[2] = 1;
        assert_ne!(different_layout, key);
    }

    fn f16_wgpu_test_device() -> Device {
        use std::sync::OnceLock;

        use burn::backend::wgpu::{
            MemoryConfiguration, RuntimeOptions, WgpuDevice, graphics::AutoGraphicsApi, init_setup,
        };

        static DEVICE: OnceLock<Device> = OnceLock::new();
        DEVICE
            .get_or_init(|| {
                let wgpu = WgpuDevice::DiscreteGpu(0);
                init_setup::<AutoGraphicsApi>(
                    &wgpu,
                    RuntimeOptions {
                        tasks_max: 32,
                        memory_config: MemoryConfiguration::SubSlices,
                    },
                );
                crate::backend_config::wgpu_device_with_precision(
                    &wgpu,
                    crate::backend_config::WgpuFloatPrecision::Fp16,
                )
                .expect("F16 WGPU test device must initialize")
            })
            .clone()
    }

    fn decoder_k7_descriptor(
        channels: usize,
        length: usize,
        dilation: usize,
    ) -> Conv1dK7Descriptor {
        Conv1dK7Descriptor {
            batch: 1,
            input_channels: channels,
            length,
            output_channels: channels,
            weight_input_channels: channels,
            kernel_size: 7,
            stride: 1,
            dilation,
            groups: 1,
            explicit_padding: Some((3 * dilation, 3 * dilation)),
            bias_channels: Some(channels),
        }
    }

    fn decoder_pointwise_descriptor(channels: usize, length: usize) -> PointwiseResidualDescriptor {
        PointwiseResidualDescriptor {
            batch: 1,
            input_channels: channels,
            length,
            output_channels: channels,
            weight_input_channels: channels,
            kernel_size: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
            explicit_padding: Some((0, 0)),
            bias_channels: Some(channels),
            packed_weight: Some([1, channels, channels]),
        }
    }

    #[test]
    fn exact_twelve_decoder_pointwise_units_select_measured_routes() {
        let cases = [
            (0, 0, 768, 600),
            (0, 1, 768, 600),
            (0, 2, 768, 600),
            (1, 0, 384, 6_000),
            (1, 1, 384, 6_000),
            (1, 2, 384, 6_000),
            (2, 0, 192, 48_000),
            (2, 1, 192, 48_000),
            (2, 2, 192, 48_000),
            (3, 0, 96, 96_000),
            (3, 1, 96, 96_000),
            (3, 2, 96, 96_000),
        ];
        for (stage, unit, channels, length) in cases {
            let expected = if stage >= 1 {
                PointwiseResidualRoute::DirectThenFinalizer
            } else {
                PointwiseResidualRoute::FusedFinalizer
            };
            assert_eq!(
                decoder_pointwise_descriptor(channels, length).route(),
                expected,
                "decoder stage={stage} unit={unit} must select its measured pointwise route"
            );
        }
    }

    #[test]
    fn pointwise_finalizer_route_falls_back_on_every_logical_contract_mismatch() {
        let supported = decoder_pointwise_descriptor(96, 96_000);
        let unsupported = [
            PointwiseResidualDescriptor {
                batch: 2,
                ..supported
            },
            PointwiseResidualDescriptor {
                length: 0,
                ..supported
            },
            PointwiseResidualDescriptor {
                output_channels: 95,
                ..supported
            },
            PointwiseResidualDescriptor {
                weight_input_channels: 95,
                ..supported
            },
            PointwiseResidualDescriptor {
                kernel_size: 3,
                ..supported
            },
            PointwiseResidualDescriptor {
                stride: 2,
                ..supported
            },
            PointwiseResidualDescriptor {
                dilation: 2,
                ..supported
            },
            PointwiseResidualDescriptor {
                groups: 2,
                ..supported
            },
            PointwiseResidualDescriptor {
                explicit_padding: Some((1, 1)),
                ..supported
            },
            PointwiseResidualDescriptor {
                bias_channels: None,
                ..supported
            },
            PointwiseResidualDescriptor {
                bias_channels: Some(95),
                ..supported
            },
            PointwiseResidualDescriptor {
                packed_weight: None,
                ..supported
            },
            PointwiseResidualDescriptor {
                packed_weight: Some([1, 96, 95]),
                ..supported
            },
        ];

        assert!(
            unsupported.into_iter().all(|descriptor| {
                descriptor.route() == PointwiseResidualRoute::ExistingFallback
            })
        );
    }

    #[test]
    fn variable_length_decoder_pointwise_units_keep_fast_routes() {
        for latent_steps in [13, 25, 50, 100, 200] {
            let stages = [
                (768, latent_steps * 12),
                (384, latent_steps * 120),
                (192, latent_steps * 960),
                (96, latent_steps * 1_920),
            ];
            for (stage, (channels, length)) in stages.into_iter().enumerate() {
                let expected = if stage >= 1 {
                    PointwiseResidualRoute::DirectThenFinalizer
                } else {
                    PointwiseResidualRoute::FusedFinalizer
                };
                assert_eq!(
                    decoder_pointwise_descriptor(channels, length).route(),
                    expected,
                    "latent_steps={latent_steps} stage={stage} C={channels} L={length}",
                );
            }
        }
    }

    #[test]
    fn official_decoder_residuals_select_measured_k7_route() {
        use crate::kernels::conv1d_k7_tiled::Conv1dK7Dilation;

        let cases = [
            (
                768,
                600,
                1,
                Conv1dK7Route::TiledO32Preferred(Conv1dK7Dilation::One),
            ),
            (
                768,
                600,
                3,
                Conv1dK7Route::TiledO16(Conv1dK7Dilation::Three),
            ),
            (
                768,
                600,
                9,
                Conv1dK7Route::TiledO64Preferred(Conv1dK7Dilation::Nine),
            ),
            (
                384,
                6_000,
                1,
                Conv1dK7Route::TiledO64Preferred(Conv1dK7Dilation::One),
            ),
            (
                384,
                6_000,
                3,
                Conv1dK7Route::TiledO64Preferred(Conv1dK7Dilation::Three),
            ),
            (
                384,
                6_000,
                9,
                Conv1dK7Route::TiledO32Preferred(Conv1dK7Dilation::Nine),
            ),
            (
                192,
                48_000,
                1,
                Conv1dK7Route::TiledO64Preferred(Conv1dK7Dilation::One),
            ),
            (
                192,
                48_000,
                3,
                Conv1dK7Route::TiledO64Preferred(Conv1dK7Dilation::Three),
            ),
            (
                192,
                48_000,
                9,
                Conv1dK7Route::TiledO32Preferred(Conv1dK7Dilation::Nine),
            ),
            (
                96,
                96_000,
                1,
                Conv1dK7Route::TiledO64Preferred(Conv1dK7Dilation::One),
            ),
            (
                96,
                96_000,
                3,
                Conv1dK7Route::TiledO64Preferred(Conv1dK7Dilation::Three),
            ),
            (
                96,
                96_000,
                9,
                Conv1dK7Route::TiledO32Preferred(Conv1dK7Dilation::Nine),
            ),
        ];
        for (channels, length, dilation, route) in cases {
            assert_eq!(
                decoder_k7_descriptor(channels, length, dilation).route(),
                route,
                "released decoder C={channels}, L={length}, dilation={dilation} must use its measured tile preference"
            );
        }
    }

    #[test]
    fn variable_length_decoder_residuals_reuse_measured_k7_tiles() {
        let reference_lengths = [(768, 600), (384, 6_000), (192, 48_000), (96, 96_000)];
        for latent_steps in [13, 25, 50, 100, 200] {
            let lengths = [
                latent_steps * 12,
                latent_steps * 120,
                latent_steps * 960,
                latent_steps * 1_920,
            ];
            for ((channels, reference_length), length) in reference_lengths.into_iter().zip(lengths)
            {
                for dilation in [1, 3, 9] {
                    let reference = decoder_k7_descriptor(channels, reference_length, dilation);
                    let candidate = decoder_k7_descriptor(channels, length, dilation);
                    assert_eq!(candidate.route(), reference.route());
                    assert_eq!(
                        candidate.measured_t128_tile(),
                        reference.measured_t128_tile()
                    );
                    assert_eq!(
                        candidate.measured_t256_snake_tile(),
                        reference.measured_t256_snake_tile(),
                    );
                    assert_eq!(
                        candidate.measured_t256_snake_vec4_store_tile(),
                        reference.measured_t256_snake_vec4_store_tile(),
                    );
                }
            }
        }
    }

    #[test]
    fn official_decoder_residuals_select_measured_t128_reduction_tile() {
        use crate::kernels::conv1d_k7_t128::Conv1dK7T128Tile;

        let cases = [
            (768, 600, 1, Conv1dK7T128Tile::Cin16),
            (768, 600, 3, Conv1dK7T128Tile::Cin16),
            (768, 600, 9, Conv1dK7T128Tile::Cin8),
            (384, 6_000, 1, Conv1dK7T128Tile::Cin16),
            (384, 6_000, 3, Conv1dK7T128Tile::Cin16),
            (384, 6_000, 9, Conv1dK7T128Tile::Cin8),
            (192, 48_000, 1, Conv1dK7T128Tile::Cin8),
            (192, 48_000, 3, Conv1dK7T128Tile::Cin8),
            (192, 48_000, 9, Conv1dK7T128Tile::Cin8),
            (96, 96_000, 1, Conv1dK7T128Tile::Cin16),
            (96, 96_000, 3, Conv1dK7T128Tile::Cin16),
            (96, 96_000, 9, Conv1dK7T128Tile::Cin8),
        ];
        for (channels, length, dilation, tile) in cases {
            assert_eq!(
                decoder_k7_descriptor(channels, length, dilation).measured_t128_tile(),
                Some(tile),
                "released decoder C={channels}, L={length}, dilation={dilation} must select its measured T128 reduction tile"
            );
        }
    }

    #[test]
    fn official_decoder_residuals_select_conservative_t256_snake_tiles() {
        use crate::kernels::conv1d_k7_t256_snake_epilogue::Conv1dK7T256Tile;

        let cases = [
            (768, 600, 1, Some(Conv1dK7T256Tile::Cin16)),
            (768, 600, 3, None),
            (768, 600, 9, Some(Conv1dK7T256Tile::Cin16)),
            (384, 6_000, 1, Some(Conv1dK7T256Tile::Cin16)),
            (384, 6_000, 3, None),
            (384, 6_000, 9, None),
            (192, 48_000, 1, Some(Conv1dK7T256Tile::Cin16)),
            (192, 48_000, 3, Some(Conv1dK7T256Tile::Cin16)),
            (192, 48_000, 9, Some(Conv1dK7T256Tile::Cin8)),
            (96, 96_000, 1, Some(Conv1dK7T256Tile::Cin16)),
            (96, 96_000, 3, Some(Conv1dK7T256Tile::Cin16)),
            (96, 96_000, 9, Some(Conv1dK7T256Tile::Cin8)),
        ];
        for (channels, length, dilation, expected) in cases {
            let descriptor = decoder_k7_descriptor(channels, length, dilation);
            assert_eq!(descriptor.measured_t256_snake_tile(), expected);
            if expected.is_none() {
                assert!(
                    descriptor.measured_t128_tile().is_some(),
                    "T256 non-selected shapes must retain the T128 selector",
                );
            }
        }
    }

    #[test]
    fn production_vec4_store_selector_keeps_exactly_eight_t256_wins() {
        use crate::kernels::conv1d_k7_t256_snake_epilogue::Conv1dK7T256Tile;

        let cases = [
            (768, 600, 1, Some(Conv1dK7T256Tile::Cin16)),
            (768, 600, 3, None),
            (768, 600, 9, None),
            (384, 6_000, 1, Some(Conv1dK7T256Tile::Cin16)),
            (384, 6_000, 3, None),
            (384, 6_000, 9, None),
            (192, 48_000, 1, Some(Conv1dK7T256Tile::Cin16)),
            (192, 48_000, 3, Some(Conv1dK7T256Tile::Cin16)),
            (192, 48_000, 9, Some(Conv1dK7T256Tile::Cin8)),
            (96, 96_000, 1, Some(Conv1dK7T256Tile::Cin16)),
            (96, 96_000, 3, Some(Conv1dK7T256Tile::Cin16)),
            (96, 96_000, 9, Some(Conv1dK7T256Tile::Cin8)),
        ];
        assert_eq!(cases.into_iter().filter(|case| case.3.is_some()).count(), 8);
        for (channels, length, dilation, expected) in cases {
            assert_eq!(
                decoder_k7_descriptor(channels, length, dilation)
                    .measured_t256_snake_vec4_store_tile(),
                expected,
            );
        }
    }

    #[test]
    fn production_residue_d1_selector_keeps_three_dilations_per_admitted_shape() {
        use crate::kernels::conv1d_k7_residue_d1_snake::ResidueDilation;

        let cases = [
            (768, 600, 1, None),
            (768, 600, 3, None),
            (768, 600, 9, None),
            (384, 6_000, 1, Some(ResidueDilation::One)),
            (384, 6_000, 3, Some(ResidueDilation::Three)),
            (384, 6_000, 9, Some(ResidueDilation::Nine)),
            (192, 48_000, 1, Some(ResidueDilation::One)),
            (192, 48_000, 3, Some(ResidueDilation::Three)),
            (192, 48_000, 9, Some(ResidueDilation::Nine)),
            (96, 96_000, 1, Some(ResidueDilation::One)),
            (96, 96_000, 3, Some(ResidueDilation::Three)),
            (96, 96_000, 9, Some(ResidueDilation::Nine)),
        ];
        assert_eq!(cases.into_iter().filter(|case| case.3.is_some()).count(), 9);
        for (channels, length, dilation, expected) in cases {
            assert_eq!(
                decoder_k7_descriptor(channels, length, dilation).measured_residue_d1_dilation(),
                expected,
            );
        }
        for length in [12_480, 24_000, 96_000, 192_000] {
            assert_eq!(
                decoder_k7_descriptor(192, length, 1).measured_residue_d1_dilation(),
                Some(ResidueDilation::One),
            );
            assert_eq!(
                decoder_k7_descriptor(192, length, 3).measured_residue_d1_dilation(),
                Some(ResidueDilation::Three),
            );
            assert_eq!(
                decoder_k7_descriptor(192, length, 9).measured_residue_d1_dilation(),
                Some(ResidueDilation::Nine),
            );
        }
        for length in [96_000, 192_000, 384_000] {
            assert_eq!(
                decoder_k7_descriptor(96, length, 1).measured_residue_d1_dilation(),
                Some(ResidueDilation::One),
            );
            assert_eq!(
                decoder_k7_descriptor(96, length, 3).measured_residue_d1_dilation(),
                Some(ResidueDilation::Three),
            );
            assert_eq!(
                decoder_k7_descriptor(96, length, 9).measured_residue_d1_dilation(),
                Some(ResidueDilation::Nine),
            );
        }
        for length in [6_000, 12_000, 24_000] {
            assert_eq!(
                decoder_k7_descriptor(384, length, 1).measured_residue_d1_dilation(),
                Some(ResidueDilation::One),
            );
            assert_eq!(
                decoder_k7_descriptor(384, length, 3).measured_residue_d1_dilation(),
                Some(ResidueDilation::Three),
            );
            assert_eq!(
                decoder_k7_descriptor(384, length, 9).measured_residue_d1_dilation(),
                Some(ResidueDilation::Nine),
            );
        }
    }

    #[test]
    fn residue_d1_preflight_failure_retains_current_t256_chain() {
        use crate::kernels::{
            conv1d_k7_residue_d1_snake::ResidueDilation,
            conv1d_k7_t256_snake_epilogue::Conv1dK7T256Tile,
        };

        let winning_descriptor = decoder_k7_descriptor(192, 48_000, 9);
        assert_eq!(
            select_compatible_conv1d_k7_residue_d1_dilation(winning_descriptor, |_| true),
            Some(ResidueDilation::Nine),
        );
        assert_eq!(
            select_compatible_conv1d_k7_residue_d1_dilation(winning_descriptor, |_| false),
            None,
            "failed residue preflight must retain the prior production chain",
        );
        assert_eq!(
            winning_descriptor.measured_t256_snake_vec4_store_tile(),
            Some(Conv1dK7T256Tile::Cin8),
        );

        let dilation_one = decoder_k7_descriptor(192, 48_000, 1);
        let mut residue_preflight_called = false;
        assert_eq!(
            select_compatible_conv1d_k7_residue_d1_dilation(dilation_one, |_| {
                residue_preflight_called = true;
                true
            }),
            Some(ResidueDilation::One),
        );
        assert!(residue_preflight_called);
        assert_eq!(
            select_compatible_conv1d_k7_residue_d1_dilation(dilation_one, |_| false),
            None,
        );
        assert_eq!(
            dilation_one.measured_t256_snake_vec4_store_tile(),
            Some(Conv1dK7T256Tile::Cin16),
        );
    }

    #[test]
    fn t128_shape_selector_is_fail_closed_on_logical_mismatches() {
        let supported = decoder_k7_descriptor(96, 96_000, 1);
        let unsupported = [
            decoder_k7_descriptor(32, 73, 1),
            Conv1dK7Descriptor {
                batch: 2,
                ..supported
            },
            Conv1dK7Descriptor {
                output_channels: 95,
                ..supported
            },
            Conv1dK7Descriptor {
                weight_input_channels: 95,
                ..supported
            },
            Conv1dK7Descriptor {
                kernel_size: 5,
                ..supported
            },
            Conv1dK7Descriptor {
                stride: 2,
                ..supported
            },
            Conv1dK7Descriptor {
                dilation: 2,
                explicit_padding: Some((6, 6)),
                ..supported
            },
            Conv1dK7Descriptor {
                groups: 2,
                ..supported
            },
            Conv1dK7Descriptor {
                explicit_padding: Some((2, 3)),
                ..supported
            },
            Conv1dK7Descriptor {
                bias_channels: None,
                ..supported
            },
        ];
        assert!(
            unsupported
                .into_iter()
                .all(|descriptor| descriptor.measured_t128_tile().is_none())
        );
        assert!(
            unsupported
                .into_iter()
                .all(|descriptor| descriptor.measured_t256_snake_tile().is_none())
        );
        assert!(
            unsupported
                .into_iter()
                .all(|descriptor| descriptor.measured_t256_snake_vec4_store_tile().is_none())
        );
        assert!(
            unsupported
                .into_iter()
                .all(|descriptor| descriptor.measured_residue_d1_dilation().is_none())
        );
    }

    #[test]
    fn vec4_store_preflight_and_policy_loss_retain_scalar_t256() {
        use crate::kernels::conv1d_k7_t256_snake_epilogue::Conv1dK7T256Tile;

        let winning_descriptor = decoder_k7_descriptor(192, 48_000, 9);
        assert_eq!(
            select_compatible_conv1d_k7_t256_snake_vec4_store_tile(winning_descriptor, |_| true,),
            Some(Conv1dK7T256Tile::Cin8),
        );
        assert_eq!(
            select_compatible_conv1d_k7_t256_snake_vec4_store_tile(winning_descriptor, |_| false,),
            None,
            "failed vec4 full-five/alignment preflight must retain scalar T256",
        );
        assert_eq!(
            select_compatible_conv1d_k7_t256_snake_tile(winning_descriptor, |_| true),
            Some(Conv1dK7T256Tile::Cin8),
        );

        let retained_scalar = decoder_k7_descriptor(768, 600, 9);
        let mut vec4_preflight_called = false;
        assert_eq!(
            select_compatible_conv1d_k7_t256_snake_vec4_store_tile(retained_scalar, |_| {
                vec4_preflight_called = true;
                true
            }),
            None,
        );
        assert!(!vec4_preflight_called);
        assert_eq!(
            select_compatible_conv1d_k7_t256_snake_tile(retained_scalar, |_| true),
            Some(Conv1dK7T256Tile::Cin16),
            "measured C768/L600/d9 loss must retain scalar T256",
        );
    }

    #[test]
    fn t256_snake_physical_preflight_falls_through_to_t128() {
        use crate::kernels::conv1d_k7_t256_snake_epilogue::Conv1dK7T256Tile;

        let descriptor = decoder_k7_descriptor(192, 48_000, 9);
        assert_eq!(
            select_compatible_conv1d_k7_t256_snake_tile(descriptor, |_| true),
            Some(Conv1dK7T256Tile::Cin8),
        );
        assert_eq!(
            select_compatible_conv1d_k7_t256_snake_tile(descriptor, |_| false),
            None,
            "failed full-five preflight must retain the current T128 chain",
        );
        assert!(descriptor.measured_t128_tile().is_some());

        let mut physical_preflight_called = false;
        assert_eq!(
            select_compatible_conv1d_k7_t256_snake_tile(
                decoder_k7_descriptor(384, 6_000, 3),
                |_| {
                    physical_preflight_called = true;
                    true
                },
            ),
            None,
        );
        assert!(
            !physical_preflight_called,
            "a T128-retained logical route must not run T256 physical preflight",
        );
    }

    #[test]
    fn t128_physical_preflight_is_required_before_route_selection() {
        use crate::kernels::conv1d_k7_t128::Conv1dK7T128Tile;

        let descriptor = decoder_k7_descriptor(768, 600, 3);
        assert_eq!(
            select_compatible_conv1d_k7_t128_tile(descriptor, |_| true),
            Some(Conv1dK7T128Tile::Cin16)
        );
        assert_eq!(
            select_compatible_conv1d_k7_t128_tile(descriptor, |_| false),
            None,
            "a failed dtype/layout/device/resource preflight must retain the old fused path"
        );

        let mut physical_preflight_called = false;
        assert_eq!(
            select_compatible_conv1d_k7_t128_tile(decoder_k7_descriptor(32, 73, 1), |_| {
                physical_preflight_called = true;
                true
            },),
            None
        );
        assert!(
            !physical_preflight_called,
            "non-released logical shapes must not reach T128 physical preflight"
        );
    }

    #[test]
    fn t128_snake_route_is_fail_closed_across_both_physical_contracts() {
        use crate::kernels::conv1d_k7_t128::Conv1dK7T128Tile;

        let tile = Conv1dK7T128Tile::Cin16;
        for (fused, raw, expected) in [
            (true, true, Conv1dK7T128SnakeRoute::Fused(tile)),
            (false, true, Conv1dK7T128SnakeRoute::Materialized(tile)),
            (true, false, Conv1dK7T128SnakeRoute::Legacy),
            (false, false, Conv1dK7T128SnakeRoute::Legacy),
        ] {
            assert_eq!(
                select_conv1d_k7_t128_snake_route(Some(tile), fused, raw),
                expected,
                "Some(tile), fused={fused}, raw={raw}"
            );
            assert_eq!(
                select_conv1d_k7_t128_snake_route(None, fused, raw),
                Conv1dK7T128SnakeRoute::Legacy,
                "None, fused={fused}, raw={raw}"
            );
        }
    }

    #[test]
    fn standalone_o64_preflight_falls_back_to_the_previous_selector() {
        use crate::kernels::{conv1d_k7_t128::Conv1dK7T128Tile, conv1d_k7_tiled::Conv1dK7Dilation};

        let preferred_o64 = Conv1dK7Route::TiledO64Preferred(Conv1dK7Dilation::One);
        let measured_o16 = Conv1dK7Route::TiledO16(Conv1dK7Dilation::Three);
        assert_eq!(
            select_conv1d_k7_standalone_tile(
                preferred_o64,
                Some(Conv1dK7T128Tile::Cin16),
                false,
                false,
            ),
            Some(Conv1dK7StandaloneTile::T128(Conv1dK7T128Tile::Cin16)),
            "a compatible T128 tile must precede the old selector"
        );
        assert_eq!(
            select_conv1d_k7_standalone_tile(
                measured_o16,
                Some(Conv1dK7T128Tile::Cin16),
                false,
                false,
            ),
            Some(Conv1dK7StandaloneTile::T128(Conv1dK7T128Tile::Cin16)),
            "the measured raw T128 tile supersedes the old O16 exception"
        );
        assert_eq!(
            select_conv1d_k7_standalone_tile(preferred_o64, None, true, true),
            Some(Conv1dK7StandaloneTile::Output64)
        );
        assert_eq!(
            select_conv1d_k7_standalone_tile(preferred_o64, None, false, true),
            Some(Conv1dK7StandaloneTile::Output32)
        );
        assert_eq!(
            select_conv1d_k7_standalone_tile(preferred_o64, None, false, false),
            Some(Conv1dK7StandaloneTile::Output16)
        );
        assert_eq!(
            select_conv1d_k7_standalone_tile(
                Conv1dK7Route::BurnFallback,
                Some(Conv1dK7T128Tile::Cin8),
                true,
                true,
            ),
            None,
            "a logically invalid route must reject even an inconsistent T128 hint"
        );
    }

    #[test]
    fn act1_epilogue_route_preserves_tile_preference_and_fallback() {
        use crate::kernels::{
            conv1d_k7_snake_epilogue::Conv1dK7SnakeTile, conv1d_k7_tiled::Conv1dK7Dilation,
        };

        let preferred_o32 = Conv1dK7Route::TiledO32Preferred(Conv1dK7Dilation::One);
        let preferred_o64 = Conv1dK7Route::TiledO64Preferred(Conv1dK7Dilation::One);
        let measured_o16 = Conv1dK7Route::TiledO16(Conv1dK7Dilation::Three);
        assert_eq!(
            select_conv1d_k7_snake_epilogue_tile(preferred_o64, true, true, true),
            Some(Conv1dK7SnakeTile::Output32),
            "an unsupported standalone O64 route must retain the prior fused O32 selector"
        );
        assert_eq!(
            select_conv1d_k7_snake_epilogue_tile(preferred_o32, true, true, true),
            Some(Conv1dK7SnakeTile::Output32)
        );
        assert_eq!(
            select_conv1d_k7_snake_epilogue_tile(preferred_o32, true, true, false),
            Some(Conv1dK7SnakeTile::Output16)
        );
        assert_eq!(
            select_conv1d_k7_snake_epilogue_tile(measured_o16, true, true, true),
            Some(Conv1dK7SnakeTile::Output16),
            "the measured O16 exception must never be promoted to O32"
        );

        for (route, contract_compatible, o16_supported, o32_supported) in [
            (preferred_o32, false, true, true),
            (preferred_o32, true, false, false),
            (measured_o16, true, false, true),
            (Conv1dK7Route::BurnFallback, true, true, true),
        ] {
            assert_eq!(
                select_conv1d_k7_snake_epilogue_tile(
                    route,
                    contract_compatible,
                    o16_supported,
                    o32_supported,
                ),
                None,
                "incompatible act1 fusion must retain standalone Snake"
            );
        }
    }

    #[test]
    fn structurally_supported_nonofficial_shape_retains_portable_o16() {
        use crate::kernels::conv1d_k7_tiled::Conv1dK7Dilation;

        assert_eq!(
            decoder_k7_descriptor(32, 73, 9).route(),
            Conv1dK7Route::TiledO16(Conv1dK7Dilation::Nine)
        );
    }

    #[test]
    fn tiled_k7_route_falls_back_outside_exact_contract() {
        let supported = decoder_k7_descriptor(96, 96_000, 1);
        let unsupported = [
            Conv1dK7Descriptor {
                batch: 2,
                ..supported
            },
            Conv1dK7Descriptor {
                input_channels: 95,
                output_channels: 95,
                weight_input_channels: 95,
                bias_channels: Some(95),
                ..supported
            },
            Conv1dK7Descriptor {
                kernel_size: 5,
                ..supported
            },
            Conv1dK7Descriptor {
                dilation: 2,
                explicit_padding: Some((6, 6)),
                ..supported
            },
            Conv1dK7Descriptor {
                explicit_padding: Some((2, 3)),
                ..supported
            },
            Conv1dK7Descriptor {
                bias_channels: None,
                ..supported
            },
        ];

        assert!(
            unsupported
                .into_iter()
                .all(|descriptor| descriptor.route() == Conv1dK7Route::BurnFallback)
        );
    }

    #[test]
    fn snake1d_identity_at_zero() {
        let dev = Default::default();
        let alpha = Tensor::<3>::ones([1, 4, 1], &dev);
        let snake = Snake1d::new(alpha);

        // x + sin²(α·x)/(α+ε) at x=0 → 0 + 0/(1+ε) = 0
        let x = Tensor::<3>::zeros([1, 4, 8], &dev);
        let out = snake.forward(x);
        let data: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(data.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn snake1d_output_shape_preserved() {
        let dev = Default::default();
        let alpha = Tensor::<3>::ones([1, 8, 1], &dev).mul_scalar(2.0);
        let snake = Snake1d::new(alpha);

        let x = Tensor::<3>::ones([2, 8, 16], &dev);
        let out = snake.forward(x);
        assert_eq!(out.dims(), [2, 8, 16]);
    }

    #[test]
    fn snake1d_positive_for_positive_input() {
        let dev = Default::default();
        let alpha = Tensor::<3>::ones([1, 4, 1], &dev);
        let snake = Snake1d::new(alpha);

        // For positive x, snake output should be >= x (since sin²(αx)/(α+ε) >= 0)
        let x = Tensor::<3>::ones([1, 4, 8], &dev);
        let out = snake.forward(x.clone());
        let diff = out - x;
        let data: Vec<f32> = diff.into_data().to_vec().unwrap();
        assert!(
            data.iter().all(|v| *v >= -1e-6),
            "snake residual must be non-negative"
        );
    }

    #[test]
    #[ignore = "requires a WGPU adapter; run manually"]
    fn cubek_post_cast_epilogue_masks_partial_tiles_before_parameter_reads() {
        let device = f16_wgpu_test_device();

        // Odd channel counts exercise N tails. Length 65 forces a later output
        // partition, so the same check also covers a non-zero logical origin.
        for channels in [1, 15, 17, 95, 97] {
            for length in [1, 15, 17, 65] {
                let conv = make_conv1d(
                    8,
                    channels,
                    7,
                    1,
                    1,
                    Tensor::<3>::zeros([channels, 8, 7], &device),
                    Some(Tensor::<1>::zeros([channels], &device)),
                    &device,
                );
                let snake = Snake1d::new(Tensor::<3>::ones([1, channels, 1], &device));
                let input = Tensor::<3>::zeros([1, length, 8], &device);
                let output = implicit_gemm_nhwc_dilated_conv1d_then_snake_wgsl(
                    &conv,
                    &snake,
                    input,
                    None,
                    false,
                    K7MultiRowsSelection::Disabled,
                    false,
                    false,
                )
                .expect("partial-tile CubeK route must launch");
                assert_eq!(output.dims(), [1, length, channels]);
                let values = output
                    .cast(FloatDType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .expect("partial-tile output must read back");
                assert!(values.iter().all(|value| value.is_finite()));
            }
        }
    }

    #[cfg(feature = "profile")]
    #[test]
    #[ignore = "requires a WGPU adapter; run manually"]
    fn cubek_k7_weight_routes_are_bitwise_equivalent() {
        let device = f16_wgpu_test_device();
        let conv = make_conv1d(
            8,
            384,
            7,
            1,
            3,
            Tensor::<3>::ones([384, 8, 7], &device),
            Some(Tensor::<1>::zeros([384], &device)),
            &device,
        );
        let snake = Snake1d::new(Tensor::<3>::ones([1, 384, 1], &device));
        let input = Tensor::<3>::ones([1, 384, 8], &device);
        let prepared =
            prepare_k7_weight_for_implicit_gemm(&conv).expect("test k7 weight must prepare");
        let repack = implicit_gemm_nhwc_dilated_conv1d_then_snake_wgsl(
            &conv,
            &snake,
            input.clone(),
            None,
            false,
            K7MultiRowsSelection::Disabled,
            false,
            false,
        )
        .expect("request repack route must launch");
        let prepared = implicit_gemm_nhwc_dilated_conv1d_then_snake_wgsl(
            &conv,
            &snake,
            input.clone(),
            Some(&prepared),
            false,
            K7MultiRowsSelection::Disabled,
            false,
            false,
        )
        .expect("prepared route must launch");
        let geometry_selected = implicit_gemm_nhwc_dilated_conv1d_then_snake_wgsl(
            &conv,
            &snake,
            input.clone(),
            None,
            false,
            K7MultiRowsSelection::GeometrySelected,
            false,
            false,
        )
        .expect("geometry-selected route must launch");
        let direct = implicit_gemm_nhwc_dilated_conv1d_then_snake_wgsl(
            &conv,
            &snake,
            input,
            None,
            true,
            K7MultiRowsSelection::Disabled,
            false,
            false,
        )
        .expect("direct OIK route must launch");
        let read = |tensor: Tensor<3>| {
            tensor
                .cast(FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .expect("route output must read back")
        };
        let expected = read(repack);
        assert_eq!(read(prepared), expected);
        assert_eq!(read(geometry_selected), expected);
        assert_eq!(read(direct), expected);
    }

    #[cfg(feature = "profile")]
    #[test]
    #[ignore = "requires a WGPU adapter; run manually"]
    fn cubek_k7_halo_matches_repack_across_k_and_partial_m_tiles() {
        let device = f16_wgpu_test_device();
        let conv = make_conv1d(
            32,
            32,
            7,
            1,
            3,
            Tensor::<3>::ones([32, 32, 7], &device),
            Some(Tensor::<1>::zeros([32], &device)),
            &device,
        );
        let snake = Snake1d::new(Tensor::<3>::ones([1, 32, 1], &device));
        let input = Tensor::<3>::ones([1, 65, 32], &device);
        let expected = implicit_gemm_nhwc_dilated_conv1d_then_snake_wgsl(
            &conv,
            &snake,
            input.clone(),
            None,
            false,
            K7MultiRowsSelection::Disabled,
            false,
            false,
        )
        .expect("request repack route must launch");
        let halo = implicit_gemm_nhwc_dilated_conv1d_then_snake_wgsl(
            &conv,
            &snake,
            input,
            None,
            false,
            K7MultiRowsSelection::Disabled,
            true,
            false,
        )
        .expect("channel-major halo route must launch");
        let read = |tensor: Tensor<3>| {
            tensor
                .cast(FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .expect("route output must read back")
        };
        let expected = read(expected);
        let halo = read(halo);
        let max_abs = halo
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 1.0e-2, "halo max_abs={max_abs}");
    }

    #[test]
    fn conv_pad_symmetric() {
        // Standard conv: kernel=7, stride=1, dilation=1 → pad = (7-1)/2 = 3
        assert_eq!(conv_pad(7, 1, 1), 3);
        // Dilated: kernel=7, stride=1, dilation=3 → pad = (7-1)*3/2 = 9
        assert_eq!(conv_pad(7, 1, 3), 9);
        // Strided: kernel=4, stride=2, dilation=1 → pad = (4-2)/2 = 1
        assert_eq!(conv_pad(4, 2, 1), 1);
    }

    #[test]
    fn conv_transpose_pad_even_strides() {
        // Even stride (most common in DACVAE decoder):
        assert_eq!(conv_transpose_pad(12), (6, 0)); // (12+1)/2=6, 12%2=0
        assert_eq!(conv_transpose_pad(10), (5, 0));
        assert_eq!(conv_transpose_pad(8), (4, 0));
        assert_eq!(conv_transpose_pad(2), (1, 0));
    }

    #[test]
    fn conv_transpose_pad_odd_stride() {
        assert_eq!(conv_transpose_pad(3), (2, 1)); // (3+1)/2=2, 3%2=1
    }

    #[test]
    fn residual_unit_preserves_shape() {
        let dev = Default::default();
        let ch = 8;

        let alpha0 = Tensor::<3>::ones([1, ch, 1], &dev);
        let alpha1 = Tensor::<3>::ones([1, ch, 1], &dev);

        let act0 = Snake1d::new(alpha0);
        let act1 = Snake1d::new(alpha1);

        // Dilated conv: kernel=7, stride=1, dilation=1 → same-size output
        let conv_dil = make_conv1d(
            ch,
            ch,
            7,
            1,
            1,
            Tensor::<3>::zeros([ch, ch, 7], &dev),
            Some(Tensor::<1>::zeros([ch], &dev)),
            &dev,
        );

        let conv_1x1 = make_conv1d(
            ch,
            ch,
            1,
            1,
            1,
            Tensor::<3>::zeros([ch, ch, 1], &dev),
            Some(Tensor::<1>::zeros([ch], &dev)),
            &dev,
        );

        let unit = ResidualUnit {
            act0,
            conv_dil,
            act1,
            conv_1x1,
            packed_conv_1x1_weight: None,
            packed_conv_dil_weight_vectors: None,
            prepared_k7_weight: None,
            #[cfg(feature = "profile")]
            prepared_k7_selector: None,
        };

        let x = Tensor::<3>::ones([1, ch, 32], &dev);
        let out = unit.forward(x);
        assert_eq!(out.dims(), [1, ch, 32]);
    }

    #[test]
    fn pointwise_conv_matches_conv1d() {
        let device = Default::default();
        let conv = Conv1dConfig::new(5, 7, 1).with_bias(true).init(&device);
        let input = Tensor::<3>::random(
            [2, 5, 11],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let expected = conv.forward(input.clone());
        let actual = pointwise_conv1d(&conv, input.clone());
        let max_abs = (expected - actual)
            .abs()
            .max()
            .into_data()
            .to_vec::<f32>()
            .expect("pointwise comparison must decode as f32")[0];
        assert!(max_abs <= 1.0e-5, "pointwise max_abs={max_abs:.3e}");

        let packed = pack_pointwise_conv1d_weight(&conv);
        let packed_expected = conv.forward(input.clone());
        let raw_nlc = pointwise_conv1d_matmul_nlc_with_weight(&conv, Some(&packed), input.clone());
        assert_eq!(raw_nlc.dims(), [2, 11, 7]);
        let Some(bias) = &conv.bias else {
            panic!("test convolution was configured with a bias");
        };
        let raw_reconstructed = (raw_nlc + bias.val().reshape([1, 1, 7]))
            .swap_dims(1, 2)
            .reshape([2, 7, 11]);
        let packed_actual = pointwise_conv1d_with_weight(&conv, Some(&packed), input);
        let raw_reconstruction_data = (raw_reconstructed - packed_actual.clone())
            .abs()
            .max()
            .into_data()
            .to_vec::<f32>();
        let Ok(raw_reconstruction_data) = raw_reconstruction_data else {
            panic!("raw pointwise reconstruction must decode as f32");
        };
        let Some(&raw_reconstruction_max_abs) = raw_reconstruction_data.first() else {
            panic!("raw pointwise reconstruction metric must contain one value");
        };
        let packed_max_abs = (packed_expected - packed_actual.clone())
            .abs()
            .max()
            .into_data()
            .to_vec::<f32>()
            .expect("packed pointwise comparison must decode as f32")[0];
        assert_eq!(packed_actual.dims(), [2, 7, 11]);
        assert_eq!(
            raw_reconstruction_max_abs, 0.0,
            "raw NLC helper must preserve the existing ordered bias/layout path"
        );
        assert!(
            packed_max_abs <= 1.0e-5,
            "packed pointwise max_abs={packed_max_abs:.3e}"
        );
    }
}
