//! Shared layers for the DACVAE codec: Snake1d activation and Snake ResidualUnit.

use burn::tensor::Device;
use burn::{
    module::{Param, ParamId},
    nn::{PaddingConfig1d, conv::Conv1d},
    prelude::*,
};

#[cfg(feature = "profile")]
use std::time::{Duration, Instant};

use crate::nvtx_range;

// ─── Snake1d ─────────────────────────────────────────────────────────────────

/// `x + sin²(α·x) / (α + ε)` activation, element-wise.
///
/// Alpha has shape `[1, channels, 1]` and is stored as a non-trained
/// inference-time constant.
#[derive(Module, Debug)]
pub(crate) struct Snake1d {
    pub(crate) alpha: Param<Tensor<3>>,
}

impl Snake1d {
    pub(crate) fn new(alpha_tensor: Tensor<3>) -> Self {
        Self {
            alpha: Param::initialized(ParamId::new(), alpha_tensor),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let alpha = self.alpha.val();
        let ax = x.clone().mul(alpha.clone());
        let sin_sq = ax.sin().powi_scalar(2);
        let denom = alpha.add_scalar(1e-9_f32);
        x + sin_sq.div(denom)
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

/// The next residual unit can consume either ordinary NCL activation or the
/// exact compact residue layout required by its measured d3/d9 core.
#[derive(Debug)]
enum PreparedActivation {
    Ncl(Tensor<3>),
    ResiduePacked {
        tensor: Tensor<1>,
        dilation: crate::kernels::conv1d_k7_residue_d1_snake::ResidueDilation,
    },
}

impl ResidualUnit {
    /// Materialize the pointwise OIK parameter as physical KCO exactly once.
    ///
    /// A stale, non-contiguous, wrong-device, or non-f32 cache is replaced.
    /// If the source or resulting allocation misses any physical contract, the
    /// cache remains absent and every forward call retains the generic path.
    pub(crate) fn prepare_for_wgsl(&mut self) {
        if !self
            .packed_conv_1x1_weight
            .as_ref()
            .is_some_and(|packed| pointwise_wgpu_pack_is_compatible(&self.conv_1x1, packed))
        {
            self.packed_conv_1x1_weight = try_pack_pointwise_conv1d_weight_wgpu(&self.conv_1x1);
        }
        if !self
            .packed_conv_dil_weight_vectors
            .as_ref()
            .is_some_and(|packed| residue_weight_vector_pack_is_compatible(&self.conv_dil, packed))
        {
            self.packed_conv_dil_weight_vectors =
                try_pack_residue_conv1d_weight_vectors_wgpu(&self.conv_dil);
        }
    }

    pub(crate) fn forward_wgsl(&self, x: Tensor<3>) -> Tensor<3> {
        let residual = x.clone();
        let activated = nvtx_range!("codec_residual_snake_0", self.act0.forward_wgsl(x));
        self.forward_wgsl_from_parts(residual, activated)
    }

    /// Execute this unit and prepare the next unit's shortcut/Snake pair.
    pub(crate) fn forward_wgsl_prepare_next(
        &self,
        x: Tensor<3>,
        next: &ResidualUnit,
    ) -> PreparedResidualPair {
        let residual = x.clone();
        let activated = nvtx_range!("codec_residual_snake_0", self.act0.forward_wgsl(x));
        self.forward_wgsl_from_parts_prepare_next(residual, activated, next)
    }

    /// Consume a prepared shortcut/Snake pair without recomputing `act0`.
    pub(crate) fn forward_wgsl_from_prepared(&self, pair: PreparedResidualPair) -> Tensor<3> {
        let y = self.dilated_from_prepared(&pair.raw, pair.activated);
        pointwise_residual_wgsl_or_fallback(
            &self.conv_1x1,
            self.packed_conv_1x1_weight.as_ref(),
            y,
            pair.raw,
        )
    }

    /// Consume one prepared pair and produce the following unit's pair.
    pub(crate) fn forward_wgsl_from_prepared_prepare_next(
        &self,
        pair: PreparedResidualPair,
        next: &ResidualUnit,
    ) -> PreparedResidualPair {
        let y = self.dilated_from_prepared(&pair.raw, pair.activated);
        pointwise_residual_snake_pair_wgsl_or_fallback(
            &self.conv_1x1,
            self.packed_conv_1x1_weight.as_ref(),
            y,
            pair.raw,
            next,
        )
    }

    fn forward_wgsl_from_parts(&self, residual: Tensor<3>, activated: Tensor<3>) -> Tensor<3> {
        let y = dilated_conv1d_act1_wgsl_or_fallback(
            &self.conv_dil,
            &self.act1,
            self.packed_conv_dil_weight_vectors.as_ref(),
            activated,
        );
        pointwise_residual_wgsl_or_fallback(
            &self.conv_1x1,
            self.packed_conv_1x1_weight.as_ref(),
            y,
            residual,
        )
    }

    fn forward_wgsl_from_parts_prepare_next(
        &self,
        residual: Tensor<3>,
        activated: Tensor<3>,
        next: &ResidualUnit,
    ) -> PreparedResidualPair {
        let y = dilated_conv1d_act1_wgsl_or_fallback(
            &self.conv_dil,
            &self.act1,
            self.packed_conv_dil_weight_vectors.as_ref(),
            activated,
        );
        pointwise_residual_snake_pair_wgsl_or_fallback(
            &self.conv_1x1,
            self.packed_conv_1x1_weight.as_ref(),
            y,
            residual,
            next,
        )
    }

    fn dilated_from_prepared(
        &self,
        residual: &Tensor<3>,
        activated: PreparedActivation,
    ) -> Tensor<3> {
        match activated {
            PreparedActivation::Ncl(activated) => dilated_conv1d_act1_wgsl_or_fallback(
                &self.conv_dil,
                &self.act1,
                self.packed_conv_dil_weight_vectors.as_ref(),
                activated,
            ),
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
                    return Tensor::from_primitive::<crate::WgpuRaw>(output);
                }
                let activated = self.act0.forward_wgsl(residual.clone());
                dilated_conv1d_act1_wgsl_or_fallback(
                    &self.conv_dil,
                    &self.act1,
                    self.packed_conv_dil_weight_vectors.as_ref(),
                    activated,
                )
            }
        }
    }

    #[cfg(feature = "profile")]
    pub(crate) fn forward_wgsl_profiled_prepare_next<E, S>(
        &self,
        x: Tensor<3>,
        next: &ResidualUnit,
        labels: [&'static str; 3],
        synchronize: &mut S,
        timings: &mut Vec<(&'static str, Duration)>,
    ) -> Result<PreparedResidualPair, E>
    where
        S: FnMut(&'static str) -> Result<(), E>,
    {
        let residual = x.clone();
        let activated = profile_residual_stage(
            labels[0],
            || self.act0.forward_wgsl(x),
            synchronize,
            timings,
        )?;
        let y = profile_residual_stage(
            labels[1],
            || {
                dilated_conv1d_act1_wgsl_or_fallback(
                    &self.conv_dil,
                    &self.act1,
                    self.packed_conv_dil_weight_vectors.as_ref(),
                    activated,
                )
            },
            synchronize,
            timings,
        )?;
        profile_residual_stage(
            labels[2],
            || {
                pointwise_residual_snake_pair_wgsl_or_fallback(
                    &self.conv_1x1,
                    self.packed_conv_1x1_weight.as_ref(),
                    y,
                    residual,
                    next,
                )
            },
            synchronize,
            timings,
        )
    }

    #[cfg(feature = "profile")]
    pub(crate) fn forward_wgsl_profiled_from_prepared_prepare_next<E, S>(
        &self,
        pair: PreparedResidualPair,
        next: &ResidualUnit,
        labels: [&'static str; 2],
        synchronize: &mut S,
        timings: &mut Vec<(&'static str, Duration)>,
    ) -> Result<PreparedResidualPair, E>
    where
        S: FnMut(&'static str) -> Result<(), E>,
    {
        let PreparedResidualPair { raw, activated } = pair;
        let y = profile_residual_stage(
            labels[0],
            || self.dilated_from_prepared(&raw, activated),
            synchronize,
            timings,
        )?;
        profile_residual_stage(
            labels[1],
            || {
                pointwise_residual_snake_pair_wgsl_or_fallback(
                    &self.conv_1x1,
                    self.packed_conv_1x1_weight.as_ref(),
                    y,
                    raw,
                    next,
                )
            },
            synchronize,
            timings,
        )
    }

    #[cfg(feature = "profile")]
    pub(crate) fn forward_wgsl_profiled_from_prepared<E, S>(
        &self,
        pair: PreparedResidualPair,
        labels: [&'static str; 2],
        synchronize: &mut S,
        timings: &mut Vec<(&'static str, Duration)>,
    ) -> Result<Tensor<3>, E>
    where
        S: FnMut(&'static str) -> Result<(), E>,
    {
        let PreparedResidualPair { raw, activated } = pair;
        let y = profile_residual_stage(
            labels[0],
            || self.dilated_from_prepared(&raw, activated),
            synchronize,
            timings,
        )?;
        profile_residual_stage(
            labels[1],
            || {
                pointwise_residual_wgsl_or_fallback(
                    &self.conv_1x1,
                    self.packed_conv_1x1_weight.as_ref(),
                    y,
                    raw,
                )
            },
            synchronize,
            timings,
        )
    }
}

#[cfg(feature = "profile")]
fn profile_residual_stage<T, E, S>(
    label: &'static str,
    operation: impl FnOnce() -> T,
    synchronize: &mut S,
    timings: &mut Vec<(&'static str, Duration)>,
) -> Result<T, E>
where
    S: FnMut(&'static str) -> Result<(), E>,
{
    let started = Instant::now();
    let output = operation();
    synchronize(label)?;
    timings.push((label, started.elapsed()));
    Ok(output)
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
    input: Tensor<3>,
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
    pointwise_residual_finalizer_wgsl_or_fallback(conv, packed_weight, input, residual)
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
fn pointwise_residual_snake_pair_wgsl_or_fallback(
    conv: &Conv1d,
    packed_weight: Option<&Tensor<3>>,
    input: Tensor<3>,
    residual: Tensor<3>,
    next: &ResidualUnit,
) -> PreparedResidualPair {
    let next_act0 = &next.act0;
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
            let next_residue_dilation =
                next.packed_conv_dil_weight_vectors.as_ref().and_then(|_| {
                    Conv1dK7Descriptor::from_conv(
                        &next.conv_dil,
                        [1, descriptor.output_channels, descriptor.length],
                    )
                    .measured_residue_d1_dilation()
                });
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
    fn production_residue_d1_selector_keeps_two_dilations_per_admitted_shape() {
        use crate::kernels::conv1d_k7_residue_d1_snake::ResidueDilation;

        let cases = [
            (768, 600, 1, None),
            (768, 600, 3, None),
            (768, 600, 9, None),
            (384, 6_000, 1, None),
            (384, 6_000, 3, Some(ResidueDilation::Three)),
            (384, 6_000, 9, Some(ResidueDilation::Nine)),
            (192, 48_000, 1, None),
            (192, 48_000, 3, Some(ResidueDilation::Three)),
            (192, 48_000, 9, Some(ResidueDilation::Nine)),
            (96, 96_000, 1, None),
            (96, 96_000, 3, Some(ResidueDilation::Three)),
            (96, 96_000, 9, Some(ResidueDilation::Nine)),
        ];
        assert_eq!(cases.into_iter().filter(|case| case.3.is_some()).count(), 6);
        for (channels, length, dilation, expected) in cases {
            assert_eq!(
                decoder_k7_descriptor(channels, length, dilation).measured_residue_d1_dilation(),
                expected,
            );
        }
        for length in [12_480, 24_000, 96_000, 192_000] {
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

        let retained_t256 = decoder_k7_descriptor(192, 48_000, 1);
        let mut residue_preflight_called = false;
        assert_eq!(
            select_compatible_conv1d_k7_residue_d1_dilation(retained_t256, |_| {
                residue_preflight_called = true;
                true
            }),
            None,
        );
        assert!(!residue_preflight_called);
        assert_eq!(
            retained_t256.measured_t256_snake_vec4_store_tile(),
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
