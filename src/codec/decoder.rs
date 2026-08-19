//! DACVAE decoder: stem Conv → 4× DecoderBlock → WmHead (no-watermark path).

use burn::{
    module::{Param, ParamId},
    nn::{
        PaddingConfig1d,
        conv::{Conv1d, ConvTranspose1d},
    },
    prelude::*,
};

use super::layers::{PreparedResidualPair, ResidualUnit, Snake1d};
#[cfg(feature = "profile")]
use crate::nvtx_range;

#[cfg(feature = "profile")]
use super::{
    algorithm::{CodecK7Algorithm, CodecPointwiseAlgorithm},
    profiling::CodecStageProfiler,
};

#[cfg(feature = "profile")]
fn profile_wgsl_stage<T, O, P>(
    label: &'static str,
    operation: O,
    profiler: &mut P,
) -> Result<T, P::Error>
where
    T: Send + 'static,
    O: FnOnce() -> T + Send,
    P: CodecStageProfiler,
{
    profiler.profile(label, operation)
}

// ─── DecoderBlock ────────────────────────────────────────────────────────────

/// One upsampling block: Snake → ConvTranspose → 3× ResidualUnit.
///
/// Only the main signal path (blocks 0,1,4,5,8,9 of the original Python
/// `ModuleList`) is implemented. Watermark-only branches are omitted.
#[derive(Module, Debug)]
pub(crate) struct DecoderBlock {
    pub(crate) act: Snake1d,
    pub(crate) conv_t: ConvTranspose1d,
    /// Inference-only `[phase, Cout, Cin, 2]` cache for the first upsampler.
    #[module(skip)]
    pub(crate) packed_conv_t_weight: Option<Tensor<4>>,
    #[module(skip)]
    pub(crate) conv_t_residency: ConvTransposeResidency,
    pub(crate) res0: ResidualUnit,
    pub(crate) res1: ResidualUnit,
    pub(crate) res2: ResidualUnit,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum ConvTransposeResidency {
    #[default]
    PortableFallback,
    Fixed112,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConvTransposeRoute {
    Case0CachedCol2ImThenPolyphase(
        crate::kernels::conv_transpose1d_polyphase::ConvTranspose1dStride,
    ),
    Polyphase(crate::kernels::conv_transpose1d_polyphase::ConvTranspose1dStride),
    CachedCol2Im(crate::kernels::conv_transpose1d_cached_col2im::CachedCol2ImCase),
    BurnFallback,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ConvTransposeModuleDescriptor {
    channels: [usize; 2],
    weight: [usize; 3],
    bias_channels: Option<usize>,
    stride: usize,
    kernel_size: usize,
    dilation: usize,
    groups: usize,
    padding: usize,
    padding_out: usize,
}

impl ConvTransposeModuleDescriptor {
    fn from_conv(conv: &ConvTranspose1d) -> Self {
        Self {
            channels: conv.channels,
            weight: conv.weight.dims(),
            bias_channels: conv.bias.as_ref().map(|bias| bias.dims()[0]),
            stride: conv.stride,
            kernel_size: conv.kernel_size,
            dilation: conv.dilation,
            groups: conv.groups,
            padding: conv.padding,
            padding_out: conv.padding_out,
        }
    }

    fn polyphase_stride(
        self,
    ) -> Option<crate::kernels::conv_transpose1d_polyphase::ConvTranspose1dStride> {
        use crate::kernels::conv_transpose1d_polyphase::ConvTranspose1dStride;

        let supported = self.channels == [1536, 768]
            && self.weight == [1536, 768, 24]
            && self.bias_channels == Some(768)
            && self.stride == 12
            && self.kernel_size == 24
            && self.dilation == 1
            && self.groups == 1
            && self.padding == 6
            && self.padding_out == 0;
        supported.then_some(ConvTranspose1dStride::Twelve)
    }

    fn cached_col2im_case(
        self,
    ) -> Option<crate::kernels::conv_transpose1d_cached_col2im::CachedCol2ImCase> {
        use crate::kernels::conv_transpose1d_cached_col2im::CachedCol2ImCase;

        let case = match self.channels {
            [768, 384] => CachedCol2ImCase::Case1,
            [384, 192] => CachedCol2ImCase::Case2,
            [192, 96] => CachedCol2ImCase::Case3,
            _ => return None,
        };
        let supported = self.weight
            == [
                case.input_channels(),
                case.output_channels(),
                case.kernel_size(),
            ]
            && self.bias_channels == Some(case.output_channels())
            && self.stride == case.stride()
            && self.kernel_size == case.kernel_size()
            && self.dilation == 1
            && self.groups == 1
            && self.padding == case.padding()
            && self.padding_out == 0;
        supported.then_some(case)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ConvTransposeLaunchDescriptor {
    module: ConvTransposeModuleDescriptor,
    batch: usize,
    input_channels: usize,
    input_length: usize,
    packed_weight: Option<[usize; 4]>,
}

impl ConvTransposeLaunchDescriptor {
    fn route(self) -> ConvTransposeRoute {
        if let Some(stride) = self.module.polyphase_stride() {
            let supported = self.batch == 1
                && self.input_channels == 1536
                && self.input_length >= 25
                && self.packed_weight == Some([12, 768, 1536, 2]);
            return if supported && self.input_length == 50 {
                ConvTransposeRoute::Case0CachedCol2ImThenPolyphase(stride)
            } else if supported {
                ConvTransposeRoute::Polyphase(stride)
            } else {
                ConvTransposeRoute::BurnFallback
            };
        }
        let Some(case) = self.module.cached_col2im_case() else {
            return ConvTransposeRoute::BurnFallback;
        };
        let supported = self.batch == 1
            && self.input_channels == case.input_channels()
            && case.supports_input_length(self.input_length)
            && self.packed_weight.is_none();
        if supported {
            ConvTransposeRoute::CachedCol2Im(case)
        } else {
            ConvTransposeRoute::BurnFallback
        }
    }
}

impl DecoderBlock {
    pub(crate) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = self.act.forward(x);
        let x = self.conv_t.forward(x);
        let x = self.res0.forward(x);
        let x = self.res1.forward(x);
        self.res2.forward(x)
    }

    #[allow(dead_code)]
    pub(crate) fn prepare_for_inference(&mut self) {
        self.res0.prepare_for_inference();
        self.res1.prepare_for_inference();
        self.res2.prepare_for_inference();
    }
}

impl DecoderBlock {
    fn lock_fixed_112_polyphase_wgsl(&mut self) -> crate::error::Result<()> {
        use crate::error::IrodoriError;

        let source_weight = self.conv_t.weight.val();
        if self.conv_t.channels != [1536, 768]
            || self.conv_t.kernel_size != 24
            || source_weight.dims() != [1536, 768, 24]
        {
            return Err(IrodoriError::Config(
                "fixed-112 codec requires the released first upsampler".to_owned(),
            ));
        }
        let packed = self.packed_conv_t_weight.as_ref().ok_or_else(|| {
            IrodoriError::Config("fixed-112 codec requires a prepared polyphase cache".to_owned())
        })?;
        if packed.dims() != [12, 768, 1536, 2] || packed.device() != source_weight.device() {
            return Err(IrodoriError::Config(
                "fixed-112 codec polyphase cache contract mismatch".to_owned(),
            ));
        }
        let tombstone = Tensor::zeros([1, 1, 1], &source_weight.device());
        self.conv_t.weight = Param::initialized(ParamId::new(), tombstone.clone());
        self.conv_t_residency = ConvTransposeResidency::Fixed112;
        Ok(())
    }

    fn forward_fixed_112_wgsl(&self, x: Tensor<3>) -> crate::error::Result<Tensor<3>> {
        use crate::error::IrodoriError;
        use crate::kernels::conv_transpose1d_polyphase::ConvTranspose1dStride;

        if self.conv_t_residency != ConvTransposeResidency::Fixed112 || x.dims() != [1, 1536, 112] {
            return Err(IrodoriError::Config(
                "fixed-112 codec received an incompatible first-upsample input".to_owned(),
            ));
        }
        let x = self.act.forward_wgsl(x);
        let x = self
            .try_polyphase_conv_transpose_wgsl(x, ConvTranspose1dStride::Twelve)
            .ok_or_else(|| {
                IrodoriError::Config(
                    "fixed-112 codec polyphase execution contract failed".to_owned(),
                )
            })?;
        let pair = self.res0.forward_wgsl_prepare_next(x, &self.res1);
        let pair = self
            .res1
            .forward_wgsl_from_prepared_prepare_next(pair, &self.res2);
        Ok(self.res2.forward_wgsl_from_prepared(pair))
    }

    fn prepare_residuals_for_wgsl(&mut self) {
        self.res0.prepare_for_wgsl();
        self.res1.prepare_for_wgsl();
        self.res2.prepare_for_wgsl();
    }

    #[cfg(feature = "profile")]
    fn prepare_residuals_for_wgsl_with_algorithm(&mut self, algorithm: CodecK7Algorithm) {
        self.res0.prepare_for_wgsl_with_algorithm(algorithm);
        self.res1.prepare_for_wgsl_with_algorithm(algorithm);
        self.res2.prepare_for_wgsl_with_algorithm(algorithm);
    }

    #[cfg(feature = "profile")]
    fn residuals(&self) -> [&ResidualUnit; 3] {
        [&self.res0, &self.res1, &self.res2]
    }

    fn prepare_conv_transpose_for_wgsl(&mut self) {
        if self.packed_conv_t_weight.is_some() {
            return;
        }
        let Some(stride) =
            ConvTransposeModuleDescriptor::from_conv(&self.conv_t).polyphase_stride()
        else {
            return;
        };

        let packed = crate::kernels::conv_transpose1d_polyphase::pack_conv_transpose1d_weight_wgsl(
            self.conv_t
                .weight
                .val()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            stride,
        );
        self.packed_conv_t_weight = Some(Tensor::from_primitive::<crate::WgpuRaw>(
            packed.into_tensor(),
        ));
    }

    #[cfg(feature = "profile")]
    fn forward_wgsl(&self, x: Tensor<3>) -> Tensor<3> {
        let x = nvtx_range!("codec_upsample_snake", self.act.forward_wgsl(x));
        let x = nvtx_range!(
            "codec_conv_transpose",
            self.conv_transpose_wgsl_or_fallback(x)
        );
        let pair = nvtx_range!(
            "codec_residual_unit_0",
            self.res0.forward_wgsl_prepare_next(x, &self.res1)
        );
        let pair = nvtx_range!(
            "codec_residual_unit_1",
            self.res1
                .forward_wgsl_from_prepared_prepare_next(pair, &self.res2)
        );
        nvtx_range!(
            "codec_residual_unit_2",
            self.res2.forward_wgsl_from_prepared(pair)
        )
    }

    /// Run one block from an already prepared upsampler activation and prepare
    /// the following block's activation in this block's final pointwise
    /// residual dispatch.
    fn forward_wgsl_from_activated_prepare_next_block(
        &self,
        activated: Tensor<3>,
        next_block_act: &Snake1d,
        conv_transpose_fusion: super::algorithm::CodecConvTransposeSnakeFusion,
    ) -> Tensor<3> {
        let pair = self.prepare_res0_after_conv_transpose(activated, conv_transpose_fusion);
        let pair = self
            .res0
            .forward_wgsl_from_prepared_prepare_next(pair, &self.res1);
        let pair = self
            .res1
            .forward_wgsl_from_prepared_prepare_next(pair, &self.res2);
        self.res2
            .forward_wgsl_from_prepared_prepare_block(pair, next_block_act)
    }

    /// Run one block from an already prepared upsampler activation.
    fn forward_wgsl_from_activated(
        &self,
        activated: Tensor<3>,
        conv_transpose_fusion: super::algorithm::CodecConvTransposeSnakeFusion,
    ) -> Tensor<3> {
        let pair = self.prepare_res0_after_conv_transpose(activated, conv_transpose_fusion);
        let pair = self
            .res0
            .forward_wgsl_from_prepared_prepare_next(pair, &self.res1);
        let pair = self
            .res1
            .forward_wgsl_from_prepared_prepare_next(pair, &self.res2);
        self.res2.forward_wgsl_from_prepared(pair)
    }

    fn forward_wgsl_prepare_next_block(
        &self,
        x: Tensor<3>,
        next_block_act: &Snake1d,
        conv_transpose_fusion: super::algorithm::CodecConvTransposeSnakeFusion,
    ) -> Tensor<3> {
        let activated = self.act.forward_wgsl(x);
        self.forward_wgsl_from_activated_prepare_next_block(
            activated,
            next_block_act,
            conv_transpose_fusion,
        )
    }

    fn prepare_res0_after_conv_transpose(
        &self,
        input: Tensor<3>,
        fusion: super::algorithm::CodecConvTransposeSnakeFusion,
    ) -> PreparedResidualPair {
        #[cfg(feature = "profile")]
        let [batch, input_channels, input_length] = input.dims();
        #[cfg(feature = "profile")]
        let descriptor = ConvTransposeLaunchDescriptor {
            module: ConvTransposeModuleDescriptor::from_conv(&self.conv_t),
            batch,
            input_channels,
            input_length,
            packed_weight: self
                .packed_conv_t_weight
                .as_ref()
                .map(|weight| weight.dims()),
        };
        #[cfg(feature = "profile")]
        {
            if let ConvTransposeRoute::CachedCol2Im(case) = descriptor.route()
                && fusion.fuses_cached_col2im(case)
                && let Some(pair) = self.try_cached_col2im_conv_transpose_snake_pair_wgsl(
                    input.clone(),
                    case,
                    &self.res0.act0,
                )
            {
                return pair;
            }
        }
        #[cfg(not(feature = "profile"))]
        let _ = fusion;
        let raw = self.conv_transpose_wgsl_or_fallback(input);
        self.res0.prepare_input_wgsl(raw)
    }

    #[cfg(feature = "profile")]
    fn forward_wgsl_profiled_residual_parts<P>(
        &self,
        x: Tensor<3>,
        labels: [&'static str; 9],
        cached_conv_labels: [&'static str; 2],
        k7_algorithm: CodecK7Algorithm,
        pointwise_algorithm: CodecPointwiseAlgorithm,
        profiler: &mut P,
    ) -> Result<Tensor<3>, P::Error>
    where
        P: CodecStageProfiler,
    {
        let x = profile_wgsl_stage(labels[0], || self.act.forward_wgsl(x), profiler)?;
        let x = self.conv_transpose_wgsl_or_fallback_profiled(
            x,
            labels[1],
            cached_conv_labels,
            profiler,
        )?;
        let pair = self.res0.forward_wgsl_profiled_prepare_next(
            x,
            &self.res1,
            [labels[2], labels[3], labels[4]],
            k7_algorithm,
            pointwise_algorithm,
            profiler,
        )?;
        let pair = self.res1.forward_wgsl_profiled_from_prepared_prepare_next(
            pair,
            &self.res2,
            [labels[5], labels[6]],
            k7_algorithm,
            pointwise_algorithm,
            profiler,
        )?;
        self.res2.forward_wgsl_profiled_from_prepared(
            pair,
            [labels[7], labels[8]],
            k7_algorithm,
            pointwise_algorithm,
            profiler,
        )
    }

    fn conv_transpose_wgsl_or_fallback(&self, input: Tensor<3>) -> Tensor<3> {
        let [batch, input_channels, input_length] = input.dims();
        let descriptor = ConvTransposeLaunchDescriptor {
            module: ConvTransposeModuleDescriptor::from_conv(&self.conv_t),
            batch,
            input_channels,
            input_length,
            packed_weight: self
                .packed_conv_t_weight
                .as_ref()
                .map(|weight| weight.dims()),
        };
        let candidate = match descriptor.route() {
            ConvTransposeRoute::Case0CachedCol2ImThenPolyphase(stride) => self
                .try_case0_cached_col2im_conv_transpose_wgsl(input.clone())
                .or_else(|| self.try_polyphase_conv_transpose_wgsl(input.clone(), stride)),
            ConvTransposeRoute::Polyphase(stride) => {
                self.try_polyphase_conv_transpose_wgsl(input.clone(), stride)
            }
            ConvTransposeRoute::CachedCol2Im(case) => {
                self.try_cached_col2im_conv_transpose_wgsl(input.clone(), case)
            }
            ConvTransposeRoute::BurnFallback => None,
        };
        candidate.unwrap_or_else(|| self.conv_t.forward(input))
    }

    #[cfg(feature = "profile")]
    fn conv_transpose_wgsl_or_fallback_profiled<P>(
        &self,
        input: Tensor<3>,
        fallback_label: &'static str,
        cached_labels: [&'static str; 2],
        profiler: &mut P,
    ) -> Result<Tensor<3>, P::Error>
    where
        P: CodecStageProfiler,
    {
        let [batch, input_channels, input_length] = input.dims();
        let descriptor = ConvTransposeLaunchDescriptor {
            module: ConvTransposeModuleDescriptor::from_conv(&self.conv_t),
            batch,
            input_channels,
            input_length,
            packed_weight: self
                .packed_conv_t_weight
                .as_ref()
                .map(|weight| weight.dims()),
        };
        if let ConvTransposeRoute::CachedCol2Im(case) = descriptor.route()
            && let Some(output) = self.try_cached_col2im_conv_transpose_wgsl_profiled(
                input.clone(),
                case,
                cached_labels,
                profiler,
            )?
        {
            return Ok(output);
        }
        profile_wgsl_stage(
            fallback_label,
            || self.conv_transpose_wgsl_or_fallback(input),
            profiler,
        )
    }

    fn try_case0_cached_col2im_conv_transpose_wgsl(&self, input: Tensor<3>) -> Option<Tensor<3>> {
        let bias = self.conv_t.bias.as_ref()?;
        let output = crate::kernels::conv_transpose1d_cached_col2im_case0::conv_transpose1d_case0_cached_col2im_wgsl(
            input.try_into_primitive::<crate::WgpuRaw>().expect("tensor must use WGPU raw backend"),
            self.conv_t.weight.val().try_into_primitive::<crate::WgpuRaw>().expect("tensor must use WGPU raw backend"),
            bias.val().try_into_primitive::<crate::WgpuRaw>().expect("tensor must use WGPU raw backend"),
        )
        .ok()?;
        Some(Tensor::from_primitive::<crate::WgpuRaw>(output))
    }

    fn try_polyphase_conv_transpose_wgsl(
        &self,
        input: Tensor<3>,
        stride: crate::kernels::conv_transpose1d_polyphase::ConvTranspose1dStride,
    ) -> Option<Tensor<3>> {
        let Some(packed_weight) = &self.packed_conv_t_weight else {
            return None;
        };
        let Some(bias) = &self.conv_t.bias else {
            return None;
        };

        let packed_weight = packed_weight
            .clone()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        let bias = bias
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        if !packed_weight.is_contiguous() || !bias.is_contiguous() {
            return None;
        }
        let packed_weight =
            crate::kernels::conv_transpose1d_polyphase::PackedConvTranspose1dWeight::from_tensor(
                packed_weight,
                stride,
            );
        let output = crate::kernels::conv_transpose1d_polyphase::conv_transpose1d_polyphase_wgsl(
            input
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            &packed_weight,
            bias,
        );
        Some(Tensor::from_primitive::<crate::WgpuRaw>(output))
    }

    fn try_cached_col2im_conv_transpose_wgsl(
        &self,
        input: Tensor<3>,
        case: crate::kernels::conv_transpose1d_cached_col2im::CachedCol2ImCase,
    ) -> Option<Tensor<3>> {
        let bias = self.conv_t.bias.as_ref()?;
        let output =
            crate::kernels::conv_transpose1d_cached_col2im::conv_transpose1d_cached_col2im_wgsl(
                input
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                self.conv_t
                    .weight
                    .val()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                bias.val()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                case,
            )
            .ok()?;
        Some(Tensor::from_primitive::<crate::WgpuRaw>(output))
    }

    #[cfg(feature = "profile")]
    fn try_cached_col2im_conv_transpose_snake_pair_wgsl(
        &self,
        input: Tensor<3>,
        case: crate::kernels::conv_transpose1d_cached_col2im::CachedCol2ImCase,
        snake: &Snake1d,
    ) -> Option<PreparedResidualPair> {
        let bias = self.conv_t.bias.as_ref()?;
        let pair = crate::kernels::conv_transpose1d_cached_col2im::conv_transpose1d_cached_col2im_snake_pair_wgsl(
            input.try_into_primitive::<crate::WgpuRaw>().ok()?,
            self.conv_t.weight.val().try_into_primitive::<crate::WgpuRaw>().ok()?,
            bias.val().try_into_primitive::<crate::WgpuRaw>().ok()?,
            snake.alpha.val().try_into_primitive::<crate::WgpuRaw>().ok()?,
            case,
        )
        .ok()?;
        PreparedResidualPair::from_ncl_nhwc(
            Tensor::from_primitive::<crate::WgpuRaw>(pair.raw_ncl),
            Tensor::from_primitive::<crate::WgpuRaw>(pair.activated_nhwc),
        )
    }

    #[cfg(feature = "profile")]
    fn try_cached_col2im_conv_transpose_wgsl_profiled<P>(
        &self,
        input: Tensor<3>,
        case: crate::kernels::conv_transpose1d_cached_col2im::CachedCol2ImCase,
        labels: [&'static str; 2],
        profiler: &mut P,
    ) -> Result<Option<Tensor<3>>, P::Error>
    where
        P: CodecStageProfiler,
    {
        let Some(bias) = self.conv_t.bias.as_ref() else {
            return Ok(None);
        };
        let bias = bias
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        let input = input
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        let weight = self
            .conv_t
            .weight
            .val()
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend");
        let bias_for_gemm = bias.clone();
        let columns = profiler.profile(labels[0], move || {
            crate::kernels::conv_transpose1d_cached_col2im::matmul_cached_col2im_columns_wgsl(
                input,
                weight,
                &bias_for_gemm,
                case,
            )
        })?;
        let Ok(columns) = columns else {
            return Ok(None);
        };

        let output = profiler.profile(labels[1], move || {
            crate::kernels::conv_transpose1d_cached_col2im::finalize_cached_col2im_wgsl(
                columns, bias, case,
            )
        })?;
        let Ok(output) = output else {
            return Ok(None);
        };
        Ok(Some(Tensor::from_primitive::<crate::WgpuRaw>(output)))
    }
}

// ─── WmHead ──────────────────────────────────────────────────────────────────

/// Watermark-less output head: `Snake → Conv(96→1) → Tanh`.
///
/// This is the `forward_no_conv` path of `WatermarkEncoderBlock.pre`:
/// the final `NormConv1d(1→32)` is replaced by identity, so we only
/// apply `[Snake, Conv(96→1), Tanh]`.
#[derive(Module, Debug)]
pub(crate) struct WmHead {
    pub(crate) act: Snake1d,
    pub(crate) conv: Conv1d,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WmHeadSnakeNlcRoute {
    Fused,
    ExistingFallback,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WmHeadSnakeNlcDescriptor {
    input: [usize; 3],
    alpha: [usize; 3],
    weight: [usize; 3],
    bias_channels: Option<usize>,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    explicit_padding: Option<(usize, usize)>,
}

impl WmHeadSnakeNlcDescriptor {
    fn from_head(head: &WmHead, input: [usize; 3]) -> Self {
        let explicit_padding = match &head.conv.padding {
            PaddingConfig1d::Explicit(left, right) => Some((*left, *right)),
            PaddingConfig1d::Same | PaddingConfig1d::Valid => None,
        };
        Self {
            input,
            alpha: head.act.alpha.dims(),
            weight: head.conv.weight.dims(),
            bias_channels: head.conv.bias.as_ref().map(|bias| bias.dims()[0]),
            kernel_size: head.conv.kernel_size,
            stride: head.conv.stride,
            dilation: head.conv.dilation,
            groups: head.conv.groups,
            explicit_padding,
        }
    }

    fn route(self) -> WmHeadSnakeNlcRoute {
        let exact = self.input[0] == 1
            && self.input[1] == 96
            && self.input[2] > 0
            && self.input[2].is_multiple_of(240)
            && self.alpha == [1, 96, 1]
            && self.weight == [1, 96, 7]
            && self.bias_channels == Some(1)
            && self.kernel_size == 7
            && self.stride == 1
            && self.dilation == 1
            && self.groups == 1
            && self.explicit_padding == Some((3, 3));
        if exact {
            WmHeadSnakeNlcRoute::Fused
        } else {
            WmHeadSnakeNlcRoute::ExistingFallback
        }
    }
}

impl WmHead {
    pub(crate) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = self.act.forward(x);
        let x = self.conv.forward(x);
        x.tanh()
    }
}

impl WmHead {
    fn forward_wgsl(&self, x: Tensor<3>) -> Tensor<3> {
        let descriptor = WmHeadSnakeNlcDescriptor::from_head(self, x.dims());
        if descriptor.route() == WmHeadSnakeNlcRoute::Fused
            && let Some(output) = self.try_fused_final_wgsl(x.clone())
        {
            return output;
        }
        let fused = (descriptor.route() == WmHeadSnakeNlcRoute::Fused)
            .then(|| self.try_snake_nlc_wgsl(x.clone()))
            .flatten();
        let x = fused.unwrap_or_else(|| self.act.forward_wgsl(x));
        let x = self.conv.forward(x);
        x.tanh()
    }

    fn try_fused_final_wgsl(&self, input: Tensor<3>) -> Option<Tensor<3>> {
        let bias = self.conv.bias.as_ref()?;
        let output =
            crate::kernels::wm_head_fused_final_t240_c16::try_wm_head_fused_final_t240_c16_wgsl(
                input
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                self.act
                    .alpha
                    .val()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                self.conv
                    .weight
                    .val()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                bias.val()
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
            )?;
        Some(Tensor::from_primitive::<crate::WgpuRaw>(output))
    }

    fn try_snake_nlc_wgsl(&self, input: Tensor<3>) -> Option<Tensor<3>> {
        let output_nlc = crate::kernels::wm_head_snake_nlc::wm_head_snake_ncl_to_nlc_wgsl(
            input
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            self.act
                .alpha
                .val()
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
        )
        .ok()?;
        let output_nlc = Tensor::from_primitive::<crate::WgpuRaw>(output_nlc);
        Some(output_nlc.swap_dims(1, 2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::nn::conv::Conv1dConfig;

    fn module_descriptor(
        input_channels: usize,
        output_channels: usize,
        stride: usize,
    ) -> ConvTransposeModuleDescriptor {
        ConvTransposeModuleDescriptor {
            channels: [input_channels, output_channels],
            weight: [input_channels, output_channels, 2 * stride],
            bias_channels: Some(output_channels),
            stride,
            kernel_size: 2 * stride,
            dilation: 1,
            groups: 1,
            padding: stride / 2,
            padding_out: 0,
        }
    }

    fn launch_descriptor(
        module: ConvTransposeModuleDescriptor,
        input_length: usize,
        packed_weight: Option<[usize; 4]>,
    ) -> ConvTransposeLaunchDescriptor {
        ConvTransposeLaunchDescriptor {
            batch: 1,
            input_channels: module.channels[0],
            input_length,
            module,
            packed_weight,
        }
    }

    fn tiny_wm_head() -> WmHead {
        let dev = Default::default();
        let channels = 8;
        WmHead {
            act: Snake1d::new(Tensor::<3>::ones([1, channels, 1], &dev)),
            conv: Conv1dConfig::new(channels, 1, 1).init(&dev),
        }
    }

    fn released_wm_head_descriptor() -> WmHeadSnakeNlcDescriptor {
        WmHeadSnakeNlcDescriptor {
            input: [1, 96, 96_000],
            alpha: [1, 96, 1],
            weight: [1, 96, 7],
            bias_channels: Some(1),
            kernel_size: 7,
            stride: 1,
            dilation: 1,
            groups: 1,
            explicit_padding: Some((3, 3)),
        }
    }

    fn released_wm_head() -> WmHead {
        let dev = Default::default();
        WmHead {
            act: Snake1d::new(Tensor::<3>::ones([1, 96, 1], &dev)),
            conv: Conv1dConfig::new(96, 1, 7)
                .with_stride(1)
                .with_dilation(1)
                .with_padding(PaddingConfig1d::Explicit(3, 3))
                .with_bias(true)
                .init(&dev),
        }
    }

    #[test]
    fn released_wm_head_selects_fused_snake_nlc_route() {
        let extracted = WmHeadSnakeNlcDescriptor::from_head(&released_wm_head(), [1, 96, 96_000]);
        assert_eq!(extracted, released_wm_head_descriptor());
        assert_eq!(extracted.route(), WmHeadSnakeNlcRoute::Fused);
    }

    #[test]
    fn hop_aligned_audio_lengths_select_the_same_fused_route() {
        let head = released_wm_head();
        for time in [24_000, 48_000, 96_000, 192_000, 384_000] {
            let descriptor = WmHeadSnakeNlcDescriptor::from_head(&head, [1, 96, time]);
            assert_eq!(descriptor.route(), WmHeadSnakeNlcRoute::Fused);
        }
    }

    #[test]
    fn wm_head_snake_nlc_route_requires_every_measured_property() {
        let supported = released_wm_head_descriptor();
        let unsupported = [
            WmHeadSnakeNlcDescriptor {
                input: [2, 96, 96_000],
                ..supported
            },
            WmHeadSnakeNlcDescriptor {
                input: [1, 95, 96_000],
                ..supported
            },
            WmHeadSnakeNlcDescriptor {
                input: [1, 96, 95_999],
                ..supported
            },
            WmHeadSnakeNlcDescriptor {
                alpha: [2, 96, 1],
                ..supported
            },
            WmHeadSnakeNlcDescriptor {
                alpha: [1, 95, 1],
                ..supported
            },
            WmHeadSnakeNlcDescriptor {
                alpha: [1, 96, 2],
                ..supported
            },
            WmHeadSnakeNlcDescriptor {
                weight: [2, 96, 7],
                ..supported
            },
            WmHeadSnakeNlcDescriptor {
                weight: [1, 95, 7],
                ..supported
            },
            WmHeadSnakeNlcDescriptor {
                weight: [1, 96, 5],
                ..supported
            },
            WmHeadSnakeNlcDescriptor {
                bias_channels: None,
                ..supported
            },
            WmHeadSnakeNlcDescriptor {
                bias_channels: Some(2),
                ..supported
            },
            WmHeadSnakeNlcDescriptor {
                kernel_size: 5,
                ..supported
            },
            WmHeadSnakeNlcDescriptor {
                stride: 2,
                ..supported
            },
            WmHeadSnakeNlcDescriptor {
                dilation: 2,
                ..supported
            },
            WmHeadSnakeNlcDescriptor {
                groups: 2,
                ..supported
            },
            WmHeadSnakeNlcDescriptor {
                explicit_padding: Some((2, 3)),
                ..supported
            },
            WmHeadSnakeNlcDescriptor {
                explicit_padding: None,
                ..supported
            },
        ];
        assert!(
            unsupported
                .into_iter()
                .all(|descriptor| descriptor.route() == WmHeadSnakeNlcRoute::ExistingFallback)
        );
    }

    #[test]
    fn released_upsamplers_select_measured_routes() {
        use crate::kernels::conv_transpose1d_cached_col2im::CachedCol2ImCase;
        use crate::kernels::conv_transpose1d_polyphase::ConvTranspose1dStride;

        let cases = [
            (
                1536,
                768,
                50,
                12,
                Some([12, 768, 1536, 2]),
                ConvTransposeRoute::Case0CachedCol2ImThenPolyphase(ConvTranspose1dStride::Twelve),
            ),
            (
                768,
                384,
                600,
                10,
                None,
                ConvTransposeRoute::CachedCol2Im(CachedCol2ImCase::Case1),
            ),
            (
                384,
                192,
                6_000,
                8,
                None,
                ConvTransposeRoute::CachedCol2Im(CachedCol2ImCase::Case2),
            ),
            (
                192,
                96,
                48_000,
                2,
                None,
                ConvTransposeRoute::CachedCol2Im(CachedCol2ImCase::Case3),
            ),
        ];
        for (input_channels, output_channels, input_length, stride, packed, expected) in cases {
            let module = module_descriptor(input_channels, output_channels, stride);
            assert_eq!(
                launch_descriptor(module, input_length, packed).route(),
                expected
            );
        }
    }

    #[test]
    fn first_upsampler_uses_case0_at_reference_length_and_polyphase_elsewhere() {
        let module = module_descriptor(1536, 768, 12);
        let supported = launch_descriptor(module, 50, Some([12, 768, 1536, 2]));
        assert_eq!(
            launch_descriptor(module, 51, Some([12, 768, 1536, 2])).route(),
            ConvTransposeRoute::Polyphase(
                crate::kernels::conv_transpose1d_polyphase::ConvTranspose1dStride::Twelve
            ),
        );
        assert_eq!(
            launch_descriptor(module, 13, Some([12, 768, 1536, 2])).route(),
            ConvTransposeRoute::BurnFallback,
        );
        let unsupported = [
            ConvTransposeLaunchDescriptor {
                batch: 2,
                ..supported
            },
            ConvTransposeLaunchDescriptor {
                input_channels: 1520,
                ..supported
            },
            ConvTransposeLaunchDescriptor {
                input_length: 0,
                ..supported
            },
            ConvTransposeLaunchDescriptor {
                packed_weight: None,
                ..supported
            },
            ConvTransposeLaunchDescriptor {
                packed_weight: Some([12, 768, 1536, 1]),
                ..supported
            },
        ];
        assert_eq!(
            supported.route(),
            ConvTransposeRoute::Case0CachedCol2ImThenPolyphase(
                crate::kernels::conv_transpose1d_polyphase::ConvTranspose1dStride::Twelve
            )
        );
        assert!(
            unsupported
                .into_iter()
                .all(|descriptor| descriptor.route() == ConvTransposeRoute::BurnFallback)
        );
    }

    #[test]
    fn first_upsampler_requires_exact_module_metadata() {
        let supported = module_descriptor(1536, 768, 12);
        let unsupported = [
            ConvTransposeModuleDescriptor {
                channels: [1536, 767],
                ..supported
            },
            ConvTransposeModuleDescriptor {
                weight: [1536, 768, 23],
                ..supported
            },
            ConvTransposeModuleDescriptor {
                bias_channels: None,
                ..supported
            },
            ConvTransposeModuleDescriptor {
                stride: 11,
                ..supported
            },
            ConvTransposeModuleDescriptor {
                dilation: 2,
                ..supported
            },
            ConvTransposeModuleDescriptor {
                groups: 2,
                ..supported
            },
            ConvTransposeModuleDescriptor {
                padding: 5,
                ..supported
            },
            ConvTransposeModuleDescriptor {
                padding_out: 1,
                ..supported
            },
        ];
        assert!(supported.polyphase_stride().is_some());
        assert!(
            unsupported
                .into_iter()
                .all(|descriptor| descriptor.polyphase_stride().is_none())
        );
    }

    #[test]
    fn cached_col2im_requires_exact_launch_without_a_persistent_cache() {
        use crate::kernels::conv_transpose1d_cached_col2im::CachedCol2ImCase;

        for case in [
            CachedCol2ImCase::Case1,
            CachedCol2ImCase::Case2,
            CachedCol2ImCase::Case3,
        ] {
            let module =
                module_descriptor(case.input_channels(), case.output_channels(), case.stride());
            let input_length = match case {
                CachedCol2ImCase::Case1 => 600,
                CachedCol2ImCase::Case2 => 6_000,
                CachedCol2ImCase::Case3 => 48_000,
            };
            let supported = launch_descriptor(module, input_length, None);
            let unsupported = [
                ConvTransposeLaunchDescriptor {
                    batch: 2,
                    ..supported
                },
                ConvTransposeLaunchDescriptor {
                    input_channels: case.input_channels() - 1,
                    ..supported
                },
                ConvTransposeLaunchDescriptor {
                    input_length: input_length + 1,
                    ..supported
                },
                ConvTransposeLaunchDescriptor {
                    packed_weight: Some([
                        case.stride(),
                        case.output_channels(),
                        case.input_channels(),
                        2,
                    ]),
                    ..supported
                },
            ];
            assert_eq!(supported.route(), ConvTransposeRoute::CachedCol2Im(case));
            assert!(
                unsupported
                    .into_iter()
                    .all(|descriptor| descriptor.route() == ConvTransposeRoute::BurnFallback)
            );
        }
    }

    #[test]
    fn variable_length_decoder_upsamplers_keep_wgsl_routes() {
        use crate::kernels::conv_transpose1d_cached_col2im::CachedCol2ImCase;
        use crate::kernels::conv_transpose1d_polyphase::ConvTranspose1dStride;

        for latent_steps in [25, 50, 100, 200] {
            let first = launch_descriptor(
                module_descriptor(1536, 768, 12),
                latent_steps,
                Some([12, 768, 1536, 2]),
            );
            let expected_first = if latent_steps == 50 {
                ConvTransposeRoute::Case0CachedCol2ImThenPolyphase(ConvTranspose1dStride::Twelve)
            } else {
                ConvTransposeRoute::Polyphase(ConvTranspose1dStride::Twelve)
            };
            assert_eq!(first.route(), expected_first);
            for (input_channels, output_channels, input_length, stride, case) in [
                (768, 384, latent_steps * 12, 10, CachedCol2ImCase::Case1),
                (384, 192, latent_steps * 120, 8, CachedCol2ImCase::Case2),
                (192, 96, latent_steps * 960, 2, CachedCol2ImCase::Case3),
            ] {
                let descriptor = launch_descriptor(
                    module_descriptor(input_channels, output_channels, stride),
                    input_length,
                    None,
                );
                assert_eq!(descriptor.route(), ConvTransposeRoute::CachedCol2Im(case));
            }
        }

        for (input_channels, output_channels, input_length, stride) in [
            (768, 384, 13 * 12, 10),
            (384, 192, 13 * 120, 8),
            (192, 96, 13 * 960, 2),
        ] {
            assert_eq!(
                launch_descriptor(
                    module_descriptor(input_channels, output_channels, stride),
                    input_length,
                    None,
                )
                .route(),
                ConvTransposeRoute::BurnFallback,
            );
        }
    }

    #[test]
    fn cached_col2im_requires_exact_module_metadata() {
        use crate::kernels::conv_transpose1d_cached_col2im::CachedCol2ImCase;

        for case in [
            CachedCol2ImCase::Case1,
            CachedCol2ImCase::Case2,
            CachedCol2ImCase::Case3,
        ] {
            let supported =
                module_descriptor(case.input_channels(), case.output_channels(), case.stride());
            let unsupported = [
                ConvTransposeModuleDescriptor {
                    channels: [case.input_channels(), case.output_channels() - 1],
                    ..supported
                },
                ConvTransposeModuleDescriptor {
                    weight: [
                        case.input_channels(),
                        case.output_channels(),
                        case.kernel_size() + 1,
                    ],
                    ..supported
                },
                ConvTransposeModuleDescriptor {
                    bias_channels: None,
                    ..supported
                },
                ConvTransposeModuleDescriptor {
                    stride: case.stride() + 1,
                    ..supported
                },
                ConvTransposeModuleDescriptor {
                    kernel_size: case.kernel_size() + 1,
                    ..supported
                },
                ConvTransposeModuleDescriptor {
                    dilation: 2,
                    ..supported
                },
                ConvTransposeModuleDescriptor {
                    groups: 2,
                    ..supported
                },
                ConvTransposeModuleDescriptor {
                    padding: case.padding() + 1,
                    ..supported
                },
                ConvTransposeModuleDescriptor {
                    padding_out: 1,
                    ..supported
                },
            ];
            assert_eq!(supported.cached_col2im_case(), Some(case));
            assert!(
                unsupported
                    .into_iter()
                    .all(|descriptor| descriptor.cached_col2im_case().is_none())
            );
        }
    }

    fn released_stem_descriptor() -> DecoderStemDescriptor {
        DecoderStemDescriptor {
            input: [1, 1_024, 50],
            weight: [1_536, 1_024, 7],
            bias_channels: Some(1_536),
            kernel_size: 7,
            stride: 1,
            dilation: 1,
            groups: 1,
            explicit_padding: Some((3, 3)),
        }
    }

    #[test]
    fn released_decoder_stem_selects_direct_route() {
        for length in [13, 25, 50, 100, 200] {
            assert_eq!(
                DecoderStemDescriptor {
                    input: [1, 1_024, length],
                    ..released_stem_descriptor()
                }
                .route(),
                DecoderStemRoute::DirectT64O32
            );
        }
    }

    #[test]
    fn decoder_stem_route_rejects_every_nonreleased_property() {
        let supported = released_stem_descriptor();
        let unsupported = [
            DecoderStemDescriptor {
                input: [2, 1_024, 50],
                ..supported
            },
            DecoderStemDescriptor {
                input: [1, 1_023, 50],
                ..supported
            },
            DecoderStemDescriptor {
                input: [1, 1_024, 0],
                ..supported
            },
            DecoderStemDescriptor {
                weight: [1_535, 1_024, 7],
                ..supported
            },
            DecoderStemDescriptor {
                weight: [1_536, 1_023, 7],
                ..supported
            },
            DecoderStemDescriptor {
                weight: [1_536, 1_024, 5],
                ..supported
            },
            DecoderStemDescriptor {
                bias_channels: None,
                ..supported
            },
            DecoderStemDescriptor {
                bias_channels: Some(1_535),
                ..supported
            },
            DecoderStemDescriptor {
                kernel_size: 5,
                ..supported
            },
            DecoderStemDescriptor {
                stride: 2,
                ..supported
            },
            DecoderStemDescriptor {
                dilation: 2,
                ..supported
            },
            DecoderStemDescriptor {
                groups: 2,
                ..supported
            },
            DecoderStemDescriptor {
                explicit_padding: Some((2, 3)),
                ..supported
            },
            DecoderStemDescriptor {
                explicit_padding: None,
                ..supported
            },
        ];
        assert!(
            unsupported
                .into_iter()
                .all(|descriptor| descriptor.route() == DecoderStemRoute::BurnFallback)
        );
    }

    #[test]
    fn wm_head_output_bounded_by_tanh() {
        let head = tiny_wm_head();
        // Use large-magnitude input to exercise tanh saturation
        let x = Tensor::<3>::ones([2, 8, 20], &Default::default()) * 100.0;
        let out = head.forward(x);
        let data: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(
            data.iter().all(|v| *v >= -1.0 && *v <= 1.0),
            "wm_head output must be in [-1, 1] due to tanh"
        );
    }

    #[test]
    fn wm_head_single_output_channel() {
        let head = tiny_wm_head();
        let x = Tensor::<3>::zeros([1, 8, 32], &Default::default());
        let out = head.forward(x);
        assert_eq!(out.dims()[1], 1, "output must have 1 channel");
    }
}

// ─── Decoder ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecoderStemRoute {
    DirectT64O32,
    BurnFallback,
}

/// Logical released-stem contract, kept separate from WGPU storage/resource
/// checks so every route condition can be tested without constructing a GPU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DecoderStemDescriptor {
    input: [usize; 3],
    weight: [usize; 3],
    bias_channels: Option<usize>,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    explicit_padding: Option<(usize, usize)>,
}

impl DecoderStemDescriptor {
    fn from_conv(conv: &Conv1d, input: [usize; 3]) -> Self {
        let explicit_padding = match &conv.padding {
            PaddingConfig1d::Explicit(left, right) => Some((*left, *right)),
            PaddingConfig1d::Same | PaddingConfig1d::Valid => None,
        };
        Self {
            input,
            weight: conv.weight.dims(),
            bias_channels: conv.bias.as_ref().map(|bias| bias.dims()[0]),
            kernel_size: conv.kernel_size,
            stride: conv.stride,
            dilation: conv.dilation,
            groups: conv.groups,
            explicit_padding,
        }
    }

    fn route(self) -> DecoderStemRoute {
        let supported = self.input[0] == 1
            && self.input[1] == 1_024
            && self.input[2] > 0
            && self.weight == [1_536, 1_024, 7]
            && self.bias_channels == Some(1_536)
            && self.kernel_size == 7
            && self.stride == 1
            && self.dilation == 1
            && self.groups == 1
            && self.explicit_padding == Some((3, 3));
        if supported {
            DecoderStemRoute::DirectT64O32
        } else {
            DecoderStemRoute::BurnFallback
        }
    }
}

/// Full DACVAE decoder (no-watermark path).
///
/// Architecture (config: `decoder_dim=1536`, rates `[12,10,8,2]`):
/// ```text
/// Conv1d(1024→1536, k=7, pad=3)
/// DecoderBlock(1536→768,  stride=12)
/// DecoderBlock(768→384,   stride=10)
/// DecoderBlock(384→192,   stride=8)
/// DecoderBlock(192→96,    stride=2)
/// WmHead(96→1)              ← forward_no_conv of WatermarkEncoderBlock
/// ```
#[derive(Module, Debug)]
pub(crate) struct Decoder {
    pub(crate) stem: Conv1d,
    pub(crate) block0: DecoderBlock,
    pub(crate) block1: DecoderBlock,
    pub(crate) block2: DecoderBlock,
    pub(crate) block3: DecoderBlock,
    pub(crate) wm_head: WmHead,
}

impl Decoder {
    pub(crate) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = self.stem.forward(x);
        let x = self.block0.forward(x);
        let x = self.block1.forward(x);
        let x = self.block2.forward(x);
        let x = self.block3.forward(x);
        self.wm_head.forward(x)
    }

    #[allow(dead_code)]
    pub(crate) fn prepare_for_inference(&mut self) {
        self.block0.prepare_for_inference();
        self.block1.prepare_for_inference();
        self.block2.prepare_for_inference();
        self.block3.prepare_for_inference();
    }
}

impl Decoder {
    pub(crate) fn prepare_for_wgsl(&mut self) {
        self.block0.prepare_residuals_for_wgsl();
        self.block1.prepare_residuals_for_wgsl();
        self.block2.prepare_residuals_for_wgsl();
        self.block3.prepare_residuals_for_wgsl();
        self.block0.prepare_conv_transpose_for_wgsl();
        // The exact WmHead fast path consumes native contiguous OIK weights,
        // alpha, and bias directly, so it has no inference cache to prepare.
    }

    #[cfg(feature = "profile")]
    pub(crate) fn prepare_for_wgsl_with_k7_algorithm(&mut self, algorithm: CodecK7Algorithm) {
        self.block0
            .prepare_residuals_for_wgsl_with_algorithm(algorithm);
        self.block1
            .prepare_residuals_for_wgsl_with_algorithm(algorithm);
        self.block2
            .prepare_residuals_for_wgsl_with_algorithm(algorithm);
        self.block3
            .prepare_residuals_for_wgsl_with_algorithm(algorithm);
        self.block0.prepare_conv_transpose_for_wgsl();
    }

    #[cfg(feature = "profile")]
    pub(crate) fn profile_k7_weight_repacks(
        &self,
    ) -> crate::error::Result<Vec<super::algorithm::K7WeightRepackReceipt>> {
        const LABELS: [&str; 12] = [
            "block0.res0",
            "block0.res1",
            "block0.res2",
            "block1.res0",
            "block1.res1",
            "block1.res2",
            "block2.res0",
            "block2.res1",
            "block2.res2",
            "block3.res0",
            "block3.res1",
            "block3.res2",
        ];
        let residuals = [
            self.block0.residuals(),
            self.block1.residuals(),
            self.block2.residuals(),
            self.block3.residuals(),
        ];
        residuals
            .into_iter()
            .flatten()
            .zip(LABELS)
            .map(|(residual, label)| residual.profile_k7_weight_repack(label))
            .collect()
    }

    pub(crate) fn lock_fixed_112_wgsl(&mut self) -> crate::error::Result<()> {
        self.block0.lock_fixed_112_polyphase_wgsl()
    }

    pub(crate) fn forward_fixed_112_wgsl(&self, x: Tensor<3>) -> crate::error::Result<Tensor<3>> {
        let x = self.stem_wgsl_or_fallback(x);
        let x = self.block0.forward_fixed_112_wgsl(x)?;
        let activated = self.block1.act.forward_wgsl(x);
        let activated = self.block1.forward_wgsl_from_activated_prepare_next_block(
            activated,
            &self.block2.act,
            super::algorithm::CodecConvTransposeSnakeFusion::Standalone,
        );
        let activated = self.block2.forward_wgsl_from_activated_prepare_next_block(
            activated,
            &self.block3.act,
            super::algorithm::CodecConvTransposeSnakeFusion::Standalone,
        );
        let x = self.block3.forward_wgsl_from_activated(
            activated,
            super::algorithm::CodecConvTransposeSnakeFusion::Standalone,
        );
        Ok(self.wm_head.forward_wgsl(x))
    }

    #[cfg(feature = "profile")]
    pub(crate) fn forward_wgsl_standalone_block_boundaries(&self, x: Tensor<3>) -> Tensor<3> {
        let x = nvtx_range!("codec_decoder_stem", self.stem_wgsl_or_fallback(x));
        let x = nvtx_range!("codec_decoder_block_0", self.block0.forward_wgsl(x));
        let x = nvtx_range!("codec_decoder_block_1", self.block1.forward_wgsl(x));
        let x = nvtx_range!("codec_decoder_block_2", self.block2.forward_wgsl(x));
        let x = nvtx_range!("codec_decoder_block_3", self.block3.forward_wgsl(x));
        nvtx_range!("codec_decoder_head", self.wm_head.forward_wgsl(x))
    }

    /// Differential route that carries the next block's prepared Snake output
    /// across block boundaries. The first block and final head retain their
    /// exact production boundaries.
    pub(crate) fn forward_wgsl_cross_block_fused(
        &self,
        x: Tensor<3>,
        policy: super::algorithm::CodecCrossBlockFusion,
    ) -> Tensor<3> {
        self.forward_wgsl_with_fusions(
            x,
            policy,
            super::algorithm::CodecConvTransposeSnakeFusion::Standalone,
        )
    }

    /// Differential route for independently selecting the two producer-side
    /// fusion families while retaining all other production algorithms.
    pub(crate) fn forward_wgsl_with_fusions(
        &self,
        x: Tensor<3>,
        cross_block_policy: super::algorithm::CodecCrossBlockFusion,
        conv_transpose_policy: super::algorithm::CodecConvTransposeSnakeFusion,
    ) -> Tensor<3> {
        let x = self.stem_wgsl_or_fallback(x);
        let activated =
            self.block0
                .forward_wgsl_prepare_next_block(x, &self.block1.act, conv_transpose_policy);
        let activated = if cross_block_policy.fuses_c384() {
            self.block1.forward_wgsl_from_activated_prepare_next_block(
                activated,
                &self.block2.act,
                conv_transpose_policy,
            )
        } else {
            let raw = self
                .block1
                .forward_wgsl_from_activated(activated, conv_transpose_policy);
            self.block2.act.forward_wgsl(raw)
        };
        let activated = if cross_block_policy.fuses_c192() {
            self.block2.forward_wgsl_from_activated_prepare_next_block(
                activated,
                &self.block3.act,
                conv_transpose_policy,
            )
        } else {
            let raw = self
                .block2
                .forward_wgsl_from_activated(activated, conv_transpose_policy);
            self.block3.act.forward_wgsl(raw)
        };
        let x = self
            .block3
            .forward_wgsl_from_activated(activated, conv_transpose_policy);
        self.wm_head.forward_wgsl(x)
    }

    pub(crate) fn forward_wgsl(&self, x: Tensor<3>) -> Tensor<3> {
        self.forward_wgsl_cross_block_fused(
            x,
            super::algorithm::CodecCrossBlockFusion::OutputC384AndC192,
        )
    }

    /// Execute the measured dynamic-stem route, falling back to Burn whenever
    /// logical metadata, physical layout, device identity, or limits disagree.
    pub(crate) fn stem_wgsl_or_fallback(&self, input: Tensor<3>) -> Tensor<3> {
        // On F16 storage, CubeCL's tuned convolution uses the device's matrix
        // acceleration and is materially faster than the scalar-F32-accumulate
        // direct shader. F32 retains the established direct route.
        if input.dtype() == burn::tensor::DType::F16 {
            return self.stem.forward(input);
        }
        let descriptor = DecoderStemDescriptor::from_conv(&self.stem, input.dims());
        if descriptor.route() == DecoderStemRoute::DirectT64O32 {
            let candidate = self.stem.bias.as_ref().and_then(|bias| {
                crate::kernels::conv1d_k7_stem_direct::try_conv1d_k7_stem_direct_wgsl(
                    input
                        .clone()
                        .try_into_primitive::<crate::WgpuRaw>()
                        .expect("tensor must use WGPU raw backend"),
                    self.stem
                        .weight
                        .val()
                        .try_into_primitive::<crate::WgpuRaw>()
                        .expect("tensor must use WGPU raw backend"),
                    bias.val()
                        .try_into_primitive::<crate::WgpuRaw>()
                        .expect("tensor must use WGPU raw backend"),
                )
            });
            if let Some(output) = candidate {
                return Tensor::from_primitive::<crate::WgpuRaw>(output);
            }
        }
        self.stem.forward(input)
    }

    #[cfg(feature = "profile")]
    pub(crate) fn forward_wgsl_profiled<P>(
        &self,
        x: Tensor<3>,
        stem_algorithm: super::algorithm::CodecStemAlgorithm,
        k7_algorithm: CodecK7Algorithm,
        pointwise_algorithm: CodecPointwiseAlgorithm,
        profiler: &mut P,
    ) -> Result<Tensor<3>, P::Error>
    where
        P: CodecStageProfiler,
    {
        let x = profile_wgsl_stage(
            "codec_decoder_stem",
            || match stem_algorithm {
                super::algorithm::CodecStemAlgorithm::AccuracyApproved => {
                    self.stem_wgsl_or_fallback(x)
                }
                super::algorithm::CodecStemAlgorithm::Burn => self.stem.forward(x),
            },
            profiler,
        )?;
        let x = self.block0.forward_wgsl_profiled_residual_parts(
            x,
            [
                "codec_block0_upsample_snake",
                "codec_block0_conv_transpose",
                "codec_block0_residual_0_act0",
                "codec_block0_residual_0_k7_act1",
                "codec_block0_residual_0_pointwise_next_act0",
                "codec_block0_residual_1_k7_act1",
                "codec_block0_residual_1_pointwise_next_act0",
                "codec_block0_residual_2_k7_act1",
                "codec_block0_residual_2_pointwise",
            ],
            [
                "codec_block0_conv_transpose_gemm",
                "codec_block0_conv_transpose_finalizer",
            ],
            k7_algorithm,
            pointwise_algorithm,
            profiler,
        )?;
        let x = self.block1.forward_wgsl_profiled_residual_parts(
            x,
            [
                "codec_block1_upsample_snake",
                "codec_block1_conv_transpose",
                "codec_block1_residual_0_act0",
                "codec_block1_residual_0_k7_act1",
                "codec_block1_residual_0_pointwise_next_act0",
                "codec_block1_residual_1_k7_act1",
                "codec_block1_residual_1_pointwise_next_act0",
                "codec_block1_residual_2_k7_act1",
                "codec_block1_residual_2_pointwise",
            ],
            [
                "codec_block1_conv_transpose_gemm",
                "codec_block1_conv_transpose_finalizer",
            ],
            k7_algorithm,
            pointwise_algorithm,
            profiler,
        )?;
        let x = self.block2.forward_wgsl_profiled_residual_parts(
            x,
            [
                "codec_block2_upsample_snake",
                "codec_block2_conv_transpose",
                "codec_block2_residual_0_act0",
                "codec_block2_residual_0_k7_act1",
                "codec_block2_residual_0_pointwise_next_act0",
                "codec_block2_residual_1_k7_act1",
                "codec_block2_residual_1_pointwise_next_act0",
                "codec_block2_residual_2_k7_act1",
                "codec_block2_residual_2_pointwise",
            ],
            [
                "codec_block2_conv_transpose_gemm",
                "codec_block2_conv_transpose_finalizer",
            ],
            k7_algorithm,
            pointwise_algorithm,
            profiler,
        )?;
        let x = self.block3.forward_wgsl_profiled_residual_parts(
            x,
            [
                "codec_block3_upsample_snake",
                "codec_block3_conv_transpose",
                "codec_block3_residual_0_act0",
                "codec_block3_residual_0_k7_act1",
                "codec_block3_residual_0_pointwise_next_act0",
                "codec_block3_residual_1_k7_act1",
                "codec_block3_residual_1_pointwise_next_act0",
                "codec_block3_residual_2_k7_act1",
                "codec_block3_residual_2_pointwise",
            ],
            [
                "codec_block3_conv_transpose_gemm",
                "codec_block3_conv_transpose_finalizer",
            ],
            k7_algorithm,
            pointwise_algorithm,
            profiler,
        )?;
        profile_wgsl_stage(
            "codec_decoder_head",
            || self.wm_head.forward_wgsl(x),
            profiler,
        )
    }
}
