//! DACVAE decoder: stem Conv → 4× DecoderBlock → WmHead (no-watermark path).

#[cfg(feature = "profile")]
use std::time::{Duration, Instant};

use burn::{
    nn::{
        PaddingConfig1d,
        conv::{Conv1d, ConvTranspose1d},
    },
    prelude::*,
};

use super::layers::{ResidualUnit, Snake1d};
use crate::nvtx_range;

#[cfg(feature = "profile")]
fn profile_wgsl_stage<T, E, O, S>(
    label: &'static str,
    operation: O,
    synchronize: &mut S,
    timings: &mut Vec<(&'static str, Duration)>,
) -> Result<T, E>
where
    O: FnOnce() -> T,
    S: FnMut(&'static str) -> Result<(), E>,
{
    let started = Instant::now();
    let output = operation();
    synchronize(label)?;
    timings.push((label, started.elapsed()));
    Ok(output)
}

// ─── DecoderBlock ────────────────────────────────────────────────────────────

/// One upsampling block: Snake → ConvTranspose → 3× ResidualUnit.
///
/// Only the main signal path (blocks 0,1,4,5,8,9 of the original Python
/// `ModuleList`) is implemented. Watermark-only branches are omitted.
#[derive(Module, Debug)]
pub(crate) struct DecoderBlock<B: Backend> {
    pub(crate) act: Snake1d<B>,
    pub(crate) conv_t: ConvTranspose1d<B>,
    /// Inference-only `[phase, Cout, Cin, 2]` cache for the first upsampler.
    #[module(skip)]
    pub(crate) packed_conv_t_weight: Option<Tensor<B, 4>>,
    pub(crate) res0: ResidualUnit<B>,
    pub(crate) res1: ResidualUnit<B>,
    pub(crate) res2: ResidualUnit<B>,
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
    fn from_conv<B: Backend>(conv: &ConvTranspose1d<B>) -> Self {
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

impl<B: Backend> DecoderBlock<B> {
    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
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

impl DecoderBlock<crate::WgpuRaw> {
    fn prepare_residuals_for_wgsl(&mut self) {
        self.res0.prepare_for_wgsl();
        self.res1.prepare_for_wgsl();
        self.res2.prepare_for_wgsl();
    }

    fn prepare_conv_transpose_for_wgsl(&mut self) {
        use burn::tensor::TensorPrimitive;

        if self.packed_conv_t_weight.is_some() {
            return;
        }
        let Some(stride) =
            ConvTransposeModuleDescriptor::from_conv(&self.conv_t).polyphase_stride()
        else {
            return;
        };

        let packed = crate::kernels::conv_transpose1d_polyphase::pack_conv_transpose1d_weight_wgsl(
            self.conv_t.weight.val().into_primitive().tensor(),
            stride,
        );
        self.packed_conv_t_weight = Some(Tensor::from_primitive(TensorPrimitive::Float(
            packed.into_tensor(),
        )));
    }

    fn forward_wgsl(&self, x: Tensor<crate::WgpuRaw, 3>) -> Tensor<crate::WgpuRaw, 3> {
        let x = nvtx_range!("codec_upsample_snake", self.act.forward_wgsl(x));
        let x = nvtx_range!(
            "codec_conv_transpose",
            self.conv_transpose_wgsl_or_fallback(x)
        );
        let pair = nvtx_range!(
            "codec_residual_unit_0",
            self.res0.forward_wgsl_prepare_next(x, &self.res1.act0)
        );
        let pair = nvtx_range!(
            "codec_residual_unit_1",
            self.res1
                .forward_wgsl_from_prepared_prepare_next(pair, &self.res2.act0)
        );
        nvtx_range!(
            "codec_residual_unit_2",
            self.res2.forward_wgsl_from_prepared(pair)
        )
    }

    #[cfg(feature = "profile")]
    fn forward_wgsl_profiled_residual_parts<E, S>(
        &self,
        x: Tensor<crate::WgpuRaw, 3>,
        labels: [&'static str; 9],
        cached_conv_labels: [&'static str; 2],
        synchronize: &mut S,
        timings: &mut Vec<(&'static str, Duration)>,
    ) -> Result<Tensor<crate::WgpuRaw, 3>, E>
    where
        S: FnMut(&'static str) -> Result<(), E>,
    {
        let x = profile_wgsl_stage(labels[0], || self.act.forward_wgsl(x), synchronize, timings)?;
        let x = self.conv_transpose_wgsl_or_fallback_profiled(
            x,
            labels[1],
            cached_conv_labels,
            synchronize,
            timings,
        )?;
        let pair = self.res0.forward_wgsl_profiled_prepare_next(
            x,
            &self.res1.act0,
            [labels[2], labels[3], labels[4]],
            synchronize,
            timings,
        )?;
        let pair = self.res1.forward_wgsl_profiled_from_prepared_prepare_next(
            pair,
            &self.res2.act0,
            [labels[5], labels[6]],
            synchronize,
            timings,
        )?;
        self.res2.forward_wgsl_profiled_from_prepared(
            pair,
            [labels[7], labels[8]],
            synchronize,
            timings,
        )
    }

    fn conv_transpose_wgsl_or_fallback(
        &self,
        input: Tensor<crate::WgpuRaw, 3>,
    ) -> Tensor<crate::WgpuRaw, 3> {
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
    fn conv_transpose_wgsl_or_fallback_profiled<E, S>(
        &self,
        input: Tensor<crate::WgpuRaw, 3>,
        fallback_label: &'static str,
        cached_labels: [&'static str; 2],
        synchronize: &mut S,
        timings: &mut Vec<(&'static str, Duration)>,
    ) -> Result<Tensor<crate::WgpuRaw, 3>, E>
    where
        S: FnMut(&'static str) -> Result<(), E>,
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
                synchronize,
                timings,
            )?
        {
            return Ok(output);
        }
        profile_wgsl_stage(
            fallback_label,
            || self.conv_transpose_wgsl_or_fallback(input),
            synchronize,
            timings,
        )
    }

    fn try_case0_cached_col2im_conv_transpose_wgsl(
        &self,
        input: Tensor<crate::WgpuRaw, 3>,
    ) -> Option<Tensor<crate::WgpuRaw, 3>> {
        use burn::tensor::TensorPrimitive;

        let bias = self.conv_t.bias.as_ref()?;
        let output = crate::kernels::conv_transpose1d_cached_col2im_case0::conv_transpose1d_case0_cached_col2im_wgsl(
            input.into_primitive().tensor(),
            self.conv_t.weight.val().into_primitive().tensor(),
            bias.val().into_primitive().tensor(),
        )
        .ok()?;
        Some(Tensor::from_primitive(TensorPrimitive::Float(output)))
    }

    fn try_polyphase_conv_transpose_wgsl(
        &self,
        input: Tensor<crate::WgpuRaw, 3>,
        stride: crate::kernels::conv_transpose1d_polyphase::ConvTranspose1dStride,
    ) -> Option<Tensor<crate::WgpuRaw, 3>> {
        use burn::tensor::TensorPrimitive;

        let Some(packed_weight) = &self.packed_conv_t_weight else {
            return None;
        };
        let Some(bias) = &self.conv_t.bias else {
            return None;
        };

        let packed_weight = packed_weight.clone().into_primitive().tensor();
        let bias = bias.val().into_primitive().tensor();
        if !packed_weight.is_contiguous() || !bias.is_contiguous() {
            return None;
        }
        let packed_weight =
            crate::kernels::conv_transpose1d_polyphase::PackedConvTranspose1dWeight::from_tensor(
                packed_weight,
                stride,
            );
        let output = crate::kernels::conv_transpose1d_polyphase::conv_transpose1d_polyphase_wgsl(
            input.into_primitive().tensor(),
            &packed_weight,
            bias,
        );
        Some(Tensor::from_primitive(TensorPrimitive::Float(output)))
    }

    fn try_cached_col2im_conv_transpose_wgsl(
        &self,
        input: Tensor<crate::WgpuRaw, 3>,
        case: crate::kernels::conv_transpose1d_cached_col2im::CachedCol2ImCase,
    ) -> Option<Tensor<crate::WgpuRaw, 3>> {
        use burn::tensor::TensorPrimitive;

        let bias = self.conv_t.bias.as_ref()?;
        let output =
            crate::kernels::conv_transpose1d_cached_col2im::conv_transpose1d_cached_col2im_wgsl(
                input.into_primitive().tensor(),
                self.conv_t.weight.val().into_primitive().tensor(),
                bias.val().into_primitive().tensor(),
                case,
            )
            .ok()?;
        Some(Tensor::from_primitive(TensorPrimitive::Float(output)))
    }

    #[cfg(feature = "profile")]
    fn try_cached_col2im_conv_transpose_wgsl_profiled<E, S>(
        &self,
        input: Tensor<crate::WgpuRaw, 3>,
        case: crate::kernels::conv_transpose1d_cached_col2im::CachedCol2ImCase,
        labels: [&'static str; 2],
        synchronize: &mut S,
        timings: &mut Vec<(&'static str, Duration)>,
    ) -> Result<Option<Tensor<crate::WgpuRaw, 3>>, E>
    where
        S: FnMut(&'static str) -> Result<(), E>,
    {
        use burn::tensor::TensorPrimitive;

        let Some(bias) = self.conv_t.bias.as_ref() else {
            return Ok(None);
        };
        let bias = bias.val().into_primitive().tensor();
        let started = Instant::now();
        let Ok(columns) =
            crate::kernels::conv_transpose1d_cached_col2im::matmul_cached_col2im_columns_wgsl(
                input.into_primitive().tensor(),
                self.conv_t.weight.val().into_primitive().tensor(),
                &bias,
                case,
            )
        else {
            return Ok(None);
        };
        synchronize(labels[0])?;
        timings.push((labels[0], started.elapsed()));

        let started = Instant::now();
        let Ok(output) =
            crate::kernels::conv_transpose1d_cached_col2im::finalize_cached_col2im_wgsl(
                columns, bias, case,
            )
        else {
            return Ok(None);
        };
        synchronize(labels[1])?;
        timings.push((labels[1], started.elapsed()));
        Ok(Some(Tensor::from_primitive(TensorPrimitive::Float(output))))
    }
}

// ─── WmHead ──────────────────────────────────────────────────────────────────

/// Watermark-less output head: `Snake → Conv(96→1) → Tanh`.
///
/// This is the `forward_no_conv` path of `WatermarkEncoderBlock.pre`:
/// the final `NormConv1d(1→32)` is replaced by identity, so we only
/// apply `[Snake, Conv(96→1), Tanh]`.
#[derive(Module, Debug)]
pub(crate) struct WmHead<B: Backend> {
    pub(crate) act: Snake1d<B>,
    pub(crate) conv: Conv1d<B>,
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
    fn from_head<B: Backend>(head: &WmHead<B>, input: [usize; 3]) -> Self {
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

impl<B: Backend> WmHead<B> {
    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.act.forward(x);
        let x = self.conv.forward(x);
        x.tanh()
    }
}

impl WmHead<crate::WgpuRaw> {
    fn forward_wgsl(&self, x: Tensor<crate::WgpuRaw, 3>) -> Tensor<crate::WgpuRaw, 3> {
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

    fn try_fused_final_wgsl(
        &self,
        input: Tensor<crate::WgpuRaw, 3>,
    ) -> Option<Tensor<crate::WgpuRaw, 3>> {
        use burn::tensor::TensorPrimitive;

        let bias = self.conv.bias.as_ref()?;
        let output =
            crate::kernels::wm_head_fused_final_t240_c16::try_wm_head_fused_final_t240_c16_wgsl(
                input.into_primitive().tensor(),
                self.act.alpha.val().into_primitive().tensor(),
                self.conv.weight.val().into_primitive().tensor(),
                bias.val().into_primitive().tensor(),
            )?;
        Some(Tensor::from_primitive(TensorPrimitive::Float(output)))
    }

    fn try_snake_nlc_wgsl(
        &self,
        input: Tensor<crate::WgpuRaw, 3>,
    ) -> Option<Tensor<crate::WgpuRaw, 3>> {
        use burn::tensor::TensorPrimitive;

        let output_nlc = crate::kernels::wm_head_snake_nlc::wm_head_snake_ncl_to_nlc_wgsl(
            input.into_primitive().tensor(),
            self.act.alpha.val().into_primitive().tensor(),
        )
        .ok()?;
        let output_nlc = Tensor::from_primitive(TensorPrimitive::Float(output_nlc));
        Some(output_nlc.swap_dims(1, 2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::nn::conv::Conv1dConfig;

    type B = NdArray;

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

    fn tiny_wm_head() -> WmHead<B> {
        let dev = Default::default();
        let channels = 8;
        WmHead {
            act: Snake1d::new(Tensor::<B, 3>::ones([1, channels, 1], &dev)),
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

    fn released_wm_head() -> WmHead<B> {
        let dev = Default::default();
        WmHead {
            act: Snake1d::new(Tensor::<B, 3>::ones([1, 96, 1], &dev)),
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
    fn wgsl_wm_head_source_keeps_candidate_then_existing_fallback_order() {
        let source = include_str!("decoder.rs");
        let implementation = source
            .split_once("impl WmHead<crate::WgpuRaw> {")
            .expect("WGPU WmHead implementation")
            .1
            .split_once("#[cfg(test)]")
            .expect("end of WGPU WmHead implementation")
            .0;

        let candidate = implementation
            .find("self.try_fused_final_wgsl(x.clone())")
            .expect("fused-final candidate route");
        let existing = implementation
            .find("self.try_snake_nlc_wgsl(x.clone())")
            .expect("existing Snake/NLC fallback");
        let generic = implementation
            .find("self.act.forward_wgsl(x)")
            .expect("generic Snake fallback");
        let conv = implementation
            .find("self.conv.forward(x)")
            .expect("existing convolution tail");
        let tanh = implementation.find("x.tanh()").expect("existing tanh tail");
        assert!(candidate < existing && existing < generic && generic < conv && conv < tanh);
        assert!(implementation[candidate..existing].contains("return output;"));
        assert!(implementation.contains("try_wm_head_fused_final_t240_c16_wgsl("));
        assert_eq!(implementation.matches("self.conv.forward(x)").count(), 1);
        assert_eq!(implementation.matches("x.tanh()").count(), 1);
        assert!(implementation.contains("output_nlc.swap_dims(1, 2)"));
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
    fn first_upsampler_source_keeps_case0_then_polyphase_then_burn_fallback_order() {
        let source = include_str!("decoder.rs");
        let function = source
            .split_once("fn conv_transpose_wgsl_or_fallback(")
            .expect("ConvTranspose production route")
            .1
            .split_once("fn try_case0_cached_col2im_conv_transpose_wgsl(")
            .expect("case-0 helper boundary")
            .0;
        let case0 = function
            .find(".try_case0_cached_col2im_conv_transpose_wgsl(input.clone())")
            .expect("case-0 candidate call");
        let polyphase = function
            .find(".or_else(|| self.try_polyphase_conv_transpose_wgsl(input.clone(), stride))")
            .expect("polyphase fallback call");
        let burn = function
            .find("candidate.unwrap_or_else(|| self.conv_t.forward(input))")
            .expect("Burn fallback call");
        assert!(case0 < polyphase && polyphase < burn);
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

    #[test]
    fn wgsl_decoder_block_prepares_only_two_intra_residual_pairs() {
        let source = include_str!("decoder.rs");
        let implementation = source
            .split_once("impl DecoderBlock<crate::WgpuRaw> {")
            .expect("WGPU DecoderBlock implementation")
            .1;
        let forward = implementation
            .split_once("fn forward_wgsl(")
            .expect("WGPU DecoderBlock forward")
            .1
            .split_once("#[cfg(feature = \"profile\")]")
            .expect("end of WGPU DecoderBlock forward")
            .0;

        assert_eq!(forward.matches("forward_wgsl_prepare_next(").count(), 1);
        assert_eq!(
            forward
                .matches("forward_wgsl_from_prepared_prepare_next(")
                .count(),
            1
        );
        assert_eq!(forward.matches("forward_wgsl_from_prepared(").count(), 1);
        assert!(forward.contains("self.res0.forward_wgsl_prepare_next(x, &self.res1.act0)"));
        assert!(
            forward.contains(".forward_wgsl_from_prepared_prepare_next(pair, &self.res2.act0)")
        );
        assert!(forward.contains("self.res2.forward_wgsl_from_prepared(pair)"));
        assert!(!forward.contains("self.res2.forward_wgsl_prepare_next"));
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
    fn wgsl_decoder_stem_keeps_explicit_burn_fallback() {
        let source = include_str!("decoder.rs");
        let implementation_marker = ["impl Decoder<crate::WgpuRaw>", " {"].concat();
        let implementation = source
            .split_once(&implementation_marker)
            .expect("WGPU Decoder implementation")
            .1;
        let forward = implementation
            .split_once("pub(crate) fn forward_wgsl(")
            .expect("production WGPU decoder forward")
            .1
            .split_once("pub(crate) fn stem_wgsl_or_fallback(")
            .expect("production decoder stem route")
            .0;
        let stem_route = implementation
            .split_once("pub(crate) fn stem_wgsl_or_fallback(")
            .expect("production decoder stem route")
            .1
            .split_once("#[cfg(feature = \"profile\")]")
            .expect("end of decoder stem route")
            .0;

        assert!(forward.contains("self.stem_wgsl_or_fallback(x)"));
        assert!(stem_route.contains("DecoderStemRoute::DirectT64O32"));
        assert!(stem_route.contains("try_conv1d_k7_stem_direct_wgsl("));
        assert!(stem_route.contains("self.stem.forward(input)"));
    }

    #[test]
    fn wm_head_output_bounded_by_tanh() {
        let head = tiny_wm_head();
        // Use large-magnitude input to exercise tanh saturation
        let x = Tensor::<B, 3>::ones([2, 8, 20], &Default::default()) * 100.0;
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
        let x = Tensor::<B, 3>::zeros([1, 8, 32], &Default::default());
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
    fn from_conv<B: Backend>(conv: &Conv1d<B>, input: [usize; 3]) -> Self {
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
pub(crate) struct Decoder<B: Backend> {
    pub(crate) stem: Conv1d<B>,
    pub(crate) block0: DecoderBlock<B>,
    pub(crate) block1: DecoderBlock<B>,
    pub(crate) block2: DecoderBlock<B>,
    pub(crate) block3: DecoderBlock<B>,
    pub(crate) wm_head: WmHead<B>,
}

impl<B: Backend> Decoder<B> {
    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
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

impl Decoder<crate::WgpuRaw> {
    pub(crate) fn prepare_for_wgsl(&mut self) {
        self.block0.prepare_residuals_for_wgsl();
        self.block1.prepare_residuals_for_wgsl();
        self.block2.prepare_residuals_for_wgsl();
        self.block3.prepare_residuals_for_wgsl();
        self.block0.prepare_conv_transpose_for_wgsl();
        // The exact WmHead fast path consumes native contiguous OIK weights,
        // alpha, and bias directly, so it has no inference cache to prepare.
    }

    pub(crate) fn forward_wgsl(&self, x: Tensor<crate::WgpuRaw, 3>) -> Tensor<crate::WgpuRaw, 3> {
        let x = nvtx_range!("codec_decoder_stem", self.stem_wgsl_or_fallback(x));
        let x = nvtx_range!("codec_decoder_block_0", self.block0.forward_wgsl(x));
        let x = nvtx_range!("codec_decoder_block_1", self.block1.forward_wgsl(x));
        let x = nvtx_range!("codec_decoder_block_2", self.block2.forward_wgsl(x));
        let x = nvtx_range!("codec_decoder_block_3", self.block3.forward_wgsl(x));
        nvtx_range!("codec_decoder_head", self.wm_head.forward_wgsl(x))
    }

    /// Execute the measured dynamic-stem route, falling back to Burn whenever
    /// logical metadata, physical layout, device identity, or limits disagree.
    pub(crate) fn stem_wgsl_or_fallback(
        &self,
        input: Tensor<crate::WgpuRaw, 3>,
    ) -> Tensor<crate::WgpuRaw, 3> {
        use burn::tensor::TensorPrimitive;

        let descriptor = DecoderStemDescriptor::from_conv(&self.stem, input.dims());
        if descriptor.route() == DecoderStemRoute::DirectT64O32 {
            let candidate = self.stem.bias.as_ref().and_then(|bias| {
                crate::kernels::conv1d_k7_stem_direct::try_conv1d_k7_stem_direct_wgsl(
                    input.clone().into_primitive().tensor(),
                    self.stem.weight.val().into_primitive().tensor(),
                    bias.val().into_primitive().tensor(),
                )
            });
            if let Some(output) = candidate {
                return Tensor::from_primitive(TensorPrimitive::Float(output));
            }
        }
        self.stem.forward(input)
    }

    #[cfg(feature = "profile")]
    pub(crate) fn forward_wgsl_profiled<E, S>(
        &self,
        x: Tensor<crate::WgpuRaw, 3>,
        synchronize: &mut S,
        timings: &mut Vec<(&'static str, Duration)>,
    ) -> Result<Tensor<crate::WgpuRaw, 3>, E>
    where
        S: FnMut(&'static str) -> Result<(), E>,
    {
        let x = profile_wgsl_stage(
            "codec_decoder_stem",
            || self.stem_wgsl_or_fallback(x),
            synchronize,
            timings,
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
            synchronize,
            timings,
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
            synchronize,
            timings,
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
            synchronize,
            timings,
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
            synchronize,
            timings,
        )?;
        profile_wgsl_stage(
            "codec_decoder_head",
            || self.wm_head.forward_wgsl(x),
            synchronize,
            timings,
        )
    }
}
