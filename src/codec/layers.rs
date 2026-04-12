//! Shared layers for the DACVAE codec: Snake1d activation and Snake ResidualUnit.

use burn::{
    module::{Param, ParamId},
    nn::{PaddingConfig1d, conv::Conv1d},
    prelude::*,
};

// ─── Snake1d ─────────────────────────────────────────────────────────────────

/// `x + sin²(α·x) / (α + ε)` activation, element-wise.
///
/// Alpha has shape `[1, channels, 1]` and is stored as a non-trained
/// inference-time constant.
#[derive(Module, Debug)]
pub(crate) struct Snake1d<B: Backend> {
    pub(crate) alpha: Param<Tensor<B, 3>>,
}

impl<B: Backend> Snake1d<B> {
    pub(crate) fn new(alpha_tensor: Tensor<B, 3>) -> Self {
        Self {
            alpha: Param::initialized(ParamId::new(), alpha_tensor),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let alpha = self.alpha.val();
        let ax = x.clone().mul(alpha.clone());
        let sin_sq = ax.sin().powi_scalar(2);
        let denom = alpha.add_scalar(1e-9_f32);
        x + sin_sq.div(denom)
    }
}

// ─── ResidualUnit ─────────────────────────────────────────────────────────────

/// Snake → dilated Conv → Snake → 1×1 Conv + identity shortcut.
///
/// Padding for dilated conv: `(kernel - 1) * dil / 2 = 3 * dil` (kernel fixed at 7).
/// All residual units in the main encoder/decoder path use `compress=1` so
/// hidden dimension equals `dim` and the shortcut is a perfect identity.
#[derive(Module, Debug)]
pub(crate) struct ResidualUnit<B: Backend> {
    pub(crate) act0: Snake1d<B>,
    pub(crate) conv_dil: Conv1d<B>,
    pub(crate) act1: Snake1d<B>,
    pub(crate) conv_1x1: Conv1d<B>,
}

impl<B: Backend> ResidualUnit<B> {
    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let y = self.act0.forward(x);
        let y = self.conv_dil.forward(y);
        let y = self.act1.forward(y);
        let y = self.conv_1x1.forward(y);
        y + residual
    }
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
pub(crate) fn make_conv1d<B: Backend>(
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    weight: Tensor<B, 3>,
    bias: Option<Tensor<B, 1>>,
    device: &B::Device,
) -> Conv1d<B> {
    let pad = conv_pad(kernel, stride, dilation);
    let mut conv = Conv1dConfig::new(in_ch, out_ch, kernel)
        .with_stride(stride)
        .with_dilation(dilation)
        .with_padding(PaddingConfig1d::Explicit(pad, pad))
        .with_bias(bias.is_some())
        .init::<B>(device);
    conv.weight = Param::initialized(ParamId::new(), weight);
    conv.bias = bias.map(|b| Param::initialized(ParamId::new(), b));
    conv
}
