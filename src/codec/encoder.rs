//! DACVAE encoder: stem Conv → 4× EncoderBlock → Snake tail → tail Conv.

use burn::{nn::conv::Conv1d, prelude::*};

use super::layers::{ResidualUnit, Snake1d};

// ─── EncoderBlock ────────────────────────────────────────────────────────────

/// One downsampling block: 3 ResidualUnits + Snake + striding Conv.
///
/// Input channels: `dim / 2`, output channels: `dim`, stride: `stride`.
/// Striding conv: `kernel = 2 * stride`, `pad = stride / 2`.
#[derive(Module, Debug)]
pub(crate) struct EncoderBlock<B: Backend> {
    pub(crate) res0: ResidualUnit<B>,
    pub(crate) res1: ResidualUnit<B>,
    pub(crate) res2: ResidualUnit<B>,
    pub(crate) tail_act: Snake1d<B>,
    pub(crate) tail_conv: Conv1d<B>,
}

impl<B: Backend> EncoderBlock<B> {
    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.res0.forward(x);
        let x = self.res1.forward(x);
        let x = self.res2.forward(x);
        let x = self.tail_act.forward(x);
        self.tail_conv.forward(x)
    }
}

// ─── Encoder ─────────────────────────────────────────────────────────────────

/// Full DACVAE encoder.
///
/// Architecture (config: `encoder_dim=64`, rates `[2,8,10,12]`, `latent_dim=1024`):
/// ```text
/// Conv1d(1→64, k=7, pad=3)
/// EncoderBlock(64→128, stride=2)
/// EncoderBlock(128→256, stride=8)
/// EncoderBlock(256→512, stride=10)
/// EncoderBlock(512→1024, stride=12)
/// Snake1d(1024)
/// Conv1d(1024→1024, k=3, pad=1)
/// ```
#[derive(Module, Debug)]
pub(crate) struct Encoder<B: Backend> {
    pub(crate) stem: Conv1d<B>,
    pub(crate) block0: EncoderBlock<B>,
    pub(crate) block1: EncoderBlock<B>,
    pub(crate) block2: EncoderBlock<B>,
    pub(crate) block3: EncoderBlock<B>,
    pub(crate) tail_act: Snake1d<B>,
    pub(crate) tail_conv: Conv1d<B>,
}

impl<B: Backend> Encoder<B> {
    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.stem.forward(x);
        let x = self.block0.forward(x);
        let x = self.block1.forward(x);
        let x = self.block2.forward(x);
        let x = self.block3.forward(x);
        let x = self.tail_act.forward(x);
        self.tail_conv.forward(x)
    }
}
