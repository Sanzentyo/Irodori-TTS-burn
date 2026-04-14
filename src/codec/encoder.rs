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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::nn::PaddingConfig1d;
    use burn::nn::conv::Conv1dConfig;

    type B = NdArray;

    /// Build a ResidualUnit with default (random) weights for testing.
    fn test_res_unit(dim: usize, dilation: usize, dev: &<B as Backend>::Device) -> ResidualUnit<B> {
        let pad = super::super::layers::conv_pad(7, 1, dilation);
        let conv_dil = Conv1dConfig::new(dim, dim, 7)
            .with_dilation(dilation)
            .with_padding(PaddingConfig1d::Explicit(pad, pad))
            .with_bias(true)
            .init::<B>(dev);
        let conv_1x1 = Conv1dConfig::new(dim, dim, 1)
            .with_bias(true)
            .init::<B>(dev);
        ResidualUnit {
            act0: Snake1d::new(Tensor::<B, 3>::ones([1, dim, 1], dev)),
            conv_dil,
            act1: Snake1d::new(Tensor::<B, 3>::ones([1, dim, 1], dev)),
            conv_1x1,
        }
    }

    /// Build an EncoderBlock with default weights for testing.
    fn test_encoder_block(
        in_dim: usize,
        out_dim: usize,
        stride: usize,
        dev: &<B as Backend>::Device,
    ) -> EncoderBlock<B> {
        let kernel = 2 * stride;
        let pad = super::super::layers::conv_pad(kernel, stride, 1);
        let tail_conv = Conv1dConfig::new(in_dim, out_dim, kernel)
            .with_stride(stride)
            .with_padding(PaddingConfig1d::Explicit(pad, pad))
            .with_bias(true)
            .init::<B>(dev);
        EncoderBlock {
            res0: test_res_unit(in_dim, 1, dev),
            res1: test_res_unit(in_dim, 3, dev),
            res2: test_res_unit(in_dim, 9, dev),
            tail_act: Snake1d::new(Tensor::<B, 3>::ones([1, in_dim, 1], dev)),
            tail_conv,
        }
    }

    #[test]
    fn encoder_block_doubles_channels() {
        let dev = Default::default();
        let block = test_encoder_block(8, 16, 2, &dev);
        let x = Tensor::<B, 3>::zeros([1, 8, 64], &dev);
        let out = block.forward(x);
        assert_eq!(out.dims()[1], 16, "output channels should double");
    }

    #[test]
    fn encoder_block_downsamples_time() {
        let dev = Default::default();
        let stride = 4;
        let block = test_encoder_block(8, 16, stride, &dev);
        let time_in = 128;
        let x = Tensor::<B, 3>::zeros([1, 8, time_in], &dev);
        let out = block.forward(x);
        assert_eq!(
            out.dims()[2],
            time_in / stride,
            "time dimension should be downsampled by stride"
        );
    }

    #[test]
    fn encoder_block_preserves_batch() {
        let dev = Default::default();
        let block = test_encoder_block(8, 16, 2, &dev);
        let x = Tensor::<B, 3>::zeros([3, 8, 64], &dev);
        let out = block.forward(x);
        assert_eq!(out.dims()[0], 3, "batch dimension should be preserved");
    }

    #[test]
    fn full_encoder_channel_progression() {
        let dev = Default::default();
        // Use small dims (4→8→16→32→64) with stride=2 for all blocks
        let stem = Conv1dConfig::new(1, 4, 7)
            .with_padding(PaddingConfig1d::Explicit(3, 3))
            .with_bias(true)
            .init::<B>(&dev);
        let encoder = Encoder {
            stem,
            block0: test_encoder_block(4, 8, 2, &dev),
            block1: test_encoder_block(8, 16, 2, &dev),
            block2: test_encoder_block(16, 32, 2, &dev),
            block3: test_encoder_block(32, 64, 2, &dev),
            tail_act: Snake1d::new(Tensor::<B, 3>::ones([1, 64, 1], &dev)),
            tail_conv: Conv1dConfig::new(64, 64, 3)
                .with_padding(PaddingConfig1d::Explicit(1, 1))
                .with_bias(true)
                .init::<B>(&dev),
        };

        // Input: [1, 1, 256] mono audio
        let x = Tensor::<B, 3>::zeros([1, 1, 256], &dev);
        let out = encoder.forward(x);
        // After 4 blocks with stride=2: 256 / 2^4 = 16
        assert_eq!(out.dims(), [1, 64, 16]);
    }
}
