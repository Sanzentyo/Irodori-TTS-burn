//! DACVAE decoder: stem Conv → 4× DecoderBlock → WmHead (no-watermark path).

use burn::{
    nn::conv::{Conv1d, ConvTranspose1d},
    prelude::*,
};

use super::layers::{ResidualUnit, Snake1d};

// ─── DecoderBlock ────────────────────────────────────────────────────────────

/// One upsampling block: Snake → ConvTranspose → 3× ResidualUnit.
///
/// Only the main signal path (blocks 0,1,4,5,8,9 of the original Python
/// `ModuleList`) is implemented. Watermark-only branches are omitted.
#[derive(Module, Debug)]
pub(crate) struct DecoderBlock<B: Backend> {
    pub(crate) act: Snake1d<B>,
    pub(crate) conv_t: ConvTranspose1d<B>,
    pub(crate) res0: ResidualUnit<B>,
    pub(crate) res1: ResidualUnit<B>,
    pub(crate) res2: ResidualUnit<B>,
}

impl<B: Backend> DecoderBlock<B> {
    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.act.forward(x);
        let x = self.conv_t.forward(x);
        let x = self.res0.forward(x);
        let x = self.res1.forward(x);
        self.res2.forward(x)
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

impl<B: Backend> WmHead<B> {
    pub(crate) fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.act.forward(x);
        let x = self.conv.forward(x);
        x.tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::nn::conv::Conv1dConfig;

    type B = NdArray;

    fn tiny_wm_head() -> WmHead<B> {
        let dev = Default::default();
        let channels = 8;
        WmHead {
            act: Snake1d::new(Tensor::<B, 3>::ones([1, channels, 1], &dev)),
            conv: Conv1dConfig::new(channels, 1, 1).init(&dev),
        }
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
}
