//! DACVAE VAE bottleneck: projection to/from the compact latent space.

use burn::{nn::conv::Conv1d, prelude::*};

use super::layers::pointwise_conv1d;

/// Projects encoder output to `[mean, scale]` and reconstructs embeddings
/// from quantized latents.
///
/// Architecture for `latent_dim=1024`, `codebook_dim=32`:
/// ```text
/// in_proj:  Conv1d(1024 → 64, k=1)   → split → mean [B, 32, T]
/// out_proj: Conv1d(32   → 1024, k=1)
/// ```
#[derive(Module, Debug)]
pub(crate) struct VaeBottleneck {
    /// Projects latent_dim → codebook_dim*2; split dim=1 gives [mean, scale].
    pub(crate) in_proj: Conv1d,
    /// Projects codebook_dim → latent_dim.
    pub(crate) out_proj: Conv1d,
    pub(crate) codebook_dim: usize,
}

impl VaeBottleneck {
    /// Deterministic encode: returns the mean of the VAE posterior.
    pub(crate) fn encode(&self, z: Tensor<3>) -> Tensor<3> {
        let ms = self.in_proj.forward(z);
        // split along channel dim: take first half (mean)
        ms.narrow(1, 0, self.codebook_dim)
    }

    /// Decode: project compact code back to latent_dim.
    pub(crate) fn decode(&self, code: Tensor<3>) -> Tensor<3> {
        self.out_proj.forward(code)
    }
}

impl VaeBottleneck {
    pub(crate) fn encode_wgsl(&self, z: Tensor<3>) -> Tensor<3> {
        pointwise_conv1d(&self.in_proj, z).narrow(1, 0, self.codebook_dim)
    }

    pub(crate) fn decode_wgsl(&self, code: Tensor<3>) -> Tensor<3> {
        pointwise_conv1d(&self.out_proj, code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::nn::conv::Conv1dConfig;

    fn tiny_bottleneck() -> VaeBottleneck {
        let dev = Default::default();
        let latent_dim = 16;
        let codebook_dim = 4;
        VaeBottleneck {
            in_proj: Conv1dConfig::new(latent_dim, codebook_dim * 2, 1).init(&dev),
            out_proj: Conv1dConfig::new(codebook_dim, latent_dim, 1).init(&dev),
            codebook_dim,
        }
    }

    #[test]
    fn encode_returns_codebook_dim_channels() {
        let bn = tiny_bottleneck();
        let z = Tensor::<3>::zeros([2, 16, 10], &Default::default());
        let code = bn.encode(z);
        assert_eq!(code.dims(), [2, 4, 10]);
    }

    #[test]
    fn decode_restores_latent_dim() {
        let bn = tiny_bottleneck();
        let code = Tensor::<3>::zeros([2, 4, 10], &Default::default());
        let z = bn.decode(code);
        assert_eq!(z.dims(), [2, 16, 10]);
    }

    #[test]
    fn encode_decode_preserves_time_dimension() {
        let bn = tiny_bottleneck();
        for t in [1, 7, 32] {
            let z = Tensor::<3>::zeros([1, 16, t], &Default::default());
            let code = bn.encode(z);
            assert_eq!(code.dims()[2], t, "time dim must be preserved by encode");
            let restored = bn.decode(code);
            assert_eq!(
                restored.dims()[2],
                t,
                "time dim must be preserved by decode"
            );
        }
    }
}
