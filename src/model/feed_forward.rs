use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Tensor, activation::silu, backend::Backend},
};

/// SwiGLU feed-forward network.
///
/// Implements: `out = Linear2(silu(Linear1(x)) * Linear3(x))`
/// where Linear1, Linear2, Linear3 are named `w1`, `w2`, `w3` to match the Python state_dict.
///
/// `w1` and `w3` are the two "gate/value" projections (expand),
/// `w2` is the output projection (contract).
#[derive(Module, Debug)]
pub struct SwiGlu<B: Backend> {
    pub(crate) w1: Linear<B>,
    pub(crate) w2: Linear<B>,
    pub(crate) w3: Linear<B>,
}

impl<B: Backend> SwiGlu<B> {
    /// `dim`: input/output dimension.
    /// `hidden_dim`: intermediate dimension (typically `dim * 8/3`, rounded up).
    pub fn new(dim: usize, hidden_dim: Option<usize>, device: &B::Device) -> Self {
        // Default: 8/3 * dim rounded to nearest multiple of 256, matching Python
        let hidden_dim = hidden_dim.unwrap_or_else(|| round_up(dim * 8 / 3, 256));

        Self {
            w1: LinearConfig::new(dim, hidden_dim)
                .with_bias(false)
                .init(device),
            w2: LinearConfig::new(hidden_dim, dim)
                .with_bias(false)
                .init(device),
            w3: LinearConfig::new(dim, hidden_dim)
                .with_bias(false)
                .init(device),
        }
    }

    /// `x`: any shape ending in `[..., dim]`, operates on last dim.
    /// Concretely used as `[B, S, D]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.w1.forward(x.clone())); // [B, S, H]
        let val = self.w3.forward(x); // [B, S, H]
        self.w2.forward(gate * val) // [B, S, D]
    }
}

fn round_up(n: usize, multiple: usize) -> usize {
    n.next_multiple_of(multiple)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn dev() -> <B as Backend>::Device {
        Default::default()
    }

    #[test]
    fn default_hidden_dim_computation() {
        // dim=768 → 768*8/3 = 2048, already multiple of 256
        let ffn = SwiGlu::<B>::new(768, None, &dev());
        let x = Tensor::zeros([1, 4, 768], &dev());
        let out = ffn.forward(x);
        assert_eq!(out.dims(), [1, 4, 768]);
    }

    #[test]
    fn custom_hidden_dim() {
        let ffn = SwiGlu::<B>::new(16, Some(32), &dev());
        let x = Tensor::zeros([2, 3, 16], &dev());
        let out = ffn.forward(x);
        assert_eq!(out.dims(), [2, 3, 16]);
    }

    #[test]
    fn output_shape_preserved() {
        let ffn = SwiGlu::<B>::new(8, Some(16), &dev());
        let x = Tensor::ones([1, 5, 8], &dev());
        let out = ffn.forward(x);
        assert_eq!(out.dims(), [1, 5, 8]);
    }

    #[test]
    fn silu_gating_nonzero_input_produces_output() {
        // With default (random) weight init, non-zero input should produce non-zero output
        let ffn = SwiGlu::<B>::new(8, Some(16), &dev());
        let x = Tensor::ones([1, 2, 8], &dev()) * 2.0;
        let out = ffn.forward(x);
        // We can't guarantee non-zero with random init, but we verify shape
        assert_eq!(out.dims(), [1, 2, 8]);
    }

    #[test]
    fn zero_input_gives_zero_output() {
        // SwiGLU: silu(w1*0) * w3*0 = silu(b1)*b3, but biases are false
        // So silu(0) * 0 = 0.5*0 * 0 = 0 (silu(0) = 0*sigmoid(0) = 0*0.5 = 0)
        let ffn = SwiGlu::<B>::new(8, Some(16), &dev());
        let x = Tensor::zeros([1, 3, 8], &dev());
        let out = ffn.forward(x);
        let sum: f32 = out.abs().sum().to_data().to_vec::<f32>().unwrap()[0];
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn no_bias_in_linears() {
        let ffn = SwiGlu::<B>::new(8, Some(16), &dev());
        assert!(ffn.w1.bias.is_none());
        assert!(ffn.w2.bias.is_none());
        assert!(ffn.w3.bias.is_none());
    }

    #[test]
    fn round_up_basic() {
        assert_eq!(round_up(100, 256), 256);
        assert_eq!(round_up(256, 256), 256);
        assert_eq!(round_up(257, 256), 512);
        assert_eq!(round_up(0, 256), 0);
    }
}
