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
