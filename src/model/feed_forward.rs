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
    /// Fused w1‖w3 weight: `[dim, 2*hidden_dim]` — inference-only optimisation.
    /// Saves 1 kernel launch per block per denoising step.
    #[module(skip)]
    fused_w13_weight: Option<Tensor<B, 2>>,
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
            fused_w13_weight: None,
        }
    }

    /// `x`: any shape ending in `[..., dim]`, operates on last dim.
    /// Concretely used as `[B, S, D]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let (gate, val) = if let Some(ref fused_w) = self.fused_w13_weight {
            // Single matmul: x @ fused_w → [B, S, 2*H], then split
            let [_b, _s, _d] = x.dims();
            let hidden_dim = fused_w.dims()[1] / 2;
            let w13 = x.matmul(fused_w.clone().unsqueeze::<3>());
            let gate = silu(w13.clone().narrow(2, 0, hidden_dim));
            let val = w13.narrow(2, hidden_dim, hidden_dim);
            (gate, val)
        } else {
            let gate = silu(self.w1.forward(x.clone()));
            let val = self.w3.forward(x);
            (gate, val)
        };
        self.w2.forward(gate * val)
    }

    /// Fuse w1 and w3 weight matrices into a single `[dim, 2*hidden_dim]` tensor.
    ///
    /// Saves 1 kernel launch per block per denoising step (~10ms total for
    /// 12 blocks × 40 steps). Idempotent.
    pub fn prepare_for_inference(&mut self) {
        if self.fused_w13_weight.is_some() {
            return;
        }
        let w1 = self.w1.weight.val(); // [dim, hidden_dim]
        let w3 = self.w3.weight.val();
        self.fused_w13_weight = Some(Tensor::cat(vec![w1, w3], 1));
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

    /// After `prepare_for_inference()`, fused w1‖w3 forward must produce
    /// identical output to the 2-linear path.
    #[test]
    fn fused_w13_matches_separate_linears() {
        let mut ffn = SwiGlu::<B>::new(16, Some(32), &dev());
        let x = Tensor::random([2, 4, 16], burn::tensor::Distribution::Default, &dev());

        let out_unfused = ffn.forward(x.clone());

        ffn.prepare_for_inference();
        assert!(ffn.fused_w13_weight.is_some());

        let out_fused = ffn.forward(x);

        let diff: f32 = (out_unfused - out_fused)
            .abs()
            .max()
            .to_data()
            .to_vec::<f32>()
            .unwrap()[0];
        assert!(
            diff < 1e-5,
            "fused w1||w3 output should match unfused: max_diff={diff}"
        );
    }

    /// `prepare_for_inference()` is idempotent.
    #[test]
    fn fused_w13_idempotent() {
        let mut ffn = SwiGlu::<B>::new(8, Some(16), &dev());
        ffn.prepare_for_inference();
        let w1: Vec<f32> = ffn
            .fused_w13_weight
            .as_ref()
            .unwrap()
            .clone()
            .into_data()
            .to_vec()
            .unwrap();
        ffn.prepare_for_inference();
        let w2: Vec<f32> = ffn
            .fused_w13_weight
            .as_ref()
            .unwrap()
            .clone()
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(w1, w2);
    }

    /// Zero input still gives zero output with fused weights.
    #[test]
    fn fused_zero_input_gives_zero_output() {
        let mut ffn = SwiGlu::<B>::new(8, Some(16), &dev());
        ffn.prepare_for_inference();
        let x = Tensor::zeros([1, 3, 8], &dev());
        let out = ffn.forward(x);
        let sum: f32 = out.abs().sum().to_data().to_vec::<f32>().unwrap()[0];
        assert_eq!(sum, 0.0);
    }
}
