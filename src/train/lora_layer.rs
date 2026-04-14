//! Low-rank adapter linear layer.
//!
//! Wraps a frozen pretrained [`Linear`] with trainable `lora_a` and `lora_b`
//! matrices.  The output is:
//!
//! ```text
//! base(x) + (x_flat @ lora_a.T @ lora_b.T) * scale
//! ```
//!
//! where `scale = alpha / r`.

use burn::{
    module::{Module, Param},
    nn::Linear,
    tensor::{Distribution, Tensor, backend::Backend},
};

/// Configuration for a LoRA-augmented linear projection.
#[derive(Debug, Clone)]
pub struct LoraLinearConfig {
    pub in_features: usize,
    pub out_features: usize,
    /// Low-rank dimension.
    pub r: usize,
    /// LoRA alpha (scale = alpha / r).
    pub alpha: f32,
}

/// Linear layer with a frozen pretrained base and trainable low-rank delta.
///
/// - `lora_a: [r, in_features]` — Kaiming-normal initialised.
/// - `lora_b: [out_features, r]` — zero initialised.
/// - `base` params: `require_grad = false` (frozen).
/// - `lora_a`, `lora_b`: `require_grad = true` (trainable).
///
/// When `lora_b` is all zeros (initial state), `forward` is numerically
/// identical to `base.forward`.
#[derive(Module, Debug)]
pub struct LoraLinear<B: Backend> {
    /// Frozen pretrained projection.
    pub(crate) base: Linear<B>,
    /// Down-projection: `[r, in_features]`.
    pub(crate) lora_a: Param<Tensor<B, 2>>,
    /// Up-projection: `[out_features, r]`.
    pub(crate) lora_b: Param<Tensor<B, 2>>,
    /// `alpha / r` — not serialised; recomputed at construction.
    scale: f32,
    /// Output feature count — not serialised; recomputed at construction.
    out_features: usize,
}

impl<B: Backend> LoraLinear<B> {
    /// Construct from a pretrained base linear.
    ///
    /// The `base` weights are immediately frozen via [`Module::no_grad`].
    /// Fresh LoRA params are Kaiming-normal (`lora_a`) and zero (`lora_b`).
    pub fn from_base(base: Linear<B>, cfg: &LoraLinearConfig, device: &B::Device) -> Self {
        let scale = cfg.alpha / cfg.r as f32;
        // Kaiming normal: std = sqrt(2 / fan_in)
        let std = (2.0_f64 / cfg.in_features as f64).sqrt();
        let lora_a = Param::from_tensor(Tensor::random(
            [cfg.r, cfg.in_features],
            Distribution::Normal(0.0, std),
            device,
        ));
        let lora_b = Param::from_tensor(Tensor::<B, 2>::zeros([cfg.out_features, cfg.r], device));
        Self {
            base: base.no_grad(),
            lora_a,
            lora_b,
            scale,
            out_features: cfg.out_features,
        }
    }

    /// Re-freeze the base weights.
    ///
    /// Must be called **after** `load_record` to ensure pretrained weights
    /// remain frozen despite the record reload.
    pub fn freeze_base(self) -> Self {
        Self {
            base: self.base.no_grad(),
            lora_a: self.lora_a,
            lora_b: self.lora_b,
            scale: self.scale,
            out_features: self.out_features,
        }
    }

    /// Forward: `base(x) + delta * scale`.
    ///
    /// Accepts `[B, S, in_features]`, returns `[B, S, out_features]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, _] = x.dims();
        let base_out = self.base.forward(x.clone());
        let delta = x
            .flatten(0, 1) // [B*S, in]
            .matmul(self.lora_a.val().transpose()) // [B*S, r]
            .matmul(self.lora_b.val().transpose()) // [B*S, out]
            .reshape([batch, seq, self.out_features])
            * self.scale;
        base_out + delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::nn::LinearConfig;

    type TestBackend = NdArray;

    fn make_lora(in_f: usize, out_f: usize, r: usize, alpha: f32) -> LoraLinear<TestBackend> {
        let device = Default::default();
        let base = LinearConfig::new(in_f, out_f)
            .with_bias(false)
            .init(&device);
        LoraLinear::from_base(
            base,
            &LoraLinearConfig {
                in_features: in_f,
                out_features: out_f,
                r,
                alpha,
            },
            &device,
        )
    }

    #[test]
    fn initial_lora_output_matches_base() {
        let device = Default::default();
        let lora = make_lora(8, 4, 2, 1.0);

        let x = Tensor::<TestBackend, 3>::ones([1, 3, 8], &device);
        let lora_out = lora.forward(x.clone());
        let base_out = lora.base.forward(x);

        let lora_data: Vec<f32> = lora_out.into_data().to_vec().unwrap();
        let base_data: Vec<f32> = base_out.into_data().to_vec().unwrap();
        for (a, b) in lora_data.iter().zip(base_data.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "with zero lora_b, output must match base"
            );
        }
    }

    #[test]
    fn forward_shape() {
        let lora = make_lora(8, 4, 2, 1.0);
        let device = Default::default();
        let x = Tensor::<TestBackend, 3>::ones([2, 5, 8], &device);
        assert_eq!(lora.forward(x).dims(), [2, 5, 4]);
    }

    #[test]
    fn nonzero_lora_b_changes_output() {
        let device = Default::default();
        let mut lora = make_lora(8, 4, 2, 1.0);
        let x = Tensor::<TestBackend, 3>::ones([1, 1, 8], &device);
        let base_out: Vec<f32> = lora.base.forward(x.clone()).into_data().to_vec().unwrap();

        // Make lora_b nonzero → delta is nonzero
        lora.lora_b = Param::from_tensor(Tensor::<TestBackend, 2>::ones([4, 2], &device));
        let lora_out: Vec<f32> = lora.forward(x).into_data().to_vec().unwrap();

        assert!(
            base_out
                .iter()
                .zip(lora_out.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "nonzero lora_b must produce different output"
        );
    }

    #[test]
    fn scale_is_alpha_over_r() {
        let lora = make_lora(8, 4, 4, 8.0);
        assert!((lora.scale - 2.0).abs() < 1e-6, "scale = alpha/r = 8/4 = 2");
    }
}
