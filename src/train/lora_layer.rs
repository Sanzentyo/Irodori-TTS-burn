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
