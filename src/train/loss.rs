//! RF flow-matching training loss.
//!
//! Implements:
//! - Logit-normal timestep sampling (`sample_logit_normal_t`)
//! - RF interpolation (`rf_interpolate`)
//! - RF velocity target (`rf_velocity_target`)
//! - Echo-style masked MSE loss (`echo_style_masked_mse`)

use burn::tensor::{Bool, Tensor, backend::Backend};

/// Sample a batch of timesteps from a logit-normal distribution.
///
/// `t ~ sigmoid(N(mean, std))`, clamped to `[t_min, t_max]`.
///
/// This matches the Python reference's `logit_normal_sample` used in
/// `IrodoriTTSTrainer`.
pub fn sample_logit_normal_t(
    batch_size: usize,
    mean: f32,
    std: f32,
    t_min: f32,
    t_max: f32,
) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..batch_size)
        .map(|_| {
            // Box-Muller: z ~ N(mean, std)
            let u1: f64 = rng.gen_range(1e-10_f64..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            let z = mean as f64
                + std as f64 * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            // sigmoid(z)
            let t = (1.0 / (1.0 + (-z).exp())) as f32;
            t.clamp(t_min, t_max)
        })
        .collect()
}

/// RF interpolation: `x_t = (1 - t) * x0 + t * noise`.
///
/// - `x0 / noise: [B, S, D]`
/// - `t: &[f32]` of length `B`
///
/// Returns `[B, S, D]`.
pub fn rf_interpolate<B: Backend>(
    x0: Tensor<B, 3>,
    noise: Tensor<B, 3>,
    t: &[f32],
    device: &B::Device,
) -> Tensor<B, 3> {
    let batch = x0.dims()[0];
    debug_assert_eq!(t.len(), batch);
    let t_tensor = Tensor::<B, 1>::from_floats(t, device).reshape([batch, 1, 1]); // [B, 1, 1]
    let one_minus_t = -t_tensor.clone() + 1.0_f32;
    x0 * one_minus_t + noise * t_tensor
}

/// RF velocity target: `v = noise - x0`.
pub fn rf_velocity_target<B: Backend>(noise: Tensor<B, 3>, x0: Tensor<B, 3>) -> Tensor<B, 3> {
    noise - x0
}

/// Echo-style masked MSE loss.
///
/// ```text
/// loss = Σ_valid ((pred - target)² mean over D) / Σ_valid + ε
/// ```
///
/// - `pred / target: [B, S, D]`
/// - `loss_mask: [B, S]` — `true` = token contributes to loss
///
/// Returns a scalar `[1]` tensor.
pub fn echo_style_masked_mse<B: Backend>(
    pred: Tensor<B, 3>,
    target: Tensor<B, 3>,
    loss_mask: Tensor<B, 2, Bool>,
) -> Tensor<B, 1> {
    // Mean squared error per token: [B, S, 1] (mean_dim preserves rank)
    let per_token = (pred - target).powf_scalar(2.0f32).mean_dim(2);
    let mask_f = loss_mask.float().unsqueeze_dim::<3>(2); // [B, S, 1]
    let weighted = per_token * mask_f.clone();
    let total = weighted.sum();
    let denom = mask_f.sum() + 1e-8f32;
    (total / denom).reshape([1])
}
