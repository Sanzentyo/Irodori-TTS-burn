//! Core rectified-flow math operations.
//!
//! These are pure tensor functions with no model dependencies.

use burn::tensor::{Tensor, backend::Backend};

/// Straight-line interpolation: `x_t = (1 - t) * x0 + t * noise`.
///
/// `t: [B]`, `x0 / noise: [B, S, D]`.
pub fn rf_interpolate<B: Backend>(
    x0: Tensor<B, 3>,
    noise: Tensor<B, 3>,
    t: Tensor<B, 1>,
) -> Tensor<B, 3> {
    let t3 = t.unsqueeze_dim::<2>(1).unsqueeze_dim::<3>(2); // [B, 1, 1]
    (Tensor::ones_like(&t3) - t3.clone()) * x0 + t3 * noise
}

/// RF velocity target: `v = noise - x0`.
pub fn rf_velocity_target<B: Backend>(x0: Tensor<B, 3>, noise: Tensor<B, 3>) -> Tensor<B, 3> {
    noise - x0
}

/// Recover clean sample from noisy + predicted velocity:
/// `x0 = x_t - t * v_pred`.
pub fn rf_predict_x0<B: Backend>(
    x_t: Tensor<B, 3>,
    v_pred: Tensor<B, 3>,
    t: Tensor<B, 1>,
) -> Tensor<B, 3> {
    let t3 = t.unsqueeze_dim::<2>(1).unsqueeze_dim::<3>(2); // [B, 1, 1]
    x_t - t3 * v_pred
}

/// Temporal score rescaling (arxiv:2510.01184).
///
/// No-op when `t >= 1`.
pub fn temporal_score_rescale<B: Backend>(
    v_pred: Tensor<B, 3>,
    x_t: Tensor<B, 3>,
    t: f32,
    rescale_k: f32,
    rescale_sigma: f32,
) -> Tensor<B, 3> {
    if t >= 1.0 {
        return v_pred;
    }
    let one_minus_t = 1.0 - t;
    let snr = (one_minus_t * one_minus_t) / (t * t);
    let sigma_sq = rescale_sigma * rescale_sigma;
    let ratio = (snr * sigma_sq + 1.0) / (snr * sigma_sq / rescale_k + 1.0);
    (v_pred * one_minus_t + x_t.clone()) * ratio / one_minus_t - x_t / one_minus_t
}
