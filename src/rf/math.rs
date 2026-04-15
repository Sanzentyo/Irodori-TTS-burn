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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn interpolate_at_t_zero_returns_x0() {
        let device = Default::default();
        let x0 = Tensor::<B, 3>::from_data([[[1.0f32, 2.0], [3.0, 4.0]]], &device);
        let noise = Tensor::<B, 3>::from_data([[[10.0f32, 20.0], [30.0, 40.0]]], &device);
        let t = Tensor::<B, 1>::from_data([0.0f32], &device);

        let result = rf_interpolate(x0.clone(), noise, t);
        let diff: f32 = (result - x0).abs().max().into_scalar();
        assert!(diff < 1e-6, "at t=0, interpolation should return x0");
    }

    #[test]
    fn interpolate_at_t_one_returns_noise() {
        let device = Default::default();
        let x0 = Tensor::<B, 3>::from_data([[[1.0f32, 2.0], [3.0, 4.0]]], &device);
        let noise = Tensor::<B, 3>::from_data([[[10.0f32, 20.0], [30.0, 40.0]]], &device);
        let t = Tensor::<B, 1>::from_data([1.0f32], &device);

        let result = rf_interpolate(x0, noise.clone(), t);
        let diff: f32 = (result - noise).abs().max().into_scalar();
        assert!(diff < 1e-6, "at t=1, interpolation should return noise");
    }

    #[test]
    fn interpolate_at_half_returns_midpoint() {
        let device = Default::default();
        let x0 = Tensor::<B, 3>::from_data([[[0.0f32, 0.0]]], &device);
        let noise = Tensor::<B, 3>::from_data([[[2.0f32, 4.0]]], &device);
        let t = Tensor::<B, 1>::from_data([0.5f32], &device);

        let result: Vec<f32> = rf_interpolate(x0, noise, t).into_data().to_vec().unwrap();
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn interpolate_batched() {
        let device = Default::default();
        // batch=2, seq=1, dim=1
        let x0 = Tensor::<B, 3>::from_data([[[0.0f32]], [[10.0]]], &device);
        let noise = Tensor::<B, 3>::from_data([[[4.0f32]], [[20.0]]], &device);
        let t = Tensor::<B, 1>::from_data([0.25f32, 0.75], &device);

        let result: Vec<f32> = rf_interpolate(x0, noise, t).into_data().to_vec().unwrap();
        // batch 0: (1-0.25)*0 + 0.25*4 = 1.0
        assert!((result[0] - 1.0).abs() < 1e-6);
        // batch 1: (1-0.75)*10 + 0.75*20 = 2.5 + 15 = 17.5
        assert!((result[1] - 17.5).abs() < 1e-6);
    }

    #[test]
    fn velocity_target_is_noise_minus_x0() {
        let device = Default::default();
        let x0 = Tensor::<B, 3>::from_data([[[1.0f32, 2.0]]], &device);
        let noise = Tensor::<B, 3>::from_data([[[5.0f32, 8.0]]], &device);

        let v: Vec<f32> = rf_velocity_target(x0, noise).into_data().to_vec().unwrap();
        assert!((v[0] - 4.0).abs() < 1e-6);
        assert!((v[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn predict_x0_inverts_interpolate() {
        let device = Default::default();
        let x0 = Tensor::<B, 3>::from_data([[[3.0f32, 7.0]]], &device);
        let noise = Tensor::<B, 3>::from_data([[[11.0f32, 13.0]]], &device);
        let t = Tensor::<B, 1>::from_data([0.4f32], &device);

        // x_t = (1-t)*x0 + t*noise
        let x_t = rf_interpolate(x0.clone(), noise.clone(), t.clone());
        // v = noise - x0
        let v = rf_velocity_target(x0.clone(), noise);
        // x0_hat = x_t - t*v  should recover x0
        let x0_hat = rf_predict_x0(x_t, v, t);

        let diff: f32 = (x0_hat - x0).abs().max().into_scalar();
        assert!(
            diff < 1e-5,
            "predict_x0 must invert interpolate, got diff={diff}"
        );
    }

    #[test]
    fn temporal_rescale_noop_at_t_one() {
        let device = Default::default();
        let v = Tensor::<B, 3>::from_data([[[1.0f32, 2.0]]], &device);
        let x_t = Tensor::<B, 3>::from_data([[[5.0f32, 6.0]]], &device);

        let result = temporal_score_rescale(v.clone(), x_t, 1.0, 1.5, 0.5);
        let diff: f32 = (result - v).abs().max().into_scalar();
        assert_eq!(diff, 0.0, "t >= 1 must be a no-op");
    }

    #[test]
    fn temporal_rescale_identity_when_k_is_one_sigma_zero() {
        let device = Default::default();
        let v = Tensor::<B, 3>::from_data([[[2.0f32, 3.0]]], &device);
        let x_t = Tensor::<B, 3>::from_data([[[1.0f32, 1.0]]], &device);

        // When sigma=0: ratio = (snr*0 + 1) / (snr*0/k + 1) = 1/1 = 1
        // So result = (v*(1-t) + x_t) * 1 / (1-t) - x_t / (1-t)
        //           = v + x_t/(1-t) - x_t/(1-t) = v
        let result = temporal_score_rescale(v.clone(), x_t, 0.5, 1.0, 0.0);
        let diff: f32 = (result - v).abs().max().into_scalar();
        assert!(diff < 1e-5, "sigma=0 should be identity, got diff={diff}");
    }

    #[test]
    fn temporal_rescale_produces_finite_output() {
        let device = Default::default();
        let v = Tensor::<B, 3>::from_data([[[1.0f32, -1.0]]], &device);
        let x_t = Tensor::<B, 3>::from_data([[[0.5f32, 0.3]]], &device);

        let result = temporal_score_rescale(v, x_t, 0.3, 2.0, 1.0);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        for val in &data {
            assert!(val.is_finite(), "rescaled output must be finite, got {val}");
        }
    }

    #[test]
    fn temporal_rescale_noop_at_t_above_one() {
        let device = Default::default();
        let v = Tensor::<B, 3>::from_data([[[3.0f32, 4.0]]], &device);
        let x_t = Tensor::<B, 3>::from_data([[[9.0f32, 10.0]]], &device);

        let result = temporal_score_rescale(v.clone(), x_t, 1.5, 2.0, 1.0);
        let diff: f32 = (result - v).abs().max().into_scalar();
        assert_eq!(diff, 0.0, "t > 1 must also be a no-op");
    }

    #[test]
    fn temporal_rescale_identity_when_k_is_one_sigma_positive() {
        let device = Default::default();
        let v = Tensor::<B, 3>::from_data([[[2.0f32, -1.0]]], &device);
        let x_t = Tensor::<B, 3>::from_data([[[0.5f32, 0.3]]], &device);

        // When k=1: ratio = (snr*σ² + 1) / (snr*σ² + 1) = 1
        // So result = (v*(1-t) + x_t)*1/(1-t) - x_t/(1-t) = v
        let result = temporal_score_rescale(v.clone(), x_t, 0.4, 1.0, 2.5);
        let diff: f32 = (result - v).abs().max().into_scalar();
        assert!(
            diff < 1e-5,
            "k=1 should be identity for any sigma, got diff={diff}"
        );
    }

    #[test]
    fn temporal_rescale_exact_value() {
        let device = Default::default();
        // Hand-computed: t=0.5, k=2, sigma=1
        // one_minus_t = 0.5, snr = 0.25/0.25 = 1.0, sigma_sq = 1.0
        // ratio = (1*1 + 1) / (1*1/2 + 1) = 2.0 / 1.5 = 4/3
        // result = (v*0.5 + x_t) * (4/3) / 0.5 - x_t / 0.5
        //        = (v*0.5 + x_t) * (8/3) - x_t * 2
        // For v=[1], x_t=[0]: (0.5)*(8/3) - 0 = 4/3
        let v = Tensor::<B, 3>::from_data([[[1.0f32]]], &device);
        let x_t = Tensor::<B, 3>::from_data([[[0.0f32]]], &device);

        let result: Vec<f32> = temporal_score_rescale(v, x_t, 0.5, 2.0, 1.0)
            .into_data()
            .to_vec()
            .unwrap();
        let expected = 4.0_f32 / 3.0;
        assert!(
            (result[0] - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            result[0]
        );
    }
}
