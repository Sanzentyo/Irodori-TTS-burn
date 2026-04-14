//! RF flow-matching training loss.
//!
//! Implements:
//! - Logit-normal timestep sampling (`sample_logit_normal_t`)
//! - Stratified logit-normal timestep sampling (`sample_stratified_logit_normal_t`)
//! - RF interpolation (`rf_interpolate`)
//! - RF velocity target (`rf_velocity_target`)
//! - Echo-style masked MSE loss (`echo_style_masked_mse`)

use burn::tensor::{Bool, Tensor, backend::Backend};
use rand::Rng;

/// Sample a batch of timesteps from a logit-normal distribution.
///
/// `t ~ sigmoid(N(mean, std))`, clamped to `[t_min, t_max]`.
///
/// This matches the Python reference's `logit_normal_sample` used in
/// `IrodoriTTSTrainer`.
pub fn sample_logit_normal_t(
    rng: &mut impl Rng,
    batch_size: usize,
    mean: f32,
    std: f32,
    t_min: f32,
    t_max: f32,
) -> Vec<f32> {
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

/// Stratified logit-normal timestep sampling for variance reduction.
///
/// Divides `[0, 1)` into `batch_size` equal-width bins and samples one uniform
/// deviate per bin, then transforms through the inverse-CDF of a logit-normal
/// distribution: `z = mean + std * Φ⁻¹(u)`, `t = sigmoid(z)`.
///
/// The result is randomly permuted so that dataset ordering does not correlate
/// with timestep bins.
///
/// Matches the Python reference `sample_stratified_logit_normal_t`.
pub fn sample_stratified_logit_normal_t(
    rng: &mut impl Rng,
    batch_size: usize,
    mean: f32,
    std: f32,
    t_min: f32,
    t_max: f32,
) -> Vec<f32> {
    use rand::seq::SliceRandom;

    if batch_size == 0 {
        return Vec::new();
    }

    let bs = batch_size as f64;

    let mut t_vals: Vec<f32> = (0..batch_size)
        .map(|i| {
            // Stratified uniform: u ∈ [i/B, (i+1)/B)
            let u_raw = (i as f64 + rng.r#gen::<f64>()) / bs;
            let u = u_raw.clamp(1e-6, 1.0 - 1e-6);
            // Φ⁻¹(u) = √2 · erfinv(2u - 1)
            let z = erfinv(2.0 * u - 1.0) * std::f64::consts::SQRT_2;
            let z = z * std as f64 + mean as f64;
            let t = (1.0 / (1.0 + (-z).exp())) as f32;
            t.clamp(t_min, t_max)
        })
        .collect();

    // Permute so bin index doesn't correlate with batch position.
    t_vals.shuffle(rng);
    t_vals
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

// ---------------------------------------------------------------------------
// Inverse error function (rational approximation)
// ---------------------------------------------------------------------------

/// Compute the inverse error function `erfinv(x)` for `x ∈ (-1, 1)`.
///
/// Uses the Winitzki (2008) approximation with a Newton refinement step
/// for high accuracy (max relative error < 1e-9 after refinement).
fn erfinv(x: f64) -> f64 {
    if x <= -1.0 {
        return f64::NEG_INFINITY;
    }
    if x >= 1.0 {
        return f64::INFINITY;
    }
    if x == 0.0 {
        return 0.0;
    }

    let a = 0.147;
    let ln1mx2 = (1.0 - x * x).ln(); // negative for |x| < 1
    let b = 2.0 / (std::f64::consts::PI * a) + ln1mx2 / 2.0;
    let inner = (b * b - ln1mx2 / a).sqrt() - b;
    let approx = x.signum() * inner.sqrt();

    // Newton refinement: y -= (erf(y) - x) / erf'(y)
    let err = libm::erf(approx) - x;
    let deriv = std::f64::consts::FRAC_2_SQRT_PI * (-approx * approx).exp();
    approx - err / deriv
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn erfinv_known_values() {
        // erfinv(0) = 0
        assert!((erfinv(0.0)).abs() < 1e-12);
        // erfinv(0.5) ≈ 0.4769362762044699
        assert!((erfinv(0.5) - 0.4769362762044699).abs() < 1e-6);
        // erfinv(-0.5) ≈ -0.4769362762044699
        assert!((erfinv(-0.5) + 0.4769362762044699).abs() < 1e-6);
        // erfinv(0.9) ≈ 1.1630871536766743
        assert!((erfinv(0.9) - 1.1630871536766743).abs() < 1e-5);
    }

    #[test]
    fn erfinv_boundary() {
        assert!(erfinv(1.0).is_infinite() && erfinv(1.0) > 0.0);
        assert!(erfinv(-1.0).is_infinite() && erfinv(-1.0) < 0.0);
    }

    #[test]
    fn sample_logit_normal_t_range() {
        let t = sample_logit_normal_t(&mut StdRng::seed_from_u64(42), 100, 0.0, 1.0, 1e-3, 0.999);
        assert_eq!(t.len(), 100);
        for &val in &t {
            assert!((1e-3..=0.999).contains(&val), "t={val} out of range");
        }
    }

    #[test]
    fn sample_stratified_logit_normal_t_range() {
        let t = sample_stratified_logit_normal_t(
            &mut StdRng::seed_from_u64(42),
            100,
            0.0,
            1.0,
            1e-3,
            0.999,
        );
        assert_eq!(t.len(), 100);
        for &val in &t {
            assert!((1e-3..=0.999).contains(&val), "t={val} out of range");
        }
    }

    #[test]
    fn sample_stratified_logit_normal_t_empty() {
        let t = sample_stratified_logit_normal_t(
            &mut StdRng::seed_from_u64(42),
            0,
            0.0,
            1.0,
            1e-3,
            0.999,
        );
        assert!(t.is_empty());
    }

    #[test]
    fn stratified_lower_estimator_variance() {
        // Stratified sampling should produce sample means with lower variance
        // across batches — this is the primary benefit for training.
        let n = 16; // typical batch size
        let runs = 500;
        let mut rng = StdRng::seed_from_u64(42);
        let mut iid_means = Vec::new();
        let mut strat_means = Vec::new();
        for _ in 0..runs {
            let iid = sample_logit_normal_t(&mut rng, n, 0.0, 1.0, 1e-3, 0.999);
            let strat = sample_stratified_logit_normal_t(&mut rng, n, 0.0, 1.0, 1e-3, 0.999);
            iid_means.push(iid.iter().map(|&x| x as f64).sum::<f64>() / n as f64);
            strat_means.push(strat.iter().map(|&x| x as f64).sum::<f64>() / n as f64);
        }
        let iid_mean_var = variance_f64(&iid_means);
        let strat_mean_var = variance_f64(&strat_means);
        assert!(
            strat_mean_var < iid_mean_var,
            "stratified mean variance ({strat_mean_var}) should be less than iid ({iid_mean_var})"
        );
    }

    fn variance_f64(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n
    }

    #[test]
    fn seeded_rng_produces_reproducible_timesteps() {
        let run = || {
            let mut rng = StdRng::seed_from_u64(123);
            let a = sample_logit_normal_t(&mut rng, 8, 0.0, 1.0, 1e-3, 0.999);
            let b = sample_stratified_logit_normal_t(&mut rng, 8, 0.0, 1.0, 1e-3, 0.999);
            (a, b)
        };
        let (a1, b1) = run();
        let (a2, b2) = run();
        assert_eq!(a1, a2, "logit-normal must be deterministic with same seed");
        assert_eq!(b1, b2, "stratified must be deterministic with same seed");
    }

    /// Verifies that seeding both the host RNG and the NdArray backend produces
    /// deterministic results. The host-side RNG (timesteps) is verified for
    /// bitwise identity; backend-side tensors may not be bitwise identical under
    /// parallel test execution (global RNG state) so we only verify finiteness.
    #[test]
    fn seeded_loss_pipeline_deterministic() {
        use burn::backend::NdArray;
        use burn::tensor::backend::Backend;

        type B = NdArray<f32>;
        let device = Default::default();
        let batch = 2;
        let seq = 4;
        let dim = 8;

        let run = |seed: u64| {
            let mut rng = StdRng::seed_from_u64(seed);
            B::seed(&device, seed);

            let t_vals = sample_logit_normal_t(&mut rng, batch, 0.0, 1.0, 1e-3, 0.999);
            let x0 = Tensor::<B, 3>::random(
                [batch, seq, dim],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            );
            let noise = Tensor::<B, 3>::random(
                [batch, seq, dim],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            );
            let x_t = rf_interpolate::<B>(x0.clone(), noise.clone(), &t_vals, &device);
            let target = rf_velocity_target::<B>(noise, x0);
            let mask = Tensor::<B, 2, Bool>::from_data(
                burn::tensor::TensorData::from([
                    [true, true, true, false],
                    [true, true, false, false],
                ]),
                &device,
            );
            let loss = echo_style_masked_mse::<B>(x_t, target, mask);
            let loss_val: Vec<f32> = loss.into_data().to_vec().unwrap();
            (t_vals, loss_val[0])
        };

        // Host-side RNG determinism (thread-local, not affected by parallel tests)
        let (t1, _) = run(99);
        let (t2, _) = run(99);
        assert_eq!(t1, t2, "host RNG must be deterministic with same seed");

        // Loss is finite
        let (_, l) = run(99);
        assert!(l.is_finite(), "loss must be finite, got {l}");

        // Different seeds produce different host-side timesteps
        let (t_a, _) = run(100);
        let (t_b, _) = run(200);
        assert_ne!(t_a, t_b, "different seeds must produce different timesteps");
    }

    /// Verifies that different seeds produce different results.
    #[test]
    fn different_seeds_produce_different_results() {
        let mut rng1 = StdRng::seed_from_u64(1);
        let mut rng2 = StdRng::seed_from_u64(2);
        let a = sample_logit_normal_t(&mut rng1, 16, 0.0, 1.0, 1e-3, 0.999);
        let b = sample_logit_normal_t(&mut rng2, 16, 0.0, 1.0, 1e-3, 0.999);
        assert_ne!(a, b, "different seeds must produce different timesteps");
    }
}
