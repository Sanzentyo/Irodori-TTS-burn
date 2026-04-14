use burn::tensor::{Tensor, backend::Backend};

/// Precomputed RoPE frequency tables for a given sequence length.
///
/// Both `cos` and `sin` have shape `[seq_len, head_dim / 2]`.
/// Intended to be precomputed once per sampling trajectory and reused
/// across all denoising steps and blocks.
#[derive(Clone, Debug)]
pub struct RopeFreqs<B: Backend> {
    pub cos: Tensor<B, 2>,
    pub sin: Tensor<B, 2>,
}

impl<B: Backend> RopeFreqs<B> {
    pub fn new(cos: Tensor<B, 2>, sin: Tensor<B, 2>) -> Self {
        Self { cos, sin }
    }
}

/// Precompute RoPE (cos, sin) frequency tables.
///
/// Returns `(cos, sin)` each of shape `[seq_len, head_dim / 2]`.
/// The Python reference uses complex-valued tensors; we store cos/sin separately.
pub fn precompute_rope_freqs<B: Backend>(
    head_dim: usize,
    seq_len: usize,
    theta: f64,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    assert!(
        head_dim.is_multiple_of(2),
        "head_dim must be even for RoPE, got {head_dim}"
    );
    let half = head_dim / 2;

    // freqs[i] = 1 / (theta ^ (2i / head_dim))
    let freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();

    let t: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();

    // outer product: [seq_len, half]
    let mut cos_data = vec![0.0f32; seq_len * half];
    let mut sin_data = vec![0.0f32; seq_len * half];
    for (si, &ti) in t.iter().enumerate() {
        for (fi, &freq) in freqs.iter().enumerate() {
            let angle = ti * freq;
            cos_data[si * half + fi] = angle.cos();
            sin_data[si * half + fi] = angle.sin();
        }
    }

    let cos = Tensor::from_floats(
        burn::tensor::TensorData::new(cos_data, [seq_len, half]),
        device,
    );
    let sin = Tensor::from_floats(
        burn::tensor::TensorData::new(sin_data, [seq_len, half]),
        device,
    );
    (cos, sin)
}

/// Precompute RoPE frequency tables, returning a [`RopeFreqs`] struct.
///
/// Convenience wrapper around [`precompute_rope_freqs`] for the cached hot path.
pub fn precompute_rope_freqs_typed<B: Backend>(
    head_dim: usize,
    seq_len: usize,
    theta: f64,
    device: &B::Device,
) -> RopeFreqs<B> {
    let (cos, sin) = precompute_rope_freqs(head_dim, seq_len, theta, device);
    RopeFreqs::new(cos, sin)
}

/// Apply RoPE to a `[batch, seq, heads, head_dim]` tensor.
///
/// Mimics the Python `apply_rotary_emb` which uses complex multiplication:
/// each pair `(x[2k], x[2k+1])` is rotated by `(cos[k], sin[k])`.
pub fn apply_rotary_emb<B: Backend>(
    x: Tensor<B, 4>,
    cos: Tensor<B, 2>, // [seq, half]
    sin: Tensor<B, 2>, // [seq, half]
) -> Tensor<B, 4> {
    let [batch, seq, heads, head_dim] = x.dims();
    let half = head_dim / 2;

    // Reshape to pair adjacent elements: [batch, seq, heads, half, 2]
    let x5: Tensor<B, 5> = x.reshape([batch, seq, heads, half, 2]);

    // Real (even) and imaginary (odd) parts: [batch, seq, heads, half]
    let x_re: Tensor<B, 4> = x5
        .clone()
        .slice([0..batch, 0..seq, 0..heads, 0..half, 0..1])
        .reshape([batch, seq, heads, half]);
    let x_im: Tensor<B, 4> = x5
        .slice([0..batch, 0..seq, 0..heads, 0..half, 1..2])
        .reshape([batch, seq, heads, half]);

    // cos/sin: [seq, half] → [1, seq, 1, half]
    let cos4: Tensor<B, 4> = cos.reshape([1, seq, 1, half]);
    let sin4: Tensor<B, 4> = sin.reshape([1, seq, 1, half]);

    // Complex multiplication: (re + i·im) · (cos + i·sin)
    let out_re = x_re.clone() * cos4.clone() - x_im.clone() * sin4.clone();
    let out_im = x_re * sin4 + x_im * cos4;

    // Interleave back: stack as [batch, seq, heads, half, 2] then flatten
    let out5: Tensor<B, 5> = Tensor::stack(vec![out_re, out_im], 4);
    out5.reshape([batch, seq, heads, head_dim])
}

/// Half-RoPE variant used in `JointAttention`.
///
/// Splits heads into two halves, applies RoPE to the first half only.
/// x: `[batch, seq, heads, head_dim]` where split is along the `heads` axis.
pub fn apply_rotary_half<B: Backend>(
    x: Tensor<B, 4>,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
) -> Tensor<B, 4> {
    let [batch, seq, heads, head_dim] = x.dims();
    let half_heads = heads / 2;

    // Split along heads dimension (dim 2)
    let x_rot: Tensor<B, 4> = x
        .clone()
        .slice([0..batch, 0..seq, 0..half_heads, 0..head_dim]);
    let x_pass: Tensor<B, 4> = x.slice([0..batch, 0..seq, half_heads..heads, 0..head_dim]);

    let x_rot_out = apply_rotary_emb(x_rot, cos, sin);
    Tensor::cat(vec![x_rot_out, x_pass], 2)
}

/// Generate sinusoidal timestep embeddings.
///
/// `timestep`: `[batch]` — values in [0, 1]  
/// Returns: `[batch, dim]`
pub fn get_timestep_embedding<B: Backend>(
    timestep: Tensor<B, 1>,
    dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    assert!(dim.is_multiple_of(2), "timestep embedding dim must be even");
    let half = dim / 2;
    let _batch = timestep.dims()[0];

    // freqs = 1000 * exp(-log(10000) * arange(half) / half)
    let log_10000 = (10000.0_f32).ln();
    let freqs_data: Vec<f32> = (0..half)
        .map(|i| 1000.0 * ((-log_10000 * i as f32) / half as f32).exp())
        .collect();

    let freqs: Tensor<B, 2> =
        Tensor::from_floats(burn::tensor::TensorData::new(freqs_data, [1, half]), device);

    // args = timestep[:, None] * freqs[None, :]  →  [batch, half]
    let t2: Tensor<B, 2> = timestep.unsqueeze_dim::<2>(1); // [batch, 1]
    let args = t2 * freqs; // broadcast → [batch, half]

    // [cos(args), sin(args)] along last dim
    Tensor::cat(vec![args.clone().cos(), args.sin()], 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn precompute_rope_freqs_shape() {
        let device = Default::default();
        let (cos, sin) = precompute_rope_freqs::<B>(8, 16, 10000.0, &device);
        assert_eq!(cos.dims(), [16, 4]); // [seq_len, head_dim/2]
        assert_eq!(sin.dims(), [16, 4]);
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let device = Default::default();
        let (cos, sin) = precompute_rope_freqs::<B>(4, 1, 10000.0, &device);
        // At position 0, angle=0 → cos=1, sin=0
        let cos_data: Vec<f32> = cos.into_data().to_vec().unwrap();
        let sin_data: Vec<f32> = sin.into_data().to_vec().unwrap();
        for v in &cos_data {
            assert!((v - 1.0).abs() < 1e-6, "cos(0) should be 1.0, got {v}");
        }
        for v in &sin_data {
            assert!(v.abs() < 1e-6, "sin(0) should be 0.0, got {v}");
        }
    }

    #[test]
    fn apply_rotary_emb_preserves_shape() {
        let device = Default::default();
        let (cos, sin) = precompute_rope_freqs::<B>(8, 4, 10000.0, &device);
        let x = Tensor::<B, 4>::ones([2, 4, 3, 8], &device);
        let y = apply_rotary_emb(x, cos, sin);
        assert_eq!(y.dims(), [2, 4, 3, 8]);
    }

    #[test]
    fn apply_rotary_emb_position_zero_identity() {
        let device = Default::default();
        let (cos, sin) = precompute_rope_freqs::<B>(4, 1, 10000.0, &device);
        // At pos 0: cos=1, sin=0 → rotation is identity
        let x = Tensor::<B, 4>::from_floats([[[[1.0, 2.0, 3.0, 4.0]]]], &device);
        let y = apply_rotary_emb(x.clone(), cos, sin);
        let x_data: Vec<f32> = x.into_data().to_vec().unwrap();
        let y_data: Vec<f32> = y.into_data().to_vec().unwrap();
        for (a, b) in x_data.iter().zip(y_data.iter()) {
            assert!((a - b).abs() < 1e-6, "identity rotation failed: {a} vs {b}");
        }
    }

    #[test]
    fn apply_rotary_half_preserves_shape() {
        let device = Default::default();
        let (cos, sin) = precompute_rope_freqs::<B>(8, 4, 10000.0, &device);
        let x = Tensor::<B, 4>::ones([2, 4, 6, 8], &device); // 6 heads
        let y = apply_rotary_half(x, cos, sin);
        assert_eq!(y.dims(), [2, 4, 6, 8]);
    }

    #[test]
    fn apply_rotary_half_second_half_unchanged() {
        let device = Default::default();
        let (cos, sin) = precompute_rope_freqs::<B>(4, 2, 10000.0, &device);
        let x = Tensor::<B, 4>::ones([1, 2, 4, 4], &device) * 5.0; // 4 heads
        let y = apply_rotary_half(x, cos, sin);
        // The second half of heads (heads 2,3) should be unchanged
        let pass: Tensor<B, 4> = y.slice([0..1, 0..2, 2..4, 0..4]);
        let data: Vec<f32> = pass.into_data().to_vec().unwrap();
        for v in &data {
            assert!((v - 5.0).abs() < 1e-6, "passthrough heads changed: {v}");
        }
    }

    #[test]
    fn timestep_embedding_shape() {
        let device = Default::default();
        let t = Tensor::<B, 1>::from_floats([0.0, 0.5, 1.0], &device);
        let emb = get_timestep_embedding(t, 16, &device);
        assert_eq!(emb.dims(), [3, 16]);
    }

    #[test]
    fn timestep_embedding_different_for_different_times() {
        let device = Default::default();
        let t0 = Tensor::<B, 1>::from_floats([0.0], &device);
        let t1 = Tensor::<B, 1>::from_floats([1.0], &device);
        let e0 = get_timestep_embedding(t0, 8, &device);
        let e1 = get_timestep_embedding(t1, 8, &device);
        let diff = (e0 - e1).abs().sum().into_scalar();
        assert!(
            diff > 1e-3,
            "different timesteps should produce different embeddings"
        );
    }
}
