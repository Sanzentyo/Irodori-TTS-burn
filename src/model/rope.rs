use burn::tensor::{Tensor, backend::Backend};

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
