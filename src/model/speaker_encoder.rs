use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Bool, Tensor, backend::Backend},
};

use crate::config::ModelConfig;

use super::{
    rope::precompute_rope_freqs,
    text_encoder::{TextBlock, bool_mask_to_float},
};

// ---------------------------------------------------------------------------
// patch_sequence_with_mask
// ---------------------------------------------------------------------------

/// Patch a sequence along the time axis.
///
/// - `seq: [B, S, D]` → `[B, S//patch, D*patch]`
/// - `mask: [B, S]` → `[B, S//patch]` (true iff all positions in window are valid)
///
/// If `patch_size == 1`, returns the inputs unchanged.
pub fn patch_sequence_with_mask<B: Backend>(
    seq: Tensor<B, 3>,
    mask: Tensor<B, 2, Bool>,
    patch_size: usize,
) -> crate::error::Result<(Tensor<B, 3>, Tensor<B, 2, Bool>)> {
    if patch_size <= 1 {
        return Ok((seq, mask));
    }
    let [batch, seq_len, dim] = seq.dims();
    let usable = (seq_len / patch_size) * patch_size;
    if usable == 0 {
        return Err(crate::error::IrodoriError::Shape(format!(
            "Sequence too short for speaker_patch_size={patch_size}: seq_len={seq_len}"
        )));
    }

    // Truncate to usable length, then reshape
    let seq_trunc = seq.slice([0..batch, 0..usable, 0..dim]);
    let mask_trunc = mask.slice([0..batch, 0..usable]);

    let seq_patched = seq_trunc.reshape([batch, usable / patch_size, dim * patch_size]);

    // mask: [B, usable] → [B, S//patch, patch] → all(dim=2)
    // In burn, "all" = product > 0, or minimum equals 1
    // We use: min over patch window; if min is 1 → all valid
    // Convert Bool to Int, reshape, take min, convert back to Bool
    let mask_int = bool_mask_to_int(mask_trunc.clone());
    // reshape to [B, S//patch, patch]
    let mask_int_3d = mask_int.reshape([batch, usable / patch_size, patch_size]);
    // min over last dim
    let mask_min = mask_int_3d.min_dim(2).reshape([batch, usable / patch_size]);
    let mask_patched: Tensor<B, 2, Bool> = mask_min.greater_elem(0);

    Ok((seq_patched, mask_patched))
}

/// Unpatchify a patched latent sequence.
///
/// Reverses the reshape done by [`patch_sequence_with_mask`] (for the sequence
/// dimension only — no mask involved):
///
/// - `patched: [B, S_pat, latent_dim * patch_size]`
/// - returns `[B, S_pat * patch_size, latent_dim]`
///
/// If `patch_size == 1`, returns the input unchanged.
pub fn unpatchify_latent<B: Backend>(
    patched: Tensor<B, 3>,
    patch_size: usize,
    latent_dim: usize,
) -> Tensor<B, 3> {
    if patch_size <= 1 {
        return patched;
    }
    let [batch, s_pat, _d_pat] = patched.dims();
    patched.reshape([batch, s_pat * patch_size, latent_dim])
}

// ---------------------------------------------------------------------------
// Reference latent encoder (speaker conditioning)
// ---------------------------------------------------------------------------

/// Speaker / style reference encoder.
///
/// Input: patched DACVAE latent sequence.
/// Processing: `/6.0` scaling → masked re-encoding via TextBlocks →
///             post-masking (keeps unconditional zeros exact).
///
/// Field names match the Python state_dict:
/// `in_proj`, `blocks`.
#[derive(Module, Debug)]
pub struct ReferenceLatentEncoder<B: Backend> {
    pub(crate) in_proj: Linear<B>,
    pub(crate) blocks: Vec<TextBlock<B>>,
    head_dim: usize,
}

impl<B: Backend> ReferenceLatentEncoder<B> {
    pub fn from_cfg(cfg: &ModelConfig, device: &B::Device) -> Self {
        let speaker_dim = cfg
            .speaker_dim
            .expect("speaker_dim required for speaker mode");
        let speaker_heads = cfg
            .speaker_heads
            .expect("speaker_heads required for speaker mode");
        let speaker_layers = cfg
            .speaker_layers
            .expect("speaker_layers required for speaker mode");
        let speaker_patched_latent_dim = cfg.speaker_patched_latent_dim();
        let mlp_ratio = cfg.speaker_mlp_ratio();

        let blocks = (0..speaker_layers)
            .map(|_| {
                TextBlock::new(
                    speaker_dim,
                    speaker_heads,
                    mlp_ratio,
                    cfg.norm_eps,
                    cfg.dropout,
                    device,
                )
            })
            .collect();

        Self {
            in_proj: LinearConfig::new(speaker_patched_latent_dim, speaker_dim)
                .with_bias(true)
                .init(device),
            blocks,
            head_dim: speaker_dim / speaker_heads,
        }
    }

    /// Encode reference latents.
    ///
    /// `latent: [B, S, D_in]` (already patched by `speaker_patch_size`),
    /// `mask: [B, S]` (True = valid frame).
    /// Returns `[B, S, D_speaker]`.
    pub fn forward(&self, latent: Tensor<B, 3>, mask: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let [_batch, seq, _] = latent.dims();
        let device = latent.device();

        // Project and scale
        let x = self.in_proj.forward(latent) / 6.0_f32;

        // Build float mask: [B, S, 1]
        let mask_f = bool_mask_to_float(mask.clone(), &device);
        // Hard-mask at input
        let mut x = x * mask_f.clone();

        // Precompute RoPE
        let (cos, sin) = precompute_rope_freqs::<B>(self.head_dim, seq, 10000.0, &device);

        for block in &self.blocks {
            x = block.forward(x, mask.clone(), cos.clone(), sin.clone());
            // Re-mask after each block (fully-masked → zero)
            x = x * mask_f.clone();
        }
        x * mask_f
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert `Tensor<B, 2, Bool>` → `Tensor<B, 2, Int>` (1 = true, 0 = false).
fn bool_mask_to_int<B: Backend>(mask: Tensor<B, 2, Bool>) -> Tensor<B, 2> {
    // Use mask_where: 1.0 where true, 0.0 where false
    let [batch, seq] = mask.dims();
    let device = mask.device();
    let ones: Tensor<B, 2> = Tensor::ones([batch, seq], &device);
    let zeros: Tensor<B, 2> = Tensor::zeros([batch, seq], &device);
    ones.mask_where(mask.bool_not(), zeros)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn dev() -> <B as Backend>::Device {
        Default::default()
    }

    // --- patch_sequence_with_mask ---

    #[test]
    fn patch_noop_when_size_1() {
        let d = dev();
        let seq = Tensor::<B, 3>::ones([1, 8, 4], &d);
        let mask = Tensor::<B, 2, Bool>::ones([1, 8], &d);
        let (ps, pm) = patch_sequence_with_mask(seq, mask, 1).unwrap();
        assert_eq!(ps.dims(), [1, 8, 4]);
        assert_eq!(pm.dims(), [1, 8]);
    }

    #[test]
    fn patch_halves_length_doubles_dim() {
        let d = dev();
        let seq = Tensor::<B, 3>::ones([2, 8, 4], &d);
        let mask = Tensor::<B, 2, Bool>::ones([2, 8], &d);
        let (ps, pm) = patch_sequence_with_mask(seq, mask, 2).unwrap();
        assert_eq!(ps.dims(), [2, 4, 8]); // S/2, D*2
        assert_eq!(pm.dims(), [2, 4]);
    }

    #[test]
    fn patch_mask_false_propagates() {
        let d = dev();
        let seq = Tensor::<B, 3>::ones([1, 4, 2], &d);
        // mask: [true, true, true, false] — patch_size=2 → patches [0:2] valid, [2:4] invalid
        let mask = Tensor::<B, 2>::from_data([[1.0f32, 1.0, 1.0, 0.0]], &d).greater_elem(0.5);
        let (_, pm) = patch_sequence_with_mask(seq, mask, 2).unwrap();
        assert_eq!(pm.dims(), [1, 2]);
        let mask_vals: Vec<bool> = pm.to_data().to_vec().unwrap();
        assert!(mask_vals[0]); // first patch: both true
        assert!(!mask_vals[1]); // second patch: one false → patch false
    }

    // --- unpatchify_latent ---

    #[test]
    fn unpatchify_noop_when_size_1() {
        let d = dev();
        let patched = Tensor::<B, 3>::ones([1, 4, 8], &d);
        let out = unpatchify_latent(patched, 1, 8);
        assert_eq!(out.dims(), [1, 4, 8]);
    }

    #[test]
    fn unpatchify_doubles_seq_halves_dim() {
        let d = dev();
        let patched = Tensor::<B, 3>::ones([2, 4, 16], &d);
        let out = unpatchify_latent(patched, 2, 8);
        assert_eq!(out.dims(), [2, 8, 8]);
    }

    // --- bool_mask_to_int ---

    #[test]
    fn bool_mask_to_int_converts_correctly() {
        let d = dev();
        let mask = Tensor::<B, 2>::from_data([[1.0f32, 0.0, 1.0, 0.0]], &d).greater_elem(0.5);
        let int_mask = bool_mask_to_int(mask);
        let vals: Vec<f32> = int_mask.to_data().to_vec().unwrap();
        assert_eq!(vals, vec![1.0, 0.0, 1.0, 0.0]);
    }

    // --- ReferenceLatentEncoder ---

    #[test]
    fn speaker_encoder_forward_shape() {
        let d = dev();
        let cfg = crate::train::tiny_model_config();
        let enc = ReferenceLatentEncoder::<B>::from_cfg(&cfg, &d);
        let input_dim = cfg.speaker_patched_latent_dim();
        let speaker_dim = cfg.speaker_dim.unwrap();

        let latent = Tensor::zeros([1, 4, input_dim], &d);
        let mask = Tensor::<B, 2, Bool>::ones([1, 4], &d);
        let out = enc.forward(latent, mask);
        assert_eq!(out.dims(), [1, 4, speaker_dim]);
    }

    #[test]
    fn speaker_encoder_masked_positions_are_zero() {
        let d = dev();
        let cfg = crate::train::tiny_model_config();
        let enc = ReferenceLatentEncoder::<B>::from_cfg(&cfg, &d);
        let input_dim = cfg.speaker_patched_latent_dim();

        let latent = Tensor::ones([1, 4, input_dim], &d);
        // mask: [true, true, false, false]
        let mask = Tensor::<B, 2>::from_data([[1.0f32, 1.0, 0.0, 0.0]], &d).greater_elem(0.5);
        let out = enc.forward(latent, mask);
        // positions 2,3 should be zero
        let pos2 = out.clone().slice([0..1, 2..3]);
        let pos3 = out.slice([0..1, 3..4]);
        let sum2: f32 = pos2.abs().sum().to_data().to_vec::<f32>().unwrap()[0];
        let sum3: f32 = pos3.abs().sum().to_data().to_vec::<f32>().unwrap()[0];
        assert_eq!(sum2, 0.0);
        assert_eq!(sum3, 0.0);
    }

    #[test]
    fn patch_too_short_returns_error() {
        let d = dev();
        // seq_len=1, patch_size=2 → usable=0 → error
        let seq = Tensor::<B, 3>::ones([1, 1, 4], &d);
        let mask = Tensor::<B, 2, Bool>::ones([1, 1], &d);
        let result = patch_sequence_with_mask(seq, mask, 2);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("too short"),
            "error should mention 'too short', got: {err_msg}"
        );
    }
}
