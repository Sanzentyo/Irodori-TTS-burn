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
) -> (Tensor<B, 3>, Tensor<B, 2, Bool>) {
    if patch_size <= 1 {
        return (seq, mask);
    }
    let [batch, seq_len, dim] = seq.dims();
    let usable = (seq_len / patch_size) * patch_size;
    assert!(
        usable > 0,
        "Sequence too short for speaker_patch_size={patch_size}: seq_len={seq_len}"
    );

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

    (seq_patched, mask_patched)
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
