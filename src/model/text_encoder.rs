use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig},
    tensor::{Bool, Int, Tensor, backend::Backend},
};

use crate::config::ModelConfig;

use super::{
    attention::SelfAttention, feed_forward::SwiGlu, norm::RmsNorm, rope::precompute_rope_freqs,
};

/// Single transformer block used in both `TextEncoder` and `ReferenceLatentEncoder`.
///
/// Field names match the Python state_dict:
/// `attention_norm`, `attention`, `mlp_norm`, `mlp`, `dropout`.
#[derive(Module, Debug)]
pub struct TextBlock<B: Backend> {
    pub attention_norm: RmsNorm<B>,
    pub attention: SelfAttention<B>,
    pub mlp_norm: RmsNorm<B>,
    pub mlp: SwiGlu<B>,
    pub dropout: Dropout,
}

impl<B: Backend> TextBlock<B> {
    pub fn new(
        dim: usize,
        heads: usize,
        mlp_ratio: f64,
        norm_eps: f64,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        let hidden_dim = ((dim as f64 * mlp_ratio) as usize).max(1);
        Self {
            attention_norm: RmsNorm::new(dim, norm_eps, device),
            attention: SelfAttention::new(dim, heads, None, device),
            mlp_norm: RmsNorm::new(dim, norm_eps, device),
            mlp: SwiGlu::new(dim, Some(hidden_dim), device),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    /// `x: [B, S, D]`, `mask: [B, S]` (True = valid), `cos/sin: [S, head_dim/2]`
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Tensor<B, 2, Bool>,
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let attn_out =
            self.attention
                .forward(self.attention_norm.forward(x.clone()), cos, sin, Some(mask));
        let x = x + self.dropout.forward(attn_out);

        let mlp_out = self.mlp.forward(self.mlp_norm.forward(x.clone()));
        x + self.dropout.forward(mlp_out)
    }
}

/// Text encoder: embedding + N×TextBlock + re-masking.
///
/// Field names match the Python state_dict:
/// `text_embedding`, `blocks`.
#[derive(Module, Debug)]
pub struct TextEncoder<B: Backend> {
    pub text_embedding: Embedding<B>,
    pub blocks: Vec<TextBlock<B>>,
    head_dim: usize,
}

impl<B: Backend> TextEncoder<B> {
    pub fn from_cfg(cfg: &ModelConfig, device: &B::Device) -> Self {
        Self::new(
            cfg.text_vocab_size,
            cfg.text_dim,
            cfg.text_layers,
            cfg.text_heads,
            cfg.text_mlp_ratio(),
            cfg.norm_eps,
            cfg.dropout,
            device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab_size: usize,
        dim: usize,
        layers: usize,
        heads: usize,
        mlp_ratio: f64,
        norm_eps: f64,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        let blocks = (0..layers)
            .map(|_| TextBlock::new(dim, heads, mlp_ratio, norm_eps, dropout, device))
            .collect();

        Self {
            text_embedding: EmbeddingConfig::new(vocab_size, dim).init(device),
            blocks,
            head_dim: dim / heads,
        }
    }

    /// Encode token ids to contextual embeddings.
    ///
    /// `input_ids: [B, S]`, `mask: [B, S]` (True = valid).  
    /// Masked positions are zeroed before and after every block (re-masking).
    /// Returns `[B, S, D]`.
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>, mask: Tensor<B, 2, Bool>) -> Tensor<B, 3> {
        let [_batch, seq] = input_ids.dims();
        let device = input_ids.device();

        let mut x = self.text_embedding.forward(input_ids); // [B, S, D]

        // Expand mask for broadcasting: [B, S, 1]
        let mask_f = bool_mask_to_float(mask.clone(), &device);

        // Hard-mask before first block
        x = x * mask_f.clone();

        // Precompute RoPE tables
        let (cos, sin) = precompute_rope_freqs::<B>(self.head_dim, seq, 10000.0, &device);

        for block in &self.blocks {
            x = block.forward(x, mask.clone(), cos.clone(), sin.clone());
            // Re-mask after each block: keeps fully-masked conditions truly zero
            x = x * mask_f.clone();
        }
        x * mask_f
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a Bool mask `[B, S]` to a Float `[B, S, 1]` for broadcasting.
pub fn bool_mask_to_float<B: Backend>(
    mask: Tensor<B, 2, Bool>,
    device: &B::Device,
) -> Tensor<B, 3> {
    let [batch, seq] = mask.dims();
    let ones: Tensor<B, 2> = Tensor::ones([batch, seq], device);
    let zeros: Tensor<B, 2> = Tensor::zeros([batch, seq], device);
    ones.mask_where(mask.bool_not(), zeros)
        .unsqueeze_dim::<3>(2) // [B, S, 1]
}
