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
    pub(crate) attention_norm: RmsNorm<B>,
    pub(crate) attention: SelfAttention<B>,
    pub(crate) mlp_norm: RmsNorm<B>,
    pub(crate) mlp: SwiGlu<B>,
    pub(crate) dropout: Dropout,
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
            attention: SelfAttention::new(dim, heads, None, norm_eps, device),
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
    pub(crate) text_embedding: Embedding<B>,
    pub(crate) blocks: Vec<TextBlock<B>>,
    head_dim: usize,
}

/// Construction parameters for [`TextEncoder::new`].
pub(crate) struct TextEncoderSpec {
    pub(crate) vocab_size: usize,
    pub(crate) dim: usize,
    pub(crate) num_layers: usize,
    pub(crate) num_heads: usize,
    pub(crate) mlp_ratio: f64,
    pub(crate) norm_eps: f64,
    pub(crate) dropout: f64,
}

impl<B: Backend> TextEncoder<B> {
    pub fn from_cfg(cfg: &ModelConfig, device: &B::Device) -> Self {
        Self::new(
            &TextEncoderSpec {
                vocab_size: cfg.text_vocab_size,
                dim: cfg.text_dim,
                num_layers: cfg.text_layers,
                num_heads: cfg.text_heads,
                mlp_ratio: cfg.text_mlp_ratio(),
                norm_eps: cfg.norm_eps,
                dropout: cfg.dropout,
            },
            device,
        )
    }

    pub(crate) fn new(spec: &TextEncoderSpec, device: &B::Device) -> Self {
        let blocks = (0..spec.num_layers)
            .map(|_| {
                TextBlock::new(
                    spec.dim,
                    spec.num_heads,
                    spec.mlp_ratio,
                    spec.norm_eps,
                    spec.dropout,
                    device,
                )
            })
            .collect();

        Self {
            text_embedding: EmbeddingConfig::new(spec.vocab_size, spec.dim).init(device),
            blocks,
            head_dim: spec.dim / spec.num_heads,
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn dev() -> <B as Backend>::Device {
        Default::default()
    }

    // --- bool_mask_to_float ---

    #[test]
    fn bool_mask_to_float_shape() {
        let d = dev();
        let mask = Tensor::<B, 2, Bool>::ones([2, 5], &d);
        let out = bool_mask_to_float(mask, &d);
        assert_eq!(out.dims(), [2, 5, 1]);
    }

    #[test]
    fn bool_mask_to_float_values() {
        let d = dev();
        let mask = Tensor::<B, 2>::from_data([[1.0f32, 0.0, 1.0]], &d).greater_elem(0.5);
        let out = bool_mask_to_float(mask, &d);
        let vals: Vec<f32> = out.reshape([3]).to_data().to_vec().unwrap();
        assert_eq!(vals, vec![1.0, 0.0, 1.0]);
    }

    // --- TextBlock ---

    #[test]
    fn text_block_forward_shape() {
        let d = dev();
        let block = TextBlock::<B>::new(16, 2, 2.0, 1e-5, 0.0, &d);
        let x = Tensor::zeros([1, 4, 16], &d);
        let mask = Tensor::<B, 2, Bool>::ones([1, 4], &d);
        let (cos, sin) = precompute_rope_freqs::<B>(8, 4, 10000.0, &d);
        let out = block.forward(x, mask, cos, sin);
        assert_eq!(out.dims(), [1, 4, 16]);
    }

    // --- TextEncoder ---

    #[test]
    fn text_encoder_from_cfg_forward_shape() {
        let d = dev();
        let cfg = crate::config::tiny_model_config();
        let enc = TextEncoder::<B>::from_cfg(&cfg, &d);

        let ids = Tensor::<B, 2, Int>::zeros([1, 6], &d);
        let mask = Tensor::<B, 2, Bool>::ones([1, 6], &d);
        let out = enc.forward(ids, mask);
        assert_eq!(out.dims(), [1, 6, cfg.text_dim]);
    }

    #[test]
    fn text_encoder_masked_positions_are_zero() {
        let d = dev();
        let cfg = crate::config::tiny_model_config();
        let enc = TextEncoder::<B>::from_cfg(&cfg, &d);

        let ids = Tensor::<B, 2, Int>::zeros([1, 4], &d);
        let mask = Tensor::<B, 2>::from_data([[1.0f32, 1.0, 0.0, 0.0]], &d).greater_elem(0.5);
        let out = enc.forward(ids, mask);
        // Positions 2,3 should be zero due to re-masking
        let pos2 = out.clone().slice([0..1, 2..3]);
        let pos3 = out.slice([0..1, 3..4]);
        let sum2: f32 = pos2.abs().sum().to_data().to_vec::<f32>().unwrap()[0];
        let sum3: f32 = pos3.abs().sum().to_data().to_vec::<f32>().unwrap()[0];
        assert_eq!(sum2, 0.0);
        assert_eq!(sum3, 0.0);
    }

    #[test]
    fn text_encoder_all_false_mask_no_nan() {
        // Verify the text encoder handles all-false masks without NaN.
        // This covers the CFG text dropout path in training, where a sample's
        // entire text mask may be zeroed to produce the unconditional pass.
        let d = dev();
        let cfg = crate::config::tiny_model_config();
        let enc = TextEncoder::<B>::from_cfg(&cfg, &d);

        let ids = Tensor::<B, 2, Int>::zeros([1, 4], &d);
        let mask = Tensor::<B, 2, Bool>::from_data([[false, false, false, false]], &d);
        let out = enc.forward(ids, mask);

        let vals: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "text encoder must not produce NaN/Inf for all-false mask (got non-finite values)"
        );
    }
}
