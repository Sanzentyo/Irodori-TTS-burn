use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig},
    tensor::{Bool, Tensor, backend::Backend},
};

use crate::{config::ModelConfig, nvtx_range};

use super::{
    attention::{CondKvCache, JointAttention, JointAttnCtx},
    condition::EncodedCondition,
    feed_forward::SwiGlu,
    norm::LowRankAdaLn,
};

/// Single diffusion transformer block.
///
/// Applies JointAttention and SwiGLU FFN, both gated via LowRankAdaLN conditioning.
/// Field names match the Python state_dict:
/// `attention`, `mlp`, `attention_adaln`, `mlp_adaln`, `dropout`.
#[derive(Module, Debug)]
pub struct DiffusionBlock<B: Backend> {
    pub(crate) attention: JointAttention<B>,
    pub(crate) mlp: SwiGlu<B>,
    pub(crate) attention_adaln: LowRankAdaLn<B>,
    pub(crate) mlp_adaln: LowRankAdaLn<B>,
    pub(crate) dropout: Dropout,
}

impl<B: Backend> DiffusionBlock<B> {
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> Self {
        let hidden_dim = ((cfg.model_dim as f64 * cfg.mlp_ratio) as usize).max(1);
        let adaln_rank = cfg.adaln_rank.max(1).min(cfg.model_dim);

        Self {
            attention: JointAttention::new(cfg, device),
            mlp: SwiGlu::new(cfg.model_dim, Some(hidden_dim), device),
            attention_adaln: LowRankAdaLn::new(cfg.model_dim, adaln_rank, cfg.norm_eps, device),
            mlp_adaln: LowRankAdaLn::new(cfg.model_dim, adaln_rank, cfg.norm_eps, device),
            dropout: DropoutConfig::new(cfg.dropout).init(),
        }
    }

    /// Hidden dimension for the SwiGLU MLP.
    pub fn hidden_dim(cfg: &ModelConfig) -> usize {
        ((cfg.model_dim as f64 * cfg.mlp_ratio) as usize).max(1)
    }

    /// Forward with encoded conditions.
    ///
    /// - `x: [B, S_lat, D]` — latent sequence
    /// - `cond_embed: [B, 1, D*3]` — timestep conditioning
    /// - `cond` — pre-encoded text/speaker/caption conditioning
    /// - `cos/sin: [S_lat, head_dim/2]` — precomputed RoPE
    /// - `kv_cache: Option<&CondKvCache>` — cached context KV projections
    ///
    /// Returns updated `[B, S_lat, D]`.
    #[allow(clippy::too_many_arguments)] // ML forward passes naturally have many inputs
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cond_embed: Tensor<B, 3>,
        cond: &EncodedCondition<B>,
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
        kv_cache: Option<&CondKvCache<B>>,
        latent_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        // Select the active auxiliary conditioning (speaker XOR caption)
        let (aux_state, aux_mask) = match &cond.aux {
            Some(aux) => {
                let (s, m) = aux.state_and_mask();
                (Some(s.clone()), Some(m.clone()))
            }
            None => (None, None),
        };

        let ctx = JointAttnCtx {
            text_state: cond.text_state.clone(),
            text_mask: cond.text_mask.clone(),
            aux_state,
            aux_mask,
            kv_cache,
        };

        // Attention path
        let (h_attn, attn_gate) = nvtx_range!(
            "adaln_attn",
            self.attention_adaln.forward(x.clone(), cond_embed.clone())
        );
        let attn_out = nvtx_range!(
            "joint_attention",
            self.attention.forward(h_attn, ctx, cos, sin, latent_mask)
        );
        let x = x + self.dropout.forward(attn_gate * attn_out);

        // MLP path
        let (h_mlp, mlp_gate) =
            nvtx_range!("adaln_mlp", self.mlp_adaln.forward(x.clone(), cond_embed));
        let mlp_out = nvtx_range!("swiglu_mlp", self.mlp.forward(h_mlp));
        x + self.dropout.forward(mlp_gate * mlp_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    fn tiny_cfg() -> ModelConfig {
        crate::train::tiny_model_config()
    }

    #[test]
    fn block_output_shape_matches_input() {
        let cfg = tiny_cfg();
        let dev = Default::default();
        let block = DiffusionBlock::<B>::new(&cfg, &dev);

        let b = 2;
        let s_lat = 8;
        let d = cfg.model_dim;
        let text_dim = cfg.text_dim;
        let speaker_dim = cfg.speaker_dim.unwrap_or(d);
        let x = Tensor::<B, 3>::zeros([b, s_lat, d], &dev);
        let cond_embed = Tensor::<B, 3>::zeros([b, 1, d * 3], &dev);

        let cond = EncodedCondition {
            text_state: Tensor::<B, 3>::zeros([b, 4, text_dim], &dev),
            text_mask: Tensor::<B, 2, Bool>::ones([b, 4], &dev),
            aux: Some(super::super::condition::AuxConditionState::Speaker {
                state: Tensor::<B, 3>::zeros([b, 3, speaker_dim], &dev),
                mask: Tensor::<B, 2, Bool>::ones([b, 3], &dev),
            }),
        };

        let (cos, sin) = {
            let half = cfg.head_dim() / 2;
            (
                Tensor::<B, 2>::zeros([s_lat, half], &dev),
                Tensor::<B, 2>::zeros([s_lat, half], &dev),
            )
        };

        let out = block.forward(x, cond_embed, &cond, cos, sin, None, None);
        assert_eq!(out.dims(), [b, s_lat, d]);
    }

    #[test]
    fn hidden_dim_calculation() {
        let cfg = tiny_cfg();
        let expected = ((cfg.model_dim as f64 * cfg.mlp_ratio) as usize).max(1);
        assert_eq!(DiffusionBlock::<B>::hidden_dim(&cfg), expected);
    }

    #[test]
    fn block_residual_connection_with_zeros() {
        let cfg = tiny_cfg();
        let dev = Default::default();
        let block = DiffusionBlock::<B>::new(&cfg, &dev);

        let b = 1;
        let s_lat = 4;
        let d = cfg.model_dim;
        let text_dim = cfg.text_dim;

        let x = Tensor::<B, 3>::zeros([b, s_lat, d], &dev);
        let cond_embed = Tensor::<B, 3>::zeros([b, 1, d * 3], &dev);

        let cond = EncodedCondition {
            text_state: Tensor::<B, 3>::zeros([b, 2, text_dim], &dev),
            text_mask: Tensor::<B, 2, Bool>::ones([b, 2], &dev),
            aux: None,
        };

        let half = cfg.head_dim() / 2;
        let cos = Tensor::<B, 2>::zeros([s_lat, half], &dev);
        let sin = Tensor::<B, 2>::zeros([s_lat, half], &dev);

        let out = block.forward(x, cond_embed, &cond, cos, sin, None, None);
        let data: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(data.iter().all(|v| v.is_finite()), "all outputs must be finite");
    }

    #[test]
    fn block_caption_conditioned_output_shape() {
        let cfg = crate::train::tiny_caption_config();
        let dev = Default::default();
        let block = DiffusionBlock::<B>::new(&cfg, &dev);

        let b = 2;
        let s_lat = 6;
        let d = cfg.model_dim;
        let text_dim = cfg.text_dim;
        let caption_dim = cfg.caption_dim();

        let x = Tensor::<B, 3>::zeros([b, s_lat, d], &dev);
        let cond_embed = Tensor::<B, 3>::zeros([b, 1, d * 3], &dev);

        let cond = EncodedCondition {
            text_state: Tensor::<B, 3>::zeros([b, 4, text_dim], &dev),
            text_mask: Tensor::<B, 2, Bool>::ones([b, 4], &dev),
            aux: Some(super::super::condition::AuxConditionState::Caption {
                state: Tensor::<B, 3>::zeros([b, 3, caption_dim], &dev),
                mask: Tensor::<B, 2, Bool>::ones([b, 3], &dev),
            }),
        };

        let half = cfg.head_dim() / 2;
        let cos = Tensor::<B, 2>::zeros([s_lat, half], &dev);
        let sin = Tensor::<B, 2>::zeros([s_lat, half], &dev);

        let out = block.forward(x, cond_embed, &cond, cos, sin, None, None);
        assert_eq!(out.dims(), [b, s_lat, d]);
    }
}
