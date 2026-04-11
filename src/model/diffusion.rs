use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig},
    tensor::{Tensor, backend::Backend},
};

use crate::config::ModelConfig;

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

    /// Forward with encoded conditions.
    ///
    /// - `x: [B, S_lat, D]` — latent sequence
    /// - `cond_embed: [B, 1, D*3]` — timestep conditioning
    /// - `cond` — pre-encoded text/speaker/caption conditioning
    /// - `cos/sin: [S_lat, head_dim/2]` — precomputed RoPE
    /// - `kv_cache: Option<&CondKvCache>` — cached context KV projections
    ///
    /// Returns updated `[B, S_lat, D]`.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cond_embed: Tensor<B, 3>,
        cond: &EncodedCondition<B>,
        cos: Tensor<B, 2>,
        sin: Tensor<B, 2>,
        kv_cache: Option<&CondKvCache<B>>,
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
        let (h_attn, attn_gate) = self.attention_adaln.forward(x.clone(), cond_embed.clone());
        let attn_out = self.attention.forward(h_attn, ctx, cos, sin);
        let x = x + self.dropout.forward(attn_gate * attn_out);

        // MLP path
        let (h_mlp, mlp_gate) = self.mlp_adaln.forward(x.clone(), cond_embed);
        let mlp_out = self.mlp.forward(h_mlp);
        x + self.dropout.forward(mlp_gate * mlp_out)
    }
}
