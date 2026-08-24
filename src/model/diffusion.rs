use burn::tensor::Device;
use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig},
    tensor::{Bool, Tensor},
};

#[cfg(feature = "profile")]
use std::time::Instant;

use crate::{config::ModelConfig, nvtx_range};

use super::{
    attention::{CondKvCache, JointAttention, JointAttnCtx},
    condition::EncodedCondition,
    feed_forward::SwiGlu,
    norm::LowRankAdaLn,
};

#[cfg(feature = "profile")]
fn profile_wgpu_stage<T, O>(
    block_index: usize,
    label: &'static str,
    batch: usize,
    sequence: usize,
    reference: &Tensor<3>,
    operation: O,
) -> T
where
    T: Send + 'static,
    O: FnOnce() -> T + Send,
{
    if super::profiling::rf_device_profile_active() {
        return if matches!(label, "adaln_attn" | "adaln_mlp") {
            super::profiling::profile_rf_stage(
                "block", label, batch, sequence, reference, operation,
            )
        } else {
            operation()
        };
    }
    if std::env::var("IRODORI_RF_STAGE_PROFILE").as_deref() != Ok("1") {
        return operation();
    }

    let device = reference.device();
    device.sync().unwrap_or_else(|error| {
        panic!("RF stage pre-sync failed for block {block_index} {label}: {error}")
    });
    let started = Instant::now();
    let output = operation();
    device.sync().unwrap_or_else(|error| {
        panic!("RF stage post-sync failed for block {block_index} {label}: {error}")
    });
    tracing::info!(
        target: "irodori_tts_burn::rf_profile",
        block = block_index,
        stage = label,
        batch,
        sequence,
        device_complete_ms = started.elapsed().as_secs_f64() * 1_000.0,
        "RF block stage profile"
    );
    output
}

#[cfg(feature = "profile")]
macro_rules! rf_profile_stage {
    ($block:expr, $label:expr, $reference:expr, $operation:expr) => {{
        let [batch, sequence, _] = $reference.dims();
        let profile_reference = $reference.clone();
        profile_wgpu_stage($block, $label, batch, sequence, &profile_reference, || {
            $operation
        })
    }};
}

#[cfg(not(feature = "profile"))]
macro_rules! rf_profile_stage {
    ($block:expr, $label:expr, $reference:expr, $operation:expr) => {
        $operation
    };
}

/// Single diffusion transformer block.
///
/// Applies JointAttention and SwiGLU FFN, both gated via LowRankAdaLN conditioning.
/// Field names match the Python state_dict:
/// `attention`, `mlp`, `attention_adaln`, `mlp_adaln`, `dropout`.
#[derive(Module, Debug)]
pub struct DiffusionBlock {
    pub(crate) attention: JointAttention,
    pub(crate) mlp: SwiGlu,
    pub(crate) attention_adaln: LowRankAdaLn,
    pub(crate) mlp_adaln: LowRankAdaLn,
    pub(crate) dropout: Dropout,
    dropout_is_identity: bool,
}

impl DiffusionBlock {
    pub fn new(cfg: &ModelConfig, device: &Device) -> Self {
        let hidden_dim = ((cfg.model_dim as f64 * cfg.mlp_ratio) as usize).max(1);
        let adaln_rank = cfg.adaln_rank.max(1).min(cfg.model_dim);

        Self {
            attention: JointAttention::new(cfg, device),
            mlp: SwiGlu::new(cfg.model_dim, Some(hidden_dim), device),
            attention_adaln: LowRankAdaLn::new(cfg.model_dim, adaln_rank, cfg.norm_eps, device),
            mlp_adaln: LowRankAdaLn::new(cfg.model_dim, adaln_rank, cfg.norm_eps, device),
            dropout: DropoutConfig::new(cfg.dropout).init(),
            dropout_is_identity: cfg.dropout == 0.0,
        }
    }

    /// Hidden dimension for the SwiGLU MLP.
    #[cfg(test)]
    pub fn hidden_dim(cfg: &ModelConfig) -> usize {
        ((cfg.model_dim as f64 * cfg.mlp_ratio) as usize).max(1)
    }

    /// Pre-fuse weight matrices in attention (QKV+gate) and MLP (w1‖w3) for inference.
    pub(crate) fn prepare_for_inference(&mut self) {
        self.attention.prepare_for_inference();
        self.mlp.prepare_for_inference();
    }

    /// Branch-free forward using pre-fused weight matrices.
    ///
    /// # Panics
    ///
    /// Panics if [`prepare_for_inference`](Self::prepare_for_inference) has not
    /// been called on this block.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_fused(
        &self,
        x: Tensor<3>,
        cond_embed: Tensor<3>,
        cond: &EncodedCondition,
        cos: Tensor<2>,
        sin: Tensor<2>,
        kv_cache: Option<&CondKvCache>,
        latent_mask: Option<Tensor<2, Bool>>,
    ) -> Tensor<3> {
        let (speaker_state, speaker_mask) = cond
            .aux
            .as_ref()
            .and_then(|aux| aux.speaker())
            .map(|(state, mask)| (Some(state.clone()), Some(mask.clone())))
            .unwrap_or((None, None));
        let (caption_state, caption_mask) = cond
            .aux
            .as_ref()
            .and_then(|aux| aux.caption())
            .map(|(state, mask)| (Some(state.clone()), Some(mask.clone())))
            .unwrap_or((None, None));

        let ctx = JointAttnCtx {
            text_state: cond.text_state.clone(),
            text_mask: cond.text_mask.clone(),
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
            kv_cache,
        };

        // Attention path (fused QKV+gate)
        let (h_attn, attn_gate) = nvtx_range!(
            "adaln_attn",
            self.attention_adaln.forward(x.clone(), cond_embed.clone())
        );
        let attn_out = nvtx_range!(
            "joint_attention_fused",
            self.attention
                .forward_fused(h_attn, ctx, cos, sin, latent_mask)
        );
        let x = x + self.dropout.forward(attn_gate * attn_out);

        // MLP path (fused w1‖w3)
        let (h_mlp, mlp_gate) =
            nvtx_range!("adaln_mlp", self.mlp_adaln.forward(x.clone(), cond_embed));
        let mlp_out = nvtx_range!("swiglu_mlp_fused", self.mlp.forward_fused(h_mlp));
        x + self.dropout.forward(mlp_gate * mlp_out)
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
        x: Tensor<3>,
        cond_embed: Tensor<3>,
        cond: &EncodedCondition,
        cos: Tensor<2>,
        sin: Tensor<2>,
        kv_cache: Option<&CondKvCache>,
        latent_mask: Option<Tensor<2, Bool>>,
    ) -> Tensor<3> {
        let (speaker_state, speaker_mask) = cond
            .aux
            .as_ref()
            .and_then(|aux| aux.speaker())
            .map(|(state, mask)| (Some(state.clone()), Some(mask.clone())))
            .unwrap_or((None, None));
        let (caption_state, caption_mask) = cond
            .aux
            .as_ref()
            .and_then(|aux| aux.caption())
            .map(|(state, mask)| (Some(state.clone()), Some(mask.clone())))
            .unwrap_or((None, None));

        let ctx = JointAttnCtx {
            text_state: cond.text_state.clone(),
            text_mask: cond.text_mask.clone(),
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
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

impl DiffusionBlock {
    /// Production WGPU inference path with measured WGSL elementwise fusions.
    ///
    /// Matmuls and attention remain on Burn/CubeCL's tuned implementations.
    /// The custom shaders fuse AdaLN normalization/modulation, SwiGLU
    /// activation, and gated residual updates, which are the kernels that won
    /// their v4-shape benchmarks on the target adapter.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_fused_wgsl(
        &self,
        _block_index: usize,
        x: Tensor<3>,
        cond_embed: Tensor<3>,
        precomputed_adaln: Option<super::adaln_cross_layer::BlockAdaLnModulations>,
        cond: &EncodedCondition,
        cos: Tensor<2>,
        sin: Tensor<2>,
        kv_cache: Option<&CondKvCache>,
        latent_mask: Option<Tensor<2, Bool>>,
    ) -> Tensor<3> {
        let (speaker_state, speaker_mask) = cond
            .aux
            .as_ref()
            .and_then(|aux| aux.speaker())
            .map(|(state, mask)| (Some(state.clone()), Some(mask.clone())))
            .unwrap_or((None, None));
        let (caption_state, caption_mask) = cond
            .aux
            .as_ref()
            .and_then(|aux| aux.caption())
            .map(|(state, mask)| (Some(state.clone()), Some(mask.clone())))
            .unwrap_or((None, None));
        let ctx = JointAttnCtx {
            text_state: cond.text_state.clone(),
            text_mask: cond.text_mask.clone(),
            speaker_state,
            speaker_mask,
            caption_state,
            caption_mask,
            kv_cache,
        };
        let (attention_modulation, mlp_modulation) = match precomputed_adaln {
            Some(modulations) => (Some(modulations.attention), Some(modulations.mlp)),
            None => (None, None),
        };

        let (h_attn, attn_gate) = nvtx_range!(
            "adaln_attn_wgsl",
            rf_profile_stage!(_block_index, "adaln_attn", x, {
                self.attention_adaln.forward_wgsl(
                    x.clone(),
                    cond_embed.clone(),
                    attention_modulation,
                )
            })
        );
        let x = if self.dropout_is_identity {
            nvtx_range!(
                "joint_attention_residual_wgsl",
                rf_profile_stage!(_block_index, "attention_residual_fused", h_attn, {
                    self.attention.forward_fused_residual_wgsl(
                        h_attn,
                        ctx,
                        cos,
                        sin,
                        latent_mask,
                        x,
                        attn_gate,
                    )
                })
            )
        } else {
            let attn_out = nvtx_range!(
                "joint_attention_fused_wgsl",
                rf_profile_stage!(_block_index, "attention", h_attn, {
                    self.attention
                        .forward_fused_wgsl(h_attn, ctx, cos, sin, latent_mask)
                })
            );
            rf_profile_stage!(_block_index, "attention_residual", x, {
                fused_residual_update(x, self.dropout.forward(attn_out), attn_gate)
            })
        };

        let (h_mlp, mlp_gate) = nvtx_range!(
            "adaln_mlp_wgsl",
            rf_profile_stage!(_block_index, "adaln_mlp", x, {
                self.mlp_adaln
                    .forward_wgsl(x.clone(), cond_embed, mlp_modulation)
            })
        );
        if self.dropout_is_identity {
            nvtx_range!(
                "swiglu_mlp_residual_wgsl",
                rf_profile_stage!(_block_index, "mlp_residual_fused", h_mlp, {
                    self.mlp.forward_fused_residual_wgsl(h_mlp, x, mlp_gate)
                })
            )
        } else {
            let mlp_out = nvtx_range!(
                "swiglu_mlp_wgsl",
                rf_profile_stage!(_block_index, "mlp", h_mlp, {
                    self.mlp.forward_fused_wgsl(h_mlp)
                })
            );
            rf_profile_stage!(_block_index, "mlp_residual", x, {
                fused_residual_update(x, self.dropout.forward(mlp_out), mlp_gate)
            })
        }
    }
}

fn fused_residual_update(residual: Tensor<3>, branch: Tensor<3>, gate: Tensor<3>) -> Tensor<3> {
    let [batch, seq_len, dim] = residual.dims();
    let output = crate::kernels::fused_residual_gate::fused_residual_gate_wgsl(
        residual
            .reshape([batch * seq_len, dim])
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
        branch
            .reshape([batch * seq_len, dim])
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
        gate.reshape([batch, dim])
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("tensor must use WGPU raw backend"),
        batch,
        seq_len,
    );
    Tensor::<2>::from_primitive::<crate::WgpuRaw>(output).reshape([batch, seq_len, dim])
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tiny_cfg() -> ModelConfig {
        crate::config::tiny_model_config()
    }

    #[test]
    fn block_output_shape_matches_input() {
        let cfg = tiny_cfg();
        let dev = Default::default();
        let block = DiffusionBlock::new(&cfg, &dev);

        let b = 2;
        let s_lat = 8;
        let d = cfg.model_dim;
        let text_dim = cfg.text_dim;
        let speaker_dim = cfg.speaker_dim.unwrap_or(d);
        let x = Tensor::<3>::zeros([b, s_lat, d], &dev);
        let cond_embed = Tensor::<3>::zeros([b, 1, d * 3], &dev);

        let cond = EncodedCondition {
            text_state: Tensor::<3>::zeros([b, 4, text_dim], &dev),
            text_mask: Tensor::<2, Bool>::ones([b, 4], &dev),
            aux: Some(super::super::condition::AuxConditionState::Speaker {
                state: Tensor::<3>::zeros([b, 3, speaker_dim], &dev),
                mask: Tensor::<2, Bool>::ones([b, 3], &dev),
            }),
        };

        let (cos, sin) = {
            let half = cfg.head_dim() / 2;
            (
                Tensor::<2>::zeros([s_lat, half], &dev),
                Tensor::<2>::zeros([s_lat, half], &dev),
            )
        };

        let out = block.forward(x, cond_embed, &cond, cos, sin, None, None);
        assert_eq!(out.dims(), [b, s_lat, d]);
    }

    #[test]
    fn block_accepts_speaker_and_caption_contexts_together() {
        let mut cfg = tiny_cfg();
        cfg.use_speaker_condition = Some(true);
        cfg.use_caption_condition = true;
        cfg.caption_dim = Some(12);
        cfg.caption_heads = Some(2);
        cfg.caption_layers = Some(1);
        let dev = Default::default();
        let block = DiffusionBlock::new(&cfg, &dev);

        let (batch, seq_lat) = (1, 4);
        let x = Tensor::<3>::zeros([batch, seq_lat, cfg.model_dim], &dev);
        let cond_embed = Tensor::<3>::zeros([batch, 1, cfg.model_dim * 3], &dev);
        let cond = EncodedCondition {
            text_state: Tensor::zeros([batch, 2, cfg.text_dim], &dev),
            text_mask: Tensor::ones([batch, 2], &dev),
            aux: Some(super::super::condition::AuxConditionState::Both {
                speaker_state: Tensor::zeros([batch, 3, cfg.speaker_dim.unwrap()], &dev),
                speaker_mask: Tensor::ones([batch, 3], &dev),
                caption_state: Tensor::zeros([batch, 5, cfg.caption_dim()], &dev),
                caption_mask: Tensor::ones([batch, 5], &dev),
            }),
        };
        let cos = Tensor::zeros([seq_lat, cfg.head_dim() / 2], &dev);
        let sin = Tensor::zeros([seq_lat, cfg.head_dim() / 2], &dev);

        let out = block.forward(x, cond_embed, &cond, cos, sin, None, None);
        assert_eq!(out.dims(), [batch, seq_lat, cfg.model_dim]);
    }

    #[test]
    fn hidden_dim_calculation() {
        let cfg = tiny_cfg();
        let expected = ((cfg.model_dim as f64 * cfg.mlp_ratio) as usize).max(1);
        assert_eq!(DiffusionBlock::hidden_dim(&cfg), expected);
    }

    #[test]
    fn block_residual_connection_with_zeros() {
        let cfg = tiny_cfg();
        let dev = Default::default();
        let block = DiffusionBlock::new(&cfg, &dev);

        let b = 1;
        let s_lat = 4;
        let d = cfg.model_dim;
        let text_dim = cfg.text_dim;

        let x = Tensor::<3>::zeros([b, s_lat, d], &dev);
        let cond_embed = Tensor::<3>::zeros([b, 1, d * 3], &dev);

        let cond = EncodedCondition {
            text_state: Tensor::<3>::zeros([b, 2, text_dim], &dev),
            text_mask: Tensor::<2, Bool>::ones([b, 2], &dev),
            aux: None,
        };

        let half = cfg.head_dim() / 2;
        let cos = Tensor::<2>::zeros([s_lat, half], &dev);
        let sin = Tensor::<2>::zeros([s_lat, half], &dev);

        let out = block.forward(x, cond_embed, &cond, cos, sin, None, None);
        let data: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(
            data.iter().all(|v| v.is_finite()),
            "all outputs must be finite"
        );
    }

    #[test]
    fn block_caption_conditioned_output_shape() {
        let cfg = crate::config::tiny_caption_config();
        let dev = Default::default();
        let block = DiffusionBlock::new(&cfg, &dev);

        let b = 2;
        let s_lat = 6;
        let d = cfg.model_dim;
        let text_dim = cfg.text_dim;
        let caption_dim = cfg.caption_dim();

        let x = Tensor::<3>::zeros([b, s_lat, d], &dev);
        let cond_embed = Tensor::<3>::zeros([b, 1, d * 3], &dev);

        let cond = EncodedCondition {
            text_state: Tensor::<3>::zeros([b, 4, text_dim], &dev),
            text_mask: Tensor::<2, Bool>::ones([b, 4], &dev),
            aux: Some(super::super::condition::AuxConditionState::Caption {
                state: Tensor::<3>::zeros([b, 3, caption_dim], &dev),
                mask: Tensor::<2, Bool>::ones([b, 3], &dev),
            }),
        };

        let half = cfg.head_dim() / 2;
        let cos = Tensor::<2>::zeros([s_lat, half], &dev);
        let sin = Tensor::<2>::zeros([s_lat, half], &dev);

        let out = block.forward(x, cond_embed, &cond, cos, sin, None, None);
        assert_eq!(out.dims(), [b, s_lat, d]);
    }
}
