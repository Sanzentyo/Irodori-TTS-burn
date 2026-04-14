//! LoRA record builders for `TensorStore`.
//!
//! All items are gated behind `#[cfg(feature = "train")]`.

#[cfg(feature = "train")]
use burn::{
    module::{EmptyRecord, Param},
    tensor::{Tensor, backend::Backend},
};

#[cfg(feature = "train")]
use super::tensor_store::TensorStore;
#[cfg(feature = "train")]
use crate::{
    config::ModelConfig,
    error::{IrodoriError, Result},
    model::dit::{AuxConditionerRecord, CaptionConditionerRecord, SpeakerConditionerRecord},
    train::{
        lora_layer::LoraLinearRecord,
        lora_model::{
            LoraDiffusionBlockRecord, LoraJointAttentionRecord, LoraTextToLatentRfDiTRecord,
        },
    },
};

#[cfg(feature = "train")]
impl TensorStore {
    /// Return the `(in_features, out_features)` for a Linear at `prefix`.
    ///
    /// The safetensors weight is stored `[d_out, d_in]` (PyTorch convention),
    /// so this reverses the order.
    pub(super) fn linear_dims(&self, prefix: &str) -> Result<(usize, usize)> {
        let key = format!("{prefix}.weight");
        let entry = self.entry(&key)?;
        if entry.shape.len() != 2 {
            return Err(IrodoriError::WrongDim(key, 2, entry.shape.len()));
        }
        Ok((entry.shape[1], entry.shape[0])) // (in_f, out_f)
    }

    /// Build a `LoraLinearRecord<B>` — base from checkpoint, LoRA params freshly initialised.
    pub(super) fn lora_linear_record<B: Backend>(
        &self,
        prefix: &str,
        r: usize,
        _alpha: f32,
        device: &B::Device,
    ) -> Result<LoraLinearRecord<B>> {
        let (in_f, out_f) = self.linear_dims(prefix)?;
        let base = self.linear(prefix, device)?;
        let std = (2.0_f64 / in_f as f64).sqrt();
        let lora_a = Param::from_tensor(Tensor::random(
            [r, in_f],
            burn::tensor::Distribution::Normal(0.0, std),
            device,
        ));
        let lora_b = Param::from_tensor(Tensor::<B, 2>::zeros([out_f, r], device));
        Ok(LoraLinearRecord {
            base,
            lora_a,
            lora_b,
            scale: EmptyRecord::new(),
            out_features: EmptyRecord::new(),
        })
    }

    /// Build a `LoraJointAttentionRecord<B>`.
    pub(super) fn lora_joint_attention<B: Backend>(
        &self,
        prefix: &str,
        r: usize,
        alpha: f32,
        device: &B::Device,
    ) -> Result<LoraJointAttentionRecord<B>> {
        let lora_lin = |sfx: &str| {
            self.lora_linear_record::<B>(format!("{prefix}.{sfx}").as_str(), r, alpha, device)
        };
        let opt_lora_lin = |sfx: &str| -> Result<Option<LoraLinearRecord<B>>> {
            let key = format!("{prefix}.{sfx}.weight");
            if self.has(&key) {
                Ok(Some(self.lora_linear_record::<B>(
                    format!("{prefix}.{sfx}").as_str(),
                    r,
                    alpha,
                    device,
                )?))
            } else {
                Ok(None)
            }
        };
        Ok(LoraJointAttentionRecord {
            wq: lora_lin("wq")?,
            wk: lora_lin("wk")?,
            wv: lora_lin("wv")?,
            wk_text: lora_lin("wk_text")?,
            wv_text: lora_lin("wv_text")?,
            wk_speaker: opt_lora_lin("wk_speaker")?,
            wv_speaker: opt_lora_lin("wv_speaker")?,
            wk_caption: opt_lora_lin("wk_caption")?,
            wv_caption: opt_lora_lin("wv_caption")?,
            gate: lora_lin("gate")?,
            wo: lora_lin("wo")?,
            q_norm: self.head_rms_norm(format!("{prefix}.q_norm").as_str(), device)?,
            k_norm: self.head_rms_norm(format!("{prefix}.k_norm").as_str(), device)?,
            num_heads: EmptyRecord::new(),
            head_dim: EmptyRecord::new(),
            scale: EmptyRecord::new(),
        })
    }

    /// Build a `LoraDiffusionBlockRecord<B>`.
    pub(super) fn lora_diffusion_block<B: Backend>(
        &self,
        prefix: &str,
        r: usize,
        alpha: f32,
        device: &B::Device,
    ) -> Result<LoraDiffusionBlockRecord<B>> {
        Ok(LoraDiffusionBlockRecord {
            attention: self.lora_joint_attention(
                format!("{prefix}.attention").as_str(),
                r,
                alpha,
                device,
            )?,
            mlp: self.swiglu(format!("{prefix}.mlp").as_str(), device)?,
            attention_adaln: self
                .low_rank_adaln(format!("{prefix}.attention_adaln").as_str(), device)?,
            mlp_adaln: self.low_rank_adaln(format!("{prefix}.mlp_adaln").as_str(), device)?,
            dropout: EmptyRecord::new(),
        })
    }

    /// Assemble a `LoraTextToLatentRfDiTRecord<B>` directly from the checkpoint.
    ///
    /// Base weights are loaded from the safetensors store; LoRA params are
    /// freshly initialised (Kaiming for `lora_a`, zeros for `lora_b`).
    pub(super) fn build_lora_model_record<B: Backend>(
        &self,
        cfg: &ModelConfig,
        r: usize,
        alpha: f32,
        device: &B::Device,
    ) -> Result<LoraTextToLatentRfDiTRecord<B>> {
        let text_encoder = self.text_encoder("text_encoder", cfg.text_layers, device)?;
        let text_norm = self.rms_norm("text_norm", cfg.norm_eps, device)?;

        let aux_conditioner = if cfg.use_speaker_condition() {
            let layers = cfg.speaker_layers.expect("speaker_layers required");
            Some(AuxConditionerRecord::Speaker(SpeakerConditionerRecord {
                encoder: self.reference_latent_encoder("speaker_encoder", layers, device)?,
                norm: self.rms_norm("speaker_norm", cfg.norm_eps, device)?,
            }))
        } else if cfg.use_caption_condition {
            let layers = cfg.caption_layers();
            Some(AuxConditionerRecord::Caption(CaptionConditionerRecord {
                encoder: self.text_encoder("caption_encoder", layers, device)?,
                norm: self.rms_norm("caption_norm", cfg.norm_eps, device)?,
            }))
        } else {
            None
        };

        let cond_module = self.cond_module(device)?;
        let in_proj = self.linear("in_proj", device)?;

        let blocks = (0..cfg.num_layers)
            .map(|i| self.lora_diffusion_block(format!("blocks.{i}").as_str(), r, alpha, device))
            .collect::<Result<Vec<_>>>()?;

        let out_norm = self.rms_norm("out_norm", cfg.norm_eps, device)?;
        let out_proj = self.linear("out_proj", device)?;

        Ok(LoraTextToLatentRfDiTRecord {
            text_encoder,
            text_norm,
            aux_conditioner,
            cond_module,
            in_proj,
            blocks,
            out_norm,
            out_proj,
            model_dim: EmptyRecord::new(),
            num_heads: EmptyRecord::new(),
            head_dim: EmptyRecord::new(),
            timestep_embed_dim: EmptyRecord::new(),
            speaker_patch_size: EmptyRecord::new(),
            patched_latent_dim: EmptyRecord::new(),
        })
    }
}

#[cfg(test)]
#[cfg(feature = "train")]
mod tests {
    use crate::weights::{TensorStore, test_helpers::*};
    use safetensors::Dtype;

    #[test]
    fn linear_dims_extract() {
        // PyTorch weight [d_out=4, d_in=8] → linear_dims returns (8, 4)
        let w = vec![0.0f32; 32];
        let file = write_safetensors(
            &[("fc.weight", f32_bytes(&w), Dtype::F32, vec![4, 8])],
            &test_config_json(),
        );
        let store = TensorStore::load(file.path()).unwrap();
        let (in_f, out_f) = store.linear_dims("fc").unwrap();
        assert_eq!(in_f, 8);
        assert_eq!(out_f, 4);
    }
}
