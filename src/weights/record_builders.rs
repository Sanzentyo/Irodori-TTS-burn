//! Base-model record builders for `TensorStore`.
//!
//! These methods construct burn `Record` hierarchies from raw safetensors data,
//! performing the necessary PyTorch→burn weight transposition for `Linear` layers.

use burn::{
    module::{EmptyRecord, Param, ParamId},
    nn::{EmbeddingRecord, LinearRecord},
    tensor::{Tensor, backend::Backend},
};

use super::tensor_store::TensorStore;
use crate::{
    config::ModelConfig,
    error::{IrodoriError, Result},
    model::{
        attention::{JointAttentionRecord, SelfAttentionRecord},
        diffusion::DiffusionBlockRecord,
        dit::{
            AuxConditionerRecord, CaptionConditionerRecord, CondModuleRecord,
            SpeakerConditionerRecord, TextToLatentRfDiTRecord,
        },
        feed_forward::SwiGluRecord,
        norm::{HeadRmsNormRecord, LowRankAdaLnRecord, RmsNormRecord},
        speaker_encoder::ReferenceLatentEncoderRecord,
        text_encoder::{TextBlockRecord, TextEncoderRecord},
    },
};

impl TensorStore {
    /// Build a `LinearRecord<B>` from weights at `prefix.weight` (and optionally `prefix.bias`).
    ///
    /// # Weight transposition
    /// PyTorch stores `nn.Linear.weight` as `[d_output, d_input]` and computes `x @ W.T`.
    /// Burn stores `Linear.weight` as `[d_input, d_output]` and computes `x @ W`.
    /// We transpose the safetensors weight before building the param.
    pub(super) fn linear<B: Backend>(
        &self,
        prefix: &str,
        device: &B::Device,
    ) -> Result<LinearRecord<B>> {
        let w_key = format!("{prefix}.weight");
        let b_key = format!("{prefix}.bias");

        let weight = {
            let entry = self.entry(&w_key)?;
            if entry.shape.len() != 2 {
                return Err(IrodoriError::WrongDim(w_key.clone(), 2, entry.shape.len()));
            }
            let [d_out, d_in] = [entry.shape[0], entry.shape[1]];
            let td = entry.to_tensor_data::<2>(&w_key)?;
            // Load as [d_out, d_in], then transpose to [d_in, d_out] for burn.
            let _ = d_out; // shape embedded in td; used only for clarity
            let _ = d_in;
            let tensor = Tensor::<B, 2>::from_data(td, device).transpose();
            Param::initialized(ParamId::new(), tensor)
        };

        let bias = if self.has(&b_key) {
            Some(self.param::<B, 1>(&b_key, device)?)
        } else {
            None
        };

        Ok(LinearRecord { weight, bias })
    }

    /// Build an `EmbeddingRecord<B>` from `prefix.weight`.
    pub(super) fn embedding<B: Backend>(
        &self,
        prefix: &str,
        device: &B::Device,
    ) -> Result<EmbeddingRecord<B>> {
        let weight = self.param::<B, 2>(&format!("{prefix}.weight"), device)?;
        Ok(EmbeddingRecord { weight })
    }

    /// Build a `RmsNormRecord<B>` from `prefix.weight` (1-D) and a constant eps.
    pub(super) fn rms_norm<B: Backend>(
        &self,
        prefix: &str,
        _eps: f64,
        device: &B::Device,
    ) -> Result<RmsNormRecord<B>> {
        Ok(RmsNormRecord {
            weight: self.param::<B, 1>(&format!("{prefix}.weight"), device)?,
            eps: EmptyRecord::new(),
        })
    }

    /// Build a `HeadRmsNormRecord<B>` from `prefix.weight` (2-D) and a constant eps.
    pub(super) fn head_rms_norm<B: Backend>(
        &self,
        prefix: &str,
        device: &B::Device,
    ) -> Result<HeadRmsNormRecord<B>> {
        Ok(HeadRmsNormRecord {
            weight: self.param::<B, 2>(&format!("{prefix}.weight"), device)?,
            eps: EmptyRecord::new(),
        })
    }

    /// Build a `LowRankAdaLnRecord<B>`.
    pub(super) fn low_rank_adaln<B: Backend>(
        &self,
        prefix: &str,
        device: &B::Device,
    ) -> Result<LowRankAdaLnRecord<B>> {
        Ok(LowRankAdaLnRecord {
            shift_down: self.linear(format!("{prefix}.shift_down").as_str(), device)?,
            scale_down: self.linear(format!("{prefix}.scale_down").as_str(), device)?,
            gate_down: self.linear(format!("{prefix}.gate_down").as_str(), device)?,
            shift_up: self.linear(format!("{prefix}.shift_up").as_str(), device)?,
            scale_up: self.linear(format!("{prefix}.scale_up").as_str(), device)?,
            gate_up: self.linear(format!("{prefix}.gate_up").as_str(), device)?,
            eps: EmptyRecord::new(),
        })
    }

    /// Build a `SwiGluRecord<B>` from `prefix.{w1,w2,w3}`.
    pub(super) fn swiglu<B: Backend>(
        &self,
        prefix: &str,
        device: &B::Device,
    ) -> Result<SwiGluRecord<B>> {
        Ok(SwiGluRecord {
            w1: self.linear(format!("{prefix}.w1").as_str(), device)?,
            w2: self.linear(format!("{prefix}.w2").as_str(), device)?,
            w3: self.linear(format!("{prefix}.w3").as_str(), device)?,
        })
    }

    /// Build a `SelfAttentionRecord<B>`.
    pub(super) fn self_attention<B: Backend>(
        &self,
        prefix: &str,
        device: &B::Device,
    ) -> Result<SelfAttentionRecord<B>> {
        Ok(SelfAttentionRecord {
            wq: self.linear(format!("{prefix}.wq").as_str(), device)?,
            wk: self.linear(format!("{prefix}.wk").as_str(), device)?,
            wv: self.linear(format!("{prefix}.wv").as_str(), device)?,
            wo: self.linear(format!("{prefix}.wo").as_str(), device)?,
            gate: self.linear(format!("{prefix}.gate").as_str(), device)?,
            q_norm: self.head_rms_norm(format!("{prefix}.q_norm").as_str(), device)?,
            k_norm: self.head_rms_norm(format!("{prefix}.k_norm").as_str(), device)?,
            num_heads: EmptyRecord::new(),
            head_dim: EmptyRecord::new(),
            scale: EmptyRecord::new(),
        })
    }

    /// Build a `JointAttentionRecord<B>`.
    ///
    /// Optional speaker/caption KV projections are loaded only if present.
    pub(super) fn joint_attention<B: Backend>(
        &self,
        prefix: &str,
        device: &B::Device,
    ) -> Result<JointAttentionRecord<B>> {
        let opt_linear = |suffix: &str| -> Result<Option<LinearRecord<B>>> {
            let key = format!("{prefix}.{suffix}.weight");
            if self.has(&key) {
                Ok(Some(
                    self.linear(format!("{prefix}.{suffix}").as_str(), device)?,
                ))
            } else {
                Ok(None)
            }
        };

        Ok(JointAttentionRecord {
            wq: self.linear(format!("{prefix}.wq").as_str(), device)?,
            wk: self.linear(format!("{prefix}.wk").as_str(), device)?,
            wv: self.linear(format!("{prefix}.wv").as_str(), device)?,
            wk_text: self.linear(format!("{prefix}.wk_text").as_str(), device)?,
            wv_text: self.linear(format!("{prefix}.wv_text").as_str(), device)?,
            wk_speaker: opt_linear("wk_speaker")?,
            wv_speaker: opt_linear("wv_speaker")?,
            wk_caption: opt_linear("wk_caption")?,
            wv_caption: opt_linear("wv_caption")?,
            gate: self.linear(format!("{prefix}.gate").as_str(), device)?,
            wo: self.linear(format!("{prefix}.wo").as_str(), device)?,
            q_norm: self.head_rms_norm(format!("{prefix}.q_norm").as_str(), device)?,
            k_norm: self.head_rms_norm(format!("{prefix}.k_norm").as_str(), device)?,
            num_heads: EmptyRecord::new(),
            head_dim: EmptyRecord::new(),
            scale: EmptyRecord::new(),
        })
    }

    /// Build a `TextBlockRecord<B>`.
    pub(super) fn text_block<B: Backend>(
        &self,
        prefix: &str,
        device: &B::Device,
    ) -> Result<TextBlockRecord<B>> {
        Ok(TextBlockRecord {
            attention_norm: self.rms_norm(
                format!("{prefix}.attention_norm").as_str(),
                1e-6,
                device,
            )?,
            attention: self.self_attention(format!("{prefix}.attention").as_str(), device)?,
            mlp_norm: self.rms_norm(format!("{prefix}.mlp_norm").as_str(), 1e-6, device)?,
            mlp: self.swiglu(format!("{prefix}.mlp").as_str(), device)?,
            dropout: EmptyRecord::new(),
        })
    }

    /// Build a `TextEncoderRecord<B>` (used for both text and caption encoders).
    pub(super) fn text_encoder<B: Backend>(
        &self,
        prefix: &str,
        num_layers: usize,
        device: &B::Device,
    ) -> Result<TextEncoderRecord<B>> {
        let blocks = (0..num_layers)
            .map(|i| self.text_block(format!("{prefix}.blocks.{i}").as_str(), device))
            .collect::<Result<Vec<_>>>()?;

        Ok(TextEncoderRecord {
            text_embedding: self.embedding(format!("{prefix}.text_embedding").as_str(), device)?,
            blocks,
            head_dim: EmptyRecord::new(),
        })
    }

    /// Build a `ReferenceLatentEncoderRecord<B>`.
    pub(super) fn reference_latent_encoder<B: Backend>(
        &self,
        prefix: &str,
        num_layers: usize,
        device: &B::Device,
    ) -> Result<ReferenceLatentEncoderRecord<B>> {
        let blocks = (0..num_layers)
            .map(|i| self.text_block(format!("{prefix}.blocks.{i}").as_str(), device))
            .collect::<Result<Vec<_>>>()?;

        Ok(ReferenceLatentEncoderRecord {
            in_proj: self.linear(format!("{prefix}.in_proj").as_str(), device)?,
            blocks,
            head_dim: EmptyRecord::new(),
        })
    }

    /// Build a `DiffusionBlockRecord<B>`.
    pub(super) fn diffusion_block<B: Backend>(
        &self,
        prefix: &str,
        device: &B::Device,
    ) -> Result<DiffusionBlockRecord<B>> {
        Ok(DiffusionBlockRecord {
            attention: self.joint_attention(format!("{prefix}.attention").as_str(), device)?,
            mlp: self.swiglu(format!("{prefix}.mlp").as_str(), device)?,
            attention_adaln: self
                .low_rank_adaln(format!("{prefix}.attention_adaln").as_str(), device)?,
            mlp_adaln: self.low_rank_adaln(format!("{prefix}.mlp_adaln").as_str(), device)?,
            dropout: EmptyRecord::new(),
        })
    }

    /// Build a `CondModuleRecord<B>`.
    pub(super) fn cond_module<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<CondModuleRecord<B>> {
        Ok(CondModuleRecord {
            linear0: self.linear("cond_module.linear0", device)?,
            linear1: self.linear("cond_module.linear1", device)?,
            linear2: self.linear("cond_module.linear2", device)?,
        })
    }

    /// Assemble the full `TextToLatentRfDiTRecord<B>`.
    pub(super) fn build_model_record<B: Backend>(
        &self,
        cfg: &ModelConfig,
        device: &B::Device,
    ) -> Result<TextToLatentRfDiTRecord<B>> {
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
            .map(|i| self.diffusion_block(format!("blocks.{i}").as_str(), device))
            .collect::<Result<Vec<_>>>()?;

        let out_norm = self.rms_norm("out_norm", cfg.norm_eps, device)?;
        let out_proj = self.linear("out_proj", device)?;

        Ok(TextToLatentRfDiTRecord {
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
mod tests {
    use super::*;
    use crate::weights::test_helpers::*;
    use burn::backend::NdArray;
    use safetensors::Dtype;

    type B = NdArray<f32>;

    #[test]
    fn linear_transpose_correctness() {
        // PyTorch linear weight: [d_out=2, d_in=3]
        // Values: row0=[1,2,3], row1=[4,5,6]
        let w_vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let file = write_safetensors(
            &[("fc.weight", f32_bytes(&w_vals), Dtype::F32, vec![2, 3])],
            &test_config_json(),
        );
        let store = TensorStore::load(file.path()).unwrap();
        let record = store.linear::<B>("fc", &Default::default()).unwrap();
        let weight = record.weight.val();
        // After transpose: burn weight should be [d_in=3, d_out=2]
        assert_eq!(weight.shape().dims(), [3, 2]);
        // Column-major readout: transposed rows become columns
        let data: Vec<f32> = weight.to_data().to_vec().unwrap();
        // Original [2,3]: [[1,2,3],[4,5,6]]
        // Transposed [3,2]: [[1,4],[2,5],[3,6]]
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn linear_with_bias() {
        let w_vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_vals = vec![0.1f32, 0.2];
        let file = write_safetensors(
            &[
                ("fc.weight", f32_bytes(&w_vals), Dtype::F32, vec![2, 2]),
                ("fc.bias", f32_bytes(&b_vals), Dtype::F32, vec![2]),
            ],
            &test_config_json(),
        );
        let store = TensorStore::load(file.path()).unwrap();
        let record = store.linear::<B>("fc", &Default::default()).unwrap();
        assert!(record.bias.is_some());
        let bias: Vec<f32> = record.bias.unwrap().val().to_data().to_vec().unwrap();
        assert_eq!(bias, b_vals);
    }

    #[test]
    fn linear_without_bias() {
        let w_vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let file = write_safetensors(
            &[("fc.weight", f32_bytes(&w_vals), Dtype::F32, vec![2, 2])],
            &test_config_json(),
        );
        let store = TensorStore::load(file.path()).unwrap();
        let record = store.linear::<B>("fc", &Default::default()).unwrap();
        assert!(record.bias.is_none());
    }

    #[test]
    fn embedding_weight_no_transpose() {
        // Embedding weight should NOT be transposed (same shape as stored)
        let vals: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let file = write_safetensors(
            &[("emb.weight", f32_bytes(&vals), Dtype::F32, vec![4, 3])],
            &test_config_json(),
        );
        let store = TensorStore::load(file.path()).unwrap();
        let record = store.embedding::<B>("emb", &Default::default()).unwrap();
        let weight = record.weight.val();
        assert_eq!(weight.shape().dims(), [4, 3]);
        let data: Vec<f32> = weight.to_data().to_vec().unwrap();
        assert_eq!(data, vals);
    }

    #[test]
    fn rms_norm_weight_1d() {
        let vals = vec![1.0f32, 1.0, 1.0];
        let file = write_safetensors(
            &[("norm.weight", f32_bytes(&vals), Dtype::F32, vec![3])],
            &test_config_json(),
        );
        let store = TensorStore::load(file.path()).unwrap();
        let record = store
            .rms_norm::<B>("norm", 1e-6, &Default::default())
            .unwrap();
        let w: Vec<f32> = record.weight.val().to_data().to_vec().unwrap();
        assert_eq!(w, vals);
    }

    #[test]
    fn missing_weight_errors() {
        let file = write_safetensors(
            &[("other.weight", f32_bytes(&[1.0]), Dtype::F32, vec![1])],
            &test_config_json(),
        );
        let store = TensorStore::load(file.path()).unwrap();
        let err = store.linear::<B>("fc", &Default::default());
        assert!(err.is_err());
    }

    #[test]
    fn bf16_linear_transpose() {
        // Test that bf16 weights are correctly decoded and transposed
        let vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let file = write_safetensors(
            &[("fc.weight", bf16_bytes(&vals), Dtype::BF16, vec![2, 2])],
            &test_config_json(),
        );
        let store = TensorStore::load(file.path()).unwrap();
        let record = store.linear::<B>("fc", &Default::default()).unwrap();
        let weight = record.weight.val();
        assert_eq!(weight.shape().dims(), [2, 2]);
        // [[1,2],[3,4]] transposed → [[1,3],[2,4]]
        let data: Vec<f32> = weight.to_data().to_vec().unwrap();
        assert!((data[0] - 1.0).abs() < 0.02);
        assert!((data[1] - 3.0).abs() < 0.02);
        assert!((data[2] - 2.0).abs() < 0.02);
        assert!((data[3] - 4.0).abs() < 0.02);
    }
}
