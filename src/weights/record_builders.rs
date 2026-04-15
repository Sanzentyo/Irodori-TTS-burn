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
    use crate::config::{tiny_caption_config, tiny_model_config};
    use crate::model::dit::TextToLatentRfDiT;
    use crate::weights::test_helpers::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::module::Module;
    use safetensors::Dtype;
    use tempfile::NamedTempFile;

    type B = NdArray<f32>;

    fn default_device() -> NdArrayDevice {
        NdArrayDevice::default()
    }

    // -----------------------------------------------------------------------
    // Unit tests for primitive builders
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Integration tests for build_model_record
    // -----------------------------------------------------------------------

    /// Sentinel fill value for a given prefix, used to verify key→tensor mapping.
    fn sentinel_for(prefix: &str) -> f32 {
        // Simple hash-like sentinel: sum of bytes mod 100 / 100
        let sum: u32 = prefix.bytes().map(u32::from).sum();
        (sum % 100) as f32 / 100.0
    }

    /// Add a linear weight entry (PyTorch layout: [d_out, d_in]) with sentinel fill.
    fn push_linear(
        entries: &mut Vec<(String, Vec<u8>, Dtype, Vec<usize>)>,
        prefix: &str,
        d_out: usize,
        d_in: usize,
        with_bias: bool,
    ) {
        let val = sentinel_for(prefix);
        let n = d_out * d_in;
        entries.push((
            format!("{prefix}.weight"),
            f32_bytes(&vec![val; n]),
            Dtype::F32,
            vec![d_out, d_in],
        ));
        if with_bias {
            entries.push((
                format!("{prefix}.bias"),
                f32_bytes(&vec![val + 0.5; d_out]),
                Dtype::F32,
                vec![d_out],
            ));
        }
    }

    /// Add an embedding weight entry (no transpose, [vocab, dim]).
    fn push_embedding(
        entries: &mut Vec<(String, Vec<u8>, Dtype, Vec<usize>)>,
        prefix: &str,
        vocab: usize,
        dim: usize,
    ) {
        let val = sentinel_for(prefix);
        entries.push((
            format!("{prefix}.weight"),
            f32_bytes(&vec![val; vocab * dim]),
            Dtype::F32,
            vec![vocab, dim],
        ));
    }

    /// Add a 1-D RmsNorm weight.
    fn push_rms_norm(
        entries: &mut Vec<(String, Vec<u8>, Dtype, Vec<usize>)>,
        prefix: &str,
        dim: usize,
    ) {
        entries.push((
            format!("{prefix}.weight"),
            f32_bytes(&vec![1.0; dim]),
            Dtype::F32,
            vec![dim],
        ));
    }

    /// Add a 2-D HeadRmsNorm weight [heads, head_dim].
    fn push_head_rms_norm(
        entries: &mut Vec<(String, Vec<u8>, Dtype, Vec<usize>)>,
        prefix: &str,
        heads: usize,
        head_dim: usize,
    ) {
        entries.push((
            format!("{prefix}.weight"),
            f32_bytes(&vec![1.0; heads * head_dim]),
            Dtype::F32,
            vec![heads, head_dim],
        ));
    }

    /// Add SwiGLU MLP weights: w1 [hidden, dim], w2 [dim, hidden], w3 [hidden, dim].
    fn push_swiglu(
        entries: &mut Vec<(String, Vec<u8>, Dtype, Vec<usize>)>,
        prefix: &str,
        dim: usize,
        hidden: usize,
    ) {
        push_linear(entries, &format!("{prefix}.w1"), hidden, dim, false);
        push_linear(entries, &format!("{prefix}.w2"), dim, hidden, false);
        push_linear(entries, &format!("{prefix}.w3"), hidden, dim, false);
    }

    /// Add a LowRankAdaLn: 3 down [rank, model_dim] no bias + 3 up [model_dim, rank] with bias.
    fn push_adaln(
        entries: &mut Vec<(String, Vec<u8>, Dtype, Vec<usize>)>,
        prefix: &str,
        model_dim: usize,
        rank: usize,
    ) {
        for name in &["shift_down", "scale_down", "gate_down"] {
            push_linear(entries, &format!("{prefix}.{name}"), rank, model_dim, false);
        }
        for name in &["shift_up", "scale_up", "gate_up"] {
            push_linear(entries, &format!("{prefix}.{name}"), model_dim, rank, true);
        }
    }

    /// Add SelfAttention weights for a text/speaker encoder block.
    fn push_self_attention(
        entries: &mut Vec<(String, Vec<u8>, Dtype, Vec<usize>)>,
        prefix: &str,
        dim: usize,
        heads: usize,
        head_dim: usize,
    ) {
        for name in &["wq", "wk", "wv", "wo", "gate"] {
            push_linear(entries, &format!("{prefix}.{name}"), dim, dim, false);
        }
        push_head_rms_norm(entries, &format!("{prefix}.q_norm"), heads, head_dim);
        push_head_rms_norm(entries, &format!("{prefix}.k_norm"), heads, head_dim);
    }

    /// Add JointAttention weights for a diffusion block.
    fn push_joint_attention(
        entries: &mut Vec<(String, Vec<u8>, Dtype, Vec<usize>)>,
        prefix: &str,
        model_dim: usize,
        num_heads: usize,
        head_dim: usize,
        is_speaker: bool,
    ) {
        for name in &["wq", "wk", "wv", "wk_text", "wv_text", "gate", "wo"] {
            push_linear(
                entries,
                &format!("{prefix}.{name}"),
                model_dim,
                model_dim,
                false,
            );
        }
        if is_speaker {
            push_linear(
                entries,
                &format!("{prefix}.wk_speaker"),
                model_dim,
                model_dim,
                false,
            );
            push_linear(
                entries,
                &format!("{prefix}.wv_speaker"),
                model_dim,
                model_dim,
                false,
            );
        } else {
            push_linear(
                entries,
                &format!("{prefix}.wk_caption"),
                model_dim,
                model_dim,
                false,
            );
            push_linear(
                entries,
                &format!("{prefix}.wv_caption"),
                model_dim,
                model_dim,
                false,
            );
        }
        push_head_rms_norm(entries, &format!("{prefix}.q_norm"), num_heads, head_dim);
        push_head_rms_norm(entries, &format!("{prefix}.k_norm"), num_heads, head_dim);
    }

    /// Add a TextBlock: attention_norm, self_attention, mlp_norm, mlp.
    fn push_text_block(
        entries: &mut Vec<(String, Vec<u8>, Dtype, Vec<usize>)>,
        prefix: &str,
        dim: usize,
        heads: usize,
        head_dim: usize,
        mlp_hidden: usize,
    ) {
        push_rms_norm(entries, &format!("{prefix}.attention_norm"), dim);
        push_self_attention(
            entries,
            &format!("{prefix}.attention"),
            dim,
            heads,
            head_dim,
        );
        push_rms_norm(entries, &format!("{prefix}.mlp_norm"), dim);
        push_swiglu(entries, &format!("{prefix}.mlp"), dim, mlp_hidden);
    }

    /// Build a complete safetensors file for `tiny_model_config()` (speaker mode).
    fn build_speaker_safetensors() -> (NamedTempFile, ModelConfig) {
        let cfg = tiny_model_config();
        let mut entries = Vec::new();

        let model_dim = cfg.model_dim; // 32
        let num_heads = cfg.num_heads; // 4
        let head_dim = cfg.head_dim(); // 8
        let text_dim = cfg.text_dim; // 16
        let text_heads = cfg.text_heads; // 2
        let text_head_dim = text_dim / text_heads; // 8
        let text_mlp_hidden = (text_dim as f64 * cfg.text_mlp_ratio()) as usize; // 32
        let text_vocab = cfg.text_vocab_size; // 64
        let speaker_dim = cfg.speaker_dim.unwrap(); // 16
        let speaker_heads = cfg.speaker_heads.unwrap(); // 2
        let speaker_head_dim = speaker_dim / speaker_heads; // 8
        let speaker_mlp_hidden = (speaker_dim as f64 * cfg.speaker_mlp_ratio()) as usize; // 32
        let speaker_patched_latent_dim = cfg.speaker_patched_latent_dim(); // 8
        let patched_latent_dim = cfg.patched_latent_dim(); // 8
        let t_embed_dim = cfg.timestep_embed_dim; // 32
        let adaln_rank = cfg.adaln_rank; // 16
        let diff_mlp_hidden = (model_dim as f64 * cfg.mlp_ratio) as usize; // 64

        // --- Text encoder ---
        push_embedding(
            &mut entries,
            "text_encoder.text_embedding",
            text_vocab,
            text_dim,
        );
        push_text_block(
            &mut entries,
            "text_encoder.blocks.0",
            text_dim,
            text_heads,
            text_head_dim,
            text_mlp_hidden,
        );
        push_rms_norm(&mut entries, "text_norm", text_dim);

        // --- Speaker encoder ---
        push_linear(
            &mut entries,
            "speaker_encoder.in_proj",
            speaker_dim,
            speaker_patched_latent_dim,
            true,
        );
        push_text_block(
            &mut entries,
            "speaker_encoder.blocks.0",
            speaker_dim,
            speaker_heads,
            speaker_head_dim,
            speaker_mlp_hidden,
        );
        push_rms_norm(&mut entries, "speaker_norm", speaker_dim);

        // --- Cond module (no bias) ---
        push_linear(
            &mut entries,
            "cond_module.linear0",
            model_dim,
            t_embed_dim,
            false,
        );
        push_linear(
            &mut entries,
            "cond_module.linear1",
            model_dim,
            model_dim,
            false,
        );
        push_linear(
            &mut entries,
            "cond_module.linear2",
            model_dim * 3,
            model_dim,
            false,
        );

        // --- in_proj (with bias) ---
        push_linear(&mut entries, "in_proj", model_dim, patched_latent_dim, true);

        // --- Diffusion block 0 ---
        push_joint_attention(
            &mut entries,
            "blocks.0.attention",
            model_dim,
            num_heads,
            head_dim,
            true,
        );
        push_swiglu(&mut entries, "blocks.0.mlp", model_dim, diff_mlp_hidden);
        push_adaln(
            &mut entries,
            "blocks.0.attention_adaln",
            model_dim,
            adaln_rank,
        );
        push_adaln(&mut entries, "blocks.0.mlp_adaln", model_dim, adaln_rank);

        // --- Output ---
        push_rms_norm(&mut entries, "out_norm", model_dim);
        push_linear(
            &mut entries,
            "out_proj",
            patched_latent_dim,
            model_dim,
            true,
        );

        let config_json = serde_json::to_string(&cfg).unwrap();
        let refs: Vec<(&str, Vec<u8>, Dtype, Vec<usize>)> = entries
            .iter()
            .map(|(n, d, dt, s)| (n.as_str(), d.clone(), *dt, s.clone()))
            .collect();
        let file = write_safetensors(&refs, &config_json);
        (file, cfg)
    }

    /// Build a complete safetensors file for `tiny_caption_config()`.
    fn build_caption_safetensors() -> (NamedTempFile, ModelConfig) {
        let cfg = tiny_caption_config();
        let mut entries = Vec::new();

        let model_dim = cfg.model_dim;
        let num_heads = cfg.num_heads;
        let head_dim = cfg.head_dim();
        let text_dim = cfg.text_dim;
        let text_heads = cfg.text_heads;
        let text_head_dim = text_dim / text_heads;
        let text_mlp_hidden = (text_dim as f64 * cfg.text_mlp_ratio()) as usize;
        let text_vocab = cfg.text_vocab_size;
        let caption_dim = cfg.caption_dim();
        let caption_heads = cfg.caption_heads();
        let caption_head_dim = caption_dim / caption_heads;
        let caption_mlp_hidden = (caption_dim as f64 * cfg.caption_mlp_ratio()) as usize;
        let caption_vocab = cfg.caption_vocab_size();
        let patched_latent_dim = cfg.patched_latent_dim();
        let t_embed_dim = cfg.timestep_embed_dim;
        let adaln_rank = cfg.adaln_rank;
        let diff_mlp_hidden = (model_dim as f64 * cfg.mlp_ratio) as usize;

        // --- Text encoder ---
        push_embedding(
            &mut entries,
            "text_encoder.text_embedding",
            text_vocab,
            text_dim,
        );
        push_text_block(
            &mut entries,
            "text_encoder.blocks.0",
            text_dim,
            text_heads,
            text_head_dim,
            text_mlp_hidden,
        );
        push_rms_norm(&mut entries, "text_norm", text_dim);

        // --- Caption encoder (same structure as text encoder) ---
        push_embedding(
            &mut entries,
            "caption_encoder.text_embedding",
            caption_vocab,
            caption_dim,
        );
        push_text_block(
            &mut entries,
            "caption_encoder.blocks.0",
            caption_dim,
            caption_heads,
            caption_head_dim,
            caption_mlp_hidden,
        );
        push_rms_norm(&mut entries, "caption_norm", caption_dim);

        // --- Cond module (no bias) ---
        push_linear(
            &mut entries,
            "cond_module.linear0",
            model_dim,
            t_embed_dim,
            false,
        );
        push_linear(
            &mut entries,
            "cond_module.linear1",
            model_dim,
            model_dim,
            false,
        );
        push_linear(
            &mut entries,
            "cond_module.linear2",
            model_dim * 3,
            model_dim,
            false,
        );

        // --- in_proj (with bias) ---
        push_linear(&mut entries, "in_proj", model_dim, patched_latent_dim, true);

        // --- Diffusion block 0 ---
        push_joint_attention(
            &mut entries,
            "blocks.0.attention",
            model_dim,
            num_heads,
            head_dim,
            false,
        );
        push_swiglu(&mut entries, "blocks.0.mlp", model_dim, diff_mlp_hidden);
        push_adaln(
            &mut entries,
            "blocks.0.attention_adaln",
            model_dim,
            adaln_rank,
        );
        push_adaln(&mut entries, "blocks.0.mlp_adaln", model_dim, adaln_rank);

        // --- Output ---
        push_rms_norm(&mut entries, "out_norm", model_dim);
        push_linear(
            &mut entries,
            "out_proj",
            patched_latent_dim,
            model_dim,
            true,
        );

        let config_json = serde_json::to_string(&cfg).unwrap();
        let refs: Vec<(&str, Vec<u8>, Dtype, Vec<usize>)> = entries
            .iter()
            .map(|(n, d, dt, s)| (n.as_str(), d.clone(), *dt, s.clone()))
            .collect();
        let file = write_safetensors(&refs, &config_json);
        (file, cfg)
    }

    #[test]
    fn build_model_record_speaker_mode() {
        let (file, cfg) = build_speaker_safetensors();
        let store = TensorStore::load(file.path()).unwrap();
        let dev = default_device();

        let record = store.build_model_record::<B>(&cfg, &dev).unwrap();

        // Verify aux_conditioner is Speaker variant
        let aux = record.aux_conditioner.as_ref().expect("aux must be Some");
        assert!(
            matches!(aux, crate::model::dit::AuxConditionerRecord::Speaker(_)),
            "expected Speaker variant"
        );

        // Load into a real model to verify all shapes match
        let model = TextToLatentRfDiT::<B>::new(&cfg, &dev);
        let _loaded = model.load_record(record);

        // Spot-check sentinel value: in_proj weight should be filled with sentinel_for("in_proj")
        let expected_val = sentinel_for("in_proj");
        let in_proj_w: Vec<f32> = _loaded.in_proj.weight.val().to_data().to_vec().unwrap();
        assert!(
            in_proj_w.iter().all(|&v| (v - expected_val).abs() < 1e-6),
            "in_proj weight sentinel mismatch"
        );

        // Spot-check: text_encoder embedding should NOT be transposed
        let emb_val = sentinel_for("text_encoder.text_embedding");
        let emb_w: Vec<f32> = _loaded
            .text_encoder
            .text_embedding
            .weight
            .val()
            .to_data()
            .to_vec()
            .unwrap();
        assert!(
            emb_w.iter().all(|&v| (v - emb_val).abs() < 1e-6),
            "text_encoder embedding sentinel mismatch"
        );

        // Spot-check: in_proj bias should exist
        assert!(_loaded.in_proj.bias.is_some(), "in_proj should have bias");
    }

    #[test]
    fn build_model_record_caption_mode() {
        let (file, cfg) = build_caption_safetensors();
        let store = TensorStore::load(file.path()).unwrap();
        let dev = default_device();

        let record = store.build_model_record::<B>(&cfg, &dev).unwrap();

        // Verify aux_conditioner is Caption variant
        let aux = record.aux_conditioner.as_ref().expect("aux must be Some");
        assert!(
            matches!(aux, crate::model::dit::AuxConditionerRecord::Caption(_)),
            "expected Caption variant"
        );

        // Load into a real model to verify all shapes match
        let model = TextToLatentRfDiT::<B>::new(&cfg, &dev);
        let _loaded = model.load_record(record);

        // Spot-check: cond_module.linear2 weight should be filled with sentinel
        let cond2_val = sentinel_for("cond_module.linear2");
        let cond2_w: Vec<f32> = _loaded
            .cond_module
            .linear2
            .weight
            .val()
            .to_data()
            .to_vec()
            .unwrap();
        assert!(
            cond2_w.iter().all(|&v| (v - cond2_val).abs() < 1e-6),
            "cond_module.linear2 weight sentinel mismatch"
        );

        // Spot-check: out_proj bias should exist
        assert!(_loaded.out_proj.bias.is_some(), "out_proj should have bias");
    }

    #[test]
    fn build_model_record_wrong_config_errors() {
        // Speaker-mode safetensors + caption config → should fail on missing caption keys
        let (file, _speaker_cfg) = build_speaker_safetensors();
        let store = TensorStore::load(file.path()).unwrap();
        let caption_cfg = tiny_caption_config();
        let dev = default_device();

        let err = store.build_model_record::<B>(&caption_cfg, &dev);
        assert!(
            err.is_err(),
            "should fail when config doesn't match weights"
        );
        let err_msg = format!("{}", err.err().expect("should be Err"));
        assert!(
            err_msg.contains("caption_encoder"),
            "error should mention missing caption_encoder key, got: {err_msg}"
        );
    }

    #[test]
    fn build_model_record_speaker_sentinel_values_round_trip() {
        // Verify that linear weights are correctly transposed and bias values survive.
        let (file, cfg) = build_speaker_safetensors();
        let store = TensorStore::load(file.path()).unwrap();
        let dev = default_device();
        let record = store.build_model_record::<B>(&cfg, &dev).unwrap();
        let model = TextToLatentRfDiT::<B>::new(&cfg, &dev).load_record(record);

        // out_proj bias sentinel: sentinel_for("out_proj") + 0.5
        let out_proj_bias_val = sentinel_for("out_proj") + 0.5;
        let bias_data: Vec<f32> = model
            .out_proj
            .bias
            .as_ref()
            .expect("out_proj bias")
            .val()
            .to_data()
            .to_vec()
            .unwrap();
        assert!(
            bias_data
                .iter()
                .all(|&v| (v - out_proj_bias_val).abs() < 1e-6),
            "out_proj bias sentinel mismatch: expected {out_proj_bias_val}, got {:?}",
            &bias_data[..bias_data.len().min(4)]
        );

        // AdaLN up bias sentinel: shift_up has bias
        let adaln_prefix = "blocks.0.attention_adaln.shift_up";
        let adaln_bias_val = sentinel_for(adaln_prefix) + 0.5;
        let adaln_bias: Vec<f32> = model.blocks[0]
            .attention_adaln
            .shift_up
            .bias
            .as_ref()
            .expect("adaln shift_up bias")
            .val()
            .to_data()
            .to_vec()
            .unwrap();
        assert!(
            adaln_bias
                .iter()
                .all(|&v| (v - adaln_bias_val).abs() < 1e-6),
            "adaln shift_up bias sentinel mismatch"
        );
    }
}
