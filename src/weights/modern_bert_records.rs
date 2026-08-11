//! Strict safetensors-to-Burn record mapping for the ModernBERT conditioner in
//! `Aratako/Irodori-TTS-v4-Small`.
//!
//! The mapping and dimensions below were checked against `model.safetensors`
//! at revision `e4aaac4df355ff560dcd35e0dae272c3a759317b`. PyTorch linear
//! weights are validated in `[output, input]` layout before the shared
//! [`TensorStore::linear`] helper transposes them to Burn's `[input, output]`
//! layout.

use burn::{
    module::EmptyRecord,
    nn::{LayerNormRecord, LinearRecord},
    tensor::backend::Backend,
};

use super::tensor_store::TensorStore;
use crate::{
    error::{IrodoriError, Result},
    model::{
        modern_bert::{
            ModernBertAttentionRecord, ModernBertEmbeddingsRecord, ModernBertEncoderLayerRecord,
            ModernBertMlpRecord, ModernBertModelRecord, PretrainedConditionProjectorRecord,
            PretrainedTextBackboneRecord, SharedModernBertConditionerRecord,
            V4_BACKBONE_WEIGHT_PREFIX, V4_CAPTION_PROJECTOR_WEIGHT_PREFIX,
            V4_TEXT_PROJECTOR_WEIGHT_PREFIX,
        },
        norm::RmsNormRecord,
    },
};

#[derive(Clone, Copy, Debug)]
struct ModernBertRecordSpec {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
}

impl ModernBertRecordSpec {
    const V4_SMALL: Self = Self {
        vocab_size: 102_400,
        hidden_size: 768,
        intermediate_size: 3_072,
        num_hidden_layers: 25,
    };
}

#[derive(Clone, Copy, Debug)]
struct ProjectorRecordSpec {
    backbone_dim: usize,
    output_dim: usize,
    hidden_dim: usize,
}

impl ProjectorRecordSpec {
    const V4_SMALL: Self = Self {
        backbone_dim: 768,
        output_dim: 512,
        hidden_dim: 1_024,
    };
}

impl TensorStore {
    /// Build the shared ModernBERT backbone record embedded in v4-Small.
    pub(super) fn v4_modern_bert_backbone<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<PretrainedTextBackboneRecord<B>> {
        self.modern_bert_backbone_with_spec(
            V4_BACKBONE_WEIGHT_PREFIX,
            ModernBertRecordSpec::V4_SMALL,
            device,
        )
    }

    /// Build either the `text_encoder` or `caption_encoder` v4-Small residual
    /// projector. The prefix must identify one of those two checkpoint trees.
    pub(super) fn v4_condition_projector<B: Backend>(
        &self,
        prefix: &str,
        device: &B::Device,
    ) -> Result<PretrainedConditionProjectorRecord<B>> {
        if prefix != V4_TEXT_PROJECTOR_WEIGHT_PREFIX && prefix != V4_CAPTION_PROJECTOR_WEIGHT_PREFIX
        {
            return Err(IrodoriError::Weight(format!(
                "unsupported v4 condition projector prefix: {prefix}"
            )));
        }
        self.condition_projector_with_spec(prefix, ProjectorRecordSpec::V4_SMALL, device)
    }

    /// Assemble the shared backbone plus independent text/caption projector
    /// records exactly as nested in the v4-Small state dict.
    pub fn v4_modern_bert_conditioner<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<SharedModernBertConditionerRecord<B>> {
        Ok(SharedModernBertConditionerRecord {
            pretrained_text_backbone: self.v4_modern_bert_backbone(device)?,
            text_encoder: self.v4_condition_projector(V4_TEXT_PROJECTOR_WEIGHT_PREFIX, device)?,
            caption_encoder: self
                .v4_condition_projector(V4_CAPTION_PROJECTOR_WEIGHT_PREFIX, device)?,
        })
    }

    fn modern_bert_backbone_with_spec<B: Backend>(
        &self,
        prefix: &str,
        spec: ModernBertRecordSpec,
        device: &B::Device,
    ) -> Result<PretrainedTextBackboneRecord<B>> {
        let embeddings_prefix = format!("{prefix}.embeddings");
        self.require_shape(
            &format!("{embeddings_prefix}.tok_embeddings.weight"),
            &[spec.vocab_size, spec.hidden_size],
        )?;
        let embeddings = ModernBertEmbeddingsRecord {
            tok_embeddings: self
                .embedding(&format!("{embeddings_prefix}.tok_embeddings"), device)?,
            norm: self.weight_only_layer_norm(
                &format!("{embeddings_prefix}.norm"),
                spec.hidden_size,
                device,
            )?,
        };

        let layers = (0..spec.num_hidden_layers)
            .map(|index| self.modern_bert_layer(prefix, index, spec, device))
            .collect::<Result<Vec<_>>>()?;

        Ok(PretrainedTextBackboneRecord {
            backbone: ModernBertModelRecord {
                embeddings,
                layers,
                final_norm: self.weight_only_layer_norm(
                    &format!("{prefix}.final_norm"),
                    spec.hidden_size,
                    device,
                )?,
                head_dim: EmptyRecord::new(),
                max_position_embeddings: EmptyRecord::new(),
                sliding_half_window: EmptyRecord::new(),
                full_rope_theta: EmptyRecord::new(),
                sliding_rope_theta: EmptyRecord::new(),
                norm_eps: EmptyRecord::new(),
            },
        })
    }

    fn modern_bert_layer<B: Backend>(
        &self,
        backbone_prefix: &str,
        index: usize,
        spec: ModernBertRecordSpec,
        device: &B::Device,
    ) -> Result<ModernBertEncoderLayerRecord<B>> {
        let prefix = format!("{backbone_prefix}.layers.{index}");
        let attn_norm = match modern_bert_attn_norm_prefix(&prefix, index) {
            Some(prefix) => Some(self.weight_only_layer_norm(&prefix, spec.hidden_size, device)?),
            None => None,
        };

        Ok(ModernBertEncoderLayerRecord {
            attn_norm,
            attn: ModernBertAttentionRecord {
                wqkv: self.checked_linear(
                    &format!("{prefix}.attn.Wqkv"),
                    spec.hidden_size * 3,
                    spec.hidden_size,
                    None,
                    device,
                )?,
                wo: self.checked_linear(
                    &format!("{prefix}.attn.Wo"),
                    spec.hidden_size,
                    spec.hidden_size,
                    None,
                    device,
                )?,
                num_heads: EmptyRecord::new(),
                head_dim: EmptyRecord::new(),
            },
            mlp_norm: self.weight_only_layer_norm(
                &format!("{prefix}.mlp_norm"),
                spec.hidden_size,
                device,
            )?,
            mlp: ModernBertMlpRecord {
                wi: self.checked_linear(
                    &format!("{prefix}.mlp.Wi"),
                    spec.intermediate_size * 2,
                    spec.hidden_size,
                    None,
                    device,
                )?,
                wo: self.checked_linear(
                    &format!("{prefix}.mlp.Wo"),
                    spec.hidden_size,
                    spec.intermediate_size,
                    None,
                    device,
                )?,
                intermediate_size: EmptyRecord::new(),
            },
            layer_type: EmptyRecord::new(),
        })
    }

    fn condition_projector_with_spec<B: Backend>(
        &self,
        prefix: &str,
        spec: ProjectorRecordSpec,
        device: &B::Device,
    ) -> Result<PretrainedConditionProjectorRecord<B>> {
        Ok(PretrainedConditionProjectorRecord {
            projector: self.checked_linear(
                &format!("{prefix}.projector"),
                spec.output_dim,
                spec.backbone_dim,
                Some(spec.output_dim),
                device,
            )?,
            residual_norm: Some(self.checked_rms_norm(
                &format!("{prefix}.residual_norm"),
                spec.backbone_dim,
                device,
            )?),
            residual_up: Some(self.checked_linear(
                &format!("{prefix}.residual_up"),
                spec.hidden_dim,
                spec.backbone_dim,
                Some(spec.hidden_dim),
                device,
            )?),
            residual_down: Some(self.checked_linear(
                &format!("{prefix}.residual_down"),
                spec.output_dim,
                spec.hidden_dim,
                Some(spec.output_dim),
                device,
            )?),
            // Dropout has no trainable state, so Burn erases the `Option<Dropout>`
            // field to one `EmptyRecord`; the model constructor retains `Some`.
            residual_dropout: EmptyRecord::new(),
        })
    }

    fn checked_linear<B: Backend>(
        &self,
        prefix: &str,
        output_dim: usize,
        input_dim: usize,
        bias_dim: Option<usize>,
        device: &B::Device,
    ) -> Result<LinearRecord<B>> {
        self.require_shape(&format!("{prefix}.weight"), &[output_dim, input_dim])?;
        let bias_key = format!("{prefix}.bias");
        match bias_dim {
            Some(dim) => self.require_shape(&bias_key, &[dim])?,
            None if self.has(&bias_key) => {
                return Err(IrodoriError::Shape(format!(
                    "{bias_key}: unexpected bias in bias-free ModernBERT layer"
                )));
            }
            None => {}
        }
        self.linear(prefix, device)
    }

    fn checked_rms_norm<B: Backend>(
        &self,
        prefix: &str,
        dim: usize,
        device: &B::Device,
    ) -> Result<RmsNormRecord<B>> {
        self.require_shape(&format!("{prefix}.weight"), &[dim])?;
        self.rms_norm(prefix, 1e-5, device)
    }

    fn weight_only_layer_norm<B: Backend>(
        &self,
        prefix: &str,
        dim: usize,
        device: &B::Device,
    ) -> Result<LayerNormRecord<B>> {
        let weight_key = format!("{prefix}.weight");
        self.require_shape(&weight_key, &[dim])?;
        let bias_key = format!("{prefix}.bias");
        if self.has(&bias_key) {
            return Err(IrodoriError::Shape(format!(
                "{bias_key}: unexpected bias for norm_bias=false"
            )));
        }

        Ok(LayerNormRecord {
            gamma: self.param::<B, 1>(&weight_key, device)?,
            beta: None,
            epsilon: EmptyRecord::new(),
        })
    }

    fn require_shape(&self, key: &str, expected: &[usize]) -> Result<()> {
        let actual = &self.entry(key)?.shape;
        if actual != expected {
            return Err(IrodoriError::Shape(format!(
                "{key}: expected {expected:?}, got {actual:?}"
            )));
        }
        Ok(())
    }
}

fn modern_bert_attn_norm_prefix(layer_prefix: &str, index: usize) -> Option<String> {
    (index != 0).then(|| format!("{layer_prefix}.attn_norm"))
}

#[cfg(test)]
mod tests {
    use burn::{backend::NdArray, tensor::Shape};
    use safetensors::Dtype;

    use super::*;
    use crate::weights::test_helpers::{f32_bytes, test_config_json, write_safetensors};

    type B = NdArray<f32>;
    type TestTensor = (String, Vec<u8>, Dtype, Vec<usize>);

    fn tensor(name: impl Into<String>, shape: &[usize]) -> TestTensor {
        let values = vec![0.0_f32; shape.iter().product()];
        (name.into(), f32_bytes(&values), Dtype::F32, shape.to_vec())
    }

    fn load_store(tensors: &[TestTensor]) -> (tempfile::NamedTempFile, TensorStore) {
        let borrowed = tensors
            .iter()
            .map(|(name, data, dtype, shape)| (name.as_str(), data.clone(), *dtype, shape.clone()))
            .collect::<Vec<_>>();
        let file = write_safetensors(&borrowed, &test_config_json());
        let store = TensorStore::load(file.path()).unwrap();
        (file, store)
    }

    fn tiny_backbone_tensors(prefix: &str) -> Vec<TestTensor> {
        let mut tensors = vec![
            tensor(
                format!("{prefix}.embeddings.tok_embeddings.weight"),
                &[7, 4],
            ),
            tensor(format!("{prefix}.embeddings.norm.weight"), &[4]),
            tensor(format!("{prefix}.final_norm.weight"), &[4]),
        ];
        for index in 0..2 {
            let layer = format!("{prefix}.layers.{index}");
            tensors.extend([
                tensor(format!("{layer}.attn.Wqkv.weight"), &[12, 4]),
                tensor(format!("{layer}.attn.Wo.weight"), &[4, 4]),
                tensor(format!("{layer}.mlp.Wi.weight"), &[12, 4]),
                tensor(format!("{layer}.mlp.Wo.weight"), &[4, 6]),
                tensor(format!("{layer}.mlp_norm.weight"), &[4]),
            ]);
            if index != 0 {
                tensors.push(tensor(format!("{layer}.attn_norm.weight"), &[4]));
            }
        }
        tensors
    }

    fn tiny_backbone_spec() -> ModernBertRecordSpec {
        ModernBertRecordSpec {
            vocab_size: 7,
            hidden_size: 4,
            intermediate_size: 6,
            num_hidden_layers: 2,
        }
    }

    #[test]
    fn layer_zero_omits_attention_norm_and_later_layers_require_it() {
        let prefix = "pretrained_text_backbone.backbone";
        let (_file, store) = load_store(&tiny_backbone_tensors(prefix));
        let record = store
            .modern_bert_backbone_with_spec::<B>(prefix, tiny_backbone_spec(), &Default::default())
            .unwrap();

        assert_eq!(record.backbone.layers.len(), 2);
        assert!(record.backbone.layers[0].attn_norm.is_none());
        assert!(record.backbone.layers[1].attn_norm.is_some());
        assert_eq!(
            record.backbone.layers[0].attn.wqkv.weight.val().shape(),
            Shape::new([4, 12])
        );
        assert_eq!(
            record.backbone.layers[0].mlp.wo.weight.val().shape(),
            Shape::new([6, 4])
        );
    }

    #[test]
    fn missing_later_attention_norm_reports_exact_hf_key() {
        let prefix = "pretrained_text_backbone.backbone";
        let mut tensors = tiny_backbone_tensors(prefix);
        let missing = format!("{prefix}.layers.1.attn_norm.weight");
        tensors.retain(|(name, ..)| name != &missing);
        let (_file, store) = load_store(&tensors);

        let error = match store.modern_bert_backbone_with_spec::<B>(
            prefix,
            tiny_backbone_spec(),
            &Default::default(),
        ) {
            Ok(_) => panic!("missing attention norm unexpectedly loaded"),
            Err(error) => error,
        };
        assert!(matches!(error, IrodoriError::Weight(key) if key == missing));
    }

    #[test]
    fn wrong_backbone_shape_is_rejected_before_record_construction() {
        let prefix = "pretrained_text_backbone.backbone";
        let mut tensors = tiny_backbone_tensors(prefix);
        let key = format!("{prefix}.embeddings.norm.weight");
        let entry = tensors.iter_mut().find(|(name, ..)| name == &key).unwrap();
        *entry = tensor(key.clone(), &[5]);
        let (_file, store) = load_store(&tensors);

        let error = match store.modern_bert_backbone_with_spec::<B>(
            prefix,
            tiny_backbone_spec(),
            &Default::default(),
        ) {
            Ok(_) => panic!("wrong ModernBERT shape unexpectedly loaded"),
            Err(error) => error,
        };
        assert!(
            matches!(error, IrodoriError::Shape(message) if message.contains(&key) && message.contains("[4]") && message.contains("[5]"))
        );
    }

    #[test]
    fn residual_projector_maps_all_seven_hf_tensors() {
        let prefix = "text_encoder";
        let tensors = vec![
            tensor(format!("{prefix}.projector.weight"), &[2, 3]),
            tensor(format!("{prefix}.projector.bias"), &[2]),
            tensor(format!("{prefix}.residual_norm.weight"), &[3]),
            tensor(format!("{prefix}.residual_up.weight"), &[4, 3]),
            tensor(format!("{prefix}.residual_up.bias"), &[4]),
            tensor(format!("{prefix}.residual_down.weight"), &[2, 4]),
            tensor(format!("{prefix}.residual_down.bias"), &[2]),
        ];
        let (_file, store) = load_store(&tensors);
        let record = store
            .condition_projector_with_spec::<B>(
                prefix,
                ProjectorRecordSpec {
                    backbone_dim: 3,
                    output_dim: 2,
                    hidden_dim: 4,
                },
                &Default::default(),
            )
            .unwrap();

        assert_eq!(record.projector.weight.val().shape(), Shape::new([3, 2]));
        assert!(record.projector.bias.is_some());
        assert_eq!(
            record.residual_norm.unwrap().weight.val().shape(),
            Shape::new([3])
        );
        assert_eq!(
            record.residual_up.unwrap().weight.val().shape(),
            Shape::new([3, 4])
        );
        assert_eq!(
            record.residual_down.unwrap().weight.val().shape(),
            Shape::new([4, 2])
        );
        let _dropout_record = record.residual_dropout;
    }

    #[test]
    fn layer_norm_schedule_matches_all_twenty_five_v4_layers() {
        let prefix = "pretrained_text_backbone.backbone";
        assert!(modern_bert_attn_norm_prefix(&format!("{prefix}.layers.0"), 0).is_none());
        for index in 1..25 {
            assert_eq!(
                modern_bert_attn_norm_prefix(&format!("{prefix}.layers.{index}"), index),
                Some(format!("{prefix}.layers.{index}.attn_norm"))
            );
        }
    }
}
