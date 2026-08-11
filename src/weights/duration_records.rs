//! Strict safetensors-to-Burn record mapping for the released v4-Small
//! automatic duration predictor.
//!
//! The source checkpoint stores PyTorch linear weights as `[output, input]`.
//! Every shape is checked in that layout before [`TensorStore::linear`]
//! transposes it to Burn's `[input, output]` representation.

use burn::{module::EmptyRecord, nn::LinearRecord, tensor::backend::Backend};

use super::tensor_store::TensorStore;
use crate::{
    error::{IrodoriError, Result},
    model::{
        duration::{DurationPredictorRecord, DurationSwiGluBlockRecord},
        feed_forward::SwiGluRecord,
        norm::RmsNormRecord,
    },
};

const V4_DURATION_PREFIX: &str = "duration_predictor";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DurationRecordSpec {
    text_dim: usize,
    hidden_dim: usize,
    layers: usize,
    speaker_dim: usize,
    caption_dim: usize,
}

impl DurationRecordSpec {
    /// Geometry measured from `Aratako/Irodori-TTS-v4-Small` revision
    /// `e4aaac4df355ff560dcd35e0dae272c3a759317b`.
    const V4_SMALL: Self = Self {
        text_dim: 512,
        hidden_dim: 1_024,
        layers: 3,
        speaker_dim: 768,
        caption_dim: 512,
    };
}

impl TensorStore {
    /// Build all 31 tensors in the released v4-Small duration record.
    pub(super) fn v4_duration_predictor<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<DurationPredictorRecord<B>> {
        self.duration_predictor_with_spec(V4_DURATION_PREFIX, DurationRecordSpec::V4_SMALL, device)
    }

    fn duration_predictor_with_spec<B: Backend>(
        &self,
        prefix: &str,
        spec: DurationRecordSpec,
        device: &B::Device,
    ) -> Result<DurationPredictorRecord<B>> {
        self.require_duration_shape(&format!("{prefix}.null_speaker"), &[spec.speaker_dim])?;
        self.require_duration_shape(&format!("{prefix}.null_caption"), &[spec.caption_dim])?;

        let token_blocks = (0..spec.layers)
            .map(|index| self.duration_block(prefix, index, spec, device))
            .collect::<Result<Vec<_>>>()?;

        Ok(DurationPredictorRecord {
            null_speaker: self.param::<B, 1>(&format!("{prefix}.null_speaker"), device)?,
            null_caption: self.param::<B, 1>(&format!("{prefix}.null_caption"), device)?,
            token_input_proj: self.duration_biased_linear(
                &format!("{prefix}.token_input_proj"),
                spec.hidden_dim,
                spec.text_dim,
                device,
            )?,
            token_blocks,
            token_out_norm: self.duration_rms_norm(
                &format!("{prefix}.token_out_norm"),
                spec.hidden_dim,
                device,
            )?,
            token_out_proj: self.duration_biased_linear(
                &format!("{prefix}.token_out_proj"),
                1,
                spec.hidden_dim,
                device,
            )?,
            text_dim: EmptyRecord::new(),
            aux_dim: EmptyRecord::new(),
            speaker_dim: EmptyRecord::new(),
            caption_dim: EmptyRecord::new(),
        })
    }

    fn duration_block<B: Backend>(
        &self,
        predictor_prefix: &str,
        index: usize,
        spec: DurationRecordSpec,
        device: &B::Device,
    ) -> Result<DurationSwiGluBlockRecord<B>> {
        let prefix = format!("{predictor_prefix}.token_blocks.{index}");
        Ok(DurationSwiGluBlockRecord {
            norm: self.duration_rms_norm(&format!("{prefix}.norm"), spec.hidden_dim, device)?,
            mlp: self.duration_swiglu(&format!("{prefix}.mlp"), spec.hidden_dim, device)?,
            dropout: EmptyRecord::new(),
            modulation: self.duration_biased_linear(
                &format!("{prefix}.modulation"),
                spec.hidden_dim * 3,
                spec.speaker_dim,
                device,
            )?,
            caption_modulation: self.duration_biased_linear(
                &format!("{prefix}.caption_modulation"),
                spec.hidden_dim * 3,
                spec.caption_dim,
                device,
            )?,
            cached_null_shift: EmptyRecord::new(),
            cached_null_scale_plus_one: EmptyRecord::new(),
            cached_null_gate_tanh: EmptyRecord::new(),
        })
    }

    fn duration_swiglu<B: Backend>(
        &self,
        prefix: &str,
        dim: usize,
        device: &B::Device,
    ) -> Result<SwiGluRecord<B>> {
        Ok(SwiGluRecord {
            w1: self.duration_bias_free_linear(&format!("{prefix}.w1"), dim, dim, device)?,
            w2: self.duration_bias_free_linear(&format!("{prefix}.w2"), dim, dim, device)?,
            w3: self.duration_bias_free_linear(&format!("{prefix}.w3"), dim, dim, device)?,
            fused_w13_weight: EmptyRecord::new(),
            packed_w2_weight_wgsl: EmptyRecord::new(),
        })
    }

    fn duration_biased_linear<B: Backend>(
        &self,
        prefix: &str,
        output_dim: usize,
        input_dim: usize,
        device: &B::Device,
    ) -> Result<LinearRecord<B>> {
        self.require_duration_shape(&format!("{prefix}.weight"), &[output_dim, input_dim])?;
        self.require_duration_shape(&format!("{prefix}.bias"), &[output_dim])?;
        self.linear(prefix, device)
    }

    fn duration_bias_free_linear<B: Backend>(
        &self,
        prefix: &str,
        output_dim: usize,
        input_dim: usize,
        device: &B::Device,
    ) -> Result<LinearRecord<B>> {
        self.require_duration_shape(&format!("{prefix}.weight"), &[output_dim, input_dim])?;
        let bias_key = format!("{prefix}.bias");
        if self.has(&bias_key) {
            return Err(IrodoriError::Shape(format!(
                "{bias_key}: unexpected bias in bias-free v4 duration SwiGLU"
            )));
        }
        self.linear(prefix, device)
    }

    fn duration_rms_norm<B: Backend>(
        &self,
        prefix: &str,
        dim: usize,
        device: &B::Device,
    ) -> Result<RmsNormRecord<B>> {
        self.require_duration_shape(&format!("{prefix}.weight"), &[dim])?;
        let bias_key = format!("{prefix}.bias");
        if self.has(&bias_key) {
            return Err(IrodoriError::Shape(format!(
                "{bias_key}: unexpected bias in v4 duration RMSNorm"
            )));
        }
        self.rms_norm(prefix, 1e-5, device)
    }

    fn require_duration_shape(&self, key: &str, expected: &[usize]) -> Result<()> {
        let actual = &self.entry(key)?.shape;
        if actual != expected {
            return Err(IrodoriError::Shape(format!(
                "{key}: expected {expected:?}, got {actual:?}"
            )));
        }
        Ok(())
    }
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

    fn tiny_spec() -> DurationRecordSpec {
        DurationRecordSpec {
            text_dim: 4,
            hidden_dim: 6,
            layers: 2,
            speaker_dim: 3,
            caption_dim: 2,
        }
    }

    fn tiny_duration_tensors(prefix: &str, spec: DurationRecordSpec) -> Vec<TestTensor> {
        let mut tensors = vec![
            tensor(format!("{prefix}.null_speaker"), &[spec.speaker_dim]),
            tensor(format!("{prefix}.null_caption"), &[spec.caption_dim]),
            tensor(
                format!("{prefix}.token_input_proj.weight"),
                &[spec.hidden_dim, spec.text_dim],
            ),
            tensor(
                format!("{prefix}.token_input_proj.bias"),
                &[spec.hidden_dim],
            ),
            tensor(
                format!("{prefix}.token_out_norm.weight"),
                &[spec.hidden_dim],
            ),
            tensor(
                format!("{prefix}.token_out_proj.weight"),
                &[1, spec.hidden_dim],
            ),
            tensor(format!("{prefix}.token_out_proj.bias"), &[1]),
        ];

        for index in 0..spec.layers {
            let block = format!("{prefix}.token_blocks.{index}");
            tensors.push(tensor(format!("{block}.norm.weight"), &[spec.hidden_dim]));
            for projection in ["w1", "w2", "w3"] {
                tensors.push(tensor(
                    format!("{block}.mlp.{projection}.weight"),
                    &[spec.hidden_dim, spec.hidden_dim],
                ));
            }
            tensors.extend([
                tensor(
                    format!("{block}.modulation.weight"),
                    &[spec.hidden_dim * 3, spec.speaker_dim],
                ),
                tensor(format!("{block}.modulation.bias"), &[spec.hidden_dim * 3]),
                tensor(
                    format!("{block}.caption_modulation.weight"),
                    &[spec.hidden_dim * 3, spec.caption_dim],
                ),
                tensor(
                    format!("{block}.caption_modulation.bias"),
                    &[spec.hidden_dim * 3],
                ),
            ]);
        }
        tensors
    }

    fn duration_tensor_count(spec: DurationRecordSpec) -> usize {
        // Seven predictor-level tensors and eight tensors in each block.
        7 + spec.layers * 8
    }

    #[test]
    fn complete_duration_tree_maps_and_transposes_every_linear_family() {
        let spec = tiny_spec();
        let (_file, store) = load_store(&tiny_duration_tensors(V4_DURATION_PREFIX, spec));
        let record = store
            .duration_predictor_with_spec::<B>(V4_DURATION_PREFIX, spec, &Default::default())
            .unwrap();

        assert_eq!(record.null_speaker.val().shape(), Shape::new([3]));
        assert_eq!(record.null_caption.val().shape(), Shape::new([2]));
        assert_eq!(
            record.token_input_proj.weight.val().shape(),
            Shape::new([4, 6])
        );
        assert!(record.token_input_proj.bias.is_some());
        assert_eq!(record.token_blocks.len(), 2);
        assert_eq!(
            record.token_blocks[0].mlp.w1.weight.val().shape(),
            Shape::new([6, 6])
        );
        assert!(record.token_blocks[0].mlp.w1.bias.is_none());
        assert_eq!(
            record.token_blocks[0].modulation.weight.val().shape(),
            Shape::new([3, 18])
        );
        assert_eq!(
            record.token_blocks[0]
                .caption_modulation
                .weight
                .val()
                .shape(),
            Shape::new([2, 18])
        );
        assert_eq!(
            record.token_out_proj.weight.val().shape(),
            Shape::new([6, 1])
        );
        assert!(record.token_out_proj.bias.is_some());
    }

    #[test]
    fn missing_tensor_reports_the_exact_v4_key() {
        let spec = tiny_spec();
        let missing = format!("{V4_DURATION_PREFIX}.token_blocks.1.caption_modulation.bias");
        let mut tensors = tiny_duration_tensors(V4_DURATION_PREFIX, spec);
        tensors.retain(|(name, ..)| name != &missing);
        let (_file, store) = load_store(&tensors);

        let error = match store.duration_predictor_with_spec::<B>(
            V4_DURATION_PREFIX,
            spec,
            &Default::default(),
        ) {
            Ok(_) => panic!("missing duration tensor unexpectedly loaded"),
            Err(error) => error,
        };
        assert!(matches!(error, IrodoriError::Weight(key) if key == missing));
    }

    #[test]
    fn wrong_shape_reports_key_expected_and_actual_geometry() {
        let spec = tiny_spec();
        let key = format!("{V4_DURATION_PREFIX}.token_blocks.0.modulation.weight");
        let mut tensors = tiny_duration_tensors(V4_DURATION_PREFIX, spec);
        let entry = tensors.iter_mut().find(|(name, ..)| name == &key).unwrap();
        *entry = tensor(key.clone(), &[17, 3]);
        let (_file, store) = load_store(&tensors);

        let error = match store.duration_predictor_with_spec::<B>(
            V4_DURATION_PREFIX,
            spec,
            &Default::default(),
        ) {
            Ok(_) => panic!("wrong duration tensor shape unexpectedly loaded"),
            Err(error) => error,
        };
        assert!(
            matches!(error, IrodoriError::Shape(message) if message.contains(&key) && message.contains("[18, 3]") && message.contains("[17, 3]"))
        );
    }

    #[test]
    fn unexpected_swiglu_bias_is_rejected() {
        let spec = tiny_spec();
        let key = format!("{V4_DURATION_PREFIX}.token_blocks.0.mlp.w1.bias");
        let mut tensors = tiny_duration_tensors(V4_DURATION_PREFIX, spec);
        tensors.push(tensor(key.clone(), &[spec.hidden_dim]));
        let (_file, store) = load_store(&tensors);

        let error = match store.duration_predictor_with_spec::<B>(
            V4_DURATION_PREFIX,
            spec,
            &Default::default(),
        ) {
            Ok(_) => panic!("biased duration SwiGLU unexpectedly loaded"),
            Err(error) => error,
        };
        assert!(
            matches!(error, IrodoriError::Shape(message) if message.contains(&key) && message.contains("unexpected bias"))
        );
    }

    #[test]
    fn released_v4_spec_is_three_blocks_and_thirty_one_tensors() {
        let spec = DurationRecordSpec::V4_SMALL;
        assert_eq!(spec.layers, 3);
        assert_eq!(spec.text_dim, 512);
        assert_eq!(spec.hidden_dim, 1_024);
        assert_eq!(spec.speaker_dim, 768);
        assert_eq!(spec.caption_dim, 512);
        assert_eq!(duration_tensor_count(spec), 31);
    }
}
