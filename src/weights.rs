//! Weight loading from safetensors checkpoints.
//!
//! Converts a Python-generated safetensors file into a fully initialised
//! `TextToLatentRfDiT<B>` model by constructing the corresponding burn Record
//! hierarchy and calling `model.load_record(record)`.
//!
//! # Key mapping
//! The Python model uses sequential indices for `cond_module` which must be
//! renamed before loading (see `scripts/convert_for_burn.py`):
//! - `cond_module.0.weight` → `cond_module.linear0.weight`
//! - `cond_module.2.weight` → `cond_module.linear1.weight`
//! - `cond_module.4.weight` → `cond_module.linear2.weight`

use std::collections::HashMap;
use std::path::Path;

use burn::{
    module::{ConstantRecord, Module, Param, ParamId},
    nn::{EmbeddingRecord, LinearRecord},
    tensor::{Tensor, TensorData, backend::Backend},
};
use half::f16;
use safetensors::{Dtype, SafeTensors};

use crate::{
    config::ModelConfig,
    error::{IrodoriError, Result},
    model::{
        TextToLatentRfDiT,
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

// ---------------------------------------------------------------------------
// Dtype conversion helpers
// ---------------------------------------------------------------------------

/// Convert a little-endian BF16 pair `[lo, hi]` to `f32`.
///
/// BF16 = the top 16 bits of F32 (same exponent, truncated mantissa).
#[inline]
fn bf16_pair_to_f32(lo: u8, hi: u8) -> f32 {
    // Zero-extend the two lower mantissa bytes that BF16 drops.
    f32::from_le_bytes([0, 0, lo, hi])
}

/// Convert a little-endian F16 byte pair `[lo, hi]` to `f32`.
#[inline]
fn f16_pair_to_f32(lo: u8, hi: u8) -> f32 {
    f16::from_le_bytes([lo, hi]).to_f32()
}

/// Convert a safetensors tensor's raw bytes to a `Vec<f32>`.
fn to_f32_vec(key: &str, dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        Dtype::BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|b| bf16_pair_to_f32(b[0], b[1]))
            .collect()),
        Dtype::F16 => Ok(bytes
            .chunks_exact(2)
            .map(|b| f16_pair_to_f32(b[0], b[1]))
            .collect()),
        other => Err(IrodoriError::Dtype(key.to_string(), format!("{other:?}"))),
    }
}

// ---------------------------------------------------------------------------
// TensorStore
// ---------------------------------------------------------------------------

/// In-memory store of all tensors from a safetensors checkpoint.
///
/// All tensors are eagerly converted to `f32` on load for simplicity and
/// cross-dtype compatibility. For large models this may use ~2× the checkpoint
/// size in RAM; this is acceptable for an inference/port workflow.
pub struct TensorStore {
    /// Flattened f32 data keyed by tensor name.
    data: HashMap<String, Vec<f32>>,
    /// Shape keyed by tensor name.
    shapes: HashMap<String, Vec<usize>>,
    /// The `config_json` metadata embedded in the checkpoint.
    pub config_json: String,
}

impl TensorStore {
    /// Load a safetensors checkpoint from `path`.
    ///
    /// Reads the entire file into memory, converts all tensors to f32, and
    /// extracts `config_json` from the user-metadata section.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;

        // Extract config_json from metadata before taking ownership.
        let config_json = {
            let (_offset, metadata) = SafeTensors::read_metadata(&bytes)?;
            let meta = metadata.metadata().as_ref().ok_or(IrodoriError::NoConfig)?;
            meta.get("config_json")
                .ok_or(IrodoriError::NoConfig)?
                .clone()
        };

        // Parse all tensors.
        let st = SafeTensors::deserialize(&bytes)?;
        let mut data = HashMap::new();
        let mut shapes = HashMap::new();
        for (name, view) in st.tensors() {
            let f32_data = to_f32_vec(&name, view.dtype(), view.data())?;
            shapes.insert(name.clone(), view.shape().to_vec());
            data.insert(name, f32_data);
        }

        Ok(Self {
            data,
            shapes,
            config_json,
        })
    }

    /// True if the store contains `key`.
    pub fn has(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Return the shape of `key`, or error if missing.
    fn shape(&self, key: &str) -> Result<&[usize]> {
        self.shapes
            .get(key)
            .map(Vec::as_slice)
            .ok_or_else(|| IrodoriError::Weight(key.to_string()))
    }

    /// Build a `Param<Tensor<B, D>>` from `key`.
    fn param<B: Backend, const D: usize>(
        &self,
        key: &str,
        device: &B::Device,
    ) -> Result<Param<Tensor<B, D>>> {
        let shape = self.shape(key)?;
        if shape.len() != D {
            return Err(IrodoriError::WrongDim(key.to_string(), D, shape.len()));
        }
        let floats = self
            .data
            .get(key)
            .ok_or_else(|| IrodoriError::Weight(key.to_string()))?;

        // Build a fixed-size shape array from the dynamic slice.
        let shape_arr: [usize; D] = shape.try_into().expect("D checked above");

        let tensor_data = TensorData::new(floats.clone(), shape_arr);
        let tensor = Tensor::<B, D>::from_data(tensor_data, device);
        Ok(Param::initialized(ParamId::new(), tensor))
    }

    /// Build a `LinearRecord<B>` from weights at `prefix.weight` (and optionally `prefix.bias`).
    ///
    /// # Weight transposition
    /// PyTorch stores `nn.Linear.weight` as `[d_output, d_input]` and computes `x @ W.T`.
    /// Burn stores `Linear.weight` as `[d_input, d_output]` and computes `x @ W`.
    /// We must transpose the safetensors weight before building the param.
    fn linear<B: Backend>(&self, prefix: &str, device: &B::Device) -> Result<LinearRecord<B>> {
        let w_key = format!("{prefix}.weight");
        let b_key = format!("{prefix}.bias");

        let weight = {
            let shape = self.shape(&w_key)?;
            if shape.len() != 2 {
                return Err(IrodoriError::WrongDim(w_key.clone(), 2, shape.len()));
            }
            let [d_out, d_in] = [shape[0], shape[1]];
            let floats = self
                .data
                .get(&w_key)
                .ok_or_else(|| IrodoriError::Weight(w_key.clone()))?;
            // Load as [d_out, d_in], then transpose to [d_in, d_out] for burn.
            let tensor_data = TensorData::new(floats.clone(), [d_out, d_in]);
            let tensor = Tensor::<B, 2>::from_data(tensor_data, device).transpose();
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
    fn embedding<B: Backend>(
        &self,
        prefix: &str,
        device: &B::Device,
    ) -> Result<EmbeddingRecord<B>> {
        let weight = self.param::<B, 2>(&format!("{prefix}.weight"), device)?;
        Ok(EmbeddingRecord { weight })
    }

    /// Build a `RmsNormRecord<B>` from `prefix.weight` (1-D) and a constant eps.
    fn rms_norm<B: Backend>(
        &self,
        prefix: &str,
        _eps: f64,
        device: &B::Device,
    ) -> Result<RmsNormRecord<B>> {
        Ok(RmsNormRecord {
            weight: self.param::<B, 1>(&format!("{prefix}.weight"), device)?,
            eps: ConstantRecord::new(),
        })
    }

    /// Build a `HeadRmsNormRecord<B>` from `prefix.weight` (2-D) and a constant eps.
    fn head_rms_norm<B: Backend>(
        &self,
        prefix: &str,
        device: &B::Device,
    ) -> Result<HeadRmsNormRecord<B>> {
        Ok(HeadRmsNormRecord {
            weight: self.param::<B, 2>(&format!("{prefix}.weight"), device)?,
            eps: ConstantRecord::new(),
        })
    }

    /// Build a `LowRankAdaLnRecord<B>`.
    fn low_rank_adaln<B: Backend>(
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
            eps: ConstantRecord::new(),
        })
    }

    /// Build a `SwiGluRecord<B>` from `prefix.{w1,w2,w3}`.
    fn swiglu<B: Backend>(&self, prefix: &str, device: &B::Device) -> Result<SwiGluRecord<B>> {
        Ok(SwiGluRecord {
            w1: self.linear(format!("{prefix}.w1").as_str(), device)?,
            w2: self.linear(format!("{prefix}.w2").as_str(), device)?,
            w3: self.linear(format!("{prefix}.w3").as_str(), device)?,
        })
    }

    /// Build a `SelfAttentionRecord<B>`.
    fn self_attention<B: Backend>(
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
            num_heads: ConstantRecord::new(),
            head_dim: ConstantRecord::new(),
            scale: ConstantRecord::new(),
        })
    }

    /// Build a `JointAttentionRecord<B>`.
    ///
    /// Optional speaker/caption KV projections are loaded only if present.
    fn joint_attention<B: Backend>(
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
            num_heads: ConstantRecord::new(),
            head_dim: ConstantRecord::new(),
            scale: ConstantRecord::new(),
        })
    }

    /// Build a `TextBlockRecord<B>`.
    fn text_block<B: Backend>(
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
            dropout: ConstantRecord::new(),
        })
    }

    /// Build a `TextEncoderRecord<B>` (used for both text and caption encoders).
    fn text_encoder<B: Backend>(
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
            head_dim: ConstantRecord::new(),
        })
    }

    /// Build a `ReferenceLatentEncoderRecord<B>`.
    fn reference_latent_encoder<B: Backend>(
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
            head_dim: ConstantRecord::new(),
        })
    }

    /// Build a `DiffusionBlockRecord<B>`.
    fn diffusion_block<B: Backend>(
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
            dropout: ConstantRecord::new(),
        })
    }

    /// Build a `CondModuleRecord<B>`.
    fn cond_module<B: Backend>(&self, device: &B::Device) -> Result<CondModuleRecord<B>> {
        Ok(CondModuleRecord {
            linear0: self.linear("cond_module.linear0", device)?,
            linear1: self.linear("cond_module.linear1", device)?,
            linear2: self.linear("cond_module.linear2", device)?,
        })
    }

    /// Assemble the full `TextToLatentRfDiTRecord<B>`.
    fn build_model_record<B: Backend>(
        &self,
        cfg: &ModelConfig,
        device: &B::Device,
    ) -> Result<TextToLatentRfDiTRecord<B>> {
        // Text encoder.
        let text_encoder = self.text_encoder("text_encoder", cfg.text_layers, device)?;
        let text_norm = self.rms_norm("text_norm", cfg.norm_eps, device)?;

        // Optional speaker or caption encoder — mutually exclusive.
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

        // Diffusion backbone.
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
            model_dim: ConstantRecord::new(),
            num_heads: ConstantRecord::new(),
            head_dim: ConstantRecord::new(),
            timestep_embed_dim: ConstantRecord::new(),
            speaker_patch_size: ConstantRecord::new(),
            patched_latent_dim: ConstantRecord::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Load a model and its configuration from a safetensors checkpoint.
///
/// The checkpoint must have been prepared by `scripts/convert_for_burn.py`,
/// which renames the `cond_module.{0,2,4}` keys to `cond_module.{linear0,linear1,linear2}`.
///
/// # Errors
/// Returns `IrodoriError::NoConfig` if `config_json` is absent from the checkpoint.
/// Returns `IrodoriError::Weight` if any required tensor is missing.
pub fn load_model<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<(TextToLatentRfDiT<B>, ModelConfig)> {
    let store = TensorStore::load(path)?;
    let cfg: ModelConfig = serde_json::from_str(&store.config_json)?;
    cfg.validate()?;
    let model = TextToLatentRfDiT::new(&cfg, device);
    let record = store.build_model_record::<B>(&cfg, device)?;
    let model = model.load_record(record);
    Ok((model, cfg))
}
