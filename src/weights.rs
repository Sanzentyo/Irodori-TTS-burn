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
    module::{EmptyRecord, Module, Param, ParamId},
    nn::{EmbeddingRecord, LinearRecord},
    tensor::{Tensor, TensorData, backend::Backend},
};
use half::{bf16, f16};
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
// TensorEntry — unified per-tensor metadata + raw bytes
// ---------------------------------------------------------------------------

/// Raw bytes and metadata for a single safetensors tensor.
struct TensorEntry {
    /// Little-endian raw bytes as stored in the safetensors file.
    bytes: Vec<u8>,
    /// Element dtype of the stored bytes.
    dtype: Dtype,
    /// Shape dimensions (innermost-last / row-major).
    shape: Vec<usize>,
}

impl TensorEntry {
    fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    fn dtype_byte_size(dtype: Dtype) -> usize {
        match dtype {
            Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => 2,
            Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
            Dtype::I8 | Dtype::U8 | Dtype::BOOL => 1,
            _ => 4, // fallback; validated separately
        }
    }

    /// Validate that `bytes.len() == numel * dtype_byte_size`.
    fn validate_byte_len(&self, key: &str) -> Result<()> {
        let expected = self.numel() * Self::dtype_byte_size(self.dtype);
        if self.bytes.len() != expected {
            return Err(IrodoriError::Weight(format!(
                "{key}: byte length mismatch — expected {expected}, got {}",
                self.bytes.len()
            )));
        }
        Ok(())
    }

    /// Decode the raw bytes into a [`TensorData`] using the checkpoint's native dtype.
    ///
    /// Burn will transparently convert to the backend's float element type when
    /// `Tensor::from_data` is called, so a bf16 checkpoint loaded into an f32
    /// backend incurs the expected bf16→f32 conversion there, not here.
    ///
    /// When the checkpoint dtype matches the backend dtype (e.g. bf16 checkpoint
    /// into bf16 backend), burn skips the conversion and does a direct copy.
    fn to_tensor_data<const D: usize>(&self, key: &str) -> Result<TensorData> {
        if self.shape.len() != D {
            return Err(IrodoriError::WrongDim(key.to_string(), D, self.shape.len()));
        }
        let shape_arr: [usize; D] = self.shape[..].try_into().expect("D validated above");

        let td = match self.dtype {
            Dtype::F32 => {
                let data: Vec<f32> = self
                    .bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                TensorData::new(data, shape_arr)
            }
            Dtype::BF16 => {
                let data: Vec<bf16> = self
                    .bytes
                    .chunks_exact(2)
                    .map(|b| bf16::from_le_bytes([b[0], b[1]]))
                    .collect();
                TensorData::new(data, shape_arr)
            }
            Dtype::F16 => {
                let data: Vec<f16> = self
                    .bytes
                    .chunks_exact(2)
                    .map(|b| f16::from_le_bytes([b[0], b[1]]))
                    .collect();
                TensorData::new(data, shape_arr)
            }
            other => return Err(IrodoriError::Dtype(key.to_string(), format!("{other:?}"))),
        };
        Ok(td)
    }
}

// ---------------------------------------------------------------------------
// TensorStore
// ---------------------------------------------------------------------------

/// In-memory store of all tensors from a safetensors checkpoint.
///
/// Raw bytes are kept in the checkpoint's native dtype (f32, bf16, or f16).
/// Conversion to the backend's element type happens lazily in [`param`] /
/// [`linear`] via burn's `Tensor::from_data`, avoiding a superfluous
/// intermediate `Vec<f32>` for bf16 checkpoints loaded into bf16 backends.
pub struct TensorStore {
    tensors: HashMap<String, TensorEntry>,
    /// The `config_json` metadata embedded in the checkpoint.
    pub config_json: String,
}

impl TensorStore {
    /// Load a safetensors checkpoint from `path`.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;

        let config_json = {
            let (_offset, metadata) = SafeTensors::read_metadata(&bytes)?;
            let meta = metadata.metadata().as_ref().ok_or(IrodoriError::NoConfig)?;
            meta.get("config_json")
                .ok_or(IrodoriError::NoConfig)?
                .clone()
        };

        let st = SafeTensors::deserialize(&bytes)?;
        let mut tensors = HashMap::new();
        for (name, view) in st.tensors() {
            let entry = TensorEntry {
                bytes: view.data().to_vec(),
                dtype: view.dtype(),
                shape: view.shape().to_vec(),
            };
            entry.validate_byte_len(&name)?;
            tensors.insert(name, entry);
        }

        Ok(Self {
            tensors,
            config_json,
        })
    }

    /// True if the store contains `key`.
    pub fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    /// Return the entry for `key`, or error if missing.
    fn entry(&self, key: &str) -> Result<&TensorEntry> {
        self.tensors
            .get(key)
            .ok_or_else(|| IrodoriError::Weight(key.to_string()))
    }

    /// Build a `Param<Tensor<B, D>>` from `key`.
    fn param<B: Backend, const D: usize>(
        &self,
        key: &str,
        device: &B::Device,
    ) -> Result<Param<Tensor<B, D>>> {
        let entry = self.entry(key)?;
        let td = entry.to_tensor_data::<D>(key)?;
        let tensor = Tensor::<B, D>::from_data(td, device);
        Ok(Param::initialized(ParamId::new(), tensor))
    }

    /// Build a `LinearRecord<B>` from weights at `prefix.weight` (and optionally `prefix.bias`).
    ///
    /// # Weight transposition
    /// PyTorch stores `nn.Linear.weight` as `[d_output, d_input]` and computes `x @ W.T`.
    /// Burn stores `Linear.weight` as `[d_input, d_output]` and computes `x @ W`.
    /// We transpose the safetensors weight before building the param.
    fn linear<B: Backend>(&self, prefix: &str, device: &B::Device) -> Result<LinearRecord<B>> {
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
            eps: EmptyRecord::new(),
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
            eps: EmptyRecord::new(),
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
            eps: EmptyRecord::new(),
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
            num_heads: EmptyRecord::new(),
            head_dim: EmptyRecord::new(),
            scale: EmptyRecord::new(),
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
            num_heads: EmptyRecord::new(),
            head_dim: EmptyRecord::new(),
            scale: EmptyRecord::new(),
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
            dropout: EmptyRecord::new(),
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
            head_dim: EmptyRecord::new(),
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
            head_dim: EmptyRecord::new(),
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
            dropout: EmptyRecord::new(),
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
