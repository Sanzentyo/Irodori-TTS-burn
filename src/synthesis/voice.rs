//! Portable reference latents and model-bound speaker-inversion embeddings.

use crate::{IrodoriError, Result};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use std::{collections::HashMap, fs::OpenOptions, io::Write, path::Path};

#[derive(Debug)]
pub(super) enum VoiceData {
    Latent {
        values: Vec<f32>,
        frames: usize,
    },
    Embedding {
        values: Vec<f32>,
        tokens: usize,
        dim: usize,
    },
}

/// A reusable voice stores FP32 values and a codec identity. Learned speaker
/// embeddings additionally bind to the exact DiT checkpoint.
#[derive(Debug)]
pub struct Voice {
    pub(super) data: VoiceData,
    pub(super) codec_sha256: String,
    pub(super) model_sha256: Option<String>,
}

impl Voice {
    pub fn load(
        path: &Path,
        codec_sha256: &str,
        model_sha256: &str,
        speaker_dim: usize,
    ) -> Result<Self> {
        let size = std::fs::metadata(path)?.len();
        if size > 64 * 1024 * 1024 {
            return Err(IrodoriError::Audio("voice file exceeds 64 MiB".into()));
        }
        let bytes = std::fs::read(path)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let (_, header) = SafeTensors::read_metadata(&bytes)?;
        let metadata = header.metadata().as_ref();
        let get = |name: &str| metadata.and_then(|m| m.get(name)).map(String::as_str);
        let (key, embedding) = if tensors.names().contains(&"speaker_embedding") {
            ("speaker_embedding", true)
        } else {
            ("reference_latent", false)
        };
        if get("irodori_voice_schema").is_some_and(|schema| schema != "1") {
            return Err(IrodoriError::Audio("unsupported voice file schema".into()));
        }
        if !embedding && get("codec_sha256") != Some(codec_sha256) {
            return Err(IrodoriError::Audio(
                "voice belongs to a different codec".into(),
            ));
        }
        if get("model_sha256").is_some_and(|sha| sha != model_sha256) {
            return Err(IrodoriError::Audio(
                "speaker embedding belongs to a different model".into(),
            ));
        }
        let tensor = tensors.tensor(key)?;
        if !matches!(tensor.dtype(), Dtype::F32 | Dtype::F16 | Dtype::BF16) {
            return Err(IrodoriError::Audio(
                "voice tensors must be FP32, FP16 or BF16".into(),
            ));
        }
        let (tokens, dim) = match tensor.shape() {
            [tokens, dim] | [1, tokens, dim] => (*tokens, *dim),
            _ => {
                return Err(IrodoriError::Audio(
                    "voice shape must be [tokens, dim] or [1, tokens, dim]".into(),
                ));
            }
        };
        if tokens == 0 || tokens > 3000 || dim != if embedding { speaker_dim } else { 32 } {
            return Err(IrodoriError::Audio(
                "voice token count or dimension is incompatible with this model".into(),
            ));
        }
        let values = if tensor.dtype() == Dtype::F32 {
            tensor
                .data()
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
                .collect::<Vec<_>>()
        } else {
            tensor
                .data()
                .chunks_exact(2)
                .map(|v| {
                    let bits = u16::from_le_bytes([v[0], v[1]]);
                    if tensor.dtype() == Dtype::F16 {
                        half::f16::from_bits(bits).to_f32()
                    } else {
                        half::bf16::from_bits(bits).to_f32()
                    }
                })
                .collect()
        };
        if values.iter().any(|v| !v.is_finite()) {
            return Err(IrodoriError::Audio(
                "voice contains non-finite values".into(),
            ));
        }
        Ok(Self {
            data: if embedding {
                VoiceData::Embedding {
                    values,
                    tokens,
                    dim,
                }
            } else {
                VoiceData::Latent {
                    values,
                    frames: tokens,
                }
            },
            codec_sha256: codec_sha256.into(),
            model_sha256: embedding.then(|| model_sha256.into()),
        })
    }

    /// Write a new safetensors voice file without overwriting an existing one.
    pub fn save(&self, path: &Path) -> Result<()> {
        let (key, values, shape) = match &self.data {
            VoiceData::Latent { values, frames } => {
                ("reference_latent", values, vec![1, *frames, 32])
            }
            VoiceData::Embedding {
                values,
                tokens,
                dim,
            } => ("speaker_embedding", values, vec![1, *tokens, *dim]),
        };
        let data = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        let view = TensorView::new(Dtype::F32, shape, &data)?;
        let mut metadata = HashMap::from([
            ("irodori_voice_schema".into(), "1".into()),
            ("codec_sha256".into(), self.codec_sha256.clone()),
        ]);
        if let Some(model) = &self.model_sha256 {
            metadata.insert("model_sha256".into(), model.clone());
        }
        let bytes = safetensors::tensor::serialize([(key, view)], Some(metadata))?;
        let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
        if let Err(error) = file.write_all(&bytes).and_then(|()| file.sync_all()) {
            drop(file);
            if let Err(cleanup) = std::fs::remove_file(path) {
                tracing::warn!(%cleanup, path = %path.display(), "failed to remove incomplete voice file");
            }
            return Err(error.into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::identity::codec_identity;
    #[test]
    fn voice_identity_is_independent_of_encoder_and_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let decoder = dir.path().join("decoder.safetensors");
        let full = dir.path().join("full.safetensors");
        let bytes = 0.125_f32.to_le_bytes();
        let view = || TensorView::new(Dtype::F32, vec![1], &bytes).unwrap();
        std::fs::write(
            &decoder,
            safetensors::tensor::serialize([("decoder.model.0.weight", view())], None).unwrap(),
        )
        .unwrap();
        std::fs::write(
            &full,
            safetensors::tensor::serialize(
                [
                    ("encoder.block.0.weight", view()),
                    ("decoder.model.0.weight", view()),
                ],
                Some(HashMap::from([("source".into(), "other-export".into())])),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            codec_identity(&decoder).unwrap(),
            codec_identity(&full).unwrap()
        );
    }

    #[test]
    fn imported_half_embedding_becomes_model_bound() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("speaker.safetensors");
        let saved = dir.path().join("voice.safetensors");
        let bytes = half::f16::from_f32(0.125).to_le_bytes().repeat(6);
        let view = TensorView::new(Dtype::F16, vec![2, 3], &bytes).unwrap();
        std::fs::write(
            &source,
            safetensors::tensor::serialize([("speaker_embedding", view)], None).unwrap(),
        )
        .unwrap();
        let voice = Voice::load(&source, "codec", "model", 3).unwrap();
        voice.save(&saved).unwrap();
        assert!(Voice::load(&saved, "codec", "different-model", 3).is_err());
        assert!(
            matches!(Voice::load(&saved, "codec", "model", 3).unwrap().data,
            VoiceData::Embedding { tokens: 2, dim: 3, values } if values == vec![0.125; 6])
        );
    }

    #[test]
    fn voice_roundtrip_validates_identity_and_never_overwrites() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("voice.safetensors");
        let voice = Voice {
            data: VoiceData::Latent {
                values: vec![0.125; 64],
                frames: 2,
            },
            codec_sha256: "codec".into(),
            model_sha256: None,
        };
        voice.save(&path).unwrap();
        let loaded = Voice::load(&path, "codec", "model", 768).unwrap();
        assert!(matches!(loaded.data, VoiceData::Latent { frames: 2, .. }));
        assert!(Voice::load(&path, "wrong-codec", "model", 768).is_err());
        assert!(voice.save(&path).is_err());
    }
}
