//! Training batch construction: padding, masking, and latent loading.

use std::path::Path;

use burn::tensor::{Bool, Int, Tensor, backend::Backend};

use super::manifest::ManifestSample;
use crate::error::IrodoriError;

// ---------------------------------------------------------------------------
// Training batch
// ---------------------------------------------------------------------------

/// A fully-tensorised training batch.
pub struct TrainBatch<B: Backend> {
    /// Target latent tokens `[B, S, D]`.
    pub latent: Tensor<B, 3>,
    /// Padding mask for latent `[B, S]` — `true` = valid token.
    pub latent_mask: Tensor<B, 2, Bool>,
    /// Loss mask `[B, S]` — same as `latent_mask` in v1.
    pub loss_mask: Tensor<B, 2, Bool>,
    /// Tokenised text `[B, T]` (int).
    pub text_ids: Tensor<B, 2, Int>,
    /// Text padding mask `[B, T]` — `true` = valid token.
    pub text_mask: Tensor<B, 2, Bool>,
    /// Optional reference latent for speaker conditioning `[B, S_ref, D]`.
    pub ref_latent: Option<Tensor<B, 3>>,
    /// Optional mask for ref latent `[B, S_ref]`.
    pub ref_latent_mask: Option<Tensor<B, 2, Bool>>,
    /// Optional tokenised caption IDs for caption conditioning `[B, C]`.
    pub caption_ids: Option<Tensor<B, 2, Int>>,
    /// Optional caption padding mask `[B, C]` — `true` = valid token.
    pub caption_mask: Option<Tensor<B, 2, Bool>>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

pub(super) fn load_latent_safetensors<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> crate::error::Result<Tensor<B, 3>> {
    let bytes = std::fs::read(path)
        .map_err(|e| IrodoriError::Dataset(format!("read latent {}: {e}", path.display())))?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| IrodoriError::Dataset(format!("parse safetensors {}: {e}", path.display())))?;

    // Prefer "latent" key, else take the only tensor in the file.
    let (name, view) = if tensors.len() == 1 {
        let (n, v) = tensors.tensors().into_iter().next().ok_or_else(|| {
            IrodoriError::Dataset(format!("empty safetensors: {}", path.display()))
        })?;
        (n, v)
    } else {
        let v = tensors.tensor("latent").map_err(|_| {
            IrodoriError::Weight(format!("'latent' key not found in {}", path.display()))
        })?;
        ("latent".to_owned(), v)
    };
    let _ = name; // used for error messages above

    let shape = view.shape();
    let data: Vec<f32> = match view.dtype() {
        safetensors::Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        safetensors::Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect(),
        safetensors::Dtype::F16 => view
            .data()
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect(),
        other => {
            return Err(IrodoriError::Dtype(
                path.display().to_string(),
                format!("{other:?}"),
            ));
        }
    };

    let tensor: Tensor<B, 3> = match shape.len() {
        2 => {
            let t: Tensor<B, 2> = Tensor::from_data(
                burn::tensor::TensorData::new(data, [shape[0], shape[1]]),
                device,
            );
            t.unsqueeze::<3>()
        }
        3 => Tensor::from_data(
            burn::tensor::TensorData::new(data, [shape[0], shape[1], shape[2]]),
            device,
        ),
        n => {
            return Err(IrodoriError::Dataset(format!(
                "latent in {} has {n} dims, expected 2 or 3",
                path.display(),
            )));
        }
    };
    Ok(tensor)
}

pub(super) fn build_batch<B: Backend>(
    samples: &[&ManifestSample],
    tokenizer: &tokenizers::Tokenizer,
    caption_tokenizer: Option<&tokenizers::Tokenizer>,
    device: &B::Device,
) -> crate::error::Result<TrainBatch<B>> {
    // ------------------------------------------------------------------
    // 1. Load latents and tokenise text (+ optional caption)
    // ------------------------------------------------------------------
    let mut latents: Vec<Tensor<B, 3>> = Vec::with_capacity(samples.len());
    let mut ref_latents: Vec<Option<Tensor<B, 3>>> = Vec::with_capacity(samples.len());
    let mut text_token_seqs: Vec<Vec<u32>> = Vec::with_capacity(samples.len());
    let mut caption_token_seqs: Vec<Option<Vec<u32>>> = Vec::with_capacity(samples.len());

    for s in samples {
        let lat = load_latent_safetensors::<B>(&s.latent_path, device)?;
        latents.push(lat);

        let ref_lat = s
            .ref_latent_path
            .as_deref()
            .map(|p| load_latent_safetensors::<B>(p, device))
            .transpose()?;
        ref_latents.push(ref_lat);

        let enc = tokenizer
            .encode(s.text.as_str(), false)
            .map_err(|e| IrodoriError::Tokenizer(format!("tokenize: {e}")))?;
        let ids = enc.get_ids().to_vec();
        if ids.is_empty() {
            return Err(IrodoriError::Dataset(format!(
                "tokenised text is empty for sample '{}' — every training sample \
                 must produce at least one token (required for safe_softmax=false in SDPA)",
                s.text,
            )));
        }
        text_token_seqs.push(ids);

        // Optional caption tokenization
        let cap_ids = s
            .caption
            .as_deref()
            .map(|cap_text| {
                let cap_tok = caption_tokenizer.unwrap_or(tokenizer);
                let enc = cap_tok
                    .encode(cap_text, false)
                    .map_err(|e| IrodoriError::Tokenizer(format!("tokenize caption: {e}")))?;
                let ids = enc.get_ids().to_vec();
                if ids.is_empty() {
                    return Err(IrodoriError::Dataset(format!(
                        "tokenised caption is empty for sample '{}' — every caption \
                         must produce at least one token",
                        cap_text,
                    )));
                }
                Ok(ids)
            })
            .transpose()?;
        caption_token_seqs.push(cap_ids);
    }

    // ------------------------------------------------------------------
    // 2. Pad latents → [B, S_max, D]  (tensor-native, dtype-agnostic)
    // ------------------------------------------------------------------
    let d = latents[0].dims()[2];
    let s_max = latents.iter().map(|t| t.dims()[1]).max().unwrap_or(0);
    let batch = samples.len();

    let mut padded_lats: Vec<Tensor<B, 3>> = Vec::with_capacity(batch);
    let mut lat_masks: Vec<Tensor<B, 2, Bool>> = Vec::with_capacity(batch);

    for lat in &latents {
        let s = lat.dims()[1];
        let pad_len = s_max - s;
        if pad_len > 0 {
            let pad = Tensor::<B, 3>::zeros([1, pad_len, d], device);
            padded_lats.push(Tensor::cat(vec![lat.clone(), pad], 1));
            let ones = Tensor::<B, 2>::ones([1, s], device).greater_elem(0.0f32);
            let zeros = Tensor::<B, 2>::zeros([1, pad_len], device).greater_elem(0.0f32);
            lat_masks.push(Tensor::cat(vec![ones, zeros], 1));
        } else {
            padded_lats.push(lat.clone());
            lat_masks.push(Tensor::<B, 2>::ones([1, s], device).greater_elem(0.0f32));
        }
    }

    let latent: Tensor<B, 3> = Tensor::cat(padded_lats, 0);
    let latent_mask: Tensor<B, 2, Bool> = Tensor::cat(lat_masks, 0);
    let loss_mask = latent_mask.clone();

    // ------------------------------------------------------------------
    // 3. Pad text → [B, T_max]
    // ------------------------------------------------------------------
    let t_max = text_token_seqs.iter().map(|v| v.len()).max().unwrap_or(0);
    let mut text_data: Vec<i32> = vec![0; batch * t_max];
    let mut text_mask_data: Vec<bool> = vec![false; batch * t_max];

    for (b, ids) in text_token_seqs.iter().enumerate() {
        for (j, &id) in ids.iter().enumerate() {
            text_data[b * t_max + j] = id as i32;
            text_mask_data[b * t_max + j] = true;
        }
    }

    let text_ids: Tensor<B, 2, Int> = Tensor::from_data(
        burn::tensor::TensorData::new(text_data, [batch, t_max]),
        device,
    );
    let text_mask: Tensor<B, 2, Bool> = Tensor::from_data(
        burn::tensor::TensorData::new(text_mask_data, [batch, t_max]),
        device,
    );

    // ------------------------------------------------------------------
    // 4. Optional ref latent → [B, S_ref_max, D]  (tensor-native)
    // ------------------------------------------------------------------
    let (ref_latent, ref_latent_mask) = if ref_latents.iter().any(|r| r.is_some()) {
        let sr_max = ref_latents
            .iter()
            .filter_map(|r| r.as_ref().map(|t| t.dims()[1]))
            .max()
            .unwrap_or(0);

        let mut padded_refs: Vec<Tensor<B, 3>> = Vec::with_capacity(batch);
        let mut ref_masks: Vec<Tensor<B, 2, Bool>> = Vec::with_capacity(batch);

        for maybe in &ref_latents {
            if let Some(t) = maybe {
                let sr = t.dims()[1];
                let pad_len = sr_max - sr;
                if pad_len > 0 {
                    let pad = Tensor::<B, 3>::zeros([1, pad_len, d], device);
                    padded_refs.push(Tensor::cat(vec![t.clone(), pad], 1));
                    let ones = Tensor::<B, 2>::ones([1, sr], device).greater_elem(0.0f32);
                    let zeros = Tensor::<B, 2>::zeros([1, pad_len], device).greater_elem(0.0f32);
                    ref_masks.push(Tensor::cat(vec![ones, zeros], 1));
                } else {
                    padded_refs.push(t.clone());
                    ref_masks.push(Tensor::<B, 2>::ones([1, sr], device).greater_elem(0.0f32));
                }
            } else {
                // No ref latent for this sample — pad with zeros / false mask.
                padded_refs.push(Tensor::<B, 3>::zeros([1, sr_max, d], device));
                ref_masks.push(Tensor::<B, 2>::zeros([1, sr_max], device).greater_elem(0.0f32));
            }
        }

        let ref_t = Tensor::cat(padded_refs, 0);
        let ref_m = Tensor::cat(ref_masks, 0);
        (Some(ref_t), Some(ref_m))
    } else {
        (None, None)
    };

    // ------------------------------------------------------------------
    // 5. Optional caption tokens → [B, C_max]
    // ------------------------------------------------------------------
    let (caption_ids, caption_mask) = if caption_token_seqs.iter().any(|c| c.is_some()) {
        let c_max = caption_token_seqs
            .iter()
            .filter_map(|c| c.as_ref().map(|v| v.len()))
            .max()
            .unwrap_or(0);

        let mut cap_data: Vec<i32> = vec![0; batch * c_max];
        let mut cap_mask_data: Vec<bool> = vec![false; batch * c_max];

        for (b, maybe_ids) in caption_token_seqs.iter().enumerate() {
            if let Some(ids) = maybe_ids {
                for (j, &id) in ids.iter().enumerate() {
                    cap_data[b * c_max + j] = id as i32;
                    cap_mask_data[b * c_max + j] = true;
                }
            }
            // Samples without caption get all-zero ids + all-false mask (already default)
        }

        let cap_ids: Tensor<B, 2, Int> = Tensor::from_data(
            burn::tensor::TensorData::new(cap_data, [batch, c_max]),
            device,
        );
        let cap_mask: Tensor<B, 2, Bool> = Tensor::from_data(
            burn::tensor::TensorData::new(cap_mask_data, [batch, c_max]),
            device,
        );
        (Some(cap_ids), Some(cap_mask))
    } else {
        (None, None)
    };

    Ok(TrainBatch {
        latent,
        latent_mask,
        loss_mask,
        text_ids,
        text_mask,
        ref_latent,
        ref_latent_mask,
        caption_ids,
        caption_mask,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use tempfile::TempDir;

    type TestBackend = burn::backend::NdArray;

    fn write_safetensors(path: &Path, seq_len: usize, dim: usize) {
        use safetensors::tensor::TensorView;

        let data: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.01).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let shape = vec![1, seq_len, dim];
        let view = TensorView::new(safetensors::Dtype::F32, shape, &bytes).expect("TensorView");
        let tensors: Vec<(&str, TensorView<'_>)> = vec![("latent", view)];
        safetensors::serialize_to_file(tensors, None, path).expect("write safetensors");
    }

    fn write_tokenizer(dir: &Path) -> std::path::PathBuf {
        use std::io::Write;

        let mut vocab = serde_json::Map::new();
        for b in 0u16..256 {
            let key = if b < 0x21 || b == 0x7f || (0x80..=0x9f).contains(&b) || b == 0xad {
                format!("byte_{b:02x}")
            } else {
                String::from(b as u8 as char)
            };
            vocab.insert(key, serde_json::Value::Number(b.into()));
        }
        vocab.insert("[UNK]".to_string(), serde_json::Value::Number(256.into()));

        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": [],
                "unk_token": "[UNK]"
            },
            "pre_tokenizer": {
                "type": "Whitespace"
            }
        });

        let path = dir.join("tokenizer.json");
        let mut f = std::fs::File::create(&path).expect("create tokenizer.json");
        f.write_all(tokenizer_json.to_string().as_bytes())
            .expect("write tokenizer.json");
        path
    }

    #[test]
    fn build_batch_pads_to_max_seq_len() {
        let dir = TempDir::new().unwrap();
        let dim = 4;
        write_safetensors(&dir.path().join("a.safetensors"), 3, dim);
        write_safetensors(&dir.path().join("b.safetensors"), 5, dim);
        let tok_path = write_tokenizer(dir.path());
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path).expect("load tokenizer");
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();

        let samples = [
            ManifestSample {
                text: "hello".to_string(),
                latent_path: dir.path().join("a.safetensors"),
                ref_latent_path: None,
                caption: None,
            },
            ManifestSample {
                text: "world".to_string(),
                latent_path: dir.path().join("b.safetensors"),
                ref_latent_path: None,
                caption: None,
            },
        ];
        let refs: Vec<&ManifestSample> = samples.iter().collect();
        let batch = build_batch::<TestBackend>(&refs, &tokenizer, None, &device).unwrap();

        assert_eq!(batch.latent.dims(), [2, 5, dim]);
        assert_eq!(batch.latent_mask.dims(), [2, 5]);

        let mask_data: Vec<bool> = batch.latent_mask.into_data().to_vec().unwrap();
        // First sample: 3 valid, 2 padded
        assert!(mask_data[0]);
        assert!(mask_data[2]);
        assert!(!mask_data[3]);
        assert!(!mask_data[4]);
        // Second sample: all 5 valid
        assert!(mask_data[5]);
        assert!(mask_data[9]);

        assert!(batch.ref_latent.is_none());
    }

    #[test]
    fn build_batch_mixed_speaker_refs() {
        let dir = TempDir::new().unwrap();
        let dim = 4;
        write_safetensors(&dir.path().join("a.safetensors"), 3, dim);
        write_safetensors(&dir.path().join("b.safetensors"), 3, dim);
        write_safetensors(&dir.path().join("ref.safetensors"), 2, dim);
        let tok_path = write_tokenizer(dir.path());
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path).expect("load tokenizer");
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();

        let samples = [
            ManifestSample {
                text: "hello".to_string(),
                latent_path: dir.path().join("a.safetensors"),
                ref_latent_path: Some(dir.path().join("ref.safetensors")),
                caption: None,
            },
            ManifestSample {
                text: "world".to_string(),
                latent_path: dir.path().join("b.safetensors"),
                ref_latent_path: None,
                caption: None,
            },
        ];
        let refs: Vec<&ManifestSample> = samples.iter().collect();
        let batch = build_batch::<TestBackend>(&refs, &tokenizer, None, &device).unwrap();

        assert!(batch.ref_latent.is_some());
        let rl = batch.ref_latent.unwrap();
        assert_eq!(rl.dims(), [2, 2, dim]);

        let rm = batch.ref_latent_mask.unwrap();
        let rm_data: Vec<bool> = rm.into_data().to_vec().unwrap();
        assert!(rm_data[0]); // [0,0] valid
        assert!(rm_data[1]); // [0,1] valid
        assert!(!rm_data[2]); // [1,0] no ref
        assert!(!rm_data[3]); // [1,1] no ref
    }
}
