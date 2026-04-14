//! Training dataset for LoRA fine-tuning.
//!
//! Reads a JSONL manifest where each line has the form:
//! ```json
//! {"text": "...", "latent_path": "...", "ref_latent_path": "..."}
//! ```
//! `ref_latent_path` is optional (speaker conditioning).
//!
//! Latent safetensors files are expected to contain a 3-D tensor of shape
//! `[1, S, D]` (or `[S, D]`), stored under either a key named `"latent"` or
//! the single tensor in the file.

use std::path::{Path, PathBuf};

use burn::tensor::{Bool, Int, Tensor, backend::Backend};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::Deserialize;

use crate::train::LoraTrainConfig;

// ---------------------------------------------------------------------------
// JSONL record
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ManifestRecord {
    text: String,
    latent_path: String,
    ref_latent_path: Option<String>,
}

// ---------------------------------------------------------------------------
// Manifest sample
// ---------------------------------------------------------------------------

/// A single training sample as loaded from the manifest.
#[derive(Debug, Clone)]
pub struct ManifestSample {
    pub text: String,
    pub latent_path: PathBuf,
    pub ref_latent_path: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

/// An in-memory dataset backed by a JSONL manifest file.
pub struct ManifestDataset {
    pub samples: Vec<ManifestSample>,
}

impl ManifestDataset {
    /// Load the dataset from a JSONL manifest.
    ///
    /// Relative paths in the manifest are resolved relative to the manifest's
    /// parent directory.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let parent = path.parent().unwrap_or(Path::new("."));
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Cannot read manifest {}: {e}", path.display()))?;

        let samples = text
            .lines()
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty())
            .map(|(i, line)| -> anyhow::Result<ManifestSample> {
                let rec: ManifestRecord = serde_json::from_str(line)
                    .map_err(|e| anyhow::anyhow!("manifest line {}: {e}", i + 1))?;
                Ok(ManifestSample {
                    text: rec.text,
                    latent_path: parent.join(&rec.latent_path),
                    ref_latent_path: rec.ref_latent_path.map(|p| parent.join(p)),
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Self { samples })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

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
}

// ---------------------------------------------------------------------------
// Batch builder
// ---------------------------------------------------------------------------

/// Iterate over [`ManifestDataset`] yielding [`TrainBatch`] tensors on
/// `device`.
///
/// Supports epoch-level shuffling (seeded for reproducibility) and in-order
/// iteration for validation.
pub struct BatchIterator<'a> {
    dataset: &'a ManifestDataset,
    cfg: &'a LoraTrainConfig,
    tokenizer: tokenizers::Tokenizer,
    /// Permutation indices for the current epoch.  Indices map positions in
    /// the iteration order to positions in `dataset.samples`.
    order: Vec<usize>,
    /// Position within `order` for the current epoch.
    cursor: usize,
    /// Epoch counter, used to derive a per-epoch shuffle seed.
    epoch: u64,
}

impl<'a> BatchIterator<'a> {
    /// Create a new iterator.
    ///
    /// `tokenizer_path` should point to a Hugging Face tokenizer JSON.
    pub fn new(
        dataset: &'a ManifestDataset,
        cfg: &'a LoraTrainConfig,
        tokenizer_path: &Path,
    ) -> anyhow::Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
        let order: Vec<usize> = (0..dataset.len()).collect();
        Ok(Self {
            dataset,
            cfg,
            tokenizer,
            order,
            cursor: 0,
            epoch: 0,
        })
    }

    /// Get the next batch, or `None` if the dataset is exhausted.
    pub fn next_batch<B: Backend>(
        &mut self,
        device: &B::Device,
    ) -> Option<anyhow::Result<TrainBatch<B>>> {
        if self.cursor >= self.order.len() {
            return None;
        }
        let end = (self.cursor + self.cfg.batch_size).min(self.order.len());
        let indices = &self.order[self.cursor..end];
        let samples: Vec<_> = indices.iter().map(|&i| &self.dataset.samples[i]).collect();
        self.cursor = end;
        Some(build_batch::<B>(&samples, &self.tokenizer, device))
    }

    /// Reset to the beginning of the dataset for a new epoch.
    ///
    /// If `cfg.shuffle` is true, the iteration order is shuffled using a
    /// deterministic per-epoch seed derived from `cfg.shuffle_seed`.
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.epoch += 1;
        if self.cfg.shuffle {
            // Derive a deterministic seed per epoch so runs are reproducible.
            let seed = self.cfg.shuffle_seed.wrapping_add(self.epoch);
            let mut rng = StdRng::seed_from_u64(seed);
            self.order.shuffle(&mut rng);
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn load_latent_safetensors<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 3>> {
    let bytes =
        std::fs::read(path).map_err(|e| anyhow::anyhow!("read latent {}: {e}", path.display()))?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| anyhow::anyhow!("parse safetensors {}: {e}", path.display()))?;

    // Prefer "latent" key, else take the only tensor in the file.
    let (name, view) = if tensors.len() == 1 {
        let (n, v) = tensors
            .tensors()
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("empty safetensors: {}", path.display()))?;
        (n, v)
    } else {
        let v = tensors
            .tensor("latent")
            .map_err(|_| anyhow::anyhow!("'latent' key not found in {}", path.display()))?;
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
        other => anyhow::bail!("unsupported dtype {:?} in {}", other, path.display()),
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
        n => anyhow::bail!("latent has {n} dims, expected 2 or 3"),
    };
    Ok(tensor)
}

fn build_batch<B: Backend>(
    samples: &[&ManifestSample],
    tokenizer: &tokenizers::Tokenizer,
    device: &B::Device,
) -> anyhow::Result<TrainBatch<B>> {
    // ------------------------------------------------------------------
    // 1. Load latents and tokenise text
    // ------------------------------------------------------------------
    let mut latents: Vec<Tensor<B, 3>> = Vec::with_capacity(samples.len());
    let mut ref_latents: Vec<Option<Tensor<B, 3>>> = Vec::with_capacity(samples.len());
    let mut text_token_seqs: Vec<Vec<u32>> = Vec::with_capacity(samples.len());

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
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let ids = enc.get_ids().to_vec();
        anyhow::ensure!(
            !ids.is_empty(),
            "tokenised text is empty for sample '{}' — every training sample \
             must produce at least one token (required for safe_softmax=false in SDPA)",
            s.text,
        );
        text_token_seqs.push(ids);
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

    Ok(TrainBatch {
        latent,
        latent_mask,
        loss_mask,
        text_ids,
        text_mask,
        ref_latent,
        ref_latent_mask,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    type TestBackend = burn::backend::NdArray;

    /// Create a minimal safetensors file with shape `[1, seq_len, dim]`.
    fn write_safetensors(path: &Path, seq_len: usize, dim: usize) {
        use safetensors::tensor::TensorView;

        let data: Vec<f32> = (0..seq_len * dim).map(|i| (i as f32) * 0.01).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let shape = vec![1, seq_len, dim];
        let view = TensorView::new(safetensors::Dtype::F32, shape, &bytes).expect("TensorView");
        let tensors: Vec<(&str, TensorView<'_>)> = vec![("latent", view)];
        safetensors::serialize_to_file(tensors, None, path).expect("write safetensors");
    }

    /// Create a minimal byte-level BPE tokenizer JSON.
    fn write_tokenizer(dir: &Path) -> PathBuf {
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

        // Use Whitespace pre-tokenizer (simpler, no add_prefix_space issues).
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

    /// Write a JSONL manifest.
    fn write_manifest(dir: &Path, entries: &[(&str, &str, Option<&str>)]) -> PathBuf {
        let manifest_path = dir.join("manifest.jsonl");
        let mut f = std::fs::File::create(&manifest_path).expect("create manifest");
        for (text, latent, ref_latent) in entries {
            let rec = if let Some(rl) = ref_latent {
                serde_json::json!({"text": text, "latent_path": latent, "ref_latent_path": rl})
            } else {
                serde_json::json!({"text": text, "latent_path": latent})
            };
            writeln!(f, "{}", rec).expect("write manifest line");
        }
        manifest_path
    }

    // -------------------------------------------------------------------
    // ManifestDataset::load
    // -------------------------------------------------------------------

    #[test]
    fn load_manifest_basic() {
        let dir = TempDir::new().unwrap();
        let manifest = write_manifest(
            dir.path(),
            &[
                ("hello", "a.safetensors", None),
                ("world", "b.safetensors", Some("ref.safetensors")),
            ],
        );
        let ds = ManifestDataset::load(&manifest).unwrap();
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.samples[0].text, "hello");
        assert!(ds.samples[0].ref_latent_path.is_none());
        assert_eq!(ds.samples[1].text, "world");
        assert!(ds.samples[1].ref_latent_path.is_some());
    }

    #[test]
    fn load_manifest_blank_lines_skipped() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("manifest.jsonl");
        std::fs::write(
            &path,
            concat!(
                "{\"text\":\"a\",\"latent_path\":\"a.safetensors\"}\n",
                "\n   \n",
                "{\"text\":\"b\",\"latent_path\":\"b.safetensors\"}\n",
            ),
        )
        .unwrap();
        let ds = ManifestDataset::load(&path).unwrap();
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn load_manifest_resolves_paths_relative_to_manifest() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        let manifest = write_manifest(&sub, &[("t", "data/a.safetensors", None)]);
        let ds = ManifestDataset::load(&manifest).unwrap();
        assert!(ds.samples[0].latent_path.starts_with(&sub));
    }

    #[test]
    fn load_manifest_empty_is_ok() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.jsonl");
        std::fs::write(&path, "").unwrap();
        let ds = ManifestDataset::load(&path).unwrap();
        assert!(ds.is_empty());
    }

    // -------------------------------------------------------------------
    // BatchIterator shuffle determinism
    // -------------------------------------------------------------------

    #[test]
    fn shuffle_is_deterministic_across_resets() {
        let dir = TempDir::new().unwrap();
        for name in &[
            "a.safetensors",
            "b.safetensors",
            "c.safetensors",
            "d.safetensors",
        ] {
            write_safetensors(&dir.path().join(name), 2, 4);
        }
        let manifest = write_manifest(
            dir.path(),
            &[
                ("a", "a.safetensors", None),
                ("b", "b.safetensors", None),
                ("c", "c.safetensors", None),
                ("d", "d.safetensors", None),
            ],
        );
        let tok_path = write_tokenizer(dir.path());
        let ds = ManifestDataset::load(&manifest).unwrap();
        let cfg = LoraTrainConfig {
            shuffle: true,
            shuffle_seed: 123,
            batch_size: 2,
            ..LoraTrainConfig::default()
        };

        let mut iter1 = BatchIterator::new(&ds, &cfg, &tok_path).unwrap();
        iter1.reset();
        let order1 = iter1.order.clone();
        iter1.reset();
        let order1_e2 = iter1.order.clone();

        let mut iter2 = BatchIterator::new(&ds, &cfg, &tok_path).unwrap();
        iter2.reset();
        let order2 = iter2.order.clone();
        iter2.reset();
        let order2_e2 = iter2.order.clone();

        assert_eq!(order1, order2, "same seed, epoch 1 should match");
        assert_eq!(order1_e2, order2_e2, "same seed, epoch 2 should match");
        assert_ne!(order1, order1_e2, "different epochs should differ");
    }

    #[test]
    fn no_shuffle_preserves_order() {
        let dir = TempDir::new().unwrap();
        for name in &["a.safetensors", "b.safetensors", "c.safetensors"] {
            write_safetensors(&dir.path().join(name), 2, 4);
        }
        let manifest = write_manifest(
            dir.path(),
            &[
                ("a", "a.safetensors", None),
                ("b", "b.safetensors", None),
                ("c", "c.safetensors", None),
            ],
        );
        let tok_path = write_tokenizer(dir.path());
        let ds = ManifestDataset::load(&manifest).unwrap();
        let cfg = LoraTrainConfig {
            shuffle: false,
            batch_size: 2,
            ..LoraTrainConfig::default()
        };

        let mut iter = BatchIterator::new(&ds, &cfg, &tok_path).unwrap();
        iter.reset();
        let expected: Vec<usize> = (0..3).collect();
        assert_eq!(iter.order, expected);
        iter.reset();
        assert_eq!(iter.order, expected);
    }

    // -------------------------------------------------------------------
    // build_batch padding
    // -------------------------------------------------------------------

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
            },
            ManifestSample {
                text: "world".to_string(),
                latent_path: dir.path().join("b.safetensors"),
                ref_latent_path: None,
            },
        ];
        let refs: Vec<&ManifestSample> = samples.iter().collect();
        let batch = build_batch::<TestBackend>(&refs, &tokenizer, &device).unwrap();

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
            },
            ManifestSample {
                text: "world".to_string(),
                latent_path: dir.path().join("b.safetensors"),
                ref_latent_path: None,
            },
        ];
        let refs: Vec<&ManifestSample> = samples.iter().collect();
        let batch = build_batch::<TestBackend>(&refs, &tokenizer, &device).unwrap();

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

    // -------------------------------------------------------------------
    // Epoch exhaustion
    // -------------------------------------------------------------------

    #[test]
    fn next_batch_returns_none_when_exhausted() {
        let dir = TempDir::new().unwrap();
        let dim = 4;
        for name in &["a.safetensors", "b.safetensors", "c.safetensors"] {
            write_safetensors(&dir.path().join(name), 2, dim);
        }
        let manifest = write_manifest(
            dir.path(),
            &[
                ("a", "a.safetensors", None),
                ("b", "b.safetensors", None),
                ("c", "c.safetensors", None),
            ],
        );
        let tok_path = write_tokenizer(dir.path());
        let ds = ManifestDataset::load(&manifest).unwrap();
        let cfg = LoraTrainConfig {
            batch_size: 2,
            shuffle: false,
            ..LoraTrainConfig::default()
        };
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();

        let mut iter = BatchIterator::new(&ds, &cfg, &tok_path).unwrap();
        let b1 = iter.next_batch::<TestBackend>(&device);
        assert!(b1.is_some());
        let b2 = iter.next_batch::<TestBackend>(&device);
        assert!(b2.is_some());
        let b3 = iter.next_batch::<TestBackend>(&device);
        assert!(b3.is_none());
    }
}
