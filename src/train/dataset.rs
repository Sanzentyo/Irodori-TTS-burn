//! Training dataset for LoRA fine-tuning.
//!
//! Split into submodules:
//! - [`manifest`]: JSONL manifest loading and in-memory dataset
//! - [`batch`]: Batch construction with padding and masking

mod batch;
mod manifest;

use std::path::Path;

use burn::tensor::backend::Backend;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::error::IrodoriError;
use crate::train::LoraTrainConfig;

pub use batch::TrainBatch;
// ManifestSample is re-exported for API completeness: it's the element type
// of `ManifestDataset::samples` (a pub field).
pub use manifest::{ManifestDataset, ManifestSample};

// ---------------------------------------------------------------------------
// Batch iterator
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
    /// Optional separate tokenizer for caption text.
    caption_tokenizer: Option<tokenizers::Tokenizer>,
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
    /// `caption_tokenizer_path` is optional -- when provided, a separate
    /// tokenizer is loaded for caption text; otherwise captions are
    /// tokenised with the main text tokenizer.
    pub fn new(
        dataset: &'a ManifestDataset,
        cfg: &'a LoraTrainConfig,
        tokenizer_path: &Path,
        caption_tokenizer_path: Option<&Path>,
    ) -> crate::error::Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| IrodoriError::Tokenizer(format!("load tokenizer: {e}")))?;
        let caption_tokenizer = caption_tokenizer_path
            .map(|p| {
                tokenizers::Tokenizer::from_file(p)
                    .map_err(|e| IrodoriError::Tokenizer(format!("load caption tokenizer: {e}")))
            })
            .transpose()?;
        let order: Vec<usize> = (0..dataset.len()).collect();
        Ok(Self {
            dataset,
            cfg,
            tokenizer,
            caption_tokenizer,
            order,
            cursor: 0,
            epoch: 0,
        })
    }

    /// Get the next batch, or `None` if the dataset is exhausted.
    pub fn next_batch<B: Backend>(
        &mut self,
        device: &B::Device,
    ) -> Option<crate::error::Result<TrainBatch<B>>> {
        if self.cursor >= self.order.len() {
            return None;
        }
        let end = (self.cursor + self.cfg.batch_size).min(self.order.len());
        let indices = &self.order[self.cursor..end];
        let samples: Vec<_> = indices.iter().map(|&i| &self.dataset.samples[i]).collect();
        self.cursor = end;
        Some(batch::build_batch::<B>(
            &samples,
            &self.tokenizer,
            self.caption_tokenizer.as_ref(),
            device,
        ))
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
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

    fn write_manifest(dir: &Path, entries: &[(&str, &str, Option<&str>)]) -> std::path::PathBuf {
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

        let mut iter1 = BatchIterator::new(&ds, &cfg, &tok_path, None).unwrap();
        iter1.reset();
        let order1 = iter1.order.clone();
        iter1.reset();
        let order1_e2 = iter1.order.clone();

        let mut iter2 = BatchIterator::new(&ds, &cfg, &tok_path, None).unwrap();
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

        let mut iter = BatchIterator::new(&ds, &cfg, &tok_path, None).unwrap();
        iter.reset();
        let expected: Vec<usize> = (0..3).collect();
        assert_eq!(iter.order, expected);
        iter.reset();
        assert_eq!(iter.order, expected);
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

        let mut iter = BatchIterator::new(&ds, &cfg, &tok_path, None).unwrap();
        let b1 = iter.next_batch::<TestBackend>(&device);
        assert!(b1.is_some());
        let b2 = iter.next_batch::<TestBackend>(&device);
        assert!(b2.is_some());
        let b3 = iter.next_batch::<TestBackend>(&device);
        assert!(b3.is_none());
    }
}
