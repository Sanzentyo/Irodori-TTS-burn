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
