//! LoRA (Low-Rank Adaptation) adapter loading and weight-merging for inference.
//!
//! This module enables transparent LoRA inference without changing the model
//! structure: LoRA weights are **merged** into the base weights at load time
//! using the formula `W_merged = W_base + (lora_alpha / r) * lora_B @ lora_A`.
//!
//! # Adapter file layout
//! An adapter directory is expected to contain:
//! - `adapter_config.json` — LoRA hyper-parameters (r, lora_alpha, etc.)
//! - `adapter_model.safetensors` (preferred) or `adapter_model.bin`
//!
//! LoRA weight keys follow the PEFT convention:
//! ```text
//! base_model.model.<module_path>.lora_A.default.weight   # [r, d_in]
//! base_model.model.<module_path>.lora_B.default.weight   # [d_out, r]
//! ```
//!
//! The corresponding base weight key in the plain (non-PEFT) safetensors is
//! `<module_path>.weight` (i.e. without the `base_model.model.` prefix).

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

use crate::error::{IrodoriError, Result};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Hyper-parameters parsed from PEFT's `adapter_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct LoraAdapterConfig {
    /// LoRA rank.
    pub r: usize,
    /// LoRA scaling factor (alpha / r gives the effective scale).
    pub lora_alpha: f64,
    /// Bias mode: `"none"`, `"all"`, or `"lora_only"`.
    #[serde(default = "default_bias")]
    pub bias: String,
    /// Module path patterns that LoRA was applied to.
    #[serde(default)]
    pub target_modules: LoraTargetModules,
}

impl LoraAdapterConfig {
    /// Effective LoRA scale: `lora_alpha / r`.
    #[inline]
    pub fn scale(&self) -> f64 {
        self.lora_alpha / self.r as f64
    }

    /// Load from `adapter_config.json` inside `adapter_dir`.
    pub fn load(adapter_dir: &Path) -> Result<Self> {
        let path = adapter_dir.join("adapter_config.json");
        let json = std::fs::read_to_string(&path).map_err(|e| {
            IrodoriError::Weight(format!(
                "cannot read adapter_config.json at {}: {e}",
                path.display()
            ))
        })?;
        let cfg: Self = serde_json::from_str(&json).map_err(|e| {
            IrodoriError::Weight(format!(
                "malformed adapter_config.json at {}: {e}",
                path.display()
            ))
        })?;
        Ok(cfg)
    }
}

fn default_bias() -> String {
    "none".to_owned()
}

/// How the target modules are specified in `adapter_config.json`.
///
/// PEFT serialises `target_modules` as either a single regex string or a list
/// of module names / regex patterns, so we accept both forms.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(untagged)]
pub enum LoraTargetModules {
    /// A list of module name patterns (most common PEFT output).
    List(Vec<String>),
    /// A single regex string.
    Regex(String),
    #[default]
    None,
}

// ---------------------------------------------------------------------------
// Adapter path resolution
// ---------------------------------------------------------------------------

/// Locate the adapter-weights file in `adapter_dir`.
///
/// Tries `adapter_model.safetensors` first, then `adapter_model.bin`.
pub fn find_adapter_weights(adapter_dir: &Path) -> Result<PathBuf> {
    for name in ["adapter_model.safetensors", "adapter_model.bin"] {
        let p = adapter_dir.join(name);
        if p.exists() {
            return Ok(p);
        }
    }
    Err(IrodoriError::Weight(format!(
        "no adapter weights found in {}",
        adapter_dir.display()
    )))
}

/// Returns `true` if `adapter_dir` looks like a PEFT LoRA adapter directory.
pub fn is_lora_adapter_dir(adapter_dir: &Path) -> bool {
    if !adapter_dir.is_dir() {
        return false;
    }
    if !adapter_dir.join("adapter_config.json").exists() {
        return false;
    }
    ["adapter_model.safetensors", "adapter_model.bin"]
        .iter()
        .any(|name| adapter_dir.join(name).exists())
}

// ---------------------------------------------------------------------------
// Raw f32 adapter tensors
// ---------------------------------------------------------------------------

/// Decode a safetensors view to a flat `Vec<f32>`, converting from its native dtype.
fn to_f32_vec(data: &[u8], dtype: Dtype) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => Ok(data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        Dtype::BF16 => Ok(data
            .chunks_exact(2)
            .map(|b| {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect()),
        Dtype::F16 => Ok(data
            .chunks_exact(2)
            .map(|b| {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                half::f16::from_bits(bits).to_f32()
            })
            .collect()),
        other => Err(IrodoriError::Dtype(String::new(), format!("{other:?}"))),
    }
}

/// Flat f32 tensor with shape metadata.
#[derive(Debug)]
struct F32Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl F32Tensor {
    fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

// ---------------------------------------------------------------------------
// The merging logic
// ---------------------------------------------------------------------------

/// Pre-scan an adapter directory to find which base weight keys will be modified
/// by `merge_lora`.  Cheaper than a full decode — only reads tensor names.
///
/// Returns a `Vec` of base weight keys (e.g. `"dit.blocks.0.attn.q_proj.weight"`)
/// that the adapter targets.
pub fn pre_scan_lora_keys(adapter_dir: &Path) -> Result<Vec<String>> {
    let adapter_path = find_adapter_weights(adapter_dir)?;
    let adapter_bytes = std::fs::read(&adapter_path).map_err(|e| {
        IrodoriError::Weight(format!(
            "cannot read adapter weights {}: {e}",
            adapter_path.display()
        ))
    })?;

    let st = SafeTensors::deserialize(&adapter_bytes)
        .map_err(|e| IrodoriError::Weight(format!("malformed adapter safetensors: {e}")))?;

    let keys = st
        .names()
        .into_iter()
        .filter(|k| k.contains(".lora_A."))
        .filter_map(|k| {
            let base = k.strip_prefix("base_model.model.").unwrap_or(k);
            let pos = base.find(".lora_A.")?;
            Some(format!("{}.weight", &base[..pos]))
        })
        .collect();

    Ok(keys)
}

/// Merge LoRA adapter weights from `adapter_dir` into the raw f32 `base_tensors` map.
///
/// # Arguments
/// * `base_tensors` — mutable map from key → (f32 data, shape); modified in-place.
/// * `base_dtype` — the dtype to re-encode the merged result as (matches base checkpoint dtype).
/// * `adapter_dir` — path to a PEFT LoRA adapter directory.
///
/// Returns the keys of base weights that were actually merged.
///
/// After merging, the caller should update the `TensorStore` bytes for each
/// affected key by re-encoding from f32 back to the original dtype.
pub fn merge_lora(
    base_tensors: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    adapter_dir: &Path,
) -> Result<Vec<String>> {
    let cfg = LoraAdapterConfig::load(adapter_dir)?;
    let adapter_path = find_adapter_weights(adapter_dir)?;
    let adapter_bytes = std::fs::read(&adapter_path).map_err(|e| {
        IrodoriError::Weight(format!(
            "cannot read adapter weights {}: {e}",
            adapter_path.display()
        ))
    })?;

    let st = SafeTensors::deserialize(&adapter_bytes)
        .map_err(|e| IrodoriError::Weight(format!("malformed adapter safetensors: {e}")))?;

    // Index adapter tensors by name.
    let mut adapter: HashMap<String, F32Tensor> = HashMap::new();
    for (name, view) in st.tensors() {
        let data = to_f32_vec(view.data(), view.dtype())
            .map_err(|e| IrodoriError::Weight(format!("{name}: {e}")))?;
        adapter.insert(
            name.clone(),
            F32Tensor {
                data,
                shape: view.shape().to_vec(),
            },
        );
    }

    let scale = cfg.scale() as f32;
    let mut merged_keys: Vec<String> = Vec::new();

    // Find all lora_A keys and merge.
    let lora_a_keys: Vec<String> = adapter
        .keys()
        .filter(|k| k.contains(".lora_A."))
        .cloned()
        .collect();

    for lora_a_key in lora_a_keys {
        // Derive lora_B and base key from the lora_A key.
        // Pattern: `base_model.model.<path>.lora_A.default.weight`
        //      →   `base_model.model.<path>.lora_B.default.weight`
        //      →   `<path>.weight`   (base key, no prefix)
        let lora_b_key = lora_a_key.replace(".lora_A.", ".lora_B.");

        let Some(lora_a) = adapter.get(&lora_a_key) else {
            continue;
        };
        let Some(lora_b) = adapter.get(&lora_b_key) else {
            continue;
        };

        // Derive the base key by stripping "base_model.model." prefix and
        // ".lora_A.default.weight" suffix.
        let base_key = lora_a_key
            .strip_prefix("base_model.model.")
            .unwrap_or(&lora_a_key);
        // Remove ".lora_A.default.weight" → find and replace the lora sub-path
        let base_key = if let Some(pos) = base_key.find(".lora_A.") {
            format!("{}.weight", &base_key[..pos])
        } else {
            continue;
        };

        let Some((base_data, base_shape)) = base_tensors.get_mut(&base_key) else {
            // This base weight doesn't exist in the store (might be bias, etc.)
            continue;
        };

        // Shapes:
        //   lora_A: [r, d_in]
        //   lora_B: [d_out, r]
        //   base_W: [d_out, d_in]
        if lora_a.shape.len() != 2 || lora_b.shape.len() != 2 {
            return Err(IrodoriError::Weight(format!(
                "LoRA rank must be 2-D for {base_key}: lora_A rank={}, lora_B rank={}",
                lora_a.shape.len(),
                lora_b.shape.len()
            )));
        }
        let r = lora_a.shape[0];
        let d_in = lora_a.shape[1];
        let d_out = lora_b.shape[0];

        if lora_a.numel() != r * d_in || lora_b.numel() != d_out * r {
            return Err(IrodoriError::Weight(format!(
                "LoRA shape mismatch for {base_key}: lora_A={:?}, lora_B={:?}",
                lora_a.shape, lora_b.shape
            )));
        }
        if base_data.len() != d_out * d_in {
            return Err(IrodoriError::Weight(format!(
                "LoRA base shape mismatch for {base_key}: base={:?}, expected [{d_out}x{d_in}]",
                base_shape
            )));
        }

        // delta = lora_B @ lora_A   (d_out x d_in)
        // Stored as row-major: delta[i*d_in + j] = sum_k lora_B[i*r+k] * lora_A[k*d_in+j]
        let delta = matmul_f32(&lora_b.data, d_out, r, &lora_a.data, r, d_in);

        // Merge: W = W + scale * delta
        for (w, d) in base_data.iter_mut().zip(delta.iter()) {
            *w += scale * d;
        }

        merged_keys.push(base_key);
    }

    Ok(merged_keys)
}

/// Row-major matrix multiply: C[m×n] = A[m×k] @ B[k×n].
fn matmul_f32(a: &[f32], m: usize, k: usize, b: &[f32], _k2: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for kk in 0..k {
            let a_val = a[i * k + kk];
            for j in 0..n {
                c[i * n + j] += a_val * b[kk * n + j];
            }
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Key-prefix handling helpers
// ---------------------------------------------------------------------------

/// Returns `true` if any key in the iterator starts with `"base_model.model."`.
pub fn has_peft_prefix<'a>(mut keys: impl Iterator<Item = &'a str>) -> bool {
    keys.any(|k| k.starts_with("base_model.model."))
}

/// Strip the `"base_model.model."` prefix from a key if present.
pub fn strip_peft_prefix(key: &str) -> &str {
    key.strip_prefix("base_model.model.").unwrap_or(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_computation() {
        let cfg = LoraAdapterConfig {
            r: 16,
            lora_alpha: 32.0,
            bias: "none".to_owned(),
            target_modules: LoraTargetModules::None,
        };
        let eps = 1e-9_f64;
        assert!((cfg.scale() - 2.0).abs() < eps);
    }

    #[test]
    fn matmul_2x2() {
        // [1,2; 3,4] @ [5,6; 7,8] = [19,22; 43,50]
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![5.0_f32, 6.0, 7.0, 8.0];
        let c = matmul_f32(&a, 2, 2, &b, 2, 2);
        assert!((c[0] - 19.0).abs() < 1e-5);
        assert!((c[1] - 22.0).abs() < 1e-5);
        assert!((c[2] - 43.0).abs() < 1e-5);
        assert!((c[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn strip_prefix() {
        assert_eq!(
            strip_peft_prefix("base_model.model.blocks.0.attention.wq.weight"),
            "blocks.0.attention.wq.weight"
        );
        assert_eq!(strip_peft_prefix("blocks.0.weight"), "blocks.0.weight");
    }
}
