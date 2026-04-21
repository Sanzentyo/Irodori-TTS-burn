//! Shared test helpers for weights submodule tests.

use half::{bf16, f16};
use safetensors::{Dtype, tensor::TensorView};
use std::collections::HashMap;
use std::path::Path;
use tempfile::NamedTempFile;

/// Create a safetensors file on disk with given tensors and config_json metadata.
pub(super) fn write_safetensors(
    tensors: &[(&str, Vec<u8>, Dtype, Vec<usize>)],
    config_json: &str,
) -> NamedTempFile {
    let views: Vec<(&str, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, data, dtype, shape)| {
            (*name, TensorView::new(*dtype, shape.clone(), data).unwrap())
        })
        .collect();

    let mut metadata = HashMap::new();
    metadata.insert("config_json".to_string(), config_json.to_string());

    let serialised = safetensors::tensor::serialize(views, Some(metadata)).expect("serialize");

    let file = NamedTempFile::new().unwrap();
    std::fs::write(file.path(), serialised).unwrap();
    file
}

/// Write a PEFT-style LoRA adapter directory under `dir`.
///
/// Creates:
/// - `adapter_config.json` with `{"r": r, "lora_alpha": alpha, "bias": "none"}`
/// - `adapter_model.safetensors` with the given tensors (no metadata required)
#[cfg(feature = "lora")]
pub(super) fn write_adapter_dir(
    dir: &Path,
    r: usize,
    alpha: f64,
    tensors: &[(&str, Vec<u8>, Dtype, Vec<usize>)],
) {
    // adapter_config.json
    let cfg = serde_json::json!({
        "r": r,
        "lora_alpha": alpha,
        "bias": "none"
    });
    std::fs::write(dir.join("adapter_config.json"), cfg.to_string()).unwrap();

    // adapter_model.safetensors (no config metadata)
    let views: Vec<(&str, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, data, dtype, shape)| {
            (*name, TensorView::new(*dtype, shape.clone(), data).unwrap())
        })
        .collect();
    let serialised =
        safetensors::tensor::serialize(views, None::<HashMap<String, String>>).unwrap();
    std::fs::write(dir.join("adapter_model.safetensors"), serialised).unwrap();
}

/// Encode f32 values to little-endian bytes.
pub(super) fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Encode f32 values to bf16 little-endian bytes.
pub(super) fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
        .collect()
}

/// Encode f32 values to f16 little-endian bytes.
pub(super) fn f16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| f16::from_f32(*v).to_le_bytes())
        .collect()
}

/// Minimal config_json for a small model (unused for unit tests,
/// but required by TensorStore::load).
pub(super) fn test_config_json() -> String {
    serde_json::json!({
        "model_dim": 64,
        "num_heads": 2,
        "head_dim": 32,
        "num_layers": 1,
        "text_vocab_size": 100,
        "text_layers": 1,
        "norm_eps": 1e-6,
        "timestep_embed_dim": 64,
        "speaker_patch_size": 4,
        "patched_latent_dim": 64,
        "condition_provider": "speaker",
        "speaker_layers": 1,
        "use_caption_condition": false,
        "rescale_k": 1.0,
        "rescale_sigma": 0.0,
        "truncation_factor": 1.0
    })
    .to_string()
}
