//! LoRA adapter checkpoint saving in PEFT-compatible format.
//!
//! Key naming convention (matches the `base_model.model.` prefix that
//! `src/lora.rs` strips when loading adapters):
//! ```text
//! base_model.model.blocks.{i}.attention.{proj}.lora_A.default.weight  // [r, in]
//! base_model.model.blocks.{i}.attention.{proj}.lora_B.default.weight  // [out, r]
//! ```
//!
//! Also writes `adapter_config.json`.

use std::path::Path;

use burn::tensor::backend::Backend;
use safetensors::{Dtype, tensor::TensorView};
use serde::Serialize;

use crate::error::IrodoriError;
use crate::train::{LoraConfig, LoraTextToLatentRfDiT, lora_layer::LoraLinear};

// ---------------------------------------------------------------------------
// adapter_config.json schema
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct AdapterConfig<'a> {
    peft_type: &'static str,
    r: usize,
    lora_alpha: f32,
    target_modules: &'a [String],
    bias: &'static str,
    task_type: &'static str,
    base_model_name_or_path: Option<String>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert f32 slice to little-endian bytes (safetensors wire format).
fn f32_to_le_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|&v| v.to_le_bytes()).collect()
}

/// Owned bytes + shape for lora_a and lora_b respectively.
type LoraBytes = (Vec<u8>, Vec<usize>, Vec<u8>, Vec<usize>);

/// Extract owned byte buffers and shape from a `LoraLinear` layer.
///
/// Weights are always serialised as F32 regardless of backend float type so that
/// the resulting adapter is loadable by both the Python PEFT library and
/// `src/lora.rs` (which expects F32 safetensors).
fn extract_lora<B: Backend>(layer: &LoraLinear<B>) -> crate::error::Result<LoraBytes> {
    let a = layer.lora_a.val();
    let b = layer.lora_b.val();
    let a_shape = a.dims().to_vec();
    let b_shape = b.dims().to_vec();
    let a_bytes = f32_to_le_bytes(
        &a.into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|e| IrodoriError::Checkpoint(format!("lora_a tensor conversion: {e:?}")))?,
    );
    let b_bytes = f32_to_le_bytes(
        &b.into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|e| IrodoriError::Checkpoint(format!("lora_b tensor conversion: {e:?}")))?,
    );
    Ok((a_bytes, a_shape, b_bytes, b_shape))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Save LoRA adapter weights and `adapter_config.json` to `output_dir`.
///
/// Tensors are written as F32 safetensors with PEFT-compatible key names so the
/// adapter can be loaded by both the Python PEFT library and Rust `src/lora.rs`.
///
/// Output directory: `{output_dir}/step-{step:07}/`.
pub fn save_lora_adapter<B: Backend>(
    model: &LoraTextToLatentRfDiT<B>,
    lora_cfg: &LoraConfig,
    output_dir: &Path,
    step: usize,
) -> crate::error::Result<()> {
    let dir = output_dir.join(format!("step-{step:07}"));
    std::fs::create_dir_all(&dir)?;

    // Phase 1: collect owned byte buffers so lifetimes outlive the TensorViews.
    type Entry = (String, Vec<u8>, Vec<usize>);
    let mut entries: Vec<Entry> = Vec::new();

    let mut push = |key: String, bytes: Vec<u8>, shape: Vec<usize>| {
        entries.push((key, bytes, shape));
    };

    for (i, block) in model.blocks.iter().enumerate() {
        let attn = &block.attention;
        let pfx = format!("base_model.model.blocks.{i}.attention");

        macro_rules! save_proj {
            ($proj:ident) => {{
                let (ab, ash, bb, bsh) = extract_lora(&attn.$proj)?;
                push(
                    format!("{pfx}.{}.lora_A.default.weight", stringify!($proj)),
                    ab,
                    ash,
                );
                push(
                    format!("{pfx}.{}.lora_B.default.weight", stringify!($proj)),
                    bb,
                    bsh,
                );
            }};
        }

        macro_rules! save_proj_opt {
            ($proj:ident) => {
                if let Some(layer) = &attn.$proj {
                    let (ab, ash, bb, bsh) = extract_lora(layer)?;
                    push(
                        format!("{pfx}.{}.lora_A.default.weight", stringify!($proj)),
                        ab,
                        ash,
                    );
                    push(
                        format!("{pfx}.{}.lora_B.default.weight", stringify!($proj)),
                        bb,
                        bsh,
                    );
                }
            };
        }

        save_proj!(wq);
        save_proj!(wk);
        save_proj!(wv);
        save_proj!(wk_text);
        save_proj!(wv_text);
        save_proj!(gate);
        save_proj!(wo);
        save_proj_opt!(wk_speaker);
        save_proj_opt!(wv_speaker);
        save_proj_opt!(wk_caption);
        save_proj_opt!(wv_caption);
    }

    // Phase 2: build TensorViews borrowing from Phase 1 data.
    let views: Vec<(String, TensorView<'_>)> = entries
        .iter()
        .map(|(key, data, shape)| {
            TensorView::new(Dtype::F32, shape.clone(), data.as_slice())
                .map(|v| (key.clone(), v))
                .map_err(|e| IrodoriError::Checkpoint(format!("TensorView: {e:?}")))
        })
        .collect::<crate::error::Result<_>>()?;

    // Phase 3: serialize.
    let out_path = dir.join("adapter_model.safetensors");
    safetensors::serialize_to_file(views, None, &out_path)
        .map_err(|e| IrodoriError::Checkpoint(format!("serialize safetensors: {e}")))?;

    let adapter_cfg = AdapterConfig {
        peft_type: "LORA",
        r: lora_cfg.r,
        lora_alpha: lora_cfg.alpha,
        target_modules: &lora_cfg.target_modules,
        bias: "none",
        task_type: "UNCONDITIONAL_GENERATION",
        base_model_name_or_path: None,
    };
    std::fs::write(
        dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&adapter_cfg)?,
    )?;

    tracing::info!(step, path = %dir.display(), "saved LoRA adapter");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use tempfile::TempDir;

    type TestBackend = NdArray;

    fn make_tiny_model() -> (LoraTextToLatentRfDiT<TestBackend>, LoraConfig) {
        let cfg = crate::train::tiny_model_config();
        let lora_cfg = LoraConfig {
            r: 2,
            alpha: 4.0,
            target_modules: vec!["wq".into(), "wk".into()],
        };
        let device = Default::default();
        let model =
            LoraTextToLatentRfDiT::<TestBackend>::new(&cfg, lora_cfg.r, lora_cfg.alpha, &device);
        (model, lora_cfg)
    }

    #[test]
    fn f32_to_le_bytes_roundtrip() {
        let vals = [1.0f32, -0.5, 3.125, 0.0];
        let bytes = f32_to_le_bytes(&vals);
        assert_eq!(bytes.len(), 4 * 4);
        let recovered: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(recovered, vals);
    }

    #[test]
    fn save_creates_correct_directory_structure() {
        let dir = TempDir::new().unwrap();
        let (model, lora_cfg) = make_tiny_model();

        save_lora_adapter(&model, &lora_cfg, dir.path(), 42).unwrap();

        let step_dir = dir.path().join("step-0000042");
        assert!(step_dir.exists(), "step directory must exist");
        assert!(
            step_dir.join("adapter_model.safetensors").exists(),
            "safetensors file must exist"
        );
        assert!(
            step_dir.join("adapter_config.json").exists(),
            "adapter_config.json must exist"
        );
    }

    #[test]
    fn saved_adapter_config_has_correct_fields() {
        let dir = TempDir::new().unwrap();
        let (model, lora_cfg) = make_tiny_model();

        save_lora_adapter(&model, &lora_cfg, dir.path(), 1).unwrap();

        let json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join("step-0000001/adapter_config.json")).unwrap(),
        )
        .unwrap();

        assert_eq!(json["peft_type"], "LORA");
        assert_eq!(json["r"], 2);
        assert_eq!(json["lora_alpha"], 4.0);
        assert_eq!(json["bias"], "none");
        let modules: Vec<String> = serde_json::from_value(json["target_modules"].clone()).unwrap();
        assert_eq!(modules, vec!["wq", "wk"]);
    }

    #[test]
    fn saved_safetensors_has_peft_keys_and_shapes() {
        let dir = TempDir::new().unwrap();
        let (model, lora_cfg) = make_tiny_model();
        let r = lora_cfg.r;

        save_lora_adapter(&model, &lora_cfg, dir.path(), 0).unwrap();

        let st_path = dir.path().join("step-0000000/adapter_model.safetensors");
        let data = std::fs::read(&st_path).unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&data).unwrap();

        // The tiny model has 1 block with speaker mode:
        // Projections: wq, wk, wv, wk_text, wv_text, gate, wo + wk_speaker, wv_speaker
        // Each has lora_A and lora_B → 9 × 2 = 18 tensors
        let names: Vec<&str> = tensors.names().into_iter().collect();
        assert!(
            names.len() >= 14,
            "expected at least 14 tensors (7 projections × 2), got {}",
            names.len()
        );

        // Verify PEFT key naming convention
        let key_a = "base_model.model.blocks.0.attention.wq.lora_A.default.weight";
        let key_b = "base_model.model.blocks.0.attention.wq.lora_B.default.weight";
        assert!(names.contains(&key_a), "missing key: {key_a}");
        assert!(names.contains(&key_b), "missing key: {key_b}");

        // Verify shapes: lora_A is [r, in_features], lora_B is [out_features, r]
        let info_a = tensors.tensor(key_a).unwrap();
        assert_eq!(info_a.shape(), &[r, 32], "lora_A shape: [r, model_dim]");

        let kv_dim = 4 * 8; // num_heads(4) * head_dim(32/4=8)
        let info_b = tensors.tensor(key_b).unwrap();
        assert_eq!(info_b.shape(), &[kv_dim, r], "lora_B shape: [kv_dim, r]");

        // All tensors should be F32
        for name in &names {
            let t = tensors.tensor(name).unwrap();
            assert_eq!(t.dtype(), Dtype::F32, "{name} must be F32");
        }
    }
}
