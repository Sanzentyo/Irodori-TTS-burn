//! LoRA model weight loading.
//!
//! Re-exports the `load_lora_model` function from [`crate::weights`],
//! keeping the training API self-contained under the `train` module.

pub use crate::weights::load_lora_model;

use std::path::Path;

use burn::{
    module::Param,
    tensor::{Tensor, TensorData, backend::AutodiffBackend},
};
use safetensors::SafeTensors;

use crate::{
    error::{IrodoriError, Result},
    train::LoraTextToLatentRfDiT,
};

/// Restore LoRA adapter weights from a checkpoint directory into a training
/// model (warm restart).
///
/// Only `lora_A` / `lora_B` params are updated.  Base weights and optimizer
/// state are left unchanged.
///
/// Key format expected in `adapter_model.safetensors`:
/// ```text
/// base_model.model.blocks.{i}.attention.{proj}.lora_A.default.weight
/// base_model.model.blocks.{i}.attention.{proj}.lora_B.default.weight
/// ```
///
/// # Validation
///
/// - Every `lora_A` / `lora_B` tensor in the checkpoint must have a matching
///   projection in the model (block index in range, known projection name).
/// - Loaded tensor shapes must match the target parameter shapes exactly.
/// - All required projections (wq, wk, wv, wk_text, wv_text, gate, wo)
///   must have both `lora_A` and `lora_B` present for every block.
pub fn apply_lora_adapter_to_model<B: AutodiffBackend>(
    mut model: LoraTextToLatentRfDiT<B>,
    adapter_path: &Path,
    device: &B::Device,
) -> Result<LoraTextToLatentRfDiT<B>> {
    use std::collections::HashSet;

    let bytes = std::fs::read(adapter_path)?;
    let st = SafeTensors::deserialize(&bytes)?;

    // Track loaded keys to detect missing tensors afterward.
    let mut loaded_keys: HashSet<String> = HashSet::new();

    for (raw_key, view) in st.iter() {
        // Strip PEFT prefix
        let key = raw_key.strip_prefix("base_model.model.").unwrap_or(raw_key);

        // Parse "blocks.{i}.attention.{proj}.lora_{A,B}.default.weight"
        let Some(rest) = key.strip_prefix("blocks.") else {
            continue;
        };
        let dot = rest.find('.').ok_or_else(|| {
            IrodoriError::Weight(format!("unexpected lora key format: {raw_key}"))
        })?;
        let block_idx: usize = rest[..dot]
            .parse()
            .map_err(|_| IrodoriError::Weight(format!("invalid block index in '{raw_key}'")))?;
        let rest = &rest[dot + 1..];

        let Some(rest) = rest.strip_prefix("attention.") else {
            continue;
        };
        let (proj, is_a) = if let Some(p) = rest.strip_suffix(".lora_A.default.weight") {
            (p, true)
        } else if let Some(p) = rest.strip_suffix(".lora_B.default.weight") {
            (p, false)
        } else {
            continue;
        };

        // Decode tensor to f32
        let raw = view.data();
        let floats: Vec<f32> = match view.dtype() {
            safetensors::Dtype::F32 => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            safetensors::Dtype::BF16 => raw
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect(),
            d => return Err(IrodoriError::Dtype(raw_key.to_string(), format!("{d:?}"))),
        };
        let shape: Vec<usize> = view.shape().to_vec();
        let data = TensorData::new(floats, shape.clone());
        let tensor: Tensor<B, 2> = Tensor::from_data(data, device);

        let block = model.blocks.get_mut(block_idx).ok_or_else(|| {
            IrodoriError::Weight(format!(
                "block index {block_idx} out of range in '{raw_key}'"
            ))
        })?;
        let attn = &mut block.attention;

        /// Validate tensor shape against target param, then assign.
        fn validate_and_set<B2: AutodiffBackend>(
            param: &mut Param<Tensor<B2, 2>>,
            tensor: Tensor<B2, 2>,
            shape: &[usize],
            raw_key: &str,
        ) -> Result<()> {
            let expected = param.val().dims();
            if shape.len() != 2 || shape[0] != expected[0] || shape[1] != expected[1] {
                return Err(IrodoriError::Weight(format!(
                    "shape mismatch for '{raw_key}': checkpoint has {shape:?}, model expects {expected:?}"
                )));
            }
            *param = Param::from_tensor(tensor);
            Ok(())
        }

        let ab = if is_a { "lora_A" } else { "lora_B" };

        macro_rules! set_proj {
            ($field:ident) => {{
                let param = if is_a {
                    &mut attn.$field.lora_a
                } else {
                    &mut attn.$field.lora_b
                };
                validate_and_set(param, tensor.clone(), &shape, raw_key)?;
            }};
        }
        macro_rules! set_proj_opt {
            ($field:ident) => {{
                if let Some(ref mut layer) = attn.$field {
                    let param = if is_a {
                        &mut layer.lora_a
                    } else {
                        &mut layer.lora_b
                    };
                    validate_and_set(param, tensor.clone(), &shape, raw_key)?;
                } else {
                    return Err(IrodoriError::Weight(format!(
                        "checkpoint has key '{raw_key}' but model has no {proj} projection"
                    )));
                }
            }};
        }

        match proj {
            "wq" => set_proj!(wq),
            "wk" => set_proj!(wk),
            "wv" => set_proj!(wv),
            "wk_text" => set_proj!(wk_text),
            "wv_text" => set_proj!(wv_text),
            "gate" => set_proj!(gate),
            "wo" => set_proj!(wo),
            "wk_speaker" => set_proj_opt!(wk_speaker),
            "wv_speaker" => set_proj_opt!(wv_speaker),
            "wk_caption" => set_proj_opt!(wk_caption),
            "wv_caption" => set_proj_opt!(wv_caption),
            other => {
                return Err(IrodoriError::Weight(format!(
                    "unknown lora projection '{other}' in key '{raw_key}'"
                )));
            }
        }

        loaded_keys.insert(format!("blocks.{block_idx}.attention.{proj}.{ab}"));
    }

    // Build expected key set from model structure and verify completeness.
    let num_blocks = model.blocks.len();
    let required_projs = ["wq", "wk", "wv", "wk_text", "wv_text", "gate", "wo"];
    let mut missing: Vec<String> = Vec::new();

    for i in 0..num_blocks {
        let attn = &model.blocks[i].attention;
        for proj in &required_projs {
            for ab in &["lora_A", "lora_B"] {
                let key = format!("blocks.{i}.attention.{proj}.{ab}");
                if !loaded_keys.contains(&key) {
                    missing.push(key);
                }
            }
        }

        // Optional projections: only require them if the model has them.
        let opt_projs: &[(&str, bool)] = &[
            ("wk_speaker", attn.wk_speaker.is_some()),
            ("wv_speaker", attn.wv_speaker.is_some()),
            ("wk_caption", attn.wk_caption.is_some()),
            ("wv_caption", attn.wv_caption.is_some()),
        ];
        for &(proj, present) in opt_projs {
            if present {
                for ab in &["lora_A", "lora_B"] {
                    let key = format!("blocks.{i}.attention.{proj}.{ab}");
                    if !loaded_keys.contains(&key) {
                        missing.push(key);
                    }
                }
            }
        }
    }

    if !missing.is_empty() {
        let preview: Vec<&str> = missing.iter().map(String::as_str).take(5).collect();
        return Err(IrodoriError::Weight(format!(
            "incomplete checkpoint: {} missing LoRA tensors (first 5: {preview:?})",
            missing.len()
        )));
    }

    Ok(model.freeze_base_weights())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use tempfile::TempDir;

    type TestBackend = burn::backend::Autodiff<NdArray>;

    fn tiny_lora_model() -> (LoraTextToLatentRfDiT<TestBackend>, crate::train::LoraConfig) {
        let cfg = crate::train::tiny_model_config();
        let lora_cfg = crate::train::LoraConfig {
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
    fn save_then_restore_roundtrip() {
        let dir = TempDir::new().unwrap();
        let (model, lora_cfg) = tiny_lora_model();

        // Save checkpoint
        crate::train::checkpoint::save_lora_adapter(&model, &lora_cfg, dir.path(), 1).unwrap();

        // Restore into a fresh model
        let (fresh_model, _) = tiny_lora_model();
        let adapter_path = dir.path().join("step-0000001/adapter_model.safetensors");
        let restored = apply_lora_adapter_to_model(fresh_model, &adapter_path, &Default::default())
            .expect("restore must succeed");

        // Verify at least one LoRA weight is non-trivially loaded
        // (not all zeros — the saved model had random init)
        let wq_a: Vec<f32> = restored.blocks[0]
            .attention
            .wq
            .lora_a
            .val()
            .into_data()
            .to_vec()
            .unwrap();
        let any_nonzero = wq_a.iter().any(|v| v.abs() > 1e-12);
        assert!(any_nonzero, "restored LoRA weight should not be all zeros");
    }

    #[test]
    fn restore_missing_file_returns_error() {
        let (model, _) = tiny_lora_model();
        let result = apply_lora_adapter_to_model(
            model,
            std::path::Path::new("/nonexistent/adapter.safetensors"),
            &Default::default(),
        );
        assert!(result.is_err());
    }

    /// Create a minimal safetensors file with only the specified keys.
    fn write_partial_safetensors(path: &Path, keys: &[(&str, Vec<usize>)]) {
        let mut entries: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
        for (key, shape) in keys {
            let numel: usize = shape.iter().product();
            let bytes: Vec<u8> = vec![0u8; numel * 4]; // f32 zeros
            entries.push((key.to_string(), bytes, shape.clone()));
        }
        serialize_entries_to_file(path, &entries);
    }

    fn serialize_entries_to_file(path: &Path, entries: &[(String, Vec<u8>, Vec<usize>)]) {
        let views: Vec<(&str, safetensors::tensor::TensorView<'_>)> = entries
            .iter()
            .map(|(k, b, s)| {
                (
                    k.as_str(),
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, s.clone(), b)
                        .unwrap(),
                )
            })
            .collect();
        let bytes = safetensors::tensor::serialize(views, None).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn restore_incomplete_checkpoint_fails() {
        let dir = TempDir::new().unwrap();
        let adapter_path = dir.path().join("adapter_model.safetensors");

        // Write only wq lora_A for block 0 — missing everything else
        write_partial_safetensors(
            &adapter_path,
            &[(
                "base_model.model.blocks.0.attention.wq.lora_A.default.weight",
                vec![2, 32],
            )],
        );

        let (model, _) = tiny_lora_model();
        let result = apply_lora_adapter_to_model(model, &adapter_path, &Default::default());
        assert!(result.is_err(), "should fail with incomplete checkpoint");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("incomplete checkpoint") || err_msg.contains("missing"),
            "error should mention missing keys, got: {err_msg}"
        );
    }

    #[test]
    fn restore_shape_mismatch_fails() {
        let dir = TempDir::new().unwrap();
        let (model, lora_cfg) = tiny_lora_model();

        // Save a valid checkpoint first
        crate::train::checkpoint::save_lora_adapter(&model, &lora_cfg, dir.path(), 1).unwrap();

        // Now corrupt it: overwrite with wrong shapes
        let adapter_path = dir.path().join("step-0000001/adapter_model.safetensors");
        let bytes = std::fs::read(&adapter_path).unwrap();
        let st = SafeTensors::deserialize(&bytes).unwrap();

        // Rebuild with one tensor having wrong shape
        let mut entries: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
        for (key, view) in st.iter() {
            let raw = view.data().to_vec();
            let mut shape = view.shape().to_vec();
            if key.contains("wq.lora_A") {
                // Corrupt shape: double the second dimension
                shape[1] *= 2;
                let extended = raw.repeat(2);
                entries.push((key.to_string(), extended, shape));
            } else {
                entries.push((key.to_string(), raw, shape));
            }
        }

        serialize_entries_to_file(&adapter_path, &entries);

        let (fresh_model, _) = tiny_lora_model();
        let result = apply_lora_adapter_to_model(fresh_model, &adapter_path, &Default::default());
        assert!(result.is_err(), "should fail with shape mismatch");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("shape mismatch"),
            "error should mention shape mismatch, got: {err_msg}"
        );
    }
}
