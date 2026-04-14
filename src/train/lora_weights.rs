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
pub fn apply_lora_adapter_to_model<B: AutodiffBackend>(
    mut model: LoraTextToLatentRfDiT<B>,
    adapter_path: &Path,
    device: &B::Device,
) -> Result<LoraTextToLatentRfDiT<B>> {
    let bytes = std::fs::read(adapter_path)?;
    let st = SafeTensors::deserialize(&bytes)?;

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
        let shape = view.shape().to_vec();
        let data = TensorData::new(floats, shape);
        let tensor: Tensor<B, 2> = Tensor::from_data(data, device);

        let block = model.blocks.get_mut(block_idx).ok_or_else(|| {
            IrodoriError::Weight(format!(
                "block index {block_idx} out of range in '{raw_key}'"
            ))
        })?;
        let attn = &mut block.attention;

        macro_rules! set_proj {
            ($field:ident) => {
                if is_a {
                    attn.$field.lora_a = Param::from_tensor(tensor.clone());
                } else {
                    attn.$field.lora_b = Param::from_tensor(tensor.clone());
                }
            };
        }
        macro_rules! set_proj_opt {
            ($field:ident) => {
                if let Some(ref mut layer) = attn.$field {
                    if is_a {
                        layer.lora_a = Param::from_tensor(tensor.clone());
                    } else {
                        layer.lora_b = Param::from_tensor(tensor.clone());
                    }
                }
            };
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
                tracing::warn!("unknown lora projection '{other}' in key '{raw_key}' — skipping");
            }
        }
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
        let model = LoraTextToLatentRfDiT::<TestBackend>::new(
            &cfg,
            lora_cfg.r,
            lora_cfg.alpha,
            &device,
        );
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
}
