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

use crate::train::LoraTextToLatentRfDiT;

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
) -> anyhow::Result<LoraTextToLatentRfDiT<B>> {
    let bytes = std::fs::read(adapter_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", adapter_path.display()))?;
    let st = SafeTensors::deserialize(&bytes)
        .map_err(|e| anyhow::anyhow!("deserialize {}: {e}", adapter_path.display()))?;

    for (raw_key, view) in st.iter() {
        // Strip PEFT prefix
        let key = raw_key.strip_prefix("base_model.model.").unwrap_or(raw_key);

        // Parse "blocks.{i}.attention.{proj}.lora_{A,B}.default.weight"
        let Some(rest) = key.strip_prefix("blocks.") else {
            continue;
        };
        let dot = rest
            .find('.')
            .ok_or_else(|| anyhow::anyhow!("unexpected lora key format: {raw_key}"))?;
        let block_idx: usize = rest[..dot]
            .parse()
            .map_err(|e| anyhow::anyhow!("parse block index in '{raw_key}': {e}"))?;
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
            d => anyhow::bail!("unsupported lora dtype {d:?} in {raw_key}"),
        };
        let shape = view.shape().to_vec();
        let data = TensorData::new(floats, shape);
        let tensor: Tensor<B, 2> = Tensor::from_data(data, device);

        let block = model.blocks.get_mut(block_idx).ok_or_else(|| {
            anyhow::anyhow!("block index {block_idx} out of range in '{raw_key}'")
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
