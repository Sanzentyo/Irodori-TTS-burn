//! LoRA adapter checkpoint saving in PEFT-compatible format.
//!
//! Key naming convention (matches the `base_model.model.` prefix that
//! `src/lora.rs` strips when loading adapters):
//! ```
//! base_model.model.blocks.{i}.attention.{proj}.lora_A.default.weight  // [r, in]
//! base_model.model.blocks.{i}.attention.{proj}.lora_B.default.weight  // [out, r]
//! ```
//!
//! Also writes `adapter_config.json`.

use std::path::Path;

use burn::tensor::backend::Backend;
use safetensors::{Dtype, tensor::TensorView};
use serde::Serialize;

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
fn extract_lora<B: Backend>(layer: &LoraLinear<B>) -> anyhow::Result<LoraBytes> {
    let a = layer.lora_a.val();
    let b = layer.lora_b.val();
    let a_shape = a.dims().to_vec();
    let b_shape = b.dims().to_vec();
    let a_bytes = f32_to_le_bytes(
        &a.into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?,
    );
    let b_bytes = f32_to_le_bytes(
        &b.into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?,
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
) -> anyhow::Result<()> {
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
                .map_err(|e| anyhow::anyhow!("TensorView: {e:?}"))
        })
        .collect::<anyhow::Result<_>>()?;

    // Phase 3: serialize.
    let out_path = dir.join("adapter_model.safetensors");
    safetensors::serialize_to_file(views, None, &out_path)
        .map_err(|e| anyhow::anyhow!("serialize safetensors: {e}"))?;

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
