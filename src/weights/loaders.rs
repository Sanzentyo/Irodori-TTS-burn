//! Public weight-loading API functions.

use std::path::Path;

use burn::{module::Module, tensor::backend::Backend};

use super::tensor_store::TensorStore;
use crate::{config::ModelConfig, error::Result, model::TextToLatentRfDiT};

/// Load a model and its configuration from a safetensors checkpoint.
///
/// The checkpoint must have been prepared by `scripts/convert_for_burn.py`,
/// which renames the `cond_module.{0,2,4}` keys to `cond_module.{linear0,linear1,linear2}`.
///
/// # Errors
/// Returns `IrodoriError::NoConfig` if `config_json` is absent from the checkpoint.
/// Returns `IrodoriError::Weight` if any required tensor is missing.
pub fn load_model<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<(TextToLatentRfDiT<B>, ModelConfig)> {
    let store = TensorStore::load(path)?;
    let cfg: ModelConfig = serde_json::from_str(&store.config_json)?;
    cfg.validate()?;
    let model = TextToLatentRfDiT::new(&cfg, device);
    let record = store.build_model_record::<B>(&cfg, device)?;
    let model = model.load_record(record);
    Ok((model, cfg))
}

/// Load model weights, optionally merging a LoRA adapter.
///
/// If `adapter_dir` is `Some`, the adapter is merged into the base weights
/// before constructing the model.  Supports PEFT-format adapters (keys with
/// the `base_model.model.` prefix are stripped automatically).
#[cfg(feature = "lora")]
pub fn load_model_with_lora<B: Backend>(
    path: &Path,
    adapter_dir: Option<&Path>,
    device: &B::Device,
) -> Result<(TextToLatentRfDiT<B>, ModelConfig)> {
    let store = TensorStore::load_with_lora(path, adapter_dir)?;
    let cfg: ModelConfig = serde_json::from_str(&store.config_json)?;
    cfg.validate()?;
    let model = TextToLatentRfDiT::new(&cfg, device);
    let record = store.build_model_record::<B>(&cfg, device)?;
    let model = model.load_record(record);
    Ok((model, cfg))
}

/// Load a LoRA training model from a base checkpoint.
///
/// Constructs a [`LoraTextToLatentRfDiT`] with frozen base weights (loaded
/// from `path`) and freshly initialised trainable LoRA params.
///
/// # Weight loading sequence
/// 1. Build fresh model + freeze base weights
/// 2. Build record directly from `TensorStore` (base from checkpoint, LoRA fresh)
/// 3. `load_record` — loads base weights while preserving frozen status
/// 4. Re-freeze (belt-and-suspenders, in case `load_record` altered grad flags)
#[cfg(feature = "train")]
pub fn load_lora_model<B: Backend>(
    path: &Path,
    r: usize,
    alpha: f32,
    device: &B::Device,
) -> Result<(crate::train::LoraTextToLatentRfDiT<B>, ModelConfig)> {
    let store = TensorStore::load(path)?;
    let cfg: ModelConfig = serde_json::from_str(&store.config_json)?;
    cfg.validate()?;
    let model = crate::train::LoraTextToLatentRfDiT::new(&cfg, r, alpha, device);
    let model = model.freeze_base_weights();
    let record = store.build_lora_model_record::<B>(&cfg, r, alpha, device)?;
    let model = model.load_record(record);
    let model = model.freeze_base_weights();
    Ok((model, cfg))
}
