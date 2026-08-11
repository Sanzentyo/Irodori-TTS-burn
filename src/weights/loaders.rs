//! Public weight-loading API functions.

use std::path::Path;

use burn::tensor::backend::Backend;

use super::tensor_store::TensorStore;
use crate::{config::ModelConfig, error::Result, model::TextToLatentRfDiT};

fn validate_pretrained_text_metadata(store: &TensorStore, cfg: &ModelConfig) -> Result<()> {
    if !cfg.use_pretrained_text_encoder() {
        return Ok(());
    }
    let json = store.metadata("text_encoder_config_json").ok_or_else(|| {
        crate::error::IrodoriError::Config(
            "pretrained text checkpoint is missing text_encoder_config_json metadata".to_owned(),
        )
    })?;
    crate::model::modern_bert::ModernBertConfig::validate_v4_metadata(json)
}

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
    validate_pretrained_text_metadata(&store, &cfg)?;
    let record = store.build_model_record::<B>(&cfg, device)?;
    let model = TextToLatentRfDiT::from_record(&cfg, record, device)?;
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
    validate_pretrained_text_metadata(&store, &cfg)?;
    let record = store.build_model_record::<B>(&cfg, device)?;
    let model = TextToLatentRfDiT::from_record(&cfg, record, device)?;
    Ok((model, cfg))
}
