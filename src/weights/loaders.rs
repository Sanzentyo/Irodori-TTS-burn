//! Public weight-loading API functions.

use std::path::Path;

use super::tensor_store::CheckpointMetadata;
#[cfg(feature = "lora")]
use super::tensor_store::TensorStore;
use crate::{config::ModelConfig, error::Result, model::TextToLatentRfDiT};
use burn::tensor::Device;
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

fn validate_pretrained_text_metadata(
    text_encoder_config_json: Option<&str>,
    cfg: &ModelConfig,
) -> Result<()> {
    if !cfg.use_pretrained_text_encoder() {
        return Ok(());
    }
    let json = text_encoder_config_json.ok_or_else(|| {
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
pub fn load_model(path: &Path, device: &Device) -> Result<(TextToLatentRfDiT, ModelConfig)> {
    let metadata = CheckpointMetadata::load(path)?;
    let cfg: ModelConfig = serde_json::from_str(&metadata.config_json)?;
    cfg.validate()?;
    validate_pretrained_text_metadata(metadata.metadata("text_encoder_config_json"), &cfg)?;
    let mut model = TextToLatentRfDiT::try_new(&cfg, device)?;
    load_checkpoint_into(&mut model, path, &cfg)?;
    Ok((model, cfg))
}

/// Load the RF model for sessions whose output geometry is always explicit.
///
/// The checkpoint configuration is preserved except for
/// `use_duration_predictor`, which is disabled before constructing either the
/// record or the model. Consequently, duration tensors are never copied to the
/// backend and the returned model cannot perform learned duration prediction.
/// Callers that accept predicted-duration requests must use [`load_model`].
pub fn load_model_exact_only(
    path: &Path,
    device: &Device,
) -> Result<(TextToLatentRfDiT, ModelConfig)> {
    let metadata = CheckpointMetadata::load(path)?;
    let mut cfg: ModelConfig = serde_json::from_str(&metadata.config_json)?;
    cfg.validate()?;
    validate_pretrained_text_metadata(metadata.metadata("text_encoder_config_json"), &cfg)?;
    cfg.use_duration_predictor = false;
    let mut model = TextToLatentRfDiT::try_new(&cfg, device)?;
    load_checkpoint_into(&mut model, path, &cfg)?;
    debug_assert!(!model.has_duration_predictor());
    Ok((model, cfg))
}

/// Load model weights, optionally merging a LoRA adapter.
///
/// If `adapter_dir` is `Some`, the adapter is merged into the base weights
/// before constructing the model.  Supports PEFT-format adapters (keys with
/// the `base_model.model.` prefix are stripped automatically).
#[cfg(feature = "lora")]
pub fn load_model_with_lora(
    path: &Path,
    adapter_dir: Option<&Path>,
    device: &Device,
) -> Result<(TextToLatentRfDiT, ModelConfig)> {
    let store = TensorStore::load_with_lora(path, adapter_dir)?;
    let cfg: ModelConfig = serde_json::from_str(&store.config_json)?;
    cfg.validate()?;
    validate_pretrained_text_metadata(store.metadata("text_encoder_config_json"), &cfg)?;
    let mut model = TextToLatentRfDiT::try_new(&cfg, device)?;
    let merged = store.to_safetensors_bytes()?;
    load_checkpoint_bytes_into(&mut model, merged, &cfg)?;
    Ok((model, cfg))
}

fn load_checkpoint_into(
    model: &mut TextToLatentRfDiT,
    path: &Path,
    cfg: &ModelConfig,
) -> Result<()> {
    let checkpoint = SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .skip_enum_variants(true);
    load_configured_checkpoint(model, checkpoint, cfg)
}

fn load_configured_checkpoint(
    model: &mut TextToLatentRfDiT,
    mut checkpoint: SafetensorsStore,
    cfg: &ModelConfig,
) -> Result<()> {
    if cfg.use_pretrained_text_encoder() {
        checkpoint = checkpoint
            .with_key_remapping(
                r"^pretrained_text_backbone\.",
                "condition_frontend.shared.pretrained_text_backbone.",
            )
            .with_key_remapping(
                r"^text_encoder\.",
                "condition_frontend.shared.text_encoder.",
            )
            .with_key_remapping(
                r"^caption_encoder\.",
                "condition_frontend.shared.caption_encoder.",
            )
            .with_key_remapping(r"^text_norm\.", "condition_frontend.text_norm.")
            .with_key_remapping(r"^speaker_encoder\.", "condition_frontend.speaker.encoder.")
            .with_key_remapping(r"^speaker_norm\.", "condition_frontend.speaker.norm.")
            .with_key_remapping(r"^caption_norm\.", "condition_frontend.caption_norm.")
            .with_key_remapping(r"\.attn\.Wqkv\.", ".attn.wqkv.")
            .with_key_remapping(r"\.attn\.Wo\.", ".attn.wo.")
            .with_key_remapping(r"\.mlp\.Wi\.", ".mlp.wi.")
            .with_key_remapping(r"\.mlp\.Wo\.", ".mlp.wo.");
    } else {
        checkpoint = checkpoint
            .with_key_remapping(r"^text_encoder\.", "condition_frontend.text_encoder.")
            .with_key_remapping(r"^text_norm\.", "condition_frontend.text_norm.");
    }

    checkpoint = checkpoint
        .with_key_remapping(r"^cond_module\.0\.", "cond_module.linear0.")
        .with_key_remapping(r"^cond_module\.2\.", "cond_module.linear1.")
        .with_key_remapping(r"^cond_module\.4\.", "cond_module.linear2.");

    let applied = model
        .load_from(&mut checkpoint)
        .map_err(|error| crate::error::IrodoriError::Store(error.to_string()))?;
    if !applied.is_success() {
        return Err(crate::error::IrodoriError::Store(applied.to_string()));
    }
    Ok(())
}

#[cfg(feature = "lora")]
fn load_checkpoint_bytes_into(
    model: &mut TextToLatentRfDiT,
    bytes: Vec<u8>,
    cfg: &ModelConfig,
) -> Result<()> {
    let checkpoint = SafetensorsStore::from_bytes(Some(bytes))
        .with_from_adapter(PyTorchToBurnAdapter)
        .skip_enum_variants(true);
    load_configured_checkpoint(model, checkpoint, cfg)
}
