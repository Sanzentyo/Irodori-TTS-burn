//! Checkpoint resume helpers for LoRA training.

use std::path::Path;

use burn::tensor::backend::AutodiffBackend;

use crate::{error::IrodoriError, train::LoraTextToLatentRfDiT};

/// Extract the step number from a checkpoint directory name `step-NNNNNNN`.
pub(super) fn parse_step_from_checkpoint_dir(
    path: &std::path::Path,
) -> crate::error::Result<usize> {
    let name = path.file_name().and_then(|n| n.to_str()).ok_or_else(|| {
        IrodoriError::Training(format!("invalid checkpoint path: {}", path.display()))
    })?;
    let digits = name.strip_prefix("step-").ok_or_else(|| {
        IrodoriError::Training(format!(
            "checkpoint dir must be named 'step-NNNNNNN', got: {name}"
        ))
    })?;
    digits
        .parse::<usize>()
        .map_err(|e| IrodoriError::Training(format!("parse step from '{name}': {e}")))
}

/// Load LoRA adapter weights from `dir/adapter_model.safetensors` into the
/// model's LoRA parameters.
///
/// Also loads `param_ids.json` (if present) and applies the original
/// [`burn::module::ParamId`]s to the restored parameters so that the
/// subsequent [`restore_optimizer_state`] call can match momentum terms
/// correctly.  When `param_ids.json` is absent (legacy checkpoint), falls
/// back to warm-restart behaviour where optimizer momentum is reset.
pub(super) fn restore_lora_weights<B: AutodiffBackend>(
    model: LoraTextToLatentRfDiT<B>,
    dir: &std::path::Path,
    device: &B::Device,
) -> crate::error::Result<LoraTextToLatentRfDiT<B>> {
    use crate::train::{
        checkpoint::load_lora_param_ids, lora_weights::apply_lora_adapter_to_model,
    };
    let adapter_path = dir.join("adapter_model.safetensors");
    if !adapter_path.exists() {
        return Err(IrodoriError::Training(format!(
            "resume checkpoint missing: {}",
            adapter_path.display()
        )));
    }
    let param_id_map = load_lora_param_ids(dir)?;
    if param_id_map.is_none() {
        tracing::warn!(
            "param_ids.json not found in checkpoint — optimizer state will be warm-restarted"
        );
    }
    apply_lora_adapter_to_model(model, &adapter_path, device, param_id_map.as_ref())
}

/// Restore optimizer state from `optimizer.mpk` in `checkpoint_dir`.
///
/// If the file does not exist (e.g., checkpoint was saved with an old version of
/// the trainer that did not persist optimizer state), logs a warning and returns
/// the original `optim` unchanged — this is a graceful degradation to a warm
/// restart.
///
/// **Important:** the model must be fully loaded (including LoRA weights) before
/// calling this function so that the `ParamId`s stored in the optimizer record
/// match those in the loaded model.
pub(crate) fn restore_optimizer_state<B, O>(
    optim: O,
    checkpoint_dir: &Path,
    device: &B::Device,
) -> crate::error::Result<O>
where
    B: AutodiffBackend,
    O: burn::optim::Optimizer<LoraTextToLatentRfDiT<B>, B>,
{
    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};

    // NamedMpkFileRecorder appends the ".mpk" extension automatically.
    let state_path = checkpoint_dir.join("optimizer.mpk");
    if !state_path.exists() {
        tracing::warn!(
            path = %state_path.display(),
            "optimizer state not found — falling back to warm restart (momentum terms reset)"
        );
        return Ok(optim);
    }

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    // Explicit type annotation avoids type inference ambiguity.
    let record: O::Record = recorder
        .load(checkpoint_dir.join("optimizer"), device)
        .map_err(|e| IrodoriError::Training(format!("load optimizer state: {e}")))?;
    Ok(optim.load_record(record))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_step_valid() {
        let p = std::path::Path::new("/tmp/checkpoints/step-0000042");
        assert_eq!(parse_step_from_checkpoint_dir(p).unwrap(), 42);
    }

    #[test]
    fn parse_step_zero_padded() {
        let p = std::path::Path::new("step-0000001");
        assert_eq!(parse_step_from_checkpoint_dir(p).unwrap(), 1);
    }

    #[test]
    fn parse_step_invalid_prefix() {
        let p = std::path::Path::new("epoch-5");
        assert!(parse_step_from_checkpoint_dir(p).is_err());
    }

    #[test]
    fn parse_step_non_numeric() {
        let p = std::path::Path::new("step-abc");
        assert!(parse_step_from_checkpoint_dir(p).is_err());
    }
}
