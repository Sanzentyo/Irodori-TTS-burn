//! Checkpoint resume helpers for LoRA training.

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
/// model's LoRA parameters (warm restart — base weights and optimizer state
/// are NOT restored).
pub(super) fn restore_lora_weights<B: AutodiffBackend>(
    model: LoraTextToLatentRfDiT<B>,
    dir: &std::path::Path,
    device: &B::Device,
) -> crate::error::Result<LoraTextToLatentRfDiT<B>> {
    use crate::train::lora_weights::apply_lora_adapter_to_model;
    let adapter_path = dir.join("adapter_model.safetensors");
    if !adapter_path.exists() {
        return Err(IrodoriError::Training(format!(
            "resume checkpoint missing: {}",
            adapter_path.display()
        )));
    }
    apply_lora_adapter_to_model(model, &adapter_path, device)
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
