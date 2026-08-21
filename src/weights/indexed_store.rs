//! Header-indexed, portable safetensors loading for the RF model.
//!
//! Burn Store's file-backed loader memory-maps the checkpoint efficiently, but
//! each lazy tensor materialization reparses the complete safetensors header
//! and looks up its name again. This loader captures validated offsets, dtype,
//! and shape once, then performs only a seek plus one exact read per tensor.
//! It deliberately uses safe `std::fs::File` I/O rather than an application
//! mmap so the same implementation works on Vulkan, Metal, and DX12 hosts.

use std::{
    cell::RefCell,
    io::{Read, Seek, SeekFrom},
    path::Path,
    rc::Rc,
};

use burn::{
    module::ParamId,
    tensor::{Bytes, DType, Shape, TensorData},
};
use burn_store::{
    FloatCastAdapter, KeyRemapper, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter,
    TensorSnapshot, TensorSnapshotError,
};
use safetensors::Dtype;

use super::tensor_store::read_checkpoint_header;
use crate::{
    config::ModelConfig,
    error::{IrodoriError, Result},
    model::TextToLatentRfDiT,
};

pub(super) fn load_checkpoint_into(
    model: &mut TextToLatentRfDiT,
    path: &Path,
    cfg: &ModelConfig,
    float_dtype: Option<DType>,
) -> Result<()> {
    let snapshots = indexed_snapshots(path)?;
    let remapper = checkpoint_key_remapper(cfg)?;
    let (snapshots, _) = remapper.remap(snapshots);
    let adapter: Box<dyn ModuleAdapter> = match float_dtype {
        Some(dtype) => Box::new(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(dtype))),
        None => Box::new(PyTorchToBurnAdapter),
    };
    let applied = model.apply(snapshots, None, Some(adapter), true);
    if !applied.is_success() {
        return Err(IrodoriError::Store(applied.to_string()));
    }
    Ok(())
}

fn indexed_snapshots(path: &Path) -> Result<Vec<TensorSnapshot>> {
    let (file, data_start, checkpoint) = read_checkpoint_header(path)?;
    let file = Rc::new(RefCell::new(file));
    let mut snapshots = Vec::with_capacity(checkpoint.offset_keys().len());

    for name in checkpoint.offset_keys() {
        let info = checkpoint
            .info(&name)
            .expect("offset key must have matching tensor metadata");
        let dtype = burn_dtype(info.dtype, &name)?;
        let shape = Shape::from(info.shape.clone());
        let (relative_start, relative_end) = info.data_offsets;
        let absolute_start = data_start
            .checked_add(u64::try_from(relative_start).map_err(|_| {
                IrodoriError::Weight(format!("tensor offset does not fit u64: {name}"))
            })?)
            .ok_or_else(|| IrodoriError::Weight(format!("tensor offset overflow: {name}")))?;
        let byte_len = relative_end
            .checked_sub(relative_start)
            .ok_or_else(|| IrodoriError::Weight(format!("tensor offsets are reversed: {name}")))?;
        let file = Rc::clone(&file);
        let tensor_name = name.clone();
        let tensor_shape = shape.clone();
        let data_fn = Rc::new(move || {
            let mut bytes = vec![0_u8; byte_len];
            let mut file = file.try_borrow_mut().map_err(|_| {
                TensorSnapshotError::IoError(format!(
                    "concurrent indexed read is unsupported for tensor {tensor_name}"
                ))
            })?;
            file.seek(SeekFrom::Start(absolute_start))
                .and_then(|_| file.read_exact(&mut bytes))
                .map_err(|error| {
                    TensorSnapshotError::IoError(format!(
                        "failed to read indexed tensor {tensor_name}: {error}"
                    ))
                })?;
            Ok(TensorData {
                bytes: Bytes::from_bytes_vec(bytes),
                shape: tensor_shape.clone(),
                dtype,
            })
        });
        let mut snapshot = TensorSnapshot::from_closure(
            data_fn,
            dtype,
            shape,
            name.split('.').map(str::to_owned).collect(),
            Vec::new(),
            ParamId::new(),
        );
        // Safetensors does not persist Burn parameter identity. Preserve the
        // freshly constructed module's ParamIds, matching Burn Store.
        snapshot.tensor_id = None;
        snapshots.push(snapshot);
    }
    Ok(snapshots)
}

fn burn_dtype(dtype: Dtype, name: &str) -> Result<DType> {
    match dtype {
        Dtype::F32 => Ok(DType::F32),
        Dtype::F16 => Ok(DType::F16),
        Dtype::BF16 => Ok(DType::BF16),
        other => Err(IrodoriError::Dtype(name.to_owned(), format!("{other:?}"))),
    }
}

fn checkpoint_key_remapper(cfg: &ModelConfig) -> Result<KeyRemapper> {
    let mut remapper = KeyRemapper::new();
    let patterns = if cfg.use_pretrained_text_encoder() {
        vec![
            (
                r"^pretrained_text_backbone\.",
                "condition_frontend.shared.pretrained_text_backbone.",
            ),
            (
                r"^text_encoder\.",
                "condition_frontend.shared.text_encoder.",
            ),
            (
                r"^caption_encoder\.",
                "condition_frontend.shared.caption_encoder.",
            ),
            (r"^text_norm\.", "condition_frontend.text_norm."),
            (r"^speaker_encoder\.", "condition_frontend.speaker.encoder."),
            (r"^speaker_norm\.", "condition_frontend.speaker.norm."),
            (r"^caption_norm\.", "condition_frontend.caption_norm."),
            (r"\.attn\.Wqkv\.", ".attn.wqkv."),
            (r"\.attn\.Wo\.", ".attn.wo."),
            (r"\.mlp\.Wi\.", ".mlp.wi."),
            (r"\.mlp\.Wo\.", ".mlp.wo."),
        ]
    } else {
        vec![
            (r"^text_encoder\.", "condition_frontend.text_encoder."),
            (r"^text_norm\.", "condition_frontend.text_norm."),
        ]
    };
    for (from, to) in patterns.into_iter().chain([
        (r"^cond_module\.0\.", "cond_module.linear0."),
        (r"^cond_module\.2\.", "cond_module.linear1."),
        (r"^cond_module\.4\.", "cond_module.linear2."),
    ]) {
        remapper = remapper.add_pattern(from, to).map_err(|error| {
            IrodoriError::Config(format!(
                "invalid checkpoint key remapping {from:?}: {error}"
            ))
        })?;
    }
    Ok(remapper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::test_helpers::{f32_bytes, test_config_json, write_safetensors};
    use safetensors::tensor::TensorView;

    #[test]
    fn indexed_snapshots_read_validated_byte_ranges() {
        let first = [1.0_f32, -2.0, 3.5, 4.0];
        let second = [8.0_f32, 9.0];
        let file = write_safetensors(
            &[
                ("a.weight", f32_bytes(&first), Dtype::F32, vec![2, 2]),
                ("z.bias", f32_bytes(&second), Dtype::F32, vec![2]),
            ],
            &test_config_json(),
        );
        let snapshots = indexed_snapshots(file.path()).unwrap();
        assert_eq!(snapshots.len(), 2);
        let mut values = snapshots
            .iter()
            .map(|snapshot| {
                (
                    snapshot.full_path(),
                    snapshot.to_data().unwrap().to_vec::<f32>().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        values.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(values[0], ("a.weight".to_owned(), first.to_vec()));
        assert_eq!(values[1], ("z.bias".to_owned(), second.to_vec()));
        assert!(
            snapshots
                .iter()
                .all(|snapshot| snapshot.tensor_id.is_none())
        );
    }

    #[test]
    fn indexed_snapshots_reject_unsupported_checkpoint_dtype() {
        let bytes = 1_i64.to_le_bytes();
        let view = TensorView::new(Dtype::I64, vec![1], &bytes).unwrap();
        let serialized = safetensors::tensor::serialize(
            [("unsupported", view)],
            Some(std::collections::HashMap::from([(
                "config_json".to_owned(),
                test_config_json(),
            )])),
        )
        .unwrap();
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), serialized).unwrap();
        let error = indexed_snapshots(file.path()).unwrap_err();
        assert!(error.to_string().contains("unsupported"));
    }
}
