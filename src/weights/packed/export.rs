//! Optional atomic model converter; only this adapter directly uses tempfile.

use super::{MODEL_SCHEMA, ModelMetadata, packed_dtype};
use crate::{
    WgpuFloatPrecision,
    error::{IrodoriError, Result},
};
use burn::tensor::Device;
use burn_store::ModuleSnapshot;
use std::{io::Write, path::Path};
use wgslpack::{Layout, TensorSpec};

/// Convert a safetensors model once, preserving exactly the weights produced by
/// the existing loader at the requested precision. The destination must not
/// exist; a same-directory temporary file is published only after a full write.
pub fn pack_checkpoint(
    source: &Path,
    destination: &Path,
    device: &Device,
    precision: WgpuFloatPrecision,
) -> Result<()> {
    if destination.exists() {
        return Err(IrodoriError::Io(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "packed destination already exists",
        )));
    }
    if destination
        .extension()
        .is_none_or(|extension| extension != "wgslpack")
    {
        return Err(IrodoriError::Config(
            "packed destination must end in .wgslpack".to_owned(),
        ));
    }
    let source_metadata = super::super::tensor_store::CheckpointMetadata::load(source)?;
    let source_sha256 = crate::sha256_file(source)?;
    let (model, config) = super::super::loaders::load_model_with_float_dtype(
        source,
        device,
        precision.tensor_dtype(),
    )?;
    let metadata = ModelMetadata {
        schema: MODEL_SCHEMA.to_owned(),
        precision,
        source_sha256,
        config,
        text_encoder_config_json: source_metadata
            .metadata("text_encoder_config_json")
            .map(str::to_owned),
    };
    let mut snapshots = model.collect(None, None, true);
    snapshots.sort_by_key(|snapshot| snapshot.full_path());
    let names = snapshots
        .iter()
        .map(|snapshot| snapshot.full_path())
        .collect::<Vec<_>>();
    let shapes = snapshots
        .iter()
        .map(|snapshot| {
            snapshot
                .shape
                .iter()
                .map(|&dimension| {
                    u32::try_from(dimension)
                        .map_err(|_| IrodoriError::Shape("packed dimension exceeds u32".to_owned()))
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    for snapshot in &snapshots {
        if snapshot.dtype != precision.tensor_dtype() {
            return Err(IrodoriError::Dtype(
                snapshot.full_path(),
                format!("{:?}", snapshot.dtype),
            ));
        }
    }
    let specs = names
        .iter()
        .zip(&shapes)
        .map(|(name, shape)| TensorSpec {
            name,
            shape,
            dtype: packed_dtype(precision),
        })
        .collect::<Vec<_>>();
    let layout = Layout::new(&specs, &serde_json::to_vec(&metadata)?)?;
    let parent = destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    file.write_all(layout.prefix())?;
    for (snapshot, descriptor) in snapshots.iter().zip(layout.index().tensors()) {
        let data = snapshot
            .to_data()
            .map_err(|error| IrodoriError::Store(error.to_string()))?;
        let range = descriptor.data_range();
        if data.bytes.len() as u64 != range.end - range.start {
            return Err(IrodoriError::Weight(format!(
                "export byte length mismatch: {}",
                descriptor.name()
            )));
        }
        file.write_all(&data.bytes)?;
        let padding = (wgslpack::ALIGNMENT as u64 - range.end % wgslpack::ALIGNMENT as u64)
            % wgslpack::ALIGNMENT as u64;
        file.write_all(&[0u8; wgslpack::ALIGNMENT][..padding as usize])?;
    }
    file.as_file().sync_all()?;
    file.persist_noclobber(destination)
        .map_err(|error| IrodoriError::Io(error.error))?;
    Ok(())
}
