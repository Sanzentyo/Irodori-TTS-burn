//! Stable identity shared by safetensors voice files and native codec exports.

use crate::{IrodoriError, Result};
use safetensors::SafeTensors;

#[cfg(feature = "synthesis")]
pub(crate) fn codec_identity(path: &std::path::Path) -> Result<String> {
    if path.extension().is_some_and(|ext| ext == "wgslpack") {
        #[cfg(feature = "packed-weights")]
        return super::packed::source_decoder_sha256(path);
        #[cfg(not(feature = "packed-weights"))]
        return Err(IrodoriError::Config(
            "native codecs require the packed-weights feature".into(),
        ));
    }
    let bytes = std::fs::read(path)?;
    decoder_digest(&SafeTensors::deserialize(&bytes)?)
}

/// Hash the original decoder names, shapes, dtypes and bytes, independent of
/// metadata ordering and optional encoder weights. Preserve the existing voice
/// identity contract when moving these bytes into a native archive.
pub(super) fn decoder_digest(tensors: &SafeTensors<'_>) -> Result<String> {
    let views = tensors
        .names()
        .into_iter()
        .map(|name| {
            let tensor = tensors.tensor(name)?;
            Ok((name, tensor))
        })
        .collect::<Result<Vec<_>>>()?;
    decoder_digest_parts(
        views
            .iter()
            .map(|(name, tensor)| (*name, tensor.dtype(), tensor.shape(), tensor.data())),
    )
}

#[cfg(feature = "synthesis")]
pub(super) fn decoder_store_digest(store: &crate::weights::TensorStore) -> Result<String> {
    decoder_digest_parts(store.raw_tensors())
}

fn decoder_digest_parts<'a>(
    tensors: impl Iterator<Item = (&'a str, safetensors::Dtype, &'a [usize], &'a [u8])>,
) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut tensors = tensors
        .filter(|(name, _, _, _)| {
            name.starts_with("decoder.model.") || name.starts_with("quantizer.out_proj.")
        })
        .collect::<Vec<_>>();
    tensors.sort_unstable_by_key(|(name, _, _, _)| *name);
    if tensors.is_empty() {
        return Err(IrodoriError::Audio(
            "codec contains no decoder tensors".into(),
        ));
    }
    let mut hash = Sha256::new();
    hash.update(b"irodori-dacvae-decoder-v1-48000-1920-32\0");
    for (name, dtype, shape, bytes) in tensors {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        let layout = serde_json::to_vec(&(dtype, shape))?;
        hash.update((layout.len() as u64).to_le_bytes());
        hash.update(layout);
        hash.update(bytes);
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(all(test, feature = "synthesis"))]
mod tests {
    use super::*;
    use safetensors::{Dtype, tensor::TensorView};

    #[test]
    fn loaded_store_preserves_identity_before_and_after_dtype_policy() {
        let bytes = [0.125_f32, -0.75]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let views = [
            "quantizer.out_proj.weight",
            "decoder.model.0.weight",
            "encoder.model.0.weight",
        ]
        .map(|name| (name, TensorView::new(Dtype::F32, vec![2], &bytes).unwrap()));
        let metadata =
            std::collections::HashMap::from([("config_json".to_owned(), "{}".to_owned())]);
        let archive = safetensors::tensor::serialize(views, Some(metadata)).unwrap();
        let expected = decoder_digest(&SafeTensors::deserialize(&archive).unwrap()).unwrap();
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), archive).unwrap();
        let store = crate::weights::TensorStore::load(file.path()).unwrap();
        assert_eq!(decoder_store_digest(&store).unwrap(), expected);
        let store = store.with_float_dtype(burn::tensor::DType::F16).unwrap();
        assert_eq!(decoder_store_digest(&store).unwrap(), expected);
    }
}
