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
    use sha2::{Digest, Sha256};
    let mut names = tensors
        .names()
        .into_iter()
        .filter(|name| {
            name.starts_with("decoder.model.") || name.starts_with("quantizer.out_proj.")
        })
        .collect::<Vec<_>>();
    names.sort_unstable();
    if names.is_empty() {
        return Err(IrodoriError::Audio(
            "codec contains no decoder tensors".into(),
        ));
    }
    let mut hash = Sha256::new();
    hash.update(b"irodori-dacvae-decoder-v1-48000-1920-32\0");
    for name in names {
        let tensor = tensors.tensor(name)?;
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        let layout = serde_json::to_vec(&(tensor.dtype(), tensor.shape()))?;
        hash.update((layout.len() as u64).to_le_bytes());
        hash.update(layout);
        hash.update(tensor.data());
    }
    Ok(format!("{:x}", hash.finalize()))
}
