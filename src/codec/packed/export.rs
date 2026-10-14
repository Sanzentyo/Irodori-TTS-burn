//! CPU-only, bit-preserving preparation of the existing WGSL layouts.

use super::schema::{self, Layout, Metadata};
use crate::{IrodoriError, Result};
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};
use std::{borrow::Cow, io::Write, path::Path};
use wgslpack::{Layout as WireLayout, TensorSpec};

/// Convert an FP32 converted DACVAE checkpoint without creating a GPU device.
/// Raw values are only permuted, never rounded. Extra encoder/watermark tensors
/// are omitted. Publish atomically, refusing to overwrite an existing file.
pub fn pack_decoder(source: &Path, destination: &Path) -> Result<()> {
    if destination.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "packed destination already exists",
        )
        .into());
    }
    if destination.extension().is_none_or(|ext| ext != "wgslpack") {
        return Err(IrodoriError::Config(
            "packed destination must end in .wgslpack".into(),
        ));
    }
    let bytes = std::fs::read(source)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let (_, header) = SafeTensors::read_metadata(&bytes)?;
    let attributes = header.metadata().as_ref().ok_or(IrodoriError::NoConfig)?;
    let config_json = attributes
        .get("config_json")
        .ok_or(IrodoriError::NoConfig)?
        .clone();
    super::super::weights::validate_codec_metadata(&config_json)?;
    super::super::weights::validate_decoder_profile(
        attributes.get("irodori_codec_profile").map(String::as_str),
    )?;
    let metadata = Metadata {
        schema: schema::SCHEMA.into(),
        config_json,
        source_sha256: format!("{:x}", Sha256::digest(&bytes)),
        source_decoder_sha256: super::super::identity::decoder_digest(&tensors)?,
    };
    let mut entries = schema::entries();
    // An optional bias may be absent; errors for present keys are never ignored.
    let source_names = tensors
        .names()
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    entries.retain(|entry| !entry.optional || source_names.contains(entry.source.as_str()));
    for entry in entries
        .iter()
        .filter(|entry| matches!(entry.layout, Layout::Original))
    {
        let tensor = tensors.tensor(&entry.source)?;
        if tensor.dtype() != Dtype::F32 {
            return Err(IrodoriError::Dtype(
                entry.source.clone(),
                "codec export requires source FP32".into(),
            ));
        }
        if !tensor
            .shape()
            .iter()
            .copied()
            .eq(entry.shape.iter().map(|&dim| dim as usize))
        {
            return Err(IrodoriError::Shape(format!(
                "codec export {} must have shape {:?}",
                entry.source, entry.shape
            )));
        }
    }
    let specs = entries
        .iter()
        .map(|entry| TensorSpec {
            name: &entry.name,
            shape: &entry.shape,
            dtype: wgslpack::Dtype::F32,
        })
        .collect::<Vec<_>>();
    let layout = WireLayout::new(&specs, &serde_json::to_vec(&metadata)?)?;
    Metadata::parse(layout.index())?;
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    file.write_all(layout.prefix())?;
    for (entry, descriptor) in entries.iter().zip(layout.index().tensors()) {
        let tensor = tensors.tensor(&entry.source)?;
        let data = match entry.layout {
            Layout::Original => Cow::Borrowed(tensor.data()),
            layout => Cow::Owned(permute(tensor.data(), layout)),
        };
        let range = descriptor.data_range();
        if data.len() as u64 != range.end - range.start {
            return Err(IrodoriError::Weight(format!(
                "packed codec length mismatch: {}",
                entry.name
            )));
        }
        file.write_all(&data)?;
        let padding = (wgslpack::ALIGNMENT as u64 - range.end % wgslpack::ALIGNMENT as u64)
            % wgslpack::ALIGNMENT as u64;
        file.write_all(&[0; wgslpack::ALIGNMENT][..padding as usize])?;
    }
    file.as_file().sync_all()?;
    file.persist_noclobber(destination)
        .map_err(|error| IrodoriError::Io(error.error))?;
    Ok(())
}

// All dimensions and byte counts have been validated against the fixed schema.
// Copy raw words so signed zero and even NaN payload bits remain unchanged.
fn permute(source: &[u8], layout: Layout) -> Vec<u8> {
    let mut packed = vec![0; source.len()];
    for (destination, word) in packed.chunks_exact_mut(4).enumerate() {
        let index = match layout {
            Layout::Original => destination,
            Layout::ConvTranspose {
                input,
                output,
                stride,
            } => {
                let tap = destination % 2;
                let input_channel = destination / 2 % input;
                let output_channel = destination / (2 * input) % output;
                let phase = destination / (2 * input * output);
                (input_channel * output + output_channel) * (2 * stride) + phase + tap * stride
            }
            Layout::Pointwise { channels } => {
                (destination % channels) * channels + destination / channels
            }
            Layout::ResidueVectors { channels } => {
                let reduction = destination / channels;
                let vector = (destination % channels) / 4;
                let lane = destination % 4;
                let output_channel = (vector / 8) * 32 + vector % 8 + lane * 8;
                (output_channel * channels + reduction / 7) * 7 + reduction % 7
            }
        };
        word.copy_from_slice(&source[index * 4..index * 4 + 4]);
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn codec_pack_pointwise_and_polyphase_preserve_exact_words() {
        let source = [0x80000000_u32, 0x7fc01234, 0xff800000, 1]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            words(&permute(&source, Layout::Pointwise { channels: 2 })),
            [0x80000000, 0xff800000, 0x7fc01234, 1]
        );
        for stride in [2, 8, 10, 12] {
            let source = (0..(2 * 3 * 2 * stride) as u32)
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>();
            let packed = words(&permute(
                &source,
                Layout::ConvTranspose {
                    input: 2,
                    output: 3,
                    stride,
                },
            ));
            // Independent scatter formulation matching the shader's source traversal.
            for input in 0..2 {
                for output in 0..3 {
                    for kernel in 0..2 * stride {
                        assert_eq!(
                            packed[(((kernel % stride) * 3 + output) * 2 + input) * 2
                                + kernel / stride],
                            ((input * 3 + output) * 2 * stride + kernel) as u32
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn codec_pack_residue_vectors_preserve_lane_order_and_are_bijective() {
        for channels in [96, 192, 384] {
            let count = channels * channels * 7;
            let source = (0..count as u32)
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>();
            let packed = words(&permute(&source, Layout::ResidueVectors { channels }));
            for input in 0..channels {
                for tap in 0..7 {
                    for output in 0..channels {
                        let vector = (output / 32) * 8 + output % 8;
                        let lane = (output % 32) / 8;
                        assert_eq!(
                            packed[(input * 7 + tap) * channels + vector * 4 + lane],
                            ((output * channels + input) * 7 + tap) as u32
                        );
                    }
                }
            }
            let mut sorted = packed;
            sorted.sort_unstable();
            assert!(sorted.iter().copied().eq(0..count as u32));
        }
    }

    #[test]
    fn codec_pack_rejects_invalid_inputs_without_publishing_or_overwriting() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("invalid.safetensors");
        let destination = dir.path().join("codec.wgslpack");
        std::fs::write(&source, b"invalid").unwrap();
        assert!(pack_decoder(&source, &destination).is_err());
        assert!(!destination.exists());
        std::fs::write(&destination, b"preserve").unwrap();
        assert!(pack_decoder(&source, &destination).is_err());
        assert_eq!(std::fs::read(destination).unwrap(), b"preserve");
    }

    #[test]
    #[ignore = "CPU-only full-file check; requires IRODORI_CODEC_TEST_CHECKPOINT"]
    fn codec_pack_fixture_preserves_payloads_and_voice_identity() {
        let source =
            std::env::var_os("IRODORI_CODEC_TEST_CHECKPOINT").expect("set source FP32 codec");
        let source = Path::new(&source);
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("codec.wgslpack");
        pack_decoder(source, &destination).unwrap();
        let source_bytes = std::fs::read(source).unwrap();
        let tensors = SafeTensors::deserialize(&source_bytes).unwrap();
        let bytes = std::fs::read(&destination).unwrap();
        let archive = wgslpack::Archive::parse(&bytes).unwrap();
        let metadata = Metadata::parse(archive.index()).unwrap();
        assert_eq!(
            metadata.source_sha256,
            format!("{:x}", Sha256::digest(&source_bytes))
        );
        assert_eq!(
            metadata.source_decoder_sha256,
            super::super::super::identity::decoder_digest(&tensors).unwrap()
        );
        for descriptor in archive
            .index()
            .tensors()
            .filter(|tensor| !tensor.name().starts_with("prepared."))
        {
            let range = descriptor.data_range();
            assert_eq!(
                &bytes[range.start as usize..range.end as usize],
                tensors.tensor(descriptor.name()).unwrap().data()
            );
        }
        #[cfg(feature = "synthesis")]
        assert_eq!(
            super::super::super::identity::codec_identity(source).unwrap(),
            super::super::super::identity::codec_identity(&destination).unwrap()
        );
    }
}
