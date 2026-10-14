//! The codec application schema is separate from WGSLPack's wire version.

use crate::{IrodoriError, Result};
use serde::{Deserialize, Serialize};
use wgslpack::{Dtype, Index};

pub(super) const SCHEMA: &str = "irodori-dacvae-f32-v1";
pub(super) const BLOCKS: [(usize, usize, usize); 4] =
    [(1536, 768, 12), (768, 384, 10), (384, 192, 8), (192, 96, 2)];

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Metadata {
    pub schema: String,
    pub config_json: String,
    pub source_sha256: String,
    // A declared source identity, not authentication of this archive's payload.
    pub source_decoder_sha256: String,
}

impl Metadata {
    pub fn parse(index: Index<'_>) -> Result<Self> {
        let metadata: Self = serde_json::from_slice(index.metadata())?;
        if metadata.schema != SCHEMA {
            return Err(IrodoriError::Config(format!(
                "unsupported packed codec schema: {}",
                metadata.schema
            )));
        }
        for hash in [&metadata.source_sha256, &metadata.source_decoder_sha256] {
            if hash.len() != 64
                || !hash
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Err(IrodoriError::Config(
                    "invalid packed codec source SHA-256".into(),
                ));
            }
        }
        super::super::weights::validate_codec_metadata(&metadata.config_json)?;
        let mut expected = entries()
            .into_iter()
            .map(|entry| (entry.name.clone(), entry))
            .collect::<std::collections::BTreeMap<_, _>>();
        for tensor in index.tensors() {
            let entry = expected.remove(tensor.name()).ok_or_else(|| {
                IrodoriError::Weight(format!("unexpected packed codec tensor: {}", tensor.name()))
            })?;
            if tensor.dtype() != Dtype::F32 {
                return Err(IrodoriError::Dtype(
                    tensor.name().into(),
                    "packed codecs require FP32".into(),
                ));
            }
            if !tensor.shape().eq(entry.shape.iter().copied()) {
                return Err(IrodoriError::Shape(format!(
                    "packed codec {} must have shape {:?}",
                    tensor.name(),
                    entry.shape
                )));
            }
        }
        if let Some(missing) = expected.values().find(|entry| !entry.optional) {
            return Err(IrodoriError::Weight(format!(
                "missing packed codec tensor: {}",
                missing.name
            )));
        }
        Ok(metadata)
    }
}

#[cfg(feature = "packed-export")]
#[derive(Clone, Copy)]
pub(super) enum Layout {
    Original,
    ConvTranspose {
        input: usize,
        output: usize,
        stride: usize,
    },
    Pointwise {
        channels: usize,
    },
    ResidueVectors {
        channels: usize,
    },
}

pub(super) struct Entry {
    pub name: String,
    #[cfg(feature = "packed-export")]
    pub source: String,
    pub shape: Vec<u32>,
    #[cfg(feature = "packed-export")]
    pub layout: Layout,
    pub optional: bool,
}

fn original(name: String, shape: &[u32], optional: bool) -> Entry {
    Entry {
        #[cfg(feature = "packed-export")]
        source: name.clone(),
        name,
        shape: shape.into(),
        #[cfg(feature = "packed-export")]
        layout: Layout::Original,
        optional,
    }
}

fn convolution(entries: &mut Vec<Entry>, prefix: &str, shape: [u32; 3], bias: u32) {
    entries.push(original(format!("{prefix}.weight"), &shape, false));
    entries.push(original(format!("{prefix}.bias"), &[bias], true));
}

pub(super) fn prepared(block: usize, suffix: &str) -> String {
    format!("prepared.decoder.block{block}.{suffix}")
}

pub(super) fn entries() -> Vec<Entry> {
    let mut entries = Vec::new();
    convolution(&mut entries, "quantizer.out_proj", [1024, 32, 1], 1024);
    convolution(&mut entries, "decoder.model.0", [1536, 1024, 7], 1536);
    for (block, (input, output, stride)) in BLOCKS.into_iter().enumerate() {
        let prefix = format!("decoder.model.{}", block + 1);
        entries.push(original(
            format!("{prefix}.block.0.alpha"),
            &[1, input as u32, 1],
            false,
        ));
        let source = format!("{prefix}.block.1");
        convolution(
            &mut entries,
            &source,
            [input as u32, output as u32, (2 * stride) as u32],
            output as u32,
        );
        entries.push(Entry {
            name: prepared(block, "conv_transpose"),
            #[cfg(feature = "packed-export")]
            source: format!("{source}.weight"),
            shape: vec![stride as u32, output as u32, input as u32, 2],
            #[cfg(feature = "packed-export")]
            layout: Layout::ConvTranspose {
                input,
                output,
                stride,
            },
            optional: false,
        });
        for (residual, position) in [4, 5, 8].into_iter().enumerate() {
            let prefix = format!("{prefix}.block.{position}.block");
            for position in [0, 2] {
                entries.push(original(
                    format!("{prefix}.{position}.alpha"),
                    &[1, output as u32, 1],
                    false,
                ));
            }
            convolution(
                &mut entries,
                &format!("{prefix}.1"),
                [output as u32, output as u32, 7],
                output as u32,
            );
            convolution(
                &mut entries,
                &format!("{prefix}.3"),
                [output as u32, output as u32, 1],
                output as u32,
            );
            entries.push(Entry {
                name: prepared(block, &format!("res{residual}.pointwise")),
                #[cfg(feature = "packed-export")]
                source: format!("{prefix}.3.weight"),
                shape: vec![1, output as u32, output as u32],
                #[cfg(feature = "packed-export")]
                layout: Layout::Pointwise { channels: output },
                optional: false,
            });
            // The C768 production path consumes OIK directly.
            if output <= 384 {
                entries.push(Entry {
                    name: prepared(block, &format!("res{residual}.k7_vectors")),
                    #[cfg(feature = "packed-export")]
                    source: format!("{prefix}.1.weight"),
                    shape: vec![output as u32, 7, output as u32],
                    #[cfg(feature = "packed-export")]
                    layout: Layout::ResidueVectors { channels: output },
                    optional: false,
                });
            }
        }
    }
    entries.push(original(
        "decoder.wm_model.encoder_block.pre.0.alpha".into(),
        &[1, 96, 1],
        false,
    ));
    convolution(
        &mut entries,
        "decoder.wm_model.encoder_block.pre.1",
        [1, 96, 7],
        1,
    );
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgslpack::{Layout as WireLayout, TensorSpec};

    fn metadata() -> Metadata {
        Metadata {
            schema: SCHEMA.into(), source_sha256: "a".repeat(64), source_decoder_sha256: "b".repeat(64),
            config_json: serde_json::json!({"encoder_dim":64,"encoder_rates":[2,8,10,12],"latent_dim":1024,
                "decoder_dim":1536,"decoder_rates":[12,10,8,2],"n_codebooks":16,"codebook_size":1024,
                "codebook_dim":32,"quantizer_dropout":false,"sample_rate":48000}).to_string(),
        }
    }

    fn check(entries: &[Entry], metadata: &Metadata, dtype: Dtype) -> Result<Metadata> {
        let specs = entries
            .iter()
            .map(|entry| TensorSpec {
                name: &entry.name,
                shape: &entry.shape,
                dtype,
            })
            .collect::<Vec<_>>();
        // Only build the index: the large production payload is never allocated.
        let layout = WireLayout::new(&specs, &serde_json::to_vec(metadata)?)?;
        Metadata::parse(layout.index())
    }

    #[test]
    fn codec_pack_schema_validates_all_shapes_and_allows_absent_bias() {
        let mut entries = entries();
        check(&entries, &metadata(), Dtype::F32).unwrap();
        entries.retain(|entry| !entry.optional);
        check(&entries, &metadata(), Dtype::F32).unwrap();
        entries[0].shape[0] += 1;
        assert!(check(&entries, &metadata(), Dtype::F32).is_err());
    }

    #[test]
    fn codec_pack_schema_rejects_missing_extra_dtype_schema_and_identity() {
        let mut entries = entries();
        entries.retain(|entry| entry.name != prepared(0, "conv_transpose"));
        assert!(check(&entries, &metadata(), Dtype::F32).is_err());
        entries = super::entries();
        entries.push(original("surprise".into(), &[1], false));
        assert!(check(&entries, &metadata(), Dtype::F32).is_err());
        entries = super::entries();
        assert!(check(&entries, &metadata(), Dtype::F16).is_err());
        let mut metadata = metadata();
        metadata.schema.push('2');
        assert!(check(&entries, &metadata, Dtype::F32).is_err());
        metadata.schema = SCHEMA.into();
        metadata.source_decoder_sha256 = "g".repeat(64);
        assert!(check(&entries, &metadata, Dtype::F32).is_err());
    }
}
