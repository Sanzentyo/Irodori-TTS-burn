//! Optional FP32 decoder archive with precomputed WGSL weight layouts.
//!
//! This adapter uses the independent Sans-I/O `wgslpack` crate. File loading
//! stages bounded batches; borrowed/owned memory entry points have no file I/O.
//! These archives contain no encoder and cannot create reference voices.

use super::{DacVaeDecoder, weights::CodecWeightSource};
use crate::{
    IrodoriError, Result,
    weights::packed::{PackedLoadOptions, read_index, upload},
};
use burn::{
    backend::wgpu::CubeTensor,
    tensor::{Bytes, DType, Device, Tensor},
};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    io::{Read, Seek, SeekFrom},
    path::Path,
};
use wgslpack::{Archive, Descriptor, Index};

#[cfg(feature = "packed-export")]
mod export;
mod schema;
#[cfg(feature = "packed-export")]
pub use export::pack_decoder;

/// Read the converter's declared original decoder identity for existing voices.
/// Validates the index/schema, but does not authenticate tensor payloads.
pub fn source_decoder_sha256(path: &Path) -> Result<String> {
    let (file, prefix) = read_index(path)?;
    Ok(
        schema::Metadata::parse(Index::parse(&prefix, file.metadata()?.len())?)?
            .source_decoder_sha256,
    )
}

/// Load FP32 weights and their prepared layouts. No dtype conversion is implicit.
pub fn load_decoder(path: &Path, device: &Device, dtype: DType) -> Result<DacVaeDecoder> {
    let (mut file, prefix) = read_index(path)?;
    let index = Index::parse(&prefix, file.metadata()?.len())?;
    load_index(index, device, dtype, |descriptor| {
        let range = descriptor.data_range();
        let length = usize::try_from(range.end - range.start)
            .map_err(|_| IrodoriError::Weight("codec tensor exceeds host address space".into()))?;
        let mut bytes = vec![0; length];
        file.seek(SeekFrom::Start(range.start))?;
        file.read_exact(&mut bytes)?;
        Ok(Bytes::from_elems(bytes))
    })
}

/// Borrow the index and copy each payload into Burn's owned GPU upload buffers.
/// The caller may release the archive after this function returns.
pub fn load_decoder_from_bytes(
    bytes: &[u8],
    device: &Device,
    dtype: DType,
) -> Result<DacVaeDecoder> {
    let archive = Archive::parse(bytes)?;
    load_index(archive.index(), device, dtype, |descriptor| {
        let range = descriptor.data_range();
        Ok(Bytes::from_elems(
            bytes[range.start as usize..range.end as usize].to_vec(),
        ))
    })
}

/// Upload shared views of an owned archive, avoiding per-tensor host copies.
/// GPU memory and driver staging are still separate allocations.
pub fn load_decoder_from_owned_bytes(
    bytes: Bytes,
    device: &Device,
    dtype: DType,
) -> Result<DacVaeDecoder> {
    let bytes = bytes.shared();
    let archive = Archive::parse(&bytes)?;
    load_index(archive.index(), device, dtype, |descriptor| {
        let range = descriptor.data_range();
        bytes
            .view(range.start as usize, range.end as usize)
            .map_err(|error| IrodoriError::Store(format!("packed codec payload view: {error:?}")))
    })
}

fn load_index<'a>(
    index: Index<'a>,
    device: &Device,
    dtype: DType,
    read: impl FnMut(Descriptor<'a>) -> Result<Bytes>,
) -> Result<DacVaeDecoder> {
    if cfg!(target_endian = "big") || dtype != DType::F32 {
        return Err(IrodoriError::Config(
            "packed codecs require an FP32 little-endian upload".into(),
        ));
    }
    // Reject all schema/shape mismatches before any device allocation.
    let metadata = schema::Metadata::parse(index)?;
    let tune_codec = crate::route_autotune::active_route_table().tunes_codec();
    let unused = (1..4)
        .map(|block| schema::prepared(block, "conv_transpose"))
        .collect::<Vec<_>>();
    let tensors = upload::tensors(
        index
            .tensors()
            .filter(|d| tune_codec || !unused.iter().any(|name| name == d.name())),
        device,
        dtype,
        PackedLoadOptions::default().upload_batch_bytes,
        read,
    )?;
    let store = PackedSource {
        tensors: RefCell::new(tensors),
        metadata,
    };
    let mut decoder = super::weights::build_decoder_only(&store, device)?;
    for (block, module) in [
        &mut decoder.decoder.block0,
        &mut decoder.decoder.block1,
        &mut decoder.decoder.block2,
        &mut decoder.decoder.block3,
    ]
    .into_iter()
    .enumerate()
    {
        if block == 0 || tune_codec {
            module.packed_conv_t_weight =
                Some(store.tensor(&schema::prepared(block, "conv_transpose"), device)?);
        }
        for (residual, unit) in [&mut module.res0, &mut module.res1, &mut module.res2]
            .into_iter()
            .enumerate()
        {
            unit.packed_conv_1x1_weight = Some(store.tensor(
                &schema::prepared(block, &format!("res{residual}.pointwise")),
                device,
            )?);
            if block > 0 {
                unit.packed_conv_dil_weight_vectors = Some(store.tensor(
                    &schema::prepared(block, &format!("res{residual}.k7_vectors")),
                    device,
                )?);
            }
        }
    }
    if !store.tensors.borrow().is_empty() {
        return Err(IrodoriError::Weight("unused packed codec tensors".into()));
    }
    Ok(decoder)
}

struct PackedSource {
    tensors: RefCell<BTreeMap<String, CubeTensor>>,
    metadata: schema::Metadata,
}

impl CodecWeightSource for PackedSource {
    fn tensor<const D: usize>(&self, name: &str, _device: &Device) -> Result<Tensor<D>> {
        // This private source is created and consumed with one device in load_index.
        let tensor = self
            .tensors
            .borrow_mut()
            .remove(name)
            .ok_or_else(|| IrodoriError::Weight(name.into()))?;
        if tensor.meta.shape().as_slice().len() != D {
            return Err(IrodoriError::Shape(format!(
                "packed codec {name}: expected rank {D}"
            )));
        }
        Ok(Tensor::from_primitive::<crate::WgpuRaw>(tensor))
    }
    fn has(&self, name: &str) -> bool {
        self.tensors.borrow().contains_key(name)
    }
    fn config_json(&self) -> &str {
        &self.metadata.config_json
    }
    fn metadata(&self, name: &str) -> Option<&str> {
        (name == "irodori_codec_profile").then_some("decoder-only")
    }
}
