//! Optional Irodori adapter for the independent, Sans-I/O `wgslpack` format.
//!
//! Conversion stores Burn parameter names, contiguous layout and precision once.
//! Loading performs no key remapping, transpose or numerical conversion. A file
//! loader reads one tensor at a time; the memory loader borrows the archive and
//! copies only the bytes required by Burn's owned upload API. Neither uses mmap.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use burn::{
    module::{Module, ModuleMapper, Param},
    tensor::{Bool, Bytes, DType, Device, Int, Tensor, TensorData},
};
use serde::{Deserialize, Serialize};
use wgslpack::{Archive, Descriptor, Dtype, Header, Index};

#[cfg(feature = "packed-export")]
mod export;
#[cfg(feature = "packed-export")]
pub use export::pack_checkpoint;

use crate::{
    WgpuFloatPrecision,
    config::ModelConfig,
    error::{IrodoriError, Result},
    model::TextToLatentRfDiT,
};

// Changing Burn parameter paths/layout requires a new application schema. The
// generic WGSLPack wire version is independent of this model-specific contract.
const MODEL_SCHEMA: &str = "irodori-burn-v1";
const MAX_PREFIX_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy)]
enum LoadScope {
    Full,
    ExactOnly,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelMetadata {
    schema: String,
    precision: WgpuFloatPrecision,
    source_sha256: String,
    config: ModelConfig,
    text_encoder_config_json: Option<String>,
}

impl ModelMetadata {
    fn parse(index: Index<'_>, expected_dtype: Option<DType>) -> Result<Self> {
        if cfg!(target_endian = "big") {
            return Err(IrodoriError::UnsupportedMode(
                "packed GPU uploads require a little-endian host".to_owned(),
            ));
        }
        let metadata: Self = serde_json::from_slice(index.metadata())?;
        if metadata.schema != MODEL_SCHEMA {
            return Err(IrodoriError::Config(format!(
                "unsupported packed model schema: {}",
                metadata.schema
            )));
        }
        if metadata.source_sha256.len() != 64
            || !metadata
                .source_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(IrodoriError::Config(
                "invalid packed source SHA-256".to_owned(),
            ));
        }
        if expected_dtype.is_some_and(|dtype| dtype != metadata.precision.tensor_dtype()) {
            return Err(IrodoriError::Config(format!(
                "packed model precision is {}; repack for the requested {expected_dtype:?}",
                metadata.precision.label()
            )));
        }
        metadata.config.validate()?;
        super::loaders::validate_pretrained_text_metadata(
            metadata.text_encoder_config_json.as_deref(),
            &metadata.config,
        )?;
        let expected = packed_dtype(metadata.precision);
        if let Some(tensor) = index.tensors().find(|tensor| tensor.dtype() != expected) {
            return Err(IrodoriError::Dtype(
                tensor.name().to_owned(),
                format!(
                    "packed tensor {:?} does not match declared {expected:?}",
                    tensor.dtype()
                ),
            ));
        }
        Ok(metadata)
    }
}

fn packed_dtype(precision: WgpuFloatPrecision) -> Dtype {
    match precision {
        WgpuFloatPrecision::Fp32 => Dtype::F32,
        WgpuFloatPrecision::Fp16 => Dtype::F16,
    }
}

fn read_index(path: &Path) -> Result<(File, Vec<u8>)> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut fixed = [0u8; wgslpack::HEADER_SIZE];
    file.read_exact(&mut fixed)?;
    let header = Header::parse(&fixed)?;
    if header.total_len() != file_len || header.prefix_len() > MAX_PREFIX_BYTES {
        return Err(IrodoriError::Weight(
            "packed file length or index limit exceeded".to_owned(),
        ));
    }
    let mut prefix = vec![0u8; header.prefix_len()];
    prefix[..fixed.len()].copy_from_slice(&fixed);
    file.read_exact(&mut prefix[fixed.len()..])?;
    Index::parse(&prefix, file_len)?;
    Ok((file, prefix))
}

/// Load a native model with memory bounded by the largest host tensor upload.
/// `expected_dtype`, when supplied, is checked rather than silently converted.
pub fn load_model(
    path: &Path,
    device: &Device,
    expected_dtype: Option<DType>,
) -> Result<(TextToLatentRfDiT, ModelConfig)> {
    load_file(path, device, expected_dtype, LoadScope::Full)
}

pub(super) fn load_model_exact_only(
    path: &Path,
    device: &Device,
) -> Result<(TextToLatentRfDiT, ModelConfig)> {
    load_file(path, device, None, LoadScope::ExactOnly)
}

fn load_file(
    path: &Path,
    device: &Device,
    expected_dtype: Option<DType>,
    scope: LoadScope,
) -> Result<(TextToLatentRfDiT, ModelConfig)> {
    let (mut file, prefix) = read_index(path)?;
    let index = Index::parse(&prefix, file.metadata()?.len())?;
    load_index(index, device, expected_dtype, scope, |descriptor| {
        let range = descriptor.data_range();
        let length = usize::try_from(range.end - range.start)
            .map_err(|_| IrodoriError::Weight("tensor exceeds host address space".to_owned()))?;
        let mut bytes = vec![0u8; length];
        file.seek(SeekFrom::Start(range.start))?;
        file.read_exact(&mut bytes)?;
        // These bytes go directly to a GPU upload, never a typed CPU slice.
        // from_bytes_vec would realign the allocation and copy the whole tensor.
        Ok(Bytes::from_elems(bytes))
    })
}

/// Load from caller-owned bytes. Parsing and indexing are borrowed; Burn's GPU
/// upload takes an owned copy of each payload. The archive can be dropped when
/// this function returns, without affecting the loaded model.
pub fn load_model_from_bytes(
    bytes: &[u8],
    device: &Device,
    expected_dtype: Option<DType>,
) -> Result<(TextToLatentRfDiT, ModelConfig)> {
    let archive = Archive::parse(bytes)?;
    load_index(
        archive.index(),
        device,
        expected_dtype,
        LoadScope::Full,
        |descriptor| {
            let range = descriptor.data_range();
            Ok(Bytes::from_elems(
                bytes[range.start as usize..range.end as usize].to_vec(),
            ))
        },
    )
}

/// Load from an owned allocation using shared, zero-copy payload views. This
/// avoids per-tensor host copies; the GPU upload retains the backing allocation
/// until it has consumed each view. Device memory and staging are still separate.
pub fn load_model_from_owned_bytes(
    bytes: Bytes,
    device: &Device,
    expected_dtype: Option<DType>,
) -> Result<(TextToLatentRfDiT, ModelConfig)> {
    let bytes = bytes.shared();
    let archive = Archive::parse(&bytes)?;
    load_index(
        archive.index(),
        device,
        expected_dtype,
        LoadScope::Full,
        |descriptor| {
            let range = descriptor.data_range();
            bytes
                .view(range.start as usize, range.end as usize)
                .map_err(|error| IrodoriError::Store(format!("packed payload view: {error:?}")))
        },
    )
}

fn load_index<'a>(
    index: Index<'a>,
    device: &Device,
    expected_dtype: Option<DType>,
    scope: LoadScope,
    read: impl FnMut(Descriptor<'a>) -> Result<Bytes>,
) -> Result<(TextToLatentRfDiT, ModelConfig)> {
    let mut metadata = ModelMetadata::parse(index, expected_dtype)?;
    if matches!(scope, LoadScope::ExactOnly) {
        metadata.config.use_duration_predictor = false;
    }
    let model = TextToLatentRfDiT::try_new(&metadata.config, device)?;
    let mut loader = NativeLoader {
        index,
        read,
        dtype: metadata.precision.tensor_dtype(),
        path: Vec::new(),
        loaded: 0,
        error: None,
    };
    let model = model.map(&mut loader);
    if let Some(error) = loader.error {
        return Err(error);
    }
    let expected_count = index
        .tensors()
        .filter(|tensor| {
            !matches!(scope, LoadScope::ExactOnly)
                || !tensor.name().starts_with("duration_predictor.")
        })
        .count();
    if loader.loaded != expected_count {
        return Err(IrodoriError::Weight(format!(
            "packed model contains unused tensors: loaded {}, stored {}",
            loader.loaded, expected_count
        )));
    }
    Ok((model, metadata.config))
}

struct NativeLoader<'a, F> {
    index: Index<'a>,
    read: F,
    dtype: DType,
    path: Vec<String>,
    loaded: usize,
    error: Option<IrodoriError>,
}

impl<'a, F: FnMut(Descriptor<'a>) -> Result<Bytes>> NativeLoader<'a, F> {
    fn tensor<const D: usize>(&mut self, param: &Param<Tensor<D>>) -> Result<Tensor<D>> {
        let path = self.path.join(".");
        let descriptor = self
            .index
            .get(&path)
            .ok_or_else(|| IrodoriError::Weight(path.clone()))?;
        let shape = descriptor
            .shape()
            .map(|dimension| dimension as usize)
            .collect::<Vec<_>>();
        if shape != param.lazy_shape().dims::<D>() {
            return Err(IrodoriError::Shape(format!(
                "packed {path}: expected {:?}, found {shape:?}",
                param.lazy_shape()
            )));
        }
        let bytes = (self.read)(descriptor)?;
        let data = TensorData {
            bytes,
            shape: shape.into(),
            dtype: self.dtype,
        };
        self.loaded += 1;
        Ok(Tensor::from_data(data, (&param.lazy_device(), self.dtype)))
    }
}

impl<'a, F: FnMut(Descriptor<'a>) -> Result<Bytes>> ModuleMapper for NativeLoader<'a, F> {
    fn enter_module(&mut self, name: &str, container_type: &str) {
        if !container_type.starts_with("Enum:") {
            self.path.push(name.to_owned());
        }
    }
    fn exit_module(&mut self, _name: &str, container_type: &str) {
        if !container_type.starts_with("Enum:") {
            self.path.pop();
        }
    }
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        if self.error.is_some() {
            return param;
        }
        match self.tensor(&param) {
            Ok(tensor) => {
                let id = param.id;
                param.transform_for_load(tensor, id)
            }
            Err(error) => {
                self.error = Some(error);
                param
            }
        }
    }
    fn map_int<const D: usize>(&mut self, param: Param<Tensor<D, Int>>) -> Param<Tensor<D, Int>> {
        self.error.get_or_insert_with(|| {
            IrodoriError::Weight("unexpected integer model parameter".to_owned())
        });
        param
    }
    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<D, Bool>>,
    ) -> Param<Tensor<D, Bool>> {
        self.error.get_or_insert_with(|| {
            IrodoriError::Weight("unexpected boolean model parameter".to_owned())
        });
        param
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use wgslpack::{Layout, TensorSpec};

    fn metadata() -> ModelMetadata {
        ModelMetadata {
            schema: MODEL_SCHEMA.to_owned(),
            precision: WgpuFloatPrecision::Fp32,
            source_sha256: "a".repeat(64),
            config: serde_json::from_str(&crate::weights::test_helpers::test_config_json())
                .unwrap(),
            text_encoder_config_json: None,
        }
    }

    fn layout(metadata: &ModelMetadata, dtype: Dtype) -> Layout {
        Layout::new(
            &[TensorSpec {
                name: "weight",
                dtype,
                shape: &[2, 3],
            }],
            &serde_json::to_vec(metadata).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn precision_and_application_schema_are_strict() {
        let mut metadata = metadata();
        let good = layout(&metadata, Dtype::F32);
        assert!(ModelMetadata::parse(good.index(), Some(DType::F32)).is_ok());
        assert!(ModelMetadata::parse(good.index(), Some(DType::F16)).is_err());
        assert!(ModelMetadata::parse(layout(&metadata, Dtype::F16).index(), None).is_err());
        metadata.schema.push_str("-unknown");
        assert!(ModelMetadata::parse(layout(&metadata, Dtype::F32).index(), None).is_err());
        metadata.schema = MODEL_SCHEMA.to_owned();
        metadata.source_sha256 = "invalid".to_owned();
        assert!(ModelMetadata::parse(layout(&metadata, Dtype::F32).index(), None).is_err());
    }

    #[test]
    fn file_adapter_reads_only_the_bounded_prefix_and_rejects_truncation() {
        let layout = layout(&metadata(), Dtype::F32);
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(layout.prefix()).unwrap();
        file.as_file().set_len(layout.total_len()).unwrap();
        let (mut opened, prefix) = read_index(file.path()).unwrap();
        assert_eq!(prefix, layout.prefix());
        assert_eq!(opened.stream_position().unwrap(), prefix.len() as u64);
        file.as_file().set_len(layout.total_len() - 1).unwrap();
        assert!(read_index(file.path()).is_err());
        let mut fixed = layout.prefix()[..wgslpack::HEADER_SIZE].to_vec();
        fixed[24..28].copy_from_slice(&(MAX_PREFIX_BYTES as u32).to_le_bytes());
        fixed[32..40].copy_from_slice(&((MAX_PREFIX_BYTES + 256) as u64).to_le_bytes());
        fixed[40..48].copy_from_slice(&((MAX_PREFIX_BYTES + 512) as u64).to_le_bytes());
        file.as_file_mut().rewind().unwrap();
        file.write_all(&fixed).unwrap();
        file.as_file()
            .set_len((MAX_PREFIX_BYTES + 512) as u64)
            .unwrap();
        assert!(read_index(file.path()).is_err());
    }
}
