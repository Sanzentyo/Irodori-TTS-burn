//! Core `TensorStore` — in-memory store of safetensors checkpoint tensors.

use std::collections::HashMap;
use std::path::Path;

use burn::{
    module::{Param, ParamId},
    tensor::{Tensor, backend::Backend},
};
use safetensors::SafeTensors;

use super::tensor_entry::TensorEntry;
use crate::error::{IrodoriError, Result};

/// In-memory store of all tensors from a safetensors checkpoint.
///
/// Raw bytes are kept in the checkpoint's native dtype (f32, bf16, or f16).
pub struct TensorStore {
    tensors: HashMap<String, TensorEntry>,
    /// The `config_json` metadata embedded in the checkpoint.
    pub config_json: String,
}

impl TensorStore {
    /// Load a safetensors checkpoint from `path`.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;

        let config_json = {
            let (_offset, metadata) = SafeTensors::read_metadata(&bytes)?;
            let meta = metadata.metadata().as_ref().ok_or(IrodoriError::NoConfig)?;
            meta.get("config_json")
                .ok_or(IrodoriError::NoConfig)?
                .clone()
        };

        let st = SafeTensors::deserialize(&bytes)?;
        let mut tensors = HashMap::new();
        for (name, view) in st.tensors() {
            let entry = TensorEntry {
                bytes: view.data().to_vec(),
                dtype: view.dtype(),
                shape: view.shape().to_vec(),
            };
            entry.validate_byte_len(&name)?;
            tensors.insert(name, entry);
        }

        Ok(Self {
            tensors,
            config_json,
        })
    }

    /// Load a safetensors checkpoint and optionally merge a LoRA adapter.
    ///
    /// If `adapter_dir` is `Some`, the LoRA weights from that directory are
    /// merged into the base weights before the model record is built.  Keys
    /// with the PEFT `base_model.model.` prefix are automatically stripped so
    /// that the resulting key map matches the plain (non-PEFT) safetensors layout.
    #[cfg(feature = "lora")]
    pub fn load_with_lora(path: &Path, adapter_dir: Option<&Path>) -> Result<Self> {
        let mut store = Self::load(path)?;

        // Strip PEFT "base_model.model." prefix and discard raw lora_ sub-keys.
        let uses_peft_prefix =
            crate::lora::has_peft_prefix(store.tensors.keys().map(String::as_str));
        if uses_peft_prefix {
            store.tensors = store
                .tensors
                .into_iter()
                .filter_map(|(k, v)| {
                    // LoRA sub-keys are handled separately via merge_lora.
                    if k.contains(".lora_") {
                        return None;
                    }
                    let new_key = crate::lora::strip_peft_prefix(&k).to_owned();
                    Some((new_key, v))
                })
                .collect();
        }

        if let Some(dir) = adapter_dir {
            let n = store.apply_lora(dir)?;
            eprintln!("[lora] merged {n} adapter layers from {}", dir.display());
        }

        Ok(store)
    }

    /// Merge a LoRA adapter from `adapter_dir` into this store in-place.
    ///
    /// Decodes only the base weights that will be modified, applies the LoRA delta,
    /// and re-encodes to the entry's original dtype.  Returns the number of
    /// layers merged.
    #[cfg(feature = "lora")]
    pub fn apply_lora(&mut self, adapter_dir: &Path) -> Result<usize> {
        use super::tensor_entry::encode_f32_to_dtype;

        // Pre-scan the adapter to find which base keys will be merged.
        let merged_keys = crate::lora::pre_scan_lora_keys(adapter_dir)?;

        // Decode only the affected base tensors to f32.
        let mut f32_map: HashMap<String, (Vec<f32>, Vec<usize>)> = merged_keys
            .iter()
            .filter_map(|k| self.tensors.get(k).map(|e| (k, e)))
            .map(|(k, e)| {
                e.to_f32_vec(k)
                    .map(|floats| (k.clone(), (floats, e.shape.clone())))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let merged = crate::lora::merge_lora(&mut f32_map, adapter_dir)?;
        let n_merged = merged.len();

        // Write back only the merged entries.
        for key in &merged {
            if let (Some((new_f32, _)), Some(entry)) = (f32_map.get(key), self.tensors.get_mut(key))
            {
                entry.bytes = encode_f32_to_dtype(new_f32, entry.dtype, key)?;
            }
        }

        Ok(n_merged)
    }

    /// True if the store contains `key`.
    pub fn has(&self, key: &str) -> bool {
        self.tensors.contains_key(key)
    }

    /// Return the entry for `key`, or error if missing.
    pub(super) fn entry(&self, key: &str) -> Result<&TensorEntry> {
        self.tensors
            .get(key)
            .ok_or_else(|| IrodoriError::Weight(key.to_string()))
    }

    /// Load a raw `Tensor<B, D>` for `key`.
    ///
    /// Used by codec weight loaders that need tensors without the `Param` wrapper.
    pub fn tensor<B: Backend, const D: usize>(
        &self,
        key: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, D>> {
        let entry = self.entry(key)?;
        let td = entry.to_tensor_data::<D>(key)?;
        Ok(Tensor::<B, D>::from_data(td, device))
    }

    /// Build a `Param<Tensor<B, D>>` from `key`.
    pub(super) fn param<B: Backend, const D: usize>(
        &self,
        key: &str,
        device: &B::Device,
    ) -> Result<Param<Tensor<B, D>>> {
        let entry = self.entry(key)?;
        let td = entry.to_tensor_data::<D>(key)?;
        let tensor = Tensor::<B, D>::from_data(td, device);
        Ok(Param::initialized(ParamId::new(), tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::test_helpers::*;
    use burn::backend::NdArray;
    use safetensors::{Dtype, tensor::TensorView};

    type B = NdArray<f32>;

    #[test]
    fn tensor_store_load_basic() {
        let vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let file = write_safetensors(
            &[("test_tensor", f32_bytes(&vals), Dtype::F32, vec![2, 2])],
            &test_config_json(),
        );
        let store = TensorStore::load(file.path()).unwrap();
        assert!(store.has("test_tensor"));
        assert!(!store.has("nonexistent"));
    }

    #[test]
    fn tensor_store_missing_config_errors() {
        // Write a safetensors file without config_json metadata
        let data = f32_bytes(&[1.0]);
        let views = vec![("t", TensorView::new(Dtype::F32, vec![1], &data).unwrap())];
        let serialised = safetensors::tensor::serialize(views, None).unwrap();

        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), serialised).unwrap();
        let err = TensorStore::load(file.path());
        assert!(err.is_err());
    }

    #[test]
    fn tensor_store_raw_tensor_read() {
        let vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let file = write_safetensors(
            &[("my.weight", f32_bytes(&vals), Dtype::F32, vec![2, 3])],
            &test_config_json(),
        );
        let store = TensorStore::load(file.path()).unwrap();
        let t: Tensor<B, 2> = store.tensor("my.weight", &Default::default()).unwrap();
        let shape = t.shape();
        assert_eq!(shape.dims(), [2, 3]);
        let result: Vec<f32> = t.to_data().to_vec().unwrap();
        assert_eq!(result, vals);
    }
}
