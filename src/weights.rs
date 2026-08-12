//! Weight loading from safetensors checkpoints.
//!
//! Converts a Python-generated safetensors file into a fully initialised
//! `TextToLatentRfDiT<B>` model by constructing the corresponding burn Record
//! hierarchy and calling `model.load_record(record)`.
//!
//! # Key mapping
//! The Python model uses sequential indices for `cond_module` which must be
//! renamed before loading (see `scripts/convert_for_burn.py`):
//! - `cond_module.0.weight` → `cond_module.linear0.weight`
//! - `cond_module.2.weight` → `cond_module.linear1.weight`
//! - `cond_module.4.weight` → `cond_module.linear2.weight`

mod duration_records;
mod loaders;
mod modern_bert_records;
mod record_builders;
mod tensor_entry;
mod tensor_store;

#[cfg(test)]
mod test_helpers;

// --- Public re-exports ---
#[cfg(feature = "lora")]
pub use loaders::load_model_with_lora;
pub use loaders::{load_model, load_model_exact_only};
pub use tensor_store::TensorStore;
