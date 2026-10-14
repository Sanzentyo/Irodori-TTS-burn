//! Weight loading from safetensors checkpoints.
//!
//! Converts a Python-generated safetensors file into a fully initialised
//! `TextToLatentRfDiT` model with Burn's direct module store.
//!
//! # Key mapping
//! The Python model uses sequential indices for `cond_module` which must be
//! renamed before loading (see `scripts/convert_for_burn.py`):
//! - `cond_module.0.weight` → `cond_module.linear0.weight`
//! - `cond_module.2.weight` → `cond_module.linear1.weight`
//! - `cond_module.4.weight` → `cond_module.linear2.weight`

#[cfg(all(feature = "synthesis", feature = "packed-weights"))]
mod identified_file;
mod indexed_store;
mod loaders;
#[cfg(feature = "packed-weights")]
pub mod packed;
mod tensor_entry;
mod tensor_store;

#[cfg(test)]
mod test_helpers;

// --- Public re-exports ---
#[cfg(feature = "synthesis")]
pub(crate) use loaders::load_model_with_identity;
#[cfg(feature = "lora")]
pub use loaders::load_model_with_lora;
pub use loaders::{
    ModelCheckpointLoader, load_model, load_model_exact_only, load_model_with_float_dtype,
    load_model_with_float_dtype_and_loader, load_model_with_loader,
};
pub use tensor_store::TensorStore;
