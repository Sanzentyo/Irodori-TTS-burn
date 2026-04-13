//! LoRA model weight loading.
//!
//! Re-exports the `load_lora_model` function from [`crate::weights`],
//! keeping the training API self-contained under the `train` module.

pub use crate::weights::load_lora_model;
