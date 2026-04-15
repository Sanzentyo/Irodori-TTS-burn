//! DiT model: text-to-latent rectified flow diffusion transformer.
//!
//! Split into submodules for clarity:
//! - [`aux_conditioner`]: Speaker/Caption conditioning modules
//! - [`model`]: Core DiT struct, CondModule, forward passes

mod aux_conditioner;
mod model;

// Re-export public API
pub(crate) use aux_conditioner::build_aux_conditioner;
pub use aux_conditioner::{AuxConditioner, CaptionConditioner, SpeakerConditioner};
pub(crate) use model::init_zero_out_proj;
pub use model::{BlockDebugOutputs, CondModule, TextToLatentRfDiT};

// Re-export burn-generated Record types used by weight loading
pub use aux_conditioner::{
    AuxConditionerRecord, CaptionConditionerRecord, SpeakerConditionerRecord,
};
pub use model::{CondModuleRecord, TextToLatentRfDiTRecord};
