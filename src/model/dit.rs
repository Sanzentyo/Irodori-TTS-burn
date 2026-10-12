//! DiT model: text-to-latent rectified flow diffusion transformer.
//!
//! Split into submodules for clarity:
//! - [`aux_conditioner`]: Speaker/Caption conditioning modules
//! - [`model`]: Core DiT struct, CondModule, forward passes

mod aux_conditioner;
mod model;

// Re-export public API
pub use aux_conditioner::{
    AuxConditioner, BothConditioner, CaptionConditioner, SpeakerConditioner,
};
pub use model::{BlockDebugOutputs, TextToLatentRfDiT};
