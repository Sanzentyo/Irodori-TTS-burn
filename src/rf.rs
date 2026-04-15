//! Rectified-flow Euler sampler with classifier-free guidance (CFG).
//!
//! Ports `irodori_tts/rf.py` to Rust + burn.
//!
//! # Usage
//!
//! ```rust,ignore
//! let params = SamplerParams::default();
//! let latent = sample_euler_rf_cfg(&model, request, &params, &device)?;
//! ```

mod euler_sampler;
mod kv_scaling;
mod math;
mod params;

pub use euler_sampler::sample_euler_rf_cfg;
pub use kv_scaling::scale_speaker_kv_cache;
pub use math::{rf_interpolate, rf_predict_x0, rf_velocity_target, temporal_score_rescale};
pub use params::{
    GuidanceConfig, SamplerParams, SamplingRequest, SpeakerKvConfig, TemporalRescaleConfig,
};
