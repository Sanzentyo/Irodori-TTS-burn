//! Resident, text-to-audio inference shared by embedders, CLI, worker and HTTP.
//! Tokenization and WAV dependencies are enabled only by the `synthesis` feature.

pub use crate::audio;
mod engine;
#[cfg(feature = "http-server")]
pub mod http;
mod text;
mod voice;
#[cfg(feature = "worker")]
pub mod worker;

pub use crate::rf::CancellationToken;
pub use engine::{SynthesisEngine, SynthesisModelPaths, SynthesisReport};
pub use text::TokenCounter;
pub use voice::Voice;

use crate::{
    CfgGuidanceMode, FlowParameterization, GuidanceConfig, IrodoriError, Result, SamplerMethod,
    SamplerParams, SpeakerKvConfig, TemporalRescaleConfig,
};
use serde::{Deserialize, Serialize};

/// User options shared by all text-to-audio entrypoints. Omitted RF options
/// use the model's defaults; explicitly supplied RF options reject MF models.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SynthesisRequest {
    pub text: String,
    pub instructions: String,
    pub voice: String,
    /// Irodori checkpoints support Japanese; `auto` selects it as well.
    pub language: String,
    pub seed: Option<u64>,
    pub steps: Option<usize>,
    pub seconds: Option<f64>,
    pub duration_scale: f64,
    pub speed: f64,
    pub keep_tail: bool,
    pub tail_window_size: usize,
    pub tail_std_threshold: f32,
    pub tail_mean_threshold: f32,
    pub cfg_scale_text: Option<f32>,
    pub cfg_scale_speaker: Option<f32>,
    pub cfg_scale_instructions: Option<f32>,
    pub cfg_guidance_mode: Option<CfgGuidanceMode>,
    pub cfg_min_t: Option<f32>,
    pub cfg_max_t: Option<f32>,
    pub truncation_factor: Option<f32>,
    pub rescale_k: Option<f32>,
    pub rescale_sigma: Option<f32>,
    pub speaker_kv_scale: Option<f32>,
    pub speaker_kv_min_t: Option<f32>,
    pub speaker_kv_max_layers: Option<usize>,
    pub sway_coeff: Option<f32>,
    pub speaker_uncond_mode: Option<crate::SpeakerUncondMode>,
}

impl Default for SynthesisRequest {
    fn default() -> Self {
        Self {
            text: String::new(),
            instructions: String::new(),
            voice: "none".into(),
            language: "auto".into(),
            seed: None,
            steps: None,
            seconds: None,
            duration_scale: 1.0,
            speed: 1.0,
            keep_tail: false,
            tail_window_size: 20,
            tail_std_threshold: 0.05,
            tail_mean_threshold: 0.1,
            cfg_scale_text: None,
            cfg_scale_speaker: None,
            cfg_scale_instructions: None,
            cfg_guidance_mode: None,
            cfg_min_t: None,
            cfg_max_t: None,
            truncation_factor: None,
            rescale_k: None,
            rescale_sigma: None,
            speaker_kv_scale: None,
            speaker_kv_min_t: None,
            speaker_kv_max_layers: None,
            sway_coeff: None,
            speaker_uncond_mode: None,
        }
    }
}

impl SynthesisRequest {
    pub fn validate(&self, flow: FlowParameterization) -> Result<()> {
        if !matches!(self.language.as_str(), "auto" | "ja") {
            return Err(IrodoriError::Config(
                "language must be 'auto' or 'ja'".into(),
            ));
        }
        if self.text.trim().is_empty() || self.text.len() > 1_048_576 {
            return Err(IrodoriError::Config(
                "text must contain 1 to 1048576 UTF-8 bytes".into(),
            ));
        }
        if !self.speed.is_finite() || !(0.25..=4.0).contains(&self.speed) {
            return Err(IrodoriError::Config("speed must be in [0.25, 4]".into()));
        }
        if !self.duration_scale.is_finite() || self.duration_scale <= 0.0 {
            return Err(IrodoriError::Config(
                "duration_scale must be finite and positive".into(),
            ));
        }
        if let Some(seconds) = self.seconds {
            if !seconds.is_finite() || !(0.5..=30.0).contains(&seconds) {
                return Err(IrodoriError::Config("seconds must be in [0.5, 30]".into()));
            }
            if self.duration_scale != 1.0 {
                return Err(IrodoriError::Config(
                    "seconds and duration_scale cannot be combined".into(),
                ));
            }
        }
        if self.tail_window_size == 0
            || self.tail_window_size > 750
            || !self.tail_std_threshold.is_finite()
            || self.tail_std_threshold < 0.0
            || !self.tail_mean_threshold.is_finite()
            || self.tail_mean_threshold < 0.0
        {
            return Err(IrodoriError::Config(
                "tail window must be in [1, 750] and thresholds finite and nonnegative".into(),
            ));
        }
        if self.keep_tail
            && (self.tail_window_size != 20
                || self.tail_std_threshold != 0.05
                || self.tail_mean_threshold != 0.1)
        {
            return Err(IrodoriError::Config(
                "custom tail thresholds have no effect with keep_tail".into(),
            ));
        }
        if self.steps.is_some_and(|steps| steps == 0 || steps > 1000) {
            return Err(IrodoriError::Config("steps must be in [1, 1000]".into()));
        }
        for (name, scale) in [
            ("cfg_scale_text", self.cfg_scale_text),
            ("cfg_scale_speaker", self.cfg_scale_speaker),
            ("cfg_scale_instructions", self.cfg_scale_instructions),
        ] {
            if scale.is_some_and(|value| !value.is_finite() || value < 0.0) {
                return Err(IrodoriError::Config(format!(
                    "{name} must be finite and nonnegative"
                )));
            }
        }
        if self.voice == "none"
            && (self.speaker_kv_scale.is_some()
                || self.speaker_uncond_mode == Some(crate::SpeakerUncondMode::Noise))
        {
            return Err(IrodoriError::Config(
                "speaker KV scaling and noise guidance require a voice".into(),
            ));
        }
        if self.voice == "none" && self.cfg_scale_speaker.is_some_and(|value| value != 0.0) {
            return Err(IrodoriError::Config(
                "speaker guidance requires a voice".into(),
            ));
        }
        if self.instructions.trim().is_empty()
            && self
                .cfg_scale_instructions
                .is_some_and(|value| value != 0.0)
        {
            return Err(IrodoriError::Config(
                "instruction guidance requires instructions".into(),
            ));
        }
        if self.rescale_k.is_some() != self.rescale_sigma.is_some() {
            return Err(IrodoriError::Config(
                "rescale_k and rescale_sigma must be supplied together".into(),
            ));
        }
        if self.speaker_kv_scale.is_none()
            && (self.speaker_kv_min_t.is_some() || self.speaker_kv_max_layers.is_some())
        {
            return Err(IrodoriError::Config(
                "speaker KV window/layers require speaker_kv_scale".into(),
            ));
        }
        if self.speaker_kv_max_layers == Some(0)
            || (self.speaker_kv_scale == Some(1.0)
                && (self.speaker_kv_min_t.is_some_and(|value| value != 0.9)
                    || self.speaker_kv_max_layers.is_some_and(|value| value != 12)))
        {
            return Err(IrodoriError::Config(
                "speaker KV layers must be positive; custom window/layers require a scale other than 1".into(),
            ));
        }
        if flow == FlowParameterization::Meanflow
            && (self.cfg_scale_text.is_some()
                || self.cfg_scale_speaker.is_some()
                || self.cfg_scale_instructions.is_some()
                || self.cfg_guidance_mode.is_some()
                || self.cfg_min_t.is_some()
                || self.cfg_max_t.is_some()
                || self.truncation_factor.is_some()
                || self.rescale_k.is_some()
                || self.speaker_kv_scale.is_some()
                || self.sway_coeff.is_some()
                || self.speaker_uncond_mode.is_some())
        {
            return Err(IrodoriError::Config(
                "MeanFlow checkpoints do not accept RF guidance or sampling controls".into(),
            ));
        }
        let params = self.sampler(
            flow,
            self.voice != "none",
            !self.instructions.trim().is_empty(),
        )?;
        if flow == FlowParameterization::RfVelocity
            && [
                params.guidance.scale_text,
                params.guidance.scale_speaker,
                params.guidance.scale_caption,
            ]
            .iter()
            .all(|value| *value == 0.0)
            && (self.cfg_guidance_mode.is_some()
                || self.cfg_min_t.is_some()
                || self.cfg_max_t.is_some()
                || self.speaker_uncond_mode.is_some())
        {
            return Err(IrodoriError::Config(
                "guidance controls require at least one active guidance scale".into(),
            ));
        }
        if params.guidance.mode == CfgGuidanceMode::Joint {
            let mut scales = [
                params.guidance.scale_text,
                params.guidance.scale_speaker,
                params.guidance.scale_caption,
            ]
            .into_iter()
            .filter(|scale| *scale > 0.0);
            if let Some(first) = scales.next()
                && scales.any(|scale| scale != first)
            {
                return Err(IrodoriError::Config(
                    "joint guidance requires equal active scales".into(),
                ));
            }
        }
        if self.speaker_uncond_mode == Some(crate::SpeakerUncondMode::Noise)
            && params.guidance.mode != CfgGuidanceMode::Joint
            && params.guidance.scale_speaker == 0.0
        {
            return Err(IrodoriError::Config(
                "noise guidance requires an active speaker guidance branch".into(),
            ));
        }
        params.validate()
    }

    fn sampler(
        &self,
        flow: FlowParameterization,
        speaker: bool,
        caption: bool,
    ) -> Result<SamplerParams> {
        if flow == FlowParameterization::Meanflow {
            return Ok(SamplerParams::meanflow(self.steps.unwrap_or(4)));
        }
        Ok(SamplerParams {
            num_steps: self.steps.unwrap_or(40),
            method: SamplerMethod::Euler,
            guidance: GuidanceConfig {
                mode: self.cfg_guidance_mode.unwrap_or_default(),
                scale_text: self.cfg_scale_text.unwrap_or(3.0),
                scale_speaker: if speaker {
                    self.cfg_scale_speaker.unwrap_or(5.0)
                } else {
                    0.0
                },
                scale_caption: if caption {
                    self.cfg_scale_instructions.unwrap_or(3.0)
                } else {
                    0.0
                },
                min_t: self.cfg_min_t.unwrap_or(0.5),
                max_t: self.cfg_max_t.unwrap_or(1.0),
            },
            truncation_factor: self.truncation_factor,
            temporal_rescale: self
                .rescale_k
                .zip(self.rescale_sigma)
                .map(|(k, sigma)| TemporalRescaleConfig { k, sigma }),
            speaker_kv: self.speaker_kv_scale.map(|scale| SpeakerKvConfig {
                scale,
                min_t: Some(self.speaker_kv_min_t.unwrap_or(0.9)),
                max_layers: Some(self.speaker_kv_max_layers.unwrap_or(12)),
            }),
            sway_coefficient: self.sway_coeff,
            speaker_uncond_mode: self.speaker_uncond_mode.unwrap_or_default(),
            ..SamplerParams::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn options_reject_ineffective_or_invalid_combinations() {
        let mut request = SynthesisRequest {
            text: "こんにちは".into(),
            ..Default::default()
        };
        assert!(request.validate(FlowParameterization::Meanflow).is_ok());
        request.cfg_scale_text = Some(3.0);
        assert!(request.validate(FlowParameterization::Meanflow).is_err());
        request.cfg_scale_text = None;
        request.rescale_k = Some(1.0);
        assert!(request.validate(FlowParameterization::RfVelocity).is_err());
        request.rescale_sigma = Some(1.0);
        assert!(request.validate(FlowParameterization::RfVelocity).is_ok());
        request.speed = f64::NAN;
        assert!(request.validate(FlowParameterization::RfVelocity).is_err());
        assert!(
            serde_json::from_str::<SynthesisRequest>(r#"{"text":"hello","unknown":true}"#).is_err()
        );
        for controls in [
            r#"{"keep_tail":true,"tail_window_size":3}"#,
            r#"{"speaker_kv_scale":1.0,"speaker_kv_min_t":0.5}"#,
            r#"{"speaker_kv_scale":1.0,"speaker_kv_max_layers":4}"#,
            r#"{"speaker_kv_scale":1.2,"speaker_kv_max_layers":0}"#,
        ] {
            let mut request: SynthesisRequest = serde_json::from_str(controls).unwrap();
            request.text = "こんにちは".into();
            request.voice = "reference".into();
            assert!(request.validate(FlowParameterization::RfVelocity).is_err());
        }
    }
}
