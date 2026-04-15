//! Inference sampling configuration.

use serde::{Deserialize, Serialize};

/// Sampling / inference hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SamplingConfig {
    pub num_steps: usize,
    /// Per-modality CFG scales (v2).
    pub cfg_scale_text: f64,
    pub cfg_scale_caption: f64,
    pub cfg_scale_speaker: f64,
    pub cfg_guidance_mode: CfgGuidanceMode,
    /// Deprecated: single-scale override (sets all scales).
    pub cfg_scale: Option<f64>,
    pub cfg_min_t: f64,
    pub cfg_max_t: f64,
    pub truncation_factor: Option<f64>,
    pub rescale_k: Option<f64>,
    pub rescale_sigma: Option<f64>,
    pub context_kv_cache: bool,
    pub speaker_kv_scale: Option<f64>,
    pub speaker_kv_min_t: Option<f64>,
    pub speaker_kv_max_layers: Option<usize>,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            num_steps: 40,
            cfg_scale_text: 3.0,
            cfg_scale_caption: 3.0,
            cfg_scale_speaker: 5.0,
            cfg_guidance_mode: CfgGuidanceMode::Independent,
            cfg_scale: None,
            cfg_min_t: 0.5,
            cfg_max_t: 1.0,
            truncation_factor: None,
            rescale_k: None,
            rescale_sigma: None,
            context_kv_cache: true,
            speaker_kv_scale: None,
            speaker_kv_min_t: Some(0.9),
            speaker_kv_max_layers: None,
            seed: 0,
        }
    }
}

/// CFG guidance application strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum CfgGuidanceMode {
    /// Separate forward pass per dropped condition (most common).
    #[default]
    Independent,
    /// Single fully-unconditional forward pass.
    Joint,
    /// Alternate which condition is dropped each step.
    Alternating,
}

impl std::fmt::Display for CfgGuidanceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Independent => write!(f, "independent"),
            Self::Joint => write!(f, "joint"),
            Self::Alternating => write!(f, "alternating"),
        }
    }
}

impl std::str::FromStr for CfgGuidanceMode {
    type Err = crate::error::IrodoriError;

    fn from_str(s: &str) -> crate::error::Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "independent" => Ok(Self::Independent),
            "joint" => Ok(Self::Joint),
            "alternating" => Ok(Self::Alternating),
            other => Err(crate::error::IrodoriError::UnsupportedMode(format!(
                "unknown cfg_guidance_mode: {other:?}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_config_default_values() {
        let cfg = SamplingConfig::default();
        assert_eq!(cfg.num_steps, 40);
        assert_eq!(cfg.cfg_scale_text, 3.0);
        assert_eq!(cfg.cfg_scale_speaker, 5.0);
        assert!(cfg.context_kv_cache);
        assert_eq!(cfg.seed, 0);
        assert!(cfg.cfg_scale.is_none());
    }

    #[test]
    fn sampling_config_serde_roundtrip() {
        let cfg = SamplingConfig {
            num_steps: 20,
            cfg_scale_text: 2.5,
            cfg_guidance_mode: CfgGuidanceMode::Joint,
            seed: 42,
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let restored: SamplingConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, restored);
    }

    #[test]
    fn sampling_config_empty_json_is_default() {
        let from_empty: SamplingConfig =
            serde_json::from_str("{}").expect("empty JSON must deserialize");
        assert_eq!(from_empty, SamplingConfig::default());
    }

    #[test]
    fn sampling_config_deserializes_from_partial_json() {
        let json = r#"{"num_steps": 10, "seed": 99}"#;
        let cfg: SamplingConfig = serde_json::from_str(json).expect("deserialize partial");
        assert_eq!(cfg.num_steps, 10);
        assert_eq!(cfg.seed, 99);
        // Defaults fill in
        assert_eq!(cfg.cfg_scale_text, 3.0);
        assert!(cfg.context_kv_cache);
    }

    #[test]
    fn cfg_guidance_mode_from_str_valid() {
        assert_eq!(
            "independent".parse::<CfgGuidanceMode>().unwrap(),
            CfgGuidanceMode::Independent
        );
        assert_eq!(
            "JOINT".parse::<CfgGuidanceMode>().unwrap(),
            CfgGuidanceMode::Joint
        );
        assert_eq!(
            " Alternating ".parse::<CfgGuidanceMode>().unwrap(),
            CfgGuidanceMode::Alternating
        );
    }

    #[test]
    fn cfg_guidance_mode_from_str_invalid() {
        assert!("random".parse::<CfgGuidanceMode>().is_err());
        assert!("".parse::<CfgGuidanceMode>().is_err());
    }

    #[test]
    fn cfg_guidance_mode_display_roundtrip() {
        for mode in [
            CfgGuidanceMode::Independent,
            CfgGuidanceMode::Joint,
            CfgGuidanceMode::Alternating,
        ] {
            let s = mode.to_string();
            let parsed: CfgGuidanceMode = s.parse().expect("display output must parse back");
            assert_eq!(parsed, mode);
        }
    }

    #[test]
    fn cfg_guidance_mode_serde_rename() {
        let json = serde_json::to_string(&CfgGuidanceMode::Independent).unwrap();
        assert_eq!(json, r#""independent""#);

        let json = serde_json::to_string(&CfgGuidanceMode::Joint).unwrap();
        assert_eq!(json, r#""joint""#);

        let parsed: CfgGuidanceMode =
            serde_json::from_str(r#""alternating""#).expect("deserialize");
        assert_eq!(parsed, CfgGuidanceMode::Alternating);
    }
}
