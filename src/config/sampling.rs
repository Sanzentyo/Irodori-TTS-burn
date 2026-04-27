//! Inference sampling configuration.

use serde::{Deserialize, Serialize};

/// ODE integration method for the RF sampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SamplerMethod {
    /// 1st-order Euler method (default).  1 NFE per step.
    #[default]
    Euler,
    /// 2nd-order Heun's method.  2 NFE per step; use half the steps for the same NFE.
    Heun,
    /// 4th-order Adams-Bashforth linear multistep method (PLMS-4).  1 NFE per step.
    ///
    /// Uses history of up to 4 past velocity evaluations with progressive startup
    /// (AB-1 → AB-2 → AB-3 → AB-4). Requires a uniform timestep schedule.
    /// **Not compatible with `CfgGuidanceMode::Alternating`**.
    PLMS4,
}

impl std::fmt::Display for SamplerMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Euler => write!(f, "euler"),
            Self::Heun => write!(f, "heun"),
            Self::PLMS4 => write!(f, "plms4"),
        }
    }
}

impl std::str::FromStr for SamplerMethod {
    type Err = crate::error::IrodoriError;

    fn from_str(s: &str) -> crate::error::Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "euler" => Ok(Self::Euler),
            "heun" => Ok(Self::Heun),
            "plms4" | "plms" => Ok(Self::PLMS4),
            other => Err(crate::error::IrodoriError::UnsupportedMode(format!(
                "unknown sampler method: {other:?}; expected euler, heun, or plms4"
            ))),
        }
    }
}

/// Sampling / inference hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SamplingConfig {
    pub num_steps: usize,
    /// ODE integration method.
    pub sampler_method: SamplerMethod,
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
            sampler_method: SamplerMethod::Euler,
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

    #[test]
    fn sampler_method_from_str_valid() {
        assert_eq!(
            "euler".parse::<SamplerMethod>().unwrap(),
            SamplerMethod::Euler
        );
        assert_eq!(
            "HEUN".parse::<SamplerMethod>().unwrap(),
            SamplerMethod::Heun
        );
        assert_eq!(
            " heun ".parse::<SamplerMethod>().unwrap(),
            SamplerMethod::Heun
        );
        assert_eq!(
            "plms4".parse::<SamplerMethod>().unwrap(),
            SamplerMethod::PLMS4
        );
        assert_eq!(
            "plms".parse::<SamplerMethod>().unwrap(),
            SamplerMethod::PLMS4
        );
        assert_eq!(
            "PLMS4".parse::<SamplerMethod>().unwrap(),
            SamplerMethod::PLMS4
        );
    }

    #[test]
    fn sampler_method_from_str_invalid() {
        assert!("runge".parse::<SamplerMethod>().is_err());
        assert!("".parse::<SamplerMethod>().is_err());
    }

    #[test]
    fn sampler_method_display_roundtrip() {
        for method in [
            SamplerMethod::Euler,
            SamplerMethod::Heun,
            SamplerMethod::PLMS4,
        ] {
            let s = method.to_string();
            let parsed: SamplerMethod = s.parse().expect("display output must parse back");
            assert_eq!(parsed, method);
        }
    }

    #[test]
    fn sampler_method_serde_rename() {
        assert_eq!(
            serde_json::to_string(&SamplerMethod::Euler).unwrap(),
            r#""euler""#
        );
        assert_eq!(
            serde_json::to_string(&SamplerMethod::Heun).unwrap(),
            r#""heun""#
        );
        assert_eq!(
            serde_json::to_string(&SamplerMethod::PLMS4).unwrap(),
            r#""plms4""#
        );
        let parsed: SamplerMethod = serde_json::from_str(r#""heun""#).unwrap();
        assert_eq!(parsed, SamplerMethod::Heun);
        let parsed_plms: SamplerMethod = serde_json::from_str(r#""plms4""#).unwrap();
        assert_eq!(parsed_plms, SamplerMethod::PLMS4);
    }

    #[test]
    fn sampling_config_with_sampler_method_roundtrip() {
        let cfg = SamplingConfig {
            num_steps: 20,
            sampler_method: SamplerMethod::Heun,
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let restored: SamplingConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, restored);
        assert_eq!(restored.sampler_method, SamplerMethod::Heun);
    }

    #[test]
    fn sampling_config_default_sampler_method_is_euler() {
        assert_eq!(
            SamplingConfig::default().sampler_method,
            SamplerMethod::Euler
        );
    }
}
