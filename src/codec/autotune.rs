//! Accuracy-approved, fail-closed codec selector manifests.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use serde::{Deserialize, Serialize};

use super::algorithm::{K7SelectorChoice, K7SelectorManifest, K7SelectorProblem};
use crate::{IrodoriError, autotune_approval::AutotuneRuntimeIdentity, validation::AudioMetrics};

pub const K7_SELECTOR_APPROVAL_SCHEMA_VERSION: u32 = 2;

/// One exact typed entry in a prepared k7 selector vector.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct K7SelectorSelection {
    pub problem: K7SelectorProblem,
    pub choice: K7SelectorChoice,
}

/// Multi-metric waveform gate recorded with a tuning case.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct K7SelectorAccuracyGate {
    pub maximum_max_abs: f64,
    pub maximum_mean_abs: f64,
    pub maximum_rmse: f64,
    pub minimum_snr_db: f64,
    pub minimum_cosine: f64,
}

impl K7SelectorAccuracyGate {
    pub const fn strict_fp16() -> Self {
        Self {
            maximum_max_abs: 0.005,
            maximum_mean_abs: 0.0005,
            maximum_rmse: 0.001,
            minimum_snr_db: 50.0,
            minimum_cosine: 0.99999,
        }
    }

    fn validate(self) -> crate::Result<()> {
        if !self.maximum_max_abs.is_finite()
            || self.maximum_max_abs < 0.0
            || !self.maximum_mean_abs.is_finite()
            || self.maximum_mean_abs < 0.0
            || !self.maximum_rmse.is_finite()
            || self.maximum_rmse < 0.0
            || !self.minimum_snr_db.is_finite()
            || !self.minimum_cosine.is_finite()
            || !(-1.0..=1.0).contains(&self.minimum_cosine)
        {
            return Err(cache_error("invalid k7 selector accuracy gate"));
        }
        Ok(())
    }

    fn accepts(self, metrics: &AudioMetrics) -> bool {
        metrics.max_abs_error.is_finite()
            && metrics.mean_abs_error.is_finite()
            && metrics.root_mean_square_error.is_finite()
            && !metrics.signal_to_noise_db.is_nan()
            && metrics.cosine_similarity.is_finite()
            && metrics.max_abs_error <= self.maximum_max_abs
            && metrics.mean_abs_error <= self.maximum_mean_abs
            && metrics.root_mean_square_error <= self.maximum_rmse
            && metrics.signal_to_noise_db >= self.minimum_snr_db
            && metrics.cosine_similarity >= self.minimum_cosine
    }
}

/// Same-process paired device-complete performance evidence.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct K7SelectorPerformanceReceipt {
    pub boundary: String,
    pub candidate_median_ms: f64,
    pub control_median_ms: f64,
    pub paired_block_delta_median_ms: f64,
    pub relative_improvement: f64,
    pub required_relative_improvement: f64,
    pub improved_blocks: usize,
    pub measured_blocks: usize,
    pub accepted: bool,
}

impl K7SelectorPerformanceReceipt {
    fn validate(&self) -> crate::Result<()> {
        if self.boundary != "device-complete"
            || !self.candidate_median_ms.is_finite()
            || self.candidate_median_ms <= 0.0
            || !self.control_median_ms.is_finite()
            || self.control_median_ms <= 0.0
            || !self.paired_block_delta_median_ms.is_finite()
            || !self.relative_improvement.is_finite()
            || !self.required_relative_improvement.is_finite()
            || !(0.0..1.0).contains(&self.required_relative_improvement)
            || self.improved_blocks > self.measured_blocks
            || self.measured_blocks == 0
        {
            return Err(cache_error("invalid k7 selector performance receipt"));
        }
        let expected_improvement =
            (-self.paired_block_delta_median_ms / self.control_median_ms).max(0.0);
        let tolerance = 1.0e-9_f64.max(expected_improvement.abs() * 1.0e-9);
        if (self.relative_improvement - expected_improvement).abs() > tolerance {
            return Err(cache_error(
                "k7 selector performance receipt has inconsistent relative improvement",
            ));
        }
        let expected_accepted = self.paired_block_delta_median_ms < 0.0
            && expected_improvement >= self.required_relative_improvement;
        if self.accepted != expected_accepted {
            return Err(cache_error(
                "k7 selector performance receipt has inconsistent acceptance decision",
            ));
        }
        Ok(())
    }
}

/// Complete evidence and final selection for one exact latent-frame shape.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct K7SelectorCaseReceipt {
    pub session_id: String,
    pub latent_frames: usize,
    pub fixture_sha256: String,
    pub precision: String,
    pub accuracy_gate: K7SelectorAccuracyGate,
    pub accuracy: AudioMetrics,
    pub candidate_waveform_sha256: String,
    pub control_waveform_sha256: String,
    pub selected_waveform_sha256: String,
    pub bitwise_equal: bool,
    pub deterministic: bool,
    pub performance: K7SelectorPerformanceReceipt,
    pub selections: Vec<K7SelectorSelection>,
}

impl K7SelectorCaseReceipt {
    pub fn selector(&self) -> crate::Result<K7SelectorManifest> {
        K7SelectorManifest::from_selections(
            self.selections
                .iter()
                .map(|entry| (entry.problem, entry.choice)),
        )
    }

    fn validate(&self) -> crate::Result<()> {
        if !valid_session_id(&self.session_id)
            || self.latent_frames == 0
            || self.precision != "fp16"
            || !valid_sha256(&self.fixture_sha256)
            || !valid_sha256(&self.candidate_waveform_sha256)
            || !valid_sha256(&self.control_waveform_sha256)
            || !valid_sha256(&self.selected_waveform_sha256)
            || !self.bitwise_equal
            || !self.deterministic
        {
            return Err(cache_error("invalid k7 selector accuracy receipt"));
        }
        self.accuracy_gate.validate()?;
        if !self.accuracy_gate.accepts(&self.accuracy) {
            return Err(cache_error("k7 selector waveform accuracy gate failed"));
        }
        self.performance.validate()?;
        let expected_selected = if self.performance.accepted {
            &self.candidate_waveform_sha256
        } else {
            &self.control_waveform_sha256
        };
        if &self.selected_waveform_sha256 != expected_selected {
            return Err(cache_error(
                "k7 selector selected waveform hash does not match its decision",
            ));
        }
        let selector = self.selector()?;
        selector.validate_decoder_shape(self.latent_frames)?;
        let released = K7SelectorManifest::released_decoder_geometry(self.latent_frames)?;
        if self.performance.accepted && selector == released {
            return Err(cache_error(
                "accepted k7 selector case must change at least one selection",
            ));
        }
        if !self.performance.accepted && selector != released {
            return Err(cache_error(
                "rejected k7 selector case must store the released geometry",
            ));
        }
        Ok(())
    }
}

/// Consensus decision for one exact decoder shape.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct K7SelectorShapeApproval {
    pub latent_frames: usize,
    pub accepted_tuning: bool,
    pub selections: Vec<K7SelectorSelection>,
    pub sessions: Vec<K7SelectorCaseReceipt>,
}

impl K7SelectorShapeApproval {
    fn derive(
        latent_frames: usize,
        minimum_fresh_sessions: usize,
        mut sessions: Vec<K7SelectorCaseReceipt>,
    ) -> crate::Result<Self> {
        if sessions.len() < minimum_fresh_sessions {
            return Err(cache_error(format!(
                "k7 selector shape {latent_frames} has {} fresh sessions; at least {minimum_fresh_sessions} are required",
                sessions.len()
            )));
        }
        sessions.sort_by(|left, right| left.session_id.cmp(&right.session_id));
        let mut session_ids = BTreeSet::new();
        for session in &sessions {
            session.validate()?;
            if session.latent_frames != latent_frames {
                return Err(cache_error(
                    "k7 selector shape contains a mismatched receipt",
                ));
            }
            if !session_ids.insert(session.session_id.as_str()) {
                return Err(cache_error(format!(
                    "duplicate k7 selector fresh session {}",
                    session.session_id
                )));
            }
        }

        let released = K7SelectorManifest::released_decoder_geometry(latent_frames)?;
        let first = sessions[0].selector()?;
        let accepted_tuning = sessions.iter().all(|session| session.performance.accepted)
            && first != released
            && sessions
                .iter()
                .skip(1)
                .all(|session| session.selector().is_ok_and(|selector| selector == first));
        let selected = if accepted_tuning { first } else { released };
        Ok(Self {
            latent_frames,
            accepted_tuning,
            selections: selected
                .selections()
                .map(|(problem, choice)| K7SelectorSelection { problem, choice })
                .collect(),
            sessions,
        })
    }

    fn selector(&self) -> crate::Result<K7SelectorManifest> {
        K7SelectorManifest::from_selections(
            self.selections
                .iter()
                .map(|entry| (entry.problem, entry.choice)),
        )
    }

    fn validate(&self, minimum_fresh_sessions: usize) -> crate::Result<()> {
        let derived = Self::derive(
            self.latent_frames,
            minimum_fresh_sessions,
            self.sessions.clone(),
        )?;
        if self.accepted_tuning != derived.accepted_tuning || self.selections != derived.selections
        {
            return Err(cache_error(
                "approved k7 selector shape does not match fresh-session consensus",
            ));
        }
        self.selector()?.validate_decoder_shape(self.latent_frames)
    }
}

/// Cross-process selector bundle sealed to one exact runtime and build.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ApprovedK7SelectorManifestSet {
    pub schema_version: u32,
    pub identity: AutotuneRuntimeIdentity,
    pub kernel_profile: String,
    pub source_sha256: String,
    pub binary_sha256: String,
    pub minimum_fresh_sessions: usize,
    pub required_latent_frames: Vec<usize>,
    pub shapes: Vec<K7SelectorShapeApproval>,
}

impl ApprovedK7SelectorManifestSet {
    pub fn seal(
        identity: AutotuneRuntimeIdentity,
        kernel_profile: String,
        source_sha256: String,
        binary_sha256: String,
        minimum_fresh_sessions: usize,
        required_latent_frames: Vec<usize>,
        cases: Vec<K7SelectorCaseReceipt>,
    ) -> crate::Result<Self> {
        if minimum_fresh_sessions == 0 {
            return Err(cache_error(
                "k7 selector approval requires at least one fresh session",
            ));
        }
        let mut grouped = BTreeMap::<usize, Vec<K7SelectorCaseReceipt>>::new();
        for case in cases {
            grouped.entry(case.latent_frames).or_default().push(case);
        }
        let shapes = grouped
            .into_iter()
            .map(|(latent_frames, sessions)| {
                K7SelectorShapeApproval::derive(latent_frames, minimum_fresh_sessions, sessions)
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let manifest = Self {
            schema_version: K7_SELECTOR_APPROVAL_SCHEMA_VERSION,
            identity,
            kernel_profile,
            source_sha256,
            binary_sha256,
            minimum_fresh_sessions,
            required_latent_frames,
            shapes,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn load(path: &Path) -> crate::Result<Self> {
        let manifest: Self = serde_json::from_slice(&fs::read(path)?).map_err(|error| {
            cache_error(format!(
                "invalid approved k7 selector manifest {}: {error}",
                path.display()
            ))
        })?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> crate::Result<()> {
        if self.schema_version != K7_SELECTOR_APPROVAL_SCHEMA_VERSION
            || self.kernel_profile.is_empty()
            || !valid_sha256(&self.source_sha256)
            || !valid_sha256(&self.binary_sha256)
            || self.minimum_fresh_sessions == 0
            || self.required_latent_frames.is_empty()
            || self.shapes.is_empty()
            || self.identity.float_dtype != "f16"
        {
            return Err(cache_error("invalid approved k7 selector manifest"));
        }
        let mut required = self.required_latent_frames.clone();
        required.sort_unstable();
        required.dedup();
        if required.len() != self.required_latent_frames.len() || required.contains(&0) {
            return Err(cache_error(
                "approved k7 selector required shape set is invalid",
            ));
        }
        let mut actual = BTreeSet::new();
        for shape in &self.shapes {
            shape.validate(self.minimum_fresh_sessions)?;
            if !actual.insert(shape.latent_frames) {
                return Err(cache_error(format!(
                    "duplicate approved k7 selector shape {}",
                    shape.latent_frames
                )));
            }
        }
        if actual.into_iter().collect::<Vec<_>>() != required {
            return Err(cache_error(
                "approved k7 selector shapes do not exactly match the required shape set",
            ));
        }
        Ok(())
    }

    pub fn verify(
        &self,
        actual_identity: &AutotuneRuntimeIdentity,
        kernel_profile: &str,
        source_sha256: &str,
        binary_sha256: &str,
        latent_frames: usize,
    ) -> crate::Result<K7SelectorVerificationReceipt> {
        self.validate()?;
        if &self.identity != actual_identity
            || self.kernel_profile != kernel_profile
            || self.source_sha256 != source_sha256
            || self.binary_sha256 != binary_sha256
        {
            return Err(cache_error(
                "approved k7 selector runtime or build identity mismatch",
            ));
        }
        let shape = self
            .shapes
            .iter()
            .find(|shape| shape.latent_frames == latent_frames)
            .ok_or_else(|| {
                cache_error(format!(
                    "approved k7 selector has no case for {latent_frames} latent frames"
                ))
            })?;
        let selector = shape.selector()?;
        Ok(K7SelectorVerificationReceipt {
            latent_frames,
            accepted_tuning: shape.accepted_tuning,
            selection_count: selector.selections().count(),
            selector,
        })
    }
}

/// Proof that a matching case was extracted after all pins were checked.
#[derive(Clone, Debug)]
pub struct K7SelectorVerificationReceipt {
    pub latent_frames: usize,
    pub accepted_tuning: bool,
    pub selection_count: usize,
    selector: K7SelectorManifest,
}

impl K7SelectorVerificationReceipt {
    pub fn selector(&self) -> &K7SelectorManifest {
        &self.selector
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_session_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
}

fn cache_error(message: impl Into<String>) -> IrodoriError {
    IrodoriError::Cache(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> AutotuneRuntimeIdentity {
        AutotuneRuntimeIdentity {
            burn_version: "0.22.0-pre.2".into(),
            burn_cubecl_version: "0.22.0-pre.2".into(),
            cubecl_version: "0.11.0-pre.2".into(),
            runtime: "wgpu-auto".into(),
            wgpu_backend: "vulkan".into(),
            gpu_name: "gpu".into(),
            driver_version: "driver".into(),
            pci_bus_id: "bus".into(),
            allocator_policy: "sub-slices".into(),
            float_dtype: "f16".into(),
            int_dtype: "i32".into(),
            bounds_check_policy: "checked".into(),
            model_revision: "model".into(),
            model_sha256: "a".repeat(64),
            codec_revision: "codec".into(),
            converted_codec_sha256: "b".repeat(64),
        }
    }

    fn case(session_id: &str, accepted: bool) -> K7SelectorCaseReceipt {
        let latent_frames = 45;
        let mut selector = K7SelectorManifest::released_decoder_geometry(latent_frames).unwrap();
        if accepted {
            let problem = selector.selections().next().unwrap().0;
            selector = selector
                .with_selection(problem, K7SelectorChoice::SingleDoublePartition)
                .unwrap();
        }
        let delta = if accepted { -0.1 } else { 0.1 };
        K7SelectorCaseReceipt {
            session_id: session_id.into(),
            latent_frames,
            fixture_sha256: "c".repeat(64),
            precision: "fp16".into(),
            accuracy_gate: K7SelectorAccuracyGate::strict_fp16(),
            accuracy: AudioMetrics {
                sample_count: 86_400,
                max_abs_error: 0.001,
                mean_abs_error: 0.0001,
                root_mean_square_error: 0.0002,
                signal_to_noise_db: 56.0,
                cosine_similarity: 0.999999,
            },
            candidate_waveform_sha256: "d".repeat(64),
            control_waveform_sha256: "d".repeat(64),
            selected_waveform_sha256: "d".repeat(64),
            bitwise_equal: true,
            deterministic: true,
            performance: K7SelectorPerformanceReceipt {
                boundary: "device-complete".into(),
                candidate_median_ms: 10.0,
                control_median_ms: 10.0,
                paired_block_delta_median_ms: delta,
                relative_improvement: (-delta / 10.0_f64).max(0.0),
                required_relative_improvement: 0.002,
                improved_blocks: usize::from(accepted) * 6,
                measured_blocks: 10,
                accepted,
            },
            selections: selector
                .selections()
                .map(|(problem, choice)| K7SelectorSelection { problem, choice })
                .collect(),
        }
    }

    #[test]
    fn approved_set_fails_closed_on_identity_and_shape() {
        let manifest = ApprovedK7SelectorManifestSet::seal(
            identity(),
            "kernel-v5".into(),
            "e".repeat(64),
            "f".repeat(64),
            3,
            vec![45],
            vec![case("s1", false), case("s2", false), case("s3", false)],
        )
        .unwrap();
        let receipt = manifest
            .verify(
                &identity(),
                "kernel-v5",
                &"e".repeat(64),
                &"f".repeat(64),
                45,
            )
            .unwrap();
        assert_eq!(receipt.selection_count, 12);
        assert!(!receipt.accepted_tuning);
        assert!(
            manifest
                .verify(
                    &identity(),
                    "kernel-v5",
                    &"e".repeat(64),
                    &"f".repeat(64),
                    489,
                )
                .is_err()
        );
    }

    #[test]
    fn rejected_case_cannot_store_a_tuned_selector() {
        let mut receipt = case("s1", false);
        receipt.selections[0].choice = K7SelectorChoice::SingleDoublePartition;
        assert!(receipt.validate().is_err());
    }

    #[test]
    fn accepted_case_must_change_the_selector_vector() {
        let mut receipt = case("s1", true);
        receipt.selections = K7SelectorManifest::released_decoder_geometry(45)
            .unwrap()
            .selections()
            .map(|(problem, choice)| K7SelectorSelection { problem, choice })
            .collect();
        assert!(receipt.validate().is_err());
    }

    #[test]
    fn consensus_requires_every_fresh_session_to_choose_the_same_plan() {
        let accepted = ApprovedK7SelectorManifestSet::seal(
            identity(),
            "kernel-v5".into(),
            "e".repeat(64),
            "f".repeat(64),
            3,
            vec![45],
            vec![case("s1", true), case("s2", true), case("s3", true)],
        )
        .unwrap();
        assert!(accepted.shapes[0].accepted_tuning);

        let rejected = ApprovedK7SelectorManifestSet::seal(
            identity(),
            "kernel-v5".into(),
            "e".repeat(64),
            "f".repeat(64),
            3,
            vec![45],
            vec![case("s1", true), case("s2", false), case("s3", true)],
        )
        .unwrap();
        assert!(!rejected.shapes[0].accepted_tuning);
    }
}
