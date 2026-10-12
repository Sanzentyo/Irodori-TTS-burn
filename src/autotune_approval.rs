//! Fail-closed sealing of numerically approved CubeCL autotune selections.
//!
//! Candidate choices can interact numerically across a full inference graph.
//! Approval therefore covers the complete cache selection vector, not one key
//! in isolation. The selected candidates are sealed only after an end-to-end
//! strict-FP32 gate passes, then verified exactly before cache restoration.

use std::{fs, path::Path};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub const AUTOTUNE_APPROVAL_SCHEMA_VERSION: u32 = 2;

/// Runtime pins which must match both sealing and restoration campaigns.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AutotuneRuntimeIdentity {
    pub burn_version: String,
    pub burn_cubecl_version: String,
    pub cubecl_version: String,
    pub runtime: String,
    pub wgpu_backend: String,
    pub gpu_name: String,
    pub driver_version: String,
    pub pci_bus_id: String,
    pub allocator_policy: String,
    pub float_dtype: String,
    pub int_dtype: String,
    pub bounds_check_policy: String,
    pub model_revision: String,
    pub model_sha256: String,
    pub codec_revision: String,
    pub converted_codec_sha256: String,
}

/// Repository policy defining the fixture and minimum numerical gate.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AutotuneAccuracyPolicy {
    pub schema_version: u32,
    pub identity: AutotuneRuntimeIdentity,
    pub required_cases: Vec<AutotuneAccuracyCasePolicy>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AutotuneAccuracyCasePolicy {
    pub fixture_sha256: String,
    pub latent_frames: usize,
    pub hard_gate: AutotuneNumericalGate,
    pub target_waveform_snr_db: f64,
}

/// A multi-metric hard gate. SNR alone is too sensitive to the fixture's
/// signal energy, while max error alone is too sensitive to one sample.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AutotuneNumericalGate {
    pub latent: AutotuneMetricGate,
    pub waveform: AutotuneMetricGate,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AutotuneMetricGate {
    pub maximum_max_abs: f64,
    pub maximum_mean_abs: f64,
    pub maximum_rmse: f64,
    pub minimum_snr_db: f64,
    pub minimum_cosine: f64,
}

impl AutotuneAccuracyPolicy {
    pub fn load(path: &Path) -> Result<Self, AutotuneApprovalError> {
        let policy: Self = serde_json::from_slice(&fs::read(path)?)?;
        policy.validate()?;
        Ok(policy)
    }

    pub fn validate(&self) -> Result<(), AutotuneApprovalError> {
        validate_schema(self.schema_version)?;
        if self.required_cases.is_empty() {
            return Err(AutotuneApprovalError::InvalidAccuracyPolicy);
        }
        for case in &self.required_cases {
            validate_sha256("policy fixture", &case.fixture_sha256)?;
            if case.latent_frames == 0
                || !case.target_waveform_snr_db.is_finite()
                || !case.hard_gate.is_valid()
                || case.target_waveform_snr_db < case.hard_gate.waveform.minimum_snr_db
            {
                return Err(AutotuneApprovalError::InvalidAccuracyPolicy);
            }
        }
        let mut identities = self
            .required_cases
            .iter()
            .map(|case| (case.fixture_sha256.as_str(), case.latent_frames))
            .collect::<Vec<_>>();
        identities.sort_unstable();
        identities.dedup();
        if identities.len() != self.required_cases.len() {
            return Err(AutotuneApprovalError::InvalidAccuracyPolicy);
        }
        Ok(())
    }

    pub fn verify_identity(
        &self,
        actual: &AutotuneRuntimeIdentity,
    ) -> Result<(), AutotuneApprovalError> {
        verify_identity(&self.identity, actual)
    }
}

/// Fresh end-to-end evidence used to approve one complete cache vector.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AutotuneAccuracyEvidence {
    pub cases: Vec<AutotuneAccuracyCaseEvidence>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AutotuneAccuracyCaseEvidence {
    pub fixture_sha256: String,
    pub latent_frames: usize,
    pub latent: AutotuneMetricEvidence,
    pub waveform: AutotuneMetricEvidence,
    pub latent_sha256: String,
    pub waveform_sha256: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AutotuneMetricEvidence {
    pub max_abs: f64,
    pub mean_abs: f64,
    pub rmse: f64,
    pub snr_db: f64,
    pub cosine: f64,
}

impl AutotuneNumericalGate {
    fn is_valid(&self) -> bool {
        self.latent.is_valid() && self.waveform.is_valid()
    }
}

impl AutotuneMetricGate {
    fn is_valid(&self) -> bool {
        self.maximum_max_abs.is_finite()
            && self.maximum_max_abs >= 0.0
            && self.maximum_mean_abs.is_finite()
            && self.maximum_mean_abs >= 0.0
            && self.maximum_rmse.is_finite()
            && self.maximum_rmse >= 0.0
            && self.minimum_snr_db.is_finite()
            && self.minimum_cosine.is_finite()
            && (-1.0..=1.0).contains(&self.minimum_cosine)
    }

    fn accepts(&self, actual: &AutotuneMetricEvidence) -> bool {
        actual.is_finite()
            && actual.max_abs <= self.maximum_max_abs
            && actual.mean_abs <= self.maximum_mean_abs
            && actual.rmse <= self.maximum_rmse
            && actual.snr_db >= self.minimum_snr_db
            && actual.cosine >= self.minimum_cosine
    }
}

impl AutotuneMetricEvidence {
    fn is_finite(&self) -> bool {
        self.max_abs.is_finite()
            && self.mean_abs.is_finite()
            && self.rmse.is_finite()
            && self.snr_db.is_finite()
            && self.cosine.is_finite()
    }
}

impl AutotuneAccuracyEvidence {
    fn validate(&self, policy: &AutotuneAccuracyPolicy) -> Result<(), AutotuneApprovalError> {
        if self.cases.len() != policy.required_cases.len() {
            return Err(AutotuneApprovalError::AccuracyCaseSetMismatch);
        }
        for required in &policy.required_cases {
            let Some(actual) = self.cases.iter().find(|actual| {
                actual.fixture_sha256 == required.fixture_sha256
                    && actual.latent_frames == required.latent_frames
            }) else {
                return Err(AutotuneApprovalError::AccuracyCaseSetMismatch);
            };
            validate_sha256("evidence fixture", &actual.fixture_sha256)?;
            validate_sha256("approved latent", &actual.latent_sha256)?;
            validate_sha256("approved waveform", &actual.waveform_sha256)?;
            if !required.hard_gate.latent.accepts(&actual.latent) {
                return Err(AutotuneApprovalError::AccuracyGateFailed {
                    latent_frames: actual.latent_frames,
                    output: "latent",
                });
            }
            if !required.hard_gate.waveform.accepts(&actual.waveform) {
                return Err(AutotuneApprovalError::AccuracyGateFailed {
                    latent_frames: actual.latent_frames,
                    output: "waveform",
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AutotuneCandidate {
    pub index: usize,
    pub name: String,
}

/// One member of the complete cache selection vector.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AutotuneSelection {
    pub cache_log_relative_path: String,
    pub cache_key: Value,
    pub selected_candidate: AutotuneCandidate,
}

/// Complete accuracy-approved cache selection vector.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ApprovedAutotuneCacheManifest {
    pub schema_version: u32,
    pub identity: AutotuneRuntimeIdentity,
    pub required_cases: Vec<AutotuneAccuracyCasePolicy>,
    pub evidence: AutotuneAccuracyEvidence,
    pub selections: Vec<AutotuneSelection>,
}

impl ApprovedAutotuneCacheManifest {
    pub fn load(path: &Path) -> Result<Self, AutotuneApprovalError> {
        let manifest: Self = serde_json::from_slice(&fs::read(path)?)?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), AutotuneApprovalError> {
        validate_schema(self.schema_version)?;
        if self.selections.is_empty() || self.required_cases.is_empty() {
            return Err(AutotuneApprovalError::InvalidApprovedManifest);
        }
        let policy = AutotuneAccuracyPolicy {
            schema_version: self.schema_version,
            identity: self.identity.clone(),
            required_cases: self.required_cases.clone(),
        };
        policy.validate()?;
        self.evidence.validate(&policy)?;
        Ok(())
    }

    pub fn verify(
        &self,
        actual_identity: &AutotuneRuntimeIdentity,
        cache_root: &Path,
    ) -> Result<AutotuneVerificationReceipt, AutotuneApprovalError> {
        self.validate()?;
        verify_identity(&self.identity, actual_identity)?;
        let actual = collect_autotune_selections(cache_root)?;
        if actual != self.selections {
            return Err(AutotuneApprovalError::SelectionVectorMismatch {
                expected_count: self.selections.len(),
                actual_count: actual.len(),
            });
        }
        Ok(AutotuneVerificationReceipt {
            selection_count: actual.len(),
            exact_match: true,
        })
    }
}

/// Seal a cache only after its fresh end-to-end evidence passes the policy.
pub fn seal_autotune_cache(
    policy: &AutotuneAccuracyPolicy,
    actual_identity: &AutotuneRuntimeIdentity,
    evidence: AutotuneAccuracyEvidence,
    cache_root: &Path,
) -> Result<ApprovedAutotuneCacheManifest, AutotuneApprovalError> {
    policy.validate()?;
    policy.verify_identity(actual_identity)?;
    evidence.validate(policy)?;
    let selections = collect_autotune_selections(cache_root)?;
    if selections.is_empty() {
        return Err(AutotuneApprovalError::EmptySelectionVector);
    }
    Ok(ApprovedAutotuneCacheManifest {
        schema_version: AUTOTUNE_APPROVAL_SCHEMA_VERSION,
        identity: actual_identity.clone(),
        required_cases: policy.required_cases.clone(),
        evidence,
        selections,
    })
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AutotuneVerificationReceipt {
    pub selection_count: usize,
    pub exact_match: bool,
}

fn collect_autotune_selections(
    cache_root: &Path,
) -> Result<Vec<AutotuneSelection>, AutotuneApprovalError> {
    let mut logs = Vec::new();
    collect_logs(cache_root, cache_root, &mut logs)?;
    logs.sort();
    let mut selections = Vec::new();
    for (relative, path) in logs {
        let source = fs::read_to_string(path)?;
        for line in source.lines().filter(|line| !line.trim().is_empty()) {
            let entry: Value = serde_json::from_str(line)?;
            let cache_key = entry
                .get("key")
                .cloned()
                .ok_or(AutotuneApprovalError::MissingCacheKey)?;
            let decision = entry.get("value").unwrap_or(&entry);
            let selected_index = decision
                .get("fastest_index")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or(AutotuneApprovalError::MissingFastestIndex)?;
            let selected_candidate = candidate(decision, selected_index)?;
            selections.push(AutotuneSelection {
                cache_log_relative_path: relative.clone(),
                cache_key,
                selected_candidate,
            });
        }
    }
    selections.sort_by(|left, right| {
        left.cache_log_relative_path
            .cmp(&right.cache_log_relative_path)
            .then_with(|| left.cache_key.to_string().cmp(&right.cache_key.to_string()))
    });
    Ok(selections)
}

fn collect_logs(
    root: &Path,
    directory: &Path,
    logs: &mut Vec<(String, std::path::PathBuf)>,
) -> Result<(), AutotuneApprovalError> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_logs(root, &path, logs)?;
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".json.log"))
        {
            let relative = path
                .strip_prefix(root)
                .map_err(|_| AutotuneApprovalError::CachePathEscaped)?
                .components()
                .map(|component| component.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            logs.push((relative, path));
        }
    }
    Ok(())
}

fn candidate(
    decision: &Value,
    target_index: usize,
) -> Result<AutotuneCandidate, AutotuneApprovalError> {
    decision
        .get("results")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|result| result.pointer("/outcome/Ok"))
        .find_map(|result| {
            let index = result.get("index")?.as_u64()?;
            if usize::try_from(index).ok()? != target_index {
                return None;
            }
            Some(AutotuneCandidate {
                index: target_index,
                name: result.get("name")?.as_str()?.to_owned(),
            })
        })
        .ok_or(AutotuneApprovalError::CandidateNotFound {
            index: target_index,
        })
}

fn verify_identity(
    expected: &AutotuneRuntimeIdentity,
    actual: &AutotuneRuntimeIdentity,
) -> Result<(), AutotuneApprovalError> {
    if expected != actual {
        return Err(AutotuneApprovalError::IdentityMismatch {
            expected: Box::new(expected.clone()),
            actual: Box::new(actual.clone()),
        });
    }
    Ok(())
}

fn validate_schema(actual: u32) -> Result<(), AutotuneApprovalError> {
    if actual != AUTOTUNE_APPROVAL_SCHEMA_VERSION {
        return Err(AutotuneApprovalError::UnsupportedSchema {
            actual,
            expected: AUTOTUNE_APPROVAL_SCHEMA_VERSION,
        });
    }
    Ok(())
}

fn validate_sha256(label: &'static str, hash: &str) -> Result<(), AutotuneApprovalError> {
    if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(AutotuneApprovalError::InvalidSha256 {
            label,
            value: hash.to_owned(),
        });
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum AutotuneApprovalError {
    #[error("I/O error while processing autotune approval: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid JSON in autotune approval input: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported autotune approval schema {actual}; expected {expected}")]
    UnsupportedSchema { actual: u32, expected: u32 },
    #[error("invalid autotune accuracy policy")]
    InvalidAccuracyPolicy,
    #[error("approved autotune manifest is internally inconsistent")]
    InvalidApprovedManifest,
    #[error("{label} SHA-256 is not 64 hexadecimal characters: {value}")]
    InvalidSha256 { label: &'static str, value: String },
    #[error("runtime identity does not match the accuracy policy")]
    IdentityMismatch {
        expected: Box<AutotuneRuntimeIdentity>,
        actual: Box<AutotuneRuntimeIdentity>,
    },
    #[error("accuracy evidence does not cover exactly the required fixture/frame cases")]
    AccuracyCaseSetMismatch,
    #[error("accuracy hard gate failed for {output} at {latent_frames} frames")]
    AccuracyGateFailed {
        latent_frames: usize,
        output: &'static str,
    },
    #[error("autotune cache contains no selectable entries")]
    EmptySelectionVector,
    #[error("autotune cache entry has no key")]
    MissingCacheKey,
    #[error("autotune cache entry has no numeric fastest_index")]
    MissingFastestIndex,
    #[error("candidate index {index} is absent from the cache evidence")]
    CandidateNotFound { index: usize },
    #[error("cache path escaped the configured root")]
    CachePathEscaped,
    #[error("restored selection vector differs: expected {expected_count}, actual {actual_count}")]
    SelectionVectorMismatch {
        expected_count: usize,
        actual_count: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> AutotuneRuntimeIdentity {
        AutotuneRuntimeIdentity {
            burn_version: "0.21".into(),
            burn_cubecl_version: "0.21".into(),
            cubecl_version: "0.10".into(),
            runtime: "wgpu".into(),
            wgpu_backend: "vulkan".into(),
            gpu_name: "gpu".into(),
            driver_version: "driver".into(),
            pci_bus_id: "bus".into(),
            allocator_policy: "exclusive-pages".into(),
            float_dtype: "f32".into(),
            int_dtype: "i32".into(),
            bounds_check_policy: "checked".into(),
            model_revision: "model-rev".into(),
            model_sha256: "a".repeat(64),
            codec_revision: "codec-rev".into(),
            converted_codec_sha256: "b".repeat(64),
        }
    }

    fn policy() -> AutotuneAccuracyPolicy {
        AutotuneAccuracyPolicy {
            schema_version: AUTOTUNE_APPROVAL_SCHEMA_VERSION,
            identity: identity(),
            required_cases: vec![AutotuneAccuracyCasePolicy {
                fixture_sha256: "c".repeat(64),
                latent_frames: 45,
                hard_gate: AutotuneNumericalGate {
                    latent: gate(90.0),
                    waveform: gate(80.0),
                },
                target_waveform_snr_db: 85.0,
            }],
        }
    }

    fn gate(minimum_snr_db: f64) -> AutotuneMetricGate {
        AutotuneMetricGate {
            maximum_max_abs: 2.0e-4,
            maximum_mean_abs: 1.0e-5,
            maximum_rmse: 2.0e-5,
            minimum_snr_db,
            minimum_cosine: 0.99999999,
        }
    }

    fn evidence(snr: f64) -> AutotuneAccuracyEvidence {
        AutotuneAccuracyEvidence {
            cases: vec![AutotuneAccuracyCaseEvidence {
                fixture_sha256: "c".repeat(64),
                latent_frames: 45,
                latent: metrics(104.0),
                waveform: metrics(snr),
                latent_sha256: "d".repeat(64),
                waveform_sha256: "e".repeat(64),
            }],
        }
    }

    fn metrics(snr_db: f64) -> AutotuneMetricEvidence {
        AutotuneMetricEvidence {
            max_abs: 1.0e-4,
            mean_abs: 2.0e-6,
            rmse: 8.0e-6,
            snr_db,
            cosine: 0.999999999,
        }
    }

    fn write_cache(root: &Path, fastest: usize) {
        let directory = root.join("autotune/device");
        fs::create_dir_all(&directory).unwrap();
        let entry = serde_json::json!({
            "key":{"key":{"shape":[4,4,64]},"checksum":"kernel"},
            "value":{"fastest_index":fastest,"results":[
                {"outcome":{"Ok":{"name":"candidate-0","index":0}}},
                {"outcome":{"Ok":{"name":"candidate-7","index":7}}}
            ]}
        });
        fs::write(directory.join("matmul.json.log"), format!("{entry}\n")).unwrap();
    }

    fn write_cubecl_011_record(root: &Path, fastest: usize) {
        let directory = root.join("autotune/device");
        fs::create_dir_all(&directory).unwrap();
        let entry = serde_json::json!({
            "key":{"schema":1,"dtype":"f16","output_length":1344},
            "fastest_index":fastest,
            "fastest_time":{"secs":0,"nanos":12000},
            "results":[
                {"outcome":{"Ok":{"name":"sync-cyclic-single-row-v1","index":0}}},
                {"outcome":{"Ok":{"name":"sync-cyclic-multi-row-v1","index":1}}}
            ],
            "log_context":null,
            "checks":null
        });
        fs::write(directory.join("k7.json.log"), format!("{entry}\n")).unwrap();
    }

    #[test]
    fn seals_and_verifies_the_complete_selection_vector() {
        let dir = tempfile::tempdir().unwrap();
        write_cache(dir.path(), 7);
        let manifest =
            seal_autotune_cache(&policy(), &identity(), evidence(86.7), dir.path()).unwrap();

        let receipt = manifest.verify(&identity(), dir.path()).unwrap();

        assert!(receipt.exact_match);
        assert_eq!(receipt.selection_count, 1);
        assert_eq!(manifest.selections[0].selected_candidate.index, 7);
    }

    #[test]
    fn seals_cubecl_011_recorder_schema() {
        let dir = tempfile::tempdir().unwrap();
        write_cubecl_011_record(dir.path(), 1);

        let manifest =
            seal_autotune_cache(&policy(), &identity(), evidence(86.7), dir.path()).unwrap();

        assert_eq!(manifest.selections.len(), 1);
        assert_eq!(manifest.selections[0].selected_candidate.index, 1);
        assert_eq!(
            manifest.selections[0].selected_candidate.name,
            "sync-cyclic-multi-row-v1"
        );
        assert!(
            manifest
                .verify(&identity(), dir.path())
                .unwrap()
                .exact_match
        );
    }

    #[test]
    fn refuses_to_seal_a_numerically_failing_cache() {
        let dir = tempfile::tempdir().unwrap();
        write_cache(dir.path(), 0);

        assert!(matches!(
            seal_autotune_cache(&policy(), &identity(), evidence(79.9), dir.path()),
            Err(AutotuneApprovalError::AccuracyGateFailed { .. })
        ));
    }

    #[test]
    fn restore_rejects_any_changed_candidate() {
        let dir = tempfile::tempdir().unwrap();
        write_cache(dir.path(), 7);
        let manifest =
            seal_autotune_cache(&policy(), &identity(), evidence(86.7), dir.path()).unwrap();
        write_cache(dir.path(), 0);

        assert!(matches!(
            manifest.verify(&identity(), dir.path()),
            Err(AutotuneApprovalError::SelectionVectorMismatch { .. })
        ));
    }
}
