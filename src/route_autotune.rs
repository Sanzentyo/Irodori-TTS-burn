//! Accuracy-approved, device-specific route selection for the WGPU graph.
//!
//! The table is resolved once before model execution. Request hot paths use a
//! direct `(batch, sequence)` index; they do not inspect adapter names, parse
//! environment variables, or hash dynamic keys. An approved manifest starts
//! from portable routes and enables only exact measured cells.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
    process::Command,
    sync::OnceLock,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{IrodoriError, Result};

pub const ROUTE_AUTOTUNE_SCHEMA_VERSION: u32 = 1;
pub const ROUTE_ABI_VERSION: &str = "v4-dit-route-1";
pub const ROUTE_MANIFEST_SET_FILE: &str = "v4-approved-routes.json";
pub const MAX_TUNED_BATCH: usize = 3;
pub const MAX_TUNED_SEQUENCE: usize = 685;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RfBatchClass {
    Single,
    GuidedPair,
    GuidedTriple,
}

impl RfBatchClass {
    pub const fn from_batch(batch: usize) -> Option<Self> {
        match batch {
            1 => Some(Self::Single),
            2 => Some(Self::GuidedPair),
            3 => Some(Self::GuidedTriple),
            _ => None,
        }
    }

    pub const fn batch(self) -> usize {
        match self {
            Self::Single => 1,
            Self::GuidedPair => 2,
            Self::GuidedTriple => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct RouteProblem {
    pub batch_class: RfBatchClass,
    pub sequence: usize,
}

impl RouteProblem {
    pub fn new(batch: usize, sequence: usize) -> Result<Self> {
        let batch_class = RfBatchClass::from_batch(batch).ok_or_else(|| {
            IrodoriError::Config(format!(
                "route problem batch must be within 1..={MAX_TUNED_BATCH}, got {batch}"
            ))
        })?;
        if !(1..=MAX_TUNED_SEQUENCE).contains(&sequence) {
            return Err(IrodoriError::Config(format!(
                "route problem sequence must be within 1..={MAX_TUNED_SEQUENCE}, got {sequence}"
            )));
        }
        Ok(Self {
            batch_class,
            sequence,
        })
    }

    pub const fn batch(self) -> usize {
        self.batch_class.batch()
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionRoute {
    DefaultGraph,
    HandwrittenT64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionOutputWeightRoute {
    SourceColumnFlat,
    PackedRowFlat,
    PackedRowRank3,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MlpContractWeightRoute {
    SourceColumnFlat,
    PackedRowFlat,
    PackedRowRank3,
}

/// Stable candidate ID. The enum variant fixes the operation, so a weight
/// route cannot accidentally be applied to a projection problem.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case", tag = "component", content = "route")]
pub enum RouteChoice {
    AttentionQkvProjection(ProjectionRoute),
    AttentionOutputProjection(ProjectionRoute),
    MlpExpandProjection(ProjectionRoute),
    MlpContract(ProjectionRoute),
    AttentionOutputWeight(AttentionOutputWeightRoute),
    MlpContractWeight(MlpContractWeightRoute),
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteOperation {
    AttentionQkvProjection,
    AttentionOutputProjection,
    MlpExpandProjection,
    MlpContract,
    AttentionOutputWeight,
    MlpContractWeight,
}

impl RouteChoice {
    pub const fn operation(self) -> RouteOperation {
        match self {
            Self::AttentionQkvProjection(_) => RouteOperation::AttentionQkvProjection,
            Self::AttentionOutputProjection(_) => RouteOperation::AttentionOutputProjection,
            Self::MlpExpandProjection(_) => RouteOperation::MlpExpandProjection,
            Self::MlpContract(_) => RouteOperation::MlpContract,
            Self::AttentionOutputWeight(_) => RouteOperation::AttentionOutputWeight,
            Self::MlpContractWeight(_) => RouteOperation::MlpContractWeight,
        }
    }

    pub const fn is_portable(self) -> bool {
        matches!(
            self,
            Self::AttentionQkvProjection(ProjectionRoute::DefaultGraph)
                | Self::AttentionOutputProjection(ProjectionRoute::DefaultGraph)
                | Self::MlpExpandProjection(ProjectionRoute::DefaultGraph)
                | Self::MlpContract(ProjectionRoute::DefaultGraph)
                | Self::AttentionOutputWeight(AttentionOutputWeightRoute::SourceColumnFlat)
                | Self::MlpContractWeight(MlpContractWeightRoute::SourceColumnFlat)
        )
    }
}

impl RouteOperation {
    pub fn candidates(self, problem: RouteProblem) -> &'static [RouteChoice] {
        const QKV_PORTABLE: [RouteChoice; 1] = [RouteChoice::AttentionQkvProjection(
            ProjectionRoute::DefaultGraph,
        )];
        const QKV_ALL: [RouteChoice; 2] = [
            RouteChoice::AttentionQkvProjection(ProjectionRoute::DefaultGraph),
            RouteChoice::AttentionQkvProjection(ProjectionRoute::HandwrittenT64),
        ];
        const ATTENTION_OUTPUT_PORTABLE: [RouteChoice; 1] =
            [RouteChoice::AttentionOutputProjection(
                ProjectionRoute::DefaultGraph,
            )];
        const ATTENTION_OUTPUT_ALL: [RouteChoice; 2] = [
            RouteChoice::AttentionOutputProjection(ProjectionRoute::DefaultGraph),
            RouteChoice::AttentionOutputProjection(ProjectionRoute::HandwrittenT64),
        ];
        const MLP_EXPAND_PORTABLE: [RouteChoice; 1] = [RouteChoice::MlpExpandProjection(
            ProjectionRoute::DefaultGraph,
        )];
        const MLP_EXPAND_ALL: [RouteChoice; 2] = [
            RouteChoice::MlpExpandProjection(ProjectionRoute::DefaultGraph),
            RouteChoice::MlpExpandProjection(ProjectionRoute::HandwrittenT64),
        ];
        const MLP_CONTRACT_PORTABLE: [RouteChoice; 1] =
            [RouteChoice::MlpContract(ProjectionRoute::DefaultGraph)];
        const MLP_CONTRACT_ALL: [RouteChoice; 2] = [
            RouteChoice::MlpContract(ProjectionRoute::DefaultGraph),
            RouteChoice::MlpContract(ProjectionRoute::HandwrittenT64),
        ];
        const ATTENTION_WEIGHTS: [RouteChoice; 3] = [
            RouteChoice::AttentionOutputWeight(AttentionOutputWeightRoute::SourceColumnFlat),
            RouteChoice::AttentionOutputWeight(AttentionOutputWeightRoute::PackedRowFlat),
            RouteChoice::AttentionOutputWeight(AttentionOutputWeightRoute::PackedRowRank3),
        ];
        const MLP_WEIGHTS: [RouteChoice; 3] = [
            RouteChoice::MlpContractWeight(MlpContractWeightRoute::SourceColumnFlat),
            RouteChoice::MlpContractWeight(MlpContractWeightRoute::PackedRowFlat),
            RouteChoice::MlpContractWeight(MlpContractWeightRoute::PackedRowRank3),
        ];

        let t64_capable = (13..=MAX_TUNED_SEQUENCE).contains(&problem.sequence);
        match (self, t64_capable) {
            (Self::AttentionQkvProjection, false) => &QKV_PORTABLE,
            (Self::AttentionQkvProjection, true) => &QKV_ALL,
            (Self::AttentionOutputProjection, false) => &ATTENTION_OUTPUT_PORTABLE,
            (Self::AttentionOutputProjection, true) => &ATTENTION_OUTPUT_ALL,
            (Self::MlpExpandProjection, false) => &MLP_EXPAND_PORTABLE,
            (Self::MlpExpandProjection, true) => &MLP_EXPAND_ALL,
            (Self::MlpContract, false) => &MLP_CONTRACT_PORTABLE,
            (Self::MlpContract, true) => &MLP_CONTRACT_ALL,
            (Self::AttentionOutputWeight, _) => &ATTENTION_WEIGHTS,
            (Self::MlpContractWeight, _) => &MLP_WEIGHTS,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RouteDeviceIdentity {
    pub adapter_name: String,
    pub backend: String,
    pub device_type: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver: String,
    pub driver_info: String,
    pub os: String,
    pub platform_version: String,
    pub architecture: String,
    pub precision: String,
    pub allocator_policy: String,
    pub compiler_policy: String,
    pub application_version: String,
    pub burn_version: String,
    pub burn_cubecl_version: String,
    pub cubecl_version: String,
    pub cubek_version: String,
    pub wgpu_version: String,
    pub model_sha256: String,
    pub codec_sha256: String,
    pub binary_sha256: String,
}

impl RouteDeviceIdentity {
    pub fn validate(&self) -> Result<()> {
        for (label, value) in [
            ("adapter_name", self.adapter_name.as_str()),
            ("backend", self.backend.as_str()),
            ("device_type", self.device_type.as_str()),
            ("os", self.os.as_str()),
            ("architecture", self.architecture.as_str()),
            ("precision", self.precision.as_str()),
            ("allocator_policy", self.allocator_policy.as_str()),
            ("compiler_policy", self.compiler_policy.as_str()),
            ("application_version", self.application_version.as_str()),
            ("burn_version", self.burn_version.as_str()),
            ("burn_cubecl_version", self.burn_cubecl_version.as_str()),
            ("cubecl_version", self.cubecl_version.as_str()),
            ("cubek_version", self.cubek_version.as_str()),
            ("wgpu_version", self.wgpu_version.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(IrodoriError::Config(format!(
                    "route identity {label} must not be empty"
                )));
            }
        }
        for (label, digest) in [
            ("model_sha256", self.model_sha256.as_str()),
            ("codec_sha256", self.codec_sha256.as_str()),
            ("binary_sha256", self.binary_sha256.as_str()),
        ] {
            if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(IrodoriError::Config(format!(
                    "route identity {label} must be a SHA-256 digest"
                )));
            }
        }
        Ok(())
    }

    /// Persistent route reuse is deliberately stricter than structural JSON
    /// validation. Some browser and compatibility adapters do not expose a
    /// stable driver/device identity; those environments may tune for the
    /// current process, but must not reuse another process's decision.
    pub fn persistent_cache_eligibility(&self) -> PersistentRouteCacheEligibility {
        let metal_identity = self.backend.eq_ignore_ascii_case("metal")
            && !self.adapter_name.eq_ignore_ascii_case("unknown");
        if self.vendor_id == 0 && self.device_id == 0 && !metal_identity {
            PersistentRouteCacheEligibility::ProcessLocalOnly(
                RouteCacheMissReason::UnstableDeviceIdentity,
            )
        } else if (self.driver.trim().is_empty() || self.driver_info.trim().is_empty())
            && (!metal_identity || self.platform_version.trim().is_empty())
        {
            PersistentRouteCacheEligibility::ProcessLocalOnly(
                RouteCacheMissReason::IncompleteDriverIdentity,
            )
        } else {
            PersistentRouteCacheEligibility::Persistent
        }
    }

    pub fn fingerprint_sha256(&self) -> Result<String> {
        self.validate()?;
        let mut hasher = Sha256::new();
        hasher.update(serde_json::to_vec(self)?);
        Ok(format!("{:x}", hasher.finalize()))
    }
}

/// Hash a source, checkpoint, or current executable without loading it all in
/// memory. This is part of exact route identity, not a content cache.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut input = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn current_binary_sha256() -> Result<String> {
    sha256_file(&std::env::current_exe()?)
}

/// Kernel/OS build identity used when a platform integrates the GPU driver
/// with the operating system (notably Metal). No shell is involved.
pub fn current_platform_version() -> Option<String> {
    #[cfg(target_family = "unix")]
    let output = Command::new("uname")
        .args(["-s", "-r", "-v"])
        .output()
        .ok()?;
    #[cfg(target_family = "windows")]
    let output = Command::new("cmd").args(["/C", "ver"]).output().ok()?;
    #[cfg(not(any(target_family = "unix", target_family = "windows")))]
    return None;

    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|version| !version.is_empty())
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteCacheMissReason {
    ManifestNotFound,
    NoExactDeviceProfile,
    UnstableDeviceIdentity,
    IncompleteDriverIdentity,
    PortableRequested,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersistentRouteCacheEligibility {
    Persistent,
    ProcessLocalOnly(RouteCacheMissReason),
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct RouteAccuracyMetrics {
    pub max_abs: f64,
    pub mean_abs: f64,
    pub rmse: f64,
    pub snr_db: f64,
    pub cosine: f64,
}

impl RouteAccuracyMetrics {
    fn finite(self) -> bool {
        self.max_abs.is_finite()
            && self.mean_abs.is_finite()
            && self.rmse.is_finite()
            && self.snr_db.is_finite()
            && self.cosine.is_finite()
    }

    fn latent_hard_pass(self) -> bool {
        self.finite()
            && self.max_abs <= 2.0e-4
            && self.mean_abs <= 1.0e-5
            && self.rmse <= 2.0e-5
            && self.snr_db >= 90.0
            && self.cosine >= 0.999_999_99
    }

    fn waveform_hard_pass(self) -> bool {
        self.finite()
            && self.max_abs <= 1.5e-4
            && self.mean_abs <= 5.0e-6
            && self.rmse <= 1.0e-5
            && self.snr_db >= 80.0
            && self.cosine >= 0.999_999_99
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RouteCandidateMeasurement {
    pub identity_sha256: String,
    pub problem: RouteProblem,
    pub choice: RouteChoice,
    /// One device-complete median per fresh process/session.
    pub fresh_session_medians_ns: Vec<u64>,
    pub measured_requests_per_session: usize,
    pub euler_steps: usize,
    pub schedule_f32_bits: Vec<u32>,
    pub forward_batches: Vec<usize>,
    pub model_layers: usize,
    pub model_block_calls: usize,
    pub local_latent: RouteAccuracyMetrics,
    pub final_waveform: RouteAccuracyMetrics,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteCandidateRejectionReason {
    Unsupported,
    CompilationFailure,
    LaunchFailure,
    OutOfMemory,
    NonFiniteOutput,
    Timeout,
    TimestampUnavailable,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RouteCandidateRejection {
    pub identity_sha256: String,
    pub problem: RouteProblem,
    pub choice: RouteChoice,
    pub reason: RouteCandidateRejectionReason,
    pub detail: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RouteTuningCase {
    pub problem: RouteProblem,
    pub operations: Vec<RouteOperation>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RouteTuningWorkload {
    pub cases: Vec<RouteTuningCase>,
}

impl RouteTuningWorkload {
    pub fn validate(&self) -> Result<()> {
        if self.cases.is_empty() {
            return Err(IrodoriError::Config(
                "route tuning workload must not be empty".to_owned(),
            ));
        }
        let mut exact_cases = BTreeSet::new();
        for case in &self.cases {
            RouteProblem::new(case.problem.batch(), case.problem.sequence)?;
            if case.operations.is_empty() {
                return Err(IrodoriError::Config(
                    "route tuning case must contain operations".to_owned(),
                ));
            }
            let mut operations = BTreeSet::new();
            for operation in &case.operations {
                if !operations.insert(*operation) || !exact_cases.insert((case.problem, *operation))
                {
                    return Err(IrodoriError::Config(
                        "route tuning workload contains a duplicate exact operation".to_owned(),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RouteCandidateRequest {
    pub problem: RouteProblem,
    pub choice: RouteChoice,
    pub policy: RouteTuningPolicy,
}

#[derive(Clone, Debug)]
pub enum RouteCandidateRun {
    Measured {
        fresh_session_medians_ns: Vec<u64>,
        measured_requests_per_session: usize,
        euler_steps: usize,
        schedule_f32_bits: Vec<u32>,
        forward_batches: Vec<usize>,
        model_layers: usize,
        model_block_calls: usize,
        local_latent: RouteAccuracyMetrics,
        final_waveform: RouteAccuracyMetrics,
    },
    Rejected {
        reason: RouteCandidateRejectionReason,
        detail: String,
    },
}

/// Model-specific execution boundary used by the generic route enumerator.
/// Implementations may use fresh child processes, but must preserve the
/// request's exact identity and timing/accuracy boundaries.
pub trait RouteCandidateRunner {
    fn run_candidate(&mut self, request: RouteCandidateRequest) -> Result<RouteCandidateRun>;
}

/// Enumerate every physically available candidate and seal the best approved
/// route. No candidate can disappear: a runner must return measured evidence
/// or an explicit rejection for each request.
pub fn autotune_routes(
    identity: RouteDeviceIdentity,
    policy: RouteTuningPolicy,
    workload: &RouteTuningWorkload,
    runner: &mut impl RouteCandidateRunner,
) -> Result<ApprovedRouteManifest> {
    workload.validate()?;
    policy.validate()?;
    let identity_sha256 = identity.fingerprint_sha256()?;
    let mut measurements = Vec::new();
    let mut rejections = Vec::new();
    for case in &workload.cases {
        for &operation in &case.operations {
            for &choice in operation.candidates(case.problem) {
                let request = RouteCandidateRequest {
                    problem: case.problem,
                    choice,
                    policy,
                };
                match runner.run_candidate(request)? {
                    RouteCandidateRun::Measured {
                        fresh_session_medians_ns,
                        measured_requests_per_session,
                        euler_steps,
                        schedule_f32_bits,
                        forward_batches,
                        model_layers,
                        model_block_calls,
                        local_latent,
                        final_waveform,
                    } => measurements.push(RouteCandidateMeasurement {
                        identity_sha256: identity_sha256.clone(),
                        problem: case.problem,
                        choice,
                        fresh_session_medians_ns,
                        measured_requests_per_session,
                        euler_steps,
                        schedule_f32_bits,
                        forward_batches,
                        model_layers,
                        model_block_calls,
                        local_latent,
                        final_waveform,
                    }),
                    RouteCandidateRun::Rejected { reason, detail } => {
                        rejections.push(RouteCandidateRejection {
                            identity_sha256: identity_sha256.clone(),
                            problem: case.problem,
                            choice,
                            reason,
                            detail,
                        });
                    }
                }
            }
        }
    }
    select_approved_routes_with_rejections(identity, policy, measurements, rejections)
}

impl RouteCandidateMeasurement {
    fn median_ns(&self) -> Option<u64> {
        median(self.fresh_session_medians_ns.clone())
    }

    fn accuracy_disposition(&self) -> AccuracyDisposition {
        if !self.local_latent.latent_hard_pass() || !self.final_waveform.waveform_hard_pass() {
            AccuracyDisposition::Reject
        } else if self.final_waveform.snr_db >= 85.0 {
            AccuracyDisposition::ApprovedTarget
        } else {
            AccuracyDisposition::ApprovedWithWarning
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AccuracyDisposition {
    Reject,
    ApprovedWithWarning,
    ApprovedTarget,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RouteTuningPolicy {
    pub minimum_fresh_sessions: usize,
    pub minimum_measured_requests_per_session: usize,
    pub minimum_improvement_basis_points: u32,
}

impl Default for RouteTuningPolicy {
    fn default() -> Self {
        Self {
            minimum_fresh_sessions: 5,
            minimum_measured_requests_per_session: 10,
            minimum_improvement_basis_points: 200,
        }
    }
}

impl RouteTuningPolicy {
    fn validate(self) -> Result<()> {
        if self.minimum_fresh_sessions == 0
            || self.minimum_measured_requests_per_session == 0
            || self.minimum_improvement_basis_points > 10_000
        {
            return Err(IrodoriError::Config(
                "invalid route tuning measurement policy".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteSelectionReason {
    FastestAccuracyApproved,
    PortableWithinNoiseFloor,
    PortableBecauseCandidatesFailedAccuracy,
    PortableBecauseBaselineFailedAccuracy,
    PortableOnlyCandidate,
    PortableBecauseCandidatesUnavailable,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ApprovedRouteSelection {
    pub problem: RouteProblem,
    pub choice: RouteChoice,
    pub reason: RouteSelectionReason,
    pub selected_median_ns: u64,
    pub portable_median_ns: u64,
    pub fresh_sessions: usize,
    pub measured_requests_per_session: usize,
    pub accuracy: AccuracyDisposition,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ApprovedRouteManifest {
    pub schema_version: u32,
    pub route_abi: String,
    pub identity: RouteDeviceIdentity,
    pub tuning_policy: RouteTuningPolicy,
    pub selections: Vec<ApprovedRouteSelection>,
}

impl ApprovedRouteManifest {
    pub fn load(path: &Path) -> Result<Self> {
        let manifest: Self = serde_json::from_slice(&fs::read(path)?)?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != ROUTE_AUTOTUNE_SCHEMA_VERSION
            || self.route_abi != ROUTE_ABI_VERSION
        {
            return Err(IrodoriError::Config(format!(
                "unsupported route manifest schema/ABI: schema={} abi={}",
                self.schema_version, self.route_abi
            )));
        }
        self.identity.validate()?;
        self.tuning_policy.validate()?;
        if self.selections.is_empty() {
            return Err(IrodoriError::Config(
                "approved route manifest must contain selections".to_owned(),
            ));
        }
        let mut keys = BTreeSet::new();
        for selection in &self.selections {
            RouteProblem::new(selection.problem.batch(), selection.problem.sequence)?;
            let key = (selection.problem, selection.choice.operation());
            if !keys.insert(key)
                || !selection
                    .choice
                    .operation()
                    .candidates(selection.problem)
                    .contains(&selection.choice)
                || selection.fresh_sessions < self.tuning_policy.minimum_fresh_sessions
                || selection.measured_requests_per_session
                    < self.tuning_policy.minimum_measured_requests_per_session
            {
                return Err(IrodoriError::Config(
                    "approved route manifest contains duplicate or under-sampled selections"
                        .to_owned(),
                ));
            }
            if !selection.choice.is_portable() && selection.accuracy == AccuracyDisposition::Reject
            {
                return Err(IrodoriError::Config(
                    "an optimized route cannot be sealed with rejected accuracy".to_owned(),
                ));
            }
        }
        Ok(())
    }

    pub fn verify_identity(&self, actual: &RouteDeviceIdentity) -> Result<()> {
        actual.validate()?;
        if &self.identity != actual {
            return Err(IrodoriError::Config(
                "route manifest device/runtime identity mismatch".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Immutable collection of independently approved GPU/driver profiles.
///
/// A service resolves one exact entry at startup. Profiles are never pooled,
/// bucketed by marketing name, or inherited across a driver update.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ApprovedRouteManifestSet {
    pub schema_version: u32,
    pub route_abi: String,
    pub profiles: Vec<ApprovedRouteManifest>,
}

impl ApprovedRouteManifestSet {
    pub fn new(profiles: Vec<ApprovedRouteManifest>) -> Result<Self> {
        let manifest_set = Self {
            schema_version: ROUTE_AUTOTUNE_SCHEMA_VERSION,
            route_abi: ROUTE_ABI_VERSION.to_owned(),
            profiles,
        };
        manifest_set.validate()?;
        Ok(manifest_set)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let manifest_set: Self = serde_json::from_slice(&fs::read(path)?)?;
        manifest_set.validate()?;
        Ok(manifest_set)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != ROUTE_AUTOTUNE_SCHEMA_VERSION
            || self.route_abi != ROUTE_ABI_VERSION
        {
            return Err(IrodoriError::Config(format!(
                "unsupported route manifest-set schema/ABI: schema={} abi={}",
                self.schema_version, self.route_abi
            )));
        }
        if self.profiles.is_empty() {
            return Err(IrodoriError::Config(
                "approved route manifest set must contain at least one profile".to_owned(),
            ));
        }
        let mut identities = Vec::with_capacity(self.profiles.len());
        for profile in &self.profiles {
            profile.validate()?;
            if profile.schema_version != self.schema_version || profile.route_abi != self.route_abi
            {
                return Err(IrodoriError::Config(
                    "route profile schema/ABI differs from its manifest set".to_owned(),
                ));
            }
            if profile.identity.persistent_cache_eligibility()
                != PersistentRouteCacheEligibility::Persistent
            {
                return Err(IrodoriError::Config(
                    "a persistent route manifest cannot contain a process-local-only identity"
                        .to_owned(),
                ));
            }
            if identities.contains(&&profile.identity) {
                return Err(IrodoriError::Config(
                    "route manifest set contains a duplicate exact identity".to_owned(),
                ));
            }
            identities.push(&profile.identity);
        }
        Ok(())
    }

    pub fn resolve<'a>(
        &'a self,
        actual: &RouteDeviceIdentity,
    ) -> Result<RouteManifestResolution<'a>> {
        self.validate()?;
        actual.validate()?;
        if let PersistentRouteCacheEligibility::ProcessLocalOnly(reason) =
            actual.persistent_cache_eligibility()
        {
            return Ok(RouteManifestResolution::Portable { reason });
        }
        Ok(self
            .profiles
            .iter()
            .find(|profile| profile.identity == *actual)
            .map_or(
                RouteManifestResolution::Portable {
                    reason: RouteCacheMissReason::NoExactDeviceProfile,
                },
                RouteManifestResolution::Approved,
            ))
    }

    pub fn profiles(&self) -> impl ExactSizeIterator<Item = &ApprovedRouteManifest> {
        self.profiles.iter()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RouteManifestResolution<'a> {
    Approved(&'a ApprovedRouteManifest),
    Portable { reason: RouteCacheMissReason },
}

/// Platform application cache location for immutable approved route profiles.
/// `cubecl_root` is normally `.../Irodori-TTS-burn/cubecl`; routes are a
/// sibling because CubeCL may replace its own versioned environment.
pub fn default_route_manifest_set_path(cubecl_root: &Path) -> PathBuf {
    cubecl_root
        .parent()
        .unwrap_or(cubecl_root)
        .join("routes")
        .join(ROUTE_MANIFEST_SET_FILE)
}

/// Select the fastest accuracy-approved candidate for every exact problem.
/// A candidate must beat the portable median by the configured margin;
/// otherwise the portable route remains selected.
pub fn select_approved_routes(
    identity: RouteDeviceIdentity,
    tuning_policy: RouteTuningPolicy,
    measurements: Vec<RouteCandidateMeasurement>,
) -> Result<ApprovedRouteManifest> {
    select_approved_routes_with_rejections(identity, tuning_policy, measurements, Vec::new())
}

/// Variant used by the automatic candidate runner. Every physically available
/// candidate must have either a valid measurement or an explicit fail-closed
/// rejection; silently unmeasured alternatives are not accepted.
pub fn select_approved_routes_with_rejections(
    identity: RouteDeviceIdentity,
    tuning_policy: RouteTuningPolicy,
    measurements: Vec<RouteCandidateMeasurement>,
    rejections: Vec<RouteCandidateRejection>,
) -> Result<ApprovedRouteManifest> {
    identity.validate()?;
    let identity_sha256 = identity.fingerprint_sha256()?;
    tuning_policy.validate()?;
    if measurements.is_empty() {
        return Err(IrodoriError::Config(
            "route tuning measurements must not be empty".to_owned(),
        ));
    }

    let mut groups = BTreeMap::<(RouteProblem, RouteOperation), Vec<_>>::new();
    for measurement in measurements {
        RouteProblem::new(measurement.problem.batch(), measurement.problem.sequence)?;
        if measurement.identity_sha256 != identity_sha256
            || measurement.euler_steps != 40
            || measurement.schedule_f32_bits != expected_linear_schedule_bits(40)
            || measurement.forward_batches.len() != 40
            || !measurement
                .forward_batches
                .iter()
                .all(|batch| (1..=MAX_TUNED_BATCH).contains(batch))
            || !measurement
                .forward_batches
                .contains(&measurement.problem.batch())
            || measurement.model_layers != 12
            || measurement.model_block_calls != 480
            || measurement.fresh_session_medians_ns.len() < tuning_policy.minimum_fresh_sessions
            || measurement.measured_requests_per_session
                < tuning_policy.minimum_measured_requests_per_session
            || measurement.median_ns().is_none_or(|median| median == 0)
        {
            return Err(IrodoriError::Config(
                "route tuning evidence identity/work manifest/timing contract is invalid"
                    .to_owned(),
            ));
        }
        groups
            .entry((measurement.problem, measurement.choice.operation()))
            .or_default()
            .push(measurement);
    }

    let mut rejected = BTreeMap::<(RouteProblem, RouteOperation), Vec<_>>::new();
    for rejection in rejections {
        RouteProblem::new(rejection.problem.batch(), rejection.problem.sequence)?;
        if rejection.identity_sha256 != identity_sha256 || rejection.detail.trim().is_empty() {
            return Err(IrodoriError::Config(
                "route candidate rejection detail must not be empty".to_owned(),
            ));
        }
        rejected
            .entry((rejection.problem, rejection.choice.operation()))
            .or_default()
            .push(rejection);
    }

    let mut selections = Vec::with_capacity(groups.len());
    for ((problem, operation), candidates) in groups {
        let mut candidate_ids = BTreeSet::new();
        if !candidates
            .iter()
            .all(|candidate| candidate_ids.insert(candidate.choice))
        {
            return Err(IrodoriError::Config(format!(
                "route problem B{} S{} contains duplicate candidate IDs",
                problem.batch(),
                problem.sequence
            )));
        }
        let rejected_candidates = rejected.remove(&(problem, operation)).unwrap_or_default();
        if !rejected_candidates
            .iter()
            .all(|candidate| candidate_ids.insert(candidate.choice))
        {
            return Err(IrodoriError::Config(format!(
                "route problem B{} S{} contains duplicate measured/rejected candidate IDs",
                problem.batch(),
                problem.sequence
            )));
        }
        let expected = operation.candidates(problem);
        if candidate_ids.len() != expected.len()
            || !expected
                .iter()
                .all(|candidate| candidate_ids.contains(candidate))
        {
            return Err(IrodoriError::Config(format!(
                "route problem B{} S{} operation {operation:?} lacks complete candidate evidence",
                problem.batch(),
                problem.sequence
            )));
        }
        let portable = candidates
            .iter()
            .find(|candidate| candidate.choice.is_portable())
            .ok_or_else(|| {
                IrodoriError::Config(format!(
                    "route problem B{} S{} is missing its portable measurement",
                    problem.batch(),
                    problem.sequence
                ))
            })?;
        let portable_median = portable
            .median_ns()
            .expect("validated measurement has a median");
        let best = candidates
            .iter()
            .filter(|candidate| candidate.accuracy_disposition() != AccuracyDisposition::Reject)
            .min_by_key(|candidate| candidate.median_ns().expect("validated median"));

        let threshold_numerator =
            u128::from(10_000 - tuning_policy.minimum_improvement_basis_points);
        let materially_faster = |candidate_ns: u64| {
            u128::from(candidate_ns) * 10_000 <= u128::from(portable_median) * threshold_numerator
        };
        let optimized_candidates = candidates
            .iter()
            .filter(|candidate| !candidate.choice.is_portable())
            .count();
        let approved_optimized = candidates
            .iter()
            .filter(|candidate| {
                !candidate.choice.is_portable()
                    && candidate.accuracy_disposition() != AccuracyDisposition::Reject
            })
            .min_by_key(|candidate| candidate.median_ns().expect("validated median"));
        let (selected, reason) = if portable.accuracy_disposition() == AccuracyDisposition::Reject {
            (
                portable,
                RouteSelectionReason::PortableBecauseBaselineFailedAccuracy,
            )
        } else if optimized_candidates == 0 && !rejected_candidates.is_empty() {
            (
                portable,
                RouteSelectionReason::PortableBecauseCandidatesUnavailable,
            )
        } else if optimized_candidates == 0 {
            (portable, RouteSelectionReason::PortableOnlyCandidate)
        } else if approved_optimized.is_none() && !rejected_candidates.is_empty() {
            (
                portable,
                RouteSelectionReason::PortableBecauseCandidatesUnavailable,
            )
        } else if approved_optimized.is_none() {
            (
                portable,
                RouteSelectionReason::PortableBecauseCandidatesFailedAccuracy,
            )
        } else {
            match best {
                Some(candidate)
                    if !candidate.choice.is_portable()
                        && materially_faster(candidate.median_ns().expect("validated median")) =>
                {
                    (candidate, RouteSelectionReason::FastestAccuracyApproved)
                }
                Some(_) => (portable, RouteSelectionReason::PortableWithinNoiseFloor),
                None => (
                    portable,
                    RouteSelectionReason::PortableBecauseCandidatesFailedAccuracy,
                ),
            }
        };
        selections.push(ApprovedRouteSelection {
            problem,
            choice: selected.choice,
            reason,
            selected_median_ns: selected.median_ns().expect("validated median"),
            portable_median_ns: portable_median,
            fresh_sessions: selected.fresh_session_medians_ns.len(),
            measured_requests_per_session: selected.measured_requests_per_session,
            accuracy: selected.accuracy_disposition(),
        });
    }

    if !rejected.is_empty() {
        return Err(IrodoriError::Config(
            "route rejection evidence has no measured portable group".to_owned(),
        ));
    }

    let manifest = ApprovedRouteManifest {
        schema_version: ROUTE_AUTOTUNE_SCHEMA_VERSION,
        route_abi: ROUTE_ABI_VERSION.to_owned(),
        identity,
        tuning_policy,
        selections,
    };
    manifest.validate()?;
    Ok(manifest)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RouteCell {
    attention_qkv_projection: ProjectionRoute,
    attention_output_projection: ProjectionRoute,
    mlp_expand: ProjectionRoute,
    mlp_contract: ProjectionRoute,
    attention_output_weight: AttentionOutputWeightRoute,
    mlp_contract_weight: MlpContractWeightRoute,
}

impl RouteCell {
    const PORTABLE: Self = Self {
        attention_qkv_projection: ProjectionRoute::DefaultGraph,
        attention_output_projection: ProjectionRoute::DefaultGraph,
        mlp_expand: ProjectionRoute::DefaultGraph,
        mlp_contract: ProjectionRoute::DefaultGraph,
        attention_output_weight: AttentionOutputWeightRoute::SourceColumnFlat,
        mlp_contract_weight: MlpContractWeightRoute::SourceColumnFlat,
    };
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedRouteTable {
    origin: RouteTableOrigin,
    cells: Box<[RouteCell]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RouteTableOrigin {
    Portable,
    ApprovedManifest,
    LegacyProduction,
    #[cfg(any(feature = "profile", test))]
    DiagnosticCandidate,
}

impl ResolvedRouteTable {
    const STRIDE: usize = MAX_TUNED_SEQUENCE + 1;
    const CELL_COUNT: usize = (MAX_TUNED_BATCH + 1) * Self::STRIDE;

    pub fn portable() -> Self {
        Self {
            origin: RouteTableOrigin::Portable,
            cells: vec![RouteCell::PORTABLE; Self::CELL_COUNT].into_boxed_slice(),
        }
    }

    /// Preserve the pre-autotune production policy when no sealed manifest is
    /// installed. This is backward compatibility, not cross-device evidence.
    pub fn production_approved() -> Self {
        let mut table = Self::portable();
        table.origin = RouteTableOrigin::LegacyProduction;
        for batch in 1..=MAX_TUNED_BATCH {
            for sequence in 1..=MAX_TUNED_SEQUENCE {
                let problem = RouteProblem::new(batch, sequence).expect("bounded route problem");
                let cell = table.cell_mut(problem);
                let b12_long = batch <= 2 && sequence >= 100;
                let b3_moderate = batch == 3 && (100..=512).contains(&sequence);
                if b12_long || b3_moderate {
                    cell.attention_qkv_projection = ProjectionRoute::HandwrittenT64;
                    cell.attention_output_projection = ProjectionRoute::HandwrittenT64;
                    cell.mlp_contract = ProjectionRoute::HandwrittenT64;
                }
                if b12_long {
                    cell.mlp_expand = ProjectionRoute::HandwrittenT64;
                }
                cell.attention_output_weight = incumbent_attention_weight(batch, sequence);
                cell.mlp_contract_weight = incumbent_mlp_weight(batch, sequence);
            }
        }
        table
    }

    #[cfg(any(feature = "profile", test))]
    pub fn extended_candidate() -> Self {
        let mut table = Self::production_approved();
        table.origin = RouteTableOrigin::DiagnosticCandidate;
        for batch in 1..=MAX_TUNED_BATCH {
            for sequence in 13..=MAX_TUNED_SEQUENCE {
                let problem = RouteProblem::new(batch, sequence).expect("bounded route problem");
                let cell = table.cell_mut(problem);
                cell.attention_qkv_projection = ProjectionRoute::HandwrittenT64;
                cell.attention_output_projection = ProjectionRoute::HandwrittenT64;
                cell.mlp_expand = ProjectionRoute::HandwrittenT64;
                cell.mlp_contract = ProjectionRoute::HandwrittenT64;
                if batch >= 2 {
                    cell.attention_output_weight = AttentionOutputWeightRoute::PackedRowFlat;
                    cell.mlp_contract_weight = MlpContractWeightRoute::PackedRowFlat;
                }
            }
        }
        table
    }

    pub fn from_manifest(
        manifest: &ApprovedRouteManifest,
        actual_identity: &RouteDeviceIdentity,
    ) -> Result<Self> {
        manifest.validate()?;
        manifest.verify_identity(actual_identity)?;
        let mut table = Self::portable();
        table.origin = RouteTableOrigin::ApprovedManifest;
        for selection in &manifest.selections {
            table.apply(selection.problem, selection.choice);
        }
        Ok(table)
    }

    pub fn attention_qkv_projection(&self, batch: usize, sequence: usize) -> ProjectionRoute {
        self.cell(batch, sequence)
            .map_or(ProjectionRoute::DefaultGraph, |cell| {
                cell.attention_qkv_projection
            })
    }

    pub fn attention_output_projection(&self, batch: usize, sequence: usize) -> ProjectionRoute {
        self.cell(batch, sequence)
            .map_or(ProjectionRoute::DefaultGraph, |cell| {
                cell.attention_output_projection
            })
    }

    pub fn mlp_expand_projection(&self, batch: usize, sequence: usize) -> ProjectionRoute {
        self.cell(batch, sequence)
            .map_or(ProjectionRoute::DefaultGraph, |cell| cell.mlp_expand)
    }

    pub fn mlp_contract(&self, batch: usize, sequence: usize) -> ProjectionRoute {
        self.cell(batch, sequence)
            .map_or(ProjectionRoute::DefaultGraph, |cell| cell.mlp_contract)
    }

    pub fn attention_output_weight(
        &self,
        batch: usize,
        sequence: usize,
    ) -> AttentionOutputWeightRoute {
        self.cell(batch, sequence)
            .map_or(AttentionOutputWeightRoute::SourceColumnFlat, |cell| {
                cell.attention_output_weight
            })
    }

    pub fn mlp_contract_weight(&self, batch: usize, sequence: usize) -> MlpContractWeightRoute {
        self.cell(batch, sequence)
            .map_or(MlpContractWeightRoute::SourceColumnFlat, |cell| {
                cell.mlp_contract_weight
            })
    }

    pub(crate) const fn permits_legacy_profile_overlay(&self) -> bool {
        matches!(self.origin, RouteTableOrigin::LegacyProduction)
    }

    fn apply(&mut self, problem: RouteProblem, choice: RouteChoice) {
        let cell = self.cell_mut(problem);
        match choice {
            RouteChoice::AttentionQkvProjection(route) => {
                cell.attention_qkv_projection = route;
            }
            RouteChoice::AttentionOutputProjection(route) => {
                cell.attention_output_projection = route;
            }
            RouteChoice::MlpExpandProjection(route) => cell.mlp_expand = route,
            RouteChoice::MlpContract(route) => cell.mlp_contract = route,
            RouteChoice::AttentionOutputWeight(route) => cell.attention_output_weight = route,
            RouteChoice::MlpContractWeight(route) => cell.mlp_contract_weight = route,
        }
    }

    fn cell(&self, batch: usize, sequence: usize) -> Option<&RouteCell> {
        if !(1..=MAX_TUNED_BATCH).contains(&batch) || !(1..=MAX_TUNED_SEQUENCE).contains(&sequence)
        {
            return None;
        }
        Some(&self.cells[batch * Self::STRIDE + sequence])
    }

    fn cell_mut(&mut self, problem: RouteProblem) -> &mut RouteCell {
        &mut self.cells[Self::index(problem)]
    }

    const fn index(problem: RouteProblem) -> usize {
        problem.batch() * Self::STRIDE + problem.sequence
    }
}

fn incumbent_attention_weight(batch: usize, sequence: usize) -> AttentionOutputWeightRoute {
    if batch == 1 && matches!(sequence, 13 | 25) {
        AttentionOutputWeightRoute::SourceColumnFlat
    } else if batch == 1 {
        AttentionOutputWeightRoute::PackedRowFlat
    } else if batch == 2 && sequence >= 200 {
        AttentionOutputWeightRoute::PackedRowRank3
    } else if batch == 2 && (sequence == 25 || sequence >= 100) {
        AttentionOutputWeightRoute::PackedRowFlat
    } else {
        AttentionOutputWeightRoute::SourceColumnFlat
    }
}

fn incumbent_mlp_weight(batch: usize, sequence: usize) -> MlpContractWeightRoute {
    match incumbent_attention_weight(batch, sequence) {
        AttentionOutputWeightRoute::SourceColumnFlat => MlpContractWeightRoute::SourceColumnFlat,
        AttentionOutputWeightRoute::PackedRowFlat => MlpContractWeightRoute::PackedRowFlat,
        AttentionOutputWeightRoute::PackedRowRank3 => MlpContractWeightRoute::PackedRowRank3,
    }
}

static ACTIVE_ROUTES: OnceLock<ResolvedRouteTable> = OnceLock::new();

pub fn active_route_table() -> &'static ResolvedRouteTable {
    ACTIVE_ROUTES.get_or_init(|| {
        #[cfg(feature = "profile")]
        if std::env::var("IRODORI_DIT_ROUTE_ENVELOPE").as_deref() == Ok("extended-candidate") {
            return ResolvedRouteTable::extended_candidate();
        }
        ResolvedRouteTable::production_approved()
    })
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RouteInstallReceipt {
    pub schema_version: u32,
    pub route_abi: String,
    pub decision: RouteInstallDecision,
}

impl Default for RouteInstallReceipt {
    fn default() -> Self {
        Self {
            schema_version: ROUTE_AUTOTUNE_SCHEMA_VERSION,
            route_abi: ROUTE_ABI_VERSION.to_owned(),
            decision: RouteInstallDecision::LegacyProduction,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum RouteInstallDecision {
    ApprovedExactDevice { selection_count: usize },
    Portable { reason: RouteCacheMissReason },
    LegacyProduction,
    ExternallyInstalled,
}

fn install_route_table(
    table: ResolvedRouteTable,
    decision: RouteInstallDecision,
) -> Result<RouteInstallReceipt> {
    if let Some(active) = ACTIVE_ROUTES.get() {
        if active != &table {
            return Err(IrodoriError::Config(
                "a different DiT route table is already active in this process".to_owned(),
            ));
        }
    } else if ACTIVE_ROUTES.set(table).is_err() {
        return Err(IrodoriError::Config(
            "failed to install the DiT route table".to_owned(),
        ));
    }
    Ok(RouteInstallReceipt {
        schema_version: ROUTE_AUTOTUNE_SCHEMA_VERSION,
        route_abi: ROUTE_ABI_VERSION.to_owned(),
        decision,
    })
}

/// Install a sealed table before model execution. A second, different table
/// is rejected; process-global route mutation while serving is not supported.
pub fn install_approved_route_manifest(
    manifest: &ApprovedRouteManifest,
    actual_identity: &RouteDeviceIdentity,
) -> Result<RouteInstallReceipt> {
    let table = ResolvedRouteTable::from_manifest(manifest, actual_identity)?;
    install_route_table(
        table,
        RouteInstallDecision::ApprovedExactDevice {
            selection_count: manifest.selections.len(),
        },
    )
}

/// Resolve a multi-device cache and install either its exact profile or the
/// portable table. A cache miss is a successful, explicit fallback.
pub fn install_route_manifest_set(
    manifest_set: &ApprovedRouteManifestSet,
    actual_identity: &RouteDeviceIdentity,
) -> Result<RouteInstallReceipt> {
    match manifest_set.resolve(actual_identity)? {
        RouteManifestResolution::Approved(manifest) => {
            install_approved_route_manifest(manifest, actual_identity)
        }
        RouteManifestResolution::Portable { reason } => install_portable_route_table(reason),
    }
}

pub fn install_portable_route_table(reason: RouteCacheMissReason) -> Result<RouteInstallReceipt> {
    install_route_table(
        ResolvedRouteTable::portable(),
        RouteInstallDecision::Portable { reason },
    )
}

/// Explicit compatibility mode for reproducing the pre-autotune policy.
/// New cross-platform applications should prefer an approved manifest set or
/// portable fallback.
pub fn install_legacy_production_route_table() -> Result<RouteInstallReceipt> {
    install_route_table(
        ResolvedRouteTable::production_approved(),
        RouteInstallDecision::LegacyProduction,
    )
}

pub fn accept_externally_installed_route_table() -> Result<RouteInstallReceipt> {
    if ACTIVE_ROUTES.get().is_none() {
        return Err(IrodoriError::Config(
            "external route policy requires a table to be installed before runtime initialization"
                .to_owned(),
        ));
    }
    Ok(RouteInstallReceipt {
        schema_version: ROUTE_AUTOTUNE_SCHEMA_VERSION,
        route_abi: ROUTE_ABI_VERSION.to_owned(),
        decision: RouteInstallDecision::ExternallyInstalled,
    })
}

fn median(mut values: Vec<u64>) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let midpoint = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[midpoint])
    } else {
        let lhs = u128::from(values[midpoint - 1]);
        let rhs = u128::from(values[midpoint]);
        u64::try_from((lhs + rhs) / 2).ok()
    }
}

fn expected_linear_schedule_bits(num_steps: usize) -> Vec<u32> {
    let steps = num_steps + 1;
    let halfway = steps / 2;
    let step = 1.0_f32 / num_steps as f32;
    (0..steps)
        .map(|index| {
            let u = if index < halfway {
                step.mul_add(index as f32, 0.0)
            } else {
                (-step).mul_add((steps - index - 1) as f32, 1.0)
            };
            ((1.0_f32 - u) * 0.999_f32).to_bits()
        })
        .collect()
}

#[cfg(test)]
fn canonical_forward_batches(problem: RouteProblem) -> Vec<usize> {
    let mut batches = vec![problem.batch(); 20];
    batches.extend([1; 20]);
    batches
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> RouteDeviceIdentity {
        RouteDeviceIdentity {
            adapter_name: "test adapter".to_owned(),
            backend: "Vulkan".to_owned(),
            device_type: "DiscreteGpu".to_owned(),
            vendor_id: 0x10de,
            device_id: 0x1234,
            driver: "test".to_owned(),
            driver_info: "1".to_owned(),
            os: "linux".to_owned(),
            platform_version: "Linux 6 test".to_owned(),
            architecture: "x86_64".to_owned(),
            precision: "fp32".to_owned(),
            allocator_policy: "exclusive_pages".to_owned(),
            compiler_policy: "wgpu_auto".to_owned(),
            application_version: env!("CARGO_PKG_VERSION").to_owned(),
            burn_version: "0.22.0-pre.2".to_owned(),
            burn_cubecl_version: "0.22.0-pre.2".to_owned(),
            cubecl_version: "0.11.0-pre.2".to_owned(),
            cubek_version: "0.3.0-pre.2".to_owned(),
            wgpu_version: "30.0.0".to_owned(),
            model_sha256: "1".repeat(64),
            codec_sha256: "2".repeat(64),
            binary_sha256: "3".repeat(64),
        }
    }

    fn metrics(snr_db: f64) -> RouteAccuracyMetrics {
        RouteAccuracyMetrics {
            max_abs: 1.0e-6,
            mean_abs: 1.0e-7,
            rmse: 1.0e-7,
            snr_db,
            cosine: 0.999_999_999,
        }
    }

    fn measurement(choice: RouteChoice, median_ns: u64, snr_db: f64) -> RouteCandidateMeasurement {
        RouteCandidateMeasurement {
            identity_sha256: identity().fingerprint_sha256().unwrap(),
            problem: RouteProblem::new(3, 489).unwrap(),
            choice,
            fresh_session_medians_ns: vec![median_ns; 5],
            measured_requests_per_session: 10,
            euler_steps: 40,
            schedule_f32_bits: expected_linear_schedule_bits(40),
            forward_batches: canonical_forward_batches(RouteProblem::new(3, 489).unwrap()),
            model_layers: 12,
            model_block_calls: 480,
            local_latent: metrics(110.0),
            final_waveform: metrics(snr_db),
        }
    }

    fn contract_measurements() -> Vec<RouteCandidateMeasurement> {
        vec![
            measurement(
                RouteChoice::MlpContract(ProjectionRoute::DefaultGraph),
                1_000,
                100.0,
            ),
            measurement(
                RouteChoice::MlpContract(ProjectionRoute::HandwrittenT64),
                1_100,
                90.0,
            ),
        ]
    }

    #[test]
    fn exact_table_defaults_uncovered_cells_to_portable() {
        let manifest = select_approved_routes(
            identity(),
            RouteTuningPolicy::default(),
            vec![
                measurement(
                    RouteChoice::MlpExpandProjection(ProjectionRoute::DefaultGraph),
                    1_000,
                    100.0,
                ),
                measurement(
                    RouteChoice::MlpExpandProjection(ProjectionRoute::HandwrittenT64),
                    800,
                    90.0,
                ),
            ],
        )
        .unwrap();
        let table = ResolvedRouteTable::from_manifest(&manifest, &identity()).unwrap();
        assert_eq!(
            table.mlp_expand_projection(3, 489),
            ProjectionRoute::HandwrittenT64
        );
        assert_eq!(
            table.mlp_expand_projection(3, 685),
            ProjectionRoute::DefaultGraph
        );
        assert_eq!(
            table.attention_output_projection(3, 489),
            ProjectionRoute::DefaultGraph
        );
    }

    #[test]
    fn accuracy_failure_keeps_portable_route() {
        let manifest = select_approved_routes(
            identity(),
            RouteTuningPolicy::default(),
            vec![
                measurement(
                    RouteChoice::MlpExpandProjection(ProjectionRoute::DefaultGraph),
                    1_000,
                    100.0,
                ),
                measurement(
                    RouteChoice::MlpExpandProjection(ProjectionRoute::HandwrittenT64),
                    700,
                    79.0,
                ),
            ],
        )
        .unwrap();
        assert_eq!(
            manifest.selections[0].choice,
            RouteChoice::MlpExpandProjection(ProjectionRoute::DefaultGraph)
        );
        assert_eq!(
            manifest.selections[0].reason,
            RouteSelectionReason::PortableBecauseCandidatesFailedAccuracy
        );
    }

    #[test]
    fn noise_floor_keeps_portable_route() {
        let manifest = select_approved_routes(
            identity(),
            RouteTuningPolicy::default(),
            vec![
                measurement(
                    RouteChoice::AttentionQkvProjection(ProjectionRoute::DefaultGraph),
                    1_000,
                    100.0,
                ),
                measurement(
                    RouteChoice::AttentionQkvProjection(ProjectionRoute::HandwrittenT64),
                    990,
                    90.0,
                ),
            ],
        )
        .unwrap();
        assert_eq!(
            manifest.selections[0].choice,
            RouteChoice::AttentionQkvProjection(ProjectionRoute::DefaultGraph)
        );
    }

    #[test]
    fn identity_change_is_a_cache_miss() {
        let manifest = select_approved_routes(
            identity(),
            RouteTuningPolicy::default(),
            contract_measurements(),
        )
        .unwrap();
        let mut changed = identity();
        changed.driver_info = "2".to_owned();
        assert!(ResolvedRouteTable::from_manifest(&manifest, &changed).is_err());
    }

    #[test]
    fn production_components_remain_independent() {
        let production = ResolvedRouteTable::production_approved();
        assert_eq!(
            production.attention_qkv_projection(3, 489),
            ProjectionRoute::HandwrittenT64
        );
        assert_eq!(
            production.mlp_contract(3, 489),
            ProjectionRoute::HandwrittenT64
        );
        assert_eq!(
            production.mlp_expand_projection(3, 489),
            ProjectionRoute::DefaultGraph
        );
        assert_eq!(
            production.attention_qkv_projection(3, 685),
            ProjectionRoute::DefaultGraph
        );
    }

    #[test]
    fn manifest_set_resolves_only_an_exact_identity() {
        let manifest = select_approved_routes(
            identity(),
            RouteTuningPolicy::default(),
            contract_measurements(),
        )
        .unwrap();
        let manifest_set = ApprovedRouteManifestSet::new(vec![manifest]).unwrap();
        assert!(matches!(
            manifest_set.resolve(&identity()).unwrap(),
            RouteManifestResolution::Approved(_)
        ));
        let mut other_generation = identity();
        other_generation.device_id += 1;
        assert_eq!(
            manifest_set.resolve(&other_generation).unwrap(),
            RouteManifestResolution::Portable {
                reason: RouteCacheMissReason::NoExactDeviceProfile
            }
        );
    }

    #[test]
    fn incomplete_driver_identity_cannot_enter_persistent_set() {
        let mut incomplete = identity();
        incomplete.driver_info.clear();
        let identity_sha256 = incomplete.fingerprint_sha256().unwrap();
        let mut measurements = contract_measurements();
        for measurement in &mut measurements {
            measurement.identity_sha256.clone_from(&identity_sha256);
        }
        let manifest =
            select_approved_routes(incomplete, RouteTuningPolicy::default(), measurements).unwrap();
        assert!(ApprovedRouteManifestSet::new(vec![manifest]).is_err());
    }

    #[test]
    fn each_component_is_selected_independently() {
        let manifest = select_approved_routes(
            identity(),
            RouteTuningPolicy::default(),
            vec![
                measurement(
                    RouteChoice::AttentionQkvProjection(ProjectionRoute::DefaultGraph),
                    1_000,
                    100.0,
                ),
                measurement(
                    RouteChoice::AttentionQkvProjection(ProjectionRoute::HandwrittenT64),
                    700,
                    90.0,
                ),
                measurement(
                    RouteChoice::MlpExpandProjection(ProjectionRoute::DefaultGraph),
                    1_000,
                    100.0,
                ),
                measurement(
                    RouteChoice::MlpExpandProjection(ProjectionRoute::HandwrittenT64),
                    1_100,
                    90.0,
                ),
            ],
        )
        .unwrap();
        let table = ResolvedRouteTable::from_manifest(&manifest, &identity()).unwrap();
        assert_eq!(
            table.attention_qkv_projection(3, 489),
            ProjectionRoute::HandwrittenT64
        );
        assert_eq!(
            table.mlp_expand_projection(3, 489),
            ProjectionRoute::DefaultGraph
        );
    }

    #[test]
    fn unavailable_candidate_requires_an_explicit_rejection() {
        let portable = measurement(
            RouteChoice::MlpExpandProjection(ProjectionRoute::DefaultGraph),
            1_000,
            100.0,
        );
        assert!(
            select_approved_routes(
                identity(),
                RouteTuningPolicy::default(),
                vec![portable.clone()],
            )
            .is_err()
        );
        let manifest = select_approved_routes_with_rejections(
            identity(),
            RouteTuningPolicy::default(),
            vec![portable],
            vec![RouteCandidateRejection {
                identity_sha256: identity().fingerprint_sha256().unwrap(),
                problem: RouteProblem::new(3, 489).unwrap(),
                choice: RouteChoice::MlpExpandProjection(ProjectionRoute::HandwrittenT64),
                reason: RouteCandidateRejectionReason::OutOfMemory,
                detail: "driver reported an allocation failure".to_owned(),
            }],
        )
        .unwrap();
        assert_eq!(
            manifest.selections[0].reason,
            RouteSelectionReason::PortableBecauseCandidatesUnavailable
        );
    }

    #[test]
    fn generic_runner_enumerates_every_candidate_and_selects_winner() {
        struct Runner {
            choices: Vec<RouteChoice>,
        }
        impl RouteCandidateRunner for Runner {
            fn run_candidate(
                &mut self,
                request: RouteCandidateRequest,
            ) -> Result<RouteCandidateRun> {
                self.choices.push(request.choice);
                let median = if request.choice.is_portable() {
                    1_000
                } else {
                    700
                };
                Ok(RouteCandidateRun::Measured {
                    fresh_session_medians_ns: vec![median; 5],
                    measured_requests_per_session: 10,
                    euler_steps: 40,
                    schedule_f32_bits: expected_linear_schedule_bits(40),
                    forward_batches: canonical_forward_batches(request.problem),
                    model_layers: 12,
                    model_block_calls: 480,
                    local_latent: metrics(110.0),
                    final_waveform: metrics(90.0),
                })
            }
        }

        let mut runner = Runner {
            choices: Vec::new(),
        };
        let manifest = autotune_routes(
            identity(),
            RouteTuningPolicy::default(),
            &RouteTuningWorkload {
                cases: vec![RouteTuningCase {
                    problem: RouteProblem::new(3, 489).unwrap(),
                    operations: vec![RouteOperation::AttentionOutputWeight],
                }],
            },
            &mut runner,
        )
        .unwrap();
        assert_eq!(runner.choices.len(), 3);
        assert_eq!(
            manifest.selections[0].choice,
            RouteChoice::AttentionOutputWeight(AttentionOutputWeightRoute::PackedRowFlat)
        );
    }
}
