//! Fresh-process, 40-step route tuner for the v4 WGPU runtime.
//!
//! The generic selector lives in [`crate::route_autotune`]. This module is the
//! concrete process boundary: every candidate is installed before WGPU
//! initialization in a child `bench_v4_residency` process, accuracy is checked
//! against explicit oracle tensors, and only measured request medians are
//! returned to the selector.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    ApprovedRouteManifest, BuiltInRouteProfile, IrodoriError, Result, RouteAccuracyMetrics,
    RouteCandidateRejectionReason, RouteCandidateRequest, RouteCandidateRun, RouteCandidateRunner,
    RouteChoice, RouteOperation, RouteOverride, RouteProblem, RouteTuningCase, RouteTuningWorkload,
    UnsealedRouteProfile, sha256_file,
};

const PRODUCT_STEPS: usize = 40;
const PRODUCT_LAYERS: usize = 12;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TuningVoice {
    TextOnly,
    Designed,
    PreparedClone,
}

impl TuningVoice {
    const fn guided_batch(self) -> usize {
        match self {
            Self::TextOnly => 2,
            Self::Designed | Self::PreparedClone => 3,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FreshProcessTuningCase {
    pub problem: RouteProblem,
    pub operations: Vec<RouteOperation>,
    pub fixture: PathBuf,
    pub references: [PathBuf; 2],
    pub voice: TuningVoice,
    /// Canonical 40-step patched RF output as little-endian f32.
    pub oracle_patched_f32le: PathBuf,
    /// Canonical decoded waveform as little-endian f32.
    pub oracle_waveform_f32le: PathBuf,
}

impl FreshProcessTuningCase {
    fn validate(&self) -> Result<()> {
        RouteProblem::new(self.problem.batch(), self.problem.sequence)?;
        if self.operations.is_empty() {
            return Err(IrodoriError::Config(
                "fresh-process tuning case has no route operations".to_owned(),
            ));
        }
        if self.problem.batch() != 1 && self.problem.batch() != self.voice.guided_batch() {
            return Err(IrodoriError::Config(format!(
                "B{} is not exercised by {:?}; use B1 or the voice's guided batch",
                self.problem.batch(),
                self.voice
            )));
        }
        for path in [
            self.fixture.as_path(),
            self.references[0].as_path(),
            self.references[1].as_path(),
            self.oracle_patched_f32le.as_path(),
            self.oracle_waveform_f32le.as_path(),
        ] {
            if !path.is_file() {
                return Err(IrodoriError::Config(format!(
                    "route tuning input is not a file: {}",
                    path.display()
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FreshProcessTuningWorkload {
    pub schema_version: u32,
    pub cases: Vec<FreshProcessTuningCase>,
}

impl FreshProcessTuningWorkload {
    pub fn load(path: &Path) -> Result<Self> {
        let workload: Self = serde_json::from_slice(&fs::read(path)?)?;
        workload.validate()?;
        Ok(workload)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 || self.cases.is_empty() {
            return Err(IrodoriError::Config(
                "fresh-process tuning workload must use schema 1 and contain cases".to_owned(),
            ));
        }
        self.cases
            .iter()
            .try_for_each(FreshProcessTuningCase::validate)?;
        let mut problems = std::collections::BTreeSet::new();
        if !self.cases.iter().all(|case| problems.insert(case.problem)) {
            return Err(IrodoriError::Config(
                "fresh-process tuning workload must use one case per exact B/S problem".to_owned(),
            ));
        }
        self.route_workload().validate()
    }

    pub fn route_workload(&self) -> RouteTuningWorkload {
        RouteTuningWorkload {
            cases: self
                .cases
                .iter()
                .map(|case| RouteTuningCase {
                    problem: case.problem,
                    operations: case.operations.clone(),
                })
                .collect(),
        }
    }

    fn case(&self, problem: RouteProblem) -> Option<&FreshProcessTuningCase> {
        self.cases.iter().find(|case| case.problem == problem)
    }
}

#[derive(Clone, Debug)]
pub struct FreshProcessRouteTunerConfig {
    pub benchmark_binary: PathBuf,
    pub checkpoint: PathBuf,
    pub codec_weights: PathBuf,
    pub output_directory: PathBuf,
    pub cubecl_cache_directory: PathBuf,
    pub adapter_index: usize,
    pub base_profile: BuiltInRouteProfile,
    pub fresh_sessions: usize,
    pub warmups: usize,
    pub measured_requests: usize,
}

impl FreshProcessRouteTunerConfig {
    fn validate(&self) -> Result<()> {
        for path in [
            self.benchmark_binary.as_path(),
            self.checkpoint.as_path(),
            self.codec_weights.as_path(),
        ] {
            if !path.is_file() {
                return Err(IrodoriError::Config(format!(
                    "route tuner executable/input is not a file: {}",
                    path.display()
                )));
            }
        }
        if self.fresh_sessions == 0 || self.measured_requests == 0 {
            return Err(IrodoriError::Config(
                "route tuner requires at least one fresh session and measured request".to_owned(),
            ));
        }
        if self.output_directory.exists() {
            return Err(IrodoriError::Config(format!(
                "route tuner output must be fresh: {}",
                self.output_directory.display()
            )));
        }
        Ok(())
    }
}

pub struct FreshProcessRouteTuner {
    configuration: FreshProcessRouteTunerConfig,
    workload: FreshProcessTuningWorkload,
    oracle_cache: BTreeMap<RouteProblem, (Vec<f32>, Vec<f32>)>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ComposedRouteValidation {
    pub problem: RouteProblem,
    pub local_latent: RouteAccuracyMetrics,
    pub final_waveform: RouteAccuracyMetrics,
    pub latent_oracle_sha256: String,
    pub waveform_oracle_sha256: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case", tag = "status")]
enum CandidateOutcomeReceipt {
    Measured {
        problem: RouteProblem,
        choice: RouteChoice,
        fresh_session_medians_ns: Vec<u64>,
        measured_requests_per_session: usize,
        local_latent: RouteAccuracyMetrics,
        final_waveform: RouteAccuracyMetrics,
    },
    Rejected {
        problem: RouteProblem,
        choice: RouteChoice,
        reason: RouteCandidateRejectionReason,
        detail: String,
    },
}

impl FreshProcessRouteTuner {
    pub fn new(
        configuration: FreshProcessRouteTunerConfig,
        workload: FreshProcessTuningWorkload,
    ) -> Result<Self> {
        configuration.validate()?;
        workload.validate()?;
        fs::create_dir_all(&configuration.output_directory)?;
        fs::create_dir_all(&configuration.cubecl_cache_directory)?;
        Ok(Self {
            configuration,
            workload,
            oracle_cache: BTreeMap::new(),
        })
    }

    fn oracle(&mut self, case: &FreshProcessTuningCase) -> Result<&(Vec<f32>, Vec<f32>)> {
        if let std::collections::btree_map::Entry::Vacant(entry) =
            self.oracle_cache.entry(case.problem)
        {
            let latent = read_f32le(&case.oracle_patched_f32le)?;
            let waveform = read_f32le(&case.oracle_waveform_f32le)?;
            if latent.is_empty() || waveform.is_empty() {
                return Err(IrodoriError::Config(
                    "route tuning oracle tensors must not be empty".to_owned(),
                ));
            }
            entry.insert((latent, waveform));
        }
        self.oracle_cache
            .get(&case.problem)
            .ok_or_else(|| IrodoriError::Config("failed to retain route tuning oracle".to_owned()))
    }

    fn candidate_directory(&self, request: RouteCandidateRequest) -> Result<PathBuf> {
        let encoded = serde_json::to_vec(&request.choice)?;
        let digest = format!("{:x}", Sha256::digest(encoded));
        Ok(self.configuration.output_directory.join(format!(
            "b{}-s{}-{:?}-{}",
            request.problem.batch(),
            request.problem.sequence,
            request.choice.operation(),
            &digest[..12]
        )))
    }

    fn run_process(
        &self,
        case: &FreshProcessTuningCase,
        route_profile: &Path,
        directory: &Path,
        requests: usize,
        warmups: usize,
        diagnostic: bool,
    ) -> std::io::Result<Output> {
        fs::create_dir_all(directory)?;
        let output_json = directory.join("result.json");
        let mut command = Command::new(&self.configuration.benchmark_binary);
        command
            .arg("--mode")
            .arg("all-resident")
            .arg("--checkpoint")
            .arg(&self.configuration.checkpoint)
            .arg("--codec-weights")
            .arg(&self.configuration.codec_weights)
            .arg("--fixture")
            .arg(&case.fixture)
            .arg("--reference")
            .arg(&case.references[0])
            .arg(&case.references[1])
            .arg("--requests")
            .arg(requests.to_string())
            .arg("--warmups")
            .arg(warmups.to_string())
            .arg("--num-steps")
            .arg(PRODUCT_STEPS.to_string())
            .arg("--cfg-caption")
            .arg("4")
            .arg("--speaker-mode")
            .arg("same")
            .arg("--length-mode")
            .arg("same")
            .arg("--adapter-index")
            .arg(self.configuration.adapter_index.to_string())
            .arg("--precision")
            .arg("fp32")
            .arg("--allocator")
            .arg("exclusive-pages")
            .arg("--codec-residency")
            .arg("decode-only")
            .arg("--load-strategy")
            .arg("parallel")
            .arg("--rf-checkpoint-loader")
            .arg("indexed-file")
            .arg("--rf-weight-residency")
            .arg("tuning-candidates")
            .arg("--cubecl-cache-dir")
            .arg(&self.configuration.cubecl_cache_directory)
            .arg("--route-profile")
            .arg(route_profile)
            .arg("--output-json")
            .arg(&output_json);
        // A typed candidate profile is the sole route authority for this
        // child. Remove legacy A/B and profiling switches inherited from the
        // caller; otherwise an environment variable could silently turn the
        // requested candidate into its fallback or add synchronization.
        for variable in [
            "IRODORI_DIT_ROUTE_ENVELOPE",
            "IRODORI_DISABLE_B3_ATTENTION_MATERIALIZATION",
            "IRODORI_DISABLE_B3_CUBEK_SWIGLU",
            "IRODORI_DISABLE_DIT_PROJECTION",
            "IRODORI_DISABLE_DIT_ATTENTION_QKV",
            "IRODORI_DISABLE_DIT_ATTENTION_OUTPUT",
            "IRODORI_DISABLE_DIT_MLP_EXPAND",
            "IRODORI_DISABLE_DIT_MLP_CONTRACT",
            "IRODORI_DISABLE_PROJECTION_SWIGLU_EPILOGUE",
            "IRODORI_RF_DETAIL_PROFILE",
            "IRODORI_RF_STAGE_PROFILE",
        ] {
            command.env_remove(variable);
        }
        match case.voice {
            TuningVoice::TextOnly => {
                command.arg("--unconditioned");
            }
            TuningVoice::Designed => {
                command.arg("--designed");
            }
            TuningVoice::PreparedClone => {}
        }
        if diagnostic {
            command
                .arg("--diagnostic-output-dir")
                .arg(directory.join("diagnostic"))
                .arg("--audio-output-dir")
                .arg(directory.join("audio"));
        }
        let output = command.output()?;
        fs::write(directory.join("stdout.log"), &output.stdout)?;
        fs::write(directory.join("stderr.log"), &output.stderr)?;
        Ok(output)
    }

    fn rejection_from_output(output: &Output) -> (RouteCandidateRejectionReason, String) {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let lowercase = stderr.to_ascii_lowercase();
        let reason = if lowercase.contains("out of memory") || lowercase.contains("oom") {
            RouteCandidateRejectionReason::OutOfMemory
        } else if lowercase.contains("compil") || lowercase.contains("shader") {
            RouteCandidateRejectionReason::CompilationFailure
        } else {
            RouteCandidateRejectionReason::LaunchFailure
        };
        let mut detail = stderr.trim().to_owned();
        if detail.len() > 4_096 {
            detail.truncate(4_096);
        }
        if detail.is_empty() {
            detail = format!("benchmark child exited with {}", output.status);
        }
        (reason, detail)
    }

    fn reject_candidate(
        directory: &Path,
        request: RouteCandidateRequest,
        output: &Output,
    ) -> Result<RouteCandidateRun> {
        let (reason, detail) = Self::rejection_from_output(output);
        fs::write(
            directory.join("candidate-outcome.json"),
            serde_json::to_vec_pretty(&CandidateOutcomeReceipt::Rejected {
                problem: request.problem,
                choice: request.choice,
                reason,
                detail: detail.clone(),
            })?,
        )?;
        Ok(RouteCandidateRun::Rejected { reason, detail })
    }

    fn reject_candidate_with_reason(
        directory: &Path,
        request: RouteCandidateRequest,
        reason: RouteCandidateRejectionReason,
        detail: String,
    ) -> Result<RouteCandidateRun> {
        fs::write(
            directory.join("candidate-outcome.json"),
            serde_json::to_vec_pretty(&CandidateOutcomeReceipt::Rejected {
                problem: request.problem,
                choice: request.choice,
                reason,
                detail: detail.clone(),
            })?,
        )?;
        Ok(RouteCandidateRun::Rejected { reason, detail })
    }

    /// Validate the fully composed selection vector, not only isolated
    /// candidates. This catches interaction and accumulated 40-step drift
    /// before an exact manifest is written to the persistent set.
    pub fn validate_composed_manifest(
        &mut self,
        manifest: &ApprovedRouteManifest,
    ) -> Result<Vec<ComposedRouteValidation>> {
        manifest.validate()?;
        let profile = UnsealedRouteProfile {
            schema_version: crate::route_autotune::ROUTE_AUTOTUNE_SCHEMA_VERSION,
            route_abi: crate::route_autotune::ROUTE_ABI_VERSION.to_owned(),
            base: manifest.base_profile,
            overrides: manifest
                .selections
                .iter()
                .map(|selection| RouteOverride {
                    problem: selection.problem,
                    choice: selection.choice,
                })
                .collect(),
        };
        profile.validate()?;
        let directory = self
            .configuration
            .output_directory
            .join("composed-validation");
        fs::create_dir(&directory)?;
        let profile_path = directory.join("route-profile.json");
        fs::write(&profile_path, serde_json::to_vec_pretty(&profile)?)?;
        let profile_sha256 = sha256_file(&profile_path)?;

        let mut receipts = Vec::with_capacity(self.workload.cases.len());
        for case in self.workload.cases.clone() {
            let case_directory = directory.join(format!(
                "b{}-s{}",
                case.problem.batch(),
                case.problem.sequence
            ));
            let output = self.run_process(&case, &profile_path, &case_directory, 1, 0, true)?;
            if !output.status.success() {
                let (reason, detail) = Self::rejection_from_output(&output);
                return Err(IrodoriError::Config(format!(
                    "composed route validation failed ({reason:?}): {detail}"
                )));
            }
            let report: BenchReport =
                serde_json::from_slice(&fs::read(case_directory.join("result.json"))?)?;
            report.validate(PRODUCT_STEPS, 0, 1, case.problem, &profile_sha256)?;
            let actual_latent = read_f32le(
                &case_directory
                    .join("diagnostic")
                    .join("rf_final_patched.f32le"),
            )?;
            let actual_waveform =
                read_f32le(&case_directory.join("audio").join("request-01.f32le"))?;
            let (oracle_latent, oracle_waveform) = self.oracle(&case)?;
            let local_latent = RouteAccuracyMetrics::compare(oracle_latent, &actual_latent)?;
            let final_waveform = RouteAccuracyMetrics::compare(oracle_waveform, &actual_waveform)?;
            if !local_latent.passes_latent_hard_gate()
                || !final_waveform.passes_waveform_hard_gate()
            {
                return Err(IrodoriError::Config(format!(
                    "composed route selection failed accuracy at B{} S{}: latent={local_latent:?} waveform={final_waveform:?}",
                    case.problem.batch(),
                    case.problem.sequence
                )));
            }
            receipts.push(ComposedRouteValidation {
                problem: case.problem,
                local_latent,
                final_waveform,
                latent_oracle_sha256: sha256_file(&case.oracle_patched_f32le)?,
                waveform_oracle_sha256: sha256_file(&case.oracle_waveform_f32le)?,
            });
        }
        Ok(receipts)
    }
}

impl RouteCandidateRunner for FreshProcessRouteTuner {
    fn run_candidate(&mut self, request: RouteCandidateRequest) -> Result<RouteCandidateRun> {
        if self.configuration.fresh_sessions < request.policy.minimum_fresh_sessions
            || self.configuration.measured_requests
                < request.policy.minimum_measured_requests_per_session
        {
            return Err(IrodoriError::Config(
                "fresh-process runner is under-sampled for the requested tuning policy".to_owned(),
            ));
        }
        let case = self
            .workload
            .case(request.problem)
            .cloned()
            .ok_or_else(|| IrodoriError::Config("route tuning case disappeared".to_owned()))?;
        if !case.operations.contains(&request.choice.operation()) {
            return Err(IrodoriError::Config(
                "route candidate operation is outside its workload case".to_owned(),
            ));
        }
        let directory = self.candidate_directory(request)?;
        fs::create_dir(&directory)?;
        let route_profile = UnsealedRouteProfile::candidate(
            self.configuration.base_profile,
            request.problem,
            request.choice,
        );
        let route_profile_path = directory.join("route-profile.json");
        fs::write(
            &route_profile_path,
            serde_json::to_vec_pretty(&route_profile)?,
        )?;
        let route_profile_sha256 = sha256_file(&route_profile_path)?;

        let accuracy_directory = directory.join("accuracy");
        let accuracy_output =
            self.run_process(&case, &route_profile_path, &accuracy_directory, 1, 0, true)?;
        if !accuracy_output.status.success() {
            return Self::reject_candidate(&directory, request, &accuracy_output);
        }
        let accuracy_report: BenchReport =
            serde_json::from_slice(&fs::read(accuracy_directory.join("result.json"))?)?;
        accuracy_report.validate(PRODUCT_STEPS, 0, 1, request.problem, &route_profile_sha256)?;
        let actual_latent = read_f32le(
            &accuracy_directory
                .join("diagnostic")
                .join("rf_final_patched.f32le"),
        )?;
        let actual_waveform =
            read_f32le(&accuracy_directory.join("audio").join("request-01.f32le"))?;
        let (oracle_latent, oracle_waveform) = self.oracle(&case)?;
        let local_latent = RouteAccuracyMetrics::compare(oracle_latent, &actual_latent)?;
        let final_waveform = RouteAccuracyMetrics::compare(oracle_waveform, &actual_waveform)?;

        let mut fresh_session_medians_ns = Vec::with_capacity(self.configuration.fresh_sessions);
        for session in 1..=self.configuration.fresh_sessions {
            let session_directory = directory.join(format!("session-{session:02}"));
            let requests = self
                .configuration
                .warmups
                .checked_add(self.configuration.measured_requests)
                .ok_or_else(|| IrodoriError::Config("route request count overflow".to_owned()))?;
            let output = self.run_process(
                &case,
                &route_profile_path,
                &session_directory,
                requests,
                self.configuration.warmups,
                false,
            )?;
            if !output.status.success() {
                return Self::reject_candidate(&directory, request, &output);
            }
            let report: BenchReport =
                serde_json::from_slice(&fs::read(session_directory.join("result.json"))?)?;
            report.validate(
                PRODUCT_STEPS,
                self.configuration.warmups,
                self.configuration.measured_requests,
                request.problem,
                &route_profile_sha256,
            )?;
            if !report.audio_is_deterministic() {
                return Self::reject_candidate_with_reason(
                    &directory,
                    request,
                    RouteCandidateRejectionReason::NonDeterministicOutput,
                    format!(
                        "same-fixture audio hashes differed within fresh session {session}: {:?}",
                        report
                            .resident_request_timings
                            .iter()
                            .map(|timing| timing.audio_f32_sha256.as_str())
                            .collect::<Vec<_>>()
                    ),
                );
            }
            let mut timings = report
                .resident_request_timings
                .into_iter()
                .filter(|timing| !timing.warmup)
                .map(|timing| seconds_to_ns(timing.rf_device_complete_seconds))
                .collect::<Result<Vec<_>>>()?;
            timings.sort_unstable();
            let median = median_ns(&timings)?;
            fresh_session_medians_ns.push(median);
        }

        let outcome = CandidateOutcomeReceipt::Measured {
            problem: request.problem,
            choice: request.choice,
            fresh_session_medians_ns: fresh_session_medians_ns.clone(),
            measured_requests_per_session: self.configuration.measured_requests,
            local_latent,
            final_waveform,
        };
        fs::write(
            directory.join("candidate-outcome.json"),
            serde_json::to_vec_pretty(&outcome)?,
        )?;
        Ok(RouteCandidateRun::Measured {
            fresh_session_medians_ns,
            measured_requests_per_session: self.configuration.measured_requests,
            euler_steps: PRODUCT_STEPS,
            schedule_f32_bits: accuracy_report.schedule_f32_bits(),
            forward_batches: accuracy_report.forward_batches,
            model_layers: PRODUCT_LAYERS,
            model_block_calls: PRODUCT_STEPS * PRODUCT_LAYERS,
            local_latent,
            final_waveform,
        })
    }
}

#[derive(Debug, Deserialize)]
struct BenchReport {
    schema_version: u32,
    latency_results_valid: bool,
    strict_fp32: bool,
    autocast: bool,
    tf32: bool,
    euler_evaluations: usize,
    forward_batches: Vec<usize>,
    layers: usize,
    block_calls: usize,
    warmups: usize,
    measured: usize,
    resident_request_timings: Vec<BenchRequestTiming>,
    work_report: Option<BenchWorkReport>,
    route_profile_sha256: Option<String>,
}

impl BenchReport {
    fn validate(
        &self,
        steps: usize,
        warmups: usize,
        measured: usize,
        problem: RouteProblem,
        expected_route_profile_sha256: &str,
    ) -> Result<()> {
        let work_report = self.work_report.as_ref().ok_or_else(|| {
            IrodoriError::Config("same-length tuning report lacks work manifest".to_owned())
        })?;
        if self.schema_version < 10
            || !self.strict_fp32
            || self.autocast
            || self.tf32
            || self.euler_evaluations != steps
            || self.layers != PRODUCT_LAYERS
            || self.block_calls != steps * PRODUCT_LAYERS
            || self.warmups != warmups
            || self.measured != measured
            || self.resident_request_timings.len() != warmups + measured
            || work_report.num_steps != steps
            || work_report.schedule_f32_bits.len() != steps + 1
            || work_report.model_layers != PRODUCT_LAYERS
            || work_report.model_block_calls != steps * PRODUCT_LAYERS
            || self.forward_batches.len() != steps
            || !self.forward_batches.contains(&problem.batch())
            || self.route_profile_sha256.as_deref() != Some(expected_route_profile_sha256)
        {
            return Err(IrodoriError::Config(
                "fresh-process route benchmark violated the 40-step work/timing contract"
                    .to_owned(),
            ));
        }
        if measured > 1 && !self.latency_results_valid {
            return Err(IrodoriError::Config(
                "performance route report marked its latency invalid".to_owned(),
            ));
        }
        Ok(())
    }

    fn schedule_f32_bits(&self) -> Vec<u32> {
        self.work_report
            .as_ref()
            .map(|report| report.schedule_f32_bits.clone())
            .unwrap_or_default()
    }

    fn audio_is_deterministic(&self) -> bool {
        let Some(first) = self.resident_request_timings.first() else {
            return false;
        };
        !first.audio_f32_sha256.is_empty()
            && self
                .resident_request_timings
                .iter()
                .all(|timing| timing.audio_f32_sha256 == first.audio_f32_sha256)
    }
}

#[derive(Debug, Deserialize)]
struct BenchWorkReport {
    num_steps: usize,
    schedule_f32_bits: Vec<u32>,
    model_layers: usize,
    model_block_calls: usize,
}

#[derive(Debug, Deserialize)]
struct BenchRequestTiming {
    warmup: bool,
    rf_device_complete_seconds: f64,
    audio_f32_sha256: String,
}

fn seconds_to_ns(seconds: f64) -> Result<u64> {
    if !seconds.is_finite() || seconds <= 0.0 || seconds > u64::MAX as f64 / 1_000_000_000.0 {
        return Err(IrodoriError::Config(
            "invalid route device-complete timing".to_owned(),
        ));
    }
    Ok((seconds * 1_000_000_000.0).round() as u64)
}

fn median_ns(sorted: &[u64]) -> Result<u64> {
    let middle = sorted.len() / 2;
    match sorted {
        [] => Err(IrodoriError::Config(
            "missing measured RF timings".to_owned(),
        )),
        values if values.len() % 2 == 1 => Ok(values[middle]),
        values => Ok(
            ((u128::from(values[middle - 1]) + u128::from(values[middle])) / 2)
                .try_into()
                .map_err(|_| IrodoriError::Config("RF timing median overflow".to_owned()))?,
        ),
    }
}

fn read_f32le(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return Err(IrodoriError::Config(format!(
            "invalid f32le tensor file: {}",
            path.display()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

pub fn campaign_sha256s(root: &Path) -> Result<Vec<(PathBuf, String)>> {
    fn visit(root: &Path, current: &Path, output: &mut Vec<(PathBuf, String)>) -> Result<()> {
        for entry in fs::read_dir(current)? {
            let path = entry?.path();
            if path.is_dir() {
                visit(root, &path, output)?;
            } else if path.file_name().is_some_and(|name| name != "SHA256SUMS") {
                let relative = path
                    .strip_prefix(root)
                    .map_err(|error| IrodoriError::Config(error.to_string()))?
                    .to_owned();
                output.push((relative, sha256_file(&path)?));
            }
        }
        Ok(())
    }
    let mut output = Vec::new();
    visit(root, root, &mut output)?;
    output.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_topology_rejects_unexercised_batch() {
        assert_eq!(TuningVoice::TextOnly.guided_batch(), 2);
        assert_eq!(TuningVoice::Designed.guided_batch(), 3);
    }

    #[test]
    fn seconds_are_converted_to_integral_nanoseconds() {
        assert_eq!(seconds_to_ns(0.123_456_789).unwrap(), 123_456_789);
        assert!(seconds_to_ns(f64::NAN).is_err());
        assert!(seconds_to_ns(0.0).is_err());
        assert_eq!(median_ns(&[10, 20, 30, 40]).unwrap(), 25);
    }

    #[test]
    fn same_fixture_requires_one_audio_hash_per_process() {
        let report = |hashes: &[&str]| BenchReport {
            schema_version: 13,
            latency_results_valid: true,
            strict_fp32: true,
            autocast: false,
            tf32: false,
            euler_evaluations: PRODUCT_STEPS,
            forward_batches: vec![3; PRODUCT_STEPS],
            layers: PRODUCT_LAYERS,
            block_calls: PRODUCT_STEPS * PRODUCT_LAYERS,
            warmups: 0,
            measured: hashes.len(),
            resident_request_timings: hashes
                .iter()
                .map(|hash| BenchRequestTiming {
                    warmup: false,
                    rf_device_complete_seconds: 1.0,
                    audio_f32_sha256: (*hash).to_owned(),
                })
                .collect(),
            work_report: None,
            route_profile_sha256: None,
        };
        assert!(report(&["same", "same"]).audio_is_deterministic());
        assert!(!report(&["first", "second"]).audio_is_deterministic());
        assert!(!report(&[]).audio_is_deterministic());
    }
}
