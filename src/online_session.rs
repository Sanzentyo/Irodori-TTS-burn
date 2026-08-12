//! Long-lived all-resident WGPU session with startup-only compile warmup.

use std::{
    collections::HashSet,
    marker::PhantomData,
    path::{Path, PathBuf},
    time::Instant,
};

use burn::tensor::{Bool, Device, Tensor, TensorData};
use serde::{Deserialize, Serialize};

use crate::{
    InferenceBuilder, IrodoriError, Result, SamplerParams, SamplingRequest, WgslInferenceEngine,
    WgslWeightProfile,
    codec::{DacVaeDecoder, load_decoder},
    model::{AuxConditionInput, unpatchify_latent},
    rf::PreparedSamplingRequest,
};

/// The session owns resident models but has not completed startup validation.
#[derive(Debug)]
pub struct Unwarmed;

/// Compile warmup and real validation have completed successfully.
#[derive(Debug)]
pub struct SessionReady;

/// Duration-model residency selected while constructing an online session.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DurationModelResidency {
    Predictive,
    ExactGeometryOnly,
}

/// Host wall timings for parallel RF/codec session construction.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SessionLoadReport {
    pub wall_seconds: f64,
    pub rf_checkpoint_seconds: f64,
    pub codec_checkpoint_seconds: f64,
    pub rf_profile_preparation_seconds: f64,
}

/// Host-visible conditioning topology used to validate compile-only warmup.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WarmupTopology {
    TextOnly,
    Designed,
    PreparedClone,
    DesignedAndClone,
}

/// Whether the startup contract includes learned duration prediction.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DurationWarmupPolicy {
    /// A predictive session must validate at least one duration request.
    #[default]
    Required,
    /// Every request supplies an exact output geometry and no duration head is resident.
    ExactGeometryOnly,
}

impl WarmupTopology {
    fn from_request(request: &PreparedSamplingRequest) -> Self {
        match (request.has_speaker_context(), request.has_caption_context()) {
            (false, false) => Self::TextOnly,
            (false, true) => Self::Designed,
            (true, false) => Self::PreparedClone,
            (true, true) => Self::DesignedAndClone,
        }
    }

    fn matches(self, request: &PreparedSamplingRequest) -> bool {
        self == Self::from_request(request)
    }
}

/// One shape/topology entry in the warmup manifest.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WarmupCaseSpec {
    pub latent_frames: usize,
    pub topology: WarmupTopology,
    /// Execute and read back this case after the compile-only pass.
    pub real_validation: bool,
    /// Run learned duration prediction using this case's explicit feature vector.
    #[serde(default)]
    pub duration_validation: bool,
}

/// Serializable manifest separated from its runtime tensor inputs.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WarmupManifest {
    pub schema_version: u32,
    pub duration_policy: DurationWarmupPolicy,
    pub cases: Vec<WarmupCaseSpec>,
}

impl WarmupManifest {
    pub const SCHEMA_VERSION: u32 = 1;

    pub fn new(cases: Vec<WarmupCaseSpec>) -> Result<Self> {
        Self::new_with_duration_policy(cases, DurationWarmupPolicy::Required)
    }

    pub fn new_with_duration_policy(
        cases: Vec<WarmupCaseSpec>,
        duration_policy: DurationWarmupPolicy,
    ) -> Result<Self> {
        if cases.is_empty() {
            return Err(IrodoriError::Config(
                "warmup manifest must contain at least one case".to_owned(),
            ));
        }
        if cases.iter().any(|case| case.latent_frames == 0) {
            return Err(IrodoriError::Config(
                "warmup latent frames must be greater than zero".to_owned(),
            ));
        }

        let mut unique = HashSet::with_capacity(cases.len());
        for case in &cases {
            if !unique.insert((case.latent_frames, case.topology)) {
                return Err(IrodoriError::Config(format!(
                    "duplicate warmup case: {} frames / {:?}",
                    case.latent_frames, case.topology
                )));
            }
        }

        let represented = cases
            .iter()
            .map(|case| case.topology)
            .collect::<HashSet<_>>();
        let validated = cases
            .iter()
            .filter(|case| case.real_validation)
            .map(|case| case.topology)
            .collect::<HashSet<_>>();
        if let Some(missing) = represented.difference(&validated).next() {
            return Err(IrodoriError::Config(format!(
                "warmup topology {missing:?} lacks a real validation case"
            )));
        }

        let duration_cases = cases
            .iter()
            .filter(|case| case.duration_validation)
            .collect::<Vec<_>>();
        if duration_cases.iter().any(|case| !case.real_validation) {
            return Err(IrodoriError::Config(
                "duration validation must also be a real validation case".to_owned(),
            ));
        }
        match duration_policy {
            DurationWarmupPolicy::Required if duration_cases.is_empty() => {
                return Err(IrodoriError::Config(
                    "predictive warmup requires a duration validation case".to_owned(),
                ));
            }
            DurationWarmupPolicy::ExactGeometryOnly if !duration_cases.is_empty() => {
                return Err(IrodoriError::Config(
                    "exact-geometry warmup cannot contain duration validation".to_owned(),
                ));
            }
            _ => {}
        }

        Ok(Self {
            schema_version: Self::SCHEMA_VERSION,
            duration_policy,
            cases,
        })
    }

    /// Required v4 RF/codec shape classes, plus one real case per online voice topology.
    pub fn v4_service() -> Self {
        let mut cases = [45, 112, 255, 333, 489, 685]
            .into_iter()
            .map(|latent_frames| WarmupCaseSpec {
                latent_frames,
                topology: WarmupTopology::TextOnly,
                real_validation: latent_frames == 112,
                duration_validation: latent_frames == 112,
            })
            .collect::<Vec<_>>();
        cases.extend([
            WarmupCaseSpec {
                latent_frames: 112,
                topology: WarmupTopology::Designed,
                real_validation: true,
                duration_validation: false,
            },
            WarmupCaseSpec {
                latent_frames: 112,
                topology: WarmupTopology::PreparedClone,
                real_validation: true,
                duration_validation: false,
            },
        ]);
        Self::new(cases).expect("built-in v4 warmup manifest is valid")
    }
}

struct WarmupCase {
    spec: WarmupCaseSpec,
    request: PreparedSamplingRequest,
    duration_features: Option<Tensor<2>>,
}

/// Runtime tensor inputs paired with one manifest entry.
pub struct WarmupInput {
    pub request: SamplingRequest,
    pub duration_features: Option<Tensor<2>>,
}

impl WarmupInput {
    pub fn exact_geometry(request: SamplingRequest) -> Self {
        Self {
            request,
            duration_features: None,
        }
    }

    pub fn predictive(request: SamplingRequest, duration_features: Tensor<2>) -> Self {
        Self {
            request,
            duration_features: Some(duration_features),
        }
    }
}

/// Validated runtime inputs paired one-to-one with a manifest.
pub struct WarmupPlan {
    manifest: WarmupManifest,
    cases: Vec<WarmupCase>,
}

impl WarmupPlan {
    pub fn prepare(
        engine: &WgslInferenceEngine,
        manifest: WarmupManifest,
        inputs: Vec<WarmupInput>,
    ) -> Result<Self> {
        if manifest.schema_version != WarmupManifest::SCHEMA_VERSION {
            return Err(IrodoriError::Config(format!(
                "unsupported warmup manifest schema {}",
                manifest.schema_version
            )));
        }
        if manifest.cases.len() != inputs.len() {
            return Err(IrodoriError::Config(format!(
                "warmup manifest has {} cases but {} requests were supplied",
                manifest.cases.len(),
                inputs.len()
            )));
        }

        let cases = manifest
            .cases
            .iter()
            .cloned()
            .zip(inputs)
            .map(|(spec, input)| {
                if spec.duration_validation != input.duration_features.is_some() {
                    return Err(IrodoriError::Config(format!(
                        "warmup duration feature presence does not match manifest for {} frames / {:?}",
                        spec.latent_frames, spec.topology
                    )));
                }
                let request = engine.prepare_sampling_request(input.request)?;
                if request.sequence_length() != spec.latent_frames {
                    return Err(IrodoriError::Shape(format!(
                        "warmup request has {} frames, manifest requires {}",
                        request.sequence_length(),
                        spec.latent_frames
                    )));
                }
                if !spec.topology.matches(&request) {
                    return Err(IrodoriError::Config(format!(
                        "warmup request conditioning does not match {:?}",
                        spec.topology
                    )));
                }
                if let Some(features) = &input.duration_features {
                    let [feature_batch, feature_width] = features.dims();
                    let expected_batch = request.request.text_ids.dims()[0];
                    let expected_width = engine.model_config().duration_aux_dim;
                    if [feature_batch, feature_width] != [expected_batch, expected_width] {
                        return Err(IrodoriError::Shape(format!(
                            "duration features must be [{expected_batch}, {expected_width}], got [{feature_batch}, {feature_width}]"
                        )));
                    }
                }
                Ok(WarmupCase {
                    spec,
                    request,
                    duration_features: input.duration_features,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { manifest, cases })
    }

    pub fn manifest(&self) -> &WarmupManifest {
        &self.manifest
    }
}

/// Startup warmup timings and real-output checks.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct WarmupReport {
    pub manifest: WarmupManifest,
    pub weight_preparation_seconds: f64,
    pub dry_run_seconds: f64,
    pub duration_validation_seconds: f64,
    pub duration_validation_cases: usize,
    pub real_validation_seconds: f64,
    pub real_validation_cases: usize,
    pub validated_audio_samples: usize,
}

/// WGPU-only all-resident online session.
pub struct OnlineSession<State> {
    engine: WgslInferenceEngine,
    codec: DacVaeDecoder,
    allowed_cases: HashSet<(usize, WarmupTopology)>,
    _state: PhantomData<State>,
}

impl OnlineSession<Unwarmed> {
    pub fn new(engine: WgslInferenceEngine, codec: DacVaeDecoder) -> Self {
        Self {
            engine,
            codec,
            allowed_cases: HashSet::new(),
            _state: PhantomData,
        }
    }

    /// Load RF and decode-only codec checkpoints concurrently, then prepare the
    /// requested RF profile and return an unwarmed all-resident session.
    ///
    /// The caller must configure the WGPU/CubeCL runtime before this method.
    /// Both workers use clones of the same high-level [`Device`], so this path
    /// is portable across the WGPU backends supported by Burn.
    #[allow(clippy::too_many_arguments)]
    pub fn load_parallel(
        device: Device,
        model_checkpoint: impl AsRef<Path>,
        codec_checkpoint: impl AsRef<Path>,
        sampling: SamplerParams,
        weight_profile: WgslWeightProfile,
        duration_residency: DurationModelResidency,
    ) -> Result<(Self, SessionLoadReport)> {
        let wall_started = Instant::now();
        let model_checkpoint = PathBuf::from(model_checkpoint.as_ref());
        let codec_checkpoint = PathBuf::from(codec_checkpoint.as_ref());
        let codec_device = device.clone();
        let builder = InferenceBuilder::<_>::new(device);

        let (loaded, codec, rf_checkpoint_seconds, codec_checkpoint_seconds) =
            std::thread::scope(|scope| -> Result<_> {
                let codec_handle = scope.spawn(move || {
                    let started = Instant::now();
                    let codec = load_decoder(&codec_checkpoint, &codec_device);
                    (codec, started.elapsed().as_secs_f64())
                });
                let rf_started = Instant::now();
                let loaded = match duration_residency {
                    DurationModelResidency::Predictive => {
                        builder.load_weights(&model_checkpoint)?
                    }
                    DurationModelResidency::ExactGeometryOnly => {
                        builder.load_weights_exact_only(&model_checkpoint)?
                    }
                };
                let rf_checkpoint_seconds = rf_started.elapsed().as_secs_f64();
                let (codec, codec_checkpoint_seconds) = codec_handle.join().map_err(|_| {
                    IrodoriError::Config("parallel codec loader panicked".to_owned())
                })?;
                Ok((
                    loaded,
                    codec?,
                    rf_checkpoint_seconds,
                    codec_checkpoint_seconds,
                ))
            })?;

        let profile_started = Instant::now();
        let engine = loaded
            .with_sampling(sampling)
            .build_wgsl_with_profile(weight_profile)?;
        let rf_profile_preparation_seconds = profile_started.elapsed().as_secs_f64();
        let report = SessionLoadReport {
            wall_seconds: wall_started.elapsed().as_secs_f64(),
            rf_checkpoint_seconds,
            codec_checkpoint_seconds,
            rf_profile_preparation_seconds,
        };
        Ok((Self::new(engine, codec), report))
    }

    /// Compile every manifest shape without executing ordinary workload dispatches,
    /// then run a small set of real end-to-end validations before becoming ready.
    pub fn warm(mut self, plan: WarmupPlan) -> Result<(OnlineSession<SessionReady>, WarmupReport)> {
        let device = self.engine.device().clone();
        let prepare_started = Instant::now();
        self.codec.prepare_for_wgsl();
        sync(&device, "after codec weight preparation")?;
        let weight_preparation_seconds = prepare_started.elapsed().as_secs_f64();

        let dry_started = Instant::now();
        {
            let _dry_run = cubecl::dry_run::DryRun::new();
            for case in &plan.cases {
                {
                    let _patched = self.engine.sample_prepared(case.request.clone())?;
                }
                device.memory_cleanup();
            }
            for case in &plan.cases {
                {
                    let batch = case.request.request.text_ids.dims()[0];
                    let latent = Tensor::<3>::zeros(
                        [
                            batch,
                            case.spec.latent_frames,
                            self.engine.model_config().latent_dim,
                        ],
                        &device,
                    );
                    let _audio = self.codec.decode_wgsl(latent);
                }
                // DryRun keeps compilation/autotune state, not tensor results.
                // RF and codec are compiled in separate phases and allocator
                // pages are released between shape classes so the startup pass
                // does not co-retain every shape's transient peak on 12 GiB.
                device.memory_cleanup();
            }
        }
        sync(&device, "after compile-only warmup")?;
        let dry_run_seconds = dry_started.elapsed().as_secs_f64();

        let duration_started = Instant::now();
        let mut duration_validation_cases = 0;
        match plan.manifest.duration_policy {
            DurationWarmupPolicy::Required if !self.engine.has_duration_predictor() => {
                return Err(IrodoriError::Config(
                    "predictive warmup requires a resident duration predictor".to_owned(),
                ));
            }
            DurationWarmupPolicy::ExactGeometryOnly if self.engine.has_duration_predictor() => {
                return Err(IrodoriError::Config(
                    "exact-geometry warmup must use a duration-free engine".to_owned(),
                ));
            }
            _ => {}
        }
        for case in plan
            .cases
            .iter()
            .filter(|case| case.spec.duration_validation)
        {
            validate_duration(&self.engine, case)?;
            duration_validation_cases += 1;
        }
        sync(&device, "after duration warmup validation")?;
        let duration_validation_seconds = duration_started.elapsed().as_secs_f64();

        let real_started = Instant::now();
        let mut real_validation_cases = 0;
        let mut validated_audio_samples = 0;
        for case in plan.cases.iter().filter(|case| case.spec.real_validation) {
            let patched = self.engine.sample_prepared(case.request.clone())?;
            let latent = unpatchify_latent(
                patched,
                self.engine.model_config().latent_patch_size,
                self.engine.model_config().latent_dim,
            );
            let audio = self.codec.decode_wgsl(latent);
            let values = audio.into_data().to_vec::<f32>().map_err(|error| {
                IrodoriError::Dtype("warmup audio".to_owned(), error.to_string())
            })?;
            if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
                return Err(IrodoriError::Config(format!(
                    "real warmup validation failed for {} frames / {:?}",
                    case.spec.latent_frames, case.spec.topology
                )));
            }
            real_validation_cases += 1;
            validated_audio_samples += values.len();
        }
        sync(&device, "after real warmup validation")?;
        let real_validation_seconds = real_started.elapsed().as_secs_f64();

        let allowed_cases = plan
            .manifest
            .cases
            .iter()
            .map(|case| (case.latent_frames, case.topology))
            .collect();
        let report = WarmupReport {
            manifest: plan.manifest,
            weight_preparation_seconds,
            dry_run_seconds,
            duration_validation_seconds,
            duration_validation_cases,
            real_validation_seconds,
            real_validation_cases,
            validated_audio_samples,
        };
        Ok((
            OnlineSession {
                engine: self.engine,
                codec: self.codec,
                allowed_cases,
                _state: PhantomData,
            },
            report,
        ))
    }
}

impl OnlineSession<SessionReady> {
    /// Synthesize while keeping RF and codec weights resident.
    pub fn synthesize(&self, request: SamplingRequest) -> Result<Tensor<3>> {
        let prepared = self.engine.prepare_sampling_request(request)?;
        let key = (
            prepared.sequence_length(),
            WarmupTopology::from_request(&prepared),
        );
        if !self.allowed_cases.contains(&key) {
            return Err(IrodoriError::Config(format!(
                "request shape/topology {key:?} was not admitted by the warmup manifest"
            )));
        }
        let patched = self.engine.sample_prepared(prepared)?;
        let latent = unpatchify_latent(
            patched,
            self.engine.model_config().latent_patch_size,
            self.engine.model_config().latent_dim,
        );
        Ok(self.codec.decode_wgsl(latent))
    }

    pub fn engine(&self) -> &WgslInferenceEngine {
        &self.engine
    }

    pub fn codec(&self) -> &DacVaeDecoder {
        &self.codec
    }
}

fn validate_duration(engine: &WgslInferenceEngine, case: &WarmupCase) -> Result<()> {
    let request = &case.request.request;
    let aux = AuxConditionInput::try_from_request(
        request.ref_latent.clone(),
        request.ref_mask.clone(),
        request.caption_ids.clone(),
        request.caption_mask.clone(),
    )?;
    let condition =
        engine.encode_conditions(request.text_ids.clone(), request.text_mask.clone(), aux)?;
    let batch = request.text_ids.dims()[0];
    let has_speaker = Tensor::<1, Bool>::from_data(
        TensorData::new(vec![case.request.has_speaker_context(); batch], [batch]),
        engine.device(),
    );
    let has_caption = Tensor::<1, Bool>::from_data(
        TensorData::new(vec![case.request.has_caption_context(); batch], [batch]),
        engine.device(),
    );
    let features = case.duration_features.clone().ok_or_else(|| {
        IrodoriError::MissingInput("duration warmup features are missing".to_owned())
    })?;
    let predicted = if !case.request.has_speaker_context()
        && !case.request.has_caption_context()
        && case.request.conditioned_text_mask_all_valid
    {
        engine.predict_duration_compact_no_aux(&condition, features, has_speaker, has_caption)?
    } else {
        engine.predict_duration_log_frames(&condition, features, has_speaker, has_caption)?
    };
    let values = predicted.into_data().to_vec::<f32>().map_err(|error| {
        IrodoriError::Dtype("duration warmup result".to_owned(), error.to_string())
    })?;
    if values.len() != batch
        || values
            .iter()
            .any(|value| !value.is_finite() || value.exp_m1() < 0.0)
    {
        return Err(IrodoriError::Config(format!(
            "duration warmup validation failed for {} frames / {:?}",
            case.spec.latent_frames, case.spec.topology
        )));
    }
    Ok(())
}

fn sync(device: &burn::tensor::Device, stage: &str) -> Result<()> {
    device.sync().map_err(|error| {
        IrodoriError::Config(format!("WGPU synchronization failed {stage}: {error}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_manifest_covers_required_v4_shapes() {
        let manifest = WarmupManifest::v4_service();
        let frames = manifest
            .cases
            .iter()
            .map(|case| case.latent_frames)
            .collect::<HashSet<_>>();
        assert_eq!(frames, HashSet::from([45, 112, 255, 333, 489, 685]));
    }

    #[test]
    fn every_represented_topology_requires_real_validation() {
        let error = WarmupManifest::new(vec![WarmupCaseSpec {
            latent_frames: 112,
            topology: WarmupTopology::TextOnly,
            real_validation: false,
            duration_validation: false,
        }])
        .expect_err("missing real validation must fail");
        assert!(error.to_string().contains("lacks a real validation"));
    }

    #[test]
    fn duplicate_shape_topology_is_rejected() {
        let case = WarmupCaseSpec {
            latent_frames: 112,
            topology: WarmupTopology::TextOnly,
            real_validation: true,
            duration_validation: true,
        };
        assert!(WarmupManifest::new(vec![case.clone(), case]).is_err());
    }
}
