pub mod autotune_approval;
pub mod backend_config;
#[cfg(feature = "codec")]
pub mod codec;
pub mod config;
pub mod error;
#[cfg(feature = "inference")]
pub mod inference;
pub mod kernels;
#[cfg(feature = "lora")]
pub mod lora;
pub mod model;
#[cfg(all(feature = "inference", feature = "codec"))]
pub mod online_session;
#[cfg(all(feature = "inference", feature = "codec"))]
pub mod phase_batch;
pub(crate) mod profiling;
pub mod rf;
pub mod route_autotune;
#[cfg(all(feature = "inference", feature = "codec"))]
pub mod runtime;
#[cfg(feature = "text-normalization")]
pub mod text_normalization;
pub mod validation;
pub mod weights;

pub use backend_config::{InferenceBackendKind, WgpuFloatPrecision, WgpuRaw};
#[cfg(feature = "codec")]
pub use codec::{load_codec, load_decoder};
pub use config::{CfgGuidanceMode, ModelConfig, SamplerMethod, SamplingConfig};
pub use error::{IrodoriError, Result};
#[cfg(feature = "inference")]
pub use inference::{
    DiagnosticForwardInput, DiagnosticForwardTrace, InferenceBuilder, InferenceEngine,
    WgslInferenceEngine, WgslWeightProfile,
};
pub use model::{
    AuxConditionInput, AuxConditionState, BlockDebugOutputs, BothConditioner, CondKvCache,
    EncodedCondition, InferenceOptimizedModel, TextToLatentRfDiT, WgslInferenceOptimizedModel,
    unpatchify_latent,
};
#[cfg(all(feature = "inference", feature = "codec"))]
pub use model::{LayoutsSelected, PreparedModel, ProfileLocked};
#[cfg(all(feature = "inference", feature = "codec"))]
pub use online_session::{
    CapturedOnlineSession, DurationModelResidency, DurationWarmupPolicy, OnlineSession,
    SessionLoadReport, SessionReady, Unwarmed as SessionUnwarmed, WarmupCaseSpec, WarmupInput,
    WarmupManifest, WarmupPlan, WarmupReport, WarmupTopology,
};
#[cfg(all(feature = "inference", feature = "codec"))]
pub use phase_batch::{
    BatchAudio, BatchItemId, CodecItemTiming, Complete as PhaseBatchComplete, LatentsResident,
    OutputGeometry, PhaseBatch, PhaseBatchMetrics, PlannedSynthesis, RfItemTiming, RfResident,
    SpeakerKey, VoiceIdentity,
};
pub use rf::{
    ConditioningGeometry, ConditioningSignal, ContextKvWorkReport,
    FixedTimestepConditionWorkReport, GuidanceConfig, PreparedSamplingRequest,
    SamplerDiagnosticForward, SamplerDiagnosticTrace, SamplerForwardEvaluation, SamplerForwardLane,
    SamplerForwardWork, SamplerParams, SamplerWorkReport, SamplingRequest, SpeakerKvConfig,
    TemporalRescaleConfig, sample_euler_rf_cfg, sample_euler_rf_cfg_wgsl,
};
pub use route_autotune::{
    AccuracyDisposition, ApprovedRouteManifest, ApprovedRouteManifestSet, ApprovedRouteSelection,
    AttentionOutputWeightRoute, MlpContractWeightRoute, PersistentRouteCacheEligibility,
    ProjectionRoute, ResolvedRouteTable, RfBatchClass, RouteAccuracyMetrics, RouteCacheMissReason,
    RouteCandidateMeasurement, RouteCandidateRejection, RouteCandidateRejectionReason,
    RouteCandidateRequest, RouteCandidateRun, RouteCandidateRunner, RouteChoice,
    RouteDeviceIdentity, RouteInstallDecision, RouteInstallReceipt, RouteManifestResolution,
    RouteOperation, RouteProblem, RouteSelectionReason, RouteTuningCase, RouteTuningPolicy,
    RouteTuningWorkload, accept_externally_installed_route_table, autotune_routes,
    current_binary_sha256, current_platform_version, default_route_manifest_set_path,
    install_approved_route_manifest, install_legacy_production_route_table,
    install_portable_route_table, install_route_manifest_set, select_approved_routes,
    select_approved_routes_with_rejections, sha256_file,
};
#[cfg(all(feature = "inference", feature = "codec"))]
pub use runtime::{
    AllocatorPolicy, DeviceSelector, EvictionReason, IdleTimeoutMillis, LatentFrames,
    MemoryPressure, RequestAdmissionPolicy, RequestClass, RequestReadiness, ResidencyPolicy,
    Runtime, RuntimeBuilder, RuntimeCachePolicy, RuntimeCacheReceipt, RuntimeCold,
    RuntimeConfiguration, RuntimeConfigured, RuntimeLoaded, RuntimeReady, RuntimeRoutePolicy,
    RuntimeStartupReport, SamplingPreset, WarmupCoverage, WarmupSelection, WeightLayout,
    WeightLayoutSet, WeightResidencyBasis, WeightResidencyPlan, WeightResidencyPolicy,
    WgpuExecutionPolicy,
};
#[cfg(feature = "text-normalization")]
pub use text_normalization::normalize_text;
pub use weights::{
    ModelCheckpointLoader, load_model, load_model_exact_only, load_model_with_float_dtype,
    load_model_with_float_dtype_and_loader, load_model_with_loader,
};
