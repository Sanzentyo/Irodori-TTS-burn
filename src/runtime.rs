//! High-level lifecycle for a cached, long-lived WGPU inference runtime.
//!
//! The lower-level [`crate::OnlineSession`] API remains available for callers
//! that already own a configured Burn device. This module owns the ordinary
//! application sequence instead: configure the persistent cache, initialize
//! WGPU, load both resident models, perform manifest-driven warmup, serve
//! admitted requests, and explicitly evict the models when policy requires it.

use std::{
    collections::{BTreeSet, HashSet},
    marker::PhantomData,
    num::NonZeroU64,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use burn::{
    backend::wgpu::{MemoryConfiguration, RuntimeOptions, graphics::AutoGraphicsApi, init_setup},
    tensor::{Device, Tensor},
};
use serde::{Deserialize, Serialize};

use crate::{
    ApprovedRouteManifestSet, IrodoriError, Result, RouteCacheMissReason, RouteDeviceIdentity,
    RouteInstallReceipt, SamplerParams, SamplingRequest, WgpuFloatPrecision, WgslWeightProfile,
    backend_config::{
        CubeClCacheReceipt, configure_cubecl_persistent_cache_for_precision,
        default_cubecl_cache_root, wgpu_device, wgpu_device_from_adapter_index,
        wgpu_device_with_precision,
    },
    online_session::{
        DurationModelResidency, DurationWarmupPolicy, OnlineSession, SessionLoadReport,
        SessionReady, Unwarmed, WarmupInput, WarmupManifest, WarmupPlan, WarmupReport,
        WarmupTopology,
    },
    route_autotune::{
        accept_externally_installed_route_table, current_binary_sha256,
        default_route_manifest_set_path, install_legacy_production_route_table,
        install_portable_route_table, install_recommended_route_table,
        install_route_manifest_set_with_defaults, sha256_file,
    },
};

/// Runtime configuration has not touched the global WGPU/CubeCL state.
#[derive(Debug)]
pub struct RuntimeCold;

/// Cache and device policy are installed, but model weights are not resident.
#[derive(Debug)]
pub struct RuntimeConfigured {
    device: Device,
    cache: RuntimeCacheReceipt,
    routes: RouteInstallReceipt,
    initialization_seconds: f64,
}

/// RF and codec models are resident but no request class is ready for traffic.
pub struct RuntimeLoaded {
    device: Device,
    cache: RuntimeCacheReceipt,
    routes: RouteInstallReceipt,
    initialization_seconds: f64,
    session: OnlineSession<Unwarmed>,
    load: SessionLoadReport,
    weight_residency: WeightResidencyPlan,
    planned_manifest: Option<WarmupManifest>,
}

/// Warmup and real validation have completed for an explicit request set.
#[derive(Debug)]
pub struct RuntimeReady;

/// Select the WGPU adapter without conflating `DefaultDevice` and adapter zero.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "index")]
pub enum DeviceSelector {
    #[default]
    Default,
    AdapterIndex(usize),
}

impl DeviceSelector {
    fn device(self) -> burn::backend::wgpu::WgpuDevice {
        match self {
            Self::Default => wgpu_device(0),
            Self::AdapterIndex(index) => wgpu_device_from_adapter_index(index),
        }
    }

    fn cache_namespace(self) -> String {
        match self {
            Self::Default => "wgpu-default-adapter".to_owned(),
            Self::AdapterIndex(index) => format!("wgpu-adapter-{index}"),
        }
    }
}

/// Where CubeCL's versioned environment should be stored.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "path")]
pub enum RuntimeCachePolicy {
    /// Use the operating system's application cache directory.
    #[default]
    PlatformDefault,
    /// Use this application-owned root and append an adapter namespace.
    Root(PathBuf),
    /// The embedding application configured CubeCL before creating this builder.
    ExternallyConfigured,
}

/// Proof of whether this builder installed the cache policy itself.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "receipt")]
pub enum RuntimeCacheReceipt {
    Managed(CubeClCacheReceipt),
    ExternallyConfigured,
}

/// Startup authority for device-specific graph routes.
///
/// `Auto` first resolves an exact manifest, then uses the shipped NVIDIA or
/// Apple family prior. The startup receipt distinguishes measured exact routes
/// from family defaults; unknown vendors remain portable.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "path")]
pub enum RuntimeRoutePolicy {
    #[default]
    Auto,
    Portable,
    ApprovedManifestSet(PathBuf),
    /// Reproduce the static pre-autotune policy for controlled comparisons.
    LegacyProduction,
    /// The embedding application installed a route table before initialize.
    ExternallyInstalled,
}

/// WGPU allocator behavior selected before runtime initialization.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AllocatorPolicy {
    #[default]
    ExclusivePages,
    SubSlices,
}

impl AllocatorPolicy {
    const fn memory_configuration(self) -> MemoryConfiguration {
        match self {
            Self::ExclusivePages => MemoryConfiguration::ExclusivePages,
            Self::SubSlices => MemoryConfiguration::SubSlices,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::ExclusivePages => "exclusive_pages",
            Self::SubSlices => "sub_slices",
        }
    }
}

/// Process-local WGPU execution settings.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WgpuExecutionPolicy {
    tasks_max: usize,
    allocator: AllocatorPolicy,
}

impl WgpuExecutionPolicy {
    pub fn new(tasks_max: usize, allocator: AllocatorPolicy) -> Result<Self> {
        if tasks_max == 0 {
            return Err(IrodoriError::Config(
                "WGPU tasks_max must be greater than zero".to_owned(),
            ));
        }
        Ok(Self {
            tasks_max,
            allocator,
        })
    }

    pub const fn tasks_max(self) -> usize {
        self.tasks_max
    }

    pub const fn allocator(self) -> AllocatorPolicy {
        self.allocator
    }
}

impl Default for WgpuExecutionPolicy {
    fn default() -> Self {
        Self {
            tasks_max: 32,
            allocator: AllocatorPolicy::ExclusivePages,
        }
    }
}

/// Named sampling policies keep benchmark-only four-step work out of defaults.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingPreset {
    /// Official CLI defaults: FP32-oriented 40-step Euler and text/caption/speaker
    /// CFG scales 3/3/5.
    #[default]
    OfficialV4,
    /// Official Voice Design UI defaults, which use caption CFG 4 instead of 3.
    OfficialVoiceDesign,
}

impl SamplingPreset {
    pub fn parameters(self) -> SamplerParams {
        let mut parameters = SamplerParams::default();
        if matches!(self, Self::OfficialVoiceDesign) {
            parameters.guidance.scale_caption = 4.0;
        }
        parameters
    }
}

/// Built-in coverage or a fully application-defined manifest.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "manifest")]
pub enum WarmupSelection {
    /// Short/medium text, Voice Design, and prepared-clone preview requests.
    Interactive,
    /// All measured v4 lengths and all four conditioning topologies.
    FullService,
    Custom(WarmupManifest),
}

impl WarmupSelection {
    fn resolve(self, residency: DurationModelResidency) -> WarmupManifest {
        let duration_policy = match residency {
            DurationModelResidency::Predictive => DurationWarmupPolicy::Required,
            DurationModelResidency::ExactGeometryOnly => DurationWarmupPolicy::ExactGeometryOnly,
        };
        match self {
            Self::Interactive => WarmupManifest::v4_interactive(duration_policy),
            Self::FullService => WarmupManifest::v4_full_service(duration_policy),
            Self::Custom(manifest) => manifest,
        }
    }
}

/// Non-zero idle duration used by residency decisions.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct IdleTimeoutMillis(NonZeroU64);

impl IdleTimeoutMillis {
    pub fn new(milliseconds: u64) -> Result<Self> {
        NonZeroU64::new(milliseconds).map(Self).ok_or_else(|| {
            IrodoriError::Config("idle timeout must be greater than zero".to_owned())
        })
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }

    fn duration(self) -> Duration {
        Duration::from_millis(self.get())
    }
}

/// Model eviction is decided explicitly by the service; no hidden timer owns
/// or drops GPU resources behind the caller's back.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "idle_timeout_ms")]
pub enum ResidencyPolicy {
    #[default]
    AlwaysResident,
    EvictAfterIdle(IdleTimeoutMillis),
    EvictOnMemoryPressure,
    EvictAfterIdleOrMemoryPressure(IdleTimeoutMillis),
}

/// Behavior for a valid shape/topology that was not part of startup warmup.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestAdmissionPolicy {
    /// Reject before sampling so a service can guarantee its warm latency set.
    StrictWarmup,
    /// Compile on the first request, then remember that class as process-warm.
    #[default]
    CompileOnDemand,
}

/// How a runtime chooses the irreversible WGPU weight layout before loading.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "profile")]
pub enum WeightResidencyPolicy {
    /// Preserve the historical explicit builder behavior.
    Explicit(WgslWeightProfile),
    /// Derive the narrowest safe profile from warmup coverage and admission.
    FromWarmupManifest,
}

impl Default for WeightResidencyPolicy {
    fn default() -> Self {
        Self::Explicit(WgslWeightProfile::default())
    }
}

/// Evidence used to select a concrete weight profile.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightResidencyBasis {
    Explicit,
    StrictManifest,
    CompileOnDemandFallback,
}

/// Logical RF weight representation retained by a concrete profile.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightLayout {
    QkvGateSource,
    QkvGateRow,
    QkvGateColumn,
    QkNormPacked,
    SwiGluSource,
    SwiGluFused,
    SwiGluInterleaved,
    AttentionOutputSource,
    AttentionOutputPacked,
    MlpContractSource,
    MlpContractPacked,
    MlpContractCubeKColumn,
}

/// Serializable receipt for the weight layout selected before model load.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WeightResidencyPlan {
    pub profile: WgslWeightProfile,
    pub basis: WeightResidencyBasis,
    pub minimum_latent_frames: usize,
    pub maximum_latent_frames: usize,
    pub topologies: Vec<WarmupTopology>,
    pub resident_layouts: Vec<WeightLayout>,
}

impl WeightResidencyPlan {
    /// Build the conservative layout receipt for an explicitly selected
    /// profile. This does not inspect the active route table.
    pub fn explicit_profile(profile: WgslWeightProfile) -> Self {
        Self::explicit(profile)
    }

    /// Derive the exact layout receipt for a manifest against the route table
    /// installed before WGPU initialization.
    pub fn derive_for_manifest(
        manifest: &WarmupManifest,
        admission: RequestAdmissionPolicy,
    ) -> Result<Self> {
        Self::derive_for_routes(
            manifest,
            admission,
            crate::route_autotune::active_route_table(),
        )
    }

    fn explicit(profile: WgslWeightProfile) -> Self {
        Self {
            profile,
            basis: WeightResidencyBasis::Explicit,
            minimum_latent_frames: 0,
            maximum_latent_frames: 0,
            topologies: Vec::new(),
            resident_layouts: resident_layouts(profile),
        }
    }

    #[cfg(test)]
    fn derive(manifest: &WarmupManifest, admission: RequestAdmissionPolicy) -> Result<Self> {
        Self::derive_for_routes(
            manifest,
            admission,
            &crate::route_autotune::ResolvedRouteTable::production_approved(),
        )
    }

    fn derive_for_routes(
        manifest: &WarmupManifest,
        admission: RequestAdmissionPolicy,
        routes: &crate::route_autotune::ResolvedRouteTable,
    ) -> Result<Self> {
        validate_residency_manifest(manifest)?;
        if manifest.cases.is_empty() {
            return Err(IrodoriError::Config(
                "weight residency requires a valid, non-empty warmup manifest".to_owned(),
            ));
        }
        let minimum_latent_frames = manifest
            .cases
            .iter()
            .map(|case| case.latent_frames)
            .min()
            .expect("non-empty manifest has a minimum");
        let maximum_latent_frames = manifest
            .cases
            .iter()
            .map(|case| case.latent_frames)
            .max()
            .expect("non-empty manifest has a maximum");
        let mut topologies = manifest
            .cases
            .iter()
            .map(|case| case.topology)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        topologies.sort_unstable();

        let (profile, basis) = if admission == RequestAdmissionPolicy::CompileOnDemand {
            (
                WgslWeightProfile::ProductionPrepared,
                WeightResidencyBasis::CompileOnDemandFallback,
            )
        } else {
            let has_combined = topologies.contains(&WarmupTopology::DesignedAndClone);
            let all_text = topologies == [WarmupTopology::TextOnly];
            let all_exact_112 = minimum_latent_frames == 112 && maximum_latent_frames == 112;
            let all_long = minimum_latent_frames >= 100;
            let profile = if all_exact_112 && !has_combined {
                WgslWeightProfile::Fixed112PackedOnly
            } else if all_exact_112 {
                WgslWeightProfile::Fixed112OneLayout
            } else if all_long && all_text {
                WgslWeightProfile::LongTextPreparedOnly
            } else if all_long && !has_combined {
                WgslWeightProfile::LongAllVoicePreparedOnly
            } else {
                WgslWeightProfile::ProductionPrepared
            };
            (profile, WeightResidencyBasis::StrictManifest)
        };
        let resident_layouts = if basis == WeightResidencyBasis::StrictManifest
            && matches!(
                profile,
                WgslWeightProfile::LongTextPreparedOnly
                    | WgslWeightProfile::LongAllVoicePreparedOnly
                    | WgslWeightProfile::Fixed112PackedOnly
            ) {
            RouteRequirementSet::for_manifest(routes, manifest)?.into_layouts()?
        } else {
            resident_layouts(profile)
        };
        Ok(Self {
            profile,
            basis,
            minimum_latent_frames,
            maximum_latent_frames,
            topologies,
            resident_layouts,
        })
    }

    /// Validated, sorted physical layout set consumed by the irreversible
    /// model-preparation transition.
    pub fn layout_set(&self) -> Result<WeightLayoutSet> {
        WeightLayoutSet::new(self.resident_layouts.iter().copied())
    }
}

/// Union of physical layouts reached by an exact, strict warmup manifest.
///
/// This is intentionally derived from resolved route variants, not from a GPU
/// marketing family or a broad length bucket. A layout is retained when any
/// admitted CFG phase reaches it. Unrepresented B4 topology is rejected by the
/// profile classifier before this type is constructed.
#[derive(Default)]
struct RouteRequirementSet(BTreeSet<WeightLayout>);

impl RouteRequirementSet {
    fn for_manifest(
        routes: &crate::route_autotune::ResolvedRouteTable,
        manifest: &WarmupManifest,
    ) -> Result<Self> {
        let mut requirements = Self::default();
        for case in &manifest.cases {
            requirements.insert_problem(routes, 1, case.latent_frames)?;
            match case.topology {
                WarmupTopology::TextOnly => {
                    requirements.insert_problem(routes, 2, case.latent_frames)?;
                }
                WarmupTopology::Designed | WarmupTopology::PreparedClone => {
                    requirements.insert_problem(routes, 3, case.latent_frames)?;
                }
                WarmupTopology::DesignedAndClone => {
                    return Err(IrodoriError::Config(
                        "exact route-derived residency does not represent B4 topology".to_owned(),
                    ));
                }
            }
        }
        Ok(requirements)
    }

    fn insert_problem(
        &mut self,
        routes: &crate::route_autotune::ResolvedRouteTable,
        batch: usize,
        sequence: usize,
    ) -> Result<()> {
        use crate::route_autotune::{
            AttentionMaterializationRoute, AttentionOutputWeightRoute, MlpContractWeightRoute,
            PostSdpaRoute, ProjectionRoute, SwiGluRoute,
        };

        crate::route_autotune::RouteProblem::new(batch, sequence)?;

        // The prepared WGPU graph always has a row-major combined projection:
        // handwritten T64 consumes it directly and the generic path keeps it
        // as its correctness fallback. The column layout is a separate tuned
        // route and is needed only for its exact admitted cells.
        self.0.insert(WeightLayout::QkvGateRow);
        let generic_qkv =
            routes.attention_qkv_projection(batch, sequence) == ProjectionRoute::DefaultGraph;
        if generic_qkv && (sequence >= 200 || (batch == 2 && sequence == 100)) {
            self.0.insert(WeightLayout::QkvGateColumn);
        }
        if matches!(
            routes.attention_materialization(batch, sequence),
            AttentionMaterializationRoute::DirectPackedKv
                | AttentionMaterializationRoute::CubeKProjectionDirectPackedKv
                | AttentionMaterializationRoute::ProjectionDirectPackedKv
                | AttentionMaterializationRoute::ProjectionDirectPackedKvSubgroup
        ) {
            self.0.insert(WeightLayout::QkNormPacked);
        }
        if routes.attention_materialization(batch, sequence)
            == AttentionMaterializationRoute::CubeKProjectionDirectPackedKv
        {
            self.0.insert(WeightLayout::QkvGateColumn);
        }

        match routes.mlp_expand(batch, sequence) {
            SwiGluRoute::DefaultGraph
            | SwiGluRoute::HandwrittenT64
            | SwiGluRoute::HandwrittenT64VectorInput
            | SwiGluRoute::HandwrittenWarp32VectorInput
            | SwiGluRoute::HandwrittenWarp32Rows128VectorInput => {
                self.0.insert(WeightLayout::SwiGluFused);
            }
            SwiGluRoute::SplitProjectionPairEpilogue => {
                self.0.insert(WeightLayout::SwiGluSource);
            }
            SwiGluRoute::CubeKCompressedInterleaved
            | SwiGluRoute::CubeKCompressedInterleavedMaxTile
            | SwiGluRoute::CubeKCompressedInterleavedDoubleUnit
            | SwiGluRoute::CubeKCompressedInterleavedPlaneVec
            | SwiGluRoute::CubeKCompressedInterleavedGemm => {
                self.0.insert(WeightLayout::SwiGluInterleaved);
            }
        }
        match routes.attention_output_weight(batch, sequence) {
            AttentionOutputWeightRoute::SourceColumnFlat => {
                self.0.insert(WeightLayout::AttentionOutputSource);
            }
            AttentionOutputWeightRoute::PackedRowFlat
            | AttentionOutputWeightRoute::PackedRowRank3 => {
                self.0.insert(WeightLayout::AttentionOutputPacked);
            }
        }
        if routes.post_sdpa(batch, sequence) == PostSdpaRoute::DirectOutputResidual {
            self.0.insert(WeightLayout::AttentionOutputPacked);
        }
        match routes.mlp_contract_weight(batch, sequence) {
            MlpContractWeightRoute::SourceColumnFlat => {
                self.0.insert(WeightLayout::MlpContractSource);
            }
            MlpContractWeightRoute::PackedRowFlat | MlpContractWeightRoute::PackedRowRank3 => {
                self.0.insert(WeightLayout::MlpContractPacked);
            }
            MlpContractWeightRoute::CubeKColumnFlat => {
                self.0.insert(WeightLayout::MlpContractCubeKColumn);
            }
        }
        Ok(())
    }

    fn into_layouts(self) -> Result<Vec<WeightLayout>> {
        let layouts = self.0.into_iter().collect::<Vec<_>>();
        WeightLayoutSet::new(layouts.iter().copied())?;
        Ok(layouts)
    }
}

/// A validated set of physical RF weight representations.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WeightLayoutSet(Vec<WeightLayout>);

impl WeightLayoutSet {
    pub fn new(layouts: impl IntoIterator<Item = WeightLayout>) -> Result<Self> {
        let mut layouts = layouts.into_iter().collect::<Vec<_>>();
        layouts.sort_unstable();
        layouts.dedup();
        let contains = |layout| layouts.binary_search(&layout).is_ok();
        let qkv = contains(WeightLayout::QkvGateSource)
            || contains(WeightLayout::QkvGateRow)
            || contains(WeightLayout::QkvGateColumn);
        let swiglu = contains(WeightLayout::SwiGluSource)
            || contains(WeightLayout::SwiGluFused)
            || contains(WeightLayout::SwiGluInterleaved);
        let attention_out = contains(WeightLayout::AttentionOutputSource)
            || contains(WeightLayout::AttentionOutputPacked);
        let mlp_out = contains(WeightLayout::MlpContractSource)
            || contains(WeightLayout::MlpContractPacked)
            || contains(WeightLayout::MlpContractCubeKColumn);
        if !qkv || !swiglu || !attention_out || !mlp_out {
            return Err(IrodoriError::Config(
                "weight layout set leaves an RF projection without a representation".to_owned(),
            ));
        }
        Ok(Self(layouts))
    }

    pub fn contains(&self, layout: WeightLayout) -> bool {
        self.0.binary_search(&layout).is_ok()
    }

    pub fn as_slice(&self) -> &[WeightLayout] {
        &self.0
    }
}

fn resident_layouts(profile: WgslWeightProfile) -> Vec<WeightLayout> {
    use WeightLayout as L;
    let mut layouts = match profile {
        WgslWeightProfile::PortableFallback => vec![
            L::QkvGateSource,
            L::QkvGateRow,
            L::QkvGateColumn,
            L::QkNormPacked,
            L::SwiGluSource,
            L::SwiGluFused,
            L::AttentionOutputSource,
            L::AttentionOutputPacked,
            L::MlpContractSource,
            L::MlpContractPacked,
        ],
        WgslWeightProfile::TuningCandidates => vec![
            L::QkvGateSource,
            L::QkvGateRow,
            L::QkvGateColumn,
            L::QkNormPacked,
            L::SwiGluSource,
            L::SwiGluFused,
            L::SwiGluInterleaved,
            L::AttentionOutputSource,
            L::AttentionOutputPacked,
            L::MlpContractSource,
            L::MlpContractPacked,
            L::MlpContractCubeKColumn,
        ],
        WgslWeightProfile::ProductionPrepared => vec![
            L::QkvGateRow,
            L::QkvGateColumn,
            L::QkNormPacked,
            L::SwiGluFused,
            L::AttentionOutputSource,
            L::AttentionOutputPacked,
            L::MlpContractSource,
            L::MlpContractPacked,
        ],
        WgslWeightProfile::LongTextPreparedOnly => vec![
            L::QkvGateRow,
            L::QkvGateColumn,
            L::QkNormPacked,
            L::SwiGluFused,
            L::AttentionOutputPacked,
            L::MlpContractPacked,
        ],
        WgslWeightProfile::LongAllVoicePreparedOnly => {
            vec![
                L::QkvGateRow,
                L::QkvGateColumn,
                L::QkNormPacked,
                L::SwiGluFused,
                L::AttentionOutputPacked,
                L::MlpContractPacked,
            ]
        }
        WgslWeightProfile::Fixed112OneLayout => vec![
            L::QkvGateSource,
            L::QkvGateRow,
            L::QkNormPacked,
            L::SwiGluSource,
            L::SwiGluFused,
            L::AttentionOutputSource,
            L::AttentionOutputPacked,
            L::MlpContractSource,
            L::MlpContractPacked,
        ],
        WgslWeightProfile::Fixed112PackedOnly => vec![
            L::QkvGateRow,
            L::QkNormPacked,
            L::SwiGluFused,
            L::AttentionOutputSource,
            L::AttentionOutputPacked,
            L::MlpContractSource,
            L::MlpContractPacked,
        ],
    };
    layouts.sort_unstable();
    layouts
}

impl Default for WeightResidencyPlan {
    fn default() -> Self {
        Self::explicit(WgslWeightProfile::default())
    }
}

fn validate_residency_manifest(manifest: &WarmupManifest) -> Result<()> {
    if manifest.schema_version != WarmupManifest::SCHEMA_VERSION {
        return Err(IrodoriError::Config(format!(
            "unsupported warmup manifest schema {}",
            manifest.schema_version
        )));
    }
    // Deserialization and public struct construction can bypass `new`; rebuild
    // once to apply duplicate, topology-validation, duration, and zero checks
    // before any irreversible weight release.
    WarmupManifest::new_with_duration_policy(manifest.cases.clone(), manifest.duration_policy)?;
    Ok(())
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPressure {
    Normal,
    Critical,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvictionReason {
    IdleTimeout,
    MemoryPressure,
}

impl ResidencyPolicy {
    pub fn eviction_reason(
        self,
        idle_for: Duration,
        memory_pressure: MemoryPressure,
    ) -> Option<EvictionReason> {
        let pressure = matches!(memory_pressure, MemoryPressure::Critical);
        match self {
            Self::AlwaysResident => None,
            Self::EvictAfterIdle(timeout) => {
                (idle_for >= timeout.duration()).then_some(EvictionReason::IdleTimeout)
            }
            Self::EvictOnMemoryPressure => pressure.then_some(EvictionReason::MemoryPressure),
            Self::EvictAfterIdleOrMemoryPressure(_) if pressure => {
                Some(EvictionReason::MemoryPressure)
            }
            Self::EvictAfterIdleOrMemoryPressure(timeout) => {
                (idle_for >= timeout.duration()).then_some(EvictionReason::IdleTimeout)
            }
        }
    }
}

/// Validated runtime construction policy retained across explicit eviction.
#[derive(Clone, Debug)]
pub struct RuntimeConfiguration {
    model_checkpoint: PathBuf,
    codec_checkpoint: PathBuf,
    device: DeviceSelector,
    cache: RuntimeCachePolicy,
    routes: RuntimeRoutePolicy,
    precision: WgpuFloatPrecision,
    execution: WgpuExecutionPolicy,
    sampling: SamplerParams,
    weight_residency: WeightResidencyPolicy,
    duration_residency: DurationModelResidency,
    admission: RequestAdmissionPolicy,
    residency: ResidencyPolicy,
}

impl RuntimeConfiguration {
    pub fn new(model_checkpoint: impl Into<PathBuf>, codec_checkpoint: impl Into<PathBuf>) -> Self {
        Self {
            model_checkpoint: model_checkpoint.into(),
            codec_checkpoint: codec_checkpoint.into(),
            device: DeviceSelector::default(),
            cache: RuntimeCachePolicy::default(),
            routes: RuntimeRoutePolicy::default(),
            precision: WgpuFloatPrecision::Fp32,
            execution: WgpuExecutionPolicy::default(),
            sampling: SamplingPreset::default().parameters(),
            weight_residency: WeightResidencyPolicy::default(),
            duration_residency: DurationModelResidency::Predictive,
            admission: RequestAdmissionPolicy::default(),
            residency: ResidencyPolicy::default(),
        }
    }

    pub fn model_checkpoint(&self) -> &Path {
        &self.model_checkpoint
    }

    pub fn codec_checkpoint(&self) -> &Path {
        &self.codec_checkpoint
    }

    pub const fn duration_residency(&self) -> DurationModelResidency {
        self.duration_residency
    }

    pub const fn residency_policy(&self) -> ResidencyPolicy {
        self.residency
    }

    pub const fn admission_policy(&self) -> RequestAdmissionPolicy {
        self.admission
    }
}

/// Type-state runtime builder. State transitions consume the previous value.
pub struct RuntimeBuilder<State> {
    configuration: RuntimeConfiguration,
    state: State,
}

impl RuntimeBuilder<RuntimeCold> {
    pub fn new(model_checkpoint: impl Into<PathBuf>, codec_checkpoint: impl Into<PathBuf>) -> Self {
        Self {
            configuration: RuntimeConfiguration::new(model_checkpoint, codec_checkpoint),
            state: RuntimeCold,
        }
    }

    pub fn from_configuration(configuration: RuntimeConfiguration) -> Self {
        Self {
            configuration,
            state: RuntimeCold,
        }
    }

    pub fn device(mut self, selector: DeviceSelector) -> Self {
        self.configuration.device = selector;
        self
    }

    pub fn cache(mut self, policy: RuntimeCachePolicy) -> Self {
        self.configuration.cache = policy;
        self
    }

    pub fn routes(mut self, policy: RuntimeRoutePolicy) -> Self {
        self.configuration.routes = policy;
        self
    }

    pub fn precision(mut self, precision: WgpuFloatPrecision) -> Self {
        self.configuration.precision = precision;
        self
    }

    pub fn execution(mut self, policy: WgpuExecutionPolicy) -> Self {
        self.configuration.execution = policy;
        self
    }

    pub fn sampling_preset(mut self, preset: SamplingPreset) -> Self {
        self.configuration.sampling = preset.parameters();
        self
    }

    pub fn sampling(mut self, parameters: SamplerParams) -> Self {
        self.configuration.sampling = parameters;
        self
    }

    pub fn weight_profile(mut self, profile: WgslWeightProfile) -> Self {
        self.configuration.weight_residency = WeightResidencyPolicy::Explicit(profile);
        self
    }

    /// Select the concrete profile from a warmup manifest before loading.
    pub fn derive_weight_profile_from_manifest(mut self) -> Self {
        self.configuration.weight_residency = WeightResidencyPolicy::FromWarmupManifest;
        self
    }

    pub fn duration_residency(mut self, residency: DurationModelResidency) -> Self {
        self.configuration.duration_residency = residency;
        self
    }

    pub fn admission(mut self, policy: RequestAdmissionPolicy) -> Self {
        self.configuration.admission = policy;
        self
    }

    pub fn residency(mut self, policy: ResidencyPolicy) -> Self {
        self.configuration.residency = policy;
        self
    }

    /// Install cache/runtime policy before the first tensor is created.
    pub fn initialize(self) -> Result<RuntimeBuilder<RuntimeConfigured>> {
        self.configuration.sampling.validate()?;
        for (label, path) in [
            ("model checkpoint", &self.configuration.model_checkpoint),
            ("codec checkpoint", &self.configuration.codec_checkpoint),
        ] {
            if !path.is_file() {
                return Err(IrodoriError::Config(format!(
                    "{label} is not a file: {}",
                    path.display()
                )));
            }
        }

        let started = Instant::now();
        let (cache, default_route_set) = match &self.configuration.cache {
            RuntimeCachePolicy::PlatformDefault => {
                let root = default_cubecl_cache_root()?;
                let route_set = default_route_manifest_set_path(&root);
                let adapter_root = root.join(self.configuration.device.cache_namespace());
                (
                    RuntimeCacheReceipt::Managed(configure_cubecl_persistent_cache_for_precision(
                        adapter_root,
                        self.configuration.precision,
                    )?),
                    Some(route_set),
                )
            }
            RuntimeCachePolicy::Root(root) => {
                if root.as_os_str().is_empty() {
                    return Err(IrodoriError::Config(
                        "runtime cache root must not be empty".to_owned(),
                    ));
                }
                let route_set = default_route_manifest_set_path(root);
                let adapter_root = root.join(self.configuration.device.cache_namespace());
                (
                    RuntimeCacheReceipt::Managed(configure_cubecl_persistent_cache_for_precision(
                        adapter_root,
                        self.configuration.precision,
                    )?),
                    Some(route_set),
                )
            }
            RuntimeCachePolicy::ExternallyConfigured => {
                (RuntimeCacheReceipt::ExternallyConfigured, None)
            }
        };

        let wgpu_device = self.configuration.device.device();
        let setup = init_setup::<AutoGraphicsApi>(
            &wgpu_device,
            RuntimeOptions {
                tasks_max: self.configuration.execution.tasks_max(),
                memory_config: self
                    .configuration
                    .execution
                    .allocator()
                    .memory_configuration(),
            },
        );
        let route_set_path = match &self.configuration.routes {
            RuntimeRoutePolicy::ApprovedManifestSet(path) => Some(path.clone()),
            RuntimeRoutePolicy::Auto => default_route_set,
            RuntimeRoutePolicy::Portable
            | RuntimeRoutePolicy::LegacyProduction
            | RuntimeRoutePolicy::ExternallyInstalled => None,
        };
        let routes = match &self.configuration.routes {
            RuntimeRoutePolicy::Portable => {
                install_portable_route_table(RouteCacheMissReason::PortableRequested)?
            }
            RuntimeRoutePolicy::LegacyProduction => install_legacy_production_route_table()?,
            RuntimeRoutePolicy::ExternallyInstalled => accept_externally_installed_route_table()?,
            RuntimeRoutePolicy::Auto | RuntimeRoutePolicy::ApprovedManifestSet(_) => {
                match route_set_path {
                    Some(path) if path.is_file() => {
                        let manifest_set = ApprovedRouteManifestSet::load(&path)?;
                        let info = setup.adapter.get_info();
                        let identity = RouteDeviceIdentity {
                            adapter_name: info.name,
                            backend: format!("{:?}", info.backend),
                            device_type: format!("{:?}", info.device_type),
                            vendor_id: info.vendor,
                            device_id: info.device,
                            driver: info.driver,
                            driver_info: info.driver_info,
                            os: std::env::consts::OS.to_owned(),
                            platform_version: crate::current_platform_version().unwrap_or_default(),
                            architecture: std::env::consts::ARCH.to_owned(),
                            precision: self.configuration.precision.label().to_owned(),
                            allocator_policy: self
                                .configuration
                                .execution
                                .allocator()
                                .label()
                                .to_owned(),
                            compiler_policy: "wgpu_auto".to_owned(),
                            application_version: env!("CARGO_PKG_VERSION").to_owned(),
                            burn_version: "0.22.0-pre.2".to_owned(),
                            burn_cubecl_version: "0.22.0-pre.2".to_owned(),
                            cubecl_version: "0.11.0-pre.2".to_owned(),
                            cubek_version: "0.3.0-pre.2".to_owned(),
                            wgpu_version: "30.0.0".to_owned(),
                            model_sha256: sha256_file(&self.configuration.model_checkpoint)?,
                            codec_sha256: sha256_file(&self.configuration.codec_checkpoint)?,
                            binary_sha256: current_binary_sha256()?,
                        };
                        install_route_manifest_set_with_defaults(
                            Some(&manifest_set),
                            &identity,
                            RouteCacheMissReason::NoExactDeviceProfile,
                        )?
                    }
                    Some(path)
                        if matches!(
                            self.configuration.routes,
                            RuntimeRoutePolicy::ApprovedManifestSet(_)
                        ) =>
                    {
                        return Err(IrodoriError::Config(format!(
                            "approved route manifest set is not a file: {}",
                            path.display()
                        )));
                    }
                    _ => {
                        let info = setup.adapter.get_info();
                        install_recommended_route_table(
                            info.vendor,
                            &format!("{:?}", info.backend),
                            std::env::consts::OS,
                            RouteCacheMissReason::ManifestNotFound,
                        )?
                    }
                }
            }
        };
        let device = wgpu_device_with_precision(&wgpu_device, self.configuration.precision)?;
        Ok(RuntimeBuilder {
            configuration: self.configuration,
            state: RuntimeConfigured {
                device,
                cache,
                routes,
                initialization_seconds: started.elapsed().as_secs_f64(),
            },
        })
    }

    /// Convenience transition for callers that do not need the configured state.
    pub fn load(self) -> Result<RuntimeBuilder<RuntimeLoaded>> {
        self.initialize()?.load()
    }

    /// Initialize, derive residency from `selection`, and load both models.
    pub fn load_for(self, selection: WarmupSelection) -> Result<RuntimeBuilder<RuntimeLoaded>> {
        self.initialize()?.load_for(selection)
    }
}

impl RuntimeBuilder<RuntimeConfigured> {
    pub fn device_ref(&self) -> &Device {
        &self.state.device
    }

    pub fn cache_receipt(&self) -> &RuntimeCacheReceipt {
        &self.state.cache
    }

    pub fn route_receipt(&self) -> &RouteInstallReceipt {
        &self.state.routes
    }

    pub fn load(self) -> Result<RuntimeBuilder<RuntimeLoaded>> {
        let profile = match self.configuration.weight_residency {
            WeightResidencyPolicy::Explicit(profile) => profile,
            WeightResidencyPolicy::FromWarmupManifest => {
                return Err(IrodoriError::Config(
                    "manifest-derived weight residency requires load_for(selection)".to_owned(),
                ));
            }
        };
        self.load_with_plan(WeightResidencyPlan::explicit(profile), None)
    }

    /// Resolve coverage before the irreversible model-load transition.
    pub fn load_for(self, selection: WarmupSelection) -> Result<RuntimeBuilder<RuntimeLoaded>> {
        let manifest = selection.resolve(self.configuration.duration_residency);
        validate_residency_manifest(&manifest)?;
        let plan = match self.configuration.weight_residency {
            WeightResidencyPolicy::Explicit(profile) => WeightResidencyPlan::explicit(profile),
            WeightResidencyPolicy::FromWarmupManifest => WeightResidencyPlan::derive_for_routes(
                &manifest,
                self.configuration.admission,
                crate::route_autotune::active_route_table(),
            )?,
        };
        self.load_with_plan(plan, Some(manifest))
    }

    fn load_with_plan(
        self,
        weight_residency: WeightResidencyPlan,
        planned_manifest: Option<WarmupManifest>,
    ) -> Result<RuntimeBuilder<RuntimeLoaded>> {
        validate_route_residency(&weight_residency, planned_manifest.as_ref())?;
        let (session, load) = OnlineSession::<Unwarmed>::load_parallel(
            self.state.device.clone(),
            &self.configuration.model_checkpoint,
            &self.configuration.codec_checkpoint,
            self.configuration.sampling.clone(),
            weight_residency.clone(),
            self.configuration.duration_residency,
        )?;
        Ok(RuntimeBuilder {
            configuration: self.configuration,
            state: RuntimeLoaded {
                device: self.state.device,
                cache: self.state.cache,
                routes: self.state.routes,
                initialization_seconds: self.state.initialization_seconds,
                session,
                load,
                weight_residency,
                planned_manifest,
            },
        })
    }
}

fn validate_route_residency(
    residency: &WeightResidencyPlan,
    planned_manifest: Option<&WarmupManifest>,
) -> Result<()> {
    validate_route_residency_for(
        crate::route_autotune::active_route_table(),
        residency,
        planned_manifest,
    )
}

fn validate_route_residency_for(
    routes: &crate::ResolvedRouteTable,
    residency: &WeightResidencyPlan,
    planned_manifest: Option<&WarmupManifest>,
) -> Result<()> {
    let layouts = residency.layout_set()?;
    let needs_attention_packed = !layouts.contains(WeightLayout::AttentionOutputSource);
    let needs_mlp_packed = !layouts.contains(WeightLayout::MlpContractSource);
    if !needs_attention_packed && !needs_mlp_packed {
        return Ok(());
    }

    if routes.permits_legacy_profile_overlay() {
        return Ok(());
    }
    let mut problems = BTreeSet::new();
    if let Some(manifest) = planned_manifest {
        for case in &manifest.cases {
            problems.insert((1, case.latent_frames));
            match case.topology {
                WarmupTopology::TextOnly => {
                    problems.insert((2, case.latent_frames));
                }
                WarmupTopology::Designed | WarmupTopology::PreparedClone => {
                    problems.insert((3, case.latent_frames));
                }
                WarmupTopology::DesignedAndClone => {
                    return Err(IrodoriError::Config(
                        "source-free output weights cannot cover B4 combined conditioning"
                            .to_owned(),
                    ));
                }
            }
        }
    } else {
        let batches: &[usize] = match residency.profile {
            WgslWeightProfile::LongTextPreparedOnly => &[1, 2],
            WgslWeightProfile::LongAllVoicePreparedOnly => &[1, 2, 3],
            _ => {
                return Err(IrodoriError::Config(
                    "source-free route residency requires an exact warmup manifest".to_owned(),
                ));
            }
        };
        for sequence in 100..=crate::route_autotune::MAX_TUNED_SEQUENCE {
            for &batch in batches {
                problems.insert((batch, sequence));
            }
        }
    }

    for (batch, sequence) in problems {
        if needs_attention_packed
            && routes.attention_output_weight(batch, sequence)
                == crate::AttentionOutputWeightRoute::SourceColumnFlat
        {
            return Err(IrodoriError::Config(format!(
                "route table requires released attention-output source weight at B{batch} S{sequence}"
            )));
        }
        if needs_mlp_packed
            && routes.mlp_contract_weight(batch, sequence)
                == crate::MlpContractWeightRoute::SourceColumnFlat
        {
            return Err(IrodoriError::Config(format!(
                "route table requires released MLP-contract source weight at B{batch} S{sequence}"
            )));
        }
    }
    Ok(())
}

impl RuntimeBuilder<RuntimeLoaded> {
    pub fn session(&self) -> &OnlineSession<Unwarmed> {
        &self.state.session
    }

    pub fn weight_residency_plan(&self) -> &WeightResidencyPlan {
        &self.state.weight_residency
    }

    /// Warm the exact manifest used to derive the loaded weight layout.
    pub fn warm_planned(self, inputs: Vec<WarmupInput>) -> Result<Runtime<RuntimeReady>> {
        let manifest = self.state.planned_manifest.clone().ok_or_else(|| {
            IrodoriError::Config("load_for(selection) must precede warm_planned".to_owned())
        })?;
        let plan = WarmupPlan::prepare(self.state.session.engine(), manifest, inputs)?;
        self.warm_with_plan(plan)
    }

    /// Resolve a built-in/custom manifest, validate its tensor inputs, and
    /// become ready only after compile warmup plus real audio validation.
    pub fn warm(
        self,
        selection: WarmupSelection,
        inputs: Vec<WarmupInput>,
    ) -> Result<Runtime<RuntimeReady>> {
        let manifest = selection.resolve(self.configuration.duration_residency);
        if self
            .state
            .planned_manifest
            .as_ref()
            .is_some_and(|planned| planned != &manifest)
        {
            return Err(IrodoriError::Config(
                "warmup selection differs from the manifest used for weight residency".to_owned(),
            ));
        }
        let plan = WarmupPlan::prepare(self.state.session.engine(), manifest, inputs)?;
        self.warm_with_plan(plan)
    }

    /// Advanced entrypoint for callers that prepared a plan themselves.
    pub fn warm_with_plan(self, plan: WarmupPlan) -> Result<Runtime<RuntimeReady>> {
        if self
            .state
            .planned_manifest
            .as_ref()
            .is_some_and(|planned| planned != plan.manifest())
        {
            return Err(IrodoriError::Config(
                "warmup plan differs from the manifest used for weight residency".to_owned(),
            ));
        }
        let warm_started = Instant::now();
        let (session, warmup) = self.state.session.warm(plan)?;
        let warmup_wall_seconds = warm_started.elapsed().as_secs_f64();
        let startup = RuntimeStartupReport {
            cache: self.state.cache.clone(),
            routes: self.state.routes.clone(),
            initialization_seconds: self.state.initialization_seconds,
            load: self.state.load,
            warmup_wall_seconds,
            warmup,
            weight_residency: self.state.weight_residency,
        };
        Ok(Runtime {
            configuration: self.configuration,
            device: self.state.device,
            session,
            coverage: WarmupCoverage::from_manifest(&startup.warmup.manifest),
            startup,
            ready_since: Instant::now(),
            last_activity: Instant::now(),
            _state: PhantomData,
        })
    }

    /// Escape hatch for applications that own a more specialized lifecycle.
    pub fn into_unwarmed_session(self) -> OnlineSession<Unwarmed> {
        self.state.session
    }
}

/// Complete startup receipt suitable for readiness diagnostics.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RuntimeStartupReport {
    pub cache: RuntimeCacheReceipt,
    #[serde(default)]
    pub routes: RouteInstallReceipt,
    pub initialization_seconds: f64,
    pub load: SessionLoadReport,
    pub warmup_wall_seconds: f64,
    pub warmup: WarmupReport,
    #[serde(default)]
    pub weight_residency: WeightResidencyPlan,
}

impl RuntimeStartupReport {
    pub fn total_wall_seconds(&self) -> f64 {
        self.initialization_seconds + self.load.wall_seconds + self.warmup_wall_seconds
    }
}

/// Non-zero latent length admitted by a warmup manifest.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct LatentFrames(usize);

impl LatentFrames {
    pub fn new(frames: usize) -> Result<Self> {
        if frames == 0 {
            return Err(IrodoriError::Config(
                "latent frames must be greater than zero".to_owned(),
            ));
        }
        Ok(Self(frames))
    }

    pub const fn get(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct RequestClass {
    pub frames: LatentFrames,
    pub topology: WarmupTopology,
}

impl RequestClass {
    pub fn new(frames: usize, topology: WarmupTopology) -> Result<Self> {
        Ok(Self {
            frames: LatentFrames::new(frames)?,
            topology,
        })
    }
}

/// Immutable set of request classes proven ready in this process.
#[derive(Clone, Debug)]
pub struct WarmupCoverage {
    classes: HashSet<RequestClass>,
}

impl WarmupCoverage {
    fn from_manifest(manifest: &WarmupManifest) -> Self {
        let classes = manifest
            .cases
            .iter()
            .map(|case| RequestClass {
                frames: LatentFrames(case.latent_frames),
                topology: case.topology,
            })
            .collect();
        Self { classes }
    }

    pub fn contains(&self, request: RequestClass) -> bool {
        self.classes.contains(&request)
    }

    pub fn iter(&self) -> impl Iterator<Item = RequestClass> + '_ {
        self.classes.iter().copied()
    }

    fn insert(&mut self, request: RequestClass) {
        self.classes.insert(request);
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestReadiness {
    Ready,
    AcceptedWithWarmup,
    Rejected,
}

/// Ready, resident runtime. Synthesis requires `&mut self` so last activity is
/// updated atomically with respect to the caller's service loop.
pub struct Runtime<State> {
    configuration: RuntimeConfiguration,
    device: Device,
    session: OnlineSession<SessionReady>,
    coverage: WarmupCoverage,
    startup: RuntimeStartupReport,
    ready_since: Instant,
    last_activity: Instant,
    _state: PhantomData<State>,
}

impl Runtime<RuntimeReady> {
    pub fn startup_report(&self) -> &RuntimeStartupReport {
        &self.startup
    }

    pub fn coverage(&self) -> &WarmupCoverage {
        &self.coverage
    }

    pub fn request_readiness(&self, request: RequestClass) -> RequestReadiness {
        if self.coverage.contains(request) {
            RequestReadiness::Ready
        } else if matches!(
            self.configuration.admission,
            RequestAdmissionPolicy::CompileOnDemand
        ) {
            RequestReadiness::AcceptedWithWarmup
        } else {
            RequestReadiness::Rejected
        }
    }

    pub fn synthesize(&mut self, request: SamplingRequest) -> Result<Tensor<3>> {
        let prepared = self.session.engine().prepare_sampling_request(request)?;
        let request_class = RequestClass {
            frames: LatentFrames(prepared.sequence_length()),
            topology: WarmupTopology::from_prepared(&prepared),
        };
        let require_admission = matches!(
            self.configuration.admission,
            RequestAdmissionPolicy::StrictWarmup
        );
        let audio = self
            .session
            .synthesize_prepared(prepared, require_admission)?;
        self.coverage.insert(request_class);
        self.last_activity = Instant::now();
        Ok(audio)
    }

    pub fn ready_for(&self) -> Duration {
        self.ready_since.elapsed()
    }

    pub fn idle_for(&self) -> Duration {
        self.last_activity.elapsed()
    }

    pub fn eviction_reason(&self, pressure: MemoryPressure) -> Option<EvictionReason> {
        self.configuration
            .residency
            .eviction_reason(self.idle_for(), pressure)
    }

    pub fn session(&self) -> &OnlineSession<SessionReady> {
        &self.session
    }

    /// Drop resident models while retaining the configured device/cache state,
    /// allowing the same process to reload without reconfiguring CubeCL.
    pub fn evict(self) -> RuntimeBuilder<RuntimeConfigured> {
        RuntimeBuilder {
            configuration: self.configuration,
            state: RuntimeConfigured {
                device: self.device,
                cache: self.startup.cache,
                routes: self.startup.routes,
                initialization_seconds: self.startup.initialization_seconds,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_manifest(cases: &[(usize, WarmupTopology)]) -> WarmupManifest {
        WarmupManifest::new_with_duration_policy(
            cases
                .iter()
                .map(|&(latent_frames, topology)| crate::WarmupCaseSpec {
                    latent_frames,
                    topology,
                    real_validation: true,
                    duration_validation: false,
                })
                .collect(),
            DurationWarmupPolicy::ExactGeometryOnly,
        )
        .expect("test manifest is valid")
    }

    #[test]
    fn official_presets_do_not_use_benchmark_four_step_sampling() {
        let cli = SamplingPreset::OfficialV4.parameters();
        let design = SamplingPreset::OfficialVoiceDesign.parameters();
        assert_eq!(cli.num_steps, 40);
        assert_eq!(cli.guidance.scale_caption, 3.0);
        assert_eq!(design.num_steps, 40);
        assert_eq!(design.guidance.scale_caption, 4.0);
    }

    #[test]
    fn portable_route_table_cannot_authorize_source_free_weight_release() {
        let residency = WeightResidencyPlan::explicit(WgslWeightProfile::LongTextPreparedOnly);
        let manifest = exact_manifest(&[(112, WarmupTopology::TextOnly)]);
        let portable = crate::ResolvedRouteTable::portable();
        assert!(validate_route_residency_for(&portable, &residency, Some(&manifest)).is_err());

        let legacy = crate::ResolvedRouteTable::production_approved();
        validate_route_residency_for(&legacy, &residency, Some(&manifest))
            .expect("legacy profile owns its historical layout overlay");
    }

    #[test]
    fn interactive_and_full_service_coverage_are_distinct() {
        let interactive = WarmupSelection::Interactive.resolve(DurationModelResidency::Predictive);
        let full = WarmupSelection::FullService.resolve(DurationModelResidency::Predictive);
        assert!(interactive.cases.len() < full.cases.len());
        assert!(
            full.cases
                .iter()
                .any(|case| case.topology == WarmupTopology::DesignedAndClone)
        );
        assert!(full.cases.iter().any(|case| case.latent_frames == 685));
    }

    #[test]
    fn exact_geometry_profile_never_claims_duration_validation() {
        let manifest =
            WarmupSelection::Interactive.resolve(DurationModelResidency::ExactGeometryOnly);
        assert_eq!(
            manifest.duration_policy,
            DurationWarmupPolicy::ExactGeometryOnly
        );
        assert!(!manifest.cases.iter().any(|case| case.duration_validation));
    }

    #[test]
    fn request_class_rejects_zero_frames() {
        assert!(RequestClass::new(0, WarmupTopology::TextOnly).is_err());
    }

    #[test]
    fn residency_policy_prioritizes_critical_memory_pressure() {
        let timeout = IdleTimeoutMillis::new(1_000).unwrap();
        let policy = ResidencyPolicy::EvictAfterIdleOrMemoryPressure(timeout);
        assert_eq!(
            policy.eviction_reason(Duration::ZERO, MemoryPressure::Critical),
            Some(EvictionReason::MemoryPressure)
        );
        assert_eq!(
            policy.eviction_reason(Duration::from_secs(2), MemoryPressure::Normal),
            Some(EvictionReason::IdleTimeout)
        );
    }

    #[test]
    fn zero_idle_timeout_is_not_representable() {
        assert!(IdleTimeoutMillis::new(0).is_err());
    }

    #[test]
    fn strict_manifest_derives_narrow_source_free_profiles() {
        let long_text = exact_manifest(&[
            (100, WarmupTopology::TextOnly),
            (685, WarmupTopology::TextOnly),
        ]);
        let plan = WeightResidencyPlan::derive(&long_text, RequestAdmissionPolicy::StrictWarmup)
            .expect("long text plan");
        assert_eq!(plan.profile, WgslWeightProfile::LongTextPreparedOnly);
        assert_eq!(plan.minimum_latent_frames, 100);
        assert_eq!(plan.maximum_latent_frames, 685);
        assert!(!plan.resident_layouts.contains(&WeightLayout::QkvGateSource));
        assert!(
            !plan
                .resident_layouts
                .contains(&WeightLayout::AttentionOutputSource)
        );

        let long_all = exact_manifest(&[
            (100, WarmupTopology::TextOnly),
            (489, WarmupTopology::Designed),
            (685, WarmupTopology::PreparedClone),
        ]);
        let plan = WeightResidencyPlan::derive(&long_all, RequestAdmissionPolicy::StrictWarmup)
            .expect("long all-voice plan");
        assert_eq!(plan.profile, WgslWeightProfile::LongAllVoicePreparedOnly);
        assert_eq!(
            plan.topologies,
            vec![
                WarmupTopology::TextOnly,
                WarmupTopology::Designed,
                WarmupTopology::PreparedClone
            ]
        );
        assert!(plan.resident_layouts.contains(&WeightLayout::SwiGluFused));
        assert!(
            !plan
                .resident_layouts
                .contains(&WeightLayout::SwiGluInterleaved)
        );
        assert_eq!(
            plan.layout_set()
                .expect("valid long-all layout set")
                .as_slice(),
            plan.resident_layouts.as_slice()
        );

        let exact_489 = exact_manifest(&[(489, WarmupTopology::Designed)]);
        let exact_plan = WeightResidencyPlan::derive_for_routes(
            &exact_489,
            RequestAdmissionPolicy::StrictWarmup,
            &crate::ResolvedRouteTable::built_in(crate::BuiltInRouteProfile::NvidiaRtx),
        )
        .expect("exact S489 NVIDIA plan");
        assert!(
            exact_plan
                .resident_layouts
                .contains(&WeightLayout::QkvGateRow)
        );
        assert!(
            !exact_plan
                .resident_layouts
                .contains(&WeightLayout::QkvGateColumn),
            "B1/B3 S489 both select the row-consuming handwritten projection"
        );
        assert!(
            exact_plan
                .resident_layouts
                .contains(&WeightLayout::QkNormPacked),
            "projection-direct packed K/V materialization owns packed Q/K norm weights"
        );
        assert!(
            !exact_plan
                .resident_layouts
                .contains(&WeightLayout::AttentionOutputSource)
        );
        assert!(
            !exact_plan
                .resident_layouts
                .contains(&WeightLayout::MlpContractSource)
        );

        let exact_685 = exact_manifest(&[(685, WarmupTopology::PreparedClone)]);
        let exact_plan = WeightResidencyPlan::derive_for_routes(
            &exact_685,
            RequestAdmissionPolicy::StrictWarmup,
            &crate::ResolvedRouteTable::built_in(crate::BuiltInRouteProfile::NvidiaRtx),
        )
        .expect("exact S685 NVIDIA plan");
        assert!(
            exact_plan
                .resident_layouts
                .contains(&WeightLayout::QkvGateColumn),
            "B3 S685 uses the generic long projection and requires its tuned column layout"
        );
    }

    #[test]
    fn direct_post_sdpa_route_adds_its_packed_weight_without_hiding_fallback() {
        let problem = crate::RouteProblem::new(3, 489).unwrap();
        let profile = crate::UnsealedRouteProfile::candidate(
            crate::BuiltInRouteProfile::Portable,
            problem,
            crate::RouteChoice::PostSdpa(crate::PostSdpaRoute::DirectOutputResidual),
        );
        let routes = crate::ResolvedRouteTable::from_unsealed_profile(&profile).unwrap();
        let manifest = exact_manifest(&[(489, WarmupTopology::Designed)]);
        let plan = WeightResidencyPlan::derive_for_routes(
            &manifest,
            RequestAdmissionPolicy::StrictWarmup,
            &routes,
        )
        .unwrap();
        assert!(
            plan.resident_layouts
                .contains(&WeightLayout::AttentionOutputPacked),
            "the direct projection owns a packed wo dependency"
        );
        assert!(
            plan.resident_layouts
                .contains(&WeightLayout::AttentionOutputSource),
            "the portable B1/fallback route must remain representable"
        );
    }

    #[test]
    fn exact_cubek_contract_keeps_only_its_column_major_representation() {
        let b1 = crate::RouteProblem::new(1, 489).unwrap();
        let b3 = crate::RouteProblem::new(3, 489).unwrap();
        let mut profile = crate::UnsealedRouteProfile::candidate(
            crate::BuiltInRouteProfile::NvidiaRtx,
            b1,
            crate::RouteChoice::MlpContract(
                crate::route_autotune::MlpContractRoute::CubeKUnitMinResidualColumn,
            ),
        );
        for (problem, choice) in [
            (
                b1,
                crate::RouteChoice::MlpContractWeight(
                    crate::MlpContractWeightRoute::CubeKColumnFlat,
                ),
            ),
            (
                b3,
                crate::RouteChoice::MlpContract(
                    crate::route_autotune::MlpContractRoute::CubeKUnitMinResidualColumn,
                ),
            ),
            (
                b3,
                crate::RouteChoice::MlpContractWeight(
                    crate::MlpContractWeightRoute::CubeKColumnFlat,
                ),
            ),
        ] {
            profile
                .overrides
                .push(crate::RouteOverride { problem, choice });
        }
        let routes = crate::ResolvedRouteTable::from_unsealed_profile(&profile).unwrap();
        let manifest = exact_manifest(&[(489, WarmupTopology::Designed)]);
        let plan = WeightResidencyPlan::derive_for_routes(
            &manifest,
            RequestAdmissionPolicy::StrictWarmup,
            &routes,
        )
        .unwrap();
        assert!(
            plan.resident_layouts
                .contains(&WeightLayout::MlpContractCubeKColumn)
        );
        assert!(
            !plan
                .resident_layouts
                .contains(&WeightLayout::MlpContractPacked)
        );
        assert!(
            !plan
                .resident_layouts
                .contains(&WeightLayout::MlpContractSource)
        );
    }

    #[test]
    fn every_selectable_direct_materialization_route_retains_qk_norm_weights() {
        use crate::AttentionMaterializationRoute as R;

        for route in [
            R::DirectPackedKv,
            R::ProjectionDirectPackedKv,
            R::CubeKProjectionDirectPackedKv,
        ] {
            let problem = crate::RouteProblem::new(3, 489).unwrap();
            let profile = crate::UnsealedRouteProfile::candidate(
                crate::BuiltInRouteProfile::NvidiaRtx,
                problem,
                crate::RouteChoice::AttentionMaterialization(route),
            );
            let routes = crate::ResolvedRouteTable::from_unsealed_profile(&profile).unwrap();
            let manifest = exact_manifest(&[(489, WarmupTopology::Designed)]);
            let plan = WeightResidencyPlan::derive_for_routes(
                &manifest,
                RequestAdmissionPolicy::StrictWarmup,
                &routes,
            )
            .unwrap();
            assert!(
                plan.resident_layouts.contains(&WeightLayout::QkNormPacked),
                "{route:?} must retain its packed Q/K norm dependency"
            );
            assert_eq!(
                plan.resident_layouts.contains(&WeightLayout::QkvGateColumn),
                route == R::CubeKProjectionDirectPackedKv,
                "only the CubeK accumulator scatter owns the column-major QKV dependency"
            );
        }
    }

    #[test]
    fn layout_set_rejects_a_missing_projection_family() {
        let error = WeightLayoutSet::new([
            WeightLayout::QkvGateRow,
            WeightLayout::SwiGluFused,
            WeightLayout::AttentionOutputPacked,
        ])
        .expect_err("MLP contraction has no representation");
        assert!(error.to_string().contains("without a representation"));
    }

    #[test]
    fn manifest_derivation_falls_back_when_future_requests_or_b4_are_admitted() {
        let combined = exact_manifest(&[(489, WarmupTopology::DesignedAndClone)]);
        let strict = WeightResidencyPlan::derive(&combined, RequestAdmissionPolicy::StrictWarmup)
            .expect("strict combined plan");
        assert_eq!(strict.profile, WgslWeightProfile::ProductionPrepared);

        let text = exact_manifest(&[(489, WarmupTopology::TextOnly)]);
        let dynamic = WeightResidencyPlan::derive(&text, RequestAdmissionPolicy::CompileOnDemand)
            .expect("dynamic plan");
        assert_eq!(dynamic.profile, WgslWeightProfile::ProductionPrepared);
        assert_eq!(dynamic.basis, WeightResidencyBasis::CompileOnDemandFallback);
        assert!(
            dynamic
                .resident_layouts
                .contains(&WeightLayout::AttentionOutputSource)
        );
    }

    #[test]
    fn exact_112_manifest_distinguishes_packed_and_b4_safe_profiles() {
        let b1_b3 = exact_manifest(&[
            (112, WarmupTopology::TextOnly),
            (112, WarmupTopology::Designed),
            (112, WarmupTopology::PreparedClone),
        ]);
        assert_eq!(
            WeightResidencyPlan::derive(&b1_b3, RequestAdmissionPolicy::StrictWarmup)
                .expect("fixed packed plan")
                .profile,
            WgslWeightProfile::Fixed112PackedOnly
        );
        let b4 = exact_manifest(&[(112, WarmupTopology::DesignedAndClone)]);
        assert_eq!(
            WeightResidencyPlan::derive(&b4, RequestAdmissionPolicy::StrictWarmup)
                .expect("fixed B4 plan")
                .profile,
            WgslWeightProfile::Fixed112OneLayout
        );
    }
}
