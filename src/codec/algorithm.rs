//! Codec execution policies shared by production and diagnostic paths.

#[cfg(feature = "profile")]
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::BufWriter,
    path::Path,
};

#[cfg(feature = "profile")]
use serde_json::Value;

/// k=7 convolution policy used by the WGPU codec decoder.
///
/// [`Self::AccuracyApproved`] is the production policy: F16 tensors use
/// CubeCL's implicit-GEMM convolution, while F32 tensors retain the established
/// packed-residue WGSL route. The explicit variants exist for differential
/// profiling and regression tests.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CodecK7Algorithm {
    /// Select the accuracy-approved implementation for the tensor dtype,
    /// including geometry-aware multi-row tiling for wide F16 convolutions.
    #[default]
    AccuracyApproved,
    /// Force the established packed-residue WGSL implementation.
    PackedResidue,
    /// Force Burn/CubeCL implicit-GEMM without materialized im2col.
    CubeClImplicitGemm,
    /// Diagnostic candidate: prepare a single physical OKI allocation and
    /// retain logical OIK only as a stride view.
    #[cfg(feature = "profile")]
    CubeClImplicitGemmSingleStorage,
    /// Use a separately prepared physical OKI weight while retaining the
    /// source OIK parameter for same-model differential profiling.
    #[cfg(feature = "profile")]
    CubeClImplicitGemmPreparedWeight(PreparedK7WeightPolicy),
    /// Consume the logical OIK-backed OKI stride view directly, without a
    /// layout copy or persistent duplicate.
    #[cfg(feature = "profile")]
    CubeClImplicitGemmDirectOik,
    /// Stage contiguous NHWC channel vectors into a shared k=7 halo and
    /// consume checkpoint-native OIK weights without a layout-copy dispatch.
    #[cfg(feature = "profile")]
    CubeClImplicitGemmK7Halo,
    /// Read each physical k=7 halo channel vector once and fan it directly
    /// into the existing CubeK stage without an intermediate shared halo.
    #[cfg(feature = "profile")]
    CubeClImplicitGemmK7Fanout,
    /// Use CubeK's generic multi-row CMMA blueprint while retaining the
    /// production weight materialization and fused Snake epilogue.
    #[cfg(feature = "profile")]
    CubeClImplicitGemmMultiRows,
    /// Select CubeK multi-row tiling only when the output matrix has at least
    /// as many rows as columns and retains a wide output-channel dimension.
    #[cfg(feature = "profile")]
    CubeClImplicitGemmGeometrySelectedMultiRows,
    /// Let CubeCL benchmark the semantically identical single-row and
    /// multi-row CubeK plans for each exact k=7 problem, then persist the
    /// selected plan in the configured device environment.
    #[cfg(feature = "profile")]
    CubeClImplicitGemmAutotuned,
    /// Dispatch a selector choice resolved once from a persisted tuning
    /// record during codec preparation. The request path performs no tuner
    /// lookup or mutex acquisition.
    #[cfg(feature = "profile")]
    CubeClImplicitGemmPreparedSelector,
    /// Replace per-output Snake division with a prepared f32 reciprocal while
    /// retaining the same convolution and geometry policy.
    #[cfg(feature = "profile")]
    CubeClImplicitGemmPreparedEpilogue,
    /// Keep prepared activations in NHWC between pointwise and k=7 stages.
    #[cfg(feature = "profile")]
    CubeClImplicitGemmInputLayoutFused,
    /// Keep the historical NHWC-to-NCHW copy before standalone Snake.
    #[cfg(feature = "profile")]
    CubeClImplicitGemmMaterialized,
    /// Force the asynchronous cyclic CMMA implicit-GEMM routine.
    #[cfg(feature = "profile")]
    CubeClImplicitGemmAsync,
    /// Force the synchronous strided CMMA implicit-GEMM routine.
    #[cfg(feature = "profile")]
    CubeClImplicitGemmSyncStrided,
    /// Force the asynchronous strided CMMA implicit-GEMM routine.
    #[cfg(feature = "profile")]
    CubeClImplicitGemmAsyncStrided,
}

/// Exact fixed-shape k=7 selector key used by a prepared codec session.
#[cfg(feature = "profile")]
#[derive(
    Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, serde::Deserialize, serde::Serialize,
)]
pub struct K7SelectorProblem {
    pub output_length: usize,
    pub output_channels: usize,
    pub dilation: usize,
}

/// Generic CubeK selector policy chosen by a fresh autotune campaign.
#[cfg(feature = "profile")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum K7SelectorChoice {
    SingleRow,
    MultiRow,
    SingleNoSwizzle,
    SingleAutoPartition,
    SingleDoublePartition,
    SingleNoSwizzleAutoPartition,
}

#[cfg(feature = "profile")]
impl K7SelectorChoice {
    pub const ALL: [Self; 6] = [
        Self::SingleRow,
        Self::MultiRow,
        Self::SingleNoSwizzle,
        Self::SingleAutoPartition,
        Self::SingleDoublePartition,
        Self::SingleNoSwizzleAutoPartition,
    ];

    fn from_candidate_name(name: &str) -> Option<Self> {
        match name {
            "production-sync-cyclic-single-row-v2" => Some(Self::SingleRow),
            "production-sync-cyclic-multi-row-v2" => Some(Self::MultiRow),
            "sync-cyclic-single-no-swizzle-v1" => Some(Self::SingleNoSwizzle),
            "sync-cyclic-single-auto-partition-v1" => Some(Self::SingleAutoPartition),
            "sync-cyclic-single-double-partition-v1" => Some(Self::SingleDoublePartition),
            "sync-cyclic-single-no-swizzle-auto-partition-v1" => {
                Some(Self::SingleNoSwizzleAutoPartition)
            }
            _ => None,
        }
    }

    fn candidate_name(self) -> &'static str {
        match self {
            Self::SingleRow => "production-sync-cyclic-single-row-v2",
            Self::MultiRow => "production-sync-cyclic-multi-row-v2",
            Self::SingleNoSwizzle => "sync-cyclic-single-no-swizzle-v1",
            Self::SingleAutoPartition => "sync-cyclic-single-auto-partition-v1",
            Self::SingleDoublePartition => "sync-cyclic-single-double-partition-v1",
            Self::SingleNoSwizzleAutoPartition => "sync-cyclic-single-no-swizzle-auto-partition-v1",
        }
    }
}

/// Typed selector vector parsed from CubeCL 0.11's machine-readable JSONL.
#[cfg(feature = "profile")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct K7SelectorManifest {
    selections: BTreeMap<K7SelectorProblem, K7SelectorChoice>,
}

#[cfg(feature = "profile")]
#[derive(serde::Deserialize, serde::Serialize)]
struct StoredK7SelectorManifest {
    schema: u16,
    selections: Vec<StoredK7SelectorEntry>,
}

#[cfg(feature = "profile")]
#[derive(serde::Deserialize, serde::Serialize)]
struct StoredK7SelectorEntry {
    problem: K7SelectorProblem,
    choice: K7SelectorChoice,
}

#[cfg(feature = "profile")]
impl K7SelectorManifest {
    /// Geometry control for every residual in the released four-block decoder.
    pub fn released_decoder_geometry(latent_frames: usize) -> crate::Result<Self> {
        let mut selections = BTreeMap::new();
        let mut output_length = latent_frames;
        for (stride, output_channels) in [(12, 768), (10, 384), (8, 192), (2, 96)] {
            output_length = output_length.checked_mul(stride).ok_or_else(|| {
                crate::IrodoriError::Config(format!(
                    "released codec selector length overflow at {output_length} * {stride}"
                ))
            })?;
            let choice = if output_length >= output_channels && output_channels >= 384 {
                K7SelectorChoice::MultiRow
            } else {
                K7SelectorChoice::SingleRow
            };
            for dilation in [1, 3, 9] {
                selections.insert(
                    K7SelectorProblem {
                        output_length,
                        output_channels,
                        dilation,
                    },
                    choice,
                );
            }
        }
        Ok(Self { selections })
    }

    /// Replace one covered choice while keeping the manifest complete.
    pub fn with_selection(
        mut self,
        problem: K7SelectorProblem,
        choice: K7SelectorChoice,
    ) -> crate::Result<Self> {
        let slot = self.selections.get_mut(&problem).ok_or_else(|| {
            crate::IrodoriError::Config(format!(
                "cannot override uncovered k7 selector problem {problem:?}"
            ))
        })?;
        *slot = choice;
        Ok(self)
    }

    /// Load diagnostic fresh-tune output. Production callers must first seal
    /// the same complete selection vector with `approve_v4_autotune`.
    pub fn from_cubecl_record(path: &Path) -> crate::Result<Self> {
        Self::from_cubecl_record_with_minimum_improvement(path, 0.02)
    }

    /// Load a selector while retaining the geometry policy unless the tuned
    /// candidate improves its median by at least `minimum_improvement`.
    pub fn from_cubecl_record_with_minimum_improvement(
        path: &Path,
        minimum_improvement: f64,
    ) -> crate::Result<Self> {
        if !minimum_improvement.is_finite() || !(0.0..1.0).contains(&minimum_improvement) {
            return Err(crate::IrodoriError::Config(format!(
                "k7 selector minimum improvement must be finite and in [0, 1), got {minimum_improvement}"
            )));
        }
        let source = fs::read_to_string(path)?;
        let mut selections = BTreeMap::new();
        for line in source.lines().filter(|line| !line.trim().is_empty()) {
            let record: Value = serde_json::from_str(line).map_err(|error| {
                crate::IrodoriError::Cache(format!(
                    "invalid CubeCL autotune record {}: {error}",
                    path.display()
                ))
            })?;
            let Some(key) = record
                .get("key")
                .filter(|key| key.get("schema").and_then(Value::as_u64) == Some(2))
            else {
                continue;
            };
            let fastest = record
                .get("fastest_index")
                .and_then(Value::as_u64)
                .and_then(|index| usize::try_from(index).ok())
                .ok_or_else(|| {
                    crate::IrodoriError::Cache("k7 autotune record has no fastest index".into())
                })?;
            let usize_field = |name: &str| {
                key.get(name)
                    .and_then(Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| {
                        crate::IrodoriError::Cache(format!(
                            "k7 autotune key is missing numeric {name}"
                        ))
                    })
            };
            let problem = K7SelectorProblem {
                output_length: usize_field("output_length")?,
                output_channels: usize_field("output_channels")?,
                dilation: usize_field("dilation")?,
            };
            let outcomes = record
                .get("results")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|result| result.pointer("/outcome/Ok"));
            let fastest_outcome = outcomes
                .clone()
                .find(|outcome| {
                    outcome.get("index").and_then(Value::as_u64) == u64::try_from(fastest).ok()
                })
                .ok_or_else(|| {
                    crate::IrodoriError::Cache(format!(
                        "k7 autotune record fastest candidate {fastest} has no successful result"
                    ))
                })?;
            let fastest_choice = fastest_outcome
                .get("name")
                .and_then(Value::as_str)
                .and_then(K7SelectorChoice::from_candidate_name)
                .ok_or_else(|| {
                    crate::IrodoriError::Cache(format!(
                        "k7 autotune record fastest candidate {fastest} is unsupported"
                    ))
                })?;
            let geometry_choice = if problem.output_length >= problem.output_channels
                && problem.output_channels >= 384
            {
                K7SelectorChoice::MultiRow
            } else {
                K7SelectorChoice::SingleRow
            };
            let median_nanos = |outcome: &Value| {
                let median = outcome.pointer("/computation/median")?;
                let seconds = median.get("secs")?.as_u64()? as u128;
                let nanos = median.get("nanos")?.as_u64()? as u128;
                seconds.checked_mul(1_000_000_000)?.checked_add(nanos)
            };
            let candidate = if fastest_choice == geometry_choice {
                fastest_choice
            } else {
                let fastest_nanos = median_nanos(fastest_outcome).ok_or_else(|| {
                    crate::IrodoriError::Cache(format!(
                        "k7 tuned candidate has no median for {problem:?}"
                    ))
                })?;
                let geometry_outcome = outcomes
                    .clone()
                    .find(|outcome| {
                        outcome.get("name").and_then(Value::as_str)
                            == Some(geometry_choice.candidate_name())
                    })
                    .ok_or_else(|| {
                        crate::IrodoriError::Cache(format!(
                            "k7 geometry control is absent for {problem:?}"
                        ))
                    })?;
                let geometry_nanos = median_nanos(geometry_outcome).ok_or_else(|| {
                    crate::IrodoriError::Cache(format!(
                        "k7 geometry control has no median for {problem:?}"
                    ))
                })?;
                let relative_improvement =
                    geometry_nanos.saturating_sub(fastest_nanos) as f64 / geometry_nanos as f64;
                if relative_improvement >= minimum_improvement {
                    fastest_choice
                } else {
                    geometry_choice
                }
            };
            if let Some(previous) = selections.insert(problem, candidate)
                && previous != candidate
            {
                return Err(crate::IrodoriError::Cache(format!(
                    "conflicting k7 autotune decisions for {problem:?}"
                )));
            }
        }
        if selections.is_empty() {
            return Err(crate::IrodoriError::Cache(format!(
                "no k7 selector decisions found in {}",
                path.display()
            )));
        }
        Ok(Self { selections })
    }

    pub fn selection(&self, problem: K7SelectorProblem) -> crate::Result<K7SelectorChoice> {
        self.selections.get(&problem).copied().ok_or_else(|| {
            crate::IrodoriError::Cache(format!("k7 selector manifest does not cover {problem:?}"))
        })
    }

    pub fn selections(&self) -> impl Iterator<Item = (K7SelectorProblem, K7SelectorChoice)> + '_ {
        self.selections
            .iter()
            .map(|(problem, choice)| (*problem, *choice))
    }

    /// Verify that this vector covers exactly the twelve residual k=7
    /// problems of one released decoder shape.
    pub(crate) fn validate_decoder_shape(&self, latent_frames: usize) -> crate::Result<()> {
        let expected = Self::released_decoder_geometry(latent_frames)?;
        if self.selections.len() != expected.selections.len()
            || self.selections.keys().ne(expected.selections.keys())
        {
            return Err(crate::IrodoriError::Cache(format!(
                "k7 selector does not cover the exact released decoder geometry for {latent_frames} latent frames"
            )));
        }
        Ok(())
    }

    pub(crate) fn from_selections(
        selections: impl IntoIterator<Item = (K7SelectorProblem, K7SelectorChoice)>,
    ) -> crate::Result<Self> {
        let mut resolved = BTreeMap::new();
        for (problem, choice) in selections {
            if let Some(previous) = resolved.insert(problem, choice)
                && previous != choice
            {
                return Err(crate::IrodoriError::Cache(format!(
                    "conflicting k7 selector decisions for {problem:?}"
                )));
            }
        }
        if resolved.is_empty() {
            return Err(crate::IrodoriError::Cache(
                "k7 selector vector is empty".into(),
            ));
        }
        Ok(Self {
            selections: resolved,
        })
    }

    pub fn from_stored_file(path: &Path) -> crate::Result<Self> {
        let source = fs::read(path)?;
        let stored: StoredK7SelectorManifest =
            serde_json::from_slice(&source).map_err(|error| {
                crate::IrodoriError::Cache(format!(
                    "invalid stored k7 selector manifest {}: {error}",
                    path.display()
                ))
            })?;
        if stored.schema != 1 {
            return Err(crate::IrodoriError::Cache(format!(
                "unsupported stored k7 selector schema {} in {}",
                stored.schema,
                path.display()
            )));
        }
        let mut selections = BTreeMap::new();
        for entry in stored.selections {
            if let Some(previous) = selections.insert(entry.problem, entry.choice)
                && previous != entry.choice
            {
                return Err(crate::IrodoriError::Cache(format!(
                    "conflicting stored k7 selector decisions for {:?}",
                    entry.problem
                )));
            }
        }
        if selections.is_empty() {
            return Err(crate::IrodoriError::Cache(format!(
                "stored k7 selector manifest is empty: {}",
                path.display()
            )));
        }
        Ok(Self { selections })
    }

    /// Persist a completed selection vector without overwriting evidence.
    pub fn write_new(&self, path: &Path) -> crate::Result<()> {
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        let stored = StoredK7SelectorManifest {
            schema: 1,
            selections: self
                .selections()
                .map(|(problem, choice)| StoredK7SelectorEntry { problem, choice })
                .collect(),
        };
        serde_json::to_writer_pretty(BufWriter::new(file), &stored).map_err(|error| {
            crate::IrodoriError::Cache(format!(
                "failed to write k7 selector manifest {}: {error}",
                path.display()
            ))
        })
    }
}

/// Generic residency policy for prepared k=7 weights.
#[cfg(feature = "profile")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PreparedK7WeightPolicy {
    min_bytes: usize,
}

#[cfg(feature = "profile")]
impl PreparedK7WeightPolicy {
    pub const fn all() -> Self {
        Self { min_bytes: 0 }
    }

    pub const fn at_least_bytes(min_bytes: usize) -> Self {
        Self { min_bytes }
    }

    pub const fn accepts(self, bytes: usize) -> bool {
        bytes >= self.min_bytes
    }
}

/// Physical-layout and GPU-copy receipt for one decoder k=7 weight.
#[cfg(feature = "profile")]
#[derive(Clone, Debug)]
pub struct K7WeightRepackReceipt {
    pub label: &'static str,
    pub source_oik_shape: [usize; 3],
    pub logical_oki_strides: [usize; 3],
    pub materialized_oki_strides: [usize; 3],
    pub logical_rhs_vector_size: usize,
    pub materialized_rhs_vector_size: usize,
    pub materialized_bytes: usize,
    pub device_duration_ms: f64,
    pub used_device_timestamps: bool,
}

/// 1×1 convolution policy used by codec differential profiling.
#[cfg(feature = "profile")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CodecPointwiseAlgorithm {
    /// Use the production packed-matmul route.
    #[default]
    AccuracyApproved,
    /// Force the production packed-matmul route.
    PackedMatmul,
    /// Use CubeCL implicit-GEMM without materialized im2col.
    CubeClImplicitGemm,
    /// Profile-only CubeK projection whose accumulator-domain store adds the
    /// shortcut and writes raw NCL plus next-Snake NHWC in one dispatch.
    CubeClAccumulatorStore,
    /// Retain the accumulator store only at the eight intra-block boundaries;
    /// block-final pointwise projections use the packed control route.
    CubeClAccumulatorPairOnly,
    /// Retain the accumulator pair store but force CubeK's single-row tiling.
    CubeClAccumulatorPairSingleRow,
    /// Use multi-row tiling only when the output matrix is at least 64 times
    /// taller than its channel width.
    CubeClAccumulatorPairTallRows,
    /// Preserve the tall-matrix row policy while selecting a cache-key-visible
    /// CubeK blueprint policy for single-row pointwise problems.
    CubeClAccumulatorPairSelector(K7SelectorChoice),
}

/// Decoder-stem policy used only for differential profiling.
#[cfg(feature = "profile")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CodecStemAlgorithm {
    /// Use the accuracy-approved direct WGSL convolution.
    #[default]
    AccuracyApproved,
    /// Use Burn/CubeCL's portable convolution implementation.
    Burn,
}

/// Cross-block pointwise/Snake fusion policy for differential profiling.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CodecCrossBlockFusion {
    #[default]
    Standalone,
    /// Fuse the C384 output from decoder block 1 into block 2's input Snake.
    #[cfg(feature = "profile")]
    OutputC384,
    /// Fuse the C192 output from decoder block 2 into block 3's input Snake.
    #[cfg(feature = "profile")]
    OutputC192,
    OutputC384AndC192,
}

impl CodecCrossBlockFusion {
    pub(crate) const fn fuses_c384(self) -> bool {
        match self {
            Self::OutputC384AndC192 => true,
            #[cfg(feature = "profile")]
            Self::OutputC384 => true,
            _ => false,
        }
    }

    pub(crate) const fn fuses_c192(self) -> bool {
        match self {
            Self::OutputC384AndC192 => true,
            #[cfg(feature = "profile")]
            Self::OutputC192 => true,
            _ => false,
        }
    }
}

/// Producer-side fusion between cached-col2im ConvTranspose finalizers and
/// the first residual unit's Snake/layout preparation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CodecConvTransposeSnakeFusion {
    /// Retain the raw finalizer followed by a standalone Snake dispatch.
    #[default]
    Standalone,
    #[cfg(feature = "profile")]
    CachedCol2ImCase1,
    #[cfg(feature = "profile")]
    CachedCol2ImCase2,
    #[cfg(feature = "profile")]
    CachedCol2ImCase3,
    /// Emit raw NCL and post-storage-cast activated NHWC from one finalizer.
    #[cfg(feature = "profile")]
    CachedCol2ImDualOutput,
}

/// Physical shortcut state retained between decoder residual units.
#[cfg(feature = "profile")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CodecResidualStateLayout {
    /// Accuracy-approved NCL shortcut state used by production.
    #[default]
    ProductionNcl,
    /// Keep shortcut and prepared activation NHWC within every block.
    NhwcWithinBlock,
}

/// Profile-only CubeK row-partition choice for the remaining C768 decoder
/// block boundary. This is explicit input to the graph tuner, not a hidden
/// device-name heuristic.
#[cfg(feature = "profile")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum C768CrossBlockRows {
    Single,
    Multi,
}

impl CodecConvTransposeSnakeFusion {
    #[cfg(feature = "profile")]
    pub(crate) const fn fuses_cached_col2im(
        self,
        case: crate::kernels::conv_transpose1d_cached_col2im::CachedCol2ImCase,
    ) -> bool {
        match self {
            #[cfg(feature = "profile")]
            Self::CachedCol2ImDualOutput => true,
            #[cfg(feature = "profile")]
            Self::CachedCol2ImCase1 => matches!(
                case,
                crate::kernels::conv_transpose1d_cached_col2im::CachedCol2ImCase::Case1
            ),
            #[cfg(feature = "profile")]
            Self::CachedCol2ImCase2 => matches!(
                case,
                crate::kernels::conv_transpose1d_cached_col2im::CachedCol2ImCase::Case2
            ),
            #[cfg(feature = "profile")]
            Self::CachedCol2ImCase3 => matches!(
                case,
                crate::kernels::conv_transpose1d_cached_col2im::CachedCol2ImCase::Case3
            ),
            Self::Standalone => false,
        }
    }
}

/// Complete codec algorithm selection for one differential run.
#[cfg(feature = "profile")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CodecAlgorithmPlan {
    pub stem: CodecStemAlgorithm,
    pub k7: CodecK7Algorithm,
    pub pointwise: CodecPointwiseAlgorithm,
}

#[cfg(feature = "profile")]
impl CodecAlgorithmPlan {
    pub const fn new(k7: CodecK7Algorithm, pointwise: CodecPointwiseAlgorithm) -> Self {
        Self {
            stem: CodecStemAlgorithm::AccuracyApproved,
            k7,
            pointwise,
        }
    }

    pub const fn with_stem(mut self, stem: CodecStemAlgorithm) -> Self {
        self.stem = stem;
        self
    }

    pub const fn accuracy_approved() -> Self {
        Self::new(
            CodecK7Algorithm::AccuracyApproved,
            CodecPointwiseAlgorithm::AccuracyApproved,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::CodecK7Algorithm;
    #[cfg(feature = "profile")]
    use super::{
        CodecAlgorithmPlan, CodecPointwiseAlgorithm, CodecStemAlgorithm, K7SelectorChoice,
        K7SelectorManifest, K7SelectorProblem,
    };

    #[test]
    fn default_is_accuracy_approved_policy() {
        assert_eq!(
            CodecK7Algorithm::default(),
            CodecK7Algorithm::AccuracyApproved
        );
    }

    #[test]
    #[cfg(feature = "profile")]
    fn default_plan_has_no_experimental_algorithm() {
        assert_eq!(
            CodecAlgorithmPlan::default(),
            CodecAlgorithmPlan::accuracy_approved()
        );
        assert_eq!(
            CodecAlgorithmPlan::default().pointwise,
            CodecPointwiseAlgorithm::AccuracyApproved
        );
        assert_eq!(
            CodecAlgorithmPlan::default().stem,
            CodecStemAlgorithm::AccuracyApproved
        );
    }

    #[test]
    #[cfg(feature = "profile")]
    fn selector_manifest_ignores_generic_records_and_resolves_exact_shape() {
        use std::io::Write;

        let mut file = tempfile::NamedTempFile::new().expect("temporary selector record");
        writeln!(
            file,
            r#"{{"key":{{"definition":{{"m":64}}}},"fastest_index":0,"results":[]}}"#
        )
        .unwrap();
        writeln!(
            file,
            r#"{{"key":{{"schema":2,"dtype":"f16","batch":1,"input_length":6000,"input_channels":384,"output_length":6000,"output_channels":384,"dilation":3,"input_strides":[2304000,384,1],"weight_strides":[2688,1,7]}},"fastest_index":1,"results":[{{"outcome":{{"Ok":{{"name":"production-sync-cyclic-single-row-v2","index":0}}}}}},{{"outcome":{{"Ok":{{"name":"production-sync-cyclic-multi-row-v2","index":1}}}}}}]}}"#
        )
        .unwrap();

        let manifest = K7SelectorManifest::from_cubecl_record(file.path()).unwrap();
        assert_eq!(
            manifest
                .selection(K7SelectorProblem {
                    output_length: 6000,
                    output_channels: 384,
                    dilation: 3,
                })
                .unwrap(),
            K7SelectorChoice::MultiRow
        );
        assert!(
            manifest
                .selection(K7SelectorProblem {
                    output_length: 6000,
                    output_channels: 384,
                    dilation: 9,
                })
                .is_err()
        );
    }

    #[test]
    #[cfg(feature = "profile")]
    fn selector_manifest_rejects_noise_but_keeps_material_improvement() {
        use std::io::Write;

        fn record(fastest_nanos: u64, geometry_nanos: u64) -> String {
            serde_json::json!({
                "key": {
                    "schema": 2,
                    "dtype": "f16",
                    "batch": 1,
                    "input_length": 48_000,
                    "input_channels": 192,
                    "output_length": 48_000,
                    "output_channels": 192,
                    "dilation": 9,
                    "input_strides": [9_216_000, 192, 1],
                    "weight_strides": [1_344, 1, 7]
                },
                "fastest_index": 2,
                "results": [
                    {"outcome": {"Ok": {
                        "name": "production-sync-cyclic-single-row-v2",
                        "index": 0,
                        "computation": {"median": {"secs": 0, "nanos": geometry_nanos}}
                    }}},
                    {"outcome": {"Ok": {
                        "name": "sync-cyclic-single-no-swizzle-v1",
                        "index": 2,
                        "computation": {"median": {"secs": 0, "nanos": fastest_nanos}}
                    }}}
                ]
            })
            .to_string()
        }

        for (fastest_nanos, expected) in [
            (990_000, K7SelectorChoice::SingleRow),
            (800_000, K7SelectorChoice::SingleNoSwizzle),
        ] {
            let mut file = tempfile::NamedTempFile::new().expect("temporary selector record");
            writeln!(file, "{}", record(fastest_nanos, 1_000_000)).unwrap();
            let manifest = K7SelectorManifest::from_cubecl_record(file.path()).unwrap();
            assert_eq!(
                manifest
                    .selection(K7SelectorProblem {
                        output_length: 48_000,
                        output_channels: 192,
                        dilation: 9,
                    })
                    .unwrap(),
                expected
            );
        }
    }

    #[test]
    #[cfg(feature = "profile")]
    fn stored_selector_manifest_round_trips_complete_geometry() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("selector.json");
        let problem = K7SelectorProblem {
            output_length: 48_000,
            output_channels: 192,
            dilation: 9,
        };
        let manifest = K7SelectorManifest::released_decoder_geometry(50)
            .unwrap()
            .with_selection(problem, K7SelectorChoice::SingleNoSwizzle)
            .unwrap();
        assert_eq!(manifest.selections().count(), 12);
        manifest.write_new(&path).unwrap();
        assert!(manifest.write_new(&path).is_err());
        let restored = K7SelectorManifest::from_stored_file(&path).unwrap();
        assert_eq!(
            restored.selection(problem).unwrap(),
            K7SelectorChoice::SingleNoSwizzle
        );
    }
}
