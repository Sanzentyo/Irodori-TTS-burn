//! Codec execution policies shared by production and diagnostic paths.

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
    /// Use CubeK's generic multi-row CMMA blueprint while retaining the
    /// production weight materialization and fused Snake epilogue.
    #[cfg(feature = "profile")]
    CubeClImplicitGemmMultiRows,
    /// Select CubeK multi-row tiling only when the output matrix has at least
    /// as many rows as columns and retains a wide output-channel dimension.
    #[cfg(feature = "profile")]
    CubeClImplicitGemmGeometrySelectedMultiRows,
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
    use super::{CodecAlgorithmPlan, CodecPointwiseAlgorithm, CodecStemAlgorithm};

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
}
