//! Codec execution policies shared by production and diagnostic paths.

/// k=7 convolution policy used by the WGPU codec decoder.
///
/// [`Self::AccuracyApproved`] is the production policy: F16 tensors use
/// CubeCL's implicit-GEMM convolution, while F32 tensors retain the established
/// packed-residue WGSL route. The explicit variants exist for differential
/// profiling and regression tests.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CodecK7Algorithm {
    /// Select the accuracy-approved implementation for the tensor dtype.
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
