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
}

#[cfg(test)]
mod tests {
    use super::CodecK7Algorithm;

    #[test]
    fn default_is_accuracy_approved_policy() {
        assert_eq!(
            CodecK7Algorithm::default(),
            CodecK7Algorithm::AccuracyApproved
        );
    }
}
