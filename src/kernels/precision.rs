//! Shared element-precision contract for handwritten WGPU kernels.
//!
//! The production default remains [`KernelFloatPrecision::F32`]. Launchers
//! derive the variant from their tensor bindings and reject mixed or unsupported
//! dtypes before allocating an output or submitting a shader.

use burn::backend::wgpu::SourceTemplate;
use burn::tensor::DType;

/// Storage precision used by a handwritten WGSL kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum KernelFloatPrecision {
    F32,
    F16,
}

impl KernelFloatPrecision {
    pub(crate) fn from_dtype(dtype: DType) -> Option<Self> {
        match dtype {
            DType::F32 => Some(Self::F32),
            DType::F16 => Some(Self::F16),
            _ => None,
        }
    }

    pub(crate) const fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
        }
    }

    pub(crate) const fn element_bytes(self) -> usize {
        match self {
            Self::F32 => size_of::<f32>(),
            Self::F16 => size_of::<half::f16>(),
        }
    }

    pub(crate) fn source(
        self,
        f32_source: &'static str,
        f16_source: &'static str,
    ) -> SourceTemplate {
        SourceTemplate::new(match self {
            Self::F32 => f32_source,
            Self::F16 => f16_source,
        })
    }
}

/// Resolve one common supported storage precision for all bindings.
pub(crate) fn common_float_precision(
    dtypes: impl IntoIterator<Item = DType>,
) -> Option<KernelFloatPrecision> {
    let mut dtypes = dtypes.into_iter();
    let precision = KernelFloatPrecision::from_dtype(dtypes.next()?)?;
    dtypes
        .all(|dtype| KernelFloatPrecision::from_dtype(dtype) == Some(precision))
        .then_some(precision)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_mixed_and_non_float_kernel_bindings() {
        assert_eq!(
            common_float_precision([DType::F16, DType::F16]),
            Some(KernelFloatPrecision::F16)
        );
        assert_eq!(
            common_float_precision([DType::F32, DType::F32]),
            Some(KernelFloatPrecision::F32)
        );
        assert_eq!(common_float_precision([DType::F16, DType::F32]), None);
        assert_eq!(common_float_precision([DType::I32]), None);
        assert_eq!(common_float_precision([]), None);
    }
}
