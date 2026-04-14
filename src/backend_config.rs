//! Backend configuration and runtime dispatch for Irodori-TTS.
//!
//! This module provides two complementary approaches to backend selection:
//!
//! ## 1. Runtime enum dispatch (recommended for binaries)
//!
//! [`InferenceBackendKind`] and [`TrainingBackendKind`] enumerate available GPU
//! backends. Use the [`dispatch_inference!`] and [`dispatch_training!`] macros to
//! call monomorphised generic functions — fully static dispatch, no `dyn`.
//!
//! ```ignore
//! let kind = InferenceBackendKind::CudaBf16;
//! let result = dispatch_inference!(kind, 0, |B, device| {
//!     let model = TextToLatentRfDiT::<B>::load(&device)?;
//!     model.forward(&input)
//! });
//! ```
//!
//! ## 2. Compile-time feature macros (deprecated, zero users)
//!
//! [`select_inference_backend!`] and [`select_train_backend!`] define a type alias
//! based on `backend_*` Cargo features. These macros have zero remaining users and
//! are retained only for backward compatibility. Prefer the enum dispatch approach
//! above.
//!
//! ## [`BackendConfig`] trait
//!
//! A thin supertrait over `Backend` that adds device construction and a
//! human-readable label. Implemented for all concrete backends and blanket-
//! implemented for `Autodiff<B>`.
//!
//! ## `--gpu-id` semantics
//! For CUDA-class backends (CubeCL CUDA, LibTorch), `gpu_id` selects the Nth device.
//! For WGPU, it selects the Nth discrete GPU (`WgpuDevice::DiscreteGpu(id)`).
//! For CPU-only backends (NdArray), the parameter is ignored and `Cpu` is returned.

use burn::tensor::backend::Backend;

/// Defines type alias `B` for the active inference backend based on Cargo feature flags.
///
/// # Deprecated
///
/// **Use [`dispatch_inference!`] instead.** This macro has zero remaining users
/// in the project. It is retained for backward compatibility but will be removed
/// in a future release.
///
/// Generates `compile_error!` if multiple backend features are active simultaneously.
/// Falls back to `NdArray<f32>` when no backend feature is selected.
///
/// # Usage
/// ```ignore
/// irodori_tts_burn::select_inference_backend!();
/// // Now `B` is available as the selected backend type.
/// ```
#[macro_export]
macro_rules! select_inference_backend {
    () => {
        #[cfg(any(
            all(feature = "backend_wgpu", feature = "backend_wgpu_f16"),
            all(feature = "backend_wgpu", feature = "backend_wgpu_bf16"),
            all(feature = "backend_wgpu_f16", feature = "backend_wgpu_bf16"),
            all(feature = "backend_wgpu", feature = "backend_cuda"),
            all(feature = "backend_wgpu", feature = "backend_cuda_bf16"),
            all(feature = "backend_wgpu", feature = "backend_tch"),
            all(feature = "backend_wgpu", feature = "backend_tch_bf16"),
            all(feature = "backend_wgpu_f16", feature = "backend_cuda"),
            all(feature = "backend_wgpu_f16", feature = "backend_cuda_bf16"),
            all(feature = "backend_wgpu_f16", feature = "backend_tch"),
            all(feature = "backend_wgpu_f16", feature = "backend_tch_bf16"),
            all(feature = "backend_wgpu_bf16", feature = "backend_cuda"),
            all(feature = "backend_wgpu_bf16", feature = "backend_cuda_bf16"),
            all(feature = "backend_wgpu_bf16", feature = "backend_tch"),
            all(feature = "backend_wgpu_bf16", feature = "backend_tch_bf16"),
            all(feature = "backend_cuda", feature = "backend_cuda_bf16"),
            all(feature = "backend_cuda", feature = "backend_tch"),
            all(feature = "backend_cuda", feature = "backend_tch_bf16"),
            all(feature = "backend_cuda_bf16", feature = "backend_tch"),
            all(feature = "backend_cuda_bf16", feature = "backend_tch_bf16"),
            all(feature = "backend_tch", feature = "backend_tch_bf16"),
        ))]
        compile_error!("backend_* features are mutually exclusive — select exactly one");

        #[cfg(feature = "backend_wgpu")]
        type B = burn::backend::Wgpu;
        #[cfg(feature = "backend_wgpu_f16")]
        type B = burn::backend::Wgpu<half::f16>;
        #[cfg(feature = "backend_wgpu_bf16")]
        type B = burn::backend::Wgpu<half::bf16>;
        #[cfg(feature = "backend_cuda")]
        type B = burn::backend::Cuda;
        #[cfg(feature = "backend_cuda_bf16")]
        type B = burn::backend::Cuda<half::bf16>;
        #[cfg(feature = "backend_tch")]
        type B = burn::backend::LibTorch;
        #[cfg(feature = "backend_tch_bf16")]
        type B = burn::backend::LibTorch<half::bf16>;
        #[cfg(not(any(
            feature = "backend_wgpu",
            feature = "backend_wgpu_f16",
            feature = "backend_wgpu_bf16",
            feature = "backend_cuda",
            feature = "backend_cuda_bf16",
            feature = "backend_tch",
            feature = "backend_tch_bf16",
        )))]
        type B = burn::backend::NdArray<f32>;
    };
}

/// Defines type alias `BaseB` for the active training backend (non-autodiff).
///
/// # Deprecated
///
/// **Use [`dispatch_training!`] instead.** This macro has zero remaining users
/// in the project. It is retained for backward compatibility but will be removed
/// in a future release.
///
/// Training only supports backends with `AutodiffBackend`: LibTorch, Cuda, NdArray.
/// WGPU is excluded because it lacks autodiff support in burn.
///
/// # Usage
/// ```ignore
/// irodori_tts_burn::select_train_backend!();
/// type B = burn::backend::Autodiff<BaseB>;
/// ```
#[macro_export]
macro_rules! select_train_backend {
    () => {
        #[cfg(any(
            feature = "backend_wgpu",
            feature = "backend_wgpu_f16",
            feature = "backend_wgpu_bf16",
        ))]
        compile_error!(
            "WGPU backends do not support autodiff and cannot be used for training. \
             Use backend_tch, backend_tch_bf16, backend_cuda, or backend_cuda_bf16."
        );

        #[cfg(any(
            all(feature = "backend_cuda", feature = "backend_cuda_bf16"),
            all(feature = "backend_cuda", feature = "backend_tch"),
            all(feature = "backend_cuda", feature = "backend_tch_bf16"),
            all(feature = "backend_cuda_bf16", feature = "backend_tch"),
            all(feature = "backend_cuda_bf16", feature = "backend_tch_bf16"),
            all(feature = "backend_tch", feature = "backend_tch_bf16"),
        ))]
        compile_error!("only one backend feature may be selected at a time");

        #[cfg(feature = "backend_cuda")]
        type BaseB = burn::backend::Cuda;
        #[cfg(feature = "backend_cuda_bf16")]
        type BaseB = burn::backend::Cuda<half::bf16>;
        #[cfg(feature = "backend_tch")]
        type BaseB = burn::backend::LibTorch;
        #[cfg(feature = "backend_tch_bf16")]
        type BaseB = burn::backend::LibTorch<half::bf16>;
        #[cfg(not(any(
            feature = "backend_cuda",
            feature = "backend_cuda_bf16",
            feature = "backend_tch",
            feature = "backend_tch_bf16",
        )))]
        type BaseB = burn::backend::NdArray;
    };
}

/// Extended backend trait for CLI/binary use.
///
/// Provides device construction and a human-readable label.
/// All implementations live in this module; the library model code
/// remains fully generic over `Backend` and never depends on this trait.
pub trait BackendConfig: Backend {
    /// Construct the device for this backend, optionally selecting by index.
    ///
    /// For GPU-capable backends this maps to "device N" (CUDA ordinal, WGPU
    /// discrete GPU ordinal, etc.).  For CPU backends the argument is ignored.
    fn device_from_id(gpu_id: u32) -> Self::Device;

    /// Return a CPU-only device for this backend.
    ///
    /// Useful for benchmarks that must run on CPU for fair comparison.
    /// For backends without a dedicated CPU mode (WGPU, CubeCL CUDA) this
    /// falls back to the default device.
    fn cpu_device() -> Self::Device;

    /// A short human-readable label shown in benchmark / CLI output.
    fn backend_label() -> &'static str;
}

// ---------------------------------------------------------------------------
// NdArray (CPU — always available)
// ---------------------------------------------------------------------------

impl BackendConfig for burn::backend::NdArray {
    fn device_from_id(_gpu_id: u32) -> Self::Device {
        burn::backend::ndarray::NdArrayDevice::Cpu
    }

    fn cpu_device() -> Self::Device {
        burn::backend::ndarray::NdArrayDevice::Cpu
    }

    fn backend_label() -> &'static str {
        "NdArray (CPU, f32)"
    }
}

// ---------------------------------------------------------------------------
// Wgpu (always compiled in our crate)
// ---------------------------------------------------------------------------

impl BackendConfig for burn::backend::Wgpu {
    fn device_from_id(gpu_id: u32) -> Self::Device {
        burn::backend::wgpu::WgpuDevice::DiscreteGpu(gpu_id as usize)
    }

    fn cpu_device() -> Self::Device {
        burn::backend::wgpu::WgpuDevice::DefaultDevice
    }

    fn backend_label() -> &'static str {
        "Wgpu (f32)"
    }
}

impl BackendConfig for burn::backend::Wgpu<half::f16> {
    fn device_from_id(gpu_id: u32) -> Self::Device {
        burn::backend::wgpu::WgpuDevice::DiscreteGpu(gpu_id as usize)
    }

    fn cpu_device() -> Self::Device {
        burn::backend::wgpu::WgpuDevice::DefaultDevice
    }

    fn backend_label() -> &'static str {
        "Wgpu (f16)"
    }
}

impl BackendConfig for burn::backend::Wgpu<half::bf16> {
    fn device_from_id(gpu_id: u32) -> Self::Device {
        burn::backend::wgpu::WgpuDevice::DiscreteGpu(gpu_id as usize)
    }

    fn cpu_device() -> Self::Device {
        burn::backend::wgpu::WgpuDevice::DefaultDevice
    }

    fn backend_label() -> &'static str {
        "Wgpu (bf16)"
    }
}

// ---------------------------------------------------------------------------
// CubeCL CUDA (always compiled in our crate)
// ---------------------------------------------------------------------------

impl BackendConfig for burn::backend::Cuda {
    fn device_from_id(gpu_id: u32) -> Self::Device {
        burn::backend::cuda::CudaDevice {
            index: gpu_id as usize,
        }
    }

    fn cpu_device() -> Self::Device {
        burn::backend::cuda::CudaDevice { index: 0 }
    }

    fn backend_label() -> &'static str {
        "Cuda (CubeCL, f32)"
    }
}

impl BackendConfig for burn::backend::Cuda<half::bf16> {
    fn device_from_id(gpu_id: u32) -> Self::Device {
        burn::backend::cuda::CudaDevice {
            index: gpu_id as usize,
        }
    }

    fn cpu_device() -> Self::Device {
        burn::backend::cuda::CudaDevice { index: 0 }
    }

    fn backend_label() -> &'static str {
        "Cuda (CubeCL, bf16)"
    }
}

// ---------------------------------------------------------------------------
// LibTorch / tch (always compiled in our crate via burn's `tch` feature)
// ---------------------------------------------------------------------------

impl BackendConfig for burn::backend::LibTorch {
    fn device_from_id(gpu_id: u32) -> Self::Device {
        burn::backend::libtorch::LibTorchDevice::Cuda(gpu_id as usize)
    }

    fn cpu_device() -> Self::Device {
        burn::backend::libtorch::LibTorchDevice::Cpu
    }

    fn backend_label() -> &'static str {
        "LibTorch (cuBLAS/FA3, f32)"
    }
}

impl BackendConfig for burn::backend::LibTorch<half::bf16> {
    fn device_from_id(gpu_id: u32) -> Self::Device {
        burn::backend::libtorch::LibTorchDevice::Cuda(gpu_id as usize)
    }

    fn cpu_device() -> Self::Device {
        burn::backend::libtorch::LibTorchDevice::Cpu
    }

    fn backend_label() -> &'static str {
        "LibTorch (cuBLAS/FA3, bf16)"
    }
}

// ---------------------------------------------------------------------------
// Blanket impl for Autodiff — delegates to the inner backend
// ---------------------------------------------------------------------------

impl<B: BackendConfig> BackendConfig for burn::backend::Autodiff<B> {
    fn device_from_id(gpu_id: u32) -> Self::Device {
        B::device_from_id(gpu_id)
    }

    fn cpu_device() -> Self::Device {
        B::cpu_device()
    }

    fn backend_label() -> &'static str {
        B::backend_label()
    }
}

// ===========================================================================
// Runtime backend dispatch (enum-based, no dynamic dispatch)
// ===========================================================================

/// Runtime-selectable inference backend.
///
/// All variants are always compiled (burn builds all backends unconditionally).
/// WGPU bf16 is excluded due to known runtime panics on most hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[serde(rename_all = "snake_case")]
pub enum InferenceBackendKind {
    /// NdArray CPU backend (f32 only). Useful for fixture validation and CI.
    #[cfg_attr(feature = "cli", value(name = "ndarray"))]
    NdArray,
    /// WGPU with f32 precision.
    #[cfg_attr(feature = "cli", value(name = "wgpu"))]
    Wgpu,
    /// WGPU with f16 precision (requires shader-f16 GPU support).
    #[cfg_attr(feature = "cli", value(name = "wgpu-f16"))]
    WgpuF16,
    /// CubeCL CUDA with f32 precision.
    #[cfg_attr(feature = "cli", value(name = "cuda"))]
    CudaF32,
    /// CubeCL CUDA with bf16 precision (Tensor Core GPUs).
    #[cfg_attr(feature = "cli", value(name = "cuda-bf16"))]
    CudaBf16,
    /// LibTorch with f32 precision (cuBLAS + FlashAttention3 via PyTorch).
    #[cfg_attr(feature = "cli", value(name = "libtorch"))]
    LibTorchF32,
    /// LibTorch with bf16 precision (cuBLAS Tensor Core + FlashAttention3).
    #[cfg_attr(feature = "cli", value(name = "libtorch-bf16"))]
    LibTorchBf16,
}

impl InferenceBackendKind {
    /// Human-readable label for logs and reports.
    pub fn label(self) -> &'static str {
        match self {
            Self::NdArray => "NdArray (CPU, f32)",
            Self::Wgpu => "Wgpu (f32)",
            Self::WgpuF16 => "Wgpu (f16)",
            Self::CudaF32 => "Cuda (CubeCL, f32)",
            Self::CudaBf16 => "Cuda (CubeCL, bf16)",
            Self::LibTorchF32 => "LibTorch (cuBLAS/FA3, f32)",
            Self::LibTorchBf16 => "LibTorch (cuBLAS/FA3, bf16)",
        }
    }

    /// Whether this backend uses reduced precision (f16/bf16).
    ///
    /// Useful for E2E tolerance selection: reduced-precision backends accumulate
    /// more floating-point error than f32 backends.
    pub fn is_reduced_precision(self) -> bool {
        matches!(self, Self::WgpuF16 | Self::CudaBf16 | Self::LibTorchBf16)
    }

    /// All available inference backend variants.
    pub fn all() -> &'static [Self] {
        &[
            Self::NdArray,
            Self::Wgpu,
            Self::WgpuF16,
            Self::CudaF32,
            Self::CudaBf16,
            Self::LibTorchF32,
            Self::LibTorchBf16,
        ]
    }
}

impl core::fmt::Display for InferenceBackendKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.label())
    }
}

/// Runtime-selectable training backend.
///
/// Only backends supporting autodiff are included. WGPU lacks autodiff support
/// in burn; NdArray is excluded because GPU training is required for practical
/// throughput. For CPU-only fallback, use [`select_train_backend!`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[serde(rename_all = "snake_case")]
pub enum TrainingBackendKind {
    /// CubeCL CUDA with f32 precision.
    #[cfg_attr(feature = "cli", value(name = "cuda"))]
    CudaF32,
    /// CubeCL CUDA with bf16 precision (Tensor Core GPUs).
    #[cfg_attr(feature = "cli", value(name = "cuda-bf16"))]
    CudaBf16,
    /// LibTorch with f32 precision (cuBLAS + FlashAttention3 via PyTorch).
    #[cfg_attr(feature = "cli", value(name = "libtorch"))]
    LibTorchF32,
    /// LibTorch with bf16 precision (cuBLAS Tensor Core + FlashAttention3).
    #[cfg_attr(feature = "cli", value(name = "libtorch-bf16"))]
    LibTorchBf16,
}

impl TrainingBackendKind {
    /// Human-readable label for logs and reports.
    pub fn label(self) -> &'static str {
        match self {
            Self::CudaF32 => "Cuda (CubeCL, f32)",
            Self::CudaBf16 => "Cuda (CubeCL, bf16)",
            Self::LibTorchF32 => "LibTorch (cuBLAS/FA3, f32)",
            Self::LibTorchBf16 => "LibTorch (cuBLAS/FA3, bf16)",
        }
    }

    /// All available training backend variants.
    pub fn all() -> &'static [Self] {
        &[
            Self::CudaF32,
            Self::CudaBf16,
            Self::LibTorchF32,
            Self::LibTorchBf16,
        ]
    }
}

impl core::fmt::Display for TrainingBackendKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.label())
    }
}

/// Dispatch a block of code with the concrete inference backend type and device.
///
/// Provides a local type alias and a pre-configured device binding in the block
/// scope. Each match arm is fully monomorphised — no dynamic dispatch.
///
/// # Forms
///
/// **With device** — most common for CLI entrypoints:
/// ```ignore
/// let result = dispatch_inference!(backend_kind, gpu_id, |B, device| {
///     let model = TextToLatentRfDiT::<B>::load(&device)?;
///     model.forward(&input)
/// });
/// ```
///
/// **Type-only** — when you need custom device setup:
/// ```ignore
/// dispatch_inference!(backend_kind, |B| {
///     let device = B::cpu_device();
///     run::<B>(&device)
/// });
/// ```
#[macro_export]
macro_rules! dispatch_inference {
    // Block form with device binding
    ($kind:expr, $gpu_id:expr, |$B:ident, $device:ident| $body:expr) => {
        match $kind {
            $crate::InferenceBackendKind::NdArray => {
                type $B = burn::backend::NdArray;
                let $device = <$B as $crate::BackendConfig>::device_from_id($gpu_id);
                $body
            }
            $crate::InferenceBackendKind::Wgpu => {
                type $B = burn::backend::Wgpu;
                let $device = <$B as $crate::BackendConfig>::device_from_id($gpu_id);
                $body
            }
            $crate::InferenceBackendKind::WgpuF16 => {
                type $B = burn::backend::Wgpu<half::f16>;
                let $device = <$B as $crate::BackendConfig>::device_from_id($gpu_id);
                $body
            }
            $crate::InferenceBackendKind::CudaF32 => {
                type $B = burn::backend::Cuda;
                let $device = <$B as $crate::BackendConfig>::device_from_id($gpu_id);
                $body
            }
            $crate::InferenceBackendKind::CudaBf16 => {
                type $B = burn::backend::Cuda<half::bf16>;
                let $device = <$B as $crate::BackendConfig>::device_from_id($gpu_id);
                $body
            }
            $crate::InferenceBackendKind::LibTorchF32 => {
                type $B = burn::backend::LibTorch;
                let $device = <$B as $crate::BackendConfig>::device_from_id($gpu_id);
                $body
            }
            $crate::InferenceBackendKind::LibTorchBf16 => {
                type $B = burn::backend::LibTorch<half::bf16>;
                let $device = <$B as $crate::BackendConfig>::device_from_id($gpu_id);
                $body
            }
        }
    };
    // Block form with type alias only (no device binding)
    ($kind:expr, |$B:ident| $body:expr) => {
        match $kind {
            $crate::InferenceBackendKind::NdArray => {
                type $B = burn::backend::NdArray;
                $body
            }
            $crate::InferenceBackendKind::Wgpu => {
                type $B = burn::backend::Wgpu;
                $body
            }
            $crate::InferenceBackendKind::WgpuF16 => {
                type $B = burn::backend::Wgpu<half::f16>;
                $body
            }
            $crate::InferenceBackendKind::CudaF32 => {
                type $B = burn::backend::Cuda;
                $body
            }
            $crate::InferenceBackendKind::CudaBf16 => {
                type $B = burn::backend::Cuda<half::bf16>;
                $body
            }
            $crate::InferenceBackendKind::LibTorchF32 => {
                type $B = burn::backend::LibTorch;
                $body
            }
            $crate::InferenceBackendKind::LibTorchBf16 => {
                type $B = burn::backend::LibTorch<half::bf16>;
                $body
            }
        }
    };
}

/// Dispatch a block of code with the concrete training backend type and device.
///
/// The type alias binds to `Autodiff<BaseBackend>`. The device is constructed
/// from the base backend's `BackendConfig` implementation.
///
/// # Forms
///
/// **With device:**
/// ```ignore
/// dispatch_training!(backend_kind, gpu_id, |B, device| {
///     let trainer = LoraTrainer::<B>::new(config, &device)?;
///     trainer.train(dataset)
/// });
/// ```
///
/// **Type-only:**
/// ```ignore
/// dispatch_training!(backend_kind, |B| {
///     LoraTrainer::<B>::supports_flash_attention()
/// });
/// ```
#[macro_export]
macro_rules! dispatch_training {
    // Block form with device binding
    ($kind:expr, $gpu_id:expr, |$B:ident, $device:ident| $body:expr) => {
        match $kind {
            $crate::TrainingBackendKind::CudaF32 => {
                type $B = burn::backend::Autodiff<burn::backend::Cuda>;
                let $device =
                    <burn::backend::Cuda as $crate::BackendConfig>::device_from_id($gpu_id);
                $body
            }
            $crate::TrainingBackendKind::CudaBf16 => {
                type $B = burn::backend::Autodiff<burn::backend::Cuda<half::bf16>>;
                let $device =
                    <burn::backend::Cuda<half::bf16> as $crate::BackendConfig>::device_from_id(
                        $gpu_id,
                    );
                $body
            }
            $crate::TrainingBackendKind::LibTorchF32 => {
                type $B = burn::backend::Autodiff<burn::backend::LibTorch>;
                let $device =
                    <burn::backend::LibTorch as $crate::BackendConfig>::device_from_id($gpu_id);
                $body
            }
            $crate::TrainingBackendKind::LibTorchBf16 => {
                type $B = burn::backend::Autodiff<burn::backend::LibTorch<half::bf16>>;
                let $device =
                    <burn::backend::LibTorch<half::bf16> as $crate::BackendConfig>::device_from_id(
                        $gpu_id,
                    );
                $body
            }
        }
    };
    // Block form with type alias only (no device binding)
    ($kind:expr, |$B:ident| $body:expr) => {
        match $kind {
            $crate::TrainingBackendKind::CudaF32 => {
                type $B = burn::backend::Autodiff<burn::backend::Cuda>;
                $body
            }
            $crate::TrainingBackendKind::CudaBf16 => {
                type $B = burn::backend::Autodiff<burn::backend::Cuda<half::bf16>>;
                $body
            }
            $crate::TrainingBackendKind::LibTorchF32 => {
                type $B = burn::backend::Autodiff<burn::backend::LibTorch>;
                $body
            }
            $crate::TrainingBackendKind::LibTorchBf16 => {
                type $B = burn::backend::Autodiff<burn::backend::LibTorch<half::bf16>>;
                $body
            }
        }
    };
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_backend_kind_labels_are_non_empty() {
        for kind in InferenceBackendKind::all() {
            assert!(!kind.label().is_empty(), "{kind:?} has empty label");
        }
    }

    #[test]
    fn training_backend_kind_labels_are_non_empty() {
        for kind in TrainingBackendKind::all() {
            assert!(!kind.label().is_empty(), "{kind:?} has empty label");
        }
    }

    #[test]
    fn inference_backend_kind_all_count() {
        assert_eq!(InferenceBackendKind::all().len(), 7);
    }

    #[test]
    fn training_backend_kind_all_count() {
        assert_eq!(TrainingBackendKind::all().len(), 4);
    }

    #[test]
    fn inference_backend_kind_display_matches_label() {
        for kind in InferenceBackendKind::all() {
            assert_eq!(kind.to_string(), kind.label());
        }
    }

    #[test]
    fn training_backend_kind_display_matches_label() {
        for kind in TrainingBackendKind::all() {
            assert_eq!(kind.to_string(), kind.label());
        }
    }

    #[test]
    fn inference_backend_kind_serde_roundtrip() {
        for kind in InferenceBackendKind::all() {
            let json = serde_json::to_string(kind).unwrap();
            let back: InferenceBackendKind = serde_json::from_str(&json).unwrap();
            assert_eq!(*kind, back);
        }
    }

    #[test]
    fn training_backend_kind_serde_roundtrip() {
        for kind in TrainingBackendKind::all() {
            let json = serde_json::to_string(kind).unwrap();
            let back: TrainingBackendKind = serde_json::from_str(&json).unwrap();
            assert_eq!(*kind, back);
        }
    }

    #[test]
    fn dispatch_inference_type_only_compiles() {
        let kind = InferenceBackendKind::Wgpu;
        let label = dispatch_inference!(kind, |B| B::backend_label());
        assert_eq!(label, "Wgpu (f32)");
    }

    #[test]
    fn dispatch_training_type_only_compiles() {
        let kind = TrainingBackendKind::LibTorchF32;
        let label = dispatch_training!(kind, |B| <B as BackendConfig>::backend_label());
        assert_eq!(label, "LibTorch (cuBLAS/FA3, f32)");
    }

    #[test]
    fn dispatch_inference_returns_result() {
        let kind = InferenceBackendKind::Wgpu;
        let result: Result<String, &str> =
            dispatch_inference!(kind, 0, |B, _device| Ok(B::backend_label().to_string()));
        assert_eq!(result.unwrap(), "Wgpu (f32)");
    }

    #[test]
    fn dispatch_training_returns_result() {
        let kind = TrainingBackendKind::CudaF32;
        let result: Result<String, &str> =
            dispatch_training!(kind, 0, |B, _device| Ok(B::backend_label().to_string()));
        assert_eq!(result.unwrap(), "Cuda (CubeCL, f32)");
    }
}
