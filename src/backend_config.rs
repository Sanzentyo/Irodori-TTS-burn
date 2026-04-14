//! Backend configuration trait and selection macros for CLI binaries.
//!
//! Provides a thin `BackendConfig` supertrait over `Backend` that encapsulates
//! backend-specific setup (device selection, display label) so that all binaries
//! can use a uniform interface instead of repeating `#[cfg]`-gated device-init blocks.
//!
//! # Backend selection macros
//!
//! [`select_inference_backend!`] defines a type alias `B` for the active backend,
//! with `compile_error!` guards against selecting multiple backends simultaneously.
//! Covers all 8 inference backends (NdArray, Wgpu ×3, Cuda ×2, LibTorch ×2).
//!
//! [`select_train_backend!`] does the same for training backends (LibTorch ×2, Cuda ×2,
//! NdArray fallback), defining `BaseB` — training only uses backends that support
//! `AutodiffBackend`.
//!
//! # `--gpu-id` semantics
//! For CUDA-class backends (CubeCL CUDA, LibTorch), `gpu_id` selects the Nth device.
//! For WGPU, it selects the Nth discrete GPU (`WgpuDevice::DiscreteGpu(id)`).
//! For CPU-only backends (NdArray), the parameter is ignored and `Cpu` is returned.

use burn::tensor::backend::Backend;

/// Defines type alias `B` for the active inference backend based on Cargo feature flags.
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

    fn backend_label() -> &'static str {
        "Wgpu (f32)"
    }
}

impl BackendConfig for burn::backend::Wgpu<half::f16> {
    fn device_from_id(gpu_id: u32) -> Self::Device {
        burn::backend::wgpu::WgpuDevice::DiscreteGpu(gpu_id as usize)
    }

    fn backend_label() -> &'static str {
        "Wgpu (f16)"
    }
}

impl BackendConfig for burn::backend::Wgpu<half::bf16> {
    fn device_from_id(gpu_id: u32) -> Self::Device {
        burn::backend::wgpu::WgpuDevice::DiscreteGpu(gpu_id as usize)
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

    fn backend_label() -> &'static str {
        "LibTorch (cuBLAS/FA3, f32)"
    }
}

impl BackendConfig for burn::backend::LibTorch<half::bf16> {
    fn device_from_id(gpu_id: u32) -> Self::Device {
        burn::backend::libtorch::LibTorchDevice::Cuda(gpu_id as usize)
    }

    fn backend_label() -> &'static str {
        "LibTorch (cuBLAS/FA3, bf16)"
    }
}
