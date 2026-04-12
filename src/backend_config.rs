//! Backend configuration trait for CLI binaries.
//!
//! Provides a thin `BackendConfig` supertrait over `Backend` that encapsulates
//! backend-specific setup (device selection, display label) so that all binaries
//! can use a uniform interface instead of repeating `#[cfg]`-gated device-init blocks.
//!
//! # `--gpu-id` semantics
//! For CUDA-class backends (CubeCL CUDA, LibTorch), `gpu_id` selects the Nth device.
//! For WGPU, it selects the Nth discrete GPU (`WgpuDevice::DiscreteGpu(id)`).
//! For CPU-only backends (NdArray), the parameter is ignored and `Cpu` is returned.

use burn::tensor::backend::Backend;

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
