//! Backend configuration and runtime dispatch for Irodori-TTS.
//!
//! Production inference is intentionally restricted to the fused FP32 WGSL
//! policy. Training retains its separate enum dispatch until the training
//! subsystem is removed or migrated to WGPU.
//!
//! ## [`BackendConfig`] trait
//!
//! A thin supertrait over `Backend` that adds device construction and a
//! human-readable label. Implemented for all concrete backends and blanket-
//! implemented for `Autodiff<B>`.
//!
//! For production WGPU, `gpu_id == 0` selects `DefaultDevice`; an explicit
//! adapter index uses [`wgpu_device_from_adapter_index`]. CUDA/LibTorch device
//! construction below is retained only by the separately feature-gated trainer.

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

    /// Return a CPU-only device for this backend.
    ///
    /// Useful for benchmarks that must run on CPU for fair comparison.
    /// For backends without a dedicated CPU mode (WGPU, CubeCL CUDA) this
    /// falls back to the default device.
    fn cpu_device() -> Self::Device;

    /// A short human-readable label shown in benchmark / CLI output.
    fn backend_label() -> &'static str;

    /// Verify that this backend is usable on the given device.
    ///
    /// Called at entry-point startup to surface a clear error message when the
    /// backend requires runtime capabilities (e.g. `SHADER_F16`) that the
    /// selected device does not expose.  Returns `Ok(())` by default; backends
    /// with non-universal requirements should override this.
    fn check_requirements(_device: &Self::Device) -> Result<(), String> {
        Ok(())
    }
}

/// Select a WGPU device by index.
///
/// `gpu_id == 0` maps to `DefaultDevice` so the platform's best available GPU
/// is chosen automatically.  This is required on Apple Silicon (M-series), where
/// there is no "discrete" GPU — `DiscreteGpu(0)` panics with "No Discrete GPU
/// device found".  On NVIDIA/AMD Linux and Windows systems `DefaultDevice` also
/// selects the primary GPU, so behaviour is equivalent for single-GPU machines.
///
/// `gpu_id > 0` maps to `DiscreteGpu(N)` for explicit multi-GPU selection on
/// Linux/Windows.  Apple Silicon users should always leave `--gpu-id` at the
/// default (0).
fn wgpu_device(gpu_id: u32) -> burn::backend::wgpu::WgpuDevice {
    if gpu_id == 0 {
        burn::backend::wgpu::WgpuDevice::DefaultDevice
    } else {
        burn::backend::wgpu::WgpuDevice::DiscreteGpu(gpu_id as usize)
    }
}

/// Select an explicit discrete WGPU adapter by WGPU enumeration index.
///
/// This is intentionally separate from the legacy [`BackendConfig::device_from_id`]
/// convention, where index zero means `DefaultDevice`. In particular, this
/// function makes `DiscreteGpu(0)` selectable. WGPU adapter order is backend
/// specific and must not be assumed to match CUDA/NVML device order.
pub fn wgpu_device_from_adapter_index(adapter_index: usize) -> burn::backend::wgpu::WgpuDevice {
    burn::backend::wgpu::WgpuDevice::DiscreteGpu(adapter_index)
}

// ---------------------------------------------------------------------------
// WgpuRaw — CubeBackend without Fusion (for custom WGSL kernels)
// ---------------------------------------------------------------------------

/// Type alias for the non-fusion WGPU backend.
///
/// `burn::backend::Wgpu` wraps `CubeBackend` in `Fusion<...>` for automatic
/// kernel fusion. This raw variant exposes `CubeBackend` directly, which is
/// required for launching custom WGSL kernels via `SourceKernel` / `client.launch()`.
pub type WgpuRaw =
    burn::backend::wgpu::CubeBackend<burn::backend::wgpu::WgpuRuntime, f32, i32, u32>;

impl BackendConfig for WgpuRaw {
    fn device_from_id(gpu_id: u32) -> Self::Device {
        wgpu_device(gpu_id)
    }

    fn cpu_device() -> Self::Device {
        burn::backend::wgpu::WgpuDevice::DefaultDevice
    }

    fn backend_label() -> &'static str {
        "WgpuRaw (no fusion, f32)"
    }
}

/// Type alias for the non-fusion WGPU backend with f16 element type.
///
/// Like [`WgpuRaw`], this bypasses burn's `Fusion` wrapper.  Using `f16` as
/// the element type enables real half-precision arithmetic on hardware that
/// exposes `wgpu::Features::SHADER_F16` (Metal on Apple Silicon, Vulkan with
/// `VK_KHR_shader_float16_int8`, DX12 with `D3D12_FEATURE_D3D12_OPTIONS4`).
///
pub type WgpuRawF16 =
    burn::backend::wgpu::CubeBackend<burn::backend::wgpu::WgpuRuntime, half::f16, i32, u32>;

impl BackendConfig for WgpuRawF16 {
    fn device_from_id(gpu_id: u32) -> Self::Device {
        wgpu_device(gpu_id)
    }

    fn cpu_device() -> Self::Device {
        burn::backend::wgpu::WgpuDevice::DefaultDevice
    }

    fn backend_label() -> &'static str {
        "WgpuRaw (validation-only f16)"
    }

    fn check_requirements(device: &Self::Device) -> Result<(), String> {
        use burn::tensor::DType;
        if !Self::supports_dtype(device, DType::F16) {
            return Err("the selected WGPU adapter does not support SHADER_F16".to_owned());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Training-only CUDA / LibTorch backends
// ---------------------------------------------------------------------------

#[cfg(feature = "train")]
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

#[cfg(feature = "train")]
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

#[cfg(feature = "train")]
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

#[cfg(feature = "train")]
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

#[cfg(feature = "train")]
impl BackendConfig for burn::backend::LibTorch<half::f16> {
    fn device_from_id(gpu_id: u32) -> Self::Device {
        burn::backend::libtorch::LibTorchDevice::Cuda(gpu_id as usize)
    }

    fn cpu_device() -> Self::Device {
        burn::backend::libtorch::LibTorchDevice::Cpu
    }

    fn backend_label() -> &'static str {
        "LibTorch (f16)"
    }
}

// ---------------------------------------------------------------------------
// Blanket impl for Autodiff — delegates to the inner backend
// ---------------------------------------------------------------------------

#[cfg(feature = "train")]
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

/// Production inference execution policy.
///
/// This branch deliberately exposes one runtime choice: the measured fused
/// FP32 WGSL path on the raw WGPU backend. Keeping the CLI value preserves
/// pinned benchmark commands while preventing accidental execution through a
/// semantically different CUDA, LibTorch, NdArray, portable-WGPU, or reduced-
/// precision path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[serde(rename_all = "snake_case")]
pub enum InferenceBackendKind {
    /// Raw FP32 WGPU with the measured production fused-WGSL policy.
    #[cfg_attr(feature = "cli", value(name = "wgpu-wgsl"))]
    WgpuWgsl,
}

impl InferenceBackendKind {
    /// Human-readable label for logs and reports.
    pub fn label(self) -> &'static str {
        match self {
            Self::WgpuWgsl => "WgpuRaw (production fused WGSL, f32)",
        }
    }

    pub const fn is_reduced_precision(self) -> bool {
        false
    }

    /// All available inference backend variants.
    pub fn all() -> &'static [Self] {
        &[Self::WgpuWgsl]
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
/// throughput.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[cfg(feature = "train")]
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

#[cfg(feature = "train")]
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

#[cfg(feature = "train")]
impl core::fmt::Display for TrainingBackendKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.label())
    }
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
#[cfg(feature = "train")]
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
    #[cfg(feature = "train")]
    fn training_backend_kind_labels_are_non_empty() {
        for kind in TrainingBackendKind::all() {
            assert!(!kind.label().is_empty(), "{kind:?} has empty label");
        }
    }

    #[test]
    fn inference_backend_kind_all_count() {
        assert_eq!(
            InferenceBackendKind::all(),
            &[InferenceBackendKind::WgpuWgsl]
        );
    }

    #[test]
    #[cfg(feature = "train")]
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
    #[cfg(feature = "train")]
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
    #[cfg(feature = "train")]
    fn training_backend_kind_serde_roundtrip() {
        for kind in TrainingBackendKind::all() {
            let json = serde_json::to_string(kind).unwrap();
            let back: TrainingBackendKind = serde_json::from_str(&json).unwrap();
            assert_eq!(*kind, back);
        }
    }

    #[test]
    #[cfg(feature = "train")]
    fn dispatch_training_type_only_compiles() {
        let kind = TrainingBackendKind::LibTorchF32;
        let label = dispatch_training!(kind, |B| <B as BackendConfig>::backend_label());
        assert_eq!(label, "LibTorch (cuBLAS/FA3, f32)");
    }

    #[test]
    fn production_wgsl_backend_is_strict_fp32() {
        let kind = InferenceBackendKind::WgpuWgsl;
        assert!(!kind.is_reduced_precision());
        assert_eq!(kind.label(), "WgpuRaw (production fused WGSL, f32)");
    }

    #[test]
    fn explicit_wgpu_adapter_zero_is_selectable() {
        assert!(matches!(
            wgpu_device_from_adapter_index(0),
            burn::backend::wgpu::WgpuDevice::DiscreteGpu(0)
        ));
    }

    #[test]
    #[cfg(feature = "train")]
    fn dispatch_training_returns_result() {
        let kind = TrainingBackendKind::CudaF32;
        let result: Result<String, &str> =
            dispatch_training!(kind, 0, |B, _device| Ok(B::backend_label().to_string()));
        assert_eq!(result.unwrap(), "Cuda (CubeCL, f32)");
    }
}
