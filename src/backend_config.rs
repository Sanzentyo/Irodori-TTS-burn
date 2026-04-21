//! Backend configuration and runtime dispatch for Irodori-TTS.
//!
//! ## Runtime enum dispatch (recommended for binaries)
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
//! ## [`BackendConfig`] trait
//!
//! A thin supertrait over `Backend` that adds device construction and a
//! human-readable label. Implemented for all concrete backends and blanket-
//! implemented for `Autodiff<B>`.
//!
//! ## `--gpu-id` semantics
//! For CUDA-class backends (CubeCL CUDA, LibTorch), `gpu_id` selects the Nth device.
//! For WGPU with `gpu_id == 0` (default): `WgpuDevice::DefaultDevice` — works on all
//! platforms including Apple Silicon (M-series) which has no discrete GPU.
//! For WGPU with `gpu_id > 0`: `WgpuDevice::DiscreteGpu(N)` for explicit multi-GPU
//! selection on Linux/Windows.
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

impl BackendConfig for burn::backend::Wgpu {
    fn device_from_id(gpu_id: u32) -> Self::Device {
        wgpu_device(gpu_id)
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
        wgpu_device(gpu_id)
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
        wgpu_device(gpu_id)
    }

    fn cpu_device() -> Self::Device {
        burn::backend::wgpu::WgpuDevice::DefaultDevice
    }

    fn backend_label() -> &'static str {
        "Wgpu (bf16)"
    }
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
/// Call [`BackendConfig::check_requirements`] before launching inference to
/// get a clear error message on hardware that lacks SHADER_F16 support.
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
        "WgpuRaw (no fusion, f16)"
    }

    fn check_requirements(device: &Self::Device) -> Result<(), String> {
        use burn::tensor::DType;
        if !Self::supports_dtype(device, DType::F16) {
            return Err("This GPU does not support SHADER_F16. \
                 Use --backend wgpu-raw for f32 inference."
                .to_owned());
        }
        Ok(())
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
    /// WGPU without kernel fusion (f32). Enables custom WGSL kernels
    /// (RMSNorm, SDPA, etc.) that bypass the Fusion layer.
    #[cfg_attr(feature = "cli", value(name = "wgpu-raw"))]
    WgpuRawF32,
    /// WGPU without kernel fusion, f16 precision.
    ///
    /// Combines the raw (no-Fusion) backend with f16 element type for real
    /// half-precision arithmetic.  Avoids the burn-fusion DACVAE crash while
    /// still delivering the f16 speed-up.  Requires `SHADER_F16` GPU support
    /// (Metal on Apple Silicon, Vulkan with float16 extension, DX12).
    #[cfg_attr(feature = "cli", value(name = "wgpu-raw-f16"))]
    WgpuRawF16,
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
            Self::WgpuRawF32 => "WgpuRaw (no fusion, f32)",
            Self::WgpuRawF16 => "WgpuRaw (no fusion, f16)",
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
        matches!(
            self,
            Self::WgpuF16 | Self::WgpuRawF16 | Self::CudaBf16 | Self::LibTorchBf16
        )
    }

    /// All available inference backend variants.
    pub fn all() -> &'static [Self] {
        &[
            Self::NdArray,
            Self::Wgpu,
            Self::WgpuF16,
            Self::WgpuRawF32,
            Self::WgpuRawF16,
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
/// throughput.
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
            $crate::InferenceBackendKind::WgpuRawF32 => {
                type $B = $crate::WgpuRaw;
                let $device = <$B as $crate::BackendConfig>::device_from_id($gpu_id);
                $body
            }
            $crate::InferenceBackendKind::WgpuRawF16 => {
                type $B = $crate::WgpuRawF16;
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
            $crate::InferenceBackendKind::WgpuRawF32 => {
                type $B = $crate::WgpuRaw;
                $body
            }
            $crate::InferenceBackendKind::WgpuRawF16 => {
                type $B = $crate::WgpuRawF16;
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
        assert_eq!(InferenceBackendKind::all().len(), 9);
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
