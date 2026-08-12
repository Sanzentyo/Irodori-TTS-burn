//! Backend configuration and runtime dispatch for Irodori-TTS.
//!
//! Production inference is intentionally restricted to the fused FP32 WGSL
//! policy.
//!
//! For production WGPU, `gpu_id == 0` selects `DefaultDevice`; an explicit
//! adapter index uses [`wgpu_device_from_adapter_index`].

use std::path::PathBuf;

/// Configure persistent CubeCL autotune and supported compilation caches
/// outside Cargo's `target` directory.
///
/// Call this exactly once, before any WGPU/CubeCL initialization. The caller
/// must include an adapter/backend fingerprint in `root`; CubeCL's internal
/// WGPU device key is not a stable cross-platform hardware identity by itself.
/// Keeping the namespace decision at the application boundary prevents cache
/// entries from being pooled across different adapters while allowing results
/// to survive `cargo clean`.
///
/// The current `wgpu_wgsl` path persists autotune decisions but does not emit a
/// reusable compiled-pipeline blob; the compilation setting is effective only
/// for CubeCL compiler paths that implement it.
pub fn configure_cubecl_persistent_cache(root: impl Into<PathBuf>) {
    let root = root.into();
    let mut config = cubecl::config::GlobalConfig::default();
    config.autotune.cache = cubecl::config::cache::CacheConfig::File(root.join("autotune"));
    config.compilation.cache = Some(cubecl::config::cache::CacheConfig::File(
        root.join("compilation"),
    ));
    cubecl::config::GlobalConfig::set(config);
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
pub fn wgpu_device(gpu_id: u32) -> burn::backend::wgpu::WgpuDevice {
    if gpu_id == 0 {
        burn::backend::wgpu::WgpuDevice::DefaultDevice
    } else {
        burn::backend::wgpu::WgpuDevice::DiscreteGpu(gpu_id as usize)
    }
}

/// Select an explicit discrete WGPU adapter by WGPU enumeration index.
///
/// This is intentionally separate from [`wgpu_device`], where index zero means
/// `DefaultDevice`. In particular, this function makes `DiscreteGpu(0)`
/// selectable. WGPU adapter order is backend specific and must not be assumed
/// to match CUDA/NVML device order.
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

// ===========================================================================
// Runtime backend dispatch (enum-based, no dynamic dispatch)
// ===========================================================================

/// Production inference execution policy.
///
/// This branch deliberately exposes one runtime choice: the measured fused
/// FP32 WGSL path on the raw WGPU backend. Keeping the CLI value preserves
/// pinned benchmark commands while preventing accidental execution through a
/// semantically different backend or reduced-precision path.
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
    fn inference_backend_kind_all_count() {
        assert_eq!(
            InferenceBackendKind::all(),
            &[InferenceBackendKind::WgpuWgsl]
        );
    }

    #[test]
    fn inference_backend_kind_display_matches_label() {
        for kind in InferenceBackendKind::all() {
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
}
