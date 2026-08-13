//! Backend configuration and runtime dispatch for Irodori-TTS.
//!
//! Production inference defaults to the measured WGPU AutoCompiler policy.
//! Burn/CubeCL generated kernels may use SPIR-V on Vulkan while handwritten
//! Irodori source kernels remain WGSL. Other WGPU platforms retain their
//! supported compiler path without adding another Burn dispatch backend.
//! Reduced precision is an explicit, separately namespaced policy and is not
//! selected implicitly by the device or checkpoint dtype.
//!
//! For production WGPU, `gpu_id == 0` selects `DefaultDevice`; an explicit
//! adapter index uses [`wgpu_device_from_adapter_index`].

use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{IrodoriError, Result};

/// Schema version for Irodori's prepared-kernel routes and warmup manifest.
pub const KERNEL_PROFILE_VERSION: &str = "v5";

/// Stable CubeCL environment identity for the current production runtime.
///
/// Adapter and driver identities remain part of CubeCL's cache keys. Keeping
/// software policy in the environment name additionally prevents accidental
/// pooling when the application changes compiler or numerical policy.
pub const CUBECL_ENVIRONMENT_NAME: &str =
    "irodori-v4-burn-0.22.0-pre.2-cubecl-0.11.0-pre.2-wgpu-auto-fp32-kernel-v5";

/// CubeCL environment identity for the experimental F16 WGPU graph.
pub const CUBECL_F16_ENVIRONMENT_NAME: &str =
    "irodori-v4-burn-0.22.0-pre.2-cubecl-0.11.0-pre.2-wgpu-auto-fp16-kernel-v5";

/// Explicit WGPU floating-point policy.
///
/// Keeping precision in a closed enum prevents an F16 checkpoint or adapter
/// default from silently weakening the production FP32 contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
#[serde(rename_all = "snake_case")]
pub enum WgpuFloatPrecision {
    Fp32,
    Fp16,
}

impl WgpuFloatPrecision {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
        }
    }

    pub const fn environment_name(self) -> &'static str {
        match self {
            Self::Fp32 => CUBECL_ENVIRONMENT_NAME,
            Self::Fp16 => CUBECL_F16_ENVIRONMENT_NAME,
        }
    }

    pub const fn float_dtype(self) -> burn::tensor::FloatDType {
        match self {
            Self::Fp32 => burn::tensor::FloatDType::F32,
            Self::Fp16 => burn::tensor::FloatDType::F16,
        }
    }

    pub const fn tensor_dtype(self) -> burn::tensor::DType {
        match self {
            Self::Fp32 => burn::tensor::DType::F32,
            Self::Fp16 => burn::tensor::DType::F16,
        }
    }
}

/// Application directory used below each operating system's user cache root.
pub const CACHE_APPLICATION_DIRECTORY: &str = "Irodori-TTS-burn";

/// Optional process override for the production CubeCL cache root.
pub const CUBECL_CACHE_DIR_ENV: &str = "IRODORI_TTS_BURN_CACHE_DIR";

/// Receipt proving which persistent CubeCL environment was installed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CubeClCacheReceipt {
    pub environment_name: String,
    pub root: PathBuf,
    pub environment_path: PathBuf,
}

/// Result of importing a pre-warmed CubeCL environment bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CubeClBundleImportReceipt {
    pub namespaces: Vec<String>,
    pub imported: usize,
    pub skipped: usize,
}

/// Resolve the production CubeCL cache outside Cargo's `target` directory.
///
/// Resolution order is [`CUBECL_CACHE_DIR_ENV`], then the platform user-cache
/// convention. CLI callers may place an explicit argument ahead of this
/// function. The returned path is not created until
/// [`configure_cubecl_persistent_cache`] is called.
pub fn default_cubecl_cache_root() -> Result<PathBuf> {
    default_cubecl_cache_root_from(std::env::consts::OS, |name| std::env::var_os(name))
}

fn default_cubecl_cache_root_from(
    os: &str,
    environment: impl Fn(&str) -> Option<OsString>,
) -> Result<PathBuf> {
    let nonempty = |name: &str| {
        environment(name)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    };
    if let Some(explicit) = nonempty(CUBECL_CACHE_DIR_ENV) {
        return Ok(explicit);
    }

    let base = match os {
        "windows" => nonempty("LOCALAPPDATA").or_else(|| nonempty("APPDATA")),
        "macos" => nonempty("HOME").map(|home| home.join("Library").join("Caches")),
        _ => nonempty("XDG_CACHE_HOME")
            .or_else(|| nonempty("HOME").map(|home| home.join(".cache"))),
    }
    .ok_or_else(|| {
        IrodoriError::Cache(format!(
            "cannot resolve a user cache directory; pass --cubecl-cache-dir or set {CUBECL_CACHE_DIR_ENV}"
        ))
    })?;

    Ok(base.join(CACHE_APPLICATION_DIRECTORY).join("cubecl"))
}

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
pub fn configure_cubecl_persistent_cache(root: impl Into<PathBuf>) -> Result<CubeClCacheReceipt> {
    configure_cubecl_persistent_cache_for_precision(root, WgpuFloatPrecision::Fp32)
}

/// Configure a precision-specific persistent CubeCL environment.
///
/// F16 and F32 never share autotune decisions even when all other runtime and
/// adapter keys match.
pub fn configure_cubecl_persistent_cache_for_precision(
    root: impl Into<PathBuf>,
    precision: WgpuFloatPrecision,
) -> Result<CubeClCacheReceipt> {
    use cubecl::config::RuntimeConfig;

    let root = root.into();
    if root.as_os_str().is_empty() {
        return Err(IrodoriError::Cache(
            "CubeCL cache root must not be empty".to_owned(),
        ));
    }
    std::fs::create_dir_all(&root)?;
    if !std::fs::metadata(&root)?.is_dir() {
        return Err(IrodoriError::Cache(format!(
            "CubeCL cache root is not a directory: {}",
            root.display()
        )));
    }

    let mut config = cubecl::config::CubeClRuntimeConfig::default();
    let environment_name = precision.environment_name();
    config.environment.name = environment_name.to_owned();
    config.environment.path = cubecl::config::cache::CacheConfig::Directory(root.clone());
    config.autotune.disable_cache = false;
    config.compilation.cache = true;
    if !cubecl::config::CubeClRuntimeConfig::try_set(config) {
        return Err(IrodoriError::Cache(
            "CubeCL runtime was initialized before persistent cache configuration".to_owned(),
        ));
    }

    Ok(CubeClCacheReceipt {
        environment_name: environment_name.to_owned(),
        environment_path: root.join(cubecl::environment::file_name(environment_name)),
        root,
    })
}

/// Import a CubeCL environment bundle into the already configured environment.
///
/// Call after [`configure_cubecl_persistent_cache`] and before WGPU runtime
/// initialization. Version and device checks remain CubeCL's responsibility;
/// write failures are treated as fatal so a service cannot claim a restored
/// warmup state that was not actually installed.
pub fn import_cubecl_environment_bundle(
    bundle_path: impl AsRef<Path>,
) -> Result<CubeClBundleImportReceipt> {
    let bundle = cubecl::bundle::open(bundle_path.as_ref()).map_err(|error| {
        IrodoriError::Cache(format!(
            "failed to open CubeCL bundle {}: {error}",
            bundle_path.as_ref().display()
        ))
    })?;
    let report = cubecl::bundle::import(bundle.as_ref());
    if report.failed != 0 {
        return Err(IrodoriError::Cache(format!(
            "CubeCL bundle import failed to persist {} entries",
            report.failed
        )));
    }
    Ok(CubeClBundleImportReceipt {
        namespaces: report.namespaces,
        imported: report.imported,
        skipped: report.skipped,
    })
}

/// Export the active CubeCL environment after warmup.
pub fn export_cubecl_environment_bundle(bundle_path: impl AsRef<Path>) -> Result<()> {
    cubecl::environment::bundle()
        .save(bundle_path.as_ref(), cubecl::bundle::BundleFormat::Sqlite)
        .map_err(|error| {
            IrodoriError::Cache(format!(
                "failed to export CubeCL bundle {}: {error}",
                bundle_path.as_ref().display()
            ))
        })?;
    Ok(())
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

/// Convert a concrete WGPU selector into Burn's high-level device and lock it
/// to the production strict-FP32 policy before the first tensor is created.
pub fn strict_fp32_device(
    device: &burn::backend::wgpu::WgpuDevice,
) -> Result<burn::tensor::Device> {
    wgpu_device_with_precision(device, WgpuFloatPrecision::Fp32)
}

/// Configure a WGPU device with an explicit floating-point policy before the
/// first tensor is created.
pub fn wgpu_device_with_precision(
    device: &burn::backend::wgpu::WgpuDevice,
    precision: WgpuFloatPrecision,
) -> Result<burn::tensor::Device> {
    use burn::tensor::IntDType;

    let mut configured: burn::tensor::Device = device.clone().into();
    configured
        .configure((precision.float_dtype(), IntDType::I32))
        .map_err(|error| {
            IrodoriError::Config(format!(
                "{} device configuration must precede tensor creation: {error}",
                precision.label()
            ))
        })?;
    let settings = configured.settings();
    if settings.float_dtype != precision.float_dtype() || settings.int_dtype != IntDType::I32 {
        return Err(IrodoriError::Config(format!(
            "{} device policy mismatch: float={:?}, int={:?}",
            precision.label(),
            settings.float_dtype,
            settings.int_dtype
        )));
    }
    Ok(configured)
}

// ---------------------------------------------------------------------------
// WgpuRaw — CubeBackend without Fusion (for custom WGSL kernels)
// ---------------------------------------------------------------------------

/// Type alias for the non-fusion WGPU backend.
///
/// `burn::backend::Wgpu` wraps `CubeBackend` in `Fusion<...>` for automatic
/// kernel fusion. This raw variant exposes `CubeBackend` directly, which is
/// required for launching custom WGSL kernels via `SourceKernel` / `client.launch()`.
pub type WgpuRaw = burn::backend::wgpu::CubeBackend<burn::backend::wgpu::WgpuRuntime>;

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
    use std::collections::HashMap;

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

    #[test]
    fn cache_environment_identity_pins_runtime_policy() {
        for (precision, environment) in [
            (WgpuFloatPrecision::Fp32, CUBECL_ENVIRONMENT_NAME),
            (WgpuFloatPrecision::Fp16, CUBECL_F16_ENVIRONMENT_NAME),
        ] {
            assert!(environment.contains("burn-0.22.0-pre.2"));
            assert!(environment.contains("cubecl-0.11.0-pre.2"));
            assert!(environment.contains("wgpu-auto"));
            assert!(environment.contains(precision.label()));
            assert!(environment.ends_with(KERNEL_PROFILE_VERSION));
            assert_eq!(precision.environment_name(), environment);
            assert!(
                environment
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
            );
        }
        assert_ne!(CUBECL_ENVIRONMENT_NAME, CUBECL_F16_ENVIRONMENT_NAME);
    }

    #[test]
    fn default_cache_roots_use_burn_specific_application_name() {
        let cases = [
            ("linux", [("HOME", "/home/test")].as_slice()),
            ("macos", [("HOME", "/Users/test")].as_slice()),
            (
                "windows",
                [("LOCALAPPDATA", "C:\\Users\\test\\AppData\\Local")].as_slice(),
            ),
        ];
        for (os, values) in cases {
            let environment = values
                .iter()
                .map(|(name, value)| ((*name).to_owned(), OsString::from(value)))
                .collect::<HashMap<_, _>>();
            let root =
                default_cubecl_cache_root_from(os, |name| environment.get(name).cloned()).unwrap();
            assert!(root.ends_with(Path::new(CACHE_APPLICATION_DIRECTORY).join("cubecl")));
        }
    }

    #[test]
    fn explicit_cache_environment_override_wins() {
        let root = default_cubecl_cache_root_from("linux", |name| {
            (name == CUBECL_CACHE_DIR_ENV).then(|| OsString::from("/cache/explicit"))
        })
        .unwrap();
        assert_eq!(root, PathBuf::from("/cache/explicit"));
    }
}
