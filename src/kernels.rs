//! Custom WGSL compute kernels for the WGPU backend.
//!
//! These kernels target the non-fusion WGPU backend (`CubeBackend<WgpuRuntime, ...>`)
//! and provide optimised implementations of hot-path operations in the DiT model.
//!
//! # Architecture
//!
//! Each sub-module contains:
//! - A `.wgsl` compute shader file
//! - A Rust launcher that implements [`KernelSource`] and handles buffer management
//!
//! # Platform compatibility
//!
//! All kernels use WGSL core features only (workgroup shared memory, barriers).
//! Vulkan/DX12/Metal-specific optimisations (subgroup ops) are commented as
//! future improvements for when the WGSL subgroups extension stabilises.

// Kernel infrastructure is not yet wired into the model — will be integrated
// when the WgpuRaw backend variant is added.
#[allow(dead_code)]
pub(crate) mod rms_norm;
