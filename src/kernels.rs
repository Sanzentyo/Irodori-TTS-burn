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
//!
//! # Known Issues
//!
//! `enable subgroups;` causes silent kernel failure (all-zero output) on
//! wgpu 29.0.1 + DX12, Vulkan, and Metal. See [`subgroup_diagnostic`] for proof.
//! `enable f16;` status is under investigation — see [`f16_diagnostic`].

// Kernel infrastructure is not yet wired into the model — will be integrated
// when the WgpuRaw backend variant is added.
#[allow(dead_code)]
pub mod f16_diagnostic;
#[allow(dead_code)]
pub mod fused_adaln;
#[allow(dead_code)]
pub mod fused_sdpa;
#[allow(dead_code)]
pub mod fused_sdpa_native;
#[allow(dead_code)]
pub mod fused_sdpa_native_f16;
#[allow(dead_code)]
pub mod fused_sdpa_tiled;
#[allow(dead_code)]
pub mod rms_norm;
#[allow(dead_code)]
pub mod subgroup_diagnostic;
