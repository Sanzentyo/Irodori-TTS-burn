# WGSL Knowledge Base

This directory contains research and reference material on WGSL (WebGPU Shading Language)
features relevant to the Irodori-TTS-burn WGPU kernel optimization work.

## Contents

| File | Topic |
|---|---|
| [extensions-status.md](extensions-status.md) | Status of key WGSL extensions (subgroups, f16, etc.) |
| [kernel-patterns.md](kernel-patterns.md) | Common WGSL compute shader patterns for ML kernels |

## Key References

- [WGSL Spec (W3C)](https://www.w3.org/TR/WGSL/) — Candidate Recommendation Draft (March 2026)
- [wgpu GitHub](https://github.com/gfx-rs/wgpu) — Rust WebGPU implementation
- [wgpu subgroup tracking](https://github.com/gfx-rs/wgpu/issues/5555) — Subgroup implementation status
- [wgpu f16 tracking](https://github.com/gfx-rs/wgpu/issues/4384) — Half-precision support
- [WebGPU subgroups proposal](https://github.com/gpuweb/gpuweb/blob/main/proposals/subgroups.md)
