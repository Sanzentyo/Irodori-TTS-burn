# Quality Review — 2026-07-17

Source: `post-macro-review` rubber-duck agent (review of backend macro centralization refactoring).

## Summary

3 findings total: 1 BLOCKING, 2 MEDIUM. All 3 resolved.

## Findings & Resolutions

| # | Severity | Area | Status |
|---|----------|------|--------|
| 1 | BLOCKING | `bench_codec` device semantic regression | ✅ Fixed (commit bb0e1b2) |
| 2 | MEDIUM | `select_train_backend!()` ignores WGPU silently | ✅ Fixed (commit bb0e1b2) |
| 3 | MEDIUM | e2e tolerance gating only covered `backend_tch_bf16` | ✅ Fixed (commit bb0e1b2) |

## Prior session findings (same day, earlier review)

7 findings from comprehensive codebase review:

| # | Severity | Area | Status |
|---|----------|------|--------|
| 1 | BLOCKING | RF sampler tensor stats at INFO level | ✅ Fixed (commit 051abfc) |
| 2 | BLOCKING | `--no-default-features --all-targets` fails | ✅ Fixed (commit 051abfc) |
| 3 | MEDIUM | `backend_cpu` feature is dead (no-op) | ✅ Documented (commit 051abfc) |
| 4 | MEDIUM | Subsystem features don't shrink deps | ⚠️ Deferred (architectural) |
| 5 | MEDIUM | Docs drift (stale API refs) | ✅ Fixed (commit 051abfc) |
| 6 | MEDIUM | Backend selection boilerplate duplication | ✅ Fixed (commits 4c00e0a, bb0e1b2) |
| 7 | LOW | Integration test gaps | ⚠️ Noted (requires model weight fixtures) |

## Changes made

### BackendConfig trait enhancement
- Added `cpu_device()` method — returns CPU-appropriate device for each backend
- LibTorch → `LibTorchDevice::Cpu`, NdArray → `NdArrayDevice::Cpu`
- WGPU/Cuda → `DefaultDevice` / `CudaDevice { index: 0 }` (no native CPU mode)

### Training macro WGPU guard
- `select_train_backend!()` now emits `compile_error!` if any `backend_wgpu*` feature is active
- Clear error message: "WGPU backends do not support autodiff and cannot be used for training"

### Tolerance gating
- `e2e_compare.rs` and `full_model_e2e.rs` now gate wider tolerances for all bf16/f16 backends
- Previously only `backend_tch_bf16` was covered

### Deprecation fix
- `WgpuDevice::BestAvailable` → `WgpuDevice::DefaultDevice` (cubecl-wgpu 0.9.0)
