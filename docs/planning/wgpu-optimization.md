# WGPU Performance Optimization Plan

## Goal

Reduce WGPU f32 inference time from **7,315ms (2.72× Python)** → target **< 4,000ms (~1.5×)**.

## Current State (2026-07-21)

### What's done

- ✅ `WgpuRaw` backend variant added (`CubeBackend<WgpuRuntime, f32>` without Fusion wrapper)
  - `InferenceBackendKind::WgpuRawF32`, `WgpuRaw` type alias in `src/backend_config.rs`
  - `just bench-wgpu-raw` recipe in justfile
  - Dispatchable via `--backend wgpu-raw` CLI flag
- ✅ Custom WGSL RMSNorm kernel (`src/kernels/rms_norm.{rs,wgsl}`)
  - Single-pass, shared-memory reduction
  - All bindings `read_write` (required for WGPU suballocator safety)
  - Template-baked `dim`, `eps`, `workgroup_size` at compile time
  - Test: matches CPU reference within 1e-4 (marked `#[ignore]` for CI — WGPU teardown SIGSEGV)
- ✅ `src/kernels/` module infrastructure

### What's NOT done

- ☐ WgpuRaw valid baseline benchmark (last run was on occupied GPU — DISCARD)
- ☐ Wgpu-f16 benchmark
- ☐ RMSNorm kernel not yet wired into model (model still uses generic burn ops)
- ☐ No WGPU-specific profiling (nsys profile was CUDA only)
- ☐ No fused SDPA kernel

## Rubber Duck Review Findings (2026-07-21)

Key findings from independent review (agent `sdpa-plan-review`):

### Decision gates (blocking — must answer before writing more kernels)

1. **Benchmark Wgpu-f16 first** — may be significantly faster due to reduced memory bandwidth;
   easier win than custom kernels.

2. **WgpuRaw go/no-go threshold**: If `WgpuRaw` (no fusion) is > ~10% slower than `Wgpu`
   (with fusion), the custom-kernel-on-WgpuRaw strategy is **not viable** without compensating
   for that fusion loss across every forward pass.
   - Tentative measured delta: WgpuRaw ~10,422ms vs Wgpu ~7,315ms ≈ **42% slower** (INVALID RUN)
   - Must re-measure on idle GPU to confirm
   - **If confirmed ~42% slower**: abort raw-kernel strategy, find within-fusion alternatives

3. **Profile WGPU specifically** before writing SDPA kernel — CUDA profile (matmul=60%,
   reduce=19%) may not reflect WGPU proportions since WGPU uses different kernel implementations.

4. **Benchmark with realistic inputs**: Current synthetic bench uses `text_len=4, ref_frames=8`
   which may understate attention cost vs real `text_len=100-200`.

### If WgpuRaw is viable (within ~10% of fusion Wgpu)

Priority order:
1. RMSNorm wiring: plug `rms_norm_wgsl()` into `src/model/norm.rs` when backend is `WgpuRaw` — isolated benchmark
2. Fused SDPA kernel — but must also fuse K/V concat + mask (not just `softmax(QKᵀ)V`)
3. Specialized matmul — potentially higher impact if attention is not the bottleneck

### If WgpuRaw is NOT viable (> 10% slower than fusion Wgpu)

Alternative approaches to investigate within the fusion layer:
1. **Wgpu-f16**: Does GPU support `shader-f16`? Check with device query
2. **Backend settings**: `MemoryConfiguration`, `RuntimeOptions` tuning
3. **Minimize layout copies**: `into_contiguous_kernel` = 2.1% on CUDA — maybe more on WGPU
4. **Upstream contribution**: Custom CubeCL ops that integrate with Fusion layer (complex)

## New Device Protocol

On a new device, run these steps before any optimization work:

```sh
# 1. Verify GPU is idle
nvidia-smi --query-gpu=name,memory.total,utilization.gpu,memory.used --format=csv,noheader

# 2. Get device spec for benchmark filename
# e.g. GPU name → "rtx-4090" or "m3-max" etc.

# 3. Create device benchmark file
cp docs/benchmarks/rtx-a6000.md docs/benchmarks/<device>.md
# Edit System section

# 4. Run all benchmarks (with GPU idle verification first)
just bench-wgpu          # Wgpu fusion f32
just bench-wgpu-f16      # Wgpu fusion f16
just bench-wgpu-raw      # WgpuRaw no-fusion f32
just bench-cuda          # CubeCL CUDA f32
just bench-tch           # LibTorch f32
just bench-tch-bf16      # LibTorch bf16

# 5. Python baseline
cd ~/Irodori-TTS && uv run python bench.py

# 6. Fill in device benchmark file, commit
```

## Key Technical Details

### WGPU buffer aliasing rule (critical)

All WGSL bindings MUST use `var<storage, read_write>`, even for read-only inputs:
- WGPU's `SlicedPool` memory manager packs multiple tensors into shared physical `wgpu::Buffer` pages
- If bindings on the same physical page have different usage flags (STORAGE_READ_ONLY vs
  STORAGE_READ_WRITE), WGPU validation rejects the dispatch
- Using `read_write` for all bindings gives them the same flag, preventing conflicts
- Source: `src/kernels/rms_norm.wgsl` lines 15-17

### WgpuRaw type

```rust
pub type WgpuRaw = burn::backend::wgpu::CubeBackend<burn::backend::wgpu::WgpuRuntime, f32, i32, u32>;
// Defined in: src/backend_config.rs
// Re-exported from: src/lib.rs
```

`burn::backend::Wgpu` = `Fusion<CubeBackend<WgpuRuntime, ...>>` when compiled with `fusion` feature.
The fusion layer provides automatic kernel fusion (~30% speedup expected) but prevents raw `client.launch()` access.

### Kernel dispatch chain

```
SourceKernel<RmsNormKernel: KernelSource>
  → CubeTask<AutoCompiler>  (via Box::new(...))
  → client.launch(kernel_box, cube_count, bindings)
```

`KernelArguments::new().with_buffer(handle.binding())` — each binding from `tensor.handle.binding()`.

### Workgroup size

Capped at 256 for WebGPU portability (`next_power_of_two(dim).min(256)`).
Do not exceed 256 without querying device limits — not guaranteed on all WebGPU implementations.

## Files

| File | Purpose |
|------|---------|
| `src/kernels.rs` | Module root (has `#[allow(dead_code)]` — not yet wired into model) |
| `src/kernels/rms_norm.rs` | RMSNorm kernel launcher + test |
| `src/kernels/rms_norm.wgsl` | WGSL compute shader |
| `src/backend_config.rs` | `WgpuRaw` type alias + `WgpuRawF32` variant |
| `src/lib.rs` | Re-exports `WgpuRaw` |
| `src/model/norm.rs` | Where custom kernel will be wired (TODO) |
| `src/model/attention.rs` | Where custom SDPA kernel will go (TODO, pending viability) |
