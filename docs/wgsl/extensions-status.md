# WGSL Extension Status (as of June 2026)

## Overview

WGSL extensions are opt-in features not in the base spec. They require runtime
feature detection and explicit `enable` declarations in shader code.

---

## 1. Subgroups (`enable subgroups;`)

### What It Does
Subgroup operations let threads within a warp/wave communicate without shared memory.
Key operations: `subgroupAdd`, `subgroupBroadcast`, `subgroupBallot`, `subgroupShuffle`.

### Spec Status
- **W3C**: Candidate Recommendation (March 2026 draft)
- **Chrome**: Shipping from v145/v146 (early 2026)
- **Firefox/Safari**: Partial/in-progress

### wgpu-rs Status
- **Basic subgroup ops**: Supported since early-mid 2024
- **Quad ops** (`quadSwizzle`, `quadBroadcast`): NOT yet implemented
- **`subgroupElect`**: NOT yet implemented
- **Inclusive scan**: Polyfilled in some backends
- **Tracking issue**: [gfx-rs/wgpu#5555](https://github.com/gfx-rs/wgpu/issues/5555)

### Native Backend Support
| Backend | Status |
|---|---|
| Vulkan | Supported (via VK_KHR_shader_subgroup) |
| DirectX 12 | Supported (SM 6.0+ wave intrinsics) |
| Metal | Supported (simd_group functions) |
| WebGPU | Supported with feature detection |

### Usage in WGSL
```wgsl
enable subgroups;

@compute @workgroup_size(256)
fn reduce(@builtin(subgroup_invocation_id) lane: u32,
          @builtin(subgroup_size) sg_size: u32) {
    var val = data[global_id.x];
    val = subgroupAdd(val);
    // Only subgroup leader writes
    if lane == 0u {
        atomicAdd(&result, val);
    }
}
```

### Impact for Irodori-TTS-burn
- **RMSNorm**: Subgroup reduction can replace shared memory reduction (fewer barriers)
- **Softmax**: Subgroup max/sum for row-wise reduction
- **Estimated speedup**: 10-20% for reduction-heavy kernels
- **WebGPU fallback**: Current shared memory approach works on all backends

### Known Issues (tested June 2026)
- **`enable subgroups;` causes silent kernel failure on wgpu 29.0.1 (DX12, Vulkan AND Metal)**:
  Merely adding the `enable subgroups;` directive (without using any subgroup ops)
  causes compute shaders to produce all-zero output.
  Tested on DX12 (RTX 5070 Ti), Vulkan (RTX 5070 Ti), and Metal (M4 Pro) — same failure on all three.
  The kernel compiles and dispatches without error, but output is incorrect.
  This is a naga codegen issue (universal, not backend-specific).
  Workaround: omit `enable subgroups;` until wgpu/naga fix lands.
  See: [gfx-rs/wgpu#5555](https://github.com/gfx-rs/wgpu/issues/5555)

---

## 2. Half-Precision (`enable f16;`)

### What It Does
Enables native 16-bit float types (`f16`, `vec4<f16>`, `mat4x4<f16>`) in compute shaders.
2× memory bandwidth, potentially 2× compute throughput on supported hardware.

### Spec Status
- **W3C**: In Candidate Recommendation
- **Chrome**: Shipping from v120+ (2024)
- **Feature name**: `"shader-f16"`

### wgpu-rs Status
- **Partial/experimental support** (wgpu 0.19-0.20+, 2025)
- **Metal**: Relatively straightforward
- **Vulkan**: Works on SM6.2+ hardware
- **DirectX 12**: Requires DXIL generation with `-enable-16bit-types` (SM6.2+)
- **Tracking issue**: [gfx-rs/wgpu#4384](https://github.com/gfx-rs/wgpu/issues/4384)

### Usage in WGSL
```wgsl
enable f16;

@group(0) @binding(0) var<storage, read> input: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f16>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let val = input[gid.x];
    output[gid.x] = val * vec4<f16>(0.5h);
}
```

### Impact for Irodori-TTS-burn
- **WGPU f16 backend already uses burn's f16**: 32% faster than f32 (4,538ms vs 6,720ms on RTX 5070 Ti)
- **Custom WGSL f16 kernels**: Could further optimize by avoiding f32→f16 conversion overhead
- **GEMM**: f16 accumulation with f32 intermediate (mixed precision) for accuracy
- **WebGPU fallback**: All kernels must have f32 fallback path

---

## 3. Subgroup ID (`enable subgroup_id;`)

### What It Does
Provides `subgroup_id` and `num_subgroups` built-in values — useful for work scheduling
across subgroups within a workgroup.

### Status
- **Chrome**: Intent to Ship in v146 (early 2026)
- **wgpu-rs**: Part of subgroup tracking issue

### Usage
```wgsl
enable subgroups;
// Note: subgroup_id requires the subgroup_id feature
// WebGPU fallback: compute from local_invocation_id / subgroup_size

@compute @workgroup_size(256)
fn main(@builtin(subgroup_id) sg_id: u32,
        @builtin(num_subgroups) num_sg: u32) {
    // Use sg_id for inter-subgroup coordination
}
```

---

## 4. Subgroup Size Control (Experimental)

### What It Does
Allows explicitly setting subgroup size (e.g., force wave32 vs wave64 on AMD).

### Status
- **Experimental** as of late 2025 — NOT in spec yet
- **Vulkan**: Supported via VK_EXT_subgroup_size_control
- **DirectX 12**: Limited support
- **NOT recommended for production use**

---

## Recommendations for Irodori-TTS-burn

### Phase 1: Current (No Extensions)
- Use shared memory reduction patterns (current RMSNorm kernel)
- All WGSL bindings `read_write` (wgpu suballocator constraint)
- Max workgroup size 256 (WebGPU portability limit)
- f32 compute; burn handles f16 at the framework level

### Phase 2: Near-Term (Subgroups Available — BLOCKED on ALL backends)
- `enable subgroups;` causes silent kernel failure on wgpu 29 + DX12, Vulkan, AND Metal
- Subgroup optimization deferred until wgpu/naga fix lands
- Current serial softmax in native FA kernel achieves 1.51× burn generic (acceptable)
- Monitor [gfx-rs/wgpu#5555](https://github.com/gfx-rs/wgpu/issues/5555) for updates

### Phase 3: Future (f16 + Subgroups)
- Native f16 WGSL kernels for memory-bound operations
- Mixed-precision GEMM kernels (f16 compute, f32 accumulate)
- Combined f16 + subgroup reductions

### WebGPU Compatibility Strategy
For every extension-using kernel, maintain two code paths:
1. **Optimized path**: Uses extensions (subgroups, f16)
2. **Fallback path**: Pure WGSL base spec (shared memory, f32)

Detection at runtime:
```rust
let has_subgroups = adapter.features().contains(wgpu::Features::SUBGROUP);
let has_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
```
