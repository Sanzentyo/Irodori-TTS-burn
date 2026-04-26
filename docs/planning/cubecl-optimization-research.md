# CubeCL Kernel Strategy Investigation

## Summary

Exhaustive investigation of CubeCL-based acceleration (CMMA / FlashUnit)
on Apple M4 Pro Metal via WGPU backend. All acceleration paths blocked
or non-viable. **Naive SDPA fallback is the correct baseline for WGPU Metal.**

## Finding 1 — CMMA Broken Through WGSL/naga Path

### Device Properties vs. Actual Capability

`cubecl-wgpu-0.10.0-pre.3/src/backend/metal.rs` registers Metal3 CMMA features:

```rust
// metal.rs: registers simdgroup_matrix hardware support in device properties
props.register_feature(Feature::Cmma { ... });
```

However, the actual shader compilation goes:

```
CubeCL IR → WGSL text → naga → MSL
```

The **WGSL compiler path in `cubecl-wgpu`** has NO CMMA instruction support.
Inspecting `src/compiler/wgsl/extension.rs` confirms: the WGSL extension
module only handles `PowfScalar`, `Powf`, `SafeTanh`, `IsNan`, `IsInf` —
no matrix-multiply / subgroup-matrix entries.

### Autotune Evidence

`target/autotune/0.10.0-pre.3/device-4-0-wgpu_wgsl_/burn_cubecl-kernel-matmul-tune-base.json.log`:

- All CMMA matmul strategies: `matmul_simple_cyclic_cmma`, `matmul_ordered_double_cmma_*`,
  `matmul_double_cyclic_cmma`, etc. → **`InvalidSamples`** (executes but produces wrong output)
- Non-CMMA unit strategies succeed: `matmul_double_unit_max_tile_size` = 2.80ms at 1024×1024×128
- **29 total strategies tested; all ~12 CMMA variants fail**

`target/autotune/0.10.0-pre.3/device-4-0-wgpu_wgsl_/burn_cubecl-kernel-attention-tune.json.log`:

- All `blackbox_accelerated_*_planes_p_*` variants → **`InvalidSamples`**
- `unit` → **`Skip`** (explained below)
- `fallback` → **`Ok`**, 6,638µs at production dims

### Root Cause

WGSL does not have a standardised `subgroup_matrix` extension in WGPU 29/naga.
The `chromium-experimental-subgroup-matrix` proposal is Chrome-specific and not
implemented in naga's MSL codegen. CubeCL detects the hardware capability (real
Metal3 simdgroup_matrix support) but cannot emit correct WGSL for it.

The compiled shader generates incorrect code → wrong output → `InvalidSamples`.
This is NOT a compile error but a semantic codegen failure.

**This is a fundamental limitation of `cubecl-wgpu` 0.10.0-pre.3 on Metal.**
It requires upstream fixes in either naga (add subgroup-matrix → MSL lowering)
or CubeCL (add passthrough MSL dialect for Metal).

---

## Finding 2 — FlashUnit FA Never Tried by Autotune

### Priority System

In `burn-cubecl-0.21.0-pre.3/src/kernel/attention/tune.rs`:

```rust
const PRIORITY_MAX: i8 = 3;
const PRIORITY_MIN: i8 = 0;

// fallback has PRIORITY_MAX for seq_q <= 4096:
set.with(Tunable::new("fallback", ...).group(&fallback, |_key| PRIORITY_MAX))

// FlashUnit has PRIORITY_MIN:
set.with(Tunable::new("unit", ...).group(&flash_attention, |_key| PRIORITY_MIN))
```

The autotune pre-initialises ALL results as `Skip`, then overwrites only the
candidates it actually executes. Since `fallback` succeeds at `PRIORITY_MAX`
and `FlashUnit` is `PRIORITY_MIN`, the autotune **exits without ever running FlashUnit**.

The `Skip` in the log does NOT mean "prepare() returned an error" — it means
"was not attempted because a higher-priority winner was found first."

### Implication

FlashUnit was never benchmarked by the autotune on any production input.
It required direct API testing (see Finding 3).

---

## Finding 3 — FlashUnit FA Correct But 8.7× Slower

### Test Method

Added `burn-cubecl = "0.21.0-pre.3"` as direct dependency and wrote
`src/bin/bench_flashunit_fa.rs` which calls:

```rust
burn_cubecl::kernel::attention::attention::<WgpuRuntime>(
    q_prim, k_prim, v_prim, None, None, options,
    AttentionStrategy::FlashUnit,
)
```

### Results (M4 Pro Metal, WgpuRaw f32)

| Scenario | Fallback (µs) | FlashUnit (µs) | Ratio |
|---|---|---|---|
| Production (1×20×750×950, D=64) | 9,878 | 85,941 | **8.70× slower** |
| Stress (1×8×1024×1024, D=64) | 3,166 | 79,180 | **25.01× slower** |
| Tiny sanity (1×4×32×32, D=64) | 1,688 | 3,542 | 2.10× slower |

**Correctness**: max_abs_diff = 2.87e-5 (f32 precision). FlashUnit produces
CORRECT output. It is slow, not broken.

### Root Cause: Tiny Tile Sizes

`AttentionTileSize::from_max_vector_sizes` for f32 WGPU produces:

- `head_dim = lcm(query_vec=4, key_vec=4) = 4`
- `val_dim = lcm(value_vec=4, out_vec=4) = 4`
- `seq_kv = lcm(key_vec=4, mask_vec=4) = 4`
- `seq_q = 8` (hardcoded)

These 4×4 tiles are **far too small** for efficient GPU execution.
Competitive FlashAttention uses 64×64 or 128×64 tiles (FA2) to maximise
the compute-to-memory ratio. With 4×4 tiles:

- Each workgroup contains only 32 threads (one SIMD lane)
- The GPU needs thousands of tiny workgroups for production shapes
- Thread-dispatch overhead dominates useful computation
- No K/V cache reuse benefit materialises

Real-world GEMM efficiency typically requires tiles ≥ 32×32 for decent GPU occupancy.

### For f16 Backend

With vec8 WGPU f16, tile sizes would be `head_dim=8, val_dim=8, seq_kv=8` —
still far too small for competitive performance.

---

## Finding 4 — Best Available Matmul Strategy

From the matmul autotune cache:

- **Winner**: `matmul_double_unit_max_tile_size` at 2.80ms for 1024×1024×128
- This is burn's double-buffered tiled GEMM with maximum tile size
- All CMMA variants fail; this is the ceiling for matmul without CMMA

The non-CMMA matmul strategies available (from cache, ordered fastest to slowest):

| Strategy | Time (ms) | Status |
|---|---|---|
| `matmul_double_unit_max_tile_size` | 2.80 | ✅ Winner |
| `matmul_simple_unit_max_tile_size` | 3.40 | ✅ |
| `matmul_double_unit_min_tile_size` | 4.75 | ✅ |
| `matmul_simple_unit_min_tile_size` | 5.48 | ✅ |
| `matmul_simple_vecmat` | 14.45 | ✅ |
| `matmul_double_vecmat` | 21.31 | ✅ |
| All CMMA variants (×12+) | — | ❌ `InvalidSamples` |

---

## Conclusion

**All CubeCL acceleration paths are blocked on Metal via WGPU 0.21/naga:**

| Path | Status | Reason |
|---|---|---|
| CMMA matmul (tensor cores) | ❌ `InvalidSamples` | WGSL/naga has no subgroup-matrix support |
| BlackboxAccelerated FA (CMMA) | ❌ `InvalidSamples` | Same root cause as CMMA matmul |
| FlashUnit FA (subgroup ops) | ❌ 8.7× slower | 4×4 tiles → terrible GPU occupancy |
| Custom WGSL FA kernels | ❌ 2.5–3.7× slower | (prior work, already exhausted) |
| Naive SDPA fallback | ✅ Current best | 6,638µs at production dims |
| `matmul_double_unit_max_tile_size` | ✅ Current best | Double-buffered tiled GEMM |

**burn's autotune makes the correct choice**: fallback (naive SDPA) wins for
all attention shapes on Metal because it's the only strategy that both works
AND is fast.

---

## Potential Future Improvements (Upstream Required)

1. **CubeCL adds Metal MSL passthrough dialect** — would unlock simdgroup_matrix
   via direct MSL injection, bypassing WGSL/naga. Estimated 2-4× matmul speedup.
2. **naga adds `subgroup-matrix` → MSL lowering** — would fix `InvalidSamples`
   for all CMMA strategies. Unlikely near-term (no Chrome WGSL spec).
3. **cubek-attention increases FlashUnit tile sizes** — tile size is device-queried
   from vector sizes; a higher-level tile override API would allow larger tiles
   that could make FlashUnit competitive.

---

## Files

- `src/bin/bench_flashunit_fa.rs` — FlashUnit vs fallback benchmark
- `target/autotune/0.10.0-pre.3/device-4-0-wgpu_wgsl_/` — autotune cache (gitignored)
  - `burn_cubecl-kernel-attention-tune.json.log` — attention strategies
  - `burn_cubecl-kernel-matmul-tune-base.json.log` — matmul strategies

---

*Investigated: 2026-Q2, M4 Pro Mac Mini, Metal backend, cubecl-wgpu 0.10.0-pre.3*
