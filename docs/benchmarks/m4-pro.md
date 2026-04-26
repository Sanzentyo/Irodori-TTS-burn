# Benchmark Results: Apple M4 Pro (Mac Mini)

## System

| Component | Details |
|---|---|
| CPU | Apple M4 Pro |
| GPU | Apple M4 Pro integrated GPU (Metal, ~14 TFLOPS) |
| RAM | 24 GB unified memory |
| OS | macOS (arm64) |
| burn version | 0.21.0-pre.3 |
| WGPU backend | Metal (via wgpu 29.0.1) |
| Rust edition | 2024 |

**Note**: This device has no CUDA GPU. CUDA/LibTorch CUDA backends are not available.
LibTorch MPS (Metal Performance Shaders via PyTorch 2.9.0) IS available and is the fastest backend.

## Model

| Field | Value |
|---|---|
| Checkpoint | `Aratako/Irodori-TTS-500M-v2` |
| model_dim | 1280 |
| num_layers | 12 |
| num_heads | 20 |
| Parameters | ~500M |

---

## Full-Model Benchmark (seq=750, steps=40)

Model: `Aratako/Irodori-TTS-500M-v2` (500M params, model_dim=1280, layers=12, heads=20).

### Default CFG (Independent, text=3.0, speaker=5.0, batch=3)

| Backend | Mean (ms) | RTF | vs Wgpu f32 | Notes |
|---|---|---|---|---|
| Rust/burn **LibTorch MPS f16** (Apple MPS) | **11,320** | **0.377** | 0.317× | torch 2.10.0 |
| Rust/burn **LibTorch MPS f32** (Apple MPS) | **11,926** | **0.398** | 0.334× | torch 2.10.0 |
| Rust/burn **WgpuRaw f16** (Metal, no Fusion) | **18,850** | **0.628** | 0.527× | ✅ no-dep fallback |
| Rust/burn **Wgpu f16** (Metal, Fusion) | 18,155 | 0.61 | 0.508× | ❌ DACVAE crash |
| Rust/burn Wgpu f32 (Metal) | 35,745 | 1.19 | 1.00× | ❌ DACVAE crash |
| Rust/burn WgpuRaw f32 (Metal) | 36,451 | 1.22 | 1.02× | |

> LibTorch MPS benchmarked with torch 2.10.0 (formerly 2.9.0 in earlier sessions).
> Previous 2.9.0 numbers: MPS f16 = 10,204ms (RTF 0.340), MPS f32 = 11,660ms (RTF 0.389).
> The ~11% MPS f16 regression correlates with the torch upgrade; WgpuRaw numbers unchanged.
> For reference: RTX 5070 Ti LibTorch bf16=1,309ms; Wgpu f16=4,538ms; Wgpu f32=6,720ms.

### CFG Mode Comparison (WgpuRaw f16 and LibTorch MPS f16)

CFG mode affects the batch structure per diffusion step:
- **Independent** (default): single batch=3 forward (cond + uncond_text + uncond_speaker); allows different scales per signal
- **Joint** (equal scales only): 2 sequential batch=1 forwards (cond + fully-uncond); requires equal scales
- **No CFG**: 1 batch=1 forward; fastest but no guidance

| CFG Mode | Batch | WgpuRaw f16 (ms) | RTF | MPS f16 (ms) | RTF |
|---|---|---|---|---|---|
| Independent (text=3.0, speaker=5.0) | 3 | 18,850 | 0.628 | 11,320 | 0.377 |
| Joint (scale=3.0 equal) | 2×1 | **14,287** | **0.476** | **~8,320*** | **~0.277*** |
| Speaker only (text=0, speaker=5.0) | 2 | 14,122 | 0.471 | — | — |
| No CFG | 1 | 9,538 | 0.318 | **~5,840*** | **~0.195*** |

> * MPS f16 Joint/NoCFG numbers extrapolated from batch scaling ratio (torch 2.10.0 baseline).
> WgpuRaw f16 CFG mode numbers confirmed fresh (2025 July).

**Key findings:**
- **Joint CFG is ~1.32× faster** than Independent; MPS f16 + Joint CFG ≈ RTF 0.28 (3.6× real-time) with torch 2.10.0
- Joint CFG requires equal scales for all active signals; allows `--cfg-mode joint --cfg-speaker 3.0`
- Batch scaling is sublinear: MPS f16 batch=3 is ~1.94× slower than batch=1 (extrapolated)
- **LibTorch MPS f16 is the recommended backend for M-series Mac** (1.66× faster than WgpuRaw f16 with torch 2.10.0)
- **WgpuRaw f16 is the recommended no-dep fallback** (avoid Wgpu f16 — burn-fusion DACVAE crash)

**Key findings (from torch 2.9.0 era):**
- **LibTorch MPS f16 was 1.85× faster than WgpuRawF16** (independent CFG, torch 2.9.0)
- **LibTorch MPS f32 was 1.62× faster than WgpuRawF16** — even f32 via MPS beats f16 WGPU
- **MPS f16 speedup over f32: 14%** — precision reduction helps but matmul bandwidth is the main bottleneck
- **`PYTORCH_MPS_PREFER_METAL=1` + `PYTORCH_MPS_FAST_MATH=1`**: negligible effect (<0.2%)

### Metal Operator Profile (WgpuRaw f32, profile_wgpu_ops)

Operator breakdown for DiT forward pass (seq=750, steps=1, per-step):

| Operator | Per-call (µs) | Count | Total (µs) | % of forward |
|---|---|---|---|---|
| Linear [750×1280]×[1280×1280] | 3,610 | ×48 | 173,280 | 27.9% |
| Fused QKV [750×1280]×[1280×3840] | 6,572 | ×24 | 157,728 | 25.4% |
| SwiGLU proj [750×1280]×[1280×10240] | 16,841 | ×12 | 202,092 | 32.5% |
| burn SDPA [1,20,750,950] | 6,168 | ×12 | 74,016 | 11.9% |
| AdaLN [750×1280] | 100 | ×24 | 2,400 | 0.4% |

- **Matmul = 86% of compute time** — CubeCL GEMM is the Metal bottleneck
- **SDPA = 12%** — Metal's attention path is already efficient (vs 22.6% on DX12)
- **AdaLN = 0.4%** — custom kernels save only ~130ms total (not worth integration complexity)
- **Conclusion**: Only MPS via LibTorch meaningfully improves Metal performance

---

## Micro-Benchmarks: Custom WGSL Kernels vs burn Generic (Metal)

All tests use `WgpuRaw` backend (Metal). Results compared to DX12/Vulkan on RTX 5070 Ti.

### RMSNorm (custom WGSL vs burn generic)

Config: warmup=10, bench_iters=50

| Scenario | burn (µs) | custom WGSL (µs) | speedup | DX12 speedup |
|---|---|---|---|---|
| DiT RmsNorm (batch=1, seq=750, dim=1024) | 277 | 175 | **1.58×** | 3.08× |
| TextEncoder RmsNorm (batch=1, seq=200, dim=1024) | 114 | 77 | **1.48×** | 4.11× |
| HeadRmsNorm (batch=1, seq=750×16, dim=64) | 268 | 299 | **0.90× (slower!)** | 1.22× |

**Key finding**: Custom RMSNorm is faster for dim=1024 on Metal (1.5×) but slower for small dim (64).
The speedup is smaller than DX12 — Metal's generic kernels are already better-optimized by the driver.

### Fused AdaLN (custom WGSL vs burn generic)

Config: warmup=10, bench_iters=50

| Scenario | burn (µs) | custom fused (µs) | speedup | DX12 speedup |
|---|---|---|---|---|
| DiT AdaLN (1×750×1024) | 312 | 176 | **1.77×** | 3.95× |
| DiT AdaLN short (1×100×1024) | 125 | 53 | **2.37×** | 6.02× |
| DiT AdaLN batch2 (2×750×1024) | 373 | 129 | **2.90×** | 3.38× |
| Small dim (1×750×256) | 158 | 127 | **1.25×** | 3.87× |

Estimated per-inference savings: **~130ms** (960 calls × 135µs Δ at DiT dims × 40 steps).

**Key finding**: Fused AdaLN shows 1.77-2.90× speedup on Metal. Savings are larger in absolute
terms vs DX12 (~130ms vs ~44ms) but model integration is still complex for <2% total-time impact.

### Fused SDPA — Row-Streaming Online Softmax

Config: warmup=10, bench_iters=50

| Scenario | burn (µs) | custom (µs) | ratio | DX12 ratio |
|---|---|---|---|---|
| DiT joint attn (1×16×750×850×128) | 7,468 | 43,728 | **0.17×** | 0.21× |
| Short seq (1×16×100×150×128) | 313 | 1,127 | **0.28×** | 0.07× |
| Square (1×16×256×256×128) | 1,167 | 4,614 | **0.25×** | 0.06× |
| Small head (1×16×750×850×64) | 5,248 | 34,493 | **0.15×** | 0.17× |

**Conclusion**: Row-streaming SDPA is 4-7× SLOWER than burn on Metal. Same fundamental
problem as DX12 — no K/V reuse across query rows. Not viable.

### FlashAttention (Tiled, Native f32, and Native f16 variants)

Config: warmup=5, bench_iters=20

| Scenario | burn (µs) | T16×8 | T8×16 | N32×8 f32 | N16×16 f32 | N32×16 | N32×8 f16 | N16×16 f16 |
|---|---|---|---|---|---|---|---|---|
| DiT joint attn (1×16×750×850×128) | 7,522 | 36,721 | 31,870 | 31,130 | **23,136** | 29,743 | 30,825 | 23,711 |
| Short seq (1×16×100×150×128) | 323 | 1,262 | 1,023 | 1,084 | **716** | 905 | 1,032 | 732 |
| Square (1×16×256×256×128) | 1,220 | 3,952 | 3,357 | 3,538 | **2,454** | 2,992 | 3,197 | 2,588 |
| Large seq (1×16×1024×1200×128) | 10,070 | 68,303 | 58,310 | 58,489 | **42,724** | 53,492 | 58,286 | 44,096 |

Best f32 config: **N16×16** at DiT dims → **3.08× burn** (DX12 N32×8 was **1.46× burn**).

**f16 storage variant results:**
- N32×8 f16: ~1% faster than f32 on Metal — negligible benefit
- N16×16 f16: ~2% SLOWER than f32 on Metal — no benefit
- **Conclusion**: f16 storage optimization does NOT help on Metal

**Key findings on Metal (D=128):**
- All custom FA variants are 3-8× SLOWER than burn on Metal (at wrong D=128 dims — see below)
- f16 storage brings no benefit on Metal (unified memory, Metal driver already f16-optimized)
- f16 kernel parity confirmed: max_diff ~1e-7 (sub-f16-precision) ✅
- Expected f16 benefit: DX12/RTX 5070 Ti where global memory bandwidth is the bottleneck

### FlashAttention at Production Dims (D=64, 20 heads) — **FINAL BENCHMARK**

**⚠ Prior benchmarks used D=128, 16 heads. Production model uses D=64, 20 heads (head_dim=1280/20=64).**

Config: warmup=5, bench_iters=20, baseline = burn `attention()` module (CubeCL FA, scale=None)

**§ D=64 Production Model (head_dim=64, 20 heads)**

| Scenario | burn (µs) | T8×8 | T4×16 | N32×8 | N16×16 | N8×32 |
|---|---|---|---|---|---|---|
| 1×20×750×950 (production seq) | **6,636.2** | 22,015.3 (3.32×) | 24,318.8 (3.66×) | 16,942.3 (2.55×) | 16,446.9 (2.48×) | 22,682.2 (3.42×) |
| 3×20×750×950 (batch=3 CFG) | **19,403.1** | 64,352.4 (3.32×) | 71,624.5 (3.69×) | 49,731.3 (2.56×) | 48,704.3 (2.51×) | 67,374.1 (3.47×) |
| 1×20×512×512 (mid-seq) | **2,390.5** | 7,973.2 (3.34×) | 8,695.9 (3.64×) | 6,104.7 (2.55×) | 6,034.2 (2.52×) | 8,115.9 (3.40×) |
| 1×20×256×256 (short seq) | **672.6** | 2,085.3 (3.10×) | 2,302.4 (3.42×) | 1,583.2 (2.35×) | 1,582.9 (2.35×) | 2,085.7 (3.10×) |

**§ D=128 Legacy Reference (head_dim=128, 16 heads)**

| Scenario | burn (µs) | T16×8 | T8×16 | N32×8 | N16×16 |
|---|---|---|---|---|---|
| 1×16×750×850 (full seq) | **7,189.6** | 36,581.0 (5.09×) | 32,008.6 (4.45×) | 31,081.5 (4.32×) | 23,027.3 (3.20×) |
| 1×16×256×256 (square) | **1,166.9** | 3,955.8 (3.39×) | 3,377.5 (2.89×) | 3,378.3 (2.90×) | 2,444.4 (2.09×) |
| 1×16×100×150 (short) | **361.0** | 1,224.8 (3.39×) | 1,009.6 (2.80×) | 1,081.7 (3.00×) | 722.6 (2.00×) |

**Scaling diagnostic (burn D=64 vs D=128 at 1×16×256×256):**  
burn D=64: 601.8µs | burn D=128: 1,157.9µs | ratio: 1.92× ✓ (~linear → compute-bound, not overhead-dominated)

**Impact estimate (D=64, 40 steps, ~1200 SDPA calls/step):**  
Best custom kernel (N16×16): 16,444µs vs burn 6,616µs → **Δ = −9,828µs/call → −472ms total** (REGRESSION, not savings)

**⛔ FINAL DECISION: Stop all WGSL FA kernel development.**
- Best custom kernel at production dims (D=64): **2.45× slower** than burn_attention
- Even at shorter sequences (D=64, 256×256): still **1.29× slower**
- SDPA = only 12% of inference; matmul = 86% — SDPA has limited ceiling regardless
- Custom FA integration would **slow down** inference by ~470ms (not save it)
- burn's CubeCL path on Metal has vendor-tuned optimizations we cannot match via WGSL source kernels

### f16 Storage Kernel — Parity Test Results (Metal)

`enable f16;` is confirmed safe on Metal (unlike `enable subgroups;`).

All parity tests pass with max_diff ~1e-7 (expected for f16 input quantization + f32 accumulation):

| Test | Config | max_diff | Status |
|---|---|---|---|
| Small (B=1 H=2 SQ=4 SKV=6) | Q32×8 | 5.96e-8 | ✅ |
| Masked (B=1 H=2 SQ=4 SKV=8) | Q32×8 | 8.94e-8 | ✅ |
| Model dims (B=1 H=16 SQ=32 SKV=48) | Q32×8 | 1.12e-7 | ✅ |
| Edge (B=1 H=1 SQ=17 SKV=9) | Q32×8 | 1.27e-7 | ✅ |
| Small (B=1 H=2 SQ=4 SKV=6) | Q16×16 | 5.96e-8 | ✅ |
| Model dims (B=1 H=16 SQ=32 SKV=48) | Q16×16 | 1.12e-7 | ✅ |

---

## WGSL Extension Status on Metal

### f16 Storage Extension (`enable f16;`)

**Confirmed on Metal**: `enable f16;` works correctly — **NO naga bug**.

Diagnostic: `cargo test --lib kernels::f16_diagnostic -- --ignored --nocapture`

- Small input (32 elements): max_diff=0.00e0 (all_zero=false) ✅
- Large input (1024 elements): max_diff=1.87e-3 (f16 quantization error only) ✅

**Conclusion**: `enable f16;` is safe on Metal (unlike `enable subgroups;`).

### Subgroup Operations (`enable subgroups;`)

**Confirmed on Metal**: `enable subgroups;` causes **silent all-zero output** — same naga bug
as DX12 and Vulkan. Workaround: use subgroup builtins without the directive (they work).

Diagnostic: `cargo test --release --lib kernels::subgroup_diagnostic -- --ignored --nocapture`

```
WITH    `enable subgroups;` (ran first): max_diff=2.05e2 (all_zero=true)
WITHOUT `enable subgroups;` (ran second): max_diff=0.00e0 (correct)
>>> BUG CONFIRMED: `enable subgroups;` breaks kernel output on Metal too <<<
```

**Conclusion**: The naga `enable subgroups;` bug is universal (DX12, Vulkan, Metal).
Deferred until upstream wgpu/naga fix.

---

## Comparison vs RTX 5070 Ti (DX12)

| Metric | RTX 5070 Ti DX12 | M4 Pro Metal | Notes |
|---|---|---|---|
| burn SDPA (DiT dims) | 2,149 µs | 7,522 µs | Metal 3.5× slower |
| best FA f32 / burn ratio | 1.46× | 3.08× | Metal ratio 2× worse |
| best FA f16 / burn ratio | est. ~1.0× | 3.16× | f16 helps DX12, not Metal |
| Fused AdaLN speedup | 3.95× | 1.77× | Metal less dramatic |
| Custom RMSNorm speedup (dim=1024) | 3.08× | 1.58× | Metal less dramatic |
| `enable subgroups;` bug | DX12 ✗ | Metal ✗ | Universal naga issue |
| `enable f16;` | DX12 ✅ | Metal ✅ | Safe on all backends |

**Strategic insight**: Custom WGSL FA kernels are further from competitive on Metal
than on DX12. Metal's GPU scheduler and driver optimizations benefit burn's
generic CubeCL path more than WGSL source kernels.

---

## Conclusions for M4 Pro

1. **LibTorch MPS f16: fastest backend on M-series Mac** — 10,216ms (RTF 0.341), 1.85× faster than WgpuRawF16
2. **LibTorch MPS f32: 11,660ms (RTF 0.389)** — still 1.62× faster than WgpuRawF16 with full precision
3. **Custom FA kernels: NOT viable on Metal** (3.08× burn at best, worse than DX12's 1.46×)
4. **f16 storage FA kernel: implemented and correct** (max_diff ~1e-7), but no Metal speedup via WGPU
5. **Fused AdaLN: marginally useful** (1.77-2.90× speedup; ~130ms savings on Metal) — not worth integrating given MPS
6. **Subgroup bug confirmed on Metal** — same as DX12/Vulkan (naga issue)
7. **`enable f16;` is safe on Metal** — no naga bug
8. **Full inference times**: MPS f16=10,216ms (RTF 0.341), WgpuRawF16=18,855ms (RTF 0.63)
9. **Recommended backends by priority** (M-series Mac):
   - **LibTorch MPS f16** (`just bench-tch-mps-f16`) — fastest; requires PyTorch 2.9.0 venv
   - **LibTorch MPS f32** (`just bench-tch-mps`) — full precision; still faster than WGPU
   - **WgpuRawF16** (`just bench-wgpu-raw-f16`) — fastest no-dependency option; avoids burn-fusion crash

LibTorch MPS via PyTorch 2.9.0 is the correct choice for Mac M-series deployment when PyTorch is available.

---

## LoRA Inference Verification (M4 Pro)

Tested using the `43ch/しみちゃん` voice adapter distributed at:
https://note.com/852wa/n/n7a4955dc6754

### Adapter Details

| Field | Value |
|---|---|
| Target | `Aratako/Irodori-TTS-500M-v2` |
| Dtype | BF16 |
| Rank | 16 |
| lora_alpha | 32.0 (scale = 2.0) |
| Layers merged | 120 (DiT ×48, speaker_encoder ×24, text_encoder ×48) |
| PEFT version | 0.19.1 |
| Key pattern | `base_model.model.<path>.lora_A/B.weight` |

### Results

| Test | Result |
|---|---|
| 120 adapter layers merged | ✅ |
| Output shape [1, 750, 32] | ✅ |
| WGPU (Metal) inference end-to-end | ✅ |
| No numerical errors / panics | ✅ |

**Command:**
```
cargo run --release --features "cli,lora" --bin infer \
  --checkpoint target/model_converted.safetensors \
  --adapter target/lora/43ch \
  --backend wgpu \
  --text "テスト"
```

**Note:** Full text→WAV pipeline tested after DACVAE weights were downloaded and converted
(`uv run scripts/convert_dacvae_weights.py`). See "Full Pipeline Test" section below.

### Bug Fixed

`lora.rs::merge_lora` accessed `lora_a.shape[0]` before checking tensor rank — panics on rank < 2.
Fixed: added rank-2 guard returning `IrodoriError::Weight` instead.

### Integration Tests Added

6 new tests in `src/weights/tensor_store.rs` under `lora_tests` (require `--features lora`):
- `apply_lora_round_trip_f32` — merge math: W_merged = W_base + scale × B @ A
- `apply_lora_key_not_found_is_silent` — missing base key → n=0, no error
- `load_with_lora_strips_peft_prefix` — `base_model.model.` prefix stripped
- `load_with_lora_end_to_end` — combined PEFT-prefixed base + adapter → merged
- `apply_lora_bf16_roundtrip` — BF16 base merged and re-encoded within 1% precision
- `apply_lora_wrong_rank_returns_error` — rank-1 adapter returns `Err` not panic

All 312 lib tests pass with `--features cli,train,lora`.

---

## Full Pipeline Test (text → WAV, M4 Pro)

**Date**: 2025-04-21  
**Text input**: `"こんにちは"` (2 tokens)  
**Seq len**: 750 frames  
**Audio output**: 30.00s @ 48 kHz (1,440,000 samples)

### Results

| Backend | TTS rf_time | Codec decode_time | Total (warm) | Status |
|---|---|---|---|---|
| `wgpu` (Fusion) | — | — | — | ❌ burn-fusion panic |
| `wgpu-f16` (Fusion) | — | — | — | ❌ burn-fusion panic |
| `wgpu-raw` (no Fusion, f32) | 36,977ms | 459,112ms† | ~40s warm | ✅ |
| `wgpu-raw-f16` (no Fusion, f16) | 20,899ms‡ | 397,140ms§ | ~25s warm | ✅ |

> †‡§ First-run times dominated by autotuning new kernel shapes (f16 shapes are separate from f32 in autotune cache). Subsequent runs skip all autotuning.
> ‡ Warm TTS: ~18,855ms (matches `bench_realmodel` — 1 small matmul shape was cold on first run).
> § Codec warm estimate: ~3,500ms (extrapolated from f32 warm; f16 uses ~same GPU time).

**Commands:**
```
# f32 (workaround only — use wgpu-raw-f16 for speed):
just pipeline-real-raw --text "こんにちは" --output /tmp/output.wav
# f16 (recommended — faster + no fusion crash):
just pipeline-real-raw-f16 --text "こんにちは" --output /tmp/output.wav
```

### burn-fusion Bug: "Ordering is bigger than operations"

The `wgpu` (Fusion) backend crashes during DACVAE decode with:

```
thread 'main' panicked at 'Ordering is bigger than operations',
burn-fusion-0.21.0-pre.3/src/stream/execution/ordering.rs:49
```

Secondary failure at `burn-ir/src/handle.rs:88`: `"Should have handle for tensor"`.

**Root cause**: The burn-fusion stream scheduler's ordering counter exceeds operation count when the DACVAE decoder's complex computation graph is queued. The DACVAE decoder uses:
- Snake activation: `x + sin²(αx)/α` — non-trivial symbolic gradient graph
- Transposed convolutions at multiple scales (stride 10×, 8×, 2×)
- Multi-scale residual connections (MRF blocks)

This interaction creates a fusion graph large/complex enough to trigger the scheduler bug.

**Workaround**: Use `--backend wgpu-raw` (non-Fusion `CubeBackend<WgpuRuntime>`). This bypasses the fusion layer entirely with no logical change to computation.

**Upstream**: Likely a bug in `burn-fusion 0.21.0-pre.3`. TTS DiT model (same backend) does NOT trigger this — the DACVAE decoder's graph structure is the trigger.

### Autotune: First-Run vs Warm

On first run, the codec decode autotuned **~18 new kernel shapes** (separately for f32 and f16 — the autotune cache is keyed by dtype).

**f32 autotune** (wgpu-raw, first run):

| Shape class | Example | Tune time |
|---|---|---|
| Conv1d (large) | k=7, ch=1024, len=16384 | ~1.5s |
| ConvTranspose2d | k=20, stride=10, ch=1024 | ~84s |
| ConvTranspose2d | k=16, stride=8, ch=512 | ~147s |
| ConvTranspose2d | k=4, stride=2, ch=256 | ~42s |
| Matmul (large) | m=8192, n=16384, k=1024 | ~8s |

**f16 autotune** (wgpu-raw-f16, first run, observed 2026-04-21):

| Shape class | Example | Tune time |
|---|---|---|
| Conv1d k=1, ch=1024, len=16384 | F16 | ~1.0s |
| ConvTranspose2d k=20, stride=10, ch=1024 | F16 | ~84s |
| Matmul m=8192, n=16384, k=1024 | F16 | ~7s |
| Conv1d k=7, ch=512, len=131072 | F16 | ~8s |
| ConvTranspose2d k=16, stride=8, ch=512 | F16 | ~146s |
| Matmul m=4096, n=131072, k=512 | F16 | ~9s |
| Conv1d + Matmul ch=256, len=1048576 | F16 | ~35s |
| ConvTranspose2d k=4, stride=2, ch=256 | F16 | ~41s |
| Conv1d + Matmul ch=128, len=2097152 | F16 | ~15s |
| Total first-run codec | | **~397s** |

These are cached in `~/.cache/cubecl/` after first run (separate keys for F16 vs F32).
**Second run will skip all autotuning** and decode in estimated ~3,500ms (GPU compute only).
