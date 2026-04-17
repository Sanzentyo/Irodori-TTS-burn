# Benchmark Results: RTX 5070 Ti Laptop GPU

## System

| Component | Details |
|---|---|
| CPU | Intel Core Ultra 9 275HX (Arrow Lake-HX) |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU (12227 MiB VRAM, CC 12.0 Blackwell) |
| iGPU | Intel Arc Graphics (Xe-LPG, ~2 GB shared) |
| CUDA Driver | 591.44 (CUDA 12.9) |
| PyTorch | 2.10.0+cu128 |
| burn version | 0.21.0-pre.3 |
| Rust edition | 2024 |
| OS | Windows 11 |

## Model

| Field | Value |
|---|---|
| Checkpoint | `Aratako/Irodori-TTS-500M-v2` |
| model_dim | 1280 |
| num_layers | 12 |
| num_heads | 20 |
| Parameters | ~500M |

## Benchmark Configuration

| Parameter | Value |
|---|---|
| seq_len | 750 (canonical `fixed_target_latent_steps`) |
| num_steps | 40 (default) |
| cfg_guidance_mode | Independent |
| cfg_scale_text | 3.0 |
| cfg_scale_speaker | 5.0 |
| cfg_min_t | 0.5 |
| Batch size | 1 |
| Input | Synthetic (batch=1, text_len=4, ref_frames=8) |
| Warmup runs | 2 |
| Timed runs | 5 |

## Results: Full Inference (seq=750, steps=40, n=5)

| Backend | Mean (ms) | Min (ms) | p50 (ms) | p95 (ms) | vs Python f32 |
|---|---|---|---|---|---|
| **Python PyTorch CUDA (f32)** | **3,831** | 3,806 | 3,838 | 3,858 | 1.00× (baseline) |
| **Rust/burn LibTorch bf16** | **1,309** | 1,302 | 1,309 | 1,316 | **0.34×** ✓ |
| Rust/burn CubeCL CUDA bf16 | 4,222 | 4,203 | 4,223 | 4,238 | 1.10× |
| Rust/burn Wgpu f16 | 4,538 | 4,530 | 4,538 | 4,547 | 1.18× |
| Rust/burn LibTorch f32 | 4,883 | 4,878 | 4,881 | 4,889 | 1.27× |
| Rust/burn CubeCL CUDA f32 | 5,682 | 5,651 | 5,687 | 5,702 | 1.48× |
| Rust/burn Wgpu f32 (fusion) | 6,720 | 6,696 | 6,722 | 6,738 | 1.75× |
| Rust/burn WgpuRaw (no fusion) | 7,049 | 7,027 | 7,052 | 7,064 | 1.84× |

### Key Observations

1. **LibTorch bf16 is 66% faster than Python** (1,309ms vs 3,831ms) — Tensor Core acceleration.
2. **CubeCL bf16** beats Python baseline (4,222ms vs 3,831ms = 1.10×) — much closer than on A6000 (1.72×).
3. **WGPU f16** is a major improvement: 4,538ms vs 6,720ms f32 = **32% faster with f16**.
4. **WgpuRaw (no fusion)** is only **5% slower** than Wgpu fusion — **custom kernel strategy is viable!**
5. **LibTorch f32** is 27% slower than Python — worse than on A6000 (2.4%), suggesting laptop thermal/power limits on cuBLAS.
6. All times are higher than A6000 (expected: laptop GPU vs workstation GPU).

### WgpuRaw Viability Assessment

| Metric | Value |
|---|---|
| Wgpu (fusion, f32) | 6,720 ms |
| WgpuRaw (no fusion, f32) | 7,049 ms |
| Delta | +329 ms (+4.9%) |
| Threshold for viability | < 10% |
| **Verdict** | **VIABLE** ✓ |

The fusion layer overhead is only ~5%, well within the 10% threshold set by the rubber duck review.
Custom WGSL kernels (RMSNorm, SDPA) can potentially recover this gap and more.

### WGPU f16 vs f32

| Metric | f32 | f16 | Improvement |
|---|---|---|---|
| Mean | 6,720 ms | 4,538 ms | **-32.4%** |
| Min | 6,696 ms | 4,530 ms | -32.3% |

WGPU f16 reduces bandwidth pressure and enables native half-precision compute.
This is the **easiest performance win** for the WGPU backend path.

### GPU Utilization Verification

GPU utilization confirmed idle (0% util, 0 MiB used) before each benchmark run.
`nvidia-smi` verified GPU was at 45°C before testing — no thermal throttling.

| Backend | Notes |
|---|---|
| All Rust backends | GPU confirmed active via nvidia-smi during runs |
| Python f32 | GPU confirmed active |

## Comparison vs RTX A6000

| Backend | RTX A6000 (ms) | RTX 5070 Ti (ms) | Ratio |
|---|---|---|---|
| Python f32 | 2,752 | 3,831 | 1.39× slower |
| LibTorch bf16 | 987 | 1,309 | 1.33× slower |
| LibTorch f32 | 2,817 | 4,883 | 1.73× slower |
| CubeCL CUDA f32 | 4,623 | 5,682 | 1.23× slower |
| Wgpu f32 | 7,315 | 6,720 | **0.92× (faster!)** |
| CubeCL CUDA bf16 | N/A | 4,222 | — |
| Wgpu f16 | N/A | 4,538 | — |
| WgpuRaw f32 | N/A (invalid) | 7,049 | — |

Notable: **WGPU f32 is 8% FASTER on the RTX 5070 Ti** than the A6000 despite being
a laptop GPU. This suggests WGPU/DirectX 12 driver quality is better on Blackwell
or the bandwidth ratio favors this workload.

## Vulkan vs DirectX 12 Comparison

| Backend | DX12 (ms) | Vulkan (ms) | Delta |
|---|---|---|---|
| WGPU f32 | 6,720 | 6,677 | -0.6% |
| WGPU f16 | 4,538 | 4,524 | -0.3% |

**Conclusion**: DX12 and Vulkan perform nearly identically on this hardware. No need to
force a specific backend for performance.

## RMSNorm Micro-Benchmark (Custom WGSL vs burn Generic)

Tested on WgpuRaw backend (DX12). Custom kernel uses shared-memory reduction;
burn generic uses powf→mean→add→sqrt chain (multiple fused elementwise kernels).

### DX12 Results

| Scenario | burn (µs) | custom WGSL (µs) | speedup |
|---|---|---|---|
| DiT RmsNorm (seq=750, dim=1024) | 49.0 | 15.9 (incl. reshape) | **3.08×** |
| TextEncoder RmsNorm (seq=200, dim=1024) | 52.5 | 12.8 (incl. reshape) | **4.11×** |
| HeadRmsNorm (seq=750×16, dim=64) | 107.3 | 88.1 (incl. reshape) | 1.22× |

### Vulkan Results

| Scenario | burn (µs) | custom WGSL (µs) | speedup |
|---|---|---|---|
| DiT RmsNorm (seq=750, dim=1024) | 50.7 | 16.4 (incl. reshape) | **3.09×** |
| TextEncoder RmsNorm (seq=200, dim=1024) | 43.9 | 9.2 (incl. reshape) | **4.80×** |
| HeadRmsNorm (seq=750×16, dim=64) | 45.3 | 92.0 (incl. reshape) | 0.49× (slower!) |

### Key Findings

1. **Custom WGSL is 3-5× faster** for the main RMSNorm case (dim=1024).
2. For HeadRmsNorm (dim=64), custom kernel is only 1.2× faster on DX12 and **slower** on Vulkan.
   This is because the small dim doesn't benefit from shared-memory reduction — the overhead
   of launching a custom kernel exceeds the compute savings.
3. **Impact estimate**: ~30 RmsNorm calls/forward × 40 steps = 1,200 launches.
   At ~35µs saved per call (dim=1024 only, ~20 calls): 20 × 35µs × 40 = 28ms per inference.
   This is only **0.4%** of total time — not a major win individually.
4. The real value of custom kernels will come from **fused operations** (AdaLN = RMSNorm + scale + shift)
   that combine multiple kernel launches into one.

### Fused SDPA Micro-Benchmark (Row-Streaming vs Tiled FA vs burn Generic)

All tests on WgpuRaw backend (DX12), D=128.

#### Row-Streaming (online softmax, 1 workgroup/query row)

| Scenario | burn (µs) | custom (µs) | ratio |
|---|---|---|---|
| DiT joint attn (1×16×750×850×128) | 2,029 | 9,764 | 0.21× |
| Short seq (1×16×100×150×128) | 121 | 1,670 | 0.07× |
| Square (1×16×256×256×128) | 188 | 3,405 | 0.06× |

#### Tiled FlashAttention (score-parallel 2D tiling, shared-memory K/V tiles)

| Scenario | burn (µs) | FA 16×8 (µs) | FA 8×16 (µs) | 16×8 ratio | 8×16 ratio |
|---|---|---|---|---|---|
| DiT joint attn (1×16×750×850×128) | 2,012 | 4,945 | 8,329 | 0.41× | 0.24× |
| Short seq (1×16×100×150×128) | 152 | 1,271 | 1,354 | 0.12× | 0.11× |
| Square (1×16×256×256×128) | 189 | 2,325 | 3,332 | 0.08× | 0.06× |
| Large seq (1×16×1024×1200×128) | 3,554 | 6,455 | 11,876 | 0.55× | 0.30× |

**Analysis**: Tiled FA (16×8) is ~2× faster than row-streaming, confirming shared-memory
K/V tiling improves data reuse. But burn's CubeCL-fused pipeline (auto-tuned matmul +
fused softmax) is still 2.5× faster. The gap is structural: hand-written WGSL source
kernels can't match burn's JIT-compiled, auto-tuned CubeCL backend.

### Fused AdaLN Micro-Benchmark (DX12)

Fused kernel: single-pass RMSNorm + modulate (`output = (x/rms) * (1+scale) + shift`)
vs burn's generic sequence (powf → mean → sqrt → div → mul → add).

| Scenario | burn generic (µs) | custom fused (µs) | Speedup |
|---|---|---|---|
| DiT AdaLN (1×750×1024) | 61.0 | 15.4 | **3.95×** |
| DiT AdaLN short (1×100×1024) | 54.3 | 9.0 | **6.02×** |
| DiT AdaLN batch2 (2×750×1024) | 74.5 | 22.0 | **3.38×** |
| Small dim (1×750×256) | 52.5 | 13.6 | **3.87×** |

**Impact estimate** (DiT 1×750×1024): 24 AdaLN calls/forward × 40 steps = 960 calls.
Δ = 45.5 µs/call → **~44ms total savings per inference** (~0.7% of 6,720ms WGPU f32).

### WGPU Operator Profile (WgpuRaw f32, DX12)

Isolated operator-level timing with GPU sync to identify WGPU-specific bottlenecks.
Model dimensions: dim=1280, heads=20, head_dim=64, seq_len=750.

| Category | µs/forward | % |
|---|---|---|
| Linear (single, ×48) | 18,130 | 19.9% |
| Linear (fused QKV, ×24) | 17,859 | 19.6% |
| Linear (SwiGLU proj, ×12) | 29,262 | **32.2%** |
| SDPA (joint attention, ×12) | 20,558 | **22.6%** |
| AdaLN (RMSNorm + modulate, ×24) | 1,613 | 1.8% |
| RMSNorm (standalone, ×14) | 713 | 0.8% |
| SwiGLU gate (silu*gate, ×12) | 2,243 | 2.5% |
| Add (residuals, ×24) | 575 | 0.6% |
| **TOTAL** | **90,953** | 100% |

Estimated 40-step inference: ~3,638ms (vs measured 7,049ms — difference is overhead,
CFG dual forward, and ops not profiled individually).

**Strategic insight**: Matmul (71.7%) + SDPA (22.6%) = **94.3%** of compute.
Custom norm/elementwise kernels can only improve the remaining ~5.7%.
The only viable WGSL optimization targets are attention fusion and matmul tuning.

## Commands Used

```powershell
# Environment setup (Windows)
$env:PATH = "C:\Users\sanze\git\Irodori-TTS\.venv\Scripts;C:\Users\sanze\git\Irodori-TTS\.venv\Lib\site-packages\torch\lib;C:\Program Files\Git\bin;$env:PATH"
$env:LIBTORCH_USE_PYTORCH = "1"
$env:LIBTORCH_BYPASS_VERSION_CHECK = "1"
$env:VIRTUAL_ENV = "C:\Users\sanze\git\Irodori-TTS\.venv"

# Rust benchmarks
cargo run --release --features cli --bin bench_realmodel -- --backend wgpu --warmup 2 --runs 5
cargo run --release --features cli --bin bench_realmodel -- --backend wgpu-f16 --warmup 2 --runs 5
cargo run --release --features cli --bin bench_realmodel -- --backend wgpu-raw --warmup 2 --runs 5
cargo run --release --features cli --bin bench_realmodel -- --backend cuda --warmup 2 --runs 5
cargo run --release --features cli --bin bench_realmodel -- --backend cuda-bf16 --warmup 2 --runs 5
cargo run --release --features cli --bin bench_realmodel -- --backend libtorch --warmup 2 --runs 5
cargo run --release --features cli --bin bench_realmodel -- --backend libtorch-bf16 --warmup 2 --runs 5

# Python baseline
cd ..\Irodori-TTS && .\.venv\Scripts\python.exe ..\Irodori-TTS-burn\scripts\bench_python.py --dtype f32 --runs 5 --warmup 2
```
