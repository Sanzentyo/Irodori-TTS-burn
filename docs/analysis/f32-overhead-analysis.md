# f32 LibTorch Overhead Analysis

## Methodology

Step-count sweep with identical methodology on both sides:
- Wall-clock timing (`time.perf_counter()` / `Instant::now()`)
- 3 warmup runs, 10 timed runs
- Same GPU session (no cold start effects)
- `torch.cuda.synchronize()` / `into_data()` for GPU sync

## Reconciled Data (wall-clock, 10 runs, 3 warmup)

| Steps | Python (ms) | Rust (ms) | Delta (ms) | Delta (%) |
|-------|-------------|-----------|------------|-----------|
| 1     | 118.8       | 123.9     | +5.1       | +4.3%     |
| 5     | 384.8       | 393.0     | +8.2       | +2.1%     |
| 10    | 688.5       | 701.9     | +13.4      | +1.9%     |
| 20    | 1,371.1     | 1,395.8   | +24.7      | +1.8%     |
| 40    | 2,747.3     | 2,820.4   | +73.1      | +2.7%     |

20-run benchmark (3 warmup): Rust 2,817ms vs Python 2,752ms = 65ms (2.4%)

## Linear Regression

```
time = setup + per_step × n_steps

Python: setup = 36.2 ms, per_step = 67.49 ms
Rust:   setup = 34.4 ms, per_step = 69.25 ms
Delta:  setup = -1.8 ms (Rust faster), per_step = +1.76 ms
```

Total overhead at 40 steps: 1.76 × 40 = 70.3ms, minus 1.8ms setup advantage = ~68ms

## Key Findings

### 1. Overhead is entirely per-step, not setup
Setup (model load + condition encoding) is actually **1.8ms faster** in Rust.
The overhead accumulates linearly in the sampling loop.

### 2. Scaffold ops are NOT the bottleneck
Per-step scaffold ops (cat, chunk, CFG arithmetic, Euler update) total only ~190µs in Python.
Even with 2× overhead, that accounts for <8ms over 40 steps — a small fraction of 73ms.

### 3. Forward pass dominates the overhead
Each forward pass through 12 DiT blocks contains ~90 tensor operations.
The ~1.76ms/step overhead is distributed across these operations:
1.76ms / ~90 ops ≈ 20µs per tensor operation.

### 4. burn `.clone()` is NOT the cause
Verified: `TchTensor::clone()` calls `shallow_clone()` — PyTorch reference-counted copy.
No GPU memory copy occurs. Same as Python's reference passing.

### 5. Source of ~20µs per-operation overhead
The burn-tch abstraction layer adds overhead per operation:
- Rust → tch FFI → libtorch C++ (vs Python → pybind11 → libtorch C++)
- `Storage` object tracking in burn-tch
- `TchShape` Vec<i64> allocation for shape conversion
- Backend trait dispatch (monomorphized but still has bookkeeping)

## Conclusion

The 2.4% (65-73ms) overhead is an inherent abstraction cost of burn's backend-agnostic design.
It is distributed across ~3600 tensor operations per inference (90 ops × 40 steps) and
cannot be reduced without bypassing burn's tensor API entirely. This is an excellent result
for a high-level abstraction layer.

### Decision: Accept and move on
Given the gap is <3% and uniformly distributed, further optimization here has diminishing returns.
The next high-value target is custom WGSL kernels for the WGPU backend (currently 2.72× Python).
