# WGSL Compute Shader Patterns for ML Kernels

Common patterns used in Irodori-TTS-burn custom WGSL kernels.

---

## 1. Shared Memory Reduction (RMSNorm Pattern)

Used for reducing across a dimension (e.g., computing mean-square for normalization).

```wgsl
var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn rms_norm(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let row = wid.x;

    // Step 1: Each thread loads and accumulates partial sum
    var sum_sq: f32 = 0.0;
    for (var i = tid; i < hidden_size; i += 256u) {
        let val = input[row * hidden_size + i];
        sum_sq += val * val;
    }
    shared_data[tid] = sum_sq;
    workgroupBarrier();

    // Step 2: Tree reduction in shared memory
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if tid < stride {
            shared_data[tid] += shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    // Step 3: Normalize
    let rms = sqrt(shared_data[0] / f32(hidden_size) + eps);
    for (var i = tid; i < hidden_size; i += 256u) {
        let idx = row * hidden_size + i;
        output[idx] = (input[idx] / rms) * weight[i];
    }
}
```

### Key Constraints
- Max workgroup size: **256** (WebGPU portability limit)
- All bindings must be `read_write` (wgpu suballocator sharing)
- `workgroupBarrier()` required between shared memory writes and reads

### With Subgroups (Extension Path)
```wgsl
enable subgroups;

// Replace shared memory reduction with subgroup reduction
var partial = /* thread's partial sum */;
partial = subgroupAdd(partial);
// Only lane 0 of each subgroup has the full subgroup sum
// Then do a final reduction across subgroups via shared memory
```

---

## 2. Row-wise Softmax

For attention score normalization (each row independently).

```wgsl
var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let row = wid.x;

    // Pass 1: Find row max
    var local_max: f32 = -1e38;
    for (var i = tid; i < seq_len; i += 256u) {
        local_max = max(local_max, data[row * seq_len + i]);
    }
    shared_max[tid] = local_max;
    workgroupBarrier();

    // Tree reduction for max
    for (var s = 128u; s > 0u; s >>= 1u) {
        if tid < s { shared_max[tid] = max(shared_max[tid], shared_max[tid + s]); }
        workgroupBarrier();
    }
    let row_max = shared_max[0];

    // Pass 2: Compute exp and sum
    var local_sum: f32 = 0.0;
    for (var i = tid; i < seq_len; i += 256u) {
        let e = exp(data[row * seq_len + i] - row_max);
        data[row * seq_len + i] = e;  // Store exp temporarily
        local_sum += e;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction for sum
    for (var s = 128u; s > 0u; s >>= 1u) {
        if tid < s { shared_sum[tid] += shared_sum[tid + s]; }
        workgroupBarrier();
    }
    let row_sum = shared_sum[0];

    // Pass 3: Normalize
    for (var i = tid; i < seq_len; i += 256u) {
        data[row * seq_len + i] /= row_sum;
    }
}
```

---

## 3. Fused Attention (SDPA) — Future Kernel

Scaled Dot-Product Attention fused into a single kernel to avoid materializing
the N×N attention matrix. This is the highest-impact optimization target.

### Simplified Tiled Approach
```
For each query tile (Bq queries):
  Initialize O[Bq] = 0, m[Bq] = -inf, l[Bq] = 0
  For each key tile (Bk keys):
    Load Q[Bq, d], K[Bk, d], V[Bk, d] into shared memory
    Compute S = Q @ K^T (Bq × Bk tile)
    Update running max, exp-sum, output accumulator (online softmax)
  Write O[Bq] normalized by l[Bq]
```

### WGSL Considerations
- Shared memory budget: typically 16 KB per workgroup
- Tile sizes limited by shared memory (e.g., Bq=32, Bk=32, d=64 → 32×64×4×3 = 24 KB)
- No tensor cores in WGSL — all math is scalar/vector
- Without f16, memory bandwidth is 2× vs Flash Attention implementations

### Decision: When to Implement
- **Only if WGPU f32/f16 benchmarks show attention as primary bottleneck**
- Current profiling (nsys, CUDA) shows matmul at 59.5% — SDPA may not be the bottleneck
  for WGPU since WGPU doesn't use the same kernel fusion as CubeCL

---

## 4. Element-wise Fused Operations

For operations that can be fused: bias + activation, scale + add, etc.

```wgsl
@compute @workgroup_size(256)
fn fused_bias_swiglu(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    if idx >= total_elements { return; }

    // SwiGLU: swish(x1) * x2 where swish(x) = x * sigmoid(x)
    let x1 = gate[idx] + bias_gate[idx % hidden_dim];
    let x2 = up[idx] + bias_up[idx % hidden_dim];

    let swish_x1 = x1 / (1.0 + exp(-x1));
    output[idx] = swish_x1 * x2;
}
```

### Note on burn's Fusion
burn's `Fusion<CubeBackend<...>>` already fuses elementwise ops automatically.
Custom WGSL elementwise kernels are only useful for **WgpuRaw** (no fusion) backend
or for fusing elementwise ops with reductions (which burn can't auto-fuse).

---

## 5. Best Practices

### Memory Access
- **Coalesced access**: Ensure adjacent threads read adjacent memory addresses
- **Vectorized loads**: Use `vec4<f32>` loads when alignment permits (4× bandwidth)
- **Avoid bank conflicts**: In shared memory, ensure stride != 32 (or add padding)

### Workgroup Size
- Use **256** as default (max WebGPU-portable size)
- For native-only (DX12/Vulkan/Metal), can use up to 1024
- Comment when using >256: `// Native-only: WebGPU max is 256`

### Dispatch
```rust
let num_workgroups = (total_elements + workgroup_size - 1) / workgroup_size;
// For 2D: dispatch (rows, 1, 1) with each workgroup handling one row
```

### Debugging
- Use `storageBarrier()` for cross-workgroup sync (rare, expensive)
- `workgroupBarrier()` for within-workgroup sync
- No `printf` in WGSL — use debug buffers to write intermediate values

### WebGPU Portability Checklist
- [ ] Max workgroup size ≤ 256
- [ ] No extensions without fallback path
- [ ] All bindings `read_write` (wgpu constraint)
- [ ] No workgroup storage > 16 KB
- [ ] Test on both DX12 and Vulkan backends
