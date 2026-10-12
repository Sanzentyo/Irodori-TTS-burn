# v4 duration predictor on RTX 3060 Ti

This note records the production FP32 duration-predictor comparison after the
compact no-auxiliary WGSL path, corrected long-text workgroup dispatch, the
measured T64 `w1 || w3` projection, and a length-selected `w2` contraction were
enabled. The current result combines the frozen Python reference from the
original full sweep with a source-pinned WGPU follow-up:

```
/tmp/irodori-v4-duration-sweep-native-input-final-attempt1-20260812
/tmp/irodori-v4-duration-dual-route-final-attempt1-20260812
```

Both recursive `SHA256SUMS` files verify. The WGPU follow-up manifest has
SHA-256 `43257648c567fd58f7a520d0ea77509a38dd1cf5f7a392011d084eb0f3e9a467`;
its 12 processes pin the production sources and binary before and after the
workload. Every evidence file is mode `0444`, every directory is mode `0555`,
and neither tree contains symlinks.

## Protocol

- GPU: NVIDIA GeForce RTX 3060 Ti, Python CUDA index 1 and WGPU Vulkan adapter
  index 0 under the campaign's pinned device policy.
- Precision: strict FP32.  PyTorch autocast and TF32 are disabled and matmul
  precision is `highest`.
- Inputs: 3, 12, 28, and 61 valid text tokens, covering predicted lengths from
  45.38 to 685.14 acoustic frames.
- Sampling: three fresh processes per runtime and input; five warmups followed
  by ten measured calls per scope, for 30 measured values per cell.
- `head`: duration blocks starting from an already encoded condition.
- `full`: condition encoding, compaction, and duration prediction.
- Primary timer: pre-sync through device completion, excluding scalar CPU
  readback.
- Secondary timer: the same work through completion of an owned, contiguous,
  one-element float32 CPU readback.
- The strict performance check is pointwise: every WGPU measurement must be
  faster than the fastest PyTorch measurement for the same input, scope, and
  timer boundary.

## Results

All times are medians in milliseconds.  `PASS` means all 30 WGPU measurements
were below the corresponding PyTorch global minimum; it is stricter than a
median-only comparison.

| Input | Valid tokens | Predicted frames | Scope | PyTorch device | WGPU device | Speedup | Device gate | PyTorch readback | WGPU readback | Readback gate |
|---|---:|---:|---|---:|---:|---:|---|---:|---:|---|
| Short | 3 | 45.38 | head | 1.595 | 1.182 | 1.349x | FAIL (26/30) | 1.627 | 1.246 | FAIL (26/30) |
| Short | 3 | 45.38 | full | 62.072 | 18.577 | 3.341x | PASS | 62.105 | 18.658 | PASS |
| Medium | 12 | 111.60 | head | 1.590 | 1.326 | 1.199x | FAIL (28/30) | 1.623 | 1.394 | FAIL (28/30) |
| Medium | 12 | 111.60 | full | 62.328 | 24.922 | 2.501x | PASS | 62.361 | 25.005 | PASS |
| Long | 28 | 333.44 | head | 1.595 | 1.412 | 1.130x | PASS | 1.627 | 1.475 | PASS |
| Long | 28 | 333.44 | full | 62.422 | 34.079 | 1.832x | PASS | 62.457 | 34.156 | PASS |
| Very long | 61 | 685.14 | head | 1.596 | 1.573 | 1.015x | FAIL (17/30) | 1.629 | 1.636 | FAIL (1/30) |
| Very long | 61 | 685.14 | full | 62.562 | 37.818 | 1.654x | PASS | 62.599 | 37.903 | PASS |

The full production duration path passes the all-sample gate for every tested
length with and without CPU readback. The isolated 28-token head also passes
both gates. At 3 and 12 tokens the WGPU medians retain substantial leads, but
one fresh process in each case contains wall-synchronization tails; performance
is reported without filtering them. The 61-token head improves from the prior
1.814 ms median to 1.573 ms and now beats the 1.596 ms PyTorch median, but its
tail and CPU-readback-inclusive distributions still fail the strict all-point
gate. This distinction must be retained: the result proves a production-path
win, not that every duration substage already wins at every synchronization
boundary.

Across all 12 WGPU result documents, the maximum absolute difference from the
paired Python duration output is `9.536743e-7`, below the enforced `1e-4`
tolerance.  Every document reports deterministic output hashes and zero WGPU
uncaptured errors.

## Implemented production optimizations

- Compact the all-valid text prefix and remove absent speaker/caption state.
- Cache the no-auxiliary duration modulation values on GPU.
- Fuse each block's RMSNorm, fixed scale, and shift preprocessing.
- Consume the combined `w1 || w3` projection in a tiled SwiGLU-plus-`w2`
  kernel without materializing the activation tensor.
- Route the preceding `w1 || w3` projection through the zero-pack T64 WGSL
  matmul for the released compact B1 extent (1--64 valid tokens).  A separate
  exhaustive screen covered every integer extent in that interval: all 64
  outputs were bit-exact and all 64 medians beat Burn, with savings ranging
  from 3.98 to 68.20 microseconds per projection.  The frozen screen is
  `/tmp/irodori-v4-duration-projection-t64-exhaustive-attempt1-20260812`
  (`SHA256SUMS` SHA-256 `b1ddb60882144b2a25d40064ec6eb5601495efd890105ed311d8880281f2e79d`).
- Project the compact text state through the checkpoint-native output-major
  `512 -> 1024` input weight without packing.  The kernel transposes each
  16-by-64 weight tile only in workgroup memory and applies the learned bias in
  the final store.  All model tensors and the result remain GPU-resident.
- Dispatch the fused SwiGLU-plus-`w2` kernel by its 16-row output tile instead
  of its eight-thread local Y dimension.  This removes redundant long-text
  workgroups without changing arithmetic or introducing a copy.
- Keep the measured O32/K32 scalar contraction below 48 valid tokens and select
  an O64/K128 `vec4<f32>` contraction from 48 through the released 64-token
  limit. The long route halves output workgroups, reduces K-loop barriers, and
  preserves ascending-K FMA order without a host copy or activation tensor.
- Fuse residual-plus-gate finalization.
- Fuse final RMSNorm, scalar projection, PyTorch-compatible softplus, token
  reduction, and `log1p` into one dispatch.

All specialized selectors validate dtype, shape, physical stride,
contiguity, device identity, and hardware limits.  A failed selector falls
back to the existing generic operation instead of copying on the hot path.

## Long-text follow-up screen

The 61-token case was also used for a set of single-process diagnostics.  They
reuse the same frozen fixture, five warmups, ten measured calls, and both timer
boundaries, but are not pooled with the formal sweep above.

| Candidate | Head device median | Outcome |
|---|---:|---|
| Production O32/K32, 16-row tile | 1.936 ms | Retained |
| CubeCL extensive autotune | 1.945 ms | Rejected |
| CubeCL full autotune | 1.953 ms | Rejected |
| Materialized activation plus tuned `w2` matmul | 2.387 ms | Rejected |
| Fused O32/K64 | 1.944 ms | Rejected |
| Fused O64/K32 | 2.524 ms | Rejected |
| Fused 32-row tile | 1.981 ms | Rejected |
| Fused eight-row tile | 2.159 ms | Rejected |
| `vec4<f32>` global loads with scalar-order FMA | 2.634 ms | Rejected |
| 64-row fused SwiGLU-plus-`w2` | 2.995 ms | Rejected |

All candidates retained the expected duration value and deterministic output
hash.  The rejected kernel variants were removed rather than left as dormant
production branches.  The diagnostic trees are under
`/tmp/irodori-v4-duration-vlong-*` and
`/tmp/irodori-v4-duration-long-tile*`; each completed tree has a verified
manifest and read-only permissions.

The vector-load screen preserved the scalar K reduction order, total shared
memory, deterministic hash, and output tolerance, but regressed the head from
the formal 1.936 ms median to 2.634 ms; its readback-inclusive median was
2.702 ms.  The frozen evidence is
`/tmp/irodori-v4-duration-vlong-vec4-load-attempt1-20260811`.

The 64-row screen reduced row workgroups and repeated `w2` reads by four, but
its eight accumulators per thread and 12 KiB shared tile reduced occupancy.  It
regressed the current 61-token head and was removed immediately.  The frozen
diagnostic is `/tmp/irodori-v4-duration-swiglu-w2-t64-very-long-attempt1-20260812`.

An instrumented CubeCL profile ranks the four duration-head matmuls at about
47% of device work and the three fused SwiGLU-plus-`w2` launches at about 43%.
Preprocessing and residual finalization are about 1% each.  Profiling changes
absolute latency substantially, so these percentages are used only to rank
future work.  The evidence is
`/tmp/irodori-v4-duration-vlong-cubecl-profile-attempt1-20260811`.

## Current length-selected follow-up

The retained O64/K128 route was selected only after measuring all four text
lengths. Applying its earlier O64/K32 predecessor globally improved the
61-token case but regressed 3, 12, and 28 tokens, so the production selector
keeps those shorter cases on O32/K32. An O128/K64 candidate was rejected after
the benchmark detected nondeterministic predictions, and a fused inter-block
residual/preprocess candidate was removed because it did not improve the
61-token median. Neither rejected kernel remains connected to production.

The WGPU follow-up contains 30 measurements per input and reports one output
hash per input across all repeats and processes. All 12 documents report zero
uncaptured WGPU errors and maximum absolute prediction error no greater than
`9.536743e-7` versus Python.

## Remaining work

The next duration target is the remaining 61-token readback-inclusive head
tail, followed by the 3/12-token fresh-process synchronization tails. The full
duration predictor already wins all 30 points at every tested text length, so
future micro-optimization must not regress that production path. Any follow-up
must keep the same four text lengths, three fresh processes, and both timer
boundaries. Adoption requires the head's maximum WGPU value to fall below the
Python minimum for every length; median improvement alone is insufficient.
