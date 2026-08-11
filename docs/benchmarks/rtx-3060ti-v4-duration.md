# v4 duration predictor on RTX 3060 Ti

This note records the production FP32 duration-predictor comparison after the
compact no-auxiliary WGSL path, corrected long-text workgroup dispatch, and
the measured T64 `w1 || w3` projection were enabled.  The controlling artifact
is:

```
/tmp/irodori-v4-duration-sweep-t64-attempt1-20260812
```

Its recursive `SHA256SUMS` verifies, `COMPLETE` is present, every file is mode
`0444`, every directory is mode `0555`, and the tree contains no symlinks.

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
| Short | 3 | 45.38 | head | 1.590 | 1.124 | 1.415x | FAIL (26/30) | 1.622 | 1.189 | FAIL (26/30) |
| Short | 3 | 45.38 | full | 61.958 | 17.113 | 3.621x | PASS | 61.991 | 17.197 | PASS |
| Medium | 12 | 111.60 | head | 1.589 | 1.330 | 1.194x | PASS | 1.621 | 1.388 | PASS |
| Medium | 12 | 111.60 | full | 62.149 | 24.776 | 2.508x | PASS | 62.182 | 24.857 | PASS |
| Long | 28 | 333.44 | head | 1.592 | 1.463 | 1.088x | PASS | 1.624 | 1.530 | PASS |
| Long | 28 | 333.44 | full | 62.283 | 34.203 | 1.821x | PASS | 62.316 | 34.289 | PASS |
| Very long | 61 | 685.14 | head | 1.594 | 1.835 | 0.869x | FAIL (0/30) | 1.628 | 1.900 | FAIL (0/30) |
| Very long | 61 | 685.14 | full | 62.447 | 38.291 | 1.631x | PASS | 62.479 | 38.377 | PASS |

The full production duration path passes the all-sample gate for every tested
length with and without CPU readback.  The isolated head now passes both
all-sample gates at 12 and 28 tokens.  At three tokens its median improves by
29%, but one fresh process contributes four early measured outliers, leaving
26/30 strict wins.  The 61-token head remains systematically slower.  This
distinction must be retained: the result proves a production-path win, not
that every duration substage is already faster than PyTorch.

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
- Dispatch the fused SwiGLU-plus-`w2` kernel by its 16-row output tile instead
  of its eight-thread local Y dimension.  This removes redundant long-text
  workgroups without changing arithmetic or introducing a copy.
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

An instrumented CubeCL profile ranks the four duration-head matmuls at about
47% of device work and the three fused SwiGLU-plus-`w2` launches at about 43%.
Preprocessing and residual finalization are about 1% each.  Profiling changes
absolute latency substantially, so these percentages are used only to rank
future work.  The evidence is
`/tmp/irodori-v4-duration-vlong-cubecl-profile-attempt1-20260811`.

## Remaining work

The next duration target is the remaining 61-token head gap, especially the
three fused SwiGLU-plus-`w2` launches and the input projection.  The short-head
fresh-process tail also needs removal rather than a looser statistic.  Any
follow-up must keep the same four text lengths, three fresh processes, and both
timer boundaries.  Adoption requires the head's maximum WGPU value to fall
below the Python minimum for every length; median improvement alone is
insufficient.
