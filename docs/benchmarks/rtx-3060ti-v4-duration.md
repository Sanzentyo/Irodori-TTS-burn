# v4 duration predictor on RTX 3060 Ti

This note records the production FP32 duration-predictor comparison after the
compact no-auxiliary WGSL path and the corrected long-text workgroup dispatch
were enabled.  The controlling artifact is:

```
/tmp/irodori-v4-duration-sweep-dispatch-fix-attempt1-20260811
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
| Short | 3 | 45.38 | head | 1.590 | 1.463 | 1.086x | PASS | 1.622 | 1.528 | PASS |
| Short | 3 | 45.38 | full | 61.982 | 19.676 | 3.150x | PASS | 62.015 | 19.788 | PASS |
| Medium | 12 | 111.60 | head | 1.589 | 1.556 | 1.022x | FAIL | 1.622 | 1.621 | FAIL |
| Medium | 12 | 111.60 | full | 62.123 | 24.927 | 2.492x | PASS | 62.156 | 25.008 | PASS |
| Long | 28 | 333.44 | head | 1.595 | 1.638 | 0.974x | FAIL | 1.627 | 1.704 | FAIL |
| Long | 28 | 333.44 | full | 62.275 | 34.327 | 1.814x | PASS | 62.309 | 34.414 | PASS |
| Very long | 61 | 685.14 | head | 1.596 | 1.936 | 0.824x | FAIL | 1.629 | 2.003 | FAIL |
| Very long | 61 | 685.14 | full | 62.361 | 38.311 | 1.628x | PASS | 62.395 | 38.405 | PASS |

The full production duration path passes the all-sample gate for every tested
length with and without CPU readback.  The isolated head now passes both
all-sample gates for the three-token input.  At 12 tokens its medians are
competitive but its slowest samples still miss the strict gate; at 28 and 61
tokens it remains slower in median.  This distinction must be retained: the
result proves a production-path win, not that every duration substage is
already faster than PyTorch.

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
- Dispatch the fused SwiGLU-plus-`w2` kernel by its 16-row output tile instead
  of its eight-thread local Y dimension.  This removes redundant long-text
  workgroups without changing arithmetic or introducing a copy.
- Fuse residual-plus-gate finalization.
- Fuse final RMSNorm, scalar projection, PyTorch-compatible softplus, token
  reduction, and `log1p` into one dispatch.

All specialized selectors validate dtype, shape, physical stride,
contiguity, device identity, and hardware limits.  A failed selector falls
back to the existing generic operation instead of copying on the hot path.

## Remaining work

The next duration target is the long-text head itself, especially the two
1024-wide block projections.  Any follow-up must keep the same four text
lengths, three fresh processes, and both timer boundaries.  Adoption requires
the head's maximum WGPU value to fall below the Python minimum for every
length; median improvement alone is insufficient.
