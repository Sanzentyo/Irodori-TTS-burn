# RTX 3060 Ti v4 length sweep

Measured on 2026-08-11 after generalizing the production fused WmHead from the
fixed 96,000-sample output to every non-zero 240-sample multiple.

## Protocol

- Strict FP32 on both runtimes; TF32 and autocast disabled
- Lengths: 0.5, 1, 2, 4, and 8 seconds at 48 kHz
- Five fresh processes per runtime and length
- Two warmups excluded, ten measured repetitions per process
- Primary: pre-sync to device-complete, preserving GPU-resident output
- Secondary: the same start through owned contiguous FP32 CPU readback
- Accuracy, determinism, RF work count, pins, and timing manifests are enforced
  for every repetition
- Performance does not filter the sweep artifact

Evidence:
`/tmp/irodori-v4-length-sweep-dynamic-wmhead-attempt1-20260811`.
The manifest verifies in full and the tree is frozen as files 0444/directories
0555.

## Device-complete medians

| Audio | PyTorch RF | WGPU RF | RF speed | PyTorch codec | WGPU codec | Codec speed |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 s | 124.154 ms | 122.540 ms | 1.013x | 21.290 ms | 35.630 ms | 0.598x |
| 1 s | 126.802 ms | 129.293 ms | 0.981x | 34.597 ms | 64.944 ms | 0.533x |
| 2 s | 136.704 ms | 122.719 ms | 1.114x | 46.536 ms | 45.177 ms | 1.030x |
| 4 s | 166.552 ms | 172.553 ms | 0.965x | 90.719 ms | 227.600 ms | 0.399x |
| 8 s | 220.893 ms | 289.575 ms | 0.763x | 189.524 ms | 438.300 ms | 0.432x |

Only the 2-second case passes the strict all-sample WGPU-below-PyTorch-minimum
gate for both RF and codec. CPU-readback-inclusive results have the same pass/
fail pattern.

## Dynamic WmHead effect

Relative to the immediately preceding pinned sweep, WGPU codec medians changed
by -0.119, -0.068, -0.093, -0.260, and -1.573 ms for 0.5/1/2/4/8 seconds.
All changes are improvements, but the short-length changes are within normal
run-to-run variation. The 8-second improvement is consistent with removing the
old generic WmHead fallback, yet it is small relative to the remaining codec
gap.

The next optimization must generalize the length-specialized residual,
pointwise, and transposed-convolution routes. A fast 2-second specialization
alone is not an acceptable production result.
