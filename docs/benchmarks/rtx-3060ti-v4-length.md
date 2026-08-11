# RTX 3060 Ti v4 length sweep

Measured on 2026-08-11 after generalizing the production fused WmHead, all
twelve decoder residual pointwise routes, and the measured K7 residual tile
policies across the supported audio lengths. The WGPU path retains GPU-resident
tensors between stages; CPU readback is performed only for the separately
reported readback-inclusive boundary.

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

Current evidence:
`/tmp/irodori-v4-length-sweep-dynamic-k7-attempt1-20260811`.
Its 564-entry manifest verifies in full, every timing family contains exactly
50 measured samples per length and runtime, and the tree is frozen as files
0444/directories 0555 without symlinks.

## Device-complete medians

| Audio | PyTorch RF | WGPU RF | RF speed | PyTorch codec | WGPU codec | Codec speed |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 s | 123.454 ms | 122.561 ms | 1.007x | 21.304 ms | 23.508 ms | 0.906x |
| 1 s | 126.765 ms | 129.399 ms | 0.980x | 34.602 ms | 37.337 ms | 0.927x |
| 2 s | 136.701 ms | 122.579 ms | 1.115x | 46.540 ms | 45.227 ms | 1.029x |
| 4 s | 166.639 ms | 172.473 ms | 0.966x | 90.733 ms | 113.380 ms | 0.800x |
| 8 s | 220.803 ms | 289.681 ms | 0.762x | 189.507 ms | 220.759 ms | 0.858x |

The 2-second RF stage passes the strict all-sample
WGPU-below-PyTorch-minimum gate at both boundaries. Codec has the faster median
at two seconds but its tail overlaps (device-complete WGPU max 46.519 ms versus
Python min 46.237 ms), so no codec length passes the strict all-point gate in
this sweep. This stricter conclusion supersedes the narrower earlier two-second
campaign and illustrates why median-only claims are insufficient.

## CPU-readback-inclusive medians

| Audio | PyTorch RF | WGPU RF | RF speed | PyTorch codec | WGPU codec | Codec speed |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 s | 123.498 ms | 122.662 ms | 1.007x | 21.355 ms | 23.692 ms | 0.901x |
| 1 s | 126.814 ms | 129.513 ms | 0.979x | 34.667 ms | 37.508 ms | 0.924x |
| 2 s | 136.746 ms | 122.711 ms | 1.114x | 46.630 ms | 45.440 ms | 1.026x |
| 4 s | 166.686 ms | 172.577 ms | 0.966x | 90.873 ms | 113.653 ms | 0.800x |
| 8 s | 220.851 ms | 289.784 ms | 0.762x | 189.750 ms | 221.062 ms | 0.858x |

## Dynamic pointwise effect

Relative to the immediately preceding dynamic-WmHead sweep, device-complete
WGPU codec medians improved by 6.058, 13.115, 0.036, 53.013, and 93.032 ms for
0.5/1/2/4/8 seconds. This is direct evidence that the former two-second-only
pointwise selectors were forcing large generic fallbacks at other lengths.
All numerical gates and output hashes remain valid after the dynamic routing
change.

The change is retained because it materially improves four of five lengths and
is neutral at the already-specialized two-second length. It is not sufficient:
the resulting K7 work below is also required.

## Dynamic K7 effect

Relative to the dynamic-pointwise sweep, reusing the measured channel/dilation
K7 tile policies at decoder-valid lengths improves WGPU codec medians by 6.064,
14.491, -0.086, 61.207, and 124.508 ms for 0.5/1/2/4/8 seconds. The small
two-second regression is within run-to-run variation; the other four changes
are large and directionally consistent. Decoder length divisibility is checked
before selecting these routes, and every numerical/hash gate passes.

The dynamic K7 change is retained. It reduces the remaining codec median gaps
to 2.204 ms (0.5 s), 2.735 ms (1 s), 22.647 ms (4 s), and 31.252 ms (8 s).
Remaining work is now dominated by transposed-convolution/other fixed decoder
routes and RF sequence-length scaling, with tail latency—not merely median—as
the acceptance criterion.
