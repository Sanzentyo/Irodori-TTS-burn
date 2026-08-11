# RTX 3060 Ti v4 length sweep

Measured on 2026-08-11 after generalizing the production fused WmHead, all
twelve decoder residual pointwise routes, the measured K7 residual tile
policies, and the decoder ConvTranspose routes across the supported audio
lengths. The WGPU path retains GPU-resident tensors between stages; CPU
readback is performed only for the separately reported readback-inclusive
boundary.

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
`/tmp/irodori-v4-length-sweep-dynamic-convt-final-attempt1-20260811`.
Its 564-entry manifest verifies in full, every timing family contains exactly
50 measured samples per length and runtime, and the tree is frozen as files
0444/directories 0555 without symlinks.

## Device-complete medians

| Audio | PyTorch RF | WGPU RF | RF speed | PyTorch codec | WGPU codec | Codec speed |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 s | 124.103 ms | 122.732 ms | 1.011x | 21.342 ms | 23.726 ms | 0.899x |
| 1 s | 126.838 ms | 129.265 ms | 0.981x | 34.609 ms | 32.515 ms | 1.064x |
| 2 s | 136.989 ms | 122.871 ms | 1.115x | 46.594 ms | 45.286 ms | 1.029x |
| 4 s | 166.605 ms | 172.453 ms | 0.966x | 90.803 ms | 111.956 ms | 0.811x |
| 8 s | 220.734 ms | 289.643 ms | 0.762x | 189.464 ms | 219.670 ms | 0.862x |

The 2-second RF stage passes the strict all-sample
WGPU-below-PyTorch-minimum gate at both boundaries. Codec passes the strict
all-sample gate at both 1 and 2 seconds: device-complete maxima are 33.110 and
46.137 ms versus Python minima 34.487 and 46.279 ms. Other lengths do not pass
both stages, so the overall multi-length goal remains open.

## CPU-readback-inclusive medians

| Audio | PyTorch RF | WGPU RF | RF speed | PyTorch codec | WGPU codec | Codec speed |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 s | 124.146 ms | 122.829 ms | 1.011x | 21.393 ms | 23.847 ms | 0.897x |
| 1 s | 126.881 ms | 129.375 ms | 0.981x | 34.673 ms | 32.665 ms | 1.061x |
| 2 s | 137.033 ms | 123.017 ms | 1.114x | 46.683 ms | 45.518 ms | 1.026x |
| 4 s | 166.652 ms | 172.562 ms | 0.966x | 90.938 ms | 112.207 ms | 0.810x |
| 8 s | 220.783 ms | 289.765 ms | 0.762x | 189.710 ms | 220.028 ms | 0.862x |

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
At that point the next fixed decoder target was ConvTranspose; the resulting
measurement follows. RF sequence-length scaling remains separately open, with
tail latency—not merely median—as the acceptance criterion.

## Dynamic ConvTranspose effect

The first upsampler now reuses its GPU-resident packed polyphase weight at
non-reference lengths. The remaining three use a zero-copy checkpoint-weight
view feeding tuned GPU GEMM and an ordered col2im finalizer. The cached-column
route is admitted only from one second upward: an initial 0.5-second diagnostic
showed a fresh-process convergence tail up to 31.860 ms, worse than the stable
K7-only fallback maximum. The final policy keeps that short route disabled.

Relative to the dynamic-K7 sweep, final WGPU codec medians change by +0.218,
-4.823, +0.059, -1.424, and -1.090 ms for 0.5/1/2/4/8 seconds. The 0.5- and
2-second differences are ordinary run variation; 1, 4, and 8 seconds improve.
Peak WGPU memory remains below 7 GiB on the 8 GiB GPU. The change is retained,
but the long-length codec gap now points primarily to the remaining residual or
materialization work rather than ConvTranspose.
