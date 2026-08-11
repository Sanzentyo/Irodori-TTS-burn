# RTX 3060 Ti v4 length sweep

Measured on 2026-08-11 after generalizing both the production fused WmHead and
all twelve decoder residual pointwise routes across the supported audio
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
`/tmp/irodori-v4-length-sweep-dynamic-pointwise-attempt1-20260811`.
Its 564-entry manifest verifies in full, every timing family contains exactly
50 measured samples per length and runtime, and the tree is frozen as files
0444/directories 0555 without symlinks.

## Device-complete medians

| Audio | PyTorch RF | WGPU RF | RF speed | PyTorch codec | WGPU codec | Codec speed |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 s | 123.663 ms | 122.650 ms | 1.008x | 21.287 ms | 29.572 ms | 0.720x |
| 1 s | 126.749 ms | 129.240 ms | 0.981x | 34.592 ms | 51.829 ms | 0.667x |
| 2 s | 136.678 ms | 122.538 ms | 1.115x | 46.554 ms | 45.141 ms | 1.031x |
| 4 s | 166.597 ms | 172.492 ms | 0.966x | 90.845 ms | 174.587 ms | 0.520x |
| 8 s | 220.778 ms | 289.611 ms | 0.762x | 189.488 ms | 345.268 ms | 0.549x |

The 2-second case passes the strict all-sample WGPU-below-PyTorch-minimum gate
at the device-complete boundary for both RF and codec. At the
readback-inclusive boundary, RF still passes but codec narrowly overlaps:
WGPU max 46.255 ms versus Python min 46.203 ms. No other length passes the
strict all-point gate.

## CPU-readback-inclusive medians

| Audio | PyTorch RF | WGPU RF | RF speed | PyTorch codec | WGPU codec | Codec speed |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 s | 123.706 ms | 122.737 ms | 1.008x | 21.337 ms | 29.702 ms | 0.718x |
| 1 s | 126.793 ms | 129.346 ms | 0.980x | 34.656 ms | 51.982 ms | 0.667x |
| 2 s | 136.722 ms | 122.663 ms | 1.115x | 46.642 ms | 45.437 ms | 1.027x |
| 4 s | 166.646 ms | 172.683 ms | 0.965x | 90.985 ms | 174.814 ms | 0.520x |
| 8 s | 220.833 ms | 289.744 ms | 0.762x | 189.734 ms | 345.591 ms | 0.549x |

## Dynamic pointwise effect

Relative to the immediately preceding dynamic-WmHead sweep, device-complete
WGPU codec medians improved by 6.058, 13.115, 0.036, 53.013, and 93.032 ms for
0.5/1/2/4/8 seconds. This is direct evidence that the former two-second-only
pointwise selectors were forcing large generic fallbacks at other lengths.
All numerical gates and output hashes remain valid after the dynamic routing
change.

The change is retained because it materially improves four of five lengths and
is neutral at the already-specialized two-second length. It is not sufficient:
long codec paths remain 1.92x (4 s) and 1.82x (8 s) slower than PyTorch by
median, and RF loses at 1, 4, and 8 seconds. The next work must target the
remaining length-specialized K7 residual and transposed-convolution paths and
the RF sequence-length scaling, with the same device/readback measurement
contract.
