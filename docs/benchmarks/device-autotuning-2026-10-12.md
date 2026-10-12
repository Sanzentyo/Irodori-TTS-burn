# RTX / Intel device tuning, 2026-10-12

This campaign adds persistent, accuracy-checked selection over 21 attention,
22 MLP contraction, 13 SwiGLU expansion, 12 QKV projection and 12 separate
attention-output projection candidates (80 operation/kernel choices, including
24 cooperative-matrix choices available only to FP16).
It also fixes an Intel numerical
error in CubeK's logical-plane reductions. FP32 and FP16 are measured separately;
the codec remains FP32. Python/XPU dependencies are confined to the reference
environment and do not become Rust library dependencies.

Raw receipts, logs, fixtures and the serial campaign driver are in
`/home/sanzentyo/benchmark-artifacts/irodori-device-tuning-20261012/`.
Earlier speech.cpp and feature-parity fixtures are in
`/home/sanzentyo/benchmark-artifacts/irodori-feature-parity-20261012/`.
The baseline for Intel is the corrected portable graph: the original graph's
incorrect output cannot serve as an accepted performance baseline.

## Hardware and reference environment

- NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12 GB, driver 595.99.02, 120 W limit.
- Intel Graphics (ARL), PCI 8086:7d67, i915, Mesa 26.0.8-1ubuntu0.3, Vulkan.
- Intel PyTorch XPU: torch/torchaudio 2.10.0+xpu, triton-xpu 3.6.0,
  Level Zero runtime 26.05.37020.3-1 and IGC 2.28.4-4. The runtime packages were
  extracted into the artifact directory; no system driver replacement was needed.
- CUDA reference: PyTorch 2.10.0+cu128. Versions and upstream revisions match
  the [preceding comparison](rtx-5070ti-synthesis-parity-2026-10-12.md).

Intel's render-node ACL was granted by the user. `probe_wgpu_adapter --integrated`
and a PyTorch XPU tensor calculation both succeeded. PyTorch's default SDPA path
failed on this GPU in both precisions with `could not create a primitive`.
The explicit `--sdpa math` reference works; no automatic backend substitution
is hidden in the benchmark.

The speech.cpp harness now prints the device description and supports
`--list-devices`. With Intel access enabled, its enumeration is Intel `Vulkan0`
and NVIDIA `Vulkan1`; prior indices must not be assumed to identify the same GPU.

`scripts/export_meanflow_oracle.py` now supports `--device cuda|xpu|cpu`, explicit
SDPA selection, warmups/repeats and several frame counts in one model load.
For XPU the campaign invokes it through the separate `pytorch-xpu` uv project,
with `LD_LIBRARY_PATH` pointing to `intel-runtime/root/usr/lib/x86_64-linux-gnu`.
The ordinary script's pinned CUDA environment is retained for reproducibility.

## Intel correctness fix

The adapter reports subgroup sizes 8–32. CubeK had treated the maximum (32) as
the actual subgroup size; Vulkan permits a shader to use a smaller subgroup.
That makes plane reductions sum only part of a logical row and makes some
shuffle-based algorithms invalid. The problem was observable before attention:
small matrix products and text-encoder states were already incorrect.

- Direct probes of `1×256×512` and `1×512×1536` products had maximum absolute
  errors of 0.1242 and 0.4796, respectively.
- The old two-second model output had latent/audio SNR of about −1.27/−1.83 dB.
- GEMM Dot now reduces each logical plane through workgroup memory when physical
  subgroup width can vary. Fixed-width adapters retain the native plane sum.
  Tail invocations participate in barriers without reading or writing invalid
  matrix coordinates. The paired Dot epilogue uses the same reduction.
- Algorithms whose shuffles require a fixed physical width reject incompatible
  devices before tuning. They are not timed or cached as valid candidates.
- FP32 kernel policy v6 and FP16 v7 invalidate old selections. Cache namespaces
  also include the actual driver and adapter, rather than an enumeration index.

The direct GEMM regression compares FP32 and FP16 partial tiles with CPU f64
products on both devices. End-to-end comparisons use the official CUDA FP32
oracle, not the previously incorrect Intel graph.

## Short-output decoder correction

Profiling found a separate latency cliff below 25 latent frames. All four
upsamplers switched from the existing WGSL routes to generic convolution.
At 13 frames on Intel, the four stages took about 113, 297, 527 and 162 ms;
the complete decode/readback took 1,185 ms, despite a 25-frame decode being
about 204 ms. This was a routing limit, not a required shader geometry limit.

The polyphase and cached-col2im routes now admit every positive released-stage
length, retaining all module, alignment, bounds and resource checks. No new
fallback layer was added. The 13-frame decode/readback takes 121.913 ms after
the change, approximately 9.7× faster, with waveform SNR 108.617 dB against the
official CUDA FP32 decoder. One-frame and first-stream-window (12-frame) oracles
also pass on both adapters. Their minimum SNR is 108.6 dB across all six checks;
each profiler run reports zero uncaptured WGPU errors. These are decoder-only
timings, distinct from complete synthesis below.

## What is tuned and stored

Attention candidates cover a strict FP32 graph, Burn SDPA, cached score softmax
with 64/128/256 lanes, fixed-32 subgroup softmax, optional UnitMin PV products,
and seven native WGSL query/key-value tile pairs. The MLP candidates cover the
FP32 graph, native-precision Burn graph, scalar/vector WGSL, K16/K32,
32/48/64/96/128-row tiles, smaller column tiles, prefetch, double buffering,
swizzled staging and split-K2.
Expansion, QKV and separate output projections share a two-input tuner, with
the operation kind included in its key. They compare the graph with scalar/vector tiles, K16/K32,
prefetch and wider row tiles. Existing fused QKV projection/materialization
routes retain their combined operation; separate projections are tuned. Existing
fused attention-output/residual routes also remain intact.

FP16 additionally measures five explicit CubeK cooperative-matrix plans:
single-row, multi-row, double buffering, and ordered buffering with four/eight
rows. These use FP32 accumulators and remain within the WGPU backend. Measuring
the actual shape avoids inheriting a poor decision from Burn's power-of-two
matmul buckets. FP32 does not use half inputs or TF32 through these candidates.

Four mixed-precision attention candidates combine cooperative FP16-input QK
matmul with FP32 score storage and the cached WGSL FP32 softmax, followed by
FP16-input PV matmul with FP32 accumulation. Probability conversion and layout
copies remain inside the measured operation. This avoids prematurely rounding
attention scores while using the adapter's matrix hardware. The single-row and
double-buffered QK/PV combinations are measured separately and must pass the
same numerical gate. Unsupported devices reject these candidates before timing.
They are admitted from 64 query frames: the short MeanFlow screen showed only
a small latency change but worse waveform agreement at 50 frames (31.19 to
28.84 dB), so that short mixed-precision route was rejected. Short requests keep
the existing accuracy-checked candidates. No tolerance was loosened to admit a
faster kernel.

Cold tuning first compiles and validates candidates outside the timer. This also
completes nested Burn tuning before CubeCL timestamp queries begin. Finite
outputs must satisfy elementwise `atol + rtol * abs(reference)`:
`2e-5 / 2e-5` for FP32, `0.002 / 0.004` for FP16. Unsupported or inaccurate
candidates are rejected explicitly. Contiguous copies and mask conversion are
included in the candidate operation being timed. Final acceptance also checks
the complete generated latent and waveform against the official reference.

CubeCL owns the timing and SQLite persistence; no second selection store or
exception-driven chain of alternative kernels was added. The key includes the
operation schema, adapter/driver, dtype, dimensions and strides. MLP keys also
include batch and sequence separately, because they change residual-gate indexing.
The CLI enables this policy by default; library users can explicitly select
`RuntimeRoutePolicy::Autotuned`. Approved whole-model profiles remain separate.
See [cache usage and invalidation](../synthesis.md#device-tuning-and-its-persistent-cache).
The Vulkan database also contains CubeCL's generated SPIR-V compilation entries;
handwritten WGSL pipelines themselves are compiled again in a new process.

The final cache audit exposed a migration issue in the development cache:
adding the four FP16 attention candidates also changes CubeCL's candidate-list
checksum for FP32. CubeCL's current hydration can select an older checksum first
when both checksums share one logical key, causing repeated tuning at startup.
The attention key schema is now advanced for both precisions. Old database rows
are retained and ignored. Candidate-list, order or acceptance-policy changes
must advance the operation schema even when a candidate is unavailable for a
particular precision. This avoids another cache owner or an upstream fork.

## Measurement boundaries

MeanFlow uses the v4.1 Small MF checkpoint, four steps, `こんにちは。`, no auxiliary
condition, and CPU-generated FP32 noise with seed 42. Lengths are 13, 25, 50,
100 and 200 latent frames (0.52, 1, 2, 4 and 8 seconds). All precisions cast the
same source noise once. RF uses the existing four-second caption-conditioned
v4.1 fixture and its original schedule.

Rust samples use two warmups and seven measured requests in a resident process.
RF timing includes condition encoding, sampling and latent CPU readback; audio
timing additionally includes FP32 decoding and waveform CPU readback. Neither
includes model loading or file writing. GPU workloads run serially, with builds
excluded from timing runs. The first cold request additionally performs tuning;
it is excluded from the warm medians.
Input tokenization is outside these fixed-tensor comparisons. PyTorch references
use two warmups and five measured requests; speech.cpp uses two and seven.

The dense microbenchmark uses alternating ABBA/BAAB order, 12 distinct layer
weights, two warmups and eight blocks. GPU timestamp and device-complete timing
are both retained. A new untimed preparation pass prevents nested Burn tuning
from invalidating outer timestamp queries. Synthetic dense results are not used
alone to accept an end-to-end route.

## Reference observations

### Accepted complete-model results

Each cell below is **RF / audio milliseconds**, median of seven warm requests.
RF includes condition encoding and latent readback; audio additionally includes
the decoder and waveform readback. These are MeanFlow's four-step requests.

| Generated seconds | RTX FP32 | RTX FP16 | Intel FP32 | Intel FP16 |
| ---: | ---: | ---: | ---: | ---: |
| 0.52 | 48.037 / 57.784 | 30.325 / 39.924 | 316.633 / 437.247 | 290.863 / 411.517 |
| 1 | 51.629 / 65.775 | 29.458 / 43.396 | 321.917 / 528.496 | 313.413 / 517.709 |
| 2 | 54.949 / 78.827 | 29.844 / 54.940 | 337.686 / 726.203 | 327.941 / 722.301 |
| 4 | 61.149 / 112.998 | 35.693 / 87.858 | 528.188 / 1333.143 | 521.288 / 1331.049 |
| 8 | 82.504 / 184.618 | 45.431 / 150.407 | 924.979 / 2553.671 | 916.164 / 2561.457 |

Against the static route controls from the beginning of this campaign, the
two-second audio boundary changes as follows. Intel's control includes the
subgroup correctness repair and uses the portable graph.

| Device / precision | Static control ms | Tuned ms | Reduction |
| --- | ---: | ---: | ---: |
| RTX FP32 | 107.478 | 78.827 | 26.7% |
| RTX FP16 | 57.618 | 54.940 | 4.6% |
| Intel FP32 | 1260.954 | 726.203 | 42.4% |
| Intel FP16 | 974.805 | 722.301 | 25.9% |

The four-second RF fixture uses 40 Euler steps with caption/text guidance:

| Device / precision | RF ms | Audio ms | Waveform SNR dB |
| --- | ---: | ---: | ---: |
| RTX FP32 | 673.694 | 726.173 | 97.670 |
| RTX FP16 | 303.868 | 354.531 | 31.980 |
| Intel FP32 | 7231.776 | 8036.166 | 101.556 |
| Intel FP16 | 7377.592 | 8174.903 | 43.960 |

The matched RTX RF static controls are 1010.548 ms in FP32 and 351.634 ms in
FP16. FP32 improves 28.1%. FP16's final 354.531 ms is 0.8% slower than its static
control, while SNR improves from 25.498 to 31.980 dB. The mixed attention path
recovers most of the earlier accuracy-checked route's 384.104 ms, but does not
establish a latency win over the static FP16 control. Intel's RF FP16 is also
slower than its FP32 path. FP16 remains an explicit choice.

Across all five MF lengths and the RF fixture, final numerical ranges are:

| Device / precision | Waveform SNR range dB | Maximum latent absolute error |
| --- | ---: | ---: |
| RTX FP32 | 85.780–107.692 | 0.000195 |
| Intel FP32 | 92.020–105.187 | 0.000151 |
| RTX FP16 | 31.188–40.698 | 0.119275 |
| Intel FP16 | 35.566–43.960 | 0.118874 |

All 12 FP32 cases meet the unchanged `max_abs < 1e-3` latent and `SNR >= 85 dB`
waveform gates. All 24 cases have bit-identical streamed/full and trimmed/full
prefixes. FP16 does not meet the FP32 error contract; the separate results above
must not be described as FP32-equivalent accuracy. No listening, MOS or speech
recognition evaluation is claimed by waveform SNR.

### Matched reference backends

Two-second MeanFlow comparisons use the same model, tokens, masks and source
noise. Speech.cpp measurements were repeated after verifying the actual selected
GPU; their GGUF model names specify weight storage precision.

| Device / implementation | Precision | RF ms | Audio ms | Waveform SNR dB |
| --- | --- | ---: | ---: | ---: |
| RTX / PyTorch CUDA | FP32 | 43.954 | 97.630 | reference |
| RTX / speech.cpp Vulkan | F32 | 69.128 | 136.223 | 94.285 |
| RTX / speech.cpp Vulkan | F16 | 23.893 | 68.695 | 46.746 |
| Intel / PyTorch XPU math SDPA | FP32 | 352.671 | 902.328 | 97.998 |
| Intel / PyTorch XPU math SDPA | FP16 | 702.982 | 1321.761 | 38.012 |
| Intel / speech.cpp Vulkan | F32 | 518.826 | 1346.929 | 89.145 |
| Intel / speech.cpp Vulkan | F16 | 440.515 | 1320.860 | 42.660 |

The tuned Rust audio boundary is faster than these matched two-second references.
The RF stage alone still trails CUDA FP32 and speech.cpp F16 on RTX. Speech.cpp
F16 retains FP32 activations in parts of its graph, so its higher RTX SNR and
the Rust FP16 speed cannot be presented as an equal-accuracy comparison.
Different prompts, checkpoints and driver versions can select different routes;
these observations do not prove a universal ordering between implementations.

### Dense candidate screen

The dense screen completed 25 paired comparisons per adapter, including
previously untried row sizes, reduction tiles, prefetch, split-K, swizzling and
CubeK alternatives. Selected FP32 GPU-timestamp medians (milliseconds) at
B3/S100 illustrate why one global tile is insufficient:

| Operation / pair | Intel control → candidate | RTX control → candidate |
| --- | ---: | ---: |
| MLP contract, Burn graph → fused WGSL | 13.452 → 4.154 | 0.609 → 0.458 |
| MLP contract, T64/C128 → C64 | 4.594 → 5.070 | 0.435 → 0.337 |
| MLP contract, WGSL → CubeK DoubleUnit | 4.170 → 23.035 | 0.458 → 0.957 |
| SwiGLU, K16 → K16 prefetch | 6.904 → 6.703 | 0.541 → 0.497 |
| QKV, K16 → K16 prefetch | 4.879 → 4.866 | 0.389 → 0.356 |
| Direct output/residual, K32 → K16 | 1.751 → 1.518 | 0.159 → 0.167 |

C64 helps NVIDIA but hurts Intel in this case; the direct output K16 result
reverses that pattern. Differences as small as the Intel QKV pair are not a
whole-model speedup claim. The new tuner measures each actual shape and checks
accuracy; existing fused output routes are retained. Full paired receipts,
including slower candidates, are in `dense-intel/` and `dense-rtx/`.

On Intel, the two-second PyTorch XPU math-SDPA reference takes 352.671 ms for RF
and 902.328 ms through audio in FP32. FP16 takes 702.982 / 1321.761 ms on this
backend. Relative to the identical CUDA FP32 oracle, waveform SNR is 97.998 dB
for XPU FP32 and 38.012 dB for XPU FP16. The input tokens, masks and initial noise
are identical; the earlier CUDA fixture and this campaign's two-second fixture
have bit-identical reference outputs.

These figures show that adapter-specific tuning matters, and that half storage
does not guarantee faster inference on every backend. They do not establish
that WGSL has reached the hardware limit.

## Persistent-cache verification

Both adapters and both precisions were restarted in a separate process after
tuning the two-second MF case. All four restorations emitted **zero fresh tuning
records**. Both `latent.f32le` and `audio.f32le` matched their original process
byte for byte. The migration test kept the older development entries in place;
it did not clear the database to obtain a passing result.

The four databases live under
`~/.cache/Irodori-TTS-burn/cubecl/adapter-<hash>/`. Intel's adapter hash begins
`44b2b7b9`; NVIDIA's begins `b69406b0`. Each directory has separate
`irodori-v4-burn-0.22.0-pre.2-cubecl-0.11.0-pre.2-wgpu-auto-fp32-kernel-v6.db`
and `irodori-v4-burn-0.22.0-pre.2-cubecl-0.11.0-pre.2-wgpu-auto-fp16-kernel-v7.db`
files. At the final audit they contained 456–614 entries each and occupied
5.8–8.9 MB, including generated SPIR-V, Burn tuning and earlier development rows.

Saved decisions differ by shape and adapter as intended. For example, the
two-second FP32 attention chooses native WGSL Q32/KV16 on RTX and Q32/KV8 on
Intel. At eight seconds, RTX chooses subgroup-32 softmax with UnitMin PV while
Intel chooses cached-128 softmax. RTX FP16 at four/eight seconds selects different
cooperative QK/PV buffering combinations. These are measured selections, not a
new vendor-specific list of exception routes.

The final receipts are:

- `final-results.json`: the 24 accepted cases, timings, numerical errors and
  streaming comparisons. FP32 comes from `versioned-*`; FP16 from `finalized-*`.
- `final-persistent-restart-proof.json`: the four process-restart pairs, zero
  fresh decisions, byte counts and SHA-256 hashes.
- `final-cache-inventory.json`: full database paths, namespace counts, current
  logical keys, candidate checksums and chosen kernel names.
- `development-cache-collisions.json`: retained evidence for the migration fix.
- `audit_final_results.py`: reproducible assertions for the matrix and cache.

Earlier `accepted-*` and `cache-proof-*` FP32 outputs belong to the development
schema and are superseded; they must not be substituted for the final restart
proof. Timing variation remains visible: the RTX FP32 two-second original and
restored medians were 78.827 and 82.199 ms with identical output and no retuning.

## Reproduction

Build the validation and production entrypoints:

```sh
cargo build --release --features synthesis,cli,profile \
  --bin validate_synthesis --bin irodori
```

For an existing official reference fixture, run a new output directory with
`--device integrated:0` or `discrete:0`, and repeat in another process with the
same cache root. The model/codec/fixture variables below name local files:

```sh
target/release/validate_synthesis \
  --checkpoint "$MF_CHECKPOINT" --codec "$DECODER" --fixture "$MF_FIXTURE" \
  --device integrated:0 --precision fp32 --autotune --warmups 2 --repeat 7 \
  --output /tmp/irodori-first --autotune-record /tmp/irodori-first.json.log

target/release/validate_synthesis \
  --checkpoint "$MF_CHECKPOINT" --codec "$DECODER" --fixture "$MF_FIXTURE" \
  --device integrated:0 --precision fp32 --autotune --warmups 2 --repeat 7 \
  --output /tmp/irodori-restored --autotune-record /tmp/irodori-restored.json.log
```

For production synthesis, choose an optional cache root with
`irodori --cache-dir PATH`; validation uses `IRODORI_TTS_BURN_CACHE_DIR=PATH`.
The root contains adapter namespaces, so one root can serve both GPUs.
No Python or speech.cpp installation is required for production inference.

In this machine's artifact directory, `uv run run_cache_revision.py` reproduces
the final FP32 sweep and `uv run run_final_precision.py` the final FP16 sweep.
They retain completed receipts; choose new output names for fresh timings.
`uv run audit_final_results.py` rechecks the saved results without GPU work.
The isolated XPU environment and the CPU-noise oracle are described above;
`scripts/export_meanflow_oracle.py --help` lists its backend/length controls.

## Regression validation

Cargo metadata and reverse dependency trees for `cubek-matmul` and `cubecl-wgpu`
were inspected. Their workspace consumer is the root crate, through CubeK,
Burn's tensor/backend layers, attention and convolution. The root library's
627 non-ignored tests are covered by 336 debug passes, 289 remaining release
passes and two repaired-test passes. This is the union recorded in
`test-coverage-union.json`, not an uninterrupted full-suite pass. The switch to
release avoids repeatedly spending about 90 seconds generating kernels for tiny
debug attention tests; previously passed tests were not rerun.

Eight shape-only tests reached their assertions successfully but exited while
GPU work was still outstanding, causing native-driver teardown crashes. They now
wait for device completion. Each repaired case passed. The new sized-softmax
test also initially initialized one device twice; its variants now share one
initialization and pass together.

Additional explicit GPU tests pass on both adapters: direct cooperative Dot
GEMM with FP32/FP16 column-major weights and partial output tiles; 64/128/256-lane
softmax at boundary lengths against an f64 oracle; and tuned attention with
strided inputs, FP16 and fully masked batches. The fixed-32 subgroup variant is
additionally checked on NVIDIA. The nine adapter-specific executions cover five
distinct ignored test cases. No numerical tolerance was weakened to fix a test.

After extending the search, 61 focused route/decoder tests passed. Attention was
also rechecked at 13 and 65 queries in FP32 and FP16 with strided inputs and a
fully masked batch on both GPUs, covering both sides of the mixed-precision
admission boundary. Six short-decoder cases (1, 12 and 13 frames on each GPU)
passed the official CUDA FP32 oracle checks. A resident CLI smoke run also
confirmed an explicit cache root, saved-cache loading and WAV output.

`cargo check --lib --no-default-features` and separate `inference`, `synthesis`,
`worker`, `http-server` feature checks pass. Clippy covers all targets with
`synthesis,cli,profile,worker,http-server`; Rust formatting and the Python
exporter's Ruff checks pass. No new Rust dependency or unsafe block is introduced.
