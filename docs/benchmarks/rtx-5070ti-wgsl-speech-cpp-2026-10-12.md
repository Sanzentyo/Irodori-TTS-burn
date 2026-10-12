# RTX 5070 Ti WGSL optimization and speech.cpp comparison — 2026-10-12

The accepted change retains strict FP32 and the existing matmul selections,
but caches softmax logits in registers and performs the same reduction tree
with subgroup shuffles. Three alternating fresh-process comparisons reduced
40-step RF time by **2.56–2.93%**, with **bitwise-identical final audio**.
Only the NVIDIA B1/B3, S489 default cells change. Other shapes remain explicit
tuning candidates until a complete-request measurement supports admission.

## Environment and artifacts

- Base: `bb547dc`, plus this working-tree change. The pre-existing edit to
  `rtx-5070ti-v4-rf-gap-followup-2026-08-30.md` was preserved.
- NVIDIA GeForce RTX 5070 Ti **Laptop** GPU, 12,227 MiB, 120 W limit,
  PCI `0000:01:00.0`, Vulkan, driver **595.99.02**.
- Intel Core Ultra 9 275HX; Ubuntu 26.04 LTS; kernel `7.0.0-38-generic`.
- Rust/Cargo 1.95.0, Burn `0.22.0-pre.2`, CubeCL `0.11.0-pre.2`, wgpu/Naga 30.
- Driver restoration and reboot preceded these measurements. Do not pool
  their absolute timings with the earlier driver or Intel measurements.
- Artifacts: `/home/sanzentyo/benchmark-artifacts/irodori-rtx-wgsl-20261012/`.
  `summary.json` contains raw-result-derived medians; `paired-*.json` contains
  every full-request sample. Binaries, profiles, logs, waveforms, diagnostic
  tensors and the imported CubeCL bundle are retained alongside them.

The model is `Aratako/Irodori-TTS-v4-Small`, revision
`e4aaac4df355ff560dcd35e0dae272c3a759317b`, checkpoint SHA-256
`5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`.
The codec is `Aratako/Semantic-DACVAE-Japanese-32dim`, revision
`47376ee24834d7a05a48ebabfe3cde29b3c5e214`. The Rust decoder-only conversion
has SHA-256 `1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231`.

## Accepted kernel

The portable cached variant reads each score and mask once and writes only
the normalized probability. It retains the original 256-lane workgroup tree.
It remains available for tuning, including on Intel, where the earlier
[screen](intel-arl-softmax-screen-2026-10-12.md) measured a 27.43% operator gain.

The accepted RTX variant assigns one 32-lane subgroup to a row and four rows
to a 128-thread workgroup. Each physical lane owns eight virtual lanes of the
old 256-lane tree. The 128/64/32-stride additions stay in registers; shuffles
perform the remaining 16/8/4/2/1 strides. This preserves addition order instead
of replacing it with an implementation-dependent `subgroupAdd` reduction.
There are no workgroup barriers or shared scratch arrays in this variant.

The entrypoint requires native execution, fixed 32-lane subgroups and reported
plane-operation support. Unsupported adapters and contexts over 2,048 keys
are rejected before launch. Masks are applied before reading logits, including
masked NaNs; fully masked rows produce zero. The existing attention fallback
handles rejected layouts or capabilities. Browser execution is not admitted.

Naga 30 accepts native subgroup builtins without `enable subgroups;`, matching
CubeCL's native WGSL emitter. The directive itself is unsupported by that
frontend. The real-device runs below establish support for this kernel; they
do not establish support for every older subgroup shader in the repository.

`SdpaRoute` exposes cached/subgroup variants with either the backend-selected
PV product or explicit strict-F32 SimpleUnit/MinTile PV. The accepted cell uses
the latter, as the incumbent did. The route ABI is now `v4-dit-route-28`, so old
sealed route decisions must be regenerated rather than silently reused.

## Operator measurements

`bench_v4_softmax` restores pristine in-place input before each timed call,
alternates ABBA/BAAB, synchronizes completion, and reports device timestamps
separately from wall time. Autotuning warmups occur outside profiling because
CubeCL timestamp scopes cannot nest. Readback and accuracy checks occur after
timing. Each row below uses 10 warmups and 100 measured invocations per route.
All outputs were finite and bitwise identical.

Heads=20, D=64, context=22; GPU timestamp medians:

| Scope | Batch | Query length | Incumbent | Subgroup | Reduction |
|---|---:|---:|---:|---:|---:|
| softmax | 1 | 112 | 0.029952 ms | 0.007936 ms | 73.50% |
| softmax | 1 | 489 | 0.138496 ms | 0.054528 ms | 60.63% |
| softmax | 1 | 685 | 0.265472 ms | 0.173312 ms | 34.72% |
| softmax | 3 | 112 | 0.074496 ms | 0.013312 ms | 82.13% |
| softmax | 3 | 489 | 0.502528 ms | 0.322304 ms | 35.86% |
| softmax | 3 | 685 | 0.918272 ms | 0.648192 ms | 29.41% |
| QK + softmax + PV | 1 | 489 | 0.414336 ms | 0.350208 ms | 15.48% |
| QK + softmax + PV | 3 | 489 | 1.343744 ms | 1.148672 ms | 14.52% |

Earlier B3/S489 screens measured 0.474 → 0.274 ms. These are separate samples;
the table uses the final paired sweep, and adoption uses full-request results.

```sh
cargo build --locked --release --features cli --bin bench_v4_softmax
target/release/bench_v4_softmax --candidate subgroup --batch 3 --sequence 489 \
  --warmups 10 --blocks 50 --output-json /tmp/softmax-new-run.json
target/release/bench_v4_softmax --candidate subgroup --scope sdpa --unit-min-pv \
  --batch 3 --sequence 489 --warmups 10 --blocks 50 \
  --output-json /tmp/sdpa-new-run.json
```

## Complete resident request

40 Euler steps, independent CFG (text=3, caption=4, active at t≥0.5), 489 latent
frames / **19.56 seconds** of 48 kHz audio. The fixture has 11 text and 11
caption tokens. Each request has 20 B3 and 20 B1 forwards, 12 layers each.
No speaker, truncation, temporal rescaling, TF32 or F16 is used.

The same release binary, same input noise, same driver-local primed CubeCL
bundle and same loader/allocator settings are used for both routes:
exclusive pages, parallel loading, indexed checkpoint loading, exact RF
weight manifest and decoder-only codec residency. Each process has two warmups
and five measured requests. Order is control/candidate, candidate/control,
control/candidate. No other GPU benchmark runs concurrently.

Times exclude loading and fixture creation. RF ends after device completion;
consumer time ends when the complete CPU waveform exists. Per-stage diagnostic
readbacks are absent from these timing runs.

| Fresh-process pair | Incumbent RF | Subgroup RF | RF reduction | Incumbent consumer | Subgroup consumer |
|---|---:|---:|---:|---:|---:|
| 1 | 2.511952 s | 2.447618 s | 2.56% | 2.807914 s | 2.743413 s |
| 2 | 2.527111 s | 2.452961 s | 2.93% | 2.822027 s | 2.749132 s |
| 3 | 2.534423 s | 2.460619 s | 2.91% | 2.830455 s | 2.762615 s |
| Pooled median, 15 requests each | 2.525855 s | 2.455046 s | 2.80% | 2.822027 s | 2.749132 s |

Consumer reduction is 2.30–2.58% per pair. Every measured waveform and every
warmup has SHA-256
`4bf6a50d2805dcd5ea7343229899e21f27c6e32c558b1f9e9858e1719b5278a2`.
The retained `run-resident.sh`, control/subgroup profiles and binaries reproduce
the comparison; use fresh output labels because artifacts are never overwritten.

Persistent allocations are unchanged: 3,556,596,160 bytes in use with the RF
and codec resident, and 3,556,110,976 bytes after the consumer. Both variants
reserve 6,516,629,760 bytes after the consumer. These are allocator snapshots,
not a sampled driver-level peak-memory claim.

## speech.cpp: matched model and numerical comparison

The clone lives **outside this repository**, at the artifact root's
`speech.cpp/`. It is not a submodule of Irodori-TTS-burn. Its runtime is unmodified:

- [speech.cpp d23c49f3233375e3ad4944c880807790b7f490be](https://github.com/nyosegawa/speech.cpp/tree/d23c49f3233375e3ad4944c880807790b7f490be)
- Its own ggml dependency: `1d73413d60fd74538099958600ca111856473e59`.
- CMake Release, `GGML_VULKAN=ON`; GCC 15; selected backend `Vulkan0`, verified
  as the same RTX 5070 Ti Laptop GPU in `speech-devices.txt`.

Current speech.cpp examples target v4.1/MF. Comparing their four-step MF or
F16-weight timings with this 40-step v4 FP32 workload would change the work.
Instead, `convert-v4.sh` uses the pinned upstream converter with only its input
model revision and version label changed. Tensor transforms are unchanged.
The resulting `Irodori-TTS-859M-v4-F32.gguf` contains the exact v4 weights in F32
and the same codec. The original model includes conditioning encoders, hence
the full-file parameter count in the name.

`scripts/bench_speech_cpp.cpp` links to the unmodified `irodoritts` library.
The matched timing mode (`tokens`) encodes the same compacted text/caption token
IDs on every request, uses the same initial noise and native 40-step sampler,
and decodes the whole 489-frame latent. Conditioning, DiT and decoder weights
are resident. Both runtimes' RF interval **includes condition encoding**.
Timing includes speech.cpp's native per-step host readbacks, graph construction
and final audio readback; it excludes tokenization, loading and diagnostics.
Thus this is a resident token-to-audio comparison, not a cold text-to-speech CLI
benchmark. A separate `encoded` mode feeds Burn's exported condition tensors
directly, for isolating DiT and decoder agreement; its timings are not used in
the matched table.

Two speech.cpp precision policies were measured, each with two warmups and
five timed requests in one fresh process. The WGSL row uses the 15-request
cohort above. A separate official PyTorch strict-FP32 diagnostic run uses
identical initial noise and disabled CUDA/cuDNN TF32. SNR below compares the
untrimmed waveform to that official reference; it is a numerical measure,
not a listening test or CER/WER score.

| Runtime / arithmetic policy | RF + conditioning median | CPU audio ready | Waveform SNR vs official FP32 | Waveform RMSE |
|---|---:|---:|---:|---:|
| Burn/WGSL, optimized strict FP32 | **2.455046 s** | **2.749132 s** | **72.19 dB** | **3.9460e-5** |
| speech.cpp, F32 storage and strict F32 execution | 3.532292 s | 4.193762 s | 62.74 dB | 1.1704e-4 |
| speech.cpp, F32 storage, default Vulkan execution | 1.515879 s | 1.959506 s | 38.66 dB | 1.8734e-3 |

These speech.cpp results are in `speech-{strict,default}-tokens-vs-oracle.json`.
Its condition-encoding medians are 22.40 ms (strict) and 13.29 ms (default).
The earlier precomputed-condition runs are retained separately: 3.549644 /
1.514502 s RF and 63.66 / 27.96 dB waveform SNR. Encoding roundoff can interact
with later sampler roundoff, so the lower-precision SNR is not a monotonic
quality ranking across those two input boundaries.

The strict speech.cpp run sets `GGML_VK_DISABLE_F16=1`,
`GGML_VK_DISABLE_COOPMAT=1`, and `GGML_VK_DISABLE_COOPMAT2=1`.
The default run sets none of these. ggml's default cooperative-matrix path
converts F32 operands to F16 for throughput; requesting F32 **accumulation**
does not prevent that input conversion. This is visible in the pinned
[Vulkan implementation](https://github.com/nyosegawa/ggml/blob/1d73413d60fd74538099958600ca111856473e59/src/ggml-vulkan/ggml-vulkan.cpp).
Consequently, F32 GGUF storage alone does not establish a strict-FP32 comparison.

Against speech.cpp's strict-FP32 policy, WGSL takes 30.50% less RF time and
34.45% less time to complete CPU audio. The default speech.cpp policy is faster,
with the larger measured numerical differences shown above. No default
precision reduction was introduced into Burn.

Additional isolation using the precomputed-condition runs, speech.cpp versus Burn:

- Each of the 40 forwards was rerun on **Burn's exact input and timestep**.
  Worst forward SNR is 106.35 dB; largest absolute error is 1.108e-4.
- Decoding Burn's exact final latent gives waveform SNR 117.95 dB and maximum
  absolute error 4.396e-6, separating decoder error from sampler drift.
- Native schedules differ by at most 5.96e-8. That difference remains in the
  native full-sampler comparison and is not silently replaced by Burn's schedule.
- Native final-latent SNR versus Burn is 80.43 dB for strict execution and
  38.92 dB for default execution.
- In the matched token run, strict speech.cpp's text and caption conditions
  agree with Burn at 122.30 and 124.78 dB SNR, respectively.
- Burn's initial noise is bitwise identical to the official PyTorch diagnostic;
  final-latent SNR versus official is 86.74 dB. The optimization leaves that
  reference agreement unchanged because final audio is identical to the control.

The reusable preparation/comparison tools are:

```sh
uv run scripts/prepare_speech_cpp_comparison.py CONTROL_DIAGNOSTIC.json INPUT_DIR \
  --fixture ORIGINAL_DESIGN_FIXTURE.safetensors
# Build the C++ harness against the pinned clone; see the commands below.
GGML_VK_VISIBLE_DEVICES=0 GGML_VK_DISABLE_F16=1 \
  GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1 \
  ./bench_speech_cpp MODEL.gguf INPUT_DIR FRESH_OUTPUT_DIR 7 2 Vulkan0 tokens
uv run scripts/compare_speech_cpp.py INPUT_DIR FRESH_OUTPUT_DIR RESULT.json \
  --python-oracle OFFICIAL_DIAGNOSTIC.safetensors
```

The preparation step requires a designed-voice residency diagnostic with
`--cfg-caption 4 --diagnostic-forward-ordinal 0`, exported audio, and the same
40-step configuration used here. It validates source tensor hashes, preserves
the float32 payload and writes a hashed input manifest. Token export additionally
checks the original fixture's SHA-256 and retains its masked-in token IDs.
It rejects other voice modes rather than inventing missing conditions.

Example build, with `speech_root` pointing to the external clone:

```sh
cmake -S "$speech_root" -B "$speech_root/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build "$speech_root/build" --target speech irodori-dit-check --parallel 10
c++ -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Werror \
  -I "$speech_root/src/common" -I "$speech_root/src/families" \
  -I "$speech_root/checks" -I "$speech_root/ggml/include" \
  scripts/bench_speech_cpp.cpp -o bench_speech_cpp \
  "$speech_root/build/libirodoritts.a" "$speech_root/build/libspeechcommon.a" \
  "$speech_root/build/ggml/src/libggml.a" -ldl \
  "$speech_root/build/ggml/src/libggml-cpu.a" \
  "$speech_root/build/ggml/src/ggml-vulkan/libggml-vulkan.a" \
  "$speech_root/build/ggml/src/libggml-base.a" -fopenmp -lpthread -lm -lvulkan
```

## Other evaluated ideas and limits

The pinned ggml `soft_max.comp` also keeps logits in a bounded private cache.
This supports the memory-traffic approach used here. Inspection of speech.cpp's
DiT and sampler also shows repeated condition projections and a host readback
after every velocity evaluation. Burn already caches conditions and keeps its
Euler loop on the GPU; those advantages are retained.

The existing Q8/KV32 subgroup FlashAttention candidate was temporarily enabled
with the native Naga-compatible syntax and measured on B3/S489. It produced
finite output with maximum absolute error 9.537e-7 on the operator input, but
took **3.945216 ms**, versus **1.341696 ms** for the incumbent two-matmul route.
It was not adopted; the trial source patch and binary remain in the artifacts.

Packing validity flags into integer bitmasks in the new subgroup kernel gave
only a small separate-process operator difference (about 0.274 → 0.269 ms),
without complete-request evidence. The simpler boolean-array implementation
used by all accepted full-request pairs is retained.

This result does not establish optimality for every GPU, voice, length or
precision policy. Complete-request admission is limited to the measured
B1/B3 S489 cells. Multiple operator lengths validate geometry and parity but
do not justify broadening production defaults. No TF32/F16 acceleration,
quantization, different model, or fewer diffusion steps is counted as an
implementation speedup.

## Validation

`cargo nextest run --locked --features cli,profile --lib --run-ignored all
-E 'test(kernels::matmul_sdpa::) | test(route_autotune::) | test(model::attention::)'
--test-threads 1 --no-tests fail` passed all **59 tests**. The selection was
checked with `cargo nextest list` first. `cargo clippy --locked --features
cli,profile --all-targets -- -D warnings`, `cargo fmt --all --check`, and
`git diff --check` passed. The release CLI binaries also built without the
`profile` feature. Both Python tools passed Ruff; the C++ harness compiled with
`-Wall -Wextra -Werror`, and preparation/comparison were executed on real data.

The final release binary built **without** `profile` was run with no route
override (`final-builtin.json`). It selected the shipped policy and reproduced
the same waveform hash on all seven requests. The five measured requests had
RF median 2.422328 s and consumer median 2.716894 s. This separate confirmation
is not pooled into the same-binary A/B timing cohort.

The GPU tests cover key counts 1, 7, 255, 256, 257, 511,
512, 513, 1025 and 2048, a partially occupied four-row workgroup, fully masked
rows, masked NaNs, the independent F64 softmax oracle, bitwise streaming parity,
invalid scales and rejection of 2049 keys. Cargo metadata and inverse dependency
inspection found no workspace consumers outside this package; CLI targets and
the attention/route consumers are included in validation.

A post-reboot Intel capability-rejection probe could not run because the
current Vulkan process enumerated no integrated adapter. Its failure log is
retained; it is not counted as a passed rejection test. The Intel cached-kernel
screen predates driver restoration and remains separate from the RTX results.
