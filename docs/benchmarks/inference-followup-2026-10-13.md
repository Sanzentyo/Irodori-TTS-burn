# Inference, loading and cache follow-up — 2026-10-13

Base: `5a4adcb1d5ef3112bbfa459a3a806452cf510de5`. Hardware and dependency pins
match the [Burn 0.22 comparison](burn-0.22-upgrade-2026-10-13.md): RTX 5070 Ti
Laptop (595.99.02), Intel ARL (Mesa 26.0.8), Burn 0.22 / CubeCL 0.11 / CubeK 0.3 /
WGPU 30. Raw commands, binaries, float outputs and rejected experiments are in
`/home/sanzentyo/benchmark-artifacts/irodori-followup-20261013/`.
The [machine-readable receipts](data/inference-followup-2026-10-13.json) retain
input/output/executable hashes, every measured timing sample, cache approvals,
rejected experiments, transcripts and the exact commands. This campaign began
on October 13 and continued into October 14 JST.

GPU work runs serially. Compilation and tests are paused during timings. The
validation harness uses Adaptive allocation, fixed source noise and an FP32
decoder. Production resident synthesis still uses its existing allocator policy.
Timing ends after CPU audio readback; model load, tokenization and file writing
are outside the synthesis boundary.

## Fixed-request graph replay

The shared graph owner now supports the sampler and sampler-plus-codec, in
addition to codec-only replay. Fixed conditions, RoPE and schedule constants
remain owned for the lifetime of the captured plan. The public output guard
permits owned CPU readback only. Input refresh and replay require mutable access.
The owner retains its capture stream and scopes input copies and output readback
to it, including after a caller switches logical streams or moves the session
to another thread.
The existing replay unsafe boundary is reused; no additional unsafe is needed.

Two different source noises produce bit-identical ordinary/replayed latent and
audio on all eight RF/MF × RTX/Intel × FP32/FP16 cells. Full versus streamed audio
and trimmed prefixes also match exactly. RF FP16 uses the explicit FP32 sampler
state in this campaign; MF retains model precision.

Warm median milliseconds, two warmups and three measured requests per path:

| Model / device / precision | Ordinary | Captured audio | Change |
|---|---:|---:|---:|
| MF, RTX, FP32 | 74.902 | 62.840 | −16.1% |
| MF, RTX, FP16 | 54.306 | 38.616 | −28.9% |
| MF, Intel, FP32 | 681.184 | 597.915 | −12.2% |
| MF, Intel, FP16 | 670.830 | 611.553 | −8.8% |
| RF, RTX, FP32 | 709.807 | 684.798 | −3.5% |
| RF, RTX, FP16 + FP32 state | 332.637 | 310.040 | −6.8% |
| RF, Intel, FP32 | 7834.483 | 7681.661 | −2.0% |
| RF, Intel, FP16 + FP32 state | 7792.364 | 7612.457 | −2.3% |

MF is a two-second, four-step fixture; RF is a four-second, forty-step fixture.
These small samples describe the measured fixed requests, not tail latency.
Graph capture has an initial cost and fixes the text, voice and geometry; it is
not automatically applied to arbitrary worker/HTTP requests. Capture cost and
sampler-only replay times are retained in each `codec-tuned-*/report.json`.
The ordinary validator also reads the latent to the CPU to record its sampling
time; the combined graph removes that synchronization. Its gain includes both
fixed condition preparation and removed dispatch/readback work. Ordinary
resident requests which avoid latent readback have a different boundary.

## Intel decoder search

The selected decoder route compares cached column-transform convolution with
six WGSL polyphase tiles. Candidate execution must pass a finite numerical
comparison before timing. Unsupported shapes and device limits are rejected
before launch. The tuner stores its selected tile in the existing CubeCL owner,
keyed by schema, hardware, dtype, shape and strides. Explicit approved profiles
and portable routes keep their existing selection policy.

The corrected standalone 50-frame Intel experiment measures 408.853 ms for the
portable tile and 381.808 ms for T64/O16/K32. The other four tested tiles measure
416.321–570.740 ms. All six produce identical waveform values. An earlier
prototype did not pack blocks 1–3 and therefore ran the old route; its timing is
retained as a failed experiment, not evidence for the polyphase implementation.
The complete MF request decreases from about 702 to 681 ms before graph replay.

Other tested codec changes were not adopted: implicit/fan-out/multi-consumer k7
convolutions took about 505 ms, pointwise implicit convolution about 409 ms and
the accumulator variant about 402 ms against an approximately 400 ms control.
The removed experimental environment switches are not production fallbacks.

## Explicit FP32 integration with FP16 forwards

Keeping the initial noise, guidance and ODE state in FP32 raises the Intel RF
waveform SNR from 33.610 to 42.591 dB against the same official FP32 oracle, with
roughly one percent latency difference in the isolated state experiment.
The older pre-upgrade result was 43.960 dB, so recovery is incomplete. This
experiment changes both initial-noise rounding and repeated integration; it
does not isolate the arithmetic responsible for the dependency regression.
Untimed old/new-stack stage captures use bit-identical input noise, but already
differ in the encoded text (maximum 0.00390625, RMSE 0.0008405). The first model
forward differs by RMSE 0.0006247 and the fortieth by 0.005280. FP32 integration
therefore mitigates a trajectory difference; it does not establish that the
FP16 neural forwards now reproduce the old stack.

RTX RF improves from 31.980 to 32.656 dB. MeanFlow is mixed: Intel changes from
39.136 to 39.398 dB, while RTX changes from 31.188 to 30.273 dB and becomes slower.
Consequently, `SamplerStatePrecision::Model` remains the default and FP32 state
is an explicit request control. An FP32 sigmoid opmath experiment produced
identical final audio in all four FP16 cells and was reverted.

These are numerical agreement measurements. They do not establish perceptual
quality or equivalence of FP16 and FP32. The original eight-cell FP32 gates are
unchanged: latent maximum absolute error below 1e-3 and audio SNR at least 85 dB.

## Bounded grouped weight uploads

WGSLPack files, borrowed archives and shared-owned archives retain the same
format and numeric values. The file loader batches uploads through CubeCL's
existing allocation API. A 1 MiB budget is the default; the loader holds one batch
plus its next tensor, and a larger tensor uploads alone. No mmap or new unsafe
boundary is introduced. Runtime/driver upload buffers are additional; the budget
does not cap total process RSS.

Five alternating fresh processes per old/new cell, warm OS file cache:

| Device / precision | Old ready ms | Grouped ready ms | Old peak RSS KiB | Grouped peak RSS KiB |
|---|---:|---:|---:|---:|
| Intel FP16 | 667.946 | 543.496 | 603820 | 560628 |
| Intel FP32 | 1331.152 | 1086.451 | 1020784 | 866852 |
| RTX FP16 | 484.745 | 429.780 | 770144 | 764820 |
| RTX FP32 | 964.985 | 867.578 | 1150416 | 1101664 |

`ready_ms` ends after completed model uploads and excludes adapter setup,
parameter verification and codec loading. A separate 48-process sweep tested
1 byte, 1 MiB, 16 MiB and 64 MiB. The larger batches increased latency/memory and
were rejected. Full 717-parameter hashes match the source and prior packed
loader in both precisions. GPU tests additionally cover odd-length tensors,
multiple dtypes and tensors exceeding the budget.
Full-model inference from grouped packed weights produces bit-identical latent
and audio to the safetensors path on RTX/Intel in FP32/FP16. This checks handwritten
kernels' binding offsets as well as parameter readback. Captured graph replay
also matches for both tested noises after loading packed weights.

## Approval of real persistent selections

`seal-bundle` binds the existing complete-trajectory accuracy manifest to an
immutable export of actual CubeCL autotune entries. Recorder logs alone omit
warm-cache hits and cannot supply that guarantee. Software/device/model identity,
execution settings and payload digest are all checked before restoration.
Conflicting or extra local selections are errors; existing values are not
overwritten. A post-workload `verify_active` detects new or changed choices.
Unseen workloads still require a new accuracy gate and approval.

Four real MF restore trials imported 183/223 RTX and 195/219 Intel entries for
FP32/FP16 respectively into separate cache roots. All replayed the 50-frame
fixture with zero fresh decisions, identical latent/audio, and an exact
post-workload database match. The replayed evidence passed sealing again.
The approval is scoped to that fixture and execution identity; additional keys
in the snapshot do not certify other lengths or voices. FP32 uses the existing
strict multi-metric policy with an 85 dB target. The explicitly experimental
FP16 policy retains the previously measured two-second compatibility envelope
(latent max 0.08 / RMSE 0.008 / SNR 40 dB; audio max 0.07 / RMSE 0.0035 / SNR
30 dB, plus mean-error/cosine checks). It is not an FP32-equivalence gate and is
not installed as a default production quality policy.

The format reader crate remains dependency-free and optional. Cache approval
uses the already-transitive `cubecl-environment` API directly; no duplicate
tuning database, Python runtime dependency or transport dependency is added.

Native driver pipeline caching is not implemented in this change. CubeCL's
saved autotune choices remain in use. Correction from the October 14 feature
audit: generated-SPIR-V persistence was configured at runtime, but its required
`cubecl-server/persistence` Cargo feature was not enabled in this build, so
generated kernels were not actually being persisted. The native-cache follow-up
keeps that setting unchanged and enables only driver-cache persistence.
WGPU's native cache restore requires a new unsafe call; the repository explicitly
requires user sign-off for that boundary. Its proposed identity, storage and
corruption checks are recorded in `native-pipeline-cache-review.md` in the
artifact directory, pending permission. Grouped loading and graph replay do
not depend on that proposal.

## Extended corpus

The corpus covers a 10-second female instruction, a 6-second male instruction,
a 4-second reference voice and a 6-second RF female instruction. Each is compared
with the pinned speech.cpp/ggml implementation and official PyTorch CUDA/XPU in
both precisions. The XPU reference explicitly uses math SDPA; the ordinary XPU
SDPA backend is unsupported on this Intel device. The reference voice is generated
test audio. No private recording is used.

Ordinary resident requests include condition encoding, the complete sampler,
decoder and final CPU waveform. Separate streaming trials include the complete
sampler before the first callback, with 12 initial and 17 subsequent latent
frames for both Rust and speech.cpp. The old decoder-only `stream_first_audio`
field is not the complete-request latency. Empirical p95 uses 20 measured MF
requests and 10 RF requests, after two warmups; this is not a production SLA.

ASR uses the same pinned faster-whisper model for every backend, with Japanese,
beam size 5, temperature zero and no VAD. The model revision is
`mobiuslabsgmbh/faster-whisper-large-v3-turbo@0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`.
NFKC-normalized character error rate omits punctuation, whitespace, symbols and
control characters. Kanji/number transcription alternatives affect the score.
ASR is an intelligibility proxy and does not replace listening or MOS.

The FP16 labels describe each implementation's precision policy, not identical
arithmetic. Rust retains FP16 model activations and an FP32 decoder; RF here
also uses FP32 sampler state. In the pinned speech.cpp implementation,
[graph inputs](https://github.com/nyosegawa/speech.cpp/blob/d23c49f3233375e3ad4944c880807790b7f490be/src/common/graph.cpp#L41)
and [sampler integration](https://github.com/nyosegawa/speech.cpp/blob/d23c49f3233375e3ad4944c880807790b7f490be/src/families/irodori-tts/sampler.cpp#L236)
remain FP32 with FP16 weights, and GGML matrix multiplication produces FP32
output tensors. These differences matter when interpreting numerical agreement.

For each sentence, all twelve implementation/device/precision combinations
produce the same recognized transcript. The 10-second MF and 6-second RF cases
have zero normalized character errors. The male case has 3 edits in 29 characters
(10.34%): ASR returns `足は` for `明日は`, and `9時` for `九時`. The reference-voice
case has one edit in 20 characters (5%): `いい` for `良い`. These include spelling
alternatives and occur in the official FP32 output too. This corpus shows no
Rust-specific ASR regression, but neither proves error-free pronunciation nor
establishes equivalent timbre/prosody.

Both the 10-second female and 6-second male MF cases fail the short-fixture FP32
numerical gate on RTX and Intel. For the female case, Rust audio SNR is 74.265 dB
on RTX and 66.957 dB on Intel, with latent maxima 0.002110 and 0.004310. The male
case reaches 78.380/76.327 dB. All four outputs are bit-identical to the saved
original executable's latent/audio, so these discrepancies predate this change.
They remain failed numerical gates; successful recognition does not waive them.
The MF reference and RF cases pass the report's FP32 gate on both Rust devices.

FP16 has larger differences. A separate two-warmup/three-repeat Intel MF state
experiment measured:

| Case | Model-state SNR dB | FP32-state SNR dB | Model-state ms | FP32-state ms |
|---|---:|---:|---:|---:|
| Female 10 s | 21.43 | 21.49 | 3019.0 | 3041.4 |
| Male 6 s | 12.28 | 12.94 | 1933.3 | 1962.7 |
| Reference 4 s | 39.87 | 41.63 | 1415.7 | 1433.8 |

The model-state column reuses the main 20-repeat campaign. All three new graphs
match their ordinary FP32-state outputs for both noises, and their ASR transcripts
remain unchanged. The small recovery does not resolve the low male-voice SNR;
FP32 model execution remains the accuracy-oriented Intel recommendation.

Rust has a lower complete-audio median than speech.cpp in all 16 matched cells.
The stricter criterion that **every** measured Rust sample is below the matching
PyTorch minimum passes 12/16 ordinary cells and 13/16 captured cells. Ordinary
Intel MF reference FP32 fails, while capture passes. RF FP32 on RTX, and RF in
both precisions on Intel, still fail even with capture. First-audio superiority
is also not universal: RTX MF FP16 is effectively tied for the female case and
slower for the male case (70.1 versus 65.6 ms).

To check whether existing route selection explains the FP32 misses, a separate
two-warmup/three-repeat control tried the portable route followed by autotuning:

| Case | Portable ms | Autotuned control ms |
|---|---:|---:|
| MF reference, Intel FP32 | 1793.5 | 1476.2 |
| RF female, RTX FP32 | 1309.6 | 1149.3 |
| RF female, Intel FP32 | 22449.8 | 11911.9 |

All six outputs pass the short-fixture numerical gate. Portable routing is slower
in every cell, so it is not promoted as a fallback. The measured RF gaps to
PyTorch remain open; graph replay and the tested tile choices do not establish
that this hardware's performance ceiling has been reached.

### Complete audio latency

Milliseconds, median / empirical p95. PyTorch uses CUDA on RTX and math SDPA on XPU.

| Case | Device | Precision | PyTorch | Rust ordinary | Rust captured | speech.cpp |
|---|---|---|---:|---:|---:|---:|
| MF female 10 s | RTX | FP32 | 401.9 / 425.9 | 245.8 / 265.9 | 212.9 / 222.6 | 489.8 / 509.4 |
| MF female 10 s | RTX | FP16 | 353.9 / 379.2 | 192.6 / 204.0 | 168.6 / 175.5 | 334.0 / 338.5 |
| MF female 10 s | INTEL | FP32 | 3811.2 / 3820.5 | 3080.8 / 3087.8 | 2836.8 / 2858.9 | 5895.8 / 5910.7 |
| MF female 10 s | INTEL | FP16 | 3986.6 / 3998.2 | 3019.0 / 3037.3 | 2835.1 / 2842.1 | 6061.3 / 6073.2 |
| MF male 6 s | RTX | FP32 | 259.9 / 291.8 | 183.8 / 202.3 | 157.8 / 162.1 | 336.3 / 343.0 |
| MF male 6 s | RTX | FP16 | 227.5 / 248.0 | 137.1 / 143.9 | 113.5 / 114.9 | 192.1 / 199.5 |
| MF male 6 s | INTEL | FP32 | 2231.6 / 2237.3 | 1952.2 / 1957.7 | 1764.2 / 1769.9 | 3718.3 / 3729.4 |
| MF male 6 s | INTEL | FP16 | 2480.0 / 2483.1 | 1933.3 / 1939.0 | 1761.9 / 1767.2 | 3868.2 / 3886.9 |
| MF reference 4 s | RTX | FP32 | 177.8 / 201.3 | 136.3 / 151.0 | 104.2 / 108.3 | 300.1 / 300.9 |
| MF reference 4 s | RTX | FP16 | 157.7 / 171.8 | 103.5 / 106.5 | 77.4 / 83.1 | 145.9 / 147.4 |
| MF reference 4 s | INTEL | FP32 | 1364.4 / 1369.9 | 1472.6 / 1485.0 | 1247.1 / 1252.2 | 2646.1 / 2651.0 |
| MF reference 4 s | INTEL | FP16 | 1640.0 / 1646.9 | 1415.7 / 1425.0 | 1240.8 / 1246.6 | 2708.1 / 2721.2 |
| RF female 6 s | RTX | FP32 | 1088.5 / 1090.6 | 1148.3 / 1152.7 | 1131.8 / 1133.4 | 1853.3 / 1857.2 |
| RF female 6 s | RTX | FP16 | 560.5 / 567.5 | 442.5 / 449.0 | 415.6 / 427.2 | 736.3 / 746.2 |
| RF female 6 s | INTEL | FP32 | 8560.6 / 8597.6 | 11917.7 / 11922.9 | 11738.5 / 11742.1 | 23643.2 / 23650.0 |
| RF female 6 s | INTEL | FP16 | 10987.4 / 10995.7 | 12369.7 / 12373.8 | 12131.8 / 12158.5 | 23336.7 / 23356.3 |

### Complete-request first audio

Milliseconds, median / empirical p95. Both implementations run the full sampler before the first decoder callback. PyTorch streaming was not measured.

| Case | Device | Precision | Rust | speech.cpp |
|---|---|---|---:|---:|
| MF female 10 s | RTX | FP32 | 118.1 / 126.2 | 175.6 / 177.7 |
| MF female 10 s | RTX | FP16 | 71.6 / 74.5 | 71.5 / 71.8 |
| MF female 10 s | INTEL | FP32 | 1284.1 / 1286.6 | 2051.0 / 2060.5 |
| MF female 10 s | INTEL | FP16 | 1229.1 / 1237.4 | 2032.1 / 2048.0 |
| MF male 6 s | RTX | FP32 | 114.5 / 121.2 | 159.9 / 162.8 |
| MF male 6 s | RTX | FP16 | 70.1 / 71.5 | 65.6 / 66.3 |
| MF male 6 s | INTEL | FP32 | 966.7 / 968.8 | 1566.5 / 1574.0 |
| MF male 6 s | INTEL | FP16 | 940.8 / 947.8 | 1602.8 / 1622.8 |
| MF reference 4 s | RTX | FP32 | 95.7 / 100.4 | 197.0 / 197.5 |
| MF reference 4 s | RTX | FP16 | 68.3 / 70.2 | 69.6 / 70.2 |
| MF reference 4 s | INTEL | FP32 | 863.4 / 866.8 | 1335.9 / 1339.9 |
| MF reference 4 s | INTEL | FP16 | 805.2 / 809.7 | 1329.8 / 1346.2 |
| RF female 6 s | RTX | FP32 | 1079.4 / 1082.3 | 1678.3 / 1681.2 |
| RF female 6 s | RTX | FP16 | 375.0 / 379.0 | 622.2 / 624.6 |
| RF female 6 s | INTEL | FP32 | 10926.1 / 10929.4 | 21492.2 / 21495.2 |
| RF female 6 s | INTEL | FP16 | 11239.1 / 11241.7 | 21078.7 / 21095.2 |

### Numerical agreement

Waveform SNR in dB against the official CUDA FP32 oracle; each cell is FP32 / FP16. Larger values indicate closer numerical agreement, not a perceptual score.

| Case | Device | PyTorch | Rust | speech.cpp |
|---|---|---:|---:|---:|
| MF female 10 s | RTX | exact / 28.44 | 74.27 / 25.66 | 74.36 / 21.04 |
| MF female 10 s | INTEL | 67.97 / 20.80 | 66.96 / 21.43 | 57.80 / 31.46 |
| MF male 6 s | RTX | exact / 10.31 | 78.38 / 25.38 | 87.71 / 18.52 |
| MF male 6 s | INTEL | 76.96 / 33.17 | 76.33 / 12.28 | 72.01 / 38.33 |
| MF reference 4 s | RTX | exact / 40.80 | 91.70 / 37.48 | 98.53 / 51.52 |
| MF reference 4 s | INTEL | 97.52 / 42.46 | 98.33 / 39.87 | 76.14 / 46.51 |
| RF female 6 s | RTX | exact / 36.64 | 94.69 / 36.15 | 91.93 / 45.53 |
| RF female 6 s | INTEL | 93.04 / 33.58 | 88.69 / 40.14 | 84.97 / 50.05 |

### Recognition of the generated speech

Normalized character error rate in percent, FP32 / FP16. Raw transcripts and audio digests are retained in the machine-readable evidence.

| Case | PyTorch RTX | PyTorch Intel | Rust RTX | Rust Intel | speech.cpp RTX | speech.cpp Intel |
|---|---:|---:|---:|---:|---:|---:|
| MF female 10 s | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| MF male 6 s | 10.34 / 10.34 | 10.34 / 10.34 | 10.34 / 10.34 | 10.34 / 10.34 | 10.34 / 10.34 | 10.34 / 10.34 |
| MF reference 4 s | 5.00 / 5.00 | 5.00 / 5.00 | 5.00 / 5.00 | 5.00 / 5.00 | 5.00 / 5.00 | 5.00 / 5.00 |
| RF female 6 s | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

## Verification

All 695 workspace library/integration tests, including the opt-in GPU and packed
model cases, have passing results in the recorded union. The broad run passed
692 and exposed the graph copy tail bug; the corrected graph/approval suite
then passed all 11 selected tests. A final seven-test approval run covers the
added binary fingerprint. The final six-test graph suite includes the new
cross-stream/cross-thread and preparation/unwind regressions. Clippy covers all workspace targets/features with
warnings denied. Core-only, inference-only, codec-only, packed-weights-only,
synthesis, worker and HTTP library builds pass independently. WGSLPack's two
doctests pass; the root's four existing ignored examples remain ignored.

The comparison harness builds with C++17, optimization and warnings-as-errors.
Python scripts run through uv; the ASR dependencies and model revision are
locked separately from the application. Validation here is Linux Vulkan on the
two named GPUs; it does not certify other GPUs or operating systems.

## Reproduction

Build the Rust tools with `cargo build --release --features
cli,profile,synthesis,packed-export --bin validate_synthesis --bin
approve_v4_autotune --bin irodori`. Generate each canonical fixture with
`scripts/export_meanflow_oracle.py` in the pinned official Irodori uv project.
The receipt's `corpus.cases` retains the exact text, caption, length and repeat
count. CUDA FP32 creates the reference voice latent once; the other PyTorch
backends use `--reference-fixture` to share it.

For a canonical MF fixture, the Rust measurement has this form:

```sh
target/release/validate_synthesis \
  --checkpoint MODEL.safetensors --codec DECODER.safetensors \
  --fixture ORACLE.safetensors --output NEW_OUTPUT_DIRECTORY \
  --device integrated:0 --precision fp32 --autotune \
  --warmups 2 --repeat 20 --stream-repeat 20 --graph-replay
```

RF uses ten measured repeats. RF FP16 additionally uses `--sampler-state fp32`;
MF FP16 retains model state. The C++ harness uses the earlier
[pinned build command](rtx-5070ti-wgsl-speech-cpp-2026-10-12.md),
the canonical fixture converted by `prepare_synthesis_comparison.py`, and
`oracle` mode. Strict FP32 sets `GGML_VK_DISABLE_F16`, `GGML_VK_DISABLE_COOPMAT`
and `GGML_VK_DISABLE_COOPMAT2` to `1`. Device visibility is selected
explicitly and the harness records the physical adapter.

The ASR manifest is a JSON array of `{label, text, audio, sample_rate}`. With
CUDA cuBLAS/cuDNN visible to CTranslate2:

```sh
uv run --locked scripts/evaluate_synthesis_intelligibility.py \
  asr-input.json NEW_ASR_OUTPUT.json \
  --revision 0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf --device cuda
```

The Rust suite uses `cargo nextest run --workspace --lib --test '*' --features
cli,profile,worker,http-server,lora,packed-export --run-ignored all
--test-threads 1 --no-tests fail`, with `IRODORI_TOKENIZER` and
`IRODORI_PACK_TEST_CHECKPOINT` pointing to the recorded opt-in fixtures. Builds
use `CARGO_BUILD_JOBS=2`; this is separate from the single GPU test worker.
