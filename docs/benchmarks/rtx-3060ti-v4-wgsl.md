# Irodori-TTS v4-Small WGSL validation on RTX 3060 Ti

## Current matched-boundary result (2026-08-12)

The current production FP32 path now passes the strict same-precision stage
campaign for both RF and codec. This result supersedes the older `122 ms` RF /
`61 ms` codec timing below; the older sections remain as optimization history.

The campaign used five fresh processes per runtime in the balanced order
`P,W,W,P,P,W,W,P,P,W`. Every process ran two excluded warmups followed by ten
measured repetitions. Both runtimes synchronized before each timer. The primary
timer stopped at device completion and excluded final tensor materialization;
the secondary timer started at the same point and included full f32 CPU readback
(1,600 RF values or 96,000 waveform values). No automatic retries were used.

| Boundary / stage | PyTorch median (range) | WGPU median (range) | Median speedup | Strict all-sample result |
|---|---:|---:|---:|---:|
| Device-complete RF | 136.813 ms (135.364-137.940) | 118.873 ms (118.049-120.065) | **1.151x** | 50/50 WGPU samples below the global PyTorch minimum |
| Device-complete codec | 46.479 ms (45.797-46.620) | 41.243 ms (40.471-41.804) | **1.127x** | 50/50 |
| Full-f32-readback RF | 136.858 ms (135.409-137.987) | 118.993 ms (118.171-120.197) | **1.150x** | 50/50 |
| Full-f32-readback codec | 46.568 ms (45.885-46.707) | 41.430 ms (40.592-42.046) | **1.124x** | 50/50 |

The RF semantic work contract was identical: four Euler model evaluations,
batches `[2,2,1,1]`, six effective rows, twelve layers, 48 block calls, CFG
active on steps `[true,true,false,false]`, and schedule bits
`[1065336439,1061146329,1056947831,1048559223,0]`. This is deliberately not a
same-graph claim: both paths request joint axis 822, while PyTorch encodes and
forwards axis 820 and the production WGPU path compacts the active graph to axis
53 and uses derived context K/V plus the fixed-condition cache.

All 60 WGPU repetitions per output (warmups included) passed the ten latent and
waveform accuracy gates, and all cross-process hash sets were singletons. The
sealed artifact is
`/tmp/irodori-v4-same-precision-stage-ab-attempt6-20260812`; its
`SHA256SUMS` digest is
`c1ab4cd6543afb1d747594d02a57866bbf0181910900c669a304ce6346be431a`.
The measured source inventory is
`cb040f65b4d4e7fb68ac64c3d6bfaf4082fc2eacb23a056740b0527f1b3007ca`,
the frozen validator is
`0bb5db109bf7b0fc380a9d9b33a98d6a584aa13abc957fd2b6d657c424d63bad`,
and the repository checkpoint before the campaign was
`97658250a98070bc75d806a409adcf1221e6eaa3`.

Measured on 2026-08-09 and 2026-08-10. This report records pinned comparisons
rather than extrapolating from tiny tensors or a different model revision.
Measurements retained from 2026-08-09 are explicitly marked **historical**;
the current FP32 result is the 2026-08-10 production-WGSL validation.

## Technical summary

- The current production WGSL path completes the strict FP32 fixed-fixture
  RF+codec replay. Against the strict PyTorch waveform, the final latent has
  max abs `1.358985901e-4` and
  cosine `0.999999999789`; the raw waveform has max abs `8.601322770e-5`,
  SNR `87.415864 dB`, and cosine `0.999999999093`.
- With production-equivalent CubeCL command aggregation (`tasks_max=32`),
  SubSlices memory configuration, and single-CCD affinity, all five fresh
  strict processes produce a median-of-session-medians of `122 ms` RF and
  `61 ms` codec. This is the current production-equivalent deployment estimate.
- The pinned strict PyTorch CUDA steady medians are `138.591 ms` RF and
  `46.220 ms` codec. WGSL RF is about `16.6 ms` faster (`1.136x`, `12.0%` lower),
  while WGSL codec is about `14.8 ms` slower (`1.320x`, `32.0%` higher). The
  component sum is `183 ms` for WGSL and `184.811 ms` for PyTorch: WGPU is
  about `1.8 ms` (`0.98%`) lower, effectively near parity at this request.
  Codec is the clearest cross-runtime gap to prioritize, but RF remains
  `122 / 183 = 66.7%` of the current WGPU component sum and needs a current-tree
  profile.
- Earlier validators hardcoded `tasks_max=1`. Their `185-226 ms` RF results
  remain useful batching and scheduling diagnostics, but they are not
  production-equivalent timing estimates. Their parity metrics remain valid.
- The production-equivalent session with NVML sampling observed a `6,672 MiB`
  maximum from a `38 MiB` device baseline. This is a **staged process peak**:
  both the validator and production CLI logically drop the RF engine before
  loading the decode codec, but SubSlices may retain released allocator pages.
  NVML therefore cannot decompose it into live RF, live codec, or simultaneous
  live-tensor residency.

## Scope and pins

| Item | Pinned value |
|---|---|
| Rust working tree | Based on HEAD `5d5656e88a3030198c3d895ae032824459a5870a`, plus an uncommitted and untracked optimization set; this is not yet a reconstructible source pin |
| Official source | `Aratako/Irodori-TTS` commit `9f19d9a9048099a4b978a762d0509228fe624e3f` |
| v4-Small revision | `e4aaac4df355ff560dcd35e0dae272c3a759317b` |
| v4 model SHA-256 | `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593` |
| Codec revision | `47376ee24834d7a05a48ebabfe3cde29b3c5e214` |
| Converted codec SHA-256 | `4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1` |
| Official-default oracle SHA-256 | `8022b2baeed05e68dd2d335bebb10392b5817d1251e006413294ff597d363fc8` |
| Strict FP32 oracle SHA-256 | `c6feac7cabf1a0ef3b264e16619de0bed8f2d7501d02bbced6c940815c181061` |
| Measured validator binary SHA-256 | `33834072a838b5da335ab76a6880f08dffdabcc50b8ce78c6389d56f4285a984` |
| Validator source SHA-256 | `879b9a0ecad320596f4f0abd98b817721249063a7ab89b14b9e6cd62d38912c0` |
| Residue d1 Rust source SHA-256 | `f75c9fda31134af1fa64b270829404f8424a8166eefc70a478923dcd81f51e0a` (`conv1d_k7_residue_d1_snake.rs`) |
| Residue d1 core WGSL SHA-256 | `4eca5625ad4c45062956fefcaa034721ce26e33ad870317137e67a4f778d60bb` (`conv1d_k7_residue_d1_snake.wgsl`) |
| Residue pack WGSL SHA-256 | `3a16ad08b73295b3deefb65a0283652c351e9896ac03d3234b9a26ef556fac93` (`conv1d_k7_residue_pack.wgsl`) |
| Codec route-selection source SHA-256 | `9a80d84b7162629a36a8e7c94d5a5f3038175cb8ae6ccf246ebffb753c69b67a` (`layers.rs`) |
| Kernel registry source SHA-256 | `d7eeca7a19d4fca09f3bfa7dd7438ffb51be336846fbcfa6d6f3407bb49df574` (`kernels.rs`) |
| GPU | NVIDIA GeForce RTX 3060 Ti 8 GiB, driver 580.126.20, Vulkan |
| WGPU selection | `DiscreteGpu(0)`, PCI `0000:07:00.0` |
| Production-equivalent CubeCL runtime | `tasks_max=32`, SubSlices memory configuration |

WGPU adapter order did not match the host's NVML ordinal order. Every accepted run
printed the adapter index, name, backend, and device type; PCI identity was audited
externally. A different RTX 3060 Ti at PCI `04:00.0` was occupied by an unrelated
process and was not used.

## Oracle request

- Text: `こんにちは。`
- No reference and empty caption; only text CFG was active
- 2.0 seconds, 96,000 samples at 48 kHz, 50 latent frames
- Four Euler steps with a linear schedule
- Independent CFG requested as text/caption/speaker = 3/3/5. No-reference
  resolves speaker scale to zero; Python retains caption scale 3 in metadata,
  while the empty all-false caption mask makes that branch inactive
- Fixed exported initial noise, context KV cache enabled, no tail trim, f32
- Watermark disabled so the Rust waveform is compared to the raw decoder signal

### Oracle identity and FP32 math policy

The two fixtures are intentionally distinct:

| Fixture | SHA-256 | PyTorch FP32 policy | Correct use |
|---|---|---|---|
| Official-default | `8022b2ba...363fc8` | `matmul.allow_tf32=false`, `cudnn.allow_tf32=true` | Reproduce the official default output and retain historical comparisons |
| Strict | `c6feac7c...c181061` | `matmul.allow_tf32=false`, `cudnn.allow_tf32=false` | Current strict FP32 parity and precision comparisons |

The RF tensors, Euler steps, and final latent are identical between these two
oracles. The expected DACVAE waveform differs because the official-default
fixture permits cuDNN TF32. Therefore, results scored against `8022...` must
not be described as strict waveform parity. The Python exporter reproduced
the official-default raw WAV byte-for-byte on two independent processes, and
both fixtures replay their own Euler recurrence and final decode exactly.

## Current FP32 production result (2026-08-10)

### Strict fixed-fixture RF+codec numerical parity

The controlled fixed-fixture runs used `validate_v4_precision` with production
WGSL, FP32, `tasks_max=32`, SubSlices, and ten repetitions. The validator verified
the strict fixture SHA, official model SHA, and converted codec SHA before replay,
then printed the selected adapter's index, name, backend, and device type; PCI
identity was audited externally. The source noise remained bit-exact after its
single f32 cast. All measured repetitions produced identical metrics.

The run is explicitly report-only: it applies no hidden numerical thresholds,
while malformed metadata, hash mismatches, invalid shapes, non-finite values,
and WGPU structural errors fail closed. The replay starts from exported token
IDs/masks, fixed noise, and sequence length, so it validates RF plus DACVAE but
is not full CLI E2E and does not validate production tokenization, RNG, or
duration prediction.

| Output | Count | Max abs | Mean abs | RMSE | SNR | Cosine |
|---|---:|---:|---:|---:|---:|---:|
| Final patched latent | 1,600 | 1.358985901e-4 | 6.610504934e-6 | 1.332637764e-5 | 93.745297 dB | 0.999999999789 |
| Raw decoded waveform | 96,000 | 8.601322770e-5 | 2.106878479e-6 | 6.249392065e-6 | 87.415864 dB | 0.999999999093 |

### Production-equivalent controlled performance (`tasks_max=32`)

CubeCL's `tasks_max` is the maximum number of compute tasks aggregated into
one GPU command. With the locked CubeCL build (`exclusive_memory_only` unset)
and `CUBECL_WGPU_MAX_TASKS` unset, a fresh production pipeline resolves to
`tasks_max=32` and SubSlices; the precision validator sets both explicitly.
The recommended protocol also pins the process to one CCD with
`taskset -c 6-11,18-23`. Session 1 ran a 100 ms NVML sampler; sessions 2-5 did
not. Its `121 ms` RF median is 1 ms below the other four sessions, so the sampler
does not show an adverse median shift in this set.

Five fresh strict processes each ran ten repetitions; repeat 1 was excluded.

| Session | Model load/build | RF median (range) | Codec load | Codec median (range) |
|---:|---:|---:|---:|---:|
| 1 | 6.432 s | 121 ms (121-134) | 0.307 s | 61 ms (61-61) |
| 2 | 5.987 s | 122 ms (121-139) | 0.332 s | 61 ms (61-62) |
| 3 | 6.410 s | 122 ms (121-130) | 0.311 s | 61 ms (61-62) |
| 4 | 5.968 s | 122 ms (121-140) | 0.323 s | 61 ms (61-62) |
| 5 | 6.413 s | 122 ms (122-134) | 0.325 s | 61 ms (61-62) |
| Median of session medians | 6.410 s | **122 ms** | 0.323 s | **61 ms** |

The independently aggregated component sum is `183 ms`. Every repetition in
every session produced the latent and waveform metrics shown above. The accepted
log basenames and SHA-256 values are:

| Session | Log basename | SHA-256 |
|---:|---|---|
| 1 | `irodori-v4-fp32-strict-residue-tasks32-subslices-s1.log` | `fde4cf2b50defe7e2844d2c040ae6bd3e4f4a1eba9bb8d5ec89cdff15588b17f` |
| 2 | `irodori-v4-fp32-strict-residue-tasks32-subslices-s2.log` | `edf0495947c1bc0f2a0294104dbb88f99b06b62caa207025a5adc3c54c3330c8` |
| 3 | `irodori-v4-fp32-strict-residue-tasks32-subslices-s3.log` | `0256cfd9d50820acf6116e1e247c6238b477d9039087d770fc709ad59bede871` |
| 4 | `irodori-v4-fp32-strict-residue-tasks32-subslices-s4.log` | `6ba90bc9561d7c28403a2997e8576e827578dfaf226bf72fce1fa2d86061897e` |
| 5 | `irodori-v4-fp32-strict-residue-tasks32-subslices-s5.log` | `cd8ae7bb3bcdd163a9c8157e5bd027afd159f137a9f9d2a4345a23e71d4f71cf` |

All five WAV files have SHA-256
`7ab5cc93479a1043284b2c9c5d57918cf9918f8f61a7624dc4ec1706cca89975`.

Session 1 telemetry observed a `6,672 MiB` peak from a `38 MiB` baseline
(`6,634 MiB` delta), with up to `100%` sampled utilization, `57 C`, and
`135.82 W`. The source basename is
`irodori-v4-fp32-strict-residue-tasks32-subslices-s1-nvml.csv` (SHA-256
`40a80721096d777f8a41eef087b9f81735f5a8a5088e12db505d4b60cc32b664`). The validator
and production CLI both call `drop(engine)` before codec loading. SubSlices
can retain those freed pages in its pool, so this production staged-process
peak is not a measurement of simultaneous live model tensors and cannot be
split into independent RF and codec residency from NVML alone.

SubSlices sensitivity runs checked that `32` is not an arbitrary maximum.
Values for `8`, `16`, `64`, and `128` are steady medians from one process;
the `32` row is the five-session headline reference above:

| `tasks_max` | RF | Codec |
|---:|---:|---:|
| 8 | 128 ms | 65 ms |
| 16 | 124 ms | 65 ms |
| 32 | 122 ms | 61 ms |
| 64 | 126 ms | 66 ms |
| 128 | 134 ms | 66 ms |

The RF screen favors the existing production default of `32`; it does not justify
a production override. The non-32 rows predate the exact residue-class codec route,
so their `65-66 ms` codec values are retained as historical sensitivity context and
are not a matched codec comparison with the current `61 ms` row. Source logs are
`/tmp/irodori-v4-fp32-strict-task-sweep-{8,16,64,128}.log`, with the controlled
five-session `32` result above as the matching reference.

### `tasks_max=1` scheduling diagnostics

Earlier validators hardcoded `tasks_max=1`, so their parity evidence remains
valid but their timing is not production-equivalent. Under the same strict
validator, ExclusivePages memory configuration, one-CCD affinity,
five-process protocol, and no external sampler, changing `tasks_max=1` to
`32` lowers the RF median from `185` to `122 ms` (`1.516x`, about `34.1%`)
and codec from `66` to `65 ms`. The `tasks_max=32` ExclusivePages control also
measured `122/65 ms`, matching the pre-residue SubSlices result from that same
tree; it is retained only as the matched batching diagnostic and is not compared
to the current `122/61 ms` production result.

The matched `tasks_max=1` scheduling experiment was:

| Protocol | RF session medians | RF median of medians | Codec session medians | Codec median of medians | RF + codec |
|---|---|---:|---|---:|---:|
| Single-CCD affinity | 186 / 184 / 185 / 186 / 185 ms | 185 ms | 65 / 66 / 65 / 66 / 66 ms | 66 ms | 251 ms |
| Unpinned control | 205 / 205 / 191 / 189 / 189 ms | 191 ms | 66 / 66 / 66 / 66 / 66 ms | 66 ms | 257 ms |

Within `tasks_max=1`, the affinity run coincides with removal of the observed
bimodality and a `6 ms`, about `3.1%` (`1.032x`) lower RF median. This is retained
as scheduling sensitivity, not an isolated causal estimate; only the matched
`tasks_max=1` versus `32` change is used for causal timing attribution. Source
logs are
`/tmp/irodori-v4-fp32-strict-affinity-s{1..5}.log` and
`/tmp/irodori-v4-fp32-strict-unpinned-s{1..5}.log`; the ExclusivePages
`tasks_max=32` control is
`/tmp/irodori-v4-fp32-strict-tasks32-affinity-s{1..5}.log`.

An earlier single-process strict `tasks_max=1` r10 replay loaded/built RF in
`6.024 s` and loaded the codec in `0.301 s`. Excluding each stage's first
repetition, RF had median `197 ms` and range `189-215 ms`; codec had median
`66 ms` and range `65-90 ms`. Its `263 ms` stage-median sum and source log
`/tmp/irodori-v4-fp32-strict-wgsl-r10.log` are retained as diagnostic evidence.

### Earlier `tasks_max=1` unpinned/external-telemetry history

The following five fresh processes used the older `validate_v4_e2e`
executable, official-default fixture, unpinned CPU scheduling, and external
NVML sampling. It also hardcoded `tasks_max=1`. Repeat 1 was excluded. RF
inputs and RF oracle tensors are
bit-identical between official-default and strict, so the runs remain useful
performance diagnostics, but their protocol is not matched to the controlled
strict result.

| Session | Model load/build | RF median (range) | Codec load | Codec median (range) | Peak device memory |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.561 s | 227 ms (226-234) | 0.312 s | 65 ms (65-96) | 4,897 MiB |
| 2 | 6.311 s | 226 ms (225-238) | 0.311 s | 65 ms (65-91) | 4,899 MiB |
| 3 | 6.124 s | 188 ms (187-190) | 0.302 s | 66 ms (65-91) | 4,909 MiB |
| 4 | 5.824 s | 226 ms (225-228) | 0.297 s | 66 ms (65-89) | 4,889 MiB |
| 5 | 5.708 s | 189 ms (187-190) | 0.320 s | 66 ms (65-91) | 4,911 MiB |
| Median of session medians | 6.124 s | **226 ms** | 0.311 s | **66 ms** | 4,899 MiB |

The sum of independently aggregated RF and codec medians is `292 ms`; the
median of the five per-session sums is `291 ms`. RF splits into a fast
`188-189 ms` group and a slow `226-227 ms` group while output metrics remain
identical. Hardware thermal slowdown was inactive, observed SM clock reached
`1,950 MHz`, maximum temperature was `59 C`, and maximum sampled power was
`139.39 W`. These observations rule out output drift and observed hardware
thermal throttling within that protocol, but do not isolate the cause of the
old timing modes.

Source logs are `/tmp/irodori-v4-fp32-final-session{1..5}.log`; sampled
telemetry is `/tmp/irodori-v4-fp32-final-session{1..5}-nvml.csv`.

All five historical outputs, as well as the earlier strict-r10 output, have
WAV SHA-256
`7ab5cc93479a1043284b2c9c5d57918cf9918f8f61a7624dc4ec1706cca89975`.
When that same WGPU waveform is scored against the official-default target,
waveform max abs is `8.134767413e-4`, RMSE is `7.565666239e-5`, SNR is
`65.755915 dB`, and cosine is `0.999999867511`. Those are
**official-default**, not strict, waveform metrics.

### PyTorch CUDA comparison

The PyTorch process reused one loaded runtime for four repetitions; the table
uses the median of repetitions 2-4. Strict mode disabled both matmul and cuDNN
TF32. WGPU uses the production-equivalent `tasks_max=32`/SubSlices,
affinity-controlled median of five fresh-process medians above. Ratios below
are elapsed-time ratios, so values below `1.0x` favor WGPU.

| Runtime | FP32 policy | RF | Codec | RF + codec | Relative to strict PyTorch |
|---|---|---:|---:|---:|---:|
| PyTorch CUDA | Strict | 138.591 ms | 46.220 ms | 184.811 ms | 1.000x |
| Production WGSL | Strict, `tasks_max=32`, SubSlices, one-CCD affinity | 122 ms | 61 ms | 183 ms | RF 0.880x; codec 1.320x; sum 0.990x |
| PyTorch CUDA | Official-default | 132.115 ms | 29.722 ms | 161.837 ms | Not a strict comparison |

WGPU RF is about `16.6 ms` faster than strict PyTorch (`1.136x` throughput,
`12.0%` less elapsed time). WGPU codec is about `14.8 ms` slower (`1.320x`,
`32.0%` more elapsed time). The component sums differ by about `1.8 ms`,
with WGPU `0.98%` lower (`1.0099x` PyTorch/WGPU). Codec is therefore the
clearest cross-runtime stage gap to prioritize. RF is still `122 / 183 = 66.7%`
of the current WGPU component sum, so the RF win does not remove the need for a
current-tree profile and optimization review.

The matched `tasks_max` experiment supports the one causal timing statement in
this comparison: under the strict validator and affinity policy, changing only
command aggregation from `tasks_max=1` to `32` reduced RF from `185` to `122 ms`
(`63 ms`, `34.1%`). Earlier semantic compaction appears in a sequential historical
path from `591` to `343 ms`; QKV post-processing and AdaLN changes were followed by
about `303 ms` under the older submission protocol. Those historical steps, the
affinity comparison, and the codec pre/post refresh are useful directional evidence,
but they do not isolate causal contributions to the current cross-runtime gap. The
remaining large GEMMs and SDPA stay on tuned Burn/CubeCL routes because handwritten
fixed-K1280 GEMMs and SDPA candidates were measured substantially slower and rejected.
The current RF path contains graph compaction, cache/materialization removal, fused
pointwise work, and submission batching, but this evidence does not assign its lead
to each change individually.

This is a product-path comparison, not a backend-only experiment. The pinned
official PyTorch path retains 256 text slots plus masked speaker and caption slots,
giving an 820-position self-plus-context attention axis. Rust right-trims to the
three valid text tokens and removes the fully masked auxiliary pairs, giving 53
positions while preserving the active semantics. A same-graph CUDA-versus-Vulkan
comparison would need the same compaction on both implementations.

The WGPU validator prints stage time to 1 ms, while the Python JSON retains
sub-millisecond values; WGPU uses five fresh processes with nine steady samples
each, while the Python anchor uses three steady repeats in one loaded process.
The about `1.8 ms` sum difference is therefore reported as near parity, not as a
statistically established advantage. The much larger, directionally opposite
RF and codec differences remain the decision-useful stage result.

These are steady fixed-fixture RF+codec measurements: model loading is excluded,
and the `183 ms` WGPU value is the sum of separately measured RF and codec medians,
not full CLI E2E or a cold full-CLI wall time.

The CPU scheduling policy is not matched across runtimes: WGPU is explicitly
single-CCD pinned, while the Python benchmark did not record an equivalent
affinity pin. Timing scopes also differ slightly: Rust RF includes the final
1,600-f32 latent readback to the CPU, while Python `sample_rf` stops after CUDA
synchronization and before a latent CPU copy. Codec includes waveform CPU transfer
in both runtimes. The near-parity conclusion therefore applies to these stated
deployment protocols, not to an isolated backend-only causal comparison.

PyTorch's strict median wall time through decode was `185.616 ms`; the
`184.811 ms` table value is the sum of its independently reported RF and codec
stage medians. PyTorch reported a maximum steady-repeat CUDA allocation of
`4,011.021 MiB` and reservation of `4,412 MiB`. These allocator numbers are
not directly comparable with either WGPU NVML peak (current `6,672 MiB` or
historical `4,899 MiB`): APIs, allocator reservation policies, and model lifetimes
differ.

## Optimization decision ledger

“Accepted” means the route is connected to the production WGSL policy and is
covered by the current strict fixed-fixture RF+codec replay. Microbenchmark
savings in this table come from different scopes and are not additive. “Rejected” means the
candidate is not selected by production; diagnostic benchmark code may remain
to preserve the negative result.

### Accepted production routes

| Area | Production decision | Evidence used for acceptance |
|---|---|---|
| Runtime submission | Match production CubeCL aggregation with `tasks_max=32` and SubSlices | Matched ExclusivePages batching control RF `185 -> 122 ms`; production SubSlices also `122 ms`; identical parity metrics |
| Conditions and context | Right-trim valid masks, remove fully masked auxiliary pairs, and retain packed-only context K/V | Historical RF `591 -> 343 ms`; calculated 12-layer cache payload `541.406 -> 1.055 MiB`; current strict fixed-fixture RF+codec replay parity |
| Attention and CFG | Combined QKV/gate preparation, direct K/V and post-SDPA handling, and exact B1-to-B2 text-CFG derivation | Historical QKV post-process microbenchmark `24.30-36.39x`; branch/mask contract checks; current strict fixed-fixture RF+codec replay parity |
| AdaLN and recurrent state | Cross-layer prepared AdaLN, ModernBERT carry reuse, and fixed-Euler condition caching | AdaLN projection aggregate B1 `918.1 -> 305.5 us`, B2 `1054.6 -> 306.7 us`; current strict fixed-fixture RF+codec replay parity |
| FFN and output projection | Flattened/fused FFN, B1 row-packed W2 cache, and shape-gated B1 row route for `wo` | W2 plus residual weighted request estimate `36.427 -> 33.142 ms`; B2 retains source-column route; current strict fixed-fixture RF+codec replay parity |
| Codec pointwise work | Paired/finalizer shaders and layout-preserving pointwise fusion | Exact-output isolated checks plus current strict waveform replay |
| ConvTranspose stage 0 | Cin32 T64/O16 polyphase tile | `2198.469 -> 2150.803 us` (`1.022x`), bit-exact |
| ConvTranspose stages 1-3 | Tuned GEMM plus cached-column col2im finalizer | Production medians `974.418 / 1966.177 / 1194.925 us`, bit-exact against Burn reference |
| Residual k7 Conv1d | Shape-selected T128/T256 and vec4 routes, plus an exact residue-class d1 pack/core route for C192/L48000 at d3 and d9 | Exact-two production sum `11.693027 -> 7.914309 ms`, saving `3.778718 ms` (`1.477x`); 18,432,000 outputs bit-identical, zero WGPU errors, and current strict fixed-fixture RF+codec replay parity |
| Waveform head | Fuse Snake with NLC write before the unchanged Conv/tanh tail | Full WmHead `1730.926 -> 1375.178 us` (`1.259x`), bit-exact |

The residue-class route was admitted only for the two measured C192/L48000
dilations. It compacts the input by dilation residue, runs the d1 core, and writes
the final Snake output directly. The isolated candidate includes pack plus core:
d3 measured `5302.842 -> 3889.304 us`, d9 `6390.186 -> 4025.005 us`, and the
two-route sum measured `11693.027 -> 7914.309 us`. The benchmark log basename is
`irodori-bench-k7-residue-d1.log` (SHA-256
`589c6a15648e351ed30c94bd4fb8eee9656ce2e66835745e5a2af9bca362673a`).

The W2 row-pack benchmark measured a one-time 12-layer break-even of about
53 requests. Its steady-state result therefore does not claim a one-shot load
latency win; production builds the cache once and reuses it.

### Rejected or deliberately bounded candidates

| Candidate | Result | Decision |
|---|---|---|
| Handwritten WGSL SDPA | `2.35-2.51x` slower than tuned Burn/CubeCL at H20/Dh64 | Reject; retain tuned SDPA |
| Strict-FP32 cooperative matrix | All six exposed configurations use F16 A/B inputs; zero exact strict-F32 configurations | Fail closed; do not call mixed-input tensor-core math strict FP32 |
| Fixed K1280 handwritten GEMM | Four-step expand `27.046 -> 206.608 ms`; raw QKV `21.850 -> 143.401 ms` | Reject; retain Burn/CubeCL GEMM |
| Fixed native-column W2 plus residual | Weighted 48-call estimate `36.427 -> 638.969 ms` | Reject; use B1 row cache and B2 source-column route |
| Cin32 polyphase retile for ConvTranspose stages 1-3 | Best candidates `3933.697 / 7982.917 / 3945.678 us` versus cached-column `974.418 / 1966.177 / 1194.925 us` | Reject; keep cached-column routes |
| One global T256/Cin16 k7 route | Exact-12 aggregate `0.991x` versus T128 baseline, with regressions on several shapes | Reject global policy; accept only measured per-shape T128/T256 selection |
| Vec4 store for C768/L600/d9 | `1922.423 -> 1924.886 us` | Reject for this shape; retain scalar-store T256 while using vec4 only on the eight measured wins |

## Historical production-kernel measurements (2026-08-09)

The following microbenchmarks predate the final 2026-08-10 production tree.
They are retained as optimization evidence and must not be read as the current
end-to-end timing.

`bench_fused_hotpath` used 10 warmups and 100 measured iterations. These are the exact
`S=50`, `D=1280`, `H=3680` shapes exercised by the pinned request.

| Operation | Shape | Portable | Fused WGSL | Speedup | max abs |
|---|---|---:|---:|---:|---:|
| SwiGLU | B1/S50/H3680 | 65.2 us | 6.9 us | 9.40x | 2.384e-6 |
| Gated residual | B1/S50/D1280 | 20.0 us | 6.8 us | 2.95x | 1.192e-7 |
| AdaLN elementwise | B1/S50/D1280 | 71.0 us | 8.1 us | 8.75x | 4.768e-7 |
| Final RMSNorm | B1/S50/D1280 | 53.2 us | 7.7 us | 6.95x | 4.768e-7 |
| QKV norm + half-RoPE | B1/S50/H20/Dh64 | 366.8 us | 10.1 us | 36.39x | 7.153e-7 |
| SwiGLU | B2/S50/H3680 | 64.2 us | 14.6 us | 4.39x | 2.384e-6 |
| Gated residual | B2/S50/D1280 | 20.5 us | 7.1 us | 2.87x | 1.192e-7 |
| AdaLN elementwise | B2/S50/D1280 | 70.4 us | 8.7 us | 8.05x | 4.768e-7 |
| Final RMSNorm | B2/S50/D1280 | 52.3 us | 8.0 us | 6.56x | 4.768e-7 |
| QKV norm + half-RoPE | B2/S50/H20/Dh64 | 331.1 us | 13.6 us | 24.30x | 7.153e-7 |
| Codec Snake1d | B1/C96/T96000 | 1005.8 us | 180.4 us | 5.57x | 4.768e-7 |

`bench_adaln_modulation` separately measured the six low-rank projection GEMMs.
Packing the shift/scale/gate weights once and using two rank-4 branch-batched matmuls
reduced B1 from 918.1 to 305.5 us (3.01x) and B2 from 1054.6 to 306.7 us
(3.44x), with max abs zero. The one-time pack measured 79.1 us per AdaLN module.

All six raw shader families were connected through the WGSL execution policy, and the
benchmarks had no WGPU or CubeCL validation errors. Projection matmuls and SDPA remain
on Burn/CubeCL; only QKV post-processing is handled by the new attention shader. The
custom SDPA candidate was measured at the actual 20-head, 64-wide head shape and was
2.35-2.51x slower than the tuned portable implementation, so it was rejected.

## Historical profile-driven context and cache optimisation (2026-08-09)

The replay and trace timings in this section used the validator's former
hardcoded `tasks_max=1`. They preserve the direction of the measured
optimization work, but are not production-equivalent absolute timings.

An Nsight Systems trace of the pre-optimisation fixed replay measured 0.308 s and
4,995 Vulkan queue submissions in condition encoding, while the four DiT evaluations
took 0.375 s and 7,259 submissions. Attention accounted for 176.8 ms (47.2%) of that
DiT interval. The request carried 256 text slots (three valid), a fully masked speaker
placeholder, and a fully masked 512-slot caption. That produced 770 context positions
and an 820-position self-plus-context attention key axis.

The sampler now right-trims masks to the last valid column before encoding and removes
fully masked auxiliary pairs. For this request, ModernBERT sees three text tokens,
caption and speaker encoders are skipped, and the attention key axis is 53 rather than
820. This is semantic compaction: no valid token or position is removed.

`CondKvCache` now retains only packed context K/V, its masks, and the optional speaker
range. It no longer keeps both split and concatenated projections. At the pinned
B1 + independent-CFG B2 shapes, the old split-plus-packed 12-layer caches account for
541.406 MiB; compact packed-only caches account for 1.055 MiB. These byte counts are
from the actual tensor dimensions and f32 element size, not an allocator estimate.

## Historical official-default numerical parity (2026-08-09)

This section is preserved unchanged in substance, but its oracle label is
corrected: every waveform metric below was scored against the
`8022b2ba...363fc8` official-default fixture with cuDNN TF32 enabled. It is
not strict waveform parity. The old validator also used `tasks_max=1`; that
does not affect these parity metrics, only its timing.

The validator streams and verifies the fixture, model, and converted-codec SHA-256
before GPU initialization. Each stage synchronizes CubeCL, checks a device-wide WGPU
uncaptured-error monitor, requires valid metrics, and applies fail-closed default
error and cosine gates. The defaults are latent max abs `5.0e-4`, waveform max abs
`2.0e-3`, latent cosine `0.999999`, and waveform cosine `0.99999`.
`--latent-max-abs` and `--waveform-max-abs` explicitly override the two default
maximum-error thresholds; the cosine floors are fixed.

The replay starts from fixture tensors: token IDs and masks, the fixed initial noise,
and the resulting sequence length are not produced by Rust during this check.
Consequently, this parity result does not validate the tokenizer, production RNG, or
duration predictor. Production `pipeline` CLI validation exercises those stages
separately and must not be conflated with the fixture replay.

| Execution policy | Final latent max abs | Latent cosine | Raw waveform max abs | Waveform SNR | Waveform cosine |
|---|---:|---:|---:|---:|---:|
| Production fused WGSL, official-default target | 1.139640808e-4 | 0.999999999828 | 8.144527674e-4 | 65.716405 dB | 0.999999866308 |
| Portable WgpuRaw, official-default target | 1.244544983e-4 | 0.999999999828 | 8.189305663e-4 | 65.751907 dB | 0.999999867396 |

The production `pipeline` independently exercised bundled-tokenizer discovery,
normalization, BOS/right-padding, the no-ref sentinel, fixed-noise loading, fused RF,
DACVAE, official PCM16 quantization, and WAV writing. Its 96,000-sample WAV versus the
official no-watermark PCM measured max abs `8.239746094e-4`, RMSE
`7.666257178e-5`, SNR `65.641192 dB`, and cosine `0.999999863959`.
Because that deterministic command supplied both `--noise-file` and `--seconds`, its
reported parity covers the tokenizer but not the production RNG or duration
predictor; those require separate CLI checks without the corresponding overrides.

Historical accepted production WAV from that optimisation run:
`/tmp/irodori-v4-production-optimized.wav` (SHA-256
`7d0a7bfd635ba4b77e9bbbe7aa304628d40c7dc7ac48ad680808cb3c980a14ef`).

## Historical timing and memory (2026-08-09)

These values describe the earlier tree and protocol. The 2026-08-10 results
above supersede them for current steady-state performance; they remain useful
as the measured optimization progression.

The repeated validator measurements and optimization-stage table below used
`tasks_max=1`. The final one-shot `pipeline` paragraph used the production
runtime rather than that validator, but still describes the historical tree.

The same loaded WGSL engine replayed the exact request four times at each optimisation
stage. The table reports steady-state medians; cold runs include pipeline creation and
autotuning and are intentionally excluded from the speedup calculation.

| Stage | Steady RF | Change from profiled baseline |
|---|---:|---:|
| Profiled fused baseline | 0.591 s | - |
| Masked-condition compaction | 0.343 s | 42.0% lower |
| + fused QKV post-processing | 0.312 s | 47.2% lower |
| + branch-batched AdaLN modulation | 0.303 s | 48.7% lower |

The final four-run process measured RF `0.841 / 0.300 / 0.303 / 0.311 s`; its model
load/build was 5.902 s. Codec decode measured `0.612 / 0.577 / 0.576 / 0.583 s`.
Snake fusion is 5.57x in isolation, but end-to-end codec improvement is small because
the transposed convolutions dominate decode.

The real one-shot production CLI, with RF model and codec resident together, measured
RF 2.241 s and codec 0.479 s with a warm autotune cache. A 100 ms NVML trace observed
a 6,740 MiB peak from a 38 MiB baseline; the process completed on the 8 GiB card and
returned to 38 MiB. These timings are reported as measured and are not compared to an
earlier cold-cache one-shot run.

## Selected v4 product-path coverage

The strict fixed-fixture RF+codec replay is intentionally text-only, no-reference,
and fixed at two seconds. It is not full E2E and does not by itself prove every
v4 product path. Code reachability, available tests, and current oracle coverage
lead to the following narrower status labels:

| Capability | Current status | What is implemented | Validation gap or limitation |
|---|---|---|---|
| Learned duration prediction | Implemented for inference; released-checkpoint reachability only | Released v4 duration topology and 31 tensors, 14-value feature vector, presence flags, `expm1`, scaling, ties-to-even rounding, bounds, and patch ceiling | This report supplies `seconds=2.0`, so the strict Python-versus-Rust fixed-fixture RF+codec replay bypasses the predictor; no Python↔Rust numerical duration oracle exists yet |
| Voice Designer core conditioning | Partial | Caption, speaker reference, both together, and neither are represented through the model and WGSL pipeline; caption CFG is connected | No official Gradio workflow, candidate generation UI, multi-reference concatenation, Speaker Inversion, or active-caption-plus-active-speaker fixed-fixture oracle |
| Voice cloning | Partial | A single WAV can be encoded by DACVAE and used as speaker conditioning in the production pipeline | Rust currently uses mono conversion, linear resampling, and trimming, not the official -16 dB audiotools normalization and torchaudio band-limited resampling; multi-reference and numerical reference-path parity are absent |
| Speaker Inversion | Not implemented | — | No speaker-state inversion model or `.speaker.safetensors` override path; legacy `--ref-latent` is a different codec-latent mechanism |
| LoRA inference | Partial | One safetensors PEFT adapter can be merged into base weights at load time before `build_wgsl()` | One fixed adapter per process; no hot swap, composition, `modules_to_save` duration payload, complete bias handling, or nonzero-v4-adapter Python/WGSL fixed-fixture oracle |
| LoRA training | Limited | CUDA or LibTorch training of the supported diffusion-attention projections, with optimizer/checkpoint machinery | Not WGPU training; no full model, frontend/MLP, duration head, DDP, correct combined active speaker+caption branch, or enforced `target_modules` filtering |

The highest-value functional validation follow-up is an actual v4 fixture with
active caption, active speaker reference, and automatic duration in the same
request. Reference-preprocessing parity and multi-reference support come next;
Speaker Inversion and broader v4 LoRA training require new implementation, not
only another benchmark.

As a reachability smoke on the released checkpoint, the production WGSL CLI
was run without `--seconds` or `--seq-len`. For `こんにちは。` it predicted
`45.381` latent frames, resolved that to 45 frames and 1.8 seconds, and wrote
an 86,400-sample WAV. The first run included previously unseen-shape autotuning,
so its timing is deliberately excluded from performance tables. This establishes
released-checkpoint reachability of the implemented inference route only; it does
not establish Python-versus-Rust numerical duration parity. The log is
`/tmp/irodori-v4-duration-smoke.log` and the WAV
SHA-256 is `0434ce45866b47cdaf46cb60c5add797b88625b2653f103c1ce7f96ec2a0737d`.

A separate released-checkpoint caption-only smoke used `落ち着いた声で`,
reported three valid caption tokens with `caption_present=true`, completed the
four-step production WGSL sampler, and wrote a 96,000-sample WAV (SHA-256
`6b8a41af2020e7705bc932c490170c3bb1677c785aff18c1bbca3682e8c9dae8`).
Its first-use caption shapes also autotuned, so `/tmp/irodori-v4-caption-smoke.log`
is reachability evidence rather than a performance or Python-parity result.

The combined `Both` state was also exercised by feeding the duration-smoke WAV
back as a single reference alongside that caption. DACVAE produced a 45-frame
reference latent; the log reported `caption_present=true` and
`speaker_present=true`, then completed RF and a 96,000-sample decode (WAV
SHA-256 `75bc6b2bdf325de11fa3823c3cf9003470dc8ca0eda710b779b34506b8cee147`).
The expected preprocessing-parity warning was emitted. Therefore
`/tmp/irodori-v4-both-smoke.log` establishes production reachability only, not
official clone quality or numerical parity.

## Post-18 lower-precision comparison placeholder

**No row in this section is a measurement yet.** Collection is intentionally
deferred until at or after 2026-08-10 18:00 JST. Each measured row must use the
same PCI `0000:07:00.0` GPU, request, model/codec pins, source noise,
fresh-process protocol, production `tasks_max=32`/SubSlices runtime policy,
and precision-specific fixed oracle. The final table must report both
performance and latent/waveform error rather than timing alone.

| Runtime/path | Precision | Oracle SHA-256 | RF | Codec | Peak memory | Latent error/cosine | Waveform error/SNR/cosine | Status |
|---|---|---|---:|---:|---:|---|---|---|
| PyTorch CUDA | Strict FP32 refresh | Pending | — | — | — | — | — | New same-GPU anchor pending |
| Production WGSL | FP32 anchor replay | Pending | — | — | — | — | — | New-anchor replay pending |
| PyTorch CUDA | FP16 | Pending | — | — | — | — | — | Post-18 measurement pending |
| Rust portable WgpuRaw | FP16 | Pending | — | — | — | — | — | Replay path implemented; qualification pending |
| PyTorch CUDA | BF16 | Pending | — | — | — | — | — | Post-18 measurement pending |
| Rust WGPU | BF16 | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | Unavailable | CLI/backend path not implemented or qualified |

The existing strict FP32 fixture `c6feac7c...c181061` remains the pinned
oracle for the current Rust validation above. It predates the hardened
cross-precision device-metadata contract and must not be mixed with new
fixtures by the post-18 comparator. A new strict FP32 fixture must be exported
on the same physical GPU alongside FP16 and BF16 before filling this table.

The post-18 runbook is fail-closed:

1. Expose exactly one CUDA device and verify model and codec placement on it.
2. Record and require stable device name, visible index, PCI identity, and
   total memory in every new fixture and benchmark result.
3. Write the Python benchmark JSON, then return nonzero unless native-latent,
   native-audio, f32-latent, and f32-audio repeat hashes are each stable.
4. Require exact device metadata equality in the cross-precision comparator;
   reject mixed legacy/new fixtures rather than inferring equivalence.
5. Run Rust on WGPU PCI `0000:07:00.0` with the FP32 protocol above and report
   accuracy and timing together.

At this snapshot, `validate_v4_precision` supports portable FP16 replay, while
production WGSL is deliberately fail-closed for any precision other than FP32.
BF16 must not be reported until both the backend path and native-dtype oracle
contract exist and pass structural validation.

## Selected replay commands

Current strict production-WGSL replay:

The `torch-sys` directory below identifies the measured local release build. A fresh
build may use a different hash-named directory; the full five-process loop, NVML
sidecar collection, and per-session `--output-wav` names are recorded in the source
logs rather than reproduced by this single replay example.

```bash
LD_LIBRARY_PATH=target/release/build/torch-sys-c3dbe1af714b189e/out/libtorch/libtorch/lib \
  taskset -c 6-11,18-23 target/release/validate_v4_precision \
  --execution wgsl --precision fp32 \
  --fixture /tmp/irodori-v4-fp32-strict-oracle.safetensors \
  --fixture-sha256 c6feac7cabf1a0ef3b264e16619de0bed8f2d7501d02bbced6c940815c181061 \
  --checkpoint /path/to/Irodori-TTS-v4-Small/model.safetensors \
  --codec-weights target/v4_dacvae_weights.safetensors \
  --adapter-index 0 --tasks-max 32 --memory-config sub-slices --repeats 10
```

Historical 2026-08-09 commands retained for reproducibility:

```bash
cargo run --release --bin bench_fused_hotpath -- 0

cargo run --release --bin bench_adaln_modulation -- 0

cargo run --release --bin validate_v4_e2e \
  --features inference,codec,cli -- \
  --checkpoint /path/to/Irodori-TTS-v4-Small/model.safetensors \
  --execution wgsl --repeats 4 --adapter-index 0 \
  --tasks-max 1 --memory-config exclusive-pages

cargo run --release --bin validate_v4_e2e \
  --features inference,codec,cli -- \
  --checkpoint /path/to/Irodori-TTS-v4-Small/model.safetensors \
  --execution portable --repeats 4 --adapter-index 0 \
  --tasks-max 1 --memory-config exclusive-pages
```

The exact production pipeline CLI command is the deterministic fixture-replay
command in the root README with the pins in this report.

## Discarded measurements and limits

- The historical `tasks_max=1` logs predate binary/source SHA recording for the
  dirty optimization tree. They are retained only as protocol diagnostics and
  cannot reconstruct an exact source artifact independently. The current
  `tasks_max=32` result pins the measured validator binary and entry-file source,
  but the full dirty and untracked transitive source tree has not yet been archived.
- A run that mapped `DiscreteGpu(1)` to the occupied PCI `04:00.0` adapter OOMed and
  produced corrupt numbers. It was discarded and never used for a parity claim.
- An early production CLI run exposed a raw-`SourceKernel` storage-usage conflict:
  read-only and read-write logical slices shared one physical CubeCL buffer. Its WAV
  had cosine 0.0133 and was discarded. Production shaders now use uniform
  `read_write` storage declarations, a shader-contract regression test enforces this,
  and every accepted log is free of WGPU validation errors.
- The validated path is text-only/no-reference. Reference WAV preprocessing currently
  differs from official audiotools normalization and torchaudio resampling; the CLI
  warns rather than claiming reference-path numerical parity.
