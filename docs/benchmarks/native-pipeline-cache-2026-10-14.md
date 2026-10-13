# Native Vulkan pipeline cache — October 14, 2026

The optional `native-pipeline-cache` feature persists driver compilation data
for both handwritten WGSL and generated SPIR-V in the existing CubeCL cache.
It targets compilation during process startup; it does not change inference
arithmetic, tensor precision, graph capture or autotune selections.

## Results

All 164 MF processes (four seeds and 160 measured fresh processes) passed exact
latent/audio comparisons with the frozen pre-change output for their precision.
No trial made a fresh autotune decision, and no trial was retried. RTX FP16
benefits even with normal driver caching: median first inference is **1.884 →
1.131 s (39.9% shorter)**. Normal-driver FP32 and Intel results are essentially
unchanged. With implicit driver disk caching disabled, all four cases benefit.

First sampler + decoder through CPU PCM readback, **after loading**, seconds;
each cell reports **median / p95** from ten fresh processes:

| Device / model | Driver disk cache | Native OFF | Native ON |
|---|---|---:|---:|
| RTX FP32 | Normal | 1.388 / 1.396 | 1.386 / 1.395 |
| RTX FP16 | Normal | 1.884 / 1.908 | 1.131 / 1.139 |
| Intel FP32 | Normal | 1.305 / 1.340 | 1.300 / 1.320 |
| Intel FP16 | Normal | 1.376 / 1.392 | 1.375 / 1.398 |
| RTX FP32 | Disabled | 5.039 / 5.152 | 1.388 / 1.398 |
| RTX FP16 | Disabled | 3.747 / 3.800 | 1.121 / 1.126 |
| Intel FP32 | Disabled | 2.187 / 2.191 | 1.294 / 1.300 |
| Intel FP16 | Disabled | 2.295 / 2.321 | 1.364 / 1.390 |

Model and codec load times below are synchronized medians under normal driver
caching. Complete first audio includes device/cache setup, both loads and first
inference. Each value is **OFF → ON**, seconds:

| Case | Model load | Codec load | Complete first audio, normal | Complete first audio, driver cache disabled |
|---|---:|---:|---:|---:|
| RTX FP32 | 1.101 → 1.098 | 0.233 → 0.222 | 3.438 → 3.388 | 7.310 → 3.419 |
| RTX FP16 | 0.603 → 0.608 | 0.236 → 0.236 | 3.414 → 2.682 | 5.591 → 2.658 |
| Intel FP32 | 1.528 → 1.519 | 0.294 → 0.289 | 3.590 → 3.562 | 4.489 → 3.538 |
| Intel FP16 | 0.811 → 0.810 | 0.304 → 0.313 | 2.932 → 2.939 | 3.959 → 2.925 |

The explicit native cache saves actual driver creation work. Total pipeline
creation across validation, with the implicit driver cache disabled, drops from
8.427 → 0.042 s (RTX FP32), 7.393 → 0.041 s (RTX FP16), 3.138 → 0.018 s
(Intel FP32), and 3.187 → 0.018 s (Intel FP16). With normal driver caching,
RTX FP16 still drops from 0.835 → 0.059 s. These totals also cover the subsequent
streaming checks, so they must not be substituted for first-request latency.

The remaining initial-request cost includes generated-IR/WGSL processing that
this driver cache does not persist. The earlier prototype's ~0.12 s RTX first
request included generated-SPIR-V persistence and is not a result of the final
native-only feature. That additional path was removed during review.

Resident inference is unchanged within observed variation: normal-driver
OFF/ON medians are 76.8/77.1 ms (RTX FP32), 54.0/53.9 ms (RTX FP16),
684.9/684.5 ms (Intel FP32), and 682.1/682.2 ms (Intel FP16).

| Case | Native entries | Encoded native bytes | Seed serialization + saving | Normal restore time | Process peak RSS, OFF → ON |
|---|---:|---:|---:|---:|---:|
| RTX FP32 | 339 | 5,133,581 | 48.1 ms | 8.5 ms | 1081.1 → 1086.3 MiB |
| RTX FP16 | 341 | 4,896,810 | 41.8 ms | 7.9 ms | 947.1 → 953.7 MiB |
| Intel FP32 | 336 | 5,289,722 | 48.4 ms | 9.9 ms | 858.3 → 869.5 MiB |
| Intel FP16 | 347 | 5,369,805 | 28.1 ms | 9.7 ms | 740.3 → 751.0 MiB |

RSS is process memory, not VRAM. Restore includes database scanning/checksums and
native cache-handle creation. Native bytes include the private envelope; all
cache files including existing tuning data and WAL/coordination files measure
6,651,368 / 6,508,552 / 6,854,320 / 7,138,080 bytes at the four seed exits.
Of the 80 restored trials, 77 save no new native entries; three compile and save
one additional pipeline during loading (8,326 / 8,343 / 3,800 encoded bytes).
Across these trials, 27,261 of 27,264 pipeline creations receive saved data.
No entry is rejected. These three samples remain in every timing statistic;
the results do not assume a perfectly prepopulated native cache.

Cache OFF, cache ON and the frozen earlier output are bitwise identical **within
each precision**. FP16's pre-existing difference from the FP32 PyTorch oracle is
preserved, not removed. MF oracle errors, unchanged across every cache mode:

| Case | Latent MAE | Audio RMSE | Audio SNR |
|---|---:|---:|---:|
| RTX FP32 | 3.222e-6 | 2.277e-6 | 93.74 dB |
| RTX FP16 | 3.655e-3 | 3.055e-3 | 31.19 dB |
| Intel FP32 | 2.016e-6 | 1.216e-6 | 99.19 dB |
| Intel FP16 | 1.627e-3 | 1.223e-3 | 39.14 dB |

Per-process timings, hashes, startup boundaries, RSS, reference errors and commands
are retained in [the machine-readable evidence](data/native-pipeline-cache-2026-10-14.json).

## Additional regression checks

Another 24 processes pass: eight RF runs (first use and restore on each
device/precision), twelve runs of the previous committed binary, and four MF
graph-replay runs. All preserve the matching frozen latent/audio output and make
zero fresh autotune decisions. Streaming/full and trimmed/full output remain
bitwise equal. Graph capture/replay matches ordinary execution with two distinct
noise inputs on both GPUs and both precisions.

The RF fixture uses 100 latent frames / four seconds of audio and 40 Euler steps.
FP16 RF explicitly retains FP32 sampler state and codec. Its restored processes
receive saved data for all 380 / 392 / 381 / 393 pipelines respectively, save no
new entries, and preserve the following differences from the FP32 PyTorch oracle:

| Case | Latent MAE | Audio RMSE | Audio SNR |
|---|---:|---:|---:|
| RTX FP32 | 1.007e-6 | 1.499e-6 | 98.19 dB |
| RTX FP16 | 1.406e-3 | 2.835e-3 | 32.66 dB |
| Intel FP32 | 1.555e-6 | 3.456e-6 | 90.94 dB |
| Intel FP16 | 1.026e-3 | 9.034e-4 | 42.59 dB |

As a separate compatibility check, the previous `6e9a3d2` release binary runs
the same packed MF workload three times per case. Complete validator process
wall-time medians, including fixture/checkpoint verification and all subsequent
validation requests, are below. These old-binary samples are not interleaved
with the ten-sample final-binary groups and lack the new synchronized startup
instrumentation. Use the same-binary OFF/ON comparisons above to attribute a
change to native caching.

| Case | Previous binary | Final binary OFF | Final binary ON |
|---|---:|---:|---:|
| RTX FP32 | 5.772 s | 5.061 s | 5.001 s |
| RTX FP16 | 5.125 s | 4.474 s | 3.720 s |
| Intel FP32 | 8.538 s | 8.636 s | 8.607 s |
| Intel FP16 | 7.450 s | 7.526 s | 7.526 s |

## Implementation and scope

The owner is private to the vendored WGPU compute server. Each pipeline has an
immutable entry; new entries are committed together at read/sync/flush boundaries.
This avoids serializing a growing device-wide blob during every autotune read.
No write or device-cache serialization occurs in steady execution once the
pipelines exist. Database locking and transactions remain CubeCL's responsibility.

A feature-graph audit found that the previous build configured compilation
caching at runtime but omitted `cubecl-server/persistence`. Autotune persistence
was working; generated-SPIR-V persistence was not. The first prototype enabled
both caches. Review removed that unnecessary coupling: the final feature enables
only `cubecl-environment/persistence` and preserves the previous SPIR-V setting.
It caches the driver's pipelines for both generated shaders and handwritten WGSL,
without adding a second change to generated-IR compilation. Final comparisons
toggle only `CUBECL_WGPU_NATIVE_PIPELINE_CACHE=0/1` in one binary.

The feature is opt-in. Vulkan adapters must expose `PIPELINE_CACHE`; other
backends continue ordinary compilation. A warm driver disk cache can already
cover the same work, leaving little additional benefit from explicit persistence.

```sh
cargo build --locked --release --features cli,synthesis,native-pipeline-cache --bin irodori
```

Cache placement follows the existing `--cache-dir` / `IRODORI_TTS_BURN_CACHE_DIR`
selection, under the adapter's CubeCL environment database. No second cache
directory, public cache API, imported executable format or extra dependency
version is introduced. The entry namespace is
`native-pipeline/wgpu-30/v1/<adapter>/<driver-hash>/<build-hash>`; the environment
already separates precision and kernel policy. Kernel identity and the linked
build identifier form the entry key.

## Unsafe boundary and review

The user authorized the single added unsafe call to
`Device::create_pipeline_cache`. Only owned, unchanged bytes produced by
`PipelineCache::get_data` can reach it through the private writer/reader. The
reader requires local origin, matching identity, a versioned envelope, exact
length and an xxh3-128 checksum. The maximum payload is 64 MiB. Imported bundle
entries are skipped and locally regenerated. Corrupt local entries are rejected
before the driver and repaired after successful compilation. A driver error
disables this cache owner for the process and logs the cause; ordinary pipeline
creation still has its existing validation scopes.

The checksum detects accidental damage, not deliberate modification by someone
able to rewrite local application state. This is the persisted-data trust
boundary documented by [WGPU's cache-creation safety contract](https://docs.rs/wgpu/30.0.0/wgpu/struct.Device.html#method.create_pipeline_cache), not a claim that arbitrary downloaded blobs are
safe. The bytes outlive the synchronous creation call. WGPU owns and drops the
resulting handle; no raw pointer, custom allocator, leaked object, asynchronous
borrow or new unsafe public API is added.

Environment generation changes discard pending entries and invalidate the
in-memory pipeline map. Each database handle remains tied to its opened file;
old pending work cannot be published into the newly selected environment.
Concurrent writers use the existing insert-only transaction rules. Cached
artifacts do not become another authority for autotune choices.

Review fixes included caching the build hash once per owner, avoiding one warning
per legitimate imported entry, separating native persistence from generated
SPIR-V persistence, and measuring synchronized loading independently of first inference.
The real-device regression test executes a WGSL kernel after cold creation,
restore, corruption repair, import rejection, disabled caching and environment
switching. Parser tests reject a bit flip at every byte position and every truncation of a
sample envelope, a wrong kernel key, an empty payload, and imported origin.
Identity tests cover changed driver, adapter and build and a non-Vulkan backend.
The complete review findings and fixes are recorded in
[the review note](../review/native-pipeline-cache-2026-10-14.md).

## Measurement method

The artifact directory is
`/home/sanzentyo/benchmark-artifacts/irodori-native-pipeline-cache-20261014`.
The base revision is `6e9a3d271c91ff4bced6b3af0f61696110eef5b9`.
All GPU work is serial, with no concurrent compilation or tests during timings.
The machine uses RTX 5070 Ti Laptop (NVIDIA 595.99.02) and Intel ARL
(Mesa 26.0.8-1ubuntu0.3), Vulkan, Rust 1.95, Burn 0.22, CubeCL 0.11 and WGPU 30.

`scripts/benchmark_native_pipeline_cache.py` runs a frozen release binary in
fresh processes. Each case starts with a private copy of an already tuned cache
and one seed process; measured pairs alternate OFF/ON and ON/OFF. Both modes have
the previous generated-SPIR-V persistence setting. No fresh autotune decisions are accepted.
The same MF fixture generates 50 latent frames / two seconds of audio, with
WGSLPack model weights, FP32 codec and FP32 or FP16 model execution.

The normal condition leaves the driver's existing disk cache enabled. The
controlled condition sets `__GL_SHADER_DISK_CACHE=0` on NVIDIA or
`MESA_SHADER_CACHE_DISABLE=true` on Intel, without deleting the user's cache.
The NVIDIA flag is documented in its [driver README](https://download.nvidia.com/XFree86/Linux-x86_64/595.99.02/README/openglenvvariables.html)
under OpenGL; its effect on this Vulkan workload is established by the measured
pipeline-creation time, not assumed to be portable to every NVIDIA environment.
Mesa documents its setting in [environment variables](https://docs.mesa3d.org/envvars.html#mesa-shader-cache-disable).
This condition isolates a miss in the driver's implicit disk cache; it is not
a cold filesystem/page-cache or a GPU reboot experiment.

`ready_seconds` includes adapter/cache setup, synchronized model and codec
loading, and request preparation. It starts after fixture parsing and checkpoint
SHA verification; those are included only in external process wall time.
`first_request_seconds` is the first sampler plus decoder, through CPU PCM
readback. `first_audio_seconds` is their sum and means complete first audio,
not the first streamed chunk. Three later requests form each process's steady
median. Streaming/full and trimmed/full equality are also checked afterward.
Pipeline telemetry covers all validation pipelines, including these later
streaming checks; it is not the same interval as first audio.

`restored` counts admitted blobs supplied to WGPU. It is not a driver-reported
cache-hit count: WGPU/driver validation can discard compatible-looking blobs.
Pipeline creation, restore/save time, database bytes and process peak RSS are
reported independently. p95 uses linear interpolation on fresh-process samples.

## Reproduction

Build and copy the binary before starting measurements:

```sh
cargo build --locked --release \
  --features cli,profile,synthesis,packed-export,native-pipeline-cache \
  --bin validate_synthesis --bin irodori
uv run scripts/benchmark_native_pipeline_cache.py SPEC.json NEW_OUTPUT_DIR --pairs 10
```

The spec names the frozen binary and, for each case, a complete
`validate_synthesis` command, a closed local seed cache, a frozen reference output
directory, and the driver-cache environment settings. The script records exact
commands, binary SHA, per-process reports, output hashes and summaries, and
refuses changed audio, missing native restore telemetry or new autotune choices.

The vendored dependency is not a workspace member, so Cargo will not select its
dev dependencies with `cargo nextest -p cubecl-wgpu`. Its private tests are run
from an isolated source copy, with the root lockfile as the version seed and an
empty `[workspace]` appended to that copy's manifest. The copied source is
compared against the production vendored tree. The retained resolved lockfile
resolves upstream optional/dev entries while retaining shared production versions.
Run its three native tests with
`--lib --no-default-features --features std,spirv,native-pipeline-cache
-E 'test(native_)' --run-ignored all --test-threads 1 --no-tests fail`.
`CUBECL_NATIVE_CACHE_TEST_DEVICE=integrated` selects the Intel roundtrip test;
`discrete` selects RTX. There are no invented payloads passed to a driver.

## Verification

- `cargo nextest run --locked -p irodori-tts-burn --lib --test '*' --features
  cli,profile,worker,http-server,lora,packed-export,native-pipeline-cache
  --run-ignored all --test-threads 1 --no-tests fail`: 686 passed, zero skipped.
  Required tokenizer and packed-model fixtures are supplied. After the final
  equivalent unit-struct constructor cleanup, the six graph tests and native
  owner tests pass again; the broad result is reused.
- Native owner: three tests pass on RTX, with the real-device lifecycle test
  also passing on Intel.
- Root workspace/all-target/all-feature Clippy and the vendored owner's Clippy
  pass with `-D warnings`; root and modified vendored Rust formatting pass.
- Core no-default-feature, minimal native-feature, and vendored WGSL-only
  native-feature builds pass. The native feature adds no package versions;
  `Cargo.lock` is unchanged and the default build keeps native caching disabled.
- All 188 final model-validation processes pass without a retry or inference
  failure. Frozen binary and source hashes are retained in the JSON evidence;
  source hashes were compared after measurement and the isolated vendored test
  source matches production.

Commands, fixture locations and the final Clippy rerun after fixing dependency
lint findings are retained in the evidence. Only Linux Vulkan was executed;
Windows, Android and macOS Vulkan were not tested. Miri cannot exercise this
native driver boundary; envelope rejection is tested separately without calling
the driver. No whole-model speedup over PyTorch or speech.cpp is claimed from a
startup-cache change, and this experiment does not rerun their benchmarks.
