# Review: optimization preparation, October 14, 2026

This records the earlier CPU-only review. Subsequent device measurements are
reported in the [acceptance campaign](../benchmarks/optimization-acceptance-2026-10-14.md);
the pending items below are historical, not a claim about that later campaign.

This is a manual review of the implementation against base `129c407`, followed
by targeted CPU validation. It is not an independent-agent review or a device
performance/precision approval. The requested scope is preparation without new
measurements; the [implementation status](../planning/optimization-preparation-2026-10-14.md)
lists the remaining acceptance work for all five candidates.

## Scope and findings

- **Feature boundaries:** SPIR-V persistence is independent of native driver
  persistence. New graph code requires `streaming-graphs` and explicit activation;
  codec export requires `codec,packed-export`, while reading needs only
  `codec,packed-weights`. Export-only layout descriptions were gated so reader
  builds do not gain unused conversion code/warnings. The existing duration
  benchmark example lacked required `cli,inference` features; default
  `--all-targets` exposed unresolved optional imports. The missing target gates
  were added, and the same default check now passes.
- **Cache ownership:** the streaming owner holds one LRU collection and immutable
  weights, drops graphs first, refreshes input on replay and exposes owned PCM.
  The key includes crop offset and length as well as input geometry. Environment
  changes invalidate all entries; eviction happens before capture. Requests that
  exceed frame bounds fail before emission. No retry or fallback hides failures.
- **Ordinary streaming:** shared planning preserves the ten-frame context,
  reflection-padding minimum and sample trimming. Shape/overflow checks happen
  before decoding; the consumer checks output length and finite PCM before the
  sink. CPU tests cover whole/partial frames, multiple window sizes and coverage.
  GPU equality remains pending.
- **Packed codec:** one validated schema drives conversion and reading; existing
  topology constructors and grouped upload code are reused. Original and prepared
  tensors stay FP32. Source shape/dtype checks precede permutation, and schema
  checks precede device allocation. CPU permutation tests use an independent
  scatter formulation for the shader layouts, including exact bit patterns.
  Publication is atomic and does not overwrite files. The shared decoder digest
  preserves the original voice contract; declared hashes do not authenticate
  archive payloads. Encoder initialization from a decoder archive is rejected.
- **Diagnostic evidence:** profiling uses the production fused codec boundary,
  not standalone blocks relabeled as the production graph. The RF scope clears
  pending state on errors/unwind. Stage manifests bind codec/fixture identity,
  order, shapes and payload checksums; traced latents must equal ordinary output.
  The offline tool permits differing model/state precision but rejects differing
  workload contracts. Its first differing stage is an observation, not a causal
  diagnosis or an approved precision tolerance.
- **Known unverified behavior:** the historical Intel Turso assertion remains
  unresolved. Neither the new cache switch nor successful CPU checks establish
  generated-SPIR-V persistence reliability. Graph reuse and prepared codec upload
  still require both RTX and Intel parity/memory/performance runs. No default
  route, tile size or arithmetic was promoted on the strength of static review.

No new unsafe block, dependency version, format-crate change or CI change was
introduced. The unrelated allocator notes in
`docs/benchmarks/rtx-5070ti-v4-rf-gap-followup-2026-08-30.md` were preserved exactly.

## Validation performed

| Check | Result |
|---|---|
| Selected root library CPU unit tests | 16 passed |
| Real FP32 codec conversion on CPU, original payload bytes and voice identity | 1 passed; no GPU/inference |
| Vendored SPIR-V switch parser | 1 passed |
| Offline stage-comparison tests | 5 passed |
| Root Clippy, all targets/all features, warnings denied | Passed |
| Vendored owner Clippy with both optional caches, warnings denied | Passed |
| Default root build, all targets | Passed after fixing the example's required features |
| No-default core; codec + packed reader; SPIR-V only; streaming graphs only | Passed |
| Vendored WGSL-only configuration | Passed |
| Rust formatting, Python Ruff checks/formatting, diff whitespace | Passed |
| New GPU acceptance integration test | Compiled only; not executed |
| GPU performance, model accuracy, shader/driver cache lifecycle campaigns | Not run, per request |

The minimal root builds retain 25 existing dead-code warnings from optional
inference internals. The same warnings occur in the recorded base core check;
all-feature Clippy is clean. Intermediate failures were the CLI graph activation
being inserted before engine construction, export-only fields present in reader
builds, and the example feature gate described above. Each was corrected and its
affected compilation repeated. Initial Python import/format findings were fixed.

The real checkpoint CPU test converts the existing 320,048,128-byte
`dacvae-decoder-only.safetensors` fixture, parses the complete new archive,
compares every original tensor payload byte for byte, checks the original file
digest, and checks voice identity across formats. It does not test uploaded
weights or generated audio. Separate synthetic tests verify each permutation.

## Reproduction and impact selection

Resolved Cargo metadata and `cargo tree --workspace --all-features --invert
cubecl-wgpu` identify CubeCL → CubeK/Burn → the root crate as the affected
consumer closure. The standalone WGSLPack crate is unchanged and is not a
reverse dependency of WGPU. Default resolved features and all 702 package IDs
match the pre-change graph; `Cargo.lock` is unchanged. CLI, worker and HTTP
consumers are compiled by the all-feature/all-target check.

These are the CPU test selectors; `cargo nextest list` was inspected before each
new selection. The root's many GPU tests were explicitly excluded.

```sh
export CARGO_BUILD_JOBS=2 CARGO_PROFILE_TEST_DEBUG=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0

cargo nextest run --locked --lib \
  --features synthesis,profile,streaming-graphs,packed-export,native-pipeline-cache,spirv-cache \
  -E 'test(codec::packed::) | test(codec::streaming::tests::) | test(codec::streaming_graph::tests::) | test(model::profiling::tests::diagnostic_scope_) | test(synthesis::voice::tests::) | test(codec::weights::tests::official_codec_metadata_) | test(codec::weights::tests::decoder_only_profile_) | test(codec::weights::tests::graph_changing_codec_metadata_)' \
  --no-tests fail

IRODORI_CODEC_TEST_CHECKPOINT=/path/to/dacvae-decoder-only.safetensors \
  cargo nextest run --locked --lib \
  --features synthesis,profile,streaming-graphs,packed-export,native-pipeline-cache,spirv-cache \
  -E 'test(codec_pack_fixture_preserves_payloads_and_voice_identity)' \
  --run-ignored only --no-tests fail

uv run scripts/test_compare_synthesis_stages.py
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo check --locked --all-targets
cargo check --locked --no-default-features
cargo check --locked --no-default-features --features codec,packed-weights
cargo check --locked --no-default-features --features spirv-cache
cargo check --locked --no-default-features --features streaming-graphs
cargo fmt --all --check
uvx ruff check --select E,F,I scripts/compare_synthesis_stages.py scripts/test_compare_synthesis_stages.py
uvx ruff format --check scripts/compare_synthesis_stages.py scripts/test_compare_synthesis_stages.py
```

The vendor's dev targets require an isolated workspace copy. Its 50 Rust source
files were verified identical to production, with only an empty `[workspace]`
added to its manifest and a separate lockfile. The CPU-only selector was
`test(spirv_cache_switch_rejects_ambiguous_values)`, with
`--no-default-features --features std,spirv,persistence,native-pipeline-cache`.
No vendor GPU test was run. Debug information and incremental compilation were
disabled for test/dev disk usage, preserving the previous validation convention;
the release profile was not changed.

Local logs, metadata, the copied vendor workspace and the preserved user diff
are under
`/home/sanzentyo/benchmark-artifacts/irodori-unmeasured-preparation-20261014/`.
No latency or accuracy result from this preparation should be interpreted as a
device optimization result; there are no new device measurements here.
