# Review of measured optimization candidates

Manual review against `129c407`, continuing the earlier
[preparation review](optimization-preparation-2026-10-14.md). This is not an
independent-agent review. The [measurement report](../benchmarks/optimization-acceptance-2026-10-14.md)
separates accepted behavior, rejected candidates and remaining precision gaps.

## Findings and resolutions

1. **Incorrect text-condition diagnostic boundary.** The old `--dump-stages`
   path independently encoded unconditioned text instead of exporting the
   condition actually consumed by the sampler. Compaction and CFG can change
   both its content and shape. Export the first traced forward's exact condition,
   identify that boundary explicitly, advance the manifest to schema 2 and
   reject obsolete captures. The sixteen corrected device captures preserve
   the frozen latent/PCM exactly; five offline comparison tests pass.
2. **Incomplete RF profile.** Autotuned SwiGLU expansion bypassed the substage
   scope, omitting the largest measured Intel component. Conditioning was also
   missing. Both now use existing profiling scopes. Ordinary arithmetic is
   unchanged, as confirmed by all sixteen corrected captures. Production codec
   profiling retains its actual cross-block fusion boundary.
3. **Misleading first-audio metric name.** The fresh-process harness initially
   called readiness plus a complete first PCM request `first_audio_seconds`.
   Rename that sum to `ready_plus_first_request_seconds`. Existing archived
   startup results are documented with their original semantics; actual stream
   TTFA is recorded separately from the first sink callback.
4. **GPU contention invalidates timing claims.** Another authorized task used
   CUDA during the rotary-arithmetic experiment. Exclude those timings; retain
   numerical comparisons. Both tasks now notify immediately with `turn/steer`
   and hold the same OS lock across GPU execution. Preserve the aggregate
   monitor's rejection and the exact accepted interval for the paused/resumed
   MLP campaign instead of relabeling the entire monitor log as successful.
5. **Mixed numerical/performance outcomes.** FP32 rotary intermediates improved
   some FP16 fixtures and degraded others. Separate MLP gate/value projections
   slowed the isolated Intel workloads. Neither experiment remains in production
   source. Do not add shape/device precision fallbacks to conceal those outcomes.
6. **Attribution of a tuning improvement.** The 64-column tile passed all 32
   local accuracy checks and won none. Its experimental schema invalidated
   existing selections, confounding overall latency. An unchanged-binary control
   reproduced the RTX FP32 RF gain by re-measuring only existing candidates in
   a private cache. Reject the added tile and schema change; preserve the
   successful re-tuned cache and report the changed FP32 rounding explicitly.
7. **Stale allocator documentation.** The vendor note still described
   ExclusivePages as the production default. Both current CLI initialization
   and the validator use upstream's default, which resolves to Adaptive in
   this native build. Correct the vendor note against CubeCL 0.11's policy;
   historical measurements with explicit allocators retain their original scope.
8. **Duplicated work on a decoder graph miss.** Capture primed the decoder,
   discarded its completed output, then replayed the same input for the first
   request. Read the priming result into owned CPU memory before recording and
   return it only after capture succeeds. The priming tensor is released before
   recording so the graph allocator can reuse it; captured output is still read
   only after a replay. Existing graph-only callers avoid this extra readback.
   A focused owner test covers readback failure cleanup, changed inputs across
   streams and preservation of the first owned result. All five owner tests and
   real-codec eviction/generation comparisons on both devices pass. The separate
   interleaved binary comparison records the latency effect.

## Contract review

- Cache layers are independently gated. The SPIR-V policy uses the upstream
  compilation store and environment invalidation. Invalid switch values disable
  only that store with a diagnostic; there is no second database owner or
  recovery path that hides store failures. Features remain opt-in.
- Graph state has a single mutable owner, immutable weights, LRU eviction before
  capture, keys including exact crop, and owned PCM leaving replay. Frame limits
  are checked before any sink emission. Environment-generation changes clear
  captured bindings. Capture/replay errors propagate. These geometric limits
  are not a hard byte budget or a claim about peak VRAM.
- Decoder topology validation is shared by storage adapters. Packed input is
  completely schema/shape/dtype checked before upload. Conversion permutes raw
  FP32 words, publishes without overwriting, and preserves the existing decoder
  identity for saved voices. The declared source digest does not authenticate
  payloads. Full codec/encoder construction from a decoder archive is rejected.
- CLI entry points delegate to library APIs. Export, reading, profiling and
  graph reuse keep their separate feature boundaries; transport dependencies do
  not become mandatory. WGSLPack's format crate and `Cargo.lock` are unchanged.
- No new unsafe code, deliberate leak or runtime precision policy was added.
  The unrelated allocator-note diff is excluded from this change.

## Verification record

The original CPU owner tests, conversion fixture, vendor switch-parser test and
feature matrix can be reused where their implementation and features are
unchanged. Their source hashes, commands and results are retained in the
preparation artifact directory. New/changed diagnostic code is compiled by the
all-feature/all-target Clippy run and exercised by the corrected device captures.

The resolved reverse-dependency closure is CubeCL WGPU → CubeCL/CubeK/Burn → the
root crate. The root CLI, worker and HTTP targets are included in Clippy. No
format-crate code or dependency graph is changed. `cargo nextest list` was
inspected for the two explicit ignored GPU integration targets before execution;
GPU invocations use one test thread and the agreed cross-process lock.

Completed final checks:

- `cargo fmt --all --check` and
  `cargo clippy --locked --all-features --all-targets -- -D warnings`: pass.
- Rebuilt release `irodori`, `validate_synthesis` and `pack_codec`, with hashes
  recorded under `final/binary-sha256.txt` in the measurement artifact directory.
- Five graph-owner tests selected with `test(/^graph::tests::/)`: pass,
  including the new priming-output behavior and existing cross-thread/error tests.
  One explicit ignored codec test selected for each device: both pass. Lists
  are checked against expected test names before execution; selecting zero tests
  is an error.
- No-default-feature `codec,packed-weights` and `streaming-graphs` library checks:
  pass. Their 25 unused-code warnings match the preparation checks exactly.
  The earlier unchanged core, SPIR-V-only and vendor WGSL checks are reused.
- Shared-lock combined startup confirmation: 48 processes, including 36 timed;
  initial streaming campaign: 72 processes, including 54 timed. All outputs
  match and no timed process makes a fresh tuning decision.
- Final interleaved priming comparison: 30 processes, including 20 timed, all
  exact with zero new timed tuning decisions. Cold streaming completion improves
  18.9–31.9%; the report retains warm-latency regressions as well as gains.
  All final GPU work finished successfully and released the common lock at
  21:47:30 JST, with immediate peer notification and zero monitor conflicts.
- Final concurrent cache acceptance: 16 processes, all exact, correctness only.
  Final CLI acceptance: 12 processes, two warmups and five requests each, all
  96,000-sample WAV files exact across ordinary/packed/graph variants for each
  device/precision pair. This single fixed-order CLI pass is a consumer regression
  check, not a separate claim of packed-format resident speedup.
- Five diagnostic-comparison Python tests passed after the schema-2 correction;
  Ruff lint and formatting checks pass for all three scripts.

The historical Intel Turso assertion and long-MF cross-backend precision gaps
are not considered fixed by these checks or by startup-cache speedups. No claim
of a fresh PyTorch/speech.cpp speed comparison or complete GPU saturation is made.
