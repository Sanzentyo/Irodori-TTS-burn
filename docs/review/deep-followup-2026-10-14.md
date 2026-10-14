# Review: measured inference follow-up

Scope: the change from `de76403` covering original-file identity during loading,
projection retuning, condition-refresh capture, numerical diagnostics and FP16
RoPE fusion.
The pre-existing user edit in the August 30 benchmark note is excluded.

## Findings and resolutions

1. **Persisted voice identity must remain the actual archive identity.**
   Packed metadata's declared source SHA is not the SHA previously stored in
   model-bound voices. The loader hashes the complete original archive, including
   gaps and trailer; the packed converter and voice schema are unchanged. Codec
   identity uses original decoder/quantizer bytes and the existing sorted layout
   encoding, independent of the GPU materialization precision.
2. **Raw hashing did not show a reliable improvement.** The interleaved raw-model
   reader was removed after measurement. Keep the simpler sequential raw hash;
   only packed model payload reads and already loaded raw codec bytes are reused.
3. **Retuning must not invalidate unrelated decisions or reward compilation.**
   CubeCL exports a new isolated snapshot, omitting only exact projection
   namespaces. The campaign seeds every case before measuring and freezes
   logical tuning values rather than physical database file bytes. Stable output
   hashes, finite positive durations and strict reference errors are required.
   Original caches remain untouched, and rejected candidates retain evidence.
4. **Changing conditions must also refresh context KV.** KV construction stays
   inside the capture. Persistent condition allocations outlive the graph;
   exact masks and all source shapes/dtypes/devices/layouts are checked before
   copying. Copies/replay/readback use the captured stream. No recapture fallback
   is added. The synthetic regression covers both auxiliary parts, cross-stream
   producers, switching back, changed masks and missing auxiliary parts.
5. **Boolean tensors use backend storage representation.** Initial trace/graph
   validation tried to read `Bool(U32)` directly as Rust `bool`. Explicit boolean
   conversion fixes both paths; the failed attempts are retained rather than
   counted as successful inference.
6. **Diagnostics must observe production execution.** Generic and short-sequence
   WGSL encoder routes share an observer with an inline no-op production instance.
   The recorder immediately copies each boundary to CPU instead of retaining GPU
   intermediates. Both Rust and official Python traces verify their final output
   against the ordinary unobserved encoder. Diagnostic methods require both
   `profile` and `inference`, so core-only profiling does not pull in unused
   inference tracing. XPU math SDPA is an explicit option after auto SDPA failed.
7. **Microbenchmark gains did not carry through to synthesis.** Half-storage MLP
   tiles could beat a slower corresponding K32 kernel, but not the already chosen
   routes. Intel full RF was 0.7% slower. The extra shader variants, API and binary
   target were removed after comparison.
8. **First-layer precision is insufficient for acceptance.** Reproducing the
   official FP16 frequency/product rounding made first-layer RoPE exact but
   worsened several whole-audio cases. This mathematical change was rejected.
9. **Lint and feature review.** Removed an unnecessary decoder-name borrow and a
   redundant result wrapper in validation. The new diagnostics are gated on
   their actual inference consumer. No new dependency, unsafe operation, manual
   lifetime extension or intentional leak is introduced.
10. **A stable cache digest does not prove successful persistence.** Review of
    `retune-rtx-v2` found runtime `database is readonly` warnings and fresh tuning
    records in every candidate process. Direct placement of exported SQLite
    bundles left an unchanged, incomplete cache. Its apparent 14% RF gain is
    excluded from accepted evidence. The tool now imports each bundle through
    CubeCL into a fresh writable runtime environment and checks imported counts
    and destination. Measured processes must produce an existing, empty tuning
    record as well as unchanged logical decisions. A regression rejects both
    missing records and fresh decisions after seeding.
    A second attempt exposed Python SQLite's inability to inspect the Turso
    runtime file. Inspection now also uses CubeCL's own database API, validating
    the scanned count and hashing sorted logical rows. A regression verifies
    insertion-order independence, excluded compilation records, changed values
    and rejection of a missing environment.
11. **FP16 fusion must preserve intermediate rounding and subnormals.**
    `quantizeToF16` flushed small products on the measured RTX path. Explicit
    conversion through `f16` preserved RTX output but was eliminated on Intel,
    producing a single-rounded result. The final kernel materializes two native
    half products in workgroup memory, synchronizes all lanes, and sums values
    read by the neighboring lane. Tail lanes participate in the barrier. This
    retains the original arithmetic on both adapters without a vendor fallback.
    The strided-QKV regression includes subnormals; ten full synthesis cases
    preserve the original latent and PCM bytes.

## Ownership and validation boundaries

`CapturedConditionedSampling` mutably borrows its output guard and retains the
prepared request and model lifetime. It extends the existing graph owner's
refresh hook without broadening its unsafe replay boundary. Mask changes fail
before replay; immutable topology is not guessed from a hash.

The change is owned by `irodori-tts-burn`. Cargo metadata and reverse-dependency
inspection find no downstream workspace Rust consumer of this package; the
independent `wgslpack` package is unchanged. The relevant external consumers are
the CLI/worker, Python comparison scripts, saved voices and CubeCL cache files.
Whole-archive hashing tests cover ordered one-pass reads, overlaps, repeated
reads, gaps, trailers, overflow and truncation. Codec tests preserve the old
identity before and after dtype policy changes. Retune policy tests reject
incomplete sessions, small gains, nonfinite errors/durations and unstable output.

The benchmark report records actual-device evidence, retained failures and the
final feature/test commands. Miri/Loom are not used: the new code adds no unsafe,
manual synchronization primitive or asynchronous task owner, and these tools
cannot execute the Vulkan graph path. GPU tests are serialized with the shared
OS lock, including tests spawned as separate nextest processes.

Final verification: 35 selected release nextest tests, five Python policy tests,
all-features/all-targets Clippy with warnings denied, formatting and the documented
feature checks pass. The accepted retune has zero fresh decisions in all twelve
measured processes. Four actual CLI WAV comparisons and four real reference-voice
condition-refresh comparisons are exact on RTX/Intel × FP32/FP16. Raw and packed
saved-voice identities remain compatible; the wrong-model rejection also passes.
See the [measurement report](../benchmarks/deep-followup-2026-10-14.md) and its
linked evidence for the final numbers and limitations.
