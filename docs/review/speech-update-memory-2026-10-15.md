# Review: speech.cpp update and decoder context

Scope: the port from `dbdc448`, shared ordinary/captured decoder window planning,
the validator's graph bound, regression coverage and matched upstream/runtime
measurements. The pre-existing user edit in the August 30 benchmark report is
excluded from this change. This is a self-review, not an independent-agent review.

## Findings and resolution

1. **The margin needs a single owner.** Derive it from the decoder's validated
   convolution metadata, rather than replacing a literal ten with a duplicated
   literal eight. Both streaming implementations consume the same plan and the
   validation tool computes its graph limit from that margin. `context_frames`
   is an additive public query for callers sizing their graph capacity; internal
   decoder topology remains private. The supported loader constructs the
   centered, even-stride geometry used in the calculation.
2. **Comparing two windowed implementations can hide the same boundary bug.**
   The GPU regression now compares against whole decoding, including 112-frame
   interiors, a one-frame first chunk and a non-frame-aligned trimmed end. An
   independent CPU index inversion covers every sample phase in the 1920-sample
   hop and demonstrates the necessity of the eighth context frame. The tests
   check output bits rather than mirroring the new radius formula.
3. **A shape-specific API is not automatically selected by that shape.** The
   initial test comment called 112 frames a specialized upsampler route. Review
   found that this test uses ordinary `decode_wgsl`, not the separate
   `Fixed112DacVaeDecoder` ownership API. The comment and report now describe
   the exercised geometry accurately; no new route is claimed or introduced.
4. **Whole-model precision and codec parity are distinct.** Fixed-noise upstream
   comparisons establish unchanged latents and tiny codec rounding changes;
   existing FP16 error against the FP32 oracle is explicitly retained. Matching
   each runtime's seeded PCM cannot establish cross-runtime accuracy, so raw
   fixed-noise fixtures are reported separately from worker timing.
5. **Memory must belong to the measured adapter and process.** An initial run
   selected Intel after Vulkan device ordering changed. The accepted speech
   run pins the NVIDIA ICD and checks the adapter name. NVML uses the exact PID
   and takes the maximum across compute/graphics appearances, not their sum.
   Polling limitations, allocator retention, startup peaks and Intel's shared
   memory are kept distinct. Normal `/proc` disappearance at process exit is
   handled; other monitoring exceptions fail the run.
6. **Interrupted measurements must not silently become complete results.**
   Resume keeps only complete old/new pairs. The partial pair is preserved but
   excluded, then rerun. Initial invalid attempts remain in the artifact record.
   The port benchmark primes each workload before its alternating measured
   processes, using isolated copies of the same tuning seed and no competing
   GPU work or heavy CPU compilation.
7. **Less arithmetic does not prove a latency or memory win everywhere.**
   The report retains the initial 2.8% short RF FP16 regression, the nearly
   neutral long Intel result and higher Intel host RSS. A focused common-cache
   follow-up checks the short RF case before adoption; it does not replace or
   hide the first cohort. The initial RTX MF long priming process peaks 73 MiB
   higher; the warm memory saving requires a subsequent process using the
   prepared cache. This limit and Intel's initial preparation delay are explicit.
   Weight residency is not claimed to shrink.

## Ownership and verification

The changed package is `irodori-tts-burn`. Full Cargo metadata and inverse-tree
inspection find no downstream workspace Rust consumer. CLI/worker synthesis and
the validator are the external boundaries exercised here. Weight formats,
voice identity, dependency features and persisted cache schemas are unchanged.
The implementation adds no unsafe operations, manual synchronization, leaks,
fallback chain or secondary cache authority.

Five selected release nextest tests pass. The real-codec integration test passes
on RTX and Intel; it also retains changed-input, bounded eviction, cancellation
and environment-generation checks. The feature matrix includes default library,
codec without defaults, codec plus streaming graphs without defaults, and all
features/all targets for Clippy. Filters were listed first and runs require
`--no-tests fail`. Minimal-feature dead-code warnings predate this change.

No Miri/Loom run is needed for this geometry-only change: it adds neither an
unsafe boundary nor a concurrent owner, and the device behavior is tested on
Vulkan. The [benchmark report](../benchmarks/speech-update-memory-2026-10-15.md)
records final performance, numerical evidence and retained failures.

Final acceptance: all twelve worker cases preserve PCM bytes, and six real-input
checks match whole decoding at both streaming window sizes and the trimmed tail.
The common-cache RF follow-up has 139 identical logical tuning entries before
and after every process and improves 1.4%; the initial regression remains in the
record. Adopt the derived context, with the observed warm memory/runtime gains
and explicitly documented first-use preparation cost. Formatting and
all-targets Clippy pass after the review comment fix. No actionable finding
remains in this change's scope.
