# Review: pointwise WGSL fusion

Scope: ModernBERT GELU/gating, residual normalization, cross-layer AdaLN gates,
diagnostic observation and runtime metadata. This is a self-review. The existing
user edit in the August 30 benchmark report is outside this change.

## Findings addressed during implementation

1. **Algebraic equivalence does not preserve stored FP16 arithmetic.** The first
   full GELU polynomial matched RTX but differed on Intel. Keep division/erf in
   Burn and fuse only the remaining pointwise operations. Shared-memory exchanges
   preserve the original storage-rounding boundaries; both precisions are checked
   bit for bit, including small values and odd tails.
2. **A fixed LayerNorm reduction changes the model.** Extending the existing
   three-token FP32 reduction tree to other shapes/FP16 introduced differences
   that amplified through synthesis. The refined candidate leaves shape/device
   reduction selection with Burn and fuses the normalization epilogue. The
   previously accepted three-token FP32 path retains its original arithmetic.
3. **An unselected kernel cannot establish accuracy.** Caption/speaker requests
   previously bypassed the specialized text route. The first long-input results
   therefore did not validate the normalization extension. Explicit routing and
   diagnostic captures exercise the intended path; those initial results remain
   in the experiment record without an accuracy claim for that extension.
4. **Diagnostic observation must record the real computation.** The GELU kernel
   optionally emits its actual pre-gate activation. The unobserved specialization
   omits that allocation/binding. Tests compare the observed and unobserved output
   and compare the recorded activation against the original graph.
5. **Gate activation needs one consistent owner.** Packed modulation now stores
   activated gates. The ordinary producer applies tanh before constructing the
   same internal type, and consumers do not apply it again. Existing stale-input,
   batch, schedule packing and module-order checks remain applicable. This is an
   in-memory inference contract, not a serialized weight/cache format change.
6. **Text length should not define a new pipeline.** Element counts and dispatch
   geometry use CubeCL's immutable metadata uniforms. Pipeline identity contains
   storage precision, feature width and observation mode, not sequence length or
   CFG batch. Tests revisit a previous shape after changing these metadata values.
7. **Timing must exclude compilation and competing work.** Speed cohorts prime
   each binary and then rotate fresh processes against identical logical cache snapshots.
   Every timed process must reproduce frozen audio/latents exactly and must not
   add tuning decisions. Shared GPU locking and immediate peer notifications
   isolate GPU execution; heavy CPU builds are also excluded from speed windows.
8. **A database copy is not a process-state snapshot.** The first Intel speed
   cohort stopped in the unchanged baseline with Turso's `reader slot released
   by non-owner` assertion. Its failed run remains in `speed-intel-v5`; no timing
   from it is accepted. The benchmark can now use CubeCL export/import to create
   a separate writable environment for every process. It verifies imported entry
   counts and logical autotune hashes, requires a present, empty decision record
   during timing, and checks that decisions remain unchanged afterward. This
   changes the measurement setup; it does not claim to fix the upstream assertion.
   The same baseline assertion recurred in `speed-intel-v6` despite fresh imports;
   that incomplete cohort is also retained and excluded. The next comparison
   disables both persistent compilation caches explicitly while retaining the
   frozen tuning decisions. Compilation occurs during warmup, outside the steady
   inference interval; normal application defaults remain unchanged.
9. **Preflight must check the epilogue's actual grid.** The extended normalization
   path dispatches three 256-element workgroups per 768-element row. Its old
   resource preflight checked only one group per row. The final guard validates
   both dispatch dimensions and padded-index overflow. A boundary test rejects a
   two-row grid height when three are needed, accepts height three, and rejects
   empty/overflowing input counts. This changes admission, not arithmetic.
10. **The original reduction choice also depends on conditioning.** Before this
    change, a three-token FP32 request with a caption or speaker used the generic
    text encoder. Extending frontend routing accidentally selected the old fixed
    reduction tree for it. A new short-caption fixture reproduced different
    latent/audio bytes on both RTX and Intel with `v7`. The final `v8` carries a
    typed reduction choice from the condition frontend through normalization and
    diagnostics: text-only requests retain the legacy short path, while requests
    with auxiliary input preserve Burn's reductions at every length. Both devices
    now reproduce the unchanged baseline exactly. This is one explicit numerical
    contract, not another device-specific fallback. The affected owner tests pass
    (12 RTX, one additional Intel test); default/minimal builds and all-target,
    all-feature Clippy also pass after the correction.

## Verification boundary

The changed owner is `irodori-tts-burn`; resolved Cargo metadata and its inverse
dependency graph identify no downstream workspace consumer. Synthesis, diagnostic
capture and graph replay are the relevant runtime boundaries. No dependency,
feature, public API, unsafe operation or independent cache authority is added.

The final `v8` source passes the affected owner selection (12 RTX tests and one
additional Intel test), default/minimal feature checks and all-features,
all-targets Clippy with warnings denied. The minimal feature check emits 25
existing dead-code warnings. The wider numerical owner runs (44 RTX/eight Intel),
16 primary and 12 additional synthesis captures and four graph-reuse cases are
reused with their source boundaries documented in the
[benchmark report](../benchmarks/fusion-followup-2026-10-15.md). The new
short-caption regression passes on both devices and retains the failing `v7`
negative control. No new unsafe, concurrency or cancellation mechanism requires
Miri/Loom coverage.

All primary speed cohorts and the four-pair confirmation completed successfully.
Every timed process reproduced the frozen latent/audio and added zero tuning
decisions. The two incomplete Turso cohorts are excluded rather than pooled with
successful timing. The review retains all three refined epilogues: RTX MF FP16
improves by 3.86%/1.45% on short/long input, while FP32 and Intel benefits are not
established. The two follow-up medians are +0.32%/+0.10%, with mixed paired signs;
small regressions remain possible. This limitation is explicit, and no
device-specific routing exception is introduced to chase those small differences.

The committed implementation is the measured `v8` source, including the typed
conditioning correction. Source/kernel/binary hashes were checked against its
frozen build after measurement. Final formatting and whitespace checks pass. No
known correctness finding from this review remains open; the existing Turso
persistent-compilation assertion and broader oracle differences remain outside
the repair and performance claims.
