# Inference follow-up review

Self-review against `5a4adcb1d5ef3112bbfa459a3a806452cf510de5`, before committing
to `master`. No independent reviewer or sub-agent was used. The existing user
edit to the August RF benchmark report is outside this change.

## Findings addressed

1. **Graph input dtype depended on the device default.** A fixed graph with
   FP32 sampler state and an FP16 model must allocate its stable input using the
   example's explicit dtype. The shared owner now does that. Shape, dtype and
   device admission precede the input copy. The graph output remains borrowed
   until owned CPU readback, and model/request lifetimes retain its constants.
2. **A fallible capture could leave recording enabled.** Capture now stops
   recording before propagating a model error. Regression tests inject a
   capture failure and then capture again; another checks rejected replay input
   followed by a valid request. The existing unsafe replay call is moved to the
   common owner, with the same lifetime/serialization contract.
3. **Host writes inside a graph were not replayed.** Fixed timestep scalar
   creation and attention masks now use device operations. RoPE and encoded
   conditions are prepared and retained outside capture. End-to-end comparison
   uses two noises so a graph retaining its original input cannot pass.
4. **Mixed precision must preserve the neural forward contract.** Model inputs
   are cast at the forward boundary, and velocities return to state precision
   before guidance/integration. A constant-velocity FP16 model exercises Euler,
   Heun, PLMS-4 and MeanFlow against their closed-form FP32 solution. The default
   remains model precision because MF did not consistently improve.
5. **A decoder prototype silently measured the old route.** Blocks 1–3 require
   packed weights before selecting a polyphase candidate. The corrected tuner
   admits only exact released geometry, checks every candidate numerically and
   retains the portable/cached choice in the same search. Invalid packed shapes
   remain rejected; the route test now also asserts the valid tuned case.
6. **Recorder files omit warm cache hits.** Bundle approval binds actual saved
   entries and owns the bytes between verification and import. It rejects
   mismatched identity/settings, tampered payloads, conflicting local choices and
   additional namespaces. Post-workload verification detects unseen choices.
   A separate executable digest prevents accepting a different application
   build merely because its Burn/CubeCL versions are unchanged; the CLI hashes
   the supplied consuming binary before any import.
   Restoration does not claim to freeze future tuning or provide global cache
   transactions; partial approved imports on I/O failure are retryable.
7. **Packed provenance is not payload authentication.** The source digest
   reader validates the bounded index/metadata but documents its declared-source
   scope. Validation records the actual archive digest separately and compares
   generated latent/audio values. Full parameter hashes are checked separately.
8. **Shape-only encoder tests exited with pending device work.** One failed
   with a native teardown SIGSEGV. The three small encoder shape tests now wait
   for device completion before assertions/process exit. Numerical tolerances
   and failure expectations are unchanged.
9. **Graph input copies counted allocator padding as logical data.** A new
   six-element regression exposed a truncated vec4 tail. The copy now derives
   its dispatch from the checked logical byte length, and rejects non-16-byte
   input lengths as its existing contract requires. Allocator padding cannot
   make an unsupported shape appear valid. The positive replay test uses an
   admitted shape, while the original six-element shape explicitly rejects.
10. **A moved session could refresh on a different logical stream.** CubeCL
    replay remains on its capture stream, whereas ordinary tensor operations
    follow the calling thread/scoped stream. Mutable borrowing alone does not
    establish their order. The common owner now retains the capture stream and
    uses the safe scoped-stream API for input copies and guarded readback. The
    input copy uses the stable destination's client. A regression exercises
    distinct capture, producer, enqueue and readback streams, followed by moving
    the graph and output guard to different threads. No new unsafe call is added.
11. **Priming failure left the client in prepared-capture state.** Preparation
    precedes recording and also needs cleanup. A scoped guard now aborts an
    incomplete window on early return or unwind; successful finalization
    disarms it. Regression coverage injects an error before recording and
    panics during both preparation and recording, then checks the capture flag
    and executes another graph. The original error/panic remains observable.

## Evidence boundaries

The [benchmark report](../benchmarks/inference-followup-2026-10-13.md) separates
capture setup, ordinary/replayed synthesis, packed loading, complete-request
first audio, numerical parity and ASR. Failed/reverted experiments remain in the
artifact directory. Larger-input numerical failures must not be reported as
passing the older short-fixture FP32 gate. Native WGPU pipeline cache work is
separate from saved autotune choices and generated SPIR-V persistence.

The completed corpus contains 48 comparisons. Each sentence has the same ASR
transcript across all twelve backends/device/precision combinations, including
nonzero errors for two sentences in the official reference. All four Rust FP32
numerical failures (long female and male, on both devices) are bit-identical to
the base executable. They are not waived. PyTorch's strict speed gate still fails
in 4/16 ordinary and 3/16 captured cells; six additional portable/autotuned
controls show that choosing the portable policy does not close those gaps.
The three additional Intel MF FP32-state trials only partly recover agreement
and do not justify changing the precision default. Final reviewed binaries
reproduce all four packed FP32/FP16 device outputs and both captured noises.

The resolved Cargo graph and reverse closure of `cubecl-environment` are saved
with the measurements. Root model, codec, sampler, graph, loading, approval and
CLI consumers are in scope. The independent WGSLPack format is unchanged.
