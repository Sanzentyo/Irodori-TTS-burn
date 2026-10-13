# Native pipeline cache review — October 14, 2026

Scope: self-review of the complete diff, the unsafe admission boundary, ownership,
environment changes, optional features, persistent storage, benchmark methodology
and affected consumers. This is not an independent human review. The user
explicitly authorized the new unsafe cache-creation call and commit/push after
review and validation.

## Findings and resolutions

1. **The prototype enabled an unrelated persistence path.** Its feature forwarded
   `cubecl-wgpu/persistence`, which enables `cubecl-server/persistence` and changes
   generated-SPIR-V storage. The prior runtime setting did not actually enable
   that Cargo feature. The final feature forwards only
   `cubecl-environment/persistence`, the database needed by native cache storage.
   Matching default-feature metadata has no added/removed package versions, and
   `cubecl-server` retains `std,storage-bytes`. Historical documentation was
   corrected. Formal OFF/ON trials use the same final binary.

2. **Persistent bytes need an explicit trust boundary.** The private writer saves
   exact WGPU `get_data` output. The private reader checks local origin, adapter,
   driver, build, kernel, schema, length and checksum before the single unsafe
   call. Imported entries cannot reach it. Checksums do not authenticate a
   malicious writer to local application state; this remaining assumption is
   documented, as required by WGPU's persisted-data safety caveat. No public
   method accepts arbitrary pipeline-cache payloads.

3. **Repeated whole-cache serialization would scale poorly during autotuning.**
   The owner stores immutable per-pipeline entries and batches writes at existing
   sync/read/flush boundaries. Restored pipelines do not serialize data again.
   Successful inference does not depend on cache writes. Storage failures are
   logged without introducing a second cache store or a retry hierarchy.

4. **Environment changes must invalidate both storage and in-memory pipelines.**
   The owner observes the existing environment generation, discards old pending
   entries and tells the server to clear its pipeline map. Database handles stay
   attached to their original paths. The real-device test switches environments
   before checkpointing a pending entry and verifies it was not published.

5. **Restore telemetry could be mistaken for a driver hit counter.** Documentation
   calls it the number of validated blobs supplied to WGPU. Actual benefit is
   established by measured pipeline creation and first-request latency. Normal
   driver caching and controlled disabled-driver-caching conditions are reported
   separately. Native persistence remains opt-in.

6. **An import could flood logs with expected rejections.** Imported entries now
   log at debug level; corrupt local entries produce one count warning. A build
   identifier is computed once per owner instead of once for every kernel.

7. **Linting the patched dependency itself exposed additional warnings.** Its
   own Clippy configuration requires backticks around `CubeCL` in doc comments;
   these were fixed in the new module and two existing comments. An existing
   derived-Default unit-struct construction was replaced with the identical
   direct `WgslCompiler` value. No lint suppression was added. Native lifecycle
   tests, six graph tests, root all-target Clippy and dependency Clippy pass again
   after this cleanup.

8. **A draft overstated restored-cache completeness.** Final telemetry auditing
   found that three of 80 restored MF processes each compiled one additional
   pipeline during loading. The other 77 wrote no entries; 27,261 of 27,264
   creations received saved data, with zero rejected entries. The report was
   corrected. All samples remain in the comparison, including their extra
   compilation and save cost; no rerun replaced an inconvenient sample.

## Verification scope

`cubecl-wgpu` feeds CubeCL, CubeK and Burn, then `irodori-tts-burn`. The reverse
dependency graph and Cargo metadata are retained with the measurements. The
affected workspace consumer has 686 selected library/integration tests, including
all ignored GPU, tokenizer and packed-model tests; all pass. The independent
WGSLPack format crate is unchanged and is not a reverse dependency of WGPU.

The private owner has two parser/identity tests and one native GPU lifecycle test.
The three tests pass on RTX; the GPU lifecycle test also passes on Intel. They
cover a real shader execution after cache creation/restore, corruption repair,
import rejection, disabled caching and environment switching. No fabricated
driver payload is passed to WGPU. Testing the vendored dependency requires an
isolated workspace copy because Cargo will not test an external workspace
dependency's dev targets. The copied Rust source is identical to production.

Test/dev debug information was disabled and incremental compilation turned off
after a debug build exhausted disk space. This changes test artifact size, not
the release benchmark profile. The earlier intermediate test run was interrupted
when feature coupling was removed; it is not counted as a final pass.

An early prototype Intel run with **native caching disabled and the additional
SPIR-V persistence enabled** hit a `turso_core` shared-reader-slot assertion.
Ten repeat runs did not reproduce it. The precise DB cause remains unproven;
it is not described as fixed, and no panic-catching/retry workaround was added.
The final feature excludes that additional SPIR-V persistence path. Its real
measurements run without retries and retain any failure logs.

The final campaign contains 164 MF processes plus 24 additional RF, old-binary
and graph-replay processes. All 188 pass exact latent/audio comparisons against
the matching previous output, streaming/trim checks and zero-new-autotune checks.
The RF FP16 runs keep FP32 sampler state. All four graph cases also match ordinary
execution with two different noises. The initial additional-run wrapper used
`--warmups 0`, which the CLI rejected before inference; the wrapper was corrected
to the supported value 1, with the rejected command/log retained. That invocation
is not counted among the 188 executed validations.

Root all-target/all-feature Clippy, owner Clippy, formatting, core feature-off,
minimal native feature and WGSL-only owner checks pass. The runtime source hashes
match the frozen release binary's build snapshot; the vendored test Rust source
matches production. No runtime code changed during the measurement campaign.
The unchanged lockfile and matching default dependency graph confirm no added
package version. Native persistence remains opt-in and the pre-existing SPIR-V
persistence setting remains unchanged.

The measured benefit justifies the optional feature: normal-driver RTX FP16
first inference falls from 1.884 to 1.131 seconds, and with the implicit driver
disk cache disabled all four GPU/precision cases improve. Normal-driver FP32
and Intel latency and steady inference show little change. FP16's existing
numerical differences from FP32 are explicitly retained in the report. No
unresolved finding requiring a runtime-code change remains from this review.

Final measurements and their executable commands are in
[the benchmark report](../benchmarks/native-pipeline-cache-2026-10-14.md).
