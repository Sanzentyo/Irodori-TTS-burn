# Upstream provenance

- Crate: `cubecl-wgpu 0.11.0`
- Repository: `https://github.com/tracel-ai/cubecl`
- Upstream commit recorded by crates.io: `52c5086d3c5e73e56b3ee4cb711bdda495a360f0`
- crates.io archive SHA-256: `2b8c6f4eb4af39ed3bbb3e4126373d8132cbe018e669172d8d4f1c775b828497`
- License: MIT OR Apache-2.0; the unmodified upstream license texts are retained.

Local changes retain the opt-in allocation high-water probe, adapter probing
before compute-server/cache initialization, and backend/device/driver identity
in the runtime fingerprint. The memory report is computed only while a probe
is active.

AutoCompiler selects WGSL for an explicitly tagged handwritten WGSL kernel,
while generated IR keeps the native compiler. The compiler instance is local
to each pipeline, so its language tag remains consistent through validation,
module creation and logging. Explicit non-Auto compilers retain upstream's
language mismatch rejection.

Software graph capture and its allocation arena now come from upstream 0.11.0;
the earlier local graph implementation has been removed. Capture is still
process-local and does not provide process-persistent pipeline serialization.
The optional `native-pipeline-cache` feature adds a private Vulkan driver-cache
owner to the compute server, shared by generated SPIR-V and handwritten WGSL
pipeline creation. Immutable per-pipeline entries are batched into the active
CubeCL environment at synchronization boundaries. Only local-origin entries
with matching adapter, driver, build, kernel and integrity checks are admitted;
imported entries are regenerated. The single added unsafe call restores bytes
returned by `wgpu::PipelineCache::get_data`. This feature enables the environment
database, independently of upstream generated-SPIR-V persistence.

The production allocator remains exclusive pages; tools can explicitly select
upstream's adaptive allocator in place of the removed SubSlices preset.

To reconstruct the upstream tree, unpack the pinned `.crate` archive and then
apply the repository diff against this directory. Verify the archive before
use:

```text
sha256sum cubecl-wgpu-0.11.0.crate
# 2b8c6f4eb4af39ed3bbb3e4126373d8132cbe018e669172d8d4f1c775b828497
```
