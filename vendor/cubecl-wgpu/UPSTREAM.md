# Upstream provenance

- Crate: `cubecl-wgpu 0.11.0-pre.2`
- Repository: `https://github.com/tracel-ai/cubecl`
- Upstream commit recorded by crates.io: `04ef8ece16c481db4ae2ee1cea7ab6eb20890b5e`
- crates.io archive SHA-256: `aa87054aea892561fb0645491c5b156286d6d25c47e6f33fec7d2eeab002be5a`
- License: MIT OR Apache-2.0; the unmodified upstream license texts are retained.

The local patch adds a process-local WGPU software graph implementation. It
records pipeline, bind-group, immediate, resource-transition, and dispatch
state; owns a dedicated reusable allocation arena for graph intermediates;
and rejects capture operations that cannot be replayed safely. The arena uses
aligned 64 MiB sliced pages so non-overlapping graph intermediates can reuse
storage, plus an exclusive fallback for allocations larger than one page. This
avoids both one-buffer-per-intermediate retention and CubeCL's device-wide
SubSlices preset, whose largest page is unnecessarily large for this graph.
The public CubeCL graph lifecycle remains unchanged.

The dedicated arena is backend-neutral WGPU code. Vulkan, Metal, DX12 and
browser WebGPU share the implementation, but the benchmark campaign in this
repository currently validates NVIDIA/Vulkan only. This patch does not claim
native driver graph capture or process-persistent pipeline serialization.

To reconstruct the upstream tree, unpack the pinned `.crate` archive and then
apply the repository diff against this directory. Verify the archive before
use:

```text
sha256sum cubecl-wgpu-0.11.0-pre.2.crate
# aa87054aea892561fb0645491c5b156286d6d25c47e6f33fec7d2eeab002be5a
```
