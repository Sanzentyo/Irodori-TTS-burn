# Vendored CubeK provenance

The repository patches two crates because the post-cast convolution epilogue
contract must change in lockstep across the matmul writer and convolution
launcher. Cargo resolves both through the root `[patch.crates-io]` entries.

| crate | exact release | crates.io archive SHA-256 | license |
|---|---|---|---|
| `cubek-matmul` | `0.3.0` | `59816f7d5baf51ca06f417899ee194d46d76f633541907c94091163b4ade20ff` | MIT OR Apache-2.0 |
| `cubek-convolution` | `0.3.0` | `8ec40639c42421af1be86c1f12da362f7f7d33a7d4dc1a7cf3c0c46b8963da85` | MIT OR Apache-2.0 |

Upstream repository metadata embedded in both archives points to
`https://github.com/tracel-ai/cubek`. The archives, not a moving branch, are
the reproducible source pin. To reproduce the base trees, download the exact
`.crate` archives, verify the hashes above, extract them, and then apply this
repository's diff against those extracted directories.

Local changes are intentionally limited to:

- threading runtime configuration and logical origin through global writers;
- a post-cast epilogue writer with writer-level edge masking;
- typed, mandatory host launch arguments and validation for parameterized
  epilogues;
- the DAC-style Snake post-cast epilogue;
- fixed subgroup-size validation for plane kernels, including Intel devices;
- a distinct column-bias accumulator footprint, matching the convolution
  reader's single-row allocation instead of charging a full M-by-N stage;
- scalar shared-memory views for the k7 fan-out reader, avoiding an invalid
  SPIR-V workgroup pointer bitcast during cooperative-matrix loads;
- direct contract tests in `cubek-convolution/tests/lib.rs`.

The stable release moves the matmul extension points under `multi_level` and
erases runtime generics. The local writers use dynamic vector extraction for
runtime lane indices; their arithmetic and output layouts are unchanged.

The source design is shared by CubeCL's Vulkan, Metal, and DX12 targets. Only
Vulkan/NVIDIA and Vulkan/Intel are covered by the dependency upgrade campaign; portability of the
source is not a claim that the other targets have passed shader or performance
validation.
