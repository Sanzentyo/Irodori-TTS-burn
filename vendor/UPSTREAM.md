# Vendored CubeK provenance

The repository patches two crates because the post-cast convolution epilogue
contract must change in lockstep across the matmul writer and convolution
launcher. Cargo resolves both through the root `[patch.crates-io]` entries.

| crate | exact release | crates.io archive SHA-256 | license |
|---|---|---|---|
| `cubek-matmul` | `0.3.0-pre.2` | `204917532e8f5bc4b440c640551d2c10d402bac7fd8062450d843752352518df` | MIT OR Apache-2.0 |
| `cubek-convolution` | `0.3.0-pre.2` | `b3c24e82dad0d0d0c6fcaf7d13376653529e91639ef401457e972f2e329fb018` | MIT OR Apache-2.0 |

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
- direct contract tests in `cubek-convolution/tests/lib.rs`.

The source design is shared by CubeCL's Vulkan, Metal, and DX12 targets. Only
Vulkan/NVIDIA has been executed in the current campaign; portability of the
source is not a claim that the other targets have passed shader or performance
validation.
