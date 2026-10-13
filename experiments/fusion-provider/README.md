# WGPU Fusion provider experiment

This crate contains the opt-in SwiGLU postprocess provider formerly exposed by
the root crate's `fusion-provider` feature. Burn 0.22.0 makes its `fusion` feature
replace the global dispatch backend. A separate workspace prevents Cargo's
feature unification from replacing production raw WGSL tensors.

Use this crate in a separate experiment process. Register its provider before
creating any Fusion tensors. It is not part of production synthesis and is
not an alternative hardware backend. Only WGPU is enabled.

```sh
cargo nextest run --manifest-path experiments/fusion-provider/Cargo.toml
cargo nextest run --manifest-path experiments/fusion-provider/Cargo.toml --run-ignored only
```

The ignored test executes the provider on the default GPU and compares its
output with scalar FP32 SwiGLU. The production dependency upgrade benchmark
uses the root crate without Fusion.
