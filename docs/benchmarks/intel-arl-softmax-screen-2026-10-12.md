# Intel ARL WGSL softmax screen — 2026-10-12

## Status

This records the initial Intel-only screen before driver restoration. The
user subsequently restored the RTX driver and rebooted. The
[RTX and speech.cpp follow-up](rtx-5070ti-wgsl-speech-cpp-2026-10-12.md) contains
the completed full-request measurements and accepted subgroup route. The
portable cached variant below remains a tuning candidate. The existing NVIDIA
benchmark document's pre-existing user edit was preserved.

## Environment and diagnosis

- Base commit: `bb547dc`.
- Ubuntu 26.04 LTS, kernel `7.0.0-38-generic`, Secure Boot enabled.
- CPU: Intel Core Ultra 9 275HX.
- Available GPU: Intel Graphics (ARL), vendor/device `8086:7d67`, Vulkan,
  Intel open-source Mesa driver `26.0.3-1ubuntu1`.
- Rust/Cargo 1.95.0; locked Burn `0.22.0-pre.2`, CubeCL `0.11.0-pre.2`.
- RTX: detected on PCI `0000:01:00.0`, but no NVIDIA kernel module is loaded
  or installed for the running kernel. `nvidia-smi` fails; the explicit
  discrete WGPU probe cannot find a device. Driver userspace `595.71.05` and
  modules for older kernels are installed.
- APT offers `linux-modules-nvidia-595-open-7.0.0-38-generic` and the matching
  `linux-modules-nvidia-595-open-generic` metapackage. A read-only installation
  simulation succeeds, updating the 595 userspace libraries and replacing the
  older NVIDIA kernel modules. No system packages were changed by this task.

After installation and reboot, both `nvidia-smi` and the explicit discrete
adapter probe succeeded with driver 595.99.02; see the follow-up report.

## Change and measurement

`try_fused_scores_softmax_cached_f32` keeps scaled logits and validity bits in
lane-private arrays, eliminating intermediate score writes and rereads. It
preserves the old 256-lane F32 max/sum reduction order, computes each
exponential once, writes one final probability, and rejects more than 2,048
keys before dispatch. The original streaming entrypoint remains available.

`bench_v4_softmax` runs both kernels in one process with alternating ABBA/BAAB
blocks. Each softmax invocation receives pristine input restored outside the
timed interval. Timings include device completion, with separate GPU timestamp
durations. Readback and accuracy comparison occur after timing. The optional
`--scope sdpa` includes QK and PV matmuls, but was not run in this screen.

Command, after `cargo build --locked --release --features cli --bin bench_v4_softmax`:

```sh
target/release/bench_v4_softmax --integrated --batch 3 --sequence 489 \
  --warmups 4 --blocks 20 \
  --output-json /home/sanzentyo/benchmark-artifacts/irodori-intel-wgsl-20261012/softmax-b3-s489-screen1.json
```

Shape: `[3,20,489,511]`, strict FP32, 14,992,740 score elements. One fresh
process, four warmups per route, 40 measured invocations per route:

| Route | GPU timestamp median | Device-complete wall median |
|---|---:|---:|
| Streaming | 13.175443 ms | 14.039438 ms |
| Cached | 9.561979 ms | 10.553509 ms |

The GPU median decreases by 27.43% in this single-process screen. Outputs are
bitwise identical, max absolute error and RMSE are zero, and no uncaptured WGPU
error occurred. This is an operator screen on Intel, not a complete model
measurement or evidence for adopting the kernel on NVIDIA.

Raw results, adapter receipt and binary/shader SHA-256 values are in:
`/home/sanzentyo/benchmark-artifacts/irodori-intel-wgsl-20261012/`.

## Validation and follow-up

- `cargo metadata --format-version 1 --offline` and
  `cargo tree --offline --workspace --invert irodori-tts-burn`: the modified
  package has no workspace reverse consumers. No dependency versions changed.
- `cargo nextest list --locked --features cli --lib --run-ignored all -E
  'test(kernels::matmul_sdpa::)'` verified the five selected tests.
- `cargo nextest run --locked --features cli --lib --run-ignored all -E
  'test(kernels::matmul_sdpa::)' --no-tests fail --test-threads 1`: **5 passed**.
  Tests cover lane boundaries (1, 7, 255, 256, 257, 511, 512, 513, 1025, 2048
  keys), fully masked rows, masked NaNs, bitwise streaming parity, an independent
  F64 softmax oracle, invalid scales and the 2049-key rejection.
- `cargo clippy --locked --features cli --all-targets -- -D warnings`: passed.
- `cargo fmt --all`: applied.

After RTX restoration, the follow-up measured paired B1/B3 operator runs at
multiple lengths, complete SDPA, and complete 40-step RF requests. Explicit
cached/subgroup candidates are now connected through `route_autotune` and
`model/attention`. Only the NVIDIA B1/B3 S489 subgroup cells were adopted,
after three alternating full-request pairs produced bitwise-identical audio.
See the linked follow-up for the measurements, reference comparison and
reproduction commands. The Intel result above remains a separate operator
screen, without an Intel full-request admission claim.
