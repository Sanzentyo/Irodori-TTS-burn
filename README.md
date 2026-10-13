# Irodori-TTS-burn

Japanese text-to-speech inference in Rust with [Burn](https://burn.dev), WGPU
and production-connected WGSL kernels. Supports Irodori v4/v4.1 rectified flow
and v4.1 MeanFlow checkpoints, reference voices, voice instructions and a
48 kHz DACVAE codec.

FP32 is the default. FP16 model inference is opt-in; the resident synthesis API
keeps the audio codec in FP32. The production backend is `wgpu-wgsl`.

Requires Rust 1.95 or newer. The pinned compute stack is Burn 0.22.0,
CubeCL 0.11.0 and CubeK 0.3.0.

On the measured Intel ARL GPU, FP16 has no consistent speed advantage over
FP32 and the stable dependency upgrade reduces FP16 numerical agreement.
Use FP32 there when accuracy matters; see the
[upgrade comparison](docs/benchmarks/burn-0.22-upgrade-2026-10-13.md).
`tts --sampler-state fp32` keeps RF integration in FP32 with an FP16 model;
this improves the measured Intel RF agreement while preserving half weights.
It is an explicit request option, and does not make FP16 equivalent to FP32.

## Text to speech

Use the original model safetensors and its adjacent `tokenizer/tokenizer.json`.
Convert the DACVAE checkpoint with `scripts/convert_dacvae_weights.py` first.

```sh
cargo build --release --features cli,synthesis --bin irodori

target/release/irodori --checkpoint MODEL.safetensors --codec CODEC.safetensors \
  --precision fp32 --wgpu-adapter-index 0 \
  tts --text 'こんにちは、世界！' --seed 42 --output out.wav
```

The resident engine selects 40 RF steps or 4 MeanFlow steps from the checkpoint.
It predicts duration, splits oversized text, trims flat latent tails and can
stream codec windows. Global model options precede the subcommand.

The CLI tunes attention, MLP and supported decoder upsampling kernels on first use and saves the choices in
the user cache directory. Use `--wgpu-device integrated:0` for an integrated GPU
and `--cache-dir PATH` to choose the persistent cache location.

See [the synthesis guide](docs/synthesis.md) for voice creation, library APIs,
request controls, the JSON Lines worker and the OpenAI-style HTTP endpoint.
The existing `pipeline` binary remains available for lower-level controls and
optional LoRA adapter merging.

## Faster model loading

The optional `packed-weights` feature accepts `.wgslpack` model checkpoints. A
one-time conversion stores native parameter names, layout and precision, so
subsequent loads skip remapping, transposes and dtype conversion:

```sh
cargo build --release --features cli,synthesis,packed-export --bin pack_checkpoint --bin irodori
target/release/pack_checkpoint --checkpoint MODEL.safetensors \
  --output MODEL.fp32.wgslpack --precision fp32 --device discrete:0
target/release/irodori --checkpoint MODEL.fp32.wgslpack --codec CODEC.safetensors \
  --tokenizer tokenizer/tokenizer.json --precision fp32 \
  tts --text 'こんにちは。' --seed 42 --output out.wav
```

Use `--precision fp16` for both conversion and inference to create/use an FP16
model. A precision mismatch is an error. Original safetensors remain supported;
the codec stays in its existing format. The converter never overwrites an output
and publishes it only after a successful write. Packed files contain a versioned
model schema and source checksum; repack if the native parameter schema changes.
LoRA merging uses the original safetensors loading path.

The independent [WGSLPack crate](crates/wgslpack/README.md) provides borrowed
in-memory views, a layout writer and matching WGSL declarations without file I/O,
GPU dependencies or unsafe code. The Irodori adapter additionally offers
`weights::packed::{load_model_from_bytes, load_model_from_owned_bytes}`. The file
loader groups uploads with a bounded 1 MiB staging budget; an oversized tensor
is uploaded alone. `load_model_with_options` exposes the budget for explicit
experiments. Callers who already own memory can choose borrowed or shared-owned
uploads explicitly. The archive format and existing converted files are unchanged.

## Optional components

```toml
[dependencies]
irodori-tts-burn = { path = "../Irodori-TTS-burn", default-features = false, features = ["synthesis"] }
```

| Feature | Default | Adds |
|---|---|---|
| `inference` | Yes | Typed model loading and sampling engine |
| `codec` | Yes | DACVAE encoder, decoder and windowed decoding |
| `text-normalization` | Yes | Japanese normalization |
| `audio-io` | No | WAV I/O, loudness normalization and sinc resampling |
| `synthesis` | No | Resident text-to-audio library; enables the above components and tokenizer |
| `worker` | No | JSON Lines transport and base64 PCM |
| `http-server` | No | HTTP transport via `tiny_http` |
| `cli` | No | Binary tooling, clap, model downloads and logging |
| `lora` | No | Loading and merging LoRA adapters |
| `profile` | No | GPU/NVTX profiling tools |
| `packed-weights` | No | WGSLPack model loading; adds only the dependency-free format crate |
| `packed-export` | No | One-time atomic conversion; enables `packed-weights` and `tempfile` |
| `vulkan-spirv` | No | Explicit SPIR-V compiler on macOS; requires Vulkan SDK |

Core model/math functionality always includes the WGPU backend. Selecting
`synthesis` alone does not include CLI, HTTP or worker dependencies. The ordinary
macOS WGPU/Metal build does not require the Vulkan SDK; non-macOS native builds
retain the SPIR-V-enabled WGPU compiler configuration.

The former `fusion-provider` feature is now an
[independent experimental crate](experiments/fusion-provider/README.md).
Its separate workspace prevents Burn's Fusion feature from changing the
production dispatch backend. Allocator tools now call the upstream adaptive
policy `adaptive`; `sub-slices` was removed by CubeCL 0.11.0.

## Measurements and validation

- [Graph replay, decoder tuning, grouped loading and extended comparisons](docs/benchmarks/inference-followup-2026-10-13.md)
- [Burn 0.22 stable: RTX / Intel speed, precision and load comparison](docs/benchmarks/burn-0.22-upgrade-2026-10-13.md)
- [RTX 5070 Ti: strict FP32 optimization and speech.cpp comparison](docs/benchmarks/rtx-5070ti-wgsl-speech-cpp-2026-10-12.md)
- [v4.1 RF/MeanFlow, FP16, voice and transport validation](docs/benchmarks/rtx-5070ti-synthesis-parity-2026-10-12.md)
- [Intel / RTX device tuning and persistent selection](docs/benchmarks/device-autotuning-2026-10-12.md)
- [Sans-I/O native checkpoint loading](docs/benchmarks/wgslpack-loading-2026-10-13.md)

Measurements distinguish model loading, warm requests, CPU audio completion and
first streamed audio. Numerical waveform agreement is reported separately from
latency; it is not a listening-quality or intelligibility score.

```sh
cargo fmt --all -- --check
cargo clippy --features cli,worker,http-server,lora --all-targets -- -D warnings
cargo nextest run --features cli,worker,http-server --lib --test-threads 1 --no-tests fail
```

GPU tests and benchmarks must run serially on the selected adapter. The oracle
and external comparison scripts use `uv`; inference itself needs no Python.

## License

MIT — see [LICENSE](LICENSE).
