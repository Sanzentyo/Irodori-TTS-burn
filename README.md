# Irodori-TTS-burn

Japanese text-to-speech inference in Rust with [Burn](https://burn.dev), WGPU
and production-connected WGSL kernels. Supports Irodori v4/v4.1 rectified flow
and v4.1 MeanFlow checkpoints, reference voices, voice instructions and a
48 kHz DACVAE codec.

FP32 is the default. FP16 model inference is opt-in; the resident synthesis API
keeps the audio codec in FP32. The production backend is `wgpu-wgsl`.

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

The CLI tunes attention and MLP kernels on first use and saves the choices in
the user cache directory. Use `--wgpu-device integrated:0` for an integrated GPU
and `--cache-dir PATH` to choose the persistent cache location.

See [the synthesis guide](docs/synthesis.md) for voice creation, library APIs,
request controls, the JSON Lines worker and the OpenAI-style HTTP endpoint.
The existing `pipeline` binary remains available for lower-level controls and
optional LoRA adapter merging.

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
| `vulkan-spirv` | No | Explicit SPIR-V compiler on macOS; requires Vulkan SDK |

Core model/math functionality always includes the WGPU backend. Selecting
`synthesis` alone does not include CLI, HTTP or worker dependencies. The ordinary
macOS WGPU/Metal build does not require the Vulkan SDK; non-macOS native builds
retain the SPIR-V-enabled WGPU compiler configuration.

## Measurements and validation

- [RTX 5070 Ti: strict FP32 optimization and speech.cpp comparison](docs/benchmarks/rtx-5070ti-wgsl-speech-cpp-2026-10-12.md)
- [v4.1 RF/MeanFlow, FP16, voice and transport validation](docs/benchmarks/rtx-5070ti-synthesis-parity-2026-10-12.md)
- [Intel / RTX device tuning and persistent selection](docs/benchmarks/device-autotuning-2026-10-12.md)

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
