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
model. A precision mismatch is an error. Original safetensors remain supported.
The converter never overwrites an output
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

Optional decoder packing also uses `packed-export` with `codec`. It stores
FP32 weights and the prepared layouts consumed by the existing WGSL kernels.
Conversion runs entirely on the CPU:

```sh
cargo run --release --features cli,codec,packed-export --bin pack_codec -- \
  --checkpoint CODEC.safetensors --output CODEC.wgslpack
```

Pass the result as `--codec CODEC.wgslpack` in a `packed-weights` build. The
library provides `codec::packed::{load_decoder_from_bytes,
load_decoder_from_owned_bytes}`. These decoder-only files support existing
voices but cannot encode new reference audio. RTX and Intel measurements retained
bit-identical output and reduced codec loading time by 28–45%, at the cost of a
34% larger decoder file; see the
[measurement report](docs/benchmarks/optimization-acceptance-2026-10-14.md).

## Optional components

```toml
[dependencies]
irodori-tts-burn = { path = "../Irodori-TTS-burn", default-features = false, features = ["synthesis"] }
```

| Feature | Default | Adds |
|---|---|---|
| `inference` | Yes | Typed model loading and sampling engine |
| `codec` | Yes | DACVAE encoder, decoder and windowed decoding |
| `streaming-graphs` | No | Bounded decoder graph cache; also requires explicit activation |
| `text-normalization` | Yes | Japanese normalization |
| `audio-io` | No | WAV I/O, loudness normalization and sinc resampling |
| `synthesis` | No | Resident text-to-audio library; enables the above components and tokenizer |
| `worker` | No | JSON Lines transport and base64 PCM |
| `http-server` | No | HTTP transport via `tiny_http` |
| `cli` | No | Binary tooling, clap, model downloads and logging |
| `lora` | No | Loading and merging LoRA adapters |
| `profile` | No | GPU/NVTX profiling tools |
| `packed-weights` | No | WGSLPack model/decoder loading; adds only the dependency-free format crate |
| `packed-export` | No | One-time atomic conversion; enables `packed-weights` and `tempfile` |
| `vulkan-spirv` | No | Explicit SPIR-V compiler on macOS; requires Vulkan SDK |
| `native-pipeline-cache` | No | Local Vulkan driver cache in the existing CubeCL environment |
| `spirv-cache` | No | Generated-SPIR-V persistence, independent of the driver cache |

Core model/math functionality always includes the WGPU backend. Selecting
`synthesis` alone does not include CLI, HTTP or worker dependencies. The ordinary
macOS WGPU/Metal build does not require the Vulkan SDK; non-macOS native builds
retain the SPIR-V-enabled WGPU compiler configuration.

For persistent Vulkan pipeline compilation, build with
`--features cli,synthesis,native-pipeline-cache`. It uses the same adapter-specific
CubeCL directory selected by `--cache-dir` or `IRODORI_TTS_BURN_CACHE_DIR`;
`CUBECL_WGPU_NATIVE_PIPELINE_CACHE=0` disables just the native driver layer.
Driver caches are trusted local state and are regenerated after a driver/build
change or a cache-bundle import. The feature stays opt-in because an already
warm driver disk cache can leave little additional startup benefit.

`spirv-cache` additionally enables CubeCL's generated-shader persistence on
SPIR-V-capable backends; `CUBECL_WGPU_SPIRV_CACHE=0` disables this layer at
process startup. It does not replace `native-pipeline-cache` and neither feature
enables the other. Build with
`--features cli,synthesis,native-pipeline-cache,spirv-cache` to use both.
Both layers reduced first-request compilation time in the
[RTX/Intel campaign](docs/benchmarks/optimization-acceptance-2026-10-14.md), with
unchanged output. This is a startup benefit, not a general resident-inference
speedup. The historical Intel database assertion did not recur in that campaign;
it is not considered a root-cause fix.

To remeasure only validated projection choices, use
`prepare_projection_retune` and `scripts/retune_projection_cache.py` as described
in the [retuning report](docs/benchmarks/deep-followup-2026-10-14.md).
The campaign preserves unrelated cache entries and records the selected cache
path only after repeated performance and accuracy checks.

With `streaming-graphs`, `irodori --codec-graph-cache-entries 4
--codec-graph-max-frames 116 ...` explicitly enables graph reuse for ordinary
streaming synthesis. Both input geometry and output crop belong to the key.
Limits bound entry count and frames, not measured VRAM bytes; oversized windows
and capture failures return errors. RTX and Intel checks preserve PCM exactly,
including changed inputs, eviction and environment switches. Resident RTX FP16
benefited in the [measured workloads](docs/benchmarks/optimization-acceptance-2026-10-14.md),
while FP32 gains were small. Initial capture still adds latency and reserved
memory, so the ordinary path remains the default.

The former `fusion-provider` feature is now an
[independent experimental crate](experiments/fusion-provider/README.md).
Its separate workspace prevents Burn's Fusion feature from changing the
production dispatch backend. Allocator tools now call the upstream adaptive
policy `adaptive`; `sub-slices` was removed by CubeCL 0.11.0.

## Measurements and validation

- [Projection retuning, faster identity hashing, mutable condition graphs and layer precision](docs/benchmarks/deep-followup-2026-10-14.md)
- [Shader persistence, packed codec, streaming graphs and rejected kernel candidates](docs/benchmarks/optimization-acceptance-2026-10-14.md)
- [Native pipeline cache: RTX / Intel startup and exact output comparison](docs/benchmarks/native-pipeline-cache-2026-10-14.md)
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
