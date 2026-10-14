# Resident Irodori synthesis

The `synthesis` library owns one resident WGPU model, a strict FP32 decoder,
a tokenizer and cached speaker conditions. The CLI, worker and HTTP server use
this same implementation. RF and MeanFlow are selected from checkpoint metadata;
MeanFlow uses its learned interval-conditioning module and does not run CFG.

## Build only what you use

```sh
# Library without transport/CLI dependencies
cargo check --no-default-features --features synthesis --lib
# CLI for WAVs and saved voices
cargo build --release --features cli,synthesis --bin irodori
# JSON Lines worker
cargo build --release --features cli,worker --bin irodori
# HTTP server; worker is not required
cargo build --release --features cli,http-server --bin irodori
```

`synthesis` enables `inference`, `codec`, `audio-io`, `text-normalization` and
`tokenizers`. `worker` alone adds `base64`; `http-server` alone adds `tiny_http`.
Neither transport requires an async runtime. The underlying core model uses
WGPU even when the inference builder and codec features are disabled.

## Models and precision

Supply the original v4/v4.1 checkpoint and a converted DACVAE safetensors file.
The tokenizer defaults to `tokenizer/tokenizer.json` beside the checkpoint;
`--tokenizer` overrides that file. A decoder-only codec is sufficient for
synthesis with existing voices; creating voices from WAVs needs the full codec.

```sh
target/release/irodori --checkpoint MODEL.safetensors --codec CODEC.safetensors \
  --wgpu-adapter-index 0 --precision fp16 \
  tts --text '明日の東京は晴れです。' \
  --instructions '低く落ち着いた男性の声で。' --seed 42 --output out.wav
```

FP32 is the default. FP16 stores model activations and weights in half precision,
uses FP32 normalization reductions and timestep/RoPE calculations, and rounds
at the reference operation boundaries in the fused kernels. The codec stays
FP32. Results differ numerically across precision policies, GPU implementations
and matrix multiplication routes. The same integer seed is not a promise of
matching PyTorch/speech.cpp noise; comparison tools inject identical source noise.

`tts --sampler-state fp32` (JSON `"sampler_state": "fp32"`) keeps the initial
noise, guidance and ODE integration in FP32 while neural forwards retain the
model precision. The default is `"model"`. On the measured Intel RF fixture this
recovers much of the agreement lost after the dependency upgrade; MeanFlow does
not consistently benefit, so no device-dependent default is applied. Euler,
Heun and PLMS-4 support this state policy; MeanFlow retains its Euler-only rule.

Configure a device's precision before allocating any tensors, and keep one
owner of seeded synthesis per device. Warmup is shape-dependent. Use
`--startup-warmup` before `worker` or `serve` to warm a two-second request before
accepting work. For resident CLI measurements, `tts --warmup 2 --repeat 5` reports
each measured request to stderr and writes the last waveform to the output.

The engine prepares timestep conditions and AdaLN for supported schedules of
up to 40 steps. Other valid schedules use the ordinary forward path. RF guidance
changes invalidate the relevant prepared schedule; MeanFlow caches both time and
interval. Cache ABI 30 separates these routes from earlier manifests; FP16
kernel policy v8 and FP32 policy v7 invalidate earlier subgroup/mask assumptions.

## Device tuning and its persistent cache

`irodori` automatically measures attention, separate QKV/output projections, and
MLP expansion/contraction kernels on the first request for a new shape. Supported
decoder transposed convolutions also compare the cached column transform with
six WGSL tile configurations, using exact input shapes and device limits. Each
candidate must pass a numerical comparison
with an FP32 graph before being timed. Both FP32 and FP16 have separate entries;
FP16 WGSL kernels retain FP32 accumulation. The candidates include Burn and WGSL
routes with different workgroup, query, key/value, row and reduction tiles.

Use `--wgpu-device integrated:0` for an Intel integrated GPU, `discrete:0` for
the first discrete GPU, or `auto` (the default). The old
`--wgpu-adapter-index 0` option still selects the first discrete GPU.

Selections are saved in CubeCL's SQLite environment database under
`~/.cache/Irodori-TTS-burn/cubecl/adapter-<adapter-and-driver-hash>/` on Linux
(`$XDG_CACHE_HOME` is honored). They survive process exits and reboots, and are
independent of Cargo's `target` directory. `--cache-dir PATH` overrides the root;
`IRODORI_TTS_BURN_CACHE_DIR=PATH` sets the default for applications using the
backend configuration API. `RUST_LOG=info` prints the selected database path.

The cache distinguishes hardware, driver, backend, kernel policy, precision,
exact tensor dimensions and strides. A driver or kernel-policy change gets a
new namespace. The first request can take substantially longer while candidates
compile and run; subsequent requests, including after restarting, load the saved
winner without repeating validation or benchmarks. Handwritten WGSL pipelines
still compile on startup; on Vulkan, CubeCL can also cache its generated SPIR-V
kernels in this database. To retune, stop applications
using the cache and choose a new `--cache-dir` (or remove the relevant namespace).
`--no-autotune` selects the established adapter routes instead of this search;
Burn's own operation tuning still applies inside those routes.

Library users can select `RuntimeRoutePolicy::Autotuned` with `RuntimeBuilder`.
Applications managing `SynthesisEngine` directly configure the persistent cache
before WGPU initialization and call `install_autotuned_route_table` before
loading weights. Explicit approved route profiles retain their own policy.
There is one tuning-cache owner (CubeCL); rejected candidates and fresh timing
receipts can be recorded using `validate_synthesis --autotune-record FILE.json.log`.

### Approve and restore complete tuning selections

Recorder logs contain fresh decisions only. For a trajectory that reused cached
choices, use `approve_v4_autotune seal-bundle` to export and bind the **actual**
autotune entries to the existing accuracy policy. The same approval owner now
supports RF, MeanFlow, FP32 and explicitly specified FP16 policies. FP16 does
not implicitly inherit a relaxed FP32 policy.

```sh
target/release/approve_v4_autotune seal-bundle \
  --policy policy.json --identity identity.json --evidence evidence.json \
  --execution execution.json --cache-database measured-environment.db \
  --production-binary target/release/irodori \
  --output-bundle approved.flat --output-manifest approved.json
target/release/approve_v4_autotune restore-bundle \
  --manifest approved.json --identity identity.json --execution execution.json \
  --bundle approved.flat --precision fp16 --cache-root restored/adapter-hash \
  --production-binary target/release/irodori \
  --receipt restored.json
# After running the same approved workload against that cache:
target/release/approve_v4_autotune restore-bundle --verify-only \
  --manifest approved.json --identity identity.json --execution execution.json \
  --bundle approved.flat --precision fp16 --cache-root restored/adapter-hash \
  --production-binary target/release/irodori \
  --receipt rechecked.json
```

The identity records software, adapter/driver, allocator, precision and model
hashes. `execution.json` contains the exact `kernel_environment`,
`production_binary_sha256` for the consuming application and a
`settings_sha256` covering sampler and route settings; the policy pins fixture
hashes and lengths. Configure this environment before creating the GPU or a
tuner. The application remains responsible for supplying its actual identity.
The immutable bundle digest covers all saved choices, including cache hits.
Restoration rejects conflicting or extra local choices rather than replacing
them. It does not prevent later tuning of unseen shapes: run `verify_active`
after the workload and require a new accuracy gate before approving added keys.
Matching saved choices alone does not admit a different input: callers must
also respect the manifest's required fixture/workload scope.
These files are portable evidence for the specified environment and fixtures,
not certification of arbitrary voices, lengths or external cache files.

### Replay a fixed inference graph

`WgslInferenceEngine::capture_sampling` captures an Euler trajectory with fixed
text, voice, geometry and sampler settings. `capture_synthesis` also captures
the prepared FP32 decoder. They require batch one, an explicit initial noise
example in the sampler state dtype and a prepared timestep cache. Speaker
unconditional noise is rejected because replay would otherwise retain a random
condition. Capture includes a priming run and has an initial cost.

The returned session borrows its model owners. Each replay accepts fresh noise
of the same shape/device/dtype and returns owned CPU values; graph output storage
cannot escape into a subsequent request. Only the noise changes. Rebuild the
capture when text, voice, schedule or shape changes. Input refresh and output
readback use the recorded stream even if the session moves between threads or
the caller enters a different scoped stream. This API is useful for
repeated fixed requests; ordinary CLI/worker synthesis keeps its general request
path. `validate_synthesis --graph-replay` checks two different noises against
ordinary execution and reports capture cost separately from replay latency.

## Voices

```sh
# Normalize each recording to -16 LUFS, encode, then concatenate its latents.
target/release/irodori --checkpoint MODEL.safetensors --codec FULL_CODEC.safetensors \
  voice --reference take1.wav --reference take2.wav --output me.voice.safetensors

# Load and encode the saved speaker condition once for the resident engine.
target/release/irodori --checkpoint MODEL.safetensors --codec DECODER.safetensors \
  --add-voice me=me.voice.safetensors \
  tts --voice me --text 'こんにちは。' --output me.wav
```

WAV input is downmixed to mono, normalized and resampled with a Kaiser-windowed
sinc. Total reference duration is limited to 120 seconds. `--target-lufs -23`
changes normalization; `--keep-loudness` retains loudness with peak protection.
Encoder windows have eight context frames on each side. Normalization follows
the official pyloudnorm block count, including its rounding for partial blocks.

Saved reference voices are independent of model execution precision. Their
identity hashes the decoder tensor content, so full and decoder-only exports
of the same codec share voices. Different decoder weights are rejected. These
are native safetensors files, not speech.cpp GGUF files.

An official speaker-inversion `.speaker.safetensors` containing
`speaker_embedding` can be passed directly to `--add-voice` or converted with
`voice --reference SPEAKER.safetensors --output VOICE.safetensors`. FP32, FP16
and BF16 imports are accepted. An unbound imported embedding is bound to the
selected checkpoint when saved; subsequent loads reject a different checkpoint.
Use the checkpoint for which the embedding was trained. Voice saving never
overwrites an existing file.

## Request options and long text

`tts --request request.json` accepts the same `SynthesisRequest` as the library
and worker. Explicit CLI options override the JSON fields. For example:

```json
{
  "text": "明日の予定をお伝えします。",
  "voice": "me",
  "instructions": "ゆっくり、落ち着いた声で。",
  "language": "ja",
  "seed": 42,
  "steps": 40,
  "speed": 1.1,
  "cfg_scale_text": 3,
  "cfg_scale_speaker": 5,
  "cfg_scale_instructions": 3,
  "speaker_uncond_mode": "noise",
  "sway_coeff": -1
}
```

Without `--text` or `--request`, `tts` reads UTF-8 text from stdin (up to 1 MiB).
`--output -` writes only WAV bytes to stdout, so file redirection and audio
players can be used without mixing the audio with logs.

| Common options | Behavior |
|---|---|
| `voice`, `instructions` | Default voice `none`; caption may accompany a reference voice |
| `language` | `auto` or `ja` |
| `steps`, `seed` | Defaults 40 RF / 4 MF; reported seed is reusable in the same runtime |
| `sampler_state` | `model` (default) or `fp32`; controls noise, guidance and integration precision |
| `seconds` | 0.5–30 seconds per segment, before speed scaling |
| `duration_scale` | Positive multiplier of predicted duration; incompatible with `seconds` |
| `speed` | 0.25–4, divides duration; effective segment duration is bounded to 0.5–30 seconds |
| `keep_tail` | Keep the requested/predicted tail; otherwise trim flat near-zero latent frames |
| `tail_window_size`, `tail_std_threshold`, `tail_mean_threshold` | Default 20, 0.05, 0.1 |

RF additionally supports independent, joint and alternating guidance,
`cfg_min_t`/`cfg_max_t`, `truncation_factor`, paired `rescale_k`/`rescale_sigma`,
`speaker_uncond_mode`, Sway and `speaker_kv_scale` with its time/layer limits.
Joint guidance requires equal positive scales. Unsupported, unknown, inactive
or contradictory controls return errors; MeanFlow rejects explicit RF controls.

Text is normalized without silently truncating tokens. A text fitting the
256-token/30-second limits is one segment. Oversized text is split at sentence,
clause or Unicode character boundaries and synthesized in order. A split ending
a sentence adds 0.9 seconds of silence before the next segment. Instructions are
limited to 512 tokens and rejected when too long. `none` may choose different
voices for different segments; use a saved voice to keep speaker conditioning.

## Library

```rust,ignore
use irodori_tts_burn::{backend_config, WgpuFloatPrecision};
use irodori_tts_burn::synthesis::{SynthesisEngine, SynthesisModelPaths, SynthesisRequest};

let precision = WgpuFloatPrecision::Fp32;
let device = backend_config::wgpu_device_with_precision(
    &backend_config::wgpu_device_from_adapter_index(0), precision)?;
let mut engine = SynthesisEngine::load(SynthesisModelPaths {
    checkpoint: "model.safetensors".into(),
    codec: "codec.safetensors".into(),
    tokenizer: None,
}, device, precision)?;
let (samples, report) = engine.synthesize(&SynthesisRequest {
    text: "こんにちは。".into(), seed: Some(42), ..Default::default()
})?;
```

`synthesize_stream` takes a `CancellationToken` and a callback receiving mono
FP32 chunks. Returning `ControlFlow::Break` cancels generation. Cancellation is
checked between denoising forwards and codec chunks. The full sentence latent
is generated before the first audio; this is decoder streaming, not incremental
text generation. Decoder windows begin at 12 frames (0.48 s) and then 96 frames,
with context derived from the decoder's convolution geometry: eight latent
frames at each boundary for the supported codec. A full interior window of
96 emitted frames therefore decodes 112 frames. Ordinary and captured decoder
execution share this plan. Trimming preserves tail context.

`SynthesisReport` covers validation/preparation through the final callback and
reports first audio, total time, segment count and emitted sample count. Sink
backpressure is included. `TokenCounter` is a CPU-only snapshot for responsive
metadata queries while the model is busy.

## JSON Lines worker

```sh
target/release/irodori --checkpoint MODEL.safetensors --codec CODEC.safetensors \
  --startup-warmup worker
```

Stdout contains only JSON Lines, logs go to stderr. The protocol-3 synthesis
vocabulary follows speech.cpp: `ready`, `synthesize`, `chunk`, `end`, `error`,
`cancelled`, `info`, `count_tokens`, `add_voice` and `cancel`. Startup failure
emits `fatal`. Without startup warmup, `ready` means loaded, not all shapes compiled.
Model metadata declares `streaming: true`, `incremental: false`, and a `voices`
array of objects with `name`, `language`, `gender` and `description` fields.

```json
{"type":"synthesize","id":"a","text":"こんにちは。","voice":"none","seed":42}
{"type":"count_tokens","id":"b","text":"次の文章です。"}
{"type":"cancel","id":"a"}
{"type":"add_voice","id":"v","name":"me","path":"me.voice.safetensors"}
```

Chunks contain sequential `seq` values and base64 mono signed 16-bit PCM at
48 kHz. Each accepted request has one terminal response. Queries and cancellation
remain responsive during inference; eight jobs may wait. Queued cancellation
terminates immediately; running synthesis stops cooperatively. Adding a voice
cannot be canceled after encoding starts. EOF drains accepted jobs. This worker
implements synthesis, not speech.cpp's ASR/VAD vocabulary or its C ABI.
The worker completes queued GPU work before moving ownership to its inference
thread and again before that thread exits.

## HTTP

```sh
target/release/irodori --checkpoint MODEL.safetensors --codec CODEC.safetensors \
  --startup-warmup serve --listen 127.0.0.1:8080

curl http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"irodori","input":"こんにちは。","voice":"none","response_format":"wav"}' \
  --output out.wav
```

`GET /health`, `GET /v1/models`, and `POST /v1/audio/speech` are available.
`response_format: "wav"` returns a complete WAV; `"pcm"` streams raw signed
16-bit little-endian mono PCM (`X-Sample-Rate: 48000`). Other output formats are
rejected. Request options match the library, with OpenAI's `input` field in
place of `text`. Load voices using `--add-voice` before starting the server.

One model owner serializes inference; a bounded PCM channel applies backpressure
and cancels on client disconnect. Validation errors before the first chunk return
HTTP 400; a full queue returns 503. The default listener is localhost. This
optional transport does not implement authentication, TLS, a browser UI or model
download management; use it behind your application's server where needed.
