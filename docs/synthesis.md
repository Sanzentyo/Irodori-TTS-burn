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

Configure a device's precision before allocating any tensors, and keep one
owner of seeded synthesis per device. Warmup is shape-dependent. Use
`--startup-warmup` before `worker` or `serve` to warm a two-second request before
accepting work. For resident CLI measurements, `tts --warmup 2 --repeat 5` reports
each measured request to stderr and writes the last waveform to the output.

The engine prepares timestep conditions and AdaLN for supported schedules of
up to 40 steps. Other valid schedules use the ordinary forward path. RF guidance
changes invalidate the relevant prepared schedule; MeanFlow caches both time and
interval. Cache ABI 29 separates these routes from earlier manifests; FP16
shader cache policy v6 is separate from unchanged FP32 policy v5.

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
with ten context frames at each boundary. Trimming preserves tail context.

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
