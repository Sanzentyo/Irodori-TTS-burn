# RTX 5070 Ti: Irodori synthesis, MeanFlow and FP16 — 2026-10-12

This change adds resident Irodori v4/v4.1 synthesis to the Rust library and
optional CLI, JSON Lines and HTTP interfaces. It extends the strict-FP32
optimization in `032b740` to MeanFlow and an explicit FP16 model policy.
The codec stays FP32. FP32 remains the default because the FP16 waveform
differences below are material; speed is not evidence of equal audio quality.

Usage and dependency boundaries are in [the synthesis guide](../synthesis.md).
The earlier v4/19.56-second comparison and subgroup-softmax optimization are in
[the preceding report](rtx-5070ti-wgsl-speech-cpp-2026-10-12.md).

## Environment and reproducibility

- NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12 GB, 120 W; driver 595.99.02;
  Intel Core Ultra 9 275HX; Linux/Vulkan, adapter 0, subgroup width 32.
- Rust 1.95; Burn 0.22.0-pre.2; CubeCL 0.11.0-pre.2; wgpu 30.
  Release builds with `cli,worker,http-server,profile`.
- Production uses raw WGPU, custom WGSL kernels and the existing CubeCL
  AutoCompiler matmul routes. CUDA/PyTorch are used only for the oracle.
- External [speech.cpp](https://github.com/nyosegawa/speech.cpp/tree/d23c49f3233375e3ad4944c880807790b7f490be)
  at `d23c49f3233375e3ad4944c880807790b7f490be`, ggml
  `1d73413d60fd74538099958600ca111856473e59`. The clone is outside this
  repository; no submodule or speech.cpp runtime dependency was added here.
- Official [Irodori-TTS](https://github.com/Aratako/Irodori-TTS/tree/89f9d8fbd4d51ea019867ee1197725ede1df13c5)
  at `89f9d8fbd4d51ea019867ee1197725ede1df13c5`, torch/torchaudio 2.10,
  transformers 5.12.1. No compilation or watermarking; TF32 disabled.
- Model SHA-256: MF
  `a3f204b3ee06058f4e639a1af38408e7e3af5b6c75176897ecb8bfaaebb4fd54`;
  v4.1 RF
  `c85de88c01700cb53538e706f128ebcb1b8513ad21d7d0e75f58bc82cdbf89f6`.
- Local evidence root:
  `/home/sanzentyo/benchmark-artifacts/irodori-feature-parity-20261012`.
  Original models are in the Hugging Face cache; the earlier external clone is
  under `/home/sanzentyo/benchmark-artifacts/irodori-rtx-wgsl-20261012/speech.cpp`.

All GPU work ran serially. Each final resident run uses two warmup requests and
five measured requests. Tables show medians. The boundary begins with token
tensors already available, includes condition encoding and sampling, and ends
with complete host waveform readback after a whole codec decode. Model loading,
tokenization, duration prediction and diagnostics are excluded from these
latency tables. Those product paths are exercised separately below.

The oracle supplies identical FP32 source noise, token IDs and masks. FP16
casts the source noise once. C++ receives compact masked-in token IDs and uses
its native sampler schedule. Matching an integer seed alone would not match
the three runtimes' random number generators. No waveform alignment, gain
fitting or sample removal is applied to the numerical comparisons.

## Matched performance and numerical error

MF: `こんにちは。`, 50 latent frames / 2 seconds, four MeanFlow steps, no
reference or caption. RF: `明日の東京は晴れです。`, caption
`低く落ち着いた男性の声で。`, 100 frames / 4 seconds, 40 steps, text/caption
CFG 3 over the native `[0.5, 1]` guidance window.

| Runtime and policy | MF sampling + conditions | MF host audio | MF waveform SNR vs official FP32 |
|---|---:|---:|---:|
| Burn, strict FP32 | 79.544 ms | **103.836 ms** | 92.70 dB |
| speech.cpp, F32 weights, strict FP32 | 67.054 ms | 133.328 ms | 94.29 dB |
| Burn, FP16 weights and activations, FP32 codec | 29.850 ms | **55.151 ms** | 30.59 dB |
| speech.cpp, F32 weights, default Vulkan arithmetic | 35.092 ms | 80.598 ms | 44.25 dB |
| speech.cpp, F16 GGUF, default Vulkan arithmetic | **23.946 ms** | 63.467 ms | 46.75 dB |

| Runtime and policy | RF sampling + conditions | RF host audio | RF waveform SNR vs official FP32 |
|---|---:|---:|---:|
| Burn, strict FP32 | **962.329 ms** | **1,015.214 ms** | 92.26 dB |
| speech.cpp, F32 weights, strict FP32 | 1,506.250 ms | 1,635.031 ms | 84.43 dB |
| Burn, FP16 weights and activations, FP32 codec | **314.710 ms** | **364.804 ms** | 25.50 dB |
| speech.cpp, F32 weights, default Vulkan arithmetic | 533.111 ms | 614.341 ms | 50.54 dB |
| speech.cpp, F16 GGUF, default Vulkan arithmetic | 453.783 ms | 534.607 ms | 50.41 dB |

Strict C++ runs set `GGML_VK_DISABLE_F16=1`, `GGML_VK_DISABLE_COOPMAT=1`
and `GGML_VK_DISABLE_COOPMAT2=1`; all C++ runs select `GGML_VK_VISIBLE_DEVICES=0`.
The F16 files were produced from the same F32 conversions using the pinned
`speech quantize ... --type f16`. Their runtime retains FP32 intermediate
values in places where Burn's FP16 activation policy rounds to half precision.
Consequently these are explicitly different arithmetic policies, not an
equal-accuracy FP16 contest. C++'s decoder follows its own default arithmetic.

Against strict C++, Burn reduces total host-audio time by about 22% for MF
and 38% for RF. C++ is faster in the short MF sampling stage; Burn's faster
codec reverses the total. Within Burn, FP16 is approximately 1.88x / 2.78x as
fast for MF / RF host audio, with the numerical tradeoff shown above.

Evidence: `mf-final-{fp32,fp16}/report.json`,
`rf41-final-{fp32,fp16}/report.json`, and
`cpp-{mf,rf41}-{strict,default,f16}/{timings,metrics}.json`.
Every report retains the individual measurements and raw latent/waveform files.

SNR is a numerical agreement measure, not intelligibility, speaker similarity
or a listening score. No corpus-level CER or human listening study was run.
In the MF control, PyTorch FP16 itself gives 35.44 dB against PyTorch FP32;
Burn FP16 versus that PyTorch FP16 waveform gives 26.96 dB. FP16 is an optional
speed/memory policy, not a claim of FP32-equivalent accuracy.

## Implemented optimization and precision contracts

- MeanFlow loads the learned `delta_cond_module`, uses the exact two-sided
  FP32 `linspace(1, 0)` schedule and evaluates one interval-conditioned forward
  per step. It does not silently run the RF sampler or CFG.
- Resident synthesis prepares timestep conditions and cross-layer AdaLN for
  eligible schedules up to 40 steps. MeanFlow keys include the interval;
  RF includes its guidance window and topology. Unsupported schedules use the
  ordinary path. Diagnostic reports confirm 4/4 MF and 40/40 RF cache hits,
  with zero ordinary timestep-condition forwards.
- FP16 uses explicit reference rounding boundaries in fused AdaLN,
  Q/K normalization before RoPE, SwiGLU and gated residuals. The SiLU kernel
  evaluates in FP32 and rounds once. RoPE tables and timestep embeddings stay
  FP32. FP32 arithmetic in these kernels is unchanged.
- The FP16 QKV postprocess accepts FP32 RoPE directly, avoiding a portable
  fallback without discarding angle precision.
- The DACVAE loader accepts an explicit floating dtype, so FP16 model sessions
  retain strict-FP32 codec weights and activations.
- Route ABI 29 rejects old route manifests. FP32 shader environment v5 is
  retained; FP16 uses v6 to invalidate previous rounding-policy shader caches.

Moving from condition-only to full AdaLN caching preserves the MF and RF
FP32 waveforms bit for bit. Their SHA-256 values are respectively
`d09f4537db7a0b651bf415ad9a2cb1a52be733545575d353e7a2ce77d1e856aa` and
`2446cda30aa9d5c408474257c4c2881a484b9775b2bdc8e2a77d4db8085b5638`.
FP16 cache variants differ numerically (MF 51.83 dB and RF 41.89 dB between
variants); no bitwise claim is made for those. See
`precision-policy-comparison.json`.

A same-binary MF FP32 cache A/B run uses the same two warmups and five
measurements. Disabling timestep/AdaLN preparation gives 83.314 ms sampling
and 107.568 ms host audio; enabling it gives 80.003 ms and 105.316 ms.
This reduces sampling time by 3.97% while preserving the waveform SHA-256
above. Reports are `mf-ab-disabled-fp32` and `mf-ab-full-fp32`; the diagnostics
change from four ordinary condition forwards to four prepared cache hits.

The original v4, 489-frame / 19.56-second strict-FP32 workload was also rerun
after the additions. All seven requests reproduce the pre-optimization and
`032b740` waveform SHA-256 exactly:
`4bf6a50d2805dcd5ea7343229899e21f27c6e32c558b1f9e9858e1719b5278a2`.
The five measured requests have median sampling time 2.419596 s and host-audio
time 2.721069 s. This is a regression check, not a new controlled speedup claim
over the previous 2.749132 s measurement. Its complete report is in the earlier
artifact root as `final-parity-regression-fp32.json`.

Promoting all half-precision attention operations to FP32 was evaluated and
discarded: the MF test became slower and its waveform SNR against the FP32
oracle fell from 30.32 to 27.30 dB. The trial remains in
`mf-validate-fp16-attention32`; it is not part of the implementation.

## Voice, streaming and frontend evidence

A six-second reference, made from the earlier generated two-second test audio,
exercises encoder window boundaries. The Rust FP32 encoder uses 96-frame
windows with eight context frames, and matches the official whole encoder at
**92.97 dB latent SNR**, maximum absolute difference `1.67e-4`.
This is a numerical encoder check with synthetic audio, not a speaker-cloning
quality evaluation. See `voice-encoder-parity.json`.

The four-second MF trajectory with that reference and caption
`落ち着いた声で。` gives **82.70 dB** waveform SNR in FP32 and **32.61 dB**
in FP16 against the official FP32 trajectory. Host-audio medians are 161.594 ms
and 111.403 ms. Both speaker and caption conditions are active. The C++ oracle
harness currently rejects reference-voice fixtures; this voice comparison is
against the official Python implementation.

Windowed decoding starts at 12 frames and then uses 96-frame windows with
ten context frames. Whole versus windowed waveforms give at least **124.5 dB**
SNR in all six final fixtures. Trimming the requested sample count retains
the original tail as convolution context, and is compared separately. The
streaming reports' first-audio number is codec-only; it must not be mistaken
for full text-to-first-audio latency.

The checked-in frontend fixture contains **164 cases** from the official
tokenizer, short Japanese phrases, speech.cpp edge cases and annotation emojis.
Normalization and all token IDs match. Long-text checks preserve normalized
content without token truncation, and oversized captions are rejected.
Five sample rates (16/22.05/44.1/48/96 kHz) match torchaudio's Kaiser resampler
within `2e-6` at the golden samples and pyloudnorm within `1e-5` LUFS. The
audio fixture was regenerated byte-for-byte with
`scripts/export_audio_frontend_oracle.py`.

## Synthesis features and dependency separation

| Irodori capability | Result in this change |
|---|---|
| v4/v4.1 RF and v4.1 MF | Checkpoint-selected sampler; FP32 and FP16 |
| Text normalization/tokenization | Official 164-case corpus; reject overlong captions |
| Duration, speed, fixed seconds, tail cutting | Shared library request API |
| Long text | Sentence/clause/character splitting, ordered output and sentence pauses |
| Reference voice creation | Multiple WAVs, loudness, sinc resampling, windowed FP32 encoder |
| Saved voice and speaker inversion | Native safetensors; import FP32/F16/BF16 speaker embeddings |
| Voice identity | Decoder-content identity, model binding for learned embeddings, no silent overwrite |
| RF controls | Independent/joint/alternating CFG, windows, noise/mask unconditioning, Sway, rescale, truncation, speaker KV scaling |
| Streaming | Overlapped codec windows, bounded transport backpressure and cooperative cancellation |
| Resident worker | Protocol-3 synthesis messages, responsive queries, bounded jobs, exactly one terminal |
| HTTP | Health/models and OpenAI-style speech endpoint; WAV and streamed PCM |
| CLI | WAV/voice commands, JSON requests, stdin text/stdout WAV and resident repeated measurements |

`synthesis` is a library feature. `worker` adds its base64 dependency;
`http-server` independently adds `tiny_http`. Neither brings in an async
runtime. CLI dependencies remain under `cli`, WAV preprocessing under
`audio-io`, and the codec under `codec`. The no-default core builds without
tokenizers, hound, HTTP, worker, hf-hub or clap. This uses feature boundaries
within the existing crate; applications can select only the needed modules.

This is Irodori synthesis parity within a Rust/WGPU application. It does not
provide speech.cpp's ASR/VAD/Qwen models, C ABI, browser application, GGUF
model/voice interchange or Q4/Q8 quantized inference. The HTTP implementation
is the optional speech API, not a multi-model hosting service. FP16 here means
the explicit policy described above; it is not GGUF F16 compatibility.

Real process tests are retained in `interfaces-final-mf-fp16` and
`interfaces-final-rf-fp32`. They cover cancellation during a 1,000-step job,
queries while busy, a following successful request, invalid-option rejection,
voice loading/info, simultaneous reference/caption conditioning, and HTTP
WAV/PCM byte equality at 48 kHz. RF additionally exercises noise guidance,
Sway, alternating guidance, rescaling, truncation and speaker KV controls.
The 353-token MF long-text request emits 64 segments, 63 pauses and 4,319,040
samples through 191 chunks. Its deliberately short fixed segment duration
tests scheduling, not long-form pronunciation quality.

After adding explicit GPU completion when handing the engine to a transport
thread and when that thread exits, both process suites were repeated in
`interfaces-drained-mf-fp16` and `interfaces-drained-rf-fp32`. They additionally
send a voice-add request followed immediately by EOF, before initialization
finishes. Both processes emit exactly `ready`, `end` and exit successfully.
The unchanged long-text case reuses the preceding successful run.

In the final runs, busy-worker token queries returned in 1.274 ms (MF/FP16)
and 0.632 ms (RF/FP32). Observed loaded-but-unwarmed `ready` times were 9.86 s
and 6.56 s; these single startup observations include existing on-disk caches
and are not cold-start medians. First unseen shapes may still compile after
`ready`.

`cli-final-mf-fp16/report.json` additionally exercises automatic duration and
streamed decoder assembly through the real CLI, with text on stdin and a WAV
on stdout. The same sentence and reference gives 216,960 samples / 4.52 s at
speed 1 and 109,440 samples / 2.28 s at speed 2 (40 ms latent-frame rounding).
With one warmup and two measured requests, full preparation-to-first-audio is
128–133 ms and preparation-to-complete-audio 241–248 ms at speed 1. These are
product-path smoke measurements and are not substituted into the matched
whole-decode tables above.

## Reproduction

```sh
cargo build --release --features cli,worker,http-server,profile \
  --bin irodori --bin validate_synthesis --bin bench_v4_residency

uv run scripts/export_meanflow_oracle.py --checkpoint MODEL.safetensors \
  --codec ORIGINAL_CODEC_WEIGHTS.pth --output oracle.safetensors \
  --precision fp32 --text 'こんにちは。' --frames 50 --steps 4

target/release/validate_synthesis --checkpoint MODEL.safetensors \
  --codec DECODER.safetensors --fixture oracle.safetensors \
  --precision fp32 --cache-policy full --warmups 2 --repeat 5 --output NEW_BURN_DIR

uv run scripts/prepare_synthesis_comparison.py oracle.safetensors NEW_INPUT_DIR
# Build bench_speech_cpp against the external pinned clone as in the preceding report.
# Convert MF or RF with its reference/irodori-tts/convert.py, then optionally:
# "$speech_root/build/speech" quantize MODEL-F32.gguf MODEL-F16.gguf --type f16
GGML_VK_VISIBLE_DEVICES=0 GGML_VK_DISABLE_F16=1 \
  GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1 \
  ./bench_speech_cpp MODEL-F32.gguf NEW_INPUT_DIR NEW_CPP_DIR 7 2 Vulkan0 oracle
uv run scripts/summarize_synthesis_comparison.py NEW_INPUT_DIR NEW_CPP_DIR

uv run scripts/smoke_synthesis_interfaces.py --binary target/release/irodori \
  --checkpoint MODEL.safetensors --codec DECODER.safetensors \
  --voice VOICE.safetensors --precision fp16 --long-text --output NEW_INTERFACE_DIR
uv run scripts/smoke_synthesis_cli.py --binary target/release/irodori \
  --checkpoint MODEL.safetensors --codec DECODER.safetensors \
  --voice VOICE.safetensors --precision fp16 --output NEW_CLI_DIR
```

The oracle scripts pin the external Python implementation and its compatible
torch dependencies in uv metadata. The local campaign used the pinned clone's
existing uv reference project with `--no-sync`.

## Validation and failure record

The selected config/inference/model/RF/weight/route suite ran 222 tests. 216
passed; six metadata/shape-only tests passed their assertions and then crashed
in NVIDIA's native driver during process exit. Restricting WGPU enumeration to
Vulkan did not fix it, so that experiment was reverted. The short tests did not
wait for asynchronous initialization/forward work; explicit device completion
was added. All eight affected/adjacent tests then passed, including all six
previous failures. The original logs and GDB output are retained; this is not
reported as an uninterrupted 222-test pass.

The final targeted MF/cache/frontend/transport/kernel suite passed **25/25**.
The precision/tokenizer suite, including explicitly selected ignored GPU and
released-tokenizer tests, passed **11/11**. It checks fused FP16 arithmetic,
FP32 RoPE inputs and every actual tokenizer ID. GPU tests used nextest with
`--test-threads 1 --no-tests fail`.
Two additional shape tests passed after adding completion at their exit.
The final transport/request regression selection passed **5/5** after the
thread-lifecycle changes. Its nextest selection is retained in
`final-transport-test-list.log` and results in `final-transport-tests.log`.
Doctests completed with zero executed and four existing examples ignored;
the real compiled CLI and process tests supply the integration evidence.

Feature checks cover default, no-default core, audio-io, codec, inference,
text-normalization, synthesis, worker, http-server, lora and synthesis+lora.
`cargo clippy --all-features --all-targets -- -D warnings` passes, including
the opt-in fusion consumer. Cargo metadata and reverse-dependency trees cover
the workspace root and the existing patched GPU crates. Some minimal builds
emit private dead-code warnings for disabled consumers; they compile without
pulling in the optional transports.
