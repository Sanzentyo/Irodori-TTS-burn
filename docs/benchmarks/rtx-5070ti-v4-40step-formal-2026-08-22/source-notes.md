# Source notes: RTX 5070 Ti 40-step formal campaign

## Authority and campaign boundary

The only authoritative dataset for this report is the sealed fresh campaign at:

```text
/home/sanzentyo/benchmark-artifacts/irodori-v4-40step-formal-20260822-attempt2
```

No result from a previous `/tmp` directory, the stopped `attempt1` campaign, or any smoke campaign is pooled into the formal aggregates. `attempt1` was stopped after 15 of 90 session pairs because the original Python report did not expose the exact schedule bits. It remains sealed as a failed audit trail rather than being repaired or reused.

The authoritative campaign contains 1,723 files covered by `SHA256SUMS`. On 2026-08-22, `sha256sum -c SHA256SUMS` verified all 1,723 entries. The SHA256 digest of the checksum manifest itself is:

```text
b5fe310825d3eeaf2b19a2a17460a9c8678c8a94bbc606e11589c9332485ecd3
```

Re-running `scripts/summarize_v4_40step_formal.py` from the sealed raw sessions reproduced `summary.json` and `condition-summary.csv` byte-for-byte.

## Pins

- Measurement source: `8a19782fd3e22017abf04b475c903c058a017af2`
- Upstream source: `9f19d9a9048099a4b978a762d0509228fe624e3f`
- Irodori-TTS-v4-Small revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- Model SHA256: `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- Semantic-DACVAE-Japanese-32dim revision: `47376ee24834d7a05a48ebabfe3cde29b3c5e214`
- Python codec input SHA256: `db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5`
- Converted decoder-only codec SHA256: `1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231`
- Rust benchmark binary SHA256: `8be5d10896334efb750646ec2de9618aa32292944fad685c0a7c012b45ca9b13`
- Frozen Python benchmark SHA256: `95123821ced722ff85691548be3fcfc4db7d7dd1985369f3cc89c609311cbc03`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- Driver: 595.71.05
- Vulkan adapter: WGPU adapter 0
- CUDA/NVML index: 0
- PCI bus ID: `00000000:01:00.0`
- Physical VRAM: 12,227 MiB; initially available: 11,774 MiB

The sealed `pins.sha256` additionally records the exact runner, Rust benchmark source, fixture creator, prepared-reference exporter, and summarizer used by the campaign.

## Measurement grain and boundaries

There are 18 independent conditions: six fixed output lengths (`45`, `112`, `255`, `333`, `489`, and `685` latent frames) by three voice topologies (text-only, voice design, and prepared clone). Each runtime/condition has five fresh processes. Every process performs two warmup requests followed by ten measured requests, for 900 measured requests and 180 warmups per runtime.

The reported condition latency first takes the median of ten measured requests within each process, then the median of five fresh-process medians. The runtime comparison uses the same boundary on both sides:

- `device-complete`: a pre-stage device synchronization through device completion;
- `readback-complete`: device-complete plus an owned contiguous CPU F32 waveform;
- `consumer-complete`: WGPU final audio accepted by the designated consumer;
- first request: recorded separately after model load and before the steady measured set.

This campaign is not a cold external process-launch-to-WAV-close E2E campaign. It also excludes duration prediction from fixed-frame request latency and excludes raw reference encoding from the prepared-clone online request.

## Semantic-work checks

Both runtimes use strict FP32, TF32 off, autocast off, 40 Euler evaluations, independent CFG, and the same fixed FP32 initial noise. The Python and Rust work manifests match in all 90 session pairs:

- 41 exact FP32 schedule bit patterns;
- 40 whole-model forwards per request;
- forward batch vector consistent with each topology;
- 60 effective rows for text-only and 80 for design/clone;
- 12 RF layers and 480 RF block calls.

The two runtimes perform the same semantic work, but not the same operator graph.

## Numerical accuracy policy

The hard numerical gate requires finite samples, waveform SNR at least 80 dB, and cosine similarity at least `0.99999999`. The target additionally requires SNR at least 85 dB and maximum absolute error no greater than `2e-4`.

This is a numerical reproducibility policy, not a listening-test threshold. No auditory preference or detectability study was conducted in this campaign.

## Report chart map

- `Readback-complete latency delta by output duration`: all 18 condition medians; zero is parity and positive values favor PyTorch.
- `Waveform SNR by output duration`: all 18 waveform comparisons with the 80 dB hard and 85 dB target reference lines.
- `NVML peak VRAM by output duration`: the maximum across three voice conditions and five fresh sessions for each runtime/length pair.
- `Exact condition-level results`: the underlying latency, accuracy, and NVML values for every condition.

The widget datasets are projected from the validated JSON rows using the exact SQLite statements embedded in `artifact.json`. The projection is checked to preserve every row without coercion before the artifact is written.

## Data-quality assessment

The data is fit to support the narrow decision reported here: current strict-FP32, 40-step, all-resident Rust/WGPU does not beat the pinned PyTorch/CUDA comparison on this GPU, and four conditions fail the declared numerical hard gate. It is not sufficient to generalize across GPU vendors, operating systems, adapters, random seeds, automatic duration, raw clone preparation, or cold full E2E.

No retry selected a successful condition. No cross-session output nondeterminism or schedule/work-manifest mismatch was observed. Non-empty Python stderr entries were the same `weight_norm` deprecation warning, not runtime errors, panics, OOMs, or tracebacks.

## Portable artifact QA

`report.html` is the single primary report surface and is self-contained. The portable artifact pipeline passed schema validation and packaging. Verification was `structural_only` because no Chromium headless executable is installed in the measurement environment; therefore browser rendering, source-dialog interaction, and viewport screenshots were not claimed as verified.
