# Session: Full Pipeline Test + LoRA .bin Dead Code Fix

**Date**: 2025-04-21  
**Hardware**: Apple M4 Pro Mac Mini (24GB unified memory, Metal GPU)  
**Handed off from**: A6000 Ubuntu machine

## User Request

> 現在、別デバイスでやられていたセッションをあなたは引き継ぐ必要があります。想定通りにモデルなどを配置したうえで、セッションを継続しなさい。a6000のubuntuからm5 mac miniになっています。
> 現在のtaskはwgpu kernelによるwgpu backendの最適化のはずです。とりあえず、サポートされるbackend(CPUを除く)に対するbenchやtestなどを回しつつ、問題がないことを確認し、あったら修正を加えつつ、作業を自律的に進めていきなさい

## Work Done

### 1. LoRA `.bin` Dead Code Fix (`src/lora.rs`)

Identified and removed dead code in adapter weight resolution:
- `SafeTensors::deserialize` cannot parse PyTorch pickle (`.bin`) format
- `find_adapter_weights` would find `.bin` but always fail to deserialize — unreachable success path
- PEFT v0.3+ (2023) uses safetensors exclusively; v0.19.1 adapter is safetensors-only
- Simplified `find_adapter_weights` and `is_lora_adapter_dir` to safetensors-only paths
- All 312 lib tests pass; clippy clean

**Commit**: `fix: remove .bin dead code from LoRA adapter resolution`

### 2. DACVAE Weights Acquisition

Downloaded and converted DACVAE codec weights:
- Model: `Aratako/Semantic-DACVAE-Japanese-32dim` (430MB `.pth` → 409.5MB `.safetensors`)
- Script: `uv run scripts/convert_dacvae_weights.py` (self-contained, no Python repo needed)
- Output: `target/dacvae_weights.safetensors` (255 tensors, 62 weight_norm pairs resolved)

### 3. Full Pipeline Test (text → WAV)

Tested `just pipeline-real --backend wgpu --text "こんにちは"`:
- **Failed**: burn-fusion panic `"Ordering is bigger than operations"` at DACVAE decode
- Root cause: burn-fusion stream scheduler bug with DACVAE's complex graph

Tested `just pipeline-real-raw --text "こんにちは"` (`wgpu-raw`, no fusion):
- **Succeeded** ✅
- TTS rf_time=36,977ms; Codec first-run=459,112ms (autotuning 18 kernel shapes)
- Audio: 30.00s @ 48kHz, 1,440,000 samples written to WAV

### 4. Justfile + Docs Updates

- Added `pipeline-real-raw` recipe (wgpu-raw workaround)
- Updated `docs/benchmarks/m4-pro.md`:
  - Fixed stale LoRA section note (pipeline IS now testable)
  - Added "Full Pipeline Test" section with timing, bug analysis, autotune details
- Committed and pushed

## Key Findings

1. **burn-fusion crashes on DACVAE decode** with `wgpu` backend — upstream bug in 0.21.0-pre.3
2. **`wgpu-raw` (no fusion) is the workaround** — full pipeline works end-to-end on M4 Pro
3. **First-run autotune is slow** (ConvTranspose2d[stride=10] takes ~84s to autotune on Metal)
4. **Subsequent runs will be much faster** — autotune cache persists at `~/.cache/cubecl/`
5. **M4 Pro WGPU benchmark**: Wgpu f16=18,155ms, f32=35,745ms, WgpuRaw=36,451ms (from prior sessions)
