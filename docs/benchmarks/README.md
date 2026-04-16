# Benchmark Results Directory

Each file in this directory contains benchmark results for a specific device.

## File Naming Convention

`<gpu-or-cpu-model>.md` — e.g.:
- `rtx-a6000.md` — NVIDIA RTX A6000 (original dev machine)
- `rtx-4090.md` — NVIDIA RTX 4090
- `m3-max.md` — Apple M3 Max (Metal/WGPU backend only)
- `cpu-intel-i9.md` — CPU-only (NdArray backend)

## Protocol: New Device

When resuming work on a **different device**, the following steps are mandatory
before running any optimization work:

1. **Identify the device**:
   ```sh
   # Linux/CUDA:
   nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
   nvcc --version
   # macOS (Metal):
   system_profiler SPDisplaysDataType | grep -E "Chipset|VRAM"
   # All:
   cat /proc/cpuinfo | grep "model name" | head -1   # Linux
   sysctl -n machdep.cpu.brand_string                # macOS
   ```

2. **Check GPU availability** before benchmarking:
   ```sh
   # Linux CUDA:
   nvidia-smi -lms 500  # watch in separate terminal while running bench
   # Or check utilization first:
   nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader
   ```

3. **Create a device-specific benchmark file** by copying the template:
   ```sh
   cp docs/benchmarks/rtx-a6000.md docs/benchmarks/<device-name>.md
   # Edit the System section with actual device specs
   ```

4. **Run all benchmarks** with proper timeout and warmup:
   ```sh
   # GPU benchmarks (adjust timeout per device speed):
   just bench-tch-bf16   # LibTorch bf16 (fastest GPU path)
   just bench-tch        # LibTorch f32
   just bench-cuda       # CubeCL CUDA f32
   just bench-wgpu       # WGPU f32 (fusion)
   just bench-wgpu-f16   # WGPU f16 (if GPU supports shader-f16)
   # Python baseline:
   cd Irodori-TTS && uv run python bench.py
   ```
   Each bench uses `--timeout` internally; also wrap with shell `timeout`.

5. **Compare results** against the reference device (`rtx-a6000.md`) and
   update the new device file with all columns filled.

6. Commit the new benchmark file: `git add docs/benchmarks/<device>.md`

## Key Metrics

- **Mean (ms)**: primary metric for comparison
- **vs Python f32**: normalized to Python baseline on the *same device*
- **RTF**: real-time factor (< 1.0 = faster than real-time audio generation)

## Current Benchmark Files

| File | Device | Best Backend | Best Time |
|------|--------|-------------|-----------|
| [rtx-a6000.md](rtx-a6000.md) | NVIDIA RTX A6000 (49GB) | LibTorch bf16 | 987ms (0.36×) |

## Notes on Invalid Benchmarks

A benchmark should be **discarded** if:
- GPU utilization was already > 10% before the run (occupied device)
- Another training/inference job was running concurrently
- The system was under thermal throttling (check with `nvidia-smi -lms 500`)
- The run timed out mid-sequence

Always verify GPU is idle before running: `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader` should show `0 %` or near-zero.
