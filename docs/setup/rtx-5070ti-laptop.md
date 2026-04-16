# Device Setup: Windows 11 — RTX 5070 Ti Laptop + Intel Core Ultra 9 275HX

## Hardware

| Component | Details |
|---|---|
| CPU | Intel Core Ultra 9 275HX (Arrow Lake-HX, 24 cores) |
| dGPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12227 MiB VRAM, CC 12.0 (Blackwell) |
| iGPU | Intel Arc Graphics (Xe-LPG, DEV_7D67, ~2 GB shared) |
| NPU | None (Arrow Lake-HX does not include Neural Processing Unit) |
| RAM | Check with: `(Get-WmiObject Win32_PhysicalMemory \| Measure-Object Capacity -Sum).Sum / 1GB` |

## Software Prerequisites

| Tool | Version | Installation |
|---|---|---|
| CUDA Toolkit | 12.9 (V12.9.86) | Pre-installed; verify with `nvcc --version` |
| NVIDIA Driver | 591.44 | Pre-installed |
| Rust | 1.94.0 | `rustup update` |
| uv | 0.10.9+ | `cargo install uv` or `pipx install uv` |
| just | 1.49.0+ | `cargo install just` |
| Git (with bash) | 2.51+ | Git for Windows; ensures `bash.exe` at `C:\Program Files\Git\bin\` |
| Python 3.10 | 3.10.20 | `uv python install 3.10` |

## Directory Layout

```
C:\Users\sanze\git\
├── Irodori-TTS\                   ← Python reference (sibling directory)
│   ├── .venv\                     ← uv-managed Python 3.10 env
│   │   ├── Scripts\               ← Windows: python.exe, pip.exe
│   │   └── Lib\site-packages\torch\lib\  ← libtorch DLLs
│   └── ...
└── Irodori-TTS-burn\              ← This Rust project
    ├── target\
    │   ├── model_converted.safetensors   ← Converted TTS weights (~2 GB)
    │   └── dacvae_weights.safetensors    ← Converted codec weights (~410 MB)
    └── ...
```

## Step-by-Step Setup

### 1. Install Python 3.10 and sync Irodori-TTS

```powershell
uv python install 3.10
cd ..\Irodori-TTS
uv sync
cd ..\Irodori-TTS-burn
```

### 2. Download and convert model weights

```powershell
# TTS model
uvx --from huggingface_hub hf download Aratako/Irodori-TTS-500M-v2 model.safetensors --local-dir target/hf_model
uv run scripts/convert_for_burn.py target/hf_model/model.safetensors target/model_converted.safetensors --apply

# DACVAE codec
uvx --from huggingface_hub hf download Aratako/Semantic-DACVAE-Japanese-32dim weights.pth --local-dir target/dacvae_hf
cd ..\Irodori-TTS
.\.venv\Scripts\python.exe ..\Irodori-TTS-burn\scripts\convert_dacvae_weights.py `
    --pth ..\Irodori-TTS-burn\target\dacvae_hf\weights.pth `
    --output ..\Irodori-TTS-burn\target\dacvae_weights.safetensors
cd ..\Irodori-TTS-burn
```

### 3. Create `.env` file

Create `.env` in the project root:

```
LIBTORCH_USE_PYTORCH=1
LIBTORCH_BYPASS_VERSION_CHECK=1
```

### 4. Set environment for building and running

```powershell
# Required for every new terminal session:
$env:PATH = "C:\Users\sanze\git\Irodori-TTS\.venv\Scripts;C:\Users\sanze\git\Irodori-TTS\.venv\Lib\site-packages\torch\lib;C:\Program Files\Git\bin;$env:PATH"
$env:LIBTORCH_USE_PYTORCH = "1"
$env:LIBTORCH_BYPASS_VERSION_CHECK = "1"
$env:VIRTUAL_ENV = "C:\Users\sanze\git\Irodori-TTS\.venv"
```

### 5. Build and test

```powershell
cargo test --quiet           # 223 tests pass, 1 ignored (WGPU teardown)
cargo clippy --all-targets   # no warnings
cargo build --release --features cli
```

### 6. Run benchmarks

```powershell
# WGPU (uses DirectX 12 on Windows)
cargo run --release --features cli --bin bench_realmodel -- --backend wgpu --warmup 2 --runs 5

# CubeCL CUDA
cargo run --release --features cli --bin bench_realmodel -- --backend cuda --warmup 2 --runs 5
cargo run --release --features cli --bin bench_realmodel -- --backend cuda-bf16 --warmup 2 --runs 5

# LibTorch (requires torch DLLs on PATH)
cargo run --release --features cli --bin bench_realmodel -- --backend libtorch --warmup 2 --runs 5
cargo run --release --features cli --bin bench_realmodel -- --backend libtorch-bf16 --warmup 2 --runs 5

# Python baseline
cd ..\Irodori-TTS
.\.venv\Scripts\python.exe ..\Irodori-TTS-burn\scripts\bench_python.py --dtype f32 --runs 5 --warmup 2
```

## Windows-Specific Notes

### LibTorch DLLs

On Windows, shared libraries are found via `PATH` (not `LD_LIBRARY_PATH`).
Since the binary links against libtorch regardless of which backend is used at runtime,
the torch DLL directory must **always** be on PATH when running any backend.

### WGPU Backend

On Windows, WGPU defaults to **DirectX 12** (not Vulkan). This can give different
performance characteristics vs Linux (Vulkan). The RTX 5070 Ti's WGPU f32 is actually
8% faster than the RTX A6000 on Linux/Vulkan despite being a laptop GPU.

To force Vulkan: set `WGPU_BACKEND=vulkan` before running.

### justfile

The justfile requires bash, which is provided by Git for Windows.
Ensure `C:\Program Files\Git\bin` is on PATH before running `just` commands.
The justfile has been updated for cross-platform path detection (Windows vs Unix).

### Known Issues

- **libtorch DLL not found**: Ensure torch DLL path is on `$env:PATH`
- **`HOME` env var missing**: The justfile uses `USERPROFILE` on Windows (fixed)
- **WGPU exit segfault**: Known issue; does not affect results
- **Thermal throttling**: Laptop GPUs may throttle under sustained load.
  Monitor with `nvidia-smi -lms 1000` in a separate terminal.

## GPU Monitor Commands

```powershell
# One-shot status
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader

# Continuous monitoring (1s interval)
nvidia-smi -lms 1000

# Intel GPU monitoring (requires Intel GPU tools)
# Not currently set up — see Intel GPU section below
```

## Intel Arc iGPU

The Intel Arc Graphics (Xe-LPG) integrated GPU is present but has limited ML support:

- **PyTorch**: Intel Extension for PyTorch (IPEX) supports Intel Arc GPUs via XPU backend
- **Rust/burn**: No native Intel GPU backend; WGPU can potentially target it via Vulkan
- **Status**: Exploration pending — see `docs/setup/intel-gpu-exploration.md` (if created)
