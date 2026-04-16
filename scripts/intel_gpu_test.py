# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.10",
# ]
#
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/xpu"]
# ///

"""
Intel GPU (XPU) inference test for Irodori-TTS-burn.

Tests whether the Intel Arc iGPU can run the DiT model via PyTorch XPU backend.
Requires Intel GPU drivers and oneAPI runtime.

Usage:
    uv run scripts/intel_gpu_test.py
"""

import sys
import time

import torch


def check_xpu_availability() -> bool:
    """Check if Intel XPU (GPU) is available."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch built with XPU: {hasattr(torch, 'xpu')}")

    if not hasattr(torch, "xpu"):
        print("ERROR: This PyTorch build does not include XPU support.")
        print("Install PyTorch with XPU: pip install torch --index-url https://download.pytorch.org/whl/xpu")
        return False

    if not torch.xpu.is_available():
        print("ERROR: XPU is not available on this system.")
        print("Possible causes:")
        print("  - Intel GPU drivers not installed")
        print("  - oneAPI runtime not installed")
        print("  - No compatible Intel GPU detected")
        return False

    device_count = torch.xpu.device_count()
    print(f"XPU devices found: {device_count}")
    for i in range(device_count):
        props = torch.xpu.get_device_properties(i)
        print(f"  Device {i}: {props.name}")
        print(f"    Driver version: {props.driver_version}")
        print(f"    Total memory: {props.total_memory / (1024**3):.1f} GB")

    return True


def benchmark_matmul(device: torch.device, size: int = 1024, runs: int = 10) -> float:
    """Benchmark matrix multiplication on the given device."""
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(3):
        _ = torch.matmul(a, b)
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(runs):
        _ = torch.matmul(a, b)
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / runs * 1000

    return elapsed


def main() -> None:
    print("=" * 60)
    print("Intel GPU (XPU) Test for Irodori-TTS-burn")
    print("=" * 60)
    print()

    if not check_xpu_availability():
        print("\nIntel XPU not available. Exiting.")
        sys.exit(1)

    print()
    print("--- Matmul Benchmark (1024x1024, f32) ---")

    # XPU benchmark
    xpu_device = torch.device("xpu:0")
    xpu_time = benchmark_matmul(xpu_device, size=1024, runs=10)
    print(f"  XPU:  {xpu_time:.2f} ms")

    # CPU baseline
    cpu_device = torch.device("cpu")
    cpu_time = benchmark_matmul(cpu_device, size=1024, runs=10)
    print(f"  CPU:  {cpu_time:.2f} ms")

    print(f"  Speedup: {cpu_time / xpu_time:.2f}x")
    print()

    # Try a larger matmul
    print("--- Matmul Benchmark (2048x2048, f32) ---")
    xpu_time_lg = benchmark_matmul(xpu_device, size=2048, runs=5)
    cpu_time_lg = benchmark_matmul(cpu_device, size=2048, runs=5)
    print(f"  XPU:  {xpu_time_lg:.2f} ms")
    print(f"  CPU:  {cpu_time_lg:.2f} ms")
    print(f"  Speedup: {cpu_time_lg / xpu_time_lg:.2f}x")

    print()
    print("Intel XPU test completed successfully!")


if __name__ == "__main__":
    main()
