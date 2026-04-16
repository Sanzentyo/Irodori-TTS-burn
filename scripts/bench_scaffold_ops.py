# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""Benchmark the per-step scaffold ops that differ between Rust/burn and Python."""
import torch
import time

def bench(name, fn, n=10000, warmup=2000):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / n * 1e6
    print(f"  {name}: {us:.1f} µs")
    return us

def main():
    device = torch.device("cuda")
    # Match inference shapes
    x = torch.randn(1, 750, 1280, device=device)
    v = torch.randn(1, 750, 1280, device=device)
    x3 = torch.cat([x, x, x], dim=0)  # pre-made for chunk
    t_scalar = torch.tensor([0.5], device=device)
    
    print("=== Python scaffold ops (µs per op) ===")
    total = 0.0
    
    # 1. cat for CFG
    total += bench("cat([x]*3, 0)", lambda: torch.cat([x, x, x], dim=0))
    
    # 2. chunk
    total += bench("chunk(3, 0)", lambda: x3.chunk(3, dim=0))
    
    # 3. t.repeat(3) - Python creates tt then repeats
    total += bench("t.repeat(3)", lambda: t_scalar.repeat(3))
    
    # 4. torch.full per-step (Python does this each step)
    total += bench("torch.full((1,), 0.5)", lambda: torch.full((1,), 0.5, device=device))
    
    # 5. CFG arithmetic: v + (v_cond - chunk) * scale (done 2x)
    a = torch.randn(1, 750, 1280, device=device)
    b = torch.randn(1, 750, 1280, device=device)
    total += bench("(a - b) * 3.0", lambda: (a - b) * 3.0) * 2  # 2 CFG dims
    total += bench("v + diff", lambda: v + a) * 2
    
    # 6. Euler step: x + v * dt
    total += bench("x + v * 0.025", lambda: x + v * 0.025)
    
    print(f"\n  TOTAL scaffold: {total:.0f} µs/step")
    print(f"  × 40 steps = {total * 40 / 1000:.1f} ms")

if __name__ == "__main__":
    main()
