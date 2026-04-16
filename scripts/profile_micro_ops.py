# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
# ]
# ///
"""Micro-benchmark tensor operations that differ between Rust/burn and Python."""
import torch
import time

def bench_op(name, fn, n=10000, warmup=1000):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        fn()
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) / n * 1000
    print(f"  {name}: {us:.1f} µs/op")
    return us

def main():
    device = torch.device("cuda")
    # Shapes matching inference: batch=1, seq=750, dim=1280
    x = torch.randn(1, 750, 1280, device=device)
    t = torch.tensor([0.5], device=device)
    
    print("=== Tensor operations (µs per op) ===")
    print(f"Shape: {x.shape}, dtype: {x.dtype}")
    
    # 1. shallow clone (equivalent to burn .clone())
    bench_op("shallow_clone (x)", lambda: x.clone())  # Actually, Python .clone() is a DEEP copy
    
    # More accurate: what does Python actually do? It passes references.
    # The equivalent in PyTorch C++ of burn's shallow_clone is just binding:
    bench_op("tensor_alias (x)", lambda: x)  # No-op, just reference
    bench_op("tensor_view (x.view(...))", lambda: x.view(1, 750, 1280))
    
    # 2. cat for CFG batching
    bench_op("cat([x]*3, dim=0)", lambda: torch.cat([x, x, x], dim=0))
    
    # 3. chunk
    x3 = torch.cat([x, x, x], dim=0)
    bench_op("chunk(3, dim=0)", lambda: x3.chunk(3, dim=0))
    
    # 4. torch.full (Python per-step) vs torch.tensor
    bench_op("torch.full((1,), 0.5)", lambda: torch.full((1,), 0.5, device=device))
    bench_op("torch.tensor([0.5])", lambda: torch.tensor([0.5], device=device))
    bench_op("t.repeat(3)", lambda: t.repeat(3))
    
    # 5. Arithmetic on output
    a = torch.randn(1, 750, 1280, device=device)
    b = torch.randn(1, 750, 1280, device=device)
    c = torch.randn(1, 750, 1280, device=device)
    bench_op("a + (a - b) * 3.0", lambda: a + (a - b) * 3.0)
    bench_op("a + (a - b) * 3.0 + (a - c) * 5.0", lambda: a + (a - b) * 3.0 + (a - c) * 5.0)
    
    # 6. Scalar creation
    bench_op("torch.tensor(0.5, dev)", lambda: torch.tensor(0.5, device=device))
    
    # 7. torch.no_grad context (what Rust does via no_grad_guard)
    bench_op("torch.no_grad().__enter__", lambda: torch.no_grad().__enter__())
    
    # 8. C++ dispatch overhead comparison
    # This measures the overhead of the Python→C++ call for a trivial op
    z = torch.zeros(1, device=device)
    bench_op("z + 0 (trivial dispatch)", lambda: z + 0)
    bench_op("z.item() (GPU→CPU sync)", lambda: z.item())
    
    print("\n=== Per-step overhead estimate (40 steps) ===")
    # In each step: 1 cat, 1 chunk, 1 full/tensor, 2-4 arithmetic ops
    # Plus: 1 forward pass (dominates)

if __name__ == "__main__":
    main()
