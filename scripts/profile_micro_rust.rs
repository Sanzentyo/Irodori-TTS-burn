//! Micro-benchmark tensor ops to find burn-tch overhead vs Python.
//! Run: cargo run --release --features cli --bin profile_micro 2>&1

use std::time::Instant;

fn main() {
    tch::no_grad_guard();
    let device = tch::Device::Cuda(0);
    
    // Match Python shapes: batch=1, seq=750, dim=1280
    let x = tch::Tensor::randn([1, 750, 1280], (tch::Kind::Float, device));
    let t = tch::Tensor::from_slice(&[0.5f32]).to_device(device);
    
    let n = 10000usize;
    let warmup = 1000usize;
    
    println!("=== Raw tch micro-ops (µs per op) ===");
    
    // 1. shallow_clone
    for _ in 0..warmup { let _ = x.shallow_clone(); }
    tch::Tensor::f_cuda_synchronize(0).unwrap();
    let t0 = Instant::now();
    for _ in 0..n { let _ = x.shallow_clone(); }
    tch::Tensor::f_cuda_synchronize(0).unwrap();
    let us = t0.elapsed().as_nanos() as f64 / n as f64 / 1000.0;
    println!("  shallow_clone: {us:.1} µs/op");
    
    // 2. cat
    for _ in 0..warmup { let _ = tch::Tensor::cat(&[&x, &x, &x], 0); }
    tch::Tensor::f_cuda_synchronize(0).unwrap();
    let t0 = Instant::now();
    for _ in 0..n { let _ = tch::Tensor::cat(&[&x, &x, &x], 0); }
    tch::Tensor::f_cuda_synchronize(0).unwrap();
    let us = t0.elapsed().as_nanos() as f64 / n as f64 / 1000.0;
    println!("  cat([x]*3, 0): {us:.1} µs/op");
    
    // 3. chunk
    let x3 = tch::Tensor::cat(&[&x, &x, &x], 0);
    for _ in 0..warmup { let _ = x3.chunk(3, 0); }
    tch::Tensor::f_cuda_synchronize(0).unwrap();
    let t0 = Instant::now();
    for _ in 0..n { let _ = x3.chunk(3, 0); }
    tch::Tensor::f_cuda_synchronize(0).unwrap();
    let us = t0.elapsed().as_nanos() as f64 / n as f64 / 1000.0;
    println!("  chunk(3, 0): {us:.1} µs/op");
    
    // 4. tensor creation
    for _ in 0..warmup { let _ = tch::Tensor::full(&[1], 0.5, (tch::Kind::Float, device)); }
    tch::Tensor::f_cuda_synchronize(0).unwrap();
    let t0 = Instant::now();
    for _ in 0..n { let _ = tch::Tensor::full(&[1], 0.5, (tch::Kind::Float, device)); }
    tch::Tensor::f_cuda_synchronize(0).unwrap();
    let us = t0.elapsed().as_nanos() as f64 / n as f64 / 1000.0;
    println!("  Tensor::full([1], 0.5): {us:.1} µs/op");
    
    // 5. repeat
    for _ in 0..warmup { let _ = t.repeat(&[3]); }
    tch::Tensor::f_cuda_synchronize(0).unwrap();
    let t0 = Instant::now();
    for _ in 0..n { let _ = t.repeat(&[3]); }
    tch::Tensor::f_cuda_synchronize(0).unwrap();
    let us = t0.elapsed().as_nanos() as f64 / n as f64 / 1000.0;
    println!("  t.repeat(3): {us:.1} µs/op");
    
    // 6. Arithmetic
    let a = tch::Tensor::randn([1, 750, 1280], (tch::Kind::Float, device));
    let b = tch::Tensor::randn([1, 750, 1280], (tch::Kind::Float, device));
    let c = tch::Tensor::randn([1, 750, 1280], (tch::Kind::Float, device));
    for _ in 0..warmup { let _ = &a + &(&a - &b) * 3.0; }
    tch::Tensor::f_cuda_synchronize(0).unwrap();
    let t0 = Instant::now();
    for _ in 0..n { let _ = &a + &(&a - &b) * 3.0; }
    tch::Tensor::f_cuda_synchronize(0).unwrap();
    let us = t0.elapsed().as_nanos() as f64 / n as f64 / 1000.0;
    println!("  a + (a - b) * 3.0: {us:.1} µs/op");
    
    for _ in 0..warmup { let _ = &a + &(&a - &b) * 3.0 + &(&a - &c) * 5.0; }
    tch::Tensor::f_cuda_synchronize(0).unwrap();
    let t0 = Instant::now();
    for _ in 0..n { let _ = &a + &(&a - &b) * 3.0 + &(&a - &c) * 5.0; }
    tch::Tensor::f_cuda_synchronize(0).unwrap();
    let us = t0.elapsed().as_nanos() as f64 / n as f64 / 1000.0;
    println!("  a + (a-b)*3 + (a-c)*5: {us:.1} µs/op");
    
    println!("\n=== Now via burn-tch wrapper ===");
    println!("(Building burn abstraction overhead is the difference)");
}
