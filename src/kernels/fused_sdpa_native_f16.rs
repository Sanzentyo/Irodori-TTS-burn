//! Native-only tiled FlashAttention SDPA — f16 Q/K/V storage + f32 accumulation.
//!
//! Variant of [`fused_sdpa_native`] that reads Q/K/V buffers as f16 (half the
//! global memory bandwidth) while all arithmetic uses f32 for precision.
//!
//! # Design
//!
//! - **Inputs**: Q, K, V as `DType::F16` CubeTensors; mask as `DType::F32`
//! - **Shared memory**: Q/K/V tiles stored as f16 (47% less shared memory)
//! - **Computation**: f16 → f32 upcast before FMA; dot products in f32
//! - **Softmax state**: f32 throughout (precision-critical)
//! - **Output**: `DType::F32` (v1 — simpler and avoids downstream f16 complexity)
//!
//! # Expected performance
//!
//! DX12/RTX 5070 Ti estimate: N32×8 improves from 1.46× burn → ~1.0× burn.
//! Metal/M4 Pro: not expected to improve beyond 2.84× burn (architecture gap).
//!
//! # Requirements
//!
//! - GPU must support `shader-f16` (`wgpu::Features::SHADER_F16`)
//! - head_dim must be a power of 2
//! - DX12, Vulkan, Metal — WebGPU: NOT supported (requires `enable f16;`)

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime, into_contiguous,
};
use burn::tensor::{DType, Shape};
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

use super::fused_sdpa_native::NativeFaConfig;

/// Shared memory padding constant (same as f32 variant).
const PAD: u32 = 1;

/// Maximum shared memory per workgroup on native backends.
const MAX_NATIVE_SHARED_BYTES: u32 = 48 * 1024;

/// f16-storage native tiled FA kernel with baked-in dimensions.
#[derive(Debug)]
struct NativeFaF16Kernel {
    tile_q: u32,
    tile_kv: u32,
    head_dim: u32,
    d_padded: u32,
    seq_q: u32,
    seq_kv: u32,
    num_heads: u32,
    workgroup_size: u32,
    dims_per_thread: u32,
    q_tile_size: u32,
    kv_tile_size: u32,
    scores_size: u32,
    log2_d: u32,
    d_mask: u32,
    scale: f64,
}

impl NativeFaF16Kernel {
    fn new(
        config: &NativeFaConfig,
        head_dim: u32,
        seq_q: u32,
        seq_kv: u32,
        num_heads: u32,
        scale: f64,
    ) -> Self {
        assert!(
            head_dim.is_power_of_two(),
            "head_dim ({head_dim}) must be a power of 2"
        );
        assert!(
            head_dim.is_multiple_of(config.tile_kv),
            "head_dim ({head_dim}) must be divisible by tile_kv ({})",
            config.tile_kv
        );

        let workgroup_size = config.tile_q * config.tile_kv;
        let d_padded = head_dim + PAD;
        let dims_per_thread = head_dim / config.tile_kv;
        let q_tile_size = config.tile_q * d_padded;
        let kv_tile_size = config.tile_kv * d_padded;
        let scores_size = config.tile_q * config.tile_kv;

        let log2_d = head_dim.trailing_zeros();
        let d_mask = head_dim - 1;

        // Shared memory: f16 tiles (2 bytes) + f32 scores/state (4 bytes)
        let shared_bytes = (q_tile_size + kv_tile_size) * 2 + (scores_size + 3 * config.tile_q) * 4;
        assert!(
            shared_bytes <= MAX_NATIVE_SHARED_BYTES,
            "shared memory ({shared_bytes} B) exceeds {MAX_NATIVE_SHARED_BYTES} B native limit"
        );

        if shared_bytes > 16384 {
            eprintln!(
                "native_fa_f16: shared memory = {shared_bytes} B (> 16384 B WebGPU limit) — native backends only"
            );
        }

        Self {
            tile_q: config.tile_q,
            tile_kv: config.tile_kv,
            head_dim,
            d_padded,
            seq_q,
            seq_kv,
            num_heads,
            workgroup_size,
            dims_per_thread,
            q_tile_size,
            kv_tile_size,
            scores_size,
            log2_d,
            d_mask,
            scale,
        }
    }
}

impl KernelSource for NativeFaF16Kernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("fused_sdpa_native_f16.wgsl"))
            .register("tile_q", self.tile_q.to_string())
            .register("tile_kv", self.tile_kv.to_string())
            .register("head_dim", self.head_dim.to_string())
            .register("d_padded", self.d_padded.to_string())
            .register("seq_q", self.seq_q.to_string())
            .register("seq_kv", self.seq_kv.to_string())
            .register("num_heads", self.num_heads.to_string())
            .register("scale", format!("{:.10e}", self.scale))
            .register("workgroup_size", self.workgroup_size.to_string())
            .register("dims_per_thread", self.dims_per_thread.to_string())
            .register("q_tile_size", self.q_tile_size.to_string())
            .register("kv_tile_size", self.kv_tile_size.to_string())
            .register("scores_size", self.scores_size.to_string())
            .register("log2_d", self.log2_d.to_string())
            .register("d_mask", self.d_mask.to_string())
    }

    fn id(&self) -> KernelId {
        // Include "f16in" discriminator to avoid cache collisions with the f32 variant.
        KernelId::new::<Self>().info((
            self.tile_q,
            self.tile_kv,
            self.head_dim,
            self.seq_q,
            self.seq_kv,
            self.num_heads,
            self.scale.to_bits(),
            "f16in_f32out",
        ))
    }
}

/// Launch native-only tiled FlashAttention SDPA with f16 Q/K/V inputs.
///
/// # Arguments
/// - `q`: `[B, H, S_Q, D]` query tensor (contiguous f16)
/// - `k`: `[B, H, S_KV, D]` key tensor (contiguous f16)
/// - `v`: `[B, H, S_KV, D]` value tensor (contiguous f16)
/// - `mask`: `[B, S_KV]` mask tensor as f32 (1.0 = attend, 0.0 = mask-out)
/// - `scale`: attention scaling factor (typically 1/√D)
/// - `config`: native tile-size configuration
///
/// # Returns
/// Output tensor `[B, H, S_Q, D]` in f32.
///
/// # Panics
/// - If Q/K/V dtype is not F16 or mask dtype is not F32
/// - If head_dim is not a power of 2
/// - If shared memory would exceed the 48 KB native limit
pub fn native_fa_sdpa_f16_in(
    q: CubeTensor<WgpuRuntime>,
    k: CubeTensor<WgpuRuntime>,
    v: CubeTensor<WgpuRuntime>,
    mask: CubeTensor<WgpuRuntime>,
    scale: f64,
    config: &NativeFaConfig,
) -> CubeTensor<WgpuRuntime> {
    for (name, tensor) in [("q", &q), ("k", &k), ("v", &v)] {
        assert_eq!(
            tensor.dtype,
            DType::F16,
            "native FA f16 kernel requires f16 {name} (got {:?})",
            tensor.dtype
        );
    }
    assert_eq!(
        mask.dtype,
        DType::F32,
        "native FA f16 kernel requires f32 mask (got {:?})",
        mask.dtype
    );

    let q = into_contiguous(q);
    let k = into_contiguous(k);
    let v = into_contiguous(v);
    let mask = into_contiguous(mask);

    assert_eq!(q.meta.num_dims(), 4, "Q must be 4D [B, H, S_Q, D]");
    assert_eq!(k.meta.num_dims(), 4, "K must be 4D [B, H, S_KV, D]");
    assert_eq!(v.meta.num_dims(), 4, "V must be 4D [B, H, S_KV, D]");
    assert_eq!(mask.meta.num_dims(), 2, "mask must be 2D [B, S_KV]");

    let batch = q.meta.shape()[0];
    let num_heads = q.meta.shape()[1];
    let seq_q = q.meta.shape()[2];
    let head_dim = q.meta.shape()[3];
    let seq_kv = k.meta.shape()[2];

    assert_eq!(k.meta.shape()[0], batch, "K batch mismatch");
    assert_eq!(k.meta.shape()[1], num_heads, "K heads mismatch");
    assert_eq!(k.meta.shape()[3], head_dim, "K head_dim mismatch");
    assert_eq!(v.meta.shape()[0], batch, "V batch mismatch");
    assert_eq!(v.meta.shape()[1], num_heads, "V heads mismatch");
    assert_eq!(v.meta.shape()[2], seq_kv, "V seq_kv mismatch");
    assert_eq!(v.meta.shape()[3], head_dim, "V head_dim mismatch");
    assert_eq!(mask.meta.shape()[0], batch, "mask batch mismatch");
    assert_eq!(mask.meta.shape()[1], seq_kv, "mask seq_kv mismatch");

    let client = q.client.clone();
    let device = q.device.clone();

    let kernel = NativeFaF16Kernel::new(
        config,
        head_dim as u32,
        seq_q as u32,
        seq_kv as u32,
        num_heads as u32,
        scale,
    );

    let num_q_tiles = (seq_q as u32).div_ceil(config.tile_q);
    let total_workgroups = (batch * num_heads) as u32 * num_q_tiles;
    let cube_count = CubeCount::new_1d(total_workgroups);
    let cube_dim = CubeDim::new_1d(kernel.workgroup_size);

    // Output is f32 (4 bytes per element)
    let output_elems = batch * num_heads * seq_q * head_dim;
    let output_handle = client.empty(output_elems * core::mem::size_of::<f32>());
    let output = CubeTensor::new_contiguous(
        client.clone(),
        device,
        Shape::from([batch, num_heads, seq_q, head_dim]),
        output_handle,
        DType::F32,
    );

    let kernel_box: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(kernel, cube_dim));

    let bindings = KernelArguments::new()
        .with_buffer(q.handle.binding())
        .with_buffer(k.handle.binding())
        .with_buffer(v.handle.binding())
        .with_buffer(mask.handle.binding())
        .with_buffer(output.handle.clone().binding());

    client.launch(kernel_box, cube_count, bindings);

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup};
    use burn::tensor::Tensor;
    use half::f16;

    type WgpuRaw = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;

    fn setup_device() -> <WgpuRaw as burn::tensor::backend::Backend>::Device {
        let device = WgpuDevice::DefaultDevice;
        init_setup::<AutoGraphicsApi>(&device, Default::default());
        device
    }

    struct TestShape {
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_kv: usize,
        head_dim: usize,
    }

    /// CPU reference SDPA operating on (f16-quantized) f32 inputs.
    #[allow(clippy::needless_range_loop)]
    fn reference_sdpa(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: &[f32],
        shape: &TestShape,
        scale: f64,
    ) -> Vec<f32> {
        let TestShape {
            batch,
            heads,
            seq_q,
            seq_kv,
            head_dim,
        } = *shape;
        let mut output = vec![0.0f32; batch * heads * seq_q * head_dim];
        let scale = scale as f32;

        for b in 0..batch {
            for h in 0..heads {
                for qi in 0..seq_q {
                    let q_base = ((b * heads + h) * seq_q + qi) * head_dim;
                    let mut scores = vec![0.0f32; seq_kv];

                    for kj in 0..seq_kv {
                        let k_base = ((b * heads + h) * seq_kv + kj) * head_dim;
                        let dot: f32 = (0..head_dim).map(|d| q[q_base + d] * k[k_base + d]).sum();
                        scores[kj] = dot * scale;
                    }

                    for kj in 0..seq_kv {
                        if mask[b * seq_kv + kj] < 0.5 {
                            scores[kj] = f32::NEG_INFINITY;
                        }
                    }

                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_s: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
                    let sum_exp: f32 = exp_s.iter().sum();

                    let out_base = ((b * heads + h) * seq_q + qi) * head_dim;
                    if sum_exp > 0.0 {
                        for d in 0..head_dim {
                            let acc: f32 = (0..seq_kv)
                                .map(|kj| {
                                    let v_base = ((b * heads + h) * seq_kv + kj) * head_dim;
                                    exp_s[kj] * v[v_base + d]
                                })
                                .sum();
                            output[out_base + d] = acc / sum_exp;
                        }
                    }
                }
            }
        }
        output
    }

    fn run_parity_test(shape: &TestShape, mask_data: &[f32], config: &NativeFaConfig, tol: f32) {
        let device = setup_device();
        let scale = (shape.head_dim as f64).powf(-0.5);

        let total_q = shape.batch * shape.heads * shape.seq_q * shape.head_dim;
        let total_kv = shape.batch * shape.heads * shape.seq_kv * shape.head_dim;

        let q_data: Vec<f32> = (0..total_q)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();
        let k_data: Vec<f32> = (0..total_kv)
            .map(|i| (i as f32 * 0.013 + 1.0).cos() * 0.5)
            .collect();
        let v_data: Vec<f32> = (0..total_kv)
            .map(|i| (i as f32 * 0.007 + 2.0).sin() * 0.5)
            .collect();

        // Reference uses the same f16-quantized inputs the kernel will see.
        let q_q: Vec<f32> = q_data.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
        let k_q: Vec<f32> = k_data.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
        let v_q: Vec<f32> = v_data.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

        let expected = reference_sdpa(&q_q, &k_q, &v_q, mask_data, shape, scale);

        // Bootstrap client from Q data (also initialises the wgpu device context).
        let dummy = Tensor::<WgpuRaw, 1>::from_floats(q_data.as_slice(), &device);
        let dummy_prim = dummy.into_primitive().tensor();
        let client = dummy_prim.client.clone();
        let dev = dummy_prim.device.clone();
        drop(dummy_prim);

        // Helper: build f16 CubeTensor from f32 host data.
        let make_f16 = |data: &[f32], s: Shape| {
            let data_f16: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
            let bytes: &[u8] = bytemuck::cast_slice(&data_f16);
            let handle = client.create_from_slice(bytes);
            CubeTensor::new_contiguous(client.clone(), dev.clone(), s, handle, DType::F16)
        };

        let q_t = make_f16(
            &q_data,
            Shape::from([shape.batch, shape.heads, shape.seq_q, shape.head_dim]),
        );
        let k_t = make_f16(
            &k_data,
            Shape::from([shape.batch, shape.heads, shape.seq_kv, shape.head_dim]),
        );
        let v_t = make_f16(
            &v_data,
            Shape::from([shape.batch, shape.heads, shape.seq_kv, shape.head_dim]),
        );

        let mask_t = Tensor::<WgpuRaw, 1>::from_floats(mask_data, &device)
            .reshape([shape.batch, shape.seq_kv]);
        let mask_prim = mask_t.into_primitive().tensor();

        let output_prim = native_fa_sdpa_f16_in(q_t, k_t, v_t, mask_prim, scale, config);
        let output_tensor =
            Tensor::<WgpuRaw, 4>::from_primitive(burn::tensor::TensorPrimitive::Float(output_prim));
        let got = output_tensor.into_data().to_vec::<f32>().unwrap();

        let mut max_diff = 0.0f32;
        for (i, (&g, &want)) in got.iter().zip(expected.iter()).enumerate() {
            let diff = (g - want).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < tol,
                "index {i}: got {g:.6}, expected {want:.6}, diff {diff:.2e}"
            );
        }
        eprintln!(
            "native_fa_f16 (Q{}×KV{}): B={} H={} SQ={} SKV={} D={} → max_diff={max_diff:.2e}",
            config.tile_q,
            config.tile_kv,
            shape.batch,
            shape.heads,
            shape.seq_q,
            shape.seq_kv,
            shape.head_dim,
        );
    }

    // --- Q32×KV8 ---

    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn native_fa_f16_small_32x8() {
        let shape = TestShape {
            batch: 1,
            heads: 2,
            seq_q: 4,
            seq_kv: 6,
            head_dim: 128,
        };
        let mask = vec![1.0f32; shape.batch * shape.seq_kv];
        run_parity_test(&shape, &mask, &NativeFaConfig::Q32_KV8, 1e-3);
    }

    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn native_fa_f16_masked_32x8() {
        let shape = TestShape {
            batch: 1,
            heads: 2,
            seq_q: 4,
            seq_kv: 8,
            head_dim: 128,
        };
        let mask = vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        run_parity_test(&shape, &mask, &NativeFaConfig::Q32_KV8, 1e-3);
    }

    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn native_fa_f16_model_dim_32x8() {
        let shape = TestShape {
            batch: 1,
            heads: 16,
            seq_q: 32,
            seq_kv: 48,
            head_dim: 128,
        };
        let mask = vec![1.0f32; shape.batch * shape.seq_kv];
        run_parity_test(&shape, &mask, &NativeFaConfig::Q32_KV8, 1e-3);
    }

    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn native_fa_f16_edge_not_divisible_32x8() {
        let shape = TestShape {
            batch: 1,
            heads: 1,
            seq_q: 17,
            seq_kv: 9,
            head_dim: 128,
        };
        let mask = vec![1.0f32; shape.batch * shape.seq_kv];
        run_parity_test(&shape, &mask, &NativeFaConfig::Q32_KV8, 1e-3);
    }

    // --- Q16×KV16 ---

    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn native_fa_f16_small_16x16() {
        let shape = TestShape {
            batch: 1,
            heads: 2,
            seq_q: 4,
            seq_kv: 6,
            head_dim: 128,
        };
        let mask = vec![1.0f32; shape.batch * shape.seq_kv];
        run_parity_test(&shape, &mask, &NativeFaConfig::Q16_KV16, 1e-3);
    }

    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn native_fa_f16_model_dim_16x16() {
        let shape = TestShape {
            batch: 1,
            heads: 16,
            seq_q: 32,
            seq_kv: 48,
            head_dim: 128,
        };
        let mask = vec![1.0f32; shape.batch * shape.seq_kv];
        run_parity_test(&shape, &mask, &NativeFaConfig::Q16_KV16, 1e-3);
    }
}
