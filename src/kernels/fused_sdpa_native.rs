//! Native-only tiled FlashAttention SDPA WGSL kernel.
//!
//! Optimised for DX12/Vulkan/Metal backends — uses >16 KB shared memory and
//! WG_SIZE decoupled from HEAD_DIM, allowing larger tile configurations that
//! reduce global memory traffic.
//!
//! Key improvements over `fused_sdpa_tiled`:
//! - WG_SIZE = TILE_Q × TILE_KV (not constrained to equal HEAD_DIM)
//! - Cooperative loads use bit-shift addressing (>> / &) for power-of-2 HEAD_DIM
//! - 4-way ILP-unrolled dot products
//! - Supports larger shared memory budgets (up to ~48 KB)
//!
//! WebGPU fallback: use `fused_sdpa_tiled` (13 KB shared, WG_SIZE=128).

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime, into_contiguous,
};
use burn::tensor::Shape;
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

/// Tile-size configuration for native-only tiled FlashAttention.
///
/// WG_SIZE = tile_q × tile_kv (no longer constrained to equal head_dim).
/// head_dim must be a power of 2 for bit-shift addressing.
#[derive(Debug, Clone, Copy)]
pub struct NativeFaConfig {
    /// Number of query rows per workgroup tile.
    pub tile_q: u32,
    /// Number of key/value rows per streaming tile.
    pub tile_kv: u32,
}

impl NativeFaConfig {
    /// 16 query rows × 8 KV rows, WG_SIZE=128.
    /// Same as TiledFaConfig::Q16_KV8 — used for debugging/comparison only.
    /// Shared memory: ~13 KB (fits WebGPU limit).
    pub const Q16_KV8: Self = Self {
        tile_q: 16,
        tile_kv: 8,
    };

    /// 32 query rows × 8 KV rows, WG_SIZE=256.
    /// 2× fewer K/V loads per Q tile vs 16×8 — best for compute-bound SDPA.
    /// Shared memory: ~22 KB (native-only).
    pub const Q32_KV8: Self = Self {
        tile_q: 32,
        tile_kv: 8,
    };

    /// 16 query rows × 16 KV rows, WG_SIZE=256.
    /// Halves the number of KV blocks → fewer barriers, better if barrier-bound.
    /// Shared memory: ~18 KB (native-only).
    pub const Q16_KV16: Self = Self {
        tile_q: 16,
        tile_kv: 16,
    };

    /// 32 query rows × 16 KV rows, WG_SIZE=512.
    /// Aggressive — high occupancy risk, but fewest barriers and best traffic.
    /// Shared memory: ~27 KB (native-only).
    pub const Q32_KV16: Self = Self {
        tile_q: 32,
        tile_kv: 16,
    };
}

/// Shared memory padding to avoid bank conflicts.
const PAD: u32 = 1;

/// Maximum shared memory per workgroup on native backends (48 KB).
/// DX12: 32 KB guaranteed, 48 KB typical for NVIDIA/AMD.
/// Vulkan: 16-48 KB depending on vendor (we use 48 KB limit conservatively).
/// Metal: 32 KB.
const MAX_NATIVE_SHARED_BYTES: u32 = 48 * 1024;

/// Native-only tiled FA kernel with baked-in dimensions.
#[derive(Debug)]
struct NativeFaSdpaKernel {
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

impl NativeFaSdpaKernel {
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
            "head_dim ({head_dim}) must be a power of 2 for bit-shift addressing"
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

        // log2 for bit-shift addressing
        let log2_d = head_dim.trailing_zeros();
        let d_mask = head_dim - 1;

        // Verify shared memory fits within native limits
        let shared_bytes = (q_tile_size + kv_tile_size + scores_size + 3 * config.tile_q) * 4;
        assert!(
            shared_bytes <= MAX_NATIVE_SHARED_BYTES,
            "shared memory ({shared_bytes} B) exceeds native {MAX_NATIVE_SHARED_BYTES} B limit"
        );

        // Warn if exceeding WebGPU limit (informational, not an error)
        if shared_bytes > 16384 {
            eprintln!(
                "native_fa: shared memory = {shared_bytes} B (>{} B WebGPU limit) — native backends only",
                16384
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

impl KernelSource for NativeFaSdpaKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("fused_sdpa_native.wgsl"))
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
            .register("elem", "f32")
    }

    fn id(&self) -> KernelId {
        // KernelId::info() REPLACES (not appends), so pack all varying
        // parameters into a single tuple to avoid cache collisions.
        KernelId::new::<Self>().info((
            self.tile_q,
            self.tile_kv,
            self.head_dim,
            self.seq_q,
            self.seq_kv,
            self.num_heads,
            self.scale.to_bits(),
        ))
    }
}

/// Launch native-only tiled FlashAttention SDPA on the WGPU backend.
///
/// Computes `softmax(Q @ K^T * scale + mask) @ V` using a single kernel with
/// larger tiles and workgroups than the WebGPU-portable variant.
///
/// # Arguments
/// - `q`: `[B, H, S_Q, D]` query tensor (contiguous f32)
/// - `k`: `[B, H, S_KV, D]` key tensor (contiguous f32)
/// - `v`: `[B, H, S_KV, D]` value tensor (contiguous f32)
/// - `mask`: `[B, S_KV]` mask tensor as f32 (1.0 = attend, 0.0 = mask-out)
/// - `scale`: attention scaling factor (typically 1/√D)
/// - `config`: native tile-size configuration
///
/// # Returns
/// Output tensor `[B, H, S_Q, D]`.
///
/// # Panics
/// - If tensor shapes don't match expectations
/// - If head_dim is not a power of 2
/// - If shared memory exceeds 48 KB native limit
pub fn native_fa_sdpa_wgsl(
    q: CubeTensor<WgpuRuntime>,
    k: CubeTensor<WgpuRuntime>,
    v: CubeTensor<WgpuRuntime>,
    mask: CubeTensor<WgpuRuntime>,
    scale: f64,
    config: &NativeFaConfig,
) -> CubeTensor<WgpuRuntime> {
    for (name, tensor) in [("q", &q), ("k", &k), ("v", &v), ("mask", &mask)] {
        assert_eq!(
            tensor.dtype,
            burn::tensor::DType::F32,
            "native FA kernel only supports f32 {name}"
        );
    }

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

    let kernel = NativeFaSdpaKernel::new(
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

    let output_elems = batch * num_heads * seq_q * head_dim;
    let output_handle = client.empty(output_elems * core::mem::size_of::<f32>());
    let output = CubeTensor::new_contiguous(
        client.clone(),
        device,
        Shape::from([batch, num_heads, seq_q, head_dim]),
        output_handle,
        q.dtype,
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
    use burn::backend::wgpu::graphics::AutoGraphicsApi;
    use burn::backend::wgpu::{WgpuDevice, init_setup};
    use burn::tensor::Tensor;

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

    /// CPU reference: softmax(Q @ K^T * scale + mask) @ V
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
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_base + d] * k[k_base + d];
                        }
                        scores[kj] = dot * scale;
                    }

                    for kj in 0..seq_kv {
                        if mask[b * seq_kv + kj] < 0.5 {
                            scores[kj] = f32::NEG_INFINITY;
                        }
                    }

                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_scores: Vec<f32> =
                        scores.iter().map(|&s| (s - max_score).exp()).collect();
                    let sum_exp: f32 = exp_scores.iter().sum();

                    let out_base = ((b * heads + h) * seq_q + qi) * head_dim;
                    if sum_exp > 0.0 {
                        for d in 0..head_dim {
                            let mut acc = 0.0f32;
                            for kj in 0..seq_kv {
                                let v_base = ((b * heads + h) * seq_kv + kj) * head_dim;
                                acc += exp_scores[kj] * v[v_base + d];
                            }
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

        let expected = reference_sdpa(&q_data, &k_data, &v_data, mask_data, shape, scale);

        let q_t = Tensor::<WgpuRaw, 1>::from_floats(q_data.as_slice(), &device).reshape([
            shape.batch,
            shape.heads,
            shape.seq_q,
            shape.head_dim,
        ]);
        let k_t = Tensor::<WgpuRaw, 1>::from_floats(k_data.as_slice(), &device).reshape([
            shape.batch,
            shape.heads,
            shape.seq_kv,
            shape.head_dim,
        ]);
        let v_t = Tensor::<WgpuRaw, 1>::from_floats(v_data.as_slice(), &device).reshape([
            shape.batch,
            shape.heads,
            shape.seq_kv,
            shape.head_dim,
        ]);
        let mask_t = Tensor::<WgpuRaw, 1>::from_floats(mask_data, &device)
            .reshape([shape.batch, shape.seq_kv]);

        let q_prim = q_t.into_primitive().tensor();
        let k_prim = k_t.into_primitive().tensor();
        let v_prim = v_t.into_primitive().tensor();
        let mask_prim = mask_t.into_primitive().tensor();

        let output_prim = native_fa_sdpa_wgsl(q_prim, k_prim, v_prim, mask_prim, scale, config);
        let output_tensor =
            Tensor::<WgpuRaw, 4>::from_primitive(burn::tensor::TensorPrimitive::Float(output_prim));
        let output_data = output_tensor.into_data().to_vec::<f32>().unwrap();

        let mut max_diff = 0.0f32;
        for (i, (&got, &want)) in output_data.iter().zip(expected.iter()).enumerate() {
            let diff = (got - want).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < tol,
                "index {i}: got {got:.6}, want {want:.6}, diff {diff:.2e}"
            );
        }
        eprintln!(
            "native_fa ({}x{}): B={} H={} SQ={} SKV={} D={} → max_diff={max_diff:.2e}",
            config.tile_q,
            config.tile_kv,
            shape.batch,
            shape.heads,
            shape.seq_q,
            shape.seq_kv,
            shape.head_dim,
        );
    }

    // ---- Q32×KV8 configuration tests (WG_SIZE=256) ----

    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn native_fa_small_32x8() {
        let shape = TestShape {
            batch: 1,
            heads: 2,
            seq_q: 4,
            seq_kv: 6,
            head_dim: 128,
        };
        let mask = vec![1.0f32; shape.batch * shape.seq_kv];
        run_parity_test(&shape, &mask, &NativeFaConfig::Q32_KV8, 1e-5);
    }

    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn native_fa_masked_32x8() {
        let shape = TestShape {
            batch: 1,
            heads: 2,
            seq_q: 4,
            seq_kv: 8,
            head_dim: 128,
        };
        let mask = vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        run_parity_test(&shape, &mask, &NativeFaConfig::Q32_KV8, 1e-5);
    }

    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn native_fa_model_dim_32x8() {
        let shape = TestShape {
            batch: 1,
            heads: 16,
            seq_q: 32,
            seq_kv: 48,
            head_dim: 128,
        };
        let mask = vec![1.0f32; shape.batch * shape.seq_kv];
        run_parity_test(&shape, &mask, &NativeFaConfig::Q32_KV8, 1e-5);
    }

    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn native_fa_edge_not_divisible_32x8() {
        // S_Q=17 not divisible by TILE_Q=32, S_KV=9 not divisible by TILE_KV=8
        let shape = TestShape {
            batch: 1,
            heads: 1,
            seq_q: 17,
            seq_kv: 9,
            head_dim: 128,
        };
        let mask = vec![1.0f32; shape.batch * shape.seq_kv];
        run_parity_test(&shape, &mask, &NativeFaConfig::Q32_KV8, 1e-5);
    }

    // ---- Q16×KV16 configuration tests (WG_SIZE=256) ----

    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn native_fa_small_16x16() {
        let shape = TestShape {
            batch: 1,
            heads: 2,
            seq_q: 4,
            seq_kv: 6,
            head_dim: 128,
        };
        let mask = vec![1.0f32; shape.batch * shape.seq_kv];
        run_parity_test(&shape, &mask, &NativeFaConfig::Q16_KV16, 1e-5);
    }

    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn native_fa_model_dim_16x16() {
        let shape = TestShape {
            batch: 1,
            heads: 16,
            seq_q: 32,
            seq_kv: 48,
            head_dim: 128,
        };
        let mask = vec![1.0f32; shape.batch * shape.seq_kv];
        run_parity_test(&shape, &mask, &NativeFaConfig::Q16_KV16, 1e-5);
    }

    // ---- Q32×KV16 configuration tests (WG_SIZE=512) ----

    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn native_fa_small_32x16() {
        let shape = TestShape {
            batch: 1,
            heads: 2,
            seq_q: 4,
            seq_kv: 6,
            head_dim: 128,
        };
        let mask = vec![1.0f32; shape.batch * shape.seq_kv];
        run_parity_test(&shape, &mask, &NativeFaConfig::Q32_KV16, 1e-5);
    }

    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn native_fa_model_dim_32x16() {
        let shape = TestShape {
            batch: 1,
            heads: 16,
            seq_q: 32,
            seq_kv: 48,
            head_dim: 128,
        };
        let mask = vec![1.0f32; shape.batch * shape.seq_kv];
        run_parity_test(&shape, &mask, &NativeFaConfig::Q32_KV16, 1e-5);
    }

    // ---- Q16×KV8 diagnostic test (WG_SIZE=128, same as original tiled FA) ----

    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn native_fa_model_dim_16x8() {
        let shape = TestShape {
            batch: 1,
            heads: 16,
            seq_q: 32,
            seq_kv: 48,
            head_dim: 128,
        };
        let mask = vec![1.0f32; shape.batch * shape.seq_kv];
        run_parity_test(&shape, &mask, &NativeFaConfig::Q16_KV8, 1e-5);
    }

    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually"]
    fn native_fa_edge_16x8() {
        let shape = TestShape {
            batch: 1,
            heads: 1,
            seq_q: 17,
            seq_kv: 9,
            head_dim: 128,
        };
        let mask = vec![1.0f32; shape.batch * shape.seq_kv];
        run_parity_test(&shape, &mask, &NativeFaConfig::Q16_KV8, 1e-5);
    }
}
