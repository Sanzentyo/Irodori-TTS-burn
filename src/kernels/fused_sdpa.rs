//! Fused SDPA WGSL kernel for the WGPU backend.
//!
//! Row-streaming online softmax: avoids materializing the S_Q × S_KV score matrix.
//! Single kernel launch replaces Q@K^T → scale → mask → softmax → @V.
//!
//! Workgroup size: 32 threads (one NVIDIA warp), each owning D/32 dimensions.
//! Dispatch: B × H × S_Q workgroups (one per query row per head per batch).
//!
//! Platform compatibility:
//! - Uses workgroup shared memory only (WGSL core).
//! - No subgroup operations — WebGPU forward-compatible.
//! - f32 only; f16 variant would add `enable f16;` + f32 accumulation.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime, into_contiguous,
};
use burn::tensor::Shape;
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

/// Workgroup size — matches NVIDIA warp size for barrier-free reduction.
/// On non-NVIDIA hardware, 32 is still an efficient workgroup size.
const WORKGROUP_SIZE: u32 = 32;

/// Fused SDPA kernel with baked-in dimensions and scale.
///
/// Template parameters determine the compiled shader pipeline:
/// `(head_dim, seq_q, seq_kv, num_heads, dims_per_thread, scale)`.
///
/// Each unique combination compiles a distinct pipeline, cached by `KernelId`.
/// For inference, all dimensions are constant across 40 diffusion steps,
/// so the pipeline is compiled once and reused.
#[derive(Debug)]
struct FusedSdpaKernel {
    head_dim: u32,
    seq_q: u32,
    seq_kv: u32,
    num_heads: u32,
    dims_per_thread: u32,
    scale: f64,
}

impl FusedSdpaKernel {
    fn new(head_dim: u32, seq_q: u32, seq_kv: u32, num_heads: u32, scale: f64) -> Self {
        assert!(
            head_dim.is_multiple_of(WORKGROUP_SIZE),
            "head_dim ({head_dim}) must be divisible by workgroup_size ({WORKGROUP_SIZE})"
        );
        let dims_per_thread = head_dim / WORKGROUP_SIZE;
        Self {
            head_dim,
            seq_q,
            seq_kv,
            num_heads,
            dims_per_thread,
            scale,
        }
    }
}

impl KernelSource for FusedSdpaKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("fused_sdpa.wgsl"))
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
            .register("head_dim", self.head_dim.to_string())
            .register("seq_q", self.seq_q.to_string())
            .register("seq_kv", self.seq_kv.to_string())
            .register("num_heads", self.num_heads.to_string())
            .register("dims_per_thread", self.dims_per_thread.to_string())
            .register("scale", format!("{:.10e}", self.scale))
            .register("elem", "f32")
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(self.head_dim)
            .info(self.seq_q)
            .info(self.seq_kv)
            .info(self.num_heads)
            .info(self.scale.to_bits())
    }
}

/// Launch fused SDPA on the WGPU backend.
///
/// Computes `softmax(Q @ K^T * scale + mask) @ V` in a single kernel launch,
/// avoiding the N×N score matrix materialization.
///
/// # Arguments
/// - `q`: `[B, H, S_Q, D]` query tensor (contiguous f32)
/// - `k`: `[B, H, S_KV, D]` key tensor (contiguous f32)
/// - `v`: `[B, H, S_KV, D]` value tensor (contiguous f32)
/// - `mask`: `[B, S_KV]` mask tensor as f32 (1.0 = attend, 0.0 = mask-out)
/// - `scale`: attention scaling factor (typically 1/√D)
///
/// # Returns
/// Output tensor `[B, H, S_Q, D]`.
///
/// # Panics
/// - If tensor shapes don't match expectations
/// - If `head_dim` is not divisible by the workgroup size (32)
/// - If any tensor is not f32
pub fn fused_sdpa_wgsl(
    q: CubeTensor<WgpuRuntime>,
    k: CubeTensor<WgpuRuntime>,
    v: CubeTensor<WgpuRuntime>,
    mask: CubeTensor<WgpuRuntime>,
    scale: f64,
) -> CubeTensor<WgpuRuntime> {
    // Validate dtypes
    for (name, tensor) in [("q", &q), ("k", &k), ("v", &v), ("mask", &mask)] {
        assert_eq!(
            tensor.dtype,
            burn::tensor::DType::F32,
            "fused SDPA kernel only supports f32 {name}"
        );
    }

    // Ensure contiguity — raw buffer indexing requires dense layout
    let q = into_contiguous(q);
    let k = into_contiguous(k);
    let v = into_contiguous(v);
    let mask = into_contiguous(mask);

    // Validate shapes: Q [B, H, S_Q, D], K/V [B, H, S_KV, D], mask [B, S_KV]
    assert_eq!(q.meta.num_dims(), 4, "Q must be 4D [B, H, S_Q, D]");
    assert_eq!(k.meta.num_dims(), 4, "K must be 4D [B, H, S_KV, D]");
    assert_eq!(v.meta.num_dims(), 4, "V must be 4D [B, H, S_KV, D]");
    assert_eq!(mask.meta.num_dims(), 2, "mask must be 2D [B, S_KV]");

    let batch = q.meta.shape()[0];
    let num_heads = q.meta.shape()[1];
    let seq_q = q.meta.shape()[2];
    let head_dim = q.meta.shape()[3];

    let seq_kv = k.meta.shape()[2];

    // Cross-validate dimensions
    assert_eq!(k.meta.shape()[0], batch, "K batch mismatch");
    assert_eq!(k.meta.shape()[1], num_heads, "K heads mismatch");
    assert_eq!(k.meta.shape()[3], head_dim, "K head_dim mismatch");
    assert_eq!(v.meta.shape()[0], batch, "V batch mismatch");
    assert_eq!(v.meta.shape()[1], num_heads, "V heads mismatch");
    assert_eq!(v.meta.shape()[2], seq_kv, "V seq_kv mismatch");
    assert_eq!(v.meta.shape()[3], head_dim, "V head_dim mismatch");
    assert_eq!(mask.meta.shape()[0], batch, "mask batch mismatch");
    assert_eq!(mask.meta.shape()[1], seq_kv, "mask seq_kv mismatch");

    assert!(
        head_dim.is_multiple_of(WORKGROUP_SIZE as usize),
        "head_dim ({head_dim}) must be divisible by workgroup_size ({WORKGROUP_SIZE})"
    );

    let client = q.client.clone();
    let device = q.device.clone();

    let output_elems = batch * num_heads * seq_q * head_dim;
    let output_handle = client.empty(output_elems * core::mem::size_of::<f32>());
    let output = CubeTensor::new_contiguous(
        client.clone(),
        device,
        Shape::from([batch, num_heads, seq_q, head_dim]),
        output_handle,
        q.dtype,
    );

    // One workgroup per (batch, head, query_position)
    let total_workgroups = (batch * num_heads * seq_q) as u32;
    let cube_count = CubeCount::new_1d(total_workgroups);
    let cube_dim = CubeDim::new_1d(WORKGROUP_SIZE);

    let kernel = FusedSdpaKernel::new(
        head_dim as u32,
        seq_q as u32,
        seq_kv as u32,
        num_heads as u32,
        scale,
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
    use burn::tensor::{Tensor, backend::Backend};

    /// Non-fusion WGPU backend for direct CubeTensor access.
    type WgpuRaw = burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;

    fn setup_device() -> <WgpuRaw as Backend>::Device {
        let device = WgpuDevice::DefaultDevice;
        init_setup::<AutoGraphicsApi>(&device, Default::default());
        device
    }

    struct SdpaShape {
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_kv: usize,
        head_dim: usize,
    }

    /// CPU reference: softmax(Q @ K^T * scale) @ V with masking.
    ///
    /// All inputs are `[B, H, S, D]` layout. mask is `[B, S_KV]` (1.0 = attend).
    #[allow(clippy::needless_range_loop)]
    fn reference_sdpa(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mask: &[f32],
        shape: &SdpaShape,
        scale: f64,
    ) -> Vec<f32> {
        let SdpaShape {
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
                    // Compute scores for this query row
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

                    // Apply mask
                    for kj in 0..seq_kv {
                        if mask[b * seq_kv + kj] < 0.5 {
                            scores[kj] = f32::NEG_INFINITY;
                        }
                    }

                    // Softmax
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_scores: Vec<f32> =
                        scores.iter().map(|&s| (s - max_score).exp()).collect();
                    let sum_exp: f32 = exp_scores.iter().sum();

                    // Weighted sum of V
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
                    // else: all masked → output stays 0.0
                }
            }
        }
        output
    }

    /// Small SDPA test: B=1, H=2, S_Q=4, S_KV=6, D=32 — all positions attend.
    ///
    /// Ignored by default: WGPU device teardown triggers SIGSEGV.
    /// Run manually: `cargo test --all-features fused_sdpa_small -- --ignored --nocapture`
    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually, not in CI"]
    fn fused_sdpa_small() {
        let device = setup_device();

        const B: usize = 1;
        const H: usize = 2;
        const S_Q: usize = 4;
        const S_KV: usize = 6;
        const D: usize = 32;

        let total_q = B * H * S_Q * D;
        let total_kv = B * H * S_KV * D;
        let total_mask = B * S_KV;

        // Deterministic inputs
        let q_data: Vec<f32> = (0..total_q)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();
        let k_data: Vec<f32> = (0..total_kv)
            .map(|i| (i as f32 * 0.013 + 1.0).cos() * 0.5)
            .collect();
        let v_data: Vec<f32> = (0..total_kv)
            .map(|i| (i as f32 * 0.007 + 2.0).sin() * 0.5)
            .collect();
        let mask_data: Vec<f32> = vec![1.0; total_mask]; // all attend

        let shape = SdpaShape {
            batch: B,
            heads: H,
            seq_q: S_Q,
            seq_kv: S_KV,
            head_dim: D,
        };
        let scale = (D as f64).powf(-0.5);
        let expected = reference_sdpa(&q_data, &k_data, &v_data, &mask_data, &shape, scale);

        let q_t =
            Tensor::<WgpuRaw, 1>::from_floats(q_data.as_slice(), &device).reshape([B, H, S_Q, D]);
        let k_t =
            Tensor::<WgpuRaw, 1>::from_floats(k_data.as_slice(), &device).reshape([B, H, S_KV, D]);
        let v_t =
            Tensor::<WgpuRaw, 1>::from_floats(v_data.as_slice(), &device).reshape([B, H, S_KV, D]);
        let mask_t =
            Tensor::<WgpuRaw, 1>::from_floats(mask_data.as_slice(), &device).reshape([B, S_KV]);

        let q_prim = q_t.into_primitive().tensor();
        let k_prim = k_t.into_primitive().tensor();
        let v_prim = v_t.into_primitive().tensor();
        let mask_prim = mask_t.into_primitive().tensor();

        let output_prim = fused_sdpa_wgsl(q_prim, k_prim, v_prim, mask_prim, scale);
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
                diff < 1e-3,
                "index {i}: got {got:.6}, want {want:.6}, diff {diff:.2e}"
            );
        }
        eprintln!("fused_sdpa_small: max_diff = {max_diff:.2e}");
    }

    /// Test with masking: some positions masked out.
    ///
    /// Run: `cargo test --all-features fused_sdpa_masked -- --ignored --nocapture`
    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually, not in CI"]
    fn fused_sdpa_masked() {
        let device = setup_device();

        const B: usize = 1;
        const H: usize = 2;
        const S_Q: usize = 4;
        const S_KV: usize = 8;
        const D: usize = 32;

        let total_q = B * H * S_Q * D;
        let total_kv = B * H * S_KV * D;

        let q_data: Vec<f32> = (0..total_q)
            .map(|i| (i as f32 * 0.02).sin() * 0.3)
            .collect();
        let k_data: Vec<f32> = (0..total_kv)
            .map(|i| (i as f32 * 0.015 + 0.5).cos() * 0.3)
            .collect();
        let v_data: Vec<f32> = (0..total_kv)
            .map(|i| (i as f32 * 0.009 + 1.5).sin() * 0.3)
            .collect();
        // Mask: first 5 positions attend, last 3 masked
        let mask_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];

        let shape = SdpaShape {
            batch: B,
            heads: H,
            seq_q: S_Q,
            seq_kv: S_KV,
            head_dim: D,
        };
        let scale = (D as f64).powf(-0.5);
        let expected = reference_sdpa(&q_data, &k_data, &v_data, &mask_data, &shape, scale);

        let q_t =
            Tensor::<WgpuRaw, 1>::from_floats(q_data.as_slice(), &device).reshape([B, H, S_Q, D]);
        let k_t =
            Tensor::<WgpuRaw, 1>::from_floats(k_data.as_slice(), &device).reshape([B, H, S_KV, D]);
        let v_t =
            Tensor::<WgpuRaw, 1>::from_floats(v_data.as_slice(), &device).reshape([B, H, S_KV, D]);
        let mask_t =
            Tensor::<WgpuRaw, 1>::from_floats(mask_data.as_slice(), &device).reshape([B, S_KV]);

        let q_prim = q_t.into_primitive().tensor();
        let k_prim = k_t.into_primitive().tensor();
        let v_prim = v_t.into_primitive().tensor();
        let mask_prim = mask_t.into_primitive().tensor();

        let output_prim = fused_sdpa_wgsl(q_prim, k_prim, v_prim, mask_prim, scale);
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
                diff < 1e-3,
                "index {i}: got {got:.6}, want {want:.6}, diff {diff:.2e}"
            );
        }
        eprintln!("fused_sdpa_masked: max_diff = {max_diff:.2e}");
    }

    /// Model-size test: B=1, H=16, S_Q=32, S_KV=48, D=128.
    ///
    /// Run: `cargo test --all-features fused_sdpa_model_dim -- --ignored --nocapture`
    #[test]
    #[ignore = "WGPU teardown SIGSEGV — run manually, not in CI"]
    fn fused_sdpa_model_dim() {
        let device = setup_device();

        const B: usize = 1;
        const H: usize = 16;
        const S_Q: usize = 32;
        const S_KV: usize = 48;
        const D: usize = 128;

        let total_q = B * H * S_Q * D;
        let total_kv = B * H * S_KV * D;

        let q_data: Vec<f32> = (0..total_q)
            .map(|i| (i as f32 * 0.003).sin() * 0.1)
            .collect();
        let k_data: Vec<f32> = (0..total_kv)
            .map(|i| (i as f32 * 0.005 + 0.7).cos() * 0.1)
            .collect();
        let v_data: Vec<f32> = (0..total_kv)
            .map(|i| (i as f32 * 0.002 + 1.3).sin() * 0.1)
            .collect();
        // Mask: first 40 attend, last 8 masked
        let mut mask_data = vec![1.0f32; S_KV];
        for m in mask_data[40..].iter_mut() {
            *m = 0.0;
        }

        let shape = SdpaShape {
            batch: B,
            heads: H,
            seq_q: S_Q,
            seq_kv: S_KV,
            head_dim: D,
        };
        let scale = (D as f64).powf(-0.5);
        let expected = reference_sdpa(&q_data, &k_data, &v_data, &mask_data, &shape, scale);

        let q_t =
            Tensor::<WgpuRaw, 1>::from_floats(q_data.as_slice(), &device).reshape([B, H, S_Q, D]);
        let k_t =
            Tensor::<WgpuRaw, 1>::from_floats(k_data.as_slice(), &device).reshape([B, H, S_KV, D]);
        let v_t =
            Tensor::<WgpuRaw, 1>::from_floats(v_data.as_slice(), &device).reshape([B, H, S_KV, D]);
        let mask_t =
            Tensor::<WgpuRaw, 1>::from_floats(mask_data.as_slice(), &device).reshape([B, S_KV]);

        let q_prim = q_t.into_primitive().tensor();
        let k_prim = k_t.into_primitive().tensor();
        let v_prim = v_t.into_primitive().tensor();
        let mask_prim = mask_t.into_primitive().tensor();

        let output_prim = fused_sdpa_wgsl(q_prim, k_prim, v_prim, mask_prim, scale);
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
                diff < 1e-3,
                "index {i}: got {got:.6}, want {want:.6}, diff {diff:.2e}"
            );
        }
        eprintln!("fused_sdpa_model_dim: max_diff = {max_diff:.2e}");
    }
}
