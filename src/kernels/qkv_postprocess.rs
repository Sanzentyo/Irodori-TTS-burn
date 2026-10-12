//! Fused QKV post-processing for joint attention on the WGPU backend.
//!
//! The input is the result of either a `Wq || Wk || Wv` projection or the
//! production `Wq || Wk || Wv || Wgate` projection. One dispatch splits it
//! into Q/K/V, applies per-head RMSNorm to Q and K, and applies adjacent-pair
//! RoPE to the first half of the heads. In the combined layout it also writes
//! sigmoid of the gate projection back to the input's final segment.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime, into_contiguous,
};
#[cfg(test)]
use burn::tensor::Device;
use burn::tensor::Shape;
use cubecl::CubeCount;
use cubecl::prelude::KernelId;
use cubecl::server::KernelArguments;

use super::precision::{KernelFloatPrecision, common_float_precision};

const MAX_WORKGROUP_SIZE: u32 = 256;

/// Three independently allocated, contiguous `[B, S, H, Dh]` tensors.
#[derive(Debug)]
pub struct QkvPostprocessOutput {
    /// Normalised and half-RoPE-rotated query tensor.
    pub q: CubeTensor<WgpuRuntime>,
    /// Normalised and half-RoPE-rotated key tensor.
    pub k: CubeTensor<WgpuRuntime>,
    /// Unmodified value tensor split from the fused projection.
    pub v: CubeTensor<WgpuRuntime>,
}

/// Q/K/V outputs plus the in-place sigmoid-transformed combined projection.
#[derive(Debug)]
pub struct QkvGatePostprocessOutput {
    /// Separate contiguous Q/K/V outputs.
    pub qkv: QkvPostprocessOutput,
    /// Contiguous `[B, S, 4 * H * Dh]`; the final `H * Dh` values per token
    /// contain sigmoid(gate), while the first three segments are unchanged.
    pub combined: CubeTensor<WgpuRuntime>,
}

#[derive(Clone, Copy, Debug)]
enum ProjectionLayout {
    Qkv,
    QkvGate,
}

impl ProjectionLayout {
    fn input_width(self, kv_dim: usize) -> usize {
        let segments = match self {
            Self::Qkv => 3,
            Self::QkvGate => 4,
        };
        kv_dim
            .checked_mul(segments)
            .expect("projection input width overflow")
    }

    fn writes_gate(self) -> bool {
        matches!(self, Self::QkvGate)
    }
}

#[derive(Debug)]
struct QkvPostprocessKernel {
    precision: KernelFloatPrecision,
    batch: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    input_width: u32,
    writes_gate: bool,
    workgroup_size: u32,
    eps: f64,
}

impl KernelSource for QkvPostprocessKernel {
    fn source(&self) -> SourceTemplate {
        self.precision
            .source(
                include_str!("qkv_postprocess.wgsl"),
                include_str!("qkv_postprocess_f16.wgsl"),
            )
            .register("seq_len", self.seq_len.to_string())
            .register("num_heads", self.num_heads.to_string())
            .register("head_dim", self.head_dim.to_string())
            .register("half_head_dim", (self.head_dim / 2).to_string())
            .register("kv_dim", (self.num_heads * self.head_dim).to_string())
            .register("input_width", self.input_width.to_string())
            .register("writes_gate", self.writes_gate.to_string())
            .register("workgroup_size", self.workgroup_size.to_string())
            .register("eps", format!("{:e}", self.eps))
    }

    fn id(&self) -> KernelId {
        // `KernelId::info` replaces earlier info, so keep every launch-varying
        // value in one tuple. Including batch is conservative: it affects the
        // dispatch even though buffer addressing only needs S/H/Dh.
        KernelId::new::<Self>().info((
            self.batch,
            self.precision,
            self.seq_len,
            self.num_heads,
            self.head_dim,
            self.input_width,
            self.writes_gate,
            self.workgroup_size,
            self.eps.to_bits(),
        ))
    }
}

/// Split and post-process a fused QKV projection in one WGPU dispatch.
///
/// Inputs:
/// - `fused_qkv`: `[B, S, 3 * H * Dh]`; this is made contiguous once.
/// - `q_weight`, `k_weight`: contiguous `[H, Dh]` RMSNorm weights.
/// - `rope_cos`, `rope_sin`: contiguous `[S, Dh / 2]` adjacent-pair tables.
/// - `eps`: finite, positive RMSNorm epsilon.
///
/// The returned Q/K/V buffers are separate and contiguous with shape
/// `[B, S, H, Dh]`. Q and K are normalised over `Dh`. RoPE is applied only
/// when `head < H / 2`; all V values and the remaining Q/K heads are copied
/// without rotation.
///
/// # Panics
///
/// Panics for an incompatible dtype, shape, device, non-contiguous auxiliary
/// input, invalid epsilon, or any size/address calculation that does not fit
/// the host address space or WGSL's `u32` storage indexing.
pub fn fused_qkv_postprocess_wgsl(
    fused_qkv: CubeTensor<WgpuRuntime>,
    q_weight: CubeTensor<WgpuRuntime>,
    k_weight: CubeTensor<WgpuRuntime>,
    rope_cos: CubeTensor<WgpuRuntime>,
    rope_sin: CubeTensor<WgpuRuntime>,
    eps: f64,
) -> QkvPostprocessOutput {
    qkv_postprocess_wgsl(
        fused_qkv,
        q_weight,
        k_weight,
        rope_cos,
        rope_sin,
        eps,
        ProjectionLayout::Qkv,
    )
    .0
}

/// Post-process combined QKV/gate projection and apply gate sigmoid in place.
///
/// `combined` must be contiguous f32 `[B, S, 4 * H * Dh]`, with Q, K, V,
/// and bias-free gate segments in that order. The returned `combined` tensor
/// owns the same shape and buffer, with only its final segment changed to
/// sigmoid(gate). Q/K/V are returned in separate contiguous buffers.
///
/// This variant keeps the shader at WebGPU's guaranteed eight storage-buffer
/// bindings: the gate is written back into the read-write projection buffer
/// rather than requiring a ninth output binding.
pub fn fused_qkv_gate_postprocess_wgsl(
    combined: CubeTensor<WgpuRuntime>,
    q_weight: CubeTensor<WgpuRuntime>,
    k_weight: CubeTensor<WgpuRuntime>,
    rope_cos: CubeTensor<WgpuRuntime>,
    rope_sin: CubeTensor<WgpuRuntime>,
    eps: f64,
) -> QkvGatePostprocessOutput {
    let (qkv, combined) = qkv_postprocess_wgsl(
        combined,
        q_weight,
        k_weight,
        rope_cos,
        rope_sin,
        eps,
        ProjectionLayout::QkvGate,
    );
    QkvGatePostprocessOutput { qkv, combined }
}

#[allow(clippy::too_many_arguments)]
fn qkv_postprocess_wgsl(
    fused_qkv: CubeTensor<WgpuRuntime>,
    q_weight: CubeTensor<WgpuRuntime>,
    k_weight: CubeTensor<WgpuRuntime>,
    rope_cos: CubeTensor<WgpuRuntime>,
    rope_sin: CubeTensor<WgpuRuntime>,
    eps: f64,
    layout: ProjectionLayout,
) -> (QkvPostprocessOutput, CubeTensor<WgpuRuntime>) {
    let bindings = [
        ("fused_qkv", &fused_qkv),
        ("q_weight", &q_weight),
        ("k_weight", &k_weight),
        ("rope_cos", &rope_cos),
        ("rope_sin", &rope_sin),
    ];
    let precision = common_float_precision(bindings.iter().map(|(_, tensor)| tensor.dtype))
        .expect("fused QKV post-process inputs must share f32 or f16 dtype");
    for (_, tensor) in bindings {
        fused_qkv.assert_is_on_same_device(tensor);
    }
    assert!(
        eps.is_finite() && eps > 0.0 && (eps as f32).is_finite() && (eps as f32) > 0.0,
        "RMSNorm epsilon must be finite, positive, and representable as f32, got {eps}"
    );

    assert_eq!(
        fused_qkv.meta.num_dims(),
        3,
        "fused_qkv must be rank 3 [B, S, 3*H*Dh]"
    );
    assert_eq!(
        q_weight.meta.num_dims(),
        2,
        "q_weight must be rank 2 [H, Dh]"
    );
    assert_eq!(
        k_weight.meta.num_dims(),
        2,
        "k_weight must be rank 2 [H, Dh]"
    );
    assert_eq!(
        rope_cos.meta.num_dims(),
        2,
        "rope_cos must be rank 2 [S, Dh/2]"
    );
    assert_eq!(
        rope_sin.meta.num_dims(),
        2,
        "rope_sin must be rank 2 [S, Dh/2]"
    );

    let batch = fused_qkv.meta.shape()[0];
    let seq_len = fused_qkv.meta.shape()[1];
    let input_width = fused_qkv.meta.shape()[2];
    let num_heads = q_weight.meta.shape()[0];
    let head_dim = q_weight.meta.shape()[1];

    assert!(batch > 0, "fused QKV post-process requires B > 0");
    assert!(seq_len > 0, "fused QKV post-process requires S > 0");
    assert!(num_heads > 0, "fused QKV post-process requires H > 0");
    assert!(
        head_dim > 0 && head_dim.is_multiple_of(2),
        "Dh must be positive and even for adjacent-pair RoPE, got {head_dim}"
    );

    assert_eq!(k_weight.meta.shape()[0], num_heads, "k_weight H mismatch");
    assert_eq!(k_weight.meta.shape()[1], head_dim, "k_weight Dh mismatch");
    assert_eq!(rope_cos.meta.shape()[0], seq_len, "rope_cos S mismatch");
    assert_eq!(
        rope_cos.meta.shape()[1],
        head_dim / 2,
        "rope_cos Dh/2 mismatch"
    );
    assert_eq!(rope_sin.meta.shape()[0], seq_len, "rope_sin S mismatch");
    assert_eq!(
        rope_sin.meta.shape()[1],
        head_dim / 2,
        "rope_sin Dh/2 mismatch"
    );

    // These tensors are model parameters/precomputed trajectory tables. Keep
    // the hot path allocation-free for them and fail loudly if that contract
    // is violated. Only the projection result may require one materialisation.
    for (name, tensor) in [
        ("q_weight", &q_weight),
        ("k_weight", &k_weight),
        ("rope_cos", &rope_cos),
        ("rope_sin", &rope_sin),
    ] {
        assert!(tensor.is_contiguous(), "{name} must be contiguous");
    }
    let fused_qkv = into_contiguous(fused_qkv);

    let kv_dim = num_heads.checked_mul(head_dim).expect("H * Dh overflow");
    let expected_input_width = layout.input_width(kv_dim);
    assert_eq!(
        input_width,
        expected_input_width,
        "projection last dimension must equal {} * H * Dh for {layout:?}",
        if layout.writes_gate() { 4 } else { 3 },
    );

    let token_count = batch.checked_mul(seq_len).expect("B * S overflow");
    let output_elements = token_count
        .checked_mul(kv_dim)
        .expect("B * S * H * Dh overflow");
    let input_elements = token_count
        .checked_mul(expected_input_width)
        .expect("B * S * projection width overflow");
    let output_bytes = output_elements
        .checked_mul(precision.element_bytes())
        .expect("Q/K/V output byte size overflow");

    for (name, value) in [
        ("B", batch),
        ("S", seq_len),
        ("H", num_heads),
        ("Dh", head_dim),
        ("H*Dh", kv_dim),
        ("projection width", expected_input_width),
        (
            "B*S*H",
            token_count.checked_mul(num_heads).expect("B*S*H overflow"),
        ),
        ("input elements", input_elements),
        ("output elements", output_elements),
    ] {
        assert!(
            u32::try_from(value).is_ok(),
            "{name}={value} exceeds WGSL u32 indexing"
        );
    }

    let total_workgroups = token_count
        .checked_mul(num_heads)
        .expect("B * S * H overflow");
    let head_dim_u32 = u32::try_from(head_dim).expect("validated Dh must fit u32");
    let pair_count = head_dim_u32 / 2;
    let workgroup_size = if pair_count >= MAX_WORKGROUP_SIZE {
        MAX_WORKGROUP_SIZE
    } else {
        pair_count.next_power_of_two()
    };
    let batch_u32 = u32::try_from(batch).expect("validated B must fit u32");
    let seq_len_u32 = u32::try_from(seq_len).expect("validated S must fit u32");
    let num_heads_u32 = u32::try_from(num_heads).expect("validated H must fit u32");
    let input_width_u32 =
        u32::try_from(expected_input_width).expect("validated projection width must fit u32");
    let total_workgroups_u32 =
        u32::try_from(total_workgroups).expect("validated B*S*H must fit u32");

    let client = fused_qkv.client.clone();
    let device = fused_qkv.device.clone();
    let shape = Shape::from([batch, seq_len, num_heads, head_dim]);
    let make_output = || {
        CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            shape.clone(),
            client.empty(output_bytes),
            precision.dtype(),
        )
    };
    let q = make_output();
    let k = make_output();
    let v = make_output();

    let kernel = QkvPostprocessKernel {
        precision,
        batch: batch_u32,
        seq_len: seq_len_u32,
        num_heads: num_heads_u32,
        head_dim: head_dim_u32,
        input_width: input_width_u32,
        writes_gate: layout.writes_gate(),
        workgroup_size,
        eps,
    };
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(kernel, CubeDim::new_1d(workgroup_size)));
    let bindings = KernelArguments::new()
        .with_buffer(fused_qkv.handle.clone().binding())
        .with_buffer(q_weight.handle.binding())
        .with_buffer(k_weight.handle.binding())
        .with_buffer(rope_cos.handle.binding())
        .with_buffer(rope_sin.handle.binding())
        .with_buffer(q.handle.clone().binding())
        .with_buffer(k.handle.clone().binding())
        .with_buffer(v.handle.clone().binding());
    client.launch(task, CubeCount::new_1d(total_workgroups_u32), bindings);

    (QkvPostprocessOutput { q, k, v }, fused_qkv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::graphics::AutoGraphicsApi;
    use burn::backend::wgpu::{WgpuDevice, init_setup};
    use burn::tensor::Tensor;

    #[derive(Clone, Copy)]
    struct ReferenceShape {
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        projection_segments: usize,
    }

    fn reference_qkv_postprocess(
        fused_qkv: &[f32],
        q_weight: &[f32],
        k_weight: &[f32],
        rope_cos: &[f32],
        rope_sin: &[f32],
        shape: ReferenceShape,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let ReferenceShape {
            batch,
            seq_len,
            num_heads,
            head_dim,
            projection_segments,
        } = shape;
        let kv_dim = num_heads * head_dim;
        let mut q_out = vec![0.0; batch * seq_len * kv_dim];
        let mut k_out = vec![0.0; q_out.len()];
        let mut v_out = vec![0.0; q_out.len()];

        for token in 0..batch * seq_len {
            let seq = token % seq_len;
            let fused_base = token * projection_segments * kv_dim;
            for head in 0..num_heads {
                let head_offset = head * head_dim;
                let q_base = fused_base + head_offset;
                let k_base = fused_base + kv_dim + head_offset;
                let v_base = fused_base + 2 * kv_dim + head_offset;
                let out_base = token * kv_dim + head_offset;

                let q_rms = (fused_qkv[q_base..q_base + head_dim]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    / head_dim as f32
                    + eps)
                    .sqrt();
                let k_rms = (fused_qkv[k_base..k_base + head_dim]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    / head_dim as f32
                    + eps)
                    .sqrt();

                for pair in 0..head_dim / 2 {
                    let even = 2 * pair;
                    let odd = even + 1;
                    let weight_base = head * head_dim;
                    let q_re = fused_qkv[q_base + even] / q_rms * q_weight[weight_base + even];
                    let q_im = fused_qkv[q_base + odd] / q_rms * q_weight[weight_base + odd];
                    let k_re = fused_qkv[k_base + even] / k_rms * k_weight[weight_base + even];
                    let k_im = fused_qkv[k_base + odd] / k_rms * k_weight[weight_base + odd];

                    if head < num_heads / 2 {
                        let rope = seq * (head_dim / 2) + pair;
                        let cos = rope_cos[rope];
                        let sin = rope_sin[rope];
                        q_out[out_base + even] = q_re * cos - q_im * sin;
                        q_out[out_base + odd] = q_re * sin + q_im * cos;
                        k_out[out_base + even] = k_re * cos - k_im * sin;
                        k_out[out_base + odd] = k_re * sin + k_im * cos;
                    } else {
                        q_out[out_base + even] = q_re;
                        q_out[out_base + odd] = q_im;
                        k_out[out_base + even] = k_re;
                        k_out[out_base + odd] = k_im;
                    }
                    v_out[out_base + even] = fused_qkv[v_base + even];
                    v_out[out_base + odd] = fused_qkv[v_base + odd];
                }
            }
        }
        (q_out, k_out, v_out)
    }

    #[test]
    fn cpu_reference_rotates_only_first_half_of_heads() {
        let shape = ReferenceShape {
            batch: 1,
            seq_len: 1,
            num_heads: 2,
            head_dim: 2,
            projection_segments: 3,
        };
        let fused = vec![
            1.0, 2.0, 3.0, 4.0, // Q
            5.0, 6.0, 7.0, 8.0, // K
            9.0, 10.0, 11.0, 12.0, // V
        ];
        let weights = vec![1.0; 4];
        let (q, k, v) =
            reference_qkv_postprocess(&fused, &weights, &weights, &[0.0], &[1.0], shape, 0.0);

        let q0_rms = ((1.0_f32 + 4.0) / 2.0).sqrt();
        let q1_rms = ((9.0_f32 + 16.0) / 2.0).sqrt();
        let k0_rms = ((25.0_f32 + 36.0) / 2.0).sqrt();
        let k1_rms = ((49.0_f32 + 64.0) / 2.0).sqrt();
        assert_eq!(
            q,
            vec![-2.0 / q0_rms, 1.0 / q0_rms, 3.0 / q1_rms, 4.0 / q1_rms]
        );
        assert_eq!(
            k,
            vec![-6.0 / k0_rms, 5.0 / k0_rms, 7.0 / k1_rms, 8.0 / k1_rms]
        );
        assert_eq!(v, vec![9.0, 10.0, 11.0, 12.0]);
    }

    fn setup_device() -> Device {
        let device = WgpuDevice::DefaultDevice;
        init_setup::<AutoGraphicsApi>(&device, Default::default());
        crate::backend_config::strict_fp32_device(&device)
            .expect("test WGPU device must support strict FP32")
    }

    #[test]
    #[ignore = "requires a WGPU adapter; run manually"]
    fn fused_qkv_postprocess_matches_cpu_reference() {
        let device = setup_device();
        let shape = ReferenceShape {
            batch: 1,
            seq_len: 3,
            num_heads: 4,
            head_dim: 8,
            projection_segments: 3,
        };
        let kv_dim = shape.num_heads * shape.head_dim;
        let fused: Vec<f32> = (0..shape.batch * shape.seq_len * 3 * kv_dim)
            .map(|index| index as f32 * 0.007 - 0.9)
            .collect();
        let q_weight: Vec<f32> = (0..kv_dim)
            .map(|index| 0.7 + index as f32 * 0.003)
            .collect();
        let k_weight: Vec<f32> = (0..kv_dim)
            .map(|index| 1.1 - index as f32 * 0.002)
            .collect();
        let rope_cos: Vec<f32> = (0..shape.seq_len * shape.head_dim / 2)
            .map(|index| (index as f32 * 0.1).cos())
            .collect();
        let rope_sin: Vec<f32> = (0..shape.seq_len * shape.head_dim / 2)
            .map(|index| (index as f32 * 0.1).sin())
            .collect();
        let eps = 1.0e-6;
        let expected = reference_qkv_postprocess(
            &fused, &q_weight, &k_weight, &rope_cos, &rope_sin, shape, eps,
        );

        let fused_tensor = Tensor::<1>::from_floats(fused.as_slice(), &device).reshape([
            shape.batch,
            shape.seq_len,
            3 * kv_dim,
        ]);
        let q_weight_tensor = Tensor::<1>::from_floats(q_weight.as_slice(), &device)
            .reshape([shape.num_heads, shape.head_dim]);
        let k_weight_tensor = Tensor::<1>::from_floats(k_weight.as_slice(), &device)
            .reshape([shape.num_heads, shape.head_dim]);
        let cos_tensor = Tensor::<1>::from_floats(rope_cos.as_slice(), &device)
            .reshape([shape.seq_len, shape.head_dim / 2]);
        let sin_tensor = Tensor::<1>::from_floats(rope_sin.as_slice(), &device)
            .reshape([shape.seq_len, shape.head_dim / 2]);

        let output = fused_qkv_postprocess_wgsl(
            fused_tensor
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            q_weight_tensor
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            k_weight_tensor
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            cos_tensor
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            sin_tensor
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("tensor must use WGPU raw backend"),
            eps as f64,
        );
        let read = |tensor| {
            Tensor::<4>::from_primitive::<crate::WgpuRaw>(tensor)
                .into_data()
                .to_vec::<f32>()
                .expect("f32 output")
        };
        let actual = (read(output.q), read(output.k), read(output.v));

        for (name, actual, expected) in [
            ("q", actual.0, expected.0),
            ("k", actual.1, expected.1),
            ("v", actual.2, expected.2),
        ] {
            for (index, (got, want)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (got - want).abs() < 1.0e-4,
                    "{name}[{index}]: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a WGPU adapter; run manually"]
    fn fused_qkv_gate_postprocess_matches_cpu_reference_for_b1_b2() {
        let device = setup_device();
        for batch in [1, 2] {
            let shape = ReferenceShape {
                batch,
                seq_len: 3,
                num_heads: 4,
                head_dim: 8,
                projection_segments: 4,
            };
            let kv_dim = shape.num_heads * shape.head_dim;
            let combined: Vec<f32> = (0..shape.batch * shape.seq_len * 4 * kv_dim)
                .map(|index| (index % 97) as f32 * 0.013 - 0.6)
                .collect();
            let q_weight: Vec<f32> = (0..kv_dim)
                .map(|index| 0.7 + index as f32 * 0.003)
                .collect();
            let k_weight: Vec<f32> = (0..kv_dim)
                .map(|index| 1.1 - index as f32 * 0.002)
                .collect();
            let rope_cos: Vec<f32> = (0..shape.seq_len * shape.head_dim / 2)
                .map(|index| (index as f32 * 0.1).cos())
                .collect();
            let rope_sin: Vec<f32> = (0..shape.seq_len * shape.head_dim / 2)
                .map(|index| (index as f32 * 0.1).sin())
                .collect();
            let eps = 1.0e-6;
            let expected_qkv = reference_qkv_postprocess(
                &combined, &q_weight, &k_weight, &rope_cos, &rope_sin, shape, eps,
            );

            let combined_tensor = Tensor::<1>::from_floats(combined.as_slice(), &device).reshape([
                shape.batch,
                shape.seq_len,
                4 * kv_dim,
            ]);
            let q_weight_tensor = Tensor::<1>::from_floats(q_weight.as_slice(), &device)
                .reshape([shape.num_heads, shape.head_dim]);
            let k_weight_tensor = Tensor::<1>::from_floats(k_weight.as_slice(), &device)
                .reshape([shape.num_heads, shape.head_dim]);
            let cos_tensor = Tensor::<1>::from_floats(rope_cos.as_slice(), &device)
                .reshape([shape.seq_len, shape.head_dim / 2]);
            let sin_tensor = Tensor::<1>::from_floats(rope_sin.as_slice(), &device)
                .reshape([shape.seq_len, shape.head_dim / 2]);

            let output = fused_qkv_gate_postprocess_wgsl(
                combined_tensor
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                q_weight_tensor
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                k_weight_tensor
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                cos_tensor
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                sin_tensor
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("tensor must use WGPU raw backend"),
                eps as f64,
            );
            let read_qkv = |tensor| {
                Tensor::<4>::from_primitive::<crate::WgpuRaw>(tensor)
                    .into_data()
                    .to_vec::<f32>()
                    .expect("f32 output")
            };
            let actual_qkv = (
                read_qkv(output.qkv.q),
                read_qkv(output.qkv.k),
                read_qkv(output.qkv.v),
            );
            for (name, actual, expected) in [
                ("q", actual_qkv.0, expected_qkv.0),
                ("k", actual_qkv.1, expected_qkv.1),
                ("v", actual_qkv.2, expected_qkv.2),
            ] {
                let max_abs = actual
                    .iter()
                    .zip(expected)
                    .map(|(got, want)| (got - want).abs())
                    .fold(0.0_f32, f32::max);
                assert!(max_abs < 1.0e-4, "batch={batch} {name} max_abs={max_abs}");
            }

            let actual_combined = Tensor::<3>::from_primitive::<crate::WgpuRaw>(output.combined)
                .into_data()
                .to_vec::<f32>()
                .expect("f32 combined output");
            let mut gate_max_abs = 0.0_f32;
            for token in 0..shape.batch * shape.seq_len {
                let row = token * 4 * kv_dim;
                assert_eq!(
                    &actual_combined[row..row + 3 * kv_dim],
                    &combined[row..row + 3 * kv_dim],
                    "batch={batch} token={token}: QKV input segments changed"
                );
                for dim in 0..kv_dim {
                    let index = row + 3 * kv_dim + dim;
                    let expected_gate = 1.0 / (1.0 + (-combined[index]).exp());
                    gate_max_abs = gate_max_abs.max((actual_combined[index] - expected_gate).abs());
                }
            }
            assert!(
                gate_max_abs <= 2.0e-7,
                "batch={batch} gate max_abs={gate_max_abs}"
            );
        }
    }
}
