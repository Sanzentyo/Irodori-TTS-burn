//! Two-matmul strict-F32 SDPA with an in-place fused softmax stage.
//!
//! The QKᵀ and softmax·V products remain ordinary Burn/CubeK matmuls, so their
//! device-specific sealed selections are preserved. A single portable WGSL
//! dispatch consumes the score matrix between them and performs scaling,
//! key-padding masking, row max/sum reductions, and normalization in place.

use burn::backend::wgpu::{
    CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime,
};
use burn::tensor::{BoolStore, DType};
use cubecl::{CubeCount, prelude::KernelId, server::KernelArguments};

const WORKGROUP_SIZE: u32 = 256;
const REQUIRED_BINDINGS: u32 = 2;
const SHARED_BYTES: usize = 2 * WORKGROUP_SIZE as usize * size_of::<f32>();

#[derive(Debug)]
struct MatmulSdpaSoftmaxKernel {
    rows: u32,
    heads: u32,
    sequence_q: u32,
    sequence_kv: u32,
    scale_bits: u32,
}

impl KernelSource for MatmulSdpaSoftmaxKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("matmul_sdpa_softmax.wgsl"))
            .register("rows", self.rows.to_string())
            .register("heads", self.heads.to_string())
            .register("sequence_q", self.sequence_q.to_string())
            .register("sequence_kv", self.sequence_kv.to_string())
            .register("scale", format!("{:.9}", f32::from_bits(self.scale_bits)))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.rows,
            self.heads,
            self.sequence_q,
            self.sequence_kv,
            self.scale_bits,
        ))
    }
}

/// Apply the strict-F32 masked softmax stage to a contiguous score matrix.
///
/// `scores` must have shape `[B,H,Sq,Skv]`. `masked_out` must have shape
/// `[B,Skv]` with `true = masked-out`, matching Burn's backend convention.
/// The score allocation is updated in place and returned without another
/// allocation or layout conversion. Every unsupported physical contract is
/// rejected before dispatch.
pub fn try_fused_scores_softmax_f32(
    scores: CubeTensor<WgpuRuntime>,
    masked_out: CubeTensor<WgpuRuntime>,
    scale: f32,
) -> Option<CubeTensor<WgpuRuntime>> {
    if scores.meta.num_dims() != 4
        || masked_out.meta.num_dims() != 2
        || scores.dtype != DType::F32
        || masked_out.dtype != DType::Bool(BoolStore::U32)
        || !scores.is_contiguous()
        || !masked_out.is_contiguous()
        || scores.device != masked_out.device
        || !core::ptr::eq(scores.client.info(), masked_out.client.info())
        || !scale.is_finite()
        || scale <= 0.0
    {
        return None;
    }
    let [batch, heads, sequence_q, sequence_kv] = scores.meta.shape().dims::<4>();
    if batch == 0
        || heads == 0
        || sequence_q == 0
        || sequence_kv == 0
        || masked_out.meta.shape().dims::<2>() != [batch, sequence_kv]
    {
        return None;
    }
    let rows = batch.checked_mul(heads)?.checked_mul(sequence_q)?;
    let rows_u32 = u32::try_from(rows).ok()?;
    let heads_u32 = u32::try_from(heads).ok()?;
    let sequence_q_u32 = u32::try_from(sequence_q).ok()?;
    let sequence_kv_u32 = u32::try_from(sequence_kv).ok()?;
    let client = scores.client.clone();
    let hardware = &client.properties().hardware;
    if hardware.max_units_per_cube < WORKGROUP_SIZE
        || hardware.max_cube_dim.0 < WORKGROUP_SIZE
        || hardware.max_cube_count.0 < rows_u32
        || hardware.max_shared_memory_size < SHARED_BYTES
        || hardware.max_bindings < REQUIRED_BINDINGS
    {
        return None;
    }

    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> =
        Box::new(SourceKernel::new(
            MatmulSdpaSoftmaxKernel {
                rows: rows_u32,
                heads: heads_u32,
                sequence_q: sequence_q_u32,
                sequence_kv: sequence_kv_u32,
                scale_bits: scale.to_bits(),
            },
            CubeDim::new_1d(WORKGROUP_SIZE),
        ));
    client.launch(
        task,
        CubeCount::new_1d(rows_u32),
        KernelArguments::new()
            .with_buffer(scores.handle.clone().binding())
            .with_buffer(masked_out.handle.binding()),
    );
    Some(scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_geometry_fits_webgpu_minimums() {
        assert_eq!(WORKGROUP_SIZE, 256);
        assert_eq!(REQUIRED_BINDINGS, 2);
        assert_eq!(SHARED_BYTES, 2_048);
    }

    #[test]
    fn shader_masks_before_both_reductions() {
        let shader = include_str!("matmul_sdpa_softmax.wgsl");
        assert!(shader.contains("masked_out[mask_index] != 0u"));
        assert!(shader.contains("max(partial_sum[0], MIN_POSITIVE)"));
        assert!(shader.contains("scores[index] = scores[index] / denominator"));
        assert_eq!(shader.matches("workgroupBarrier();").count(), 4);
    }

    #[test]
    #[ignore = "requires a WGPU adapter"]
    fn device_kernel_matches_masked_softmax_and_zeroes_fully_masked_rows() {
        use burn::backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup};
        use burn::tensor::{Bool, Tensor, TensorData};

        let raw_device = WgpuDevice::DefaultDevice;
        init_setup::<AutoGraphicsApi>(&raw_device, Default::default());
        let device = crate::backend_config::wgpu_device_with_precision(
            &raw_device,
            crate::WgpuFloatPrecision::Fp32,
        )
        .unwrap();
        let shape = [2, 3, 5, 7];
        let values = (0..shape.into_iter().product::<usize>())
            .map(|index| (index as f32 * 0.013).sin())
            .collect::<Vec<_>>();
        let scores =
            Tensor::<1>::from_data(TensorData::new(values.clone(), [values.len()]), &device)
                .reshape(shape);
        let mask = Tensor::<2, Bool>::from_bool(
            [
                [false, true, false, false, true, false, false],
                [true, true, true, true, true, true, true],
            ],
            &device,
        );
        let actual = try_fused_scores_softmax_f32(
            scores.try_into_primitive::<crate::WgpuRaw>().unwrap(),
            mask.try_into_primitive::<crate::WgpuRaw>().unwrap(),
            0.125,
        )
        .expect("exact fused softmax contract");
        let actual = Tensor::<4>::from_primitive::<crate::WgpuRaw>(actual)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for row in 0..2 * 3 * 5 {
            let batch = row / (3 * 5);
            let row_values = &values[row * 7..row * 7 + 7];
            let valid = if batch == 0 {
                [true, false, true, true, false, true, true]
            } else {
                [false; 7]
            };
            let max = row_values
                .iter()
                .zip(valid)
                .filter_map(|(value, valid)| valid.then_some(value * 0.125))
                .fold(f32::NEG_INFINITY, f32::max);
            let denominator = row_values
                .iter()
                .zip(valid)
                .filter_map(|(value, valid)| valid.then_some((value * 0.125 - max).exp()))
                .sum::<f32>();
            for key in 0..7 {
                let expected = if valid[key] {
                    (row_values[key] * 0.125 - max).exp() / denominator
                } else {
                    0.0
                };
                let found = actual[row * 7 + key];
                assert!(
                    (found - expected).abs() <= 1.0e-6,
                    "row {row} key {key}: {found} vs {expected}"
                );
            }
        }
    }
}
