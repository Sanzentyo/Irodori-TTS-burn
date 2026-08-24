//! Strict-F32 plane SDPA expressed in the CubeCL DSL.
//!
//! One 32-lane plane owns one `(batch, head, query)` row.  Every lane keeps
//! two head-dimension values, while [`plane_sum`] performs the D=64 dot
//! reduction.  The softmax is updated online, so no score matrix or other
//! sequence-squared temporary is materialized.

use burn::backend::wgpu::{CubeTensor, WgpuRuntime};
use burn::tensor::{DType, Shape};
use cubecl::{CubeCount, cube, ir::features::Plane, prelude::*};

const PLANE_SIZE: u32 = 32;
const HEAD_DIM: usize = 64;
const SCALE: f32 = 0.125;

#[cube(launch)]
fn plane_sdpa_f32_kernel(
    q: &Tensor<f32>,
    k: &Tensor<f32>,
    v: &Tensor<f32>,
    attend_mask: &Tensor<u32>,
    output: &mut Tensor<f32>,
) {
    let query_row = CUBE_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let sequence_q = q.shape(2);
    let sequence_kv = k.shape(2);
    let heads = q.shape(1);
    let batch_head = query_row / sequence_q;
    let query = query_row % sequence_q;
    let batch = batch_head / heads;
    let head = batch_head % heads;

    let q_base = batch * q.stride(0) + head * q.stride(1) + query * q.stride(2);
    let q0 = q[q_base + lane * q.stride(3)];
    let q1 = q[q_base + (lane + PLANE_SIZE as usize) * q.stride(3)];
    let mut row_max = f32::new(-3.402_823_5e38_f32);
    let mut row_sum = f32::new(0.0_f32);
    let mut value0 = f32::new(0.0_f32);
    let mut value1 = f32::new(0.0_f32);

    for key in 0..sequence_kv {
        let mask_offset = batch * attend_mask.stride(0) + key * attend_mask.stride(1);
        if attend_mask[mask_offset] != 0 {
            let kv_base = batch * k.stride(0) + head * k.stride(1) + key * k.stride(2);
            let partial = q0 * k[kv_base + lane * k.stride(3)]
                + q1 * k[kv_base + (lane + PLANE_SIZE as usize) * k.stride(3)];
            let score = plane_sum(partial) * f32::new(SCALE);
            let next_max = max(row_max, score);
            let previous_scale = (row_max - next_max).exp();
            let weight = (score - next_max).exp();
            let v_base = batch * v.stride(0) + head * v.stride(1) + key * v.stride(2);
            value0 = value0 * previous_scale + weight * v[v_base + lane * v.stride(3)];
            value1 = value1 * previous_scale
                + weight * v[v_base + (lane + PLANE_SIZE as usize) * v.stride(3)];
            row_sum = row_sum * previous_scale + weight;
            row_max = next_max;
        }
    }

    let out_base = query_row * HEAD_DIM;
    if row_sum > 0.0 {
        output[out_base + lane] = value0 / row_sum;
        output[out_base + lane + PLANE_SIZE as usize] = value1 / row_sum;
    } else {
        // Match the portable masked-attention contract without emitting NaN
        // when a caller supplies an entirely masked key row.
        output[out_base + lane] = 0.0;
        output[out_base + lane + PLANE_SIZE as usize] = 0.0;
    }
}

/// Launch the plane candidate when its complete physical contract is met.
///
/// `attend_mask` is U32-backed and uses `nonzero = attend`.  Callers retain a
/// portable fallback for every rejected shape or adapter.
pub fn try_plane_sdpa_f32(
    q: CubeTensor<WgpuRuntime>,
    k: CubeTensor<WgpuRuntime>,
    v: CubeTensor<WgpuRuntime>,
    attend_mask: CubeTensor<WgpuRuntime>,
) -> Option<CubeTensor<WgpuRuntime>> {
    if q.meta.num_dims() != 4
        || k.meta.num_dims() != 4
        || v.meta.num_dims() != 4
        || attend_mask.meta.num_dims() != 2
        || [q.dtype, k.dtype, v.dtype]
            .into_iter()
            .any(|dtype| dtype != DType::F32)
        || attend_mask.dtype != DType::Bool(burn::tensor::BoolStore::U32)
        || q.device != k.device
        || q.device != v.device
        || q.device != attend_mask.device
    {
        return None;
    }
    let q_shape = q.meta.shape().dims::<4>();
    let k_shape = k.meta.shape().dims::<4>();
    let v_shape = v.meta.shape().dims::<4>();
    let mask_shape = attend_mask.meta.shape().dims::<2>();
    let [batch, heads, sequence_q, head_dim] = q_shape;
    let [k_batch, k_heads, sequence_kv, k_head_dim] = k_shape;
    if batch == 0
        || heads == 0
        || sequence_q == 0
        || sequence_kv == 0
        || head_dim != HEAD_DIM
        || k_shape != v_shape
        || [k_batch, k_heads, k_head_dim] != [batch, heads, HEAD_DIM]
        || mask_shape != [batch, sequence_kv]
    {
        return None;
    }
    let client = q.client.clone();
    let properties = client.properties();
    let hardware = &properties.hardware;
    let query_rows = batch.checked_mul(heads)?.checked_mul(sequence_q)?;
    let cube_count = u32::try_from(query_rows).ok()?;
    if hardware.plane_size_min != PLANE_SIZE
        || hardware.plane_size_max != PLANE_SIZE
        || !properties.features.plane.contains(Plane::Ops)
        || hardware.max_units_per_cube < PLANE_SIZE
        || hardware.max_cube_dim.0 < PLANE_SIZE
        || hardware.max_cube_count.0 < cube_count
        || hardware.max_bindings < 5
    {
        return None;
    }

    let output: CubeTensor<WgpuRuntime> = burn_cubecl::ops::numeric::empty_device_dtype(
        client.clone(),
        q.device.clone(),
        Shape::new(q_shape),
        DType::F32,
    );
    plane_sdpa_f32_kernel::launch::<WgpuRuntime>(
        &client,
        CubeCount::new_1d(cube_count),
        CubeDim::new_1d(PLANE_SIZE),
        q.into_tensor_arg(),
        k.into_tensor_arg(),
        v.into_tensor_arg(),
        attend_mask.into_tensor_arg(),
        output.clone().into_tensor_arg(),
    );
    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_contract_is_explicit() {
        assert_eq!(PLANE_SIZE, 32);
        assert_eq!(HEAD_DIM, 2 * PLANE_SIZE as usize);
        assert_eq!(
            SCALE.to_bits(),
            (1.0_f32 / (HEAD_DIM as f32).sqrt()).to_bits()
        );
    }

    #[test]
    #[ignore = "requires a plane-capable WGPU adapter"]
    fn device_kernel_respects_the_attend_mask() {
        use burn::backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup};
        use burn::tensor::{Bool, Tensor as BurnTensor, TensorData};

        let raw_device = WgpuDevice::DefaultDevice;
        init_setup::<AutoGraphicsApi>(&raw_device, Default::default());
        let device = crate::backend_config::wgpu_device_with_precision(
            &raw_device,
            crate::WgpuFloatPrecision::Fp32,
        )
        .unwrap();
        let q = BurnTensor::<1>::zeros([2 * HEAD_DIM], &device).reshape([1, 1, 2, HEAD_DIM]);
        let k = BurnTensor::<1>::zeros([3 * HEAD_DIM], &device).reshape([1, 1, 3, HEAD_DIM]);
        let values = (0..3 * HEAD_DIM)
            .map(|index| (index / HEAD_DIM) as f32 + (index % HEAD_DIM) as f32 / 100.0)
            .collect::<Vec<_>>();
        let v = BurnTensor::<1>::from_data(TensorData::new(values, [3 * HEAD_DIM]), &device)
            .reshape([1, 1, 3, HEAD_DIM]);
        let mask = BurnTensor::<2, Bool>::from_bool([[true, false, true]], &device);

        let output = try_plane_sdpa_f32(
            q.try_into_primitive::<crate::WgpuRaw>().unwrap(),
            k.try_into_primitive::<crate::WgpuRaw>().unwrap(),
            v.try_into_primitive::<crate::WgpuRaw>().unwrap(),
            mask.try_into_primitive::<crate::WgpuRaw>().unwrap(),
        )
        .expect("default test adapter must satisfy the 32-lane plane contract");
        let output = BurnTensor::<4>::from_primitive::<crate::WgpuRaw>(output)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        for query in 0..2 {
            for dim in 0..HEAD_DIM {
                let expected = 1.0 + dim as f32 / 100.0;
                assert!((output[query * HEAD_DIM + dim] - expected).abs() < 1.0e-5);
            }
        }
    }
}
