//! Two-matmul strict-F32 SDPA with an in-place fused softmax stage.
//!
//! The QKᵀ and softmax·V products remain ordinary Burn/CubeK matmuls, so their
//! device-specific sealed selections are preserved. A single portable WGSL
//! dispatch consumes the score matrix between them and performs scaling,
//! key-padding masking, row max/sum reductions, and normalization in place.

use burn::backend::wgpu::{CubeDim, CubeTensor, KernelSource, SourceKernel, SourceTemplate};
use burn::tensor::{BoolStore, DType};
use cubecl::{CubeCount, ir::features::Plane, prelude::KernelId, server::KernelArguments};

const WORKGROUP_SIZE: u32 = 256;
const REQUIRED_BINDINGS: u32 = 2;
// Bound private storage independently of the request length. Longer contexts
// continue to use the streaming implementation instead of spilling a large
// per-invocation array to device memory.
const MAX_CACHED_KEYS: usize = 2_048;
const SUBGROUP_SIZE: u32 = 32;
const ROWS_PER_SUBGROUP_WORKGROUP: u32 = 4;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum ScoreStorage {
    Streaming,
    Cached,
    Cached64,
    Cached128,
    CachedSubgroup32,
}

impl ScoreStorage {
    const fn reduction_lanes(self) -> u32 {
        match self {
            Self::Cached64 => 64,
            Self::Cached128 => 128,
            _ => WORKGROUP_SIZE,
        }
    }
}

#[derive(Debug)]
struct MatmulSdpaSoftmaxKernel {
    storage: ScoreStorage,
    rows: u32,
    heads: u32,
    sequence_q: u32,
    sequence_kv: u32,
    scale_bits: u32,
}

impl KernelSource for MatmulSdpaSoftmaxKernel {
    fn lang(&self) -> &'static str {
        "wgsl"
    }

    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(match self.storage {
            ScoreStorage::Streaming => include_str!("matmul_sdpa_softmax.wgsl"),
            ScoreStorage::Cached | ScoreStorage::Cached64 | ScoreStorage::Cached128 => {
                include_str!("matmul_sdpa_softmax_cached.wgsl")
            }
            ScoreStorage::CachedSubgroup32 => include_str!("matmul_sdpa_softmax_subgroup.wgsl"),
        })
        .register("rows", self.rows.to_string())
        .register("heads", self.heads.to_string())
        .register("sequence_q", self.sequence_q.to_string())
        .register("sequence_kv", self.sequence_kv.to_string())
        .register("workgroup_size", self.storage.reduction_lanes().to_string())
        .register(
            "items_per_lane",
            self.sequence_kv
                .div_ceil(self.storage.reduction_lanes())
                .to_string(),
        )
        .register("scale", format!("{:.9}", f32::from_bits(self.scale_bits)))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info((
            self.storage,
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
    scores: CubeTensor,
    masked_out: CubeTensor,
    scale: f32,
) -> Option<CubeTensor> {
    try_scores_softmax_f32(scores, masked_out, scale, ScoreStorage::Streaming)
}

/// Fused strict-F32 softmax retaining each lane's scores and mask in registers.
///
/// This has the same input and in-place ownership contract as
/// [`try_fused_scores_softmax_f32`], including fully masked rows producing zero.
/// It preserves the streaming kernel's reduction order, but reads each score
/// and mask once and writes only the final probability. Contexts longer than
/// 2,048 keys are rejected before launch to bound per-invocation storage.
pub fn try_fused_scores_softmax_cached_f32(
    scores: CubeTensor,
    masked_out: CubeTensor,
    scale: f32,
) -> Option<CubeTensor> {
    try_scores_softmax_f32(scores, masked_out, scale, ScoreStorage::Cached)
}

/// Cached FP32 softmax with an explicit 64, 128 or 256 lane reduction.
/// Smaller groups change addition order and must pass the caller's accuracy gate.
pub fn try_fused_scores_softmax_cached_sized_f32(
    scores: CubeTensor,
    masked_out: CubeTensor,
    scale: f32,
    workgroup_size: u32,
) -> Option<CubeTensor> {
    let storage = match workgroup_size {
        64 => ScoreStorage::Cached64,
        128 => ScoreStorage::Cached128,
        256 => ScoreStorage::Cached,
        _ => return None,
    };
    try_scores_softmax_f32(scores, masked_out, scale, storage)
}

/// Register-cached softmax using one 32-lane subgroup per row.
///
/// Eight virtual lanes per invocation preserve the streaming implementation's
/// 256-lane addition tree. Requires native WGSL subgroup operations and a fixed
/// subgroup width of 32; other adapters and contexts over 2,048 keys are rejected.
/// The in-place ownership and masking contract matches
/// [`try_fused_scores_softmax_f32`].
pub fn try_fused_scores_softmax_subgroup_f32(
    scores: CubeTensor,
    masked_out: CubeTensor,
    scale: f32,
) -> Option<CubeTensor> {
    try_scores_softmax_f32(scores, masked_out, scale, ScoreStorage::CachedSubgroup32)
}

fn try_scores_softmax_f32(
    scores: CubeTensor,
    masked_out: CubeTensor,
    scale: f32,
    storage: ScoreStorage,
) -> Option<CubeTensor> {
    if scores.meta.num_dims() != 4
        || masked_out.meta.num_dims() != 2
        || scores.dtype != DType::F32
        || masked_out.dtype != DType::Bool(BoolStore::U32)
        || !scores.is_contiguous()
        || !masked_out.is_contiguous()
        || scores.device != masked_out.device
        || scores.client.service_id() != masked_out.client.service_id()
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
        || (storage != ScoreStorage::Streaming && sequence_kv > MAX_CACHED_KEYS)
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
    let properties = client.properties();
    let hardware = &properties.hardware;
    let subgroup = storage == ScoreStorage::CachedSubgroup32;
    if subgroup
        && (cfg!(target_family = "wasm")
            || hardware.plane_size_min != SUBGROUP_SIZE
            || hardware.plane_size_max != SUBGROUP_SIZE
            || !properties.features.plane.contains(Plane::Ops))
    {
        return None;
    }
    let (workgroup_size, groups, shared_bytes) = if subgroup {
        (
            SUBGROUP_SIZE * ROWS_PER_SUBGROUP_WORKGROUP,
            rows_u32.div_ceil(ROWS_PER_SUBGROUP_WORKGROUP),
            0,
        )
    } else {
        (
            storage.reduction_lanes(),
            rows_u32,
            2 * storage.reduction_lanes() as usize * size_of::<f32>(),
        )
    };
    if hardware.max_units_per_cube < workgroup_size
        || hardware.max_cube_dim.0 < workgroup_size
        || hardware.max_cube_count.0 < groups
        || hardware.max_shared_memory_size < shared_bytes
        || hardware.max_bindings < REQUIRED_BINDINGS
    {
        return None;
    }

    let task: Box<dyn cubecl::CubeKernel> = Box::new(SourceKernel::new(
        MatmulSdpaSoftmaxKernel {
            storage,
            rows: rows_u32,
            heads: heads_u32,
            sequence_q: sequence_q_u32,
            sequence_kv: sequence_kv_u32,
            scale_bits: scale.to_bits(),
        },
        CubeDim::new_1d(workgroup_size),
    ));
    client.launch(
        task,
        CubeCount::new_1d(groups),
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

        let raw_device = WgpuDevice::default();
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
            .try_to_vec::<f32>()
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

    #[test]
    #[ignore = "requires a WGPU adapter"]
    fn cached_softmax_matches_streaming_and_f64_oracle_at_lane_boundaries() {
        check_cached_softmax(&[ScoreStorage::Cached]);
    }

    #[test]
    #[ignore = "requires a WGPU adapter"]
    fn sized_softmax_matches_f64_at_lane_boundaries() {
        check_cached_softmax(&[ScoreStorage::Cached64, ScoreStorage::Cached128]);
    }

    #[test]
    #[ignore = "requires a WGPU adapter with fixed 32-lane subgroups"]
    fn subgroup_softmax_matches_streaming_and_f64_oracle_at_lane_boundaries() {
        check_cached_softmax(&[ScoreStorage::CachedSubgroup32]);
    }

    fn check_cached_softmax(storages: &[ScoreStorage]) {
        use burn::backend::wgpu::{graphics::AutoGraphicsApi, init_setup};
        use burn::tensor::{Bool, Tensor, TensorData};

        let raw_device = std::env::var("IRODORI_TEST_DEVICE")
            .unwrap_or_else(|_| "auto".to_owned())
            .parse::<crate::backend_config::DeviceSelector>()
            .unwrap()
            .device();
        init_setup::<AutoGraphicsApi>(&raw_device, Default::default());
        let device = crate::backend_config::strict_fp32_device(&raw_device).unwrap();
        for &storage in storages {
            for keys in [
                1,
                7,
                63,
                64,
                65,
                127,
                128,
                129,
                255,
                256,
                257,
                511,
                512,
                513,
                1_025,
                MAX_CACHED_KEYS,
            ] {
                // Six rows also exercise a partially occupied four-row workgroup.
                let shape = [2, 1, 3, keys];
                let rows_per_batch = shape[1] * shape[2];
                let mask = (0..2 * keys)
                    .map(|i| i >= keys || i % keys % 17 == 1)
                    .collect::<Vec<_>>();
                let values = (0..shape.into_iter().product::<usize>())
                    .map(|i| {
                        let batch = i / (rows_per_batch * keys);
                        if mask[batch * keys + i % keys] {
                            // Masked NaNs must never enter either reduction.
                            f32::NAN
                        } else {
                            (i as f32 * 0.73).sin() * 90.0
                        }
                    })
                    .collect::<Vec<_>>();
                let run = |storage| {
                    let scores =
                        Tensor::<4>::from_data(TensorData::new(values.clone(), shape), &device);
                    let masked_out = Tensor::<2, Bool>::from_data(
                        TensorData::new(mask.clone(), [2, keys]),
                        &device,
                    );
                    let output = try_scores_softmax_f32(
                        scores.try_into_primitive::<crate::WgpuRaw>().unwrap(),
                        masked_out.try_into_primitive::<crate::WgpuRaw>().unwrap(),
                        0.125,
                        storage,
                    )
                    .expect("valid softmax geometry");
                    Tensor::<4>::from_primitive::<crate::WgpuRaw>(output)
                        .into_data()
                        .try_to_vec::<f32>()
                        .unwrap()
                };
                let streaming = run(ScoreStorage::Streaming);
                let cached = run(storage);
                if matches!(
                    storage,
                    ScoreStorage::Cached | ScoreStorage::CachedSubgroup32
                ) {
                    assert!(
                        streaming
                            .iter()
                            .zip(&cached)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "keys={keys}: reduction order changed"
                    );
                }
                for (row, found) in cached.chunks_exact(keys).enumerate() {
                    let batch = row / rows_per_batch;
                    let valid = |key: usize| !mask[batch * keys + key];
                    let logits = &values[row * keys..(row + 1) * keys];
                    let max = (0..keys)
                        .filter(|&k| valid(k))
                        .map(|k| f64::from(logits[k]) * 0.125)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let numerator = |k: usize| {
                        if valid(k) {
                            (f64::from(logits[k]) * 0.125 - max).exp()
                        } else {
                            0.0
                        }
                    };
                    let sum = (0..keys).map(numerator).sum::<f64>();
                    for (key, &actual) in found.iter().enumerate() {
                        let expected = if valid(key) {
                            numerator(key) / sum
                        } else {
                            0.0
                        };
                        assert!(actual.is_finite(), "keys={keys}, row={row}, key={key}");
                        assert!(
                            (f64::from(actual) - expected).abs() <= 2.0e-6,
                            "keys={keys}, row={row}, key={key}: {actual} vs {expected}"
                        );
                        if !valid(key) {
                            assert_eq!(actual, 0.0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a WGPU adapter"]
    fn cached_softmax_rejects_unbounded_context_and_invalid_scale() {
        use burn::backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup};
        use burn::tensor::{Bool, Tensor};

        let raw_device = WgpuDevice::default();
        init_setup::<AutoGraphicsApi>(&raw_device, Default::default());
        let device = crate::backend_config::strict_fp32_device(&raw_device).unwrap();
        for (keys, scale) in [
            (MAX_CACHED_KEYS + 1, 0.125),
            (7, 0.0),
            (7, -1.0),
            (7, f32::NAN),
            (7, f32::INFINITY),
        ] {
            for storage in [ScoreStorage::Cached, ScoreStorage::CachedSubgroup32] {
                let scores = Tensor::<4>::zeros([1, 2, 3, keys], &device);
                let mask = Tensor::<2, Bool>::from_data(
                    burn::tensor::TensorData::new(vec![false; keys], [1, keys]),
                    &device,
                );
                assert!(
                    try_scores_softmax_f32(
                        scores.try_into_primitive::<crate::WgpuRaw>().unwrap(),
                        mask.try_into_primitive::<crate::WgpuRaw>().unwrap(),
                        scale,
                        storage,
                    )
                    .is_none()
                );
            }
        }
    }
}
