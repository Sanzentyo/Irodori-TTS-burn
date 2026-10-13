//! Exact-shape attention tuning over the existing WGSL and Burn kernels.
//!
//! CubeCL owns timing and persistence. Only cold tuning inputs carry an
//! accuracy guard; a restored winner performs no readback or repeated check.
//! Every candidate is compared to an explicit FP32 attention graph before it
//! can win. Unsupported or inaccurate candidates return a tuning error.

use std::sync::Arc;

use burn::{
    backend::wgpu::{CubeTensor, into_contiguous},
    tensor::{Bool, DType, Tensor, activation::softmax, ops::AttentionModuleOptions},
};
use cubecl::{
    CubeTuneId,
    tune::{LocalTuner, Tunable, TunableSet, local_tuner},
};

use super::{
    fused_sdpa_native::{NativeFaConfig, native_fa_sdpa_wgsl, supports_native_fa_sdpa_wgsl},
    matmul_cmma::{self, CmmaPlan},
    matmul_sdpa,
    tuning_validation::check_accuracy,
};

#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
struct AttentionKey {
    schema: u16,
    device: String,
    dtype: String,
    shapes: [[usize; 4]; 3],
    strides: [Vec<usize>; 4],
}

impl std::fmt::Display for AttentionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sdpa-v{}-{}-{}-{:?}-{:?}",
            self.schema, self.device, self.dtype, self.shapes, self.strides
        )
    }
}
impl cubecl::tune::AutotuneKey for AttentionKey {}

#[derive(Clone)]
struct Input {
    q: CubeTensor,
    k: CubeTensor,
    v: CubeTensor,
    masked_out: CubeTensor,
    validation: Option<Arc<Vec<Result<(), String>>>>,
}

#[derive(Clone, Copy)]
enum Candidate {
    Reference,
    Burn,
    CachedSoftmax(u32, bool),
    SubgroupSoftmax(bool),
    Native(NativeFaConfig),
    MixedPrecision { qk: CmmaPlan, pv: CmmaPlan },
}

const CANDIDATES: &[(&str, Candidate)] = &[
    ("fp32-reference", Candidate::Reference),
    ("burn-attention", Candidate::Burn),
    ("matmul-cached-64", Candidate::CachedSoftmax(64, false)),
    ("matmul-cached-128", Candidate::CachedSoftmax(128, false)),
    ("matmul-cached-256", Candidate::CachedSoftmax(256, false)),
    ("matmul-subgroup-32", Candidate::SubgroupSoftmax(false)),
    (
        "matmul-cached-64-unit-pv",
        Candidate::CachedSoftmax(64, true),
    ),
    (
        "matmul-cached-128-unit-pv",
        Candidate::CachedSoftmax(128, true),
    ),
    (
        "matmul-cached-256-unit-pv",
        Candidate::CachedSoftmax(256, true),
    ),
    (
        "matmul-subgroup-32-unit-pv",
        Candidate::SubgroupSoftmax(true),
    ),
    (
        "wgsl-q4-kv16",
        Candidate::Native(NativeFaConfig {
            tile_q: 4,
            tile_kv: 16,
        }),
    ),
    (
        "wgsl-q8-kv8",
        Candidate::Native(NativeFaConfig {
            tile_q: 8,
            tile_kv: 8,
        }),
    ),
    ("wgsl-q16-kv8", Candidate::Native(NativeFaConfig::Q16_KV8)),
    ("wgsl-q32-kv8", Candidate::Native(NativeFaConfig::Q32_KV8)),
    ("wgsl-q16-kv16", Candidate::Native(NativeFaConfig::Q16_KV16)),
    ("wgsl-q32-kv16", Candidate::Native(NativeFaConfig::Q32_KV16)),
    ("wgsl-q8-kv32", Candidate::Native(NativeFaConfig::Q8_KV32)),
    (
        "mixed-cmma-single-single",
        Candidate::MixedPrecision {
            qk: CmmaPlan::SingleRow,
            pv: CmmaPlan::SingleRow,
        },
    ),
    (
        "mixed-cmma-single-double",
        Candidate::MixedPrecision {
            qk: CmmaPlan::SingleRow,
            pv: CmmaPlan::Double,
        },
    ),
    (
        "mixed-cmma-double-single",
        Candidate::MixedPrecision {
            qk: CmmaPlan::Double,
            pv: CmmaPlan::SingleRow,
        },
    ),
    (
        "mixed-cmma-double-double",
        Candidate::MixedPrecision {
            qk: CmmaPlan::Double,
            pv: CmmaPlan::Double,
        },
    ),
];

fn key(input: &Input) -> AttentionKey {
    AttentionKey {
        // Bump for every candidate-set or accuracy-policy change, for both dtypes.
        // CubeCL validates candidate checksums, but can hydrate an older checksum
        // first when two candidate sets were stored under the same logical key.
        schema: 4,
        device: input.q.client.properties().identity.fingerprint.clone(),
        dtype: input.q.dtype.name().to_owned(),
        shapes: [&input.q, &input.k, &input.v].map(|x| x.meta.shape().dims::<4>()),
        strides: [&input.q, &input.k, &input.v, &input.masked_out]
            .map(|x| x.meta.strides().to_vec()),
    }
}

fn tensor(x: &CubeTensor) -> Tensor<4> {
    Tensor::from_primitive::<crate::WgpuRaw>(x.clone())
}

fn reference(input: &Input) -> CubeTensor {
    let q = tensor(&input.q).cast(DType::F32);
    let k = tensor(&input.k).cast(DType::F32);
    let v = tensor(&input.v).cast(DType::F32);
    let mask = Tensor::<2, Bool>::from_primitive::<crate::WgpuRaw>(input.masked_out.clone())
        .unsqueeze_dim::<3>(1)
        .unsqueeze_dim::<4>(2);
    let scores = q.matmul(k.swap_dims(2, 3)) * (1.0 / (input.q.meta.shape()[3] as f32).sqrt());
    // The final mask also makes fully masked rows exactly zero.
    let probabilities = softmax(scores.mask_fill(mask.clone(), f32::MIN), 3).mask_fill(mask, 0.0);
    probabilities
        .matmul(v)
        .cast(input.q.dtype)
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("attention uses WGPU tensors")
}

fn launch(candidate: Candidate, input: &Input) -> Result<CubeTensor, String> {
    let scale = (input.q.meta.shape()[3] as f64).powf(-0.5);
    match candidate {
        Candidate::Reference => Ok(reference(input)),
        Candidate::MixedPrecision { qk, pv } => {
            // At short lengths the full-model screen found negligible savings
            // and worse waveform agreement from probability rounding. Admit
            // these matrix plans only once a 64-row query tile is full.
            if input.q.dtype != DType::F16 || input.q.meta.shape()[2] < 64 {
                return Err(
                    "mixed cooperative attention requires FP16 and at least 64 queries".to_owned(),
                );
            }
            let transposed_k = tensor(&input.k)
                .swap_dims(2, 3)
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU keys");
            let scores = matmul_cmma::matmul_f32_output(
                into_contiguous(input.q.clone()),
                into_contiguous(transposed_k),
                qk,
            )?;
            let probabilities = matmul_sdpa::try_fused_scores_softmax_cached_sized_f32(
                scores,
                input.masked_out.clone(),
                scale as f32,
                128,
            )
            .ok_or_else(|| "mixed attention softmax contract unsupported".to_owned())?;
            let probabilities = tensor(&probabilities)
                .cast(DType::F16)
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU probabilities");
            matmul_cmma::matmul(probabilities, into_contiguous(input.v.clone()), pv)
        }
        Candidate::Burn => {
            let mask =
                Tensor::<2, Bool>::from_primitive::<crate::WgpuRaw>(input.masked_out.clone())
                    .unsqueeze_dim::<3>(1)
                    .unsqueeze_dim::<4>(2)
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU mask");
            burn_cubecl::kernel::attention::attention(
                input.q.clone(),
                input.k.clone(),
                input.v.clone(),
                Some(mask),
                None,
                AttentionModuleOptions {
                    scale: None,
                    softcap: None,
                    is_causal: false,
                },
                burn_cubecl::kernel::attention::AttentionStrategy::Autotune,
            )
            .map_err(|error| format!("{error:?}"))
        }
        Candidate::CachedSoftmax(_, unit_pv) | Candidate::SubgroupSoftmax(unit_pv) => {
            if input.q.dtype != DType::F32 {
                return Err("fused score storage requires FP32".to_owned());
            }
            let scores = tensor(&input.q)
                .matmul(tensor(&input.k).swap_dims(2, 3))
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU scores");
            let probabilities = match candidate {
                Candidate::CachedSoftmax(lanes, _) => {
                    matmul_sdpa::try_fused_scores_softmax_cached_sized_f32(
                        scores,
                        input.masked_out.clone(),
                        scale as f32,
                        lanes,
                    )
                }
                _ => matmul_sdpa::try_fused_scores_softmax_subgroup_f32(
                    scores,
                    input.masked_out.clone(),
                    scale as f32,
                ),
            }
            .ok_or_else(|| "softmax launch contract unsupported".to_owned())?;
            if unit_pv {
                return super::matmul_unit::try_matmul_unit_min_f32(
                    probabilities,
                    into_contiguous(input.v.clone()),
                )
                .ok_or_else(|| "unit PV launch contract unsupported".to_owned());
            }
            Ok(Tensor::<4>::from_primitive::<crate::WgpuRaw>(probabilities)
                .matmul(tensor(&input.v))
                .try_into_primitive::<crate::WgpuRaw>()
                .expect("WGPU output"))
        }
        Candidate::Native(config) => {
            let mask =
                Tensor::<2, Bool>::from_primitive::<crate::WgpuRaw>(input.masked_out.clone())
                    .bool_not()
                    .float()
                    .cast(input.q.dtype)
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU mask");
            // Copies are part of candidate timing, never hidden in preparation.
            let q = into_contiguous(input.q.clone());
            let k = into_contiguous(input.k.clone());
            let v = into_contiguous(input.v.clone());
            if !supports_native_fa_sdpa_wgsl(&q, &k, &v, &mask, scale, &config) {
                return Err("native attention launch contract unsupported".to_owned());
            }
            Ok(native_fa_sdpa_wgsl(q, k, v, mask, scale, &config))
        }
    }
}

fn read(output: CubeTensor) -> Result<Vec<f32>, String> {
    super::tuning_validation::read::<4>(output)
}

fn tuning_inputs(_: &AttentionKey, input: &Input) -> Input {
    let expected = read(reference(input)).expect("attention tuning reference readback");
    let half = input.q.dtype == DType::F16;
    check_accuracy(&expected, &expected, half).expect("attention tuning reference must be finite");
    // Complete compilation, nested Burn tuning and numerical readbacks before
    // CubeCL opens its timing queries. Measuring a nested tuner corrupts the
    // query scope; including the numerical check biases candidate selection.
    let checks = CANDIDATES
        .iter()
        .map(|&(_, candidate)| {
            launch(candidate, input)
                .and_then(read)
                .and_then(|actual| check_accuracy(&expected, &actual, half))
        })
        .collect();
    Input {
        validation: Some(Arc::new(checks)),
        ..input.clone()
    }
}

/// Select and persist a validated route for this exact device, dtype and layout.
/// The mask always uses `true = masked out`; callers normalize it once.
pub(crate) fn attention(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    masked_out: Tensor<2, Bool>,
) -> Tensor<4> {
    static TUNER: LocalTuner<AttentionKey, CubeTuneId> = local_tuner!("validated-sdpa-v1");
    let input = Input {
        q: q.try_into_primitive::<crate::WgpuRaw>()
            .expect("WGPU query"),
        k: k.try_into_primitive::<crate::WgpuRaw>().expect("WGPU key"),
        v: v.try_into_primitive::<crate::WgpuRaw>()
            .expect("WGPU value"),
        masked_out: masked_out
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("WGPU mask"),
        validation: None,
    };
    let operations = TUNER.init(&CubeTuneId::new(&input.q.client, &input.q.device), || {
        CANDIDATES.iter().enumerate().fold(
            TunableSet::new(key, tuning_inputs).with_short_circuit(false),
            |set, (index, &(name, candidate))| {
                set.with(Tunable::new(name, move |input: Input| {
                    if let Some(validation) = &input.validation {
                        validation[index].clone()?;
                    }
                    launch(candidate, &input)
                }))
            },
        )
    });
    let client = input.q.client.clone();
    let id = CubeTuneId::new(&client, &input.q.device);
    Tensor::from_primitive::<crate::WgpuRaw>(TUNER.execute(&id, &client, operations, input))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a WGPU adapter; benchmarks candidate kernels"]
    fn tuned_attention_handles_strides_half_and_fully_masked_batches() {
        use burn::tensor::TensorData;
        let raw_device = std::env::var("IRODORI_TEST_DEVICE")
            .unwrap_or_else(|_| "auto".to_owned())
            .parse::<crate::backend_config::DeviceSelector>()
            .unwrap()
            .device();
        let device = crate::backend_config::strict_fp32_device(&raw_device).unwrap();
        for (dtype, sequence) in [DType::F32, DType::F16]
            .into_iter()
            .flat_map(|dtype| [13, 65].map(|sequence| (dtype, sequence)))
        {
            let make = |sequence: usize, phase: f32| {
                Tensor::<4>::from_data(
                    TensorData::new(
                        (0..2 * sequence * 2 * 64)
                            .map(|i| (i as f32 * 0.17 + phase).sin() * 0.3)
                            .collect::<Vec<_>>(),
                        [2, sequence, 2, 64],
                    ),
                    (&device, dtype),
                )
                .swap_dims(1, 2)
            };
            let q = make(sequence, 0.0);
            let k = make(17, 1.0);
            let v = make(17, 2.0);
            let mask = Tensor::<2, Bool>::from_data(
                TensorData::new(
                    (0..34).map(|i| i >= 17 || i % 5 == 0).collect::<Vec<_>>(),
                    [2, 17],
                ),
                &device,
            );
            let input = Input {
                q: q.clone().try_into_primitive::<crate::WgpuRaw>().unwrap(),
                k: k.clone().try_into_primitive::<crate::WgpuRaw>().unwrap(),
                v: v.clone().try_into_primitive::<crate::WgpuRaw>().unwrap(),
                masked_out: mask.clone().try_into_primitive::<crate::WgpuRaw>().unwrap(),
                validation: None,
            };
            let expected = read(reference(&input)).unwrap();
            let run = || {
                read(
                    attention(q.clone(), k.clone(), v.clone(), mask.clone())
                        .try_into_primitive::<crate::WgpuRaw>()
                        .unwrap(),
                )
                .unwrap()
            };
            let actual = run();
            check_accuracy(&expected, &actual, dtype == DType::F16).unwrap();
            assert!(actual[2 * sequence * 64..].iter().all(|&x| x == 0.0));
            assert_eq!(actual, run(), "cached winner must preserve output");
        }
    }

    #[test]
    fn accuracy_gate_rejects_nonfinite_and_partial_reductions() {
        assert!(check_accuracy(&[0.3, -0.8], &[0.3, -0.8], false).is_ok());
        assert!(check_accuracy(&[0.3, -0.8], &[0.15, -0.4], false).is_err());
        assert!(check_accuracy(&[0.3], &[f32::NAN], false).is_err());
        assert!(check_accuracy(&[f32::INFINITY], &[f32::INFINITY], true).is_err());
        assert!(check_accuracy(&[], &[], true).is_err());
    }

    #[test]
    fn persistent_key_separates_driver_dtype_and_exact_layout() {
        let base = AttentionKey {
            schema: 1,
            device: "driver-a".to_owned(),
            dtype: "f32".to_owned(),
            shapes: [[1, 20, 50, 64], [1, 20, 53, 64], [1, 20, 53, 64]],
            strides: [
                vec![64000, 64, 1280, 1],
                vec![67840, 64, 1280, 1],
                vec![67840, 64, 1280, 1],
                vec![53, 1],
            ],
        };
        let mut changes = vec![base.clone(); 5];
        changes[0].device = "driver-b".to_owned();
        changes[1].dtype = "f16".to_owned();
        changes[2].shapes[0][2] = 51;
        changes[3].strides[0] = vec![64000, 3200, 64, 1];
        changes[4].schema += 1;
        let encoded = serde_json::to_string(&base).unwrap();
        assert_eq!(
            serde_json::from_str::<AttentionKey>(&encoded).unwrap(),
            base
        );
        for other in changes {
            assert_ne!(base, other);
            assert_ne!(encoded, serde_json::to_string(&other).unwrap());
        }
    }
}
