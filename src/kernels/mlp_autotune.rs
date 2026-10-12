//! Accuracy-gated MLP contraction tuning, using CubeCL's persistent cache.

use super::{
    dit_mlp_contract_residual::*,
    matmul_cmma::{self, CmmaPlan},
    tuning_validation::{check_accuracy, read},
};
use burn::{
    backend::wgpu::{CubeTensor, WgpuRuntime},
    tensor::{DType, Tensor},
};
use cubecl::{
    CubeTuneId,
    tune::{LocalTuner, Tunable, TunableSet, local_tuner},
};
use std::sync::Arc;

type GpuTensor = CubeTensor<WgpuRuntime>;
type Launcher = fn(GpuTensor, GpuTensor, GpuTensor, GpuTensor, usize, usize) -> Option<GpuTensor>;

#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
struct ContractKey {
    schema: u16,
    device: String,
    dtype: String,
    batch: usize,
    sequence: usize,
    shapes: [[usize; 2]; 4],
    strides: [Vec<usize>; 4],
}
impl std::fmt::Display for ContractKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mlp-v{}-{}-{}-b{}s{}-{:?}-{:?}",
            self.schema,
            self.device,
            self.dtype,
            self.batch,
            self.sequence,
            self.shapes,
            self.strides
        )
    }
}
impl cubecl::tune::AutotuneKey for ContractKey {}

#[derive(Clone)]
struct Input {
    activated: GpuTensor,
    weight: GpuTensor,
    residual: GpuTensor,
    gate: GpuTensor,
    batch: usize,
    sequence: usize,
    validation: Option<Arc<Vec<Result<(), String>>>>,
}

#[derive(Clone, Copy)]
enum Candidate {
    Reference,
    Burn,
    Cmma(CmmaPlan),
    Wgsl(Launcher),
}
const CANDIDATES: &[(&str, Candidate)] = &[
    ("fp32-reference", Candidate::Reference),
    ("burn-matmul-residual", Candidate::Burn),
    (
        "wgsl-scalar-k32",
        Candidate::Wgsl(try_dit_mlp_contract_residual_wgsl),
    ),
    (
        "wgsl-vec4-k32",
        Candidate::Wgsl(try_dit_mlp_contract_residual_vec4_wgsl),
    ),
    (
        "wgsl-vec4-k16",
        Candidate::Wgsl(try_dit_mlp_contract_residual_vec4_k16_wgsl),
    ),
    (
        "wgsl-rows32",
        Candidate::Wgsl(try_dit_mlp_contract_residual_rows32_vec4_wgsl),
    ),
    (
        "wgsl-rows48",
        Candidate::Wgsl(try_dit_mlp_contract_residual_rows48_vec4_wgsl),
    ),
    (
        "wgsl-rows48-k16",
        Candidate::Wgsl(try_dit_mlp_contract_residual_rows48_vec4_k16_wgsl),
    ),
    (
        "wgsl-double-buffer-k16",
        Candidate::Wgsl(try_dit_mlp_contract_residual_double_buffer_vec4_k16_wgsl),
    ),
    (
        "wgsl-prefetch-k16",
        Candidate::Wgsl(try_dit_mlp_contract_residual_prefetch_vec4_k16_wgsl),
    ),
    (
        "wgsl-rows96-k16",
        Candidate::Wgsl(try_dit_mlp_contract_residual_rows96_prefetch_vec4_k16_wgsl),
    ),
    (
        "wgsl-swizzled-k16",
        Candidate::Wgsl(try_dit_mlp_contract_residual_swizzled_vec4_k16_wgsl),
    ),
    (
        "wgsl-c64",
        Candidate::Wgsl(try_dit_mlp_contract_residual_c64_vec4_wgsl),
    ),
    (
        "wgsl-warp32",
        Candidate::Wgsl(try_dit_mlp_contract_residual_warp32_wgsl),
    ),
    (
        "wgsl-warp32-k16",
        Candidate::Wgsl(try_dit_mlp_contract_residual_warp32_k16_wgsl),
    ),
    (
        "wgsl-rows128",
        Candidate::Wgsl(try_dit_mlp_contract_residual_warp32_rows128_wgsl),
    ),
    (
        "wgsl-split-k2",
        Candidate::Wgsl(
            super::dit_mlp_contract_split_k::try_dit_mlp_contract_residual_split_k2_wgsl,
        ),
    ),
    ("cmma-single-row", Candidate::Cmma(CmmaPlan::SingleRow)),
    ("cmma-multi-row", Candidate::Cmma(CmmaPlan::MultiRow)),
    ("cmma-double", Candidate::Cmma(CmmaPlan::Double)),
    (
        "cmma-ordered-rows4",
        Candidate::Cmma(CmmaPlan::OrderedRows4),
    ),
    (
        "cmma-ordered-rows8",
        Candidate::Cmma(CmmaPlan::OrderedRows8),
    ),
];

fn key(input: &Input) -> ContractKey {
    let tensors = [
        &input.activated,
        &input.weight,
        &input.residual,
        &input.gate,
    ];
    ContractKey {
        schema: 2,
        device: input
            .activated
            .client
            .properties()
            .identity
            .fingerprint
            .clone(),
        dtype: input.activated.dtype.name().to_owned(),
        batch: input.batch,
        sequence: input.sequence,
        shapes: tensors.map(|x| x.meta.shape().dims::<2>()),
        strides: tensors.map(|x| x.meta.strides().to_vec()),
    }
}

fn graph(input: &Input, fp32: bool) -> GpuTensor {
    let dtype = if fp32 {
        DType::F32
    } else {
        input.activated.dtype
    };
    let tensor =
        |x: &GpuTensor| Tensor::<2>::from_primitive::<crate::WgpuRaw>(x.clone()).cast(dtype);
    let branch = tensor(&input.activated).matmul(tensor(&input.weight));
    let output = super::fused_residual_gate::fused_residual_gate_wgsl(
        tensor(&input.residual)
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("WGPU residual"),
        branch
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("WGPU branch"),
        tensor(&input.gate)
            .try_into_primitive::<crate::WgpuRaw>()
            .expect("WGPU gate"),
        input.batch,
        input.sequence,
    );
    Tensor::<2>::from_primitive::<crate::WgpuRaw>(output)
        .cast(input.activated.dtype)
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("WGPU output")
}

fn launch(candidate: Candidate, input: &Input) -> Result<GpuTensor, String> {
    match candidate {
        Candidate::Reference => Ok(graph(input, true)),
        Candidate::Burn => Ok(graph(input, false)),
        Candidate::Cmma(plan) => {
            let branch = matmul_cmma::matmul(input.activated.clone(), input.weight.clone(), plan)?;
            Ok(super::fused_residual_gate::fused_residual_gate_wgsl(
                input.residual.clone(),
                branch,
                input.gate.clone(),
                input.batch,
                input.sequence,
            ))
        }
        Candidate::Wgsl(launch) => launch(
            input.activated.clone(),
            input.weight.clone(),
            input.residual.clone(),
            input.gate.clone(),
            input.batch,
            input.sequence,
        )
        .ok_or_else(|| "MLP launch contract unsupported".to_owned()),
    }
}

fn tuning_inputs(_: &ContractKey, input: &Input) -> Input {
    let expected = read::<2>(graph(input, true)).expect("MLP tuning reference readback");
    let half = input.activated.dtype == DType::F16;
    check_accuracy(&expected, &expected, half).expect("MLP reference must be finite");
    // Compilation and nested matmul tuning must finish outside timing queries.
    let checks = CANDIDATES
        .iter()
        .map(|&(_, candidate)| {
            launch(candidate, input)
                .and_then(read::<2>)
                .and_then(|actual| check_accuracy(&expected, &actual, half))
        })
        .collect();
    Input {
        validation: Some(Arc::new(checks)),
        ..input.clone()
    }
}

pub(crate) fn contract(
    activated: Tensor<2>,
    weight: Tensor<2>,
    residual: Tensor<2>,
    gate: Tensor<2>,
    batch: usize,
    sequence: usize,
) -> Tensor<2> {
    static TUNER: LocalTuner<ContractKey, CubeTuneId> = local_tuner!("validated-mlp-v1");
    let primitive = |x: Tensor<2>| {
        x.try_into_primitive::<crate::WgpuRaw>()
            .expect("WGPU contraction input")
    };
    let input = Input {
        activated: primitive(activated),
        weight: primitive(weight),
        residual: primitive(residual),
        gate: primitive(gate),
        batch,
        sequence,
        validation: None,
    };
    let operations = TUNER.init(|| {
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
    let client = input.activated.client.clone();
    let id = CubeTuneId::new(&client, &input.activated.device);
    Tensor::from_primitive::<crate::WgpuRaw>(TUNER.execute(&id, &client, operations, input))
}
