//! Shared exact-layout tuning for QKV, attention-output and SwiGLU projections.

use super::{
    dit_projection_t64::*,
    matmul_cmma::{self, CmmaPlan},
    tuning_validation::{check_accuracy, read},
};
use burn::{
    backend::wgpu::CubeTensor,
    tensor::{DType, Tensor},
};
use cubecl::{
    CubeTuneId,
    tune::{LocalTuner, Tunable, TunableSet, local_tuner},
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub(crate) enum ProjectionKind {
    Qkv,
    SwiGlu,
    AttentionOutput,
}

type GpuTensor = CubeTensor;
type Launcher = fn(GpuTensor, GpuTensor) -> Option<GpuTensor>;

#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
struct Key {
    schema: u16,
    kind: ProjectionKind,
    device: String,
    dtype: String,
    shapes: [[usize; 2]; 2],
    strides: [Vec<usize>; 2],
}
impl std::fmt::Display for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "projection-v{}-{:?}-{}-{}-{:?}-{:?}",
            self.schema, self.kind, self.device, self.dtype, self.shapes, self.strides
        )
    }
}
impl cubecl::tune::AutotuneKey for Key {}

#[derive(Clone)]
struct Input {
    x: GpuTensor,
    weight: GpuTensor,
    kind: ProjectionKind,
    validation: Option<Arc<Vec<Result<(), String>>>>,
}

#[derive(Clone, Copy)]
enum Candidate {
    Reference,
    Burn,
    Cmma(CmmaPlan),
    Wgsl {
        qkv: Option<Launcher>,
        output: Option<ProjectionTileLayout>,
        swiglu: Launcher,
    },
}
const CANDIDATES: &[(&str, Candidate)] = &[
    ("fp32-reference", Candidate::Reference),
    ("burn-graph", Candidate::Burn),
    (
        "wgsl-scalar-k32",
        Candidate::Wgsl {
            qkv: Some(try_dit_attention_qkv_gate_t64_wgsl),
            output: Some(ProjectionTileLayout::C128ScalarK32),
            swiglu: try_dit_mlp_expand_swiglu_c128_wgsl,
        },
    ),
    (
        "wgsl-vector-k32",
        Candidate::Wgsl {
            qkv: Some(try_dit_attention_qkv_gate_c128_vec4_wgsl),
            output: Some(ProjectionTileLayout::C128VectorK32),
            swiglu: try_dit_mlp_expand_swiglu_c128_vec4_wgsl,
        },
    ),
    (
        "wgsl-vector-k16",
        Candidate::Wgsl {
            qkv: Some(try_dit_attention_qkv_gate_c128_k16_wgsl),
            output: Some(ProjectionTileLayout::C128VectorK16),
            swiglu: try_dit_mlp_expand_swiglu_c128_vec4_k16_wgsl,
        },
    ),
    (
        "wgsl-prefetch-k16",
        Candidate::Wgsl {
            qkv: Some(try_dit_attention_qkv_gate_c128_k16_prefetch_wgsl),
            output: Some(ProjectionTileLayout::C128VectorK16Prefetched),
            swiglu: try_dit_mlp_expand_swiglu_c128_vec4_k16_prefetch_wgsl,
        },
    ),
    (
        "wgsl-rows128",
        Candidate::Wgsl {
            qkv: Some(try_dit_attention_qkv_gate_c128_rows128_k16_wgsl),
            output: Some(ProjectionTileLayout::C128VectorRows128K16),
            swiglu: try_dit_mlp_expand_swiglu_warp32_rows128_wgsl,
        },
    ),
    (
        "wgsl-warp32",
        Candidate::Wgsl {
            qkv: None,
            output: None,
            swiglu: try_dit_mlp_expand_swiglu_warp32_wgsl,
        },
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

fn key(input: &Input) -> Key {
    let tensors = [&input.x, &input.weight];
    Key {
        schema: 2,
        kind: input.kind,
        device: input.x.client.properties().identity.fingerprint.clone(),
        dtype: input.x.dtype.name().to_owned(),
        shapes: tensors.map(|x| x.meta.shape().dims::<2>()),
        strides: tensors.map(|x| x.meta.strides().to_vec()),
    }
}

fn graph(input: &Input, fp32: bool) -> GpuTensor {
    let dtype = if fp32 { DType::F32 } else { input.x.dtype };
    let tensor =
        |x: &GpuTensor| Tensor::<2>::from_primitive::<crate::WgpuRaw>(x.clone()).cast(dtype);
    let projected = tensor(&input.x).matmul(tensor(&input.weight));
    let output = match input.kind {
        ProjectionKind::Qkv | ProjectionKind::AttentionOutput => projected,
        ProjectionKind::SwiGlu => {
            Tensor::from_primitive::<crate::WgpuRaw>(super::fused_swiglu::fused_swiglu_wgsl(
                projected
                    .try_into_primitive::<crate::WgpuRaw>()
                    .expect("WGPU projection"),
            ))
        }
    };
    output
        .cast(input.x.dtype)
        .try_into_primitive::<crate::WgpuRaw>()
        .expect("WGPU projection output")
}

fn launch(candidate: Candidate, input: &Input) -> Result<GpuTensor, String> {
    match candidate {
        Candidate::Reference => Ok(graph(input, true)),
        Candidate::Burn => Ok(graph(input, false)),
        Candidate::Cmma(plan) => {
            let projected = matmul_cmma::matmul(input.x.clone(), input.weight.clone(), plan)?;
            Ok(match input.kind {
                ProjectionKind::SwiGlu => super::fused_swiglu::fused_swiglu_wgsl(projected),
                _ => projected,
            })
        }
        Candidate::Wgsl {
            qkv,
            output,
            swiglu,
        } => {
            let unsupported = || "tile does not implement this projection".to_owned();
            let result = match input.kind {
                ProjectionKind::Qkv => {
                    qkv.ok_or_else(unsupported)?(input.x.clone(), input.weight.clone())
                }
                ProjectionKind::SwiGlu => swiglu(input.x.clone(), input.weight.clone()),
                ProjectionKind::AttentionOutput => try_attention_output_with_tile(
                    input.x.clone(),
                    input.weight.clone(),
                    output.ok_or_else(unsupported)?,
                ),
            };
            result.ok_or_else(|| "projection launch contract unsupported".to_owned())
        }
    }
}

fn tuning_inputs(_: &Key, input: &Input) -> Input {
    let expected = read::<2>(graph(input, true)).expect("projection tuning reference readback");
    let half = input.x.dtype == DType::F16;
    check_accuracy(&expected, &expected, half).expect("projection reference must be finite");
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

pub(crate) fn project(x: Tensor<2>, weight: Tensor<2>, kind: ProjectionKind) -> Tensor<2> {
    static TUNER: LocalTuner<Key, CubeTuneId> = local_tuner!("validated-projection-v1");
    let primitive = |x: Tensor<2>| {
        x.try_into_primitive::<crate::WgpuRaw>()
            .expect("WGPU projection input")
    };
    let input = Input {
        x: primitive(x),
        weight: primitive(weight),
        kind,
        validation: None,
    };
    let operations = TUNER.init(&CubeTuneId::new(&input.x.client, &input.x.device), || {
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
    let client = input.x.client.clone();
    let id = CubeTuneId::new(&client, &input.x.device);
    Tensor::from_primitive::<crate::WgpuRaw>(TUNER.execute(&id, &client, operations, input))
}
