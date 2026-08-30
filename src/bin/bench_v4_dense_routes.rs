//! Same-process paired timing for exact-shape handwritten RF dense routes.

use std::{
    fs::OpenOptions,
    io::BufWriter,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use burn::{
    backend::wgpu::{
        AutoCompiler, CubeTensor, MemoryConfiguration, RuntimeOptions, WgpuDevice, WgpuRuntime,
        graphics::AutoGraphicsApi, init_setup,
    },
    tensor::Tensor,
};
use clap::{Parser, ValueEnum};
use cubecl::{
    future,
    prelude::Runtime,
    profile::{ProfileDuration, TimingMethod},
};
use irodori_tts_burn::{
    WgpuRaw,
    kernels::cubek_mlp_contract::{CubeKMlpContractAlgorithm, try_cubek_mlp_contract_residual},
    kernels::dit_mlp_contract_residual::{
        try_dit_attention_output_direct_residual_vec4_k16_prefetch_wgsl,
        try_dit_attention_output_direct_residual_vec4_k16_wgsl,
        try_dit_attention_output_direct_residual_vec4_wgsl,
        try_dit_mlp_contract_residual_c64_vec4_wgsl,
        try_dit_mlp_contract_residual_double_buffer_vec4_k16_wgsl,
        try_dit_mlp_contract_residual_prefetch_vec4_k16_wgsl,
        try_dit_mlp_contract_residual_rows32_vec4_wgsl,
        try_dit_mlp_contract_residual_rows48_vec4_k16_wgsl,
        try_dit_mlp_contract_residual_rows48_vec4_wgsl,
        try_dit_mlp_contract_residual_rows96_prefetch_vec4_k16_wgsl,
        try_dit_mlp_contract_residual_swizzled_vec4_k16_wgsl,
        try_dit_mlp_contract_residual_vec4_k16_wgsl, try_dit_mlp_contract_residual_vec4_wgsl,
        try_dit_mlp_contract_residual_warp32_k16_wgsl,
        try_dit_mlp_contract_residual_warp32_rows128_wgsl,
    },
    kernels::dit_mlp_contract_split_k::try_dit_mlp_contract_residual_split_k2_wgsl,
    kernels::dit_projection_t64::{
        ATTENTION_QKV_GATE_K, ATTENTION_QKV_GATE_N, EXPAND_K, EXPAND_N,
        try_dit_attention_qkv_gate_c128_k16_prefetch_wgsl,
        try_dit_attention_qkv_gate_c128_k16_wgsl, try_dit_attention_qkv_gate_c128_vec4_wgsl,
        try_dit_mlp_expand_swiglu_c128_vec4_k16_prefetch_wgsl,
        try_dit_mlp_expand_swiglu_c128_vec4_k16_wgsl, try_dit_mlp_expand_swiglu_c128_vec4_wgsl,
    },
    kernels::fused_residual_gate::fused_residual_gate_wgsl,
    kernels::fused_swiglu::try_fused_swiglu_pitched_in_place_wgsl,
};
use serde::Serialize;

type WgpuRt = WgpuRuntime<AutoCompiler>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum DensePair {
    QkvK16,
    QkvPrefetchK16,
    MlpExpandVector,
    MlpExpandK16,
    MlpExpandPrefetchK16,
    MlpContractK16,
    MlpContractRows32,
    MlpContractRows48,
    MlpContractRows48K16,
    MlpContractRows48K32VsK16,
    MlpContractDoubleBufferK16,
    MlpContractDoubleBufferCurrent,
    MlpContractPrefetchCurrent,
    MlpContractSplitK2Current,
    MlpContractRows128Current,
    MlpContractRows96Current,
    MlpContractC64,
    MlpContractWarp32K16,
    MlpContractSwizzledK16,
    MlpContractBurnCore,
    MlpContractBurnGraph,
    MlpContractCubeKDoubleUnit,
    DirectOutputK16,
    DirectOutputPrefetchK32,
    DirectOutputPrefetchK16,
}

impl DensePair {
    const fn label(self) -> &'static str {
        match self {
            Self::QkvK16 => "qkv_k16",
            Self::QkvPrefetchK16 => "qkv_prefetch_k16",
            Self::MlpExpandVector => "mlp_expand_vector",
            Self::MlpExpandK16 => "mlp_expand_k16",
            Self::MlpExpandPrefetchK16 => "mlp_expand_prefetch_k16",
            Self::MlpContractK16 => "mlp_contract_k16",
            Self::MlpContractRows32 => "mlp_contract_rows32",
            Self::MlpContractRows48 => "mlp_contract_rows48",
            Self::MlpContractRows48K16 => "mlp_contract_rows48_k16",
            Self::MlpContractRows48K32VsK16 => "mlp_contract_rows48_k32_vs_k16",
            Self::MlpContractDoubleBufferK16 => "mlp_contract_double_buffer_k16",
            Self::MlpContractDoubleBufferCurrent => "mlp_contract_double_buffer_current",
            Self::MlpContractPrefetchCurrent => "mlp_contract_prefetch_current",
            Self::MlpContractSplitK2Current => "mlp_contract_split_k2_current",
            Self::MlpContractRows128Current => "mlp_contract_rows128_current",
            Self::MlpContractRows96Current => "mlp_contract_rows96_current",
            Self::MlpContractC64 => "mlp_contract_c64",
            Self::MlpContractWarp32K16 => "mlp_contract_warp32_k16",
            Self::MlpContractSwizzledK16 => "mlp_contract_swizzled_k16",
            Self::MlpContractBurnCore => "mlp_contract_burn_core",
            Self::MlpContractBurnGraph => "mlp_contract_burn_graph",
            Self::MlpContractCubeKDoubleUnit => "mlp_contract_cubek_double_unit",
            Self::DirectOutputK16 => "direct_output_k16",
            Self::DirectOutputPrefetchK32 => "direct_output_prefetch_k32",
            Self::DirectOutputPrefetchK16 => "direct_output_prefetch_k16",
        }
    }

    const fn route_label(self, route: Route) -> &'static str {
        match (self, route) {
            (Self::QkvK16, Route::Control) => "handwritten_c128_vector_input",
            (Self::QkvK16, Route::Candidate) => "handwritten_c128_k16",
            (Self::QkvPrefetchK16, Route::Control) => "handwritten_c128_k16",
            (Self::QkvPrefetchK16, Route::Candidate) => "handwritten_c128_k16_prefetched",
            (Self::MlpExpandVector, Route::Control) => "burn_projection_pitched_swiglu",
            (Self::MlpExpandVector, Route::Candidate) => "handwritten_c128_vector_input",
            (Self::MlpExpandK16, Route::Control) => "handwritten_c128_vector_input",
            (Self::MlpExpandK16, Route::Candidate) => "handwritten_c128_k16",
            (Self::MlpExpandPrefetchK16, Route::Control) => "handwritten_c128_k16",
            (Self::MlpExpandPrefetchK16, Route::Candidate) => "handwritten_c128_k16_prefetched",
            (Self::MlpContractK16, Route::Control) => "handwritten_t64_pitched_vector_input",
            (Self::MlpContractK16, Route::Candidate) => "handwritten_k16_pitched_vector_input",
            (Self::MlpContractRows32, Route::Control) => "handwritten_t64_pitched_vector_input",
            (Self::MlpContractRows32, Route::Candidate) => {
                "handwritten_rows32_pitched_vector_input"
            }
            (Self::MlpContractRows48, Route::Control) => "handwritten_t64_pitched_vector_input",
            (Self::MlpContractRows48, Route::Candidate) => {
                "handwritten_rows48_pitched_vector_input"
            }
            (Self::MlpContractRows48K16, Route::Control) => "handwritten_k16_pitched_vector_input",
            (Self::MlpContractRows48K16, Route::Candidate) => {
                "handwritten_rows48_k16_pitched_vector_input"
            }
            (Self::MlpContractRows48K32VsK16, Route::Control) => {
                "handwritten_k16_pitched_vector_input"
            }
            (Self::MlpContractRows48K32VsK16, Route::Candidate) => {
                "handwritten_rows48_k32_pitched_vector_input"
            }
            (Self::MlpContractDoubleBufferK16, Route::Control) => {
                "handwritten_k16_pitched_vector_input"
            }
            (Self::MlpContractDoubleBufferK16, Route::Candidate) => {
                "handwritten_k16_double_buffer_pitched_vector_input"
            }
            (Self::MlpContractDoubleBufferCurrent, Route::Control) => {
                "handwritten_production_incumbent"
            }
            (Self::MlpContractDoubleBufferCurrent, Route::Candidate) => {
                "handwritten_k16_double_buffer_pitched_vector_input"
            }
            (Self::MlpContractPrefetchCurrent, Route::Control) => {
                "handwritten_production_incumbent"
            }
            (Self::MlpContractPrefetchCurrent, Route::Candidate) => {
                "handwritten_k16_prefetched_pitched_vector_input"
            }
            (Self::MlpContractSplitK2Current, Route::Control) => {
                "handwritten_k16_prefetched_pitched_vector_input"
            }
            (Self::MlpContractSplitK2Current, Route::Candidate) => {
                "handwritten_global_split_k2_prefetched"
            }
            (Self::MlpContractRows128Current, Route::Control) => {
                "handwritten_k16_prefetched_pitched_vector_input"
            }
            (Self::MlpContractRows128Current, Route::Candidate) => {
                "handwritten_warp32_rows128_pitched_vector_input"
            }
            (Self::MlpContractRows96Current, Route::Control) => {
                "handwritten_k16_prefetched_pitched_vector_input"
            }
            (Self::MlpContractRows96Current, Route::Candidate) => {
                "handwritten_rows96_k16_prefetched_pitched_vector_input"
            }
            (Self::MlpContractC64, Route::Control) => "handwritten_t64_pitched_vector_input",
            (Self::MlpContractC64, Route::Candidate) => "handwritten_c64_pitched_vector_input",
            (Self::MlpContractWarp32K16, Route::Control) => "handwritten_k16_pitched_vector_input",
            (Self::MlpContractWarp32K16, Route::Candidate) => {
                "handwritten_warp32_k16_pitched_vector_input"
            }
            (Self::MlpContractSwizzledK16, Route::Control) => {
                "handwritten_k16_pitched_vector_input"
            }
            (Self::MlpContractSwizzledK16, Route::Candidate) => {
                "handwritten_k16_swizzled_pitched_vector_input"
            }
            (Self::MlpContractBurnCore, Route::Control) => {
                "handwritten_incumbent_zero_residual_unit_gate"
            }
            (Self::MlpContractBurnCore, Route::Candidate) => "burn_matmul_only",
            (Self::MlpContractBurnGraph, Route::Control) => {
                "handwritten_incumbent_fused_residual_gate"
            }
            (Self::MlpContractBurnGraph, Route::Candidate) => "burn_matmul_plus_wgsl_residual_gate",
            (Self::MlpContractCubeKDoubleUnit, Route::Control) => "handwritten_exact_shape",
            (Self::MlpContractCubeKDoubleUnit, Route::Candidate) => {
                "cubek_double_unit_accumulator_transform"
            }
            (Self::DirectOutputK16, Route::Control) => "direct_output_residual_vector_input",
            (Self::DirectOutputK16, Route::Candidate) => "direct_output_residual_k16_vector_input",
            (Self::DirectOutputPrefetchK32, Route::Control) => {
                "direct_output_residual_vector_input"
            }
            (Self::DirectOutputPrefetchK32, Route::Candidate) => {
                "direct_output_residual_k16_prefetched_vector_input"
            }
            (Self::DirectOutputPrefetchK16, Route::Control) => {
                "direct_output_residual_k16_vector_input"
            }
            (Self::DirectOutputPrefetchK16, Route::Candidate) => {
                "direct_output_residual_k16_prefetched_vector_input"
            }
        }
    }
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, value_enum)]
    pair: DensePair,
    #[arg(long)]
    output_json: PathBuf,
    #[arg(long, default_value_t = 3)]
    batch: usize,
    #[arg(long, default_value_t = 489)]
    sequence: usize,
    #[arg(long, default_value_t = 4)]
    warmups: usize,
    #[arg(long, default_value_t = 10)]
    blocks: usize,
    #[arg(long, default_value_t = 0)]
    adapter_index: usize,
    /// Number of distinct weight buffers rotated through the timed loop.
    /// Twelve models the RF stack's per-layer weight working set and avoids
    /// reporting an unrealistically L2-hot single-weight microbenchmark.
    #[arg(long, default_value_t = 1)]
    weight_working_set: usize,
}

#[derive(Debug, Serialize)]
struct AdapterReceipt {
    name: String,
    vendor_id: u32,
    device_id: u32,
    backend: String,
    driver: String,
    driver_info: String,
}

#[derive(Debug, Serialize)]
struct Sample {
    block: usize,
    slot: usize,
    route: &'static str,
    device_complete_ms: f64,
    gpu_elapsed_ms: f64,
    gpu_timing_source: &'static str,
}

#[derive(Debug, Serialize)]
struct Summary {
    route: &'static str,
    samples: usize,
    median_ms: f64,
    minimum_ms: f64,
    maximum_ms: f64,
    gpu_median_ms: f64,
    gpu_minimum_ms: f64,
    gpu_maximum_ms: f64,
}

#[derive(Debug, Serialize)]
struct AccuracyReceipt {
    elements: usize,
    bitwise_equal: bool,
    maximum_absolute_error: f64,
    rmse: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    timing_boundary: &'static str,
    gpu_timing_boundary: &'static str,
    precision: &'static str,
    pair: &'static str,
    batch: usize,
    sequence: usize,
    rows: usize,
    warmups_per_route: usize,
    blocks: usize,
    weight_working_set: usize,
    adapter: AdapterReceipt,
    samples: Vec<Sample>,
    summaries: Vec<Summary>,
    candidate_minus_control_median_ms: f64,
    candidate_minus_control_gpu_median_ms: f64,
    accuracy: AccuracyReceipt,
    allocator_bytes_in_use: u64,
    allocator_bytes_reserved: u64,
    uncaptured_wgpu_errors: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Route {
    Control,
    Candidate,
}

fn sync(device: &WgpuDevice) -> Result<()> {
    cubecl::future::block_on(WgpuRt::client(device).sync()).context("WGPU sync failed")
}

fn launch_qkv(route: Route, input: &Tensor<2>, weight: &Tensor<2>) -> Result<Tensor<2>> {
    let input = input
        .clone()
        .try_into_primitive::<WgpuRaw>()
        .map_err(|error| anyhow::anyhow!("QKV input is not a WGPU tensor: {error:?}"))?;
    let weight = weight
        .clone()
        .try_into_primitive::<WgpuRaw>()
        .map_err(|error| anyhow::anyhow!("QKV weight is not a WGPU tensor: {error:?}"))?;
    let output = match route {
        Route::Control => try_dit_attention_qkv_gate_c128_vec4_wgsl(input, weight),
        Route::Candidate => try_dit_attention_qkv_gate_c128_k16_wgsl(input, weight),
    }
    .with_context(|| {
        format!(
            "{} rejected the exact QKV shape",
            DensePair::QkvK16.route_label(route)
        )
    })?;
    Ok(Tensor::<2>::from_primitive::<WgpuRaw>(output))
}

fn launch_qkv_prefetch_k16(
    route: Route,
    input: &Tensor<2>,
    weight: &Tensor<2>,
) -> Result<Tensor<2>> {
    let input = input
        .clone()
        .try_into_primitive::<WgpuRaw>()
        .map_err(|error| anyhow::anyhow!("QKV input is not a WGPU tensor: {error:?}"))?;
    let weight = weight
        .clone()
        .try_into_primitive::<WgpuRaw>()
        .map_err(|error| anyhow::anyhow!("QKV weight is not a WGPU tensor: {error:?}"))?;
    let output = match route {
        Route::Control => try_dit_attention_qkv_gate_c128_k16_wgsl(input, weight),
        Route::Candidate => try_dit_attention_qkv_gate_c128_k16_prefetch_wgsl(input, weight),
    }
    .with_context(|| {
        format!(
            "{} rejected the exact QKV shape",
            DensePair::QkvPrefetchK16.route_label(route)
        )
    })?;
    Ok(Tensor::<2>::from_primitive::<WgpuRaw>(output))
}

fn launch_mlp_expand(route: Route, input: &Tensor<2>, weight: &Tensor<2>) -> Result<Tensor<2>> {
    let rows = input.dims()[0];
    match route {
        Route::Control => {
            let projected = input.clone().matmul(weight.clone());
            let pitched = try_fused_swiglu_pitched_in_place_wgsl(into_cube(
                projected,
                "Burn MLP projection",
            )?)
            .context("pitched SwiGLU rejected the Burn projection")?;
            Ok(Tensor::<2>::from_primitive::<WgpuRaw>(pitched).slice([0..rows, 0..EXPAND_N / 2]))
        }
        Route::Candidate => {
            let output = try_dit_mlp_expand_swiglu_c128_vec4_wgsl(
                into_cube(input.clone(), "MLP expand input")?,
                into_cube(weight.clone(), "MLP expand weight")?,
            )
            .context("vector-input MLP expand rejected the exact shape")?;
            Ok(Tensor::<2>::from_primitive::<WgpuRaw>(output))
        }
    }
}

fn launch_mlp_expand_k16(route: Route, input: &Tensor<2>, weight: &Tensor<2>) -> Result<Tensor<2>> {
    let input = into_cube(input.clone(), "MLP expand input")?;
    let weight = into_cube(weight.clone(), "MLP expand weight")?;
    let output = match route {
        Route::Control => try_dit_mlp_expand_swiglu_c128_vec4_wgsl(input, weight),
        Route::Candidate => try_dit_mlp_expand_swiglu_c128_vec4_k16_wgsl(input, weight),
    }
    .with_context(|| {
        format!(
            "{} rejected the exact MLP expand shape",
            DensePair::MlpExpandK16.route_label(route)
        )
    })?;
    Ok(Tensor::<2>::from_primitive::<WgpuRaw>(output))
}

fn launch_mlp_expand_prefetch_k16(
    route: Route,
    input: &Tensor<2>,
    weight: &Tensor<2>,
) -> Result<Tensor<2>> {
    let input = into_cube(input.clone(), "MLP expand input")?;
    let weight = into_cube(weight.clone(), "MLP expand weight")?;
    let output = match route {
        Route::Control => try_dit_mlp_expand_swiglu_c128_vec4_k16_wgsl(input, weight),
        Route::Candidate => try_dit_mlp_expand_swiglu_c128_vec4_k16_prefetch_wgsl(input, weight),
    }
    .with_context(|| {
        format!(
            "{} rejected the exact MLP expand shape",
            DensePair::MlpExpandPrefetchK16.route_label(route)
        )
    })?;
    Ok(Tensor::<2>::from_primitive::<WgpuRaw>(output))
}

#[allow(clippy::too_many_arguments)]
fn launch_mlp_contract(
    pair: DensePair,
    route: Route,
    activated: &Tensor<2>,
    weight: &Tensor<2>,
    residual: &Tensor<2>,
    gate: &Tensor<2>,
    batch: usize,
    sequence: usize,
) -> Result<Tensor<2>> {
    if pair == DensePair::MlpContractBurnCore && route == Route::Candidate {
        return Ok(activated.clone().matmul(weight.clone()));
    }
    if pair == DensePair::MlpContractBurnGraph && route == Route::Candidate {
        let branch = activated.clone().matmul(weight.clone());
        let output = fused_residual_gate_wgsl(
            into_cube(residual.clone(), "MLP residual")?,
            into_cube(branch, "Burn MLP contract output")?,
            into_cube(gate.clone(), "MLP gate")?,
            batch,
            sequence,
        );
        return Ok(Tensor::<2>::from_primitive::<WgpuRaw>(output));
    }
    let activated = into_cube(activated.clone(), "MLP activation")?;
    let weight = into_cube(weight.clone(), "MLP contract weight")?;
    let residual = into_cube(residual.clone(), "MLP residual")?;
    let gate = into_cube(gate.clone(), "MLP gate")?;
    let output = match (pair, route) {
        (
            DensePair::MlpContractK16
            | DensePair::MlpContractRows32
            | DensePair::MlpContractRows48
            | DensePair::MlpContractC64,
            Route::Control,
        ) => try_dit_mlp_contract_residual_vec4_wgsl(
            activated, weight, residual, gate, batch, sequence,
        ),
        (DensePair::MlpContractK16, Route::Candidate) => {
            try_dit_mlp_contract_residual_vec4_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractRows32, Route::Candidate) => {
            try_dit_mlp_contract_residual_rows32_vec4_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractRows48, Route::Candidate) => {
            try_dit_mlp_contract_residual_rows48_vec4_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractRows48K16, Route::Control) => {
            try_dit_mlp_contract_residual_vec4_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractRows48K16, Route::Candidate) => {
            try_dit_mlp_contract_residual_rows48_vec4_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractRows48K32VsK16, Route::Control) => {
            try_dit_mlp_contract_residual_vec4_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractRows48K32VsK16, Route::Candidate) => {
            try_dit_mlp_contract_residual_rows48_vec4_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractDoubleBufferK16, Route::Control) => {
            try_dit_mlp_contract_residual_vec4_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractDoubleBufferK16, Route::Candidate) => {
            try_dit_mlp_contract_residual_double_buffer_vec4_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractDoubleBufferCurrent, Route::Control) if batch == 1 => {
            try_dit_mlp_contract_residual_vec4_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractDoubleBufferCurrent, Route::Control) => {
            try_dit_mlp_contract_residual_vec4_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractDoubleBufferCurrent, Route::Candidate) => {
            try_dit_mlp_contract_residual_double_buffer_vec4_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractPrefetchCurrent, Route::Control) if batch == 1 => {
            try_dit_mlp_contract_residual_vec4_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractPrefetchCurrent, Route::Control) => {
            try_dit_mlp_contract_residual_vec4_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractPrefetchCurrent, Route::Candidate) => {
            try_dit_mlp_contract_residual_prefetch_vec4_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractSplitK2Current, Route::Control) => {
            try_dit_mlp_contract_residual_prefetch_vec4_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractSplitK2Current, Route::Candidate) => {
            try_dit_mlp_contract_residual_split_k2_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractRows128Current, Route::Control) => {
            try_dit_mlp_contract_residual_prefetch_vec4_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractRows128Current, Route::Candidate) => {
            try_dit_mlp_contract_residual_warp32_rows128_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractRows96Current, Route::Control) => {
            try_dit_mlp_contract_residual_prefetch_vec4_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractRows96Current, Route::Candidate) => {
            try_dit_mlp_contract_residual_rows96_prefetch_vec4_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractC64, Route::Candidate) => {
            try_dit_mlp_contract_residual_c64_vec4_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractWarp32K16, Route::Control) => {
            try_dit_mlp_contract_residual_vec4_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractWarp32K16, Route::Candidate) => {
            try_dit_mlp_contract_residual_warp32_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractSwizzledK16, Route::Control) => {
            try_dit_mlp_contract_residual_vec4_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractSwizzledK16, Route::Candidate) => {
            try_dit_mlp_contract_residual_swizzled_vec4_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractBurnCore | DensePair::MlpContractBurnGraph, Route::Control)
            if batch == 3 =>
        {
            try_dit_mlp_contract_residual_vec4_k16_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        (DensePair::MlpContractBurnCore | DensePair::MlpContractBurnGraph, Route::Control) => {
            try_dit_mlp_contract_residual_vec4_wgsl(
                activated, weight, residual, gate, batch, sequence,
            )
        }
        _ => unreachable!("non-contract pair passed to the contract launcher"),
    }
    .with_context(|| {
        format!(
            "{} rejected the exact MLP contract shape",
            pair.route_label(route)
        )
    })?;
    Ok(Tensor::<2>::from_primitive::<WgpuRaw>(output))
}

#[allow(clippy::too_many_arguments)]
fn launch_mlp_contract_cubek_double_unit(
    route: Route,
    activated: &Tensor<2>,
    row_weight: &Tensor<2>,
    column_weight: &Tensor<2>,
    residual: &Tensor<2>,
    gate: &Tensor<2>,
    batch: usize,
    sequence: usize,
) -> Result<Tensor<2>> {
    let activated = into_cube(activated.clone(), "MLP activation")?;
    let residual = into_cube(residual.clone(), "MLP residual")?;
    let gate = into_cube(gate.clone(), "MLP gate")?;
    let output = match route {
        Route::Control if batch == 3 => try_dit_mlp_contract_residual_vec4_k16_wgsl(
            activated,
            into_cube(row_weight.clone(), "row-major MLP contract weight")?,
            residual,
            gate,
            batch,
            sequence,
        ),
        Route::Control => try_dit_mlp_contract_residual_vec4_wgsl(
            activated,
            into_cube(row_weight.clone(), "row-major MLP contract weight")?,
            residual,
            gate,
            batch,
            sequence,
        ),
        Route::Candidate => try_cubek_mlp_contract_residual(
            activated,
            into_cube(
                column_weight.clone(),
                "column-major CubeK MLP contract weight",
            )?,
            residual,
            gate,
            batch,
            sequence,
            CubeKMlpContractAlgorithm::DoubleUnit,
        ),
    }
    .context("exact MLP contract route rejected its physical contract")?;
    Ok(Tensor::<2>::from_primitive::<WgpuRaw>(output))
}

#[allow(clippy::too_many_arguments)]
fn launch_direct_output(
    pair: DensePair,
    route: Route,
    attention: &Tensor<4>,
    attention_gate: &Tensor<3>,
    weight: &Tensor<2>,
    residual: &Tensor<2>,
    block_gate: &Tensor<2>,
    batch: usize,
    sequence: usize,
) -> Result<Tensor<2>> {
    let attention = into_cube(attention.clone(), "head-major attention")?;
    let attention_gate = into_cube(attention_gate.clone(), "attention gate")?;
    let weight = into_cube(weight.clone(), "attention output weight")?;
    let residual = into_cube(residual.clone(), "attention residual")?;
    let block_gate = into_cube(block_gate.clone(), "attention block gate")?;
    let output = match (pair, route) {
        (DensePair::DirectOutputK16, Route::Control)
        | (DensePair::DirectOutputPrefetchK32, Route::Control) => {
            try_dit_attention_output_direct_residual_vec4_wgsl(
                attention,
                attention_gate,
                weight,
                residual,
                block_gate,
                batch,
                sequence,
            )
        }
        (DensePair::DirectOutputK16, Route::Candidate)
        | (DensePair::DirectOutputPrefetchK16, Route::Control) => {
            try_dit_attention_output_direct_residual_vec4_k16_wgsl(
                attention,
                attention_gate,
                weight,
                residual,
                block_gate,
                batch,
                sequence,
            )
        }
        (DensePair::DirectOutputPrefetchK16, Route::Candidate)
        | (DensePair::DirectOutputPrefetchK32, Route::Candidate) => {
            try_dit_attention_output_direct_residual_vec4_k16_prefetch_wgsl(
                attention,
                attention_gate,
                weight,
                residual,
                block_gate,
                batch,
                sequence,
            )
        }
        _ => anyhow::bail!("{pair:?} is not a direct-output pair"),
    }
    .with_context(|| {
        format!(
            "{} rejected the exact direct-output shape",
            pair.route_label(route)
        )
    })?;
    Ok(Tensor::<2>::from_primitive::<WgpuRaw>(output))
}

fn into_cube<const D: usize>(tensor: Tensor<D>, label: &str) -> Result<CubeTensor<WgpuRt>> {
    tensor
        .try_into_primitive::<WgpuRaw>()
        .map_err(|error| anyhow::anyhow!("{label} is not a WGPU tensor: {error:?}"))
}

struct Measurement {
    device_complete_ms: f64,
    gpu_elapsed_ms: f64,
    gpu_timing_source: &'static str,
}

fn resolve_profile_duration(duration: ProfileDuration) -> MeasurementGpuDuration {
    let timing_source = match duration.timing_method() {
        TimingMethod::Device => "device_timestamp",
        TimingMethod::System => "synchronized_system_clock",
    };
    let elapsed_ms = future::block_on(duration.resolve())
        .duration()
        .as_secs_f64()
        * 1_000.0;
    MeasurementGpuDuration {
        elapsed_ms,
        timing_source,
    }
}

struct MeasurementGpuDuration {
    elapsed_ms: f64,
    timing_source: &'static str,
}

fn measure<F>(device: &WgpuDevice, route: Route, launch: &F) -> Result<Measurement>
where
    F: Fn(Route) -> Result<Tensor<2>> + Sync,
{
    sync(device)?;
    let client = WgpuRt::client(device);
    let started = Instant::now();
    let (output, duration) = client
        .profile(|| launch(route), "paired_dense_route")
        .context("dense-route GPU timestamp scope failed")?;
    let output = output?;
    sync(device)?;
    std::hint::black_box(output.dims());
    let device_complete_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let gpu = resolve_profile_duration(duration);
    Ok(Measurement {
        device_complete_ms,
        gpu_elapsed_ms: gpu.elapsed_ms,
        gpu_timing_source: gpu.timing_source,
    })
}

fn median(values: &[f64]) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let middle = ordered.len() / 2;
    if ordered.len().is_multiple_of(2) {
        (ordered[middle - 1] + ordered[middle]) / 2.0
    } else {
        ordered[middle]
    }
}

fn summary(route_label: &'static str, samples: &[Sample]) -> Summary {
    let values = samples
        .iter()
        .filter(|sample| sample.route == route_label)
        .map(|sample| sample.device_complete_ms)
        .collect::<Vec<_>>();
    let gpu_values = samples
        .iter()
        .filter(|sample| sample.route == route_label)
        .map(|sample| sample.gpu_elapsed_ms)
        .collect::<Vec<_>>();
    Summary {
        route: route_label,
        samples: values.len(),
        median_ms: median(&values),
        minimum_ms: values.iter().copied().fold(f64::INFINITY, f64::min),
        maximum_ms: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        gpu_median_ms: median(&gpu_values),
        gpu_minimum_ms: gpu_values.iter().copied().fold(f64::INFINITY, f64::min),
        gpu_maximum_ms: gpu_values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

fn accuracy<F>(launch: &F) -> Result<AccuracyReceipt>
where
    F: Fn(Route) -> Result<Tensor<2>> + Sync,
{
    let control = launch(Route::Control)?.into_data().to_vec::<f32>()?;
    let candidate = launch(Route::Candidate)?.into_data().to_vec::<f32>()?;
    ensure!(
        control.len() == candidate.len(),
        "dense-route output length mismatch"
    );
    let (maximum_absolute_error, squared_error) = control.iter().zip(&candidate).fold(
        (0.0_f64, 0.0_f64),
        |(maximum, squared), (&lhs, &rhs)| {
            let difference = f64::from(lhs) - f64::from(rhs);
            (
                maximum.max(difference.abs()),
                squared + difference * difference,
            )
        },
    );
    Ok(AccuracyReceipt {
        elements: control.len(),
        bitwise_equal: control == candidate,
        maximum_absolute_error,
        rmse: (squared_error / control.len() as f64).sqrt(),
    })
}

fn run_pair<F>(
    args: &Args,
    device: &WgpuDevice,
    launch: F,
) -> Result<(Vec<Sample>, AccuracyReceipt)>
where
    F: Fn(Route) -> Result<Tensor<2>> + Sync,
{
    for _ in 0..args.warmups {
        for route in [Route::Control, Route::Candidate] {
            let _ = measure(device, route, &launch)?;
        }
    }

    let mut samples = Vec::with_capacity(args.blocks * 4);
    for block in 0..args.blocks {
        let order = if block.is_multiple_of(2) {
            [
                Route::Control,
                Route::Candidate,
                Route::Candidate,
                Route::Control,
            ]
        } else {
            [
                Route::Candidate,
                Route::Control,
                Route::Control,
                Route::Candidate,
            ]
        };
        for (slot, route) in order.into_iter().enumerate() {
            let measurement = measure(device, route, &launch)?;
            let route_label = args.pair.route_label(route);
            tracing::info!(
                block,
                slot,
                route = route_label,
                device_complete_ms = measurement.device_complete_ms,
                gpu_elapsed_ms = measurement.gpu_elapsed_ms,
                gpu_timing_source = measurement.gpu_timing_source,
                "paired dense-route sample"
            );
            samples.push(Sample {
                block,
                slot,
                route: route_label,
                device_complete_ms: measurement.device_complete_ms,
                gpu_elapsed_ms: measurement.gpu_elapsed_ms,
                gpu_timing_source: measurement.gpu_timing_source,
            });
        }
    }
    Ok((samples, accuracy(&launch)?))
}

fn main() -> Result<()> {
    irodori_tts_burn::backend_config::initialize_cli_tracing("info")?;
    let args = Args::parse();
    ensure!(args.batch > 0, "batch must be positive");
    ensure!(args.sequence > 0, "sequence must be positive");
    ensure!(args.warmups > 0, "warmups must be positive");
    ensure!(args.blocks > 0, "blocks must be positive");
    ensure!(
        args.weight_working_set > 0,
        "weight-working-set must be positive"
    );
    ensure!(
        !args.output_json.exists(),
        "refusing to overwrite {}",
        args.output_json.display()
    );
    let wgpu_device = WgpuDevice::DiscreteGpu(args.adapter_index);
    let setup = init_setup::<AutoGraphicsApi>(
        &wgpu_device,
        RuntimeOptions {
            tasks_max: 32,
            memory_config: MemoryConfiguration::ExclusivePages,
        },
    );
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    let callback_errors = Arc::clone(&errors);
    setup.device.on_uncaptured_error(Arc::new(move |error| {
        if let Ok(mut errors) = callback_errors.lock() {
            errors.push(error.to_string());
        }
    }));
    let device = irodori_tts_burn::backend_config::strict_fp32_device(&wgpu_device)?;
    let rows = args
        .batch
        .checked_mul(args.sequence)
        .context("batch * sequence overflow")?;
    let (samples, accuracy) = match args.pair {
        DensePair::QkvK16 => {
            let input = Tensor::<2>::ones([rows, ATTENTION_QKV_GATE_K], &device);
            let weight = Tensor::<2>::ones([ATTENTION_QKV_GATE_K, ATTENTION_QKV_GATE_N], &device);
            run_pair(&args, &wgpu_device, |route| {
                launch_qkv(route, &input, &weight)
            })?
        }
        DensePair::QkvPrefetchK16 => {
            let input = Tensor::<2>::ones([rows, ATTENTION_QKV_GATE_K], &device);
            let weight = Tensor::<2>::ones([ATTENTION_QKV_GATE_K, ATTENTION_QKV_GATE_N], &device);
            run_pair(&args, &wgpu_device, |route| {
                launch_qkv_prefetch_k16(route, &input, &weight)
            })?
        }
        DensePair::MlpExpandVector => {
            let input = Tensor::<2>::ones([rows, EXPAND_K], &device);
            let weight = Tensor::<2>::ones([EXPAND_K, EXPAND_N], &device);
            run_pair(&args, &wgpu_device, |route| {
                launch_mlp_expand(route, &input, &weight)
            })?
        }
        DensePair::MlpExpandK16 => {
            let input = Tensor::<2>::ones([rows, EXPAND_K], &device);
            let weight = Tensor::<2>::ones([EXPAND_K, EXPAND_N], &device);
            run_pair(&args, &wgpu_device, |route| {
                launch_mlp_expand_k16(route, &input, &weight)
            })?
        }
        DensePair::MlpExpandPrefetchK16 => {
            let input = Tensor::<2>::ones([rows, EXPAND_K], &device);
            let weight = Tensor::<2>::ones([EXPAND_K, EXPAND_N], &device);
            run_pair(&args, &wgpu_device, |route| {
                launch_mlp_expand_prefetch_k16(route, &input, &weight)
            })?
        }
        DensePair::MlpContractK16
        | DensePair::MlpContractRows32
        | DensePair::MlpContractRows48
        | DensePair::MlpContractRows48K16
        | DensePair::MlpContractRows48K32VsK16
        | DensePair::MlpContractDoubleBufferK16
        | DensePair::MlpContractDoubleBufferCurrent
        | DensePair::MlpContractPrefetchCurrent
        | DensePair::MlpContractSplitK2Current
        | DensePair::MlpContractRows128Current
        | DensePair::MlpContractRows96Current
        | DensePair::MlpContractC64
        | DensePair::MlpContractWarp32K16
        | DensePair::MlpContractSwizzledK16
        | DensePair::MlpContractBurnCore
        | DensePair::MlpContractBurnGraph => {
            const INPUT_DIM: usize = 3_680;
            const OUTPUT_DIM: usize = 1_280;
            let activated = if matches!(
                args.pair,
                DensePair::MlpContractBurnCore | DensePair::MlpContractBurnGraph
            ) {
                Tensor::<2>::ones([rows, INPUT_DIM], &device)
            } else if args.pair == DensePair::MlpContractSplitK2Current {
                (Tensor::<2>::ones([rows, 2 * INPUT_DIM], &device) * 0.003_141_592_7_f32)
                    .slice([0..rows, 0..INPUT_DIM])
            } else {
                Tensor::<2>::ones([rows, 2 * INPUT_DIM], &device).slice([0..rows, 0..INPUT_DIM])
            };
            let weights = (0..args.weight_working_set)
                .map(|_| {
                    let weight = Tensor::<2>::ones([INPUT_DIM, OUTPUT_DIM], &device);
                    if args.pair == DensePair::MlpContractSplitK2Current {
                        weight * 0.001_234_567_f32
                    } else {
                        weight
                    }
                })
                .collect::<Vec<_>>();
            let next_weight = AtomicUsize::new(0);
            let residual = if matches!(args.pair, DensePair::MlpContractBurnCore) {
                Tensor::<2>::zeros([rows, OUTPUT_DIM], &device)
            } else if args.pair == DensePair::MlpContractSplitK2Current {
                Tensor::<2>::ones([rows, OUTPUT_DIM], &device) * 0.031_25_f32
            } else {
                Tensor::<2>::ones([rows, OUTPUT_DIM], &device)
            };
            let gate = if args.pair == DensePair::MlpContractSplitK2Current {
                Tensor::<2>::ones([args.batch, OUTPUT_DIM], &device) * 0.125_f32
            } else {
                Tensor::<2>::ones([args.batch, OUTPUT_DIM], &device)
            };
            run_pair(&args, &wgpu_device, |route| {
                let weight = &weights[next_weight.fetch_add(1, Ordering::Relaxed) % weights.len()];
                launch_mlp_contract(
                    args.pair,
                    route,
                    &activated,
                    weight,
                    &residual,
                    &gate,
                    args.batch,
                    args.sequence,
                )
            })?
        }
        DensePair::MlpContractCubeKDoubleUnit => {
            const INPUT_DIM: usize = 3_680;
            const OUTPUT_DIM: usize = 1_280;
            let activated = Tensor::<2>::ones([rows, INPUT_DIM], &device);
            let row_weight = Tensor::<2>::ones([INPUT_DIM, OUTPUT_DIM], &device);
            let column_weight = Tensor::<2>::ones([OUTPUT_DIM, INPUT_DIM], &device).transpose();
            let residual = Tensor::<2>::ones([rows, OUTPUT_DIM], &device);
            let gate = Tensor::<2>::ones([args.batch, OUTPUT_DIM], &device);
            run_pair(&args, &wgpu_device, |route| {
                launch_mlp_contract_cubek_double_unit(
                    route,
                    &activated,
                    &row_weight,
                    &column_weight,
                    &residual,
                    &gate,
                    args.batch,
                    args.sequence,
                )
            })?
        }
        DensePair::DirectOutputK16
        | DensePair::DirectOutputPrefetchK32
        | DensePair::DirectOutputPrefetchK16 => {
            const OUTPUT_DIM: usize = 1_280;
            let attention = Tensor::<4>::ones([args.batch, 20, args.sequence, 64], &device);
            let attention_gate =
                Tensor::<3>::ones([args.batch, args.sequence, 4 * OUTPUT_DIM], &device);
            let weight = Tensor::<2>::ones([OUTPUT_DIM, OUTPUT_DIM], &device);
            let residual = Tensor::<2>::ones([rows, OUTPUT_DIM], &device);
            let block_gate = Tensor::<2>::ones([args.batch, OUTPUT_DIM], &device);
            run_pair(&args, &wgpu_device, |route| {
                launch_direct_output(
                    args.pair,
                    route,
                    &attention,
                    &attention_gate,
                    &weight,
                    &residual,
                    &block_gate,
                    args.batch,
                    args.sequence,
                )
            })?
        }
    };
    let control_label = args.pair.route_label(Route::Control);
    let candidate_label = args.pair.route_label(Route::Candidate);
    let control = summary(control_label, &samples);
    let candidate = summary(candidate_label, &samples);
    sync(&wgpu_device)?;
    let usage = WgpuRt::client(&wgpu_device)
        .memory_usage()
        .context("WGPU memory query failed")?;
    let errors = errors.lock().expect("WGPU error monitor poisoned").clone();
    let info = setup.adapter.get_info();
    let report = Report {
        schema_version: 2,
        timing_boundary: "pre-start device sync through post-kernel device completion",
        gpu_timing_boundary: "CubeCL timestamp scope around route enqueue and GPU execution",
        precision: "strict_fp32",
        pair: args.pair.label(),
        batch: args.batch,
        sequence: args.sequence,
        rows,
        warmups_per_route: args.warmups,
        blocks: args.blocks,
        weight_working_set: args.weight_working_set,
        adapter: AdapterReceipt {
            name: info.name,
            vendor_id: info.vendor,
            device_id: info.device,
            backend: format!("{:?}", info.backend),
            driver: info.driver,
            driver_info: info.driver_info,
        },
        candidate_minus_control_median_ms: candidate.median_ms - control.median_ms,
        candidate_minus_control_gpu_median_ms: candidate.gpu_median_ms - control.gpu_median_ms,
        summaries: vec![control, candidate],
        samples,
        accuracy,
        allocator_bytes_in_use: usage.bytes_in_use,
        allocator_bytes_reserved: usage.bytes_reserved,
        uncaptured_wgpu_errors: errors,
    };
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output_json)
        .with_context(|| format!("failed to create {}", args.output_json.display()))?;
    serde_json::to_writer_pretty(BufWriter::new(output), &report)?;
    Ok(())
}
