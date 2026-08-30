//! Same-process paired timing for exact-shape handwritten RF dense routes.

use std::{
    fs::OpenOptions,
    io::BufWriter,
    path::PathBuf,
    sync::{Arc, Mutex},
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
use cubecl::prelude::Runtime;
use irodori_tts_burn::{
    WgpuRaw,
    kernels::dit_mlp_contract_residual::{
        try_dit_attention_output_direct_residual_vec4_k16_wgsl,
        try_dit_attention_output_direct_residual_vec4_wgsl,
        try_dit_mlp_contract_residual_vec4_k16_wgsl, try_dit_mlp_contract_residual_vec4_wgsl,
    },
    kernels::dit_projection_t64::{
        ATTENTION_QKV_GATE_K, ATTENTION_QKV_GATE_N, EXPAND_K, EXPAND_N,
        try_dit_attention_qkv_gate_c128_k16_wgsl, try_dit_attention_qkv_gate_c128_vec4_wgsl,
        try_dit_mlp_expand_swiglu_c128_vec4_wgsl,
    },
    kernels::fused_swiglu::try_fused_swiglu_pitched_in_place_wgsl,
};
use serde::Serialize;

type WgpuRt = WgpuRuntime<AutoCompiler>;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DensePair {
    QkvK16,
    MlpExpandVector,
    MlpContractK16,
    DirectOutputK16,
}

impl DensePair {
    const fn label(self) -> &'static str {
        match self {
            Self::QkvK16 => "qkv_k16",
            Self::MlpExpandVector => "mlp_expand_vector",
            Self::MlpContractK16 => "mlp_contract_k16",
            Self::DirectOutputK16 => "direct_output_k16",
        }
    }

    const fn route_label(self, route: Route) -> &'static str {
        match (self, route) {
            (Self::QkvK16, Route::Control) => "handwritten_c128_vector_input",
            (Self::QkvK16, Route::Candidate) => "handwritten_c128_k16",
            (Self::MlpExpandVector, Route::Control) => "burn_projection_pitched_swiglu",
            (Self::MlpExpandVector, Route::Candidate) => "handwritten_c128_vector_input",
            (Self::MlpContractK16, Route::Control) => "handwritten_t64_pitched_vector_input",
            (Self::MlpContractK16, Route::Candidate) => "handwritten_k16_pitched_vector_input",
            (Self::DirectOutputK16, Route::Control) => "direct_output_residual_vector_input",
            (Self::DirectOutputK16, Route::Candidate) => "direct_output_residual_k16_vector_input",
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
}

#[derive(Debug, Serialize)]
struct Summary {
    route: &'static str,
    samples: usize,
    median_ms: f64,
    minimum_ms: f64,
    maximum_ms: f64,
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
    precision: &'static str,
    pair: &'static str,
    batch: usize,
    sequence: usize,
    rows: usize,
    warmups_per_route: usize,
    blocks: usize,
    adapter: AdapterReceipt,
    samples: Vec<Sample>,
    summaries: Vec<Summary>,
    candidate_minus_control_median_ms: f64,
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

#[allow(clippy::too_many_arguments)]
fn launch_mlp_contract(
    route: Route,
    activated: &Tensor<2>,
    weight: &Tensor<2>,
    residual: &Tensor<2>,
    gate: &Tensor<2>,
    batch: usize,
    sequence: usize,
) -> Result<Tensor<2>> {
    let activated = into_cube(activated.clone(), "MLP activation")?;
    let weight = into_cube(weight.clone(), "MLP contract weight")?;
    let residual = into_cube(residual.clone(), "MLP residual")?;
    let gate = into_cube(gate.clone(), "MLP gate")?;
    let output = match route {
        Route::Control => try_dit_mlp_contract_residual_vec4_wgsl(
            activated, weight, residual, gate, batch, sequence,
        ),
        Route::Candidate => try_dit_mlp_contract_residual_vec4_k16_wgsl(
            activated, weight, residual, gate, batch, sequence,
        ),
    }
    .with_context(|| {
        format!(
            "{} rejected the exact MLP contract shape",
            DensePair::MlpContractK16.route_label(route)
        )
    })?;
    Ok(Tensor::<2>::from_primitive::<WgpuRaw>(output))
}

#[allow(clippy::too_many_arguments)]
fn launch_direct_output(
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
    let output = match route {
        Route::Control => try_dit_attention_output_direct_residual_vec4_wgsl(
            attention,
            attention_gate,
            weight,
            residual,
            block_gate,
            batch,
            sequence,
        ),
        Route::Candidate => try_dit_attention_output_direct_residual_vec4_k16_wgsl(
            attention,
            attention_gate,
            weight,
            residual,
            block_gate,
            batch,
            sequence,
        ),
    }
    .with_context(|| {
        format!(
            "{} rejected the exact direct-output shape",
            DensePair::DirectOutputK16.route_label(route)
        )
    })?;
    Ok(Tensor::<2>::from_primitive::<WgpuRaw>(output))
}

fn into_cube<const D: usize>(tensor: Tensor<D>, label: &str) -> Result<CubeTensor<WgpuRt>> {
    tensor
        .try_into_primitive::<WgpuRaw>()
        .map_err(|error| anyhow::anyhow!("{label} is not a WGPU tensor: {error:?}"))
}

fn measure<F>(device: &WgpuDevice, route: Route, launch: &F) -> Result<f64>
where
    F: Fn(Route) -> Result<Tensor<2>>,
{
    sync(device)?;
    let started = Instant::now();
    let output = launch(route)?;
    sync(device)?;
    std::hint::black_box(output.dims());
    Ok(started.elapsed().as_secs_f64() * 1_000.0)
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
    Summary {
        route: route_label,
        samples: values.len(),
        median_ms: median(&values),
        minimum_ms: values.iter().copied().fold(f64::INFINITY, f64::min),
        maximum_ms: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

fn accuracy<F>(launch: &F) -> Result<AccuracyReceipt>
where
    F: Fn(Route) -> Result<Tensor<2>>,
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
    F: Fn(Route) -> Result<Tensor<2>>,
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
            let device_complete_ms = measure(device, route, &launch)?;
            let route_label = args.pair.route_label(route);
            tracing::info!(
                block,
                slot,
                route = route_label,
                device_complete_ms,
                "paired dense-route sample"
            );
            samples.push(Sample {
                block,
                slot,
                route: route_label,
                device_complete_ms,
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
        DensePair::MlpExpandVector => {
            let input = Tensor::<2>::ones([rows, EXPAND_K], &device);
            let weight = Tensor::<2>::ones([EXPAND_K, EXPAND_N], &device);
            run_pair(&args, &wgpu_device, |route| {
                launch_mlp_expand(route, &input, &weight)
            })?
        }
        DensePair::MlpContractK16 => {
            const INPUT_DIM: usize = 3_680;
            const OUTPUT_DIM: usize = 1_280;
            let activated =
                Tensor::<2>::ones([rows, 2 * INPUT_DIM], &device).slice([0..rows, 0..INPUT_DIM]);
            let weight = Tensor::<2>::ones([INPUT_DIM, OUTPUT_DIM], &device);
            let residual = Tensor::<2>::ones([rows, OUTPUT_DIM], &device);
            let gate = Tensor::<2>::ones([args.batch, OUTPUT_DIM], &device);
            run_pair(&args, &wgpu_device, |route| {
                launch_mlp_contract(
                    route,
                    &activated,
                    &weight,
                    &residual,
                    &gate,
                    args.batch,
                    args.sequence,
                )
            })?
        }
        DensePair::DirectOutputK16 => {
            const OUTPUT_DIM: usize = 1_280;
            let attention = Tensor::<4>::ones([args.batch, 20, args.sequence, 64], &device);
            let attention_gate =
                Tensor::<3>::ones([args.batch, args.sequence, 4 * OUTPUT_DIM], &device);
            let weight = Tensor::<2>::ones([OUTPUT_DIM, OUTPUT_DIM], &device);
            let residual = Tensor::<2>::ones([rows, OUTPUT_DIM], &device);
            let block_gate = Tensor::<2>::ones([args.batch, OUTPUT_DIM], &device);
            run_pair(&args, &wgpu_device, |route| {
                launch_direct_output(
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
        schema_version: 1,
        timing_boundary: "pre-start device sync through post-kernel device completion",
        precision: "strict_fp32",
        pair: args.pair.label(),
        batch: args.batch,
        sequence: args.sequence,
        rows,
        warmups_per_route: args.warmups,
        blocks: args.blocks,
        adapter: AdapterReceipt {
            name: info.name,
            vendor_id: info.vendor,
            device_id: info.device,
            backend: format!("{:?}", info.backend),
            driver: info.driver,
            driver_info: info.driver_info,
        },
        candidate_minus_control_median_ms: candidate.median_ms - control.median_ms,
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
