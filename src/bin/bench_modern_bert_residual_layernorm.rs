//! Production-disconnected benchmark for a v4 ModernBERT boundary fusion.
//!
//! The current boundary materializes `updated = residual + branch`, retains it
//! for the following residual connection, and gives a clone to Burn LayerNorm.
//! The candidate emits both `updated` and `normalized` from one exact-shape
//! SourceKernel. It is specialized to the fixed replay's B1/S3/D768 f32 shape.
//! The 50-boundary aggregate is the released encoder's 25 attention-residual
//! to `mlp_norm` boundaries, 24 MLP-residual to next-layer `attn_norm`
//! boundaries, and one final MLP-residual to `final_norm` boundary. Except for
//! that terminal boundary, `updated` remains live for the following residual;
//! retaining both outputs for all 50 is therefore a conservative uniform
//! production contract.
//!
//! This binary and its sibling WGSL file are deliberately not registered in
//! Cargo and are not connected to `kernels.rs` or production routing. After a
//! separate registration decision, the intended invocation is:
//!
//! `cargo run --release --bin bench_modern_bert_residual_layernorm -- 0`

use std::{
    error::Error,
    fmt, io,
    sync::{Arc, Mutex},
    time::Instant,
};

use burn::{
    backend::wgpu::{
        CubeDim, CubeTensor, KernelSource, RuntimeOptions, SourceKernel, SourceTemplate,
        WgpuDevice, WgpuRuntime, graphics::AutoGraphicsApi, init_setup,
    },
    module::EmptyRecord,
    module::{Module, Param},
    nn::{LayerNorm, LayerNormConfig, LayerNormRecord},
    tensor::{DType, Distribution, Shape, Tensor, TensorPrimitive, backend::Backend},
};
use cubecl::{CubeCount, prelude::KernelId, prelude::Runtime, server::KernelArguments};
use irodori_tts_wgpu::WgpuRaw;

type B = WgpuRaw;

const BATCH: usize = 1;
const SEQUENCE: usize = 3;
const WIDTH: usize = 768;
const ROWS: usize = BATCH * SEQUENCE;
const ELEMENTS: usize = ROWS * WIDTH;
const BOUNDARIES: usize = 50;
const EPSILON: f64 = 1.0e-5;
const WORKGROUP_SIZE: u32 = 256;
const REQUIRED_BINDINGS: u32 = 5;
const SHARED_BYTES: usize = WORKGROUP_SIZE as usize * core::mem::size_of::<f32>();
const F32_BYTES: usize = core::mem::size_of::<f32>();
const WARMUP: usize = 10;
const ITERATIONS: usize = 100;
const TRIALS: usize = 5;
const SEED: u64 = 0;
const NORMALIZED_MAX_ABS: f32 = 5.0e-5;

// WgpuRaw does not fuse these backend operations. Burn's var_mean_bias is
// mean_dim followed by sub/square/sum/div, not a one-pass E[x^2]-mean^2.
const CURRENT_BACKEND_OPS_PER_BOUNDARY: usize = 11;
const CANDIDATE_DISPATCHES_PER_BOUNDARY: usize = 1;

#[derive(Debug)]
struct CandidateError {
    message: String,
}

impl CandidateError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for CandidateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for CandidateError {}

#[derive(Debug)]
struct Args {
    adapter_index: usize,
}

enum ParseOutcome {
    Run(Args),
    Help,
}

struct BoundaryFixture {
    residual: Tensor<B, 3>,
    branch: Tensor<B, 3>,
    gamma: Tensor<B, 1>,
    norm: LayerNorm<B>,
}

#[derive(Clone)]
struct BoundaryOutputs {
    updated: Tensor<B, 3>,
    normalized: Tensor<B, 3>,
}

#[derive(Clone, Copy, Debug, Default)]
struct Comparison {
    elements: usize,
    bit_mismatches: usize,
    max_abs: f32,
}

#[derive(Clone, Copy, Debug)]
struct Timing {
    median_us: f64,
    min_us: f64,
    max_us: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Path {
    Current,
    Fused,
}

impl Path {
    const fn label(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Fused => "fused",
        }
    }
}

const PATHS: [Path; 2] = [Path::Current, Path::Fused];

#[derive(Debug)]
struct WgpuErrorMonitor {
    errors: Arc<Mutex<Vec<String>>>,
}

impl WgpuErrorMonitor {
    fn new() -> Self {
        Self {
            errors: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn callback_sink(&self) -> Arc<Mutex<Vec<String>>> {
        Arc::clone(&self.errors)
    }

    fn check(&self, stage: &str) -> Result<(), Box<dyn Error>> {
        let mut errors = self.errors.lock().map_err(|_| {
            io::Error::other(format!(
                "WGPU error monitor lock was poisoned after {stage}"
            ))
        })?;
        if errors.is_empty() {
            return Ok(());
        }
        let count = errors.len();
        let details = errors.drain(..).collect::<Vec<_>>().join("\n---\n");
        Err(io::Error::other(format!(
            "WGPU reported {count} uncaptured error(s) during {stage}:\n{details}"
        ))
        .into())
    }
}

#[derive(Debug)]
struct ResidualLayerNormKernel;

impl KernelSource for ResidualLayerNormKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(include_str!("bench_modern_bert_residual_layernorm.wgsl"))
            .register("width", WIDTH.to_string())
            .register("workgroup_size", WORKGROUP_SIZE.to_string())
            .register("epsilon", format!("{EPSILON:e}"))
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

fn usage() -> &'static str {
    "usage: bench_modern_bert_residual_layernorm <adapter-index>"
}

fn parse_args() -> Result<ParseOutcome, Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let Some(argument) = args.next() else {
        return Err(io::Error::other(format!("missing adapter index; {}", usage())).into());
    };
    if matches!(argument.as_str(), "--help" | "-h") {
        return Ok(ParseOutcome::Help);
    }
    if argument.starts_with('-') {
        return Err(io::Error::other(format!("unknown option {argument:?}; {}", usage())).into());
    }
    let adapter_index = argument.parse::<usize>().map_err(|error| {
        io::Error::other(format!(
            "invalid adapter index {argument:?}: {error}; {}",
            usage()
        ))
    })?;
    if let Some(extra) = args.next() {
        return Err(io::Error::other(format!("unexpected argument {extra:?}; {}", usage())).into());
    }
    Ok(ParseOutcome::Run(Args { adapter_index }))
}

fn initialize_wgpu(adapter_index: usize) -> (WgpuDevice, WgpuErrorMonitor) {
    let device = WgpuDevice::DiscreteGpu(adapter_index);
    let setup = init_setup::<AutoGraphicsApi>(&device, RuntimeOptions::default());
    let monitor = WgpuErrorMonitor::new();
    let callback_errors = monitor.callback_sink();
    setup.device.on_uncaptured_error(Arc::new(move |error| {
        if let Ok(mut errors) = callback_errors.lock() {
            errors.push(error.to_string());
        }
    }));
    let info = setup.adapter.get_info();
    println!(
        "wgpu_adapter: index={adapter_index} name={:?} backend={:?} device_type={:?}",
        info.name, info.backend, info.device_type
    );
    (device, monitor)
}

fn synchronize_and_check_wgpu(
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    stage: &str,
) -> Result<(), Box<dyn Error>> {
    let client = WgpuRuntime::client(device);
    let sync_result = cubecl::future::block_on(client.sync());
    monitor.check(stage)?;
    sync_result.map_err(|error| {
        io::Error::other(format!(
            "CubeCL synchronization failed after {stage}: {error}"
        ))
        .into()
    })
}

fn make_fixture(device: &<B as Backend>::Device) -> BoundaryFixture {
    let residual = Tensor::random(
        [BATCH, SEQUENCE, WIDTH],
        Distribution::Uniform(-0.75, 0.75),
        device,
    );
    let branch = Tensor::random(
        [BATCH, SEQUENCE, WIDTH],
        Distribution::Uniform(-0.5, 0.5),
        device,
    );
    let gamma = Tensor::random([WIDTH], Distribution::Uniform(0.75, 1.25), device);
    let norm = LayerNormConfig::new(WIDTH)
        .with_epsilon(EPSILON)
        .with_bias(false)
        .init(device)
        .load_record(LayerNormRecord {
            gamma: Param::from_tensor(gamma.clone()),
            beta: None,
            epsilon: EmptyRecord::new(),
        });
    debug_assert!(norm.beta.is_none());
    BoundaryFixture {
        residual,
        branch,
        gamma,
        norm,
    }
}

fn make_fixtures(device: &<B as Backend>::Device) -> Vec<BoundaryFixture> {
    (0..BOUNDARIES).map(|_| make_fixture(device)).collect()
}

fn rank3_strides(tensor: &CubeTensor<WgpuRuntime>) -> [usize; 3] {
    let strides = tensor.meta.strides();
    [strides[0], strides[1], strides[2]]
}

fn validate_rank3_input(
    name: &str,
    tensor: &CubeTensor<WgpuRuntime>,
    reference: &CubeTensor<WgpuRuntime>,
) -> Result<(), CandidateError> {
    if tensor.dtype != DType::F32 {
        return Err(CandidateError::new(format!(
            "{name} must be f32, got {:?}",
            tensor.dtype
        )));
    }
    if tensor.device != reference.device {
        return Err(CandidateError::new(format!(
            "{name} is on a different WGPU device"
        )));
    }
    if tensor.meta.num_dims() != 3 {
        return Err(CandidateError::new(format!(
            "{name} must be rank 3 [1,3,768], got rank {}",
            tensor.meta.num_dims()
        )));
    }
    let shape = tensor.meta.shape();
    if [shape[0], shape[1], shape[2]] != [BATCH, SEQUENCE, WIDTH] {
        return Err(CandidateError::new(format!(
            "{name} shape mismatch: expected [1,3,768], got {shape:?}"
        )));
    }
    let expected_strides = [ELEMENTS, WIDTH, 1];
    if !tensor.is_contiguous() || rank3_strides(tensor) != expected_strides {
        return Err(CandidateError::new(format!(
            "{name} layout mismatch: expected contiguous strides {expected_strides:?}, got {:?}",
            rank3_strides(tensor)
        )));
    }
    Ok(())
}

fn validate_contract(
    residual: &CubeTensor<WgpuRuntime>,
    branch: &CubeTensor<WgpuRuntime>,
    gamma: &CubeTensor<WgpuRuntime>,
) -> Result<(usize, u32), CandidateError> {
    validate_rank3_input("residual", residual, residual)?;
    validate_rank3_input("branch", branch, residual)?;

    if gamma.dtype != DType::F32 {
        return Err(CandidateError::new(format!(
            "gamma must be f32, got {:?}",
            gamma.dtype
        )));
    }
    if gamma.device != residual.device {
        return Err(CandidateError::new("gamma is on a different WGPU device"));
    }
    if gamma.meta.num_dims() != 1 || gamma.meta.shape()[0] != WIDTH {
        return Err(CandidateError::new(format!(
            "gamma shape mismatch: expected [768], got {:?}",
            gamma.meta.shape()
        )));
    }
    if !gamma.is_contiguous() || gamma.meta.strides()[0] != 1 {
        return Err(CandidateError::new(format!(
            "gamma layout mismatch: expected contiguous stride [1], got {:?}",
            gamma.meta.strides()
        )));
    }

    let output_bytes = ELEMENTS
        .checked_mul(F32_BYTES)
        .ok_or_else(|| CandidateError::new("output byte count overflows usize"))?;
    let output_bytes_u64 = u64::try_from(output_bytes)
        .map_err(|_| CandidateError::new("output byte count exceeds u64"))?;
    let rows = u32::try_from(ROWS)
        .map_err(|_| CandidateError::new("row count exceeds the dispatch u32 range"))?;
    let properties = residual.client.properties();
    if output_bytes_u64 > properties.memory.max_page_size {
        return Err(CandidateError::new(format!(
            "each output requires {output_bytes} bytes, device max page is {} bytes",
            properties.memory.max_page_size
        )));
    }
    let hardware = &properties.hardware;
    if hardware.max_bindings < REQUIRED_BINDINGS {
        return Err(CandidateError::new(format!(
            "candidate requires {REQUIRED_BINDINGS} storage bindings, device exposes {}",
            hardware.max_bindings
        )));
    }
    if hardware.max_shared_memory_size < SHARED_BYTES {
        return Err(CandidateError::new(format!(
            "candidate requires {SHARED_BYTES} shared bytes, device exposes {}",
            hardware.max_shared_memory_size
        )));
    }
    if hardware.max_units_per_cube < WORKGROUP_SIZE
        || hardware.max_cube_dim.0 < WORKGROUP_SIZE
        || hardware.max_cube_dim.1 < 1
        || hardware.max_cube_dim.2 < 1
    {
        return Err(CandidateError::new(format!(
            "candidate requires workgroup [{WORKGROUP_SIZE},1,1], device limits units={} dim={:?}",
            hardware.max_units_per_cube, hardware.max_cube_dim
        )));
    }
    if hardware.max_cube_count.0 < rows
        || hardware.max_cube_count.1 < 1
        || hardware.max_cube_count.2 < 1
    {
        return Err(CandidateError::new(format!(
            "candidate requires dispatch [{rows},1,1], device limit is {:?}",
            hardware.max_cube_count
        )));
    }
    Ok((output_bytes, rows))
}

fn fused_residual_layer_norm(
    residual: CubeTensor<WgpuRuntime>,
    branch: CubeTensor<WgpuRuntime>,
    gamma: CubeTensor<WgpuRuntime>,
) -> Result<(CubeTensor<WgpuRuntime>, CubeTensor<WgpuRuntime>), CandidateError> {
    let (output_bytes, rows) = validate_contract(&residual, &branch, &gamma)?;
    let client = residual.client.clone();
    let device = residual.device.clone();
    let updated = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from([BATCH, SEQUENCE, WIDTH]),
        client.empty(output_bytes),
        DType::F32,
    );
    let normalized = CubeTensor::new_contiguous(
        client.clone(),
        device,
        Shape::from([BATCH, SEQUENCE, WIDTH]),
        client.empty(output_bytes),
        DType::F32,
    );
    let task: Box<dyn cubecl::CubeTask<burn::backend::wgpu::AutoCompiler>> = Box::new(
        SourceKernel::new(ResidualLayerNormKernel, CubeDim::new_1d(WORKGROUP_SIZE)),
    );
    let bindings = KernelArguments::new()
        .with_buffer(residual.handle.binding())
        .with_buffer(branch.handle.binding())
        .with_buffer(gamma.handle.binding())
        .with_buffer(updated.handle.clone().binding())
        .with_buffer(normalized.handle.clone().binding());
    client.launch(task, CubeCount::new_1d(rows), bindings);
    Ok((updated, normalized))
}

fn current_boundary(fixture: &BoundaryFixture) -> BoundaryOutputs {
    // This is the exact ownership pattern in ModernBertEncoderLayer: retain the
    // updated residual and pass its clone into bias-free LayerNorm.
    let updated = fixture.residual.clone() + fixture.branch.clone();
    let normalized = fixture.norm.forward(updated.clone());
    BoundaryOutputs {
        updated,
        normalized,
    }
}

fn fused_boundary(fixture: &BoundaryFixture) -> Result<BoundaryOutputs, Box<dyn Error>> {
    let (updated, normalized) = fused_residual_layer_norm(
        fixture.residual.clone().into_primitive().tensor(),
        fixture.branch.clone().into_primitive().tensor(),
        fixture.gamma.clone().into_primitive().tensor(),
    )?;
    Ok(BoundaryOutputs {
        updated: Tensor::from_primitive(TensorPrimitive::Float(updated)),
        normalized: Tensor::from_primitive(TensorPrimitive::Float(normalized)),
    })
}

fn compare_tensor(
    name: &str,
    expected: Tensor<B, 3>,
    actual: Tensor<B, 3>,
) -> Result<Comparison, Box<dyn Error>> {
    if expected.dims() != actual.dims() {
        return Err(io::Error::other(format!(
            "{name} shape mismatch: current={:?}, fused={:?}",
            expected.dims(),
            actual.dims()
        ))
        .into());
    }
    let expected = expected.into_data().to_vec::<f32>()?;
    let actual = actual.into_data().to_vec::<f32>()?;
    if expected.len() != actual.len() {
        return Err(io::Error::other(format!(
            "{name} length mismatch: current={} fused={}",
            expected.len(),
            actual.len()
        ))
        .into());
    }

    let mut comparison = Comparison {
        elements: expected.len(),
        ..Comparison::default()
    };
    for (index, (&expected, &actual)) in expected.iter().zip(&actual).enumerate() {
        if !expected.is_finite() || !actual.is_finite() {
            return Err(io::Error::other(format!(
                "{name} contains non-finite output at {index}: current={expected:?} fused={actual:?}"
            ))
            .into());
        }
        comparison.bit_mismatches += usize::from(expected.to_bits() != actual.to_bits());
        comparison.max_abs = comparison.max_abs.max((expected - actual).abs());
    }
    if !comparison.max_abs.is_finite() {
        return Err(io::Error::other(format!("{name} max_abs is non-finite")).into());
    }
    Ok(comparison)
}

fn compare_outputs(
    label: &str,
    current: BoundaryOutputs,
    fused: BoundaryOutputs,
) -> Result<(Comparison, Comparison), Box<dyn Error>> {
    let updated = compare_tensor(
        &format!("{label} updated residual"),
        current.updated,
        fused.updated,
    )?;
    let normalized = compare_tensor(
        &format!("{label} normalized"),
        current.normalized,
        fused.normalized,
    )?;
    if updated.bit_mismatches != 0 || updated.max_abs != 0.0 {
        return Err(io::Error::other(format!(
            "{label} updated residual must be bit-identical: mismatches={} max_abs={:.9e}",
            updated.bit_mismatches, updated.max_abs
        ))
        .into());
    }
    if normalized.max_abs > NORMALIZED_MAX_ABS {
        return Err(io::Error::other(format!(
            "{label} normalized max_abs={:.9e} exceeds {:.9e}",
            normalized.max_abs, NORMALIZED_MAX_ABS
        ))
        .into());
    }
    Ok((updated, normalized))
}

fn collect_outputs(
    fixtures: &[BoundaryFixture],
    path: Path,
) -> Result<BoundaryOutputs, Box<dyn Error>> {
    if fixtures.is_empty() {
        return Err(io::Error::other("cannot collect an empty fixture set").into());
    }
    let outputs = fixtures
        .iter()
        .map(|fixture| match path {
            Path::Current => Ok(current_boundary(fixture)),
            Path::Fused => fused_boundary(fixture),
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let updated = Tensor::cat(
        outputs
            .iter()
            .map(|output| output.updated.clone())
            .collect(),
        0,
    );
    let normalized = Tensor::cat(
        outputs
            .into_iter()
            .map(|output| output.normalized)
            .collect(),
        0,
    );
    Ok(BoundaryOutputs {
        updated,
        normalized,
    })
}

fn validate_correctness(fixtures: &[BoundaryFixture]) -> Result<(), Box<dyn Error>> {
    let first = fixtures
        .first()
        .ok_or_else(|| io::Error::other("correctness requires at least one fixture"))?;
    let (single_updated, single_normalized) =
        compare_outputs("single", current_boundary(first), fused_boundary(first)?)?;
    println!(
        "correctness single: updated elements={} bit_mismatch={} max_abs={:.9e} finite=true; normalized elements={} bit_mismatch={} max_abs={:.9e} finite=true",
        single_updated.elements,
        single_updated.bit_mismatches,
        single_updated.max_abs,
        single_normalized.elements,
        single_normalized.bit_mismatches,
        single_normalized.max_abs,
    );

    let (aggregate_updated, aggregate_normalized) = compare_outputs(
        "aggregate-50",
        collect_outputs(fixtures, Path::Current)?,
        collect_outputs(fixtures, Path::Fused)?,
    )?;
    println!(
        "correctness aggregate-50: updated elements={} bit_mismatch={} max_abs={:.9e} finite=true; normalized elements={} bit_mismatch={} max_abs={:.9e} finite=true",
        aggregate_updated.elements,
        aggregate_updated.bit_mismatches,
        aggregate_updated.max_abs,
        aggregate_normalized.elements,
        aggregate_normalized.bit_mismatches,
        aggregate_normalized.max_abs,
    );
    Ok(())
}

fn run_path(fixtures: &[BoundaryFixture], path: Path) -> Result<BoundaryOutputs, Box<dyn Error>> {
    let mut last = None;
    for fixture in fixtures {
        last = Some(match path {
            Path::Current => current_boundary(fixture),
            Path::Fused => fused_boundary(fixture)?,
        });
    }
    last.ok_or_else(|| io::Error::other("timed path received no fixtures").into())
}

fn synchronize_output(outputs: BoundaryOutputs) -> Result<(), Box<dyn Error>> {
    if outputs.updated.dims() != [BATCH, SEQUENCE, WIDTH]
        || outputs.normalized.dims() != [BATCH, SEQUENCE, WIDTH]
    {
        return Err(io::Error::other(format!(
            "timed output shape mismatch: updated={:?} normalized={:?}",
            outputs.updated.dims(),
            outputs.normalized.dims()
        ))
        .into());
    }
    let values = outputs
        .normalized
        .slice([0..1, 0..1, 0..1])
        .into_data()
        .to_vec::<f32>()?;
    let value = values
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("timed synchronization returned no value"))?;
    if !value.is_finite() {
        return Err(io::Error::other(format!(
            "timed synchronization returned non-finite value {value:?}"
        ))
        .into());
    }
    Ok(())
}

fn warm_up(fixtures: &[BoundaryFixture], path: Path) -> Result<(), Box<dyn Error>> {
    let output = (0..WARMUP)
        .map(|_| run_path(fixtures, path))
        .last()
        .ok_or_else(|| io::Error::other("WARMUP must be non-zero"))??;
    synchronize_output(output)
}

fn measure(fixtures: &[BoundaryFixture], path: Path) -> Result<f64, Box<dyn Error>> {
    let started = Instant::now();
    let output = (0..ITERATIONS)
        .map(|_| run_path(fixtures, path))
        .last()
        .ok_or_else(|| io::Error::other("ITERATIONS must be non-zero"))??;
    synchronize_output(output)?;
    Ok(started.elapsed().as_secs_f64() * 1_000_000.0 / ITERATIONS as f64)
}

fn summarize_samples(samples: &[f64]) -> Result<Timing, Box<dyn Error>> {
    if samples.is_empty() {
        return Err(io::Error::other("cannot summarize an empty timing sample set").into());
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(io::Error::other("timing samples contain a non-finite value").into());
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    Ok(Timing {
        median_us: sorted[sorted.len() / 2],
        min_us: sorted[0],
        max_us: sorted[sorted.len() - 1],
    })
}

fn benchmark_workload(
    label: &str,
    fixtures: &[BoundaryFixture],
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
) -> Result<(), Box<dyn Error>> {
    for path in PATHS {
        warm_up(fixtures, path)?;
        synchronize_and_check_wgpu(device, monitor, &format!("{label} {} warmup", path.label()))?;
    }

    let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::with_capacity(TRIALS));
    for trial in 0..TRIALS {
        for offset in 0..PATHS.len() {
            let path_index = (trial + offset) % PATHS.len();
            let path = PATHS[path_index];
            let sample = measure(fixtures, path)?;
            synchronize_and_check_wgpu(
                device,
                monitor,
                &format!("{label} {} trial {}", path.label(), trial + 1),
            )?;
            samples[path_index].push(sample);
        }
    }
    let current = summarize_samples(&samples[0])?;
    let fused = summarize_samples(&samples[1])?;
    let boundaries = fixtures.len() as f64;
    println!("timing {label}: boundaries={}", fixtures.len());
    println!(
        "  current: median={:.3} us/op range=[{:.3},{:.3}] per_boundary={:.3} us",
        current.median_us,
        current.min_us,
        current.max_us,
        current.median_us / boundaries,
    );
    println!(
        "  fused  : median={:.3} us/op range=[{:.3},{:.3}] per_boundary={:.3} us speedup={:.3}x saving={:.3} us/op",
        fused.median_us,
        fused.min_us,
        fused.max_us,
        fused.median_us / boundaries,
        current.median_us / fused.median_us,
        current.median_us - fused.median_us,
    );
    Ok(())
}

fn current_traffic_per_boundary() -> Result<usize, Box<dyn Error>> {
    let activation = ELEMENTS
        .checked_mul(F32_BYTES)
        .ok_or_else(|| io::Error::other("activation byte count overflows usize"))?;
    let row_stats = ROWS
        .checked_mul(F32_BYTES)
        .ok_or_else(|| io::Error::other("row-stat byte count overflows usize"))?;
    let gamma = WIDTH
        .checked_mul(F32_BYTES)
        .ok_or_else(|| io::Error::other("gamma byte count overflows usize"))?;

    // Physical-tensor traffic lower bound for the exact Burn operation graph:
    // residual add; mean; variance sub/square/sum/div; normalized
    // sub/eps-add/sqrt/div/gamma-mul. Broadcast rereads and reduction scratch
    // are deliberately not inflated.
    [
        3 * activation,
        activation + row_stats,
        2 * activation + row_stats,
        2 * activation,
        activation + row_stats,
        2 * row_stats,
        2 * activation + row_stats,
        2 * row_stats,
        2 * row_stats,
        2 * activation + row_stats,
        2 * activation + gamma,
    ]
    .into_iter()
    .try_fold(0_usize, |total, bytes| {
        total.checked_add(bytes).ok_or_else(|| {
            Box::<dyn Error>::from(io::Error::other("current traffic overflows usize"))
        })
    })
}

fn candidate_traffic_per_boundary() -> Result<usize, Box<dyn Error>> {
    let activation = ELEMENTS
        .checked_mul(F32_BYTES)
        .ok_or_else(|| io::Error::other("activation byte count overflows usize"))?;
    let gamma = WIDTH
        .checked_mul(F32_BYTES)
        .ok_or_else(|| io::Error::other("gamma byte count overflows usize"))?;
    activation
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(gamma))
        .ok_or_else(|| io::Error::other("candidate traffic overflows usize").into())
}

fn print_static_accounting() -> Result<(), Box<dyn Error>> {
    let current_bytes = current_traffic_per_boundary()?;
    let candidate_bytes = candidate_traffic_per_boundary()?;
    let saved_bytes = current_bytes
        .checked_sub(candidate_bytes)
        .ok_or_else(|| io::Error::other("candidate traffic exceeds current traffic"))?;
    let aggregate_current = current_bytes
        .checked_mul(BOUNDARIES)
        .ok_or_else(|| io::Error::other("aggregate current traffic overflows usize"))?;
    let aggregate_candidate = candidate_bytes
        .checked_mul(BOUNDARIES)
        .ok_or_else(|| io::Error::other("aggregate candidate traffic overflows usize"))?;
    let aggregate_saved = saved_bytes
        .checked_mul(BOUNDARIES)
        .ok_or_else(|| io::Error::other("aggregate saved traffic overflows usize"))?;
    println!(
        "semantics: updated=residual+branch; mean=sum(x)/768; biased_var=sum((x-mean)^2)/768; eps=1e-5; gamma=true; beta=false; outputs=updated+normalized"
    );
    println!(
        "dispatch accounting: current_dispatch_floor={CURRENT_BACKEND_OPS_PER_BOUNDARY}/boundary {}/50; fused_sourcekernel={CANDIDATE_DISPATCHES_PER_BOUNDARY}/boundary {}/50; saved_at_least={}/50",
        CURRENT_BACKEND_OPS_PER_BOUNDARY * BOUNDARIES,
        CANDIDATE_DISPATCHES_PER_BOUNDARY * BOUNDARIES,
        (CURRENT_BACKEND_OPS_PER_BOUNDARY - CANDIDATE_DISPATCHES_PER_BOUNDARY) * BOUNDARIES,
    );
    println!(
        "traffic lower-bound: current={current_bytes} B/boundary ({:.6} MiB/50); fused={candidate_bytes} B/boundary ({:.6} MiB/50); saved={saved_bytes} B/boundary ({:.6} MiB/50)",
        aggregate_current as f64 / (1024.0 * 1024.0),
        aggregate_candidate as f64 / (1024.0 * 1024.0),
        aggregate_saved as f64 / (1024.0 * 1024.0),
    );
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = match parse_args()? {
        ParseOutcome::Run(args) => args,
        ParseOutcome::Help => {
            println!("{}", usage());
            return Ok(());
        }
    };
    let (device, monitor) = initialize_wgpu(args.adapter_index);
    B::seed(&device, SEED);
    println!(
        "isolated ModernBERT residual+LayerNorm benchmark: B={BATCH} S={SEQUENCE} D={WIDTH} boundaries={BOUNDARIES} warmup={WARMUP} iterations={ITERATIONS} x {TRIALS} trials seed={SEED}"
    );
    print_static_accounting()?;

    let fixtures = make_fixtures(&device);
    validate_correctness(&fixtures)?;
    synchronize_and_check_wgpu(&device, &monitor, "correctness")?;

    benchmark_workload("single-boundary", &fixtures[..1], &device, &monitor)?;
    benchmark_workload("aggregate-50", &fixtures, &device, &monitor)?;

    synchronize_and_check_wgpu(&device, &monitor, "benchmark completion")?;
    println!("wgpu_uncaptured_errors=0");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_shape_accounting_is_exact() {
        assert_eq!(ROWS, 3);
        assert_eq!(ELEMENTS, 2_304);
        assert_eq!(SHARED_BYTES, 1_024);
        assert_eq!(current_traffic_per_boundary().unwrap(), 141_444);
        assert_eq!(candidate_traffic_per_boundary().unwrap(), 39_936);
        assert_eq!(CURRENT_BACKEND_OPS_PER_BOUNDARY * BOUNDARIES, 550);
        assert_eq!(CANDIDATE_DISPATCHES_PER_BOUNDARY * BOUNDARIES, 50);
    }

    #[test]
    fn sourcekernel_storage_access_is_uniform() {
        let shader = include_str!("bench_modern_bert_residual_layernorm.wgsl");
        assert_eq!(shader.matches("var<storage, read_write>").count(), 5);
        assert!(!shader.contains("var<storage, read>"));
    }
}
