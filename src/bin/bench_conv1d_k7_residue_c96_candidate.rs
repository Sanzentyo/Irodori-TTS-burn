//! Isolated production-weight screen for the C96/L96000 k=7 residue candidate.
//!
//! This binary is intentionally absent from `Cargo.toml`.  It compares the
//! exact current production T256+Snake vec4 launcher with one candidate that
//! reuses the accepted residue pack/core WGSL for decoder block-3 d3 and d9.
//! Promotion still requires a separate production route change and full codec
//! waveform/hash validation.

#[path = "../kernels/conv1d_k7_residue_c96_candidate.rs"]
mod conv1d_k7_residue_c96_candidate;

use std::{
    fs::File,
    hint::black_box,
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use burn::{
    backend::wgpu::{
        MemoryConfiguration, RuntimeOptions, WgpuDevice, WgpuRuntime, graphics::AutoGraphicsApi,
        init_setup,
    },
    tensor::{DType, Distribution, Tensor, TensorPrimitive, backend::Backend},
};
use clap::Parser;
use conv1d_k7_residue_c96_candidate::{
    ResidueCandidateDescriptor, ResidueCandidateDilation, ResidueCandidateGeometry,
    ResidueCandidateInputs, ResidueCandidateTraffic, residue_candidate_contract_is_compatible,
    try_conv1d_k7_residue_c96_candidate,
};
use cubecl::prelude::Runtime;
use irodori_tts_wgpu::{
    WgpuRaw,
    kernels::{
        conv1d_k7_t256_snake_epilogue::{Conv1dK7T256Tile, LaunchGeometry},
        conv1d_k7_t256_snake_vec4_store::{
            conv1d_k7_t256_snake_vec4_store_contract_is_compatible, production_tile_for_shape,
            try_conv1d_k7_same_t256_snake_vec4_store_wgsl,
        },
        conv1d_k7_tiled::Conv1dK7Dilation,
    },
    weights::TensorStore,
};
use sha2::{Digest, Sha256};

type B = WgpuRaw;

const CHANNELS: usize = 96;
const LENGTH: usize = 96_000;
const KERNEL_SIZE: usize = 7;
const ELEMENTS: usize = CHANNELS * LENGTH;
const F32_BYTES: usize = size_of::<f32>();
const WARMUP: usize = 10;
const ITERATIONS: usize = 100;
const TRIALS: usize = 5;
const VARIANT_COUNT: usize = 2;
const PINNED_CODEC_SHA256: &str =
    "4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1";
// Filled only from `bench_python_k7_snake_c96_control.py` after its pinned
// production-weight GPU run.  `None` keeps this acceptance binary fail-closed;
// the historical conv+bias-only sum is not a same-work substitute.
const PYTORCH_SAME_WORK_SUITE_GLOBAL_MIN_US: Option<f64> = None;

#[derive(Debug, Parser)]
#[command(about = "C96/L96000 d3+d9 residue-class production-weight screen")]
struct Args {
    #[arg(long, default_value = "target/v4_dacvae_weights.safetensors")]
    codec_weights: PathBuf,
    #[arg(long, default_value = PINNED_CODEC_SHA256)]
    codec_weights_sha256: String,
    #[arg(long, default_value_t = 0)]
    adapter_index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ConvCase {
    label: &'static str,
    production_dilation: Conv1dK7Dilation,
    candidate_dilation: ResidueCandidateDilation,
    production_tile: Conv1dK7T256Tile,
    weight_key: &'static str,
    bias_key: &'static str,
    alpha_key: &'static str,
    seed: u64,
    pytorch_conv_only_diagnostic_us: f64,
    pytorch_same_work_global_min_us: Option<f64>,
}

const CASES: [ConvCase; 2] = [
    ConvCase {
        label: "decoder_block3_res1_c96_l96000_d3",
        production_dilation: Conv1dK7Dilation::Three,
        candidate_dilation: ResidueCandidateDilation::Three,
        production_tile: Conv1dK7T256Tile::Cin16,
        weight_key: "decoder.model.4.block.5.block.1.weight",
        bias_key: "decoder.model.4.block.5.block.1.bias",
        alpha_key: "decoder.model.4.block.5.block.2.alpha",
        seed: 0x0960_0003,
        pytorch_conv_only_diagnostic_us: 1_716.429,
        pytorch_same_work_global_min_us: None,
    },
    ConvCase {
        label: "decoder_block3_res2_c96_l96000_d9",
        production_dilation: Conv1dK7Dilation::Nine,
        candidate_dilation: ResidueCandidateDilation::Nine,
        production_tile: Conv1dK7T256Tile::Cin8,
        weight_key: "decoder.model.4.block.8.block.1.weight",
        bias_key: "decoder.model.4.block.8.block.1.bias",
        alpha_key: "decoder.model.4.block.8.block.2.alpha",
        seed: 0x0960_0009,
        pytorch_conv_only_diagnostic_us: 1_717.125,
        pytorch_same_work_global_min_us: None,
    },
];

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Variant {
    CurrentProduction,
    ResidueC96Candidate,
}

impl Variant {
    const ALL: [Self; VARIANT_COUNT] = [Self::CurrentProduction, Self::ResidueC96Candidate];

    const fn label(self) -> &'static str {
        match self {
            Self::CurrentProduction => "current_t256_snake_vec4",
            Self::ResidueC96Candidate => "candidate_residue_pack_d1_snake",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::CurrentProduction => 0,
            Self::ResidueC96Candidate => 1,
        }
    }
}

struct CaseFixture {
    case: ConvCase,
    input: Tensor<B, 3>,
    weight: Tensor<B, 3>,
    bias: Tensor<B, 1>,
    alpha: Tensor<B, 3>,
}

#[derive(Clone, Copy, Debug)]
struct Comparison {
    elements: usize,
    bit_mismatches: usize,
    max_abs: f32,
    mean_abs: f64,
    finite: bool,
}

#[derive(Clone, Copy, Debug)]
struct Timing {
    median_us: f64,
    min_us: f64,
    max_us: f64,
}

#[derive(Clone, Copy, Debug)]
struct CaseResult {
    case: ConvCase,
    timings: [Timing; VARIANT_COUNT],
    comparison: Comparison,
    accepted: bool,
}

#[derive(Clone, Copy, Debug)]
struct CurrentTraffic {
    input_read_bytes: usize,
    weight_read_bytes: usize,
    bias_read_bytes: usize,
    alpha_read_bytes: usize,
    output_write_bytes: usize,
    total_bytes: usize,
}

#[derive(Clone, Default)]
struct WgpuErrorMonitor {
    errors: Arc<Mutex<Vec<String>>>,
}

impl WgpuErrorMonitor {
    fn callback_sink(&self) -> Arc<Mutex<Vec<String>>> {
        Arc::clone(&self.errors)
    }

    fn check(&self, stage: &str) -> Result<()> {
        let mut errors = self
            .errors
            .lock()
            .map_err(|_| anyhow::anyhow!("WGPU error monitor lock poisoned after {stage}"))?;
        ensure!(errors.is_empty(), "WGPU errors after {stage}: {errors:?}");
        errors.clear();
        Ok(())
    }
}

fn initialize_wgpu(adapter_index: usize) -> (WgpuDevice, WgpuErrorMonitor) {
    let device = WgpuDevice::DiscreteGpu(adapter_index);
    let setup = init_setup::<AutoGraphicsApi>(
        &device,
        RuntimeOptions {
            tasks_max: 32,
            memory_config: MemoryConfiguration::SubSlices,
        },
    );
    let monitor = WgpuErrorMonitor::default();
    let callback_errors = monitor.callback_sink();
    setup.device.on_uncaptured_error(Arc::new(move |error| {
        if let Ok(mut errors) = callback_errors.lock() {
            errors.push(error.to_string());
        }
    }));
    let info = setup.adapter.get_info();
    println!(
        "wgpu_adapter: index={adapter_index} name={:?} backend={:?} device_type={:?} tasks_max=32 memory_config=sub-slices",
        info.name, info.backend, info.device_type,
    );
    (device, monitor)
}

fn synchronize_and_check_wgpu(
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    stage: &str,
) -> Result<()> {
    let sync_result = cubecl::future::block_on(WgpuRuntime::client(device).sync());
    monitor.check(stage)?;
    sync_result.with_context(|| format!("CubeCL synchronization failed after {stage}"))
}

fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    ensure!(
        expected.len() == 64 && expected.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "codec SHA-256 must contain exactly 64 hex digits"
    );
    let file = File::open(path)
        .with_context(|| format!("failed to open codec weights {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .with_context(|| format!("failed to hash codec weights {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let actual = format!("{:x}", hasher.finalize());
    ensure!(
        actual == expected.to_ascii_lowercase(),
        "codec SHA-256 mismatch: got {actual}, expected {expected}"
    );
    println!("codec_weights_sha256={actual} path={}", path.display());
    Ok(())
}

fn ensure_exact_cube_layout<const D: usize>(
    label: &str,
    tensor: &burn::backend::wgpu::CubeTensor<WgpuRuntime>,
    expected_shape: [usize; D],
) -> Result<()> {
    ensure!(
        tensor.meta.num_dims() == D
            && tensor.meta.shape().dims::<D>() == expected_shape
            && tensor.dtype == DType::F32
            && tensor.is_contiguous(),
        "{label}: expected shape={expected_shape:?} dtype=F32 contiguous=true; got rank={} shape={:?} strides={:?} dtype={:?} contiguous={}",
        tensor.meta.num_dims(),
        tensor.meta.shape(),
        tensor.meta.strides(),
        tensor.dtype,
        tensor.is_contiguous(),
    );
    Ok(())
}

fn make_fixture(store: &TensorStore, device: &WgpuDevice, case: ConvCase) -> Result<CaseFixture> {
    let weight: Tensor<B, 3> = store
        .tensor(case.weight_key, device)
        .with_context(|| format!("load {}", case.weight_key))?;
    let bias: Tensor<B, 1> = store
        .tensor(case.bias_key, device)
        .with_context(|| format!("load {}", case.bias_key))?;
    let alpha: Tensor<B, 3> = store
        .tensor(case.alpha_key, device)
        .with_context(|| format!("load {}", case.alpha_key))?;
    ensure!(
        weight.dims() == [CHANNELS, CHANNELS, KERNEL_SIZE],
        "{} shape mismatch: {:?}",
        case.weight_key,
        weight.dims(),
    );
    ensure!(
        bias.dims() == [CHANNELS],
        "{} shape mismatch: {:?}",
        case.bias_key,
        bias.dims(),
    );
    ensure!(
        alpha.dims() == [1, CHANNELS, 1],
        "{} shape mismatch: {:?}",
        case.alpha_key,
        alpha.dims(),
    );
    B::seed(device, case.seed);
    let input = Tensor::<B, 3>::random(
        [1, CHANNELS, LENGTH],
        Distribution::Uniform(-0.5, 0.5),
        device,
    );
    Ok(CaseFixture {
        case,
        input,
        weight,
        bias,
        alpha,
    })
}

fn candidate_descriptor(case: ConvCase) -> ResidueCandidateDescriptor {
    ResidueCandidateDescriptor::c96_l96000(case.candidate_dilation)
}

fn candidate_inputs(fixture: &CaseFixture) -> ResidueCandidateInputs {
    ResidueCandidateInputs::new(
        fixture.input.clone().into_primitive().tensor(),
        fixture.weight.clone().into_primitive().tensor(),
        fixture.bias.clone().into_primitive().tensor(),
        fixture.alpha.clone().into_primitive().tensor(),
    )
}

fn validate_fixture_contract(fixture: &CaseFixture) -> Result<()> {
    let input = fixture.input.clone().into_primitive().tensor();
    let weight = fixture.weight.clone().into_primitive().tensor();
    let bias = fixture.bias.clone().into_primitive().tensor();
    let alpha = fixture.alpha.clone().into_primitive().tensor();
    ensure_exact_cube_layout("input", &input, [1, CHANNELS, LENGTH])?;
    ensure_exact_cube_layout(
        fixture.case.weight_key,
        &weight,
        [CHANNELS, CHANNELS, KERNEL_SIZE],
    )?;
    ensure_exact_cube_layout(fixture.case.bias_key, &bias, [CHANNELS])?;
    ensure_exact_cube_layout(fixture.case.alpha_key, &alpha, [1, CHANNELS, 1])?;
    ensure!(
        production_tile_for_shape(CHANNELS, LENGTH, fixture.case.production_dilation,)
            == Some(fixture.case.production_tile),
        "{} is not the direct current production vec4 route",
        fixture.case.label,
    );
    ensure!(
        conv1d_k7_t256_snake_vec4_store_contract_is_compatible(
            &input,
            &weight,
            &bias,
            &alpha,
            fixture.case.production_dilation,
            fixture.case.production_tile,
        ),
        "{} current production physical contract failed",
        fixture.case.label,
    );
    let candidate = candidate_inputs(fixture);
    let descriptor = candidate_descriptor(fixture.case);
    let shape = descriptor.shape();
    ensure!(
        shape.channels() == CHANNELS
            && shape.length() == LENGTH
            && descriptor.dilation() == fixture.case.candidate_dilation,
        "{} candidate descriptor drifted from C96/L96000/d{}",
        fixture.case.label,
        fixture.case.candidate_dilation.value(),
    );
    let final_packed_index = descriptor
        .packed_index(CHANNELS - 1, LENGTH - 1)
        .context("final candidate packed index is not representable")?;
    ensure!(
        final_packed_index < shape.elements(),
        "{} final packed index {} exceeds {} elements",
        fixture.case.label,
        final_packed_index,
        shape.elements(),
    );
    ensure!(
        residue_candidate_contract_is_compatible(&candidate, descriptor),
        "{} residue candidate physical contract failed",
        fixture.case.label,
    );
    println!(
        "fixture case={} seed=0x{:08x} input_distribution=uniform[-0.5,0.5) input_shape=[1,{CHANNELS},{LENGTH}] weight_key={} bias_key={} alpha_key={} checkpoint_weights=true dtype=f32 contiguous=true current_contract=true candidate_contract=true",
        fixture.case.label,
        fixture.case.seed,
        fixture.case.weight_key,
        fixture.case.bias_key,
        fixture.case.alpha_key,
    );
    Ok(())
}

fn current_forward(fixture: &CaseFixture) -> Result<Tensor<B, 3>> {
    let output = try_conv1d_k7_same_t256_snake_vec4_store_wgsl(
        fixture.input.clone().into_primitive().tensor(),
        fixture.weight.clone().into_primitive().tensor(),
        fixture.bias.clone().into_primitive().tensor(),
        fixture.alpha.clone().into_primitive().tensor(),
        fixture.case.production_dilation,
        fixture.case.production_tile,
    )
    .with_context(|| format!("{} current production launch rejected", fixture.case.label))?;
    Ok(Tensor::from_primitive(TensorPrimitive::Float(output)))
}

fn candidate_forward(fixture: &CaseFixture) -> Result<Tensor<B, 3>> {
    let output = try_conv1d_k7_residue_c96_candidate(
        candidate_inputs(fixture),
        candidate_descriptor(fixture.case),
    )
    .with_context(|| format!("{} residue candidate launch rejected", fixture.case.label))?;
    Ok(Tensor::from_primitive(TensorPrimitive::Float(output)))
}

fn variant_forward(variant: Variant, fixture: &CaseFixture) -> Result<Tensor<B, 3>> {
    match variant {
        Variant::CurrentProduction => current_forward(fixture),
        Variant::ResidueC96Candidate => candidate_forward(fixture),
    }
}

fn output_to_host(output: Tensor<B, 3>, case: ConvCase) -> Result<Vec<f32>> {
    ensure!(
        output.dims() == [1, CHANNELS, LENGTH],
        "{} output shape mismatch: {:?}",
        case.label,
        output.dims(),
    );
    let values = output
        .into_data()
        .to_vec::<f32>()
        .with_context(|| format!("{} full output readback", case.label))?;
    ensure!(
        values.len() == ELEMENTS,
        "{} output element mismatch: got {}, expected {ELEMENTS}",
        case.label,
        values.len(),
    );
    Ok(values)
}

fn compare_values(reference: &[f32], candidate: &[f32]) -> Result<Comparison> {
    ensure!(
        reference.len() == candidate.len(),
        "comparison length mismatch: current={} candidate={}",
        reference.len(),
        candidate.len(),
    );
    let mut bit_mismatches = 0_usize;
    let mut max_abs = 0.0_f32;
    let mut absolute_sum = 0.0_f64;
    let mut finite = true;
    for (&expected, &actual) in reference.iter().zip(candidate) {
        finite &= expected.is_finite() && actual.is_finite();
        bit_mismatches += usize::from(expected.to_bits() != actual.to_bits());
        let difference = (expected - actual).abs();
        max_abs = max_abs.max(difference);
        absolute_sum += f64::from(difference);
    }
    Ok(Comparison {
        elements: reference.len(),
        bit_mismatches,
        max_abs,
        mean_abs: absolute_sum / reference.len() as f64,
        finite,
    })
}

fn sha256_f32_le(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn verify_full_output(
    fixture: &CaseFixture,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
) -> Result<Comparison> {
    let current = output_to_host(current_forward(fixture)?, fixture.case)?;
    synchronize_and_check_wgpu(
        device,
        monitor,
        &format!("{} current full readback", fixture.case.label),
    )?;
    let candidate = output_to_host(candidate_forward(fixture)?, fixture.case)?;
    synchronize_and_check_wgpu(
        device,
        monitor,
        &format!("{} candidate full readback", fixture.case.label),
    )?;
    let comparison = compare_values(&current, &candidate)?;
    let current_sha256 = sha256_f32_le(&current);
    let candidate_sha256 = sha256_f32_le(&candidate);
    println!(
        "correctness case={} elements={} finite={} bit_mismatch={} max_abs={:.9e} mean_abs={:.9e} hash_encoding=f32_ieee754_le current_sha256={} candidate_sha256={} hash_equal={} expected_gate=finite_bit0_hash_equal_maxabs0_meanabs0",
        fixture.case.label,
        comparison.elements,
        comparison.finite,
        comparison.bit_mismatches,
        comparison.max_abs,
        comparison.mean_abs,
        current_sha256,
        candidate_sha256,
        current_sha256 == candidate_sha256,
    );
    ensure!(
        comparison.finite
            && comparison.bit_mismatches == 0
            && comparison.max_abs == 0.0
            && comparison.mean_abs == 0.0
            && current_sha256 == candidate_sha256,
        "{} failed the full 9,216,000-element bit-exact gate",
        fixture.case.label,
    );
    Ok(comparison)
}

fn warm_fixture(
    fixture: &CaseFixture,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
) -> Result<()> {
    let mut retained = None;
    for round in 0..WARMUP {
        for offset in 0..VARIANT_COUNT {
            let variant = Variant::ALL[(round + offset) % VARIANT_COUNT];
            retained = Some(variant_forward(variant, fixture)?);
        }
    }
    black_box(&retained);
    synchronize_and_check_wgpu(
        device,
        monitor,
        &format!("{} rotating warmup", fixture.case.label),
    )?;
    drop(retained);
    Ok(())
}

fn measure_variant(
    variant: Variant,
    fixture: &CaseFixture,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    trial: usize,
) -> Result<f64> {
    synchronize_and_check_wgpu(
        device,
        monitor,
        &format!(
            "{} {} trial {trial} pre-sync",
            fixture.case.label,
            variant.label()
        ),
    )?;
    let started = Instant::now();
    let mut retained = None;
    for _ in 0..ITERATIONS {
        retained = Some(variant_forward(variant, fixture)?);
    }
    black_box(&retained);
    synchronize_and_check_wgpu(
        device,
        monitor,
        &format!(
            "{} {} trial {trial} post-sync",
            fixture.case.label,
            variant.label()
        ),
    )?;
    let elapsed_us = started.elapsed().as_secs_f64() * 1_000_000.0 / ITERATIONS as f64;
    drop(retained);
    Ok(elapsed_us)
}

fn summarize(samples: &[f64]) -> Timing {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    Timing {
        median_us: sorted[sorted.len() / 2],
        min_us: sorted[0],
        max_us: sorted[sorted.len() - 1],
    }
}

fn current_geometry(case: ConvCase) -> LaunchGeometry {
    LaunchGeometry::new(
        CHANNELS,
        LENGTH,
        case.production_dilation,
        case.production_tile,
    )
    .expect("released current production geometry must be valid")
}

fn current_traffic(case: ConvCase) -> CurrentTraffic {
    let geometry = current_geometry(case);
    let workgroups = geometry.workgroups().expect("workgroups fit usize");
    let padding = 3 * case.production_dilation.value();
    let mut valid_input_times = 0_usize;
    for tile in 0..geometry.time_tiles as usize {
        let time_base = tile * 256;
        let source_start = time_base.saturating_sub(padding);
        let source_end = (time_base + geometry.input_span)
            .saturating_sub(padding)
            .min(LENGTH);
        valid_input_times += source_end.saturating_sub(source_start);
    }
    let output_channel_tiles = geometry.output_channel_tiles as usize;
    let input_read_bytes = valid_input_times * CHANNELS * output_channel_tiles * F32_BYTES;
    let reduction_tiles = CHANNELS / case.production_tile.input_channel_tile();
    let weight_read_bytes = workgroups * reduction_tiles * geometry.weight_tile_size * F32_BYTES;
    let bias_read_bytes = workgroups * 256 * 2 * F32_BYTES;
    let alpha_read_bytes = ELEMENTS * F32_BYTES;
    let output_write_bytes = ELEMENTS * F32_BYTES;
    let total_bytes = input_read_bytes
        + weight_read_bytes
        + bias_read_bytes
        + alpha_read_bytes
        + output_write_bytes;
    CurrentTraffic {
        input_read_bytes,
        weight_read_bytes,
        bias_read_bytes,
        alpha_read_bytes,
        output_write_bytes,
        total_bytes,
    }
}

fn print_static_case(case: ConvCase) {
    let current_geometry = current_geometry(case);
    let current_workgroups = current_geometry.workgroups().expect("workgroups fit usize");
    let current_barriers =
        current_workgroups * 2 * (CHANNELS / case.production_tile.input_channel_tile());
    let current_traffic = current_traffic(case);
    let descriptor = candidate_descriptor(case);
    let candidate_geometry = ResidueCandidateGeometry::new(descriptor);
    let candidate_traffic = ResidueCandidateTraffic::new(descriptor);
    let candidate_shape = candidate_geometry.descriptor.shape();
    println!(
        "static_current case={} tile={} dispatches=1 workgroups={} barriers={} shared_bytes={} temp_bytes=0 persistent_bytes=0 output_unpack_dispatches=0 input_read_bytes={} weight_read_bytes={} bias_read_bytes={} alpha_read_bytes={} output_write_bytes={} total_semantic_buffer_bytes={} total_mib={:.6}",
        case.label,
        case.production_tile.label(),
        current_workgroups,
        current_barriers,
        current_geometry.shared_bytes,
        current_traffic.input_read_bytes,
        current_traffic.weight_read_bytes,
        current_traffic.bias_read_bytes,
        current_traffic.alpha_read_bytes,
        current_traffic.output_write_bytes,
        current_traffic.total_bytes,
        current_traffic.total_bytes as f64 / 1_048_576.0,
    );
    println!(
        "static_candidate case={} descriptor={} channels={} length={} dilation={} tile=T256/O32/Cin16/WG16x16 dispatches={} pack_workgroups={} core_workgroups={} core_grid=[{},{},{}] core_barriers={} core_shared_bytes={} pack_elements={} temporary_bytes={} persistent_bytes={} output_unpack_dispatches={} pack_read_bytes={} pack_write_bytes={} core_input_read_bytes={} core_weight_read_bytes={} core_bias_read_bytes={} core_alpha_read_bytes={} core_output_write_bytes={} total_semantic_buffer_bytes={} total_mib={:.6}",
        case.label,
        case.candidate_dilation.label(),
        candidate_shape.channels(),
        candidate_shape.length(),
        case.candidate_dilation.value(),
        candidate_geometry.dispatches,
        candidate_geometry.pack_workgroups,
        candidate_geometry.core_workgroups,
        candidate_geometry.core_time_tiles,
        candidate_geometry.core_output_channel_tiles,
        candidate_geometry.core_residues,
        candidate_geometry.core_barriers,
        candidate_geometry.core_shared_bytes,
        candidate_geometry.packed_elements,
        candidate_geometry.temporary_bytes,
        candidate_geometry.persistent_bytes,
        candidate_geometry.output_unpack_dispatches,
        candidate_traffic.pack_read_bytes,
        candidate_traffic.pack_write_bytes,
        candidate_traffic.core_input_read_bytes,
        candidate_traffic.core_weight_read_bytes,
        candidate_traffic.core_bias_read_bytes,
        candidate_traffic.core_alpha_read_bytes,
        candidate_traffic.core_output_write_bytes,
        candidate_traffic.total_bytes,
        candidate_traffic.total_bytes as f64 / 1_048_576.0,
    );
}

fn benchmark_case(
    fixture: &CaseFixture,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
) -> Result<CaseResult> {
    let comparison = verify_full_output(fixture, device, monitor)?;
    warm_fixture(fixture, device, monitor)?;
    let mut samples: [Vec<f64>; VARIANT_COUNT] =
        std::array::from_fn(|_| Vec::with_capacity(TRIALS));
    for trial in 0..TRIALS {
        for offset in 0..VARIANT_COUNT {
            let variant_index = (trial + offset) % VARIANT_COUNT;
            let variant = Variant::ALL[variant_index];
            let sample = measure_variant(variant, fixture, device, monitor, trial)?;
            println!(
                "timing_sample case={} trial={} rotation_position={} variant={} iterations={} mean_us={:.3}",
                fixture.case.label,
                trial,
                offset,
                variant.label(),
                ITERATIONS,
                sample,
            );
            samples[variant_index].push(sample);
        }
    }
    let timings = std::array::from_fn(|index| summarize(&samples[index]));
    let current = timings[Variant::CurrentProduction.index()];
    let candidate = timings[Variant::ResidueC96Candidate.index()];
    let max_below_current_min = candidate.max_us < current.min_us;
    let median_below_conv_only_diagnostic =
        candidate.median_us < fixture.case.pytorch_conv_only_diagnostic_us;
    let max_below_pytorch_same_work_global_min = fixture
        .case
        .pytorch_same_work_global_min_us
        .is_some_and(|target_us| candidate.max_us < target_us);
    let accepted = max_below_current_min && max_below_pytorch_same_work_global_min;
    for variant in Variant::ALL {
        let timing = timings[variant.index()];
        println!(
            "timing case={} variant={} median_us={:.3} range_us=[{:.3},{:.3}] speedup_vs_current={:.4}",
            fixture.case.label,
            variant.label(),
            timing.median_us,
            timing.min_us,
            timing.max_us,
            current.median_us / timing.median_us,
        );
    }
    println!(
        "shape_verdict case={} candidate_max_us={:.3} current_min_us={:.3} candidate_max_below_current_min={} candidate_median_us={:.3} pytorch_conv_only_diagnostic_us={:.3} candidate_median_below_conv_only_diagnostic={} conv_only_diagnostic_is_same_work=false conv_only_diagnostic_is_hard_gate=false pytorch_same_work_global_min_us={:?} same_work_target_fixed={} candidate_max_below_pytorch_same_work_global_min={} pytorch_scope=conv_plus_bias_then_production_snake candidate_scope=production_weight_pack_plus_k7_plus_snake final_same_work_hard_gate=candidate_max_below_pytorch_global_min accepted={}",
        fixture.case.label,
        candidate.max_us,
        current.min_us,
        max_below_current_min,
        candidate.median_us,
        fixture.case.pytorch_conv_only_diagnostic_us,
        median_below_conv_only_diagnostic,
        fixture.case.pytorch_same_work_global_min_us,
        fixture.case.pytorch_same_work_global_min_us.is_some(),
        max_below_pytorch_same_work_global_min,
        accepted,
    );
    Ok(CaseResult {
        case: fixture.case,
        timings,
        comparison,
        accepted,
    })
}

fn print_aggregate(results: &[CaseResult]) -> bool {
    let current_median_sum = results
        .iter()
        .map(|result| result.timings[Variant::CurrentProduction.index()].median_us)
        .sum::<f64>();
    let current_min_sum = results
        .iter()
        .map(|result| result.timings[Variant::CurrentProduction.index()].min_us)
        .sum::<f64>();
    let candidate_median_sum = results
        .iter()
        .map(|result| result.timings[Variant::ResidueC96Candidate.index()].median_us)
        .sum::<f64>();
    let candidate_max_sum = results
        .iter()
        .map(|result| result.timings[Variant::ResidueC96Candidate.index()].max_us)
        .sum::<f64>();
    let pytorch_conv_only_diagnostic_sum = results
        .iter()
        .map(|result| result.case.pytorch_conv_only_diagnostic_us)
        .sum::<f64>();
    let elements = results
        .iter()
        .map(|result| result.comparison.elements)
        .sum::<usize>();
    let bit_mismatches = results
        .iter()
        .map(|result| result.comparison.bit_mismatches)
        .sum::<usize>();
    let max_abs = results
        .iter()
        .map(|result| result.comparison.max_abs)
        .fold(0.0_f32, f32::max);
    let mean_abs_sum = results
        .iter()
        .map(|result| result.comparison.mean_abs)
        .sum::<f64>();
    let finite = results.iter().all(|result| result.comparison.finite);
    let every_shape_passed = results.iter().all(|result| result.accepted);
    let aggregate_max_below_current_min = candidate_max_sum < current_min_sum;
    let aggregate_max_below_pytorch_same_work_suite_global_min =
        PYTORCH_SAME_WORK_SUITE_GLOBAL_MIN_US
            .is_some_and(|target_us| candidate_max_sum < target_us);
    let accepted = every_shape_passed
        && aggregate_max_below_current_min
        && aggregate_max_below_pytorch_same_work_suite_global_min
        && finite
        && bit_mismatches == 0
        && max_abs == 0.0
        && mean_abs_sum == 0.0;
    println!(
        "aggregate_timing shapes={} current_median_sum_us={:.3} current_min_sum_us={:.3} candidate_median_sum_us={:.3} candidate_max_sum_us={:.3} pytorch_conv_only_diagnostic_sum_us={:.3} conv_only_diagnostic_is_same_work=false conv_only_diagnostic_is_hard_gate=false pytorch_same_work_suite_global_min_us={:?} same_work_suite_target_fixed={} measured_speedup={:.4} candidate_max_sum_below_current_min_sum={} candidate_max_sum_below_pytorch_same_work_suite_global_min={} every_shape_passed={} accepted={}",
        results.len(),
        current_median_sum,
        current_min_sum,
        candidate_median_sum,
        candidate_max_sum,
        pytorch_conv_only_diagnostic_sum,
        PYTORCH_SAME_WORK_SUITE_GLOBAL_MIN_US,
        PYTORCH_SAME_WORK_SUITE_GLOBAL_MIN_US.is_some(),
        current_median_sum / candidate_median_sum,
        aggregate_max_below_current_min,
        aggregate_max_below_pytorch_same_work_suite_global_min,
        every_shape_passed,
        accepted,
    );
    println!(
        "aggregate_correctness shapes={} elements={} finite={} bit_mismatch={} max_abs={:.9e} mean_abs_sum={:.9e} hard_gate={}",
        results.len(),
        elements,
        finite,
        bit_mismatches,
        max_abs,
        mean_abs_sum,
        if finite && bit_mismatches == 0 && max_abs == 0.0 && mean_abs_sum == 0.0 {
            "pass"
        } else {
            "fail"
        },
    );
    accepted
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        CASES
            .iter()
            .all(|case| case.pytorch_same_work_global_min_us.is_some())
            && PYTORCH_SAME_WORK_SUITE_GLOBAL_MIN_US.is_some(),
        "same-work PyTorch targets are not fixed: run the pinned Python Conv1d+bias->production-Snake control, then insert each case and two-call-suite global minimum before any Rust GPU acceptance run"
    );
    ensure!(
        args.codec_weights_sha256
            .eq_ignore_ascii_case(PINNED_CODEC_SHA256),
        "acceptance is pinned to codec SHA-256 {PINNED_CODEC_SHA256}"
    );
    verify_sha256(&args.codec_weights, &args.codec_weights_sha256)?;
    println!(
        "benchmark_protocol precision=f32 cases=2 candidates=1 no_c384_d1_anomaly=true current_launcher=direct_production_t256_snake_vec4 candidate_launcher=isolated_residue_c96 actual_checkpoint_weights=true deterministic_inputs=true warmup={} iterations={} trials={} rotating_order=true full_output_elements_per_shape={} expected_correctness=finite_bit0_hash_equal_maxabs0_meanabs0",
        WARMUP, ITERATIONS, TRIALS, ELEMENTS,
    );
    println!(
        "candidate_semantics pack=[residue][channel][q] compact_ragged=true core_dilation=1 padding=3 reduction_order=input_channel_then_tap0_through_6_fma snake_order=scalar_production direct_scatter=t_residue_plus_q_times_d output_unpack=0 candidate_count=1 t128_variant_skipped=would_require_new_core_not_localized"
    );
    for case in CASES {
        print_static_case(case);
    }

    let (device, monitor) = initialize_wgpu(args.adapter_index);
    let store = TensorStore::load(&args.codec_weights)
        .with_context(|| format!("load codec store {}", args.codec_weights.display()))?;
    let fixtures = CASES
        .iter()
        .map(|case| make_fixture(&store, &device, *case))
        .collect::<Result<Vec<_>>>()?;
    synchronize_and_check_wgpu(&device, &monitor, "fixture construction")?;
    for fixture in &fixtures {
        validate_fixture_contract(fixture)?;
    }

    let results = fixtures
        .iter()
        .map(|fixture| benchmark_case(fixture, &device, &monitor))
        .collect::<Result<Vec<_>>>()?;
    let accepted = print_aggregate(&results);
    monitor.check("benchmark completion")?;
    println!("wgpu_errors=0");
    std::io::stdout()
        .flush()
        .context("flush completed benchmark report")?;
    ensure!(
        accepted,
        "C96 residue acceptance failed: each shape requires candidate max < current min and candidate max < its same-work PyTorch global minimum; aggregate requires candidate max-sum < current min-sum and < the same-work PyTorch two-call-suite global minimum"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cases_are_exactly_released_c96_d3_and_d9() {
        assert_eq!(CASES.len(), 2);
        assert_eq!(CASES[0].candidate_dilation, ResidueCandidateDilation::Three);
        assert_eq!(CASES[1].candidate_dilation, ResidueCandidateDilation::Nine);
        assert_eq!(CASES[0].production_tile, Conv1dK7T256Tile::Cin16);
        assert_eq!(CASES[1].production_tile, Conv1dK7T256Tile::Cin8);
        assert_eq!(
            CASES.map(|case| case.pytorch_conv_only_diagnostic_us),
            [1_716.429, 1_717.125],
        );
        assert!(
            CASES
                .iter()
                .all(|case| case.pytorch_same_work_global_min_us.is_none())
        );
        assert_eq!(PYTORCH_SAME_WORK_SUITE_GLOBAL_MIN_US, None);
        assert_eq!(ELEMENTS, 9_216_000);
        assert_eq!(WARMUP, 10);
        assert_eq!(ITERATIONS, 100);
        assert_eq!(TRIALS, 5);
    }

    #[test]
    fn checkpoint_keys_match_decoder_block3_residual_layout() {
        assert_eq!(
            CASES.map(|case| case.weight_key),
            [
                "decoder.model.4.block.5.block.1.weight",
                "decoder.model.4.block.8.block.1.weight",
            ],
        );
        assert_eq!(
            CASES.map(|case| case.alpha_key),
            [
                "decoder.model.4.block.5.block.2.alpha",
                "decoder.model.4.block.8.block.2.alpha",
            ],
        );
    }

    #[test]
    fn exact_static_accounting_matches_the_sources() {
        let d3_current = current_traffic(CASES[0]);
        assert_eq!(d3_current.input_read_bytes, 118_347_264);
        assert_eq!(d3_current.weight_read_bytes, 96_768_000);
        assert_eq!(d3_current.bias_read_bytes, 2_304_000);
        assert_eq!(d3_current.total_bytes, 291_147_264);
        let d9_current = current_traffic(CASES[1]);
        assert_eq!(d9_current.input_read_bytes, 133_857_792);
        assert_eq!(d9_current.weight_read_bytes, 96_768_000);
        assert_eq!(d9_current.bias_read_bytes, 2_304_000);
        assert_eq!(d9_current.total_bytes, 306_657_792);

        let candidate = CASES.map(|case| {
            let descriptor = candidate_descriptor(case);
            (
                ResidueCandidateGeometry::new(descriptor),
                ResidueCandidateTraffic::new(descriptor),
            )
        });
        assert_eq!(candidate[0].0.core_workgroups, 1_125);
        assert_eq!(candidate[0].0.core_barriers, 13_500);
        assert_eq!(candidate[0].1.total_bytes, 359_691_264);
        assert_eq!(candidate[1].0.core_workgroups, 1_134);
        assert_eq!(candidate[1].0.core_barriers, 13_608);
        assert_eq!(candidate[1].1.total_bytes, 360_463_104);
    }

    #[test]
    fn rotation_is_balanced_and_alternates_first_variant() {
        let orders = (0..TRIALS)
            .map(|trial| {
                (0..VARIANT_COUNT)
                    .map(|offset| Variant::ALL[(trial + offset) % VARIANT_COUNT])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            orders[0],
            [Variant::CurrentProduction, Variant::ResidueC96Candidate],
        );
        assert_eq!(
            orders[1],
            [Variant::ResidueC96Candidate, Variant::CurrentProduction],
        );
        assert_eq!(
            orders
                .iter()
                .flatten()
                .filter(|variant| **variant == Variant::CurrentProduction)
                .count(),
            TRIALS,
        );
        assert_eq!(
            orders
                .iter()
                .flatten()
                .filter(|variant| **variant == Variant::ResidueC96Candidate)
                .count(),
            TRIALS,
        );
    }
}
