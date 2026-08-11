//! Production-weight regression screen for all six released pointwise calls.
//!
//! This isolated screen compares the current production graph
//! `packed matmul -> prepared-pair/finalizer` with the production
//! T64/O96/K32 vec4 winner imported from the library. The strict full-decoder
//! waveform and hash gates remain separate from this kernel-level screen.

use std::{
    collections::BTreeMap,
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
        init_setup, into_contiguous,
    },
    tensor::{DType, Distribution, Tensor, TensorPrimitive, backend::Backend},
};
use clap::Parser;
use cubecl::prelude::Runtime;
use irodori_tts_wgpu::{
    WgpuRaw,
    kernels::{
        pointwise_residual_direct_tiled::{
            PointwiseKTile, PointwiseResidualDirectInputs, RELEASED_SHAPES,
            pointwise_residual_direct_contract_is_compatible, pointwise_residual_direct_raw_wgsl,
            pointwise_residual_direct_snake_pair_wgsl,
        },
        pointwise_residual_finalizer::pointwise_residual_finalizer_wgsl,
        pointwise_residual_snake_pair::pointwise_residual_snake_pair_wgsl,
    },
    weights::TensorStore,
};
use sha2::{Digest, Sha256};

type B = WgpuRaw;

const PINNED_CODEC_SHA256: &str =
    "4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1";
const ELEMENTS_PER_CALL: usize = 9_216_000;
const F32_BYTES: usize = size_of::<f32>();
const CURRENT_POINTWISE_ESTIMATE_MS: f64 = 12.77;
const PYTORCH_SAME_WORK_DEVICE_MEDIAN_MS: f64 = 8.588_503;
const PYTORCH_SAME_WORK_DEVICE_GLOBAL_MIN_MS: f64 = 8.584_956_665;
const PYTORCH_K1_CONV_ONLY_SIX_CALL_MS: f64 = 3.155_006;
const REQUIRED_SAVING_MS: f64 = 9.8;
const DIRECT_FUSED_SUITE_TARGET_MS: f64 = 3.0;

#[derive(Debug, Parser)]
#[command(about = "Direct pointwise tile screen over all six C192/C96 calls")]
struct Args {
    #[arg(long, default_value = "target/v4_dacvae_weights.safetensors")]
    codec_weights: PathBuf,
    #[arg(long, default_value = PINNED_CODEC_SHA256)]
    codec_weights_sha256: String,
    #[arg(long, default_value_t = 0)]
    adapter_index: usize,
    #[arg(long, default_value_t = 10)]
    warmup: usize,
    #[arg(long, default_value_t = 100)]
    iterations: usize,
    #[arg(long, default_value_t = 5)]
    trials: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BoundaryKind {
    Pair { next_alpha_key: &'static str },
    Raw,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PointwiseCase {
    label: &'static str,
    channels: usize,
    length: usize,
    weight_key: &'static str,
    bias_key: &'static str,
    boundary: BoundaryKind,
    seed: u64,
}

const CASES: [PointwiseCase; 6] = [
    PointwiseCase {
        label: "block2_res0_pair",
        channels: 192,
        length: 48_000,
        weight_key: "decoder.model.3.block.4.block.3.weight",
        bias_key: "decoder.model.3.block.4.block.3.bias",
        boundary: BoundaryKind::Pair {
            next_alpha_key: "decoder.model.3.block.5.block.0.alpha",
        },
        seed: 0x1920_0000,
    },
    PointwiseCase {
        label: "block2_res1_pair",
        channels: 192,
        length: 48_000,
        weight_key: "decoder.model.3.block.5.block.3.weight",
        bias_key: "decoder.model.3.block.5.block.3.bias",
        boundary: BoundaryKind::Pair {
            next_alpha_key: "decoder.model.3.block.8.block.0.alpha",
        },
        seed: 0x1920_0001,
    },
    PointwiseCase {
        label: "block2_res2_raw",
        channels: 192,
        length: 48_000,
        weight_key: "decoder.model.3.block.8.block.3.weight",
        bias_key: "decoder.model.3.block.8.block.3.bias",
        boundary: BoundaryKind::Raw,
        seed: 0x1920_0002,
    },
    PointwiseCase {
        label: "block3_res0_pair",
        channels: 96,
        length: 96_000,
        weight_key: "decoder.model.4.block.4.block.3.weight",
        bias_key: "decoder.model.4.block.4.block.3.bias",
        boundary: BoundaryKind::Pair {
            next_alpha_key: "decoder.model.4.block.5.block.0.alpha",
        },
        seed: 0x0960_0000,
    },
    PointwiseCase {
        label: "block3_res1_pair",
        channels: 96,
        length: 96_000,
        weight_key: "decoder.model.4.block.5.block.3.weight",
        bias_key: "decoder.model.4.block.5.block.3.bias",
        boundary: BoundaryKind::Pair {
            next_alpha_key: "decoder.model.4.block.8.block.0.alpha",
        },
        seed: 0x0960_0001,
    },
    PointwiseCase {
        label: "block3_res2_raw",
        channels: 96,
        length: 96_000,
        weight_key: "decoder.model.4.block.8.block.3.weight",
        bias_key: "decoder.model.4.block.8.block.3.bias",
        boundary: BoundaryKind::Raw,
        seed: 0x0960_0002,
    },
];

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Variant {
    CurrentProduction,
    ProductionDirect,
}

impl Variant {
    const ALL: [Self; 2] = [Self::CurrentProduction, Self::ProductionDirect];

    const CANDIDATES: [Self; 1] = [Self::ProductionDirect];

    const fn label(self) -> &'static str {
        match self {
            Self::CurrentProduction => "current_packed_matmul_finalizer",
            Self::ProductionDirect => "production_direct_t64_o96_k32_wg32x8_vec4",
        }
    }

    const fn tile(self, _case: PointwiseCase) -> Option<PointwiseKTile> {
        match self {
            Self::CurrentProduction => None,
            Self::ProductionDirect => Some(PointwiseKTile::PRODUCTION),
        }
    }
}

struct CaseFixture {
    case: PointwiseCase,
    input_ncl: Tensor<B, 3>,
    packed_weight_kco: Tensor<B, 3>,
    one_time_pack_us: f64,
    bias: Tensor<B, 1>,
    residual_ncl: Tensor<B, 3>,
    alpha: Option<Tensor<B, 3>>,
}

struct CaseOutput {
    case: PointwiseCase,
    raw: Tensor<B, 3>,
    activated: Option<Tensor<B, 3>>,
}

struct HostCaseOutput {
    case: PointwiseCase,
    raw: Vec<f32>,
    activated: Option<Vec<f32>>,
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
        info.name, info.backend, info.device_type
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

fn ensure_dims<const D: usize>(
    label: &str,
    actual: [usize; D],
    expected: [usize; D],
) -> Result<()> {
    ensure!(
        actual == expected,
        "{label}: expected {expected:?}, got {actual:?}"
    );
    Ok(())
}

fn ensure_exact_cube_layout<const D: usize>(
    label: &str,
    tensor: &burn::backend::wgpu::CubeTensor<WgpuRuntime>,
    expected_shape: [usize; D],
    expected_strides: [usize; D],
) -> Result<()> {
    let actual_shape = tensor.meta.shape().dims::<D>();
    let actual_strides = tensor.meta.strides();
    ensure!(
        tensor.meta.num_dims() == D
            && actual_shape == expected_shape
            && &actual_strides[..] == expected_strides.as_slice()
            && tensor.dtype == DType::F32
            && tensor.is_contiguous(),
        "{label}: expected shape={expected_shape:?} strides={expected_strides:?} dtype=F32 contiguous=true; got rank={} shape={:?} strides={actual_strides:?} dtype={:?} contiguous={}",
        tensor.meta.num_dims(),
        tensor.meta.shape(),
        tensor.dtype,
        tensor.is_contiguous(),
    );
    Ok(())
}

fn pack_pointwise_weight_contiguous(
    weight_oik: Tensor<B, 3>,
    case: PointwiseCase,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
) -> Result<(Tensor<B, 3>, f64)> {
    let channels = case.channels;
    let weight_elements = channels * channels;
    let source_raw = weight_oik.clone().into_primitive().tensor();
    ensure_exact_cube_layout(
        &format!("{} source OIK", case.label),
        &source_raw,
        [channels, channels, 1],
        [channels, 1, 1],
    )?;

    // `transpose` is intentionally only a logical KCO view here. Burn's
    // elementwise no-op optimizations are not a materialization contract, so
    // the one-time physical pack below uses CubeCL's explicit copy primitive.
    let logical_kco = weight_oik
        .squeeze_dim::<2>(2)
        .transpose()
        .unsqueeze_dim::<3>(0);
    let logical_raw = logical_kco.clone().into_primitive().tensor();
    ensure!(
        logical_raw.meta.shape().dims::<3>() == [1, channels, channels]
            && &logical_raw.meta.strides()[..] == [weight_elements, 1, channels].as_slice()
            && logical_raw.dtype == DType::F32
            && !logical_raw.is_contiguous(),
        "{} logical KCO view: expected shape=[1,{channels},{channels}] strides=[{weight_elements},1,{channels}] dtype=F32 contiguous=false; got shape={:?} strides={:?} dtype={:?} contiguous={}",
        case.label,
        logical_raw.meta.shape(),
        logical_raw.meta.strides(),
        logical_raw.dtype,
        logical_raw.is_contiguous(),
    );

    synchronize_and_check_wgpu(device, monitor, &format!("{} pack pre-sync", case.label))?;
    let started = Instant::now();
    let packed_raw = into_contiguous(logical_kco.into_primitive().tensor());
    synchronize_and_check_wgpu(
        device,
        monitor,
        &format!("{} explicit weight pack", case.label),
    )?;
    let one_time_pack_us = started.elapsed().as_secs_f64() * 1_000_000.0;
    ensure_exact_cube_layout(
        &format!("{} packed KCO", case.label),
        &packed_raw,
        [1, channels, channels],
        [weight_elements, channels, 1],
    )?;
    ensure!(
        packed_raw.device == source_raw.device,
        "{} explicit pack changed device: source={:?} packed={:?}",
        case.label,
        source_raw.device,
        packed_raw.device,
    );
    let packed_weight_kco = Tensor::from_primitive(TensorPrimitive::Float(packed_raw));

    // Full-element checkpoint gate. OIK is physical `o*C+k`; the direct
    // kernels consume physical KCO `k*C+o`. An explicit copy must preserve
    // every f32 bit under that transpose.
    let source = weight_oik_to_host(source_raw)?;
    let packed = packed_weight_kco
        .clone()
        .into_data()
        .to_vec::<f32>()
        .with_context(|| format!("{} packed KCO readback", case.label))?;
    synchronize_and_check_wgpu(
        device,
        monitor,
        &format!("{} weight-pack correctness readback", case.label),
    )?;
    ensure!(
        source.len() == weight_elements && packed.len() == weight_elements,
        "{} weight-pack element count drift: source={} packed={} expected={weight_elements}",
        case.label,
        source.len(),
        packed.len(),
    );
    let mut bit_mismatches = 0_usize;
    let mut max_abs = 0.0_f32;
    let mut finite = true;
    for input_channel in 0..channels {
        for output_channel in 0..channels {
            let expected = source[output_channel * channels + input_channel];
            let actual = packed[input_channel * channels + output_channel];
            finite &= expected.is_finite() && actual.is_finite();
            bit_mismatches += usize::from(expected.to_bits() != actual.to_bits());
            max_abs = max_abs.max((expected - actual).abs());
        }
    }
    ensure!(
        finite && bit_mismatches == 0,
        "{} explicit weight pack failed: finite={finite} bit_mismatches={bit_mismatches} max_abs={max_abs:.9e}",
        case.label,
    );
    println!(
        "weight_pack case={} method=burn_wgpu_into_contiguous source=OIK[o*C+k] destination=KCO[k*C+o] elements={} one_time_us={:.3} timing_scope=pre_synced_copy_plus_post_sync benchmark_included=false full_gate=bit_exact bit_mismatch={} max_abs={:.9e} finite={}",
        case.label, weight_elements, one_time_pack_us, bit_mismatches, max_abs, finite,
    );
    Ok((packed_weight_kco, one_time_pack_us))
}

fn weight_oik_to_host(
    source_raw: burn::backend::wgpu::CubeTensor<WgpuRuntime>,
) -> Result<Vec<f32>> {
    Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(source_raw))
        .into_data()
        .to_vec::<f32>()
        .context("source OIK readback")
}

fn make_fixture(
    store: &TensorStore,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    case: PointwiseCase,
) -> Result<CaseFixture> {
    let weight_oik: Tensor<B, 3> = store
        .tensor(case.weight_key, device)
        .with_context(|| format!("load {}", case.weight_key))?;
    let bias: Tensor<B, 1> = store
        .tensor(case.bias_key, device)
        .with_context(|| format!("load {}", case.bias_key))?;
    ensure_dims(
        case.weight_key,
        weight_oik.dims(),
        [case.channels, case.channels, 1],
    )?;
    ensure_dims(case.bias_key, bias.dims(), [case.channels])?;
    let (packed_weight_kco, one_time_pack_us) =
        pack_pointwise_weight_contiguous(weight_oik, case, device, monitor)?;
    ensure_dims(
        "packed production weight",
        packed_weight_kco.dims(),
        [1, case.channels, case.channels],
    )?;
    let alpha = match case.boundary {
        BoundaryKind::Pair { next_alpha_key } => {
            let alpha: Tensor<B, 3> = store
                .tensor(next_alpha_key, device)
                .with_context(|| format!("load {next_alpha_key}"))?;
            ensure_dims(next_alpha_key, alpha.dims(), [1, case.channels, 1])?;
            Some(alpha)
        }
        BoundaryKind::Raw => None,
    };
    B::seed(device, case.seed);
    let input_ncl = Tensor::<B, 3>::random(
        [1, case.channels, case.length],
        Distribution::Uniform(-0.5, 0.5),
        device,
    );
    let residual_ncl = Tensor::<B, 3>::random(
        [1, case.channels, case.length],
        Distribution::Uniform(-0.5, 0.5),
        device,
    );
    Ok(CaseFixture {
        case,
        input_ncl,
        packed_weight_kco,
        one_time_pack_us,
        bias,
        residual_ncl,
        alpha,
    })
}

fn direct_inputs(fixture: &CaseFixture) -> PointwiseResidualDirectInputs {
    PointwiseResidualDirectInputs::new(
        fixture.input_ncl.clone().into_primitive().tensor(),
        fixture.packed_weight_kco.clone().into_primitive().tensor(),
        fixture.bias.clone().into_primitive().tensor(),
        fixture.residual_ncl.clone().into_primitive().tensor(),
    )
}

fn current_forward(fixture: &CaseFixture) -> Result<CaseOutput> {
    let branch_nlc = fixture
        .input_ncl
        .clone()
        .swap_dims(1, 2)
        .matmul(fixture.packed_weight_kco.clone());
    match (&fixture.case.boundary, &fixture.alpha) {
        (BoundaryKind::Pair { .. }, Some(alpha)) => {
            let pair = pointwise_residual_snake_pair_wgsl(
                branch_nlc.into_primitive().tensor(),
                fixture.bias.clone().into_primitive().tensor(),
                fixture.residual_ncl.clone().into_primitive().tensor(),
                alpha.clone().into_primitive().tensor(),
            )?;
            let (raw, activated) = pair.into_tensors();
            Ok(CaseOutput {
                case: fixture.case,
                raw: Tensor::from_primitive(TensorPrimitive::Float(raw)),
                activated: Some(Tensor::from_primitive(TensorPrimitive::Float(activated))),
            })
        }
        (BoundaryKind::Raw, None) => {
            let raw = pointwise_residual_finalizer_wgsl(
                branch_nlc.into_primitive().tensor(),
                fixture.bias.clone().into_primitive().tensor(),
                fixture.residual_ncl.clone().into_primitive().tensor(),
            )?;
            Ok(CaseOutput {
                case: fixture.case,
                raw: Tensor::from_primitive(TensorPrimitive::Float(raw)),
                activated: None,
            })
        }
        _ => anyhow::bail!("{} boundary/alpha fixture mismatch", fixture.case.label),
    }
}

fn direct_forward(fixture: &CaseFixture, tile: PointwiseKTile) -> Result<CaseOutput> {
    match (&fixture.case.boundary, &fixture.alpha) {
        (BoundaryKind::Pair { .. }, Some(alpha)) => {
            let inputs = direct_inputs(fixture);
            let alpha = alpha.clone().into_primitive().tensor();
            let pair = pointwise_residual_direct_snake_pair_wgsl(inputs, alpha, tile)
                .with_context(|| {
                    format!(
                        "direct pair launch failed for case={} tile={}",
                        fixture.case.label,
                        tile.label()
                    )
                })?;
            let (raw, activated) = pair.into_tensors();
            Ok(CaseOutput {
                case: fixture.case,
                raw: Tensor::from_primitive(TensorPrimitive::Float(raw)),
                activated: Some(Tensor::from_primitive(TensorPrimitive::Float(activated))),
            })
        }
        (BoundaryKind::Raw, None) => {
            let inputs = direct_inputs(fixture);
            let raw = pointwise_residual_direct_raw_wgsl(inputs, tile).with_context(|| {
                format!(
                    "direct raw launch failed for case={} tile={}",
                    fixture.case.label,
                    tile.label()
                )
            })?;
            Ok(CaseOutput {
                case: fixture.case,
                raw: Tensor::from_primitive(TensorPrimitive::Float(raw)),
                activated: None,
            })
        }
        _ => anyhow::bail!("{} boundary/alpha fixture mismatch", fixture.case.label),
    }
}

fn print_cube_tensor_contract(
    case: &str,
    name: &str,
    tensor: &burn::backend::wgpu::CubeTensor<WgpuRuntime>,
) {
    println!(
        "fixture_tensor case={case} name={name} shape={:?} strides={:?} dtype={:?} contiguous={} device={:?}",
        tensor.meta.shape(),
        tensor.meta.strides(),
        tensor.dtype,
        tensor.is_contiguous(),
        tensor.device,
    );
}

fn validate_and_print_fixture_contracts(fixtures: &[CaseFixture]) -> Result<()> {
    if let Some(first) = fixtures.first() {
        let reference = first.input_ncl.clone().into_primitive().tensor();
        let properties = reference.client.properties();
        let hardware = &properties.hardware;
        println!(
            "device_contract bindings={} shared_bytes={} units={} dim={:?} count={:?} page_bytes={}",
            hardware.max_bindings,
            hardware.max_shared_memory_size,
            hardware.max_units_per_cube,
            hardware.max_cube_dim,
            hardware.max_cube_count,
            properties.memory.max_page_size,
        );
    }
    let expected_contracts = fixtures.len() * Variant::CANDIDATES.len();
    let mut compatible_contracts = 0_usize;
    for fixture in fixtures {
        let input = fixture.input_ncl.clone().into_primitive().tensor();
        let packed_weight = fixture.packed_weight_kco.clone().into_primitive().tensor();
        let bias = fixture.bias.clone().into_primitive().tensor();
        let residual = fixture.residual_ncl.clone().into_primitive().tensor();
        print_cube_tensor_contract(fixture.case.label, "input_ncl", &input);
        print_cube_tensor_contract(fixture.case.label, "packed_weight_kco", &packed_weight);
        print_cube_tensor_contract(fixture.case.label, "bias", &bias);
        print_cube_tensor_contract(fixture.case.label, "residual_ncl", &residual);
        let alpha = fixture
            .alpha
            .as_ref()
            .map(|alpha| alpha.clone().into_primitive().tensor());
        if let Some(alpha) = &alpha {
            print_cube_tensor_contract(fixture.case.label, "alpha", alpha);
        }
        for variant in Variant::CANDIDATES {
            let tile = variant.tile(fixture.case).ok_or_else(|| {
                anyhow::anyhow!(
                    "candidate {} has no tile for released case {} C{}",
                    variant.label(),
                    fixture.case.label,
                    fixture.case.channels,
                )
            })?;
            let inputs = direct_inputs(fixture);
            let compatible =
                pointwise_residual_direct_contract_is_compatible(&inputs, alpha.as_ref(), tile);
            compatible_contracts += usize::from(compatible);
            println!(
                "fixture_contract case={} variant={} tile={} pair={} compatible={}",
                fixture.case.label,
                variant.label(),
                tile.label(),
                alpha.is_some(),
                compatible,
            );
        }
    }
    let all_compatible = compatible_contracts == expected_contracts;
    println!(
        "fixture_contract_summary compatible={} total={} expected=6 all_compatible={} candidate_dispatches_before_gate=0",
        compatible_contracts, expected_contracts, all_compatible,
    );
    std::io::stdout()
        .flush()
        .context("flush fixture contract preflight before candidate gate")?;
    ensure!(
        expected_contracts == CASES.len() && all_compatible,
        "candidate preflight rejected: compatible={compatible_contracts}/{expected_contracts}, expected all 6; candidate dispatches=0"
    );
    Ok(())
}

fn run_suite(variant: Variant, fixtures: &[CaseFixture]) -> Result<Vec<CaseOutput>> {
    fixtures
        .iter()
        .map(|fixture| {
            if variant == Variant::CurrentProduction {
                return current_forward(fixture);
            }
            let tile = variant.tile(fixture.case).ok_or_else(|| {
                anyhow::anyhow!(
                    "candidate {} has no fail-closed tile for case={} C{} L{}",
                    variant.label(),
                    fixture.case.label,
                    fixture.case.channels,
                    fixture.case.length,
                )
            })?;
            direct_forward(fixture, tile)
        })
        .collect()
}

fn to_host(outputs: Vec<CaseOutput>) -> Result<Vec<HostCaseOutput>> {
    outputs
        .into_iter()
        .map(|output| {
            Ok(HostCaseOutput {
                case: output.case,
                raw: output.raw.into_data().to_vec::<f32>()?,
                activated: output
                    .activated
                    .map(|tensor| tensor.into_data().to_vec::<f32>())
                    .transpose()?,
            })
        })
        .collect()
}

fn compare_values(reference: &[f32], candidate: &[f32]) -> Result<Comparison> {
    ensure!(
        reference.len() == candidate.len(),
        "comparison length mismatch"
    );
    let mut bit_mismatches = 0;
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

fn compare_host_suite(
    variant: Variant,
    reference: &[HostCaseOutput],
    candidate: &[HostCaseOutput],
) -> Result<()> {
    ensure!(reference.len() == CASES.len() && candidate.len() == CASES.len());
    for (expected, actual) in reference.iter().zip(candidate) {
        ensure!(expected.case == actual.case, "case ordering drift");
        let raw = compare_values(&expected.raw, &actual.raw)?;
        ensure!(
            raw.finite,
            "{} {} raw is non-finite",
            variant.label(),
            actual.case.label
        );
        let raw_status = if raw.bit_mismatches == 0 {
            "bit_exact"
        } else {
            "finite_recorded_delta"
        };
        let reference_sha256 = sha256_f32_le(&expected.raw);
        let candidate_sha256 = sha256_f32_le(&actual.raw);
        println!(
            "correctness variant={} case={} output=raw elements={} status={} bit_mismatch={} max_abs={:.9e} mean_abs={:.9e} hash_encoding=f32_ieee754_le reference_sha256={} candidate_sha256={}",
            variant.label(),
            actual.case.label,
            raw.elements,
            raw_status,
            raw.bit_mismatches,
            raw.max_abs,
            raw.mean_abs,
            reference_sha256,
            candidate_sha256,
        );
        match (&expected.activated, &actual.activated) {
            (Some(expected), Some(actual_values)) => {
                let activated = compare_values(expected, actual_values)?;
                ensure!(
                    activated.finite,
                    "{} {} activated output is non-finite",
                    variant.label(),
                    actual.case.label
                );
                let status = if activated.bit_mismatches == 0 {
                    "bit_exact"
                } else {
                    "finite_recorded_delta"
                };
                let reference_sha256 = sha256_f32_le(expected);
                let candidate_sha256 = sha256_f32_le(actual_values);
                println!(
                    "correctness variant={} case={} output=activated elements={} status={} bit_mismatch={} max_abs={:.9e} mean_abs={:.9e} hash_encoding=f32_ieee754_le reference_sha256={} candidate_sha256={}",
                    variant.label(),
                    actual.case.label,
                    activated.elements,
                    status,
                    activated.bit_mismatches,
                    activated.max_abs,
                    activated.mean_abs,
                    reference_sha256,
                    candidate_sha256,
                );
            }
            (None, None) => {}
            _ => anyhow::bail!("{} activated output presence mismatch", actual.case.label),
        }
    }
    Ok(())
}

fn warm_all(
    fixtures: &[CaseFixture],
    args: &Args,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
) -> Result<()> {
    let mut retained = None;
    for round in 0..args.warmup {
        for offset in 0..Variant::ALL.len() {
            let variant = Variant::ALL[(round + offset) % Variant::ALL.len()];
            retained = Some(run_suite(variant, fixtures)?);
        }
    }
    black_box(&retained);
    synchronize_and_check_wgpu(device, monitor, "rotating warmup")?;
    drop(retained);
    Ok(())
}

fn measure_variant(
    variant: Variant,
    fixtures: &[CaseFixture],
    args: &Args,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
) -> Result<f64> {
    let started = Instant::now();
    let mut retained = None;
    for _ in 0..args.iterations {
        retained = Some(run_suite(variant, fixtures)?);
    }
    black_box(&retained);
    synchronize_and_check_wgpu(device, monitor, &format!("{} timed trial", variant.label()))?;
    let elapsed_us = started.elapsed().as_secs_f64() * 1_000_000.0 / args.iterations as f64;
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

fn print_static_accounting() {
    let calls = CASES.len();
    let pair_calls = CASES
        .iter()
        .filter(|case| matches!(case.boundary, BoundaryKind::Pair { .. }))
        .count();
    let macs = CASES.iter().fold(0_u64, |total, case| {
        total + (case.channels as u64 * case.channels as u64 * case.length as u64)
    });
    let branch_transient_bytes = calls * 2 * ELEMENTS_PER_CALL * F32_BYTES;
    let live_output_bytes = (calls + pair_calls) * ELEMENTS_PER_CALL * F32_BYTES;
    println!(
        "graph_contract calls={calls} pair_calls={pair_calls} raw_calls={} dtype=f32 input=NCL packed_weight=[1,K,O]/offset=k*C+o reduction=ascending-K bias=last residual=after-bias",
        calls - pair_calls
    );
    println!("dispatch_accounting current=12 candidate=6 macs_per_suite={macs}",);
    println!(
        "traffic_accounting eliminated_branch_write_read={} bytes ({:.3} MiB) live_raw_plus_activated={} bytes ({:.3} MiB)",
        branch_transient_bytes,
        branch_transient_bytes as f64 / (1024.0 * 1024.0),
        live_output_bytes,
        live_output_bytes as f64 / (1024.0 * 1024.0),
    );
    let tile = PointwiseKTile::PRODUCTION;
    println!(
        "production_source_tile tile={} shared_bytes={} K={} time={} output={} workgroup=[{},{},1] vector_width={} fixed_across_released_shapes=true",
        tile.label(),
        tile.shared_memory_bytes(),
        tile.reduction(),
        tile.time_tile(),
        tile.output_tile(),
        tile.workgroup_x(),
        tile.workgroup_y(),
        tile.vector_width(),
    );
    for variant in Variant::CANDIDATES {
        let workgroups = CASES.iter().fold(0_usize, |total, case| {
            let tile = variant
                .tile(*case)
                .expect("all benchmark candidates cover all released cases");
            total + (case.length / tile.time_tile()) * case.channels.div_ceil(tile.output_tile())
        });
        for &(channels, length) in &RELEASED_SHAPES {
            let representative = CASES
                .iter()
                .find(|case| case.channels == channels && case.length == length)
                .expect("released benchmark shape has a fixture");
            let tile = variant
                .tile(*representative)
                .expect("all benchmark candidates cover all released cases");
            let reduction_passes = channels / tile.reduction();
            println!(
                "tile_accounting variant={} shape=C{channels}L{length} tile={} shared_bytes={} K={} reduction_passes={} barriers_per_call={} time={} output={} workgroup=[{},{},1] adjacent_times_per_local_x={} output_channels_per_thread={} outputs_per_thread={} accumulator_f32_per_thread={} vector_width={} fma_statements_per_k={} workgroups_per_suite={workgroups}",
                variant.label(),
                tile.label(),
                tile.shared_memory_bytes(),
                tile.reduction(),
                reduction_passes,
                2 * reduction_passes,
                tile.time_tile(),
                tile.output_tile(),
                tile.workgroup_x(),
                tile.workgroup_y(),
                tile.local_time_outputs(),
                tile.local_output_channels(),
                tile.outputs_per_thread(),
                tile.outputs_per_thread(),
                tile.vector_width(),
                tile.fma_statements_per_reduction_step(),
            );
        }
    }
    let compute_floor_ms = (macs as f64 * 2.0) / 16.2e12 * 1_000.0;
    println!(
        "acceptance_headline current_estimate_ms={CURRENT_POINTWISE_ESTIMATE_MS:.3} pytorch_same_work_device_median_ms={PYTORCH_SAME_WORK_DEVICE_MEDIAN_MS:.6} pytorch_same_work_device_global_min_ms={PYTORCH_SAME_WORK_DEVICE_GLOBAL_MIN_MS:.9} pytorch_same_work_scope=six_conv_bias_plus_six_residual_plus_four_snake hard_candidate_gate=max_below_same_work_global_min pytorch_k1_conv_only_six_call_median_ms={PYTORCH_K1_CONV_ONLY_SIX_CALL_MS:.6} conv_only_cross_scope_diagnostic=true aspirational_direct_fused_suite_target_ms={DIRECT_FUSED_SUITE_TARGET_MS:.3} required_full_codec_suite_saving_ms={REQUIRED_SAVING_MS:.3} rtx3060ti_peak_compute_floor_ms={compute_floor_ms:.3} absolute_saving_ceiling_ms={:.3} production_acceptance=full-decoder-all-steady-below-46.189ms-plus-strict-waveform-and-hash",
        CURRENT_POINTWISE_ESTIMATE_MS - compute_floor_ms,
    );
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.warmup > 0, "--warmup must be positive");
    ensure!(args.iterations > 0, "--iterations must be positive");
    ensure!(
        args.trials > 0 && args.trials % 2 == 1,
        "--trials must be positive and odd"
    );
    verify_sha256(&args.codec_weights, &args.codec_weights_sha256)?;
    print_static_accounting();

    let (device, monitor) = initialize_wgpu(args.adapter_index);
    let store = TensorStore::load(&args.codec_weights)
        .with_context(|| format!("load codec store {}", args.codec_weights.display()))?;
    let fixtures = CASES
        .iter()
        .map(|case| make_fixture(&store, &device, &monitor, *case))
        .collect::<Result<Vec<_>>>()?;
    synchronize_and_check_wgpu(&device, &monitor, "production fixture construction")?;
    println!(
        "weight_pack_summary calls={} aggregate_one_time_us={:.3} timing_excluded=true",
        fixtures.len(),
        fixtures
            .iter()
            .map(|fixture| fixture.one_time_pack_us)
            .sum::<f64>(),
    );
    validate_and_print_fixture_contracts(&fixtures)?;
    println!(
        "benchmark_protocol production_weights=true deterministic_inputs=true cases=6 variants=2 candidates=1 expected_contracts=6 warmup={} iterations={} trials={} expected_timing_samples={} rotating_variants=true full_output_screen=true production_direct_import=true",
        args.warmup,
        args.iterations,
        args.trials,
        args.trials * Variant::ALL.len(),
    );

    let reference = to_host(run_suite(Variant::CurrentProduction, &fixtures)?)?;
    synchronize_and_check_wgpu(&device, &monitor, "current correctness readback")?;
    for variant in Variant::CANDIDATES {
        let candidate = to_host(run_suite(variant, &fixtures)?)?;
        synchronize_and_check_wgpu(
            &device,
            &monitor,
            &format!("{} correctness readback", variant.label()),
        )?;
        compare_host_suite(variant, &reference, &candidate)?;
    }

    warm_all(&fixtures, &args, &device, &monitor)?;
    let mut samples = BTreeMap::<Variant, Vec<f64>>::new();
    for trial in 0..args.trials {
        for offset in 0..Variant::ALL.len() {
            let variant = Variant::ALL[(trial + offset) % Variant::ALL.len()];
            let sample = measure_variant(variant, &fixtures, &args, &device, &monitor)?;
            println!(
                "timing_sample trial={} rotation_position={} variant={} six_call_suite_us={:.3}",
                trial,
                offset,
                variant.label(),
                sample,
            );
            samples.entry(variant).or_default().push(sample);
        }
    }
    let summaries = Variant::ALL
        .into_iter()
        .map(|variant| (variant, summarize(&samples[&variant])))
        .collect::<BTreeMap<_, _>>();
    let baseline = summaries[&Variant::CurrentProduction];
    for variant in Variant::ALL {
        let timing = summaries[&variant];
        println!(
            "timing variant={} six_call_suite_median_us={:.3} range_us=[{:.3},{:.3}] speedup_vs_current={:.4} saving_ms={:.6}",
            variant.label(),
            timing.median_us,
            timing.min_us,
            timing.max_us,
            baseline.median_us / timing.median_us,
            (baseline.median_us - timing.median_us) / 1_000.0,
        );
    }
    let current_min_us = baseline.min_us;
    let mut accepted_variants = 0_usize;
    for variant in Variant::CANDIDATES {
        let timing = summaries[&variant];
        let below_current_min = timing.max_us < current_min_us;
        let aspirational_median_target = timing.median_us <= DIRECT_FUSED_SUITE_TARGET_MS * 1_000.0;
        let below_pytorch_conv_only_max =
            timing.max_us < PYTORCH_K1_CONV_ONLY_SIX_CALL_MS * 1_000.0;
        let below_pytorch_same_work_global_min =
            timing.max_us < PYTORCH_SAME_WORK_DEVICE_GLOBAL_MIN_MS * 1_000.0;
        let accepted = below_pytorch_same_work_global_min;
        accepted_variants += usize::from(accepted);
        println!(
            "timing_verdict variant={} candidate_max_us={:.3} current_min_us={:.3} candidate_max_below_current_min={} candidate_median_us={:.3} aspirational_candidate_median_le_3000us={} candidate_max_below_pytorch_conv_only_3155.006us={} conv_only_cross_scope_diagnostic_only=true pytorch_same_work_global_min_us={:.6} candidate_max_below_pytorch_same_work_global_min={} pytorch_same_work_scope=six_conv_bias_plus_six_residual_plus_four_snake candidate_scope=direct_six_call_suite_bias_residual_plus_four_next_snakes hard_acceptance_gate=same_work_global_min accepted={}",
            variant.label(),
            timing.max_us,
            current_min_us,
            below_current_min,
            timing.median_us,
            aspirational_median_target,
            below_pytorch_conv_only_max,
            PYTORCH_SAME_WORK_DEVICE_GLOBAL_MIN_MS * 1_000.0,
            below_pytorch_same_work_global_min,
            accepted,
        );
    }
    println!(
        "timing_acceptance_summary accepted_variants={} total_candidates=1 hard_requirement=production_direct_max_below_pytorch_same_work_device_global_min_8584.956665us conv_only_3155.006us_and_median_3000us_are_diagnostic_only",
        accepted_variants,
    );
    monitor.check("benchmark completion")?;
    println!("wgpu_errors=0");
    std::io::stdout()
        .flush()
        .context("flush completed benchmark report")?;
    ensure!(
        accepted_variants > 0,
        "timing acceptance failed: no candidate maximum beat the same-work PyTorch device-time global minimum"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_variant_is_fixed_and_covers_only_released_cases() {
        for case in CASES {
            let tile = Variant::ProductionDirect
                .tile(case)
                .expect("released case must use the production tile");
            assert_eq!(tile, PointwiseKTile::PRODUCTION);
            assert!(RELEASED_SHAPES.contains(&(case.channels, case.length)));
            assert!(case.channels.is_multiple_of(tile.reduction()));
            assert!(case.length.is_multiple_of(tile.time_tile()));
        }
    }
}
