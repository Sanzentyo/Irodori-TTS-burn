//! Isolated stage profiler for the exact production WGSL DACVAE decoder path.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use burn::{
    backend::wgpu::{
        AutoCompiler, MemoryConfiguration, RuntimeOptions, WgpuDevice, WgpuRuntime,
        graphics::AutoGraphicsApi, init_setup,
    },
    tensor::{FloatDType, Tensor, TensorData},
};
use clap::{Parser, ValueEnum};
use cubecl::prelude::Runtime;
use irodori_tts_burn::{
    backend_config::{
        WgpuFloatPrecision, configure_cubecl_persistent_cache_for_precision,
        configure_cubecl_persistent_cache_for_precision_with_record, default_cubecl_cache_root,
        wgpu_device_with_precision,
    },
    codec::{
        CodecAlgorithmPlan, CodecConvTransposeSnakeFusion, CodecCrossBlockFusion, CodecK7Algorithm,
        CodecPointwiseAlgorithm, CodecResidualStateLayout, CodecStageTiming, CodecStemAlgorithm,
        CodecTimingSource, PreparedK7WeightPolicy, load_codec,
    },
    validation::AudioMetrics,
};
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};

type WgpuRt = WgpuRuntime<AutoCompiler>;

#[derive(Debug, Parser)]
#[command(about = "Profile exact production WGSL codec stages from a precision oracle")]
struct Args {
    /// WGPU storage precision used by the codec and handwritten kernels.
    #[arg(long, value_enum, default_value = "fp32")]
    precision: WgpuFloatPrecision,

    /// Native dtype stored in the oracle. Defaults to `--precision`.
    /// This permits an F16 execution to be checked against an independently
    /// pinned F32 oracle without rewriting that source artifact.
    #[arg(long, value_enum)]
    fixture_precision: Option<WgpuFloatPrecision>,

    /// Precision-oracle fixture containing the exact final latent.
    #[arg(long)]
    fixture: PathBuf,

    /// Required out-of-band SHA-256 for the fixture.
    #[arg(long)]
    fixture_sha256: String,

    /// Rust-converted Semantic-DACVAE weights.
    #[arg(long)]
    codec_weights: PathBuf,

    /// Persistent CubeCL environment root. Defaults to
    /// `IRODORI_TTS_BURN_CACHE_DIR` or the platform cache directory.
    #[arg(long)]
    cubecl_cache_dir: Option<PathBuf>,

    /// Append fresh CubeCL autotune decisions as machine-readable JSONL.
    /// The path must end in `.json.log` and should live inside the campaign.
    #[arg(long)]
    autotune_record: Option<PathBuf>,

    /// Explicit WGPU discrete-adapter enumeration index.
    #[arg(long, default_value_t = 0)]
    adapter_index: usize,

    /// Maximum compute tasks aggregated into one WGPU command buffer.
    #[arg(long, default_value_t = 32)]
    tasks_max: usize,

    /// Untimed production-path warmups.
    #[arg(long, default_value_t = 2)]
    warmup: usize,

    /// Timed unchanged production-path repetitions.
    #[arg(long, default_value_t = 10)]
    repeats: usize,

    /// Timed stage profiling repetitions. Zero disables stage profiling.
    #[arg(long, default_value_t = 5)]
    profile_repeats: usize,

    /// Stage measurement method. Device timestamps avoid per-stage waits.
    #[arg(long, value_enum, default_value_t = StageProfileMethod::Device)]
    stage_profile_method: StageProfileMethod,

    /// k=7 implementation used by the timed decode and stage profiler.
    #[arg(long, value_enum, default_value_t = K7ProfileAlgorithm::Production)]
    k7_algorithm: K7ProfileAlgorithm,

    /// Pointwise implementation used by the timed decode and stage profiler.
    #[arg(long, value_enum, default_value_t = PointwiseProfileAlgorithm::Production)]
    pointwise_algorithm: PointwiseProfileAlgorithm,

    /// Decoder-stem implementation used by the timed decode and stage profiler.
    #[arg(long, value_enum, default_value_t = StemProfileAlgorithm::Production)]
    stem_algorithm: StemProfileAlgorithm,

    /// Decoder-block boundary implementation used by the timed decode.
    #[arg(long, value_enum, default_value_t = BlockBoundaryProfileAlgorithm::FusedC384AndC192)]
    block_boundary_algorithm: BlockBoundaryProfileAlgorithm,

    /// ConvTranspose finalizer to first-residual Snake boundary.
    #[arg(long, value_enum, default_value_t = ConvTransposeSnakeProfileAlgorithm::Standalone)]
    conv_transpose_snake_algorithm: ConvTransposeSnakeProfileAlgorithm,

    /// Physical shortcut layout retained within decoder residual blocks.
    #[arg(long, value_enum, default_value_t = ResidualStateProfileLayout::ProductionNcl)]
    residual_state_layout: ResidualStateProfileLayout,

    /// Run same-process ABBA/BAAB blocks comparing prepared single-storage k7
    /// weights against the request-time repack control.
    #[arg(long)]
    paired_single_storage: bool,

    /// Compare a same-model prepared OKI binding against request-time repack
    /// using alternating ABBA/BAAB blocks.
    #[arg(long)]
    paired_prepared_weight: bool,

    /// Compare the production geometry-selected multi-row k7 tiling against
    /// an explicit single-row control in alternating same-model blocks.
    #[arg(long)]
    paired_geometry_multi_rows: bool,

    /// Compare the restored per-shape CubeCL k7 autotune choice against the
    /// production geometry heuristic in alternating same-model blocks.
    #[arg(long)]
    paired_autotuned_multi_rows: bool,

    /// Compare model-prepared Snake reciprocals against per-output division
    /// with otherwise identical geometry-selected k7 routing.
    #[arg(long)]
    paired_prepared_epilogue: bool,

    /// Compare the accumulator-domain CubeK pointwise store transform against
    /// the existing packed-matmul pointwise route in alternating blocks.
    #[arg(long)]
    paired_pointwise_accumulator_store: bool,

    /// Isolate the four block-final residual stores by comparing all twelve
    /// accumulator stores against the already-adopted eight pair stores.
    #[arg(long)]
    paired_pointwise_residual_store: bool,

    /// Compare CubeK accumulator-domain activated-only cross-block stores
    /// against the adopted direct-WGSL cross-block producers.
    #[arg(long)]
    paired_cross_block_accumulator_store: bool,

    /// Compare the final pointwise-residual/WmHead fusion against production.
    #[arg(long)]
    paired_pointwise_head_fusion: bool,

    /// Profile only the twelve k=7 weight-layout materializations.
    #[arg(long)]
    profile_k7_weight_repack: bool,

    /// Compare WGPU process-local software-graph replay against normal
    /// fixed-shape graph construction in alternating ABBA/BAAB blocks.
    #[arg(long)]
    paired_software_graph: bool,

    /// Minimum physical weight bytes routed through prepared OKI during the
    /// same-model paired sweep. Zero selects all twelve weights.
    #[arg(long, default_value_t = 0)]
    prepared_k7_min_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum StageProfileMethod {
    #[default]
    Device,
    Synchronized,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum K7ProfileAlgorithm {
    #[default]
    Production,
    PackedResidue,
    ImplicitGemm,
    ImplicitGemmSingleStorage,
    ImplicitGemmPreparedWeight,
    ImplicitGemmDirectOik,
    ImplicitGemmK7Halo,
    ImplicitGemmMultiRows,
    ImplicitGemmGeometrySelectedMultiRows,
    ImplicitGemmAutotuned,
    ImplicitGemmPreparedEpilogue,
    ImplicitGemmInputLayoutFused,
    ImplicitGemmMaterialized,
    ImplicitGemmAsync,
    ImplicitGemmSyncStrided,
    ImplicitGemmAsyncStrided,
}

impl From<K7ProfileAlgorithm> for CodecK7Algorithm {
    fn from(value: K7ProfileAlgorithm) -> Self {
        match value {
            K7ProfileAlgorithm::Production => Self::AccuracyApproved,
            K7ProfileAlgorithm::PackedResidue => Self::PackedResidue,
            K7ProfileAlgorithm::ImplicitGemm => Self::CubeClImplicitGemm,
            K7ProfileAlgorithm::ImplicitGemmSingleStorage => Self::CubeClImplicitGemmSingleStorage,
            K7ProfileAlgorithm::ImplicitGemmPreparedWeight => {
                Self::CubeClImplicitGemmPreparedWeight(PreparedK7WeightPolicy::all())
            }
            K7ProfileAlgorithm::ImplicitGemmDirectOik => Self::CubeClImplicitGemmDirectOik,
            K7ProfileAlgorithm::ImplicitGemmK7Halo => Self::CubeClImplicitGemmK7Halo,
            K7ProfileAlgorithm::ImplicitGemmMultiRows => Self::CubeClImplicitGemmMultiRows,
            K7ProfileAlgorithm::ImplicitGemmGeometrySelectedMultiRows => {
                Self::CubeClImplicitGemmGeometrySelectedMultiRows
            }
            K7ProfileAlgorithm::ImplicitGemmAutotuned => Self::CubeClImplicitGemmAutotuned,
            K7ProfileAlgorithm::ImplicitGemmPreparedEpilogue => {
                Self::CubeClImplicitGemmPreparedEpilogue
            }
            K7ProfileAlgorithm::ImplicitGemmInputLayoutFused => {
                Self::CubeClImplicitGemmInputLayoutFused
            }
            K7ProfileAlgorithm::ImplicitGemmMaterialized => Self::CubeClImplicitGemmMaterialized,
            K7ProfileAlgorithm::ImplicitGemmAsync => Self::CubeClImplicitGemmAsync,
            K7ProfileAlgorithm::ImplicitGemmSyncStrided => Self::CubeClImplicitGemmSyncStrided,
            K7ProfileAlgorithm::ImplicitGemmAsyncStrided => Self::CubeClImplicitGemmAsyncStrided,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum PointwiseProfileAlgorithm {
    #[default]
    Production,
    PackedMatmul,
    ImplicitGemm,
    AccumulatorStore,
    AccumulatorPairOnly,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum StemProfileAlgorithm {
    #[default]
    Production,
    Burn,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum BlockBoundaryProfileAlgorithm {
    Standalone,
    FusedC384,
    FusedC192,
    #[default]
    FusedC384AndC192,
}

impl From<BlockBoundaryProfileAlgorithm> for CodecCrossBlockFusion {
    fn from(value: BlockBoundaryProfileAlgorithm) -> Self {
        match value {
            BlockBoundaryProfileAlgorithm::Standalone => Self::Standalone,
            BlockBoundaryProfileAlgorithm::FusedC384 => Self::OutputC384,
            BlockBoundaryProfileAlgorithm::FusedC192 => Self::OutputC192,
            BlockBoundaryProfileAlgorithm::FusedC384AndC192 => Self::OutputC384AndC192,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum ConvTransposeSnakeProfileAlgorithm {
    #[default]
    Standalone,
    CachedCol2ImCase1,
    CachedCol2ImCase2,
    CachedCol2ImCase3,
    CachedCol2ImDualOutput,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum ResidualStateProfileLayout {
    #[default]
    ProductionNcl,
    NhwcWithinBlock,
}

impl From<ResidualStateProfileLayout> for CodecResidualStateLayout {
    fn from(value: ResidualStateProfileLayout) -> Self {
        match value {
            ResidualStateProfileLayout::ProductionNcl => Self::ProductionNcl,
            ResidualStateProfileLayout::NhwcWithinBlock => Self::NhwcWithinBlock,
        }
    }
}

impl From<ConvTransposeSnakeProfileAlgorithm> for CodecConvTransposeSnakeFusion {
    fn from(value: ConvTransposeSnakeProfileAlgorithm) -> Self {
        match value {
            ConvTransposeSnakeProfileAlgorithm::Standalone => Self::Standalone,
            ConvTransposeSnakeProfileAlgorithm::CachedCol2ImCase1 => Self::CachedCol2ImCase1,
            ConvTransposeSnakeProfileAlgorithm::CachedCol2ImCase2 => Self::CachedCol2ImCase2,
            ConvTransposeSnakeProfileAlgorithm::CachedCol2ImCase3 => Self::CachedCol2ImCase3,
            ConvTransposeSnakeProfileAlgorithm::CachedCol2ImDualOutput => {
                Self::CachedCol2ImDualOutput
            }
        }
    }
}

impl From<StemProfileAlgorithm> for CodecStemAlgorithm {
    fn from(value: StemProfileAlgorithm) -> Self {
        match value {
            StemProfileAlgorithm::Production => Self::AccuracyApproved,
            StemProfileAlgorithm::Burn => Self::Burn,
        }
    }
}

impl From<PointwiseProfileAlgorithm> for CodecPointwiseAlgorithm {
    fn from(value: PointwiseProfileAlgorithm) -> Self {
        match value {
            PointwiseProfileAlgorithm::Production => Self::AccuracyApproved,
            PointwiseProfileAlgorithm::PackedMatmul => Self::PackedMatmul,
            PointwiseProfileAlgorithm::ImplicitGemm => Self::CubeClImplicitGemm,
            PointwiseProfileAlgorithm::AccumulatorStore => Self::CubeClAccumulatorStore,
            PointwiseProfileAlgorithm::AccumulatorPairOnly => Self::CubeClAccumulatorPairOnly,
        }
    }
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
        let errors = self
            .errors
            .lock()
            .map_err(|_| anyhow::anyhow!("WGPU error monitor lock poisoned after {stage}"))?;
        ensure!(errors.is_empty(), "WGPU errors after {stage}: {errors:?}");
        Ok(())
    }
}

fn initialize_wgpu(adapter_index: usize, tasks_max: usize) -> (WgpuDevice, WgpuErrorMonitor) {
    let device = WgpuDevice::DiscreteGpu(adapter_index);
    let setup = init_setup::<AutoGraphicsApi>(
        &device,
        RuntimeOptions {
            tasks_max,
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
        "wgpu_adapter: index={adapter_index} name={:?} backend={:?} device_type={:?} tasks_max={tasks_max} memory_config=sub-slices",
        info.name, info.backend, info.device_type,
    );
    (device, monitor)
}

fn synchronize_and_check_wgpu(
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    stage: &str,
) -> Result<()> {
    let client = WgpuRt::client(device);
    let sync_result = cubecl::future::block_on(client.sync());
    monitor.check(stage)?;
    sync_result.with_context(|| format!("CubeCL synchronization failed after {stage}"))
}

fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    ensure!(
        expected.len() == 64,
        "fixture SHA-256 must have 64 hex digits"
    );
    ensure!(
        expected.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "fixture SHA-256 contains a non-hex digit"
    );
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read fixture for hashing: {}", path.display()))?;
    let actual = format!("{:x}", Sha256::digest(bytes));
    ensure!(
        actual == expected.to_ascii_lowercase(),
        "fixture SHA-256 mismatch: got {actual}, expected {expected}"
    );
    println!("sha256: precision_fixture={actual} path={}", path.display());
    Ok(())
}

fn read_float_tensor(
    tensors: &SafeTensors<'_>,
    key: &str,
    precision: WgpuFloatPrecision,
) -> Result<(Vec<usize>, Vec<f32>)> {
    let view = tensors
        .tensor(key)
        .with_context(|| format!("fixture tensor {key:?} is missing"))?;
    let expected_dtype = match precision {
        WgpuFloatPrecision::Fp32 => Dtype::F32,
        WgpuFloatPrecision::Fp16 => Dtype::F16,
    };
    ensure!(
        view.dtype() == expected_dtype,
        "fixture tensor {key:?} has dtype {:?}, expected {expected_dtype:?}",
        view.dtype()
    );
    let shape = view.shape().to_vec();
    let values = match precision {
        WgpuFloatPrecision::Fp32 => view
            .data()
            .chunks_exact(size_of::<f32>())
            .map(|chunk| {
                let bytes: [u8; size_of::<f32>()] = chunk
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("invalid f32 bytes in {key:?}"))?;
                Ok(f32::from_le_bytes(bytes))
            })
            .collect::<Result<Vec<_>>>()?,
        WgpuFloatPrecision::Fp16 => view
            .data()
            .chunks_exact(size_of::<half::f16>())
            .map(|chunk| {
                let bytes: [u8; size_of::<half::f16>()] = chunk
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("invalid f16 bytes in {key:?}"))?;
                Ok(half::f16::from_le_bytes(bytes).to_f32())
            })
            .collect::<Result<Vec<_>>>()?,
    };
    Ok((shape, values))
}

fn load_oracle_tensors(
    path: &Path,
    precision: WgpuFloatPrecision,
) -> Result<(Vec<f32>, usize, Vec<f32>)> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read precision fixture {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("malformed precision fixture {}", path.display()))?;
    let (latent_shape, latent) = read_float_tensor(&tensors, "final_patched_latent", precision)?;
    let (waveform_shape, waveform) =
        read_float_tensor(&tensors, "raw_decoded_waveform", precision)?;
    ensure!(
        latent_shape.len() == 3
            && latent_shape[0] == 1
            && latent_shape[1] > 0
            && latent_shape[2] == 32,
        "final_patched_latent shape {latent_shape:?} must be [1, positive_steps, 32]"
    );
    ensure!(
        waveform_shape.len() == 2 && waveform_shape[0] == 1 && waveform_shape[1] > 0,
        "raw_decoded_waveform shape {waveform_shape:?} must be [1, positive_samples]"
    );
    ensure!(
        latent
            .iter()
            .chain(&waveform)
            .all(|value| value.is_finite()),
        "oracle tensors contain non-finite values"
    );
    Ok((latent, latent_shape[1], waveform))
}

fn sha256_f32_le(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    values
        .iter()
        .for_each(|value| hasher.update(value.to_bits().to_le_bytes()));
    format!("{:x}", hasher.finalize())
}

fn waveform_gate(
    reference: &[f32],
    actual: &[f32],
    label: &str,
    precision: WgpuFloatPrecision,
) -> Result<()> {
    let metrics = AudioMetrics::compare(reference, actual)?;
    println!(
        "{label}: count={} max_abs={:.9e} mean_abs={:.9e} rmse={:.9e} snr_db={:.6} cosine={:.12}",
        metrics.sample_count,
        metrics.max_abs_error,
        metrics.mean_abs_error,
        metrics.root_mean_square_error,
        metrics.signal_to_noise_db,
        metrics.cosine_similarity
    );
    let (max_abs, mean_abs, rmse, snr, cosine) = match precision {
        WgpuFloatPrecision::Fp32 => (0.00015, 0.000005, 0.00001, 85.0, 0.99999999),
        WgpuFloatPrecision::Fp16 => (0.005, 0.0005, 0.001, 50.0, 0.99999),
    };
    ensure!(
        metrics.max_abs_error <= max_abs,
        "{label} max_abs gate failed"
    );
    ensure!(
        metrics.mean_abs_error <= mean_abs,
        "{label} mean_abs gate failed"
    );
    ensure!(
        metrics.root_mean_square_error <= rmse,
        "{label} RMSE gate failed"
    );
    ensure!(metrics.signal_to_noise_db >= snr, "{label} SNR gate failed");
    ensure!(
        metrics.cosine_similarity >= cosine,
        "{label} cosine gate failed"
    );
    Ok(())
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    } else {
        sorted[middle]
    }
}

fn print_summary(label: &str, values_ms: &[f64]) {
    if values_ms.is_empty() {
        return;
    }
    let minimum = values_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = values_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "timing_summary stage={label} median_ms={:.6} range_ms=[{minimum:.6},{maximum:.6}] samples={}",
        median(values_ms),
        values_ms.len()
    );
}

#[allow(clippy::too_many_arguments)]
fn run_paired_single_storage(
    prepared: &irodori_tts_burn::codec::DacVaeCodec,
    repack: &irodori_tts_burn::codec::DacVaeCodec,
    latent: &Tensor<3>,
    expected_waveform: &[f32],
    precision: WgpuFloatPrecision,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    warmup: usize,
    blocks: usize,
) -> Result<()> {
    let repack_plan = CodecAlgorithmPlan::new(
        CodecK7Algorithm::CubeClImplicitGemmSingleStorage,
        CodecPointwiseAlgorithm::AccuracyApproved,
    );
    for repetition in 1..=warmup {
        drop(prepared.decode_wgsl(latent.clone()));
        drop(repack.decode_wgsl_with_plan(latent.clone(), repack_plan));
        synchronize_and_check_wgpu(device, monitor, &format!("paired warmup {repetition}"))?;
    }

    let mut prepared_device = Vec::with_capacity(blocks * 4);
    let mut prepared_readback = Vec::with_capacity(blocks * 4);
    let mut repack_device = Vec::with_capacity(blocks * 4);
    let mut repack_readback = Vec::with_capacity(blocks * 4);
    let mut prepared_hash = None;
    let mut repack_hash = None;

    for block in 1..=blocks {
        let order = if block % 2 == 1 {
            [true, false, false, true]
        } else {
            [false, true, true, false]
        };
        for (slot, is_prepared) in order.into_iter().enumerate() {
            synchronize_and_check_wgpu(device, monitor, "paired pre-start")?;
            let started = Instant::now();
            let output = if is_prepared {
                prepared.decode_wgsl(latent.clone())
            } else {
                repack.decode_wgsl_with_plan(latent.clone(), repack_plan)
            };
            synchronize_and_check_wgpu(device, monitor, "paired device completion")?;
            let device_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let values = output
                .cast(FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .context("failed paired readback")?;
            synchronize_and_check_wgpu(device, monitor, "paired readback completion")?;
            let readback_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let hash = sha256_f32_le(&values);
            waveform_gate(
                expected_waveform,
                &values,
                if is_prepared {
                    "paired_single_storage"
                } else {
                    "paired_request_repack"
                },
                precision,
            )?;
            let stable_hash = if is_prepared {
                &mut prepared_hash
            } else {
                &mut repack_hash
            };
            if let Some(expected) = stable_hash.as_ref() {
                ensure!(
                    &hash == expected,
                    "paired route output was nondeterministic"
                );
            } else {
                *stable_hash = Some(hash.clone());
            }
            if is_prepared {
                prepared_device.push(device_ms);
                prepared_readback.push(readback_ms);
            } else {
                repack_device.push(device_ms);
                repack_readback.push(readback_ms);
            }
            println!(
                "paired_sample block={block}/{blocks} slot={} route={} device_complete_ms={device_ms:.6} readback_complete_ms={readback_ms:.6} sha256={hash}",
                slot + 1,
                if is_prepared {
                    "single-storage"
                } else {
                    "request-repack"
                }
            );
        }
    }
    print_summary("paired_single_storage_device_complete", &prepared_device);
    print_summary(
        "paired_single_storage_readback_complete",
        &prepared_readback,
    );
    print_summary("paired_request_repack_device_complete", &repack_device);
    print_summary("paired_request_repack_readback_complete", &repack_readback);
    println!(
        "paired_hashes single_storage={} request_repack={} bitwise_equal={}",
        prepared_hash.as_deref().unwrap_or("missing"),
        repack_hash.as_deref().unwrap_or("missing"),
        prepared_hash == repack_hash
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_paired_prepared_weight(
    codec: &irodori_tts_burn::codec::DacVaeCodec,
    latent: &Tensor<3>,
    expected_waveform: &[f32],
    precision: WgpuFloatPrecision,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    warmup: usize,
    blocks: usize,
    prepared_k7_min_bytes: usize,
) -> Result<()> {
    let prepared_plan = CodecAlgorithmPlan::new(
        CodecK7Algorithm::CubeClImplicitGemmPreparedWeight(PreparedK7WeightPolicy::at_least_bytes(
            prepared_k7_min_bytes,
        )),
        CodecPointwiseAlgorithm::AccuracyApproved,
    );
    for repetition in 1..=warmup {
        drop(codec.decode_wgsl_with_plan(latent.clone(), prepared_plan));
        drop(codec.decode_wgsl(latent.clone()));
        synchronize_and_check_wgpu(device, monitor, &format!("paired warmup {repetition}"))?;
    }

    let mut prepared_device = Vec::with_capacity(blocks * 2);
    let mut prepared_readback = Vec::with_capacity(blocks * 2);
    let mut repack_device = Vec::with_capacity(blocks * 2);
    let mut repack_readback = Vec::with_capacity(blocks * 2);
    let mut prepared_hash = None;
    let mut repack_hash = None;

    for block in 1..=blocks {
        let order = if block % 2 == 1 {
            [true, false, false, true]
        } else {
            [false, true, true, false]
        };
        for (slot, is_prepared) in order.into_iter().enumerate() {
            synchronize_and_check_wgpu(device, monitor, "paired pre-start")?;
            let started = Instant::now();
            let output = if is_prepared {
                codec.decode_wgsl_with_plan(latent.clone(), prepared_plan)
            } else {
                codec.decode_wgsl(latent.clone())
            };
            synchronize_and_check_wgpu(device, monitor, "paired device completion")?;
            let device_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let values = output
                .cast(FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .context("failed paired readback")?;
            synchronize_and_check_wgpu(device, monitor, "paired readback completion")?;
            let readback_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let hash = sha256_f32_le(&values);
            waveform_gate(
                expected_waveform,
                &values,
                if is_prepared {
                    "paired_prepared_weight"
                } else {
                    "paired_request_repack"
                },
                precision,
            )?;
            let stable_hash = if is_prepared {
                &mut prepared_hash
            } else {
                &mut repack_hash
            };
            if let Some(expected) = stable_hash.as_ref() {
                ensure!(
                    &hash == expected,
                    "paired route output was nondeterministic"
                );
            } else {
                *stable_hash = Some(hash.clone());
            }
            let (device_samples, readback_samples) = if is_prepared {
                (&mut prepared_device, &mut prepared_readback)
            } else {
                (&mut repack_device, &mut repack_readback)
            };
            device_samples.push(device_ms);
            readback_samples.push(readback_ms);
            println!(
                "paired_sample block={block}/{blocks} slot={} route={} device_complete_ms={device_ms:.6} readback_complete_ms={readback_ms:.6} sha256={hash}",
                slot + 1,
                if is_prepared {
                    "prepared-oki"
                } else {
                    "request-repack"
                }
            );
        }
    }
    print_summary("paired_prepared_oki_device_complete", &prepared_device);
    print_summary("paired_prepared_oki_readback_complete", &prepared_readback);
    print_summary("paired_request_repack_device_complete", &repack_device);
    print_summary("paired_request_repack_readback_complete", &repack_readback);
    println!(
        "paired_hashes prepared_oki={} request_repack={} bitwise_equal={}",
        prepared_hash.as_deref().unwrap_or("missing"),
        repack_hash.as_deref().unwrap_or("missing"),
        prepared_hash == repack_hash
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_paired_software_graph(
    codec: &irodori_tts_burn::codec::DacVaeCodec,
    latent: &Tensor<3>,
    expected_waveform: &[f32],
    precision: WgpuFloatPrecision,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    warmup: usize,
    blocks: usize,
) -> Result<()> {
    let primitive = latent
        .clone()
        .try_into_primitive::<irodori_tts_burn::WgpuRaw>()
        .map_err(|_| anyhow::anyhow!("software graph requires a WGPU latent"))?;
    let client = primitive.client;
    let stable_latent = client
        .memory_persistent_allocation((), |()| Tensor::<3>::zeros(latent.dims(), &latent.device()))
        .context("failed to allocate stable software graph input")?;
    copy_software_graph_input(latent, &stable_latent)?;
    let before_capture = client
        .memory_usage()
        .context("failed to query pre-capture WGPU memory")?;

    client
        .graph_prepare()
        .context("failed to prepare WGPU software graph memory")?;
    synchronize_and_check_wgpu(device, monitor, "software graph input initialization")?;
    let priming_output = codec
        .decode_wgsl(stable_latent.clone())
        .cast(FloatDType::F32);
    synchronize_and_check_wgpu(device, monitor, "software graph priming")?;
    drop(priming_output);
    client
        .start_capture()
        .context("failed to start WGPU software graph capture")?;
    let captured_output = codec
        .decode_wgsl(stable_latent.clone())
        .cast(FloatDType::F32);
    let graph = client
        .stop_capture()
        .context("failed to stop WGPU software graph capture")?;
    let after_capture = client
        .memory_usage()
        .context("failed to query post-capture WGPU memory")?;
    println!(
        "software_graph_memory before_in_use_bytes={} before_reserved_bytes={} after_in_use_bytes={} after_reserved_bytes={} delta_in_use_bytes={} delta_reserved_bytes={}",
        before_capture.bytes_in_use,
        before_capture.bytes_reserved,
        after_capture.bytes_in_use,
        after_capture.bytes_reserved,
        after_capture
            .bytes_in_use
            .saturating_sub(before_capture.bytes_in_use),
        after_capture
            .bytes_reserved
            .saturating_sub(before_capture.bytes_reserved),
    );
    // A graph owns its complete scratch arena. Release now-unused pages from
    // the ordinary allocator before admitting traffic; keeping both scratch
    // sets would overstate the production resident configuration.
    client.memory_cleanup();
    synchronize_and_check_wgpu(device, monitor, "software graph main-pool cleanup")?;
    let after_cleanup = client
        .memory_usage()
        .context("failed to query post-cleanup WGPU memory")?;
    println!(
        "software_graph_post_cleanup in_use_bytes={} reserved_bytes={} delta_in_use_from_before_bytes={} delta_reserved_from_before_bytes={}",
        after_cleanup.bytes_in_use,
        after_cleanup.bytes_reserved,
        after_cleanup
            .bytes_in_use
            .saturating_sub(before_capture.bytes_in_use),
        after_cleanup
            .bytes_reserved
            .saturating_sub(before_capture.bytes_reserved),
    );

    for repetition in 1..=warmup {
        copy_software_graph_input(latent, &stable_latent)?;
        // SAFETY: model, latent, captured output, and the graph-owned
        // intermediate allocations remain live; all work uses this one stream.
        unsafe { graph.replay() };
        synchronize_and_check_wgpu(device, monitor, &format!("graph warmup {repetition}"))?;
        drop(codec.decode_wgsl(latent.clone()).cast(FloatDType::F32));
        synchronize_and_check_wgpu(device, monitor, &format!("control warmup {repetition}"))?;
    }

    let mut graph_enqueue = Vec::with_capacity(blocks * 2);
    let mut graph_device = Vec::with_capacity(blocks * 2);
    let mut graph_readback = Vec::with_capacity(blocks * 2);
    let mut control_enqueue = Vec::with_capacity(blocks * 2);
    let mut control_device = Vec::with_capacity(blocks * 2);
    let mut control_readback = Vec::with_capacity(blocks * 2);
    let mut block_device_deltas = Vec::with_capacity(blocks);
    let mut graph_hash = None;
    let mut control_hash = None;

    for block in 1..=blocks {
        let order = if block % 2 == 1 {
            [true, false, false, true]
        } else {
            [false, true, true, false]
        };
        for (slot, is_graph) in order.into_iter().enumerate() {
            synchronize_and_check_wgpu(device, monitor, "software graph paired pre-start")?;
            let started = Instant::now();
            let output = if is_graph {
                copy_software_graph_input(latent, &stable_latent)?;
                // SAFETY: the captured buffers are held by `graph`, while the
                // model, latent and captured output outlive every replay.
                unsafe { graph.replay() };
                captured_output.clone()
            } else {
                codec.decode_wgsl(latent.clone()).cast(FloatDType::F32)
            };
            let enqueue_ms = started.elapsed().as_secs_f64() * 1_000.0;
            synchronize_and_check_wgpu(device, monitor, "software graph device completion")?;
            let device_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let values = output
                .into_data()
                .to_vec::<f32>()
                .context("failed software graph paired readback")?;
            synchronize_and_check_wgpu(device, monitor, "software graph readback completion")?;
            let readback_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let hash = sha256_f32_le(&values);
            waveform_gate(
                expected_waveform,
                &values,
                if is_graph {
                    "paired_software_graph"
                } else {
                    "paired_normal_graph"
                },
                precision,
            )?;
            let stable_hash = if is_graph {
                &mut graph_hash
            } else {
                &mut control_hash
            };
            if let Some(expected) = stable_hash.as_ref() {
                ensure!(
                    &hash == expected,
                    "paired graph output was nondeterministic"
                );
            } else {
                *stable_hash = Some(hash.clone());
            }
            let (enqueue_samples, device_samples, readback_samples) = if is_graph {
                (&mut graph_enqueue, &mut graph_device, &mut graph_readback)
            } else {
                (
                    &mut control_enqueue,
                    &mut control_device,
                    &mut control_readback,
                )
            };
            enqueue_samples.push(enqueue_ms);
            device_samples.push(device_ms);
            readback_samples.push(readback_ms);
            println!(
                "paired_sample block={block}/{blocks} slot={} route={} enqueue_complete_ms={enqueue_ms:.6} device_complete_ms={device_ms:.6} readback_complete_ms={readback_ms:.6} sha256={hash}",
                slot + 1,
                if is_graph {
                    "software-graph"
                } else {
                    "normal-graph"
                }
            );
        }
        let graph_mean =
            (graph_device[graph_device.len() - 2] + graph_device[graph_device.len() - 1]) * 0.5;
        let control_mean = (control_device[control_device.len() - 2]
            + control_device[control_device.len() - 1])
            * 0.5;
        let delta = graph_mean - control_mean;
        block_device_deltas.push(delta);
        println!(
            "paired_block_summary block={block}/{blocks} software_graph_minus_normal_graph_device_ms={delta:.6}"
        );
    }
    print_summary("paired_software_graph_enqueue_complete", &graph_enqueue);
    print_summary("paired_software_graph_device_complete", &graph_device);
    print_summary("paired_software_graph_readback_complete", &graph_readback);
    print_summary("paired_normal_graph_enqueue_complete", &control_enqueue);
    print_summary("paired_normal_graph_device_complete", &control_device);
    print_summary("paired_normal_graph_readback_complete", &control_readback);
    print_summary(
        "paired_block_software_graph_minus_normal_graph_device",
        &block_device_deltas,
    );
    println!(
        "paired_hashes software_graph={} normal_graph={} bitwise_equal={}",
        graph_hash.as_deref().unwrap_or("missing"),
        control_hash.as_deref().unwrap_or("missing"),
        graph_hash == control_hash
    );
    Ok(())
}

fn copy_software_graph_input(source: &Tensor<3>, destination: &Tensor<3>) -> Result<()> {
    let source = source
        .clone()
        .try_into_primitive::<irodori_tts_burn::WgpuRaw>()
        .map_err(|_| anyhow::anyhow!("software graph source must use WGPU"))?;
    let destination = destination
        .clone()
        .try_into_primitive::<irodori_tts_burn::WgpuRaw>()
        .map_err(|_| anyhow::anyhow!("software graph destination must use WGPU"))?;
    irodori_tts_burn::kernels::contiguous_copy::copy_contiguous_into_wgsl(source, destination)
        .map_err(anyhow::Error::msg)
}

#[allow(clippy::too_many_arguments)]
fn run_paired_geometry_multi_rows(
    codec: &irodori_tts_burn::codec::DacVaeCodec,
    latent: &Tensor<3>,
    expected_waveform: &[f32],
    precision: WgpuFloatPrecision,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    warmup: usize,
    blocks: usize,
) -> Result<()> {
    let geometry_plan = CodecAlgorithmPlan::accuracy_approved();
    let control_plan = CodecAlgorithmPlan::new(
        CodecK7Algorithm::CubeClImplicitGemm,
        CodecPointwiseAlgorithm::AccuracyApproved,
    );
    for repetition in 1..=warmup {
        drop(codec.decode_wgsl_with_plan(latent.clone(), geometry_plan));
        drop(codec.decode_wgsl_with_plan(latent.clone(), control_plan));
        synchronize_and_check_wgpu(device, monitor, &format!("paired warmup {repetition}"))?;
    }

    let mut geometry_device = Vec::with_capacity(blocks * 2);
    let mut geometry_readback = Vec::with_capacity(blocks * 2);
    let mut control_device = Vec::with_capacity(blocks * 2);
    let mut control_readback = Vec::with_capacity(blocks * 2);
    let mut geometry_selected_k7 = Vec::with_capacity(blocks * 2);
    let mut geometry_all_k7 = Vec::with_capacity(blocks * 2);
    let mut geometry_all_stages = Vec::with_capacity(blocks * 2);
    let mut control_selected_k7 = Vec::with_capacity(blocks * 2);
    let mut control_all_k7 = Vec::with_capacity(blocks * 2);
    let mut control_all_stages = Vec::with_capacity(blocks * 2);
    let mut geometry_hash = None;
    let mut control_hash = None;

    for block in 1..=blocks {
        let order = if block % 2 == 1 {
            [true, false, false, true]
        } else {
            [false, true, true, false]
        };
        for (slot, is_geometry) in order.into_iter().enumerate() {
            synchronize_and_check_wgpu(device, monitor, "paired pre-start")?;
            let started = Instant::now();
            let (output, timings) = codec.decode_wgsl_device_profiled_with_plan(
                latent.clone(),
                if is_geometry {
                    geometry_plan
                } else {
                    control_plan
                },
            )?;
            synchronize_and_check_wgpu(device, monitor, "paired device completion")?;
            let device_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let (selected_k7_ms, all_k7_ms, all_stages_ms) = summarize_k7_timings(&timings)?;
            let values = output
                .cast(FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .context("failed paired readback")?;
            synchronize_and_check_wgpu(device, monitor, "paired readback completion")?;
            let readback_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let hash = sha256_f32_le(&values);
            waveform_gate(
                expected_waveform,
                &values,
                if is_geometry {
                    "paired_geometry_multi_rows"
                } else {
                    "paired_single_row_control"
                },
                precision,
            )?;
            let stable_hash = if is_geometry {
                &mut geometry_hash
            } else {
                &mut control_hash
            };
            if let Some(expected) = stable_hash.as_ref() {
                ensure!(
                    &hash == expected,
                    "paired route output was nondeterministic"
                );
            } else {
                *stable_hash = Some(hash.clone());
            }
            let (device_samples, readback_samples) = if is_geometry {
                (&mut geometry_device, &mut geometry_readback)
            } else {
                (&mut control_device, &mut control_readback)
            };
            device_samples.push(device_ms);
            readback_samples.push(readback_ms);
            let (selected_k7_samples, all_k7_samples, all_stage_samples) = if is_geometry {
                (
                    &mut geometry_selected_k7,
                    &mut geometry_all_k7,
                    &mut geometry_all_stages,
                )
            } else {
                (
                    &mut control_selected_k7,
                    &mut control_all_k7,
                    &mut control_all_stages,
                )
            };
            selected_k7_samples.push(selected_k7_ms);
            all_k7_samples.push(all_k7_ms);
            all_stage_samples.push(all_stages_ms);
            println!(
                "paired_sample block={block}/{blocks} slot={} route={} device_complete_ms={device_ms:.6} readback_complete_ms={readback_ms:.6} selected_k7_device_ms={selected_k7_ms:.6} all_k7_device_ms={all_k7_ms:.6} all_stages_device_ms={all_stages_ms:.6} sha256={hash}",
                slot + 1,
                if is_geometry {
                    "geometry-multi-rows"
                } else {
                    "single-row-control"
                }
            );
        }
    }
    print_summary(
        "paired_geometry_multi_rows_device_complete",
        &geometry_device,
    );
    print_summary(
        "paired_geometry_multi_rows_readback_complete",
        &geometry_readback,
    );
    print_summary("paired_single_row_control_device_complete", &control_device);
    print_summary(
        "paired_single_row_control_readback_complete",
        &control_readback,
    );
    print_summary(
        "paired_geometry_multi_rows_selected_k7_device",
        &geometry_selected_k7,
    );
    print_summary(
        "paired_single_row_control_selected_k7_device",
        &control_selected_k7,
    );
    print_summary("paired_geometry_multi_rows_all_k7_device", &geometry_all_k7);
    print_summary("paired_single_row_control_all_k7_device", &control_all_k7);
    print_summary(
        "paired_geometry_multi_rows_all_stages_device",
        &geometry_all_stages,
    );
    print_summary(
        "paired_single_row_control_all_stages_device",
        &control_all_stages,
    );
    println!(
        "paired_hashes geometry_multi_rows={} single_row_control={} bitwise_equal={}",
        geometry_hash.as_deref().unwrap_or("missing"),
        control_hash.as_deref().unwrap_or("missing"),
        geometry_hash == control_hash
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_paired_k7_plans(
    codec: &irodori_tts_burn::codec::DacVaeCodec,
    latent: &Tensor<3>,
    expected_waveform: &[f32],
    precision: WgpuFloatPrecision,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    warmup: usize,
    blocks: usize,
    candidate_plan: CodecAlgorithmPlan,
    control_plan: CodecAlgorithmPlan,
    candidate_label: &'static str,
    control_label: &'static str,
) -> Result<()> {
    for repetition in 1..=warmup {
        drop(codec.decode_wgsl_with_plan(latent.clone(), candidate_plan));
        drop(codec.decode_wgsl_with_plan(latent.clone(), control_plan));
        synchronize_and_check_wgpu(device, monitor, &format!("paired warmup {repetition}"))?;
    }

    let mut candidate_device = Vec::with_capacity(blocks * 2);
    let mut candidate_readback = Vec::with_capacity(blocks * 2);
    let mut control_device = Vec::with_capacity(blocks * 2);
    let mut control_readback = Vec::with_capacity(blocks * 2);
    let mut candidate_hash = None;
    let mut control_hash = None;
    let mut block_device_deltas = Vec::with_capacity(blocks);

    for block in 1..=blocks {
        let order = if block % 2 == 1 {
            [true, false, false, true]
        } else {
            [false, true, true, false]
        };
        for (slot, is_candidate) in order.into_iter().enumerate() {
            synchronize_and_check_wgpu(device, monitor, "paired pre-start")?;
            let started = Instant::now();
            let output = codec.decode_wgsl_with_plan(
                latent.clone(),
                if is_candidate {
                    candidate_plan
                } else {
                    control_plan
                },
            );
            synchronize_and_check_wgpu(device, monitor, "paired device completion")?;
            let device_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let values = output
                .cast(FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .context("failed paired k7-plan readback")?;
            synchronize_and_check_wgpu(device, monitor, "paired readback completion")?;
            let readback_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let hash = sha256_f32_le(&values);
            let label = if is_candidate {
                candidate_label
            } else {
                control_label
            };
            waveform_gate(expected_waveform, &values, label, precision)?;
            let stable_hash = if is_candidate {
                &mut candidate_hash
            } else {
                &mut control_hash
            };
            if let Some(expected) = stable_hash.as_ref() {
                ensure!(
                    &hash == expected,
                    "paired route output was nondeterministic"
                );
            } else {
                *stable_hash = Some(hash.clone());
            }
            if is_candidate {
                candidate_device.push(device_ms);
                candidate_readback.push(readback_ms);
            } else {
                control_device.push(device_ms);
                control_readback.push(readback_ms);
            }
            println!(
                "paired_sample block={block}/{blocks} slot={} route={label} device_complete_ms={device_ms:.6} readback_complete_ms={readback_ms:.6} sha256={hash}",
                slot + 1
            );
        }
        let candidate_mean = (candidate_device[candidate_device.len() - 2]
            + candidate_device[candidate_device.len() - 1])
            * 0.5;
        let control_mean = (control_device[control_device.len() - 2]
            + control_device[control_device.len() - 1])
            * 0.5;
        let delta = candidate_mean - control_mean;
        block_device_deltas.push(delta);
        println!(
            "paired_block_summary block={block}/{blocks} {candidate_label}_minus_{control_label}_device_ms={delta:.6}"
        );
    }
    print_summary(
        &format!("paired_{candidate_label}_device_complete"),
        &candidate_device,
    );
    print_summary(
        &format!("paired_{candidate_label}_readback_complete"),
        &candidate_readback,
    );
    print_summary(
        &format!("paired_{control_label}_device_complete"),
        &control_device,
    );
    print_summary(
        &format!("paired_{control_label}_readback_complete"),
        &control_readback,
    );
    print_summary(
        &format!("paired_block_{candidate_label}_minus_{control_label}_device"),
        &block_device_deltas,
    );
    println!(
        "paired_improvement candidate_label={candidate_label} improved_blocks={}/{}",
        block_device_deltas
            .iter()
            .filter(|delta| **delta < 0.0)
            .count(),
        block_device_deltas.len(),
    );
    println!(
        "paired_hashes candidate_label={candidate_label} candidate={} control_label={control_label} control={} bitwise_equal={}",
        candidate_hash.as_deref().unwrap_or("missing"),
        control_hash.as_deref().unwrap_or("missing"),
        candidate_hash == control_hash
    );
    Ok(())
}

#[derive(Clone, Copy)]
enum PairedTailCandidate {
    CrossBlockAccumulator,
    PointwiseHeadFusion,
}

impl PairedTailCandidate {
    fn label(self) -> &'static str {
        match self {
            Self::CrossBlockAccumulator => "cross-block-accumulator",
            Self::PointwiseHeadFusion => "pointwise-head-fusion",
        }
    }

    fn decode(
        self,
        codec: &irodori_tts_burn::codec::DacVaeCodec,
        latent: Tensor<3>,
    ) -> Result<Tensor<3>> {
        match self {
            Self::CrossBlockAccumulator => Ok(codec.decode_wgsl_cross_block_accumulator(latent)?),
            Self::PointwiseHeadFusion => Ok(codec.decode_wgsl_pointwise_head_fused(latent)?),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_paired_tail_candidate(
    codec: &irodori_tts_burn::codec::DacVaeCodec,
    latent: &Tensor<3>,
    expected_waveform: &[f32],
    precision: WgpuFloatPrecision,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    warmup: usize,
    blocks: usize,
    candidate: PairedTailCandidate,
) -> Result<()> {
    let candidate_label = candidate.label();
    for repetition in 1..=warmup {
        drop(candidate.decode(codec, latent.clone())?);
        drop(codec.decode_wgsl(latent.clone()));
        synchronize_and_check_wgpu(
            device,
            monitor,
            &format!("paired cross-block warmup {repetition}"),
        )?;
    }

    let mut candidate_device = Vec::with_capacity(blocks * 2);
    let mut candidate_readback = Vec::with_capacity(blocks * 2);
    let mut control_device = Vec::with_capacity(blocks * 2);
    let mut control_readback = Vec::with_capacity(blocks * 2);
    let mut candidate_hash = None;
    let mut control_hash = None;
    let mut block_device_deltas = Vec::with_capacity(blocks);
    for block in 1..=blocks {
        let order = if block % 2 == 1 {
            [true, false, false, true]
        } else {
            [false, true, true, false]
        };
        for (slot, is_candidate) in order.into_iter().enumerate() {
            synchronize_and_check_wgpu(device, monitor, "paired cross-block pre-start")?;
            let started = Instant::now();
            let output = if is_candidate {
                candidate.decode(codec, latent.clone())?
            } else {
                codec.decode_wgsl(latent.clone())
            };
            synchronize_and_check_wgpu(device, monitor, "paired cross-block device completion")?;
            let device_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let values = output
                .cast(FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .context("failed cross-block accumulator readback")?;
            synchronize_and_check_wgpu(device, monitor, "paired cross-block readback completion")?;
            let readback_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let hash = sha256_f32_le(&values);
            let label = if is_candidate {
                candidate_label
            } else {
                "production"
            };
            waveform_gate(expected_waveform, &values, label, precision)?;
            let stable_hash = if is_candidate {
                &mut candidate_hash
            } else {
                &mut control_hash
            };
            if let Some(expected) = stable_hash.as_ref() {
                ensure!(
                    &hash == expected,
                    "paired route output was nondeterministic"
                );
            } else {
                *stable_hash = Some(hash.clone());
            }
            if is_candidate {
                candidate_device.push(device_ms);
                candidate_readback.push(readback_ms);
            } else {
                control_device.push(device_ms);
                control_readback.push(readback_ms);
            }
            println!(
                "paired_sample block={block}/{blocks} slot={} route={label} device_complete_ms={device_ms:.6} readback_complete_ms={readback_ms:.6} sha256={hash}",
                slot + 1
            );
        }
        let candidate_mean = (candidate_device[candidate_device.len() - 2]
            + candidate_device[candidate_device.len() - 1])
            * 0.5;
        let control_mean = (control_device[control_device.len() - 2]
            + control_device[control_device.len() - 1])
            * 0.5;
        let delta = candidate_mean - control_mean;
        block_device_deltas.push(delta);
        println!(
            "paired_block_summary block={block}/{blocks} {candidate_label}_minus_production_device_ms={delta:.6}"
        );
    }
    print_summary(
        &format!("paired_{candidate_label}_device_complete"),
        &candidate_device,
    );
    print_summary(
        &format!("paired_{candidate_label}_readback_complete"),
        &candidate_readback,
    );
    print_summary("paired_production_device_complete", &control_device);
    print_summary("paired_production_readback_complete", &control_readback);
    print_summary(
        &format!("paired_block_{candidate_label}_minus_production_device"),
        &block_device_deltas,
    );
    println!(
        "paired_improvement candidate_label={candidate_label} improved_blocks={}/{}",
        block_device_deltas
            .iter()
            .filter(|delta| **delta < 0.0)
            .count(),
        block_device_deltas.len(),
    );
    println!(
        "paired_hashes candidate={} control={} bitwise_equal={}",
        candidate_hash.as_deref().unwrap_or("missing"),
        control_hash.as_deref().unwrap_or("missing"),
        candidate_hash == control_hash
    );
    Ok(())
}

fn summarize_k7_timings(timings: &[CodecStageTiming]) -> Result<(f64, f64, f64)> {
    ensure!(
        !timings.is_empty(),
        "device stage profiler returned no timings"
    );
    ensure!(
        timings
            .iter()
            .all(|timing| timing.source == CodecTimingSource::DeviceTimestamp),
        "paired geometry stage comparison requires device timestamps"
    );
    let milliseconds = |timing: &CodecStageTiming| timing.duration.as_secs_f64() * 1_000.0;
    let selected_k7_ms = timings
        .iter()
        .filter(|timing| {
            timing.label.ends_with("_k7_act1")
                && (timing.label.starts_with("codec_block0_")
                    || timing.label.starts_with("codec_block1_"))
        })
        .map(milliseconds)
        .sum();
    let all_k7_ms = timings
        .iter()
        .filter(|timing| timing.label.ends_with("_k7_act1"))
        .map(milliseconds)
        .sum();
    let all_stages_ms = timings.iter().map(milliseconds).sum();
    Ok((selected_k7_ms, all_k7_ms, all_stages_ms))
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.warmup > 0, "--warmup must be positive");
    ensure!(args.repeats > 0, "--repeats must be positive");
    ensure!(args.tasks_max > 0, "--tasks-max must be positive");
    ensure!(
        args.stage_profile_method == StageProfileMethod::Device
            || (args.k7_algorithm == K7ProfileAlgorithm::Production
                && args.pointwise_algorithm == PointwiseProfileAlgorithm::Production
                && args.stem_algorithm == StemProfileAlgorithm::Production
                && args.block_boundary_algorithm == BlockBoundaryProfileAlgorithm::Standalone
                && args.conv_transpose_snake_algorithm
                    == ConvTransposeSnakeProfileAlgorithm::Standalone),
        "explicit codec algorithm comparison requires --stage-profile-method device"
    );
    verify_sha256(&args.fixture, &args.fixture_sha256)?;
    let cache_root = match &args.cubecl_cache_dir {
        Some(path) => path.clone(),
        None => default_cubecl_cache_root()?,
    };
    let cache = if let Some(record) = &args.autotune_record {
        configure_cubecl_persistent_cache_for_precision_with_record(
            &cache_root,
            args.precision,
            Some(record),
        )?
    } else {
        configure_cubecl_persistent_cache_for_precision(&cache_root, args.precision)?
    };
    println!(
        "cubecl_cache environment={} root={} path={}",
        cache.environment_name,
        cache.root.display(),
        cache.environment_path.display()
    );
    let fixture_precision = args.fixture_precision.unwrap_or(args.precision);
    let (latent_values, latent_steps, expected_waveform) =
        load_oracle_tensors(&args.fixture, fixture_precision)?;
    println!(
        "profile_shape latent_steps={latent_steps} waveform_samples={} execution_precision={} fixture_precision={}",
        expected_waveform.len(),
        args.precision.label(),
        fixture_precision.label()
    );
    let (device, monitor) = initialize_wgpu(args.adapter_index, args.tasks_max);
    let tensor_device = wgpu_device_with_precision(&device, args.precision)?;

    let mut codec = load_codec(&args.codec_weights, &tensor_device)
        .with_context(|| format!("failed to load codec {}", args.codec_weights.display()))?;
    codec.prepare_decoder_for_wgsl_with_k7_algorithm(args.k7_algorithm.into());
    synchronize_and_check_wgpu(&device, &monitor, "codec load and preparation")?;
    let latent = Tensor::<3>::from_data(
        TensorData::new(latent_values, [1, latent_steps, 32]),
        &tensor_device,
    );

    if args.paired_cross_block_accumulator_store || args.paired_pointwise_head_fusion {
        ensure!(
            args.precision == WgpuFloatPrecision::Fp16
                && args.k7_algorithm == K7ProfileAlgorithm::Production
                && args.pointwise_algorithm == PointwiseProfileAlgorithm::Production
                && args.stem_algorithm == StemProfileAlgorithm::Production
                && args.block_boundary_algorithm == BlockBoundaryProfileAlgorithm::FusedC384AndC192
                && args.conv_transpose_snake_algorithm
                    == ConvTransposeSnakeProfileAlgorithm::Standalone
                && args.residual_state_layout == ResidualStateProfileLayout::ProductionNcl,
            "paired tail candidates require the F16 production route"
        );
        ensure!(
            !(args.paired_cross_block_accumulator_store && args.paired_pointwise_head_fusion),
            "select only one paired tail candidate"
        );
        let candidate = if args.paired_cross_block_accumulator_store {
            PairedTailCandidate::CrossBlockAccumulator
        } else {
            PairedTailCandidate::PointwiseHeadFusion
        };
        run_paired_tail_candidate(
            &codec,
            &latent,
            &expected_waveform,
            args.precision,
            &device,
            &monitor,
            args.warmup,
            args.repeats,
            candidate,
        )?;
        monitor.check("paired tail candidate completion")?;
        println!("wgpu_uncaptured_errors=0");
        return Ok(());
    }

    if args.paired_software_graph {
        ensure!(
            args.residual_state_layout == ResidualStateProfileLayout::ProductionNcl
                && args.k7_algorithm == K7ProfileAlgorithm::Production
                && args.pointwise_algorithm == PointwiseProfileAlgorithm::Production
                && args.stem_algorithm == StemProfileAlgorithm::Production
                && args.block_boundary_algorithm == BlockBoundaryProfileAlgorithm::FusedC384AndC192
                && args.conv_transpose_snake_algorithm
                    == ConvTransposeSnakeProfileAlgorithm::Standalone,
            "--paired-software-graph requires all production algorithm, layout, and fusion selections"
        );
        run_paired_software_graph(
            &codec,
            &latent,
            &expected_waveform,
            args.precision,
            &device,
            &monitor,
            args.warmup,
            args.repeats,
        )?;
        monitor.check("paired software-graph completion")?;
        println!("wgpu_uncaptured_errors=0");
        return Ok(());
    }

    if args.profile_k7_weight_repack {
        for warmup in 1..=args.warmup {
            let receipts = codec.profile_k7_weight_repacks()?;
            println!(
                "k7_repack_warmup={warmup}/{} copies={}",
                args.warmup,
                receipts.len()
            );
        }
        for repetition in 1..=args.repeats {
            let receipts = codec.profile_k7_weight_repacks()?;
            let total_ms: f64 = receipts
                .iter()
                .map(|receipt| receipt.device_duration_ms)
                .sum();
            for receipt in &receipts {
                println!(
                    "k7_repack repetition={repetition}/{} label={} source_oik={:?} logical_oki_strides={:?} materialized_oki_strides={:?} logical_rhs_vector={} materialized_rhs_vector={} bytes={} duration_ms={:.6} device_timestamp={}",
                    args.repeats,
                    receipt.label,
                    receipt.source_oik_shape,
                    receipt.logical_oki_strides,
                    receipt.materialized_oki_strides,
                    receipt.logical_rhs_vector_size,
                    receipt.materialized_rhs_vector_size,
                    receipt.materialized_bytes,
                    receipt.device_duration_ms,
                    receipt.used_device_timestamps,
                );
            }
            println!(
                "k7_repack_summary repetition={repetition}/{} copies={} total_device_ms={total_ms:.6}",
                args.repeats,
                receipts.len(),
            );
        }
        monitor.check("k7 repack profiling completion")?;
        println!("wgpu_uncaptured_errors=0");
        return Ok(());
    }

    if args.paired_geometry_multi_rows {
        ensure!(
            args.precision == WgpuFloatPrecision::Fp16,
            "--paired-geometry-multi-rows is an F16 k7 comparison"
        );
        ensure!(
            args.k7_algorithm == K7ProfileAlgorithm::Production
                && args.pointwise_algorithm == PointwiseProfileAlgorithm::Production
                && args.stem_algorithm == StemProfileAlgorithm::Production,
            "--paired-geometry-multi-rows requires all production algorithm selections"
        );
        run_paired_geometry_multi_rows(
            &codec,
            &latent,
            &expected_waveform,
            args.precision,
            &device,
            &monitor,
            args.warmup,
            args.repeats,
        )?;
        monitor.check("paired geometry multi-row completion")?;
        println!("wgpu_uncaptured_errors=0");
        return Ok(());
    }

    if args.paired_autotuned_multi_rows {
        ensure!(
            args.precision == WgpuFloatPrecision::Fp16,
            "--paired-autotuned-multi-rows is an F16 k7 comparison"
        );
        ensure!(
            args.k7_algorithm == K7ProfileAlgorithm::Production
                && args.pointwise_algorithm == PointwiseProfileAlgorithm::Production
                && args.stem_algorithm == StemProfileAlgorithm::Production,
            "--paired-autotuned-multi-rows requires all production algorithm selections"
        );
        run_paired_k7_plans(
            &codec,
            &latent,
            &expected_waveform,
            args.precision,
            &device,
            &monitor,
            args.warmup,
            args.repeats,
            CodecAlgorithmPlan::new(
                CodecK7Algorithm::CubeClImplicitGemmAutotuned,
                CodecPointwiseAlgorithm::AccuracyApproved,
            ),
            CodecAlgorithmPlan::accuracy_approved(),
            "autotuned-multi-rows",
            "geometry-heuristic",
        )?;
        monitor.check("paired autotuned multi-row completion")?;
        println!("wgpu_uncaptured_errors=0");
        return Ok(());
    }

    if args.paired_prepared_epilogue {
        ensure!(
            args.precision == WgpuFloatPrecision::Fp16,
            "--paired-prepared-epilogue is an F16 k7 comparison"
        );
        ensure!(
            args.k7_algorithm == K7ProfileAlgorithm::Production
                && args.pointwise_algorithm == PointwiseProfileAlgorithm::Production
                && args.stem_algorithm == StemProfileAlgorithm::Production,
            "--paired-prepared-epilogue requires all production algorithm selections"
        );
        let candidate = CodecK7Algorithm::CubeClImplicitGemmPreparedEpilogue;
        codec.prepare_decoder_for_wgsl_with_k7_algorithm(candidate);
        synchronize_and_check_wgpu(&device, &monitor, "prepared Snake reciprocal creation")?;
        run_paired_k7_plans(
            &codec,
            &latent,
            &expected_waveform,
            args.precision,
            &device,
            &monitor,
            args.warmup,
            args.repeats,
            CodecAlgorithmPlan::new(candidate, CodecPointwiseAlgorithm::AccuracyApproved),
            CodecAlgorithmPlan::new(
                CodecK7Algorithm::CubeClImplicitGemmGeometrySelectedMultiRows,
                CodecPointwiseAlgorithm::AccuracyApproved,
            ),
            "prepared-epilogue",
            "scalar-epilogue",
        )?;
        monitor.check("paired prepared-epilogue completion")?;
        println!("wgpu_uncaptured_errors=0");
        return Ok(());
    }

    if args.paired_pointwise_accumulator_store {
        ensure!(
            args.precision == WgpuFloatPrecision::Fp16,
            "--paired-pointwise-accumulator-store is an F16 pointwise comparison"
        );
        ensure!(
            args.k7_algorithm == K7ProfileAlgorithm::Production
                && args.pointwise_algorithm == PointwiseProfileAlgorithm::Production
                && args.stem_algorithm == StemProfileAlgorithm::Production,
            "--paired-pointwise-accumulator-store requires all production algorithm selections"
        );
        run_paired_k7_plans(
            &codec,
            &latent,
            &expected_waveform,
            args.precision,
            &device,
            &monitor,
            args.warmup,
            args.repeats,
            CodecAlgorithmPlan::new(
                CodecK7Algorithm::AccuracyApproved,
                CodecPointwiseAlgorithm::CubeClAccumulatorStore,
            ),
            CodecAlgorithmPlan::new(
                CodecK7Algorithm::AccuracyApproved,
                CodecPointwiseAlgorithm::PackedMatmul,
            ),
            "pointwise-accumulator-store",
            "packed-pointwise-control",
        )?;
        monitor.check("paired pointwise accumulator-store completion")?;
        println!("wgpu_uncaptured_errors=0");
        return Ok(());
    }

    if args.paired_pointwise_residual_store {
        ensure!(
            args.precision == WgpuFloatPrecision::Fp16,
            "--paired-pointwise-residual-store is an F16 pointwise comparison"
        );
        ensure!(
            args.k7_algorithm == K7ProfileAlgorithm::Production
                && args.pointwise_algorithm == PointwiseProfileAlgorithm::Production
                && args.stem_algorithm == StemProfileAlgorithm::Production,
            "--paired-pointwise-residual-store requires all production algorithm selections"
        );
        run_paired_k7_plans(
            &codec,
            &latent,
            &expected_waveform,
            args.precision,
            &device,
            &monitor,
            args.warmup,
            args.repeats,
            CodecAlgorithmPlan::new(
                CodecK7Algorithm::AccuracyApproved,
                CodecPointwiseAlgorithm::CubeClAccumulatorStore,
            ),
            CodecAlgorithmPlan::new(
                CodecK7Algorithm::AccuracyApproved,
                CodecPointwiseAlgorithm::CubeClAccumulatorPairOnly,
            ),
            "all-pointwise-accumulator-store",
            "pair-only-accumulator-store",
        )?;
        monitor.check("paired pointwise residual-store completion")?;
        println!("wgpu_uncaptured_errors=0");
        return Ok(());
    }

    if args.paired_prepared_weight {
        ensure!(
            args.precision == WgpuFloatPrecision::Fp16,
            "--paired-prepared-weight is an F16 k7 comparison"
        );
        ensure!(
            args.k7_algorithm == K7ProfileAlgorithm::Production
                && args.pointwise_algorithm == PointwiseProfileAlgorithm::Production
                && args.stem_algorithm == StemProfileAlgorithm::Production,
            "--paired-prepared-weight requires all production algorithm selections"
        );
        codec.prepare_decoder_for_wgsl_with_k7_algorithm(
            CodecK7Algorithm::CubeClImplicitGemmPreparedWeight(PreparedK7WeightPolicy::all()),
        );
        synchronize_and_check_wgpu(&device, &monitor, "prepared OKI materialization")?;
        run_paired_prepared_weight(
            &codec,
            &latent,
            &expected_waveform,
            args.precision,
            &device,
            &monitor,
            args.warmup,
            args.repeats,
            args.prepared_k7_min_bytes,
        )?;
        monitor.check("paired prepared-weight completion")?;
        println!("wgpu_uncaptured_errors=0");
        return Ok(());
    }

    if args.paired_single_storage {
        ensure!(
            args.precision == WgpuFloatPrecision::Fp16,
            "--paired-single-storage is an F16 k7 comparison"
        );
        ensure!(
            args.k7_algorithm == K7ProfileAlgorithm::Production
                && args.pointwise_algorithm == PointwiseProfileAlgorithm::Production
                && args.stem_algorithm == StemProfileAlgorithm::Production,
            "--paired-single-storage requires all production algorithm selections"
        );
        let mut repack = load_codec(&args.codec_weights, &tensor_device).with_context(|| {
            format!(
                "failed to load repack control {}",
                args.codec_weights.display()
            )
        })?;
        repack.prepare_decoder_for_wgsl_with_k7_algorithm(
            CodecK7Algorithm::CubeClImplicitGemmSingleStorage,
        );
        synchronize_and_check_wgpu(&device, &monitor, "paired codec preparation")?;
        run_paired_single_storage(
            &repack,
            &codec,
            &latent,
            &expected_waveform,
            args.precision,
            &device,
            &monitor,
            args.warmup,
            args.repeats,
        )?;
        monitor.check("paired completion")?;
        println!("wgpu_uncaptured_errors=0");
        return Ok(());
    }

    if args.residual_state_layout == ResidualStateProfileLayout::NhwcWithinBlock {
        ensure!(
            args.k7_algorithm == K7ProfileAlgorithm::Production
                && args.pointwise_algorithm == PointwiseProfileAlgorithm::Production
                && args.stem_algorithm == StemProfileAlgorithm::Production
                && args.block_boundary_algorithm == BlockBoundaryProfileAlgorithm::FusedC384AndC192
                && args.conv_transpose_snake_algorithm
                    == ConvTransposeSnakeProfileAlgorithm::Standalone,
            "--residual-state-layout nhwc-within-block requires all production algorithm and fusion selections"
        );
        ensure!(
            args.profile_repeats == 0,
            "NHWC residual-state screening reports whole-decode boundaries; use --profile-repeats 0"
        );
    }
    let plan = CodecAlgorithmPlan::new(args.k7_algorithm.into(), args.pointwise_algorithm.into())
        .with_stem(args.stem_algorithm.into());

    let decode_selected = |latent| -> Result<Tensor<3>> {
        if args.residual_state_layout == ResidualStateProfileLayout::NhwcWithinBlock {
            return Ok(
                codec.decode_wgsl_with_residual_state(latent, args.residual_state_layout.into())?
            );
        }
        if args.block_boundary_algorithm != BlockBoundaryProfileAlgorithm::Standalone
            || args.conv_transpose_snake_algorithm != ConvTransposeSnakeProfileAlgorithm::Standalone
        {
            return Ok(codec.decode_wgsl_with_fusions(
                latent,
                args.block_boundary_algorithm.into(),
                args.conv_transpose_snake_algorithm.into(),
            ));
        }
        match (
            args.k7_algorithm,
            args.pointwise_algorithm,
            args.stem_algorithm,
        ) {
            (
                K7ProfileAlgorithm::Production,
                PointwiseProfileAlgorithm::Production,
                StemProfileAlgorithm::Production,
            ) => Ok(codec.decode_wgsl_standalone_block_boundaries(latent)),
            _ => Ok(codec.decode_wgsl_with_plan(latent, plan)),
        }
    };

    for warmup in 1..=args.warmup {
        let output = decode_selected(latent.clone())?;
        synchronize_and_check_wgpu(&device, &monitor, &format!("warmup {warmup}"))?;
        drop(output);
    }

    let mut production_enqueue_ms = Vec::with_capacity(args.repeats);
    let mut production_device_ms = Vec::with_capacity(args.repeats);
    let mut production_readback_ms = Vec::with_capacity(args.repeats);
    let mut production_hash = None;
    for repetition in 1..=args.repeats {
        let started = Instant::now();
        let output = decode_selected(latent.clone())?;
        let enqueue_complete_ms = started.elapsed().as_secs_f64() * 1_000.0;
        synchronize_and_check_wgpu(
            &device,
            &monitor,
            &format!("production device completion {repetition}"),
        )?;
        let device_complete_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let values = output
            .cast(FloatDType::F32)
            .into_data()
            .to_vec::<f32>()
            .with_context(|| format!("failed production readback {repetition}"))?;
        synchronize_and_check_wgpu(
            &device,
            &monitor,
            &format!("production repetition {repetition}"),
        )?;
        let readback_complete_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let hash = sha256_f32_le(&values);
        if let Some(expected_hash) = &production_hash {
            ensure!(
                &hash == expected_hash,
                "production waveform changed at repetition {repetition}"
            );
        } else {
            production_hash = Some(hash.clone());
        }
        waveform_gate(
            &expected_waveform,
            &values,
            &format!("production_waveform[{repetition}]"),
            args.precision,
        )?;
        println!(
            "production_repeat={repetition}/{} decode_enqueue_complete_ms={enqueue_complete_ms:.6} decode_device_complete_ms={device_complete_ms:.6} decode_and_readback_ms={readback_complete_ms:.6} sha256={hash}",
            args.repeats
        );
        production_enqueue_ms.push(enqueue_complete_ms);
        production_device_ms.push(device_complete_ms);
        production_readback_ms.push(readback_complete_ms);
    }
    print_summary("production_decode_enqueue_complete", &production_enqueue_ms);
    print_summary("production_decode_device_complete", &production_device_ms);
    print_summary("production_decode_and_readback", &production_readback_ms);

    if matches!(
        args.k7_algorithm,
        K7ProfileAlgorithm::ImplicitGemm
            | K7ProfileAlgorithm::ImplicitGemmInputLayoutFused
            | K7ProfileAlgorithm::ImplicitGemmPreparedWeight
            | K7ProfileAlgorithm::ImplicitGemmDirectOik
            | K7ProfileAlgorithm::ImplicitGemmK7Halo
            | K7ProfileAlgorithm::ImplicitGemmMultiRows
            | K7ProfileAlgorithm::ImplicitGemmGeometrySelectedMultiRows
            | K7ProfileAlgorithm::ImplicitGemmPreparedEpilogue
            | K7ProfileAlgorithm::ImplicitGemmMaterialized
            | K7ProfileAlgorithm::ImplicitGemmAsync
            | K7ProfileAlgorithm::ImplicitGemmSyncStrided
            | K7ProfileAlgorithm::ImplicitGemmAsyncStrided
    ) || args.pointwise_algorithm == PointwiseProfileAlgorithm::ImplicitGemm
    {
        for warmup in 1..=args.warmup {
            let (output, _) = codec.decode_wgsl_device_profiled_with_plan(latent.clone(), plan)?;
            let values = output
                .cast(FloatDType::F32)
                .into_data()
                .to_vec::<f32>()
                .with_context(|| format!("failed implicit-gemm warmup readback {warmup}"))?;
            synchronize_and_check_wgpu(
                &device,
                &monitor,
                &format!("implicit-gemm warmup {warmup}"),
            )?;
            waveform_gate(
                &expected_waveform,
                &values,
                &format!("implicit_gemm_warmup[{warmup}]"),
                args.precision,
            )?;
            println!(
                "candidate_warmup={warmup}/{} k7_algorithm={:?} pointwise_algorithm={:?} sha256={}",
                args.warmup,
                args.k7_algorithm,
                args.pointwise_algorithm,
                sha256_f32_le(&values)
            );
        }
    }

    let mut stage_samples: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    let mut profiled_total_ms = Vec::with_capacity(args.profile_repeats);
    for repetition in 1..=args.profile_repeats {
        let started = Instant::now();
        let (output, timings) = match (
            args.stage_profile_method,
            args.block_boundary_algorithm,
            args.conv_transpose_snake_algorithm,
        ) {
            (
                StageProfileMethod::Device,
                boundary @ (BlockBoundaryProfileAlgorithm::FusedC384
                | BlockBoundaryProfileAlgorithm::FusedC192
                | BlockBoundaryProfileAlgorithm::FusedC384AndC192),
                conv_transpose,
            ) => codec.decode_wgsl_with_fusions_device_profiled(
                latent.clone(),
                boundary.into(),
                conv_transpose.into(),
            )?,
            (
                StageProfileMethod::Device,
                BlockBoundaryProfileAlgorithm::Standalone,
                conv_transpose @ (ConvTransposeSnakeProfileAlgorithm::CachedCol2ImCase1
                | ConvTransposeSnakeProfileAlgorithm::CachedCol2ImCase2
                | ConvTransposeSnakeProfileAlgorithm::CachedCol2ImCase3
                | ConvTransposeSnakeProfileAlgorithm::CachedCol2ImDualOutput),
            ) => codec.decode_wgsl_with_fusions_device_profiled(
                latent.clone(),
                CodecCrossBlockFusion::Standalone,
                conv_transpose.into(),
            )?,
            (
                StageProfileMethod::Device,
                BlockBoundaryProfileAlgorithm::Standalone,
                ConvTransposeSnakeProfileAlgorithm::Standalone,
            ) => codec.decode_wgsl_standalone_device_profiled_with_plan(latent.clone(), plan)?,
            (
                StageProfileMethod::Synchronized,
                BlockBoundaryProfileAlgorithm::Standalone,
                ConvTransposeSnakeProfileAlgorithm::Standalone,
            ) => codec.decode_wgsl_standalone_profiled(latent.clone(), |stage| {
                synchronize_and_check_wgpu(&device, &monitor, stage)
            })?,
            (StageProfileMethod::Synchronized, _, _) => {
                unreachable!("fused-boundary profiling requires device timestamps")
            }
        };
        let device_complete_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let values = output
            .cast(FloatDType::F32)
            .into_data()
            .to_vec::<f32>()
            .with_context(|| format!("failed profiled readback {repetition}"))?;
        synchronize_and_check_wgpu(
            &device,
            &monitor,
            &format!("profiled repetition {repetition}"),
        )?;
        waveform_gate(
            &expected_waveform,
            &values,
            &format!("profiled_waveform[{repetition}]"),
            args.precision,
        )?;
        let profiled_hash = sha256_f32_le(&values);
        if args.k7_algorithm == K7ProfileAlgorithm::Production
            && args.pointwise_algorithm == PointwiseProfileAlgorithm::Production
        {
            ensure!(
                production_hash.as_deref() == Some(&profiled_hash),
                "profiled waveform differs bitwise from production"
            );
        }
        for timing in timings {
            let source = match timing.source {
                CodecTimingSource::DeviceTimestamp => "device-timestamp",
                CodecTimingSource::SynchronizedSystemClock => "synchronized-system-clock",
            };
            println!(
                "stage_profile repetition={repetition} stage={} source={source} duration_ms={:.6}",
                timing.label,
                timing.duration.as_secs_f64() * 1_000.0
            );
            stage_samples
                .entry(timing.label)
                .or_default()
                .push(timing.duration.as_secs_f64() * 1_000.0);
        }
        let readback_complete_ms = started.elapsed().as_secs_f64() * 1_000.0;
        println!(
            "profiled_repeat={repetition}/{} method={:?} k7_algorithm={:?} pointwise_algorithm={:?} profile_wall_complete_ms={device_complete_ms:.6} profile_and_readback_ms={readback_complete_ms:.6} sha256={profiled_hash}",
            args.profile_repeats,
            args.stage_profile_method,
            args.k7_algorithm,
            args.pointwise_algorithm
        );
        profiled_total_ms.push(device_complete_ms);
    }

    let mut summaries: Vec<_> = stage_samples
        .iter()
        .map(|(&label, values)| (label, median(values), values))
        .collect();
    summaries.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));
    for (label, _, values) in summaries {
        print_summary(label, values);
    }
    print_summary("profiled_wall_complete", &profiled_total_ms);
    monitor.check("profile completion")?;
    println!("wgpu_uncaptured_errors=0");
    Ok(())
}
