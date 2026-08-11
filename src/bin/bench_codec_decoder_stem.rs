//! Production-weight A/B for the current decoder-stem route and Burn fallback.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, Read},
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
    tensor::{Tensor, TensorData},
};
use clap::Parser;
use cubecl::prelude::Runtime;
use irodori_tts_wgpu::{WgpuRaw, codec::load_codec};
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};

const INPUT_CHANNELS: usize = 1_024;
const OUTPUT_CHANNELS: usize = 1_536;
const LENGTH: usize = 50;
const KERNEL_SIZE: usize = 7;
const ISOLATED_MAX_ABS: f32 = 2.0e-4;

#[derive(Debug, Parser)]
#[command(about = "A/B exact production decoder-stem candidates")]
struct Args {
    #[arg(long)]
    fixture: PathBuf,
    #[arg(long)]
    fixture_sha256: String,
    #[arg(long)]
    codec_weights: PathBuf,
    #[arg(long)]
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

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Variant {
    BurnReference,
    ProductionRoute,
}

impl Variant {
    const ALL: [Self; 2] = [Self::BurnReference, Self::ProductionRoute];

    const fn label(self) -> &'static str {
        match self {
            Self::BurnReference => "burn_reference",
            Self::ProductionRoute => "production_stem_route",
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
    let client = WgpuRuntime::client(device);
    let sync_result = cubecl::future::block_on(client.sync());
    monitor.check(stage)?;
    sync_result.with_context(|| format!("CubeCL synchronization failed after {stage}"))
}

fn verify_sha256(label: &str, path: &Path, expected: &str) -> Result<()> {
    ensure!(
        expected.len() == 64 && expected.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{label} SHA-256 must contain exactly 64 hex digits"
    );
    let file = File::open(path)
        .with_context(|| format!("failed to open {label} for hashing: {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .with_context(|| format!("failed to hash {label}: {}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let actual = format!("{:x}", hasher.finalize());
    ensure!(
        actual == expected.to_ascii_lowercase(),
        "{label} SHA-256 mismatch: got {actual}, expected {expected}"
    );
    println!("sha256: {label}={actual} path={}", path.display());
    Ok(())
}

fn load_exact_latent(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read precision fixture {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("malformed precision fixture {}", path.display()))?;
    let view = tensors
        .tensor("final_patched_latent")
        .context("fixture tensor final_patched_latent is missing")?;
    ensure!(view.dtype() == Dtype::F32, "final latent must be f32");
    ensure!(
        view.shape() == [1, LENGTH, 32],
        "final latent has unexpected shape {:?}",
        view.shape()
    );
    view.data()
        .chunks_exact(size_of::<f32>())
        .map(|chunk| {
            let bytes: [u8; size_of::<f32>()] = chunk
                .try_into()
                .map_err(|_| anyhow::anyhow!("invalid f32 bytes in final latent"))?;
            Ok(f32::from_le_bytes(bytes))
        })
        .collect()
}

fn sha256_f32_le(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    values
        .iter()
        .for_each(|value| hasher.update(value.to_bits().to_le_bytes()));
    format!("{:x}", hasher.finalize())
}

struct Comparison {
    elements: usize,
    finite: bool,
    bit_mismatch: usize,
    max_abs: f32,
    mean_abs: f64,
}

fn compare_full_output(reference: &[f32], candidate: &[f32]) -> Result<Comparison> {
    ensure!(
        reference.len() == candidate.len(),
        "stem output length mismatch: reference={} candidate={}",
        reference.len(),
        candidate.len()
    );
    let mut finite = true;
    let mut bit_mismatch = 0;
    let mut max_abs = 0.0_f32;
    let mut absolute_sum = 0.0_f64;
    for (&expected, &actual) in reference.iter().zip(candidate) {
        finite &= expected.is_finite() && actual.is_finite();
        bit_mismatch += usize::from(expected.to_bits() != actual.to_bits());
        let difference = (expected - actual).abs();
        max_abs = max_abs.max(difference);
        absolute_sum += f64::from(difference);
    }
    Ok(Comparison {
        elements: reference.len(),
        finite,
        bit_mismatch,
        max_abs,
        mean_abs: absolute_sum / reference.len() as f64,
    })
}

fn run_variant(
    variant: Variant,
    codec: &irodori_tts_wgpu::codec::DacVaeCodec<WgpuRaw>,
    input: &Tensor<WgpuRaw, 3>,
) -> Result<Tensor<WgpuRaw, 3>> {
    match variant {
        Variant::BurnReference => Ok(codec.decoder_stem_burn_reference_wgsl(input.clone())),
        Variant::ProductionRoute => Ok(codec.decoder_stem_current_wgsl(input.clone())),
    }
}

fn warm_variant(
    variant: Variant,
    codec: &irodori_tts_wgpu::codec::DacVaeCodec<WgpuRaw>,
    input: &Tensor<WgpuRaw, 3>,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    warmup: usize,
) -> Result<()> {
    let mut output = None;
    for _ in 0..warmup {
        output = Some(run_variant(variant, codec, input)?);
    }
    ensure!(output.is_some(), "warmup must execute at least once");
    synchronize_and_check_wgpu(device, monitor, &format!("{} warmup", variant.label()))?;
    drop(output);
    Ok(())
}

fn measure_trial(
    variant: Variant,
    codec: &irodori_tts_wgpu::codec::DacVaeCodec<WgpuRaw>,
    input: &Tensor<WgpuRaw, 3>,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    iterations: usize,
) -> Result<f64> {
    let started = Instant::now();
    let mut output = None;
    for _ in 0..iterations {
        output = Some(run_variant(variant, codec, input)?);
    }
    ensure!(output.is_some(), "trial must execute at least once");
    synchronize_and_check_wgpu(device, monitor, &format!("{} trial", variant.label()))?;
    drop(output);
    Ok(started.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64)
}

fn summary(values: &[f64]) -> (f64, f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    (
        sorted[sorted.len() / 2],
        sorted[0],
        sorted[sorted.len() - 1],
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.warmup > 0, "--warmup must be positive");
    ensure!(args.iterations > 0, "--iterations must be positive");
    ensure!(
        args.trials > 0 && args.trials % 2 == 1,
        "--trials must be positive and odd"
    );
    verify_sha256("precision_fixture", &args.fixture, &args.fixture_sha256)?;
    verify_sha256(
        "converted_codec",
        &args.codec_weights,
        &args.codec_weights_sha256,
    )?;
    let latent_values = load_exact_latent(&args.fixture)?;
    let (device, monitor) = initialize_wgpu(args.adapter_index);
    let codec = load_codec::<WgpuRaw>(&args.codec_weights, &device)
        .with_context(|| format!("failed to load codec {}", args.codec_weights.display()))?;
    synchronize_and_check_wgpu(&device, &monitor, "codec load")?;

    let latent =
        Tensor::<WgpuRaw, 3>::from_data(TensorData::new(latent_values, [1, LENGTH, 32]), &device);
    let stem_input = codec.decoder_stem_input_wgsl(latent);
    synchronize_and_check_wgpu(&device, &monitor, "exact stem input")?;
    ensure!(
        stem_input.dims() == [1, INPUT_CHANNELS, LENGTH],
        "stem input shape mismatch"
    );

    println!(
        "static_contract input=[1,{INPUT_CHANNELS},{LENGTH}] output=[1,{OUTPUT_CHANNELS},{LENGTH}] k={KERNEL_SIZE} stride=1 dilation=1 groups=1 padding=3 bias={OUTPUT_CHANNELS} dtype=f32 source_weight=checkpoint-native-contiguous-OIK"
    );
    println!(
        "direct_accounting tile=T64/O32/Cin16 workgroups=48 shared_bytes=18816 computed_time_slots=64 valid_time_slots=50 guarded_tail_slots=14 bias_order=last"
    );
    println!(
        "isolated_accuracy_gate=screening max_abs<={ISOLATED_MAX_ABS:.9e} finite=true production_adoption_requires_full_decode_strict_waveform_gate_and_hash_stability"
    );

    let burn_reference = run_variant(Variant::BurnReference, &codec, &stem_input)?
        .into_data()
        .to_vec::<f32>()?;
    ensure!(
        burn_reference.iter().all(|value| value.is_finite()),
        "Burn stem reference output is non-finite"
    );
    println!(
        "burn_reference_output elements={} finite=true sha256={}",
        burn_reference.len(),
        sha256_f32_le(&burn_reference)
    );
    let production = run_variant(Variant::ProductionRoute, &codec, &stem_input)?
        .into_data()
        .to_vec::<f32>()?;
    let comparison = compare_full_output(&burn_reference, &production)?;
    println!(
        "correctness variant={} elements={} finite={} bit_mismatch={} max_abs={:.9e} mean_abs={:.9e} sha256={}",
        Variant::ProductionRoute.label(),
        comparison.elements,
        comparison.finite,
        comparison.bit_mismatch,
        comparison.max_abs,
        comparison.mean_abs,
        sha256_f32_le(&production)
    );
    ensure!(comparison.finite, "production stem output is non-finite");
    ensure!(
        comparison.max_abs <= ISOLATED_MAX_ABS,
        "production stem max_abs {:.9e} exceeds isolated gate {:.9e}",
        comparison.max_abs,
        ISOLATED_MAX_ABS
    );

    for variant in Variant::ALL {
        warm_variant(variant, &codec, &stem_input, &device, &monitor, args.warmup)?;
    }

    let mut samples: BTreeMap<Variant, Vec<f64>> = Variant::ALL
        .into_iter()
        .map(|variant| (variant, Vec::with_capacity(args.trials)))
        .collect();
    for trial in 0..args.trials {
        let order = [
            Variant::ALL[trial % Variant::ALL.len()],
            Variant::ALL[(trial + 1) % Variant::ALL.len()],
        ];
        for variant in order {
            let elapsed_us = measure_trial(
                variant,
                &codec,
                &stem_input,
                &device,
                &monitor,
                args.iterations,
            )?;
            println!(
                "trial={} variant={} average_us={elapsed_us:.6}",
                trial + 1,
                variant.label()
            );
            samples
                .get_mut(&variant)
                .expect("all variants must have a sample vector")
                .push(elapsed_us);
        }
    }

    let burn_summary = summary(
        samples
            .get(&Variant::BurnReference)
            .expect("Burn reference samples must exist"),
    );
    for variant in Variant::ALL {
        let (median, minimum, maximum) =
            summary(samples.get(&variant).expect("variant samples must exist"));
        println!(
            "timing variant={} median_us={median:.6} range_us=[{minimum:.6},{maximum:.6}] speedup={:.6}x saving_us={:.6} all_range_below_burn={}",
            variant.label(),
            burn_summary.0 / median,
            burn_summary.0 - median,
            variant == Variant::BurnReference || maximum < burn_summary.1
        );
    }
    monitor.check("benchmark completion")?;
    println!("wgpu_uncaptured_errors=0");
    Ok(())
}
