//! WmHead Snake/layout production-kernel benchmark.
//!
//! The legacy path writes contiguous NCL Snake output, after which Burn
//! materializes a contiguous NLC copy before its unchanged direct Conv1d. The
//! production kernel evaluates the identical Snake expression while writing a
//! contiguous NLC allocation directly. Its logical NCL view is then passed to
//! the same Burn Conv1d and tanh operations.
//!
//! Run: `cargo run --release --bin bench_wm_head_snake_nlc -- 0`.

use std::{
    error::Error,
    hint::black_box,
    io,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};

use burn::{
    backend::wgpu::{
        RuntimeOptions, WgpuDevice, WgpuRuntime, graphics::AutoGraphicsApi, init_setup,
    },
    module::{Param, ParamId},
    nn::{
        PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{Tensor, TensorData, TensorPrimitive},
};
use cubecl::prelude::Runtime;
use irodori_tts_wgpu::{
    WgpuRaw,
    kernels::{
        snake::snake_wgsl,
        wm_head_snake_nlc::{BATCH, CHANNELS, TILE, TIME, wm_head_snake_ncl_to_nlc_wgsl},
    },
    weights::TensorStore,
};

type B = WgpuRaw;

const DEFAULT_WARMUP: usize = 10;
const DEFAULT_ITERATIONS: usize = 100;
const DEFAULT_TRIALS: usize = 5;
const DEFAULT_CODEC_WEIGHTS: &str = "target/v4_dacvae_weights.safetensors";
const ALPHA_KEY: &str = "decoder.wm_model.encoder_block.pre.0.alpha";
const WEIGHT_KEY: &str = "decoder.wm_model.encoder_block.pre.1.weight";
const BIAS_KEY: &str = "decoder.wm_model.encoder_block.pre.1.bias";
const KERNEL_SIZE: usize = 7;
const OUTPUT_CHANNELS: usize = 1;
const PADDING: usize = 3;
const F32_BYTES: usize = size_of::<f32>();

#[derive(Clone, Debug)]
enum WeightSource {
    AutoDefault,
    Checkpoint(PathBuf),
    Deterministic,
}

#[derive(Debug)]
struct Args {
    adapter_index: usize,
    warmup: usize,
    iterations: usize,
    trials: usize,
    weight_source: WeightSource,
}

enum ParseOutcome {
    Run(Args),
    Help,
}

#[derive(Clone, Copy, Debug)]
struct Timing {
    median_us: f64,
    min_us: f64,
    max_us: f64,
}

#[derive(Clone, Copy, Debug)]
struct PairTiming {
    legacy: Timing,
    production: Timing,
}

#[derive(Clone, Copy, Debug)]
struct Comparison {
    elements: usize,
    bit_mismatches: usize,
    max_abs: f32,
    finite: bool,
}

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

struct WmHeadParameters {
    alpha: Tensor<B, 3>,
    conv: Conv1d<B>,
    source: String,
}

fn usage() -> &'static str {
    "usage: bench_wm_head_snake_nlc <adapter-index> \
     [--codec-weights PATH | --deterministic] [--warmup N] \
     [--iterations N] [--trials N]"
}

fn next_positive_usize(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<usize, Box<dyn Error>> {
    let text = args
        .next()
        .ok_or_else(|| io::Error::other(format!("{option} requires a value")))?;
    let value = text.parse::<usize>().map_err(|error| {
        io::Error::other(format!("invalid value {text:?} for {option}: {error}"))
    })?;
    if value == 0 {
        return Err(io::Error::other(format!("{option} must be greater than zero")).into());
    }
    Ok(value)
}

fn parse_args() -> Result<ParseOutcome, Box<dyn Error>> {
    let mut adapter_index = None;
    let mut warmup = DEFAULT_WARMUP;
    let mut iterations = DEFAULT_ITERATIONS;
    let mut trials = DEFAULT_TRIALS;
    let mut weight_source = WeightSource::AutoDefault;
    let mut explicit_weight_source = false;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--codec-weights" => {
                if explicit_weight_source {
                    return Err(io::Error::other(
                        "choose exactly one of --codec-weights and --deterministic",
                    )
                    .into());
                }
                let path = args.next().ok_or_else(|| {
                    io::Error::other("--codec-weights requires a filesystem path")
                })?;
                weight_source = WeightSource::Checkpoint(PathBuf::from(path));
                explicit_weight_source = true;
            }
            "--deterministic" => {
                if explicit_weight_source {
                    return Err(io::Error::other(
                        "choose exactly one of --codec-weights and --deterministic",
                    )
                    .into());
                }
                weight_source = WeightSource::Deterministic;
                explicit_weight_source = true;
            }
            "--warmup" => warmup = next_positive_usize(&mut args, "--warmup")?,
            "--iterations" => iterations = next_positive_usize(&mut args, "--iterations")?,
            "--trials" => trials = next_positive_usize(&mut args, "--trials")?,
            "--help" | "-h" => return Ok(ParseOutcome::Help),
            _ if argument.starts_with('-') => {
                return Err(
                    io::Error::other(format!("unknown option {argument:?}; {}", usage())).into(),
                );
            }
            _ if adapter_index.is_none() => {
                adapter_index = Some(argument.parse::<usize>().map_err(|error| {
                    io::Error::other(format!(
                        "invalid adapter index {argument:?}: {error}; {}",
                        usage()
                    ))
                })?);
            }
            _ => {
                return Err(io::Error::other(format!(
                    "unexpected positional argument {argument:?}; {}",
                    usage()
                ))
                .into());
            }
        }
    }
    Ok(ParseOutcome::Run(Args {
        adapter_index: adapter_index
            .ok_or_else(|| io::Error::other(format!("missing adapter index; {}", usage())))?,
        warmup,
        iterations,
        trials,
        weight_source,
    }))
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

fn deterministic_values(length: usize, seed: u64, low: f32, high: f32) -> Vec<f32> {
    let mut state = seed;
    let scale = high - low;
    (0..length)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let fraction = ((state >> 40) as u32) as f32 / 16_777_215.0;
            low + scale * fraction
        })
        .collect()
}

fn ensure_dims<const D: usize>(
    label: &str,
    actual: [usize; D],
    expected: [usize; D],
) -> Result<(), Box<dyn Error>> {
    if actual != expected {
        return Err(io::Error::other(format!(
            "{label} shape mismatch: expected {expected:?}, got {actual:?}"
        ))
        .into());
    }
    Ok(())
}

fn make_conv(weight: Tensor<B, 3>, bias: Tensor<B, 1>, device: &WgpuDevice) -> Conv1d<B> {
    let mut conv = Conv1dConfig::new(CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE)
        .with_stride(1)
        .with_dilation(1)
        .with_padding(PaddingConfig1d::Explicit(PADDING, PADDING))
        .with_bias(true)
        .init::<B>(device);
    conv.weight = Param::initialized(ParamId::new(), weight);
    conv.bias = Some(Param::initialized(ParamId::new(), bias));
    conv
}

fn deterministic_parameters(device: &WgpuDevice) -> WmHeadParameters {
    let alpha = Tensor::<B, 3>::from_data(
        TensorData::new(
            deterministic_values(CHANNELS, 0xA1FA_0001, 0.25, 2.0),
            [BATCH, CHANNELS, 1],
        ),
        device,
    );
    let weight = Tensor::<B, 3>::from_data(
        TensorData::new(
            deterministic_values(
                OUTPUT_CHANNELS * CHANNELS * KERNEL_SIZE,
                0xC0A7_0002,
                -0.025,
                0.025,
            ),
            [OUTPUT_CHANNELS, CHANNELS, KERNEL_SIZE],
        ),
        device,
    );
    let bias =
        Tensor::<B, 1>::from_data(TensorData::new(vec![0.0125_f32], [OUTPUT_CHANNELS]), device);
    WmHeadParameters {
        alpha,
        conv: make_conv(weight, bias, device),
        source: "deterministic exact-layout tensors".to_owned(),
    }
}

fn checkpoint_parameters(
    path: &Path,
    device: &WgpuDevice,
) -> Result<WmHeadParameters, Box<dyn Error>> {
    let store = TensorStore::load(path)?;
    let alpha: Tensor<B, 3> = store.tensor(ALPHA_KEY, device)?;
    let weight: Tensor<B, 3> = store.tensor(WEIGHT_KEY, device)?;
    let bias: Tensor<B, 1> = store.tensor(BIAS_KEY, device)?;
    ensure_dims(ALPHA_KEY, alpha.dims(), [BATCH, CHANNELS, 1])?;
    ensure_dims(
        WEIGHT_KEY,
        weight.dims(),
        [OUTPUT_CHANNELS, CHANNELS, KERNEL_SIZE],
    )?;
    ensure_dims(BIAS_KEY, bias.dims(), [OUTPUT_CHANNELS])?;
    Ok(WmHeadParameters {
        alpha,
        conv: make_conv(weight, bias, device),
        source: format!("production checkpoint {}", path.display()),
    })
}

fn load_parameters(
    source: &WeightSource,
    device: &WgpuDevice,
) -> Result<WmHeadParameters, Box<dyn Error>> {
    match source {
        WeightSource::AutoDefault => {
            let path = Path::new(DEFAULT_CODEC_WEIGHTS);
            if path.is_file() {
                checkpoint_parameters(path, device)
            } else {
                eprintln!(
                    "default codec checkpoint {} is absent; using deterministic exact-layout tensors",
                    path.display()
                );
                Ok(deterministic_parameters(device))
            }
        }
        WeightSource::Checkpoint(path) => checkpoint_parameters(path, device),
        WeightSource::Deterministic => Ok(deterministic_parameters(device)),
    }
}

fn make_input(device: &WgpuDevice) -> Tensor<B, 3> {
    Tensor::from_data(
        TensorData::new(
            deterministic_values(BATCH * CHANNELS * TIME, 0x1A2B_3C4D, -0.75, 0.75),
            [BATCH, CHANNELS, TIME],
        ),
        device,
    )
}

fn current_snake(input: &Tensor<B, 3>, alpha: &Tensor<B, 3>) -> Tensor<B, 3> {
    let output = snake_wgsl(
        input.clone().into_primitive().tensor(),
        alpha.clone().into_primitive().tensor(),
    );
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn production_snake(
    input: &Tensor<B, 3>,
    alpha: &Tensor<B, 3>,
) -> Result<Tensor<B, 3>, Box<dyn Error>> {
    let output_nlc = wm_head_snake_ncl_to_nlc_wgsl(
        input.clone().into_primitive().tensor(),
        alpha.clone().into_primitive().tensor(),
    )?;
    let output_nlc: Tensor<B, 3> = Tensor::from_primitive(TensorPrimitive::Float(output_nlc));
    Ok(output_nlc.swap_dims(1, 2))
}

fn validate_physical_layouts(
    legacy: &Tensor<B, 3>,
    production: &Tensor<B, 3>,
) -> Result<(), Box<dyn Error>> {
    let legacy_raw = legacy.clone().into_primitive().tensor();
    let production_raw = production.clone().into_primitive().tensor();
    let expected_shape = [BATCH, CHANNELS, TIME];
    let expected_legacy_strides = [CHANNELS * TIME, TIME, 1];
    let expected_production_strides = [CHANNELS * TIME, 1, CHANNELS];
    if legacy_raw.meta.shape().dims::<3>() != expected_shape
        || &legacy_raw.meta.strides()[..] != expected_legacy_strides.as_slice()
        || !legacy_raw.is_contiguous()
    {
        return Err(io::Error::other(format!(
            "legacy Snake layout mismatch: shape={:?} strides={:?} contiguous={}",
            legacy_raw.meta.shape(),
            legacy_raw.meta.strides(),
            legacy_raw.is_contiguous()
        ))
        .into());
    }
    if production_raw.meta.shape().dims::<3>() != expected_shape
        || &production_raw.meta.strides()[..] != expected_production_strides.as_slice()
        || production_raw.is_contiguous()
    {
        return Err(io::Error::other(format!(
            "production logical NCL layout mismatch: shape={:?} strides={:?} contiguous={}",
            production_raw.meta.shape(),
            production_raw.meta.strides(),
            production_raw.is_contiguous()
        ))
        .into());
    }

    let production_nlc = production.clone().swap_dims(1, 2).into_primitive().tensor();
    let expected_nlc_strides = [CHANNELS * TIME, CHANNELS, 1];
    if production_nlc.meta.shape().dims::<3>() != [BATCH, TIME, CHANNELS]
        || &production_nlc.meta.strides()[..] != expected_nlc_strides.as_slice()
        || !production_nlc.is_contiguous()
    {
        return Err(io::Error::other(format!(
            "production physical NLC layout mismatch: shape={:?} strides={:?} contiguous={}",
            production_nlc.meta.shape(),
            production_nlc.meta.strides(),
            production_nlc.is_contiguous()
        ))
        .into());
    }
    println!(
        "layout: legacy logical NCL strides={expected_legacy_strides:?} contiguous=true; production logical NCL strides={expected_production_strides:?}, physical NLC strides={expected_nlc_strides:?} contiguous=true"
    );
    Ok(())
}

fn compare_tensors(
    label: &str,
    expected: Tensor<B, 3>,
    actual: Tensor<B, 3>,
) -> Result<Comparison, Box<dyn Error>> {
    let expected_shape = expected.dims();
    let actual_shape = actual.dims();
    if expected_shape != actual_shape {
        return Err(io::Error::other(format!(
            "{label} shape mismatch: legacy={expected_shape:?}, production={actual_shape:?}"
        ))
        .into());
    }
    let expected = expected.into_data().to_vec::<f32>()?;
    let actual = actual.into_data().to_vec::<f32>()?;
    if expected.len() != actual.len() {
        return Err(io::Error::other(format!(
            "{label} element count mismatch: legacy={}, production={}",
            expected.len(),
            actual.len()
        ))
        .into());
    }

    let mut bit_mismatches = 0_usize;
    let mut max_abs = 0.0_f32;
    for (index, (&expected, &actual)) in expected.iter().zip(&actual).enumerate() {
        if !expected.is_finite() || !actual.is_finite() {
            return Err(io::Error::other(format!(
                "{label} contains non-finite data at {index}: legacy={expected:?}, production={actual:?}"
            ))
            .into());
        }
        bit_mismatches += usize::from(expected.to_bits() != actual.to_bits());
        max_abs = max_abs.max((expected - actual).abs());
    }
    Ok(Comparison {
        elements: expected.len(),
        bit_mismatches,
        max_abs,
        finite: max_abs.is_finite(),
    })
}

fn require_exact(label: &str, comparison: Comparison) -> Result<(), Box<dyn Error>> {
    println!(
        "correctness {label}: elements={} bit_mismatch={} max_abs={:.9e} finite={}",
        comparison.elements, comparison.bit_mismatches, comparison.max_abs, comparison.finite
    );
    if comparison.bit_mismatches != 0 || comparison.max_abs != 0.0 || !comparison.finite {
        return Err(io::Error::other(format!(
            "{label} failed exact production gate: {comparison:?}"
        ))
        .into());
    }
    Ok(())
}

fn validate_correctness(
    input: &Tensor<B, 3>,
    parameters: &WmHeadParameters,
) -> Result<(), Box<dyn Error>> {
    let legacy_activated = current_snake(input, &parameters.alpha);
    let production_activated = production_snake(input, &parameters.alpha)?;
    validate_physical_layouts(&legacy_activated, &production_activated)?;
    require_exact(
        "Snake logical NCL",
        compare_tensors(
            "Snake logical NCL",
            legacy_activated.clone(),
            production_activated.clone(),
        )?,
    )?;

    let legacy_pre_tanh = parameters.conv.forward(legacy_activated);
    let production_pre_tanh = parameters.conv.forward(production_activated);
    require_exact(
        "Conv pre-tanh",
        compare_tensors(
            "Conv pre-tanh",
            legacy_pre_tanh.clone(),
            production_pre_tanh.clone(),
        )?,
    )?;
    require_exact(
        "final tanh",
        compare_tensors(
            "final tanh",
            legacy_pre_tanh.tanh(),
            production_pre_tanh.tanh(),
        )?,
    )
}

fn warm_up<T, F>(
    count: usize,
    operation: &mut F,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    label: &str,
) -> Result<(), Box<dyn Error>>
where
    F: FnMut() -> Result<T, Box<dyn Error>>,
{
    let mut output = None;
    for _ in 0..count {
        output = Some(operation()?);
    }
    black_box(output.as_ref());
    synchronize_and_check_wgpu(device, monitor, &format!("{label} warmup"))?;
    drop(output);
    Ok(())
}

fn measure<T, F>(
    iterations: usize,
    operation: &mut F,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    label: &str,
) -> Result<f64, Box<dyn Error>>
where
    F: FnMut() -> Result<T, Box<dyn Error>>,
{
    let started = Instant::now();
    let mut output = None;
    for _ in 0..iterations {
        output = Some(operation()?);
    }
    black_box(output.as_ref());
    synchronize_and_check_wgpu(device, monitor, label)?;
    let elapsed_us = started.elapsed().as_secs_f64() * 1_000_000.0;
    drop(output);
    Ok(elapsed_us / iterations as f64)
}

fn summarize(samples: &[f64]) -> Result<Timing, Box<dyn Error>> {
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return Err(io::Error::other("timing samples must be non-empty and finite").into());
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    Ok(Timing {
        median_us: sorted[sorted.len() / 2],
        min_us: sorted[0],
        max_us: sorted[sorted.len() - 1],
    })
}

fn benchmark_pair<T, Legacy, Production>(
    label: &str,
    args: &Args,
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    legacy: &mut Legacy,
    production: &mut Production,
) -> Result<PairTiming, Box<dyn Error>>
where
    Legacy: FnMut() -> Result<T, Box<dyn Error>>,
    Production: FnMut() -> Result<T, Box<dyn Error>>,
{
    warm_up(
        args.warmup,
        legacy,
        device,
        monitor,
        &format!("{label} legacy"),
    )?;
    warm_up(
        args.warmup,
        production,
        device,
        monitor,
        &format!("{label} production"),
    )?;

    let mut samples: [Vec<f64>; 2] = std::array::from_fn(|_| Vec::with_capacity(args.trials));
    for trial in 0..args.trials {
        for offset in 0..2 {
            let variant = (trial + offset) % 2;
            let sample_label = format!("{label} trial {trial} variant {variant}");
            let sample = if variant == 0 {
                measure(args.iterations, legacy, device, monitor, &sample_label)?
            } else {
                measure(args.iterations, production, device, monitor, &sample_label)?
            };
            samples[variant].push(sample);
        }
    }
    let pair = PairTiming {
        legacy: summarize(&samples[0])?,
        production: summarize(&samples[1])?,
    };
    println!(
        "timing {label} legacy={:.3}us [{:.3},{:.3}] production={:.3}us [{:.3},{:.3}] speedup={:.3}x saving={:.3}us",
        pair.legacy.median_us,
        pair.legacy.min_us,
        pair.legacy.max_us,
        pair.production.median_us,
        pair.production.min_us,
        pair.production.max_us,
        pair.legacy.median_us / pair.production.median_us,
        pair.legacy.median_us - pair.production.median_us,
    );
    Ok(pair)
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn print_static_accounting() -> Result<(), Box<dyn Error>> {
    let elements = BATCH
        .checked_mul(CHANNELS)
        .and_then(|value| value.checked_mul(TIME))
        .ok_or_else(|| io::Error::other("WmHead element count overflow"))?;
    let tensor_bytes = elements
        .checked_mul(F32_BYTES)
        .ok_or_else(|| io::Error::other("WmHead tensor byte count overflow"))?;
    let legacy_boundary_traffic = tensor_bytes
        .checked_mul(4)
        .ok_or_else(|| io::Error::other("legacy boundary traffic overflow"))?;
    let production_boundary_traffic = tensor_bytes
        .checked_mul(2)
        .ok_or_else(|| io::Error::other("production boundary traffic overflow"))?;
    let conv_mac = TIME
        .checked_mul(CHANNELS)
        .and_then(|value| value.checked_mul(KERNEL_SIZE))
        .ok_or_else(|| io::Error::other("WmHead Conv MAC count overflow"))?;
    println!(
        "static exact B={BATCH} C={CHANNELS} T={TIME}: elements={elements}, tensor={tensor_bytes}B ({:.3}MiB), tile={TILE}",
        mib(tensor_bytes)
    );
    println!(
        "Snake/layout boundary traffic before mandatory Conv reads: legacy={legacy_boundary_traffic}B ({:.3}MiB, Snake read/write + transpose read/write) -> production={production_boundary_traffic}B ({:.3}MiB, fused read/write), saved={}B ({:.3}MiB)",
        mib(legacy_boundary_traffic),
        mib(production_boundary_traffic),
        legacy_boundary_traffic - production_boundary_traffic,
        mib(legacy_boundary_traffic - production_boundary_traffic),
    );
    println!(
        "dispatch: Stage A 1->1, WmHead full 5->4 (Snake + Burn input-layout copy + weight-layout copy + Conv + tanh); Stage-A workgroups={} -> {}; removed transient={}B ({:.3}MiB), persistent_delta=0",
        elements.div_ceil(256),
        (TIME / TILE) * (CHANNELS / TILE),
        tensor_bytes,
        mib(tensor_bytes),
    );
    println!(
        "unchanged Conv: weight=[1,{CHANNELS},{KERNEL_SIZE}], output=[1,1,{TIME}], conventional_MAC={conv_mac}; tanh elements={TIME}"
    );
    Ok(())
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    println!(
        "WmHead production-kernel A/B: warmup={} iterations={} trials={} variants=2 rotated",
        args.warmup, args.iterations, args.trials
    );
    print_static_accounting()?;
    let (device, monitor) = initialize_wgpu(args.adapter_index);
    let parameters = load_parameters(&args.weight_source, &device)?;
    println!("parameter_source={}", parameters.source);
    let input = make_input(&device);
    synchronize_and_check_wgpu(&device, &monitor, "input and parameter upload")?;

    validate_correctness(&input, &parameters)?;
    synchronize_and_check_wgpu(&device, &monitor, "three-stage exact parity")?;

    let mut legacy_stage =
        || -> Result<Tensor<B, 3>, Box<dyn Error>> { Ok(current_snake(&input, &parameters.alpha)) };
    let mut production_stage = || production_snake(&input, &parameters.alpha);
    let stage = benchmark_pair(
        "Snake+layout stage",
        &args,
        &device,
        &monitor,
        &mut legacy_stage,
        &mut production_stage,
    )?;

    let legacy_activated = current_snake(&input, &parameters.alpha);
    let production_activated = production_snake(&input, &parameters.alpha)?;
    synchronize_and_check_wgpu(&device, &monitor, "split-tail input preparation")?;
    let mut legacy_tail = || -> Result<Tensor<B, 3>, Box<dyn Error>> {
        Ok(parameters.conv.forward(legacy_activated.clone()).tanh())
    };
    let mut production_tail = || -> Result<Tensor<B, 3>, Box<dyn Error>> {
        Ok(parameters.conv.forward(production_activated.clone()).tanh())
    };
    let tail = benchmark_pair(
        "unchanged Burn Conv+tanh tail",
        &args,
        &device,
        &monitor,
        &mut legacy_tail,
        &mut production_tail,
    )?;

    let mut legacy_full = || -> Result<Tensor<B, 3>, Box<dyn Error>> {
        Ok(parameters
            .conv
            .forward(current_snake(&input, &parameters.alpha))
            .tanh())
    };
    let mut production_full = || -> Result<Tensor<B, 3>, Box<dyn Error>> {
        Ok(parameters
            .conv
            .forward(production_snake(&input, &parameters.alpha)?)
            .tanh())
    };
    let full = benchmark_pair(
        "full WmHead",
        &args,
        &device,
        &monitor,
        &mut legacy_full,
        &mut production_full,
    )?;

    let clear_win = full.production.max_us < full.legacy.min_us
        && full.production.median_us <= full.legacy.median_us * 0.98;
    println!(
        "summary split_savings_us: stage={:.3} tail={:.3} summed={:.3}; full_saving_us={:.3}; full_speedup={:.3}x; clear_win_nonoverlap_and_2pct={clear_win}",
        stage.legacy.median_us - stage.production.median_us,
        tail.legacy.median_us - tail.production.median_us,
        stage.legacy.median_us + tail.legacy.median_us
            - stage.production.median_us
            - tail.production.median_us,
        full.legacy.median_us - full.production.median_us,
        full.legacy.median_us / full.production.median_us,
    );
    monitor.check("benchmark completion")?;
    println!("WGPU errors=0");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    match parse_args()? {
        ParseOutcome::Run(args) => run(args),
        ParseOutcome::Help => {
            println!("{}", usage());
            Ok(())
        }
    }
}
