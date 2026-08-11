//! Validate and measure the production DACVAE polyphase ConvTranspose1d path.
//!
//! This binary is intentionally kept out of `Cargo.toml` until its design is
//! reviewed. Once registered, run all released shapes with:
//!
//! `cargo run --release --bin bench_conv_transpose1d_polyphase -- 0 --nvml-index 1`
//!
//! For the cleanest NVML delta, run one shape in a fresh process with
//! `--case 0`, `--case 1`, `--case 2`, or `--case 3`.

use std::{
    error::Error,
    io,
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use burn::{
    backend::wgpu::{WgpuDevice, graphics::AutoGraphicsApi, init_setup},
    module::{Param, ParamId},
    nn::{Initializer, conv::ConvTranspose1d, conv::ConvTranspose1dConfig},
    tensor::{Distribution, Tensor, TensorData, TensorPrimitive, backend::Backend},
};
use irodori_tts_wgpu::{
    WgpuRaw,
    kernels::conv_transpose1d_polyphase::{
        ConvTranspose1dStride, PackedConvTranspose1dWeight, conv_transpose1d_polyphase_wgsl,
        pack_conv_transpose1d_weight_wgsl,
    },
};

type B = WgpuRaw;

const DEFAULT_WARMUP: usize = 10;
const DEFAULT_TRIALS: usize = 5;
const CPU_INPUT_CHANNELS: usize = 16;
const CPU_OUTPUT_CHANNELS: usize = 16;
const CPU_INPUT_LENGTH: usize = 5;
const CPU_TOLERANCE: f32 = 2.0e-5;
const VRAM_PROBE_ITERATIONS: usize = 10;

#[derive(Clone, Copy, Debug)]
struct ConvTransposeCase {
    input_channels: usize,
    output_channels: usize,
    input_length: usize,
    stride: ConvTranspose1dStride,
    iterations: usize,
}

impl ConvTransposeCase {
    fn kernel_size(self) -> usize {
        2 * self.stride.value()
    }

    fn output_length(self) -> usize {
        self.input_length * self.stride.value()
    }
}

const PRODUCTION_CASES: [ConvTransposeCase; 4] = [
    ConvTransposeCase {
        input_channels: 1536,
        output_channels: 768,
        input_length: 50,
        stride: ConvTranspose1dStride::Twelve,
        iterations: 100,
    },
    ConvTransposeCase {
        input_channels: 768,
        output_channels: 384,
        input_length: 600,
        stride: ConvTranspose1dStride::Ten,
        iterations: 50,
    },
    ConvTransposeCase {
        input_channels: 384,
        output_channels: 192,
        input_length: 6_000,
        stride: ConvTranspose1dStride::Eight,
        iterations: 10,
    },
    ConvTransposeCase {
        input_channels: 192,
        output_channels: 96,
        input_length: 48_000,
        stride: ConvTranspose1dStride::Two,
        iterations: 10,
    },
];

#[derive(Debug)]
struct Args {
    adapter_index: Option<usize>,
    nvml_index: Option<usize>,
    case_index: Option<usize>,
    warmup: usize,
    iterations: Option<usize>,
    trials: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            adapter_index: None,
            nvml_index: None,
            case_index: None,
            warmup: DEFAULT_WARMUP,
            iterations: None,
            trials: DEFAULT_TRIALS,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Timing {
    median_us: f64,
    min_us: f64,
    max_us: f64,
}

#[derive(Clone, Copy, Debug)]
struct MemoryAccounting {
    common_resident_bytes: usize,
    burn_weight_reorder_bytes: usize,
    burn_columns_bytes: usize,
    custom_packed_weight_bytes: usize,
}

#[derive(Debug)]
struct NvmlPeakSampler {
    stop: Arc<AtomicBool>,
    peak_mib: Arc<AtomicU64>,
    worker: JoinHandle<()>,
}

impl NvmlPeakSampler {
    fn start(nvml_index: usize, initial_mib: u64) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak_mib = Arc::new(AtomicU64::new(initial_mib));
        let worker_stop = Arc::clone(&stop);
        let worker_peak = Arc::clone(&peak_mib);
        let worker = thread::spawn(move || {
            while !worker_stop.load(Ordering::Relaxed) {
                if let Ok(used_mib) = query_total_vram_mib(nvml_index) {
                    worker_peak.fetch_max(used_mib, Ordering::Relaxed);
                }
                thread::sleep(Duration::from_millis(10));
            }
        });
        Self {
            stop,
            peak_mib,
            worker,
        }
    }

    fn finish(self) -> Result<u64, Box<dyn Error>> {
        self.stop.store(true, Ordering::Relaxed);
        self.worker
            .join()
            .map_err(|_| io::Error::other("NVML sampler thread panicked"))?;
        Ok(self.peak_mib.load(Ordering::Relaxed))
    }
}

fn usage() -> &'static str {
    "usage: bench_conv_transpose1d_polyphase [adapter-index] \
     [--nvml-index N] [--case 0|1|2|3] [--warmup N] \
     [--iterations N] [--trials N]"
}

fn next_usize(
    args: &mut impl Iterator<Item = String>,
    option: &str,
    require_positive: bool,
) -> Result<usize, Box<dyn Error>> {
    let value = args
        .next()
        .ok_or_else(|| io::Error::other(format!("{option} requires a value")))?;
    let parsed = value.parse::<usize>().map_err(|error| {
        io::Error::other(format!("invalid value {value:?} for {option}: {error}"))
    })?;
    if require_positive && parsed == 0 {
        return Err(io::Error::other(format!("{option} must be greater than zero")).into());
    }
    Ok(parsed)
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut parsed = Args::default();
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--nvml-index" => {
                parsed.nvml_index = Some(next_usize(&mut args, "--nvml-index", false)?)
            }
            "--case" => parsed.case_index = Some(next_usize(&mut args, "--case", false)?),
            "--warmup" => parsed.warmup = next_usize(&mut args, "--warmup", true)?,
            "--iterations" => {
                parsed.iterations = Some(next_usize(&mut args, "--iterations", true)?)
            }
            "--trials" => parsed.trials = next_usize(&mut args, "--trials", true)?,
            "--help" | "-h" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            _ if argument.starts_with('-') => {
                return Err(
                    io::Error::other(format!("unknown option {argument:?}; {}", usage())).into(),
                );
            }
            _ if parsed.adapter_index.is_none() => {
                parsed.adapter_index = Some(argument.parse::<usize>().map_err(|error| {
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
    if let Some(index) = parsed.case_index
        && index >= PRODUCTION_CASES.len()
    {
        return Err(io::Error::other(format!(
            "--case must be in 0..{}, got {index}",
            PRODUCTION_CASES.len()
        ))
        .into());
    }
    Ok(parsed)
}

fn conv_with_data(
    device: &<B as Backend>::Device,
    input_channels: usize,
    output_channels: usize,
    stride: ConvTranspose1dStride,
    weight: Tensor<B, 3>,
    bias: Tensor<B, 1>,
) -> ConvTranspose1d<B> {
    let stride = stride.value();
    let mut conv = ConvTranspose1dConfig::new([input_channels, output_channels], 2 * stride)
        .with_stride(stride)
        .with_padding(stride / 2)
        .with_padding_out(0)
        .with_dilation(1)
        .with_groups(1)
        .with_bias(true)
        .with_initializer(Initializer::Zeros)
        .init::<B>(device);
    conv.weight = Param::initialized(ParamId::new(), weight);
    conv.bias = Some(Param::initialized(ParamId::new(), bias));
    conv
}

fn pack_weight(
    conv: &ConvTranspose1d<B>,
    stride: ConvTranspose1dStride,
) -> PackedConvTranspose1dWeight {
    pack_conv_transpose1d_weight_wgsl(conv.weight.val().into_primitive().tensor(), stride)
}

fn custom_forward(
    conv: &ConvTranspose1d<B>,
    packed_weight: &PackedConvTranspose1dWeight,
    input: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let bias = conv
        .bias
        .as_ref()
        .expect("benchmark ConvTranspose1d must have a bias");
    let output = conv_transpose1d_polyphase_wgsl(
        input.into_primitive().tensor(),
        packed_weight,
        bias.val().into_primitive().tensor(),
    );
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn synchronize(tensor: Tensor<B, 3>) {
    let _ = tensor.slice([0..1, 0..1, 0..1]).into_data();
}

fn synchronize_packed_weight(packed: &PackedConvTranspose1dWeight) {
    let tensor: Tensor<B, 4> = Tensor::from_primitive(TensorPrimitive::Float(packed.tensor()));
    let _ = tensor.slice([0..1, 0..1, 0..1, 0..1]).into_data();
}

fn measure<F>(warmup: usize, iterations: usize, trials: usize, mut operation: F) -> Timing
where
    F: FnMut() -> Tensor<B, 3>,
{
    let mut warmup_output = None;
    for _ in 0..warmup {
        warmup_output = Some(operation());
    }
    synchronize(warmup_output.expect("warmup count must be non-zero"));

    let mut samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let started = Instant::now();
        let mut output = None;
        for _ in 0..iterations {
            output = Some(operation());
        }
        synchronize(output.expect("iteration count must be non-zero"));
        samples.push(started.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64);
    }
    samples.sort_by(f64::total_cmp);
    Timing {
        median_us: samples[samples.len() / 2],
        min_us: samples[0],
        max_us: samples[samples.len() - 1],
    }
}

fn max_abs_diff(lhs: Tensor<B, 3>, rhs: Tensor<B, 3>) -> Result<f32, Box<dyn Error>> {
    let values = (lhs - rhs).abs().max().into_data().to_vec::<f32>()?;
    let maximum = values
        .first()
        .copied()
        .ok_or_else(|| io::Error::other("maximum reduction returned no values"))?;
    if maximum.is_finite() {
        Ok(maximum)
    } else {
        Err(io::Error::other(format!("non-finite max_abs result {maximum}")).into())
    }
}

fn max_abs_diff_cpu(actual: Tensor<B, 3>, expected: &[f32]) -> Result<f32, Box<dyn Error>> {
    let actual = actual.into_data().to_vec::<f32>()?;
    if actual.len() != expected.len() {
        return Err(io::Error::other(format!(
            "CPU reference length mismatch: actual {}, expected {}",
            actual.len(),
            expected.len()
        ))
        .into());
    }
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max))
}

fn deterministic_values(length: usize, modulus: usize, multiplier: usize, scale: f32) -> Vec<f32> {
    let centre = (modulus / 2) as f32;
    (0..length)
        .map(|index| (((index % modulus) * multiplier) % modulus) as f32 - centre)
        .map(|value| value * scale)
        .collect()
}

/// Independent scalar scatter definition for PyTorch-compatible ConvTranspose1d.
#[allow(clippy::too_many_arguments)]
fn cpu_reference_conv_transpose1d(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    input_channels: usize,
    output_channels: usize,
    input_length: usize,
    stride: usize,
) -> Vec<f32> {
    let kernel_size = 2 * stride;
    let padding = stride / 2;
    let output_length = input_length * stride;
    let mut output = vec![0.0_f32; output_channels * output_length];
    for output_channel in 0..output_channels {
        output[output_channel * output_length..(output_channel + 1) * output_length]
            .fill(bias[output_channel]);
    }

    for input_channel in 0..input_channels {
        for input_time in 0..input_length {
            let input_value = input[input_channel * input_length + input_time];
            for output_channel in 0..output_channels {
                for kernel_index in 0..kernel_size {
                    let output_time = input_time as isize * stride as isize + kernel_index as isize
                        - padding as isize;
                    if (0..output_length as isize).contains(&output_time) {
                        let weight_index = (input_channel * output_channels + output_channel)
                            * kernel_size
                            + kernel_index;
                        let output_index = output_channel * output_length + output_time as usize;
                        output[output_index] =
                            input_value.mul_add(weight[weight_index], output[output_index]);
                    }
                }
            }
        }
    }
    output
}

fn validate_cpu_reference(device: &<B as Backend>::Device) -> Result<(), Box<dyn Error>> {
    let input_values =
        deterministic_values(CPU_INPUT_CHANNELS * CPU_INPUT_LENGTH, 29, 11, 1.0 / 16.0);
    let bias_values = deterministic_values(CPU_OUTPUT_CHANNELS, 17, 5, 1.0 / 128.0);

    for stride in [
        ConvTranspose1dStride::Two,
        ConvTranspose1dStride::Eight,
        ConvTranspose1dStride::Ten,
        ConvTranspose1dStride::Twelve,
    ] {
        let kernel_size = 2 * stride.value();
        let weight_values = deterministic_values(
            CPU_INPUT_CHANNELS * CPU_OUTPUT_CHANNELS * kernel_size,
            31,
            7,
            1.0 / 256.0,
        );
        let input = Tensor::<B, 3>::from_data(
            TensorData::new(
                input_values.clone(),
                [1, CPU_INPUT_CHANNELS, CPU_INPUT_LENGTH],
            ),
            device,
        );
        let weight = Tensor::<B, 3>::from_data(
            TensorData::new(
                weight_values.clone(),
                [CPU_INPUT_CHANNELS, CPU_OUTPUT_CHANNELS, kernel_size],
            ),
            device,
        );
        let bias = Tensor::<B, 1>::from_data(
            TensorData::new(bias_values.clone(), [CPU_OUTPUT_CHANNELS]),
            device,
        );
        let conv = conv_with_data(
            device,
            CPU_INPUT_CHANNELS,
            CPU_OUTPUT_CHANNELS,
            stride,
            weight,
            bias,
        );
        let packed = pack_weight(&conv, stride);
        let expected = cpu_reference_conv_transpose1d(
            &input_values,
            &weight_values,
            &bias_values,
            CPU_INPUT_CHANNELS,
            CPU_OUTPUT_CHANNELS,
            CPU_INPUT_LENGTH,
            stride.value(),
        );
        let burn_error = max_abs_diff_cpu(conv.forward(input.clone()), &expected)?;
        let custom_error = max_abs_diff_cpu(custom_forward(&conv, &packed, input), &expected)?;
        if burn_error > CPU_TOLERANCE || custom_error > CPU_TOLERANCE {
            return Err(io::Error::other(format!(
                "CPU-reference s={}: Burn max_abs={burn_error:.3e}, custom max_abs={custom_error:.3e}, tolerance={CPU_TOLERANCE:.3e}",
                stride.value()
            ))
            .into());
        }
        println!(
            "CPU reference Cin={CPU_INPUT_CHANNELS} Cout={CPU_OUTPUT_CHANNELS} \
             Lin={CPU_INPUT_LENGTH} s={}: Burn max_abs={burn_error:.3e}, \
             polyphase max_abs={custom_error:.3e}",
            stride.value()
        );
    }
    Ok(())
}

fn nominal_macs(case: ConvTransposeCase) -> u128 {
    case.output_length() as u128 * case.output_channels as u128 * (2 * case.input_channels) as u128
}

fn useful_macs(case: ConvTransposeCase) -> u128 {
    case.stride.value() as u128
        * case.output_channels as u128
        * case.input_channels as u128
        * (2 * case.input_length - 1) as u128
}

fn gflops(macs: u128, time_us: f64) -> f64 {
    2.0 * macs as f64 / (time_us * 1_000.0)
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn memory_accounting(case: ConvTransposeCase) -> MemoryAccounting {
    let scalar_bytes = core::mem::size_of::<f32>();
    let input_bytes = case.input_channels * case.input_length * scalar_bytes;
    let weight_bytes =
        case.input_channels * case.output_channels * case.kernel_size() * scalar_bytes;
    let bias_bytes = case.output_channels * scalar_bytes;
    let output_bytes = case.output_channels * case.output_length() * scalar_bytes;
    let columns_bytes =
        case.output_channels * case.kernel_size() * case.input_length * scalar_bytes;
    MemoryAccounting {
        common_resident_bytes: input_bytes + weight_bytes + bias_bytes + output_bytes,
        burn_weight_reorder_bytes: weight_bytes,
        burn_columns_bytes: columns_bytes,
        custom_packed_weight_bytes: weight_bytes,
    }
}

fn query_total_vram_mib(nvml_index: usize) -> Result<u64, Box<dyn Error>> {
    let output = Command::new("nvidia-smi")
        .arg(format!("--id={nvml_index}"))
        .arg("--query-gpu=memory.used")
        .arg("--format=csv,noheader,nounits")
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "nvidia-smi failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
        .into());
    }
    let value = String::from_utf8(output.stdout)?;
    Ok(value.trim().parse::<u64>()?)
}

fn report_nvml(label: &str, nvml_index: Option<usize>, baseline: Option<u64>) {
    let Some(index) = nvml_index else {
        return;
    };
    match query_total_vram_mib(index) {
        Ok(used) => match baseline {
            Some(baseline) => println!(
                "NVML GPU {index} {label}: total_used={used} MiB, delta_from_initialized={:+} MiB",
                used as i64 - baseline as i64
            ),
            None => println!("NVML GPU {index} {label}: total_used={used} MiB"),
        },
        Err(error) => eprintln!("warning: could not query NVML GPU {index} at {label}: {error}"),
    }
}

fn benchmark_case(
    device: &<B as Backend>::Device,
    case_index: usize,
    case: ConvTransposeCase,
    args: &Args,
    nvml_baseline: Option<u64>,
) -> Result<(), Box<dyn Error>> {
    let conv = ConvTranspose1dConfig::new(
        [case.input_channels, case.output_channels],
        case.kernel_size(),
    )
    .with_stride(case.stride.value())
    .with_padding(case.stride.value() / 2)
    .with_padding_out(0)
    .with_dilation(1)
    .with_groups(1)
    .with_bias(true)
    .init::<B>(device);
    let input = Tensor::<B, 3>::random(
        [1, case.input_channels, case.input_length],
        Distribution::Uniform(-1.0, 1.0),
        device,
    );
    let pack_started = Instant::now();
    let packed = pack_weight(&conv, case.stride);
    synchronize_packed_weight(&packed);
    let pack_wall_ms = pack_started.elapsed().as_secs_f64() * 1_000.0;

    let burn_output = conv.forward(input.clone());
    let custom_output = custom_forward(&conv, &packed, input.clone());
    let error = max_abs_diff(burn_output, custom_output)?;
    report_nvml("after correctness", args.nvml_index, nvml_baseline);

    let iterations = args.iterations.unwrap_or(case.iterations);
    let burn = measure(args.warmup, iterations, args.trials, || {
        conv.forward(input.clone())
    });
    report_nvml("after Burn trials", args.nvml_index, nvml_baseline);
    let custom = measure(args.warmup, iterations, args.trials, || {
        custom_forward(&conv, &packed, input.clone())
    });
    report_nvml("after custom trials", args.nvml_index, nvml_baseline);

    let nvml_peak = if let Some(nvml_index) = args.nvml_index {
        let before_probe = query_total_vram_mib(nvml_index)?;
        let sampler = NvmlPeakSampler::start(nvml_index, before_probe);
        let mut output = None;
        for _ in 0..VRAM_PROBE_ITERATIONS {
            output = Some(conv.forward(input.clone()));
        }
        synchronize(output.expect("VRAM probe iteration count must be non-zero"));
        let mut output = None;
        for _ in 0..VRAM_PROBE_ITERATIONS {
            output = Some(custom_forward(&conv, &packed, input.clone()));
        }
        synchronize(output.expect("VRAM probe iteration count must be non-zero"));
        let peak = sampler.finish()?.max(query_total_vram_mib(nvml_index)?);
        let after = query_total_vram_mib(nvml_index)?;
        Some((peak, after))
    } else {
        None
    };

    let nominal = nominal_macs(case);
    let useful = useful_macs(case);
    let memory = memory_accounting(case);
    println!(
        "case={case_index} Cin={:4} Cout={:3} Lin={:5} Lout={:5} \
         s={:2} k={:2} p={} op=0: max_abs={error:.3e}, one-time pack cold wall={pack_wall_ms:.3} ms\n\
         nominal MAC={:.6}G, boundary-valid MAC={:.6}G\n\
         current Burn:    median={:10.1} us [{:10.1}, {:10.1}], {:8.2} GFLOP/s\n\
         polyphase WGSL: median={:10.1} us [{:10.1}, {:10.1}], {:8.2} GFLOP/s, speedup={:5.2}x\n\
         VRAM common={:.3} MiB; Burn known workspace lower bound: weight reorder={:.3} MiB + columns={:.3} MiB; \
         custom persistent packed={:.3} MiB (reported object={:.3} MiB), global dispatch workspace=0 MiB, \
         production shared: case0 Cin32=12.125 KiB when supported, portable fallback=4.062 KiB/workgroup",
        case.input_channels,
        case.output_channels,
        case.input_length,
        case.output_length(),
        case.stride.value(),
        case.kernel_size(),
        case.stride.value() / 2,
        nominal as f64 / 1.0e9,
        useful as f64 / 1.0e9,
        burn.median_us,
        burn.min_us,
        burn.max_us,
        gflops(nominal, burn.median_us),
        custom.median_us,
        custom.min_us,
        custom.max_us,
        gflops(nominal, custom.median_us),
        burn.median_us / custom.median_us,
        mib(memory.common_resident_bytes),
        mib(memory.burn_weight_reorder_bytes),
        mib(memory.burn_columns_bytes),
        mib(memory.custom_packed_weight_bytes),
        mib(packed.bytes()),
    );
    if let (Some(baseline), Some((peak, after))) = (nvml_baseline, nvml_peak) {
        println!(
            "NVML peak probe: before={baseline} MiB, peak={peak} MiB, delta={:+} MiB, after={after} MiB (timed steady completed before polling)",
            peak as i64 - baseline as i64
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let device = args
        .adapter_index
        .map(WgpuDevice::DiscreteGpu)
        .unwrap_or(WgpuDevice::DefaultDevice);
    init_setup::<AutoGraphicsApi>(&device, Default::default());
    B::seed(&device, 0);
    let nvml_baseline = args
        .nvml_index
        .map(query_total_vram_mib)
        .transpose()
        .map_err(|error| io::Error::other(format!("initial NVML query failed: {error}")))?;

    println!(
        "Polyphase DACVAE ConvTranspose1d device={device:?}, warmup={}, trials={}, iterations={}, cases={}",
        args.warmup,
        args.trials,
        args.iterations
            .map_or_else(|| "per-shape".to_owned(), |value| value.to_string()),
        args.case_index
            .map_or_else(|| "all 4".to_owned(), |index| format!("case {index}"))
    );
    report_nvml("initialized baseline", args.nvml_index, None);
    validate_cpu_reference(&device)?;

    let cases = PRODUCTION_CASES
        .into_iter()
        .enumerate()
        .filter(|(index, _)| args.case_index.is_none_or(|selected| selected == *index));
    for (index, case) in cases {
        benchmark_case(&device, index, case, &args, nvml_baseline)?;
    }
    Ok(())
}
