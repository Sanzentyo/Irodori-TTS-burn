//! Exact-shape benchmark for ConvTranspose1d col2im weight layouts.
//!
//! Released decoder cases 1--3 currently use Burn's transposed-convolution
//! col2im route. Burn converts `[Cin, Cout, kernel]` weights to contiguous
//! row-major `[Cout * kernel, Cin]` on every forward, runs its tuned batched
//! GEMM, then applies a generic col2im kernel. This isolated benchmark compares
//! that complete route with:
//!
//! - the existing direct polyphase candidate;
//! - a one-time row-major weight copy plus tuned GEMM and exact 1D finalizer;
//! - the production checkpoint-allocation zero-copy column view, tuned GEMM,
//!   and exact finalizer.
//!
//! The full cached-column path calls the production helper directly. Its GEMM
//! and finalizer stages are also timed separately against the retained
//! row-major experiment.
//!
//! Once explicitly registered, use one fresh process per shape for meaningful
//! cold-pack and NVML numbers:
//! `cargo run --release --bin bench_conv_transpose1d_cached_col2im -- 0 --nvml-index 1 --case 1`.

use std::{
    error::Error,
    io,
    process::Command,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use burn::{
    backend::wgpu::{
        RuntimeOptions, WgpuDevice, WgpuRuntime, graphics::AutoGraphicsApi, init_setup,
    },
    nn::conv::{ConvTranspose1d, ConvTranspose1dConfig},
    tensor::{Distribution, Tensor, TensorPrimitive, backend::Backend},
};
use burn_cubecl::kernel::into_contiguous_aligned;
use cubecl::prelude::Runtime;
use irodori_tts_wgpu::{
    WgpuRaw,
    kernels::{
        conv_transpose1d_cached_col2im::{
            CachedCol2ImCase, conv_transpose1d_cached_col2im_wgsl, finalize_cached_col2im_wgsl,
        },
        conv_transpose1d_polyphase::{
            ConvTranspose1dStride, PackedConvTranspose1dWeight, conv_transpose1d_polyphase_wgsl,
            pack_conv_transpose1d_weight_wgsl,
        },
    },
};

type B = WgpuRaw;

const DEFAULT_WARMUP: usize = 10;
const DEFAULT_TRIALS: usize = 5;
const SEED: u64 = 0;
const F32_BYTES: usize = size_of::<f32>();
const VRAM_PROBE_ITERATIONS: usize = 10;

#[derive(Clone, Copy, Debug)]
struct BenchCase {
    index: usize,
    candidate: CachedCol2ImCase,
    polyphase_stride: ConvTranspose1dStride,
    default_iterations: usize,
    pytorch_strict_us: f64,
    pytorch_official_default_us: f64,
}

impl BenchCase {
    const fn input_channels(self) -> usize {
        self.candidate.input_channels()
    }

    const fn output_channels(self) -> usize {
        self.candidate.output_channels()
    }

    const fn input_length(self) -> usize {
        self.candidate.input_length()
    }

    const fn output_length(self) -> usize {
        self.candidate.output_length()
    }

    const fn stride(self) -> usize {
        self.candidate.stride()
    }

    const fn kernel_size(self) -> usize {
        self.candidate.kernel_size()
    }

    const fn weight_elements(self) -> usize {
        self.input_channels() * self.output_channels() * self.kernel_size()
    }
}

const CASES: [BenchCase; 3] = [
    BenchCase {
        index: 1,
        candidate: CachedCol2ImCase::Case1,
        polyphase_stride: ConvTranspose1dStride::Ten,
        default_iterations: 50,
        pytorch_strict_us: 1_187.12,
        pytorch_official_default_us: 698.25,
    },
    BenchCase {
        index: 2,
        candidate: CachedCol2ImCase::Case2,
        polyphase_stride: ConvTranspose1dStride::Eight,
        default_iterations: 10,
        pytorch_strict_us: 2_165.34,
        pytorch_official_default_us: 1_570.26,
    },
    BenchCase {
        index: 3,
        candidate: CachedCol2ImCase::Case3,
        polyphase_stride: ConvTranspose1dStride::Two,
        default_iterations: 10,
        pytorch_strict_us: 1_509.31,
        pytorch_official_default_us: 1_210.57,
    },
];

#[derive(Debug)]
struct Args {
    adapter_index: usize,
    nvml_index: Option<usize>,
    case_index: Option<usize>,
    warmup: usize,
    iterations: Option<usize>,
    trials: usize,
}

#[derive(Clone, Copy, Debug)]
struct Timing {
    median_us: f64,
    min_us: f64,
    max_us: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Comparison {
    elements: usize,
    mismatched_bits: usize,
    max_abs: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CachedLayout {
    RowMajor,
    ColumnMajor,
}

impl CachedLayout {
    const fn label(self) -> &'static str {
        match self {
            Self::RowMajor => "cached-row",
            Self::ColumnMajor => "cached-col",
        }
    }
}

#[derive(Clone)]
struct CachedWeight {
    tensor: Tensor<B, 2>,
    layout: CachedLayout,
}

#[derive(Clone, Copy, Debug)]
enum PackPath {
    Polyphase,
    CachedRow,
    CachedColumn,
}

impl PackPath {
    const fn label(self) -> &'static str {
        match self {
            Self::Polyphase => "polyphase",
            Self::CachedRow => "cached-row",
            Self::CachedColumn => "cached-col-zero-copy",
        }
    }
}

const PACK_PATHS: [PackPath; 3] = [
    PackPath::Polyphase,
    PackPath::CachedRow,
    PackPath::CachedColumn,
];

enum PackedArtifact {
    Polyphase(PackedConvTranspose1dWeight),
    Cached(CachedWeight),
}

struct PackMeasurement {
    artifact: PackedArtifact,
    cold_us: f64,
    steady: Timing,
}

#[derive(Clone, Copy, Debug)]
enum FullPath {
    Burn,
    Polyphase,
    CachedRow,
    CachedColumn,
}

impl FullPath {
    const fn label(self) -> &'static str {
        match self {
            Self::Burn => "current-burn-full",
            Self::Polyphase => "old-polyphase",
            Self::CachedRow => "cached-row-full",
            Self::CachedColumn => "cached-col-full",
        }
    }
}

const FULL_PATHS: [FullPath; 4] = [
    FullPath::Burn,
    FullPath::Polyphase,
    FullPath::CachedRow,
    FullPath::CachedColumn,
];

#[derive(Clone, Copy, Debug)]
enum GemmPath {
    CachedRow,
    CachedColumn,
}

impl GemmPath {
    const fn label(self) -> &'static str {
        match self {
            Self::CachedRow => "cached-row-gemm",
            Self::CachedColumn => "cached-col-gemm",
        }
    }
}

const GEMM_PATHS: [GemmPath; 2] = [GemmPath::CachedRow, GemmPath::CachedColumn];

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
struct NvmlPeakSampler {
    stop: Arc<AtomicBool>,
    peak_mib: Arc<AtomicU64>,
    worker: Option<JoinHandle<()>>,
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
            worker: Some(worker),
        }
    }

    fn finish(mut self) -> Result<u64, Box<dyn Error>> {
        self.stop.store(true, Ordering::Relaxed);
        let worker = self
            .worker
            .take()
            .ok_or_else(|| io::Error::other("NVML sampler worker was already consumed"))?;
        worker
            .join()
            .map_err(|_| io::Error::other("NVML sampler thread panicked"))?;
        Ok(self.peak_mib.load(Ordering::Relaxed))
    }
}

impl Drop for NvmlPeakSampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn usage() -> &'static str {
    "usage: bench_conv_transpose1d_cached_col2im <adapter-index> \
     [--nvml-index N] [--case 1|2|3] [--warmup N] \
     [--iterations N] [--trials N]"
}

fn next_usize(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
    require_positive: bool,
) -> Result<usize, Box<dyn Error>> {
    let text = arguments
        .next()
        .ok_or_else(|| io::Error::other(format!("{option} requires a value")))?;
    let value = text.parse::<usize>().map_err(|error| {
        io::Error::other(format!("invalid value {text:?} for {option}: {error}"))
    })?;
    if require_positive && value == 0 {
        return Err(io::Error::other(format!("{option} must be greater than zero")).into());
    }
    Ok(value)
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut adapter_index = None;
    let mut nvml_index = None;
    let mut case_index = None;
    let mut warmup = DEFAULT_WARMUP;
    let mut iterations = None;
    let mut trials = DEFAULT_TRIALS;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--nvml-index" => nvml_index = Some(next_usize(&mut arguments, "--nvml-index", false)?),
            "--case" => case_index = Some(next_usize(&mut arguments, "--case", true)?),
            "--warmup" => warmup = next_usize(&mut arguments, "--warmup", true)?,
            "--iterations" => iterations = Some(next_usize(&mut arguments, "--iterations", true)?),
            "--trials" => trials = next_usize(&mut arguments, "--trials", true)?,
            "--help" | "-h" => {
                println!("{}", usage());
                std::process::exit(0);
            }
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
    if let Some(index) = case_index
        && !(1..=3).contains(&index)
    {
        return Err(io::Error::other(format!("--case must be 1, 2, or 3; got {index}")).into());
    }
    Ok(Args {
        adapter_index: adapter_index
            .ok_or_else(|| io::Error::other(format!("missing adapter index; {}", usage())))?,
        nvml_index,
        case_index,
        warmup,
        iterations,
        trials,
    })
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
    Ok(String::from_utf8(output.stdout)?.trim().parse::<u64>()?)
}

fn report_nvml(label: &str, index: Option<usize>, baseline: Option<u64>) {
    let Some(index) = index else {
        return;
    };
    match query_total_vram_mib(index) {
        Ok(used) => match baseline {
            Some(baseline) => println!(
                "NVML GPU {index} {label}: total_used={used} MiB, delta={:+} MiB",
                used as i64 - baseline as i64
            ),
            None => println!("NVML GPU {index} {label}: total_used={used} MiB"),
        },
        Err(error) => eprintln!("warning: NVML query failed at {label}: {error}"),
    }
}

fn create_conv(device: &<B as Backend>::Device, case: BenchCase) -> ConvTranspose1d<B> {
    ConvTranspose1dConfig::new(
        [case.input_channels(), case.output_channels()],
        case.kernel_size(),
    )
    .with_stride(case.stride())
    .with_padding(case.stride() / 2)
    .with_padding_out(0)
    .with_dilation(1)
    .with_groups(1)
    .with_bias(true)
    .init::<B>(device)
}

fn validate_source_weight(
    conv: &ConvTranspose1d<B>,
    case: BenchCase,
) -> Result<(), Box<dyn Error>> {
    let source = conv.weight.val().into_primitive().tensor();
    if source.dtype != burn::tensor::DType::F32 {
        return Err(io::Error::other(format!(
            "source weight must be f32, got {}",
            source.dtype.name()
        ))
        .into());
    }
    if source.meta.num_dims() != 3 || !source.is_contiguous() {
        return Err(io::Error::other(format!(
            "source weight must be contiguous rank-3, shape={:?}, strides={:?}",
            source.meta.shape(),
            source.meta.strides()
        ))
        .into());
    }
    let expected = [
        case.input_channels(),
        case.output_channels(),
        case.kernel_size(),
    ];
    if source.meta.shape().dims::<3>() != expected {
        return Err(io::Error::other(format!(
            "source weight shape mismatch: expected {expected:?}, got {:?}",
            source.meta.shape()
        ))
        .into());
    }
    let weight_bytes = source
        .meta
        .num_elements()
        .checked_mul(F32_BYTES)
        .ok_or_else(|| io::Error::other("source weight byte count overflow"))?;
    let weight_bytes_u64 = u64::try_from(weight_bytes)
        .map_err(|_| io::Error::other("source weight byte count exceeds u64"))?;
    let page_limit = source.client.properties().memory.max_page_size;
    if weight_bytes_u64 > page_limit {
        return Err(io::Error::other(format!(
            "source weight requires {weight_bytes} bytes, device page limit is {page_limit}"
        ))
        .into());
    }
    Ok(())
}

fn prepare_cached_weight(
    conv: &ConvTranspose1d<B>,
    case: BenchCase,
    layout: CachedLayout,
) -> Result<CachedWeight, Box<dyn Error>> {
    validate_source_weight(conv, case)?;
    let rows = case.candidate.columns_rows();
    let column_view: Tensor<B, 2> = conv
        .weight
        .val()
        .reshape([case.input_channels(), rows])
        .transpose();
    let tensor = match layout {
        CachedLayout::ColumnMajor => column_view,
        CachedLayout::RowMajor => Tensor::from_primitive(TensorPrimitive::Float(
            into_contiguous_aligned(column_view.into_primitive().tensor()),
        )),
    };
    validate_cached_weight(&tensor, case, layout)?;
    Ok(CachedWeight { tensor, layout })
}

fn validate_cached_weight(
    weight: &Tensor<B, 2>,
    case: BenchCase,
    layout: CachedLayout,
) -> Result<(), Box<dyn Error>> {
    let raw = weight.clone().into_primitive().tensor();
    if raw.dtype != burn::tensor::DType::F32 || raw.meta.num_dims() != 2 {
        return Err(
            io::Error::other(format!("{} weight must be rank-2 f32", layout.label())).into(),
        );
    }
    let expected_shape = [case.candidate.columns_rows(), case.input_channels()];
    if raw.meta.shape().dims::<2>() != expected_shape {
        return Err(io::Error::other(format!(
            "{} shape mismatch: expected {expected_shape:?}, got {:?}",
            layout.label(),
            raw.meta.shape()
        ))
        .into());
    }
    let expected_strides = match layout {
        CachedLayout::RowMajor => [case.input_channels(), 1],
        CachedLayout::ColumnMajor => [1, case.candidate.columns_rows()],
    };
    if &raw.meta.strides()[..] != expected_strides.as_slice() {
        return Err(io::Error::other(format!(
            "{} stride mismatch: expected {expected_strides:?}, got {:?}",
            layout.label(),
            raw.meta.strides()
        ))
        .into());
    }
    if raw.is_contiguous() != (layout == CachedLayout::RowMajor) {
        return Err(io::Error::other(format!(
            "{} contiguity does not match declared layout",
            layout.label()
        ))
        .into());
    }
    Ok(())
}

fn validate_input_and_device(
    input: &Tensor<B, 3>,
    weight: &CachedWeight,
    case: BenchCase,
) -> Result<(), Box<dyn Error>> {
    let input_raw = input.clone().into_primitive().tensor();
    let weight_raw = weight.tensor.clone().into_primitive().tensor();
    if input_raw.dtype != burn::tensor::DType::F32
        || input_raw.meta.num_dims() != 3
        || !input_raw.is_contiguous()
    {
        return Err(io::Error::other(format!(
            "input must be contiguous rank-3 f32, got shape={:?} strides={:?} dtype={}",
            input_raw.meta.shape(),
            input_raw.meta.strides(),
            input_raw.dtype.name()
        ))
        .into());
    }
    let expected = [1, case.input_channels(), case.input_length()];
    if input_raw.meta.shape().dims::<3>() != expected {
        return Err(io::Error::other(format!(
            "input shape mismatch: expected {expected:?}, got {:?}",
            input_raw.meta.shape()
        ))
        .into());
    }
    if input_raw.device != weight_raw.device {
        return Err(io::Error::other(format!(
            "input/weight device mismatch: {:?} != {:?}",
            input_raw.device, weight_raw.device
        ))
        .into());
    }
    let properties = input_raw.client.properties();
    let page_limit = properties.memory.max_page_size;
    let buffers = [
        ("input", input_raw.meta.num_elements()),
        ("weight", weight_raw.meta.num_elements()),
        ("columns", case.candidate.columns_elements()),
    ];
    for (label, elements) in buffers {
        let bytes = elements
            .checked_mul(F32_BYTES)
            .ok_or_else(|| io::Error::other(format!("{label} byte count overflow")))?;
        let bytes_u64 = u64::try_from(bytes)
            .map_err(|_| io::Error::other(format!("{label} byte count exceeds u64")))?;
        if bytes_u64 > page_limit {
            return Err(io::Error::other(format!(
                "{label} requires {bytes} bytes, device page limit is {page_limit}"
            ))
            .into());
        }
    }
    validate_cached_weight(&weight.tensor, case, weight.layout)
}

fn cached_columns(
    input: Tensor<B, 3>,
    weight: &CachedWeight,
    case: BenchCase,
) -> Result<Tensor<B, 2>, Box<dyn Error>> {
    validate_input_and_device(&input, weight, case)?;
    let columns = weight
        .tensor
        .clone()
        .unsqueeze::<3>()
        .matmul(input)
        .reshape([case.candidate.columns_rows(), case.input_length()]);
    Ok(Tensor::from_primitive(TensorPrimitive::Float(
        into_contiguous_aligned(columns.into_primitive().tensor()),
    )))
}

fn validate_bias_for_weight(
    conv: &ConvTranspose1d<B>,
    weight: &CachedWeight,
    case: BenchCase,
) -> Result<(), Box<dyn Error>> {
    let bias = conv
        .bias
        .as_ref()
        .ok_or_else(|| io::Error::other("benchmark ConvTranspose1d requires bias"))?;
    let bias_raw = bias.val().into_primitive().tensor();
    let weight_raw = weight.tensor.clone().into_primitive().tensor();
    if bias_raw.dtype != burn::tensor::DType::F32
        || bias_raw.meta.num_dims() != 1
        || !bias_raw.is_contiguous()
    {
        return Err(io::Error::other(format!(
            "bias must be contiguous rank-1 f32, got shape={:?} strides={:?} dtype={}",
            bias_raw.meta.shape(),
            bias_raw.meta.strides(),
            bias_raw.dtype.name()
        ))
        .into());
    }
    let expected = [case.output_channels()];
    if bias_raw.meta.shape().dims::<1>() != expected {
        return Err(io::Error::other(format!(
            "bias shape mismatch: expected {expected:?}, got {:?}",
            bias_raw.meta.shape()
        ))
        .into());
    }
    if bias_raw.device != weight_raw.device {
        return Err(io::Error::other(format!(
            "bias/weight device mismatch: {:?} != {:?}",
            bias_raw.device, weight_raw.device
        ))
        .into());
    }
    Ok(())
}

fn cached_finalize(
    conv: &ConvTranspose1d<B>,
    columns: Tensor<B, 2>,
    case: BenchCase,
) -> Result<Tensor<B, 3>, Box<dyn Error>> {
    let bias = conv
        .bias
        .as_ref()
        .ok_or_else(|| io::Error::other("benchmark ConvTranspose1d requires bias"))?;
    let output = finalize_cached_col2im_wgsl(
        columns.into_primitive().tensor(),
        bias.val().into_primitive().tensor(),
        case.candidate,
    )?;
    Ok(Tensor::from_primitive(TensorPrimitive::Float(output)))
}

fn cached_forward(
    conv: &ConvTranspose1d<B>,
    input: Tensor<B, 3>,
    weight: &CachedWeight,
    case: BenchCase,
) -> Result<Tensor<B, 3>, Box<dyn Error>> {
    validate_bias_for_weight(conv, weight, case)?;
    cached_finalize(conv, cached_columns(input, weight, case)?, case)
}

fn production_cached_column_forward(
    conv: &ConvTranspose1d<B>,
    input: Tensor<B, 3>,
    case: BenchCase,
) -> Result<Tensor<B, 3>, Box<dyn Error>> {
    let bias = conv
        .bias
        .as_ref()
        .ok_or_else(|| io::Error::other("benchmark ConvTranspose1d requires bias"))?;
    let output = conv_transpose1d_cached_col2im_wgsl(
        input.into_primitive().tensor(),
        conv.weight.val().into_primitive().tensor(),
        bias.val().into_primitive().tensor(),
        case.candidate,
    )?;
    Ok(Tensor::from_primitive(TensorPrimitive::Float(output)))
}

fn polyphase_forward(
    conv: &ConvTranspose1d<B>,
    input: Tensor<B, 3>,
    packed: &PackedConvTranspose1dWeight,
) -> Result<Tensor<B, 3>, Box<dyn Error>> {
    let bias = conv
        .bias
        .as_ref()
        .ok_or_else(|| io::Error::other("benchmark ConvTranspose1d requires bias"))?;
    let output = conv_transpose1d_polyphase_wgsl(
        input.into_primitive().tensor(),
        packed,
        bias.val().into_primitive().tensor(),
    );
    Ok(Tensor::from_primitive(TensorPrimitive::Float(output)))
}

fn sync_rank1(tensor: Tensor<B, 1>) {
    let _ = tensor.narrow(0, 0, 1).into_data();
}

fn sync_rank2(tensor: Tensor<B, 2>) {
    let _ = tensor.slice([0..1, 0..1]).into_data();
}

fn sync_rank3(tensor: Tensor<B, 3>) {
    let _ = tensor.slice([0..1, 0..1, 0..1]).into_data();
}

fn sync_rank4(tensor: Tensor<B, 4>) {
    let _ = tensor.slice([0..1, 0..1, 0..1, 0..1]).into_data();
}

fn synchronize_artifact(artifact: &PackedArtifact) {
    match artifact {
        PackedArtifact::Polyphase(packed) => sync_rank4(Tensor::from_primitive(
            TensorPrimitive::Float(packed.tensor()),
        )),
        PackedArtifact::Cached(weight) => sync_rank2(weight.tensor.clone()),
    }
}

fn pack_artifact(
    conv: &ConvTranspose1d<B>,
    case: BenchCase,
    path: PackPath,
) -> Result<PackedArtifact, Box<dyn Error>> {
    match path {
        PackPath::Polyphase => Ok(PackedArtifact::Polyphase(
            pack_conv_transpose1d_weight_wgsl(
                conv.weight.val().into_primitive().tensor(),
                case.polyphase_stride,
            ),
        )),
        PackPath::CachedRow => Ok(PackedArtifact::Cached(prepare_cached_weight(
            conv,
            case,
            CachedLayout::RowMajor,
        )?)),
        PackPath::CachedColumn => Ok(PackedArtifact::Cached(prepare_cached_weight(
            conv,
            case,
            CachedLayout::ColumnMajor,
        )?)),
    }
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

fn measure_pack(
    conv: &ConvTranspose1d<B>,
    case: BenchCase,
    path: PackPath,
    trials: usize,
) -> Result<PackMeasurement, Box<dyn Error>> {
    let started = Instant::now();
    let artifact = pack_artifact(conv, case, path)?;
    if !matches!(path, PackPath::CachedColumn) {
        synchronize_artifact(&artifact);
    }
    let cold_us = started.elapsed().as_secs_f64() * 1_000_000.0;

    let mut samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let started = Instant::now();
        let sample = pack_artifact(conv, case, path)?;
        if !matches!(path, PackPath::CachedColumn) {
            synchronize_artifact(&sample);
        }
        samples.push(started.elapsed().as_secs_f64() * 1_000_000.0);
    }
    Ok(PackMeasurement {
        artifact,
        cold_us,
        steady: summarize(&samples),
    })
}

fn comparison(expected: &[f32], actual: &[f32]) -> Result<Comparison, Box<dyn Error>> {
    if expected.len() != actual.len() {
        return Err(io::Error::other(format!(
            "comparison length mismatch: expected {}, actual {}",
            expected.len(),
            actual.len()
        ))
        .into());
    }
    expected
        .iter()
        .zip(actual)
        .try_fold(Comparison::default(), |mut result, (&lhs, &rhs)| {
            if !lhs.is_finite() || !rhs.is_finite() {
                return Err(io::Error::other(format!(
                    "non-finite comparison pair: {lhs:?}, {rhs:?}"
                )));
            }
            result.elements += 1;
            result.mismatched_bits += usize::from(lhs.to_bits() != rhs.to_bits());
            result.max_abs = result.max_abs.max((lhs - rhs).abs());
            Ok(result)
        })
        .map_err(|error| Box::new(error) as Box<dyn Error>)
}

fn compare_rank2(
    expected: Tensor<B, 2>,
    actual: Tensor<B, 2>,
) -> Result<Comparison, Box<dyn Error>> {
    comparison(
        &expected.into_data().to_vec::<f32>()?,
        &actual.into_data().to_vec::<f32>()?,
    )
}

fn compare_rank3(expected: &[f32], actual: Tensor<B, 3>) -> Result<Comparison, Box<dyn Error>> {
    comparison(expected, &actual.into_data().to_vec::<f32>()?)
}

fn print_comparison(label: &str, result: Comparison) {
    println!(
        "  correctness {label:<24} elements={:>9} bit_mismatch={:>9} max_abs={:.9e}",
        result.elements, result.mismatched_bits, result.max_abs
    );
}

struct FullPathContext<'a> {
    conv: &'a ConvTranspose1d<B>,
    input: &'a Tensor<B, 3>,
    polyphase: &'a PackedConvTranspose1dWeight,
    row: &'a CachedWeight,
    case: BenchCase,
}

impl FullPathContext<'_> {
    fn run(&self, path: FullPath) -> Result<Tensor<B, 3>, Box<dyn Error>> {
        match path {
            FullPath::Burn => Ok(self.conv.forward(self.input.clone())),
            FullPath::Polyphase => polyphase_forward(self.conv, self.input.clone(), self.polyphase),
            FullPath::CachedRow => {
                cached_forward(self.conv, self.input.clone(), self.row, self.case)
            }
            FullPath::CachedColumn => {
                production_cached_column_forward(self.conv, self.input.clone(), self.case)
            }
        }
    }
}

fn benchmark_full_paths(
    context: &FullPathContext<'_>,
    warmup: usize,
    iterations: usize,
    trials: usize,
) -> Result<[Timing; FULL_PATHS.len()], Box<dyn Error>> {
    for path in FULL_PATHS {
        let mut output = None;
        for _ in 0..warmup {
            output = Some(context.run(path)?);
        }
        sync_rank3(output.ok_or_else(|| io::Error::other("full-path warmup was empty"))?);
    }

    let mut samples: [Vec<f64>; FULL_PATHS.len()] =
        std::array::from_fn(|_| Vec::with_capacity(trials));
    for trial in 0..trials {
        for offset in 0..FULL_PATHS.len() {
            let path_index = (trial + offset) % FULL_PATHS.len();
            let path = FULL_PATHS[path_index];
            let started = Instant::now();
            let mut output = None;
            for _ in 0..iterations {
                output = Some(context.run(path)?);
            }
            sync_rank3(output.ok_or_else(|| io::Error::other("full-path trial was empty"))?);
            samples[path_index]
                .push(started.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64);
        }
    }
    Ok(std::array::from_fn(|index| summarize(&samples[index])))
}

fn gemm_output(
    path: GemmPath,
    input: &Tensor<B, 3>,
    row: &CachedWeight,
    column: &CachedWeight,
    case: BenchCase,
) -> Result<Tensor<B, 2>, Box<dyn Error>> {
    match path {
        GemmPath::CachedRow => cached_columns(input.clone(), row, case),
        GemmPath::CachedColumn => cached_columns(input.clone(), column, case),
    }
}

fn benchmark_gemm_paths(
    input: &Tensor<B, 3>,
    row: &CachedWeight,
    column: &CachedWeight,
    case: BenchCase,
    warmup: usize,
    iterations: usize,
    trials: usize,
) -> Result<[Timing; GEMM_PATHS.len()], Box<dyn Error>> {
    for path in GEMM_PATHS {
        let mut output = None;
        for _ in 0..warmup {
            output = Some(gemm_output(path, input, row, column, case)?);
        }
        sync_rank2(output.ok_or_else(|| io::Error::other("GEMM warmup was empty"))?);
    }

    let mut samples: [Vec<f64>; GEMM_PATHS.len()] =
        std::array::from_fn(|_| Vec::with_capacity(trials));
    for trial in 0..trials {
        for offset in 0..GEMM_PATHS.len() {
            let path_index = (trial + offset) % GEMM_PATHS.len();
            let path = GEMM_PATHS[path_index];
            let started = Instant::now();
            let mut output = None;
            for _ in 0..iterations {
                output = Some(gemm_output(path, input, row, column, case)?);
            }
            sync_rank2(output.ok_or_else(|| io::Error::other("GEMM trial was empty"))?);
            samples[path_index]
                .push(started.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64);
        }
    }
    Ok(std::array::from_fn(|index| summarize(&samples[index])))
}

fn benchmark_finalizer(
    conv: &ConvTranspose1d<B>,
    columns: &Tensor<B, 2>,
    case: BenchCase,
    warmup: usize,
    iterations: usize,
    trials: usize,
) -> Result<Timing, Box<dyn Error>> {
    let mut output = None;
    for _ in 0..warmup {
        output = Some(cached_finalize(conv, columns.clone(), case)?);
    }
    sync_rank3(output.ok_or_else(|| io::Error::other("finalizer warmup was empty"))?);

    let mut samples = Vec::with_capacity(trials);
    for _ in 0..trials {
        let started = Instant::now();
        let mut output = None;
        for _ in 0..iterations {
            output = Some(cached_finalize(conv, columns.clone(), case)?);
        }
        sync_rank3(output.ok_or_else(|| io::Error::other("finalizer trial was empty"))?);
        samples.push(started.elapsed().as_secs_f64() * 1_000_000.0 / iterations as f64);
    }
    Ok(summarize(&samples))
}

fn break_even_requests(pack_us: f64, savings_us: f64) -> f64 {
    if savings_us <= 0.0 {
        f64::INFINITY
    } else {
        (pack_us / savings_us).ceil()
    }
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn report_static_accounting(case: BenchCase) {
    let weight_bytes = case.weight_elements() * F32_BYTES;
    let columns_bytes = case.candidate.columns_elements() * F32_BYTES;
    let output_bytes = case.candidate.output_elements() * F32_BYTES;
    let finalizer_workgroups = case.candidate.output_elements().div_ceil(256);
    println!(
        "  exact GEMM: [{},{}] @ [{},{}] -> [{},{}]; finalizer_workgroups={finalizer_workgroups}",
        case.candidate.columns_rows(),
        case.input_channels(),
        case.input_channels(),
        case.input_length(),
        case.candidate.columns_rows(),
        case.input_length(),
    );
    println!(
        "  logical VRAM: source_weight={} bytes ({:.6} MiB), Burn per-call reorder={} bytes, \
         columns={} bytes ({:.6} MiB), output={} bytes ({:.6} MiB)",
        weight_bytes,
        mib(weight_bytes),
        weight_bytes,
        columns_bytes,
        mib(columns_bytes),
        output_bytes,
        mib(output_bytes),
    );
    println!(
        "  cache delta: cached-row=+{} bytes ({:.6} MiB); cached-col=0 bytes \
         (zero-copy source view); old-poly=+{} bytes ({:.6} MiB). \
         Candidate active columns remain {:.6} MiB.",
        weight_bytes,
        mib(weight_bytes),
        weight_bytes,
        mib(weight_bytes),
        mib(columns_bytes),
    );
    println!(
        "  isolated retained-cache total (row + old-poly)={} bytes ({:.6} MiB); \
         production would retain only a per-shape winner",
        2 * weight_bytes,
        mib(2 * weight_bytes),
    );
}

fn print_timing(label: &str, timing: Timing) {
    println!(
        "  {label:<25} median={:10.3} us [{:10.3}, {:10.3}]",
        timing.median_us, timing.min_us, timing.max_us
    );
}

fn probe_vram(
    nvml_index: usize,
    context: &FullPathContext<'_>,
) -> Result<(u64, u64, u64), Box<dyn Error>> {
    let before = query_total_vram_mib(nvml_index)?;
    let sampler = NvmlPeakSampler::start(nvml_index, before);
    for path in FULL_PATHS {
        let mut output = None;
        for _ in 0..VRAM_PROBE_ITERATIONS {
            output = Some(context.run(path)?);
        }
        sync_rank3(output.ok_or_else(|| io::Error::other("VRAM probe was empty"))?);
    }
    let peak = sampler.finish()?.max(query_total_vram_mib(nvml_index)?);
    let after = query_total_vram_mib(nvml_index)?;
    Ok((before, peak, after))
}

fn benchmark_case(
    device: &WgpuDevice,
    monitor: &WgpuErrorMonitor,
    args: &Args,
    case: BenchCase,
    nvml_baseline: Option<u64>,
) -> Result<(), Box<dyn Error>> {
    let iterations = args.iterations.unwrap_or(case.default_iterations);
    println!(
        "\ncase={} Cin={} Cout={} Lin={} Lout={} stride={} kernel={} padding={} \
         warmup={} iterations={} trials={} PyTorch_strict={:.2} us official_default={:.2} us",
        case.index,
        case.input_channels(),
        case.output_channels(),
        case.input_length(),
        case.output_length(),
        case.stride(),
        case.kernel_size(),
        case.stride() / 2,
        args.warmup,
        iterations,
        args.trials,
        case.pytorch_strict_us,
        case.pytorch_official_default_us,
    );
    report_static_accounting(case);

    let conv = create_conv(device, case);
    validate_source_weight(&conv, case)?;
    let input = Tensor::<B, 3>::random(
        [1, case.input_channels(), case.input_length()],
        Distribution::Uniform(-1.0, 1.0),
        device,
    );
    sync_rank3(input.clone());
    sync_rank3(conv.weight.val());
    sync_rank1(
        conv.bias
            .as_ref()
            .ok_or_else(|| io::Error::other("benchmark ConvTranspose1d requires bias"))?
            .val(),
    );
    synchronize_and_check_wgpu(device, monitor, "case input initialization")?;

    let mut pack_measurements = Vec::with_capacity(PACK_PATHS.len());
    for path in PACK_PATHS {
        let measurement = measure_pack(&conv, case, path, args.trials)?;
        println!(
            "  pack {:<20} cold={:10.3} us steady={:10.3} us [{:10.3}, {:10.3}]",
            path.label(),
            measurement.cold_us,
            measurement.steady.median_us,
            measurement.steady.min_us,
            measurement.steady.max_us,
        );
        pack_measurements.push(measurement);
        synchronize_and_check_wgpu(device, monitor, path.label())?;
    }
    let [poly_measurement, row_measurement, column_measurement]: [PackMeasurement; 3] =
        pack_measurements
            .try_into()
            .map_err(|measurements: Vec<_>| {
                io::Error::other(format!(
                    "expected three pack measurements, got {}",
                    measurements.len()
                ))
            })?;
    let polyphase = match poly_measurement.artifact {
        PackedArtifact::Polyphase(packed) => packed,
        PackedArtifact::Cached(_) => {
            return Err(io::Error::other("polyphase pack returned cached weight").into());
        }
    };
    let row = match row_measurement.artifact {
        PackedArtifact::Cached(weight) if weight.layout == CachedLayout::RowMajor => weight,
        _ => return Err(io::Error::other("cached-row pack returned wrong artifact").into()),
    };
    let column = match column_measurement.artifact {
        PackedArtifact::Cached(weight) if weight.layout == CachedLayout::ColumnMajor => weight,
        _ => return Err(io::Error::other("cached-col pack returned wrong artifact").into()),
    };
    report_nvml("after retained packs", args.nvml_index, nvml_baseline);

    let weight_comparison = compare_rank2(row.tensor.clone(), column.tensor.clone())?;
    print_comparison("row-weight vs col-view", weight_comparison);

    let row_columns = cached_columns(input.clone(), &row, case)?;
    let column_columns = cached_columns(input.clone(), &column, case)?;
    print_comparison(
        "row-GEMM vs col-GEMM",
        compare_rank2(row_columns.clone(), column_columns.clone())?,
    );

    let burn_values = conv.forward(input.clone()).into_data().to_vec::<f32>()?;
    print_comparison(
        "Burn vs old-poly",
        compare_rank3(
            &burn_values,
            polyphase_forward(&conv, input.clone(), &polyphase)?,
        )?,
    );
    print_comparison(
        "Burn vs cached-row",
        compare_rank3(
            &burn_values,
            cached_finalize(&conv, row_columns.clone(), case)?,
        )?,
    );
    print_comparison(
        "Burn vs cached-col",
        compare_rank3(
            &burn_values,
            production_cached_column_forward(&conv, input.clone(), case)?,
        )?,
    );
    synchronize_and_check_wgpu(device, monitor, "correctness readback")?;
    report_nvml("after correctness", args.nvml_index, nvml_baseline);

    let full_context = FullPathContext {
        conv: &conv,
        input: &input,
        polyphase: &polyphase,
        row: &row,
        case,
    };
    let full = benchmark_full_paths(&full_context, args.warmup, iterations, args.trials)?;
    synchronize_and_check_wgpu(device, monitor, "full-path timings")?;
    println!("  full-path timings:");
    for (path, timing) in FULL_PATHS.into_iter().zip(full) {
        print_timing(path.label(), timing);
    }

    let gemm = benchmark_gemm_paths(
        &input,
        &row,
        &column,
        case,
        args.warmup,
        iterations,
        args.trials,
    )?;
    synchronize_and_check_wgpu(device, monitor, "GEMM timings")?;
    println!("  GEMM-only timings:");
    for (path, timing) in GEMM_PATHS.into_iter().zip(gemm) {
        print_timing(path.label(), timing);
    }

    let row_finalizer = benchmark_finalizer(
        &conv,
        &row_columns,
        case,
        args.warmup,
        iterations,
        args.trials,
    )?;
    let column_finalizer = benchmark_finalizer(
        &conv,
        &column_columns,
        case,
        args.warmup,
        iterations,
        args.trials,
    )?;
    synchronize_and_check_wgpu(device, monitor, "finalizer timings")?;
    println!("  finalizer-only timings:");
    print_timing("row-columns finalizer", row_finalizer);
    print_timing("col-columns finalizer", column_finalizer);

    let burn_us = full[0].median_us;
    let row_savings = burn_us - full[2].median_us;
    let column_savings = burn_us - full[3].median_us;
    println!(
        "  break-even cached-row: savings={row_savings:.3} us/request, \
         cold_pack={:.3} us -> {:.0} requests, steady_pack={:.3} us -> {:.0} requests",
        row_measurement.cold_us,
        break_even_requests(row_measurement.cold_us, row_savings),
        row_measurement.steady.median_us,
        break_even_requests(row_measurement.steady.median_us, row_savings),
    );
    println!(
        "  break-even cached-col: savings={column_savings:.3} us/request, \
         metadata_pack={:.3} us -> {:.0} requests (additional cache bytes=0)",
        column_measurement.steady.median_us,
        break_even_requests(column_measurement.steady.median_us, column_savings),
    );
    println!(
        "  ratios: Burn/PyTorch-strict={:.4}x row/PyTorch-strict={:.4}x \
         col/PyTorch-strict={:.4}x",
        burn_us / case.pytorch_strict_us,
        full[2].median_us / case.pytorch_strict_us,
        full[3].median_us / case.pytorch_strict_us,
    );

    if let Some(nvml_index) = args.nvml_index {
        let (before, peak, after) = probe_vram(nvml_index, &full_context)?;
        println!(
            "  NVML peak probe: before={before} MiB peak={peak} MiB delta={:+} MiB after={after} MiB",
            peak as i64 - before as i64
        );
    }
    synchronize_and_check_wgpu(device, monitor, "case completion")?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let (device, monitor) = initialize_wgpu(args.adapter_index);
    B::seed(&device, SEED);
    let nvml_baseline = args
        .nvml_index
        .map(query_total_vram_mib)
        .transpose()
        .map_err(|error| io::Error::other(format!("initial NVML query failed: {error}")))?;
    report_nvml("initialized baseline", args.nvml_index, None);
    println!(
        "cached col2im isolated benchmark: device={device:?}, seed={SEED}, \
         warmup={}, trials={}, iterations={} cases={}",
        args.warmup,
        args.trials,
        args.iterations.map_or_else(
            || "per-shape safe default".to_owned(),
            |value| value.to_string()
        ),
        args.case_index
            .map_or_else(|| "1,2,3".to_owned(), |value| value.to_string()),
    );
    let all_case_weight_bytes = CASES
        .iter()
        .map(|case| case.weight_elements() * F32_BYTES)
        .sum::<usize>();
    println!(
        "all cases 1-3 production cache delta: cached-row={} bytes ({:.6} MiB), \
         cached-col=0 bytes; transient columns remain per call",
        all_case_weight_bytes,
        mib(all_case_weight_bytes),
    );

    for case in CASES
        .into_iter()
        .filter(|case| args.case_index.is_none_or(|index| index == case.index))
    {
        benchmark_case(&device, &monitor, &args, case, nvml_baseline)?;
    }
    synchronize_and_check_wgpu(&device, &monitor, "benchmark completion")?;
    report_nvml("final", args.nvml_index, nvml_baseline);
    println!("wgpu_uncaptured_errors=0");
    Ok(())
}
