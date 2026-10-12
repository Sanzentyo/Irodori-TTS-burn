//! Paired streaming/cached WGSL softmax and complete SDPA measurements.

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
        AutoCompiler, MemoryConfiguration, RuntimeOptions, WgpuDevice, WgpuRuntime,
        graphics::AutoGraphicsApi, init_setup,
    },
    tensor::{Bool, Device, Tensor, TensorData},
};
use clap::{Parser, ValueEnum};
use cubecl::{future::block_on, prelude::Runtime, profile::TimingMethod};
use irodori_tts_burn::{
    WgpuRaw,
    kernels::matmul_sdpa::{
        try_fused_scores_softmax_cached_f32, try_fused_scores_softmax_f32,
        try_fused_scores_softmax_subgroup_f32,
    },
    route_autotune::RouteAccuracyMetrics,
};
use serde::Serialize;

type Rt = WgpuRuntime<AutoCompiler>;

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum Scope {
    Softmax,
    Sdpa,
}

#[derive(Debug, Parser, Serialize)]
struct Args {
    #[arg(long)]
    output_json: PathBuf,
    #[arg(long)]
    integrated: bool,
    #[arg(long, default_value_t = 0)]
    adapter_index: usize,
    #[arg(long, value_enum, default_value_t = Scope::Softmax)]
    scope: Scope,
    #[arg(long, value_enum, default_value_t = Route::Cached)]
    candidate: Route,
    #[arg(long, default_value_t = 3)]
    batch: usize,
    #[arg(long, default_value_t = 489)]
    sequence: usize,
    #[arg(long, default_value_t = 22)]
    context: usize,
    #[arg(long, default_value_t = 4)]
    warmups: usize,
    #[arg(long, default_value_t = 20)]
    blocks: usize,
    /// Use the production strict-F32 SimpleUnit/MinTile probability-value matmul.
    #[arg(long)]
    unit_min_pv: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum Route {
    Streaming,
    Cached,
    Subgroup,
}

struct Inputs {
    unit_min_pv: bool,
    device: Device,
    scores: TensorData,
    mask: Tensor<2, Bool>,
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
}

impl Inputs {
    fn new(args: &Args, device: Device) -> Result<Self> {
        let keys = args
            .sequence
            .checked_add(args.context)
            .context("key count overflow")?;
        ensure!(
            (1..=2_048).contains(&keys),
            "cached softmax supports 1..=2048 keys"
        );
        let shape = [args.batch, 20, args.sequence, keys];
        let count = shape
            .into_iter()
            .try_fold(1usize, usize::checked_mul)
            .context("score size overflow")?;
        let values = (0..count)
            .map(|i| ((i as f32 * 0.017).sin() * 16.0) + ((i % 31) as f32 - 15.0))
            .collect::<Vec<_>>();
        let scores = TensorData::new(values, shape);
        let mask = Tensor::<2, Bool>::from_data(
            TensorData::new(
                (0..args.batch * keys)
                    .map(|i| (i % keys) % 17 == 1)
                    .collect::<Vec<_>>(),
                [args.batch, keys],
            ),
            &device,
        );
        let tensor = |shape: [usize; 4], phase: f32| {
            let count = shape.into_iter().product();
            Tensor::<4>::from_data(
                TensorData::new(
                    (0..count)
                        .map(|i| (i as f32 * 0.013 + phase).sin())
                        .collect::<Vec<_>>(),
                    shape,
                ),
                &device,
            )
        };
        let q = tensor([args.batch, 20, args.sequence, 64], 0.1);
        let k = tensor([args.batch, 20, keys, 64], 1.3).swap_dims(2, 3);
        let v = tensor([args.batch, 20, keys, 64], 2.7);
        Ok(Self {
            unit_min_pv: args.unit_min_pv,
            device,
            scores,
            mask,
            q,
            k,
            v,
        })
    }

    fn prepare(&self, scope: Scope) -> Option<Tensor<4>> {
        match scope {
            // Restore pristine scores before timing: the kernel mutates them.
            Scope::Softmax => Some(Tensor::from_data(self.scores.clone(), &self.device)),
            Scope::Sdpa => None,
        }
    }

    fn launch(&self, scope: Scope, route: Route, prepared: Option<Tensor<4>>) -> Result<Tensor<4>> {
        let scores = match prepared {
            Some(scores) => scores,
            None => self.q.clone().matmul(self.k.clone()),
        }
        .try_into_primitive::<WgpuRaw>()
        .map_err(|error| anyhow::anyhow!("scores must use raw WGPU: {error:?}"))?;
        let mask = self
            .mask
            .clone()
            .try_into_primitive::<WgpuRaw>()
            .map_err(|error| anyhow::anyhow!("mask must use raw WGPU: {error:?}"))?;
        let output = match route {
            Route::Streaming => try_fused_scores_softmax_f32(scores, mask, 0.125),
            Route::Cached => try_fused_scores_softmax_cached_f32(scores, mask, 0.125),
            Route::Subgroup => try_fused_scores_softmax_subgroup_f32(scores, mask, 0.125),
        }
        .context("softmax launch contract rejected")?;
        let probabilities = Tensor::<4>::from_primitive::<WgpuRaw>(output);
        Ok(match scope {
            Scope::Softmax => probabilities,
            Scope::Sdpa if self.unit_min_pv => {
                let lhs = probabilities
                    .try_into_primitive::<WgpuRaw>()
                    .map_err(|error| {
                        anyhow::anyhow!("probabilities must use raw WGPU: {error:?}")
                    })?;
                let rhs = self
                    .v
                    .clone()
                    .try_into_primitive::<WgpuRaw>()
                    .map_err(|error| anyhow::anyhow!("values must use raw WGPU: {error:?}"))?;
                Tensor::from_primitive::<WgpuRaw>(
                    irodori_tts_burn::kernels::matmul_unit::try_matmul_unit_min_f32(lhs, rhs)
                        .context("strict-F32 PV matmul contract rejected")?,
                )
            }
            Scope::Sdpa => probabilities.matmul(self.v.clone()),
        })
    }
}

#[derive(Serialize)]
struct Sample {
    block: usize,
    slot: usize,
    route: Route,
    device_complete_ms: f64,
    gpu_elapsed_ms: f64,
    timing_source: &'static str,
}

fn sync(device: &WgpuDevice) -> Result<()> {
    block_on(Rt::client(device).sync()).context("device synchronization failed")
}

fn measure(
    inputs: &Inputs,
    device: &WgpuDevice,
    scope: Scope,
    route: Route,
    block: usize,
    slot: usize,
) -> Result<Sample> {
    let prepared = inputs.prepare(scope);
    sync(device)?;
    let client = Rt::client(device);
    let started = Instant::now();
    let (output, duration) = client
        .profile(|| inputs.launch(scope, route, prepared), "paired_softmax")
        .context("GPU timestamp scope failed")?;
    let output = output?;
    sync(device)?;
    std::hint::black_box(output.dims());
    let device_complete_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let timing_source = match duration.timing_method() {
        TimingMethod::Device => "device_timestamp",
        TimingMethod::System => "synchronized_system_clock",
    };
    let gpu_elapsed_ms = block_on(duration.resolve()).duration().as_secs_f64() * 1_000.0;
    Ok(Sample {
        block,
        slot,
        route,
        device_complete_ms,
        gpu_elapsed_ms,
        timing_source,
    })
}

fn main() -> Result<()> {
    irodori_tts_burn::backend_config::initialize_cli_tracing("warn")?;
    let args = Args::parse();
    ensure!(
        args.batch > 0 && args.sequence > 0 && args.warmups > 0 && args.blocks > 0,
        "batch, sequence, warmups and blocks must be positive"
    );
    ensure!(
        !args.output_json.exists(),
        "refusing to overwrite {}",
        args.output_json.display()
    );
    let raw_device = if args.integrated {
        WgpuDevice::IntegratedGpu(args.adapter_index)
    } else {
        WgpuDevice::DiscreteGpu(args.adapter_index)
    };
    let setup = init_setup::<AutoGraphicsApi>(
        &raw_device,
        RuntimeOptions {
            tasks_max: 32,
            memory_config: MemoryConfiguration::ExclusivePages,
        },
    );
    let info = setup.adapter.get_info();
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    let callback_errors = Arc::clone(&errors);
    setup.device.on_uncaptured_error(Arc::new(move |error| {
        callback_errors
            .lock()
            .expect("error monitor poisoned")
            .push(error.to_string());
    }));
    let device = irodori_tts_burn::backend_config::strict_fp32_device(&raw_device)?;
    let inputs = Inputs::new(&args, device)?;
    for _ in 0..args.warmups {
        for route in [Route::Streaming, args.candidate] {
            // Autotuning profiles its own candidates. Complete it before an
            // outer timestamp scope, which cannot nest those query sets.
            let output = inputs.launch(args.scope, route, inputs.prepare(args.scope))?;
            sync(&raw_device)?;
            std::hint::black_box(output.dims());
        }
    }
    let mut samples = Vec::new();
    for block in 0..args.blocks {
        let order = if block.is_multiple_of(2) {
            [
                Route::Streaming,
                args.candidate,
                args.candidate,
                Route::Streaming,
            ]
        } else {
            [
                args.candidate,
                Route::Streaming,
                Route::Streaming,
                args.candidate,
            ]
        };
        for (slot, route) in order.into_iter().enumerate() {
            samples.push(measure(
                &inputs,
                &raw_device,
                args.scope,
                route,
                block,
                slot,
            )?);
        }
    }
    let read = |route| -> Result<Vec<f32>> {
        Ok(inputs
            .launch(args.scope, route, inputs.prepare(args.scope))?
            .into_data()
            .to_vec::<f32>()?)
    };
    let reference = read(Route::Streaming)?;
    let candidate = read(args.candidate)?;
    let accuracy = RouteAccuracyMetrics::compare(&reference, &candidate)?;
    let bitwise_equal = reference
        .iter()
        .zip(&candidate)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    sync(&raw_device)?;
    let errors = errors.lock().expect("error monitor poisoned");
    ensure!(errors.is_empty(), "uncaptured WGPU errors: {errors:?}");
    let report = serde_json::json!({
        "schema_version": 1,
        "args": args,
        "precision": "strict_fp32",
        "timing_boundary": "device-complete; input restore outside timing; readback after timing",
        "adapter": { "name": info.name, "vendor_id": info.vendor, "device_id": info.device,
            "device_type": format!("{:?}", info.device_type), "backend": format!("{:?}", info.backend),
            "driver": info.driver, "driver_info": info.driver_info },
        "samples": samples,
        "accuracy": accuracy,
        "elements": reference.len(),
        "bitwise_equal": bitwise_equal,
        "uncaptured_wgpu_errors": *errors,
    });
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output_json)?;
    serde_json::to_writer_pretty(BufWriter::new(output), &report)?;
    println!(
        "{}: {} elements, bitwise_equal={bitwise_equal}, max_abs={:.3e}",
        args.output_json.display(),
        reference.len(),
        accuracy.max_abs
    );
    Ok(())
}
