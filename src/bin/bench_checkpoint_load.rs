//! One isolated model load. Run separate processes for unbiased peak RSS.
use anyhow::Result;
use clap::Parser;
use irodori_tts_burn::{WgpuFloatPrecision, backend_config, weights};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long, value_enum, default_value = "fp32")]
    precision: WgpuFloatPrecision,
    #[arg(long, default_value = "discrete:0")]
    device: backend_config::DeviceSelector,
    /// Compare streamed, borrowed and shared-owned WGSLPack host memory.
    #[arg(long, value_enum, default_value = "file")]
    mode: Mode,
    /// Read back a canonical parameter digest after timing (not part of load).
    #[arg(long)]
    verify: bool,
}

#[derive(Clone, Copy, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    File,
    Borrowed,
    Shared,
}

fn main() -> Result<()> {
    backend_config::initialize_cli_tracing("warn")?;
    let args = Args::parse();
    let start = Instant::now();
    let device = backend_config::wgpu_device_with_precision(&args.device.device(), args.precision)?;
    device.sync()?;
    let setup_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let bytes = (!matches!(args.mode, Mode::File))
        .then(|| std::fs::read(&args.checkpoint))
        .transpose()?;
    let read_ms = start.elapsed().as_secs_f64() * 1000.0;
    let load_start = Instant::now();
    let (model, _) = match (args.mode, bytes) {
        (Mode::Borrowed, Some(bytes)) => weights::packed::load_model_from_bytes(
            &bytes,
            &device,
            Some(args.precision.tensor_dtype()),
        )?,
        (Mode::Shared, Some(bytes)) => weights::packed::load_model_from_owned_bytes(
            burn::tensor::Bytes::from_elems(bytes),
            &device,
            Some(args.precision.tensor_dtype()),
        )?,
        (Mode::File, None) => weights::load_model_with_float_dtype(
            &args.checkpoint,
            &device,
            args.precision.tensor_dtype(),
        )?,
        _ => unreachable!("file I/O follows selected mode"),
    };
    let submitted_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    device.sync()?;
    let ready_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let peak_rss_kib = peak_rss_kib()?;
    let digest = if args.verify {
        use burn_store::ModuleSnapshot;
        use sha2::{Digest, Sha256};
        let mut snapshots = model.collect(None, None, true);
        snapshots.sort_by_key(|snapshot| snapshot.full_path());
        let mut hash = Sha256::new();
        for snapshot in snapshots {
            let data = snapshot.to_data()?;
            hash.update(snapshot.full_path().as_bytes());
            hash.update([0]);
            hash.update(format!("{:?}:{:?}", data.dtype, data.shape).as_bytes());
            hash.update(&*data.bytes);
        }
        Some(format!("{:x}", hash.finalize()))
    } else {
        None
    };
    println!(
        "{}",
        serde_json::json!({
            "checkpoint": args.checkpoint, "device": format!("{:?}", args.device),
        "precision": args.precision, "mode": args.mode, "setup_ms": setup_ms,
            "host_read_ms": read_ms, "submitted_ms": submitted_ms, "ready_ms": ready_ms,
            "total_ms": total_ms, "peak_rss_kib": peak_rss_kib, "parameter_sha256": digest,
        })
    );
    Ok(())
}

fn peak_rss_kib() -> Result<Option<u64>> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status")?;
        let value = status
            .lines()
            .find_map(|line| line.strip_prefix("VmHWM:"))
            .and_then(|value| value.split_whitespace().next())
            .map(str::parse::<u64>)
            .transpose()?;
        Ok(value)
    }
    #[cfg(not(target_os = "linux"))]
    Ok(None)
}
