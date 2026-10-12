//! One-time conversion to the optional native model format.
use anyhow::Result;
use clap::Parser;
use irodori_tts_burn::{WgpuFloatPrecision, backend_config, weights::packed};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum, default_value = "fp32")]
    precision: WgpuFloatPrecision,
    #[arg(long, default_value = "discrete:0")]
    device: backend_config::DeviceSelector,
}

fn main() -> Result<()> {
    backend_config::initialize_cli_tracing("warn")?;
    let args = Args::parse();
    let device = backend_config::wgpu_device_with_precision(&args.device.device(), args.precision)?;
    let start = Instant::now();
    packed::pack_checkpoint(&args.checkpoint, &args.output, &device, args.precision)?;
    println!(
        "{}",
        serde_json::json!({
            "output": args.output, "precision": args.precision,
            "elapsed_ms": start.elapsed().as_secs_f64() * 1000.0,
            "bytes": std::fs::metadata(&args.output)?.len(),
        })
    );
    Ok(())
}
