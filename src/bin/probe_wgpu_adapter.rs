//! Read-only WGPU adapter identity probe for benchmark environment manifests.

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use burn::backend::wgpu::{
    AutoCompiler, MemoryConfiguration, RuntimeOptions, WgpuDevice, WgpuRuntime,
    graphics::AutoGraphicsApi, init_setup,
};
use clap::Parser;
use cubecl::prelude::Runtime;
use serde::Serialize;

type WgpuRt = WgpuRuntime<AutoCompiler>;

#[derive(Debug, Parser)]
struct Args {
    /// Probe an integrated GPU instead of a discrete GPU.
    #[arg(long)]
    integrated: bool,
    #[arg(long, default_value_t = 0)]
    adapter_index: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
struct AdapterProbe {
    schema_version: u32,
    adapter_index: usize,
    name: String,
    backend: String,
    device_type: String,
    driver: String,
    driver_info: String,
    vendor_id: u32,
    device_id: u32,
    allocator_number_allocs: u64,
    allocator_bytes_in_use: u64,
    allocator_bytes_reserved: u64,
}

fn main() -> Result<()> {
    irodori_tts_burn::backend_config::initialize_cli_tracing("info")?;
    let args = Args::parse();
    let device = if args.integrated {
        WgpuDevice::IntegratedGpu(args.adapter_index)
    } else {
        WgpuDevice::DiscreteGpu(args.adapter_index)
    };
    let setup = init_setup::<AutoGraphicsApi>(
        &device,
        RuntimeOptions {
            tasks_max: 32,
            memory_config: MemoryConfiguration::SubSlices,
        },
    );
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    let callback_errors = Arc::clone(&errors);
    setup.device.on_uncaptured_error(Arc::new(move |error| {
        if let Ok(mut values) = callback_errors.lock() {
            values.push(error.to_string());
        }
    }));
    let client = WgpuRt::client(&device);
    cubecl::future::block_on(client.sync()).context("WGPU adapter probe sync failed")?;
    let errors = errors.lock().expect("WGPU error monitor poisoned");
    ensure!(errors.is_empty(), "uncaptured WGPU errors: {errors:?}");
    let usage = client
        .memory_usage()
        .context("failed to query initial WGPU allocator usage")?;
    let info = setup.adapter.get_info();
    let probe = AdapterProbe {
        schema_version: 1,
        adapter_index: args.adapter_index,
        name: info.name,
        backend: format!("{:?}", info.backend),
        device_type: format!("{:?}", info.device_type),
        driver: info.driver,
        driver_info: info.driver_info,
        vendor_id: info.vendor,
        device_id: info.device,
        allocator_number_allocs: usage.number_allocs,
        allocator_bytes_in_use: usage.bytes_in_use,
        allocator_bytes_reserved: usage.bytes_reserved,
    };
    serde_json::to_writer_pretty(std::io::stdout().lock(), &probe)?;
    Ok(())
}
