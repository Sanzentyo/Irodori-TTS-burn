//! Build and optionally install one exact GPU/driver route profile.

use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use burn::backend::wgpu::{
    MemoryConfiguration, RuntimeOptions, graphics::AutoGraphicsApi, init_setup,
};
use clap::{Parser, ValueEnum};
use irodori_tts_burn::{
    ApprovedRouteManifestSet, BuiltInRouteProfile, FreshProcessRouteTuner,
    FreshProcessRouteTunerConfig, FreshProcessTuningWorkload, RouteDeviceIdentity,
    RouteTuningPolicy, autotune_routes_on_base, campaign_sha256s,
};
use serde::Serialize;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BaseProfile {
    Auto,
    Portable,
    NvidiaRtx,
    AppleM5,
}

impl BaseProfile {
    fn resolve(self, vendor_id: u32, backend: &str) -> BuiltInRouteProfile {
        match self {
            Self::Auto => {
                BuiltInRouteProfile::for_adapter(vendor_id, backend, std::env::consts::OS)
            }
            Self::Portable => BuiltInRouteProfile::Portable,
            Self::NvidiaRtx => BuiltInRouteProfile::NvidiaRtx,
            Self::AppleM5 => BuiltInRouteProfile::AppleM5,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "tune_v4_routes",
    about = "Measure, accuracy-check, and seal exact v4 WGPU routes"
)]
struct Args {
    /// JSON workload with exact B/S operations, fixtures, references, and
    /// canonical 40-step latent/waveform oracle files.
    #[arg(long)]
    workload: PathBuf,
    /// `bench_v4_residency` built from the same source with `--features profile`.
    #[arg(long, default_value = "target/release/bench_v4_residency")]
    benchmark_binary: PathBuf,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    codec_weights: PathBuf,
    /// Exact application binary that will consume the resulting profile.
    #[arg(long, default_value = "target/release/pipeline")]
    production_binary: PathBuf,
    /// Must not already exist. Raw child logs and evidence remain here.
    #[arg(long)]
    output_directory: PathBuf,
    /// Shared CubeCL environment. Pipeline objects remain process-local.
    #[arg(long)]
    cubecl_cache_directory: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    adapter_index: usize,
    #[arg(long, value_enum, default_value = "auto")]
    base_profile: BaseProfile,
    #[arg(long, default_value_t = 5)]
    fresh_sessions: usize,
    #[arg(long, default_value_t = 2)]
    warmups: usize,
    #[arg(long, default_value_t = 10)]
    measured: usize,
    #[arg(long, default_value_t = 200)]
    minimum_improvement_basis_points: u32,
    /// Merge the exact profile into this route set. Requires --install.
    #[arg(long)]
    manifest_set: Option<PathBuf>,
    /// Persist after successful composed 40-step validation. Without this,
    /// the campaign is generated but no application cache is changed.
    #[arg(long)]
    install: bool,
}

#[derive(Serialize)]
struct IdentityReceipt<'a> {
    identity_sha256: String,
    identity: &'a RouteDeviceIdentity,
}

#[derive(Serialize)]
struct CampaignPins {
    workload_sha256: String,
    benchmark_binary_sha256: String,
    production_binary_sha256: String,
    model_sha256: String,
    codec_sha256: String,
}

fn write_new_json(path: &Path, value: &impl Serialize) -> Result<()> {
    ensure!(!path.exists(), "output already exists: {}", path.display());
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&serde_json::to_vec_pretty(value)?)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(())
}

fn write_sha256s(root: &Path) -> Result<()> {
    let payload = campaign_sha256s(root)?
        .into_iter()
        .map(|(path, digest)| format!("{digest}  {}\n", path.display()))
        .collect::<String>();
    let path = root.join("SHA256SUMS");
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(payload.as_bytes())?;
    output.sync_all()?;
    Ok(())
}

struct CacheLock {
    path: PathBuf,
}

impl CacheLock {
    fn acquire(target: &Path) -> Result<Self> {
        let path = target.with_extension("lock");
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("route cache lock is held: {}", path.display()))?;
        Ok(Self { path })
    }
}

impl Drop for CacheLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn install_manifest_set(
    path: &Path,
    profile: irodori_tts_burn::ApprovedRouteManifest,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let _lock = CacheLock::acquire(path)?;
    let mut profiles = if path.is_file() {
        ApprovedRouteManifestSet::load(path)?
            .profiles()
            .cloned()
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    profiles.retain(|existing| existing.identity != profile.identity);
    profiles.push(profile);
    let set = ApprovedRouteManifestSet::new(profiles)?;
    let temporary = path.with_extension(format!("new-{}", std::process::id()));
    write_new_json(&temporary, &set)?;
    #[cfg(not(windows))]
    fs::rename(&temporary, path)?;
    #[cfg(windows)]
    {
        let backup = path.with_extension(format!("backup-{}", std::process::id()));
        if path.exists() {
            fs::rename(path, &backup)?;
        }
        if let Err(error) = fs::rename(&temporary, path) {
            if backup.exists() {
                let _ = fs::rename(&backup, path);
            }
            return Err(error.into());
        }
        if backup.exists() {
            fs::remove_file(backup)?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    irodori_tts_burn::backend_config::initialize_cli_tracing("info")?;
    let args = Args::parse();
    ensure!(
        !args.output_directory.exists(),
        "output directory must be fresh: {}",
        args.output_directory.display()
    );
    ensure!(
        args.install || args.manifest_set.is_none(),
        "--manifest-set requires --install"
    );
    for path in [
        args.workload.as_path(),
        args.benchmark_binary.as_path(),
        args.checkpoint.as_path(),
        args.codec_weights.as_path(),
        args.production_binary.as_path(),
    ] {
        ensure!(
            path.is_file(),
            "required input is not a file: {}",
            path.display()
        );
    }
    let workload = FreshProcessTuningWorkload::load(&args.workload)?;
    let device =
        irodori_tts_burn::backend_config::wgpu_device_from_adapter_index(args.adapter_index);
    let setup = init_setup::<AutoGraphicsApi>(
        &device,
        RuntimeOptions {
            tasks_max: 32,
            memory_config: MemoryConfiguration::ExclusivePages,
        },
    );
    let adapter = setup.adapter.get_info();
    // The coordinator never executes candidates itself. Release its probe
    // device before child processes start so it cannot consume VRAM or skew
    // clocks during the fresh-session measurements.
    drop(setup);
    let base_profile = args
        .base_profile
        .resolve(adapter.vendor, &format!("{:?}", adapter.backend));
    let identity = RouteDeviceIdentity {
        adapter_name: adapter.name,
        backend: format!("{:?}", adapter.backend),
        device_type: format!("{:?}", adapter.device_type),
        vendor_id: adapter.vendor,
        device_id: adapter.device,
        driver: adapter.driver,
        driver_info: adapter.driver_info,
        os: std::env::consts::OS.to_owned(),
        platform_version: irodori_tts_burn::current_platform_version().unwrap_or_default(),
        architecture: std::env::consts::ARCH.to_owned(),
        precision: "fp32".to_owned(),
        allocator_policy: "exclusive_pages".to_owned(),
        compiler_policy: "wgpu_auto".to_owned(),
        application_version: env!("CARGO_PKG_VERSION").to_owned(),
        burn_version: "0.22.0-pre.2".to_owned(),
        burn_cubecl_version: "0.22.0-pre.2".to_owned(),
        cubecl_version: "0.11.0-pre.2".to_owned(),
        cubek_version: "0.3.0-pre.2".to_owned(),
        wgpu_version: "30.0.0".to_owned(),
        model_sha256: irodori_tts_burn::sha256_file(&args.checkpoint)?,
        codec_sha256: irodori_tts_burn::sha256_file(&args.codec_weights)?,
        binary_sha256: irodori_tts_burn::sha256_file(&args.production_binary)?,
    };
    identity.validate()?;
    let identity_sha256 = identity.fingerprint_sha256()?;
    let cache = args.cubecl_cache_directory.unwrap_or_else(|| {
        args.output_directory
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("Irodori-TTS-burn-route-tuning-cubecl")
    });
    let policy = RouteTuningPolicy {
        minimum_fresh_sessions: args.fresh_sessions,
        minimum_measured_requests_per_session: args.measured,
        minimum_improvement_basis_points: args.minimum_improvement_basis_points,
    };
    let pins = CampaignPins {
        workload_sha256: irodori_tts_burn::sha256_file(&args.workload)?,
        benchmark_binary_sha256: irodori_tts_burn::sha256_file(&args.benchmark_binary)?,
        production_binary_sha256: identity.binary_sha256.clone(),
        model_sha256: identity.model_sha256.clone(),
        codec_sha256: identity.codec_sha256.clone(),
    };
    let route_workload = workload.route_workload();
    let mut tuner = FreshProcessRouteTuner::new(
        FreshProcessRouteTunerConfig {
            benchmark_binary: args.benchmark_binary,
            checkpoint: args.checkpoint,
            codec_weights: args.codec_weights,
            output_directory: args.output_directory.clone(),
            cubecl_cache_directory: cache,
            adapter_index: args.adapter_index,
            base_profile,
            fresh_sessions: args.fresh_sessions,
            warmups: args.warmups,
            measured_requests: args.measured,
        },
        workload,
    )?;
    write_new_json(
        &args.output_directory.join("identity.json"),
        &IdentityReceipt {
            identity_sha256,
            identity: &identity,
        },
    )?;
    write_new_json(&args.output_directory.join("pins.json"), &pins)?;
    let manifest =
        autotune_routes_on_base(identity, policy, base_profile, &route_workload, &mut tuner)?;
    let composed = tuner.validate_composed_manifest(&manifest)?;
    write_new_json(
        &args.output_directory.join("approved-route-manifest.json"),
        &manifest,
    )?;
    write_new_json(
        &args.output_directory.join("composed-validation.json"),
        &composed,
    )?;
    write_new_json(
        &args.output_directory.join("route-set.json"),
        &ApprovedRouteManifestSet::new(vec![manifest.clone()])?,
    )?;
    if args.install {
        let target =
            args.manifest_set
                .unwrap_or(irodori_tts_burn::default_route_manifest_set_path(
                    &irodori_tts_burn::backend_config::default_cubecl_cache_root()?,
                ));
        install_manifest_set(&target, manifest)?;
        tracing::info!("installed_route_manifest_set={}", target.display());
    }
    write_sha256s(&args.output_directory)?;
    tracing::info!("route_campaign={}", args.output_directory.display());
    Ok(())
}
