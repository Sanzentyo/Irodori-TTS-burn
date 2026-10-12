//! Seal or verify a complete accuracy-approved CubeCL selection vector.

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use burn::backend::wgpu::{
    MemoryConfiguration, RuntimeOptions, graphics::AutoGraphicsApi, init_setup,
};
use clap::{Parser, Subcommand};
use irodori_tts_burn::autotune_approval::{
    ApprovedAutotuneCacheManifest, AutotuneAccuracyEvidence, AutotuneAccuracyPolicy,
    AutotuneRuntimeIdentity, seal_autotune_cache,
};
#[cfg(feature = "profile")]
use irodori_tts_burn::codec::{ApprovedK7SelectorManifestSet, K7SelectorCaseReceipt};
use irodori_tts_burn::route_autotune::{
    ApprovedRouteManifest, ApprovedRouteManifestSet, ResolvedRouteTable, RouteCandidateMeasurement,
    RouteCandidateRejection, RouteDeviceIdentity, RouteManifestResolution, RouteTuningPolicy,
    select_approved_routes_with_rejections,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(
    name = "approve_v4_autotune",
    about = "Seal and verify complete accuracy-approved CubeCL cache vectors"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Seal the exact complete selection vector after an E2E accuracy pass.
    Seal {
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        identity: PathBuf,
        #[arg(long)]
        evidence: PathBuf,
        #[arg(long)]
        cache_root: PathBuf,
        #[arg(long)]
        output_manifest: PathBuf,
    },
    /// Verify a restored cache exactly matches its approved selection vector.
    Verify {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        identity: PathBuf,
        #[arg(long)]
        cache_root: PathBuf,
        #[arg(long)]
        receipt: PathBuf,
    },
    /// Select exact per-device routes from 40-step fresh-session evidence.
    SelectRoutes {
        #[arg(long)]
        identity: PathBuf,
        /// Optional JSON policy; defaults to 5 fresh sessions, 10 requests,
        /// and a 2% minimum improvement.
        #[arg(long)]
        tuning_policy: Option<PathBuf>,
        #[arg(long = "measurement", required = true)]
        measurements: Vec<PathBuf>,
        /// Explicit fail-closed evidence for candidates that could not run.
        #[arg(long = "rejection")]
        rejections: Vec<PathBuf>,
        #[arg(long)]
        output_manifest: PathBuf,
    },
    /// Verify an exact device identity can resolve a sealed route table.
    VerifyRoutes {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        identity: PathBuf,
        #[arg(long)]
        receipt: PathBuf,
    },
    /// Assemble independently approved devices into one immutable lookup set.
    AssembleRouteSet {
        #[arg(long = "manifest", required = true)]
        manifests: Vec<PathBuf>,
        #[arg(long)]
        output_set: PathBuf,
    },
    /// Resolve a multi-device set exactly as a production binary would.
    ResolveRouteSet {
        #[arg(long)]
        manifest_set: PathBuf,
        #[arg(long)]
        identity: PathBuf,
        #[arg(long)]
        receipt: PathBuf,
    },
    /// Capture the exact runtime identity used by route measurements.
    BuildRouteIdentity {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        codec_weights: PathBuf,
        /// Exact release binary that will consume the approved route set.
        #[arg(long)]
        production_binary: PathBuf,
        #[arg(long, default_value_t = 0)]
        adapter_index: usize,
        #[arg(long, value_enum, default_value = "fp32")]
        precision: irodori_tts_burn::WgpuFloatPrecision,
        #[arg(long, default_value = "exclusive_pages")]
        allocator_policy: String,
        #[arg(long)]
        output_identity: PathBuf,
    },
    /// Seal exact codec k7 selector cases to one runtime and build identity.
    #[cfg(feature = "profile")]
    SealK7Selectors {
        #[arg(long)]
        identity: PathBuf,
        #[arg(long)]
        kernel_profile: String,
        #[arg(long)]
        source_sha256: String,
        #[arg(long)]
        binary_sha256: String,
        #[arg(long, default_value_t = 5)]
        minimum_fresh_sessions: usize,
        #[arg(long, value_delimiter = ',', default_value = "45,112,255,333,489,685")]
        required_latent_frames: Vec<usize>,
        #[arg(long = "case", required = true)]
        cases: Vec<PathBuf>,
        #[arg(long)]
        output_manifest: PathBuf,
    },
    /// Verify all k7 selector pins and extract one exact prepared shape.
    #[cfg(feature = "profile")]
    VerifyK7Selectors {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        identity: PathBuf,
        #[arg(long)]
        kernel_profile: String,
        #[arg(long)]
        source_sha256: String,
        #[arg(long)]
        binary_sha256: String,
        #[arg(long)]
        latent_frames: usize,
        #[arg(long)]
        receipt: PathBuf,
    },
}

#[cfg(feature = "profile")]
#[derive(Serialize)]
struct K7VerificationOutput {
    latent_frames: usize,
    accepted_tuning: bool,
    selection_count: usize,
    exact_identity_match: bool,
}

#[derive(Serialize)]
struct RouteVerificationOutput {
    schema_version: u32,
    route_abi: String,
    selection_count: usize,
    exact_identity_match: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case", tag = "decision")]
enum RouteSetResolutionOutput {
    Approved {
        schema_version: u32,
        route_abi: String,
        selection_count: usize,
    },
    Portable {
        reason: irodori_tts_burn::RouteCacheMissReason,
    },
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_slice(&fs::read(path).with_context(|| format!("read {}", path.display()))?)
        .with_context(|| format!("invalid JSON {}", path.display()))
}

fn write_new_json(path: &Path, value: &impl Serialize) -> Result<()> {
    ensure!(!path.exists(), "output already exists: {}", path.display());
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    output.write_all(&serde_json::to_vec_pretty(value)?)?;
    output.write_all(b"\n")?;
    Ok(())
}

fn main() -> Result<()> {
    irodori_tts_burn::backend_config::initialize_cli_tracing("info")?;
    match Args::parse().command {
        Command::Seal {
            policy,
            identity,
            evidence,
            cache_root,
            output_manifest,
        } => {
            let policy = AutotuneAccuracyPolicy::load(&policy)?;
            let identity: AutotuneRuntimeIdentity = read_json(&identity)?;
            let evidence: AutotuneAccuracyEvidence = read_json(&evidence)?;
            let manifest = seal_autotune_cache(&policy, &identity, evidence, &cache_root)?;
            write_new_json(&output_manifest, &manifest)?;
            serde_json::to_writer(std::io::stdout().lock(), &manifest)?;
        }
        Command::Verify {
            manifest,
            identity,
            cache_root,
            receipt,
        } => {
            let manifest = ApprovedAutotuneCacheManifest::load(&manifest)?;
            let identity: AutotuneRuntimeIdentity = read_json(&identity)?;
            let verification = manifest.verify(&identity, &cache_root)?;
            write_new_json(&receipt, &verification)?;
            serde_json::to_writer(std::io::stdout().lock(), &verification)?;
        }
        Command::SelectRoutes {
            identity,
            tuning_policy,
            measurements,
            rejections,
            output_manifest,
        } => {
            let identity: RouteDeviceIdentity = read_json(&identity)?;
            let policy = tuning_policy
                .as_deref()
                .map(read_json::<RouteTuningPolicy>)
                .transpose()?
                .unwrap_or_default();
            let measurements = measurements
                .iter()
                .map(|path| read_json::<RouteCandidateMeasurement>(path))
                .collect::<Result<Vec<_>>>()?;
            let rejections = rejections
                .iter()
                .map(|path| read_json::<RouteCandidateRejection>(path))
                .collect::<Result<Vec<_>>>()?;
            let manifest =
                select_approved_routes_with_rejections(identity, policy, measurements, rejections)?;
            write_new_json(&output_manifest, &manifest)?;
            serde_json::to_writer(std::io::stdout().lock(), &manifest)?;
        }
        Command::VerifyRoutes {
            manifest,
            identity,
            receipt,
        } => {
            let manifest = ApprovedRouteManifest::load(&manifest)?;
            let identity: RouteDeviceIdentity = read_json(&identity)?;
            let _routes = ResolvedRouteTable::from_manifest(&manifest, &identity)?;
            let output = RouteVerificationOutput {
                schema_version: manifest.schema_version,
                route_abi: manifest.route_abi.clone(),
                selection_count: manifest.selections.len(),
                exact_identity_match: true,
            };
            write_new_json(&receipt, &output)?;
            serde_json::to_writer(std::io::stdout().lock(), &output)?;
        }
        Command::AssembleRouteSet {
            manifests,
            output_set,
        } => {
            let profiles = manifests
                .iter()
                .map(|path| ApprovedRouteManifest::load(path))
                .collect::<irodori_tts_burn::Result<Vec<_>>>()?;
            let manifest_set = ApprovedRouteManifestSet::new(profiles)?;
            write_new_json(&output_set, &manifest_set)?;
            serde_json::to_writer(std::io::stdout().lock(), &manifest_set)?;
        }
        Command::ResolveRouteSet {
            manifest_set,
            identity,
            receipt,
        } => {
            let manifest_set = ApprovedRouteManifestSet::load(&manifest_set)?;
            let identity: RouteDeviceIdentity = read_json(&identity)?;
            let output = match manifest_set.resolve(&identity)? {
                RouteManifestResolution::Approved(manifest) => RouteSetResolutionOutput::Approved {
                    schema_version: manifest.schema_version,
                    route_abi: manifest.route_abi.clone(),
                    selection_count: manifest.selections.len(),
                },
                RouteManifestResolution::Portable { reason } => {
                    RouteSetResolutionOutput::Portable { reason }
                }
            };
            write_new_json(&receipt, &output)?;
            serde_json::to_writer(std::io::stdout().lock(), &output)?;
        }
        Command::BuildRouteIdentity {
            checkpoint,
            codec_weights,
            production_binary,
            adapter_index,
            precision,
            allocator_policy,
            output_identity,
        } => {
            ensure!(checkpoint.is_file(), "checkpoint is not a file");
            ensure!(codec_weights.is_file(), "codec weights are not a file");
            ensure!(
                production_binary.is_file(),
                "production binary is not a file"
            );
            ensure!(
                matches!(allocator_policy.as_str(), "exclusive_pages" | "sub_slices"),
                "allocator policy must be exclusive_pages or sub_slices"
            );
            let device =
                irodori_tts_burn::backend_config::wgpu_device_from_adapter_index(adapter_index);
            let setup = init_setup::<AutoGraphicsApi>(
                &device,
                RuntimeOptions {
                    tasks_max: 32,
                    memory_config: match allocator_policy.as_str() {
                        "exclusive_pages" => MemoryConfiguration::ExclusivePages,
                        "sub_slices" => MemoryConfiguration::SubSlices,
                        _ => unreachable!("validated allocator policy"),
                    },
                },
            );
            let info = setup.adapter.get_info();
            let identity = RouteDeviceIdentity {
                adapter_name: info.name,
                backend: format!("{:?}", info.backend),
                device_type: format!("{:?}", info.device_type),
                vendor_id: info.vendor,
                device_id: info.device,
                driver: info.driver,
                driver_info: info.driver_info,
                os: std::env::consts::OS.to_owned(),
                platform_version: irodori_tts_burn::current_platform_version().unwrap_or_default(),
                architecture: std::env::consts::ARCH.to_owned(),
                precision: precision.label().to_owned(),
                allocator_policy,
                compiler_policy: "wgpu_auto".to_owned(),
                application_version: env!("CARGO_PKG_VERSION").to_owned(),
                burn_version: "0.22.0-pre.2".to_owned(),
                burn_cubecl_version: "0.22.0-pre.2".to_owned(),
                cubecl_version: "0.11.0-pre.2".to_owned(),
                cubek_version: "0.3.0-pre.2".to_owned(),
                wgpu_version: "30.0.0".to_owned(),
                model_sha256: irodori_tts_burn::sha256_file(&checkpoint)?,
                codec_sha256: irodori_tts_burn::sha256_file(&codec_weights)?,
                binary_sha256: irodori_tts_burn::sha256_file(&production_binary)?,
            };
            identity.validate()?;
            write_new_json(&output_identity, &identity)?;
            serde_json::to_writer(std::io::stdout().lock(), &identity)?;
        }
        #[cfg(feature = "profile")]
        Command::SealK7Selectors {
            identity,
            kernel_profile,
            source_sha256,
            binary_sha256,
            minimum_fresh_sessions,
            required_latent_frames,
            cases,
            output_manifest,
        } => {
            let identity: AutotuneRuntimeIdentity = read_json(&identity)?;
            let cases = cases
                .iter()
                .map(|path| read_json::<K7SelectorCaseReceipt>(path))
                .collect::<Result<Vec<_>>>()?;
            let manifest = ApprovedK7SelectorManifestSet::seal(
                identity,
                kernel_profile,
                source_sha256,
                binary_sha256,
                minimum_fresh_sessions,
                required_latent_frames,
                cases,
            )?;
            write_new_json(&output_manifest, &manifest)?;
            serde_json::to_writer(std::io::stdout().lock(), &manifest)?;
        }
        #[cfg(feature = "profile")]
        Command::VerifyK7Selectors {
            manifest,
            identity,
            kernel_profile,
            source_sha256,
            binary_sha256,
            latent_frames,
            receipt,
        } => {
            let manifest = ApprovedK7SelectorManifestSet::load(&manifest)?;
            let identity: AutotuneRuntimeIdentity = read_json(&identity)?;
            let verification = manifest.verify(
                &identity,
                &kernel_profile,
                &source_sha256,
                &binary_sha256,
                latent_frames,
            )?;
            let output = K7VerificationOutput {
                latent_frames: verification.latent_frames,
                accepted_tuning: verification.accepted_tuning,
                selection_count: verification.selection_count,
                exact_identity_match: true,
            };
            write_new_json(&receipt, &output)?;
            serde_json::to_writer(std::io::stdout().lock(), &output)?;
        }
    }
    Ok(())
}
