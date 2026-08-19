//! Seal or verify a complete accuracy-approved CubeCL selection vector.

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use irodori_tts_burn::autotune_approval::{
    ApprovedAutotuneCacheManifest, AutotuneAccuracyEvidence, AutotuneAccuracyPolicy,
    AutotuneRuntimeIdentity, seal_autotune_cache,
};
#[cfg(feature = "profile")]
use irodori_tts_burn::codec::{ApprovedK7SelectorManifestSet, K7SelectorCaseReceipt};
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
            println!("{}", serde_json::to_string(&manifest)?);
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
            println!("{}", serde_json::to_string(&verification)?);
        }
        #[cfg(feature = "profile")]
        Command::SealK7Selectors {
            identity,
            kernel_profile,
            source_sha256,
            binary_sha256,
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
                cases,
            )?;
            write_new_json(&output_manifest, &manifest)?;
            println!("{}", serde_json::to_string(&manifest)?);
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
            println!("{}", serde_json::to_string(&output)?);
        }
    }
    Ok(())
}
