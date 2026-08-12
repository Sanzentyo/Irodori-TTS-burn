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
    }
    Ok(())
}
