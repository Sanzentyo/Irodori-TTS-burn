//! Prepare isolated, persistent retuning inputs without initializing a GPU.
use clap::{Parser, Subcommand};
use irodori_tts_burn::{WgpuFloatPrecision, backend_config};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Snapshot a CubeCL environment with and without projection decisions")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Hash logical autotune values in a closed environment through CubeCL.
    Inspect {
        #[arg(long)]
        source: PathBuf,
    },
    /// Export closed baseline and candidate bundles without changing the source.
    Snapshot {
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Import a bundle into a fresh writable runtime environment, without a GPU.
    Import {
        #[arg(long)]
        bundle: PathBuf,
        /// Adapter-specific cache directory; the environment file must not exist.
        #[arg(long)]
        cache_dir: PathBuf,
        #[arg(long)]
        precision: WgpuFloatPrecision,
    },
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    backend_config::initialize_cli_tracing("warn")?;
    let receipt = match args.command {
        Command::Inspect { source } => {
            serde_json::to_value(backend_config::inspect_autotune_cache(&source)?)?
        }
        Command::Snapshot { source, output } => {
            serde_json::to_value(backend_config::prepare_projection_retune(&source, &output)?)?
        }
        Command::Import {
            bundle,
            cache_dir,
            precision,
        } => {
            let environment = backend_config::configure_cubecl_persistent_cache_for_precision(
                cache_dir, precision,
            )?;
            anyhow::ensure!(
                !environment.environment_path.exists(),
                "import requires a new environment file"
            );
            let imported = backend_config::import_cubecl_environment_bundle(&bundle)?;
            serde_json::json!({"environment": environment, "imported": imported})
        }
    };
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
