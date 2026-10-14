//! CPU-only conversion of decoder weights and prepared WGSL layouts.
use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    irodori_tts_burn::codec::packed::pack_decoder(&args.checkpoint, &args.output)?;
    println!("{}", args.output.display());
    Ok(())
}
