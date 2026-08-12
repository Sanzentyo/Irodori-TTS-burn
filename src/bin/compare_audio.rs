use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use irodori_tts_burn::validation::AudioMetrics;

#[derive(Debug, Parser)]
#[command(about = "Measure sample-aligned parity between two WAV files")]
struct Args {
    /// Authoritative reference WAV.
    #[arg(long)]
    reference: PathBuf,
    /// Candidate WAV produced by this runtime.
    #[arg(long)]
    candidate: PathBuf,
    /// Emit machine-readable JSON.
    #[arg(long)]
    json: bool,
}

struct Waveform {
    sample_rate: u32,
    samples: Vec<f32>,
}

fn read_mono(path: &Path) -> Result<Waveform> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open WAV {}", path.display()))?;
    let spec = reader.spec();
    ensure!(spec.channels > 0, "WAV has no channels: {}", path.display());

    let interleaved = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("failed to decode float WAV")?,
        hound::SampleFormat::Int => {
            let denominator = (1_u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / denominator))
                .collect::<Result<Vec<_>, _>>()
                .context("failed to decode integer WAV")?
        }
    };
    let channels = usize::from(spec.channels);
    let inverse_channels = 1.0 / channels as f32;
    let samples = interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() * inverse_channels)
        .collect();
    Ok(Waveform {
        sample_rate: spec.sample_rate,
        samples,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let reference = read_mono(&args.reference)?;
    let candidate = read_mono(&args.candidate)?;
    ensure!(
        reference.sample_rate == candidate.sample_rate,
        "sample-rate mismatch: reference={} Hz, candidate={} Hz",
        reference.sample_rate,
        candidate.sample_rate,
    );
    let metrics = AudioMetrics::compare(&reference.samples, &candidate.samples)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&metrics)?);
    } else {
        println!("sample_rate_hz: {}", reference.sample_rate);
        println!("sample_count: {}", metrics.sample_count);
        println!("max_abs_error: {:.9}", metrics.max_abs_error);
        println!("mean_abs_error: {:.9}", metrics.mean_abs_error);
        println!("rmse: {:.9}", metrics.root_mean_square_error);
        println!("snr_db: {:.6}", metrics.signal_to_noise_db);
        println!("cosine_similarity: {:.9}", metrics.cosine_similarity);
    }
    Ok(())
}
