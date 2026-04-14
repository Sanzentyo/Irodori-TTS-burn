//! E2E parity test: compare Rust DACVAE encoder output against Python reference.
//!
//! Usage:
//!   cargo run --bin codec_e2e --release -- \
//!     --weights target/dacvae_weights.safetensors \
//!     --ref-latent /tmp/py_latent.npy \
//!     [--audio path/to/audio.wav]
//!
//! The reference latent is produced by `scripts/codec_e2e_ref.py`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use burn::backend::NdArray;
use burn::prelude::*;
use clap::Parser;

use irodori_tts_burn::load_codec;

type B = NdArray<f32>;

#[derive(Parser)]
#[command(about = "DACVAE E2E parity test vs Python reference")]
struct Args {
    /// Path to the converted safetensors weights file.
    #[arg(long, default_value = "target/dacvae_weights.safetensors")]
    weights: PathBuf,

    /// Path to Python-produced reference latent (.npy, shape [1,T,32]).
    #[arg(long)]
    ref_latent: Option<PathBuf>,

    /// Path to audio file to encode (WAV, 48 kHz mono recommended).
    #[arg(long, default_value = "target/test_audio.wav")]
    audio: PathBuf,

    /// Tolerance for absolute mean error against reference.
    #[arg(long, default_value_t = 1e-4)]
    atol: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Default::default();

    eprintln!(
        "[codec_e2e] Loading weights from {} ...",
        args.weights.display()
    );
    let codec = load_codec::<B>(&args.weights, &device).context("Failed to load DACVAE weights")?;

    eprintln!(
        "[codec_e2e] Loading audio from {} ...",
        args.audio.display()
    );
    let wav = load_wav_as_tensor(&args.audio, &device)?;
    let wav_shape = wav.dims();
    eprintln!(
        "[codec_e2e] Audio shape: [{}, {}, {}]",
        wav_shape[0], wav_shape[1], wav_shape[2]
    );

    eprintln!("[codec_e2e] Encoding ...");
    let latent = codec.encode(wav);
    let lat_shape = latent.dims();
    eprintln!(
        "[codec_e2e] Latent shape: [{}, {}, {}]",
        lat_shape[0], lat_shape[1], lat_shape[2]
    );

    // Print first few latent values for visual inspection.
    let flat: Vec<f32> = latent.clone().into_data().to_vec().unwrap();
    eprintln!(
        "[codec_e2e] Latent[0,0,:8] = {:?}",
        &flat[..8.min(flat.len())]
    );

    // Decode back and print audio shape.
    eprintln!("[codec_e2e] Decoding ...");
    let recon = codec.decode(latent.clone());
    let rec_shape = recon.dims();
    eprintln!(
        "[codec_e2e] Reconstructed audio shape: [{}, {}, {}]",
        rec_shape[0], rec_shape[1], rec_shape[2]
    );

    if let Some(ref_path) = &args.ref_latent {
        eprintln!("[codec_e2e] Comparing against Python reference latent ...");
        let ref_lat = load_npy_latent(ref_path, &device)?;
        let ref_shape = ref_lat.dims();
        eprintln!(
            "[codec_e2e] Reference shape: [{}, {}, {}]",
            ref_shape[0], ref_shape[1], ref_shape[2]
        );

        let t_min = lat_shape[1].min(ref_shape[1]);
        let d = lat_shape[2];
        let rust_trim = latent.narrow(1, 0, t_min);
        let ref_trim = ref_lat.narrow(1, 0, t_min);

        // Per-frame error detail.
        let rust_flat: Vec<f32> = rust_trim.clone().into_data().to_vec().unwrap();
        let ref_flat: Vec<f32> = ref_trim.clone().into_data().to_vec().unwrap();
        for frame in 0..t_min {
            let r = &rust_flat[frame * d..(frame + 1) * d];
            let p = &ref_flat[frame * d..(frame + 1) * d];
            let frame_err: f32 =
                r.iter().zip(p).map(|(a, b)| (a - b).abs()).sum::<f32>() / d as f32;
            eprintln!(
                "[codec_e2e] frame {:2}: rust[0]={:.4} py[0]={:.4} mean_abs_err={:.6}",
                frame, r[0], p[0], frame_err
            );
        }

        let diff = rust_trim.sub(ref_trim).abs();
        let diff_data: Vec<f32> = diff.clone().into_data().to_vec().unwrap();
        let mean_err = diff_data.iter().copied().sum::<f32>() / diff_data.len() as f32;
        let max_err = diff_data.iter().copied().fold(0.0_f32, f32::max);

        eprintln!("[codec_e2e] Mean abs error: {mean_err:.6}");
        eprintln!("[codec_e2e] Max  abs error: {max_err:.6}");

        if mean_err as f64 > args.atol {
            anyhow::bail!(
                "Parity check FAILED: mean error {mean_err:.6} > atol {}",
                args.atol
            );
        }
        eprintln!(
            "[codec_e2e] Parity check PASSED (mean error {mean_err:.6} ≤ {})",
            args.atol
        );
    } else {
        eprintln!("[codec_e2e] No reference latent provided; skipping parity check.");
    }

    Ok(())
}

// ─── Audio loading ───────────────────────────────────────────────────────────

fn load_wav_as_tensor(path: &PathBuf, device: &<B as Backend>::Device) -> Result<Tensor<B, 3>> {
    // Read WAV with hound, normalise integer samples to [-1, 1] by bit depth.
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("Cannot open WAV: {}", path.display()))?;
    let spec = reader.spec();
    let scale = 2.0_f32.powi(spec.bits_per_sample as i32 - 1); // e.g. 32768 for 16-bit
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?,
        hound::SampleFormat::Int => reader
            .into_samples::<i32>()
            .map(|s| s.map(|v| v as f32 / scale))
            .map(|s| s.map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?,
    };

    // Take mono (first channel) if stereo.
    let mono: Vec<f32> = if spec.channels == 1 {
        samples
    } else {
        samples
            .chunks(spec.channels as usize)
            .map(|ch| ch[0])
            .collect()
    };

    let n = mono.len();
    let data = TensorData::new(mono, Shape::new([1usize, 1, n]));
    Ok(Tensor::<B, 3>::from_data(data, device))
}

// ─── NPY loading (simple float32 only) ───────────────────────────────────────

fn load_npy_latent(path: &PathBuf, device: &<B as Backend>::Device) -> Result<Tensor<B, 3>> {
    let bytes =
        std::fs::read(path).with_context(|| format!("Cannot read npy: {}", path.display()))?;
    let (_header, data_bytes) =
        parse_npy_f32(&bytes).with_context(|| format!("Cannot parse npy: {}", path.display()))?;
    let n = data_bytes.len() / 4;
    let floats: Vec<f32> = data_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    // Assume shape [1, T, 32].
    let t = n / 32;
    let data = TensorData::new(floats, Shape::new([1usize, t, 32]));
    Ok(Tensor::<B, 3>::from_data(data, device))
}

/// Minimal .npy parser — handles only 1-D or N-D float32 C-order arrays.
fn parse_npy_f32(bytes: &[u8]) -> Result<(&str, &[u8])> {
    anyhow::ensure!(bytes.len() > 10, "npy too short");
    anyhow::ensure!(&bytes[0..6] == b"\x93NUMPY", "not a .npy file");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header_start = 10;
    let data_start = header_start + header_len;
    anyhow::ensure!(bytes.len() > data_start, "npy header truncated");
    let header =
        std::str::from_utf8(&bytes[header_start..data_start]).context("npy header is not UTF-8")?;
    Ok((header, &bytes[data_start..]))
}
