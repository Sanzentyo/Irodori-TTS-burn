//! Reference preprocessing and PCM output, independent of the GPU runtime.

use std::{f64::consts::PI, io::Write, path::Path};

use crate::{IrodoriError, Result};

pub fn read_wav(path: &Path, max_seconds: f64) -> Result<(Vec<f32>, u32)> {
    let mut reader =
        hound::WavReader::open(path).map_err(|e| IrodoriError::Audio(e.to_string()))?;
    let spec = reader.spec();
    if spec.channels == 0
        || spec.sample_rate == 0
        || !(max_seconds.is_finite() && max_seconds > 0.0)
    {
        return Err(IrodoriError::Audio(
            "invalid sample rate, channels or reference duration limit".into(),
        ));
    }
    if f64::from(reader.duration()) / f64::from(spec.sample_rate) > max_seconds {
        return Err(IrodoriError::Audio(format!(
            "reference exceeds {max_seconds} seconds"
        )));
    }
    let interleaved = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>(),
        hound::SampleFormat::Int => {
            if !(1..=32).contains(&spec.bits_per_sample) {
                return Err(IrodoriError::Audio("unsupported PCM bit depth".into()));
            }
            let denominator = (1_u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / denominator))
                .collect()
        }
    }
    .map_err(|e| IrodoriError::Audio(e.to_string()))?;
    if interleaved.is_empty()
        || !interleaved.len().is_multiple_of(usize::from(spec.channels))
        || interleaved.iter().any(|v| !v.is_finite())
    {
        return Err(IrodoriError::Audio(
            "reference audio must contain finite, complete sample frames".into(),
        ));
    }
    let mono = interleaved
        .chunks_exact(usize::from(spec.channels))
        .map(|frame| {
            (frame.iter().map(|v| f64::from(*v)).sum::<f64>() / f64::from(spec.channels)) as f32
        })
        .collect();
    Ok((mono, spec.sample_rate))
}

fn i0(x: f64) -> f64 {
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..128 {
        term *= x * x / (4.0 * f64::from(k * k));
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// Kaiser-windowed sinc matching the official torchaudio reference policy.
pub fn resample(samples: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == 0 || to == 0 || samples.iter().any(|v| !v.is_finite()) {
        return Err(IrodoriError::Audio(
            "resampling requires positive rates and finite samples".into(),
        ));
    }
    if from == to {
        return Ok(samples.to_vec());
    }
    let (mut a, mut b) = (from, to);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    let (from, to) = (from / a, to / a);
    if from > 4096 || to > 4096 {
        return Err(IrodoriError::Audio(
            "sample-rate ratio exceeds 4096; convert to a standard recording rate first".into(),
        ));
    }
    let base = f64::from(from.min(to)) * 0.947_593_716_739_959_6;
    let width = (64.0 * f64::from(from) / base).ceil() as i64;
    let beta = f64::from(14.769_656_459_379_492_f64 as f32);
    let peak = f64::from(i0(beta) as f32);
    let mut phases = Vec::with_capacity(to as usize);
    for phase in 0..to {
        let taps = (-width..width + i64::from(from))
            .filter_map(|offset| {
                let t = (offset as f64 / f64::from(from) - f64::from(phase) / f64::from(to)) * base;
                if t.abs() >= 64.0 {
                    return None;
                }
                let x = PI * t;
                let sinc = if x == 0.0 { 1.0 } else { x.sin() / x };
                let window = i0(beta * (1.0 - (t / 64.0).powi(2)).sqrt()) / peak;
                Some((offset, sinc * window * base / f64::from(from)))
            })
            .collect::<Vec<_>>();
        phases.push(taps);
    }
    let count = ((samples.len() as f64 * f64::from(to) / f64::from(from)) as f32).ceil() as usize;
    Ok((0..count)
        .map(|index| {
            let start = (index / to as usize) as i64 * i64::from(from);
            phases[index % to as usize]
                .iter()
                .filter_map(|(offset, weight)| {
                    usize::try_from(start + offset)
                        .ok()
                        .and_then(|i| samples.get(i))
                        .map(|v| f64::from(*v) * weight)
                })
                .sum::<f64>() as f32
        })
        .collect())
}

struct Biquad {
    b: [f64; 3],
    a: [f64; 2],
}

impl Biquad {
    fn run(&self, samples: &mut [f64]) {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        for sample in samples {
            let y = self.b[0] * *sample + self.b[1] * x1 + self.b[2] * x2
                - self.a[0] * y1
                - self.a[1] * y2;
            (x2, x1, y2, y1) = (x1, *sample, y1, y);
            *sample = y;
        }
    }
}

/// Integrated BS.1770 loudness with pyloudnorm's K-weighting and block gates.
pub fn integrated_loudness(samples: &[f32], sample_rate: u32) -> Result<f64> {
    if sample_rate < 8000 || samples.is_empty() || samples.iter().any(|v| !v.is_finite()) {
        return Err(IrodoriError::Audio(
            "loudness requires finite audio at 8000 Hz or higher".into(),
        ));
    }
    let mut data = samples.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
    data.resize(data.len().max(sample_rate as usize / 2), 0.0);
    let a = 10.0_f64.powf(4.0 / 40.0);
    let w = 2.0 * PI * 1500.0 / f64::from(sample_rate);
    let c = w.cos();
    let s = 2.0 * a.sqrt() * w.sin() / (2.0 / 2.0_f64.sqrt());
    let a0 = a + 1.0 - (a - 1.0) * c + s;
    Biquad {
        b: [
            a * (a + 1.0 + (a - 1.0) * c + s) / a0,
            -2.0 * a * (a - 1.0 + (a + 1.0) * c) / a0,
            a * (a + 1.0 + (a - 1.0) * c - s) / a0,
        ],
        a: [
            2.0 * (a - 1.0 - (a + 1.0) * c) / a0,
            (a + 1.0 - (a - 1.0) * c - s) / a0,
        ],
    }
    .run(&mut data);
    let w = 2.0 * PI * 38.0 / f64::from(sample_rate);
    let c = w.cos();
    let a0 = 1.0 + w.sin();
    Biquad {
        b: [(1.0 + c) / 2.0 / a0, -(1.0 + c) / a0, (1.0 + c) / 2.0 / a0],
        a: [-2.0 * c / a0, (1.0 - w.sin()) / a0],
    }
    .run(&mut data);
    let block = (0.4 * f64::from(sample_rate)) as usize;
    let step = (0.1 * f64::from(sample_rate)) as usize;
    let blocks =
        ((data.len() as f64 / f64::from(sample_rate) - 0.4) / 0.1).round_ties_even() as usize + 1;
    data.resize((blocks - 1) * step + block, 0.0);
    let power = (0..blocks)
        .map(|i| {
            data[i * step..i * step + block]
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                / (0.4 * f64::from(sample_rate))
        })
        .collect::<Vec<_>>();
    let gated = |threshold: f64| {
        let mut count = 0;
        let sum = power
            .iter()
            .copied()
            .filter(|p| -0.691 + 10.0 * p.log10() > threshold)
            .inspect(|_| count += 1)
            .sum::<f64>();
        if count == 0 { 0.0 } else { sum / count as f64 }
    };
    let relative = -0.691 + 10.0 * gated(-70.0).log10() - 10.0;
    Ok((-0.691 + 10.0 * gated(relative.max(-70.0)).log10()).max(-70.0))
}

pub fn normalize_loudness(samples: &mut [f32], rate: u32, target: Option<f64>) -> Result<()> {
    if let Some(target) = target {
        if !target.is_finite() || !(-70.0..=0.0).contains(&target) {
            return Err(IrodoriError::Audio(
                "reference loudness must be in [-70, 0] LUFS".into(),
            ));
        }
        let gain =
            ((target - integrated_loudness(samples, rate)?) * 10.0_f64.ln() / 20.0).exp() as f32;
        samples.iter_mut().for_each(|v| *v *= gain);
    }
    let peak = samples.iter().map(|v| v.abs()).fold(1.0_f32, f32::max);
    samples.iter_mut().for_each(|v| *v /= peak);
    Ok(())
}

pub fn pcm16(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|sample| ((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes())
        .collect()
}

pub fn write_wav(mut output: impl Write, samples: &[f32], rate: u32) -> Result<()> {
    if rate == 0 || rate > u32::MAX / 2 || samples.iter().any(|value| !value.is_finite()) {
        return Err(IrodoriError::Audio(
            "WAV requires a valid sample rate and finite samples".into(),
        ));
    }
    let bytes = samples
        .len()
        .checked_mul(2)
        .and_then(|n| u32::try_from(n).ok())
        .filter(|n| *n <= u32::MAX - 36)
        .ok_or_else(|| IrodoriError::Audio("WAV exceeds the RIFF size limit".into()))?;
    output.write_all(b"RIFF")?;
    output.write_all(&(bytes + 36).to_le_bytes())?;
    output.write_all(b"WAVEfmt ")?;
    output.write_all(&16_u32.to_le_bytes())?;
    output.write_all(&1_u16.to_le_bytes())?;
    output.write_all(&1_u16.to_le_bytes())?;
    output.write_all(&rate.to_le_bytes())?;
    output.write_all(&(rate * 2).to_le_bytes())?;
    output.write_all(&2_u16.to_le_bytes())?;
    output.write_all(&16_u16.to_le_bytes())?;
    output.write_all(b"data")?;
    output.write_all(&bytes.to_le_bytes())?;
    output.write_all(&pcm16(samples))?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preprocessing_matches_official_torchaudio_and_pyloudnorm() {
        #[derive(serde::Deserialize)]
        struct Case {
            rate: u32,
            count: usize,
            indices: Vec<usize>,
            expected: Vec<f32>,
            output_count: usize,
            loudness: f64,
        }
        #[derive(serde::Deserialize)]
        struct Fixture {
            cases: Vec<Case>,
        }
        let fixture: Fixture =
            serde_json::from_str(include_str!("../tests/fixtures/audio-preprocessing.json"))
                .unwrap();
        for case in fixture.cases {
            let input = (0..case.count)
                .map(|i| {
                    let i = i as f64;
                    ((0.15 * (2.0 * PI * 173.0 * i / f64::from(case.rate)).sin()
                        + 0.07 * (2.0 * PI * 17731.0 * i / f64::from(case.rate)).sin())
                        * (0.7 + 0.3 * (PI * i / case.count as f64).sin()))
                        as f32
                })
                .collect::<Vec<_>>();
            let output = resample(&input, case.rate, 48000).unwrap();
            assert_eq!(output.len(), case.output_count);
            for (index, expected) in case.indices.into_iter().zip(case.expected) {
                assert!(
                    (output[index] - expected).abs() < 2e-6,
                    "rate={} sample={index} actual={} expected={expected}",
                    case.rate,
                    output[index]
                );
            }
            let loudness = integrated_loudness(&input, case.rate).unwrap();
            assert!(
                (loudness - case.loudness).abs() < 1e-5,
                "rate={} actual={loudness} expected={}",
                case.rate,
                case.loudness
            );
        }
    }
    #[test]
    fn resampling_rejects_aliasing_and_preserves_dc() {
        let input = (0..4800)
            .map(|i| (2.0 * PI * 18000.0 * i as f64 / 48000.0).sin() as f32)
            .collect::<Vec<_>>();
        let output = resample(&input, 48000, 16000).unwrap();
        assert_eq!(output.len(), 1600);
        assert!(output[100..1500].iter().all(|v| v.abs() < 1e-4));
        let dc = resample(&vec![0.5; 1000], 16000, 48000).unwrap();
        assert!(dc[200..2800].iter().all(|v| (v - 0.5).abs() < 1e-4));
    }
    #[test]
    fn loudness_normalizes_to_target_and_silence_remains_finite() {
        let mut data = (0..48000)
            .map(|i| 0.01 * (2.0 * PI * 1000.0 * i as f64 / 48000.0).sin() as f32)
            .collect::<Vec<_>>();
        normalize_loudness(&mut data, 48000, Some(-16.0)).unwrap();
        assert!((integrated_loudness(&data, 48000).unwrap() + 16.0).abs() < 1e-4);
        let mut silence = vec![0.0; 100];
        normalize_loudness(&mut silence, 48000, Some(-16.0)).unwrap();
        assert!(silence.iter().all(|v| *v == 0.0));
    }
    #[test]
    fn wav_has_correct_length_rate_and_signed_samples() {
        let mut bytes = Vec::new();
        write_wav(&mut bytes, &[-1.0, 0.0, 1.0], 48000).unwrap();
        let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(reader.spec().sample_rate, 48000);
        assert_eq!(
            reader
                .samples::<i16>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .unwrap(),
            [-32767, 0, 32767]
        );
    }
}
