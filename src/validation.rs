//! Reproducible numerical metrics used for waveform parity checks.

use serde::{Deserialize, Serialize};

/// Direct, sample-aligned comparison of two finite mono waveforms.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct AudioMetrics {
    pub sample_count: usize,
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
    pub root_mean_square_error: f64,
    pub signal_to_noise_db: f64,
    pub cosine_similarity: f64,
}

impl AudioMetrics {
    /// Compare equal-length, sample-aligned signals.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, unequal-length, or non-finite input.
    pub fn compare(reference: &[f32], candidate: &[f32]) -> Result<Self, AudioMetricError> {
        if reference.is_empty() {
            return Err(AudioMetricError::Empty);
        }
        if reference.len() != candidate.len() {
            return Err(AudioMetricError::LengthMismatch {
                reference: reference.len(),
                candidate: candidate.len(),
            });
        }

        let sums = reference.iter().zip(candidate).try_fold(
            MetricSums::default(),
            |mut sums, (&expected, &actual)| {
                if !expected.is_finite() || !actual.is_finite() {
                    return Err(AudioMetricError::NonFinite);
                }
                let expected = f64::from(expected);
                let actual = f64::from(actual);
                let error = actual - expected;
                sums.max_abs_error = sums.max_abs_error.max(error.abs());
                sums.absolute_error += error.abs();
                sums.squared_error += error * error;
                sums.reference_energy += expected * expected;
                sums.candidate_energy += actual * actual;
                sums.dot_product += expected * actual;
                Ok(sums)
            },
        )?;

        let count = reference.len() as f64;
        let root_mean_square_error = (sums.squared_error / count).sqrt();
        let signal_to_noise_db = if sums.squared_error == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (sums.reference_energy / sums.squared_error).log10()
        };
        let cosine_denominator = (sums.reference_energy * sums.candidate_energy).sqrt();
        let cosine_similarity = if cosine_denominator == 0.0 {
            if sums.reference_energy == sums.candidate_energy {
                1.0
            } else {
                0.0
            }
        } else {
            sums.dot_product / cosine_denominator
        };

        Ok(Self {
            sample_count: reference.len(),
            max_abs_error: sums.max_abs_error,
            mean_abs_error: sums.absolute_error / count,
            root_mean_square_error,
            signal_to_noise_db,
            cosine_similarity,
        })
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AudioMetricError {
    #[error("waveforms must not be empty")]
    Empty,
    #[error("waveform length mismatch: reference={reference}, candidate={candidate}")]
    LengthMismatch { reference: usize, candidate: usize },
    #[error("waveforms must contain only finite samples")]
    NonFinite,
}

#[derive(Default)]
struct MetricSums {
    max_abs_error: f64,
    absolute_error: f64,
    squared_error: f64,
    reference_energy: f64,
    candidate_energy: f64,
    dot_product: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_signal_has_exact_metrics() {
        let signal = [0.25, -0.5, 1.0, 0.0];
        let metrics = AudioMetrics::compare(&signal, &signal).unwrap();
        assert_eq!(metrics.max_abs_error, 0.0);
        assert_eq!(metrics.mean_abs_error, 0.0);
        assert_eq!(metrics.root_mean_square_error, 0.0);
        assert_eq!(metrics.signal_to_noise_db, f64::INFINITY);
        assert!((metrics.cosine_similarity - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn known_error_metrics_match_hand_calculation() {
        let reference = [1.0, -1.0];
        let candidate = [0.5, -1.5];
        let metrics = AudioMetrics::compare(&reference, &candidate).unwrap();
        assert_eq!(metrics.max_abs_error, 0.5);
        assert_eq!(metrics.mean_abs_error, 0.5);
        assert_eq!(metrics.root_mean_square_error, 0.5);
        assert!((metrics.signal_to_noise_db - 6.020_599_913).abs() < 1e-9);
        assert!((metrics.cosine_similarity - 0.894_427_191).abs() < 1e-9);
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert_eq!(
            AudioMetrics::compare(&[], &[]),
            Err(AudioMetricError::Empty)
        );
        assert_eq!(
            AudioMetrics::compare(&[0.0], &[0.0, 1.0]),
            Err(AudioMetricError::LengthMismatch {
                reference: 1,
                candidate: 2,
            })
        );
        assert_eq!(
            AudioMetrics::compare(&[f32::NAN], &[0.0]),
            Err(AudioMetricError::NonFinite)
        );
    }
}
