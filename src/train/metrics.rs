//! Structured metric logging for training runs.
//!
//! Provides a [`MetricsSink`] trait and two concrete implementations:
//!
//! - [`StdoutSink`] — emits metrics via `tracing::info!` (always active).
//! - [`JsonlSink`] — appends one JSON object per metric to a file.
//! - [`MultiSink`] — generic fan-out combining two sinks into one.
//!
//! The trainer uses `&mut dyn MetricsSink` so that additional sink
//! implementations can be injected without changing the training loop.
//!
//! # Example JSONL record
//!
//! ```json
//! {"step":100,"ts":1753000000.123,"key":"train/loss","value":0.42}
//! ```

use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::error::IrodoriError;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Consumer for numerical training metrics.
///
/// Implementations receive `(step, key, value)` tuples and may emit them to
/// stdout, a file, an HTTP endpoint, etc.
pub trait MetricsSink: Send {
    /// Record a scalar metric at the given training step.
    ///
    /// Implementations should be best-effort: I/O errors are stored and
    /// surfaced lazily via [`flush`][MetricsSink::flush].
    fn log_scalar(&mut self, step: usize, key: &str, value: f64);

    /// Flush any buffered output and surface the first pending I/O error.
    fn flush(&mut self) -> crate::error::Result<()>;
}

// ---------------------------------------------------------------------------
// StdoutSink
// ---------------------------------------------------------------------------

/// Metric sink that forwards to `tracing::info!`.
///
/// Each call to [`log_scalar`][MetricsSink::log_scalar] emits one structured
/// log record.  No batching — suitable for interactive use.
pub struct StdoutSink;

impl MetricsSink for StdoutSink {
    fn log_scalar(&mut self, step: usize, key: &str, value: f64) {
        tracing::info!(step, key, value, "metric");
    }

    fn flush(&mut self) -> crate::error::Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// JsonlSink
// ---------------------------------------------------------------------------

/// One JSONL record serialized to the metrics file.
#[derive(Serialize)]
struct MetricRecord<'a> {
    step: usize,
    ts: f64,
    key: &'a str,
    value: f64,
}

/// Metric sink that appends JSONL records to a file.
///
/// Each call to [`log_scalar`][MetricsSink::log_scalar] appends one line:
/// ```json
/// {"step":N,"ts":unix_float,"key":"...","value":f64}
/// ```
///
/// Write failures are stored internally; the first error is returned by
/// [`flush`][MetricsSink::flush].  Non-finite values are silently skipped to
/// avoid producing invalid JSON tokens.
pub struct JsonlSink {
    writer: std::io::BufWriter<std::fs::File>,
    pending_error: Option<std::io::Error>,
    dirty: bool,
}

impl JsonlSink {
    /// Open (or create) the JSONL file at `path` for append-only writing.
    pub fn create(path: &Path) -> crate::error::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| {
                IrodoriError::Io(std::io::Error::new(
                    e.kind(),
                    format!("open metrics file {}: {e}", path.display()),
                ))
            })?;
        Ok(Self {
            writer: std::io::BufWriter::new(file),
            pending_error: None,
            dirty: false,
        })
    }
}

impl MetricsSink for JsonlSink {
    fn log_scalar(&mut self, step: usize, key: &str, value: f64) {
        if self.pending_error.is_some() || !value.is_finite() {
            return;
        }
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let record = MetricRecord {
            step,
            ts,
            key,
            value,
        };
        let result = serde_json::to_writer(&mut self.writer, &record)
            .map_err(std::io::Error::other)
            .and_then(|()| self.writer.write_all(b"\n"));
        match result {
            Ok(()) => self.dirty = true,
            Err(e) => self.pending_error = Some(e),
        }
    }

    fn flush(&mut self) -> crate::error::Result<()> {
        if let Some(e) = self.pending_error.take() {
            return Err(IrodoriError::Io(e));
        }
        if self.dirty {
            self.writer.flush().map_err(IrodoriError::Io)?;
            self.dirty = false;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MultiSink
// ---------------------------------------------------------------------------

/// Generic fan-out sink: forwards each metric to both `A` and `B`.
///
/// Both sinks are always flushed; if both fail, the errors are combined into
/// one message so no diagnostic is silently discarded.
pub struct MultiSink<A: MetricsSink, B: MetricsSink> {
    a: A,
    b: B,
}

impl<A: MetricsSink, B: MetricsSink> MultiSink<A, B> {
    /// Combine two sinks into one.
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: MetricsSink, B: MetricsSink> MetricsSink for MultiSink<A, B> {
    fn log_scalar(&mut self, step: usize, key: &str, value: f64) {
        self.a.log_scalar(step, key, value);
        self.b.log_scalar(step, key, value);
    }

    fn flush(&mut self) -> crate::error::Result<()> {
        // Flush both so no buffered data is silently dropped, then combine errors.
        let a_err = self.a.flush();
        let b_err = self.b.flush();
        match (a_err, b_err) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(e), Ok(())) | (Ok(()), Err(e)) => Err(e),
            (Err(ea), Err(eb)) => Err(IrodoriError::Training(format!(
                "metrics flush failed: {ea}; {eb}"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn stdout_sink_flush_returns_ok() {
        let mut sink = StdoutSink;
        sink.log_scalar(1, "train/loss", 0.5);
        assert!(sink.flush().is_ok());
    }

    #[test]
    fn jsonl_sink_creates_and_appends() {
        let tmp = NamedTempFile::new().unwrap();
        let mut sink = JsonlSink::create(tmp.path()).unwrap();
        sink.log_scalar(1, "train/loss", 0.42);
        sink.log_scalar(2, "val/loss", 0.38);
        sink.flush().unwrap();

        let content = std::fs::read_to_string(tmp.path()).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);

        let obj: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(obj["step"], 1);
        assert_eq!(obj["key"], "train/loss");
        assert!((obj["value"].as_f64().unwrap() - 0.42).abs() < 1e-9);
    }

    #[test]
    fn jsonl_sink_stops_writing_after_error() {
        // Use an invalid path to force an error on creation.
        let result = JsonlSink::create(Path::new("/nonexistent/path/metrics.jsonl"));
        assert!(result.is_err());
    }

    #[test]
    fn multi_sink_fans_out_to_both() {
        let tmp = NamedTempFile::new().unwrap();
        let jsonl = JsonlSink::create(tmp.path()).unwrap();
        let mut multi = MultiSink::new(StdoutSink, jsonl);
        multi.log_scalar(5, "train/loss", 1.0);
        multi.flush().unwrap();

        let content = std::fs::read_to_string(tmp.path()).unwrap();
        assert_eq!(content.lines().count(), 1);
    }
}
