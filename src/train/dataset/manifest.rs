//! JSONL manifest loading and in-memory dataset.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::IrodoriError;

// ---------------------------------------------------------------------------
// JSONL record (internal)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ManifestRecord {
    text: String,
    latent_path: String,
    ref_latent_path: Option<String>,
    /// Free-form caption for caption-conditioned models (VoiceDesign mode).
    caption: Option<String>,
}

// ---------------------------------------------------------------------------
// Manifest sample
// ---------------------------------------------------------------------------

/// A single training sample as loaded from the manifest.
#[derive(Debug, Clone)]
pub struct ManifestSample {
    pub text: String,
    pub latent_path: PathBuf,
    pub ref_latent_path: Option<PathBuf>,
    /// Optional caption for caption-conditioned training.
    pub caption: Option<String>,
}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

/// An in-memory dataset backed by a JSONL manifest file.
pub struct ManifestDataset {
    pub samples: Vec<ManifestSample>,
}

impl ManifestDataset {
    /// Load the dataset from a JSONL manifest.
    ///
    /// Relative paths in the manifest are resolved relative to the manifest's
    /// parent directory.
    pub fn load(path: &Path) -> crate::error::Result<Self> {
        let parent = path.parent().unwrap_or(Path::new("."));
        let text = std::fs::read_to_string(path).map_err(|e| {
            IrodoriError::Dataset(format!("cannot read manifest {}: {e}", path.display()))
        })?;

        let samples = text
            .lines()
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty())
            .map(|(i, line)| -> crate::error::Result<ManifestSample> {
                let rec: ManifestRecord = serde_json::from_str(line)
                    .map_err(|e| IrodoriError::Dataset(format!("manifest line {}: {e}", i + 1)))?;
                Ok(ManifestSample {
                    text: rec.text,
                    latent_path: parent.join(&rec.latent_path),
                    ref_latent_path: rec.ref_latent_path.map(|p| parent.join(p)),
                    caption: rec.caption.filter(|c| !c.trim().is_empty()),
                })
            })
            .collect::<crate::error::Result<Vec<_>>>()?;

        Ok(Self { samples })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn write_manifest(dir: &Path, entries: &[(&str, &str, Option<&str>)]) -> PathBuf {
        let manifest_path = dir.join("manifest.jsonl");
        let mut f = std::fs::File::create(&manifest_path).expect("create manifest");
        for (text, latent, ref_latent) in entries {
            let rec = if let Some(rl) = ref_latent {
                serde_json::json!({"text": text, "latent_path": latent, "ref_latent_path": rl})
            } else {
                serde_json::json!({"text": text, "latent_path": latent})
            };
            writeln!(f, "{}", rec).expect("write manifest line");
        }
        manifest_path
    }

    #[test]
    fn load_manifest_basic() {
        let dir = TempDir::new().unwrap();
        let manifest = write_manifest(
            dir.path(),
            &[
                ("hello", "a.safetensors", None),
                ("world", "b.safetensors", Some("ref.safetensors")),
            ],
        );
        let ds = ManifestDataset::load(&manifest).unwrap();
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.samples[0].text, "hello");
        assert!(ds.samples[0].ref_latent_path.is_none());
        assert_eq!(ds.samples[1].text, "world");
        assert!(ds.samples[1].ref_latent_path.is_some());
    }

    #[test]
    fn load_manifest_blank_lines_skipped() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("manifest.jsonl");
        std::fs::write(
            &path,
            concat!(
                "{\"text\":\"a\",\"latent_path\":\"a.safetensors\"}\n",
                "\n   \n",
                "{\"text\":\"b\",\"latent_path\":\"b.safetensors\"}\n",
            ),
        )
        .unwrap();
        let ds = ManifestDataset::load(&path).unwrap();
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn load_manifest_resolves_paths_relative_to_manifest() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        let manifest = write_manifest(&sub, &[("t", "data/a.safetensors", None)]);
        let ds = ManifestDataset::load(&manifest).unwrap();
        assert!(ds.samples[0].latent_path.starts_with(&sub));
    }

    #[test]
    fn load_manifest_empty_is_ok() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.jsonl");
        std::fs::write(&path, "").unwrap();
        let ds = ManifestDataset::load(&path).unwrap();
        assert!(ds.is_empty());
    }
}
