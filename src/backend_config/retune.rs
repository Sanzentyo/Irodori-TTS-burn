//! Isolated projection retuning using CubeCL's own namespace export API.
//! This is a CPU operation. The source environment is never modified.

use crate::{IrodoriError, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize)]
pub struct ProjectionRetuneSeed {
    pub baseline: PathBuf,
    pub candidate: PathBuf,
    pub baseline_sha256: String,
    pub candidate_sha256: String,
    pub projection_namespaces: Vec<String>,
    pub projection_entries: u64,
    pub retained_entries: u64,
}

#[derive(Debug, Serialize)]
pub struct AutotuneCacheSnapshot {
    pub sha256: String,
    pub entries: u64,
}

/// Inspect logical decisions through the same database engine as the runtime.
/// The source must be closed. Reading a Turso WAL with another SQLite engine is
/// not a supported way to inspect an active-format environment.
pub fn inspect_autotune_cache(path: &Path) -> Result<AutotuneCacheSnapshot> {
    use cubecl_environment::persistence::{Access, Database};
    use sha2::{Digest, Sha256};

    let database = Database::open_at(path, Access::ReadOnly)
        .map_err(|error| IrodoriError::Cache(format!("inspect autotune cache: {error}")))?;
    let mut namespaces = database
        .summary()
        .into_iter()
        .filter(|row| row.namespace.starts_with("autotune/"))
        .collect::<Vec<_>>();
    namespaces.sort_unstable_by(|a, b| a.namespace.cmp(&b.namespace));
    let mut hash = Sha256::new();
    let mut entries = 0;
    for row in namespaces {
        let mut values = Vec::new();
        database.scan(&row.namespace, &mut |key, value| {
            values.push((key.to_vec(), value.to_vec()));
        });
        if values.len() as u64 != row.entries {
            return Err(IrodoriError::Cache(format!(
                "incomplete autotune namespace read: {}",
                row.namespace
            )));
        }
        values.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        for (key, value) in values {
            for bytes in [row.namespace.as_bytes(), key.as_slice(), value.as_slice()] {
                hash.update((bytes.len() as u64).to_le_bytes());
                hash.update(bytes);
            }
            entries += 1;
        }
    }
    if entries == 0 {
        return Err(IrodoriError::Cache(
            "cache has no autotune decisions".into(),
        ));
    }
    Ok(AutotuneCacheSnapshot {
        sha256: format!("{:x}", hash.finalize()),
        entries,
    })
}

fn projection_namespace(namespace: &str) -> bool {
    namespace.starts_with("autotune/0.11.0/")
        && namespace.rsplit('/').next()
            == Some("irodori_tts_burn-kernels-projection_autotune-validated-projection-v1")
}

/// Snapshot a closed source environment and export a second snapshot without
/// projection selections. All compilation and unrelated tuning entries stay.
/// The output directory must be new. Partial evidence is retained on failure.
pub fn prepare_projection_retune(source: &Path, output: &Path) -> Result<ProjectionRetuneSeed> {
    if !source.is_file() {
        return Err(IrodoriError::Cache(format!(
            "source environment is not a file: {}",
            source.display()
        )));
    }
    let name = source
        .file_name()
        .ok_or_else(|| IrodoriError::Cache("missing environment filename".into()))?;
    std::fs::create_dir(output)?;
    let baseline = output.join("baseline.db");
    let candidate = output.join(name);
    if candidate == baseline {
        return Err(IrodoriError::Cache(
            "source environment cannot be named baseline.db".into(),
        ));
    }
    let options = cubecl::bundle::ExportOptions {
        name: "irodori-projection-retune".into(),
        ..Default::default()
    };
    let error = |error| IrodoriError::Cache(format!("projection cache snapshot: {error}"));
    cubecl::bundle::export(&[source], &baseline, &options).map_err(error)?;
    let summary = cubecl::bundle::SqliteBundle::open(&baseline)
        .map_err(error)?
        .summary();
    let projection_namespaces = summary
        .iter()
        .filter(|row| projection_namespace(&row.namespace))
        .map(|row| row.namespace.clone())
        .collect::<Vec<_>>();
    let projection_entries = summary
        .iter()
        .filter(|row| projection_namespace(&row.namespace))
        .map(|row| row.entries)
        .sum();
    let retained_entries = summary
        .iter()
        .filter(|row| !projection_namespace(&row.namespace))
        .map(|row| row.entries)
        .sum();
    if projection_namespaces.is_empty() || projection_entries == 0 {
        return Err(IrodoriError::Cache(
            "source has no validated projection decisions to retune".into(),
        ));
    }
    cubecl::bundle::export(
        &[&baseline],
        &candidate,
        &cubecl::bundle::ExportOptions {
            excluded_namespaces: projection_namespaces.clone(),
            ..options
        },
    )
    .map_err(error)?;
    let candidate_summary = cubecl::bundle::SqliteBundle::open(&candidate)
        .map_err(error)?
        .summary();
    if candidate_summary
        .iter()
        .any(|row| projection_namespace(&row.namespace))
        || candidate_summary.iter().map(|row| row.entries).sum::<u64>() != retained_entries
    {
        return Err(IrodoriError::Cache(
            "projection snapshot retained unexpected namespaces or entry counts".into(),
        ));
    }
    let receipt = ProjectionRetuneSeed {
        baseline_sha256: crate::sha256_file(&baseline)?,
        candidate_sha256: crate::sha256_file(&candidate)?,
        baseline,
        candidate,
        projection_namespaces,
        projection_entries,
        retained_entries,
    };
    std::fs::write(
        output.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_snapshot_ignores_insertion_order_and_compilation_records() {
        use cubecl_environment::persistence::{Access, Database, Insertion, Origin};

        let directory = tempfile::tempdir().unwrap();
        let paths = [directory.path().join("a.db"), directory.path().join("b.db")];
        for (index, path) in paths.iter().enumerate() {
            let database = Database::open_at(path, Access::ReadWrite).unwrap();
            let keys = if index == 0 { [1, 2] } else { [2, 1] };
            for key in keys {
                assert!(matches!(
                    database.insert(
                        "autotune/0.11.0/device/test",
                        &[key],
                        &[key + 1],
                        Origin::Local
                    ),
                    Insertion::Stored
                ));
            }
            assert!(matches!(
                database.insert("compilation", b"shader", &[index as u8], Origin::Local),
                Insertion::Stored
            ));
        }
        let first = inspect_autotune_cache(&paths[0]).unwrap();
        assert_eq!(first.entries, 2);
        assert_eq!(
            first.sha256,
            inspect_autotune_cache(&paths[1]).unwrap().sha256
        );
        {
            let database = Database::open_at(&paths[1], Access::ReadWrite).unwrap();
            assert!(matches!(
                database.replace("autotune/0.11.0/device/test", &[1], &[9], Origin::Local),
                Insertion::Stored
            ));
        }
        assert_ne!(
            first.sha256,
            inspect_autotune_cache(&paths[1]).unwrap().sha256
        );
        assert!(inspect_autotune_cache(&directory.path().join("missing.db")).is_err());
    }

    #[test]
    fn selects_only_current_projection_tuners_on_any_device() {
        let name = "irodori_tts_burn-kernels-projection_autotune-validated-projection-v1";
        assert!(projection_namespace(&format!(
            "autotune/0.11.0/device-1-wgpu<spirv>/{name}"
        )));
        for wrong in [
            format!("autotune/0.10.0/device/{name}"),
            format!("compilation/0.11.0/device/{name}"),
            format!("autotune/0.11.0/device/{name}-other"),
            format!("autotune/0.11.0/device/{name}/child"),
        ] {
            assert!(!projection_namespace(&wrong));
        }
    }
}
