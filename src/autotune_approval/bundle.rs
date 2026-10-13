//! Bind approval to the actual persistent entries, including warm-cache hits.
use super::*;
use cubecl::bundle::{Bundle, EmbeddedBundle};
use std::collections::BTreeMap;

/// Execution settings not described by the device/model identity. The digest
/// covers sampler and route settings; fixture hashes bind inputs and geometry.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AutotuneExecutionIdentity {
    pub kernel_environment: String,
    /// Exact application executable which will consume these selections.
    pub production_binary_sha256: String,
    pub settings_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AutotuneBundleBinding {
    pub execution: AutotuneExecutionIdentity,
    pub payload_sha256: String,
    pub entry_count: usize,
}

impl AutotuneBundleBinding {
    pub(super) fn validate(&self) -> Result<(), AutotuneApprovalError> {
        validate_sha256("autotune bundle", &self.payload_sha256)?;
        validate_sha256("execution settings", &self.execution.settings_sha256)?;
        validate_sha256(
            "production binary",
            &self.execution.production_binary_sha256,
        )?;
        if self.entry_count == 0 || self.execution.kernel_environment.is_empty() {
            return Err(AutotuneApprovalError::InvalidApprovedManifest);
        }
        Ok(())
    }
}

/// A verified, immutable snapshot. This owns the bytes which were checked so a
/// file replacement between verification and import cannot change selections.
#[derive(Debug)]
pub struct VerifiedAutotuneBundle {
    bundle: EmbeddedBundle,
    entries: usize,
}

fn open(bytes: Vec<u8>) -> Result<EmbeddedBundle, AutotuneApprovalError> {
    EmbeddedBundle::open(cubecl_environment::bytes::Bytes::from_bytes_vec(bytes))
        .map_err(|error| AutotuneApprovalError::Bundle(error.to_string()))
}

fn entry_count(bundle: &dyn Bundle) -> Result<usize, AutotuneApprovalError> {
    let mut count = 0;
    let namespaces = bundle.namespaces();
    for namespace in namespaces {
        if !namespace.starts_with("autotune/") {
            return Err(AutotuneApprovalError::Bundle(format!(
                "non-autotune namespace: {namespace}"
            )));
        }
        bundle.scan(&namespace, &mut |_, _| count += 1);
    }
    if count == 0 {
        return Err(AutotuneApprovalError::EmptySelectionVector);
    }
    Ok(count)
}

/// Seal exported CubeCL flat-bundle bytes after a complete RF or MF trajectory
/// passes the supplied precision-specific policy. Unlike recorder logs, this
/// also includes decisions reused from an earlier process.
pub fn seal_autotune_bundle(
    policy: &AutotuneAccuracyPolicy,
    actual_identity: &AutotuneRuntimeIdentity,
    evidence: AutotuneAccuracyEvidence,
    execution: AutotuneExecutionIdentity,
    bytes: Vec<u8>,
) -> Result<ApprovedAutotuneCacheManifest, AutotuneApprovalError> {
    policy.validate()?;
    policy.verify_identity(actual_identity)?;
    evidence.validate(policy)?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    let bundle = open(bytes)?;
    let binding = AutotuneBundleBinding {
        execution,
        payload_sha256: digest,
        entry_count: entry_count(&bundle)?,
    };
    binding.validate()?;
    Ok(ApprovedAutotuneCacheManifest {
        schema_version: AUTOTUNE_APPROVAL_SCHEMA_VERSION,
        identity: actual_identity.clone(),
        required_cases: policy.required_cases.clone(),
        evidence,
        selections: Vec::new(),
        bundle: Some(binding),
    })
}

impl ApprovedAutotuneCacheManifest {
    pub fn verify_bundle(
        &self,
        actual_identity: &AutotuneRuntimeIdentity,
        execution: &AutotuneExecutionIdentity,
        bytes: Vec<u8>,
    ) -> Result<VerifiedAutotuneBundle, AutotuneApprovalError> {
        self.validate()?;
        verify_identity(&self.identity, actual_identity)?;
        let binding = self.bundle.as_ref().ok_or_else(|| {
            AutotuneApprovalError::Bundle("approval has no payload binding".into())
        })?;
        if &binding.execution != execution
            || binding.payload_sha256 != format!("{:x}", Sha256::digest(&bytes))
        {
            return Err(AutotuneApprovalError::Bundle(
                "payload or execution settings differ from approval".into(),
            ));
        }
        let bundle = open(bytes)?;
        if entry_count(&bundle)? != binding.entry_count {
            return Err(AutotuneApprovalError::InvalidApprovedManifest);
        }
        Ok(VerifiedAutotuneBundle {
            bundle,
            entries: binding.entry_count,
        })
    }
}

impl VerifiedAutotuneBundle {
    pub fn entry_count(&self) -> usize {
        self.entries
    }

    /// Import into the already configured CubeCL environment before creating
    /// any GPU/tuner. Conflicting or extra selections are rejected. Successful
    /// restore verifies every persisted value, not just the recorder log.
    ///
    /// This does not freeze future tuning: an unseen workload can add entries
    /// and must pass a new trajectory gate before that enlarged cache is sealed.
    /// An I/O failure can leave a partial import of approved entries; it returns
    /// an error and is safe to retry. Existing different entries are never erased.
    pub fn restore(&self) -> Result<(), AutotuneApprovalError> {
        self.check_active(false)?;
        let result = cubecl::bundle::import(&self.bundle);
        if result.failed != 0 {
            return Err(AutotuneApprovalError::Bundle(format!(
                "{} cache entries could not be persisted",
                result.failed
            )));
        }
        self.check_active(true)
    }

    /// Recheck the current persistent selection vector after a workload. It
    /// detects additional keys and changed choices; call outside timed work.
    pub fn verify_active(&self) -> Result<(), AutotuneApprovalError> {
        self.check_active(true)
    }

    fn check_active(&self, require_complete: bool) -> Result<(), AutotuneApprovalError> {
        let namespaces = self.bundle.namespaces();
        for summary in cubecl::environment::namespaces() {
            let namespace = summary.namespace;
            if namespace.starts_with("autotune/") && !namespaces.contains(&namespace) {
                return Err(AutotuneApprovalError::Bundle(format!(
                    "unapproved namespace: {namespace}"
                )));
            }
        }
        for namespace in namespaces {
            let mut expected = BTreeMap::new();
            self.bundle.scan(&namespace, &mut |key, value| {
                expected.insert(key.to_vec(), value.to_vec());
            });
            let storage = cubecl_environment::persistence::storage::open(&namespace);
            let mut actual = BTreeMap::new();
            storage.scan(&mut |key, value| {
                actual.insert(key.to_vec(), value.to_vec());
            });
            if actual
                .iter()
                .any(|(key, value)| expected.get(key) != Some(value))
                || (require_complete && actual != expected)
            {
                return Err(AutotuneApprovalError::Bundle(format!(
                    "persistent selections differ in {namespace}"
                )));
            }
        }
        Ok(())
    }
}
