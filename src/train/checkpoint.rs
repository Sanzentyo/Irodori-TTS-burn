//! LoRA adapter checkpoint saving in PEFT-compatible format.
//!
//! Two public entry points:
//!
//! * [`save_lora_adapter`] — writes only adapter weights + `adapter_config.json`
//!   (PEFT-compatible; no optimizer state; suitable for distributing final adapters).
//! * [`save_checkpoint`] — full training checkpoint: adapter weights, adapter config,
//!   optimizer state, and [`TrainingMeta`]; everything written atomically under one
//!   temp-dir → rename so no partial checkpoint is ever visible.
//!
//! Key naming convention (matches the `base_model.model.` prefix that
//! `src/lora.rs` strips when loading adapters):
//! ```text
//! base_model.model.blocks.{i}.attention.{proj}.lora_A.default.weight  // [r, in]
//! base_model.model.blocks.{i}.attention.{proj}.lora_B.default.weight  // [out, r]
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend;
use safetensors::{Dtype, tensor::TensorView};
use serde::{Deserialize, Serialize};

use crate::error::IrodoriError;
use crate::train::{LoraConfig, LoraTextToLatentRfDiT, lora_layer::LoraLinear};

// ---------------------------------------------------------------------------
// Training metadata
// ---------------------------------------------------------------------------

/// Minimal training state persisted alongside each full checkpoint.
///
/// This is written as `training_meta.json` and is used by the resume path to
/// restore the step counter and re-derive the training RNG seed.
///
/// **RNG note:** the persisted seed is used to derive a shifted seed at resume
/// time (`training_seed ^ step`) to avoid replaying the exact same random
/// sequence — however this is *not* an exact replay of the original RNG state.
/// For exact reproducibility within a run, use the same seed and restart from
/// scratch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMeta {
    /// The optimizer step at which this checkpoint was saved.
    pub step: usize,
    /// The seed originally passed to the training config (used to derive the
    /// resume RNG seed deterministically).
    pub training_seed: u64,
}

/// Load [`TrainingMeta`] from `training_meta.json` inside `checkpoint_dir`.
///
/// Returns `None` if the file does not exist (e.g. old checkpoints produced
/// before this feature was added).
pub fn load_training_meta(checkpoint_dir: &Path) -> crate::error::Result<Option<TrainingMeta>> {
    let path = checkpoint_dir.join("training_meta.json");
    if !path.exists() {
        return Ok(None);
    }
    let json = std::fs::read_to_string(&path)?;
    let meta: TrainingMeta = serde_json::from_str(&json)
        .map_err(|e| IrodoriError::Checkpoint(format!("parse training_meta.json: {e}")))?;
    Ok(Some(meta))
}

/// Collect a mapping from PEFT checkpoint key to raw [`ParamId`] value for
/// every LoRA parameter in `model`.
///
/// The key format matches `adapter_model.safetensors`
/// (e.g. `base_model.model.blocks.0.attention.wq.lora_A.default.weight`).
/// Storing this mapping alongside the optimizer state lets the resume path
/// re-apply the *original* [`ParamId`]s so that the optimizer's momentum
/// terms (keyed by [`ParamId`]) still match the restored model.
fn collect_lora_param_ids<B: Backend>(model: &LoraTextToLatentRfDiT<B>) -> HashMap<String, u64> {
    let mut map = HashMap::new();

    for (i, block) in model.blocks.iter().enumerate() {
        let attn = &block.attention;
        let pfx = format!("base_model.model.blocks.{i}.attention");

        macro_rules! collect_proj {
            ($proj:ident) => {{
                map.insert(
                    format!("{pfx}.{}.lora_A.default.weight", stringify!($proj)),
                    attn.$proj.lora_a.id.val(),
                );
                map.insert(
                    format!("{pfx}.{}.lora_B.default.weight", stringify!($proj)),
                    attn.$proj.lora_b.id.val(),
                );
            }};
        }
        macro_rules! collect_proj_opt {
            ($proj:ident) => {
                if let Some(layer) = &attn.$proj {
                    map.insert(
                        format!("{pfx}.{}.lora_A.default.weight", stringify!($proj)),
                        layer.lora_a.id.val(),
                    );
                    map.insert(
                        format!("{pfx}.{}.lora_B.default.weight", stringify!($proj)),
                        layer.lora_b.id.val(),
                    );
                }
            };
        }

        collect_proj!(wq);
        collect_proj!(wk);
        collect_proj!(wv);
        collect_proj!(wk_text);
        collect_proj!(wv_text);
        collect_proj!(gate);
        collect_proj!(wo);
        collect_proj_opt!(wk_speaker);
        collect_proj_opt!(wv_speaker);
        collect_proj_opt!(wk_caption);
        collect_proj_opt!(wv_caption);
    }

    map
}

/// Load the `param_ids.json` sidecar written by [`save_checkpoint`].
///
/// Returns `None` if the file does not exist (old checkpoint produced before
/// ParamId tracking was added — the resume path falls back to a warm restart
/// of optimizer state).
///
/// Returns an error if the file exists but is unparseable, or if duplicate
/// [`ParamId`] values are detected (which would indicate a corrupt checkpoint).
pub fn load_lora_param_ids(
    checkpoint_dir: &Path,
) -> crate::error::Result<Option<HashMap<String, u64>>> {
    let path = checkpoint_dir.join("param_ids.json");
    if !path.exists() {
        return Ok(None);
    }
    let json = std::fs::read_to_string(&path)?;
    let map: HashMap<String, u64> = serde_json::from_str(&json)
        .map_err(|e| IrodoriError::Checkpoint(format!("parse param_ids.json: {e}")))?;

    // Duplicate ParamId values indicate a corrupt or incorrectly-generated
    // checkpoint; fail early rather than silently mis-assign momentum terms.
    let mut seen = std::collections::HashSet::new();
    for &v in map.values() {
        if !seen.insert(v) {
            return Err(IrodoriError::Checkpoint(format!(
                "param_ids.json contains duplicate ParamId {v} — checkpoint may be corrupt"
            )));
        }
    }

    Ok(Some(map))
}

// ---------------------------------------------------------------------------
// adapter_config.json schema
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct AdapterConfig<'a> {
    peft_type: &'static str,
    r: usize,
    lora_alpha: f32,
    target_modules: &'a [String],
    bias: &'static str,
    task_type: &'static str,
    base_model_name_or_path: Option<String>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert f32 slice to little-endian bytes (safetensors wire format).
fn f32_to_le_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|&v| v.to_le_bytes()).collect()
}

/// Owned bytes + shape for lora_a and lora_b respectively.
type LoraBytes = (Vec<u8>, Vec<usize>, Vec<u8>, Vec<usize>);

/// Extract owned byte buffers and shape from a `LoraLinear` layer.
///
/// Weights are always serialised as F32 regardless of backend float type so that
/// the resulting adapter is loadable by both the Python PEFT library and
/// `src/lora.rs` (which expects F32 safetensors).
fn extract_lora<B: Backend>(layer: &LoraLinear<B>) -> crate::error::Result<LoraBytes> {
    let a = layer.lora_a.val();
    let b = layer.lora_b.val();
    let a_shape = a.dims().to_vec();
    let b_shape = b.dims().to_vec();
    let a_bytes = f32_to_le_bytes(
        &a.into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|e| IrodoriError::Checkpoint(format!("lora_a tensor conversion: {e:?}")))?,
    );
    let b_bytes = f32_to_le_bytes(
        &b.into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|e| IrodoriError::Checkpoint(format!("lora_b tensor conversion: {e:?}")))?,
    );
    Ok((a_bytes, a_shape, b_bytes, b_shape))
}

/// Write adapter weights (`adapter_model.safetensors`) and `adapter_config.json`
/// to an already-existing directory `dir`.
///
/// This is the shared inner implementation used by both [`save_lora_adapter`]
/// and [`save_checkpoint`].
fn write_adapter_files<B: Backend>(
    model: &LoraTextToLatentRfDiT<B>,
    lora_cfg: &LoraConfig,
    dir: &Path,
) -> crate::error::Result<()> {
    // Phase 1: collect owned byte buffers so lifetimes outlive the TensorViews.
    type Entry = (String, Vec<u8>, Vec<usize>);
    let mut entries: Vec<Entry> = Vec::new();

    let mut push = |key: String, bytes: Vec<u8>, shape: Vec<usize>| {
        entries.push((key, bytes, shape));
    };

    for (i, block) in model.blocks.iter().enumerate() {
        let attn = &block.attention;
        let pfx = format!("base_model.model.blocks.{i}.attention");

        macro_rules! save_proj {
            ($proj:ident) => {{
                let (ab, ash, bb, bsh) = extract_lora(&attn.$proj)?;
                push(
                    format!("{pfx}.{}.lora_A.default.weight", stringify!($proj)),
                    ab,
                    ash,
                );
                push(
                    format!("{pfx}.{}.lora_B.default.weight", stringify!($proj)),
                    bb,
                    bsh,
                );
            }};
        }

        macro_rules! save_proj_opt {
            ($proj:ident) => {
                if let Some(layer) = &attn.$proj {
                    let (ab, ash, bb, bsh) = extract_lora(layer)?;
                    push(
                        format!("{pfx}.{}.lora_A.default.weight", stringify!($proj)),
                        ab,
                        ash,
                    );
                    push(
                        format!("{pfx}.{}.lora_B.default.weight", stringify!($proj)),
                        bb,
                        bsh,
                    );
                }
            };
        }

        save_proj!(wq);
        save_proj!(wk);
        save_proj!(wv);
        save_proj!(wk_text);
        save_proj!(wv_text);
        save_proj!(gate);
        save_proj!(wo);
        save_proj_opt!(wk_speaker);
        save_proj_opt!(wv_speaker);
        save_proj_opt!(wk_caption);
        save_proj_opt!(wv_caption);
    }

    // Phase 2: build TensorViews borrowing from Phase 1 data.
    let views: Vec<(String, TensorView<'_>)> = entries
        .iter()
        .map(|(key, data, shape)| {
            TensorView::new(Dtype::F32, shape.clone(), data.as_slice())
                .map(|v| (key.clone(), v))
                .map_err(|e| IrodoriError::Checkpoint(format!("TensorView: {e:?}")))
        })
        .collect::<crate::error::Result<_>>()?;

    // Phase 3: serialize safetensors.
    safetensors::serialize_to_file(views, None, &dir.join("adapter_model.safetensors"))
        .map_err(|e| IrodoriError::Checkpoint(format!("serialize safetensors: {e}")))?;

    let adapter_cfg = AdapterConfig {
        peft_type: "LORA",
        r: lora_cfg.r,
        lora_alpha: lora_cfg.alpha,
        target_modules: &lora_cfg.target_modules,
        bias: "none",
        task_type: "UNCONDITIONAL_GENERATION",
        base_model_name_or_path: None,
    };
    std::fs::write(
        dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&adapter_cfg)?,
    )?;

    Ok(())
}

/// Perform the atomic temp-dir → rename pattern shared by both save functions.
///
/// Returns the final (committed) directory path.
fn atomic_save<F>(output_dir: &Path, step: usize, write_fn: F) -> crate::error::Result<PathBuf>
where
    F: FnOnce(&Path) -> crate::error::Result<()>,
{
    let final_dir = output_dir.join(format!("step-{step:07}"));
    let tmp_dir = output_dir.join(format!("step-{step:07}.tmp"));

    // Clean up any leftover temp dir from a previous crash.
    if tmp_dir.exists() {
        std::fs::remove_dir_all(&tmp_dir)?;
    }
    std::fs::create_dir_all(&tmp_dir)?;

    write_fn(&tmp_dir)?;

    // Atomic rename — remove existing final dir first if needed.
    if final_dir.exists() {
        std::fs::remove_dir_all(&final_dir)?;
    }
    std::fs::rename(&tmp_dir, &final_dir).map_err(|e| {
        IrodoriError::Checkpoint(format!(
            "rename {} → {}: {e}",
            tmp_dir.display(),
            final_dir.display()
        ))
    })?;

    Ok(final_dir)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Save LoRA adapter weights and `adapter_config.json` to `output_dir`.
///
/// Tensors are written as F32 safetensors with PEFT-compatible key names so the
/// adapter can be loaded by both the Python PEFT library and Rust `src/lora.rs`.
///
/// Output directory: `{output_dir}/step-{step:07}/`.
///
/// Uses atomic temp-dir + rename: writes to a `.tmp` directory first, then
/// renames to the final name.  No optimizer state is saved — use
/// [`save_checkpoint`] for full training checkpoints.
pub fn save_lora_adapter<B: Backend>(
    model: &LoraTextToLatentRfDiT<B>,
    lora_cfg: &LoraConfig,
    output_dir: &Path,
    step: usize,
) -> crate::error::Result<PathBuf> {
    let dir = atomic_save(output_dir, step, |tmp| {
        write_adapter_files(model, lora_cfg, tmp)
    })?;
    tracing::info!(step, path = %dir.display(), "saved LoRA adapter");
    Ok(dir)
}

/// Save a full training checkpoint atomically.
///
/// Writes the following files under `{output_dir}/step-{step:07}/` in a single
/// atomic temp-dir → rename so no partial checkpoint is ever visible:
///
/// * `adapter_model.safetensors` — PEFT-compatible LoRA weights (F32)
/// * `adapter_config.json` — PEFT adapter config
/// * `training_meta.json` — step counter and training seed for resume
/// * `param_ids.json` — `ParamId` map for accurate optimizer-state restore
/// * `optimizer.mpk` — optimizer state (burn `NamedMpkFileRecorder`)
///
/// On resume, the optimizer state is restored automatically by the trainer
/// when `LoraTrainConfig::resume_from` is set.
pub fn save_checkpoint<B, O>(
    model: &LoraTextToLatentRfDiT<B>,
    optim: &O,
    lora_cfg: &LoraConfig,
    step: usize,
    training_seed: u64,
    output_dir: &Path,
) -> crate::error::Result<PathBuf>
where
    B: burn::tensor::backend::AutodiffBackend,
    O: burn::optim::Optimizer<LoraTextToLatentRfDiT<B>, B>,
{
    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};

    let dir = atomic_save(output_dir, step, |tmp| {
        write_adapter_files(model, lora_cfg, tmp)?;

        // Training metadata (step + seed for RNG re-derivation at resume).
        let meta = TrainingMeta {
            step,
            training_seed,
        };
        std::fs::write(
            tmp.join("training_meta.json"),
            serde_json::to_string_pretty(&meta)?,
        )?;

        // ParamId map — lets the resume path restore original ParamIds so that
        // optimizer momentum terms (keyed by ParamId) match the loaded model.
        let param_ids = collect_lora_param_ids(model);
        std::fs::write(
            tmp.join("param_ids.json"),
            serde_json::to_string_pretty(&param_ids)?,
        )?;

        // Optimizer state — use NamedMpkFileRecorder for better cross-version
        // compatibility compared to BinFileRecorder.
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        recorder
            .record(optim.to_record(), tmp.join("optimizer"))
            .map_err(|e| IrodoriError::Checkpoint(format!("save optimizer state: {e}")))?;

        Ok(())
    })?;

    tracing::info!(step, path = %dir.display(), "saved full training checkpoint");
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use tempfile::TempDir;

    type TestBackend = NdArray;

    fn make_tiny_model() -> (LoraTextToLatentRfDiT<TestBackend>, LoraConfig) {
        let cfg = crate::config::tiny_model_config();
        let lora_cfg = LoraConfig {
            r: 2,
            alpha: 4.0,
            target_modules: vec!["wq".into(), "wk".into()],
        };
        let device = Default::default();
        let model =
            LoraTextToLatentRfDiT::<TestBackend>::new(&cfg, lora_cfg.r, lora_cfg.alpha, &device);
        (model, lora_cfg)
    }

    #[test]
    fn f32_to_le_bytes_roundtrip() {
        let vals = [1.0f32, -0.5, 3.125, 0.0];
        let bytes = f32_to_le_bytes(&vals);
        assert_eq!(bytes.len(), 4 * 4);
        let recovered: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(recovered, vals);
    }

    #[test]
    fn save_creates_correct_directory_structure() {
        let dir = TempDir::new().unwrap();
        let (model, lora_cfg) = make_tiny_model();

        save_lora_adapter(&model, &lora_cfg, dir.path(), 42).unwrap();

        let step_dir = dir.path().join("step-0000042");
        assert!(step_dir.exists(), "step directory must exist");
        assert!(
            step_dir.join("adapter_model.safetensors").exists(),
            "safetensors file must exist"
        );
        assert!(
            step_dir.join("adapter_config.json").exists(),
            "adapter_config.json must exist"
        );
    }

    #[test]
    fn saved_adapter_config_has_correct_fields() {
        let dir = TempDir::new().unwrap();
        let (model, lora_cfg) = make_tiny_model();

        save_lora_adapter(&model, &lora_cfg, dir.path(), 1).unwrap();

        let json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join("step-0000001/adapter_config.json")).unwrap(),
        )
        .unwrap();

        assert_eq!(json["peft_type"], "LORA");
        assert_eq!(json["r"], 2);
        assert_eq!(json["lora_alpha"], 4.0);
        assert_eq!(json["bias"], "none");
        let modules: Vec<String> = serde_json::from_value(json["target_modules"].clone()).unwrap();
        assert_eq!(modules, vec!["wq", "wk"]);
    }

    #[test]
    fn saved_safetensors_has_peft_keys_and_shapes() {
        let dir = TempDir::new().unwrap();
        let (model, lora_cfg) = make_tiny_model();
        let r = lora_cfg.r;

        save_lora_adapter(&model, &lora_cfg, dir.path(), 0).unwrap();

        let st_path = dir.path().join("step-0000000/adapter_model.safetensors");
        let data = std::fs::read(&st_path).unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&data).unwrap();

        // The tiny model has 1 block with speaker mode:
        // Projections: wq, wk, wv, wk_text, wv_text, gate, wo + wk_speaker, wv_speaker
        // Each has lora_A and lora_B → 9 × 2 = 18 tensors
        let names: Vec<&str> = tensors.names().into_iter().collect();
        assert!(
            names.len() >= 14,
            "expected at least 14 tensors (7 projections × 2), got {}",
            names.len()
        );

        // Verify PEFT key naming convention
        let key_a = "base_model.model.blocks.0.attention.wq.lora_A.default.weight";
        let key_b = "base_model.model.blocks.0.attention.wq.lora_B.default.weight";
        assert!(names.contains(&key_a), "missing key: {key_a}");
        assert!(names.contains(&key_b), "missing key: {key_b}");

        // Verify shapes: lora_A is [r, in_features], lora_B is [out_features, r]
        let info_a = tensors.tensor(key_a).unwrap();
        assert_eq!(info_a.shape(), &[r, 32], "lora_A shape: [r, model_dim]");

        let kv_dim = 4 * 8; // num_heads(4) * head_dim(32/4=8)
        let info_b = tensors.tensor(key_b).unwrap();
        assert_eq!(info_b.shape(), &[kv_dim, r], "lora_B shape: [kv_dim, r]");

        // All tensors should be F32
        for name in &names {
            let t = tensors.tensor(name).unwrap();
            assert_eq!(t.dtype(), Dtype::F32, "{name} must be F32");
        }
    }

    #[test]
    fn save_cleans_up_stale_tmp_dir() {
        let dir = TempDir::new().unwrap();
        let (model, lora_cfg) = make_tiny_model();

        // Simulate a leftover .tmp dir from a prior crash.
        let tmp_dir = dir.path().join("step-0000010.tmp");
        std::fs::create_dir_all(&tmp_dir).unwrap();
        std::fs::write(tmp_dir.join("garbage"), b"leftover").unwrap();

        // Save should clean up the stale .tmp dir and succeed.
        save_lora_adapter(&model, &lora_cfg, dir.path(), 10).unwrap();

        assert!(!tmp_dir.exists(), ".tmp dir must be removed");
        assert!(
            dir.path().join("step-0000010").exists(),
            "final dir must exist"
        );
    }

    #[test]
    fn save_overwrites_existing_checkpoint() {
        let dir = TempDir::new().unwrap();
        let (model, lora_cfg) = make_tiny_model();

        // Save at step 5 twice — should not fail.
        save_lora_adapter(&model, &lora_cfg, dir.path(), 5).unwrap();
        save_lora_adapter(&model, &lora_cfg, dir.path(), 5).unwrap();

        let step_dir = dir.path().join("step-0000005");
        assert!(step_dir.exists());
        assert!(step_dir.join("adapter_model.safetensors").exists());
        assert!(step_dir.join("adapter_config.json").exists());
    }

    #[test]
    fn no_tmp_dir_remains_after_successful_save() {
        let dir = TempDir::new().unwrap();
        let (model, lora_cfg) = make_tiny_model();

        save_lora_adapter(&model, &lora_cfg, dir.path(), 99).unwrap();

        // After a successful save, no .tmp directory should exist.
        let tmp_dir = dir.path().join("step-0000099.tmp");
        assert!(
            !tmp_dir.exists(),
            ".tmp dir must not remain after successful save"
        );
    }

    /// `save_checkpoint` must produce a `param_ids.json` with non-empty
    /// content that round-trips through `load_lora_param_ids` and has values
    /// that match the model's actual `ParamId`s.
    #[test]
    fn save_checkpoint_produces_param_ids_json() {
        use burn::backend::Autodiff;
        use burn::optim::AdamWConfig;

        type TrainBackend = Autodiff<NdArray>;

        let dir = TempDir::new().unwrap();

        let cfg = crate::config::tiny_model_config();
        let lora_cfg = crate::train::LoraConfig {
            r: 2,
            alpha: 4.0,
            target_modules: vec!["wq".into(), "wk".into()],
        };
        let device: <TrainBackend as burn::tensor::backend::Backend>::Device = Default::default();
        let model =
            LoraTextToLatentRfDiT::<TrainBackend>::new(&cfg, lora_cfg.r, lora_cfg.alpha, &device);

        // Record the actual ParamIds before save.
        let expected_wq_a_id = model.blocks[0].attention.wq.lora_a.id.val();
        let expected_wq_b_id = model.blocks[0].attention.wq.lora_b.id.val();

        // Build a minimal (empty-record) optimizer — save_checkpoint calls
        // optim.to_record() which is valid even without a training step.
        let optim = AdamWConfig::new().init::<TrainBackend, LoraTextToLatentRfDiT<TrainBackend>>();

        save_checkpoint(&model, &optim, &lora_cfg, 3, 42, dir.path()).unwrap();

        let step_dir = dir.path().join("step-0000003");
        assert!(
            step_dir.join("param_ids.json").exists(),
            "param_ids.json must exist"
        );

        let loaded = load_lora_param_ids(&step_dir)
            .expect("load_lora_param_ids must succeed")
            .expect("param_ids.json must be present");

        assert!(!loaded.is_empty(), "param_ids map must not be empty");

        let key_a = "base_model.model.blocks.0.attention.wq.lora_A.default.weight";
        let key_b = "base_model.model.blocks.0.attention.wq.lora_B.default.weight";
        assert_eq!(
            loaded[key_a], expected_wq_a_id,
            "wq.lora_A ParamId must match model"
        );
        assert_eq!(
            loaded[key_b], expected_wq_b_id,
            "wq.lora_B ParamId must match model"
        );

        // No duplicate IDs.
        let mut vals: Vec<u64> = loaded.values().copied().collect();
        vals.sort_unstable();
        let before = vals.len();
        vals.dedup();
        assert_eq!(vals.len(), before, "param_ids must have no duplicates");
    }
}
