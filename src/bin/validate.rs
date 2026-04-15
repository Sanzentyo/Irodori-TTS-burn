//! Numerical validation binary.
//!
//! Reads the fixture files written by `scripts/validate_numerics.py`, runs the
//! same forward pass through the Rust/burn model, and asserts that every output
//! tensor matches the Python reference within a tight tolerance.
//!
//! Checks:
//! - encode_conditions: text_state, speaker_state / caption_state
//! - per-DiT-block outputs (block_0_out, block_1_out, ...) — catches layer-level bugs
//! - final v_pred
//! - KV-cache consistency: cached forward must match uncached forward
//!
//! # Usage
//! ```sh
//! just validate    # generates fixtures first, then runs this
//! ```

use std::{collections::HashMap, path::Path};

use anyhow::{Context, Result};
use burn::{
    backend::NdArray,
    tensor::{Bool, Int, Tensor, TensorData, backend::Backend},
};
use safetensors::SafeTensors;

use irodori_tts_burn::{AuxConditionInput, AuxConditionState, load_model};

type B = NdArray;

// ---------------------------------------------------------------------------
// Tolerance
// ---------------------------------------------------------------------------

/// Maximum allowed absolute difference between Python and Rust outputs.
const ABS_TOL: f32 = 1e-3;

// ---------------------------------------------------------------------------
// Helpers: safetensors → Burn tensors
// ---------------------------------------------------------------------------

type TensorsResult = (
    Vec<u8>,
    HashMap<String, Vec<f32>>,
    HashMap<String, Vec<usize>>,
);

/// Read a safetensors file, returning `(file_bytes, data_map, shape_map)`.
///
/// The bytes must outlive the `SafeTensors` view.
fn read_tensors(path: &str) -> Result<TensorsResult> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("cannot read {path} — run `just validate-fixtures` first"))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("malformed safetensors file: {path}"))?;

    let mut data_map = HashMap::new();
    let mut shape_map = HashMap::new();

    for (name, view) in tensors.tensors() {
        let bytes_data = view.data();
        let f32_data: Vec<f32> = bytes_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        data_map.insert(name.to_string(), f32_data);
        shape_map.insert(name.to_string(), view.shape().to_vec());
    }

    Ok((bytes, data_map, shape_map))
}

fn to_tensor_f32(data: &[f32], shape: &[usize], device: &<B as Backend>::Device) -> Tensor<B, 3> {
    let td = TensorData::new(data.to_vec(), shape.to_vec());
    Tensor::from_data(td, device)
}

fn to_tensor_f32_1d(
    data: &[f32],
    shape: &[usize],
    device: &<B as Backend>::Device,
) -> Tensor<B, 1> {
    let td = TensorData::new(data.to_vec(), shape.to_vec());
    Tensor::from_data(td, device)
}

fn to_tensor_int(
    data: &[f32],
    shape: &[usize],
    device: &<B as Backend>::Device,
) -> Tensor<B, 2, Int> {
    let int_data: Vec<i32> = data.iter().map(|&x| x as i32).collect();
    let td = TensorData::new(int_data, shape.to_vec());
    Tensor::from_data(td, device)
}

fn to_tensor_bool_2d(
    data: &[f32],
    shape: &[usize],
    device: &<B as Backend>::Device,
) -> Tensor<B, 2, Bool> {
    // Python saved bool masks as float (1.0 = True, 0.0 = False)
    let bool_data: Vec<bool> = data.iter().map(|&x| x > 0.5).collect();
    let td = TensorData::new(bool_data, shape.to_vec());
    Tensor::from_data(td, device)
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

fn max_abs_diff_3d_flat(a_data: &[f32], b: &Tensor<B, 3>) -> f32 {
    let flat: Vec<f32> = b.to_data().to_vec().unwrap();
    a_data
        .iter()
        .zip(flat.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn check(name: &str, ref_data: &[f32], rust_tensor: &Tensor<B, 3>) -> bool {
    let diff = max_abs_diff_3d_flat(ref_data, rust_tensor);
    let pass = diff <= ABS_TOL;
    let status = if pass { "✓ PASS" } else { "✗ FAIL" };
    println!("  {status}  {name:<20}  max_abs_diff = {diff:.2e}  (tol={ABS_TOL:.0e})");
    pass
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> anyhow::Result<()> {
    let device = Default::default();

    let speaker_pass = validate_speaker(&device)?;
    println!();
    let caption_pass = validate_caption(&device)?;
    println!();
    let kv_pass = validate_kv_cache_consistency(&device)?;

    println!();
    if speaker_pass && caption_pass && kv_pass {
        println!("All checks PASSED ✓");
        Ok(())
    } else {
        anyhow::bail!("One or more checks FAILED ✗")
    }
}

// ---------------------------------------------------------------------------
// Speaker-conditioned validation
// ---------------------------------------------------------------------------

fn validate_speaker(
    device: &<B as burn::tensor::backend::Backend>::Device,
) -> anyhow::Result<bool> {
    // ------------------------------------------------------------------
    // Load model from fixture weights
    // ------------------------------------------------------------------
    let weights_path = "target/validate_weights.safetensors";
    let (model, cfg) = load_model::<B>(Path::new(weights_path), device).with_context(|| {
        format!("load_model failed from {weights_path} — run `just validate-fixtures` first")
    })?;
    println!(
        "[speaker] Model loaded  (dim={}, layers={}, heads={})",
        cfg.model_dim, cfg.num_layers, cfg.num_heads
    );

    // ------------------------------------------------------------------
    // Load reference tensors
    // ------------------------------------------------------------------
    let tensors_path = "target/validate_tensors.safetensors";
    let (_bytes, data, shapes) = read_tensors(tensors_path)?;

    let get = |name: &str| -> anyhow::Result<(&Vec<f32>, &Vec<usize>)> {
        let d = data
            .get(name)
            .with_context(|| format!("missing tensor '{name}' in fixture"))?;
        let s = shapes
            .get(name)
            .with_context(|| format!("missing shape for '{name}' in fixture"))?;
        Ok((d, s))
    };

    let (text_ids_d, text_ids_s) = get("text_ids")?;
    let (text_mask_d, text_mask_s) = get("text_mask")?;
    let (x_t_d, x_t_s) = get("x_t")?;
    let (t_d, t_s) = get("t")?;
    let (ref_latent_d, ref_latent_s) = get("ref_latent")?;
    let (ref_mask_d, ref_mask_s) = get("ref_mask")?;
    let (ref_text_state, _) = get("text_state")?;
    let (ref_speaker_state, _) = get("speaker_state")?;
    let (ref_v_pred, _) = get("v_pred")?;

    let text_ids: Tensor<B, 2, Int> = to_tensor_int(text_ids_d, text_ids_s, device);
    let text_mask: Tensor<B, 2, Bool> = to_tensor_bool_2d(text_mask_d, text_mask_s, device);
    let x_t = to_tensor_f32(x_t_d, x_t_s, device);
    let t: Tensor<B, 1> = to_tensor_f32_1d(t_d, t_s, device);
    let ref_latent = to_tensor_f32(ref_latent_d, ref_latent_s, device);
    let ref_mask_bool: Tensor<B, 2, Bool> = to_tensor_bool_2d(ref_mask_d, ref_mask_s, device);

    println!("\n=== [speaker] encode_conditions ===");

    let encoded = model.encode_conditions(
        text_ids,
        text_mask.clone(),
        AuxConditionInput::Speaker {
            ref_latent,
            ref_mask: ref_mask_bool,
        },
    )?;

    let mut all_pass = true;
    all_pass &= check("text_state", ref_text_state, &encoded.text_state);

    match &encoded.aux {
        Some(AuxConditionState::Speaker { state, .. }) => {
            all_pass &= check("speaker_state", ref_speaker_state, state);
        }
        _ => {
            println!("  ✗ FAIL  speaker_state         (expected Some(Speaker), got other)");
            all_pass = false;
        }
    }

    println!("\n=== [speaker] per-block outputs + v_pred ===");

    let [_, seq_lat, _] = x_t.dims();
    let lat_rope = model.precompute_latent_rope(seq_lat, device);
    let (v_pred, debug) = model.forward_with_cond_debug(x_t, t, &encoded, &lat_rope);

    // Check per-block outputs against Python fixtures.
    // Fixture keys: "block_0_out", "block_1_out", ...
    for (i, block_out) in debug.block_outputs.iter().enumerate() {
        let key = format!("block_{i}_out");
        match get(&key) {
            Ok((ref_data, _)) => {
                all_pass &= check(&key, ref_data, block_out);
            }
            Err(_) => {
                // Older fixtures may not have per-block dumps; skip gracefully.
                println!("  (skip)  {key:<20}  not in fixture (run `just validate-fixtures`)");
            }
        }
    }

    // Also check after_in_proj if present
    match get("after_in_proj") {
        Ok((ref_data, _)) => {
            all_pass &= check("after_in_proj", ref_data, &debug.after_in_proj);
        }
        Err(_) => {
            println!(
                "  (skip)  after_in_proj         not in fixture (run `just validate-fixtures`)"
            );
        }
    }

    let diff_vpred = max_abs_diff_3d_flat(ref_v_pred, &v_pred);
    let pass_vpred = diff_vpred <= ABS_TOL;
    let status = if pass_vpred { "✓ PASS" } else { "✗ FAIL" };
    println!(
        "  {status}  {:<20}  max_abs_diff = {:.2e}  (tol={ABS_TOL:.0e})",
        "v_pred", diff_vpred
    );
    all_pass &= pass_vpred;

    Ok(all_pass)
}

// ---------------------------------------------------------------------------
// Caption-conditioned validation (voice-design path)
// ---------------------------------------------------------------------------

fn validate_caption(
    device: &<B as burn::tensor::backend::Backend>::Device,
) -> anyhow::Result<bool> {
    // ------------------------------------------------------------------
    // Load model from caption fixture weights
    // ------------------------------------------------------------------
    let weights_path = "target/validate_caption_weights.safetensors";
    let (model, cfg) = load_model::<B>(Path::new(weights_path), device).with_context(|| {
        format!("load_model failed from {weights_path} — run `just validate-fixtures` first")
    })?;
    println!(
        "[caption] Model loaded  (dim={}, layers={}, heads={}, use_caption={})",
        cfg.model_dim, cfg.num_layers, cfg.num_heads, cfg.use_caption_condition
    );
    if !cfg.use_caption_condition {
        anyhow::bail!("caption fixture config has use_caption_condition=false — fixture mismatch");
    }

    // ------------------------------------------------------------------
    // Load reference tensors
    // ------------------------------------------------------------------
    let tensors_path = "target/validate_caption_tensors.safetensors";
    let (_bytes, data, shapes) = read_tensors(tensors_path)?;

    let get = |name: &str| -> anyhow::Result<(&Vec<f32>, &Vec<usize>)> {
        let d = data
            .get(name)
            .with_context(|| format!("missing tensor '{name}' in caption fixture"))?;
        let s = shapes
            .get(name)
            .with_context(|| format!("missing shape for '{name}' in caption fixture"))?;
        Ok((d, s))
    };

    let (text_ids_d, text_ids_s) = get("text_ids")?;
    let (text_mask_d, text_mask_s) = get("text_mask")?;
    let (caption_ids_d, caption_ids_s) = get("caption_ids")?;
    let (caption_mask_d, caption_mask_s) = get("caption_mask")?;
    let (x_t_d, x_t_s) = get("x_t")?;
    let (t_d, t_s) = get("t")?;
    let (ref_text_state, _) = get("text_state")?;
    let (ref_caption_state, _) = get("caption_state")?;
    let (ref_v_pred, _) = get("v_pred")?;

    let text_ids: Tensor<B, 2, Int> = to_tensor_int(text_ids_d, text_ids_s, device);
    let text_mask: Tensor<B, 2, Bool> = to_tensor_bool_2d(text_mask_d, text_mask_s, device);
    let caption_ids: Tensor<B, 2, Int> = to_tensor_int(caption_ids_d, caption_ids_s, device);
    let caption_mask: Tensor<B, 2, Bool> =
        to_tensor_bool_2d(caption_mask_d, caption_mask_s, device);
    let x_t = to_tensor_f32(x_t_d, x_t_s, device);
    let t: Tensor<B, 1> = to_tensor_f32_1d(t_d, t_s, device);

    println!("\n=== [caption] encode_conditions ===");

    let encoded = model.encode_conditions(
        text_ids,
        text_mask.clone(),
        AuxConditionInput::Caption {
            ids: caption_ids,
            mask: caption_mask,
        },
    )?;

    let mut all_pass = true;
    all_pass &= check("text_state", ref_text_state, &encoded.text_state);

    match &encoded.aux {
        Some(AuxConditionState::Caption { state, .. }) => {
            all_pass &= check("caption_state", ref_caption_state, state);
        }
        _ => {
            println!("  ✗ FAIL  caption_state        (expected Some(Caption), got other)");
            all_pass = false;
        }
    }

    println!("\n=== [caption] forward_with_cond ===");

    let v_pred = model.forward_with_cond(x_t, t, &encoded, None, None);
    let diff_vpred = max_abs_diff_3d_flat(ref_v_pred, &v_pred);
    let pass_vpred = diff_vpred <= ABS_TOL;
    let status = if pass_vpred { "✓ PASS" } else { "✗ FAIL" };
    println!(
        "  {status}  {:<20}  max_abs_diff = {:.2e}  (tol={ABS_TOL:.0e})",
        "v_pred", diff_vpred
    );
    all_pass &= pass_vpred;

    Ok(all_pass)
}

// ---------------------------------------------------------------------------
// KV-cache consistency: cached forward must match uncached forward
// ---------------------------------------------------------------------------

/// Verify that using a pre-built KV cache produces the same v_pred as not using one.
///
/// This is a Rust-internal invariant test (no Python fixture needed).  A failure
/// here indicates a bug in the KV cache construction or application logic.
fn validate_kv_cache_consistency(
    device: &<B as burn::tensor::backend::Backend>::Device,
) -> anyhow::Result<bool> {
    println!("=== [kv-cache] cached vs uncached consistency ===");

    let weights_path = "target/validate_weights.safetensors";
    let (model, _cfg) = load_model::<B>(Path::new(weights_path), device)
        .with_context(|| format!("load_model failed from {weights_path}"))?;

    let tensors_path = "target/validate_tensors.safetensors";
    let (_bytes, data, shapes) = read_tensors(tensors_path)?;

    let get = |name: &str| -> anyhow::Result<(&Vec<f32>, &Vec<usize>)> {
        let d = data
            .get(name)
            .with_context(|| format!("missing tensor '{name}' in fixture"))?;
        let s = shapes
            .get(name)
            .with_context(|| format!("missing shape for '{name}'"))?;
        Ok((d, s))
    };

    let (text_ids_d, text_ids_s) = get("text_ids")?;
    let (text_mask_d, text_mask_s) = get("text_mask")?;
    let (x_t_d, x_t_s) = get("x_t")?;
    let (t_d, t_s) = get("t")?;
    let (ref_latent_d, ref_latent_s) = get("ref_latent")?;
    let (ref_mask_d, ref_mask_s) = get("ref_mask")?;

    let text_ids: Tensor<B, 2, Int> = to_tensor_int(text_ids_d, text_ids_s, device);
    let text_mask: Tensor<B, 2, Bool> = to_tensor_bool_2d(text_mask_d, text_mask_s, device);
    let x_t = to_tensor_f32(x_t_d, x_t_s, device);
    let t: Tensor<B, 1> = to_tensor_f32_1d(t_d, t_s, device);
    let ref_latent = to_tensor_f32(ref_latent_d, ref_latent_s, device);
    let ref_mask_bool: Tensor<B, 2, Bool> = to_tensor_bool_2d(ref_mask_d, ref_mask_s, device);

    let encoded = model.encode_conditions(
        text_ids,
        text_mask.clone(),
        AuxConditionInput::Speaker {
            ref_latent,
            ref_mask: ref_mask_bool,
        },
    )?;

    // Uncached forward
    let v_uncached = model.forward_with_cond(x_t.clone(), t.clone(), &encoded, None, None);

    // Cached forward — build KV cache then run
    let [_, seq_lat, _] = x_t.dims();
    let kv_caches = model.build_kv_caches(&encoded, Some(seq_lat));
    let lat_rope = model.precompute_latent_rope(seq_lat, device);
    let v_cached =
        model.forward_with_cond_cached(x_t, t, &encoded, None, Some(&kv_caches), &lat_rope);

    let uncached_data: Vec<f32> = v_uncached.to_data().to_vec().unwrap();
    let diff = max_abs_diff_3d_flat(&uncached_data, &v_cached);
    // KV cache should give EXACTLY the same result (same arithmetic, just reordered)
    let tol = 1e-5_f32;
    let pass = diff <= tol;
    let status = if pass { "✓ PASS" } else { "✗ FAIL" };
    println!(
        "  {status}  {:<20}  max_abs_diff = {:.2e}  (tol={tol:.0e})",
        "cached_vs_uncached", diff
    );

    Ok(pass)
}
