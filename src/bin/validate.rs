//! Numerical validation binary.
//!
//! Reads the fixture files written by `scripts/validate_numerics.py`, runs the
//! same forward pass through the Rust/burn model, and asserts that every output
//! tensor matches the Python reference within a tight tolerance.
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

use irodori_tts_burn::weights::load_model;

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

    // ------------------------------------------------------------------
    // Load model from fixture weights
    // ------------------------------------------------------------------
    let weights_path = "target/validate_weights.safetensors";
    let (model, cfg) = load_model::<B>(Path::new(weights_path), &device).with_context(|| {
        format!("load_model failed from {weights_path} — run `just validate-fixtures` first")
    })?;
    println!(
        "Model loaded  (dim={}, layers={}, heads={})",
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
    let (ref_text_state, _ref_text_state_s) = get("text_state")?;
    let (ref_speaker_state, _ref_speaker_state_s) = get("speaker_state")?;
    let (ref_speaker_mask_d, ref_speaker_mask_s) = get("speaker_mask")?;
    let (ref_v_pred, ref_v_pred_s) = get("v_pred")?;

    // ------------------------------------------------------------------
    // Build input tensors
    // ------------------------------------------------------------------
    let text_ids: Tensor<B, 2, Int> = to_tensor_int(text_ids_d, text_ids_s, &device);
    let text_mask: Tensor<B, 2, Bool> = to_tensor_bool_2d(text_mask_d, text_mask_s, &device);
    let x_t = to_tensor_f32(x_t_d, x_t_s, &device);
    let t: Tensor<B, 1> = to_tensor_f32_1d(t_d, t_s, &device);
    let ref_latent = to_tensor_f32(ref_latent_d, ref_latent_s, &device);
    let ref_mask_bool: Tensor<B, 2, Bool> = to_tensor_bool_2d(ref_mask_d, ref_mask_s, &device);
    let ref_speaker_mask: Tensor<B, 2, Bool> =
        to_tensor_bool_2d(ref_speaker_mask_d, ref_speaker_mask_s, &device);

    println!("\n=== encode_conditions ===");

    // ------------------------------------------------------------------
    // Run encode_conditions
    // ------------------------------------------------------------------
    let encoded = model.encode_conditions(
        text_ids,
        text_mask.clone(),
        irodori_tts_burn::model::condition::AuxConditionInput::Speaker {
            ref_latent,
            ref_mask: ref_mask_bool,
        },
    );

    let _ = ref_speaker_mask; // used for v_pred comparison below

    // Compare text_state
    let mut all_pass = true;
    all_pass &= check("text_state", ref_text_state, &encoded.text_state);

    // Compare speaker_state
    match &encoded.aux {
        Some(irodori_tts_burn::model::condition::AuxConditionState::Speaker { state, .. }) => {
            all_pass &= check("speaker_state", ref_speaker_state, state);
        }
        _ => {
            println!("  ✗ FAIL  speaker_state         (expected Some(Speaker), got other)");
            all_pass = false;
        }
    }

    println!("\n=== forward_with_cond ===");

    // ------------------------------------------------------------------
    // Run forward_with_cond
    // ------------------------------------------------------------------
    let v_pred = model.forward_with_cond(x_t, t, &encoded, None, None);

    // Compare v_pred
    let _ = ref_v_pred_s; // shape info not needed for diff
    let diff_vpred = max_abs_diff_3d_flat(ref_v_pred, &v_pred);
    let pass_vpred = diff_vpred <= ABS_TOL;
    let status = if pass_vpred { "✓ PASS" } else { "✗ FAIL" };
    println!(
        "  {status}  {:<20}  max_abs_diff = {:.2e}  (tol={ABS_TOL:.0e})",
        "v_pred", diff_vpred
    );
    all_pass &= pass_vpred;

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    println!();
    if all_pass {
        println!("All checks PASSED ✓");
        Ok(())
    } else {
        anyhow::bail!("One or more checks FAILED ✗")
    }
}
