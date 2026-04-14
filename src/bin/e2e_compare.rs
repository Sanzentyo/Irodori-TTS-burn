//! E2E comparison binary.
//!
//! Loads the tiny validation model, reads the fixture inputs written by
//! `scripts/e2e_compare.py`, runs `sample_euler_rf_cfg` with a fixed initial
//! noise, then compares the Rust output against the Python reference.
//!
//! # Usage
//! ```sh
//! just e2e                # NdArray (CPU)
//! just e2e-tch            # LibTorch CUDA (correctness on GPU path)
//! ```

// ── Backend selection ─────────────────────────────────────────────────────
irodori_tts_burn::select_inference_backend!();

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};
use safetensors::SafeTensors;

use irodori_tts_burn::{
    CfgGuidanceMode, GuidanceConfig, SamplerParams, SamplingRequest, backend_config::BackendConfig,
    sample_euler_rf_cfg, weights::load_model,
};

// bf16 has ~2 significant decimal digits (~0.004 relative error), so tolerance must be wider.
// This applies to all bf16 backends (LibTorch, Cuda, Wgpu).
#[cfg(any(
    feature = "backend_tch_bf16",
    feature = "backend_cuda_bf16",
    feature = "backend_wgpu_bf16",
))]
const ABS_TOL: f32 = 5e-2;

// f16 also has reduced precision (Wgpu f16).
#[cfg(feature = "backend_wgpu_f16")]
const ABS_TOL: f32 = 5e-2;

#[cfg(not(any(
    feature = "backend_tch_bf16",
    feature = "backend_cuda_bf16",
    feature = "backend_wgpu_bf16",
    feature = "backend_wgpu_f16",
)))]
const ABS_TOL: f32 = 1e-3;

// ---------------------------------------------------------------------------
// Fixture loading helpers
// ---------------------------------------------------------------------------

type FixtureMap = HashMap<String, (Vec<f32>, Vec<usize>)>;

fn load_safetensors_as_f32(path: &str) -> Result<FixtureMap> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("cannot read '{path}' — run `just e2e-fixtures` first"))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("malformed safetensors: {path}"))?;

    let mut map = FixtureMap::new();
    for (name, view) in tensors.tensors() {
        let f32s: Vec<f32> = view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        map.insert(name.to_string(), (f32s, view.shape().to_vec()));
    }
    Ok(map)
}

fn get<'a>(map: &'a FixtureMap, key: &str) -> Result<(&'a Vec<f32>, &'a Vec<usize>)> {
    let (d, s) = map
        .get(key)
        .with_context(|| format!("missing tensor '{key}' in fixture"))?;
    Ok((d, s))
}

fn as_tensor_3d(data: &[f32], shape: &[usize], device: &<B as Backend>::Device) -> Tensor<B, 3> {
    let td = TensorData::new(data.to_vec(), shape.to_vec())
        .convert::<<B as burn::tensor::backend::Backend>::FloatElem>();
    Tensor::from_data(td, device)
}

fn as_tensor_int(
    data: &[f32],
    shape: &[usize],
    device: &<B as Backend>::Device,
) -> Tensor<B, 2, Int> {
    let ints: Vec<i32> = data.iter().map(|&x| x as i32).collect();
    Tensor::from_data(TensorData::new(ints, shape.to_vec()), device)
}

fn as_tensor_bool_2d(
    data: &[f32],
    shape: &[usize],
    device: &<B as Backend>::Device,
) -> Tensor<B, 2, Bool> {
    let bools: Vec<bool> = data.iter().map(|&x| x > 0.5).collect();
    Tensor::from_data(TensorData::new(bools, shape.to_vec()), device)
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

fn max_abs_diff(ref_data: &[f32], rust_tensor: &Tensor<B, 3>) -> f32 {
    // `.convert::<f32>()` handles any backend float element type (f32 or bf16).
    let flat: Vec<f32> = rust_tensor.to_data().convert::<f32>().to_vec().unwrap();
    ref_data
        .iter()
        .zip(&flat)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

fn report(label: &str, diff: f32, tol: f32) -> bool {
    let pass = diff <= tol;
    let sym = if pass { "✓ PASS" } else { "✗ FAIL" };
    println!("  {sym}  {label:<20}  max_abs_diff = {diff:.2e}  (tol={tol:.0e})",);
    pass
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let device = B::device_from_id(0);

    // ------------------------------------------------------------------
    // Load model
    // ------------------------------------------------------------------
    let weights_path = "target/validate_weights.safetensors";
    let (model, cfg) = load_model::<B>(std::path::Path::new(weights_path), &device)
        .with_context(|| "load_model failed — run `just validate-fixtures` first".to_string())?;
    println!(
        "Model loaded  (dim={}, layers={}, heads={})",
        cfg.model_dim, cfg.num_layers, cfg.num_heads
    );

    // ------------------------------------------------------------------
    // Load Python fixture inputs
    // ------------------------------------------------------------------
    let inputs = load_safetensors_as_f32("target/e2e_inputs.safetensors")?;
    let outputs_ref = load_safetensors_as_f32("target/e2e_output.safetensors")?;

    let (x_t_init_d, x_t_init_s) = get(&inputs, "x_t_init")?;
    let (text_ids_d, text_ids_s) = get(&inputs, "text_ids")?;
    let (text_mask_d, text_mask_s) = get(&inputs, "text_mask")?;
    let (ref_latent_d, ref_latent_s) = get(&inputs, "ref_latent")?;
    let (ref_mask_d, ref_mask_s) = get(&inputs, "ref_mask")?;

    // Derive sequence_length from the initial noise shape — [B, S, D].
    let sequence_length = x_t_init_s[1];

    let x_t_init = as_tensor_3d(x_t_init_d, x_t_init_s, &device);
    let text_ids = as_tensor_int(text_ids_d, text_ids_s, &device);
    let text_mask = as_tensor_bool_2d(text_mask_d, text_mask_s, &device);
    let ref_latent = as_tensor_3d(ref_latent_d, ref_latent_s, &device);
    let ref_mask = as_tensor_bool_2d(ref_mask_d, ref_mask_s, &device);

    println!("Loaded fixtures: x_t_init shape={x_t_init_s:?}, seq_len={sequence_length}");

    // ------------------------------------------------------------------
    // Build sampler params matching Python script exactly
    // ------------------------------------------------------------------
    let params = SamplerParams {
        num_steps: 4,
        guidance: GuidanceConfig {
            mode: CfgGuidanceMode::Independent,
            scale_text: 3.0,
            scale_speaker: 5.0,
            scale_caption: 0.0,
            min_t: 0.0,
            max_t: 1.0,
        },
        truncation_factor: None,
        temporal_rescale: None,
        speaker_kv: None,
        use_context_kv_cache: true,
    };

    let request = SamplingRequest {
        text_ids,
        text_mask,
        ref_latent: Some(ref_latent),
        ref_mask: Some(ref_mask),
        sequence_length,
        caption_ids: None,
        caption_mask: None,
        initial_noise: Some(x_t_init),
    };

    // ------------------------------------------------------------------
    // Run the Rust sampler
    // ------------------------------------------------------------------
    println!("\n=== sample_euler_rf_cfg (4 steps, Independent CFG) ===");
    let rust_output =
        sample_euler_rf_cfg(&model, request, &params, &device).context("sampler failed")?;

    // ------------------------------------------------------------------
    // Compare against Python reference
    // ------------------------------------------------------------------
    println!("\n=== Comparison ===");
    let (ref_output_d, _) = get(&outputs_ref, "output")?;
    let diff = max_abs_diff(ref_output_d, &rust_output);
    let all_pass = report("final output", diff, ABS_TOL);

    // Note: Python also saves per-step x_t_{i} and v_cond_{i} tensors in the
    // fixture; these are useful for manual debugging if the final diff fails but
    // cannot be compared here without exposing intermediate state from the sampler.

    println!();
    if all_pass {
        println!("E2E check PASSED ✓  (max_abs_diff = {diff:.2e})");
        Ok(())
    } else {
        bail!(
            "E2E check FAILED ✗  (max_abs_diff = {diff:.2e}, tol = {ABS_TOL:.0e})\n\
             Re-run `just e2e` with RUST_LOG=debug for more detail."
        )
    }
}
