//! E2E comparison binary.
//!
//! Loads the tiny validation model, reads the fixture inputs written by
//! `scripts/e2e_compare.py`, runs `sample_euler_rf_cfg` with a fixed initial
//! noise, then compares the Rust output against the Python reference.
//!
//! # Usage
//! ```sh
//! just e2e                # NdArray (CPU, default)
//! just e2e-tch            # --backend libtorch
//! just e2e-tch-bf16       # --backend libtorch-bf16
//! ```

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};
use clap::Parser;
use safetensors::SafeTensors;

use irodori_tts_burn::{
    CfgGuidanceMode, GuidanceConfig, InferenceBackendKind, InferenceOptimizedModel, SamplerMethod,
    SamplerParams, SamplingRequest, backend_config::BackendConfig, dispatch_inference, load_model,
    sample_euler_rf_cfg,
};

#[derive(Parser)]
#[command(about = "E2E sampler comparison against Python fixtures")]
struct Args {
    /// Backend to use for inference.
    #[arg(long, default_value = "ndarray")]
    backend: InferenceBackendKind,
    /// GPU device index (ignored for NdArray).
    #[arg(long, default_value_t = 0)]
    gpu_id: u32,
}

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

fn as_tensor_3d<B: Backend>(data: &[f32], shape: &[usize], device: &B::Device) -> Tensor<B, 3> {
    let td = TensorData::new(data.to_vec(), shape.to_vec()).convert::<B::FloatElem>();
    Tensor::from_data(td, device)
}

fn as_tensor_int<B: Backend>(
    data: &[f32],
    shape: &[usize],
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let ints: Vec<i32> = data.iter().map(|&x| x as i32).collect();
    Tensor::from_data(TensorData::new(ints, shape.to_vec()), device)
}

fn as_tensor_bool_2d<B: Backend>(
    data: &[f32],
    shape: &[usize],
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    let bools: Vec<bool> = data.iter().map(|&x| x > 0.5).collect();
    Tensor::from_data(TensorData::new(bools, shape.to_vec()), device)
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

fn max_abs_diff<B: Backend>(ref_data: &[f32], rust_tensor: &Tensor<B, 3>) -> f32 {
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
    println!("  {sym}  {label:<20}  max_abs_diff = {diff:.2e}  (tol={tol:.0e})");
    pass
}

// ---------------------------------------------------------------------------
// Generic run function
// ---------------------------------------------------------------------------

fn run<B: BackendConfig>(backend: InferenceBackendKind, device: B::Device) -> Result<()> {
    // bf16/f16 have ~2 significant decimal digits; use wider tolerance.
    let abs_tol = if backend.is_reduced_precision() {
        5e-2
    } else {
        1e-3
    };
    println!("Backend: {}  tolerance: {abs_tol:.0e}", backend.label());

    // Load model
    let weights_path = "target/validate_weights.safetensors";
    let (model, cfg) = load_model::<B>(std::path::Path::new(weights_path), &device)
        .with_context(|| "load_model failed — run `just validate-fixtures` first".to_string())?;
    let model = InferenceOptimizedModel::new(model);
    println!(
        "Model loaded  (dim={}, layers={}, heads={})",
        cfg.model_dim, cfg.num_layers, cfg.num_heads
    );

    // Load Python fixture inputs
    let inputs = load_safetensors_as_f32("target/e2e_inputs.safetensors")?;
    let outputs_euler = load_safetensors_as_f32("target/e2e_output.safetensors")?;
    let outputs_heun = load_safetensors_as_f32("target/e2e_heun_output.safetensors")?;

    let (x_t_init_d, x_t_init_s) = get(&inputs, "x_t_init")?;
    let (text_ids_d, text_ids_s) = get(&inputs, "text_ids")?;
    let (text_mask_d, text_mask_s) = get(&inputs, "text_mask")?;
    let (ref_latent_d, ref_latent_s) = get(&inputs, "ref_latent")?;
    let (ref_mask_d, ref_mask_s) = get(&inputs, "ref_mask")?;

    let sequence_length = x_t_init_s[1];
    println!("Loaded fixtures: x_t_init shape={x_t_init_s:?}, seq_len={sequence_length}");

    // Shared CFG guidance config (same for Euler and Heun, matching Python exactly).
    let guidance = GuidanceConfig {
        mode: CfgGuidanceMode::Independent,
        scale_text: 3.0,
        scale_speaker: 5.0,
        scale_caption: 0.0,
        min_t: 0.0,
        max_t: 1.0,
    };
    let base_params = SamplerParams {
        guidance,
        truncation_factor: None,
        temporal_rescale: None,
        speaker_kv: None,
        use_context_kv_cache: true,
        ..SamplerParams::default()
    };

    // -----------------------------------------------------------------------
    // Euler 4-step
    // -----------------------------------------------------------------------
    let params_euler = SamplerParams {
        num_steps: 4,
        method: SamplerMethod::Euler,
        ..base_params.clone()
    };
    let request_euler = SamplingRequest {
        text_ids: as_tensor_int::<B>(text_ids_d, text_ids_s, &device),
        text_mask: as_tensor_bool_2d::<B>(text_mask_d, text_mask_s, &device),
        ref_latent: Some(as_tensor_3d::<B>(ref_latent_d, ref_latent_s, &device)),
        ref_mask: Some(as_tensor_bool_2d::<B>(ref_mask_d, ref_mask_s, &device)),
        sequence_length,
        caption_ids: None,
        caption_mask: None,
        initial_noise: Some(as_tensor_3d::<B>(x_t_init_d, x_t_init_s, &device)),
    };

    println!("\n=== sample_euler_rf_cfg (4 steps Euler, NFE=4) ===");
    let rust_euler = sample_euler_rf_cfg(&model, request_euler, &params_euler, &device)
        .context("Euler sampler failed")?;

    // -----------------------------------------------------------------------
    // Heun 2-step (NFE=4 — same total forward passes as Euler 4-step)
    // -----------------------------------------------------------------------
    let params_heun = SamplerParams {
        num_steps: 2,
        method: SamplerMethod::Heun,
        ..base_params
    };
    let request_heun = SamplingRequest {
        text_ids: as_tensor_int::<B>(text_ids_d, text_ids_s, &device),
        text_mask: as_tensor_bool_2d::<B>(text_mask_d, text_mask_s, &device),
        ref_latent: Some(as_tensor_3d::<B>(ref_latent_d, ref_latent_s, &device)),
        ref_mask: Some(as_tensor_bool_2d::<B>(ref_mask_d, ref_mask_s, &device)),
        sequence_length,
        caption_ids: None,
        caption_mask: None,
        initial_noise: Some(as_tensor_3d::<B>(x_t_init_d, x_t_init_s, &device)),
    };

    println!("\n=== sample_euler_rf_cfg (2 steps Heun, NFE=4) ===");
    let rust_heun = sample_euler_rf_cfg(&model, request_heun, &params_heun, &device)
        .context("Heun sampler failed")?;

    // -----------------------------------------------------------------------
    // Compare both against Python references
    // -----------------------------------------------------------------------
    println!("\n=== Comparison ===");

    let (ref_euler_d, _) = get(&outputs_euler, "output")?;
    let diff_euler = max_abs_diff::<B>(ref_euler_d, &rust_euler);
    let pass_euler = report("Euler final output", diff_euler, abs_tol);

    let (ref_heun_d, _) = get(&outputs_heun, "output")?;
    let diff_heun = max_abs_diff::<B>(ref_heun_d, &rust_heun);
    let pass_heun = report("Heun final output ", diff_heun, abs_tol);

    // Per-step comparison for Heun to aid localisation on failure.
    let mut pass_heun_steps = true;
    for i in 0..2 {
        if let Ok((ref_xt_d, _)) = get(&outputs_heun, &format!("x_t_{i}")) {
            // Load the Rust per-step x_t by re-running? No — we don't have
            // per-step tensors from Rust yet.  Just check final for now.
            let _ = (ref_xt_d, &mut pass_heun_steps);
        }
    }

    println!();
    let all_pass = pass_euler && pass_heun;
    if all_pass {
        println!(
            "E2E check PASSED ✓  (euler max_abs_diff={diff_euler:.2e}, heun max_abs_diff={diff_heun:.2e})"
        );
        Ok(())
    } else {
        bail!(
            "E2E check FAILED ✗  (euler={diff_euler:.2e}, heun={diff_heun:.2e}, tol={abs_tol:.0e})\n\
             Re-run with RUST_LOG=debug for more detail."
        )
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();
    let backend = args.backend;
    let gpu_id = args.gpu_id;
    dispatch_inference!(backend, gpu_id, |B, device| run::<B>(backend, device))
}
