//! Full-model E2E comparison binary.
//!
//! Loads the full ~500M TextToLatentRfDiT from `target/model_converted.safetensors`,
//! reads fixture inputs written by `scripts/full_model_e2e.py`, runs
//! `sample_euler_rf_cfg` with the same parameters and initial noise, then
//! compares the Rust output against the Python reference.
//!
//! **Scope**: validates Rust sampler parity with the explicitly-unrolled 3-pass
//! Python Independent-CFG loop. Both use f32, the same weights, and identical
//! initial noise — so the expected discrepancy is at float rounding level.
//!
//! # Usage
//! ```sh
//! just full-e2e              # NdArray (CPU, default)
//! just full-e2e-tch          # --backend libtorch
//! just full-e2e-tch-bf16     # --backend libtorch-bf16
//! ```

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use burn::tensor::{Bool, Int, Tensor, TensorData, backend::Backend};
use clap::Parser;
use safetensors::SafeTensors;

use irodori_tts_burn::{
    CfgGuidanceMode, GuidanceConfig, InferenceBackendKind, InferenceOptimizedModel, SamplerParams,
    SamplingRequest, backend_config::BackendConfig, dispatch_inference, load_model,
    sample_euler_rf_cfg,
};

#[derive(Parser)]
#[command(about = "Full-model E2E sampler comparison against Python fixtures")]
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
        .with_context(|| format!("cannot read '{path}' — run `just full-e2e-fixtures` first"))?;
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

fn diff_stats<B: Backend>(ref_data: &[f32], rust_tensor: &Tensor<B, 3>) -> (f32, f32) {
    let flat: Vec<f32> = rust_tensor.to_data().convert::<f32>().to_vec().unwrap();
    let max_abs = ref_data
        .iter()
        .zip(&flat)
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let mean_abs = ref_data
        .iter()
        .zip(&flat)
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f32>()
        / ref_data.len() as f32;
    (max_abs, mean_abs)
}

fn report(label: &str, max_abs: f32, mean_abs: f32, tol: f32) -> bool {
    let pass = max_abs <= tol;
    let sym = if pass { "✓ PASS" } else { "✗ FAIL" };
    println!(
        "  {sym}  {label:<25}  max_abs={max_abs:.2e}  mean_abs={mean_abs:.2e}  (tol={tol:.0e})"
    );
    pass
}

// ---------------------------------------------------------------------------
// Generic run function
// ---------------------------------------------------------------------------

fn run<B: BackendConfig>(backend: InferenceBackendKind, device: B::Device) -> Result<()> {
    // bf16/f16 vs f32 reference: expect ~0.1-0.3 max diff over 10 steps due to dtype.
    // Python bf16 is broken (cuBLAS CUDA_R_16BF limitation), so we compare Rust
    // reduced-precision against Python f32 with wider tolerance.
    let abs_tol = if backend.is_reduced_precision() {
        3e-1
    } else {
        1e-2
    };
    println!("Backend: {}  tolerance: {abs_tol:.0e}", backend.label());

    // Load full model
    let checkpoint = "target/model_converted.safetensors";
    let (model, cfg) = load_model::<B>(std::path::Path::new(checkpoint), &device)
        .with_context(|| format!("load_model failed — check {checkpoint}"))?;
    let model = InferenceOptimizedModel::new(model);
    println!(
        "Model loaded  dim={} layers={} heads={} latent_dim={} params≈{}M",
        cfg.model_dim,
        cfg.num_layers,
        cfg.num_heads,
        cfg.latent_dim,
        (cfg.model_dim * cfg.model_dim * 4 * cfg.num_layers + cfg.model_dim * 1000) / 1_000_000,
    );

    // Load Python fixture inputs
    let inputs = load_safetensors_as_f32("target/full_e2e_inputs.safetensors")?;
    let outputs_ref = load_safetensors_as_f32("target/full_e2e_output.safetensors")?;

    let (x_t_init_d, x_t_init_s) = get(&inputs, "x_t_init")?;
    let (text_ids_d, text_ids_s) = get(&inputs, "text_ids")?;
    let (text_mask_d, text_mask_s) = get(&inputs, "text_mask")?;
    let (ref_latent_d, ref_latent_s) = get(&inputs, "ref_latent")?;
    let (ref_mask_d, ref_mask_s) = get(&inputs, "ref_mask")?;

    let sequence_length = x_t_init_s[1];

    let x_t_init = as_tensor_3d::<B>(x_t_init_d, x_t_init_s, &device);
    let text_ids = as_tensor_int::<B>(text_ids_d, text_ids_s, &device);
    let text_mask = as_tensor_bool_2d::<B>(text_mask_d, text_mask_s, &device);
    let ref_latent = as_tensor_3d::<B>(ref_latent_d, ref_latent_s, &device);
    let ref_mask = as_tensor_bool_2d::<B>(ref_mask_d, ref_mask_s, &device);

    println!(
        "Fixtures loaded: x_t_init={x_t_init_s:?}, text_ids={text_ids_s:?}, \
        ref_latent={ref_latent_s:?}, seq_len={sequence_length}"
    );

    // Build sampler params — must match scripts/full_model_e2e.py exactly
    let params = SamplerParams {
        num_steps: 10,
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

    // Run the Rust sampler
    println!("\n=== sample_euler_rf_cfg (10 steps, Independent CFG) ===");
    let rust_output =
        sample_euler_rf_cfg(&model, request, &params, &device).context("sampler failed")?;

    let rust_flat: Vec<f32> = rust_output.to_data().convert::<f32>().to_vec().unwrap();
    println!(
        "Rust output: min={:.4}  max={:.4}  mean={:.6}",
        rust_flat.iter().cloned().fold(f32::INFINITY, f32::min),
        rust_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        rust_flat.iter().sum::<f32>() / rust_flat.len() as f32,
    );

    // Compare against Python reference
    println!("\n=== Comparison (Python unrolled loop vs Rust) ===");

    let (ref_output_d, _) = get(&outputs_ref, "output")?;
    let (max_abs, mean_abs) = diff_stats::<B>(ref_output_d, &rust_output);
    let all_pass = report("final output (10 steps)", max_abs, mean_abs, abs_tol);

    // Also report step-0 velocity for quick debugging if the final fails.
    if let (Ok((v0_d, v0_s)), _) = (get(&outputs_ref, "v_cond_0"), ()) {
        let v0_rust_proxy = as_tensor_3d::<B>(v0_d, v0_s, &device);
        let v0_flat: Vec<f32> = v0_rust_proxy.to_data().convert::<f32>().to_vec().unwrap();
        println!(
            "  (info) Python v_cond_0: min={:.4}  max={:.4}  mean={:.6}",
            v0_flat.iter().cloned().fold(f32::INFINITY, f32::min),
            v0_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            v0_flat.iter().sum::<f32>() / v0_flat.len() as f32,
        );
    }

    println!();
    if all_pass {
        println!("Full-model E2E check PASSED ✓  (max_abs={max_abs:.2e}, mean_abs={mean_abs:.2e})");
        Ok(())
    } else {
        bail!(
            "Full-model E2E check FAILED ✗  (max_abs={max_abs:.2e}, tol={abs_tol:.0e})\n\
             Per-step fixtures (v_cond_i, v_text_unc_i, v_spk_unc_i, x_t_i) are saved in\n\
             target/full_e2e_output.safetensors for manual debugging."
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
