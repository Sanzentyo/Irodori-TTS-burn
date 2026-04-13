//! LoRA fine-tuning training loop.
//!
//! The training loop:
//! 1. Load base checkpoint → freeze base weights → optionally restore LoRA params from
//!    a resume checkpoint (warm restart)
//! 2. Build AdamW over **trainable** (LoRA) params only via [`GradientsParams`]
//! 3. Iterate over the dataset in shuffled epoch order
//! 4. Gradient accumulation: accumulate over `cfg.grad_accum_steps` micro-batches,
//!    flush with one optimiser step per accumulation window
//! 5. Sample `t ~ logit-normal`, interpolate noised input, predict velocity
//! 6. Compute echo-style masked MSE loss → backward → accumulate via
//!    [`GradientsAccumulator`]
//! 7. Validation: every `cfg.val_every` steps, run `cfg.val_batches` batches of the
//!    validation set on the inner (non-AD) backend via `model.valid()`
//! 8. Save adapter checkpoint every `cfg.save_every` steps

use burn::{
    optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer},
    prelude::ElementConversion,
    tensor::{Tensor, backend::AutodiffBackend},
};

use crate::{
    model::condition::AuxConditionInput,
    train::{
        LoraTextToLatentRfDiT, LoraTrainConfig,
        checkpoint::save_lora_adapter,
        dataset::{BatchIterator, ManifestDataset},
        loss::{echo_style_masked_mse, rf_interpolate, rf_velocity_target, sample_logit_normal_t},
        lr_schedule::WarmupCosineSchedule,
    },
    weights::load_lora_model,
};

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Run LoRA fine-tuning.
pub fn train_lora<B: AutodiffBackend>(
    cfg: &LoraTrainConfig,
    device: &B::Device,
) -> anyhow::Result<()> {
    // -----------------------------------------------------------------------
    // 1. Load base model with frozen weights + fresh LoRA params
    // -----------------------------------------------------------------------
    let lora_cfg = &cfg.lora;
    let (model, model_cfg) =
        load_lora_model::<B>(&cfg.base_model_path, lora_cfg.r, lora_cfg.alpha, device)?;
    tracing::info!(
        num_blocks = model_cfg.num_layers,
        r = lora_cfg.r,
        alpha = lora_cfg.alpha,
        "LoRA model loaded"
    );

    // -----------------------------------------------------------------------
    // 1b. Optional: warm restart from checkpoint
    // -----------------------------------------------------------------------
    let (mut model, mut step) = if let Some(ref resume_path) = cfg.resume_from {
        let step = parse_step_from_checkpoint_dir(resume_path)?;
        tracing::info!(
            step,
            path = %resume_path.display(),
            "resuming from checkpoint (warm restart — optimizer state resets)"
        );
        let model = restore_lora_weights(model, resume_path, device)?;
        (model, step)
    } else {
        (model, 0usize)
    };

    // -----------------------------------------------------------------------
    // 2. Training dataset + batch iterator
    // -----------------------------------------------------------------------
    let dataset = ManifestDataset::load(&cfg.manifest_path)?;
    tracing::info!(samples = dataset.len(), "train dataset loaded");
    anyhow::ensure!(!dataset.is_empty(), "training manifest is empty");

    let mut iter = BatchIterator::new(&dataset, cfg, &cfg.tokenizer_path)?;

    // -----------------------------------------------------------------------
    // 2b. Optional validation dataset
    // -----------------------------------------------------------------------
    let val_dataset: Option<ManifestDataset> = if let Some(ref val_path) = cfg.val_manifest {
        let val_ds = ManifestDataset::load(val_path)?;
        tracing::info!(samples = val_ds.len(), "validation dataset loaded");
        anyhow::ensure!(!val_ds.is_empty(), "validation manifest is empty");
        Some(val_ds)
    } else {
        None
    };

    // -----------------------------------------------------------------------
    // 3. Optimizer (AdamW)
    // -----------------------------------------------------------------------
    let optim_cfg = AdamWConfig::new()
        .with_weight_decay(cfg.weight_decay as f32)
        .with_epsilon(1e-8);
    let mut optim = optim_cfg.init::<B, LoraTextToLatentRfDiT<B>>();

    // -----------------------------------------------------------------------
    // 4. LR schedule
    // -----------------------------------------------------------------------
    let schedule = WarmupCosineSchedule {
        warmup_steps: cfg.warmup_steps,
        total_steps: cfg.max_steps,
        base_lr: cfg.lr,
        min_lr_scale: 0.1,
    };

    // -----------------------------------------------------------------------
    // 5. Training loop
    // -----------------------------------------------------------------------
    let output_dir = &cfg.output_dir;
    let grad_accum = cfg.grad_accum_steps.max(1);
    let mut accumulator: GradientsAccumulator<LoraTextToLatentRfDiT<B>> =
        GradientsAccumulator::new();
    let mut micro_step = 0usize; // micro-batches accumulated since last flush

    'outer: loop {
        iter.reset();
        while let Some(batch_result) = iter.next_batch::<B>(device) {
            if step >= cfg.max_steps {
                break 'outer;
            }

            let batch = batch_result?;

            // Sample timesteps and noise
            let t_vals =
                sample_logit_normal_t(batch.latent.dims()[0], cfg.t_mean, cfg.t_std, 1e-3, 0.999);

            let noise = Tensor::<B, 3>::random(
                batch.latent.dims(),
                burn::tensor::Distribution::Normal(0.0, 1.0),
                device,
            );

            let x0 = batch.latent.clone();
            let x_t = rf_interpolate::<B>(x0.clone(), noise.clone(), &t_vals, device);
            let t_tensor = Tensor::<B, 1>::from_floats(t_vals.as_slice(), device);

            let aux_input = if let (Some(ref_lat), Some(ref_mask)) =
                (batch.ref_latent, batch.ref_latent_mask)
            {
                AuxConditionInput::Speaker {
                    ref_latent: ref_lat,
                    ref_mask,
                }
            } else {
                AuxConditionInput::None
            };

            // Forward
            let pred = model.forward_train(
                x_t,
                t_tensor,
                batch.text_ids,
                batch.text_mask,
                aux_input,
                Some(batch.latent_mask),
            );

            let target = rf_velocity_target::<B>(noise, x0);

            // Scale loss by 1/grad_accum so accumulated gradients ≈ mean gradient.
            let loss = echo_style_masked_mse(pred, target, batch.loss_mask) / (grad_accum as f64);
            let loss_val: f32 = loss.clone().into_scalar().elem::<f32>() * grad_accum as f32;

            // Backward and accumulate
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            accumulator.accumulate(&model, grads);
            micro_step += 1;

            // Flush accumulated gradients every `grad_accum` micro-batches, or
            // when we've hit max_steps on the next real step.
            if micro_step >= grad_accum {
                let lr = schedule.lr_at_step(step);
                let flushed_grads = accumulator.grads();
                model = optim.step(lr, model, flushed_grads);
                micro_step = 0;
                step += 1;

                if step.is_multiple_of(cfg.log_every) {
                    tracing::info!(step, loss = loss_val, lr, "train step");
                }

                if step.is_multiple_of(cfg.save_every) {
                    save_lora_adapter(&model, lora_cfg, output_dir, step)?;
                }

                // Validation
                if cfg.val_every > 0 && step.is_multiple_of(cfg.val_every) {
                    if let Some(ref val_ds) = val_dataset {
                        let mut val_iter = BatchIterator::new(val_ds, cfg, &cfg.tokenizer_path)?;
                        let val_loss = run_validation::<B>(&model, &mut val_iter, cfg, device)?;
                        tracing::info!(step, val_loss, "validation");
                    }
                }
            }
        }

        // Flush any remaining accumulated gradients at end of epoch.
        if micro_step > 0 && step < cfg.max_steps {
            let lr = schedule.lr_at_step(step);
            let flushed_grads = accumulator.grads();
            model = optim.step(lr, model, flushed_grads);
            micro_step = 0;
            step += 1;
        }
    }

    // Final checkpoint
    save_lora_adapter(&model, lora_cfg, output_dir, step)?;
    tracing::info!(steps = step, "training complete");
    Ok(())
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Run validation on the inner (non-AD) backend to avoid building the
/// computation graph.  Returns mean loss over at most `cfg.val_batches` batches
/// (or the full validation set if `val_batches == 0`).
fn run_validation<B: AutodiffBackend>(
    model: &LoraTextToLatentRfDiT<B>,
    val_iter: &mut BatchIterator<'_>,
    cfg: &LoraTrainConfig,
    _device: &B::Device,
) -> anyhow::Result<f32> {
    use burn::module::AutodiffModule;
    type IB<B> = <B as AutodiffBackend>::InnerBackend;

    let inner_model = model.valid();
    val_iter.reset();

    let max_batches = if cfg.val_batches == 0 {
        usize::MAX
    } else {
        cfg.val_batches
    };
    let inner_device: <IB<B> as burn::tensor::backend::Backend>::Device =
        burn::module::Module::devices(&inner_model)
            .into_iter()
            .next()
            .unwrap_or_default();

    let mut total_loss = 0.0f32;
    let mut count = 0usize;

    while count < max_batches {
        let Some(batch_result) = val_iter.next_batch::<IB<B>>(&inner_device) else {
            break;
        };
        let batch = batch_result?;

        let t_vals =
            sample_logit_normal_t(batch.latent.dims()[0], cfg.t_mean, cfg.t_std, 1e-3, 0.999);
        let noise = Tensor::<IB<B>, 3>::random(
            batch.latent.dims(),
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &inner_device,
        );
        let x0 = batch.latent.clone();
        let x_t = rf_interpolate::<IB<B>>(x0.clone(), noise.clone(), &t_vals, &inner_device);
        let t_tensor = Tensor::<IB<B>, 1>::from_floats(t_vals.as_slice(), &inner_device);

        let aux_input =
            if let (Some(ref_lat), Some(ref_mask)) = (batch.ref_latent, batch.ref_latent_mask) {
                AuxConditionInput::Speaker {
                    ref_latent: ref_lat,
                    ref_mask,
                }
            } else {
                AuxConditionInput::None
            };

        let pred = inner_model.forward_train(
            x_t,
            t_tensor,
            batch.text_ids,
            batch.text_mask,
            aux_input,
            Some(batch.latent_mask),
        );
        let target = rf_velocity_target::<IB<B>>(noise, x0);
        let loss = echo_style_masked_mse(pred, target, batch.loss_mask);
        total_loss += loss.into_scalar().elem::<f32>();
        count += 1;
    }

    Ok(if count > 0 {
        total_loss / count as f32
    } else {
        0.0
    })
}

// ---------------------------------------------------------------------------
// Resume helpers
// ---------------------------------------------------------------------------

/// Extract the step number from a checkpoint directory name `step-NNNNNNN`.
fn parse_step_from_checkpoint_dir(path: &std::path::Path) -> anyhow::Result<usize> {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow::anyhow!("invalid checkpoint path: {}", path.display()))?;
    let digits = name.strip_prefix("step-").ok_or_else(|| {
        anyhow::anyhow!("checkpoint dir must be named 'step-NNNNNNN', got: {name}")
    })?;
    digits
        .parse::<usize>()
        .map_err(|e| anyhow::anyhow!("parse step from '{name}': {e}"))
}

/// Load LoRA adapter weights from `dir/adapter_model.safetensors` into the
/// model's LoRA parameters (warm restart — base weights and optimizer state
/// are NOT restored).
fn restore_lora_weights<B: AutodiffBackend>(
    model: LoraTextToLatentRfDiT<B>,
    dir: &std::path::Path,
    device: &B::Device,
) -> anyhow::Result<LoraTextToLatentRfDiT<B>> {
    use crate::train::lora_weights::apply_lora_adapter_to_model;
    let adapter_path = dir.join("adapter_model.safetensors");
    anyhow::ensure!(
        adapter_path.exists(),
        "resume checkpoint missing: {}",
        adapter_path.display()
    );
    apply_lora_adapter_to_model(model, &adapter_path, device)
}
