//! LoRA fine-tuning training loop.
//!
//! The training loop:
//! 1. Load base checkpoint → freeze base weights → load record → re-freeze
//! 2. Build AdamW over **trainable** (LoRA) params only via [`GradientsParams`]
//! 3. Iterate over the dataset in manifest order, epoch after epoch
//! 4. Sample `t ~ logit-normal`, interpolate noised input, predict velocity
//! 5. Compute echo-style masked MSE loss
//! 6. Backward → AdamW step with warmup-cosine LR
//! 7. Save adapter checkpoint every `cfg.save_every` steps

use burn::{
    optim::{AdamWConfig, GradientsParams, Optimizer},
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
    // 2. Dataset + batch iterator
    // -----------------------------------------------------------------------
    let dataset = ManifestDataset::load(&cfg.manifest_path)?;
    tracing::info!(samples = dataset.len(), "dataset loaded");
    anyhow::ensure!(!dataset.is_empty(), "manifest is empty");

    let mut iter = BatchIterator::new(&dataset, cfg, &cfg.tokenizer_path)?;

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
    let mut model = model;
    let mut step = 0usize;

    'outer: loop {
        iter.reset();
        while let Some(batch_result) = iter.next_batch::<B>(device) {
            if step >= cfg.max_steps {
                break 'outer;
            }

            let batch = batch_result?;

            // Sample timesteps
            let t_vals =
                sample_logit_normal_t(batch.latent.dims()[0], cfg.t_mean, cfg.t_std, 1e-3, 0.999);

            // Sample noise
            let noise = Tensor::<B, 3>::random(
                batch.latent.dims(),
                burn::tensor::Distribution::Normal(0.0, 1.0),
                device,
            );

            // RF interpolation: x_t = (1-t)*x0 + t*noise
            let x0 = batch.latent.clone();
            let x_t = rf_interpolate::<B>(x0.clone(), noise.clone(), &t_vals, device);

            // Timestep tensor [B]
            let t_tensor = Tensor::<B, 1>::from_floats(t_vals.as_slice(), device);

            // Condition input
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
            let pred =
                model.forward_train(x_t, t_tensor, batch.text_ids, batch.text_mask, aux_input);

            // Target velocity: noise - x0
            let target = rf_velocity_target::<B>(noise, x0);

            // Loss
            let loss = echo_style_masked_mse(pred, target, batch.loss_mask);
            let loss_val: f32 = loss.clone().into_scalar().elem::<f32>();

            // Backward
            let grads = loss.backward();

            // Filter gradients to trainable (LoRA) params only
            let grads = GradientsParams::from_grads(grads, &model);

            // Apply LR for this step
            let lr = schedule.lr_at_step(step);

            // Step optimizer
            model = optim.step(lr, model, grads);

            step += 1;

            if step.is_multiple_of(cfg.log_every) {
                tracing::info!(step, loss = loss_val, lr, "train step");
            }

            if step.is_multiple_of(cfg.save_every) {
                save_lora_adapter(&model, lora_cfg, output_dir, step)?;
            }
        }
    }

    // Final checkpoint
    save_lora_adapter(&model, lora_cfg, output_dir, step)?;
    tracing::info!(steps = step, "training complete");
    Ok(())
}
