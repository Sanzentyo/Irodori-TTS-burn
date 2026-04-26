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

mod condition_dropout;
mod detached_encoding;
mod gradient_clipping;
mod resume;
mod validation;

use condition_dropout::{apply_caption_dropout_post_encode, apply_condition_dropout};
use detached_encoding::encode_conditions_detached;
use gradient_clipping::clip_grad_norm_global;
use resume::{parse_step_from_checkpoint_dir, restore_lora_weights};
use validation::run_validation;

use burn::{
    optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer},
    prelude::ElementConversion,
    tensor::{Tensor, backend::AutodiffBackend},
};
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    error::IrodoriError,
    model::condition::AuxConditionInput,
    train::{
        LoraTextToLatentRfDiT, LoraTrainConfig,
        checkpoint::save_lora_adapter,
        dataset::{BatchIterator, ManifestDataset},
        loss::{
            echo_style_masked_mse, rf_interpolate, rf_velocity_target, sample_logit_normal_t,
            sample_stratified_logit_normal_t,
        },
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
) -> crate::error::Result<()> {
    // -----------------------------------------------------------------------
    // 0. Validate config
    // -----------------------------------------------------------------------
    cfg.validate()?;

    // -----------------------------------------------------------------------
    // 0b. Seed the backend RNG for reproducible LoRA init and noise sampling.
    //     This covers Tensor::random() calls in lora_layer.rs (LoRA A kaiming
    //     init) and trainer.rs (Gaussian noise for RF interpolation).
    // -----------------------------------------------------------------------
    B::seed(device, cfg.training_seed);

    // -----------------------------------------------------------------------
    // 1. Load base model with frozen weights + fresh LoRA params
    // -----------------------------------------------------------------------
    let lora_cfg = &cfg.lora;
    let (model, model_cfg) =
        load_lora_model::<B>(&cfg.base_model_path, lora_cfg.r, lora_cfg.alpha, device)?;

    // Caption-conditioned and speaker-conditioned are mutually exclusive.
    let is_caption_mode = model_cfg.use_caption_condition;

    if is_caption_mode && cfg.caption_tokenizer_path.is_none() {
        tracing::warn!(
            "caption-conditioned model detected but no caption_tokenizer_path specified; \
             falling back to text tokenizer — this is only safe if caption and text share \
             the same vocabulary"
        );
    }

    let total_params = burn::module::Module::num_params(&model);
    tracing::info!(
        num_blocks = model_cfg.num_layers,
        r = lora_cfg.r,
        alpha = lora_cfg.alpha,
        total_params,
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
            "resuming from checkpoint (warm restart — optimizer + RNG state reset, \
             data order may differ from original run)"
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
    if dataset.is_empty() {
        return Err(IrodoriError::Dataset("training manifest is empty".into()));
    }

    let mut iter = BatchIterator::new(
        &dataset,
        cfg,
        &cfg.tokenizer_path,
        cfg.caption_tokenizer_path.as_deref(),
    )?;

    // -----------------------------------------------------------------------
    // 2b. Optional validation dataset
    // -----------------------------------------------------------------------
    let val_dataset: Option<ManifestDataset> = if let Some(ref val_path) = cfg.val_manifest {
        let val_ds = ManifestDataset::load(val_path)?;
        tracing::info!(samples = val_ds.len(), "validation dataset loaded");
        if val_ds.is_empty() {
            return Err(IrodoriError::Dataset("validation manifest is empty".into()));
        }
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
    let mut last_saved_step = 0usize; // avoids duplicate final save

    // Timing accumulators for profiling
    let mut data_time_ms = 0.0f64;
    let mut forward_time_ms = 0.0f64;
    let mut backward_time_ms = 0.0f64;
    let mut optim_time_ms = 0.0f64;
    let wall_start = std::time::Instant::now();

    // Seeded RNG for reproducible timestep sampling and condition dropout.
    let mut train_rng = StdRng::seed_from_u64(cfg.training_seed);

    'outer: loop {
        iter.reset();
        while let Some(batch_result) = {
            let data_start = std::time::Instant::now();
            let r = iter.next_batch::<B>(device);
            data_time_ms += data_start.elapsed().as_secs_f64() * 1000.0;
            r
        } {
            if step >= cfg.max_steps {
                break 'outer;
            }

            let batch = batch_result?;

            let compute_start = std::time::Instant::now();

            // Sample timesteps (stratified or i.i.d.)
            let bsz = batch.latent.dims()[0];
            let t_vals = if cfg.timestep_stratified {
                sample_stratified_logit_normal_t(
                    &mut train_rng,
                    bsz,
                    cfg.t_mean,
                    cfg.t_std,
                    1e-3,
                    0.999,
                )
            } else {
                sample_logit_normal_t(&mut train_rng, bsz, cfg.t_mean, cfg.t_std, 1e-3, 0.999)
            };

            let noise = Tensor::<B, 3>::random(
                batch.latent.dims(),
                burn::tensor::Distribution::Normal(0.0, 1.0),
                device,
            );

            let x0 = batch.latent.clone();
            let x_t = rf_interpolate::<B>(x0.clone(), noise.clone(), &t_vals, device);
            let t_tensor = Tensor::<B, 1>::from_floats(t_vals.as_slice(), device);

            let aux_input = if is_caption_mode {
                if let (Some(caption_ids), Some(caption_mask)) =
                    (batch.caption_ids, batch.caption_mask)
                {
                    AuxConditionInput::Caption {
                        ids: caption_ids,
                        mask: caption_mask,
                    }
                } else {
                    AuxConditionInput::None
                }
            } else if let (Some(ref_lat), Some(ref_mask)) =
                (batch.ref_latent, batch.ref_latent_mask)
            {
                AuxConditionInput::Speaker {
                    ref_latent: ref_lat,
                    ref_mask,
                }
            } else {
                AuxConditionInput::None
            };

            // Condition dropout: randomly zero out text/speaker masks per sample
            // to enable classifier-free guidance at inference time.
            // Caption dropout is applied AFTER encoding (see below).
            let (text_mask, aux_input) = apply_condition_dropout(
                &mut train_rng,
                batch.text_mask,
                aux_input,
                bsz,
                cfg.text_condition_dropout,
                cfg.speaker_condition_dropout,
                device,
            );

            // Forward — encode conditions on the inner (non-AD) backend to
            // skip ~270 autodiff dispatch calls for frozen components.
            let mut cond =
                encode_conditions_detached(&model, batch.text_ids, text_mask, aux_input)?;

            // Caption dropout is applied post-encoding to avoid NaN risk from
            // all-false masks fed to the caption TextEncoder.
            if is_caption_mode {
                cond.aux = apply_caption_dropout_post_encode(
                    &mut train_rng,
                    cond.aux,
                    bsz,
                    cfg.caption_condition_dropout,
                    device,
                );
            }
            let pred = model.forward_backbone(x_t, t_tensor, &cond, Some(batch.latent_mask));

            let target = rf_velocity_target::<B>(noise, x0);

            // Scale loss by 1/grad_accum so accumulated gradients ≈ mean gradient.
            let loss = echo_style_masked_mse(pred, target, batch.loss_mask) / (grad_accum as f64);

            // Only extract the scalar when we actually need to log — extracting
            // forces a GPU→CPU sync which serialises the pipeline.
            let should_log =
                micro_step + 1 >= grad_accum && (step + 1).is_multiple_of(cfg.log_every);
            let loss_val: Option<f32> = if should_log {
                Some(loss.clone().into_scalar().elem::<f32>() * grad_accum as f32)
            } else {
                None
            };

            forward_time_ms += compute_start.elapsed().as_secs_f64() * 1000.0;

            // Guard: abort early if loss is non-finite. This check is zero-cost
            // when we've already paid the GPU→CPU sync for logging; other steps
            // go unchecked to avoid serialising the pipeline.
            if let Some(val) = loss_val
                && !val.is_finite()
            {
                return Err(IrodoriError::Training(format!(
                    "loss is not finite at step {step} (loss={val}); \
                     check model inputs, learning rate, and weight initialization"
                )));
            }

            // Backward and accumulate
            let backward_start = std::time::Instant::now();
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            accumulator.accumulate(&model, grads);
            backward_time_ms += backward_start.elapsed().as_secs_f64() * 1000.0;
            micro_step += 1;

            // Flush accumulated gradients every `grad_accum` micro-batches, or
            // when we've hit max_steps on the next real step.
            if micro_step >= grad_accum {
                let optim_start = std::time::Instant::now();
                let lr = schedule.lr_at_step(step);
                let flushed_grads = accumulator.grads();

                // Global gradient norm clipping (matches PyTorch clip_grad_norm_)
                let flushed_grads = if let Some(max_norm) = cfg.grad_clip_norm {
                    clip_grad_norm_global(flushed_grads, &model, max_norm)
                } else {
                    flushed_grads
                };

                model = optim.step(lr, model, flushed_grads);
                optim_time_ms += optim_start.elapsed().as_secs_f64() * 1000.0;
                micro_step = 0;
                step += 1;

                if let Some(loss_val) = loss_val {
                    let total_steps = step as f64;
                    tracing::info!(
                        step,
                        loss = loss_val,
                        lr,
                        data_ms = data_time_ms / total_steps,
                        fwd_ms = forward_time_ms / total_steps,
                        bwd_ms = backward_time_ms / total_steps,
                        optim_ms = optim_time_ms / total_steps,
                        "train step"
                    );
                }

                if step.is_multiple_of(cfg.save_every) {
                    save_lora_adapter(&model, lora_cfg, output_dir, step)?;
                    last_saved_step = step;
                }

                // Validation
                if cfg.val_every > 0
                    && step.is_multiple_of(cfg.val_every)
                    && let Some(ref val_ds) = val_dataset
                {
                    {
                        let mut val_iter = BatchIterator::new(
                            val_ds,
                            cfg,
                            &cfg.tokenizer_path,
                            cfg.caption_tokenizer_path.as_deref(),
                        )?;
                        let val_loss = run_validation::<B>(
                            &model,
                            &mut val_iter,
                            cfg,
                            is_caption_mode,
                            device,
                        )?;
                        tracing::info!(step, val_loss, "validation");
                    }
                }
            }
        }

        // Flush any remaining accumulated gradients at end of epoch.
        if micro_step > 0 && step < cfg.max_steps {
            let lr = schedule.lr_at_step(step);
            let flushed_grads = accumulator.grads();
            let flushed_grads = if let Some(max_norm) = cfg.grad_clip_norm {
                clip_grad_norm_global(flushed_grads, &model, max_norm)
            } else {
                flushed_grads
            };
            model = optim.step(lr, model, flushed_grads);
            micro_step = 0;
            step += 1;
        }
    }

    // Final checkpoint (skip if already saved at this step)
    if step != last_saved_step {
        save_lora_adapter(&model, lora_cfg, output_dir, step)?;
    }

    // Wall-clock summary (all steps, no warmup subtraction — wall_start
    // includes the very first step so the measurement is honest).
    let wall_secs = wall_start.elapsed().as_secs_f64();
    let steps_per_sec = if wall_secs > 0.0 {
        step as f64 / wall_secs
    } else {
        0.0
    };
    tracing::info!(
        steps = step,
        wall_secs = format_args!("{wall_secs:.2}"),
        steps_per_sec = format_args!("{steps_per_sec:.2}"),
        ms_per_step = format_args!("{:.1}", wall_secs * 1000.0 / step.max(1) as f64),
        "training complete"
    );
    Ok(())
}
