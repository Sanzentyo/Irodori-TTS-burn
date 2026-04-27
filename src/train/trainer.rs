//! LoRA fine-tuning training loop.
//!
//! The training loop:
//! 1. Load base checkpoint → freeze base weights → optionally restore LoRA params from
//!    a resume checkpoint (warm restart)
//! 2. Build optimizer (AdamW or Muon) over **trainable** (LoRA) params only via
//!    [`GradientsParams`]
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
use resume::{parse_step_from_checkpoint_dir, restore_lora_weights, restore_optimizer_state};
use validation::run_validation;

use burn::{
    optim::{
        AdamWConfig, GradientsAccumulator, GradientsParams, MuonConfig, Optimizer,
        decay::WeightDecayConfig, momentum::MomentumConfig,
    },
    prelude::ElementConversion,
    tensor::{Tensor, backend::AutodiffBackend},
};
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    config::{AdjustLrPolicy, OptimizerKind},
    error::IrodoriError,
    model::condition::AuxConditionInput,
    train::{
        LoraTextToLatentRfDiT, LoraTrainConfig,
        checkpoint::save_checkpoint,
        dataset::{BatchIterator, ManifestDataset},
        loss::{
            echo_style_masked_mse, rf_interpolate, rf_velocity_target, sample_logit_normal_t,
            sample_stratified_logit_normal_t,
        },
        lr_schedule::WarmupCosineSchedule,
        metrics::{JsonlSink, MetricsSink, MultiSink, StdoutSink},
    },
    weights::load_lora_model,
};

// ---------------------------------------------------------------------------
// LR policy conversion
// ---------------------------------------------------------------------------

fn adjust_lr_to_burn(policy: AdjustLrPolicy) -> burn::optim::AdjustLrFn {
    match policy {
        AdjustLrPolicy::Original => burn::optim::AdjustLrFn::Original,
        AdjustLrPolicy::MatchRmsAdamW => burn::optim::AdjustLrFn::MatchRmsAdamW,
    }
}

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
    // -----------------------------------------------------------------------
    B::seed(device, cfg.training_seed);

    // -----------------------------------------------------------------------
    // 1. Load base model with frozen weights + fresh LoRA params
    // -----------------------------------------------------------------------
    let lora_cfg = &cfg.lora;
    let (model, model_cfg) =
        load_lora_model::<B>(&cfg.base_model_path, lora_cfg.r, lora_cfg.alpha, device)?;

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
    // 1b. Optional: resume from checkpoint
    // -----------------------------------------------------------------------
    let (model, step, resume_dir) = if let Some(ref resume_path) = cfg.resume_from {
        let step = parse_step_from_checkpoint_dir(resume_path)?;
        tracing::info!(
            step,
            path = %resume_path.display(),
            "resuming from checkpoint"
        );
        let model = restore_lora_weights(model, resume_path, device)?;
        (model, step, Some(resume_path.as_path()))
    } else {
        (model, 0usize, None)
    };

    // -----------------------------------------------------------------------
    // 2. Training dataset
    // -----------------------------------------------------------------------
    let dataset = ManifestDataset::load(&cfg.manifest_path)?;
    tracing::info!(samples = dataset.len(), "train dataset loaded");
    if dataset.is_empty() {
        return Err(IrodoriError::Dataset("training manifest is empty".into()));
    }

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
    // 3. Metric sink
    // -----------------------------------------------------------------------
    let mut sink: Box<dyn MetricsSink> = if let Some(ref metrics_file) = cfg.metrics_file {
        Box::new(MultiSink::new(StdoutSink, JsonlSink::create(metrics_file)?))
    } else {
        Box::new(StdoutSink)
    };

    // -----------------------------------------------------------------------
    // 4. Optimizer — dispatch on kind, then delegate to the generic inner loop
    // -----------------------------------------------------------------------
    let rng_seed = if step > 0 {
        cfg.training_seed.wrapping_add(step as u64)
    } else {
        cfg.training_seed
    };
    let train_rng = StdRng::seed_from_u64(rng_seed);

    match &cfg.optimizer {
        OptimizerKind::AdamW => {
            let optim_cfg = AdamWConfig::new()
                .with_weight_decay(cfg.weight_decay as f32)
                .with_epsilon(1e-8);
            let mut optim = optim_cfg.init::<B, LoraTextToLatentRfDiT<B>>();
            if let Some(dir) = resume_dir {
                optim = restore_optimizer_state(optim, dir, device)?;
            }
            train_lora_inner(
                cfg,
                device,
                model,
                optim,
                step,
                is_caption_mode,
                lora_cfg,
                &dataset,
                val_dataset.as_ref(),
                train_rng,
                &mut *sink,
            )
        }
        OptimizerKind::Muon(muon_cfg) => {
            let wd = if cfg.weight_decay > 0.0 {
                Some(WeightDecayConfig::new(cfg.weight_decay as f32))
            } else {
                None
            };
            let optim_cfg = MuonConfig::new()
                .with_momentum(MomentumConfig {
                    momentum: muon_cfg.momentum,
                    dampening: 0.0,
                    nesterov: true,
                })
                .with_ns_steps(muon_cfg.ns_steps)
                .with_adjust_lr_fn(adjust_lr_to_burn(muon_cfg.adjust_lr_fn))
                .with_weight_decay(wd);
            let mut optim = optim_cfg.init::<B, LoraTextToLatentRfDiT<B>>();
            if let Some(dir) = resume_dir {
                optim = restore_optimizer_state(optim, dir, device)?;
            }
            train_lora_inner(
                cfg,
                device,
                model,
                optim,
                step,
                is_caption_mode,
                lora_cfg,
                &dataset,
                val_dataset.as_ref(),
                train_rng,
                &mut *sink,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Generic training loop (optimizer-agnostic)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn train_lora_inner<B, O>(
    cfg: &LoraTrainConfig,
    device: &B::Device,
    mut model: LoraTextToLatentRfDiT<B>,
    mut optim: O,
    mut step: usize,
    is_caption_mode: bool,
    lora_cfg: &crate::config::LoraConfig,
    dataset: &ManifestDataset,
    val_dataset: Option<&ManifestDataset>,
    mut train_rng: StdRng,
    sink: &mut dyn MetricsSink,
) -> crate::error::Result<()>
where
    B: AutodiffBackend,
    O: Optimizer<LoraTextToLatentRfDiT<B>, B>,
{
    let mut iter = BatchIterator::new(
        dataset,
        cfg,
        &cfg.tokenizer_path,
        cfg.caption_tokenizer_path.as_deref(),
    )?;

    // -----------------------------------------------------------------------
    // LR schedule
    // -----------------------------------------------------------------------
    let schedule = WarmupCosineSchedule {
        warmup_steps: cfg.warmup_steps,
        total_steps: cfg.max_steps,
        base_lr: cfg.lr,
        min_lr_scale: 0.1,
    };

    let output_dir = &cfg.output_dir;
    let grad_accum = cfg.grad_accum_steps.max(1);
    let mut accumulator: GradientsAccumulator<LoraTextToLatentRfDiT<B>> =
        GradientsAccumulator::new();
    let mut micro_step = 0usize;
    let mut last_saved_step = 0usize;

    // Timing accumulators for profiling
    let mut data_time_ms = 0.0f64;
    let mut forward_time_ms = 0.0f64;
    let mut backward_time_ms = 0.0f64;
    let mut optim_time_ms = 0.0f64;
    let wall_start = std::time::Instant::now();

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

            let (text_mask, aux_input) = apply_condition_dropout(
                &mut train_rng,
                batch.text_mask,
                aux_input,
                bsz,
                cfg.text_condition_dropout,
                cfg.speaker_condition_dropout,
                device,
            );

            let mut cond =
                encode_conditions_detached(&model, batch.text_ids, text_mask, aux_input)?;

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

            let loss = echo_style_masked_mse(pred, target, batch.loss_mask) / (grad_accum as f64);

            let should_log =
                micro_step + 1 >= grad_accum && (step + 1).is_multiple_of(cfg.log_every);
            let loss_val: Option<f32> = if should_log {
                Some(loss.clone().into_scalar().elem::<f32>() * grad_accum as f32)
            } else {
                None
            };

            forward_time_ms += compute_start.elapsed().as_secs_f64() * 1000.0;

            if let Some(val) = loss_val
                && !val.is_finite()
            {
                return Err(IrodoriError::Training(format!(
                    "loss is not finite at step {step} (loss={val}); \
                     check model inputs, learning rate, and weight initialization"
                )));
            }

            let backward_start = std::time::Instant::now();
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            accumulator.accumulate(&model, grads);
            backward_time_ms += backward_start.elapsed().as_secs_f64() * 1000.0;
            micro_step += 1;

            if micro_step >= grad_accum {
                let optim_start = std::time::Instant::now();
                let lr = schedule.lr_at_step(step);
                let flushed_grads = accumulator.grads();

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
                    sink.log_scalar(step, "train/loss", loss_val as f64);
                    sink.log_scalar(step, "train/lr", lr);
                }

                if step.is_multiple_of(cfg.save_every) {
                    save_checkpoint(
                        &model,
                        &optim,
                        lora_cfg,
                        step,
                        cfg.training_seed,
                        output_dir,
                    )?;
                    last_saved_step = step;
                }

                if cfg.val_every > 0
                    && step.is_multiple_of(cfg.val_every)
                    && let Some(val_ds) = val_dataset
                {
                    let mut val_iter = BatchIterator::new(
                        val_ds,
                        cfg,
                        &cfg.tokenizer_path,
                        cfg.caption_tokenizer_path.as_deref(),
                    )?;
                    let val_loss =
                        run_validation::<B>(&model, &mut val_iter, cfg, is_caption_mode, device)?;
                    tracing::info!(step, val_loss, "validation");
                    sink.log_scalar(step, "val/loss", val_loss as f64);
                }

                sink.flush().map_err(|e| {
                    IrodoriError::Training(format!("metrics sink flush failed: {e}"))
                })?;
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
        save_checkpoint(
            &model,
            &optim,
            lora_cfg,
            step,
            cfg.training_seed,
            output_dir,
        )?;
    }

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
