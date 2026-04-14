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
    module::AutodiffModule,
    optim::{AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer},
    prelude::ElementConversion,
    tensor::{Tensor, backend::AutodiffBackend},
};

use crate::{
    model::condition::{AuxConditionInput, AuxConditionState, EncodedCondition},
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
) -> anyhow::Result<()> {
    // -----------------------------------------------------------------------
    // 0. Validate config
    // -----------------------------------------------------------------------
    cfg.validate()?;

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

    // Timing accumulators for profiling
    let mut data_time_ms = 0.0f64;
    let mut forward_time_ms = 0.0f64;
    let mut backward_time_ms = 0.0f64;
    let mut optim_time_ms = 0.0f64;

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
                sample_stratified_logit_normal_t(bsz, cfg.t_mean, cfg.t_std, 1e-3, 0.999)
            } else {
                sample_logit_normal_t(bsz, cfg.t_mean, cfg.t_std, 1e-3, 0.999)
            };

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

            // Condition dropout: randomly zero out text/speaker masks per sample
            // to enable classifier-free guidance at inference time.
            let (text_mask, aux_input) = apply_condition_dropout(
                batch.text_mask,
                aux_input,
                bsz,
                cfg.text_condition_dropout,
                cfg.speaker_condition_dropout,
                device,
            );

            // Forward — encode conditions on the inner (non-AD) backend to
            // skip ~270 autodiff dispatch calls for frozen components.
            let cond = encode_conditions_detached(&model, batch.text_ids, text_mask, aux_input);
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
                }

                // Validation
                if cfg.val_every > 0
                    && step.is_multiple_of(cfg.val_every)
                    && let Some(ref val_ds) = val_dataset
                {
                    {
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
    type IB<B> = <B as AutodiffBackend>::InnerBackend;

    let inner_model = model.valid();
    // Note: no reset() — the iterator is created fresh before each validation
    // run with sequential order, which is the correct behavior for validation
    // (deterministic, no shuffling).

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
// Condition dropout
// ---------------------------------------------------------------------------

/// Apply per-sample condition dropout by zeroing out conditioning masks.
///
/// For each sample in the batch, independently:
/// - With probability `text_dropout_prob`, set the entire text mask row to `false`
/// - With probability `speaker_dropout_prob`, set the speaker ref_mask row to `false`
///   and zero out the speaker ref_latent
///
/// This trains the model to handle missing conditioning, enabling classifier-free
/// guidance at inference time.
fn apply_condition_dropout<B: burn::tensor::backend::Backend>(
    text_mask: burn::tensor::Tensor<B, 2, burn::tensor::Bool>,
    aux_input: AuxConditionInput<B>,
    batch_size: usize,
    text_dropout_prob: f64,
    speaker_dropout_prob: f64,
    device: &B::Device,
) -> (
    burn::tensor::Tensor<B, 2, burn::tensor::Bool>,
    AuxConditionInput<B>,
) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Text condition dropout: zero out text_mask for dropped samples.
    // NOTE: when `AuxConditionInput::Speaker` is present, some per-sample
    // entries may have missing speaker data represented as zero ref_latent +
    // all-false ref_mask.  Multiplying zeros/all-false by the keep mask is a
    // no-op, so the result is equivalent to Python's `has_speaker & ~drop`.
    let text_mask = if text_dropout_prob > 0.0 {
        let drop_flags: Vec<bool> = (0..batch_size)
            .map(|_| rng.r#gen::<f64>() < text_dropout_prob)
            .collect();
        if drop_flags.iter().any(|&f| f) {
            // Build a [B, 1] bool mask where true = KEEP (not dropped)
            let keep: Vec<f32> = drop_flags
                .iter()
                .map(|&dropped| if dropped { 0.0 } else { 1.0 })
                .collect();
            let keep_mask = burn::tensor::Tensor::<B, 2>::from_data(
                burn::tensor::TensorData::new(keep, [batch_size, 1]),
                device,
            );
            // Multiply float text_mask by keep_mask, then convert back to bool.
            // text_mask is [B, T], keep_mask is [B, 1] → broadcast.
            let masked_float = text_mask.float() * keep_mask;
            masked_float.greater_elem(0.5)
        } else {
            text_mask
        }
    } else {
        text_mask
    };

    // Speaker condition dropout: zero out ref_mask and ref_latent for dropped samples.
    let aux_input = match aux_input {
        AuxConditionInput::Speaker {
            ref_latent,
            ref_mask,
        } if speaker_dropout_prob > 0.0 => {
            let drop_flags: Vec<bool> = (0..batch_size)
                .map(|_| rng.r#gen::<f64>() < speaker_dropout_prob)
                .collect();
            if drop_flags.iter().any(|&f| f) {
                let keep: Vec<f32> = drop_flags
                    .iter()
                    .map(|&dropped| if dropped { 0.0 } else { 1.0 })
                    .collect();
                // ref_mask: [B, T_ref], ref_latent: [B, T_ref, D]
                let keep_2d = burn::tensor::Tensor::<B, 2>::from_data(
                    burn::tensor::TensorData::new(keep.clone(), [batch_size, 1]),
                    device,
                );
                let keep_3d = burn::tensor::Tensor::<B, 3>::from_data(
                    burn::tensor::TensorData::new(keep, [batch_size, 1, 1]),
                    device,
                );
                let ref_mask = (ref_mask.float() * keep_2d).greater_elem(0.5);
                let ref_latent = ref_latent * keep_3d;
                AuxConditionInput::Speaker {
                    ref_latent,
                    ref_mask,
                }
            } else {
                AuxConditionInput::Speaker {
                    ref_latent,
                    ref_mask,
                }
            }
        }
        other => other,
    };

    (text_mask, aux_input)
}

// ---------------------------------------------------------------------------
// Global gradient norm clipping
// ---------------------------------------------------------------------------

/// Clip gradients by global L2 norm, matching PyTorch's `clip_grad_norm_`.
///
/// Computes the total L2 norm across **all** parameters, then scales each
/// gradient by `min(1, max_norm / (total_norm + ε))` if the total norm exceeds
/// `max_norm`.
///
/// This differs from burn's built-in per-parameter clipping — here we compute
/// one global norm and apply one uniform scale factor to all gradients.
fn clip_grad_norm_global<B: AutodiffBackend>(
    mut grads: GradientsParams,
    model: &LoraTextToLatentRfDiT<B>,
    max_norm: f64,
) -> GradientsParams {
    use burn::module::{Module, ModuleVisitor, Param};
    type IB<B> = <B as AutodiffBackend>::InnerBackend;

    // Pass 1: compute total squared norm.
    struct NormComputer<'a, B: AutodiffBackend> {
        grads: &'a GradientsParams,
        sq_sum: f64,
        _b: std::marker::PhantomData<B>,
    }
    impl<B: AutodiffBackend> ModuleVisitor<B> for NormComputer<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if let Some(grad) = self.grads.get::<IB<B>, D>(param.id) {
                // NOTE: squaring happens in the backend's native dtype.  For
                // bf16/f16 with very large gradient values this could overflow,
                // but typical LoRA gradients are O(1e-3) — well within range.
                let sq: f64 = grad.clone().mul(grad).sum().into_scalar().elem();
                self.sq_sum += sq;
            }
        }
    }

    let mut computer = NormComputer::<B> {
        grads: &grads,
        sq_sum: 0.0,
        _b: std::marker::PhantomData,
    };
    model.visit(&mut computer);

    let total_norm = computer.sq_sum.sqrt();
    let clip_coef = (max_norm / (total_norm + 1e-6)).min(1.0);

    if clip_coef >= 1.0 {
        return grads;
    }

    // Pass 2: scale all gradients by clip_coef.
    struct NormScaler<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        scale: f32,
        _b: std::marker::PhantomData<B>,
    }
    impl<B: AutodiffBackend> ModuleVisitor<B> for NormScaler<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if let Some(grad) = self.grads.remove::<IB<B>, D>(param.id) {
                self.grads.register::<IB<B>, D>(param.id, grad * self.scale);
            }
        }
    }

    let mut scaler = NormScaler::<B> {
        grads: &mut grads,
        scale: clip_coef as f32,
        _b: std::marker::PhantomData,
    };
    model.visit(&mut scaler);

    grads
}

// ---------------------------------------------------------------------------
// Detached conditioning (inner-backend optimization)
// ---------------------------------------------------------------------------

/// Encode conditions on the inner (non-AD) backend to avoid autodiff dispatch
/// overhead for the frozen text encoder and aux conditioner.
///
/// Since these components are entirely frozen (no trainable parameters), running
/// them through the AD graph adds ~270 unnecessary dispatch calls.  This helper
/// strips the AD wrapper, runs encode on the raw backend, and wraps the results
/// back as AD leaf tensors (no gradient flows through conditioning).
///
/// # Safety invariant
///
/// This is correct **only while** `text_encoder`, `text_norm`, and
/// `aux_conditioner` remain fully frozen (no trainable parameters).
/// If any of these modules are later LoRA-wrapped or unfrozen, this
/// function will silently cut their gradients — replace with
/// `model.encode_conditions(...)` in that case.
fn encode_conditions_detached<B: AutodiffBackend>(
    model: &LoraTextToLatentRfDiT<B>,
    text_input_ids: burn::tensor::Tensor<B, 2, burn::tensor::Int>,
    text_mask: burn::tensor::Tensor<B, 2, burn::tensor::Bool>,
    aux_input: AuxConditionInput<B>,
) -> EncodedCondition<B> {
    use burn::tensor::TensorPrimitive;
    type IB<B> = <B as AutodiffBackend>::InnerBackend;

    // Float tensor conversion helpers that handle TensorPrimitive::Float wrapping.
    let to_inner_float = |t: Tensor<B, 3>| -> Tensor<IB<B>, 3> {
        let prim = match t.into_primitive() {
            TensorPrimitive::Float(p) => p,
            _ => unreachable!("expected Float primitive"),
        };
        Tensor::from_primitive(TensorPrimitive::Float(B::inner(prim)))
    };
    let from_inner_float = |t: Tensor<IB<B>, 3>| -> Tensor<B, 3> {
        let prim = match t.into_primitive() {
            TensorPrimitive::Float(p) => p,
            _ => unreachable!("expected Float primitive"),
        };
        Tensor::from_primitive(TensorPrimitive::Float(B::from_inner(prim)))
    };

    let inner_model = model.valid();

    // Int and Bool tensors are pass-through for the Autodiff backend.
    let inner_text_ids = burn::tensor::Tensor::<IB<B>, 2, burn::tensor::Int>::from_primitive(
        B::int_inner(text_input_ids.into_primitive()),
    );
    let inner_text_mask = burn::tensor::Tensor::<IB<B>, 2, burn::tensor::Bool>::from_primitive(
        B::bool_inner(text_mask.into_primitive()),
    );

    let inner_aux = match aux_input {
        AuxConditionInput::Speaker {
            ref_latent,
            ref_mask,
        } => AuxConditionInput::Speaker {
            ref_latent: to_inner_float(ref_latent),
            ref_mask: burn::tensor::Tensor::<IB<B>, 2, burn::tensor::Bool>::from_primitive(
                B::bool_inner(ref_mask.into_primitive()),
            ),
        },
        AuxConditionInput::Caption { ids, mask } => AuxConditionInput::Caption {
            ids: burn::tensor::Tensor::<IB<B>, 2, burn::tensor::Int>::from_primitive(B::int_inner(
                ids.into_primitive(),
            )),
            mask: burn::tensor::Tensor::<IB<B>, 2, burn::tensor::Bool>::from_primitive(
                B::bool_inner(mask.into_primitive()),
            ),
        },
        AuxConditionInput::None => AuxConditionInput::None,
    };

    let inner_cond = inner_model.encode_conditions(inner_text_ids, inner_text_mask, inner_aux);

    // Wrap results back as AD leaf tensors (no gradient).
    let text_state = from_inner_float(inner_cond.text_state);
    let text_mask = burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::from_primitive(
        B::bool_from_inner(inner_cond.text_mask.into_primitive()),
    );
    let aux = inner_cond.aux.map(|a| match a {
        AuxConditionState::Speaker { state, mask } => AuxConditionState::Speaker {
            state: from_inner_float(state),
            mask: burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::from_primitive(
                B::bool_from_inner(mask.into_primitive()),
            ),
        },
        AuxConditionState::Caption { state, mask } => AuxConditionState::Caption {
            state: from_inner_float(state),
            mask: burn::tensor::Tensor::<B, 2, burn::tensor::Bool>::from_primitive(
                B::bool_from_inner(mask.into_primitive()),
            ),
        },
    });

    EncodedCondition {
        text_state,
        text_mask,
        aux,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    // -----------------------------------------------------------------------
    // parse_step_from_checkpoint_dir
    // -----------------------------------------------------------------------

    #[test]
    fn parse_step_valid() {
        let p = std::path::Path::new("/tmp/checkpoints/step-0000042");
        assert_eq!(parse_step_from_checkpoint_dir(p).unwrap(), 42);
    }

    #[test]
    fn parse_step_zero_padded() {
        let p = std::path::Path::new("step-0000001");
        assert_eq!(parse_step_from_checkpoint_dir(p).unwrap(), 1);
    }

    #[test]
    fn parse_step_invalid_prefix() {
        let p = std::path::Path::new("epoch-5");
        assert!(parse_step_from_checkpoint_dir(p).is_err());
    }

    #[test]
    fn parse_step_non_numeric() {
        let p = std::path::Path::new("step-abc");
        assert!(parse_step_from_checkpoint_dir(p).is_err());
    }

    // -----------------------------------------------------------------------
    // apply_condition_dropout
    // -----------------------------------------------------------------------

    #[test]
    fn condition_dropout_prob_zero_is_noop() {
        let device = Default::default();
        let batch = 3;
        let _seq = 4;

        let text_mask = burn::tensor::Tensor::<TestBackend, 2, burn::tensor::Bool>::from_data(
            burn::tensor::TensorData::from([
                [true, true, true, false],
                [true, true, false, false],
                [true, true, true, true],
            ]),
            &device,
        );
        let ref_latent = burn::tensor::Tensor::<TestBackend, 3>::ones([batch, 2, 8], &device);
        let ref_mask = burn::tensor::Tensor::<TestBackend, 2, burn::tensor::Bool>::from_data(
            burn::tensor::TensorData::from([[true, true], [true, false], [true, true]]),
            &device,
        );
        let aux = AuxConditionInput::Speaker {
            ref_latent: ref_latent.clone(),
            ref_mask: ref_mask.clone(),
        };

        let (out_mask, out_aux) =
            apply_condition_dropout(text_mask.clone(), aux, batch, 0.0, 0.0, &device);

        // With prob=0, masks should be identical
        let orig: Vec<bool> = text_mask.into_data().to_vec().unwrap();
        let result: Vec<bool> = out_mask.into_data().to_vec().unwrap();
        assert_eq!(orig, result, "text_mask unchanged with prob=0");

        if let AuxConditionInput::Speaker {
            ref_latent: out_rl,
            ref_mask: out_rm,
        } = out_aux
        {
            let orig_rm: Vec<bool> = ref_mask.into_data().to_vec().unwrap();
            let out_rm_v: Vec<bool> = out_rm.into_data().to_vec().unwrap();
            assert_eq!(orig_rm, out_rm_v, "ref_mask unchanged with prob=0");

            let orig_rl: Vec<f32> = ref_latent.into_data().to_vec().unwrap();
            let out_rl_v: Vec<f32> = out_rl.into_data().to_vec().unwrap();
            assert_eq!(orig_rl, out_rl_v, "ref_latent unchanged with prob=0");
        } else {
            panic!("expected Speaker variant");
        }
    }

    #[test]
    fn condition_dropout_prob_one_drops_all() {
        let device = Default::default();
        let batch = 2;

        let text_mask = burn::tensor::Tensor::<TestBackend, 2, burn::tensor::Bool>::from_data(
            burn::tensor::TensorData::from([[true, true, true], [true, true, false]]),
            &device,
        );
        let ref_latent = burn::tensor::Tensor::<TestBackend, 3>::ones([batch, 2, 4], &device);
        let ref_mask = burn::tensor::Tensor::<TestBackend, 2, burn::tensor::Bool>::from_data(
            burn::tensor::TensorData::from([[true, true], [true, false]]),
            &device,
        );

        let aux = AuxConditionInput::Speaker {
            ref_latent,
            ref_mask,
        };

        let (out_mask, out_aux) = apply_condition_dropout(text_mask, aux, batch, 1.0, 1.0, &device);

        // With prob=1, ALL text mask entries should be false
        let tm: Vec<bool> = out_mask.into_data().to_vec().unwrap();
        assert!(
            tm.iter().all(|&v| !v),
            "all text_mask entries must be false"
        );

        if let AuxConditionInput::Speaker {
            ref_latent: out_rl,
            ref_mask: out_rm,
        } = out_aux
        {
            let rm: Vec<bool> = out_rm.into_data().to_vec().unwrap();
            assert!(rm.iter().all(|&v| !v), "all ref_mask entries must be false");

            let rl: Vec<f32> = out_rl.into_data().to_vec().unwrap();
            assert!(
                rl.iter().all(|&v| v == 0.0),
                "all ref_latent entries must be zero"
            );
        } else {
            panic!("expected Speaker variant");
        }
    }

    #[test]
    fn condition_dropout_caption_variant_unchanged() {
        let device = Default::default();
        let batch = 2;

        let text_mask = burn::tensor::Tensor::<TestBackend, 2, burn::tensor::Bool>::from_data(
            burn::tensor::TensorData::from([[true, true], [true, false]]),
            &device,
        );
        let cap_ids =
            burn::tensor::Tensor::<TestBackend, 2, burn::tensor::Int>::ones([batch, 3], &device);
        let cap_mask = burn::tensor::Tensor::<TestBackend, 2, burn::tensor::Bool>::from_data(
            burn::tensor::TensorData::from([[true, true, true], [true, true, false]]),
            &device,
        );

        let aux = AuxConditionInput::Caption {
            ids: cap_ids.clone(),
            mask: cap_mask.clone(),
        };

        // speaker_dropout_prob=1.0 should not affect Caption variant
        let (_out_mask, out_aux) =
            apply_condition_dropout(text_mask, aux, batch, 0.0, 1.0, &device);

        if let AuxConditionInput::Caption {
            ids: out_ids,
            mask: out_m,
        } = out_aux
        {
            let orig_m: Vec<bool> = cap_mask.into_data().to_vec().unwrap();
            let out_m_v: Vec<bool> = out_m.into_data().to_vec().unwrap();
            assert_eq!(orig_m, out_m_v, "caption mask unchanged by speaker dropout");

            let orig_ids: Vec<i64> = cap_ids.into_data().to_vec().unwrap();
            let out_ids_v: Vec<i64> = out_ids.into_data().to_vec().unwrap();
            assert_eq!(
                orig_ids, out_ids_v,
                "caption ids unchanged by speaker dropout"
            );
        } else {
            panic!("expected Caption variant");
        }
    }

    #[test]
    fn condition_dropout_none_variant_unchanged() {
        let device = Default::default();
        let text_mask = burn::tensor::Tensor::<TestBackend, 2, burn::tensor::Bool>::from_data(
            burn::tensor::TensorData::from([[true, true]]),
            &device,
        );
        let aux = AuxConditionInput::<TestBackend>::None;

        let (_out_mask, out_aux) = apply_condition_dropout(text_mask, aux, 1, 0.0, 1.0, &device);

        assert!(
            matches!(out_aux, AuxConditionInput::None),
            "None variant must remain None"
        );
    }
}
