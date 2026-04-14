//! Validation loop for LoRA fine-tuning.

use burn::{
    module::AutodiffModule,
    prelude::ElementConversion,
    tensor::{
        Tensor,
        backend::{AutodiffBackend, Backend},
    },
};
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    model::condition::AuxConditionInput,
    train::{
        LoraTextToLatentRfDiT, LoraTrainConfig,
        dataset::BatchIterator,
        loss::{echo_style_masked_mse, rf_interpolate, rf_velocity_target, sample_logit_normal_t},
    },
};

type IB<B> = <B as AutodiffBackend>::InnerBackend;

/// Run validation on the inner (non-AD) backend to avoid building the
/// computation graph.  Returns mean loss over at most `cfg.val_batches` batches
/// (or the full validation set if `val_batches == 0`).
pub(super) fn run_validation<B: AutodiffBackend>(
    model: &LoraTextToLatentRfDiT<B>,
    val_iter: &mut BatchIterator<'_>,
    cfg: &LoraTrainConfig,
    _device: &B::Device,
) -> crate::error::Result<f32> {
    let inner_model = model.valid();

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

    // Seed the inner backend for deterministic validation noise.
    IB::<B>::seed(&inner_device, 0xCAFE);

    let mut total_loss = 0.0f32;
    let mut count = 0usize;
    // Fixed-seed RNG for deterministic validation loss.
    let mut val_rng = StdRng::seed_from_u64(0xCAFE);

    while count < max_batches {
        let Some(batch_result) = val_iter.next_batch::<IB<B>>(&inner_device) else {
            break;
        };
        let batch = batch_result?;

        let t_vals = sample_logit_normal_t(
            &mut val_rng,
            batch.latent.dims()[0],
            cfg.t_mean,
            cfg.t_std,
            1e-3,
            0.999,
        );
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
        )?;
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
