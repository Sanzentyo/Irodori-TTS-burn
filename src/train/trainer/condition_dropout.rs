//! Per-sample condition dropout for classifier-free guidance training.

use burn::tensor::{Bool, Tensor, TensorData, backend::Backend};

use crate::model::condition::AuxConditionInput;

/// Apply per-sample condition dropout by zeroing out conditioning masks.
///
/// For each sample in the batch, independently:
/// - With probability `text_dropout_prob`, set the entire text mask row to `false`
/// - With probability `speaker_dropout_prob`, set the speaker ref_mask row to `false`
///   and zero out the speaker ref_latent
///
/// This trains the model to handle missing conditioning, enabling classifier-free
/// guidance at inference time.
pub(super) fn apply_condition_dropout<B: Backend>(
    rng: &mut impl rand::Rng,
    text_mask: Tensor<B, 2, Bool>,
    aux_input: AuxConditionInput<B>,
    batch_size: usize,
    text_dropout_prob: f64,
    speaker_dropout_prob: f64,
    device: &B::Device,
) -> (Tensor<B, 2, Bool>, AuxConditionInput<B>) {
    let text_mask = apply_text_dropout(rng, text_mask, batch_size, text_dropout_prob, device);
    let aux_input = apply_speaker_dropout(rng, aux_input, batch_size, speaker_dropout_prob, device);
    (text_mask, aux_input)
}

fn apply_text_dropout<B: Backend>(
    rng: &mut impl rand::Rng,
    text_mask: Tensor<B, 2, Bool>,
    batch_size: usize,
    prob: f64,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    if prob <= 0.0 {
        return text_mask;
    }
    let drop_flags: Vec<bool> = (0..batch_size).map(|_| rng.r#gen::<f64>() < prob).collect();
    if !drop_flags.iter().any(|&f| f) {
        return text_mask;
    }
    let keep: Vec<f32> = drop_flags
        .iter()
        .map(|&dropped| if dropped { 0.0 } else { 1.0 })
        .collect();
    let keep_mask = Tensor::<B, 2>::from_data(TensorData::new(keep, [batch_size, 1]), device);
    (text_mask.float() * keep_mask).greater_elem(0.5)
}

fn apply_speaker_dropout<B: Backend>(
    rng: &mut impl rand::Rng,
    aux_input: AuxConditionInput<B>,
    batch_size: usize,
    prob: f64,
    device: &B::Device,
) -> AuxConditionInput<B> {
    match aux_input {
        AuxConditionInput::Speaker {
            ref_latent,
            ref_mask,
        } if prob > 0.0 => {
            let drop_flags: Vec<bool> =
                (0..batch_size).map(|_| rng.r#gen::<f64>() < prob).collect();
            if !drop_flags.iter().any(|&f| f) {
                return AuxConditionInput::Speaker {
                    ref_latent,
                    ref_mask,
                };
            }
            let keep: Vec<f32> = drop_flags
                .iter()
                .map(|&dropped| if dropped { 0.0 } else { 1.0 })
                .collect();
            let keep_2d =
                Tensor::<B, 2>::from_data(TensorData::new(keep.clone(), [batch_size, 1]), device);
            let keep_3d =
                Tensor::<B, 3>::from_data(TensorData::new(keep, [batch_size, 1, 1]), device);
            let ref_mask = (ref_mask.float() * keep_2d).greater_elem(0.5);
            let ref_latent = ref_latent * keep_3d;
            AuxConditionInput::Speaker {
                ref_latent,
                ref_mask,
            }
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use rand::{SeedableRng, rngs::StdRng};

    type TestBackend = NdArray;

    #[test]
    fn condition_dropout_prob_zero_is_noop() {
        let device = Default::default();
        let batch = 3;

        let text_mask = Tensor::<TestBackend, 2, Bool>::from_data(
            TensorData::from([
                [true, true, true, false],
                [true, true, false, false],
                [true, true, true, true],
            ]),
            &device,
        );
        let ref_latent = Tensor::<TestBackend, 3>::ones([batch, 2, 8], &device);
        let ref_mask = Tensor::<TestBackend, 2, Bool>::from_data(
            TensorData::from([[true, true], [true, false], [true, true]]),
            &device,
        );
        let aux = AuxConditionInput::Speaker {
            ref_latent: ref_latent.clone(),
            ref_mask: ref_mask.clone(),
        };

        let (out_mask, out_aux) = apply_condition_dropout(
            &mut StdRng::seed_from_u64(0),
            text_mask.clone(),
            aux,
            batch,
            0.0,
            0.0,
            &device,
        );

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

        let text_mask = Tensor::<TestBackend, 2, Bool>::from_data(
            TensorData::from([[true, true, true], [true, true, false]]),
            &device,
        );
        let ref_latent = Tensor::<TestBackend, 3>::ones([batch, 2, 4], &device);
        let ref_mask = Tensor::<TestBackend, 2, Bool>::from_data(
            TensorData::from([[true, true], [true, false]]),
            &device,
        );

        let aux = AuxConditionInput::Speaker {
            ref_latent,
            ref_mask,
        };

        let (out_mask, out_aux) = apply_condition_dropout(
            &mut StdRng::seed_from_u64(0),
            text_mask,
            aux,
            batch,
            1.0,
            1.0,
            &device,
        );

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

        let text_mask = Tensor::<TestBackend, 2, Bool>::from_data(
            TensorData::from([[true, true], [true, false]]),
            &device,
        );
        let cap_ids = Tensor::<TestBackend, 2, burn::tensor::Int>::ones([batch, 3], &device);
        let cap_mask = Tensor::<TestBackend, 2, Bool>::from_data(
            TensorData::from([[true, true, true], [true, true, false]]),
            &device,
        );

        let aux = AuxConditionInput::Caption {
            ids: cap_ids.clone(),
            mask: cap_mask.clone(),
        };

        let (_out_mask, out_aux) = apply_condition_dropout(
            &mut StdRng::seed_from_u64(0),
            text_mask,
            aux,
            batch,
            0.0,
            1.0,
            &device,
        );

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
        let text_mask =
            Tensor::<TestBackend, 2, Bool>::from_data(TensorData::from([[true, true]]), &device);
        let aux = AuxConditionInput::<TestBackend>::None;

        let (_out_mask, out_aux) = apply_condition_dropout(
            &mut StdRng::seed_from_u64(0),
            text_mask,
            aux,
            1,
            0.0,
            1.0,
            &device,
        );

        assert!(
            matches!(out_aux, AuxConditionInput::None),
            "None variant must remain None"
        );
    }
}
