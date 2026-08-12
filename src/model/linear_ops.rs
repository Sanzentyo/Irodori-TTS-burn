use burn::tensor::{Tensor, module::linear};

/// Apply a linear projection to `[batch, sequence, input]` as one rank-2
/// matrix multiplication.
///
/// CubeCL tunes rank-3 matmul from the final matrix dimensions only, so a
/// broadcasted batch can select a kernel that processes each batch item
/// independently. Flattening the two leading dimensions exposes the complete
/// row count while preserving the mathematical result.
///
/// This helper is intentionally limited to inference call sites whose input is
/// known to be a dense `[B, S, K]` tensor. Shape checks happen before reshape so
/// an invalid projection fails closed instead of relying on broadcasting.
#[track_caller]
pub(super) fn linear_rank3_flattened(
    input: Tensor<3>,
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
) -> Tensor<3> {
    let [batch, sequence, input_features] = input.dims();
    let [weight_input_features, output_features] = weight.dims();

    assert!(
        batch != 0 && sequence != 0 && input_features != 0 && output_features != 0,
        "flattened rank-3 linear requires non-zero dimensions; input={:?}, weight={:?}",
        [batch, sequence, input_features],
        [weight_input_features, output_features]
    );
    assert_eq!(
        input_features, weight_input_features,
        "flattened rank-3 linear input/weight mismatch: input K={input_features}, weight K={weight_input_features}"
    );
    if let Some(bias) = bias.as_ref() {
        let [bias_features] = bias.dims();
        assert_eq!(
            bias_features, output_features,
            "flattened rank-3 linear weight/bias mismatch: weight N={output_features}, bias N={bias_features}"
        );
    }

    let rows = batch.checked_mul(sequence).unwrap_or_else(|| {
        panic!("flattened rank-3 linear row count overflow: batch={batch}, sequence={sequence}")
    });
    linear(input.reshape([rows, input_features]), weight, bias).reshape([
        batch,
        sequence,
        output_features,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{TensorData, module::linear};

    #[test]
    fn matches_rank3_linear_for_b1_b2_with_and_without_bias() {
        let device = Default::default();
        let sequence = 3;
        let input_features = 4;
        let output_features = 5;
        let weight_values = (1..=input_features * output_features)
            .map(|value| value as f32 * 0.025)
            .collect::<Vec<_>>();
        let bias_values = (1..=output_features)
            .map(|value| value as f32 * 0.125)
            .collect::<Vec<_>>();

        for batch in [1, 2] {
            for with_bias in [false, true] {
                let input_values = (1..=batch * sequence * input_features)
                    .map(|value| value as f32 * 0.05)
                    .collect::<Vec<_>>();
                let input = Tensor::<3>::from_data(
                    TensorData::new(input_values, [batch, sequence, input_features]),
                    &device,
                );
                let weight = Tensor::<2>::from_data(
                    TensorData::new(weight_values.clone(), [input_features, output_features]),
                    &device,
                );
                let bias = with_bias.then(|| {
                    Tensor::<1>::from_data(
                        TensorData::new(bias_values.clone(), [output_features]),
                        &device,
                    )
                });

                let expected = linear(input.clone(), weight.clone(), bias.clone());
                let actual = linear_rank3_flattened(input, weight, bias);
                assert_eq!(actual.dims(), [batch, sequence, output_features]);
                let magnitude: f32 = actual.clone().abs().sum().into_scalar();
                assert!(
                    magnitude > 0.0,
                    "batch={batch}, with_bias={with_bias}: output must be non-zero"
                );

                let max_abs: f32 = (expected - actual).abs().max().into_scalar();
                assert!(
                    max_abs <= 1.0e-6,
                    "batch={batch}, with_bias={with_bias}: max_abs={max_abs}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "input/weight mismatch")]
    fn rejects_incompatible_weight_shape() {
        let device = Default::default();
        let input = Tensor::<3>::ones([2, 3, 4], &device);
        let weight = Tensor::<2>::ones([5, 6], &device);
        let _ = linear_rank3_flattened(input, weight, None);
    }

    #[test]
    #[should_panic(expected = "weight/bias mismatch")]
    fn rejects_incompatible_bias_shape() {
        let device = Default::default();
        let input = Tensor::<3>::ones([2, 3, 4], &device);
        let weight = Tensor::<2>::ones([4, 6], &device);
        let bias = Tensor::<1>::ones([5], &device);
        let _ = linear_rank3_flattened(input, weight, Some(bias));
    }
}
