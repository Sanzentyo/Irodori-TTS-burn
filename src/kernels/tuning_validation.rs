//! Shared cold-tuning readback and numerical acceptance policy.

use burn::{
    backend::wgpu::{CubeTensor, WgpuRuntime},
    tensor::Tensor,
};

pub(super) fn read<const D: usize>(output: CubeTensor<WgpuRuntime>) -> Result<Vec<f32>, String> {
    Tensor::<D>::from_primitive::<crate::WgpuRaw>(output)
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|error| format!("tuning readback: {error}"))
}

pub(super) fn check_accuracy(expected: &[f32], actual: &[f32], half: bool) -> Result<(), String> {
    if expected.len() != actual.len() || expected.is_empty() {
        return Err("tuning output shape mismatch".to_owned());
    }
    let (atol, rtol) = if half {
        (0.002, 0.004)
    } else {
        (0.00002, 0.00002)
    };
    for (&expected, &actual) in expected.iter().zip(actual) {
        if !expected.is_finite()
            || !actual.is_finite()
            || (actual - expected).abs() > atol + rtol * expected.abs()
        {
            return Err(format!(
                "accuracy gate: expected {expected}, got {actual} (atol={atol}, rtol={rtol})"
            ));
        }
    }
    Ok(())
}
