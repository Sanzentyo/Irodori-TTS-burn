//! Criterion benchmarks for the Irodori-TTS-burn inference pipeline.
//!
//! Uses the small validation model (`target/validate_weights.safetensors`).
//! Generate it first with `just validate-fixtures`, then run with `just bench`.
//!
//! Benchmarks:
//! - `encode_conditions` — text + speaker encoding
//! - `forward_with_cond` — single diffusion step
//! - `sample_4steps` — full Euler sampler, 4 steps

#![recursion_limit = "256"]

use burn::{
    backend::NdArray,
    tensor::{Bool, Int, Tensor, TensorData, backend::Backend},
};
use criterion::{Criterion, criterion_group, criterion_main};

use irodori_tts_burn::{
    InferenceBuilder,
    model::condition::AuxConditionInput,
    rf::{SamplerParams, SamplingRequest},
};

// ---------------------------------------------------------------------------
// Shared setup
// ---------------------------------------------------------------------------

const WEIGHTS_PATH: &str = "target/validate_weights.safetensors";
const SEQ_LEN: usize = 16;
const TEXT_LEN: usize = 4;

type B = NdArray<f32>;

/// Load the small validation model and build dummy inputs.
///
/// Tensor materialisation is forced here so setup cost is excluded from
/// benchmark timing.
fn setup() -> (
    irodori_tts_burn::InferenceEngine<B>,
    Tensor<B, 2, Int>,
    Tensor<B, 2, Bool>,
) {
    let device: <B as Backend>::Device = Default::default();

    let engine = InferenceBuilder::<B, _>::new(device)
        .load_weights(WEIGHTS_PATH)
        .expect("load model — run `just validate-fixtures` first")
        .with_default_sampling()
        .build();

    let text_ids = Tensor::<B, 2, Int>::from_data(
        TensorData::new(vec![0_i32, 1, 2, 3], [1, TEXT_LEN]),
        &device,
    );
    let text_mask: Tensor<B, 2, Bool> =
        Tensor::<B, 2>::ones([1, TEXT_LEN], &device).greater_elem(0.0f32);

    let _ = text_ids.clone().into_data();
    let _ = text_mask.clone().into_data();

    (engine, text_ids, text_mask)
}

fn make_request(text_ids: Tensor<B, 2, Int>, text_mask: Tensor<B, 2, Bool>) -> SamplingRequest<B> {
    SamplingRequest {
        text_ids,
        text_mask,
        ref_latent: None,
        ref_mask: None,
        sequence_length: SEQ_LEN,
        caption_ids: None,
        caption_mask: None,
        initial_noise: None,
    }
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_encode_conditions(c: &mut Criterion) {
    let (engine, text_ids, text_mask) = setup();

    c.bench_function("encode_conditions", |b| {
        b.iter(|| {
            let cond = engine.model().encode_conditions(
                text_ids.clone(),
                text_mask.clone(),
                AuxConditionInput::None,
            );
            let _ = cond.text_state.into_data();
        });
    });
}

fn bench_forward_with_cond(c: &mut Criterion) {
    let (engine, text_ids, text_mask) = setup();
    let device: <B as Backend>::Device = Default::default();

    let cond = engine.model().encode_conditions(
        text_ids.clone(),
        text_mask.clone(),
        AuxConditionInput::None,
    );
    let x_t = Tensor::<B, 3>::zeros([1, SEQ_LEN, engine.model().patched_latent_dim()], &device);
    let t = Tensor::<B, 1>::from_data(TensorData::new(vec![0.5_f32], [1]), &device);

    c.bench_function("forward_with_cond", |b| {
        b.iter(|| {
            let v = engine
                .model()
                .forward_with_cond(x_t.clone(), t.clone(), &cond, None, None);
            let _ = v.into_data();
        });
    });
}

fn bench_sample_4steps(c: &mut Criterion) {
    let device: <B as Backend>::Device = Default::default();

    let engine = InferenceBuilder::<B, _>::new(device)
        .load_weights(WEIGHTS_PATH)
        .expect("load model — run `just validate-fixtures` first")
        .with_sampling(SamplerParams {
            num_steps: 4,
            ..SamplerParams::default()
        })
        .build();

    let text_ids = Tensor::<B, 2, Int>::from_data(
        TensorData::new(vec![0_i32, 1, 2, 3], [1, TEXT_LEN]),
        &device,
    );
    let text_mask: Tensor<B, 2, Bool> =
        Tensor::<B, 2>::ones([1, TEXT_LEN], &device).greater_elem(0.0f32);

    c.bench_function("sample_4steps", |b| {
        b.iter(|| {
            let out = engine
                .sample(make_request(text_ids.clone(), text_mask.clone()))
                .expect("sampler failed");
            let _ = out.into_data();
        });
    });
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_encode_conditions,
    bench_forward_with_cond,
    bench_sample_4steps
);
criterion_main!(benches);
