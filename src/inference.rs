//! Type-state builder for constructing an [`InferenceEngine`].
//!
//! The builder enforces the correct construction order **at compile time**
//! using phantom marker types:
//!
//! ```text
//! Unconfigured → (load_weights) → Loaded → (with_sampling) → Ready → (build) → InferenceEngine
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use std::path::Path;
//! use burn::backend::NdArray;
//! use irodori_tts_burn::inference::InferenceBuilder;
//! use irodori_tts_burn::rf::SamplerParams;
//!
//! let engine = InferenceBuilder::<NdArray, _>::new(Default::default())
//!     .load_weights(Path::new("weights.safetensors"))?
//!     .with_default_sampling()
//!     .build();
//!
//! let latent = engine.sample(request);
//! ```

use std::marker::PhantomData;
use std::path::Path;

use burn::tensor::backend::Backend;

use crate::{
    config::ModelConfig,
    error::Result,
    model::TextToLatentRfDiT,
    rf::{SamplerParams, SamplingRequest, sample_euler_rf_cfg},
    weights::{load_model, load_model_with_lora},
};

// ---------------------------------------------------------------------------
// Sealed trait — prevents external implementors of `BuilderState`
// ---------------------------------------------------------------------------

mod sealed {
    pub trait Sealed {}
}

/// Marker trait for the type-state positions of [`InferenceBuilder`].
///
/// This trait is **sealed** — it cannot be implemented outside this crate.
pub trait BuilderState: sealed::Sealed {}

/// The builder has no weights loaded yet.
#[derive(Debug)]
pub struct Unconfigured;

/// Weights have been loaded; a [`SamplerParams`] is still needed.
#[derive(Debug)]
pub struct Loaded;

/// Weights and sampling parameters are both present; ready to [`build`](InferenceBuilder::build).
#[derive(Debug)]
pub struct Ready;

impl sealed::Sealed for Unconfigured {}
impl sealed::Sealed for Loaded {}
impl sealed::Sealed for Ready {}

impl BuilderState for Unconfigured {}
impl BuilderState for Loaded {}
impl BuilderState for Ready {}

// ---------------------------------------------------------------------------
// InferenceBuilder
// ---------------------------------------------------------------------------

/// A type-state builder for constructing an [`InferenceEngine`].
///
/// Each method that advances the state consumes `self` and returns a new
/// `InferenceBuilder` at the next state, making it impossible to call
/// methods out of order.
pub struct InferenceBuilder<B: Backend, S: BuilderState> {
    device: B::Device,
    model: Option<TextToLatentRfDiT<B>>,
    config: Option<ModelConfig>,
    params: Option<SamplerParams>,
    _state: PhantomData<S>,
}

impl<B: Backend> InferenceBuilder<B, Unconfigured> {
    /// Create a new builder targeting `device`.
    pub fn new(device: B::Device) -> Self {
        Self {
            device,
            model: None,
            config: None,
            params: None,
            _state: PhantomData,
        }
    }

    /// Load model weights from a safetensors checkpoint.
    ///
    /// Reads the `config_json` metadata embedded in the checkpoint and
    /// advances the builder to the [`Loaded`] state.
    pub fn load_weights(self, path: impl AsRef<Path>) -> Result<InferenceBuilder<B, Loaded>> {
        let (model, config) = load_model::<B>(path.as_ref(), &self.device)?;
        Ok(InferenceBuilder {
            device: self.device,
            model: Some(model),
            config: Some(config),
            params: None,
            _state: PhantomData,
        })
    }

    /// Load model weights and merge a PEFT LoRA adapter.
    ///
    /// `adapter_dir` must contain `adapter_config.json` and
    /// `adapter_model.safetensors` (or `adapter_model.bin`).
    /// The LoRA delta is merged into the base weights at load time
    /// so inference is transparent.
    pub fn load_weights_with_adapter(
        self,
        path: impl AsRef<Path>,
        adapter_dir: impl AsRef<Path>,
    ) -> Result<InferenceBuilder<B, Loaded>> {
        let (model, config) =
            load_model_with_lora::<B>(path.as_ref(), Some(adapter_dir.as_ref()), &self.device)?;
        Ok(InferenceBuilder {
            device: self.device,
            model: Some(model),
            config: Some(config),
            params: None,
            _state: PhantomData,
        })
    }
}

impl<B: Backend> InferenceBuilder<B, Loaded> {
    /// Return the model configuration read from the checkpoint.
    pub fn model_config(&self) -> &ModelConfig {
        self.config
            .as_ref()
            .expect("config is always Some in Loaded state")
    }

    /// Set custom sampling parameters and advance to [`Ready`].
    pub fn with_sampling(self, params: SamplerParams) -> InferenceBuilder<B, Ready> {
        InferenceBuilder {
            device: self.device,
            model: self.model,
            config: self.config,
            params: Some(params),
            _state: PhantomData,
        }
    }

    /// Use the default [`SamplerParams`] and advance to [`Ready`].
    pub fn with_default_sampling(self) -> InferenceBuilder<B, Ready> {
        self.with_sampling(SamplerParams::default())
    }
}

impl<B: Backend> InferenceBuilder<B, Ready> {
    /// Replace the sampling parameters before building.
    pub fn with_sampling(self, params: SamplerParams) -> Self {
        Self {
            params: Some(params),
            ..self
        }
    }

    /// Consume the builder and produce an [`InferenceEngine`].
    ///
    /// # Panics
    ///
    /// Panics if internal invariants are violated (should be impossible via
    /// the type-state transitions).
    pub fn build(self) -> InferenceEngine<B> {
        InferenceEngine {
            model: self.model.expect("model is always Some in Ready state"),
            config: self.config.expect("config is always Some in Ready state"),
            params: self.params.expect("params is always Some in Ready state"),
            device: self.device,
        }
    }
}

// ---------------------------------------------------------------------------
// InferenceEngine
// ---------------------------------------------------------------------------

/// A fully configured inference engine produced by [`InferenceBuilder`].
pub struct InferenceEngine<B: Backend> {
    model: TextToLatentRfDiT<B>,
    config: ModelConfig,
    params: SamplerParams,
    device: B::Device,
}

impl<B: Backend> InferenceEngine<B> {
    /// Run the rectified-flow Euler sampler with classifier-free guidance.
    ///
    /// Returns the denoised latent: `[batch, sequence_length, patched_latent_dim]`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::IrodoriError::Config`] if the sampling parameters
    /// are invalid (e.g. `num_steps == 0` or Joint CFG with mismatched scales).
    pub fn sample(
        &self,
        request: SamplingRequest<B>,
    ) -> crate::error::Result<burn::tensor::Tensor<B, 3>> {
        sample_euler_rf_cfg(&self.model, request, &self.params, &self.device)
    }

    /// Replace the sampling parameters (e.g., to change `num_steps` or CFG scales)
    /// and return a new engine with the same loaded model.
    pub fn with_sampling(self, params: SamplerParams) -> Self {
        Self { params, ..self }
    }

    /// The model configuration read from the checkpoint.
    pub fn model_config(&self) -> &ModelConfig {
        &self.config
    }

    /// The active sampling parameters.
    pub fn sampling_params(&self) -> &SamplerParams {
        &self.params
    }

    /// The device this engine runs on.
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Access the underlying model directly (e.g., for `encode_conditions`).
    pub fn model(&self) -> &TextToLatentRfDiT<B> {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    fn tiny_config() -> ModelConfig {
        crate::train::tiny_model_config()
    }

    fn make_loaded_builder() -> InferenceBuilder<B, Loaded> {
        let dev: <B as Backend>::Device = Default::default();
        let cfg = tiny_config();
        let model = TextToLatentRfDiT::<B>::new(&cfg, &dev);
        InferenceBuilder {
            device: dev,
            model: Some(model),
            config: Some(cfg),
            params: None,
            _state: PhantomData,
        }
    }

    #[test]
    fn builder_new_creates_unconfigured() {
        let dev: <B as Backend>::Device = Default::default();
        let builder = InferenceBuilder::<B, Unconfigured>::new(dev);
        assert!(builder.model.is_none());
        assert!(builder.config.is_none());
        assert!(builder.params.is_none());
    }

    #[test]
    fn loaded_state_provides_model_config() {
        let builder = make_loaded_builder();
        let cfg = builder.model_config();
        assert!(cfg.model_dim > 0);
    }

    #[test]
    fn with_default_sampling_transitions_to_ready() {
        let builder = make_loaded_builder();
        let ready = builder.with_default_sampling();
        assert!(ready.params.is_some());
        let params = ready.params.as_ref().unwrap();
        assert!(params.num_steps > 0);
    }

    #[test]
    fn with_custom_sampling_transitions_to_ready() {
        let builder = make_loaded_builder();
        let params = SamplerParams {
            num_steps: 10,
            ..SamplerParams::default()
        };
        let ready = builder.with_sampling(params);
        assert_eq!(ready.params.as_ref().unwrap().num_steps, 10);
    }

    #[test]
    fn ready_can_replace_sampling_params() {
        let builder = make_loaded_builder();
        let ready = builder.with_default_sampling();
        let old_steps = ready.params.as_ref().unwrap().num_steps;
        let ready = ready.with_sampling(SamplerParams {
            num_steps: old_steps + 5,
            ..SamplerParams::default()
        });
        assert_eq!(ready.params.as_ref().unwrap().num_steps, old_steps + 5);
    }

    #[test]
    fn build_produces_engine_with_correct_accessors() {
        let builder = make_loaded_builder();
        let engine = builder.with_default_sampling().build();
        assert!(engine.model_config().model_dim > 0);
        assert!(engine.sampling_params().num_steps > 0);
    }

    #[test]
    fn engine_with_sampling_replaces_params() {
        let builder = make_loaded_builder();
        let engine = builder.with_default_sampling().build();
        let new_engine = engine.with_sampling(SamplerParams {
            num_steps: 7,
            ..SamplerParams::default()
        });
        assert_eq!(new_engine.sampling_params().num_steps, 7);
    }
}
