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
//! use burn::tensor::Device;
//! use irodori_tts_burn::inference::InferenceBuilder;
//! use irodori_tts_burn::rf::SamplerParams;
//!
//! let device: Device = Default::default();
//! let engine = InferenceBuilder::new(device)
//!     .load_weights(Path::new("weights.safetensors"))?
//!     .with_default_sampling()
//!     .build();
//!
//! let latent = engine.sample(request);
//! ```

use burn::tensor::Device;
use std::marker::PhantomData;
use std::path::Path;

use burn::tensor::{Bool, DType, Int, Tensor};

#[cfg(feature = "lora")]
use crate::weights::load_model_with_lora;
use crate::{
    config::ModelConfig,
    error::Result,
    model::{
        AuxConditionInput, BlockDebugOutputs, EncodedCondition, InferenceOptimizedModel,
        TextToLatentRfDiT, WgslInferenceOptimizedModel,
        timestep_condition::{FixedEulerCondCache, supports_fixed_euler_params},
    },
    rf::{
        PreparedSamplingRequest, SamplerDiagnosticTrace, SamplerParams, SamplerWorkReport,
        SamplingRequest, sample_euler_rf_cfg, sample_euler_rf_cfg_reported,
        sample_euler_rf_cfg_wgsl_cached_diagnostic_prepared,
        sample_euler_rf_cfg_wgsl_cached_prepared,
        sample_euler_rf_cfg_wgsl_cached_reported_prepared,
    },
    weights::{load_model, load_model_exact_only},
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

/// WGPU weight residency policy selected at the engine type boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WgslWeightProfile {
    /// Preserve source weights and both measured QKV layouts for arbitrary
    /// supported output lengths.
    #[default]
    PortableFallback,
    /// Accept every supported frame count, keep every measured production
    /// layout, and release only logical QKV and w1/w3 sources unreachable from
    /// the WGSL production graph.
    ProductionPrepared,
    /// Accept batch-one text-only requests of at least 100 latent frames. The
    /// resulting B2/B1 CFG route uses prepared `wo`/`w2` layouts exclusively,
    /// so their learned source layouts can also be released.
    LongTextPreparedOnly,
    /// Accept exactly 112 latent frames and retain learned source weights, but
    /// release the unused long-sequence QKV+gate layout.
    Fixed112OneLayout,
    /// Accept exactly 112 latent frames, retain the measured packed/fused
    /// layouts, and release learned projections unused by that route.
    Fixed112PackedOnly,
}

impl WgslWeightProfile {
    const fn fixed_frames(self) -> Option<usize> {
        match self {
            Self::PortableFallback | Self::ProductionPrepared | Self::LongTextPreparedOnly => None,
            Self::Fixed112OneLayout | Self::Fixed112PackedOnly => Some(112),
        }
    }

    /// Whether the profile's irreversible weight layout still covers this
    /// request class. Tensor shape/value validation remains a separate step.
    pub const fn admits_request_class(
        self,
        batch: usize,
        frames: usize,
        has_speaker: bool,
        has_caption: bool,
    ) -> bool {
        if batch == 0 || frames == 0 {
            return false;
        }
        match self {
            Self::PortableFallback | Self::ProductionPrepared => true,
            Self::LongTextPreparedOnly => {
                batch == 1 && frames >= 100 && !has_speaker && !has_caption
            }
            Self::Fixed112OneLayout | Self::Fixed112PackedOnly => frames == 112,
        }
    }
}

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
pub struct InferenceBuilder<S: BuilderState> {
    device: Device,
    model: Option<TextToLatentRfDiT>,
    config: Option<ModelConfig>,
    params: Option<SamplerParams>,
    _state: PhantomData<S>,
}

impl InferenceBuilder<Unconfigured> {
    /// Create a new builder targeting `device`.
    pub fn new(device: Device) -> Self {
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
    pub fn load_weights(self, path: impl AsRef<Path>) -> Result<InferenceBuilder<Loaded>> {
        let (model, config) = load_model(path.as_ref(), &self.device)?;
        Ok(InferenceBuilder {
            device: self.device,
            model: Some(model),
            config: Some(config),
            params: None,
            _state: PhantomData,
        })
    }

    /// Load weights with an explicit host checkpoint reader.
    pub fn load_weights_with_loader(
        self,
        path: impl AsRef<Path>,
        loader: crate::ModelCheckpointLoader,
    ) -> Result<InferenceBuilder<Loaded>> {
        let (model, config) =
            crate::weights::load_model_with_loader(path.as_ref(), &self.device, loader)?;
        Ok(InferenceBuilder {
            device: self.device,
            model: Some(model),
            config: Some(config),
            params: None,
            _state: PhantomData,
        })
    }

    /// Load weights while casting every floating-point checkpoint tensor to
    /// the requested dtype before installing it in the module.
    ///
    /// Callers must configure `self`'s device to the same dtype first. This
    /// explicit method keeps reduced precision out of the default production
    /// load path.
    pub fn load_weights_with_float_dtype(
        self,
        path: impl AsRef<Path>,
        float_dtype: burn::tensor::DType,
    ) -> Result<InferenceBuilder<Loaded>> {
        let (model, config) =
            crate::weights::load_model_with_float_dtype(path.as_ref(), &self.device, float_dtype)?;
        Ok(InferenceBuilder {
            device: self.device,
            model: Some(model),
            config: Some(config),
            params: None,
            _state: PhantomData,
        })
    }

    /// Load reduced-precision weights with an explicit host checkpoint reader.
    ///
    /// This exists so startup campaigns and embedders can select the portable
    /// header-indexed reader without changing the ordinary inference graph.
    pub fn load_weights_with_float_dtype_and_loader(
        self,
        path: impl AsRef<Path>,
        float_dtype: burn::tensor::DType,
        loader: crate::ModelCheckpointLoader,
    ) -> Result<InferenceBuilder<Loaded>> {
        let (model, config) = crate::weights::load_model_with_float_dtype_and_loader(
            path.as_ref(),
            &self.device,
            float_dtype,
            loader,
        )?;
        Ok(InferenceBuilder {
            device: self.device,
            model: Some(model),
            config: Some(config),
            params: None,
            _state: PhantomData,
        })
    }

    /// Load an exact-geometry inference model without duration-predictor
    /// weights.
    ///
    /// This residency profile is intended for requests whose frame count is
    /// supplied by the caller. Learned duration prediction is unavailable on
    /// the resulting engine; use [`Self::load_weights`] for predictive
    /// sessions.
    pub fn load_weights_exact_only(
        self,
        path: impl AsRef<Path>,
    ) -> Result<InferenceBuilder<Loaded>> {
        let (model, config) = load_model_exact_only(path.as_ref(), &self.device)?;
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
    #[cfg(feature = "lora")]
    pub fn load_weights_with_adapter(
        self,
        path: impl AsRef<Path>,
        adapter_dir: impl AsRef<Path>,
    ) -> Result<InferenceBuilder<Loaded>> {
        let (model, config) =
            load_model_with_lora(path.as_ref(), Some(adapter_dir.as_ref()), &self.device)?;
        Ok(InferenceBuilder {
            device: self.device,
            model: Some(model),
            config: Some(config),
            params: None,
            _state: PhantomData,
        })
    }
}

impl InferenceBuilder<Loaded> {
    /// Return the model configuration read from the checkpoint.
    pub fn model_config(&self) -> &ModelConfig {
        self.config
            .as_ref()
            .expect("config is always Some in Loaded state")
    }

    /// Set custom sampling parameters and advance to [`Ready`].
    pub fn with_sampling(self, params: SamplerParams) -> InferenceBuilder<Ready> {
        InferenceBuilder {
            device: self.device,
            model: self.model,
            config: self.config,
            params: Some(params),
            _state: PhantomData,
        }
    }

    /// Use the default [`SamplerParams`] and advance to [`Ready`].
    pub fn with_default_sampling(self) -> InferenceBuilder<Ready> {
        self.with_sampling(SamplerParams::default())
    }
}

impl InferenceBuilder<Ready> {
    /// Replace the sampling parameters before building.
    pub fn with_sampling(self, params: SamplerParams) -> Self {
        Self {
            params: Some(params),
            ..self
        }
    }

    /// Consume the builder and produce an [`InferenceEngine`].
    ///
    /// Fuses weight matrices (QKV, SwiGLU w1‖w3) for optimal kernel-launch
    /// efficiency during inference via [`InferenceOptimizedModel`].
    /// This is an inference-only optimisation that does not affect the
    /// serialised model record.
    ///
    /// # Panics
    ///
    /// Panics if internal invariants are violated (should be impossible via
    /// the type-state transitions).
    pub fn build(self) -> InferenceEngine {
        let model = self.model.expect("model is always Some in Ready state");
        InferenceEngine {
            model: InferenceOptimizedModel::from(model),
            config: self.config.expect("config is always Some in Ready state"),
            params: self.params.expect("params is always Some in Ready state"),
            device: self.device,
        }
    }
}

impl InferenceBuilder<Ready> {
    /// Build an engine whose DiT hot path uses the measured fused WGSL policy.
    ///
    /// This explicit transition is available only for raw f32 WGPU. Portable
    /// callers continue to use [`Self::build`].
    pub fn build_wgsl(mut self) -> WgslInferenceEngine {
        let model = WgslInferenceOptimizedModel::from(
            self.model
                .take()
                .expect("model is always Some in Ready state"),
        );
        self.finish_wgsl(model, WgslWeightProfile::PortableFallback)
    }

    /// Build a WGPU engine with an explicit weight-residency profile.
    pub fn build_wgsl_with_profile(
        mut self,
        profile: WgslWeightProfile,
    ) -> Result<WgslInferenceEngine> {
        let model = WgslInferenceOptimizedModel::from(
            self.model
                .take()
                .expect("model is always Some in Ready state"),
        );
        let model = match profile {
            WgslWeightProfile::PortableFallback => model,
            WgslWeightProfile::ProductionPrepared => model.release_production_sources()?,
            WgslWeightProfile::LongTextPreparedOnly => model.lock_long_text_prepared_only()?,
            WgslWeightProfile::Fixed112OneLayout => model.lock_fixed_112_profile(false)?,
            WgslWeightProfile::Fixed112PackedOnly => model.lock_fixed_112_profile(true)?,
        };
        let engine = self.finish_wgsl(model, profile);
        if matches!(
            profile,
            WgslWeightProfile::ProductionPrepared | WgslWeightProfile::LongTextPreparedOnly
        ) {
            // Source parameters were just made unreachable. Return their pages
            // to the backend here so callers do not need a hidden post-build
            // allocator ritual to realize the profile's physical VRAM saving.
            engine.device.memory_cleanup();
        }
        Ok(engine)
    }

    fn finish_wgsl(
        self,
        model: WgslInferenceOptimizedModel,
        profile: WgslWeightProfile,
    ) -> WgslInferenceEngine {
        let params = self.params.expect("params is always Some in Ready state");
        let fixed_euler_cond_cache = if supports_fixed_euler_params(&params) {
            model.try_build_fixed_euler_cond_cache().map(Box::new)
        } else {
            None
        };
        WgslInferenceEngine {
            model,
            config: self.config.expect("config is always Some in Ready state"),
            params,
            device: self.device,
            fixed_euler_cond_cache,
            weight_profile: profile,
        }
    }
}

// ---------------------------------------------------------------------------
// InferenceEngine
// ---------------------------------------------------------------------------

/// A fully configured inference engine produced by [`InferenceBuilder`].
///
/// Wraps an [`InferenceOptimizedModel`] — the model is guaranteed to have
/// fused weight matrices for branch-free inference.
pub struct InferenceEngine {
    model: InferenceOptimizedModel,
    config: ModelConfig,
    params: SamplerParams,
    device: Device,
}

/// Fully configured f32 WGPU engine using production fused WGSL kernels.
pub struct WgslInferenceEngine {
    model: WgslInferenceOptimizedModel,
    config: ModelConfig,
    params: SamplerParams,
    device: Device,
    fixed_euler_cond_cache: Option<Box<FixedEulerCondCache>>,
    weight_profile: WgslWeightProfile,
}

/// Exact tensors for one diagnostic DiT forward. This bypasses the Euler
/// integrator so both runtimes can consume the same latent, condition, and
/// timestep without changing the production sampling API.
pub struct DiagnosticForwardInput {
    latent: Tensor<3>,
    timestep: Tensor<1>,
    condition: EncodedCondition,
}

impl DiagnosticForwardInput {
    pub fn new(latent: Tensor<3>, timestep: Tensor<1>, condition: EncodedCondition) -> Self {
        Self {
            latent,
            timestep,
            condition,
        }
    }
}

/// Retained production intermediates from one diagnostic DiT forward.
pub struct DiagnosticForwardTrace {
    pub output: Tensor<3>,
    pub after_input_projection: Tensor<3>,
    pub block_outputs: Vec<Tensor<3>>,
}

impl WgslInferenceEngine {
    fn validate_request_contract(&self, request: &SamplingRequest) -> crate::error::Result<()> {
        self.validate_sequence_length(request.sequence_length)?;
        let text_batch = request.text_ids.dims()[0];
        if self.weight_profile == WgslWeightProfile::LongTextPreparedOnly
            && (text_batch != 1 || request.sequence_length < 100)
        {
            return Err(crate::error::IrodoriError::Config(format!(
                "long-text prepared-only profile requires batch-one input and at least 100 latent frames before condition compaction; got batch={text_batch}, frames={}",
                request.sequence_length,
            )));
        }
        Ok(())
    }

    fn validate_prepared_request_contract(
        &self,
        request: &PreparedSamplingRequest,
    ) -> crate::error::Result<()> {
        self.validate_request_contract(&request.request)?;
        let batch = request.request.text_ids.dims()[0];
        if !self.weight_profile.admits_request_class(
            batch,
            request.sequence_length(),
            request.has_speaker_context(),
            request.has_caption_context(),
        ) {
            return Err(crate::error::IrodoriError::Config(format!(
                "WGSL weight profile {:?} rejects prepared batch={batch}, frames={}, speaker={}, caption={}",
                self.weight_profile,
                request.sequence_length(),
                request.has_speaker_context(),
                request.has_caption_context(),
            )));
        }
        Ok(())
    }

    fn validate_sequence_length(&self, sequence_length: usize) -> crate::error::Result<()> {
        if self
            .weight_profile
            .fixed_frames()
            .is_some_and(|frames| frames != sequence_length)
        {
            return Err(crate::error::IrodoriError::Config(format!(
                "fixed-112 WGPU profile rejects {sequence_length} latent frames"
            )));
        }
        Ok(())
    }

    /// Run rectified-flow sampling through the WGSL execution policy.
    pub fn sample(
        &self,
        request: SamplingRequest,
    ) -> crate::error::Result<burn::tensor::Tensor<3>> {
        let request = self.prepare_sampling_request(request)?;
        self.sample_prepared(request)
    }

    /// Resolve every data-dependent request preparation step before warmup.
    pub fn prepare_sampling_request(
        &self,
        request: SamplingRequest,
    ) -> crate::error::Result<PreparedSamplingRequest> {
        self.validate_request_contract(&request)?;
        let request = request.prepare(self.model.patched_latent_dim())?;
        self.validate_prepared_request_contract(&request)?;
        Ok(request)
    }

    /// Sample a request whose data-dependent mask compaction completed before
    /// a possible compile-only warmup guard was entered.
    pub fn sample_prepared(
        &self,
        request: PreparedSamplingRequest,
    ) -> crate::error::Result<burn::tensor::Tensor<3>> {
        self.validate_prepared_request_contract(&request)?;
        sample_euler_rf_cfg_wgsl_cached_prepared(
            &self.model,
            request,
            &self.params,
            &self.device,
            self.fixed_euler_cond_cache.as_deref(),
        )
    }

    /// Run sampling while returning a machine-readable account of issued RF work.
    ///
    /// This explicit validation path leaves [`Self::sample`] unchanged.
    pub fn sample_with_work_report(
        &self,
        request: SamplingRequest,
    ) -> crate::error::Result<(burn::tensor::Tensor<3>, SamplerWorkReport)> {
        let request = self.prepare_sampling_request(request)?;
        sample_euler_rf_cfg_wgsl_cached_reported_prepared(
            &self.model,
            request,
            &self.params,
            &self.device,
            self.fixed_euler_cond_cache.as_deref(),
        )
    }

    /// Run a diagnostic-only production sample while retaining every whole-model
    /// output on the GPU for post-request localization.
    ///
    /// Retained tensors change allocator lifetime; timings from this call are
    /// not valid performance measurements.
    pub fn sample_with_diagnostic_trace(
        &self,
        request: SamplingRequest,
    ) -> crate::error::Result<(
        burn::tensor::Tensor<3>,
        SamplerWorkReport,
        SamplerDiagnosticTrace,
    )> {
        let request = self.prepare_sampling_request(request)?;
        sample_euler_rf_cfg_wgsl_cached_diagnostic_prepared(
            &self.model,
            request,
            &self.params,
            &self.device,
            self.fixed_euler_cond_cache.as_deref(),
        )
    }

    /// Run exactly one production DiT forward while retaining every block
    /// boundary. No Euler update, codec decode, or CPU readback is performed.
    /// Callers must exclude this retained-tensor path from latency results.
    pub fn diagnostic_forward(
        &self,
        input: DiagnosticForwardInput,
    ) -> crate::error::Result<DiagnosticForwardTrace> {
        let [batch, frames, latent_dim] = input.latent.dims();
        let [condition_batch, text_tokens, text_width] = input.condition.text_state.dims();
        if batch == 0
            || frames == 0
            || latent_dim != self.model.patched_latent_dim()
            || input.timestep.dims() != [batch]
            || condition_batch != batch
            || input.condition.text_mask.dims() != [batch, text_tokens]
            || text_width != self.config.text_dim
            || input.latent.dtype() != DType::F32
            || input.timestep.dtype() != DType::F32
            || input.condition.text_state.dtype() != DType::F32
        {
            return Err(crate::error::IrodoriError::Shape(format!(
                "diagnostic forward requires latent=[B,S,{}], timestep=[B], and condition batch B; got latent={:?}, timestep={:?}, text_state={:?}, text_mask={:?}",
                self.model.patched_latent_dim(),
                input.latent.dims(),
                input.timestep.dims(),
                input.condition.text_state.dims(),
                input.condition.text_mask.dims(),
            )));
        }
        self.validate_sequence_length(frames)?;
        if self.weight_profile == WgslWeightProfile::LongTextPreparedOnly
            && (frames < 100 || batch > 2 || input.condition.aux.is_some())
        {
            return Err(crate::error::IrodoriError::Config(
                "long-text prepared-only diagnostic forward admits only B1/B2 text conditions at 100+ frames"
                    .to_owned(),
            ));
        }
        if input.latent.device() != self.device
            || input.timestep.device() != self.device
            || input.condition.text_state.device() != self.device
            || input.condition.text_mask.device() != self.device
        {
            return Err(crate::error::IrodoriError::Config(
                "diagnostic forward tensors must reside on the engine device".to_owned(),
            ));
        }
        for (name, expected_width, context) in [
            (
                "speaker",
                self.config.speaker_dim,
                input.condition.aux.as_ref().and_then(|aux| aux.speaker()),
            ),
            (
                "caption",
                self.config.caption_dim,
                input.condition.aux.as_ref().and_then(|aux| aux.caption()),
            ),
        ] {
            if let Some((state, mask)) = context {
                let [context_batch, context_tokens, context_width] = state.dims();
                if context_batch != batch
                    || mask.dims() != [batch, context_tokens]
                    || expected_width != Some(context_width)
                    || state.dtype() != DType::F32
                    || state.device() != self.device
                    || mask.device() != self.device
                {
                    return Err(crate::error::IrodoriError::Shape(format!(
                        "diagnostic {name} condition must be FP32 [B,T,{expected_width:?}] with mask [B,T] on the engine device"
                    )));
                }
            }
        }
        let rope = self.model.precompute_latent_rope(frames, &self.device);
        let kv_caches = self.model.build_kv_caches(&input.condition, Some(frames));
        let (
            output,
            BlockDebugOutputs {
                after_in_proj,
                block_outputs,
            },
        ) = self.model.forward_with_cond_cached_debug(
            input.latent,
            input.timestep,
            &input.condition,
            None,
            Some(&kv_caches),
            &rope,
        );
        Ok(DiagnosticForwardTrace {
            output,
            after_input_projection: after_in_proj,
            block_outputs,
        })
    }

    /// Profile-only RF path that fuses single-signal Independent CFG combine
    /// with the Euler update while preserving the ordinary work manifest.
    #[cfg(feature = "profile")]
    pub fn sample_with_work_report_fused_cfg_euler(
        &self,
        request: SamplingRequest,
    ) -> crate::error::Result<(burn::tensor::Tensor<3>, SamplerWorkReport)> {
        if self.weight_profile == WgslWeightProfile::LongTextPreparedOnly {
            return Err(crate::error::IrodoriError::Config(
                "profile-only fused CFG/Euler differential is unavailable after long-text source release"
                    .to_owned(),
            ));
        }
        self.validate_request_contract(&request)?;
        crate::rf::sample_euler_rf_cfg_wgsl_cached_reported_fused_cfg_euler(
            &self.model,
            request,
            &self.params,
            &self.device,
            self.fixed_euler_cond_cache.as_deref(),
        )
    }

    pub fn with_sampling(mut self, params: SamplerParams) -> Self {
        self.fixed_euler_cond_cache = if supports_fixed_euler_params(&params) {
            match self.fixed_euler_cond_cache.take() {
                Some(cache) if self.model.fixed_euler_cond_cache_matches(cache.as_ref()) => {
                    Some(cache)
                }
                _ => self.model.try_build_fixed_euler_cond_cache().map(Box::new),
            }
        } else {
            None
        };
        self.params = params;
        self
    }

    pub fn model_config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn sampling_params(&self) -> &SamplerParams {
        &self.params
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Access the model only while every portable fallback weight/layout is
    /// resident. Profile-locked engines intentionally do not expose a direct
    /// forward escape hatch around their frame contract.
    pub fn portable_model(&self) -> Option<&WgslInferenceOptimizedModel> {
        (self.weight_profile == WgslWeightProfile::PortableFallback).then_some(&self.model)
    }

    pub fn encode_conditions(
        &self,
        text_input_ids: Tensor<2, Int>,
        text_mask: Tensor<2, Bool>,
        aux_input: AuxConditionInput,
    ) -> crate::error::Result<EncodedCondition> {
        self.model
            .encode_conditions(text_input_ids, text_mask, aux_input)
    }

    pub fn predict_duration_log_frames(
        &self,
        cond: &EncodedCondition,
        duration_features: Tensor<2>,
        has_speaker: Tensor<1, Bool>,
        has_caption: Tensor<1, Bool>,
    ) -> crate::error::Result<Tensor<1>> {
        self.model
            .predict_duration_log_frames(cond, duration_features, has_speaker, has_caption)
    }

    pub fn predict_duration_compact_no_aux(
        &self,
        cond: &EncodedCondition,
        duration_features: Tensor<2>,
        has_speaker: Tensor<1, Bool>,
        has_caption: Tensor<1, Bool>,
    ) -> crate::error::Result<Tensor<1>> {
        self.model.predict_duration_compact_no_aux_wgsl(
            cond,
            duration_features,
            has_speaker,
            has_caption,
        )
    }

    pub fn has_duration_predictor(&self) -> bool {
        self.model.has_duration_predictor()
    }

    pub const fn weight_profile(&self) -> WgslWeightProfile {
        self.weight_profile
    }
}

impl InferenceEngine {
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
        request: SamplingRequest,
    ) -> crate::error::Result<burn::tensor::Tensor<3>> {
        sample_euler_rf_cfg(&self.model, request, &self.params, &self.device)
    }

    /// Run sampling while returning a machine-readable account of issued RF work.
    ///
    /// This explicit validation path leaves [`Self::sample`] unchanged.
    pub fn sample_with_work_report(
        &self,
        request: SamplingRequest,
    ) -> crate::error::Result<(burn::tensor::Tensor<3>, SamplerWorkReport)> {
        sample_euler_rf_cfg_reported(&self.model, request, &self.params, &self.device)
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
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Access the underlying optimized model.
    pub fn model(&self) -> &InferenceOptimizedModel {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tiny_config() -> ModelConfig {
        crate::config::tiny_model_config()
    }

    fn make_loaded_builder() -> InferenceBuilder<Loaded> {
        let dev: Device = Default::default();
        let cfg = tiny_config();
        let model = TextToLatentRfDiT::new(&cfg, &dev);
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
        let dev: Device = Default::default();
        let builder = InferenceBuilder::<Unconfigured>::new(dev);
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

    #[test]
    fn wgsl_weight_profiles_make_their_frame_contract_explicit() {
        assert_eq!(WgslWeightProfile::PortableFallback.fixed_frames(), None);
        assert_eq!(WgslWeightProfile::ProductionPrepared.fixed_frames(), None);
        assert_eq!(WgslWeightProfile::LongTextPreparedOnly.fixed_frames(), None);
        assert_eq!(
            WgslWeightProfile::Fixed112OneLayout.fixed_frames(),
            Some(112)
        );
        assert_eq!(
            WgslWeightProfile::Fixed112PackedOnly.fixed_frames(),
            Some(112)
        );
    }

    #[test]
    fn long_text_prepared_profile_admits_only_source_free_routes() {
        let profile = WgslWeightProfile::LongTextPreparedOnly;
        for frames in [100, 112, 255, 333, 489, 685] {
            assert!(profile.admits_request_class(1, frames, false, false));
        }
        for rejected in [
            (0, 112, false, false),
            (1, 99, false, false),
            (2, 112, false, false),
            (1, 112, true, false),
            (1, 112, false, true),
        ] {
            assert!(!profile.admits_request_class(rejected.0, rejected.1, rejected.2, rejected.3));
        }
    }
}
