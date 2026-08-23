//! Euler sampler over the RF ODE with classifier-free guidance (CFG).

use std::collections::VecDeque;

use burn::tensor::Device;
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};

use crate::{
    config::{CfgGuidanceMode, SamplerMethod},
    model::{
        EncodedCondition, InferenceOptimizedModel, WgslInferenceOptimizedModel,
        condition::{AuxConditionInput, AuxConditionState},
        rope::RopeFreqs,
        timestep_condition::{
            ModelGeneration, PreparedEulerCondCache, PreparedEulerCondition,
            reference_linear_schedule, supports_prepared_euler_params,
        },
        wgsl::TextOnlyCfgCacheProof,
    },
    nvtx_range,
};

use super::kv_scaling::scale_speaker_kv_cache;
use super::math::temporal_score_rescale;
use super::params::{PreparedSamplingRequest, SamplerParams, SamplingRequest};

use crate::model::attention::{CondKvCache, TextCfgKvCachePair};

/// Conditioning geometry observed by the RF sampler.
///
/// `joint_axis` is the latent sequence plus every retained context sequence.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ConditioningGeometry {
    pub batch_rows: usize,
    pub latent_sequence: usize,
    pub latent_dim: usize,
    pub text_tokens: usize,
    pub speaker_tokens: Option<usize>,
    pub caption_tokens: Option<usize>,
    pub joint_axis: usize,
}

/// Conditioning signal that remained eligible for CFG after masks were resolved.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditioningSignal {
    Text,
    Speaker,
    Caption,
}

/// Position of a whole-model evaluation within an ODE step.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplerForwardEvaluation {
    Primary,
    HeunCorrector,
}

/// Condition bundle used by a whole-model evaluation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplerForwardLane {
    BatchedIndependent,
    Conditional,
    JointUnconditional,
    AlternatingUnconditional,
}

/// One actual whole-model forward issued by the sampler.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct SamplerForwardWork {
    pub step_index: usize,
    pub evaluation: SamplerForwardEvaluation,
    pub timestep_f32_bits: u32,
    pub cfg_active: bool,
    pub lane: SamplerForwardLane,
    pub batch_rows: usize,
    pub latent_sequence: usize,
    pub latent_dim: usize,
    pub text_tokens: usize,
    pub speaker_tokens: Option<usize>,
    pub caption_tokens: Option<usize>,
    pub joint_axis: usize,
    pub context_kv_layers: usize,
    pub fixed_cond_lookup_attempted: bool,
    pub fixed_cond_lookup_hit: bool,
    pub precomputed_cond_forward_used: bool,
    pub precomputed_adaln_used: bool,
}

/// Context-K/V work selected for one sampling request.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ContextKvWorkReport {
    pub enabled: bool,
    pub derived_text_cfg_pair_used: bool,
    pub conditional_layers: usize,
    pub batched_cfg_layers: usize,
}

/// Timestep-condition cache work selected for one sampling request.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct FixedTimestepConditionWorkReport {
    pub engine_cache_supplied: bool,
    pub request_selected: bool,
    pub lookup_attempts: usize,
    pub lookup_hits: usize,
    pub precomputed_forward_hits: usize,
    pub precomputed_adaln_hits: usize,
    pub ordinary_cond_forwards: usize,
}

/// Machine-readable account of the work actually issued by one RF sample.
///
/// The normal sampler API does not construct this report. It is available via
/// the explicit reported engine path used by precision validation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct SamplerWorkReport {
    pub schema_version: u32,
    pub method: SamplerMethod,
    pub guidance_mode: CfgGuidanceMode,
    pub num_steps: usize,
    pub schedule_f32_bits: Vec<u32>,
    pub requested: ConditioningGeometry,
    pub compacted: ConditioningGeometry,
    pub encoded: ConditioningGeometry,
    pub conditioned_text_mask_all_valid: bool,
    pub enabled_cfg: Vec<ConditioningSignal>,
    pub has_speaker_context: bool,
    pub has_caption_context: bool,
    pub context_kv: ContextKvWorkReport,
    pub fixed_timestep_condition: FixedTimestepConditionWorkReport,
    pub model_layers: usize,
    pub whole_model_forwards: usize,
    pub model_block_calls: usize,
    pub forwards: Vec<SamplerForwardWork>,
}

impl SamplerWorkReport {
    fn new(params: &SamplerParams) -> Self {
        Self {
            schema_version: 1,
            method: params.method,
            guidance_mode: params.guidance.mode,
            num_steps: params.num_steps,
            schedule_f32_bits: Vec::with_capacity(params.num_steps + 1),
            requested: ConditioningGeometry::default(),
            compacted: ConditioningGeometry::default(),
            encoded: ConditioningGeometry::default(),
            conditioned_text_mask_all_valid: false,
            enabled_cfg: Vec::new(),
            has_speaker_context: false,
            has_caption_context: false,
            context_kv: ContextKvWorkReport::default(),
            fixed_timestep_condition: FixedTimestepConditionWorkReport::default(),
            model_layers: 0,
            whole_model_forwards: 0,
            model_block_calls: 0,
            forwards: Vec::new(),
        }
    }

    /// Total batch rows processed across all whole-model forwards.
    pub fn effective_model_rows(&self) -> usize {
        self.forwards.iter().map(|forward| forward.batch_rows).sum()
    }
}

trait SamplerWorkRecorder {
    fn report(&mut self) -> Option<&mut SamplerWorkReport>;

    #[inline(always)]
    fn record_forward_input(
        &mut self,
        _meta: ForwardWorkMeta,
        _input: &Tensor<3>,
        _condition: &EncodedCondition,
    ) {
    }

    #[inline(always)]
    fn record_forward_output(&mut self, _meta: ForwardWorkMeta, _output: &Tensor<3>) {}
}

struct NoSamplerWorkReport;

impl SamplerWorkRecorder for NoSamplerWorkReport {
    #[inline(always)]
    fn report(&mut self) -> Option<&mut SamplerWorkReport> {
        None
    }
}

impl SamplerWorkRecorder for SamplerWorkReport {
    #[inline(always)]
    fn report(&mut self) -> Option<&mut SamplerWorkReport> {
        Some(self)
    }
}

/// One retained production-model output from a diagnostic-only RF request.
///
/// Keeping the GPU tensor alive changes allocator lifetime, so callers must
/// never treat a request carrying this trace as a latency measurement.
pub struct SamplerDiagnosticForward {
    pub ordinal: usize,
    pub step_index: usize,
    pub timestep_f32_bits: u32,
    pub batch_rows: usize,
    pub input: Tensor<3>,
    /// Exact encoded condition consumed by this forward, including CFG rows.
    pub condition: EncodedCondition,
    pub output: Tensor<3>,
}

/// Diagnostic-only production RF trace, paired with [`SamplerWorkReport`].
#[derive(Default)]
pub struct SamplerDiagnosticTrace {
    pub forwards: Vec<SamplerDiagnosticForward>,
}

struct SamplerDiagnosticRecorder {
    report: SamplerWorkReport,
    trace: SamplerDiagnosticTrace,
    pending_input: Option<(ForwardWorkMeta, Tensor<3>, EncodedCondition)>,
}

impl SamplerWorkRecorder for SamplerDiagnosticRecorder {
    fn report(&mut self) -> Option<&mut SamplerWorkReport> {
        Some(&mut self.report)
    }

    fn record_forward_input(
        &mut self,
        meta: ForwardWorkMeta,
        input: &Tensor<3>,
        condition: &EncodedCondition,
    ) {
        assert!(
            self.pending_input.is_none(),
            "diagnostic forward input must be paired before the next forward"
        );
        self.pending_input = Some((meta, input.clone(), condition.clone()));
    }

    fn record_forward_output(&mut self, meta: ForwardWorkMeta, output: &Tensor<3>) {
        let (input_meta, input, condition) = self
            .pending_input
            .take()
            .expect("diagnostic forward output must have a paired input");
        assert_eq!(input_meta.step_index, meta.step_index);
        assert_eq!(input_meta.timestep_f32_bits, meta.timestep_f32_bits);
        assert_eq!(input.dims()[0], output.dims()[0]);
        self.trace.forwards.push(SamplerDiagnosticForward {
            ordinal: self.trace.forwards.len(),
            step_index: meta.step_index,
            timestep_f32_bits: meta.timestep_f32_bits,
            batch_rows: output.dims()[0],
            input,
            condition,
            output: output.clone(),
        });
    }
}

#[derive(Clone, Copy)]
struct ForwardWorkMeta {
    step_index: usize,
    evaluation: SamplerForwardEvaluation,
    timestep_f32_bits: u32,
    cfg_active: bool,
    lane: SamplerForwardLane,
    fixed_cond_lookup_attempted: bool,
}

fn request_geometry(request: &SamplingRequest, latent_dim: usize) -> ConditioningGeometry {
    let [batch_rows, text_tokens] = request.text_ids.dims();
    let speaker_tokens = request.ref_latent.as_ref().map(|state| state.dims()[1]);
    let caption_tokens = request.caption_ids.as_ref().map(|ids| ids.dims()[1]);
    ConditioningGeometry {
        batch_rows,
        latent_sequence: request.sequence_length,
        latent_dim,
        text_tokens,
        speaker_tokens,
        caption_tokens,
        joint_axis: request.sequence_length
            + text_tokens
            + speaker_tokens.unwrap_or(0)
            + caption_tokens.unwrap_or(0),
    }
}

fn requested_geometry(
    request: &PreparedSamplingRequest,
    latent_dim: usize,
) -> ConditioningGeometry {
    let batch_rows = request.request.text_ids.dims()[0];
    let latent_sequence = request.request.sequence_length;
    ConditioningGeometry {
        batch_rows,
        latent_sequence,
        latent_dim,
        text_tokens: request.requested_text_tokens,
        speaker_tokens: request.requested_speaker_tokens,
        caption_tokens: request.requested_caption_tokens,
        joint_axis: latent_sequence
            + request.requested_text_tokens
            + request.requested_speaker_tokens.unwrap_or(0)
            + request.requested_caption_tokens.unwrap_or(0),
    }
}

fn encoded_geometry(
    cond: &EncodedCondition,
    latent_sequence: usize,
    latent_dim: usize,
) -> ConditioningGeometry {
    let [batch_rows, text_tokens, _] = cond.text_state.dims();
    let speaker_tokens = cond
        .aux
        .as_ref()
        .and_then(AuxConditionState::speaker)
        .map(|(state, _)| state.dims()[1]);
    let caption_tokens = cond
        .aux
        .as_ref()
        .and_then(AuxConditionState::caption)
        .map(|(state, _)| state.dims()[1]);
    ConditioningGeometry {
        batch_rows,
        latent_sequence,
        latent_dim,
        text_tokens,
        speaker_tokens,
        caption_tokens,
        joint_axis: latent_sequence
            + text_tokens
            + speaker_tokens.unwrap_or(0)
            + caption_tokens.unwrap_or(0),
    }
}

/// Apply the same speaker K/V multiplier to every cache participating in the
/// current CFG strategy.
///
/// In Alternating CFG, the conditional and single-signal-unconditional passes
/// must see the same speaker scaling whenever speaker context is retained.
fn scale_speaker_cache_set<const N: usize>(
    caches: [&mut Option<Vec<CondKvCache>>; N],
    scale: f32,
    max_layers: Option<usize>,
) {
    for cache in caches {
        if let Some(current) = cache.take() {
            *cache = Some(scale_speaker_kv_cache(current, scale, max_layers));
        }
    }
}

// ---------------------------------------------------------------------------
// Private CFG helpers
// ---------------------------------------------------------------------------

/// Which conditioning signal to drop for a particular CFG bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
enum CfgName {
    Text,
    Speaker,
    Caption,
}

fn cfg_scale_for(name: &CfgName, text: f32, speaker: f32, caption: f32) -> f32 {
    match name {
        CfgName::Text => text,
        CfgName::Speaker => speaker,
        CfgName::Caption => caption,
    }
}

#[allow(clippy::too_many_arguments)]
fn supports_prepared_euler_cond_cache_request(
    params: &SamplerParams,
    batch_size: usize,
    cfg_batch_mult: usize,
    enabled_cfg: &[CfgName],
    has_model_generation: bool,
) -> bool {
    supports_prepared_euler_params(params)
        && batch_size == 1
        && (1..=3).contains(&cfg_batch_mult)
        && enabled_cfg.len() < 3
        && cfg_batch_mult == 1 + enabled_cfg.len()
        && has_model_generation
}

/// Build an `EncodedCondition` that nullifies only the text signal.
fn make_text_uncond(cond: &EncodedCondition, uncond: &EncodedCondition) -> EncodedCondition {
    EncodedCondition {
        text_state: uncond.text_state.clone(),
        text_mask: uncond.text_mask.clone(),
        aux: cond.aux.clone(),
    }
}

/// Build an `EncodedCondition` that nullifies only speaker conditioning.
fn make_speaker_uncond(cond: &EncodedCondition, device: &Device) -> EncodedCondition {
    EncodedCondition {
        text_state: cond.text_state.clone(),
        text_mask: cond.text_mask.clone(),
        aux: cond
            .aux
            .as_ref()
            .map(|aux| aux.speaker_unconditional(device)),
    }
}

/// Build an `EncodedCondition` that nullifies only caption conditioning.
fn make_caption_uncond(cond: &EncodedCondition, device: &Device) -> EncodedCondition {
    EncodedCondition {
        text_state: cond.text_state.clone(),
        text_mask: cond.text_mask.clone(),
        aux: cond
            .aux
            .as_ref()
            .map(|aux| aux.caption_unconditional(device)),
    }
}

/// Build an `EncodedCondition` that nullifies a single named signal.
fn make_single_uncond(
    name: &CfgName,
    cond: &EncodedCondition,
    uncond: &EncodedCondition,
    device: &Device,
) -> EncodedCondition {
    match name {
        CfgName::Text => make_text_uncond(cond, uncond),
        CfgName::Speaker => make_speaker_uncond(cond, device),
        CfgName::Caption => make_caption_uncond(cond, device),
    }
}

// ---------------------------------------------------------------------------
// PLMS-4 Adams-Bashforth helper
// ---------------------------------------------------------------------------

/// Compute the Adams-Bashforth extrapolated effective velocity from history.
///
/// `history` stores velocity estimates newest-first: `[v_n, v_{n-1}, ...]`.
/// Coefficients are for constant step size (uniform schedule).
///
/// | len | order | coefficients |
/// |-----|-------|----------------------------------------------|
/// |  1  | AB-1  | [1]                                          |
/// |  2  | AB-2  | [3/2, −1/2]                                  |
/// |  3  | AB-3  | [23/12, −16/12, 5/12]                        |
/// | ≥4  | AB-4  | [55/24, −59/24, 37/24, −9/24]                |
fn ab_extrapolate(history: &VecDeque<Tensor<3>>) -> Tensor<3> {
    debug_assert!(!history.is_empty(), "PLMS history must not be empty");
    match history.len() {
        1 => history[0].clone(),
        2 => history[0].clone() * 1.5 + history[1].clone() * -0.5_f32,
        3 => {
            history[0].clone() * (23.0_f32 / 12.0)
                + history[1].clone() * (-16.0_f32 / 12.0)
                + history[2].clone() * (5.0_f32 / 12.0)
        }
        _ => {
            history[0].clone() * (55.0_f32 / 24.0)
                + history[1].clone() * (-59.0_f32 / 24.0)
                + history[2].clone() * (37.0_f32 / 24.0)
                + history[3].clone() * (-9.0_f32 / 24.0)
        }
    }
}

// ---------------------------------------------------------------------------
// Main sampler
// ---------------------------------------------------------------------------

/// Internal execution contract shared by the portable and WGSL model wrappers.
///
/// Keeping this trait private preserves the stable public sampler API while
/// allowing a measured backend-specific execution path to reuse the complete
/// CFG/KV-cache/ODE implementation.
trait SamplerModel {
    fn encode_conditions(
        &self,
        text_input_ids: burn::tensor::Tensor<2, burn::tensor::Int>,
        text_mask: burn::tensor::Tensor<2, burn::tensor::Bool>,
        aux_input: AuxConditionInput,
    ) -> crate::error::Result<EncodedCondition>;

    fn forward_with_cond_cached(
        &self,
        x_t: Tensor<3>,
        t: Tensor<1>,
        cond: &EncodedCondition,
        latent_mask: Option<burn::tensor::Tensor<2, burn::tensor::Bool>>,
        kv_caches: Option<&[CondKvCache]>,
        lat_rope: &RopeFreqs,
    ) -> Tensor<3>;

    fn model_generation(&self) -> Option<ModelGeneration> {
        None
    }

    fn try_forward_with_precomputed_cond_cached(
        &self,
        _x_t: Tensor<3>,
        _condition: PreparedEulerCondition,
        _cond: &EncodedCondition,
        _latent_mask: Option<burn::tensor::Tensor<2, burn::tensor::Bool>>,
        _kv_caches: Option<&[CondKvCache]>,
        _lat_rope: &RopeFreqs,
    ) -> Option<Tensor<3>> {
        None
    }

    fn precompute_latent_rope(&self, seq_lat: usize, device: &Device) -> RopeFreqs;

    fn build_kv_caches(&self, cond: &EncodedCondition, seq_lat: Option<usize>) -> Vec<CondKvCache>;

    fn try_build_text_cfg_kv_caches(
        &self,
        _cond: &EncodedCondition,
        _batched_cfg: &EncodedCondition,
        _seq_lat: usize,
        _proof: Option<&TextOnlyCfgCacheProof>,
    ) -> Option<TextCfgKvCachePair> {
        None
    }

    fn use_speaker_condition(&self) -> bool;

    fn use_caption_condition(&self) -> bool;

    fn patched_latent_dim(&self) -> usize;

    fn try_independent_cfg_euler_update(
        &self,
        _x_t: Tensor<3>,
        _velocities: Tensor<3>,
        _cfg_scale: f32,
        _dt: f32,
    ) -> Option<Tensor<3>> {
        None
    }
}

impl SamplerModel for InferenceOptimizedModel {
    fn encode_conditions(
        &self,
        text_input_ids: burn::tensor::Tensor<2, burn::tensor::Int>,
        text_mask: burn::tensor::Tensor<2, burn::tensor::Bool>,
        aux_input: AuxConditionInput,
    ) -> crate::error::Result<EncodedCondition> {
        InferenceOptimizedModel::encode_conditions(self, text_input_ids, text_mask, aux_input)
    }

    fn forward_with_cond_cached(
        &self,
        x_t: Tensor<3>,
        t: Tensor<1>,
        cond: &EncodedCondition,
        latent_mask: Option<burn::tensor::Tensor<2, burn::tensor::Bool>>,
        kv_caches: Option<&[CondKvCache]>,
        lat_rope: &RopeFreqs,
    ) -> Tensor<3> {
        InferenceOptimizedModel::forward_with_cond_cached(
            self,
            x_t,
            t,
            cond,
            latent_mask,
            kv_caches,
            lat_rope,
        )
    }

    fn precompute_latent_rope(&self, seq_lat: usize, device: &Device) -> RopeFreqs {
        InferenceOptimizedModel::precompute_latent_rope(self, seq_lat, device)
    }

    fn build_kv_caches(&self, cond: &EncodedCondition, seq_lat: Option<usize>) -> Vec<CondKvCache> {
        InferenceOptimizedModel::build_kv_caches(self, cond, seq_lat)
    }

    fn use_speaker_condition(&self) -> bool {
        InferenceOptimizedModel::use_speaker_condition(self)
    }

    fn use_caption_condition(&self) -> bool {
        InferenceOptimizedModel::use_caption_condition(self)
    }

    fn patched_latent_dim(&self) -> usize {
        InferenceOptimizedModel::patched_latent_dim(self)
    }
}

impl SamplerModel for WgslInferenceOptimizedModel {
    fn encode_conditions(
        &self,
        text_input_ids: burn::tensor::Tensor<2, burn::tensor::Int>,
        text_mask: burn::tensor::Tensor<2, burn::tensor::Bool>,
        aux_input: AuxConditionInput,
    ) -> crate::error::Result<EncodedCondition> {
        WgslInferenceOptimizedModel::encode_conditions(self, text_input_ids, text_mask, aux_input)
    }

    fn forward_with_cond_cached(
        &self,
        x_t: Tensor<3>,
        t: Tensor<1>,
        cond: &EncodedCondition,
        latent_mask: Option<burn::tensor::Tensor<2, burn::tensor::Bool>>,
        kv_caches: Option<&[CondKvCache]>,
        lat_rope: &RopeFreqs,
    ) -> Tensor<3> {
        WgslInferenceOptimizedModel::forward_with_cond_cached(
            self,
            x_t,
            t,
            cond,
            latent_mask,
            kv_caches,
            lat_rope,
        )
    }

    fn model_generation(&self) -> Option<ModelGeneration> {
        Some(WgslInferenceOptimizedModel::model_generation(self))
    }

    fn try_forward_with_precomputed_cond_cached(
        &self,
        x_t: Tensor<3>,
        condition: PreparedEulerCondition,
        cond: &EncodedCondition,
        latent_mask: Option<burn::tensor::Tensor<2, burn::tensor::Bool>>,
        kv_caches: Option<&[CondKvCache]>,
        lat_rope: &RopeFreqs,
    ) -> Option<Tensor<3>> {
        WgslInferenceOptimizedModel::try_forward_with_precomputed_cond_cached(
            self,
            x_t,
            condition.cond_embed,
            condition.adaln,
            cond,
            latent_mask,
            kv_caches,
            lat_rope,
        )
    }

    fn precompute_latent_rope(&self, seq_lat: usize, device: &Device) -> RopeFreqs {
        WgslInferenceOptimizedModel::precompute_latent_rope(self, seq_lat, device)
    }

    fn build_kv_caches(&self, cond: &EncodedCondition, seq_lat: Option<usize>) -> Vec<CondKvCache> {
        WgslInferenceOptimizedModel::build_kv_caches(self, cond, seq_lat)
    }

    fn try_build_text_cfg_kv_caches(
        &self,
        cond: &EncodedCondition,
        batched_cfg: &EncodedCondition,
        seq_lat: usize,
        proof: Option<&TextOnlyCfgCacheProof>,
    ) -> Option<(Vec<CondKvCache>, Vec<CondKvCache>)> {
        WgslInferenceOptimizedModel::try_build_text_cfg_kv_caches(
            self,
            cond,
            batched_cfg,
            seq_lat,
            proof,
        )
    }

    fn use_speaker_condition(&self) -> bool {
        WgslInferenceOptimizedModel::use_speaker_condition(self)
    }

    fn use_caption_condition(&self) -> bool {
        WgslInferenceOptimizedModel::use_caption_condition(self)
    }

    fn patched_latent_dim(&self) -> usize {
        WgslInferenceOptimizedModel::patched_latent_dim(self)
    }

    fn try_independent_cfg_euler_update(
        &self,
        x_t: Tensor<3>,
        velocities: Tensor<3>,
        cfg_scale: f32,
        dt: f32,
    ) -> Option<Tensor<3>> {
        let x_t = x_t.try_into_primitive::<crate::WgpuRaw>().ok()?;
        let velocities = velocities.try_into_primitive::<crate::WgpuRaw>().ok()?;
        crate::kernels::rf_cfg_euler_update::try_rf_independent_cfg_euler_update_wgsl(
            x_t, velocities, cfg_scale, dt,
        )
        .map(Tensor::from_primitive::<crate::WgpuRaw>)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum SamplerElementwisePolicy {
    #[default]
    Reference,
    #[cfg(feature = "profile")]
    FusedIndependentCfgEuler,
}

impl SamplerElementwisePolicy {
    fn fuses_independent_cfg_euler(self) -> bool {
        match self {
            Self::Reference => false,
            #[cfg(feature = "profile")]
            Self::FusedIndependentCfgEuler => true,
        }
    }
}

enum SamplerStepValue {
    Velocity(Tensor<3>),
    UpdatedLatent(Tensor<3>),
}

trait TimestepCondCache {
    fn step(
        &self,
        generation: ModelGeneration,
        index: usize,
        timestep_bits: u32,
        batch: usize,
        device: &Device,
    ) -> Option<PreparedEulerCondition>;
}

impl TimestepCondCache for PreparedEulerCondCache {
    fn step(
        &self,
        generation: ModelGeneration,
        index: usize,
        timestep_bits: u32,
        batch: usize,
        device: &Device,
    ) -> Option<PreparedEulerCondition> {
        PreparedEulerCondCache::step(self, generation, index, timestep_bits, batch, device)
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_sampler_model<M: SamplerModel, R: SamplerWorkRecorder>(
    model: &M,
    x_t: Tensor<3>,
    t: Tensor<1>,
    precomputed_cond: Option<PreparedEulerCondition>,
    cond: &EncodedCondition,
    latent_mask: Option<burn::tensor::Tensor<2, burn::tensor::Bool>>,
    kv_caches: Option<&[CondKvCache]>,
    lat_rope: &RopeFreqs,
    recorder: &mut R,
    meta: ForwardWorkMeta,
) -> Tensor<3> {
    let fixed_cond_lookup_hit = precomputed_cond.is_some();
    if let Some(report) = recorder.report() {
        let geometry = encoded_geometry(cond, x_t.dims()[1], x_t.dims()[2]);
        report.forwards.push(SamplerForwardWork {
            step_index: meta.step_index,
            evaluation: meta.evaluation,
            timestep_f32_bits: meta.timestep_f32_bits,
            cfg_active: meta.cfg_active,
            lane: meta.lane,
            batch_rows: x_t.dims()[0],
            latent_sequence: geometry.latent_sequence,
            latent_dim: geometry.latent_dim,
            text_tokens: geometry.text_tokens,
            speaker_tokens: geometry.speaker_tokens,
            caption_tokens: geometry.caption_tokens,
            joint_axis: geometry.joint_axis,
            context_kv_layers: kv_caches.map_or(0, |caches| caches.len()),
            fixed_cond_lookup_attempted: meta.fixed_cond_lookup_attempted,
            fixed_cond_lookup_hit,
            precomputed_cond_forward_used: false,
            precomputed_adaln_used: false,
        });
        if meta.fixed_cond_lookup_attempted {
            report.fixed_timestep_condition.lookup_attempts += 1;
        }
        if fixed_cond_lookup_hit {
            report.fixed_timestep_condition.lookup_hits += 1;
        }
    }

    // The default recorder compiles to a no-op. Only the explicit diagnostic
    // recorder clones and retains this tensor, keeping production lifetimes
    // and dispatches unchanged.
    recorder.record_forward_input(meta, &x_t, cond);

    if let Some(condition) = precomputed_cond {
        let precomputed_adaln_used = condition.adaln.is_some();
        if let Some(output) = model.try_forward_with_precomputed_cond_cached(
            x_t.clone(),
            condition,
            cond,
            latent_mask.clone(),
            kv_caches,
            lat_rope,
        ) {
            if let Some(report) = recorder.report() {
                let forward = report
                    .forwards
                    .last_mut()
                    .expect("reported forward was recorded before execution");
                forward.precomputed_cond_forward_used = true;
                forward.precomputed_adaln_used = precomputed_adaln_used;
                report.fixed_timestep_condition.precomputed_forward_hits += 1;
                report.fixed_timestep_condition.precomputed_adaln_hits +=
                    usize::from(precomputed_adaln_used);
            }
            recorder.record_forward_output(meta, &output);
            return output;
        }
    }
    if let Some(report) = recorder.report() {
        report.fixed_timestep_condition.ordinary_cond_forwards += 1;
    }
    let output = model.forward_with_cond_cached(x_t, t, cond, latent_mask, kv_caches, lat_rope);
    recorder.record_forward_output(meta, &output);
    output
}

///
/// Returns the denoised latent: `[batch, sequence_length, patched_latent_dim]`.
///
/// # Parameters
/// - `request.initial_noise`: supply pre-generated noise for reproducibility; if `None`
///   a standard Gaussian is sampled from burn's RNG.
///
/// # Errors
///
/// Returns [`IrodoriError::Config`] if `params` fails validation (e.g. `num_steps == 0`,
/// Joint mode with mismatched CFG scales).
pub fn sample_euler_rf_cfg(
    model: &InferenceOptimizedModel,
    request: SamplingRequest,
    params: &SamplerParams,
    device: &Device,
) -> crate::error::Result<Tensor<3>> {
    let mut recorder = NoSamplerWorkReport;
    let request = request.prepare(model.patched_latent_dim())?;
    sample_euler_rf_cfg_impl(
        model,
        request,
        params,
        device,
        None,
        &mut recorder,
        SamplerElementwisePolicy::Reference,
    )
}

pub(crate) fn sample_euler_rf_cfg_reported(
    model: &InferenceOptimizedModel,
    request: SamplingRequest,
    params: &SamplerParams,
    device: &Device,
) -> crate::error::Result<(Tensor<3>, SamplerWorkReport)> {
    let mut report = SamplerWorkReport::new(params);
    let request = request.prepare(model.patched_latent_dim())?;
    let output = sample_euler_rf_cfg_impl(
        model,
        request,
        params,
        device,
        None,
        &mut report,
        SamplerElementwisePolicy::Reference,
    )?;
    Ok((output, report))
}

/// Sample with the production WGSL execution policy on the raw f32 WGPU
/// backend.
pub fn sample_euler_rf_cfg_wgsl(
    model: &WgslInferenceOptimizedModel,
    request: SamplingRequest,
    params: &SamplerParams,
    device: &Device,
) -> crate::error::Result<Tensor<3>> {
    let mut recorder = NoSamplerWorkReport;
    let request = request.prepare(model.patched_latent_dim())?;
    sample_euler_rf_cfg_impl(
        model,
        request,
        params,
        device,
        None,
        &mut recorder,
        SamplerElementwisePolicy::Reference,
    )
}

pub(crate) fn sample_euler_rf_cfg_wgsl_cached_prepared(
    model: &WgslInferenceOptimizedModel,
    request: PreparedSamplingRequest,
    params: &SamplerParams,
    device: &Device,
    fixed_cond_cache: Option<&PreparedEulerCondCache>,
) -> crate::error::Result<Tensor<3>> {
    let fixed_cond_cache = fixed_cond_cache.map(|cache| cache as &dyn TimestepCondCache);
    let mut recorder = NoSamplerWorkReport;
    sample_euler_rf_cfg_impl(
        model,
        request,
        params,
        device,
        fixed_cond_cache,
        &mut recorder,
        SamplerElementwisePolicy::Reference,
    )
}

pub(crate) fn sample_euler_rf_cfg_wgsl_cached_reported_prepared(
    model: &WgslInferenceOptimizedModel,
    request: PreparedSamplingRequest,
    params: &SamplerParams,
    device: &Device,
    fixed_cond_cache: Option<&PreparedEulerCondCache>,
) -> crate::error::Result<(Tensor<3>, SamplerWorkReport)> {
    let fixed_cond_cache = fixed_cond_cache.map(|cache| cache as &dyn TimestepCondCache);
    let mut report = SamplerWorkReport::new(params);
    let output = sample_euler_rf_cfg_impl(
        model,
        request,
        params,
        device,
        fixed_cond_cache,
        &mut report,
        SamplerElementwisePolicy::Reference,
    )?;
    Ok((output, report))
}

pub(crate) fn sample_euler_rf_cfg_wgsl_cached_diagnostic_prepared(
    model: &WgslInferenceOptimizedModel,
    request: PreparedSamplingRequest,
    params: &SamplerParams,
    device: &Device,
    fixed_cond_cache: Option<&PreparedEulerCondCache>,
) -> crate::error::Result<(Tensor<3>, SamplerWorkReport, SamplerDiagnosticTrace)> {
    let fixed_cond_cache = fixed_cond_cache.map(|cache| cache as &dyn TimestepCondCache);
    let mut recorder = SamplerDiagnosticRecorder {
        report: SamplerWorkReport::new(params),
        trace: SamplerDiagnosticTrace::default(),
        pending_input: None,
    };
    let output = sample_euler_rf_cfg_impl(
        model,
        request,
        params,
        device,
        fixed_cond_cache,
        &mut recorder,
        SamplerElementwisePolicy::Reference,
    )?;
    Ok((output, recorder.report, recorder.trace))
}

#[cfg(feature = "profile")]
pub(crate) fn sample_euler_rf_cfg_wgsl_cached_reported_fused_cfg_euler(
    model: &WgslInferenceOptimizedModel,
    request: SamplingRequest,
    params: &SamplerParams,
    device: &Device,
    fixed_cond_cache: Option<&PreparedEulerCondCache>,
) -> crate::error::Result<(Tensor<3>, SamplerWorkReport)> {
    let fixed_cond_cache = fixed_cond_cache.map(|cache| cache as &dyn TimestepCondCache);
    let mut report = SamplerWorkReport::new(params);
    let request = request.prepare(model.patched_latent_dim())?;
    let output = sample_euler_rf_cfg_impl(
        model,
        request,
        params,
        device,
        fixed_cond_cache,
        &mut report,
        SamplerElementwisePolicy::FusedIndependentCfgEuler,
    )?;
    Ok((output, report))
}

fn sample_euler_rf_cfg_impl<M: SamplerModel, R: SamplerWorkRecorder>(
    model: &M,
    request: PreparedSamplingRequest,
    params: &SamplerParams,
    device: &Device,
    fixed_cond_cache: Option<&dyn TimestepCondCache>,
    recorder: &mut R,
    elementwise_policy: SamplerElementwisePolicy,
) -> crate::error::Result<Tensor<3>> {
    use crate::error::IrodoriError;

    params.validate()?;

    let latent_dim = model.patched_latent_dim();
    if let Some(report) = recorder.report() {
        report.requested = requested_geometry(&request, latent_dim);
        report.fixed_timestep_condition.engine_cache_supplied = fixed_cond_cache.is_some();
    }
    let PreparedSamplingRequest {
        request,
        requested_text_tokens: _,
        requested_speaker_tokens: _,
        requested_caption_tokens: _,
        conditioned_text_mask_all_valid,
        has_speaker_context,
        has_caption_context,
    } = request;
    if let Some(report) = recorder.report() {
        report.compacted = request_geometry(&request, latent_dim);
        report.conditioned_text_mask_all_valid = conditioned_text_mask_all_valid;
    }
    if request.ref_latent.is_some() && !model.use_speaker_condition() {
        return Err(IrodoriError::Config(
            "reference input was supplied to a model without speaker conditioning".to_string(),
        ));
    }
    if request.caption_ids.is_some() && !model.use_caption_condition() {
        return Err(IrodoriError::Config(
            "caption input was supplied to a model without caption conditioning".to_string(),
        ));
    }
    let batch_size = request.text_ids.dims()[0];

    // --- Initial noise ---
    let mut x_t = request.initial_noise.unwrap_or_else(|| {
        Tensor::random(
            [batch_size, request.sequence_length, latent_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        )
    });
    if let Some(k) = params.truncation_factor {
        x_t = x_t * k;
    }

    let g = &params.guidance;

    // Resolve effective CFG scales
    let cfg_scale_text = g.scale_text;
    let cfg_scale_caption = g.scale_caption;
    let cfg_scale_speaker = if model.use_speaker_condition() {
        g.scale_speaker
    } else {
        0.0
    };

    // --- Encode conditioned state once ---
    let aux_input = AuxConditionInput::try_from_request(
        request.ref_latent,
        request.ref_mask,
        request.caption_ids,
        request.caption_mask,
    )?;
    let cond = model.encode_conditions(request.text_ids, request.text_mask, aux_input)?;
    let uncond = cond.zeros_like(device);
    if let Some(report) = recorder.report() {
        report.encoded = encoded_geometry(&cond, request.sequence_length, latent_dim);
    }

    // Precompute RoPE tables for the latent sequence once — reused across all 40 × 3 forward passes.
    let lat_rope = model.precompute_latent_rope(request.sequence_length, device);

    // Which CFG signals are active?
    let has_text_cfg = cfg_scale_text > 0.0;
    // Speaker CFG is only meaningful when a reference audio was actually provided.
    let has_speaker_cfg = cfg_scale_speaker > 0.0 && has_speaker_context;
    let has_caption_cfg = cfg_scale_caption > 0.0 && has_caption_context;

    // Build list of active CFG names (determines alternating order)
    let mut enabled_cfg: Vec<CfgName> = Vec::new();
    if has_text_cfg {
        enabled_cfg.push(CfgName::Text);
    }
    if has_speaker_cfg {
        enabled_cfg.push(CfgName::Speaker);
    }
    if has_caption_cfg {
        enabled_cfg.push(CfgName::Caption);
    }
    if let Some(report) = recorder.report() {
        report.enabled_cfg = enabled_cfg
            .iter()
            .map(|name| match name {
                CfgName::Text => ConditioningSignal::Text,
                CfgName::Speaker => ConditioningSignal::Speaker,
                CfgName::Caption => ConditioningSignal::Caption,
            })
            .collect();
        report.has_speaker_context = has_speaker_context;
        report.has_caption_context = has_caption_context;
    }

    // Joint CFG requires all active guidance scales to be equal.
    if matches!(g.mode, CfgGuidanceMode::Joint) && !enabled_cfg.is_empty() {
        let active_scales: Vec<f32> = enabled_cfg
            .iter()
            .map(|n| cfg_scale_for(n, cfg_scale_text, cfg_scale_speaker, cfg_scale_caption))
            .collect();
        let min_s = active_scales.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_s = active_scales
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        if max_s - min_s >= 1e-6 {
            return Err(IrodoriError::Config(format!(
                "cfg_guidance_mode=Joint requires all active cfg scales to be equal, \
                 but got text={}, speaker={}, caption={}. \
                 Pass a single equal value for all active signals.",
                g.scale_text, g.scale_speaker, g.scale_caption
            )));
        }
    }

    // --- Precompute KV caches ---
    let effective_kv_cache = params.use_context_kv_cache || params.speaker_kv.is_some();
    let seq_lat = request.sequence_length;

    // Joint mode: one shared fully-unconditioned pass per step.
    let kv_uncond: Option<Vec<CondKvCache>> = if effective_kv_cache
        && matches!(g.mode, CfgGuidanceMode::Joint)
        && !enabled_cfg.is_empty()
    {
        Some(model.build_kv_caches(&uncond, Some(seq_lat)))
    } else {
        None
    };

    // --- Batched Independent CFG ---
    //
    // Instead of N sequential forward passes (1 conditioned + per-signal unconditioned),
    // concatenate all conditioning variants along the batch dimension and run a single
    // forward pass with batch=cfg_batch_mult.  This matches the Python implementation
    // and significantly reduces GPU kernel launch overhead (~20% per step).
    let use_batched_independent =
        matches!(g.mode, CfgGuidanceMode::Independent) && !enabled_cfg.is_empty();
    let cfg_batch_mult = if use_batched_independent {
        1 + enabled_cfg.len() // conditioned + one per enabled signal
    } else {
        1
    };
    let fixed_cond_cache = fixed_cond_cache.filter(|_| {
        supports_prepared_euler_cond_cache_request(
            params,
            batch_size,
            cfg_batch_mult,
            &enabled_cfg,
            model.model_generation().is_some(),
        )
    });
    if let Some(report) = recorder.report() {
        report.fixed_timestep_condition.request_selected = fixed_cond_cache.is_some();
    }

    // Pre-concatenated condition for batched Independent CFG.
    let batched_cfg_cond: Option<EncodedCondition> = use_batched_independent.then(|| {
        let uncond_bundles: Vec<EncodedCondition> = enabled_cfg
            .iter()
            .map(|name| make_single_uncond(name, &cond, &uncond, device))
            .collect();
        let mut refs: Vec<&EncodedCondition> = Vec::with_capacity(cfg_batch_mult);
        refs.push(&cond);
        refs.extend(uncond_bundles.iter());
        EncodedCondition::cat_batch(&refs)
    });

    // The opaque proof captures facts established by the host control flow:
    // `uncond` came from `zeros_like`, the conditioned row was concatenated
    // first, every retained text token is valid, and mask readback found no
    // active speaker/caption context. The latent sequence length is deliberately
    // not part of the proof: the derived tensors contain context K/V only, while
    // the length-dependent joint mask is constructed below for each request.
    // Every other model returns `None` from the hook and takes the ordinary pair
    // of cache builds below.
    let text_cfg_cache_proof = TextOnlyCfgCacheProof::try_new(
        supports_prepared_euler_cond_cache_request(
            params,
            batch_size,
            cfg_batch_mult,
            &enabled_cfg,
            model.model_generation().is_some(),
        ),
        enabled_cfg.len() == 1 && enabled_cfg.first() == Some(&CfgName::Text),
        true,
        conditioned_text_mask_all_valid,
        true,
        !has_speaker_context && !has_caption_context,
    );
    let derived_text_cfg_caches = batched_cfg_cond
        .as_ref()
        .filter(|_| effective_kv_cache)
        .and_then(|batched| {
            model.try_build_text_cfg_kv_caches(
                &cond,
                batched,
                seq_lat,
                text_cfg_cache_proof.as_ref(),
            )
        });
    let derived_text_cfg_pair_used = derived_text_cfg_caches.is_some();
    let (mut kv_cond, mut kv_batched_cfg) =
        if let Some((conditional, batched)) = derived_text_cfg_caches {
            (Some(conditional), Some(batched))
        } else {
            (
                effective_kv_cache.then(|| model.build_kv_caches(&cond, Some(seq_lat))),
                batched_cfg_cond
                    .as_ref()
                    .filter(|_| effective_kv_cache)
                    .map(|batched| model.build_kv_caches(batched, Some(seq_lat))),
            )
        };
    if let Some(report) = recorder.report() {
        report.context_kv = ContextKvWorkReport {
            enabled: effective_kv_cache,
            derived_text_cfg_pair_used,
            conditional_layers: kv_cond.as_ref().map_or(0, Vec::len),
            batched_cfg_layers: kv_batched_cfg.as_ref().map_or(0, Vec::len),
        };
        report.model_layers = report
            .context_kv
            .conditional_layers
            .max(report.context_kv.batched_cfg_layers);
    }

    // Alternating mode: per-signal unconditioned caches (not used by Independent).
    let use_alt_caches = effective_kv_cache
        && matches!(g.mode, CfgGuidanceMode::Alternating)
        && !enabled_cfg.is_empty();
    let mut kv_alt_text: Option<Vec<CondKvCache>> = use_alt_caches
        .then(|| {
            has_text_cfg.then(|| {
                let uncond_text = make_text_uncond(&cond, &uncond);
                model.build_kv_caches(&uncond_text, Some(seq_lat))
            })
        })
        .flatten();
    let mut kv_alt_speaker: Option<Vec<CondKvCache>> = use_alt_caches
        .then(|| {
            has_speaker_cfg.then(|| {
                let uncond_spk = make_speaker_uncond(&cond, device);
                model.build_kv_caches(&uncond_spk, Some(seq_lat))
            })
        })
        .flatten();
    let mut kv_alt_caption: Option<Vec<CondKvCache>> = use_alt_caches
        .then(|| {
            has_caption_cfg.then(|| {
                let uncond_cap = make_caption_uncond(&cond, device);
                model.build_kv_caches(&uncond_cap, Some(seq_lat))
            })
        })
        .flatten();

    // Match the reference cache policy exactly: force-speaker scaling applies
    // to the conditional cache, the batched Independent cache, and every
    // Alternating cache. Joint-unconditional speaker state is fully zeroed and
    // is intentionally left unscaled.
    if let Some(ref skv) = params.speaker_kv
        && has_speaker_context
    {
        scale_speaker_cache_set(
            [
                &mut kv_cond,
                &mut kv_batched_cfg,
                &mut kv_alt_text,
                &mut kv_alt_speaker,
                &mut kv_alt_caption,
            ],
            skv.scale,
            skv.max_layers,
        );
    }

    // --- Timestep schedule: linearly spaced [0.999, 0] ---
    // Pre-compute all timestep tensors on-device to avoid per-step CPU→GPU copies.
    let t_schedule = reference_linear_schedule(params.num_steps);
    if let Some(report) = recorder.report() {
        report.schedule_f32_bits = t_schedule.iter().copied().map(f32::to_bits).collect();
    }

    // Pre-allocate timestep tensors: tt_base[i] = [t_schedule[i]; batch_size] on device,
    // and tt_cfg[i] = tt_base[i].repeat(cfg_batch_mult) for batched Independent CFG.
    // Allocate num_steps+1 entries so the endpoint (t=0, index num_steps) is available
    // for the Heun corrector's second evaluation at t_next = t_schedule[num_steps].
    let tt_base: Vec<Tensor<1>> = t_schedule
        .iter()
        .map(|&t| Tensor::from_floats([t].repeat(batch_size).as_slice(), device))
        .collect();
    let tt_cfg: Vec<Tensor<1>> = if cfg_batch_mult > 1 {
        tt_base
            .iter()
            .map(|tt| tt.clone().repeat(&[cfg_batch_mult]))
            .collect()
    } else {
        Vec::new()
    };

    let mut speaker_kv_active = params.speaker_kv.is_some() && has_speaker_context;

    // PLMS-4: velocity history (newest-first) and regime tracking.
    // History is reset when the effective ODE RHS changes (CFG on↔off or speaker KV deactivated).
    let mut plms_history: VecDeque<Tensor<3>> = VecDeque::with_capacity(4);
    let mut plms_prev_regime = {
        let init_t = t_schedule[0];
        let init_cfg = !enabled_cfg.is_empty() && g.min_t <= init_t && init_t <= g.max_t;
        (init_cfg, speaker_kv_active)
    };

    // --- Euler / Heun / PLMS-4 ODE loop ---
    for i in 0..params.num_steps {
        let t = t_schedule[i];
        let t_next = t_schedule[i + 1];
        let tt = tt_base[i].clone();

        {
            if tracing::enabled!(tracing::Level::DEBUG) {
                let x_data: Vec<f32> = x_t.clone().into_data().convert::<f32>().to_vec().unwrap();
                let mean = x_data.iter().sum::<f32>() / x_data.len() as f32;
                let std = (x_data.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
                    / x_data.len() as f32)
                    .sqrt();
                let min = x_data.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = x_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                tracing::debug!(
                    "RF step {i}: t={t:.4} x_t min={min:.4} max={max:.4} mean={mean:.4} std={std:.4}"
                );
            }
        }

        let use_cfg = !enabled_cfg.is_empty() && g.min_t <= t && t <= g.max_t;

        // PLMS-4 regime change detection: reset history when the effective ODE RHS changes.
        if matches!(params.method, SamplerMethod::PLMS4) {
            let regime = (use_cfg, speaker_kv_active);
            if regime != plms_prev_regime {
                plms_history.clear();
                plms_prev_regime = regime;
            }
        }

        let kv_cond_ref = kv_cond.as_deref();

        #[cfg(feature = "profile")]
        let _step_label = format!("euler_step_{i}");
        #[cfg(not(feature = "profile"))]
        let _step_label = "";
        let step_value = nvtx_range!(&_step_label, {
            if use_cfg {
                match g.mode {
                    CfgGuidanceMode::Independent => {
                        // Batched forward: one pass with batch=cfg_batch_mult
                        // instead of cfg_batch_mult sequential passes.
                        let batched_cond =
                            batched_cfg_cond.as_ref().expect("batched cond must exist");
                        let x_t_cfg = Tensor::cat(vec![x_t.clone(); cfg_batch_mult], 0);
                        let tt_cfg_step = tt_cfg[i].clone();
                        let fixed_cond_lookup_attempted = fixed_cond_cache.is_some();
                        let precomputed_cond = fixed_cond_cache.and_then(|cache| {
                            cache.step(
                                model.model_generation()?,
                                i,
                                t.to_bits(),
                                x_t_cfg.dims()[0],
                                device,
                            )
                        });

                        let kv_ref = kv_batched_cfg.as_deref();
                        let v_out = nvtx_range!(
                            "forward_batched_cfg",
                            forward_sampler_model(
                                model,
                                x_t_cfg,
                                tt_cfg_step,
                                precomputed_cond,
                                batched_cond,
                                None,
                                kv_ref,
                                &lat_rope,
                                recorder,
                                ForwardWorkMeta {
                                    step_index: i,
                                    evaluation: SamplerForwardEvaluation::Primary,
                                    timestep_f32_bits: t.to_bits(),
                                    cfg_active: true,
                                    lane: SamplerForwardLane::BatchedIndependent,
                                    fixed_cond_lookup_attempted,
                                },
                            )
                        );

                        let fused_update = if elementwise_policy.fuses_independent_cfg_euler()
                            && matches!(params.method, SamplerMethod::Euler)
                            && params.temporal_rescale.is_none()
                            && enabled_cfg.len() == 1
                            && cfg_batch_mult == 2
                        {
                            let scale = cfg_scale_for(
                                &enabled_cfg[0],
                                cfg_scale_text,
                                cfg_scale_speaker,
                                cfg_scale_caption,
                            );
                            model.try_independent_cfg_euler_update(
                                x_t.clone(),
                                v_out.clone(),
                                scale,
                                t_next - t,
                            )
                        } else {
                            None
                        };
                        if let Some(updated) = fused_update {
                            SamplerStepValue::UpdatedLatent(updated)
                        } else {
                            // Split output: chunks[0] = conditioned, chunks[1..] = unconditioned
                            let chunks = v_out.chunk(cfg_batch_mult, 0);
                            let v_cond = &chunks[0];
                            let mut v = v_cond.clone();
                            for (idx, name) in enabled_cfg.iter().enumerate() {
                                let scale = cfg_scale_for(
                                    name,
                                    cfg_scale_text,
                                    cfg_scale_speaker,
                                    cfg_scale_caption,
                                );
                                v = v + (v_cond.clone() - chunks[idx + 1].clone()) * scale;
                            }
                            SamplerStepValue::Velocity(v)
                        }
                    }
                    CfgGuidanceMode::Joint => {
                        let v_cond = nvtx_range!(
                            "forward_cond",
                            forward_sampler_model(
                                model,
                                x_t.clone(),
                                tt.clone(),
                                None,
                                &cond,
                                None,
                                kv_cond_ref,
                                &lat_rope,
                                recorder,
                                ForwardWorkMeta {
                                    step_index: i,
                                    evaluation: SamplerForwardEvaluation::Primary,
                                    timestep_f32_bits: t.to_bits(),
                                    cfg_active: true,
                                    lane: SamplerForwardLane::Conditional,
                                    fixed_cond_lookup_attempted: false,
                                },
                            )
                        );
                        let v = if enabled_cfg.is_empty() {
                            v_cond
                        } else {
                            let joint_scale = cfg_scale_for(
                                &enabled_cfg[0],
                                cfg_scale_text,
                                cfg_scale_speaker,
                                cfg_scale_caption,
                            );
                            let v_uncond = nvtx_range!(
                                "forward_uncond",
                                forward_sampler_model(
                                    model,
                                    x_t.clone(),
                                    tt.clone(),
                                    None,
                                    &uncond,
                                    None,
                                    kv_uncond.as_deref(),
                                    &lat_rope,
                                    recorder,
                                    ForwardWorkMeta {
                                        step_index: i,
                                        evaluation: SamplerForwardEvaluation::Primary,
                                        timestep_f32_bits: t.to_bits(),
                                        cfg_active: true,
                                        lane: SamplerForwardLane::JointUnconditional,
                                        fixed_cond_lookup_attempted: false,
                                    },
                                )
                            );
                            v_cond.clone() + (v_cond - v_uncond) * joint_scale
                        };
                        SamplerStepValue::Velocity(v)
                    }
                    CfgGuidanceMode::Alternating => {
                        let v_cond = nvtx_range!(
                            "forward_cond",
                            forward_sampler_model(
                                model,
                                x_t.clone(),
                                tt.clone(),
                                None,
                                &cond,
                                None,
                                kv_cond_ref,
                                &lat_rope,
                                recorder,
                                ForwardWorkMeta {
                                    step_index: i,
                                    evaluation: SamplerForwardEvaluation::Primary,
                                    timestep_f32_bits: t.to_bits(),
                                    cfg_active: true,
                                    lane: SamplerForwardLane::Conditional,
                                    fixed_cond_lookup_attempted: false,
                                },
                            )
                        );
                        let v = if enabled_cfg.is_empty() {
                            v_cond
                        } else {
                            let alt_name = &enabled_cfg[i % enabled_cfg.len()];
                            let alt_cond = make_single_uncond(alt_name, &cond, &uncond, device);
                            let kv_alt_ref: Option<&[CondKvCache]> = match alt_name {
                                CfgName::Text => kv_alt_text.as_deref(),
                                CfgName::Speaker => kv_alt_speaker.as_deref(),
                                CfgName::Caption => kv_alt_caption.as_deref(),
                            };
                            let v_alt = nvtx_range!(
                                "forward_uncond",
                                forward_sampler_model(
                                    model,
                                    x_t.clone(),
                                    tt.clone(),
                                    None,
                                    &alt_cond,
                                    None,
                                    kv_alt_ref,
                                    &lat_rope,
                                    recorder,
                                    ForwardWorkMeta {
                                        step_index: i,
                                        evaluation: SamplerForwardEvaluation::Primary,
                                        timestep_f32_bits: t.to_bits(),
                                        cfg_active: true,
                                        lane: SamplerForwardLane::AlternatingUnconditional,
                                        fixed_cond_lookup_attempted: false,
                                    },
                                )
                            );
                            let scale = cfg_scale_for(
                                alt_name,
                                cfg_scale_text,
                                cfg_scale_speaker,
                                cfg_scale_caption,
                            );
                            v_cond.clone() + (v_cond - v_alt) * scale
                        };
                        SamplerStepValue::Velocity(v)
                    }
                }
            } else {
                let fixed_cond_lookup_attempted = fixed_cond_cache.is_some();
                let precomputed_cond = fixed_cond_cache.and_then(|cache| {
                    cache.step(
                        model.model_generation()?,
                        i,
                        t.to_bits(),
                        x_t.dims()[0],
                        device,
                    )
                });
                SamplerStepValue::Velocity(nvtx_range!(
                    "forward_uncfg",
                    forward_sampler_model(
                        model,
                        x_t.clone(),
                        tt.clone(),
                        precomputed_cond,
                        &cond,
                        None,
                        kv_cond_ref,
                        &lat_rope,
                        recorder,
                        ForwardWorkMeta {
                            step_index: i,
                            evaluation: SamplerForwardEvaluation::Primary,
                            timestep_f32_bits: t.to_bits(),
                            cfg_active: false,
                            lane: SamplerForwardLane::Conditional,
                            fixed_cond_lookup_attempted,
                        },
                    )
                ))
            }
        });

        // Temporal score rescaling for v1
        let step_value = match step_value {
            SamplerStepValue::Velocity(v) => {
                SamplerStepValue::Velocity(if let Some(trc) = params.temporal_rescale {
                    temporal_score_rescale(v, x_t.clone(), t, trc.k, trc.sigma)
                } else {
                    v
                })
            }
            updated @ SamplerStepValue::UpdatedLatent(_) => updated,
        };

        // Disable force-speaker scaling once t crosses the threshold.
        // Runs before the Heun corrector so v2 sees the updated KV state.
        if speaker_kv_active
            && let Some(ref skv) = params.speaker_kv
            && let Some(_) = skv.min_t.filter(|&min_t| t_next < min_t && t >= min_t)
        {
            let inv_scale = 1.0 / skv.scale;
            scale_speaker_cache_set(
                [
                    &mut kv_cond,
                    &mut kv_batched_cfg,
                    &mut kv_alt_text,
                    &mut kv_alt_speaker,
                    &mut kv_alt_caption,
                ],
                inv_scale,
                skv.max_layers,
            );
            speaker_kv_active = false;
        }

        if let SamplerStepValue::Velocity(v) = &step_value
            && tracing::enabled!(tracing::Level::DEBUG)
        {
            let v_data: Vec<f32> = v.clone().into_data().convert::<f32>().to_vec().unwrap();
            let mean = v_data.iter().sum::<f32>() / v_data.len() as f32;
            let std = (v_data.iter().map(|a| (a - mean).powi(2)).sum::<f32>()
                / v_data.len() as f32)
                .sqrt();
            let min = v_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = v_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            tracing::debug!("RF step {i}: v min={min:.4} max={max:.4} mean={mean:.4} std={std:.4}");
        }

        let v = match step_value {
            SamplerStepValue::Velocity(v) => v,
            SamplerStepValue::UpdatedLatent(updated) => {
                x_t = updated;
                continue;
            }
        };

        // ODE step: Euler or Heun's trapezoidal corrector.
        let dt = t_next - t;
        let v_eff = match params.method {
            SamplerMethod::Euler => v,
            SamplerMethod::Heun => {
                // --- Heun's method: corrector step ---
                // Predictor: x_pred = x_t + v1 * dt
                let v1 = v;
                let x_pred = x_t.clone() + v1.clone() * dt;

                // Evaluate model at (x_pred, t_next) using the same CFG strategy as v1.
                // All time-dependent guards (CFG window, temporal rescale) use t_next.
                let use_cfg_v2 = !enabled_cfg.is_empty() && g.min_t <= t_next && t_next <= g.max_t;
                let tt_next = tt_base[i + 1].clone(); // always available (num_steps+1 entries)
                let kv_cond_ref_v2 = kv_cond.as_deref();

                let v2_raw = nvtx_range!("heun_corrector", {
                    if use_cfg_v2 {
                        match g.mode {
                            CfgGuidanceMode::Independent => {
                                let batched_cond =
                                    batched_cfg_cond.as_ref().expect("batched cond must exist");
                                let x_pred_cfg =
                                    Tensor::cat(vec![x_pred.clone(); cfg_batch_mult], 0);
                                let tt_next_cfg = tt_cfg[i + 1].clone();
                                let v_out = forward_sampler_model(
                                    model,
                                    x_pred_cfg,
                                    tt_next_cfg,
                                    None,
                                    batched_cond,
                                    None,
                                    kv_batched_cfg.as_deref(),
                                    &lat_rope,
                                    recorder,
                                    ForwardWorkMeta {
                                        step_index: i,
                                        evaluation: SamplerForwardEvaluation::HeunCorrector,
                                        timestep_f32_bits: t_next.to_bits(),
                                        cfg_active: true,
                                        lane: SamplerForwardLane::BatchedIndependent,
                                        fixed_cond_lookup_attempted: false,
                                    },
                                );
                                let chunks = v_out.chunk(cfg_batch_mult, 0);
                                let v_cond2 = &chunks[0];
                                let mut v2 = v_cond2.clone();
                                for (idx, name) in enabled_cfg.iter().enumerate() {
                                    let scale = cfg_scale_for(
                                        name,
                                        cfg_scale_text,
                                        cfg_scale_speaker,
                                        cfg_scale_caption,
                                    );
                                    v2 = v2 + (v_cond2.clone() - chunks[idx + 1].clone()) * scale;
                                }
                                v2
                            }
                            CfgGuidanceMode::Joint => {
                                let v_cond2 = forward_sampler_model(
                                    model,
                                    x_pred.clone(),
                                    tt_next.clone(),
                                    None,
                                    &cond,
                                    None,
                                    kv_cond_ref_v2,
                                    &lat_rope,
                                    recorder,
                                    ForwardWorkMeta {
                                        step_index: i,
                                        evaluation: SamplerForwardEvaluation::HeunCorrector,
                                        timestep_f32_bits: t_next.to_bits(),
                                        cfg_active: true,
                                        lane: SamplerForwardLane::Conditional,
                                        fixed_cond_lookup_attempted: false,
                                    },
                                );
                                if enabled_cfg.is_empty() {
                                    v_cond2
                                } else {
                                    let joint_scale = cfg_scale_for(
                                        &enabled_cfg[0],
                                        cfg_scale_text,
                                        cfg_scale_speaker,
                                        cfg_scale_caption,
                                    );
                                    let v_uncond2 = forward_sampler_model(
                                        model,
                                        x_pred.clone(),
                                        tt_next.clone(),
                                        None,
                                        &uncond,
                                        None,
                                        kv_uncond.as_deref(),
                                        &lat_rope,
                                        recorder,
                                        ForwardWorkMeta {
                                            step_index: i,
                                            evaluation: SamplerForwardEvaluation::HeunCorrector,
                                            timestep_f32_bits: t_next.to_bits(),
                                            cfg_active: true,
                                            lane: SamplerForwardLane::JointUnconditional,
                                            fixed_cond_lookup_attempted: false,
                                        },
                                    );
                                    v_cond2.clone() + (v_cond2 - v_uncond2) * joint_scale
                                }
                            }
                            CfgGuidanceMode::Alternating => {
                                let v_cond2 = forward_sampler_model(
                                    model,
                                    x_pred.clone(),
                                    tt_next.clone(),
                                    None,
                                    &cond,
                                    None,
                                    kv_cond_ref_v2,
                                    &lat_rope,
                                    recorder,
                                    ForwardWorkMeta {
                                        step_index: i,
                                        evaluation: SamplerForwardEvaluation::HeunCorrector,
                                        timestep_f32_bits: t_next.to_bits(),
                                        cfg_active: true,
                                        lane: SamplerForwardLane::Conditional,
                                        fixed_cond_lookup_attempted: false,
                                    },
                                );
                                if enabled_cfg.is_empty() {
                                    v_cond2
                                } else {
                                    // Use same dropped-condition index as v1 within this step
                                    // (index i, not i+1) to keep the step internally consistent.
                                    let alt_name = &enabled_cfg[i % enabled_cfg.len()];
                                    let alt_cond =
                                        make_single_uncond(alt_name, &cond, &uncond, device);
                                    let kv_alt_ref: Option<&[CondKvCache]> = match alt_name {
                                        CfgName::Text => kv_alt_text.as_deref(),
                                        CfgName::Speaker => kv_alt_speaker.as_deref(),
                                        CfgName::Caption => kv_alt_caption.as_deref(),
                                    };
                                    let v_alt2 = forward_sampler_model(
                                        model,
                                        x_pred.clone(),
                                        tt_next.clone(),
                                        None,
                                        &alt_cond,
                                        None,
                                        kv_alt_ref,
                                        &lat_rope,
                                        recorder,
                                        ForwardWorkMeta {
                                            step_index: i,
                                            evaluation: SamplerForwardEvaluation::HeunCorrector,
                                            timestep_f32_bits: t_next.to_bits(),
                                            cfg_active: true,
                                            lane: SamplerForwardLane::AlternatingUnconditional,
                                            fixed_cond_lookup_attempted: false,
                                        },
                                    );
                                    let scale = cfg_scale_for(
                                        alt_name,
                                        cfg_scale_text,
                                        cfg_scale_speaker,
                                        cfg_scale_caption,
                                    );
                                    v_cond2.clone() + (v_cond2 - v_alt2) * scale
                                }
                            }
                        }
                    } else {
                        // CFG inactive at t_next — single conditioned pass
                        forward_sampler_model(
                            model,
                            x_pred.clone(),
                            tt_next,
                            None,
                            &cond,
                            None,
                            kv_cond_ref_v2,
                            &lat_rope,
                            recorder,
                            ForwardWorkMeta {
                                step_index: i,
                                evaluation: SamplerForwardEvaluation::HeunCorrector,
                                timestep_f32_bits: t_next.to_bits(),
                                cfg_active: false,
                                lane: SamplerForwardLane::Conditional,
                                fixed_cond_lookup_attempted: false,
                            },
                        )
                    }
                });

                // Temporal rescale for v2 uses t_next and x_pred
                let v2 = if let Some(trc) = params.temporal_rescale {
                    temporal_score_rescale(v2_raw, x_pred, t_next, trc.k, trc.sigma)
                } else {
                    v2_raw
                };

                // Trapezoidal average of the two velocity estimates
                (v1 + v2) / 2.0
            }
            SamplerMethod::PLMS4 => {
                // --- Adams-Bashforth multistep extrapolation ---
                // Push current velocity to front (newest-first), cap history at 4.
                plms_history.push_front(v);
                if plms_history.len() > 4 {
                    plms_history.pop_back();
                }
                ab_extrapolate(&plms_history)
            }
        };

        x_t = x_t + v_eff * dt;
    }

    if let Some(report) = recorder.report() {
        report.whole_model_forwards = report.forwards.len();
        report.model_block_calls = report.whole_model_forwards * report.model_layers;
    }
    Ok(x_t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfg_scale_for_returns_correct_scale() {
        assert_eq!(cfg_scale_for(&CfgName::Text, 3.0, 5.0, 2.0), 3.0);
        assert_eq!(cfg_scale_for(&CfgName::Speaker, 3.0, 5.0, 2.0), 5.0);
        assert_eq!(cfg_scale_for(&CfgName::Caption, 3.0, 5.0, 2.0), 2.0);
    }

    #[test]
    fn prepared_euler_cond_cache_selector_covers_b1_b2_b3_and_fails_closed() {
        let params = SamplerParams {
            num_steps: 4,
            ..Default::default()
        };
        let text_only = [CfgName::Text];

        assert!(supports_prepared_euler_cond_cache_request(
            &params,
            1,
            1,
            &[],
            true,
        ));
        assert!(supports_prepared_euler_cond_cache_request(
            &params, 1, 2, &text_only, true,
        ));
        assert!(supports_prepared_euler_cond_cache_request(
            &params,
            1,
            3,
            &[CfgName::Text, CfgName::Caption],
            true,
        ));
        assert!(!supports_prepared_euler_cond_cache_request(
            &params, 2, 2, &text_only, true,
        ));
        assert!(!supports_prepared_euler_cond_cache_request(
            &params,
            1,
            4,
            &[CfgName::Text, CfgName::Speaker, CfgName::Caption],
            true,
        ));
        assert!(!supports_prepared_euler_cond_cache_request(
            &params, 1, 2, &text_only, false,
        ));

        let unsupported = SamplerParams {
            method: SamplerMethod::Heun,
            ..params
        };
        assert!(!supports_prepared_euler_cond_cache_request(
            &unsupported,
            1,
            2,
            &text_only,
            true,
        ));
    }

    #[test]
    fn independent_cfg_drops_only_named_context_when_both_are_present() {
        let device = Device::default();
        let cond = EncodedCondition {
            text_state: Tensor::ones([1, 2, 4], &device),
            text_mask: Tensor::ones([1, 2], &device),
            aux: Some(AuxConditionState::Both {
                speaker_state: Tensor::ones([1, 3, 4], &device) * 2.0,
                speaker_mask: Tensor::ones([1, 3], &device),
                caption_state: Tensor::ones([1, 5, 4], &device) * 3.0,
                caption_mask: Tensor::ones([1, 5], &device),
            }),
        };
        let uncond = cond.zeros_like(&device);

        let speaker = make_single_uncond(&CfgName::Speaker, &cond, &uncond, &device);
        let speaker_aux = speaker.aux.unwrap();
        assert_eq!(
            speaker_aux
                .speaker()
                .unwrap()
                .0
                .clone()
                .abs()
                .sum()
                .into_scalar::<f32>(),
            0.0
        );
        assert_eq!(
            speaker_aux
                .caption()
                .unwrap()
                .0
                .clone()
                .min()
                .into_scalar::<f32>(),
            3.0
        );

        let caption = make_single_uncond(&CfgName::Caption, &cond, &uncond, &device);
        let caption_aux = caption.aux.unwrap();
        assert_eq!(
            caption_aux
                .speaker()
                .unwrap()
                .0
                .clone()
                .min()
                .into_scalar::<f32>(),
            2.0
        );
        assert_eq!(
            caption_aux
                .caption()
                .unwrap()
                .0
                .clone()
                .abs()
                .sum()
                .into_scalar::<f32>(),
            0.0
        );
    }

    #[test]
    fn timestep_schedule_shape_and_endpoints() {
        let num_steps = 40;
        let t_schedule = reference_linear_schedule(num_steps);

        assert_eq!(t_schedule.len(), num_steps + 1);
        assert!(
            (t_schedule[0] - 0.999).abs() < 1e-6,
            "first step should be 0.999"
        );
        assert!(
            (t_schedule[num_steps]).abs() < 1e-6,
            "last step should be ~0"
        );
        // Monotonically decreasing
        for w in t_schedule.windows(2) {
            assert!(w[0] > w[1], "schedule must be strictly decreasing");
        }
    }

    #[test]
    fn timestep_schedule_uniform_spacing() {
        let num_steps = 10;
        let t_schedule = reference_linear_schedule(num_steps);

        let dt = t_schedule[0] - t_schedule[1];
        for w in t_schedule.windows(2) {
            assert!((w[0] - w[1] - dt).abs() < 1e-6, "spacing should be uniform");
        }
    }

    #[test]
    fn timestep_schedule_matches_pytorch_cuda_fp32_bits() {
        let actual: Vec<u32> = reference_linear_schedule(40)
            .into_iter()
            .map(f32::to_bits)
            .collect();
        assert_eq!(
            actual,
            [
                1065336439, 1064917428, 1064498417, 1064079406, 1063660395, 1063241384, 1062822374,
                1062403362, 1061984351, 1061565340, 1061146329, 1060727319, 1060308307, 1059889296,
                1059470285, 1059051274, 1058632264, 1058213252, 1057794241, 1057375230, 1056947831,
                1056109810, 1055271787, 1054433766, 1053595742, 1052757721, 1051919700, 1051081677,
                1050243656, 1049405633, 1048559223, 1046883181, 1045207134, 1043531092, 1041855046,
                1040170615, 1036818530, 1033466438, 1028429922, 1020041298, 0,
            ]
        );
    }

    #[test]
    fn alternating_cfg_selection_cycles() {
        let enabled = [CfgName::Text, CfgName::Speaker, CfgName::Caption];
        let selected: Vec<&CfgName> = (0..9).map(|i| &enabled[i % enabled.len()]).collect();
        assert_eq!(
            selected,
            [
                &CfgName::Text,
                &CfgName::Speaker,
                &CfgName::Caption,
                &CfgName::Text,
                &CfgName::Speaker,
                &CfgName::Caption,
                &CfgName::Text,
                &CfgName::Speaker,
                &CfgName::Caption,
            ]
        );
    }

    #[test]
    fn alternating_single_signal_always_same() {
        let enabled = [CfgName::Text];
        let selected: Vec<&CfgName> = (0..5).map(|i| &enabled[i % enabled.len()]).collect();
        assert!(
            selected.iter().all(|n| **n == CfgName::Text),
            "single-signal alternating should always pick the same signal"
        );
    }

    #[test]
    fn use_cfg_check_respects_t_range() {
        let enabled_cfg = [CfgName::Text];
        let min_t = 0.1_f32;
        let max_t = 0.9_f32;

        // In range
        let t = 0.5;
        assert!(
            !enabled_cfg.is_empty() && min_t <= t && t <= max_t,
            "should use cfg"
        );
        // Below min_t
        let t = 0.05;
        assert!(
            !(min_t <= t && t <= max_t),
            "below min_t should not use cfg"
        );
        // Above max_t
        let t = 0.95;
        assert!(
            !(min_t <= t && t <= max_t),
            "above max_t should not use cfg"
        );
        // Empty cfg
        let empty: Vec<CfgName> = vec![];
        assert!(!(!empty.is_empty() && min_t <= 0.5 && 0.5 <= max_t));
    }

    // -----------------------------------------------------------------------
    // Integration tests: run `sample_euler_rf_cfg` with a tiny model
    // -----------------------------------------------------------------------
    use super::super::params::{GuidanceConfig, SamplerParams, SamplingRequest, SpeakerKvConfig};
    use crate::config::SamplerMethod;
    use crate::model::attention::SpeakerKvRange;
    use crate::model::{InferenceOptimizedModel, TextToLatentRfDiT};

    fn unit_speaker_cache(device: &Device) -> Vec<CondKvCache> {
        let text_k = Tensor::<4>::ones([1, 2, 1, 2], device);
        let text_v = Tensor::<4>::ones([1, 2, 1, 2], device);
        let speaker_k = Tensor::<4>::ones([1, 2, 1, 2], device);
        let speaker_v = Tensor::<4>::ones([1, 2, 1, 2], device);
        let ctx_k = Tensor::cat(vec![text_k, speaker_k], 1);
        let ctx_v = Tensor::cat(vec![text_v, speaker_v], 1);
        vec![CondKvCache {
            ctx_k,
            ctx_v,
            ctx_mask: Tensor::<2, burn::tensor::Bool>::full([1, 4], true, device),
            joint_mask: None,
            speaker_range: Some(SpeakerKvRange::from_start_len(2, 2)),
            packed_ctx_kv_wgsl: None,
            joint_mask_wgsl: None,
            joint_attend_mask_wgsl: None,
        }]
    }

    fn speaker_cache_value(cache: &Option<Vec<CondKvCache>>) -> f32 {
        cache
            .as_ref()
            .and_then(|layers| layers.first())
            .and_then(|layer| {
                layer
                    .speaker_range
                    .map(|range| layer.ctx_k.clone().narrow(1, range.start(), range.len()))
            })
            .expect("test cache must contain a speaker range")
            .max()
            .into_scalar()
    }

    #[test]
    fn speaker_kv_scale_set_keeps_conditional_and_alternating_caches_symmetric() {
        let device = Device::default();
        let mut conditional = Some(unit_speaker_cache(&device));
        let mut alternating_text = Some(unit_speaker_cache(&device));
        let mut alternating_caption = Some(unit_speaker_cache(&device));

        scale_speaker_cache_set(
            [
                &mut conditional,
                &mut alternating_text,
                &mut alternating_caption,
            ],
            2.0,
            None,
        );
        assert_eq!(speaker_cache_value(&conditional), 2.0);
        assert_eq!(speaker_cache_value(&alternating_text), 2.0);
        assert_eq!(speaker_cache_value(&alternating_caption), 2.0);

        scale_speaker_cache_set(
            [
                &mut conditional,
                &mut alternating_text,
                &mut alternating_caption,
            ],
            0.5,
            None,
        );
        assert_eq!(speaker_cache_value(&conditional), 1.0);
        assert_eq!(speaker_cache_value(&alternating_text), 1.0);
        assert_eq!(speaker_cache_value(&alternating_caption), 1.0);
    }

    fn tiny_model_and_request() -> (InferenceOptimizedModel, SamplingRequest, Device) {
        use crate::config::tiny_model_config;

        let device = Device::default();
        let cfg = tiny_model_config();
        let model = TextToLatentRfDiT::new(&cfg, &device);

        let (batch, seq_txt, seq_lat) = (1, 4, 6);
        let text_ids = Tensor::<2, burn::tensor::Int>::zeros([batch, seq_txt], &device);
        let text_mask: Tensor<2, burn::tensor::Bool> =
            Tensor::<2>::ones([batch, seq_txt], &device).greater_elem(0.0);

        let speaker_dim = cfg.speaker_patched_latent_dim();
        let ref_lat = Tensor::<3>::ones([batch, 3, speaker_dim], &device);
        let ref_mask: Tensor<2, burn::tensor::Bool> =
            Tensor::<2>::ones([batch, 3], &device).greater_elem(0.0);

        let noise = Tensor::<3>::ones([batch, seq_lat, cfg.patched_latent_dim()], &device);

        let request = SamplingRequest {
            text_ids,
            text_mask,
            ref_latent: Some(ref_lat),
            ref_mask: Some(ref_mask),
            sequence_length: seq_lat,
            caption_ids: None,
            caption_mask: None,
            initial_noise: Some(noise),
        };

        (InferenceOptimizedModel::new(model), request, device)
    }

    fn tiny_caption_model_and_request() -> (InferenceOptimizedModel, SamplingRequest, Device) {
        use crate::config::tiny_caption_config;

        let device = Device::default();
        let cfg = tiny_caption_config();
        let model = TextToLatentRfDiT::new(&cfg, &device);

        let (batch, seq_txt, seq_cap, seq_lat) = (1, 4, 3, 6);
        let text_ids = Tensor::<2, burn::tensor::Int>::zeros([batch, seq_txt], &device);
        let text_mask: Tensor<2, burn::tensor::Bool> =
            Tensor::<2>::ones([batch, seq_txt], &device).greater_elem(0.0);

        // Caption tokens (vocab indices 1..=seq_cap to avoid pad=0)
        let cap_ids = Tensor::<2, burn::tensor::Int>::ones([batch, seq_cap], &device);
        let cap_mask: Tensor<2, burn::tensor::Bool> =
            Tensor::<2>::ones([batch, seq_cap], &device).greater_elem(0.0);

        let noise = Tensor::<3>::ones([batch, seq_lat, cfg.patched_latent_dim()], &device);

        let request = SamplingRequest {
            text_ids,
            text_mask,
            ref_latent: None,
            ref_mask: None,
            sequence_length: seq_lat,
            caption_ids: Some(cap_ids),
            caption_mask: Some(cap_mask),
            initial_noise: Some(noise),
        };

        (InferenceOptimizedModel::new(model), request, device)
    }

    #[test]
    fn sampler_no_cfg_produces_finite_output() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 2,
            guidance: GuidanceConfig {
                scale_text: 0.0,
                scale_speaker: 0.0,
                scale_caption: 0.0,
                ..Default::default()
            },
            use_context_kv_cache: false,
            ..Default::default()
        };

        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        let [b, s, _d] = out.dims();
        assert_eq!(b, 1);
        assert_eq!(s, 6);
        let vals: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "output must be all finite"
        );
    }

    #[test]
    fn sampler_rejects_mismatched_request_before_encoding() {
        let (model, mut request, device) = tiny_model_and_request();
        request.text_mask = Tensor::<2>::ones([1, 3], &device).greater_elem(0.0);

        let result = sample_euler_rf_cfg(&model, request, &SamplerParams::default(), &device);
        assert!(matches!(result, Err(crate::error::IrodoriError::Shape(_))));
    }

    #[test]
    fn sampler_rejects_half_reference_pair_before_encoding() {
        let (model, mut request, device) = tiny_model_and_request();
        request.ref_mask = None;

        let result = sample_euler_rf_cfg(&model, request, &SamplerParams::default(), &device);
        assert!(matches!(
            result,
            Err(crate::error::IrodoriError::MissingInput(_))
        ));
    }

    #[test]
    fn sampler_rejects_conditions_unsupported_by_model_before_encoding() {
        let (speaker_model, mut speaker_request, speaker_device) = tiny_model_and_request();
        speaker_request.caption_ids = Some(Tensor::zeros([1, 3], &speaker_device));
        speaker_request.caption_mask =
            Some(Tensor::<2>::ones([1, 3], &speaker_device).greater_elem(0.0));
        let speaker_result = sample_euler_rf_cfg(
            &speaker_model,
            speaker_request,
            &SamplerParams::default(),
            &speaker_device,
        );
        assert!(matches!(
            speaker_result,
            Err(crate::error::IrodoriError::Config(_))
        ));

        let (caption_model, mut caption_request, caption_device) = tiny_caption_model_and_request();
        caption_request.ref_latent = Some(Tensor::zeros([1, 2, 8], &caption_device));
        caption_request.ref_mask =
            Some(Tensor::<2>::ones([1, 2], &caption_device).greater_elem(0.0));
        let caption_result = sample_euler_rf_cfg(
            &caption_model,
            caption_request,
            &SamplerParams::default(),
            &caption_device,
        );
        assert!(matches!(
            caption_result,
            Err(crate::error::IrodoriError::Config(_))
        ));
    }

    #[test]
    fn sampler_independent_cfg_runs_without_error() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 2,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Independent,
                scale_text: 3.0,
                scale_speaker: 5.0,
                scale_caption: 0.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: false,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        let [b, s, _d] = out.dims();
        assert_eq!(b, 1);
        assert_eq!(s, 6);
        let vals: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sampler_alternating_cfg_runs_without_error() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 4, // enough steps to cycle through text + speaker
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Alternating,
                scale_text: 2.0,
                scale_speaker: 3.0,
                scale_caption: 0.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: false,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        let [b, s, _d] = out.dims();
        assert_eq!(b, 1);
        assert_eq!(s, 6);
        let vals: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sampler_alternating_cfg_with_speaker_kv_scaling_runs() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 4,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Alternating,
                scale_text: 2.0,
                scale_speaker: 3.0,
                scale_caption: 0.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            speaker_kv: Some(SpeakerKvConfig {
                scale: 2.0,
                max_layers: None,
                min_t: Some(0.5),
            }),
            use_context_kv_cache: true,
            ..Default::default()
        };

        let values = sample_euler_rf_cfg(&model, request, &params, &device)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn sampler_independent_cfg_cached_runs() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 2,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Independent,
                scale_text: 3.0,
                scale_speaker: 5.0,
                scale_caption: 0.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: true,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        assert!(
            out.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn reported_four_step_text_cfg_counts_actual_model_work() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 4,
            method: SamplerMethod::Euler,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Independent,
                scale_text: 3.0,
                scale_speaker: 0.0,
                scale_caption: 0.0,
                min_t: 0.5,
                max_t: 1.0,
            },
            use_context_kv_cache: true,
            ..Default::default()
        };

        let expected = sample_euler_rf_cfg(&model, request.clone(), &params, &device)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let (actual, report) =
            sample_euler_rf_cfg_reported(&model, request, &params, &device).unwrap();
        assert_eq!(actual.into_data().to_vec::<f32>().unwrap(), expected);
        assert_eq!(
            report.schedule_f32_bits,
            (0..=4)
                .map(|index| 0.999_f32 * (1.0 - index as f32 / 4.0))
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            report
                .forwards
                .iter()
                .map(|forward| forward.batch_rows)
                .collect::<Vec<_>>(),
            [2, 2, 1, 1]
        );
        assert_eq!(
            report
                .forwards
                .iter()
                .map(|forward| forward.cfg_active)
                .collect::<Vec<_>>(),
            [true, true, false, false]
        );
        assert_eq!(report.whole_model_forwards, 4);
        assert_eq!(report.effective_model_rows(), 6);
        assert_eq!(report.model_block_calls, report.model_layers * 4);
        assert_eq!(report.fixed_timestep_condition.ordinary_cond_forwards, 4);
        assert_eq!(
            serde_json::from_str::<SamplerWorkReport>(&serde_json::to_string(&report).unwrap())
                .unwrap(),
            report
        );
    }

    #[test]
    fn sampler_speaker_kv_deactivation() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 4,
            guidance: GuidanceConfig {
                scale_text: 0.0,
                scale_speaker: 0.0,
                scale_caption: 0.0,
                ..Default::default()
            },
            speaker_kv: Some(SpeakerKvConfig {
                scale: 2.0,
                max_layers: None,
                min_t: Some(0.5), // should deactivate mid-schedule
            }),
            use_context_kv_cache: true,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        assert!(
            out.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn sampler_joint_unequal_scales_errors() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 2,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Joint,
                scale_text: 3.0,
                scale_speaker: 5.0, // unequal → should error
                scale_caption: 0.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: false,
            ..Default::default()
        };

        let result = sample_euler_rf_cfg(&model, request, &params, &device);
        assert!(
            result.is_err(),
            "Joint mode with unequal text/speaker scales must error"
        );
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("Joint"),
            "error should mention Joint mode: {msg}"
        );
    }

    #[test]
    fn sampler_cached_matches_uncached() {
        let (model, request, device) = tiny_model_and_request();

        let base_params = SamplerParams {
            num_steps: 2,
            guidance: GuidanceConfig {
                scale_text: 0.0,
                scale_speaker: 0.0,
                scale_caption: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };

        // Uncached
        let request_uncached = SamplingRequest {
            text_ids: request.text_ids.clone(),
            text_mask: request.text_mask.clone(),
            ref_latent: request.ref_latent.clone(),
            ref_mask: request.ref_mask.clone(),
            sequence_length: request.sequence_length,
            caption_ids: None,
            caption_mask: None,
            initial_noise: request.initial_noise.clone(),
        };
        let params_uncached = SamplerParams {
            use_context_kv_cache: false,
            ..base_params.clone()
        };
        let out_uncached =
            sample_euler_rf_cfg(&model, request_uncached, &params_uncached, &device).unwrap();

        // Cached
        let params_cached = SamplerParams {
            use_context_kv_cache: true,
            ..base_params
        };
        let out_cached = sample_euler_rf_cfg(&model, request, &params_cached, &device).unwrap();

        let diff: f32 = (out_uncached - out_cached).abs().max().into_scalar();
        assert_eq!(
            diff, 0.0,
            "cached and uncached should produce identical output on NdArray"
        );
    }

    // --- Rubber duck finding #1: Joint happy-path ---
    #[test]
    fn sampler_joint_cfg_happy_path() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 2,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Joint,
                scale_text: 3.0,
                scale_speaker: 3.0, // must equal text for Joint
                scale_caption: 0.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: true,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        let [b, s, _d] = out.dims();
        assert_eq!((b, s), (1, 6));
        assert!(
            out.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    // --- Rubber duck finding #2: Caption CFG ---
    #[test]
    fn sampler_caption_independent_cfg_runs() {
        let (model, request, device) = tiny_caption_model_and_request();
        let params = SamplerParams {
            num_steps: 2,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Independent,
                scale_text: 3.0,
                scale_speaker: 0.0,
                scale_caption: 2.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: false,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        let [b, s, _d] = out.dims();
        assert_eq!((b, s), (1, 6));
        assert!(
            out.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn sampler_caption_alternating_cfg_runs() {
        let (model, request, device) = tiny_caption_model_and_request();
        let params = SamplerParams {
            num_steps: 4,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Alternating,
                scale_text: 2.0,
                scale_speaker: 0.0,
                scale_caption: 3.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: true,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        assert!(
            out.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    // --- Rubber duck finding #3: Cached-vs-uncached under CFG ---
    #[test]
    fn sampler_independent_cfg_cached_matches_uncached() {
        let (model, request, device) = tiny_model_and_request();
        let guidance = GuidanceConfig {
            mode: CfgGuidanceMode::Independent,
            scale_text: 3.0,
            scale_speaker: 5.0,
            scale_caption: 0.0,
            min_t: 0.0,
            max_t: 1.0,
        };

        let request2 = SamplingRequest {
            text_ids: request.text_ids.clone(),
            text_mask: request.text_mask.clone(),
            ref_latent: request.ref_latent.clone(),
            ref_mask: request.ref_mask.clone(),
            sequence_length: request.sequence_length,
            caption_ids: None,
            caption_mask: None,
            initial_noise: request.initial_noise.clone(),
        };

        let out_uncached = sample_euler_rf_cfg(
            &model,
            request2,
            &SamplerParams {
                num_steps: 2,
                guidance: guidance.clone(),
                use_context_kv_cache: false,
                ..Default::default()
            },
            &device,
        )
        .unwrap();

        let out_cached = sample_euler_rf_cfg(
            &model,
            request,
            &SamplerParams {
                num_steps: 2,
                guidance,
                use_context_kv_cache: true,
                ..Default::default()
            },
            &device,
        )
        .unwrap();

        let diff: f32 = (out_uncached - out_cached).abs().max().into_scalar();
        assert_eq!(
            diff, 0.0,
            "Independent CFG cached and uncached must match on NdArray"
        );
    }

    #[test]
    fn sampler_alternating_cfg_cached_matches_uncached() {
        let (model, request, device) = tiny_model_and_request();
        let guidance = GuidanceConfig {
            mode: CfgGuidanceMode::Alternating,
            scale_text: 2.0,
            scale_speaker: 3.0,
            scale_caption: 0.0,
            min_t: 0.0,
            max_t: 1.0,
        };

        let request2 = SamplingRequest {
            text_ids: request.text_ids.clone(),
            text_mask: request.text_mask.clone(),
            ref_latent: request.ref_latent.clone(),
            ref_mask: request.ref_mask.clone(),
            sequence_length: request.sequence_length,
            caption_ids: None,
            caption_mask: None,
            initial_noise: request.initial_noise.clone(),
        };

        let out_uncached = sample_euler_rf_cfg(
            &model,
            request2,
            &SamplerParams {
                num_steps: 4,
                guidance: guidance.clone(),
                use_context_kv_cache: false,
                ..Default::default()
            },
            &device,
        )
        .unwrap();

        let out_cached = sample_euler_rf_cfg(
            &model,
            request,
            &SamplerParams {
                num_steps: 4,
                guidance,
                use_context_kv_cache: true,
                ..Default::default()
            },
            &device,
        )
        .unwrap();

        let diff: f32 = (out_uncached - out_cached).abs().max().into_scalar();
        assert_eq!(
            diff, 0.0,
            "Alternating CFG cached and uncached must match on NdArray"
        );
    }

    // --- Rubber duck finding #4: Speaker KV deactivation with both paths ---
    #[test]
    fn sampler_speaker_kv_with_and_without_min_t_both_succeed() {
        let (model, request, device) = tiny_model_and_request();

        let request2 = SamplingRequest {
            text_ids: request.text_ids.clone(),
            text_mask: request.text_mask.clone(),
            ref_latent: request.ref_latent.clone(),
            ref_mask: request.ref_mask.clone(),
            sequence_length: request.sequence_length,
            caption_ids: None,
            caption_mask: None,
            initial_noise: request.initial_noise.clone(),
        };

        // scale=2.0, min_t=None → scaling stays for all steps
        let out_always = sample_euler_rf_cfg(
            &model,
            request2,
            &SamplerParams {
                num_steps: 4,
                guidance: GuidanceConfig {
                    scale_text: 0.0,
                    scale_speaker: 0.0,
                    scale_caption: 0.0,
                    ..Default::default()
                },
                speaker_kv: Some(SpeakerKvConfig {
                    scale: 2.0,
                    max_layers: None,
                    min_t: None,
                }),
                use_context_kv_cache: true,
                ..Default::default()
            },
            &device,
        )
        .unwrap();
        assert!(
            out_always
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );

        // scale=2.0, min_t=Some(0.5) → deactivation branch fires at step 2
        let out_reverted = sample_euler_rf_cfg(
            &model,
            request,
            &SamplerParams {
                num_steps: 4,
                guidance: GuidanceConfig {
                    scale_text: 0.0,
                    scale_speaker: 0.0,
                    scale_caption: 0.0,
                    ..Default::default()
                },
                speaker_kv: Some(SpeakerKvConfig {
                    scale: 2.0,
                    max_layers: None,
                    min_t: Some(0.5),
                }),
                use_context_kv_cache: true,
                ..Default::default()
            },
            &device,
        )
        .unwrap();
        assert!(
            out_reverted
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    // --- num_steps=1 edge case (regression for precomputed timestep Vecs) ---
    #[test]
    fn sampler_single_step_independent_cfg() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 1,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Independent,
                scale_text: 3.0,
                scale_speaker: 5.0,
                scale_caption: 0.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: true,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        let [b, s, _d] = out.dims();
        assert_eq!(b, 1);
        assert_eq!(s, 6);
        let vals: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sampler_single_step_no_cfg() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 1,
            guidance: GuidanceConfig {
                scale_text: 0.0,
                scale_speaker: 0.0,
                scale_caption: 0.0,
                ..Default::default()
            },
            use_context_kv_cache: false,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        let vals: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    // -----------------------------------------------------------------------
    // Heun's method tests
    // -----------------------------------------------------------------------

    #[test]
    fn heun_no_cfg_produces_finite_output() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 2,
            method: SamplerMethod::Heun,
            guidance: GuidanceConfig {
                scale_text: 0.0,
                scale_speaker: 0.0,
                scale_caption: 0.0,
                ..Default::default()
            },
            use_context_kv_cache: false,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        let [b, s, _d] = out.dims();
        assert_eq!((b, s), (1, 6));
        assert!(
            out.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn heun_independent_cfg_produces_finite_output() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 2,
            method: SamplerMethod::Heun,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Independent,
                scale_text: 3.0,
                scale_speaker: 5.0,
                scale_caption: 0.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: true,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        let [b, s, _d] = out.dims();
        assert_eq!((b, s), (1, 6));
        assert!(
            out.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn heun_joint_cfg_produces_finite_output() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 2,
            method: SamplerMethod::Heun,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Joint,
                scale_text: 3.0,
                scale_speaker: 3.0,
                scale_caption: 0.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: true,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        assert!(
            out.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn heun_alternating_cfg_produces_finite_output() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 4,
            method: SamplerMethod::Heun,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Alternating,
                scale_text: 2.0,
                scale_speaker: 3.0,
                scale_caption: 0.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: false,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        assert!(
            out.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    /// Single-step Heun's uses tt_base[1] (the endpoint t=0) for the corrector.
    /// Verify this edge case doesn't panic.
    #[test]
    fn heun_single_step_edge_case() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 1,
            method: SamplerMethod::Heun,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Independent,
                scale_text: 3.0,
                scale_speaker: 5.0,
                scale_caption: 0.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            use_context_kv_cache: true,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        let [b, s, _d] = out.dims();
        assert_eq!((b, s), (1, 6));
        assert!(
            out.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn heun_temporal_rescale_at_zero_endpoint_stays_finite() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 1,
            method: SamplerMethod::Heun,
            guidance: GuidanceConfig {
                scale_text: 0.0,
                scale_speaker: 0.0,
                scale_caption: 0.0,
                ..Default::default()
            },
            temporal_rescale: Some(super::super::params::TemporalRescaleConfig {
                k: 2.0,
                sigma: 1.0,
            }),
            use_context_kv_cache: false,
            ..Default::default()
        };

        let values = sample_euler_rf_cfg(&model, request, &params, &device)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(
            values.iter().all(|value| value.is_finite()),
            "Heun's t_next=0 corrector must not introduce NaNs"
        );
    }

    /// Heun CFG-window crossing: with min_t=0.5 and 4 steps, the corrector
    /// at the step that crosses the boundary should fall outside the CFG window
    /// (use_cfg_v2 = false while use_cfg_v1 = true).  Verify no panic.
    #[test]
    fn heun_cfg_window_crossing_no_panic() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 4,
            method: SamplerMethod::Heun,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Independent,
                scale_text: 3.0,
                scale_speaker: 5.0,
                scale_caption: 0.0,
                min_t: 0.5,
                max_t: 1.0,
            },
            use_context_kv_cache: true,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        assert!(
            out.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    /// Heun with speaker KV deactivation: v2 must use the post-deactivation cache.
    #[test]
    fn heun_speaker_kv_deactivation_no_panic() {
        let (model, request, device) = tiny_model_and_request();
        let params = SamplerParams {
            num_steps: 4,
            method: SamplerMethod::Heun,
            guidance: GuidanceConfig {
                scale_text: 0.0,
                scale_speaker: 0.0,
                scale_caption: 0.0,
                ..Default::default()
            },
            speaker_kv: Some(SpeakerKvConfig {
                scale: 2.0,
                max_layers: None,
                min_t: Some(0.5),
            }),
            use_context_kv_cache: true,
            ..Default::default()
        };
        let out = sample_euler_rf_cfg(&model, request, &params, &device).unwrap();
        assert!(
            out.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    /// Heun 20 steps produces same shape as Euler 40 steps with same NFE budget.
    #[test]
    fn heun_20_steps_same_shape_as_euler_40() {
        let (model, request, device) = tiny_model_and_request();
        let request2 = SamplingRequest {
            text_ids: request.text_ids.clone(),
            text_mask: request.text_mask.clone(),
            ref_latent: request.ref_latent.clone(),
            ref_mask: request.ref_mask.clone(),
            sequence_length: request.sequence_length,
            caption_ids: None,
            caption_mask: None,
            initial_noise: request.initial_noise.clone(),
        };

        let euler_out = sample_euler_rf_cfg(
            &model,
            request2,
            &SamplerParams {
                num_steps: 4,
                method: SamplerMethod::Euler,
                guidance: GuidanceConfig {
                    scale_text: 0.0,
                    scale_speaker: 0.0,
                    scale_caption: 0.0,
                    ..Default::default()
                },
                use_context_kv_cache: false,
                ..Default::default()
            },
            &device,
        )
        .unwrap();

        let heun_out = sample_euler_rf_cfg(
            &model,
            request,
            &SamplerParams {
                num_steps: 2, // half the steps = same NFE
                method: SamplerMethod::Heun,
                guidance: GuidanceConfig {
                    scale_text: 0.0,
                    scale_speaker: 0.0,
                    scale_caption: 0.0,
                    ..Default::default()
                },
                use_context_kv_cache: false,
                ..Default::default()
            },
            &device,
        )
        .unwrap();

        assert_eq!(
            euler_out.dims(),
            heun_out.dims(),
            "Heun and Euler output shapes must match"
        );
    }

    // ------------------------------------------------------------------
    // PLMS-4 unit tests
    // ------------------------------------------------------------------

    #[test]
    fn plms4_alternating_cfg_rejected() {
        use crate::config::CfgGuidanceMode;
        use crate::rf::params::{GuidanceConfig, SamplerParams};

        let params = SamplerParams {
            method: SamplerMethod::PLMS4,
            guidance: GuidanceConfig {
                mode: CfgGuidanceMode::Alternating,
                scale_text: 3.0,
                scale_speaker: 5.0,
                scale_caption: 2.0,
                min_t: 0.0,
                max_t: 1.0,
            },
            ..SamplerParams::default()
        };
        assert!(
            params.validate().is_err(),
            "PLMS4 + Alternating must be rejected"
        );
    }

    #[test]
    fn ab_extrapolate_order_progression() {
        use burn::tensor::Device;
        let device: Device = Default::default();

        let shape = [1usize, 4, 8];
        let mk = |val: f32| Tensor::<3>::full(shape, val, &device);

        let mut h: VecDeque<Tensor<3>> = VecDeque::new();

        // AB-1 (Euler): single entry
        h.push_front(mk(2.0));
        let result = ab_extrapolate(&h);
        let data: Vec<f32> = result.into_data().convert::<f32>().to_vec().unwrap();
        assert!(
            data.iter().all(|&v| (v - 2.0).abs() < 1e-5),
            "AB-1 must return v unchanged"
        );

        // AB-2: two entries
        h.push_front(mk(3.0)); // history: [3, 2]
        let result = ab_extrapolate(&h);
        let data: Vec<f32> = result.into_data().convert::<f32>().to_vec().unwrap();
        let expected = 3.0 * 1.5 + 2.0 * -0.5; // 3.5
        assert!(
            data.iter().all(|&v| (v - expected).abs() < 1e-4),
            "AB-2 expected {expected}, got {}",
            data[0]
        );

        // AB-3: three entries
        h.push_front(mk(4.0)); // history: [4, 3, 2]
        let result = ab_extrapolate(&h);
        let data: Vec<f32> = result.into_data().convert::<f32>().to_vec().unwrap();
        let expected = 4.0 * (23.0 / 12.0) + 3.0 * (-16.0 / 12.0) + 2.0 * (5.0 / 12.0);
        assert!(
            data.iter().all(|&v| (v - expected).abs() < 1e-4),
            "AB-3 expected {expected}, got {}",
            data[0]
        );

        // AB-4: four entries
        h.push_front(mk(5.0)); // history: [5, 4, 3, 2]
        let result = ab_extrapolate(&h);
        let data: Vec<f32> = result.into_data().convert::<f32>().to_vec().unwrap();
        let expected =
            5.0 * (55.0 / 24.0) + 4.0 * (-59.0 / 24.0) + 3.0 * (37.0 / 24.0) + 2.0 * (-9.0 / 24.0);
        assert!(
            data.iter().all(|&v| (v - expected).abs() < 1e-4),
            "AB-4 expected {expected}, got {}",
            data[0]
        );

        // AB-4 capped: pushing a 5th entry should still use only the 4 most recent
        h.push_front(mk(6.0));
        if h.len() > 4 {
            h.pop_back();
        }
        assert_eq!(h.len(), 4, "history must be capped at 4");
        let result = ab_extrapolate(&h);
        let data: Vec<f32> = result.into_data().convert::<f32>().to_vec().unwrap();
        let expected =
            6.0 * (55.0 / 24.0) + 5.0 * (-59.0 / 24.0) + 4.0 * (37.0 / 24.0) + 3.0 * (-9.0 / 24.0);
        assert!(
            data.iter().all(|&v| (v - expected).abs() < 1e-4),
            "AB-4 capped: expected {expected}, got {}",
            data[0]
        );
    }

    #[test]
    fn plms4_regime_reset_clears_history() {
        // Verify that when the regime key changes, history is cleared.
        // We test the logic directly (not via full inference) to keep this fast.
        use burn::tensor::Device;
        let device: Device = Default::default();

        let shape = [1usize, 4, 8];
        let mk = |val: f32| Tensor::<3>::full(shape, val, &device);

        let mut history: VecDeque<Tensor<3>> = VecDeque::with_capacity(4);
        let mut prev_regime = (true, true); // use_cfg=true, speaker_kv_active=true

        // Simulate 3 steps building up history
        for val in [1.0_f32, 2.0, 3.0] {
            let regime = (true, true);
            if regime != prev_regime {
                history.clear();
                prev_regime = regime;
            }
            history.push_front(mk(val));
            if history.len() > 4 {
                history.pop_back();
            }
        }
        assert_eq!(
            history.len(),
            3,
            "history should have 3 entries before regime change"
        );

        // Regime changes: speaker_kv_active becomes false
        let new_regime = (true, false);
        if new_regime != prev_regime {
            history.clear();
            prev_regime = new_regime;
        }
        assert_eq!(history.len(), 0, "history must be cleared on regime change");
        assert_eq!(prev_regime, (true, false));

        // Push one more entry after reset — should be AB-1 (Euler)
        history.push_front(mk(5.0));
        let result = ab_extrapolate(&history);
        let data: Vec<f32> = result.into_data().convert::<f32>().to_vec().unwrap();
        assert!(
            data.iter().all(|&v| (v - 5.0).abs() < 1e-5),
            "After reset, first step must use AB-1 (Euler): expected 5.0, got {}",
            data[0]
        );
    }
}
