use burn::tensor::{Bool, DType, Device, Tensor, TensorData};
use std::{
    collections::{BTreeMap, VecDeque},
    ops::ControlFlow,
    path::{Path, PathBuf},
    time::Instant,
};

use super::{
    CancellationToken, SynthesisRequest, Voice, audio, text::TextFrontend, voice::VoiceData,
};
use crate::codec::{DACVAE_HOP_LENGTH, DACVAE_SAMPLE_RATE, DacVaeDecoder, DecodeWindow};
use crate::inference::text as duration;
use crate::{
    AuxConditionInput, AuxConditionState, InferenceBuilder, IrodoriError, ModelConfig, Result,
    SamplingRequest, WgpuFloatPrecision, WgslInferenceEngine,
};

pub struct SynthesisModelPaths {
    pub checkpoint: PathBuf,
    pub codec: PathBuf,
    pub tokenizer: Option<PathBuf>,
}

struct CachedVoice {
    state: Tensor<3>,
    mask: Tensor<2, Bool>,
}

/// One model and decoder remain resident across requests. A mutable borrow
/// serializes request-specific sampling configuration. Use one engine owner
/// per device when relying on seeded device RNG.
pub struct SynthesisEngine {
    model: WgslInferenceEngine,
    codec: DacVaeDecoder,
    frontend: TextFrontend,
    config: ModelConfig,
    device: Device,
    precision: WgpuFloatPrecision,
    codec_path: PathBuf,
    codec_sha256: String,
    model_sha256: String,
    voices: BTreeMap<String, CachedVoice>,
}

#[derive(Debug, serde::Serialize)]
pub struct SynthesisReport {
    pub sample_rate: usize,
    pub samples: usize,
    pub segments: usize,
    pub seed: u64,
    pub first_audio_seconds: Option<f64>,
    pub total_seconds: f64,
}

impl SynthesisEngine {
    /// The caller configures the WGPU device and precision before loading.
    /// The codec always materializes as FP32, including FP16 DiT sessions.
    pub fn load(
        paths: SynthesisModelPaths,
        device: Device,
        precision: WgpuFloatPrecision,
    ) -> Result<Self> {
        if device.settings().float_dtype != precision.float_dtype() {
            return Err(IrodoriError::Config(
                "device precision does not match the requested model precision".into(),
            ));
        }
        let tokenizer = paths.tokenizer.unwrap_or_else(|| {
            paths
                .checkpoint
                .parent()
                .unwrap_or(Path::new("."))
                .join("tokenizer/tokenizer.json")
        });
        if !tokenizer.is_file() {
            return Err(IrodoriError::Tokenizer(format!(
                "tokenizer not found: {}",
                tokenizer.display()
            )));
        }
        let loaded = InferenceBuilder::new(device.clone())
            .with_timestep_condition_cache(crate::TimestepConditionCachePolicy::ConditionAndAdaLn)
            .load_weights_with_float_dtype(&paths.checkpoint, precision.tensor_dtype())?;
        let config = loaded.model_config().clone();
        if config.latent_dim != 32 || config.latent_patch_size != 1 {
            return Err(IrodoriError::Config(
                "text synthesis requires the released v4/v4.1 32-channel codec geometry".into(),
            ));
        }
        if config.caption_tokenizer_repo() != config.text_tokenizer_repo
            || config.caption_add_bos() != config.text_add_bos
        {
            return Err(IrodoriError::Config(
                "this text frontend requires the released shared text/caption tokenizer".into(),
            ));
        }
        let frontend = TextFrontend::load(&tokenizer, &config)?;
        let model = loaded.with_default_sampling().build_wgsl();
        let mut codec =
            crate::codec::load_decoder_with_float_dtype(&paths.codec, &device, DType::F32)?;
        codec.prepare_for_wgsl();
        let model_sha256 = crate::sha256_file(&paths.checkpoint)?;
        let codec_sha256 = super::voice::codec_identity(&paths.codec)?;
        Ok(Self {
            model,
            codec,
            frontend,
            config,
            device,
            precision,
            codec_path: paths.codec,
            codec_sha256,
            model_sha256,
            voices: BTreeMap::new(),
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    pub fn precision(&self) -> WgpuFloatPrecision {
        self.precision
    }
    pub fn voice_names(&self) -> impl Iterator<Item = &str> {
        std::iter::once("none").chain(self.voices.keys().map(String::as_str))
    }
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        self.frontend.count(crate::normalize_text(text).trim())
    }

    pub fn token_counter(&self) -> super::TokenCounter {
        super::TokenCounter(self.frontend.clone())
    }

    /// Drain this owner's stream before transferring ownership to a transport
    /// thread, or ending that thread after cancellation/metadata-only work.
    #[cfg(any(feature = "worker", feature = "http-server"))]
    pub(super) fn synchronize(&self) -> Result<()> {
        self.device
            .sync()
            .map_err(|error| IrodoriError::Execution(error.to_string()))
    }

    /// Information shared by the optional process and HTTP transports.
    pub fn info(&self) -> serde_json::Value {
        serde_json::json!({
            "id": "irodori", "name": "Irodori-TTS (WGSL)",
            "architecture": "irodori-tts", "task": "synthesis", "sample_rate": DACVAE_SAMPLE_RATE,
            "flow_parameterization": self.config.flow_parameterization,
            "precision": self.precision, "codec_precision": "fp32",
            "languages": ["ja"], "channels": 1,
            "default_steps": if self.config.flow_parameterization == crate::FlowParameterization::Meanflow {4} else {40},
            "max_text_tokens": self.config.max_text_len.unwrap_or(256),
            "max_instruction_tokens": self.config.max_caption_len.unwrap_or(512),
            "voices": self.voice_names().map(|name| serde_json::json!({
                "name": name, "language": "ja", "gender": "", "description": "",
            })).collect::<Vec<_>>(),
            "voice_files": self.config.use_speaker_condition(),
            "voice_codec": self.codec_sha256, "incremental": false,
            "streaming": true, "formats": ["wav", "pcm"],
        })
    }

    pub fn load_voice(&self, path: &Path) -> Result<Voice> {
        Voice::load(
            path,
            &self.codec_sha256,
            &self.model_sha256,
            self.config.speaker_dim.unwrap_or(self.config.text_dim),
        )
    }

    /// Encode each reference at its own loudness, then concatenate its latents.
    pub fn create_voice(&self, recordings: &[PathBuf], target_lufs: Option<f64>) -> Result<Voice> {
        if recordings.is_empty() || !self.config.use_speaker_condition() {
            return Err(IrodoriError::Config(
                "voice creation needs recordings and a speaker-conditioned model".into(),
            ));
        }
        let max_seconds = self.config.ref_max_seconds.unwrap_or(120.0).min(120.0);
        let mut prepared = Vec::with_capacity(recordings.len());
        let mut total_seconds = 0.0;
        for path in recordings {
            let (mut mono, rate) = audio::read_wav(path, max_seconds)?;
            total_seconds += mono.len() as f64 / f64::from(rate);
            if total_seconds > max_seconds {
                return Err(IrodoriError::Audio(format!(
                    "references exceed {max_seconds} seconds together"
                )));
            }
            audio::normalize_loudness(&mut mono, rate, target_lufs)?;
            prepared.push(audio::resample(&mono, rate, DACVAE_SAMPLE_RATE as u32)?);
        }
        let mut encoder =
            crate::codec::load_codec_with_float_dtype(&self.codec_path, &self.device, DType::F32)?;
        encoder.prepare_encoder_for_wgsl();
        let mut values = Vec::new();
        for mut audio in prepared {
            let count = audio.len();
            let padding = (DACVAE_HOP_LENGTH - count % DACVAE_HOP_LENGTH) % DACVAE_HOP_LENGTH;
            if padding >= count {
                return Err(IrodoriError::Audio(
                    "reference is too short for codec reflection padding".into(),
                ));
            }
            for index in 0..padding {
                audio.push(audio[count - 2 - index]);
            }
            let count = audio.len();
            let tensor = Tensor::from_data(
                TensorData::new(audio, [1, 1, count]),
                (&self.device, DType::F32),
            );
            let frames = count / DACVAE_HOP_LENGTH;
            for first in (0..frames).step_by(96) {
                let last = (first + 96).min(frames);
                let from = first.saturating_sub(8);
                let to = (last + 8).min(frames);
                let latent = encoder
                    .encode_wgsl(tensor.clone().narrow(
                        2,
                        from * DACVAE_HOP_LENGTH,
                        (to - from) * DACVAE_HOP_LENGTH,
                    ))
                    .narrow(1, first - from, last - first)
                    .into_data()
                    .convert::<f32>()
                    .try_to_vec::<f32>()
                    .map_err(|e| {
                        IrodoriError::Audio(format!("reference encoding readback failed: {e}"))
                    })?;
                if latent.iter().any(|v| !v.is_finite()) {
                    return Err(IrodoriError::Audio(
                        "reference encoder returned non-finite values".into(),
                    ));
                }
                values.extend(latent);
            }
        }
        let frames = values.len() / 32;
        if frames > (max_seconds * 25.0).ceil() as usize {
            return Err(IrodoriError::Audio(
                "padded references exceed the model's latent limit".into(),
            ));
        }
        Ok(Voice {
            data: VoiceData::Latent { values, frames },
            codec_sha256: self.codec_sha256.clone(),
            model_sha256: None,
        })
    }

    pub fn add_voice(&mut self, name: &str, voice: Voice) -> Result<()> {
        if name.is_empty() || name == "none" || name.len() > 128 {
            return Err(IrodoriError::Config(
                "voice name must be nonempty, at most 128 bytes, and not 'none'".into(),
            ));
        }
        if voice.codec_sha256 != self.codec_sha256
            || voice
                .model_sha256
                .as_ref()
                .is_some_and(|sha| sha != &self.model_sha256)
        {
            return Err(IrodoriError::Config(
                "voice model/codec identity mismatch".into(),
            ));
        }
        let state = match voice.data {
            VoiceData::Embedding {
                values,
                tokens,
                dim,
            } => Tensor::from_data(
                TensorData::new(values, [1, tokens, dim]),
                (&self.device, self.precision.tensor_dtype()),
            ),
            VoiceData::Latent { values, frames } => {
                let latent = Tensor::from_data(
                    TensorData::new(values, [1, frames, 32]),
                    (&self.device, self.precision.tensor_dtype()),
                );
                let mask = Tensor::from_data(
                    TensorData::new(vec![true; frames], [1, frames]),
                    &self.device,
                );
                let text = self.frontend.encode("", false, &self.device)?;
                let encoded = self.model.encode_conditions(
                    text.ids,
                    text.mask,
                    AuxConditionInput::Speaker {
                        ref_latent: latent,
                        ref_mask: mask,
                    },
                )?;
                encoded
                    .aux
                    .and_then(|aux| match aux {
                        AuxConditionState::Speaker { state, .. } => Some(state),
                        AuxConditionState::Both { speaker_state, .. } => Some(speaker_state),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        IrodoriError::Config("reference encoding produced no speaker state".into())
                    })?
            }
        };
        let [batch, tokens, _] = state.dims();
        let mask = Tensor::from_data(
            TensorData::new(vec![true; batch * tokens], [batch, tokens]),
            &self.device,
        );
        self.voices
            .insert(name.to_owned(), CachedVoice { state, mask });
        Ok(())
    }

    pub fn remove_voice(&mut self, name: &str) -> bool {
        self.voices.remove(name).is_some()
    }

    pub fn synthesize(
        &mut self,
        request: &SynthesisRequest,
    ) -> Result<(Vec<f32>, SynthesisReport)> {
        let mut samples = Vec::new();
        let report = self.synthesize_stream(request, &CancellationToken::default(), |chunk| {
            samples.extend_from_slice(chunk);
            Ok(ControlFlow::Continue(()))
        })?;
        Ok((samples, report))
    }

    pub fn synthesize_stream(
        &mut self,
        request: &SynthesisRequest,
        cancellation: &CancellationToken,
        mut sink: impl FnMut(&[f32]) -> Result<ControlFlow<()>>,
    ) -> Result<SynthesisReport> {
        let started = Instant::now();
        request.validate(self.config.flow_parameterization)?;
        cancellation.check()?;
        let speaker =
            if request.voice == "none" {
                None
            } else {
                Some(self.voices.get(&request.voice).ok_or_else(|| {
                    IrodoriError::Config(format!("unknown voice: {}", request.voice))
                })?)
            };
        let speaker = speaker.map(|v| (v.state.clone(), v.mask.clone()));
        let has_speaker = speaker.is_some();
        let caption_text = request.instructions.trim();
        if !caption_text.is_empty() && !self.config.use_caption_condition {
            return Err(IrodoriError::Config(
                "model does not support voice instructions".into(),
            ));
        }
        let caption = self.frontend.encode(caption_text, true, &self.device)?;
        let has_caption = caption.count > 0;
        let params =
            request.sampler(self.config.flow_parameterization, has_speaker, has_caption)?;
        self.model.set_sampling_params(params)?;
        let seed = request.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |time| time.as_nanos() as u64)
        });
        let mut report = SynthesisReport {
            sample_rate: DACVAE_SAMPLE_RATE,
            samples: 0,
            segments: 0,
            seed,
            first_audio_seconds: None,
            total_seconds: 0.0,
        };
        let mut pending = VecDeque::from(self.frontend.split(&request.text)?);
        while let Some(text) = pending.pop_front() {
            cancellation.check()?;
            let tokens = self.frontend.encode(&text, false, &self.device)?;
            let aux = if has_caption {
                AuxConditionInput::Caption {
                    ids: caption.ids.clone(),
                    mask: caption.mask.clone(),
                }
            } else {
                AuxConditionInput::None
            };
            let mut condition =
                self.model
                    .encode_conditions(tokens.ids.clone(), tokens.mask.clone(), aux)?;
            if let Some((state, mask)) = &speaker {
                condition.aux = Some(match condition.aux.as_ref().and_then(|aux| aux.caption()) {
                    Some((caption_state, caption_mask)) => AuxConditionState::Both {
                        speaker_state: state.clone(),
                        speaker_mask: mask.clone(),
                        caption_state: caption_state.clone(),
                        caption_mask: caption_mask.clone(),
                    },
                    None => AuxConditionState::Speaker {
                        state: state.clone(),
                        mask: mask.clone(),
                    },
                });
            }
            let frames = if let Some(seconds) = request.seconds {
                (seconds / request.speed * 25.0).ceil().clamp(13.0, 750.0) as usize
            } else {
                let features = duration::build_duration_features(
                    &text,
                    tokens.count,
                    self.config.max_text_len.unwrap_or(256),
                    has_speaker,
                    self.config.duration_aux_dim,
                )?;
                let features = Tensor::from_data(
                    TensorData::new(features, [1, self.config.duration_aux_dim]),
                    &self.device,
                );
                let present_speaker = Tensor::from_data([has_speaker], &self.device);
                let present_caption = Tensor::from_data([has_caption], &self.device);
                let predicted = f64::from(
                    self.model
                        .predict_duration_log_frames(
                            &condition,
                            features,
                            present_speaker,
                            present_caption,
                        )?
                        .into_data()
                        .convert::<f32>()
                        .try_to_vec::<f32>()
                        .map_err(|e| {
                            IrodoriError::Audio(format!("duration readback failed: {e}"))
                        })?[0],
                )
                .exp_m1();
                let scaled = predicted * request.duration_scale / request.speed;
                if !scaled.is_finite() || scaled < 0.0 {
                    return Err(IrodoriError::Audio(
                        "duration predictor returned an invalid length".into(),
                    ));
                }
                if scaled > 750.0 {
                    let (first, second) = super::text::split_piece(&text)?;
                    pending.push_front(second.to_owned());
                    pending.push_front(first.to_owned());
                    continue;
                }
                scaled.round_ties_even().max(13.0) as usize
            };
            self.device.seed(seed.wrapping_add(report.segments as u64));
            let request_tensors = SamplingRequest {
                text_ids: tokens.ids,
                text_mask: tokens.mask,
                ref_latent: None,
                ref_mask: None,
                caption_ids: None,
                caption_mask: None,
                sequence_length: frames,
                initial_noise: None,
            };
            let latent = self
                .model
                .sample_conditioned(request_tensors, condition, cancellation)?;
            cancellation.check()?;
            let mut limit = if let Some(seconds) = request.seconds {
                ((seconds / request.speed).clamp(0.5, 30.0) * DACVAE_SAMPLE_RATE as f64).round()
                    as usize
            } else {
                frames * DACVAE_HOP_LENGTH
            };
            if !request.keep_tail {
                let data = latent
                    .clone()
                    .into_data()
                    .convert::<f32>()
                    .try_to_vec::<f32>()
                    .map_err(|e| IrodoriError::Audio(format!("latent readback failed: {e}")))?;
                let end = duration::find_flattening_point(
                    &data,
                    frames,
                    32,
                    request.tail_window_size,
                    request.tail_std_threshold,
                    request.tail_mean_threshold,
                );
                if end > 0 {
                    limit = limit.min(end * DACVAE_HOP_LENGTH);
                }
            }
            let decoded = self.codec.decode_stream(
                latent,
                Some(limit),
                DecodeWindow::default(),
                |chunk| {
                    cancellation.check()?;
                    report
                        .first_audio_seconds
                        .get_or_insert_with(|| started.elapsed().as_secs_f64());
                    report.samples += chunk.len();
                    sink(chunk)
                },
            )?;
            if decoded.cancelled {
                return Err(IrodoriError::Cancelled);
            }
            report.segments += 1;
            if !pending.is_empty() && text.trim_end().ends_with(['。', '!', '?', '.']) {
                cancellation.check()?;
                let silence = vec![0.0; DACVAE_SAMPLE_RATE * 9 / 10];
                if sink(&silence)?.is_break() {
                    return Err(IrodoriError::Cancelled);
                }
                report.samples += silence.len();
            }
        }
        report.total_seconds = started.elapsed().as_secs_f64();
        Ok(report)
    }
}
