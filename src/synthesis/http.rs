//! Optional OpenAI-style speech HTTP transport using a single resident model.

use super::{CancellationToken, SynthesisEngine, SynthesisRequest, audio};
use crate::{FlowParameterization, IrodoriError, Result};
use serde_json::{Value, json};
use std::{
    io::{self, Cursor, Read},
    net::SocketAddr,
    ops::ControlFlow,
    sync::mpsc,
    time::Duration,
};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

const MAX_BODY: u64 = 1_100_000;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Wav,
    Pcm,
}
struct Job {
    http: Request,
    synthesis: SynthesisRequest,
    format: Format,
}

fn header(name: &'static str, value: &'static str) -> Header {
    Header::from_bytes(name, value).expect("static HTTP header is valid")
}
fn respond_json(request: Request, status: u16, value: Value) -> io::Result<()> {
    request.respond(
        Response::from_string(value.to_string())
            .with_status_code(status)
            .with_header(header("Content-Type", "application/json")),
    )
}
fn error_body(message: impl std::fmt::Display) -> Value {
    json!({"error":{"message":message.to_string(),"type":"invalid_request_error","code":null}})
}
fn parse(
    mut value: Value,
    flow: FlowParameterization,
    voices: &[String],
) -> Result<(SynthesisRequest, Format)> {
    let fields = value
        .as_object_mut()
        .ok_or_else(|| IrodoriError::Config("expected a JSON object".into()))?;
    fields.retain(|_, value| !value.is_null());
    if fields
        .remove("model")
        .is_some_and(|model| model != "irodori")
    {
        return Err(IrodoriError::Config("model must be 'irodori'".into()));
    }
    if fields.contains_key("text") {
        return Err(IrodoriError::Config("use 'input' for speech text".into()));
    }
    let input = fields
        .remove("input")
        .ok_or_else(|| IrodoriError::MissingInput("input is required".into()))?;
    fields.insert("text".into(), input);
    let format = match fields.remove("response_format") {
        None => Format::Wav,
        Some(Value::String(value)) if value == "wav" => Format::Wav,
        Some(Value::String(value)) if value == "pcm" => Format::Pcm,
        _ => {
            return Err(IrodoriError::Config(
                "response_format must be 'wav' or 'pcm'".into(),
            ));
        }
    };
    let request: SynthesisRequest = serde_json::from_value(value)?;
    request.validate(flow)?;
    if !voices.contains(&request.voice) {
        return Err(IrodoriError::Config(format!(
            "unknown voice: {}",
            request.voice
        )));
    }
    Ok((request, format))
}

/// Serve until the process exits. Bind to loopback unless remote access is
/// intended. WAV is a complete response; PCM streams mono s16le at 48 kHz.
pub fn serve(engine: SynthesisEngine, address: SocketAddr) -> Result<()> {
    serve_until(engine, address, &CancellationToken::default())
}

/// Stop accepting on cancellation, then drain already accepted requests.
/// Queue capacity is eight; health and model queries do not wait for inference.
pub fn serve_until(
    mut engine: SynthesisEngine,
    address: SocketAddr,
    shutdown: &CancellationToken,
) -> Result<()> {
    engine.synchronize()?;
    let server = Server::http(address)
        .map_err(|e| IrodoriError::Config(format!("HTTP bind failed: {e}")))?;
    tracing::info!(%address, "speech server listening");
    let info = engine.info();
    let flow = engine.config().flow_parameterization;
    let voices = engine.voice_names().map(str::to_owned).collect::<Vec<_>>();
    let (sender, receiver) = mpsc::sync_channel::<Job>(8);
    std::thread::scope(|scope| -> Result<()> {
        let worker = scope.spawn(move || {
            while let Ok(job) = receiver.recv() {
                if let Err(error) = handle(&mut engine, job) {
                    tracing::warn!(%error, "speech response failed");
                }
            }
            engine.synchronize()
        });
        let result = (|| -> Result<()> {
            while !shutdown.is_cancelled() {
                let Some(mut request) = server.recv_timeout(Duration::from_millis(100))? else {
                    continue;
                };
                match (request.method(), request.url()) {
                    (&Method::Get, "/health") => {
                        let _ = respond_json(request, 200, json!({"status":"ok"}));
                    }
                    (&Method::Get, "/v1/models") => {
                        let _ = respond_json(
                            request,
                            200,
                            json!({"object":"list","data":[{"id":"irodori","object":"model","owned_by":"local","capabilities":info}]}),
                        );
                    }
                    (&Method::Post, "/v1/audio/speech") => {
                        if request
                            .body_length()
                            .is_some_and(|len| len as u64 > MAX_BODY)
                        {
                            let _ = respond_json(
                                request,
                                413,
                                error_body("request exceeds 1100000 bytes"),
                            );
                            continue;
                        }
                        let mut bytes = Vec::new();
                        if let Err(error) = request
                            .as_reader()
                            .take(MAX_BODY + 1)
                            .read_to_end(&mut bytes)
                        {
                            let _ = respond_json(request, 400, error_body(error));
                            continue;
                        }
                        if bytes.len() as u64 > MAX_BODY {
                            let _ = respond_json(
                                request,
                                413,
                                error_body("request exceeds 1100000 bytes"),
                            );
                            continue;
                        }
                        let parsed = serde_json::from_slice(&bytes)
                            .map_err(IrodoriError::from)
                            .and_then(|value| parse(value, flow, &voices));
                        let (synthesis, format) = match parsed {
                            Ok(value) => value,
                            Err(error) => {
                                let _ = respond_json(request, 400, error_body(error));
                                continue;
                            }
                        };
                        if let Err(error) = sender.try_send(Job {
                            http: request,
                            synthesis,
                            format,
                        }) {
                            let job = match error {
                                mpsc::TrySendError::Full(job)
                                | mpsc::TrySendError::Disconnected(job) => job,
                            };
                            let _ = respond_json(
                                job.http,
                                503,
                                error_body("synthesis queue unavailable"),
                            );
                        }
                    }
                    _ => {
                        let _ = respond_json(request, 404, error_body("unknown endpoint"));
                    }
                }
            }
            Ok(())
        })();
        drop(sender);
        let completed = worker
            .join()
            .map_err(|_| IrodoriError::Config("HTTP inference worker panicked".into()))?;
        result.and(completed)
    })
}

enum Piece {
    Audio(Vec<u8>),
    Error(String),
}
struct PcmReader {
    receiver: mpsc::Receiver<Piece>,
    current: Cursor<Vec<u8>>,
    cancellation: CancellationToken,
}
impl Read for PcmReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        loop {
            let count = self.current.read(buffer)?;
            if count > 0 {
                return Ok(count);
            }
            match self.receiver.recv() {
                Ok(Piece::Audio(bytes)) => self.current = Cursor::new(bytes),
                Ok(Piece::Error(message)) => return Err(io::Error::other(message)),
                Err(_) => return Ok(0),
            }
        }
    }
}
impl Drop for PcmReader {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

fn handle(engine: &mut SynthesisEngine, job: Job) -> Result<()> {
    if job.format == Format::Wav {
        return match engine.synthesize(&job.synthesis) {
            Ok((samples, _)) => {
                let mut wav = Vec::new();
                audio::write_wav(&mut wav, &samples, 48_000)?;
                job.http.respond(
                    Response::from_data(wav).with_header(header("Content-Type", "audio/wav")),
                )?;
                Ok(())
            }
            Err(error) => {
                respond_json(job.http, 400, error_body(error))?;
                Ok(())
            }
        };
    }
    let (sender, receiver) = mpsc::sync_channel::<Piece>(2);
    let cancellation = CancellationToken::default();
    std::thread::scope(|scope| -> Result<()> {
        let token = cancellation.clone();
        let response = scope.spawn(move || -> io::Result<()> {
            // Delay status 200 until the first valid PCM chunk is available.
            match receiver.recv() {
                Ok(Piece::Audio(first)) => {
                    let reader = PcmReader {
                        receiver,
                        current: Cursor::new(first),
                        cancellation: token,
                    };
                    job.http.respond(Response::new(
                        StatusCode(200),
                        vec![
                            header("Content-Type", "audio/pcm"),
                            header("X-Sample-Rate", "48000"),
                        ],
                        reader,
                        None,
                        None,
                    ))
                }
                Ok(Piece::Error(error)) => respond_json(job.http, 400, error_body(error)),
                Err(error) => respond_json(job.http, 500, error_body(error)),
            }
        });
        let result = engine.synthesize_stream(&job.synthesis, &cancellation, |chunk| {
            sender
                .send(Piece::Audio(audio::pcm16(chunk)))
                .map_err(|_| IrodoriError::Cancelled)?;
            Ok(ControlFlow::Continue(()))
        });
        if let Err(error) = result {
            let _ = sender.send(Piece::Error(error.to_string()));
        }
        drop(sender);
        response
            .join()
            .map_err(|_| IrodoriError::Config("HTTP response thread panicked".into()))??;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn speech_json_rejects_ignored_options_and_maps_openai_fields() {
        let voices = vec!["none".into()];
        let (request, format) = parse(
            json!({"model":"irodori","input":"こんにちは","response_format":"pcm","speed":1.5}),
            FlowParameterization::Meanflow,
            &voices,
        )
        .unwrap();
        assert_eq!(request.text, "こんにちは");
        assert_eq!(format, Format::Pcm);
        for body in [
            json!({"input":"x","response_format":"mp3"}),
            json!({"input":"x","unknown":1}),
            json!({"input":"x","voice":"missing"}),
            json!({"input":"x","cfg_scale_text":3}),
            json!({"input":"x","response_format":2}),
        ] {
            assert!(parse(body, FlowParameterization::Meanflow, &voices).is_err());
        }
    }
    #[test]
    fn pcm_reader_propagates_failure_and_cancels_on_disconnect() {
        let (tx, rx) = mpsc::channel();
        let token = CancellationToken::default();
        tx.send(Piece::Audio(vec![1, 2, 3])).unwrap();
        tx.send(Piece::Error("decode failed".into())).unwrap();
        let mut reader = PcmReader {
            receiver: rx,
            current: Cursor::new(Vec::new()),
            cancellation: token.clone(),
        };
        let mut bytes = [0; 8];
        assert_eq!(reader.read(&mut bytes).unwrap(), 3);
        assert!(reader.read(&mut bytes).is_err());
        drop(reader);
        assert!(token.is_cancelled());
    }
}
