//! Bounded JSON Lines synthesis worker. The model has one owner; metadata
//! queries and cancellation remain responsive during denoising.

use super::{CancellationToken, SynthesisEngine, SynthesisRequest, audio};
use crate::{IrodoriError, Result};
use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{BufRead, Read, Write},
    ops::ControlFlow,
    path::PathBuf,
    sync::{Mutex, mpsc},
};

const MAX_LINE: usize = 1_100_000;
const QUEUE_CAPACITY: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Queued,
    Running,
    AddingVoice,
}
struct Active {
    generation: u64,
    status: Status,
    token: CancellationToken,
}
#[derive(Default)]
struct Registry {
    generation: u64,
    active: BTreeMap<String, Active>,
}
struct Job {
    id: String,
    generation: u64,
    token: CancellationToken,
    action: Action,
}
enum Action {
    Synthesize(Box<SynthesisRequest>),
    AddVoice { name: String, path: PathBuf },
}

fn lock<T>(mutex: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>> {
    mutex
        .lock()
        .map_err(|_| IrodoriError::Config("worker state was poisoned".into()))
}
fn emit(output: &Mutex<impl Write>, value: &Value) -> Result<()> {
    let mut output = lock(output)?;
    serde_json::to_writer(&mut *output, value)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}
fn error(id: Option<&str>, code: &str, message: impl std::fmt::Display) -> Value {
    let mut value =
        json!({"type":"error", "error":{"code":code,"option":null,"message":message.to_string()}});
    if let Some(id) = id {
        value["id"] = id.into();
    }
    value
}
fn finish(
    registry: &Mutex<Registry>,
    output: &Mutex<impl Write>,
    job: &Job,
    value: Value,
) -> Result<()> {
    let mut registry = lock(registry)?;
    if registry
        .active
        .get(&job.id)
        .is_some_and(|entry| entry.generation == job.generation)
    {
        emit(output, &value)?;
        registry.active.remove(&job.id);
    }
    Ok(())
}
fn cancel(registry: &Mutex<Registry>, output: &Mutex<impl Write>, id: &str) -> Result<()> {
    let mut registry = lock(registry)?;
    if let Some(active) = registry.active.get(id) {
        match active.status {
            Status::Queued => {
                active.token.cancel();
                emit(output, &json!({"type":"cancelled", "id":id}))?;
                registry.active.remove(id);
            }
            Status::Running => active.token.cancel(),
            Status::AddingVoice => {}
        }
    }
    Ok(())
}

/// Serve the speech.cpp protocol-3 synthesis vocabulary. No GPU work runs on
/// the input reader. Eight jobs may wait; overload receives an explicit error.
/// EOF drains accepted jobs. `add_voice` is not cancellable after it starts.
/// Logs must be directed to stderr by the embedding application.
pub fn run(
    mut engine: SynthesisEngine,
    mut input: impl BufRead,
    output: impl Write + Send,
) -> Result<()> {
    let output = Mutex::new(output);
    if let Err(cause) = engine.synchronize() {
        emit(
            &output,
            &json!({"type":"fatal","error":{"code":"execution","option":null,"message":cause.to_string()}}),
        )?;
        return Err(cause);
    }
    let registry = Mutex::new(Registry::default());
    let info = Mutex::new(engine.info());
    let counter = engine.token_counter();
    let (sender, receiver) = mpsc::sync_channel::<Job>(QUEUE_CAPACITY);
    emit(
        &output,
        &json!({"type":"ready", "protocol":3, "version":env!("CARGO_PKG_VERSION"), "defaults":{}, "model":lock(&info)?.clone()}),
    )?;
    std::thread::scope(|scope| -> Result<()> {
        let worker = {
            let output = &output;
            let registry = &registry;
            let info = &info;
            scope.spawn(move || -> Result<()> {
                let result = (|| -> Result<()> {
                    while let Ok(job) = receiver.recv() {
                        {
                            let mut registry = lock(registry)?;
                            let Some(active) = registry
                                .active
                                .get_mut(&job.id)
                                .filter(|entry| entry.generation == job.generation)
                            else {
                                continue;
                            };
                            active.status = match job.action {
                                Action::Synthesize(_) => Status::Running,
                                Action::AddVoice { .. } => Status::AddingVoice,
                            };
                        }
                        let result = match &job.action {
                            Action::Synthesize(request) => {
                                let mut seq = 0;
                                engine
                                    .synthesize_stream(request, &job.token, |chunk| {
                                        emit(
                                            output,
                                            &json!({"type":"chunk", "id":job.id, "seq":seq, "pcm":STANDARD.encode(audio::pcm16(chunk))}),
                                        )?;
                                        seq += 1;
                                        Ok(ControlFlow::Continue(()))
                                    })
                                    .map(|report| {
                                        json!({"type":"end", "id":job.id, "seed":report.seed, "samples":report.samples, "stop":"complete"})
                                    })
                            }
                            Action::AddVoice { name, path } => (|| {
                                let voice = if path
                                    .extension()
                                    .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
                                {
                                    engine.create_voice(std::slice::from_ref(path), Some(-16.0))?
                                } else {
                                    engine.load_voice(path)?
                                };
                                engine.add_voice(name, voice)?;
                                *lock(info)? = engine.info();
                                Ok(json!({"type":"end", "id":job.id}))
                            })(),
                        };
                        let terminal = match result {
                            Ok(value) => value,
                            Err(IrodoriError::Cancelled) => {
                                json!({"type":"cancelled", "id":job.id})
                            }
                            Err(cause) => error(Some(&job.id), "invalid_argument", cause),
                        };
                        finish(registry, output, &job, terminal)?;
                    }
                    Ok(())
                })();
                let completed = engine.synchronize();
                result.and(completed)
            })
        };
        let read_result = (|| -> Result<()> {
            loop {
                let mut line = Vec::new();
                let read = (&mut input)
                    .take((MAX_LINE + 1) as u64)
                    .read_until(b'\n', &mut line)?;
                if read == 0 {
                    break;
                }
                if line.len() > MAX_LINE {
                    // Drain an oversized record without retaining its contents.
                    if line.last() != Some(&b'\n') {
                        input.skip_until(b'\n')?;
                    }
                    emit(
                        &output,
                        &error(None, "out_of_range", "line exceeds 1100000 bytes"),
                    )?;
                    continue;
                }
                let mut fields = match serde_json::from_slice::<Value>(&line) {
                    Ok(Value::Object(fields)) => fields,
                    Ok(_) => {
                        emit(
                            &output,
                            &error(None, "invalid_argument", "expected a JSON object"),
                        )?;
                        continue;
                    }
                    Err(cause) => {
                        emit(&output, &error(None, "invalid_argument", cause))?;
                        continue;
                    }
                };
                fields.retain(|_, value| !value.is_null());
                let id = match fields.remove("id") {
                    Some(Value::String(id)) if !id.is_empty() && id.len() <= 256 => id,
                    _ => {
                        emit(
                            &output,
                            &error(
                                None,
                                "invalid_argument",
                                "id must be a nonempty string of at most 256 bytes",
                            ),
                        )?;
                        continue;
                    }
                };
                let kind = fields.remove("type");
                if kind.as_ref().and_then(Value::as_str) == Some("cancel") {
                    if fields.is_empty() {
                        cancel(&registry, &output, &id)?;
                    } else {
                        emit(
                            &output,
                            &error(None, "invalid_argument", "cancel only accepts type and id"),
                        )?;
                    }
                    continue;
                }
                if lock(&registry)?.active.contains_key(&id) {
                    emit(
                        &output,
                        &error(None, "invalid_argument", "id is already in flight"),
                    )?;
                    continue;
                }
                match kind.as_ref().and_then(Value::as_str) {
                    Some("info") if fields.is_empty() => {
                        emit(
                            &output,
                            &json!({"type":"end", "id":id, "defaults":{}, "model":lock(&info)?.clone()}),
                        )?;
                        continue;
                    }
                    Some("count_tokens")
                        if fields.len() == 1
                            && fields.get("text").is_some_and(Value::is_string) =>
                    {
                        let text = fields["text"].as_str().expect("checked string");
                        let value = match counter.count(text) {
                            Ok(tokens) => json!({"type":"end", "id":id, "tokens":tokens}),
                            Err(cause) => error(Some(&id), "invalid_argument", cause),
                        };
                        emit(&output, &value)?;
                        continue;
                    }
                    _ => {}
                }
                let action = match kind.as_ref().and_then(Value::as_str) {
                    Some("synthesize") => serde_json::from_value(Value::Object(fields))
                        .map(|request| Action::Synthesize(Box::new(request)))
                        .map_err(|e| e.to_string()),
                    Some("add_voice") => {
                        #[derive(serde::Deserialize)]
                        #[serde(deny_unknown_fields)]
                        struct AddVoice {
                            name: String,
                            path: PathBuf,
                        }
                        serde_json::from_value::<AddVoice>(Value::Object(fields))
                            .map(|v| Action::AddVoice {
                                name: v.name,
                                path: v.path,
                            })
                            .map_err(|e| e.to_string())
                    }
                    Some("info" | "count_tokens") => Err("invalid query members".into()),
                    _ => {
                        emit(
                            &output,
                            &error(Some(&id), "unsupported", "unknown synthesis message type"),
                        )?;
                        continue;
                    }
                };
                let action = match action {
                    Ok(action) => action,
                    Err(cause) => {
                        emit(&output, &error(Some(&id), "invalid_argument", cause))?;
                        continue;
                    }
                };
                let job = {
                    let mut registry = lock(&registry)?;
                    registry.generation = registry.generation.checked_add(1).ok_or_else(|| {
                        IrodoriError::Config("worker generation exhausted".into())
                    })?;
                    let generation = registry.generation;
                    let token = CancellationToken::default();
                    registry.active.insert(
                        id.clone(),
                        Active {
                            generation,
                            status: Status::Queued,
                            token: token.clone(),
                        },
                    );
                    Job {
                        id,
                        generation,
                        token,
                        action,
                    }
                };
                if let Err(cause) = sender.try_send(job) {
                    let (job, message) = match cause {
                        mpsc::TrySendError::Full(job) => (job, "worker queue is full"),
                        mpsc::TrySendError::Disconnected(job) => (job, "worker stopped"),
                    };
                    finish(
                        &registry,
                        &output,
                        &job,
                        error(Some(&job.id), "unavailable", message),
                    )?;
                }
            }
            Ok(())
        })();
        if read_result.is_err() {
            for active in lock(&registry)?.active.values() {
                active.token.cancel();
            }
        }
        drop(sender);
        let work_result = worker
            .join()
            .map_err(|_| IrodoriError::Config("synthesis worker panicked".into()))?;
        read_result.and(work_result)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn queued_cancel_terminates_once_and_reused_ids_ignore_old_completion() {
        let output = Mutex::new(Vec::new());
        let registry = Mutex::new(Registry::default());
        let token = CancellationToken::default();
        let job = Job {
            id: "a".into(),
            generation: 1,
            token: token.clone(),
            action: Action::Synthesize(Box::default()),
        };
        lock(&registry).unwrap().active.insert(
            "a".into(),
            Active {
                generation: 1,
                status: Status::Queued,
                token: token.clone(),
            },
        );
        cancel(&registry, &output, "a").unwrap();
        cancel(&registry, &output, "a").unwrap();
        assert!(token.is_cancelled());
        lock(&registry).unwrap().active.insert(
            "a".into(),
            Active {
                generation: 2,
                status: Status::Queued,
                token: CancellationToken::default(),
            },
        );
        finish(&registry, &output, &job, json!({"type":"end","id":"a"})).unwrap();
        assert_eq!(lock(&registry).unwrap().active["a"].generation, 2);
        let data = lock(&output).unwrap();
        let records = data
            .split(|b| *b == b'\n')
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 1);
        assert_eq!(
            serde_json::from_slice::<Value>(records[0]).unwrap()["type"],
            "cancelled"
        );
    }
    #[test]
    fn running_cancel_waits_for_owner_and_does_not_interrupt_voice_mutation() {
        let output = Mutex::new(Vec::new());
        let registry = Mutex::new(Registry::default());
        let token = CancellationToken::default();
        lock(&registry).unwrap().active.insert(
            "a".into(),
            Active {
                generation: 1,
                status: Status::Running,
                token: token.clone(),
            },
        );
        cancel(&registry, &output, "a").unwrap();
        assert!(token.is_cancelled());
        assert!(lock(&output).unwrap().is_empty());
        let token = CancellationToken::default();
        lock(&registry).unwrap().active.insert(
            "v".into(),
            Active {
                generation: 2,
                status: Status::AddingVoice,
                token: token.clone(),
            },
        );
        cancel(&registry, &output, "v").unwrap();
        assert!(!token.is_cancelled());
    }
}
