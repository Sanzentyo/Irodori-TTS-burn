//! Thin entrypoint for the optional resident synthesis API.

use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand};
use irodori_tts_burn::{
    WgpuFloatPrecision, backend_config,
    synthesis::{SynthesisEngine, SynthesisModelPaths, SynthesisRequest, audio},
};
use std::{
    fs::File,
    io::{self, BufWriter, IsTerminal, Read, Write},
    path::PathBuf,
};

#[derive(Parser)]
#[command(about = "Irodori v4/v4.1 RF and MeanFlow inference with WGSL")]
struct Cli {
    #[command(flatten)]
    model: ModelArgs,
    #[command(subcommand)]
    command: Command,
}

#[derive(Args)]
struct ModelArgs {
    #[arg(long)]
    checkpoint: PathBuf,
    /// Full converted codec for voice creation; decoder-only also supports TTS.
    #[arg(long)]
    codec: PathBuf,
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "fp32")]
    precision: WgpuFloatPrecision,
    #[arg(long)]
    wgpu_adapter_index: Option<usize>,
    /// Adapter class and class-local index: auto, discrete:N, integrated:N, cpu.
    #[arg(long, default_value = "auto", conflicts_with = "wgpu_adapter_index")]
    wgpu_device: backend_config::DeviceSelector,
    /// Use established adapter defaults without tuning attention or MLP tiles.
    #[arg(long)]
    no_autotune: bool,
    /// Persistent tuning cache root (default: the platform user cache directory).
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    /// Load a saved voice or encode a WAV once: NAME=PATH.
    #[arg(long, value_parser = voice_pair)]
    add_voice: Vec<(String, PathBuf)>,
    /// Warm a two-second request before serving; shader compilation is shape-dependent.
    #[arg(long)]
    startup_warmup: bool,
    /// Evaluate decoder graph reuse with this maximum number of cached windows.
    #[cfg(feature = "streaming-graphs")]
    #[arg(long)]
    codec_graph_cache_entries: Option<std::num::NonZeroUsize>,
    /// Bound the input geometry of each cached decoder graph (not a VRAM byte cap).
    #[cfg(feature = "streaming-graphs")]
    #[arg(long, default_value = "116")]
    codec_graph_max_frames: std::num::NonZeroUsize,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a WAV; --request accepts every library sampling option as JSON.
    Tts {
        /// Text to synthesize; otherwise read stdin when --request is absent.
        #[arg(long)]
        text: Option<String>,
        #[arg(long)]
        request: Option<PathBuf>,
        #[arg(long)]
        instructions: Option<String>,
        #[arg(long)]
        voice: Option<String>,
        #[arg(long)]
        seconds: Option<f64>,
        #[arg(long)]
        steps: Option<usize>,
        /// ODE/CFG state precision; neural forwards retain model precision.
        #[arg(long, value_enum)]
        sampler_state: Option<irodori_tts_burn::SamplerStatePrecision>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long)]
        speed: Option<f64>,
        #[arg(long)]
        keep_tail: bool,
        /// WAV destination, or - for stdout.
        #[arg(short, long)]
        output: PathBuf,
        /// Warm requests run on the same resident engine before measurements.
        #[arg(long, default_value_t = 0)]
        warmup: usize,
        #[arg(long, default_value_t = 1)]
        repeat: usize,
    },
    /// Save concatenated reference latents, normalized to -16 LUFS per clip.
    Voice {
        #[arg(long, required = true)]
        reference: Vec<PathBuf>,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, default_value_t = -16.0, allow_hyphen_values = true)]
        target_lufs: f64,
        #[arg(long)]
        keep_loudness: bool,
    },
    #[cfg(feature = "worker")]
    /// Serve synthesis JSON Lines on stdin/stdout.
    Worker,
    #[cfg(feature = "http-server")]
    /// Serve /v1/audio/speech, /v1/models and /health.
    Serve {
        #[arg(long, default_value = "127.0.0.1:8080")]
        listen: std::net::SocketAddr,
    },
}

fn voice_pair(value: &str) -> std::result::Result<(String, PathBuf), String> {
    let (name, path) = value.split_once('=').ok_or("use NAME=PATH")?;
    if name.is_empty() || path.is_empty() {
        return Err("use nonempty NAME=PATH".into());
    }
    Ok((name.into(), path.into()))
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let cli = Cli::parse();
    #[cfg(feature = "worker")]
    let is_worker = matches!(&cli.command, Command::Worker);
    let start_engine = || -> Result<SynthesisEngine> {
        let selection = cli.model.wgpu_adapter_index.map_or(
            cli.model.wgpu_device,
            backend_config::DeviceSelector::AdapterIndex,
        );
        let selected = selection.device();
        let namespace = backend_config::wgpu_cache_namespace(&selected)?;
        let cache_root = match &cli.model.cache_dir {
            Some(path) => path.clone(),
            None => backend_config::default_cubecl_cache_root()?,
        };
        let cache = backend_config::configure_cubecl_persistent_cache_for_precision(
            cache_root.join(&namespace),
            cli.model.precision,
        )?;
        tracing::info!(path = %cache.environment_path.display(), "persistent kernel tuning cache");
        use burn::backend::wgpu::{RuntimeOptions, graphics::AutoGraphicsApi, init_setup};
        let setup = init_setup::<AutoGraphicsApi>(&selected, RuntimeOptions::default());
        let info = setup.adapter.get_info();
        if cli.model.no_autotune {
            irodori_tts_burn::install_recommended_route_table(
                info.vendor,
                &format!("{:?}", info.backend),
                std::env::consts::OS,
                irodori_tts_burn::RouteCacheMissReason::PortableRequested,
            )?;
        } else {
            irodori_tts_burn::install_autotuned_route_table(
                info.vendor,
                &format!("{:?}", info.backend),
                std::env::consts::OS,
            )?;
        }
        let device = backend_config::wgpu_device_with_precision(&selected, cli.model.precision)?;
        let mut engine = SynthesisEngine::load(
            SynthesisModelPaths {
                checkpoint: cli.model.checkpoint,
                codec: cli.model.codec,
                tokenizer: cli.model.tokenizer,
            },
            device,
            cli.model.precision,
        )?;
        #[cfg(feature = "streaming-graphs")]
        if let Some(entries) = cli.model.codec_graph_cache_entries {
            engine =
                engine.with_codec_graph_cache(irodori_tts_burn::codec::DecoderGraphLimits::new(
                    entries,
                    cli.model.codec_graph_max_frames,
                )?)?;
        }
        for (name, path) in cli.model.add_voice {
            let voice = if path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
            {
                engine.create_voice(&[path], Some(-16.0))?
            } else {
                engine.load_voice(&path)?
            };
            engine.add_voice(&name, voice)?;
        }
        if cli.model.startup_warmup {
            engine.synthesize(&SynthesisRequest {
                text: "こんにちは。".into(),
                seconds: Some(2.0),
                seed: Some(42),
                keep_tail: true,
                ..Default::default()
            })?;
        }
        Ok(engine)
    };
    let mut engine = match start_engine() {
        Ok(engine) => engine,
        Err(error) => {
            #[cfg(feature = "worker")]
            if is_worker {
                println!(
                    "{}",
                    serde_json::json!({"type":"fatal","error":{"code":"model_file","option":null,"message":error.to_string()}})
                );
            }
            return Err(error);
        }
    };
    match cli.command {
        Command::Tts {
            text,
            request,
            instructions,
            voice,
            seconds,
            steps,
            sampler_state,
            seed,
            speed,
            keep_tail,
            output,
            warmup,
            repeat,
        } => {
            ensure!(
                (1..=100).contains(&repeat) && warmup <= 100,
                "repeat must be 1..100 and warmup 0..100"
            );
            let read_stdin = text.is_none() && request.is_none();
            let mut request: SynthesisRequest = match request {
                Some(path) => serde_json::from_reader(File::open(path)?)?,
                None => SynthesisRequest::default(),
            };
            if let Some(text) = text {
                request.text = text;
            }
            if read_stdin {
                ensure!(
                    !io::stdin().is_terminal(),
                    "provide --text, --request or UTF-8 text on stdin"
                );
                io::stdin()
                    .take(1_048_577)
                    .read_to_string(&mut request.text)
                    .context("cannot read UTF-8 text from stdin")?;
            }
            if let Some(instructions) = instructions {
                request.instructions = instructions;
            }
            if let Some(voice) = voice {
                request.voice = voice;
            }
            if let Some(seconds) = seconds {
                request.seconds = Some(seconds);
            }
            if let Some(steps) = steps {
                request.steps = Some(steps);
            }
            if let Some(precision) = sampler_state {
                request.sampler_state = precision;
            }
            if let Some(seed) = seed {
                request.seed = Some(seed);
            }
            if let Some(speed) = speed {
                request.speed = speed;
            }
            request.keep_tail |= keep_tail;
            request.validate(engine.config().flow_parameterization)?;
            for _ in 0..warmup {
                engine.synthesize(&request)?;
            }
            let mut samples = Vec::new();
            for index in 0..repeat {
                let (audio, report) = engine.synthesize(&request)?;
                samples = audio;
                eprintln!(
                    "{}",
                    serde_json::json!({"iteration":index, "precision":engine.precision(), "report": report})
                );
            }
            let destination: Box<dyn Write> = if output.as_os_str() == "-" {
                Box::new(io::stdout().lock())
            } else {
                Box::new(
                    File::create(&output)
                        .with_context(|| format!("cannot create {}", output.display()))?,
                )
            };
            let mut file = BufWriter::new(destination);
            audio::write_wav(&mut file, &samples, 48_000)?;
            file.flush()?;
        }
        Command::Voice {
            reference,
            output,
            target_lufs,
            keep_loudness,
        } => {
            let voice = if reference.len() == 1
                && reference[0]
                    .extension()
                    .is_some_and(|ext| ext == "safetensors")
            {
                engine.load_voice(&reference[0])?
            } else {
                engine.create_voice(&reference, (!keep_loudness).then_some(target_lufs))?
            };
            voice.save(&output)?;
        }
        #[cfg(feature = "worker")]
        Command::Worker => irodori_tts_burn::synthesis::worker::run(
            engine,
            std::io::stdin().lock(),
            std::io::stdout(),
        )?,
        #[cfg(feature = "http-server")]
        Command::Serve { listen } => irodori_tts_burn::synthesis::http::serve(engine, listen)?,
    }
    Ok(())
}
