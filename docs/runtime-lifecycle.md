# Long-lived WGPU runtime lifecycle

`runtime` is the application-facing lifecycle above the lower-level
`InferenceBuilder` and `OnlineSession` APIs. Its state transitions make the
following ordering explicit:

```text
RuntimeBuilder<RuntimeCold>
    -> RuntimeBuilder<RuntimeConfigured>
    -> RuntimeBuilder<RuntimeLoaded>
    -> Runtime<RuntimeReady>
```

Cache configuration and WGPU policy are installed before the first tensor is
created. RF and codec checkpoints are loaded in parallel. A ready runtime is
only returned after compile-only warmup and real final-audio validation.

## Ordinary service construction

```rust,ignore
use irodori_tts_burn::{
    RequestAdmissionPolicy, ResidencyPolicy, RuntimeBuilder, SamplingPreset,
    WarmupSelection,
};

let loaded = RuntimeBuilder::new(model_path, decoder_path)
    .sampling_preset(SamplingPreset::OfficialV4)
    .admission(RequestAdmissionPolicy::CompileOnDemand)
    .residency(ResidencyPolicy::AlwaysResident)
    .load()?;

// Tokenized/encoded inputs are application data. They are validated one-to-one
// against the selected manifest before DryRun begins.
let mut runtime = loaded.warm(WarmupSelection::Interactive, warmup_inputs)?;
```

`SamplingPreset::OfficialV4` is the practical 40-step Euler policy used by the
official CLI defaults. `SamplingPreset::OfficialVoiceDesign` changes caption
CFG from 3 to the official Voice Design UI value 4. Four-step Euler exists in
benchmark harnesses to compare equal RF work; it is not a user-quality preset.

## Request admission

Two policies cover different latency contracts:

- `StrictWarmup` rejects a valid request class that was not in the startup
  manifest. This is appropriate for latency-SLO services and fixed-shape jobs.
- `CompileOnDemand` accepts an uncommon predicted duration, pays the compile
  cost once, and records the class as process-warm. This is the ergonomic
  default for local tools with variable duration.

`Runtime::request_readiness` reports `Ready`, `AcceptedWithWarmup`, or
`Rejected`, so a GUI or HTTP server can queue an uncommon request instead of
silently appearing hung.

`WarmupSelection::Interactive` covers 45/112-frame preview requests for
text-only, Voice Design, and prepared clone. `FullService` covers the measured
45/112/255/333/489/685 shapes and all four conditioning topologies, including
combined Voice Design plus clone. `Custom(WarmupManifest)` remains available
for a deployment-specific shape set.

`FullService` describes coverage; it is not a promise that every precision and
weight profile fits a 12 GiB adapter during one startup pass. On constrained
devices, use `Interactive + CompileOnDemand` or a smaller custom manifest and
observe the returned startup/allocator receipts.

## Cache and process lifetime

`RuntimeCachePolicy::PlatformDefault` uses the OS application cache directory
and appends a WGPU adapter namespace. `Root(path)` relocates that hierarchy.
`ExternallyConfigured` lets an embedding host install CubeCL policy itself.

The persistent environment stores autotune decisions and supported compiler
artifacts. WGPU pipeline objects remain process-local, so a separate warmup
subprocess cannot make a later service process warm. Keep `Runtime<RuntimeReady>`
alive and let CLI/GUI clients talk to that same process.

## Residency

`ResidencyPolicy` distinguishes always-resident, idle-timeout, memory-pressure,
and combined policies without paired `Option` values. The library does not run
a hidden timer: the service asks `Runtime::eviction_reason` and explicitly
consumes the runtime with `Runtime::evict`. Eviction returns
`RuntimeBuilder<RuntimeConfigured>`, which can reload models without illegally
reconfiguring process-global CubeCL state.
