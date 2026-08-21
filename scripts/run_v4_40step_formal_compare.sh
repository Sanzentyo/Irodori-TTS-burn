#!/usr/bin/env bash
# Fresh strict-FP32 production comparison: 40 Euler evaluations, six lengths,
# three conditioning topologies, five fresh processes per runtime/condition.

set -Eeuo pipefail
IFS=$'\n\t'

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
USER_ROOT=$(realpath -- "$ROOT/../..")
UPSTREAM=$(realpath -- "$ROOT/../Irodori-TTS")
OUT=
SMOKE=0

while (($#)); do
  case "$1" in
    --output-dir) OUT=${2:?--output-dir requires a path}; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    -h|--help)
      printf 'usage: %s --output-dir FRESH_PATH [--smoke]\n' "$0"
      exit 0
      ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ -n $OUT ]] || { printf 'error: --output-dir is required\n' >&2; exit 2; }
OUT=$(realpath -m -- "$OUT")
[[ ! -e $OUT && ! -L $OUT ]] || { printf 'error: output exists: %s\n' "$OUT" >&2; exit 1; }

MODEL_REV=e4aaac4df355ff560dcd35e0dae272c3a759317b
CODEC_REV=47376ee24834d7a05a48ebabfe3cde29b3c5e214
MODEL="$USER_ROOT/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4-Small/snapshots/$MODEL_REV/model.safetensors"
MODEL_SAMPLES="$USER_ROOT/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4-Small/snapshots/$MODEL_REV/samples"
PY_CODEC="$USER_ROOT/.cache/huggingface/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/$CODEC_REV/weights.pth"
WG_CODEC="$USER_ROOT/benchmark-artifacts/irodori-v4-load-opt-20260813-attempt1/models/dacvae-decoder-only.safetensors"
REF1="$MODEL_SAMPLES/clone_ref1.wav"
REF2="$MODEL_SAMPLES/clone_ref2.wav"
WG_BIN="$ROOT/target/release/bench_v4_residency"
PY_BENCH="$ROOT/scripts/bench_python_runtime_scenarios.py"
SOURCE_CREATOR="$ROOT/scripts/create_v4_source_fixture.py"
REF_EXPORTER="$ROOT/scripts/export_prepared_reference_latents.py"

MODEL_SHA=5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593
PY_CODEC_SHA=db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5
WG_CODEC_SHA=1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231
UPSTREAM_COMMIT=9f19d9a9048099a4b978a762d0509228fe624e3f
GPU_NAME='NVIDIA GeForce RTX 5070 Ti Laptop GPU'
GPU_PCI=00000000:01:00.0
GPU_VRAM_MIB=12227
NVML_INDEX=0
WGPU_ADAPTER=0
CPU_SET=0-11
LOCK=/tmp/irodori-v4-40step-formal-gpu0.lock
ACTIVE_MONITOR=
CURRENT_PHASE=preflight
COMPLETE=0

FRAMES=(45 112 255 333 489 685)
SLUGS=(f045 f112 f255 f333 f489 f685)
VOICES=(text design clone)
SESSIONS=5
WARMUPS=2
MEASURED=10
if ((SMOKE)); then
  FRAMES=(112)
  SLUGS=(f112)
  SESSIONS=1
  WARMUPS=1
  MEASURED=1
fi

say() { printf '[40step-formal] %s\n' "$1"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
sha() { sha256sum -- "$1" | awk '{print $1}'; }
stop_monitor() {
  if [[ -n $ACTIVE_MONITOR ]]; then
    kill "$ACTIVE_MONITOR" 2>/dev/null || true
    wait "$ACTIVE_MONITOR" 2>/dev/null || true
    ACTIVE_MONITOR=
  fi
}
seal() {
  local status=$1
  stop_monitor
  [[ -d $OUT && ! -L $OUT ]] || return 0
  printf 'status=%s\nphase=%s\nautomatic_retries=0\noutput_reuse=false\n' \
    "$status" "$CURRENT_PHASE" >"$OUT/$status"
  (
    cd "$OUT"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
  )
}
on_exit() {
  local status=$?
  stop_monitor
  if ((status != 0 && ! COMPLETE)); then
    set +e
    seal FAILURE
  fi
  return "$status"
}
trap on_exit EXIT

for command in cargo flock git jq nvidia-smi taskset uv vulkaninfo; do
  command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
done
for path in "$MODEL" "$PY_CODEC" "$WG_CODEC" "$REF1" "$REF2" "$WG_BIN" \
  "$PY_BENCH" "$SOURCE_CREATOR" "$REF_EXPORTER"; do
  [[ -f $path && -s $path ]] || die "missing input: $path"
done
[[ $(sha "$MODEL") == "$MODEL_SHA" ]] || die 'model SHA mismatch'
[[ $(sha "$PY_CODEC") == "$PY_CODEC_SHA" ]] || die 'Python codec SHA mismatch'
[[ $(sha "$WG_CODEC") == "$WG_CODEC_SHA" ]] || die 'WGPU codec SHA mismatch'
[[ $(git -C "$UPSTREAM" rev-parse HEAD) == "$UPSTREAM_COMMIT" ]] || die 'upstream commit mismatch'
[[ $(git -C "$ROOT" branch --show-current) == codex/v4-post-seal-priority-1-4 ]] || die 'unexpected Rust branch'
[[ -z $(git -C "$ROOT" status --short) ]] || die 'Rust source tree must be clean'

gpu_row=$(nvidia-smi -i "$NVML_INDEX" --query-gpu=name,pci.bus_id,memory.total,driver_version --format=csv,noheader,nounits)
IFS=',' read -r measured_name measured_pci measured_vram measured_driver <<<"$gpu_row"
measured_name=${measured_name## }; measured_name=${measured_name%% }
measured_pci=${measured_pci//[[:space:]]/}
measured_vram=${measured_vram//[[:space:]]/}
measured_driver=${measured_driver//[[:space:]]/}
[[ $measured_name == "$GPU_NAME" && ${measured_pci^^} == "$GPU_PCI" && $measured_vram == "$GPU_VRAM_MIB" ]] || \
  die "GPU identity mismatch: $gpu_row"

mkdir -p "$OUT/build" "$OUT/inputs" "$OUT/preparation" "$OUT/prime" "$OUT/sessions"
install -m 0555 "$WG_BIN" "$OUT/build/bench_v4_residency"
install -m 0444 "$PY_BENCH" "$OUT/build/bench_python_runtime_scenarios.py"
install -m 0444 "$SOURCE_CREATOR" "$OUT/build/create_v4_source_fixture.py"
install -m 0444 "$REF_EXPORTER" "$OUT/build/export_prepared_reference_latents.py"
install -m 0444 "$ROOT/src/bin/bench_v4_residency.rs" "$OUT/build/bench_v4_residency.rs"
install -m 0444 "$0" "$OUT/build/runner.sh"
git -C "$ROOT" rev-parse HEAD >"$OUT/source-head.txt"
git -C "$UPSTREAM" rev-parse HEAD >"$OUT/upstream-head.txt"
sha256sum "$OUT/build"/* "$MODEL" "$PY_CODEC" "$WG_CODEC" "$REF1" "$REF2" >"$OUT/pins.sha256"
nvidia-smi -q >"$OUT/nvidia-smi-q.txt"
vulkaninfo --summary >"$OUT/vulkan-summary.txt" 2>&1
rustc -Vv >"$OUT/rustc.txt"
cargo -Vv >"$OUT/cargo.txt"
uv --version >"$OUT/uv.txt"
jq -n \
  --arg format irodori-v4-40step-formal-v1 \
  --arg mode "$([[ $SMOKE == 1 ]] && printf smoke || printf formal)" \
  --arg started "$(date --iso-8601=seconds)" \
  --arg source "$(git -C "$ROOT" rev-parse HEAD)" \
  --arg upstream "$UPSTREAM_COMMIT" \
  --arg model_revision "$MODEL_REV" \
  --arg codec_revision "$CODEC_REV" \
  --arg gpu "$GPU_NAME" --arg pci "$GPU_PCI" --arg driver "$measured_driver" \
  --arg cpu_set "$CPU_SET" --argjson vram "$GPU_VRAM_MIB" \
  --argjson sessions "$SESSIONS" --argjson warmups "$WARMUPS" --argjson measured "$MEASURED" \
  '{format:$format,mode:$mode,started_at:$started,precision:"fp32",steps:40,
    sampler:"euler",schedule:"linear",cfg:{mode:"independent",text:3,caption:4,speaker:5,min_t:0.5,max_t:1},
    tail_trim:false,watermark:false,source_commit:$source,upstream_commit:$upstream,
    model_revision:$model_revision,codec_revision:$codec_revision,
    process_contract:{fresh_sessions_per_runtime_condition:$sessions,warmups:$warmups,measured:$measured,automatic_retries:0},
    timing:{device_complete:"pre-start device sync through GPU completion",readback_complete:"device work plus owned contiguous CPU f32 audio",cold_and_first_request:"kept separate from steady medians"},
    cache:{cubecl:"one fresh campaign bundle primed before measurement",driver:"one campaign-local XDG cache primed before measurement",wgpu_pipelines:"process-local and rebuilt in every fresh process"},
    hardware:{gpu:$gpu,pci_bus_id:$pci,driver:$driver,vram_mib:$vram,nvml_index:0,wgpu_adapter_index:0},cpu_affinity:$cpu_set}' \
  >"$OUT/protocol.json"

exec 9>>"$LOCK"
flock -n 9 || die 'formal campaign GPU lock is held'

wait_idle() {
  local quiet=0 telemetry processes
  for _ in $(seq 1 60); do
    telemetry=$(nvidia-smi -i "$NVML_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi -i "$NVML_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ $telemetry =~ ^([0-9]+),[[:space:]]*([0-9]+)$ ]] \
      && ((BASH_REMATCH[1] <= 512 && BASH_REMATCH[2] <= 5)) \
      && [[ ! $processes =~ [0-9] ]]; then
      ((quiet += 1))
      ((quiet >= 2)) && return 0
    else
      quiet=0
    fi
    sleep 1
  done
  die 'GPU did not provide two consecutive idle samples'
}

run_monitored() {
  local dir=$1
  shift
  nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader,nounits -lms 100 -f "$dir/nvml.csv" &
  ACTIVE_MONITOR=$!
  set +e
  /usr/bin/time -o "$dir/wall.txt" -f 'exit_status=%x\nelapsed_seconds=%e\nmax_rss_kib=%M' \
    "$@" >"$dir/stdout.log" 2>"$dir/stderr.log"
  local status=$?
  set -e
  stop_monitor
  return "$status"
}

CURRENT_PHASE=source-fixture
uv run "$OUT/build/create_v4_source_fixture.py" --output "$OUT/inputs/source-noise.safetensors" \
  >"$OUT/preparation/source-noise.log" 2>&1
SOURCE_FIXTURE="$OUT/inputs/source-noise.safetensors"
SOURCE_SHA=$(sha "$SOURCE_FIXTURE")

say 'preparing fresh request fixtures and reference latents'
for index in "${!FRAMES[@]}"; do
  frames=${FRAMES[$index]}
  slug=${SLUGS[$index]}
  dir="$OUT/preparation/$slug"
  mkdir "$dir"
  CURRENT_PHASE="prepare-$slug"
  wait_idle
  run_monitored "$dir" env -u LD_LIBRARY_PATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    taskset -c "$CPU_SET" uv run "$OUT/build/bench_python_runtime_scenarios.py" \
      --upstream "$UPSTREAM" --checkpoint "$MODEL" --codec "$PY_CODEC" --ref1 "$REF1" --ref2 "$REF2" \
      --output "$dir/result.json" --work-dir "$dir/work" --fixture-dir "$dir/fixtures" \
      --source-fixture "$SOURCE_FIXTURE" --latent-frames "$frames" --num-steps 1 \
      --cfg-scale-caption 4 --warmups 1 --measured 1 --precision fp32 \
      --expected-pci "$GPU_PCI" --expected-gpu-name "$GPU_NAME" --scenario text_only_fixed \
    || die "fixture preparation failed without retry: $slug"
  jq -e --argjson frames "$frames" --arg source "$SOURCE_SHA" '
    .parameters.latent_frames == $frames and .parameters.source_fixture_sha256 == $source and
    .environment.strict_fp32 and (.rust_request_fixtures | keys == ["design","text"]) and
    (.prepared_reference.tensor_f32_sha256 | length == 2)
  ' "$dir/result.json" >/dev/null || die "fixture preparation gate failed: $slug"
done

CURRENT_PHASE=reference-export
uv run "$OUT/build/export_prepared_reference_latents.py" \
  --input "$OUT/preparation/${SLUGS[0]}/work/ref1-latent.pt" \
  --input "$OUT/preparation/${SLUGS[0]}/work/ref2-latent.pt" \
  --output-dir "$OUT/inputs/references" >"$OUT/preparation/reference-export.log" 2>&1
REF1_PREP="$OUT/inputs/references/ref1.safetensors"
REF2_PREP="$OUT/inputs/references/ref2.safetensors"
EXPECTED_REF_HASHES=$(jq -c '.prepared_reference.tensor_f32_sha256' "$OUT/preparation/${SLUGS[0]}/result.json")
for slug in "${SLUGS[@]}"; do
  jq -e --argjson expected "$EXPECTED_REF_HASHES" \
    '.prepared_reference.tensor_f32_sha256 == $expected' "$OUT/preparation/$slug/result.json" >/dev/null \
    || die "prepared reference tensor changed across fresh processes: $slug"
done

fixture_args=()
for slug in "${SLUGS[@]}"; do
  fixture_args+=(--fixture "$OUT/preparation/$slug/fixtures/text.safetensors")
done

say 'priming one campaign-local CubeCL bundle and vendor driver cache'
mkdir "$OUT/prime/xdg" "$OUT/prime/cubecl"
prime_length_mode=mixed
((${#FRAMES[@]} == 1)) && prime_length_mode=same
for voice in "${VOICES[@]}"; do
  dir="$OUT/prime/$voice"
  mkdir "$dir"
  CURRENT_PHASE="prime-$voice"
  flags=()
  case "$voice" in
    text) flags=(--unconditioned) ;;
    design)
      flags=(--designed)
      fixture_args=()
      for slug in "${SLUGS[@]}"; do fixture_args+=(--fixture "$OUT/preparation/$slug/fixtures/design.safetensors"); done
      ;;
    clone)
      flags=()
      fixture_args=()
      for slug in "${SLUGS[@]}"; do fixture_args+=(--fixture "$OUT/preparation/$slug/fixtures/text.safetensors"); done
      ;;
  esac
  bundle_out=()
  [[ $voice == clone ]] && bundle_out=(--cubecl-bundle-out "$OUT/prime/environment.cubecl")
  wait_idle
  run_monitored "$dir" env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID WGPU_BACKEND=vulkan \
    XDG_CACHE_HOME="$OUT/prime/xdg" taskset -c "$CPU_SET" "$OUT/build/bench_v4_residency" \
      --mode all-resident --checkpoint "$MODEL" --codec-weights "$WG_CODEC" \
      "${fixture_args[@]}" --reference "$REF1_PREP" "$REF2_PREP" \
      --requests "${#FRAMES[@]}" --warmups 0 --num-steps 40 --cfg-caption 4 \
      "${flags[@]}" --speaker-mode same --length-mode "$prime_length_mode" --adapter-index "$WGPU_ADAPTER" \
      --precision fp32 --allocator exclusive-pages --codec-residency decode-only \
      --load-strategy parallel --rf-checkpoint-loader indexed-file \
      --cubecl-cache-dir "$OUT/prime/cubecl" "${bundle_out[@]}" --output-json "$dir/result.json" \
    || die "cache prime failed without retry: $voice"
  jq -e --argjson requests "${#FRAMES[@]}" '
    .schema_version == 9 and .euler_evaluations == 40 and .block_calls == 480 and
    .requests == $requests and (.work_reports | length == $requests)
  ' "$dir/result.json" >/dev/null || die "cache prime result gate failed: $voice"
done
[[ -f $OUT/prime/environment.cubecl && -s $OUT/prime/environment.cubecl ]] || die 'CubeCL bundle export missing'

run_python() {
  local dir=$1 frames=$2 scenario=$3
  mkdir -p "$dir/audio"
  run_monitored "$dir" env -u LD_LIBRARY_PATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    taskset -c "$CPU_SET" uv run "$OUT/build/bench_python_runtime_scenarios.py" \
      --upstream "$UPSTREAM" --checkpoint "$MODEL" --codec "$PY_CODEC" --ref1 "$REF1" --ref2 "$REF2" \
      --output "$dir/result.json" --work-dir "$dir/work" --audio-output-dir "$dir/audio" \
      --source-fixture "$SOURCE_FIXTURE" --latent-frames "$frames" --num-steps 40 \
      --cfg-scale-caption 4 --warmups "$WARMUPS" --measured "$MEASURED" --precision fp32 \
      --expected-pci "$GPU_PCI" --expected-gpu-name "$GPU_NAME" --scenario "$scenario"
}

run_wgpu() {
  local dir=$1 frames=$2 voice=$3 fixture=$4
  local total=$((WARMUPS + MEASURED))
  mkdir -p "$dir/audio" "$dir/cubecl"
  local flags=()
  case "$voice" in
    text) flags=(--unconditioned) ;;
    design) flags=(--designed) ;;
    clone) flags=() ;;
  esac
  run_monitored "$dir" env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID WGPU_BACKEND=vulkan \
    XDG_CACHE_HOME="$OUT/prime/xdg" taskset -c "$CPU_SET" "$OUT/build/bench_v4_residency" \
      --mode all-resident --checkpoint "$MODEL" --codec-weights "$WG_CODEC" \
      --fixture "$fixture" --reference "$REF1_PREP" "$REF2_PREP" \
      --requests "$total" --warmups "$WARMUPS" --num-steps 40 --cfg-caption 4 \
      "${flags[@]}" --speaker-mode same --length-mode same --adapter-index "$WGPU_ADAPTER" \
      --precision fp32 --allocator exclusive-pages --codec-residency decode-only \
      --load-strategy parallel --rf-checkpoint-loader indexed-file \
      --cubecl-cache-dir "$dir/cubecl" --cubecl-bundle-in "$OUT/prime/environment.cubecl" \
      --audio-output-dir "$dir/audio" --output-json "$dir/result.json"
}

pair_index=0
for index in "${!FRAMES[@]}"; do
  frames=${FRAMES[$index]}
  slug=${SLUGS[$index]}
  for voice in "${VOICES[@]}"; do
    case "$voice" in
      text)
        scenario=text_only_fixed
        fixture="$OUT/preparation/$slug/fixtures/text.safetensors"
        expected_rows=60
        ;;
      design)
        scenario=design_fixed
        fixture="$OUT/preparation/$slug/fixtures/design.safetensors"
        expected_rows=80
        ;;
      clone)
        scenario=clone_prepared_fixed
        fixture="$OUT/preparation/$slug/fixtures/text.safetensors"
        expected_rows=80
        ;;
    esac
    for session in $(seq 1 "$SESSIONS"); do
      condition="$slug-$voice-s$session"
      base="$OUT/sessions/$condition"
      py="$base/python"
      wg="$base/wgpu"
      mkdir -p "$py" "$wg"
      CURRENT_PHASE="$condition"
      say "$condition ($(($pair_index + 1))/$(( ${#FRAMES[@]} * ${#VOICES[@]} * SESSIONS )))"
      if (((pair_index + session) % 2 == 0)); then
        wait_idle
        run_python "$py" "$frames" "$scenario" || die "Python failed without retry: $condition"
        wait_idle
        run_wgpu "$wg" "$frames" "$voice" "$fixture" || die "WGPU failed without retry: $condition"
      else
        wait_idle
        run_wgpu "$wg" "$frames" "$voice" "$fixture" || die "WGPU failed without retry: $condition"
        wait_idle
        run_python "$py" "$frames" "$scenario" || die "Python failed without retry: $condition"
      fi
      jq -e --arg scenario "$scenario" --argjson frames "$frames" --arg source "$SOURCE_SHA" \
        --argjson warmups "$WARMUPS" --argjson measured "$MEASURED" --argjson refs "$EXPECTED_REF_HASHES" '
        .environment.strict_fp32 and (.environment.matmul_tf32|not) and (.environment.cudnn_tf32|not) and
        (.environment.autocast|not) and .parameters.num_steps == 40 and
        .parameters.cfg_scale_caption == 4 and .parameters.latent_frames == $frames and
        .parameters.warmups == $warmups and .parameters.measured == $measured and
        .parameters.source_fixture_sha256 == $source and .prepared_reference.tensor_f32_sha256 == $refs and
        .scenarios[$scenario].summary.measured_requests == $measured and
        .scenarios[$scenario].summary.deterministic_per_voice and
        (.scenarios[$scenario].rows | length == ($warmups + $measured))
      ' "$py/result.json" >/dev/null || die "Python result gate failed: $condition"
      jq -e --argjson frames "$frames" --argjson warmups "$WARMUPS" --argjson measured "$MEASURED" \
        --argjson rows "$expected_rows" '
        .schema_version == 9 and .strict_fp32 and (.autocast|not) and (.tf32|not) and
        .euler_evaluations == 40 and .cfg_caption == 4 and .block_calls == 480 and
        .effective_rows == $rows and .warmups == $warmups and .measured == $measured and
        (.resident_request_timings | length == ($warmups + $measured)) and
        (.work_report.num_steps == 40) and (.work_report.schedule_f32_bits | length == 41) and
        (.work_report.forwards | length == 40) and
        (.items | all(.frames == $frames and .samples == ($frames * 1920))) and
        ([.items[].audio_f32_sha256] | unique | length == 1)
      ' "$wg/result.json" >/dev/null || die "WGPU result gate failed: $condition"
      printf 'complete\n' >"$base/COMPLETE"
      ((pair_index += 1))
    done
  done
done

CURRENT_PHASE=complete
wait_idle
COMPLETE=1
seal COMPLETE
say "complete: $OUT"
