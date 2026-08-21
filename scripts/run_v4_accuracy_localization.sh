#!/usr/bin/env bash
# Fresh diagnostic-only localization of the strict-FP32 40-step divergence.

set -Eeuo pipefail
IFS=$'\n\t'

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
USER_ROOT=$(realpath -- "$ROOT/../..")
UPSTREAM=$(realpath -- "$ROOT/../Irodori-TTS")
OUT=
while (($#)); do
  case "$1" in
    --output-dir) OUT=${2:?--output-dir requires a path}; shift 2 ;;
    -h|--help) printf 'usage: %s --output-dir FRESH_PATH\n' "$0"; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n $OUT ]] || { printf 'error: --output-dir is required\n' >&2; exit 2; }
OUT=$(realpath -m -- "$OUT")
[[ ! -e $OUT && ! -L $OUT ]] || { printf 'error: output exists: %s\n' "$OUT" >&2; exit 1; }

MODEL_REV=e4aaac4df355ff560dcd35e0dae272c3a759317b
CODEC_REV=47376ee24834d7a05a48ebabfe3cde29b3c5e214
MODEL="$USER_ROOT/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4-Small/snapshots/$MODEL_REV/model.safetensors"
SAMPLES="$USER_ROOT/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4-Small/snapshots/$MODEL_REV/samples"
PY_CODEC="$USER_ROOT/.cache/huggingface/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/$CODEC_REV/weights.pth"
WG_CODEC="$USER_ROOT/benchmark-artifacts/irodori-v4-load-opt-20260813-attempt1/models/dacvae-decoder-only.safetensors"
REF1="$SAMPLES/clone_ref1.wav"
REF2="$SAMPLES/clone_ref2.wav"
WG_BIN="$ROOT/target/release/bench_v4_residency"
PY_BENCH="$ROOT/scripts/bench_python_runtime_scenarios.py"
COMPARE="$ROOT/scripts/compare_v4_diagnostic_tensors.py"
SOURCE_CREATOR="$ROOT/scripts/create_v4_source_fixture.py"
REF_EXPORTER="$ROOT/scripts/export_prepared_reference_latents.py"
GPU_NAME='NVIDIA GeForce RTX 5070 Ti Laptop GPU'
GPU_PCI=00000000:01:00.0
GPU_VRAM_MIB=12227
MODEL_SHA=5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593
PY_CODEC_SHA=db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5
WG_CODEC_SHA=1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231
UPSTREAM_COMMIT=9f19d9a9048099a4b978a762d0509228fe624e3f
NVML_INDEX=0
WGPU_ADAPTER=0
CPU_SET=0-11
LOCK=/tmp/irodori-v4-accuracy-localization-gpu0.lock
CURRENT_PHASE=preflight
ACTIVE_MONITOR=
COMPLETE=0

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
  printf 'status=%s\nphase=%s\nautomatic_retries=0\nlatency_results_valid=false\n' \
    "$status" "$CURRENT_PHASE" >"$OUT/$status"
  (cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
}
on_exit() {
  local status=$?
  stop_monitor
  if ((status != 0 && ! COMPLETE)); then set +e; seal FAILURE; fi
  return "$status"
}
trap on_exit EXIT

for command in flock git jq nvidia-smi taskset uv; do
  command -v "$command" >/dev/null || die "missing command: $command"
done
for path in "$MODEL" "$PY_CODEC" "$WG_CODEC" "$REF1" "$REF2" "$WG_BIN" \
  "$PY_BENCH" "$COMPARE" "$SOURCE_CREATOR" "$REF_EXPORTER"; do
  [[ -f $path && -s $path ]] || die "missing input: $path"
done
[[ -z $(git -C "$ROOT" status --short) ]] || die 'Rust source tree must be clean'
[[ $(sha "$MODEL") == "$MODEL_SHA" ]] || die 'model SHA mismatch'
[[ $(sha "$PY_CODEC") == "$PY_CODEC_SHA" ]] || die 'Python codec SHA mismatch'
[[ $(sha "$WG_CODEC") == "$WG_CODEC_SHA" ]] || die 'WGPU codec SHA mismatch'
[[ $(git -C "$UPSTREAM" rev-parse HEAD) == "$UPSTREAM_COMMIT" ]] || die 'upstream commit mismatch'
gpu_row=$(nvidia-smi -i "$NVML_INDEX" --query-gpu=name,pci.bus_id,memory.total,driver_version --format=csv,noheader,nounits)
IFS=',' read -r measured_name measured_pci measured_vram measured_driver <<<"$gpu_row"
measured_name=${measured_name## }; measured_name=${measured_name%% }
measured_pci=${measured_pci//[[:space:]]/}; measured_vram=${measured_vram//[[:space:]]/}
[[ $measured_name == "$GPU_NAME" && ${measured_pci^^} == "$GPU_PCI" && $measured_vram == "$GPU_VRAM_MIB" ]] \
  || die "GPU identity mismatch: $gpu_row"

mkdir -p "$OUT/build" "$OUT/inputs" "$OUT/preparation" "$OUT/prime" "$OUT/cases"
install -m 0555 "$WG_BIN" "$OUT/build/bench_v4_residency"
install -m 0444 "$PY_BENCH" "$COMPARE" "$SOURCE_CREATOR" "$REF_EXPORTER" "$OUT/build/"
install -m 0444 "$0" "$OUT/build/runner.sh"
git -C "$ROOT" rev-parse HEAD >"$OUT/source-head.txt"
git -C "$UPSTREAM" rev-parse HEAD >"$OUT/upstream-head.txt"
nvidia-smi -q >"$OUT/nvidia-smi-q.txt"
sha256sum "$OUT/build"/* "$MODEL" "$PY_CODEC" "$WG_CODEC" "$REF1" "$REF2" >"$OUT/pins.sha256"

exec 9>>"$LOCK"
flock -n 9 || die 'accuracy-localization GPU lock is held'
wait_idle() {
  local quiet=0 telemetry processes
  for _ in $(seq 1 60); do
    telemetry=$(nvidia-smi -i "$NVML_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi -i "$NVML_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ $telemetry =~ ^([0-9]+),[[:space:]]*([0-9]+)$ ]] \
      && ((BASH_REMATCH[1] <= 512 && BASH_REMATCH[2] <= 5)) && [[ ! $processes =~ [0-9] ]]; then
      ((quiet += 1)); ((quiet >= 2)) && return 0
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
  nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.free,utilization.gpu \
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

FRAMES=(112 333 489 685)
for frames in "${FRAMES[@]}"; do
  slug=$(printf 'f%03d' "$frames")
  dir="$OUT/preparation/$slug"
  mkdir "$dir"
  CURRENT_PHASE="prepare-$slug"
  wait_idle
  run_monitored "$dir" env -u LD_LIBRARY_PATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    taskset -c "$CPU_SET" uv run --python 3.10 "$OUT/build/bench_python_runtime_scenarios.py" \
      --upstream "$UPSTREAM" --checkpoint "$MODEL" --codec "$PY_CODEC" --ref1 "$REF1" --ref2 "$REF2" \
      --output "$dir/result.json" --work-dir "$dir/work" --fixture-dir "$dir/fixtures" \
      --source-fixture "$SOURCE_FIXTURE" --latent-frames "$frames" --num-steps 1 \
      --cfg-scale-caption 4 --warmups 1 --measured 1 --precision fp32 \
      --expected-pci "$GPU_PCI" --expected-gpu-name "$GPU_NAME" --scenario text_only_fixed \
    || die "fixture preparation failed: $slug"
done

CURRENT_PHASE=reference-export
uv run --python 3.10 "$OUT/build/export_prepared_reference_latents.py" \
  --input "$OUT/preparation/f112/work/ref1-latent.pt" \
  --input "$OUT/preparation/f112/work/ref2-latent.pt" \
  --output-dir "$OUT/inputs/references" >"$OUT/preparation/reference-export.log" 2>&1
REF1_PREP="$OUT/inputs/references/ref1.safetensors"
REF2_PREP="$OUT/inputs/references/ref2.safetensors"

mkdir "$OUT/prime/xdg" "$OUT/prime/cubecl"
for voice in text clone; do
  dir="$OUT/prime/$voice"
  mkdir "$dir"
  flags=(--unconditioned)
  [[ $voice == clone ]] && flags=()
  fixtures=()
  for frames in "${FRAMES[@]}"; do
    slug=$(printf 'f%03d' "$frames")
    fixtures+=(--fixture "$OUT/preparation/$slug/fixtures/text.safetensors")
  done
  bundle=()
  [[ $voice == clone ]] && bundle=(--cubecl-bundle-out "$OUT/prime/environment.cubecl")
  CURRENT_PHASE="prime-$voice"
  wait_idle
  run_monitored "$dir" env -u CUDA_VISIBLE_DEVICES WGPU_BACKEND=vulkan XDG_CACHE_HOME="$OUT/prime/xdg" \
    taskset -c "$CPU_SET" "$OUT/build/bench_v4_residency" --mode all-resident \
      --checkpoint "$MODEL" --codec-weights "$WG_CODEC" "${fixtures[@]}" \
      --reference "$REF1_PREP" "$REF2_PREP" --requests 4 --warmups 0 --num-steps 40 \
      --cfg-caption 4 "${flags[@]}" --length-mode mixed --precision fp32 \
      --allocator exclusive-pages --codec-residency decode-only --load-strategy parallel \
      --rf-checkpoint-loader indexed-file --cubecl-cache-dir "$OUT/prime/cubecl" \
      "${bundle[@]}" --output-json "$dir/result.json" || die "prime failed: $voice"
done

CASES=('f112-text' 'f333-clone' 'f489-design' 'f489-clone' 'f685-clone')
for name in "${CASES[@]}"; do
  slug=${name%-*}
  voice=${name#*-}
  frames=$((10#${slug#f}))
  dir="$OUT/cases/$name"
  py="$dir/python"
  wg="$dir/wgpu"
  mkdir -p "$py" "$wg/cubecl"
  scenario=text_only_fixed
  fixture="$OUT/preparation/$slug/fixtures/text.safetensors"
  wg_flags=(--unconditioned)
  case "$voice" in
    text) ;;
    design)
      scenario=design_fixed
      fixture="$OUT/preparation/$slug/fixtures/design.safetensors"
      wg_flags=(--designed)
      ;;
    clone)
      scenario=clone_prepared_fixed
      wg_flags=()
      ;;
    *) die "unknown voice: $voice" ;;
  esac
  CURRENT_PHASE="$name-python"
  wait_idle
  run_monitored "$py" env -u LD_LIBRARY_PATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    taskset -c "$CPU_SET" uv run --python 3.10 "$OUT/build/bench_python_runtime_scenarios.py" \
      --upstream "$UPSTREAM" --checkpoint "$MODEL" --codec "$PY_CODEC" --ref1 "$REF1" --ref2 "$REF2" \
      --output "$py/result.json" --work-dir "$py/work" --audio-output-dir "$py/audio" \
      --diagnostic-output-dir "$py/diagnostic" --source-fixture "$SOURCE_FIXTURE" \
      --latent-frames "$frames" --num-steps 40 --cfg-scale-caption 4 --warmups 1 --measured 1 \
      --precision fp32 --expected-pci "$GPU_PCI" --expected-gpu-name "$GPU_NAME" --scenario "$scenario" \
    || die "Python diagnostic failed: $name"
  CURRENT_PHASE="$name-wgpu"
  wait_idle
  run_monitored "$wg" env -u CUDA_VISIBLE_DEVICES WGPU_BACKEND=vulkan XDG_CACHE_HOME="$OUT/prime/xdg" \
    taskset -c "$CPU_SET" "$OUT/build/bench_v4_residency" --mode all-resident \
      --checkpoint "$MODEL" --codec-weights "$WG_CODEC" --fixture "$fixture" \
      --reference "$REF1_PREP" "$REF2_PREP" --requests 2 --warmups 1 --num-steps 40 \
      --cfg-caption 4 "${wg_flags[@]}" --precision fp32 --allocator exclusive-pages \
      --codec-residency decode-only --load-strategy parallel --rf-checkpoint-loader indexed-file \
      --cubecl-cache-dir "$wg/cubecl" --cubecl-bundle-in "$OUT/prime/environment.cubecl" \
      --audio-output-dir "$wg/audio" --diagnostic-output-dir "$wg/diagnostic" \
      --output-json "$wg/result.json" || die "WGPU diagnostic failed: $name"
  jq -e '.latency_results_valid == false and .diagnostic_artifacts != null' \
    "$py/result.json" "$wg/result.json" >/dev/null || die "diagnostic manifest gate failed: $name"
  uv run --python 3.10 "$OUT/build/compare_v4_diagnostic_tensors.py" \
    --python-artifact "$py/diagnostic/$scenario.safetensors" --wgpu-report "$wg/result.json" \
    --output "$dir/comparison.json"
done

jq -s '{format:"irodori-v4-accuracy-localization-summary-v1",latency_values_used:false,
  cases:map({case:(.pins.wgpu_report|split("/")|.[-3]),comparisons:.comparisons})}' \
  "$OUT"/cases/*/comparison.json >"$OUT/summary.json"
CURRENT_PHASE=complete
COMPLETE=1
seal COMPLETE
printf 'complete: %s\n' "$OUT"
