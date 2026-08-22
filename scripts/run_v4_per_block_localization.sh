#!/usr/bin/env bash
# Exact-input, exact-condition per-block diagnostic for the worst 489-frame design case.

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
REF1="$SAMPLES/clone_ref1.wav"
REF2="$SAMPLES/clone_ref2.wav"
PREVIOUS="$USER_ROOT/benchmark-artifacts/irodori-v4-same-input-localization-20260822-attempt1"
SOURCE_FIXTURE="$PREVIOUS/inputs/source-noise.safetensors"
TEACHER_REPORT="$PREVIOUS/cases/f489-design/wgpu/result.json"
BUNDLE="$PREVIOUS/prime/environment.cubecl"
PY_BENCH="$ROOT/scripts/bench_python_runtime_scenarios.py"
COMPARE="$ROOT/scripts/compare_v4_forward_blocks.py"
WG_BIN="$ROOT/target/release/diagnose_v4_forward"
GPU_NAME='NVIDIA GeForce RTX 5070 Ti Laptop GPU'
GPU_PCI=00000000:01:00.0
GPU_VRAM_MIB=12227
MODEL_SHA=5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593
PY_CODEC_SHA=db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5
SOURCE_SHA=17e9016569e9e087001bebde393d7039d84e0beaee81a3fef7438a91bcdf186b
TEACHER_SHA=637200252de86b658374df82f9bcb5d7e0122ef0b166188c5e7b1ec093c23c56
BUNDLE_SHA=033727598e9a39d49d4da71ce0ff93dcfe3c93bdc64eb448f91b80ed0e439c8b
PREVIOUS_SUMS_SHA=cf5932b063de09c61cba81935f944a51f722312fd229fa2df0ba92ace72148c4
UPSTREAM_COMMIT=9f19d9a9048099a4b978a762d0509228fe624e3f
NVML_INDEX=0
CPU_SET=0-11
LOCK=/tmp/irodori-v4-per-block-localization-gpu0.lock
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
for path in "$MODEL" "$PY_CODEC" "$REF1" "$REF2" "$SOURCE_FIXTURE" \
  "$TEACHER_REPORT" "$BUNDLE" "$PY_BENCH" "$COMPARE" "$WG_BIN"; do
  [[ -f $path && -s $path ]] || die "missing input: $path"
done
[[ -z $(git -C "$ROOT" status --short) ]] || die 'Rust source tree must be clean'
[[ $(sha "$MODEL") == "$MODEL_SHA" ]] || die 'model SHA mismatch'
[[ $(sha "$PY_CODEC") == "$PY_CODEC_SHA" ]] || die 'Python codec SHA mismatch'
[[ $(sha "$SOURCE_FIXTURE") == "$SOURCE_SHA" ]] || die 'source fixture SHA mismatch'
[[ $(sha "$TEACHER_REPORT") == "$TEACHER_SHA" ]] || die 'teacher report SHA mismatch'
[[ $(sha "$BUNDLE") == "$BUNDLE_SHA" ]] || die 'CubeCL bundle SHA mismatch'
[[ $(sha "$PREVIOUS/SHA256SUMS") == "$PREVIOUS_SUMS_SHA" ]] || die 'prior campaign manifest SHA mismatch'
(cd "$PREVIOUS" && sha256sum -c SHA256SUMS >/dev/null) || die 'prior campaign verification failed'
[[ $(git -C "$UPSTREAM" rev-parse HEAD) == "$UPSTREAM_COMMIT" ]] || die 'upstream commit mismatch'
gpu_row=$(nvidia-smi -i "$NVML_INDEX" --query-gpu=name,pci.bus_id,memory.total,driver_version --format=csv,noheader,nounits)
IFS=',' read -r measured_name measured_pci measured_vram measured_driver <<<"$gpu_row"
measured_name=${measured_name## }; measured_name=${measured_name%% }
measured_pci=${measured_pci//[[:space:]]/}; measured_vram=${measured_vram//[[:space:]]/}
[[ $measured_name == "$GPU_NAME" && ${measured_pci^^} == "$GPU_PCI" && $measured_vram == "$GPU_VRAM_MIB" ]] \
  || die "GPU identity mismatch: $gpu_row"

mkdir -p "$OUT/build" "$OUT/python" "$OUT/wgpu/cubecl" "$OUT/inputs"
install -m 0555 "$WG_BIN" "$OUT/build/diagnose_v4_forward"
install -m 0444 "$PY_BENCH" "$COMPARE" "$0" "$OUT/build/"
printf 'source=%s\nupstream=%s\nmodel_revision=%s\ncodec_revision=%s\ngpu=%s\npci=%s\ndriver=%s\n' \
  "$(git -C "$ROOT" rev-parse HEAD)" "$UPSTREAM_COMMIT" "$MODEL_REV" "$CODEC_REV" \
  "$measured_name" "$measured_pci" "${measured_driver//[[:space:]]/}" >"$OUT/pins.txt"
sha256sum "$OUT/build"/* "$MODEL" "$PY_CODEC" "$REF1" "$REF2" \
  "$SOURCE_FIXTURE" "$TEACHER_REPORT" "$BUNDLE" >"$OUT/pins.sha256"
nvidia-smi -q >"$OUT/nvidia-smi-q.txt"

exec 9>>"$LOCK"
flock -n 9 || die 'per-block diagnostic GPU lock is held'
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

CURRENT_PHASE=python-exact-input-capture
wait_idle
run_monitored "$OUT/python" env -u LD_LIBRARY_PATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  taskset -c "$CPU_SET" uv run --python 3.10 "$OUT/build/bench_python_runtime_scenarios.py" \
    --upstream "$UPSTREAM" --checkpoint "$MODEL" --codec "$PY_CODEC" --ref1 "$REF1" --ref2 "$REF2" \
    --output "$OUT/python/result.json" --work-dir "$OUT/python/work" \
    --diagnostic-output-dir "$OUT/python/diagnostic" --source-fixture "$SOURCE_FIXTURE" \
    --diagnostic-forward-input-report "$TEACHER_REPORT" --diagnostic-block-forward-ordinal 7 \
    --latent-frames 489 --num-steps 40 --cfg-scale-caption 4 --warmups 1 --measured 1 \
    --precision fp32 --expected-pci "$GPU_PCI" --expected-gpu-name "$GPU_NAME" \
    --scenario design_fixed || die 'Python exact-input block capture failed'
PY_ARTIFACT="$OUT/python/diagnostic/design_fixed.safetensors"
PY_ARTIFACT_SHA=$(sha "$PY_ARTIFACT")

CURRENT_PHASE=wgpu-exact-input-forward
wait_idle
run_monitored "$OUT/wgpu" env -u CUDA_VISIBLE_DEVICES WGPU_BACKEND=vulkan XDG_CACHE_HOME="$OUT/wgpu/xdg" \
  taskset -c "$CPU_SET" "$OUT/build/diagnose_v4_forward" --checkpoint "$MODEL" \
    --input "$PY_ARTIFACT" --input-sha256 "$PY_ARTIFACT_SHA" \
    --output-dir "$OUT/wgpu/tensors" --cubecl-cache-dir "$OUT/wgpu/cubecl" \
    --cubecl-bundle-in "$BUNDLE" || die 'WGPU exact-input block forward failed'
jq -e '.diagnostic_only == true and .latency_results_valid == false and .block_count == 12' \
  "$OUT/wgpu/tensors/report.json" >/dev/null || die 'WGPU report gate failed'

CURRENT_PHASE=compare
uv run --python 3.10 "$OUT/build/compare_v4_forward_blocks.py" \
  --python-artifact "$PY_ARTIFACT" --wgpu-report "$OUT/wgpu/tensors/report.json" \
  --output "$OUT/comparison.json"
CURRENT_PHASE=complete
COMPLETE=1
seal COMPLETE
printf 'complete: %s\n' "$OUT"
