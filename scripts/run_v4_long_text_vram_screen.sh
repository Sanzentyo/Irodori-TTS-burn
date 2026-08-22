#!/usr/bin/env bash
# Fresh paired screen for the source-free long-text serving profile.

set -Eeuo pipefail
IFS=$'\n\t'

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
USER_ROOT=$(realpath -- "$ROOT/../..")
OUT=
FIXTURE=
REFERENCE_DIR=
BUNDLE=
while (($#)); do
  case "$1" in
    --output-dir) OUT=${2:?}; shift 2 ;;
    --fixture) FIXTURE=${2:?}; shift 2 ;;
    --reference-dir) REFERENCE_DIR=${2:?}; shift 2 ;;
    --cubecl-bundle-in) BUNDLE=${2:?}; shift 2 ;;
    -h|--help)
      printf 'usage: %s --output-dir FRESH --fixture TEXT.safetensors --reference-dir DIR --cubecl-bundle-in FILE\n' "$0"
      exit 0
      ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n $OUT && -n $FIXTURE && -n $REFERENCE_DIR && -n $BUNDLE ]] || {
  printf 'error: all arguments are required\n' >&2
  exit 2
}
OUT=$(realpath -m -- "$OUT")
FIXTURE=$(realpath -- "$FIXTURE")
REFERENCE_DIR=$(realpath -- "$REFERENCE_DIR")
BUNDLE=$(realpath -- "$BUNDLE")
[[ ! -e $OUT && ! -L $OUT ]] || { printf 'error: output exists: %s\n' "$OUT" >&2; exit 1; }

MODEL_REV=e4aaac4df355ff560dcd35e0dae272c3a759317b
MODEL="$USER_ROOT/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4-Small/snapshots/$MODEL_REV/model.safetensors"
CODEC="$USER_ROOT/benchmark-artifacts/irodori-v4-load-opt-20260813-attempt1/models/dacvae-decoder-only.safetensors"
REF1="$REFERENCE_DIR/ref1.safetensors"
REF2="$REFERENCE_DIR/ref2.safetensors"
BIN="$ROOT/target/release/bench_v4_residency"
MODEL_SHA=5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593
CODEC_SHA=1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231
GPU_NAME='NVIDIA GeForce RTX 5070 Ti Laptop GPU'
GPU_PCI=00000000:01:00.0
GPU_VRAM_MIB=12227
ACTIVE_MONITOR=
CURRENT_PHASE=preflight
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
  printf 'status=%s\nphase=%s\nautomatic_retries=0\n' "$status" "$CURRENT_PHASE" >"$OUT/$status"
  (cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
}
on_exit() {
  local status=$?
  stop_monitor
  if ((status != 0 && ! COMPLETE)); then set +e; seal FAILURE; fi
  return "$status"
}
trap on_exit EXIT

for command in flock git jq nvidia-smi taskset; do command -v "$command" >/dev/null || die "missing $command"; done
for path in "$MODEL" "$CODEC" "$REF1" "$REF2" "$FIXTURE" "$BUNDLE" "$BIN"; do
  [[ -f $path && -s $path ]] || die "missing input: $path"
done
[[ -z $(git -C "$ROOT" status --short) ]] || die 'source tree must be clean'
[[ $(sha "$MODEL") == "$MODEL_SHA" ]] || die 'model SHA mismatch'
[[ $(sha "$CODEC") == "$CODEC_SHA" ]] || die 'codec SHA mismatch'
gpu_row=$(nvidia-smi -i 0 --query-gpu=name,pci.bus_id,memory.total --format=csv,noheader,nounits)
IFS=',' read -r measured_name measured_pci measured_vram <<<"$gpu_row"
measured_name=${measured_name## }; measured_name=${measured_name%% }
measured_pci=${measured_pci//[[:space:]]/}; measured_vram=${measured_vram//[[:space:]]/}
[[ $measured_name == "$GPU_NAME" && ${measured_pci^^} == "$GPU_PCI" && $measured_vram == "$GPU_VRAM_MIB" ]] \
  || die "GPU identity mismatch: $gpu_row"

mkdir -p "$OUT/build" "$OUT/sessions"
install -m 0555 "$BIN" "$OUT/build/bench_v4_residency"
install -m 0444 "$0" "$OUT/build/runner.sh"
git -C "$ROOT" rev-parse HEAD >"$OUT/source-head.txt"
nvidia-smi -q >"$OUT/nvidia-smi-q.txt"
sha256sum "$OUT/build"/* "$MODEL" "$CODEC" "$FIXTURE" "$REF1" "$REF2" "$BUNDLE" >"$OUT/pins.sha256"

exec 9>>/tmp/irodori-v4-long-text-vram-gpu0.lock
flock -n 9 || die 'GPU lock is held'
wait_idle() {
  local quiet=0 row processes
  for _ in $(seq 1 60); do
    row=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ $row =~ ^([0-9]+),[[:space:]]*([0-9]+)$ ]] \
      && ((BASH_REMATCH[1] <= 512 && BASH_REMATCH[2] <= 5)) && [[ ! $processes =~ [0-9] ]]; then
      ((quiet += 1)); ((quiet >= 2)) && return 0
    else quiet=0; fi
    sleep 1
  done
  die 'GPU did not become idle'
}

for session in 1 2 3; do
  if ((session % 2)); then profiles=(production-prepared long-text-prepared-only); else profiles=(long-text-prepared-only production-prepared); fi
  for profile in "${profiles[@]}"; do
    name="s${session}-${profile}"
    dir="$OUT/sessions/$name"
    mkdir -p "$dir/cache"
    CURRENT_PHASE=$name
    wait_idle
    nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits -lms 100 -f "$dir/nvml.csv" &
    ACTIVE_MONITOR=$!
    set +e
    env -u CUDA_VISIBLE_DEVICES WGPU_BACKEND=vulkan XDG_CACHE_HOME="$dir/xdg" \
      taskset -c 0-11 "$OUT/build/bench_v4_residency" --mode all-resident \
        --checkpoint "$MODEL" --codec-weights "$CODEC" --fixture "$FIXTURE" \
        --reference "$REF1" "$REF2" --requests 6 --warmups 1 --num-steps 40 \
        --unconditioned --precision fp32 --allocator exclusive-pages \
        --codec-residency decode-only --load-strategy parallel --rf-checkpoint-loader indexed-file \
        --rf-weight-residency "$profile" --cubecl-cache-dir "$dir/cache" \
        --cubecl-bundle-in "$BUNDLE" --audio-output-dir "$dir/audio" \
        --output-json "$dir/result.json" >"$dir/stdout.log" 2>"$dir/stderr.log"
    status=$?
    set -e
    stop_monitor
    ((status == 0)) || die "session failed: $name"
  done
done

jq -s '
  def median: sort | .[length/2];
  {format:"irodori-v4-long-text-vram-screen-v1", sessions: map(
    . as $r | ($r.audio_artifacts[0].path|split("/")|.[-3]) as $session |
    {session:$session, profile:.rf_weight_residency,
     persistent:(.memory[]|select(.stage=="rf_duration_codec_resident")|{bytes_in_use,bytes_reserved}),
     consumer_median_seconds:([.resident_request_timings[]|select(.warmup|not)|.consumer_complete_seconds]|median),
     measured_hashes:[.resident_request_timings[]|select(.warmup|not)|.audio_f32_sha256]|unique})
  }' "$OUT"/sessions/*/result.json >"$OUT/summary.json"
[[ $(jq '[.sessions[].measured_hashes|length]|max' "$OUT/summary.json") == 1 ]] || die 'nondeterministic output'

CURRENT_PHASE=complete
COMPLETE=1
seal COMPLETE
printf 'complete: %s\n' "$OUT"
