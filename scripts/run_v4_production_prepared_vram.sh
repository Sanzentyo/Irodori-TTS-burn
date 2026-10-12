#!/usr/bin/env bash
# A/B the general-length production-prepared RF residency profile.

set -Eeuo pipefail
IFS=$'\n\t'

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
USER_ROOT=$(realpath -- "$ROOT/../..")
OUT=
INPUT=
while (($#)); do
  case "$1" in
    --output-dir) OUT=${2:?}; shift 2 ;;
    --input-campaign) INPUT=${2:?}; shift 2 ;;
    -h|--help) printf 'usage: %s --output-dir FRESH --input-campaign SEALED_ACCURACY_CAMPAIGN\n' "$0"; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n $OUT && -n $INPUT ]] || { printf 'error: both paths are required\n' >&2; exit 2; }
OUT=$(realpath -m -- "$OUT")
INPUT=$(realpath -- "$INPUT")
[[ ! -e $OUT && ! -L $OUT ]] || { printf 'error: output exists: %s\n' "$OUT" >&2; exit 1; }
[[ -f $INPUT/COMPLETE && -f $INPUT/SHA256SUMS ]] || { printf 'error: input campaign is not sealed\n' >&2; exit 1; }
(cd "$INPUT" && sha256sum -c SHA256SUMS >/dev/null) || { printf 'error: input campaign SHA failure\n' >&2; exit 1; }

MODEL="$USER_ROOT/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4-Small/snapshots/e4aaac4df355ff560dcd35e0dae272c3a759317b/model.safetensors"
CODEC="$USER_ROOT/benchmark-artifacts/irodori-v4-load-opt-20260813-attempt1/models/dacvae-decoder-only.safetensors"
BIN="$ROOT/target/release/bench_v4_residency"
REF1="$INPUT/inputs/references/ref1.safetensors"
REF2="$INPUT/inputs/references/ref2.safetensors"
BUNDLE="$INPUT/prime/environment.cubecl"
LOCK=/tmp/irodori-v4-production-prepared-vram-gpu0.lock
ACTIVE_MONITOR=
CURRENT_PHASE=preflight
COMPLETE=0

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
stop_monitor() { if [[ -n $ACTIVE_MONITOR ]]; then kill "$ACTIVE_MONITOR" 2>/dev/null || true; wait "$ACTIVE_MONITOR" 2>/dev/null || true; ACTIVE_MONITOR=; fi; }
seal() {
  local status=$1
  stop_monitor
  [[ -d $OUT ]] || return 0
  printf 'status=%s\nphase=%s\nautomatic_retries=0\n' "$status" "$CURRENT_PHASE" >"$OUT/$status"
  (cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
}
on_exit() { local s=$?; stop_monitor; if ((s != 0 && ! COMPLETE)); then set +e; seal FAILURE; fi; return "$s"; }
trap on_exit EXIT

for path in "$MODEL" "$CODEC" "$BIN" "$REF1" "$REF2" "$BUNDLE"; do [[ -f $path && -s $path ]] || die "missing $path"; done
[[ -z $(git -C "$ROOT" status --short) ]] || die 'source tree must be clean'
mkdir -p "$OUT/build" "$OUT/sessions" "$OUT/xdg"
install -m 0555 "$BIN" "$OUT/build/bench_v4_residency"
install -m 0444 "$0" "$OUT/build/runner.sh"
git -C "$ROOT" rev-parse HEAD >"$OUT/source-head.txt"
sha256sum "$OUT/build"/* "$MODEL" "$CODEC" "$REF1" "$REF2" "$BUNDLE" >"$OUT/pins.sha256"
nvidia-smi -q >"$OUT/nvidia-smi-q.txt"

exec 9>>"$LOCK"
flock -n 9 || die 'VRAM campaign GPU lock is held'
wait_idle() {
  local quiet=0 row pids
  for _ in $(seq 1 60); do
    row=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    pids=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ $row =~ ^([0-9]+),[[:space:]]*([0-9]+)$ ]] && ((BASH_REMATCH[1] <= 512 && BASH_REMATCH[2] <= 5)) && [[ ! $pids =~ [0-9] ]]; then
      ((quiet += 1)); ((quiet >= 2)) && return 0
    else quiet=0; fi
    sleep 1
  done
  die 'GPU did not become idle'
}
run_one() {
  local case_name=$1 profile=$2 session=$3 frames fixture flags
  case "$case_name" in
    f112-text) frames=112; fixture="$INPUT/preparation/f112/fixtures/text.safetensors"; flags=(--unconditioned) ;;
    f489-design) frames=489; fixture="$INPUT/preparation/f489/fixtures/design.safetensors"; flags=(--designed) ;;
    f685-clone) frames=685; fixture="$INPUT/preparation/f685/fixtures/text.safetensors"; flags=() ;;
    *) die "unknown case $case_name" ;;
  esac
  local dir="$OUT/sessions/$case_name-$profile-s$session"
  mkdir -p "$dir/cubecl"
  CURRENT_PHASE="$case_name-$profile-s$session"
  wait_idle
  nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits -lms 100 -f "$dir/nvml.csv" & ACTIVE_MONITOR=$!
  set +e
  /usr/bin/time -o "$dir/wall.txt" -f 'exit_status=%x\nelapsed_seconds=%e\nmax_rss_kib=%M' \
    env -u CUDA_VISIBLE_DEVICES WGPU_BACKEND=vulkan XDG_CACHE_HOME="$OUT/xdg" \
    taskset -c 0-11 "$OUT/build/bench_v4_residency" --mode all-resident \
      --checkpoint "$MODEL" --codec-weights "$CODEC" --fixture "$fixture" \
      --reference "$REF1" "$REF2" --requests 7 --warmups 2 --num-steps 40 --cfg-caption 4 \
      "${flags[@]}" --precision fp32 --allocator exclusive-pages --codec-residency decode-only \
      --load-strategy parallel --rf-checkpoint-loader indexed-file --rf-weight-residency "$profile" \
      --cubecl-cache-dir "$dir/cubecl" --cubecl-bundle-in "$BUNDLE" --output-json "$dir/result.json" \
      >"$dir/stdout.log" 2>"$dir/stderr.log"
  local status=$?
  set -e
  stop_monitor
  ((status == 0)) || die "condition failed: $CURRENT_PHASE"
  jq -e --argjson frames "$frames" --arg profile "${profile//-/_}" '
    .latency_results_valid and .euler_evaluations==40 and .rf_weight_residency==$profile and
    (.items|all(.frames==$frames)) and ([.items[].audio_f32_sha256]|unique|length==1)
  ' "$dir/result.json" >/dev/null || die "result gate failed: $CURRENT_PHASE"
}

for case_name in f112-text f489-design f685-clone; do
  for session in 1 2 3; do
    if (((session + ${#case_name}) % 2)); then profiles=(portable-fallback production-prepared); else profiles=(production-prepared portable-fallback); fi
    for profile in "${profiles[@]}"; do run_one "$case_name" "$profile" "$session"; done
  done
done

jq -s '{format:"irodori-v4-production-prepared-vram-v1",steps:40,warmups:2,measured:5,
  rows:map(. as $r|($r.memory[]|select(.stage=="rf_duration_codec_resident")) as $m|
    {case:($r.items[0].frames|tostring)+"-"+(if $r.designed then "design" elif $r.unconditioned then "text" else "clone" end),
     profile:$r.rf_weight_residency,persistent_in_use_bytes:$m.bytes_in_use,persistent_reserved_bytes:$m.bytes_reserved,
     consumer_seconds:[$r.resident_request_timings[]|select(.warmup|not)|.consumer_complete_seconds],
     audio_hash:($r.items[0].audio_f32_sha256)})}' "$OUT"/sessions/*/result.json >"$OUT/summary.json"
jq -e '([.rows|group_by(.case)[]|([.[].audio_hash]|unique|length==1)]|all) and
  ([.rows[]|select(.profile=="production_prepared")|.persistent_in_use_bytes] | all(. < 4500000000))' \
  "$OUT/summary.json" >/dev/null || die 'summary accuracy/VRAM gate failed'
CURRENT_PHASE=complete
COMPLETE=1
seal COMPLETE
printf 'complete: %s\n' "$OUT"
