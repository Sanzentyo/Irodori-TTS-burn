#!/usr/bin/env bash
# Fresh-process paired allocator campaign for the 40-step f489 Voice Design path.

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
    -h|--help)
      printf 'usage: %s --output-dir FRESH --input-campaign SEALED_40STEP_CAMPAIGN\n' "$0"
      exit 0
      ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n $OUT && -n $INPUT ]] || { printf 'error: both paths are required\n' >&2; exit 2; }
OUT=$(realpath -m -- "$OUT")
INPUT=$(realpath -- "$INPUT")
[[ ! -e $OUT && ! -L $OUT ]] || { printf 'error: output exists: %s\n' "$OUT" >&2; exit 1; }
[[ -f $INPUT/COMPLETE && -f $INPUT/SHA256SUMS ]] || {
  printf 'error: input campaign is not sealed\n' >&2
  exit 1
}

MODEL="$USER_ROOT/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4-Small/snapshots/e4aaac4df355ff560dcd35e0dae272c3a759317b/model.safetensors"
CODEC="$USER_ROOT/benchmark-artifacts/irodori-v4-load-opt-20260813-attempt1/models/dacvae-decoder-only.safetensors"
FIXTURE="$INPUT/preparation/f489/fixtures/design.safetensors"
REF1="$INPUT/inputs/references/ref1.safetensors"
REF2="$INPUT/inputs/references/ref2.safetensors"
BUNDLE="$INPUT/prime/environment.cubecl"
BIN="$ROOT/target/release/bench_v4_residency"
LOCK=/tmp/irodori-v4-allocator-ab-gpu0.lock
ACTIVE_MONITOR=
CURRENT_PHASE=preflight
COMPLETE=0

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
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
  [[ -d $OUT ]] || return 0
  printf 'status=%s\nphase=%s\nautomatic_retries=0\n' "$status" "$CURRENT_PHASE" >"$OUT/$status"
  (cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
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

for command in cargo flock git jq nvidia-smi taskset; do
  command -v "$command" >/dev/null || die "missing command: $command"
done
for path in "$MODEL" "$CODEC" "$FIXTURE" "$REF1" "$REF2" "$BUNDLE"; do
  [[ -f $path && -s $path ]] || die "missing input: $path"
done
(cd "$INPUT" && sha256sum -c SHA256SUMS >/dev/null) || die 'input campaign SHA check failed'
[[ -z $(git -C "$ROOT" status --short) ]] || die 'source tree must be clean'

cargo build --release --locked --features inference,codec,cli,profile --bin bench_v4_residency
mkdir -p "$OUT/build" "$OUT/sessions"
install -m 0555 "$BIN" "$OUT/build/bench_v4_residency"
install -m 0444 "$0" "$OUT/build/runner.sh"
git -C "$ROOT" rev-parse HEAD >"$OUT/source-head.txt"
git -C "$ROOT" status --short >"$OUT/source-status.txt"
nvidia-smi -q >"$OUT/nvidia-smi-q.txt"
sha256sum "$OUT/build"/* "$MODEL" "$CODEC" "$FIXTURE" "$REF1" "$REF2" "$BUNDLE" >"$OUT/pins.sha256"

exec 9>>"$LOCK"
flock -n 9 || die 'allocator campaign GPU lock is held'

wait_idle() {
  local quiet=0 row pids
  for _ in $(seq 1 60); do
    row=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    pids=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ $row =~ ^([0-9]+),[[:space:]]*([0-9]+)$ ]] &&
       ((BASH_REMATCH[1] <= 512 && BASH_REMATCH[2] <= 5)) &&
       [[ ! $pids =~ [0-9] ]]; then
      ((quiet += 1))
      ((quiet >= 2)) && return 0
    else
      quiet=0
    fi
    sleep 1
  done
  die 'GPU did not become idle'
}

run_one() {
  local allocator=$1 session=$2
  local dir="$OUT/sessions/$allocator-s$session"
  CURRENT_PHASE="$allocator-s$session"
  mkdir -p "$dir/cubecl" "$dir/xdg"
  wait_idle
  nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits -lms 50 -f "$dir/nvml.csv" &
  ACTIVE_MONITOR=$!
  set +e
  /usr/bin/time -o "$dir/wall.txt" -f 'exit_status=%x\nelapsed_seconds=%e\nmax_rss_kib=%M' \
    env -u CUDA_VISIBLE_DEVICES WGPU_BACKEND=vulkan XDG_CACHE_HOME="$dir/xdg" \
    taskset -c 0-11 "$OUT/build/bench_v4_residency" --mode all-resident \
      --checkpoint "$MODEL" --codec-weights "$CODEC" --fixture "$FIXTURE" \
      --reference "$REF1" "$REF2" --requests 7 --warmups 2 --num-steps 40 --cfg-caption 4 \
      --designed --precision fp32 --allocator "$allocator" --codec-residency decode-only \
      --load-strategy parallel --rf-checkpoint-loader indexed-file \
      --rf-weight-residency exact-manifest --cubecl-cache-dir "$dir/cubecl" \
      --cubecl-bundle-in "$BUNDLE" --output-json "$dir/result.json" \
      >"$dir/stdout.log" 2>"$dir/stderr.log"
  local status=$?
  set -e
  stop_monitor
  ((status == 0)) || die "condition failed: $CURRENT_PHASE"
  jq -e --arg allocator "${allocator//-/_}" '
    .latency_results_valid and .euler_evaluations == 40 and .block_calls == 480 and
    .allocator == $allocator and .precision == "fp32" and .designed and
    .rf_weight_residency == "exact_manifest" and
    ([.resident_request_timings[].audio_f32_sha256] | unique | length == 1)
  ' "$dir/result.json" >/dev/null || die "result gate failed: $CURRENT_PHASE"
}

for session in 1 2 3 4 5; do
  if ((session % 2)); then
    allocators=(exclusive-pages sub-slices)
  else
    allocators=(sub-slices exclusive-pages)
  fi
  for allocator in "${allocators[@]}"; do
    run_one "$allocator" "$session"
  done
done

jq -s '
  def median: sort | .[length / 2 | floor];
  {
    format: "irodori-v4-allocator-ab-v1",
    steps: 40,
    warmups_per_process: 2,
    measured_per_process: 5,
    fresh_sessions_per_allocator: 5,
    rows: map(
      (.memory[] | select(.stage == "rf_duration_codec_resident")) as $idle |
      (.memory[] | select(.stage == "all_resident_after_consumer")) as $after |
      {
        allocator,
        rf_device_ms: [.resident_request_timings[] | select(.warmup | not) | .rf_device_complete_seconds * 1000],
        consumer_ms: [.resident_request_timings[] | select(.warmup | not) | .consumer_complete_seconds * 1000],
        idle_reserved_bytes: $idle.bytes_reserved,
        after_reserved_bytes: $after.bytes_reserved,
        audio_hash: .resident_request_timings[-1].audio_f32_sha256
      }
    ),
    aggregate: (
      map(
        (.memory[] | select(.stage == "rf_duration_codec_resident")) as $idle |
        (.memory[] | select(.stage == "all_resident_after_consumer")) as $after |
        {
          allocator,
          rf_median_ms: ([.resident_request_timings[] | select(.warmup | not) | .rf_device_complete_seconds * 1000] | median),
          consumer_median_ms: ([.resident_request_timings[] | select(.warmup | not) | .consumer_complete_seconds * 1000] | median),
          idle_reserved_bytes: $idle.bytes_reserved,
          after_reserved_bytes: $after.bytes_reserved
        }
      ) | group_by(.allocator) | map({
        allocator: .[0].allocator,
        session_rf_median_ms: [.[].rf_median_ms],
        rf_median_of_medians_ms: ([.[].rf_median_ms] | median),
        consumer_median_of_medians_ms: ([.[].consumer_median_ms] | median),
        idle_reserved_median_bytes: ([.[].idle_reserved_bytes] | median),
        after_reserved_median_bytes: ([.[].after_reserved_bytes] | median)
      })
    )
  }
' "$OUT"/sessions/*/result.json >"$OUT/summary.json"

jq -e '
  (.rows | length == 10) and
  ([.rows[].audio_hash] | unique | length == 1) and
  ([.aggregate[].allocator] | sort == ["exclusive_pages", "sub_slices"])
' "$OUT/summary.json" >/dev/null || die 'summary gate failed'

CURRENT_PHASE=complete
COMPLETE=1
seal COMPLETE
printf 'complete: %s\n' "$OUT"
