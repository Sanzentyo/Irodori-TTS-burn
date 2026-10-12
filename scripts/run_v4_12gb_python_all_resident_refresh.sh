#!/usr/bin/env bash
# Fresh Python all-resident comparison arm: 5 sessions x 12 requests.

set -Eeuo pipefail
IFS=$'\n\t'

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
OUT=${1:-}
[[ -n $OUT ]] || { printf 'usage: %s FRESH_OUTPUT_DIR\n' "$0" >&2; exit 2; }
OUT=$(realpath -m -- "$OUT")
[[ ! -e $OUT && ! -L $OUT ]] || { printf 'error: output exists: %s\n' "$OUT" >&2; exit 1; }

BASE=$HOME/benchmark-artifacts/irodori-v4-12gb-baseline-20260812-attempt1
FIXTURE=$BASE/accuracy-campaign/lengths/s4p48/oracle.safetensors
MODEL=$HOME/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4-Small/snapshots/e4aaac4df355ff560dcd35e0dae272c3a759317b/model.safetensors
CODEC=$HOME/.cache/huggingface/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/47376ee24834d7a05a48ebabfe3cde29b3c5e214/weights.pth
UPSTREAM=$ROOT/../Irodori-TTS
SCRIPT=$ROOT/scripts/bench_python_e2e_precision.py
LOCK=/tmp/irodori-v4-12gb-gpu0.lock
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
  printf 'status=%s\nphase=%s\nautomatic_retries=0\noutput_reuse=false\nold_measurements_pooled=false\n' \
    "$status" "$CURRENT_PHASE" >"$OUT/$status"
  (cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
}
on_exit() {
  local status=$?
  stop_monitor
  if ((status != 0 && ! COMPLETE)) && [[ -d $OUT ]]; then
    set +e
    seal FAILURE
  fi
  return "$status"
}
trap on_exit EXIT

for path in "$FIXTURE" "$MODEL" "$CODEC" "$SCRIPT"; do
  [[ -f $path && -s $path ]] || die "missing input: $path"
done
[[ $(sha256sum -- "$MODEL" | cut -d' ' -f1) == 5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593 ]] || die 'model SHA mismatch'
[[ $(sha256sum -- "$CODEC" | cut -d' ' -f1) == db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5 ]] || die 'codec SHA mismatch'
[[ $(git -C "$UPSTREAM" rev-parse HEAD) == 9f19d9a9048099a4b978a762d0509228fe624e3f ]] || die 'upstream mismatch'
gpu=$(nvidia-smi -i 0 --query-gpu=name,pci.bus_id,memory.total --format=csv,noheader,nounits)
[[ $gpu == 'NVIDIA GeForce RTX 5070 Ti Laptop GPU, 00000000:01:00.0, 12227' ]] || die "GPU mismatch: $gpu"

mkdir -p "$OUT/build" "$OUT/sessions"
install -m 0444 "$SCRIPT" "$OUT/build/bench_python_e2e_precision.py"
install -m 0444 "$ROOT/scripts/run_v4_12gb_python_all_resident_refresh.sh" "$OUT/build/runner.sh"
sha256sum "$OUT/build"/* "$FIXTURE" "$MODEL" "$CODEC" >"$OUT/pins.sha256"
printf 'source_head=%s\nupstream_commit=%s\nprecision=strict-fp32\ntf32=false\nautocast=false\nseconds=4.48\nframes=112\nvoice=unconditioned\nwarmups=2\nmeasured=10\nfresh_sessions=5\n' \
  "$(git -C "$ROOT" rev-parse HEAD)" "$(git -C "$UPSTREAM" rev-parse HEAD)" >"$OUT/protocol.txt"

exec 9>>"$LOCK"
flock -n 9 || die 'GPU0 campaign lock is held'
wait_idle() {
  local count=0 telemetry processes
  for _ in $(seq 1 30); do
    telemetry=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ $telemetry =~ ^([0-9]+),[[:space:]]*([0-9]+)$ ]] \
      && ((BASH_REMATCH[1] <= 128 && BASH_REMATCH[2] <= 5)) && [[ ! $processes =~ [0-9] ]]; then
      ((count += 1)); ((count >= 2)) && return 0
    else
      count=0
    fi
    sleep 1
  done
  die 'GPU did not settle'
}

fixture_sha=$(sha256sum -- "$FIXTURE" | cut -d' ' -f1)
for session in 1 2 3 4 5; do
  dir=$OUT/sessions/python-$session
  mkdir "$dir"
  CURRENT_PHASE=python-$session
  wait_idle
  nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader,nounits -lms 100 -f "$dir/nvml.csv" &
  ACTIVE_MONITOR=$!
  set +e
  /usr/bin/time -o "$dir/wall.txt" -f 'exit_status=%x\nelapsed_seconds=%e\nmax_rss_kib=%M' \
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 \
    uv run --python 3.10 "$OUT/build/bench_python_e2e_precision.py" \
      --precision fp32 --upstream "$UPSTREAM" --checkpoint "$MODEL" --codec "$CODEC" \
      --source-fixture "$FIXTURE" --source-fixture-sha256 "$fixture_sha" --seconds 4.48 \
      --repeats 12 --consumer-only-boundary --model-device cuda:0 --codec-device cuda:0 \
      --json-out "$dir/result.json" >"$dir/stdout.log" 2>"$dir/stderr.log"
  status=$?
  set -e
  stop_monitor
  ((status == 0)) || die "Python session failed without retry: $session"
  jq -e '.repeats == 12 and (.repeat_results | length) == 12 and .runtime_reused_across_repeats
    and .parameters.consumer_only_boundary and .parameters.precision == "fp32"' "$dir/result.json" >/dev/null \
    || die "Python result gate failed: $session"
  printf 'complete\n' >"$dir/COMPLETE"
  (cd "$dir" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
done

CURRENT_PHASE=complete
wait_idle
COMPLETE=1
seal COMPLETE
printf 'python_all_resident_refresh_complete=%s\n' "$OUT"
