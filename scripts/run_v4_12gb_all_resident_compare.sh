#!/usr/bin/env bash
# Fresh paired all-resident comparison: 5 sessions, 2 warmups, 10 measured.

set -Eeuo pipefail
IFS=$'\n\t'

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
OUT=${1:-}
[[ -n $OUT ]] || { printf 'usage: %s FRESH_OUTPUT_DIR\n' "$0" >&2; exit 2; }
OUT=$(realpath -m -- "$OUT")
[[ ! -e $OUT && ! -L $OUT ]] || { printf 'error: output exists: %s\n' "$OUT" >&2; exit 1; }

BASE=$HOME/benchmark-artifacts/irodori-v4-12gb-baseline-20260812-attempt1
FIXTURE=$BASE/accuracy-campaign/lengths/s4p48/oracle.safetensors
REF_DIR=$BASE/phase-batch/prepared-references
REF1=$REF_DIR/ref1.safetensors
REF2=$REF_DIR/ref2.safetensors
MODEL=$HOME/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4-Small/snapshots/e4aaac4df355ff560dcd35e0dae272c3a759317b/model.safetensors
PY_CODEC=$HOME/.cache/huggingface/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/47376ee24834d7a05a48ebabfe3cde29b3c5e214/weights.pth
WG_CODEC=$ROOT/target/v4_dacvae_weights.safetensors
UPSTREAM=$ROOT/../Irodori-TTS
WG_BIN=$ROOT/target/release/bench_v4_residency
PY_SCRIPT=$ROOT/scripts/bench_python_e2e_precision.py
MODEL_SHA=5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593
PY_CODEC_SHA=db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5
WG_CODEC_SHA=4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1
GPU_NAME='NVIDIA GeForce RTX 5070 Ti Laptop GPU'
GPU_PCI=00000000:01:00.0
LOCK=/tmp/irodori-v4-12gb-gpu0.lock
ACTIVE_MONITOR=
CURRENT_PHASE=preflight
COMPLETE=0

sha() { sha256sum -- "$1" | awk '{print $1}'; }
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
  [[ -d $OUT && ! -L $OUT ]] || return 0
  printf 'status=%s\nphase=%s\nautomatic_retries=0\noutput_reuse=false\n' "$status" "$CURRENT_PHASE" >"$OUT/$status"
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

for path in "$FIXTURE" "$REF1" "$REF2" "$MODEL" "$PY_CODEC" "$WG_CODEC" "$WG_BIN" "$PY_SCRIPT"; do
  [[ -f $path && -s $path ]] || die "missing input: $path"
done
[[ $(sha "$MODEL") == "$MODEL_SHA" ]] || die 'model SHA mismatch'
[[ $(sha "$PY_CODEC") == "$PY_CODEC_SHA" ]] || die 'Python codec SHA mismatch'
[[ $(sha "$WG_CODEC") == "$WG_CODEC_SHA" ]] || die 'WGPU codec SHA mismatch'
[[ $(git -C "$UPSTREAM" rev-parse HEAD) == 9f19d9a9048099a4b978a762d0509228fe624e3f ]] || die 'upstream mismatch'
row=$(nvidia-smi -i 0 --query-gpu=name,pci.bus_id,memory.total --format=csv,noheader,nounits)
[[ $row == "$GPU_NAME, $GPU_PCI, 12227" ]] || die "GPU identity mismatch: $row"

mkdir -p "$OUT/build" "$OUT/sessions"
install -m 0555 "$WG_BIN" "$OUT/build/bench_v4_residency"
install -m 0444 "$PY_SCRIPT" "$OUT/build/bench_python_e2e_precision.py"
install -m 0444 "$ROOT/src/bin/bench_v4_residency.rs" "$OUT/build/bench_v4_residency.rs"
install -m 0444 "$ROOT/scripts/run_v4_12gb_all_resident_compare.sh" "$OUT/build/runner.sh"
sha256sum "$OUT/build"/* "$FIXTURE" "$REF1" "$REF2" "$MODEL" "$PY_CODEC" "$WG_CODEC" >"$OUT/pins.sha256"
git -C "$ROOT" diff --binary >"$OUT/source.diff"
printf 'source_head=%s\nsource_diff_sha256=%s\nupstream_commit=%s\ngpu_name=%s\ngpu_pci=%s\nnvml_index=0\nwgpu_adapter_index=0\nprecision=strict-fp32\ntf32=false\nautocast=false\nseconds=4.48\nframes=112\nvoice=unconditioned\nwarmups=2\nmeasured=10\nfresh_sessions=5\n' \
  "$(git -C "$ROOT" rev-parse HEAD)" "$(sha "$OUT/source.diff")" "$(git -C "$UPSTREAM" rev-parse HEAD)" "$GPU_NAME" "$GPU_PCI" >"$OUT/protocol.txt"

exec 9>>"$LOCK"
flock -n 9 || die 'GPU0 campaign lock is held'

wait_idle() {
  local count=0 telemetry processes
  for _ in $(seq 1 30); do
    telemetry=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ $telemetry =~ ^([0-9]+),[[:space:]]*([0-9]+)$ ]] && ((BASH_REMATCH[1] <= 128 && BASH_REMATCH[2] <= 5)) && [[ ! $processes =~ [0-9] ]]; then
      ((count += 1))
      ((count >= 2)) && return 0
    else
      count=0
    fi
    sleep 1
  done
  die 'GPU did not settle'
}

run_monitored() {
  local dir=$1
  shift
  nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits -lms 100 -f "$dir/nvml.csv" &
  ACTIVE_MONITOR=$!
  set +e
  /usr/bin/time -o "$dir/wall.txt" -f 'exit_status=%x\nelapsed_seconds=%e\nmax_rss_kib=%M' "$@" >"$dir/stdout.log" 2>"$dir/stderr.log"
  local status=$?
  set -e
  stop_monitor
  return "$status"
}

fixture_sha=$(sha "$FIXTURE")
for session in 1 2 3 4 5; do
  py=$OUT/sessions/python-$session
  mkdir "$py"
  CURRENT_PHASE=python-$session
  wait_idle
  run_monitored "$py" env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 \
    uv run --python 3.10 "$OUT/build/bench_python_e2e_precision.py" \
      --precision fp32 --upstream "$UPSTREAM" --checkpoint "$MODEL" --codec "$PY_CODEC" \
      --source-fixture "$FIXTURE" --source-fixture-sha256 "$fixture_sha" --seconds 4.48 \
      --repeats 12 --consumer-only-boundary --model-device cuda:0 --codec-device cuda:0 \
      --json-out "$py/result.json" \
    || die "Python session failed without retry: $session"
  jq -e '.repeats == 12 and (.repeat_results | length) == 12 and .runtime_reused_across_repeats and .parameters.consumer_only_boundary' "$py/result.json" >/dev/null || die "Python result gate failed: $session"
  printf 'complete\n' >"$py/COMPLETE"
  (cd "$py" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)

  wg=$OUT/sessions/wgpu-$session
  mkdir "$wg" "$wg/xdg-cache"
  CURRENT_PHASE=wgpu-$session
  wait_idle
  run_monitored "$wg" env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID WGPU_BACKEND=vulkan XDG_CACHE_HOME="$wg/xdg-cache" \
    "$OUT/build/bench_v4_residency" --mode all-resident --checkpoint "$MODEL" --codec-weights "$WG_CODEC" \
      --fixture "$FIXTURE" --reference "$REF1" "$REF2" --requests 12 --warmups 2 --unconditioned \
      --speaker-mode same --length-mode same --adapter-index 0 --output-json "$wg/result.json" \
    || die "WGPU session failed without retry: $session"
  jq -e '.requests == 12 and .warmups == 2 and .measured == 10 and .unconditioned and (.resident_request_timings | length) == 12 and .work_report.num_steps == 4' "$wg/result.json" >/dev/null || die "WGPU result gate failed: $session"
  printf 'complete\n' >"$wg/COMPLETE"
  (cd "$wg" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
done

CURRENT_PHASE=complete
wait_idle
COMPLETE=1
seal COMPLETE
printf 'all_resident_compare_complete=%s\n' "$OUT"
