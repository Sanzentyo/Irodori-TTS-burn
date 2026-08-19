#!/usr/bin/env bash
# Fresh all-resident eager/captured codec A/B with identical restored tune state.

set -Eeuo pipefail
IFS=$'\n\t'

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
OUT=${1:-}
[[ -n $OUT ]] || { printf 'usage: %s FRESH_OUTPUT_DIR\n' "$0" >&2; exit 2; }
OUT=$(realpath -m -- "$OUT")
[[ ! -e $OUT && ! -L $OUT ]] || { printf 'error: output exists: %s\n' "$OUT" >&2; exit 1; }

LOAD_ARTIFACT=$HOME/benchmark-artifacts/irodori-v4-load-opt-20260813-attempt1
MODEL=$HOME/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4-Small/snapshots/e4aaac4df355ff560dcd35e0dae272c3a759317b/model.safetensors
CODEC=$LOAD_ARTIFACT/models/dacvae-decoder-only.safetensors
FIXTURE=$LOAD_ARTIFACT/inputs/fixture-112.safetensors
REF1=$LOAD_ARTIFACT/inputs/ref1.safetensors
REF2=$LOAD_ARTIFACT/inputs/ref2.safetensors
BIN=$ROOT/target/release/bench_v4_residency
MODEL_SHA=5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593
CODEC_SHA=1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231
FIXTURE_SHA=f90e785823da3a0ec05caddadfc3d337bf833ad003daa9ff968f42086043d032
GPU_NAME='NVIDIA GeForce RTX 5070 Ti Laptop GPU'
GPU_PCI=00000000:01:00.0
LOCK=/tmp/irodori-v4-captured-all-resident-gpu0.lock
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
  printf 'status=%s\nphase=%s\nautomatic_retries=0\noutput_reuse=false\n' \
    "$status" "$CURRENT_PHASE" >"$OUT/$status"
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

for path in "$MODEL" "$CODEC" "$FIXTURE" "$REF1" "$REF2" "$BIN"; do
  [[ -f $path && -s $path ]] || die "missing input: $path"
done
[[ $(sha "$MODEL") == "$MODEL_SHA" ]] || die 'model SHA mismatch'
[[ $(sha "$CODEC") == "$CODEC_SHA" ]] || die 'codec SHA mismatch'
[[ $(sha "$FIXTURE") == "$FIXTURE_SHA" ]] || die 'fixture SHA mismatch'
gpu_row=$(nvidia-smi -i 0 --query-gpu=name,driver_version,pci.bus_id,memory.total,index --format=csv,noheader,nounits)
[[ $gpu_row == "$GPU_NAME, 595.71.05, $GPU_PCI, 12227, 0" ]] || die "GPU identity mismatch: $gpu_row"

mkdir -p "$OUT/build" "$OUT/cache" "$OUT/environment" "$OUT/prime" "$OUT/sessions"
install -m 0555 "$BIN" "$OUT/build/bench_v4_residency"
install -m 0444 "$ROOT/src/bin/bench_v4_residency.rs" "$OUT/build/bench_v4_residency.rs"
install -m 0444 "$ROOT/src/codec/graph.rs" "$OUT/build/codec-graph.rs"
install -m 0444 "$ROOT/vendor/cubecl-wgpu/src/compute/server.rs" "$OUT/build/cubecl-wgpu-server.rs"
install -m 0444 "$ROOT/vendor/cubecl-wgpu/src/compute/stream.rs" "$OUT/build/cubecl-wgpu-stream.rs"
install -m 0444 "$ROOT/scripts/run_v4_captured_all_resident_ab.sh" "$OUT/build/runner.sh"
git -C "$ROOT" diff --binary >"$OUT/source.diff"
sha256sum "$OUT/build"/* "$MODEL" "$CODEC" "$FIXTURE" "$REF1" "$REF2" >"$OUT/pins.sha256"
nvidia-smi -q >"$OUT/environment/nvidia-smi-q.txt"
nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id,memory.total,memory.free --format=csv,noheader,nounits \
  >"$OUT/environment/nvidia-smi.csv"
if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary >"$OUT/environment/vulkan-summary.txt" 2>&1
else
  printf 'vulkaninfo=not-installed\nadapter_source=bench_v4_residency WGPU initialization\n' \
    >"$OUT/environment/vulkan-summary.txt"
fi
rustc -Vv >"$OUT/environment/rustc.txt"
cargo -V >"$OUT/environment/cargo.txt"
printf 'source_head=%s\nsource_diff_sha256=%s\nprecision=strict-fp32\ntf32=false\nautocast=false\nseconds=4.48\nframes=112\nvoice=unconditioned\nwarmups=2\nmeasured=10\nfresh_sessions=5\ncodec_weights=decoder-only\nrf_weights=fixed112-packed-only\ncodec_weights_profile=portable-fallback\nold_measurements_pooled=false\nautomatic_retries=0\n' \
  "$(git -C "$ROOT" rev-parse HEAD)" "$(sha "$OUT/source.diff")" >"$OUT/protocol.txt"

exec 9>>"$LOCK"
flock -n 9 || die 'GPU0 campaign lock is held'

wait_idle() {
  local count=0 telemetry processes
  for _ in $(seq 1 30); do
    telemetry=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ $telemetry =~ ^([0-9]+),[[:space:]]*([0-9]+)$ ]] \
      && ((BASH_REMATCH[1] <= 640 && BASH_REMATCH[2] <= 5)) \
      && [[ ! $processes =~ [0-9] ]]; then
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

common_args=(
  --mode all-resident
  --checkpoint "$MODEL"
  --codec-weights "$CODEC"
  --fixture "$FIXTURE"
  --reference "$REF1" "$REF2"
  --unconditioned
  --speaker-mode same
  --length-mode same
  --adapter-index 0
  --allocator exclusive-pages
  --codec-residency decode-only
  --duration-residency exact-only
  --rf-weight-residency fixed112-packed-only
  --codec-weight-residency portable-fallback
  --startup-warmup dry-run
)

CURRENT_PHASE=prime
mkdir "$OUT/cache/prime" "$OUT/prime/xdg-cache"
wait_idle
run_monitored "$OUT/prime" env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID WGPU_BACKEND=vulkan \
  XDG_CACHE_HOME="$OUT/prime/xdg-cache" "$OUT/build/bench_v4_residency" "${common_args[@]}" \
  --codec-execution eager --requests 1 --warmups 0 --output-json "$OUT/prime/result.json" \
  --cubecl-cache-dir "$OUT/cache/prime" --cubecl-bundle-out "$OUT/prime/environment.sqlite" \
  || die 'prime failed without retry'
jq -e '.schema_version == 6 and .requests == 1 and .codec_execution == "eager"' \
  "$OUT/prime/result.json" >/dev/null || die 'prime result gate failed'

run_condition() {
  local condition=$1 session=$2 dir=$OUT/sessions/$condition-$session
  mkdir "$dir" "$dir/xdg-cache" "$OUT/cache/$condition-$session"
  CURRENT_PHASE=$condition-$session
  wait_idle
  run_monitored "$dir" env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID WGPU_BACKEND=vulkan \
    XDG_CACHE_HOME="$dir/xdg-cache" "$OUT/build/bench_v4_residency" "${common_args[@]}" \
    --codec-execution "$condition" --requests 12 --warmups 2 --output-json "$dir/result.json" \
    --cubecl-cache-dir "$OUT/cache/$condition-$session" --cubecl-bundle-in "$OUT/prime/environment.sqlite" \
    || die "$condition session failed without retry: $session"
  jq -e --arg condition "$condition" \
    '.schema_version == 6 and .requests == 12 and .warmups == 2 and .measured == 10
     and .codec_execution == ($condition | gsub("-"; "_"))
     and (.resident_request_timings | length) == 12
     and ([.resident_request_timings[] | .codec_readback_complete_seconds >= .codec_device_complete_seconds] | all)
     and ([.resident_request_timings[].audio_f32_sha256] | unique | length) == 1
     and .work_report.num_steps == 4
     and (if $condition == "captured-graph" then .load_timing.codec_graph_capture_seconds > 0
          else .load_timing.codec_graph_capture_seconds == null end)' \
    "$dir/result.json" >/dev/null || die "$condition result gate failed: $session"
  [[ ! -s $dir/stderr.log ]] || die "$condition emitted stderr: $session"
  printf 'complete\n' >"$dir/COMPLETE"
  (cd "$dir" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
}

for session in 1 2 3 4 5; do
  if ((session % 2 == 1)); then
    order=(eager captured-graph)
  else
    order=(captured-graph eager)
  fi
  for condition in "${order[@]}"; do
    run_condition "$condition" "$session"
  done
  eager_hash=$(jq -r '[.resident_request_timings[].audio_f32_sha256] | unique | .[0]' "$OUT/sessions/eager-$session/result.json")
  captured_hash=$(jq -r '[.resident_request_timings[].audio_f32_sha256] | unique | .[0]' "$OUT/sessions/captured-graph-$session/result.json")
  [[ $eager_hash == "$captured_hash" ]] || die "paired audio hash mismatch in session $session"
done

CURRENT_PHASE=complete
wait_idle
COMPLETE=1
seal COMPLETE
printf 'captured_all_resident_complete=%s\n' "$OUT"
