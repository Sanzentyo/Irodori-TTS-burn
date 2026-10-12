#!/usr/bin/env bash
# Fresh WGPU all-resident VRAM optimization A/B: 5 sessions per condition.

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
CODEC=$ROOT/target/v4_dacvae_weights.safetensors
BIN=$ROOT/target/release/bench_v4_residency
MODEL_SHA=5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593
CODEC_SHA=4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1
AUDIO_SHA=faf8ea4008458d8e1ff3c2c70254c48fa02b181d873d41f99745ed84ce479e3d
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

for path in "$FIXTURE" "$REF1" "$REF2" "$MODEL" "$CODEC" "$BIN"; do
  [[ -f $path && -s $path ]] || die "missing input: $path"
done
[[ $(sha "$MODEL") == "$MODEL_SHA" ]] || die 'model SHA mismatch'
[[ $(sha "$CODEC") == "$CODEC_SHA" ]] || die 'codec SHA mismatch'
row=$(nvidia-smi -i 0 --query-gpu=name,pci.bus_id,memory.total --format=csv,noheader,nounits)
[[ $row == "$GPU_NAME, $GPU_PCI, 12227" ]] || die "GPU identity mismatch: $row"

mkdir -p "$OUT/build" "$OUT/sessions"
install -m 0555 "$BIN" "$OUT/build/bench_v4_residency"
install -m 0444 "$ROOT/src/bin/bench_v4_residency.rs" "$OUT/build/bench_v4_residency.rs"
install -m 0444 "$ROOT/src/codec/model.rs" "$OUT/build/codec-model.rs"
install -m 0444 "$ROOT/src/codec/weights.rs" "$OUT/build/codec-weights.rs"
install -m 0444 "$ROOT/scripts/run_v4_12gb_vram_optimization.sh" "$OUT/build/runner.sh"
install -m 0444 "$ROOT/scripts/summarize_v4_12gb_vram_optimization.py" "$OUT/build/summarizer.py"
sha256sum "$OUT/build"/* "$FIXTURE" "$REF1" "$REF2" "$MODEL" "$CODEC" >"$OUT/pins.sha256"
git -C "$ROOT" diff --binary >"$OUT/source.diff"
printf 'source_head=%s\nsource_diff_sha256=%s\ngpu_name=%s\ngpu_pci=%s\nnvml_index=0\nwgpu_adapter_index=0\nprecision=strict-fp32\ntf32=false\nautocast=false\nseconds=4.48\nframes=112\nvoice=unconditioned\nwarmups=2\nmeasured=10\nfresh_sessions=5\n' \
  "$(git -C "$ROOT" rev-parse HEAD)" "$(sha "$OUT/source.diff")" "$GPU_NAME" "$GPU_PCI" >"$OUT/protocol.txt"

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

run_condition() {
  local condition=$1 session=$2 codec_residency allocator cleanup dir
  dir=$OUT/sessions/$condition-$session
  codec_residency=decode-only
  allocator=sub-slices
  cleanup=()
  case $condition in
    control-full) codec_residency=full ;;
    decode-only) ;;
    decode-cleanup) cleanup=(--cleanup-after-warmup) ;;
    decode-exclusive) allocator=exclusive-pages ;;
    *) die "unknown condition: $condition" ;;
  esac
  mkdir "$dir" "$dir/xdg-cache"
  CURRENT_PHASE=$condition-$session
  wait_idle
  run_monitored "$dir" env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID WGPU_BACKEND=vulkan XDG_CACHE_HOME="$dir/xdg-cache" \
    "$OUT/build/bench_v4_residency" --mode all-resident --checkpoint "$MODEL" --codec-weights "$CODEC" \
      --fixture "$FIXTURE" --reference "$REF1" "$REF2" --requests 12 --warmups 2 --unconditioned \
      --speaker-mode same --length-mode same --adapter-index 0 --codec-residency "$codec_residency" \
      --allocator "$allocator" "${cleanup[@]}" --output-json "$dir/result.json" \
    || die "$condition session failed without retry: $session"
  jq -e --arg codec "$codec_residency" --arg allocator "$allocator" --arg hash "$AUDIO_SHA" \
    '.requests == 12 and .warmups == 2 and .measured == 10 and .unconditioned
     and .codec_residency == ($codec | gsub("-"; "_"))
     and .allocator == ($allocator | gsub("-"; "_"))
     and (.resident_request_timings | length) == 12
     and ([.resident_request_timings[].audio_f32_sha256] | unique) == [$hash]
     and .work_report.num_steps == 4' "$dir/result.json" >/dev/null \
    || die "$condition result gate failed: $session"
  printf 'complete\n' >"$dir/COMPLETE"
  (cd "$dir" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
}

for session in 1 2 3 4 5; do
  for condition in control-full decode-only decode-cleanup decode-exclusive; do
    run_condition "$condition" "$session"
  done
done

CURRENT_PHASE=complete
wait_idle
COMPLETE=1
seal COMPLETE
printf 'vram_optimization_complete=%s\n' "$OUT"
