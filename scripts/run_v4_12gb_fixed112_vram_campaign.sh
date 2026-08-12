#!/usr/bin/env bash
# Fresh 112-frame WGPU residency decomposition: 6 conditions x 5 sessions.

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
GPU_PCI=00000000:01:00.0
GPU_MEMORY_MIB=12227
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
  printf 'status=%s\nphase=%s\nautomatic_retries=0\noutput_reuse=false\nold_measurements_pooled=false\n' \
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

for path in "$FIXTURE" "$REF1" "$REF2" "$MODEL" "$CODEC" "$BIN"; do
  [[ -f $path && -s $path ]] || die "missing input: $path"
done
[[ $(sha "$MODEL") == "$MODEL_SHA" ]] || die 'model SHA mismatch'
[[ $(sha "$CODEC") == "$CODEC_SHA" ]] || die 'codec SHA mismatch'

gpu_row=$(nvidia-smi -i 0 --query-gpu=name,driver_version,pci.bus_id,memory.total,memory.free,index --format=csv,noheader,nounits)
IFS=',' read -r gpu_name gpu_driver gpu_pci gpu_memory gpu_free gpu_index <<<"$gpu_row"
gpu_name=${gpu_name# }; gpu_name=${gpu_name% }
gpu_driver=${gpu_driver# }; gpu_driver=${gpu_driver% }
gpu_pci=${gpu_pci# }; gpu_pci=${gpu_pci% }
gpu_memory=${gpu_memory# }; gpu_memory=${gpu_memory% }
gpu_free=${gpu_free# }; gpu_free=${gpu_free% }
gpu_index=${gpu_index# }; gpu_index=${gpu_index% }
[[ $gpu_pci == "$GPU_PCI" && $gpu_memory == "$GPU_MEMORY_MIB" && $gpu_index == 0 ]] \
  || die "GPU identity mismatch: $gpu_row"

mkdir -p "$OUT/build" "$OUT/sessions" "$OUT/cache-prime" "$OUT/cubecl-cache" "$OUT/driver-cache"
install -m 0555 "$BIN" "$OUT/build/bench_v4_residency"
for source in \
  src/bin/bench_v4_residency.rs src/inference.rs src/model/optimized.rs \
  src/model/attention.rs src/model/feed_forward.rs src/codec/decoder.rs \
  src/codec/model.rs src/backend_config.rs scripts/run_v4_12gb_fixed112_vram_campaign.sh \
  scripts/summarize_v4_12gb_fixed112_vram_campaign.py; do
  install -m 0444 "$ROOT/$source" "$OUT/build/$(basename "$source")"
done
sha256sum "$OUT/build"/* "$FIXTURE" "$REF1" "$REF2" "$MODEL" "$CODEC" >"$OUT/pins.sha256"
git -C "$ROOT" diff --binary >"$OUT/source.diff"
printf 'source_head=%s\nsource_diff_sha256=%s\ngpu_name=%s\ngpu_driver=%s\ngpu_pci=%s\nnvml_index=0\ncuda_index=0\nwgpu_adapter_index=0\nvram_total_mib=%s\nvram_free_preflight_mib=%s\nprecision=strict-fp32\ntf32=false\nautocast=false\nseconds=4.48\nframes=112\nvoice=unconditioned\nwarmups=2\nmeasured=10\nfresh_sessions=5\nshared_campaign_cache=true\n' \
  "$(git -C "$ROOT" rev-parse HEAD)" "$(sha "$OUT/source.diff")" "$gpu_name" "$gpu_driver" "$gpu_pci" \
  "$gpu_memory" "$gpu_free" >"$OUT/protocol.txt"

exec 9>>"$LOCK"
flock -n 9 || die 'GPU0 campaign lock is held'

wait_idle() {
  local count=0 telemetry processes
  for _ in $(seq 1 30); do
    telemetry=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ $telemetry =~ ^([0-9]+),[[:space:]]*([0-9]+)$ ]] \
      && ((BASH_REMATCH[1] <= 128 && BASH_REMATCH[2] <= 5)) && [[ ! $processes =~ [0-9] ]]; then
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
  --mode all-resident --checkpoint "$MODEL" --codec-weights "$CODEC"
  --fixture "$FIXTURE" --reference "$REF1" "$REF2" --unconditioned
  --speaker-mode same --length-mode same --adapter-index 0 --codec-residency decode-only
  --allocator exclusive-pages --cubecl-cache-dir "$OUT/cubecl-cache"
)

CURRENT_PHASE=cache-prime
wait_idle
run_monitored "$OUT/cache-prime" env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
  WGPU_BACKEND=vulkan XDG_CACHE_HOME="$OUT/driver-cache" \
  "$OUT/build/bench_v4_residency" "${common_args[@]}" --requests 1 --warmups 0 \
  --duration-residency predictive --rf-weight-residency portable-fallback \
  --codec-weight-residency portable-fallback --output-json "$OUT/cache-prime/result.json" \
  || die 'cache prime failed without retry'
CAMPAIGN_AUDIO_SHA=$(jq -er '.items[0].audio_f32_sha256' "$OUT/cache-prime/result.json")
printf '%s  campaign-control-audio-f32\n' "$CAMPAIGN_AUDIO_SHA" >"$OUT/campaign-audio.sha256"

run_condition() {
  local condition=$1 session=$2 duration rf_weights codec_weights dir
  dir=$OUT/sessions/$condition-$session
  duration=predictive
  rf_weights=portable-fallback
  codec_weights=portable-fallback
  case $condition in
    control) ;;
    exact-only) duration=exact-only ;;
    rf-one-layout) rf_weights=fixed112-one-layout ;;
    rf-packed-only) rf_weights=fixed112-packed-only ;;
    codec-packed-only) codec_weights=fixed112-packed-only ;;
    combined-packed-only)
      duration=exact-only
      rf_weights=fixed112-packed-only
      codec_weights=fixed112-packed-only
      ;;
    *) die "unknown condition: $condition" ;;
  esac
  mkdir "$dir"
  CURRENT_PHASE=$condition-$session
  wait_idle
  run_monitored "$dir" env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
    WGPU_BACKEND=vulkan XDG_CACHE_HOME="$OUT/driver-cache" \
    "$OUT/build/bench_v4_residency" "${common_args[@]}" --requests 12 --warmups 2 \
    --duration-residency "$duration" --rf-weight-residency "$rf_weights" \
    --codec-weight-residency "$codec_weights" --output-json "$dir/result.json" \
    || die "$condition session failed without retry: $session"
  jq -e --arg duration "$duration" --arg rf "$rf_weights" --arg codec "$codec_weights" \
    --arg hash "$CAMPAIGN_AUDIO_SHA" \
    '.requests == 12 and .warmups == 2 and .measured == 10 and .unconditioned
     and .duration_residency == ($duration | gsub("-"; "_"))
     and .rf_weight_residency == ($rf | gsub("-"; "_"))
     and .codec_weight_residency == ($codec | gsub("-"; "_"))
     and (.resident_request_timings | length) == 12
     and ([.resident_request_timings[].audio_f32_sha256] | unique) == [$hash]
     and .strict_fp32 and (.autocast | not) and (.tf32 | not)
     and .work_report.num_steps == 4
     and .work_report.model_layers == 12
     and .work_report.model_block_calls == 48' "$dir/result.json" >/dev/null \
    || die "$condition result gate failed: $session"
  printf 'complete\n' >"$dir/COMPLETE"
  (cd "$dir" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
}

for session in 1 2 3 4 5; do
  for condition in control exact-only rf-one-layout rf-packed-only codec-packed-only combined-packed-only; do
    run_condition "$condition" "$session"
  done
done

CURRENT_PHASE=summarize
wait_idle
uv run "$OUT/build/summarize_v4_12gb_fixed112_vram_campaign.py" \
  "$OUT" "$OUT/summary.json"
COMPLETE=1
CURRENT_PHASE=complete
seal COMPLETE
printf 'fixed112_vram_campaign_complete=%s\n' "$OUT"
