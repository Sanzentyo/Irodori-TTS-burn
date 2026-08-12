#!/usr/bin/env bash
# Fresh WGPU all-resident and type-state phase-batch campaign for the 12 GiB GPU.

set -Eeuo pipefail
IFS=$'\n\t'

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
OUT=${1:-}
REF_DIR=${2:-}
[[ -n $OUT && -n $REF_DIR ]] || { printf 'usage: %s FRESH_OUTPUT_DIR PREPARED_REFERENCE_DIR\n' "$0" >&2; exit 2; }
OUT=$(realpath -m -- "$OUT")
REF_DIR=$(realpath -- "$REF_DIR")
[[ ! -e $OUT && ! -L $OUT ]] || { printf 'error: output exists: %s\n' "$OUT" >&2; exit 1; }

BASE=$HOME/benchmark-artifacts/irodori-v4-12gb-baseline-20260812-attempt1
ACCURACY=$BASE/accuracy-campaign/lengths
MODEL=$HOME/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4-Small/snapshots/e4aaac4df355ff560dcd35e0dae272c3a759317b/model.safetensors
CODEC=$ROOT/target/v4_dacvae_weights.safetensors
BIN=$ROOT/target/release/bench_v4_residency
REF1=$REF_DIR/ref1.safetensors
REF2=$REF_DIR/ref2.safetensors
FIXTURES=(
  "$ACCURACY/s4p48/oracle.safetensors"
  "$ACCURACY/s1p8/oracle.safetensors"
  "$ACCURACY/s10p2/oracle.safetensors"
  "$ACCURACY/s13p32/oracle.safetensors"
  "$ACCURACY/s19p56/oracle.safetensors"
  "$ACCURACY/s27p4/oracle.safetensors"
)
GPU_NAME='NVIDIA GeForce RTX 5070 Ti Laptop GPU'
GPU_PCI=00000000:01:00.0
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

for path in "$MODEL" "$CODEC" "$BIN" "$REF1" "$REF2" "${FIXTURES[@]}"; do
  [[ -f $path && -s $path ]] || die "missing input: $path"
done
[[ $(sha256sum -- "$MODEL" | awk '{print $1}') == 5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593 ]] || die 'model SHA mismatch'
[[ $(sha256sum -- "$CODEC" | awk '{print $1}') == 4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1 ]] || die 'codec SHA mismatch'
[[ $(git -C "$ROOT" rev-parse HEAD) == b275147b63542d37be20e28e89b39bf2ed9421d6 ]] || die 'source HEAD changed'
row=$(nvidia-smi -i 0 --query-gpu=name,pci.bus_id,memory.total --format=csv,noheader,nounits)
[[ $row == "$GPU_NAME, $GPU_PCI, 12227" ]] || die "GPU identity mismatch: $row"

mkdir -p "$OUT/build" "$OUT/conditions"
install -m 0555 "$BIN" "$OUT/build/bench_v4_residency"
install -m 0444 "$ROOT/src/phase_batch.rs" "$OUT/build/phase_batch.rs"
install -m 0444 "$ROOT/scripts/run_v4_12gb_residency.sh" "$OUT/build/run_v4_12gb_residency.sh"
sha256sum "$OUT/build"/* "$MODEL" "$CODEC" "$REF1" "$REF2" "${FIXTURES[@]}" >"$OUT/pins.sha256"
printf 'source_commit=%s\ngpu_name=%s\ngpu_pci=%s\nnvml_index=0\nwgpu_adapter_index=0\nprecision=strict-fp32\ntf32=false\nautocast=false\nphase_type_states=RfResident,LatentsResident,CodecResident,Complete\n' \
  "$(git -C "$ROOT" rev-parse HEAD)" "$GPU_NAME" "$GPU_PCI" >"$OUT/protocol.txt"

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

run_condition() {
  local name=$1 mode=$2 requests=$3 speaker=$4 length=$5
  local dir=$OUT/conditions/$name
  mkdir "$dir" "$dir/xdg-cache"
  CURRENT_PHASE=$name
  wait_idle
  local fixture_args=(--fixture "${FIXTURES[0]}")
  if [[ $length == mixed ]]; then
    fixture_args=()
    for fixture in "${FIXTURES[@]}"; do fixture_args+=(--fixture "$fixture"); done
  fi
  nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits -lms 100 -f "$dir/nvml.csv" &
  ACTIVE_MONITOR=$!
  set +e
  /usr/bin/time -o "$dir/wall.txt" -f 'exit_status=%x\nelapsed_seconds=%e\nmax_rss_kib=%M' \
    env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID WGPU_BACKEND=vulkan XDG_CACHE_HOME="$dir/xdg-cache" \
    "$OUT/build/bench_v4_residency" --mode "$mode" --checkpoint "$MODEL" --codec-weights "$CODEC" \
      "${fixture_args[@]}" --reference "$REF1" "$REF2" --requests "$requests" \
      --speaker-mode "$speaker" --length-mode "$length" --adapter-index 0 --output-json "$dir/result.json" \
      >"$dir/stdout.log" 2>"$dir/stderr.log"
  local status=$?
  set -e
  stop_monitor
  ((status == 0)) || die "condition failed without retry: $name"
  jq -e --argjson n "$requests" '.requests == $n and (.items | length) == $n' "$dir/result.json" >/dev/null || die "result gate failed: $name"
  printf 'complete\n' >"$dir/COMPLETE"
  (cd "$dir" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
}

run_condition all-resident all-resident 1 same same
for requests in 1 2 4 8 12; do
  run_condition "phase-same-speaker-same-length-n$requests" phase-batch "$requests" same same
  run_condition "phase-same-speaker-mixed-length-n$requests" phase-batch "$requests" same mixed
  if ((requests > 1)); then
    run_condition "phase-multi-speaker-same-length-n$requests" phase-batch "$requests" alternating same
    run_condition "phase-multi-speaker-mixed-length-n$requests" phase-batch "$requests" alternating mixed
  fi
done

CURRENT_PHASE=complete
wait_idle
COMPLETE=1
seal COMPLETE
printf 'residency_campaign_complete=%s\n' "$OUT"
