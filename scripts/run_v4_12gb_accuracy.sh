#!/usr/bin/env bash
# Fresh strict-FP32 RF/codec baseline on the pinned 12 GiB device.

set -Eeuo pipefail
IFS=$'\n\t'

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
OUT=${1:-}
[[ -n $OUT ]] || { printf 'usage: %s FRESH_OUTPUT_DIR\n' "$0" >&2; exit 2; }
[[ $OUT == /* ]] || OUT=$PWD/$OUT
OUT=$(realpath -m -- "$OUT")
[[ ! -e $OUT && ! -L $OUT ]] || { printf 'error: output exists: %s\n' "$OUT" >&2; exit 1; }

MODEL=$HOME/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4-Small/snapshots/e4aaac4df355ff560dcd35e0dae272c3a759317b/model.safetensors
PY_CODEC=$HOME/.cache/huggingface/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/47376ee24834d7a05a48ebabfe3cde29b3c5e214/weights.pth
RUST_CODEC=$ROOT/target/v4_dacvae_weights.safetensors
SOURCE=$HOME/benchmark-artifacts/irodori-v4-12gb-baseline-20260812-attempt1/accuracy/source-noise.safetensors
UPSTREAM=$ROOT/../Irodori-TTS
VALIDATOR=$ROOT/target/release/validate_v4_precision
MODEL_SHA=5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593
PY_CODEC_SHA=db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5
RUST_CODEC_SHA=4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1
GPU_PCI=00000000:01:00.0
GPU_NAME='NVIDIA GeForce RTX 5070 Ti Laptop GPU'
LENGTHS=(1.8 4.48 10.2 13.32 19.56 27.4)
SLUGS=(s1p8 s4p48 s10p2 s13p32 s19p56 s27p4)
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

for path in "$MODEL" "$PY_CODEC" "$RUST_CODEC" "$SOURCE" "$VALIDATOR"; do
  [[ -f $path && -s $path ]] || die "missing input: $path"
done
[[ $(sha "$MODEL") == "$MODEL_SHA" ]] || die 'model SHA mismatch'
[[ $(sha "$PY_CODEC") == "$PY_CODEC_SHA" ]] || die 'Python codec SHA mismatch'
[[ $(sha "$RUST_CODEC") == "$RUST_CODEC_SHA" ]] || die 'Rust codec SHA mismatch'
[[ $(git -C "$ROOT" rev-parse HEAD) == b275147b63542d37be20e28e89b39bf2ed9421d6 ]] || die 'source HEAD changed'
[[ $(git -C "$UPSTREAM" rev-parse HEAD) == 9f19d9a9048099a4b978a762d0509228fe624e3f ]] || die 'upstream HEAD mismatch'
row=$(nvidia-smi -i 0 --query-gpu=name,pci.bus_id,memory.total --format=csv,noheader,nounits)
[[ $row == "$GPU_NAME, $GPU_PCI, 12227" ]] || die "GPU identity mismatch: $row"

mkdir -p "$OUT/build" "$OUT/lengths"
install -m 0555 "$VALIDATOR" "$OUT/build/validate_v4_precision"
install -m 0444 "$ROOT/scripts/export_v4_precision_oracle.py" "$OUT/build/export_v4_precision_oracle.py"
install -m 0444 "$ROOT/scripts/bench_python_e2e_precision.py" "$OUT/build/bench_python_e2e_precision.py"
sha256sum "$OUT/build"/* "$MODEL" "$PY_CODEC" "$RUST_CODEC" "$SOURCE" >"$OUT/pins.sha256"
printf 'source_commit=%s\nupstream_commit=%s\ngpu_name=%s\ngpu_pci=%s\nnvml_index=0\nwgpu_adapter_index=0\nprecision=strict-fp32\ntf32=false\nautocast=false\nwarmups=2\nmeasured=10\n' \
  "$(git -C "$ROOT" rev-parse HEAD)" "$(git -C "$UPSTREAM" rev-parse HEAD)" "$GPU_NAME" "$GPU_PCI" >"$OUT/protocol.txt"

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
  local telemetry=$1 wall=$2 stdout=$3 stderr=$4
  shift 4
  nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits -lms 100 -f "$telemetry" &
  ACTIVE_MONITOR=$!
  set +e
  /usr/bin/time -o "$wall" -f 'exit_status=%x\nelapsed_seconds=%e\nmax_rss_kib=%M' "$@" >"$stdout" 2>"$stderr"
  local status=$?
  set -e
  stop_monitor
  return "$status"
}

for index in "${!LENGTHS[@]}"; do
  seconds=${LENGTHS[$index]}
  slug=${SLUGS[$index]}
  dir=$OUT/lengths/$slug
  mkdir "$dir"

  CURRENT_PHASE=oracle-$slug
  wait_idle
  run_monitored "$dir/oracle-nvml.csv" "$dir/oracle-wall.txt" "$dir/oracle.stdout.log" "$dir/oracle.stderr.log" \
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 \
    uv run --python 3.10 "$OUT/build/export_v4_precision_oracle.py" \
      --precision fp32 --upstream "$UPSTREAM" --checkpoint "$MODEL" --codec "$PY_CODEC" \
      --source-fixture "$SOURCE" --seconds "$seconds" --model-device cuda:0 --codec-device cuda:0 \
      --output "$dir/oracle.safetensors" --manifest-out "$dir/oracle.json" --verification-wav "$dir/oracle.wav" \
    || die "oracle failed without retry: $slug"
  oracle_sha=$(sha "$dir/oracle.safetensors")

  CURRENT_PHASE=python-$slug
  wait_idle
  run_monitored "$dir/python-nvml.csv" "$dir/python-wall.txt" "$dir/python.stdout.log" "$dir/python.stderr.log" \
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 \
    uv run --python 3.10 "$OUT/build/bench_python_e2e_precision.py" \
      --precision fp32 --upstream "$UPSTREAM" --checkpoint "$MODEL" --codec "$PY_CODEC" \
      --source-fixture "$dir/oracle.safetensors" --source-fixture-sha256 "$oracle_sha" \
      --seconds "$seconds" --repeats 12 --model-device cuda:0 --codec-device cuda:0 \
      --json-out "$dir/python.json" \
    || die "Python benchmark failed without retry: $slug"

  CURRENT_PHASE=wgpu-$slug
  wait_idle
  run_monitored "$dir/wgpu-nvml.csv" "$dir/wgpu-wall.txt" "$dir/wgpu.stdout.log" "$dir/wgpu.stderr.log" \
    env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID WGPU_BACKEND=vulkan RUST_LOG=warn \
    "$OUT/build/validate_v4_precision" --execution wgsl --precision fp32 \
      --fixture "$dir/oracle.safetensors" --fixture-sha256 "$oracle_sha" \
      --checkpoint "$MODEL" --codec-weights "$RUST_CODEC" --adapter-index 0 \
      --tasks-max 32 --memory-config sub-slices --repeats 12 \
    || die "WGPU benchmark failed without retry: $slug"
  grep -F 'backend=Vulkan device_type=DiscreteGpu' "$dir/wgpu.stdout.log" >/dev/null || die "WGPU adapter gate failed: $slug"
  [[ $(grep -c '^rf_repeat=' "$dir/wgpu.stdout.log") == 12 ]] || die "WGPU RF repeat count failed: $slug"
  [[ $(grep -c '^codec_repeat=' "$dir/wgpu.stdout.log") == 12 ]] || die "WGPU codec repeat count failed: $slug"
  printf 'complete\n' >"$dir/COMPLETE"
  (cd "$dir" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
done

CURRENT_PHASE=complete
wait_idle
COMPLETE=1
seal COMPLETE
printf 'accuracy_campaign_complete=%s\n' "$OUT"
