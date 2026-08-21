#!/usr/bin/env bash
# External process-launch through final WAV close, separated from steady runs.

set -Eeuo pipefail
IFS=$'\n\t'

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
USER_ROOT=$(realpath -- "$ROOT/../..")
UPSTREAM=$(realpath -- "$ROOT/../Irodori-TTS")
OUT=
while (($#)); do
  case "$1" in
    --output-dir) OUT=${2:?--output-dir requires a path}; shift 2 ;;
    -h|--help) printf 'usage: %s --output-dir FRESH_PATH\n' "$0"; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n $OUT ]] || { printf 'error: --output-dir is required\n' >&2; exit 2; }
OUT=$(realpath -m -- "$OUT")
[[ ! -e $OUT && ! -L $OUT ]] || { printf 'error: output exists: %s\n' "$OUT" >&2; exit 1; }

MODEL_REV=e4aaac4df355ff560dcd35e0dae272c3a759317b
CODEC_REV=47376ee24834d7a05a48ebabfe3cde29b3c5e214
MODEL="$USER_ROOT/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4-Small/snapshots/$MODEL_REV/model.safetensors"
SAMPLES="$USER_ROOT/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4-Small/snapshots/$MODEL_REV/samples"
PY_CODEC="$USER_ROOT/.cache/huggingface/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/$CODEC_REV/weights.pth"
WG_CODEC="$ROOT/target/v4_dacvae_weights.safetensors"
REF="$SAMPLES/clone_ref1.wav"
WG_BIN="$ROOT/target/release/pipeline"
PY_RUNNER="$ROOT/scripts/run_python_v4_cold_e2e.py"
MODEL_SHA=5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593
PY_CODEC_SHA=db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5
WG_CODEC_SHA=4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1
GPU_NAME='NVIDIA GeForce RTX 5070 Ti Laptop GPU'
GPU_PCI=00000000:01:00.0
LOCK=/tmp/irodori-v4-cold-e2e-gpu0.lock
CPU_SET=0-11
ACTIVE_MONITOR=
CURRENT_PHASE=preflight
COMPLETE=0
TEXT='これは音声合成の実運用に近い条件を確認するためのサンプルです。'
DESIGN='落ち着いた自然な日本語の声で、明瞭かつ穏やかに話す。'

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
  [[ -d $OUT ]] || return 0
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

for command in ffprobe flock git jq nvidia-smi taskset uv; do command -v "$command" >/dev/null || die "missing $command"; done
for path in "$MODEL" "$PY_CODEC" "$WG_CODEC" "$REF" "$WG_BIN" "$PY_RUNNER"; do
  [[ -f $path && -s $path ]] || die "missing input: $path"
done
[[ $(sha "$MODEL") == "$MODEL_SHA" && $(sha "$PY_CODEC") == "$PY_CODEC_SHA" \
  && $(sha "$WG_CODEC") == "$WG_CODEC_SHA" ]] || die 'model/codec SHA mismatch'
[[ -z $(git -C "$ROOT" status --short) ]] || die 'Rust source tree must be clean'
gpu_row=$(nvidia-smi -i 0 --query-gpu=name,pci.bus_id,memory.total,driver_version --format=csv,noheader,nounits)
[[ $gpu_row == "$GPU_NAME, $GPU_PCI, 12227,"* ]] || die "GPU identity mismatch: $gpu_row"

mkdir -p "$OUT/build" "$OUT/voices"
install -m 0555 "$WG_BIN" "$OUT/build/pipeline"
install -m 0444 "$PY_RUNNER" "$OUT/build/run_python_v4_cold_e2e.py"
install -m 0444 "$0" "$OUT/build/runner.sh"
git -C "$ROOT" rev-parse HEAD >"$OUT/source-head.txt"
git -C "$UPSTREAM" rev-parse HEAD >"$OUT/upstream-head.txt"
nvidia-smi -q >"$OUT/nvidia-smi-q.txt"
sha256sum "$OUT/build"/* "$MODEL" "$PY_CODEC" "$WG_CODEC" "$REF" >"$OUT/pins.sha256"

exec 9>>"$LOCK"
flock -n 9 || die 'cold E2E GPU lock is held'
wait_idle() {
  local quiet=0 telemetry processes
  for _ in $(seq 1 60); do
    telemetry=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ $telemetry =~ ^([0-9]+),[[:space:]]*([0-9]+)$ ]] \
      && ((BASH_REMATCH[1] <= 512 && BASH_REMATCH[2] <= 5)) && [[ ! $processes =~ [0-9] ]]; then
      ((quiet += 1)); ((quiet >= 2)) && return 0
    else quiet=0; fi
    sleep 1
  done
  die 'GPU did not become idle'
}
run_monitored() {
  local dir=$1
  shift
  nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.free,utilization.gpu \
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
write_receipt() {
  local dir=$1 runtime=$2 voice=$3 session=$4 cache_state=$5
  local wall samples rate seconds peak
  wall=$(sed -n 's/^elapsed_seconds=//p' "$dir/wall.txt")
  samples=$(ffprobe -v error -select_streams a:0 -show_entries stream=duration_ts -of csv=p=0 "$dir/output.wav")
  rate=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of csv=p=0 "$dir/output.wav")
  seconds=$(awk -v n="$samples" -v r="$rate" 'BEGIN {printf "%.9f", n/r}')
  peak=$(awk -F',' 'BEGIN{m=0}{gsub(/ /,"",$4);if($4+0>m)m=$4+0}END{print m}' "$dir/nvml.csv")
  jq -n --arg runtime "$runtime" --arg voice "$voice" --arg cache "$cache_state" \
    --arg wav "$(realpath -- "$dir/output.wav")" --arg sha "$(sha "$dir/output.wav")" \
    --argjson session "$session" --argjson wall "$wall" --argjson samples "$samples" \
    --argjson rate "$rate" --argjson seconds "$seconds" --argjson peak "$peak" \
    '{runtime:$runtime,voice:$voice,session:$session,cache_state:$cache,
      boundary:"external process launch through exit immediately after final WAV close",
      cold_e2e_seconds:$wall,output:{wav:$wav,sha256:$sha,samples:$samples,sample_rate:$rate,seconds:$seconds},
      nvml_peak_mib:$peak}' >"$dir/receipt.json"
}

for voice in text design clone; do
  voice_root="$OUT/voices/$voice"
  mkdir -p "$voice_root/python" "$voice_root/wgpu" "$voice_root/wgpu-cache" "$voice_root/xdg"
  # Cold E2E is a one-shot distribution: keep one fresh-cache launch and one
  # restored-cache launch per runtime/voice. It is never pooled with steady data.
  for session in 1 2; do
    cache_state=restored_campaign_cache
    ((session == 1)) && cache_state=fresh_campaign_cache
    for runtime in python wgpu; do
      dir="$voice_root/$runtime/s$session"
      mkdir -p "$dir"
      CURRENT_PHASE="$voice-$runtime-s$session"
      wait_idle
      if [[ $runtime == python ]]; then
        ref_args=()
        [[ $voice == clone ]] && ref_args=(--ref-wav "$REF")
        run_monitored "$dir" env -u LD_LIBRARY_PATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
          PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
          taskset -c "$CPU_SET" uv run --python 3.10 "$OUT/build/run_python_v4_cold_e2e.py" \
            --upstream "$UPSTREAM" --checkpoint "$MODEL" --codec "$PY_CODEC" --voice "$voice" \
            "${ref_args[@]}" --output-wav "$dir/output.wav" --output-json "$dir/internal.json" \
            --expected-pci "$GPU_PCI" --expected-gpu-name "$GPU_NAME" --num-steps 40 \
          || die "Python cold E2E failed: $voice/s$session"
      else
        voice_args=()
        [[ $voice == design ]] && voice_args=(--caption "$DESIGN" --cfg-caption 4)
        [[ $voice == clone ]] && voice_args=(--ref-audio "$REF")
        run_monitored "$dir" env -u CUDA_VISIBLE_DEVICES WGPU_BACKEND=vulkan \
          XDG_CACHE_HOME="$voice_root/xdg" RUST_LOG=info HF_HUB_OFFLINE=1 \
          taskset -c "$CPU_SET" "$OUT/build/pipeline" --backend wgpu-wgsl \
            --checkpoint "$MODEL" --codec-weights "$WG_CODEC" --text "$TEXT" \
            "${voice_args[@]}" --output "$dir/output.wav" --num-steps 40 --cfg-text 3 \
            --cfg-speaker 5 --cfg-mode independent --cfg-min-t 0.5 --cfg-max-t 1 \
            --trim-tail false --seed 42 --wgpu-adapter-index 0 \
            --cubecl-cache-dir "$voice_root/wgpu-cache" --rf-work-manifest-out "$dir/rf-work.json" \
          || die "WGPU cold E2E failed: $voice/s$session"
        jq -e '.num_steps == 40 and .whole_model_forwards == 40 and .model_block_calls == 480' \
          "$dir/rf-work.json" >/dev/null || die "WGPU RF work gate failed: $voice/s$session"
      fi
      [[ -s $dir/output.wav ]] || die "missing WAV: $voice/$runtime/s$session"
      write_receipt "$dir" "$runtime" "$voice" "$session" "$cache_state"
    done
  done
done

jq -s '{format:"irodori-v4-cold-e2e-campaign-v1",steps:40,duration:"predict",tail_trim:false,
  cache_note:"session 1 starts with a fresh per-voice CubeCL/vendor cache; session 2 restores it; process-local pipelines are always rebuilt",
  results:.,summaries:(group_by([.voice,.runtime])|map({voice:.[0].voice,runtime:.[0].runtime,
    restored_cold_e2e_seconds:([.[]|select(.cache_state=="restored_campaign_cache")|.cold_e2e_seconds]|sort),
    fresh_cold_e2e_seconds:[.[]|select(.cache_state=="fresh_campaign_cache")|.cold_e2e_seconds],
    nvml_peak_mib:([.[].nvml_peak_mib]|max)}))}' \
  "$OUT"/voices/*/{python,wgpu}/s*/receipt.json >"$OUT/summary.json"
CURRENT_PHASE=complete
COMPLETE=1
seal COMPLETE
printf 'complete: %s\n' "$OUT"
