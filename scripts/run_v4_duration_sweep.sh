#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
DEFAULT_OUT=/tmp/irodori-v4-duration-sweep-20260811
OUT=$DEFAULT_OUT
DRY_RUN=0
SELF_TEST=0
LOCK=/tmp/irodori-v4-post18-gpu1.lock
MODEL=$HOME/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4-Small/snapshots/e4aaac4df355ff560dcd35e0dae272c3a759317b/model.safetensors
CODEC=$HOME/.cache/huggingface/hub/models--Aratako--Semantic-DACVAE-Japanese-32dim/snapshots/47376ee24834d7a05a48ebabfe3cde29b3c5e214/weights.pth
MODEL_SHA=5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593
CODEC_SHA=db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5
GPU_INDEX=1
GPU_PCI=00000000:07:00.0
PYTHON_SCRIPT=$ROOT/scripts/bench_python_duration.py
BINARY=$ROOT/target/release/examples/bench_v4_duration

usage() {
  cat <<'EOF'
Usage: scripts/run_v4_duration_sweep.sh [OPTIONS]
  --output-dir PATH  Fresh output root
  --dry-run          Print the protocol without build/model/GPU work
  --self-test        Run CPU-only CLI and case-table checks
  -h, --help         Show this help

There is no retry, resume, overwrite, or performance-filtering mode.
EOF
}

while (($#)); do
  case "$1" in
    --output-dir)
      (($# >= 2)) || { printf 'error: --output-dir requires a value\n' >&2; exit 1; }
      OUT=$2
      shift 2
      ;;
    --output-dir=*) OUT=${1#*=}; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --self-test) SELF_TEST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; usage >&2; exit 1 ;;
  esac
done
((DRY_RUN + SELF_TEST <= 1)) || { printf 'error: --dry-run and --self-test are mutually exclusive\n' >&2; exit 1; }
[[ -n $OUT ]] || { printf 'error: --output-dir must not be empty\n' >&2; exit 1; }
[[ $OUT == /* ]] || OUT=$PWD/$OUT
OUT=$(realpath -m -- "$OUT")

case_names=(short medium long very_long)
case_texts=(
  'こんにちは。'
  '今日は晴れているので、近所の公園までゆっくり散歩に行きます。'
  '音声合成の性能を正しく評価するため、短い文だけでなく、句読点を含む少し長い文章でも、推定時間と生成音声の長さを確認します。'
  'この測定では、実際の利用場面に近い長めの文章も対象にします。文章が長くなると、長さ推定モデルが処理する有効トークン数と予測フレーム数の両方が増えます。その変化に対して、GPU上の計算時間、CPUへの読み戻し時間、出力の再現性がどのように変わるかを、同じ条件で丁寧に比較します。'
)

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
sha() { sha256sum -- "$1" | awk '{print $1}'; }

if ((SELF_TEST)); then
  ((${#case_names[@]} == 4 && ${#case_texts[@]} == 4)) || die "duration case table is inconsistent"
  [[ ${case_names[*]} == 'short medium long very_long' ]] || die "duration case ordering changed"
  python3 - <<'PY'
import math

sample_rate = 48_000
hop_length = 1_920
for predicted, frames, samples, seconds in (
    (45.38101521433686, 45, 86_400, 1.8),
    (111.60224961624918, 112, 215_040, 4.48),
    (333.4430534902918, 333, 639_360, 13.32),
    (685.1357384411837, 685, 1_315_200, 27.4),
):
    resolved = min(max(round(predicted), math.ceil(0.5 * sample_rate / hop_length)), math.floor(30.0 * sample_rate / hop_length))
    assert (resolved, resolved * hop_length, resolved * hop_length / sample_rate) == (frames, samples, seconds)
PY
  printf 'duration_sweep_self_test=passed cases=%s gpu_workload=false output_created=false\n' "${#case_names[@]}"
  exit 0
fi

if ((DRY_RUN)); then
  printf 'duration_sweep_dry_run=ready output=%s cases=%s fresh_processes_per_runtime=3 warmups=5 measured=10 gpu_workload=false output_created=false\n' \
    "$OUT" "${case_names[*]}"
  printf 'timer_primary=pre-sync_to_device-complete timer_secondary=owned-contiguous-f32_CPU-readback-complete\n'
  exit 0
fi

[[ ! -e $OUT ]] || die "output already exists: $OUT"
[[ -f $MODEL && ! -L $MODEL ]] || [[ -L $MODEL ]] || die "model is missing"
[[ -f $CODEC && ! -L $CODEC ]] || [[ -L $CODEC ]] || die "codec is missing"
[[ $(sha "$MODEL") == "$MODEL_SHA" ]] || die "model SHA mismatch"
[[ $(sha "$CODEC") == "$CODEC_SHA" ]] || die "codec SHA mismatch"
[[ -f $PYTHON_SCRIPT && ! -L $PYTHON_SCRIPT ]] || die "Python benchmark is not regular"
[[ -e $LOCK && ( ! -f $LOCK || -L $LOCK ) ]] && die "unsafe lock path"

cd "$ROOT"
cargo build --release --locked --features cli --example bench_v4_duration
[[ -x $BINARY && ! -L $BINARY ]] || die "duration benchmark binary is missing"

mkdir -- "$OUT"
mkdir -- "$OUT/build" "$OUT/sessions"
install -m 0555 -- "$BINARY" "$OUT/build/bench_v4_duration"
install -m 0444 -- "$PYTHON_SCRIPT" "$OUT/build/bench_python_duration.py"
printf 'runner_sha256=%s\npython_sha256=%s\nbinary_sha256=%s\nmodel_sha256=%s\ncodec_sha256=%s\n' \
  "$(sha "${BASH_SOURCE[0]}")" "$(sha "$PYTHON_SCRIPT")" "$(sha "$BINARY")" "$MODEL_SHA" "$CODEC_SHA" >"$OUT/pins.txt"

exec 9>>"$LOCK"
flock -n 9 || die "GPU1 campaign lock is held"

gpu_row() {
  nvidia-smi -i "$GPU_INDEX" --query-gpu=index,pci.bus_id,memory.used,utilization.gpu --format=csv,noheader,nounits
}

settle_gpu() {
  local label=$1 row index pci memory util compute_pids
  for _ in $(seq 1 60); do
    row=$(gpu_row) || die "NVML failed during $label"
    IFS=',' read -r index pci memory util <<<"$row"
    index=${index//[[:space:]]/}; pci=${pci//[[:space:]]/}
    memory=${memory//[[:space:]]/}; util=${util//[[:space:]]/}
    [[ $index == "$GPU_INDEX" && $pci == "$GPU_PCI" ]] || die "GPU identity mismatch during $label"
    if (( memory <= 128 && util <= 5 )); then
      compute_pids=$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits) \
        || die "NVML process query failed during $label"
      if ! grep -Eq '[0-9]' <<<"$compute_pids"; then
        return 0
      fi
    fi
    sleep 1
  done
  die "GPU did not settle during $label"
}

for case_index in "${!case_names[@]}"; do
  name=${case_names[$case_index]}
  text=${case_texts[$case_index]}
  mkdir -- "$OUT/sessions/$name"
  for session in 1 2 3; do
    session_dir=$OUT/sessions/$name/s$session
    mkdir -- "$session_dir"
    fixture=$session_dir/fixture.safetensors
    python_json=$session_dir/python.json
    wgpu_json=$session_dir/wgpu.json

    settle_gpu "python-$name-s$session"
    set +e
    env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      taskset -c 6-11,18-23 \
      uv run --offline --script "$OUT/build/bench_python_duration.py" \
        --upstream "$ROOT/../Irodori-TTS" \
        --checkpoint "$MODEL" \
        --codec "$CODEC" \
        --text "$text" \
        --fixture-out "$fixture" \
        --json-out "$python_json" \
        >"$session_dir/python.log" 2>&1
    python_status=$?
    set -e
    printf '%s\n' "$python_status" >"$session_dir/python.exit"
    (( python_status == 0 )) || die "Python failed for $name session $session"
    fixture_sha=$(sha "$fixture")
    python_sha=$(sha "$python_json")

    settle_gpu "wgpu-$name-s$session"
    set +e
    env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
      CUBECL_WGPU_MAX_TASKS=32 \
      taskset -c 6-11,18-23 \
      "$OUT/build/bench_v4_duration" \
        --checkpoint "$MODEL" \
        --checkpoint-sha256 "$MODEL_SHA" \
        --fixture "$fixture" \
        --fixture-sha256 "$fixture_sha" \
        --python-json "$python_json" \
        --python-json-sha256 "$python_sha" \
        --adapter-index 0 \
        --json-out "$wgpu_json" \
        >"$session_dir/wgpu.log" 2>&1
    wgpu_status=$?
    set -e
    printf '%s\n' "$wgpu_status" >"$session_dir/wgpu.exit"
    (( wgpu_status == 0 )) || die "WGPU failed for $name session $session"
  done
done

settle_gpu final

jq -n --arg format irodori-v4-duration-sweep-v1 \
  --arg timer_primary 'pre-sync to device complete; scalar CPU readback excluded' \
  --arg timer_secondary 'owned contiguous float32 one-element CPU readback complete' \
  --argjson cases "$(
    for name in "${case_names[@]}"; do
      jq -s --arg name "$name" '
        def stats: sort as $v | {min:$v[0], median:(if ($v|length)%2==1 then $v[($v|length)/2|floor] else (($v[($v|length)/2-1]+$v[($v|length)/2])/2) end), max:$v[-1]};
        {name:$name,
         text:.[0].input.text,
         text_valid_tokens:.[0].input.text_valid_tokens,
         predicted_frames:.[0].scopes.full.predicted_frames,
         resolved_length:.[0].resolved_length,
         resolved_length_equal_across_runtimes:
           (([.[]|.resolved_length]|unique|length) == 1),
         python:{
           head_device:([.[]|select(.format=="irodori-v4-python-duration-benchmark-v1")|.repeats[]|select(.scope=="head" and (.cold|not))|.timing.device_complete_seconds]|stats),
           head_readback:([.[]|select(.format=="irodori-v4-python-duration-benchmark-v1")|.repeats[]|select(.scope=="head" and (.cold|not))|.timing.readback_complete_seconds]|stats),
           full_device:([.[]|select(.format=="irodori-v4-python-duration-benchmark-v1")|.repeats[]|select(.scope=="full" and (.cold|not))|.timing.device_complete_seconds]|stats),
           full_readback:([.[]|select(.format=="irodori-v4-python-duration-benchmark-v1")|.repeats[]|select(.scope=="full" and (.cold|not))|.timing.readback_complete_seconds]|stats)},
         wgpu:{
           head_device:([.[]|select(.format=="irodori-v4-wgpu-duration-benchmark-v1")|.repeats[]|select(.scope=="head" and (.cold|not))|.timing.device_complete_seconds]|stats),
           head_readback:([.[]|select(.format=="irodori-v4-wgpu-duration-benchmark-v1")|.repeats[]|select(.scope=="head" and (.cold|not))|.timing.readback_complete_seconds]|stats),
           full_device:([.[]|select(.format=="irodori-v4-wgpu-duration-benchmark-v1")|.repeats[]|select(.scope=="full" and (.cold|not))|.timing.device_complete_seconds]|stats),
           full_readback:([.[]|select(.format=="irodori-v4-wgpu-duration-benchmark-v1")|.repeats[]|select(.scope=="full" and (.cold|not))|.timing.readback_complete_seconds]|stats)},
         all_point_wins:{
           head_device:(([.[]|select(.format=="irodori-v4-wgpu-duration-benchmark-v1")|.repeats[]|select(.scope=="head" and (.cold|not))|.timing.device_complete_seconds]|max) < ([.[]|select(.format=="irodori-v4-python-duration-benchmark-v1")|.repeats[]|select(.scope=="head" and (.cold|not))|.timing.device_complete_seconds]|min)),
           head_readback:(([.[]|select(.format=="irodori-v4-wgpu-duration-benchmark-v1")|.repeats[]|select(.scope=="head" and (.cold|not))|.timing.readback_complete_seconds]|max) < ([.[]|select(.format=="irodori-v4-python-duration-benchmark-v1")|.repeats[]|select(.scope=="head" and (.cold|not))|.timing.readback_complete_seconds]|min)),
           full_device:(([.[]|select(.format=="irodori-v4-wgpu-duration-benchmark-v1")|.repeats[]|select(.scope=="full" and (.cold|not))|.timing.device_complete_seconds]|max) < ([.[]|select(.format=="irodori-v4-python-duration-benchmark-v1")|.repeats[]|select(.scope=="full" and (.cold|not))|.timing.device_complete_seconds]|min)),
           full_readback:(([.[]|select(.format=="irodori-v4-wgpu-duration-benchmark-v1")|.repeats[]|select(.scope=="full" and (.cold|not))|.timing.readback_complete_seconds]|max) < ([.[]|select(.format=="irodori-v4-python-duration-benchmark-v1")|.repeats[]|select(.scope=="full" and (.cold|not))|.timing.readback_complete_seconds]|min))}}
      ' "$OUT"/sessions/"$name"/s{1,2,3}/python.json "$OUT"/sessions/"$name"/s{1,2,3}/wgpu.json
    done | jq -s .
  )" \
  '{format:$format,timer_contract:{primary:$timer_primary,secondary:$timer_secondary},fresh_processes_per_runtime_per_case:3,warmups_per_scope:5,measured_per_scope:10,cases:$cases}' \
  >"$OUT/summary.json"

jq -e '
  (.cases | length) == 4 and
  all(.cases[]; .resolved_length_equal_across_runtimes) and
  [.cases[].resolved_length.latent_frames] == [45,112,333,685] and
  [.cases[].resolved_length.target_samples] == [86400,215040,639360,1315200] and
  [.cases[].resolved_length.seconds] == [1.8,4.48,13.32,27.4]
' "$OUT/summary.json" >/dev/null || die "resolved duration aggregation failed"

printf 'complete\n' >"$OUT/COMPLETE"
(
  cd "$OUT"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
  sha256sum --check SHA256SUMS >/dev/null
)
find "$OUT" -type f -exec chmod 0444 {} +
find "$OUT" -type d -exec chmod 0555 {} +
printf 'duration_sweep_complete=%s\n' "$OUT"
