#!/usr/bin/env bash
# Nsight Systems profile of the matched 40-step production WGPU RF path.

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
    -h|--help) printf 'usage: %s --output-dir FRESH --input-campaign SEALED_ACCURACY_CAMPAIGN\n' "$0"; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
[[ -n $OUT && -n $INPUT ]] || { printf 'error: both paths are required\n' >&2; exit 2; }
OUT=$(realpath -m -- "$OUT")
INPUT=$(realpath -- "$INPUT")
[[ ! -e $OUT && ! -L $OUT ]] || { printf 'error: output exists: %s\n' "$OUT" >&2; exit 1; }
[[ -f $INPUT/COMPLETE && -f $INPUT/SHA256SUMS ]] || { printf 'error: input is not sealed\n' >&2; exit 1; }

MODEL="$USER_ROOT/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4-Small/snapshots/e4aaac4df355ff560dcd35e0dae272c3a759317b/model.safetensors"
CODEC="$USER_ROOT/benchmark-artifacts/irodori-v4-load-opt-20260813-attempt1/models/dacvae-decoder-only.safetensors"
FIXTURE="$INPUT/preparation/f489/fixtures/design.safetensors"
REF1="$INPUT/inputs/references/ref1.safetensors"
REF2="$INPUT/inputs/references/ref2.safetensors"
BUNDLE="$INPUT/prime/environment.cubecl"
BIN="$ROOT/target/release/bench_v4_residency"
LOCK=/tmp/irodori-v4-40step-rf-profile-gpu0.lock
COMPLETE=0

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
seal() {
  local status=$1
  [[ -d $OUT ]] || return 0
  printf 'status=%s\nautomatic_retries=0\nprofile_latency_is_not_formal_latency=true\n' "$status" >"$OUT/$status"
  (cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
}
on_exit() { local s=$?; if ((s != 0 && ! COMPLETE)); then set +e; seal FAILURE; fi; return "$s"; }
trap on_exit EXIT

for command in cargo flock git jq nsys nvidia-smi taskset; do command -v "$command" >/dev/null || die "missing $command"; done
for path in "$MODEL" "$CODEC" "$FIXTURE" "$REF1" "$REF2" "$BUNDLE"; do [[ -f $path && -s $path ]] || die "missing $path"; done
[[ -z $(git -C "$ROOT" status --short) ]] || die 'source tree must be clean'

cargo build --release --locked --features inference,codec,cli,profile --bin bench_v4_residency
mkdir -p "$OUT/build" "$OUT/cubecl" "$OUT/xdg"
install -m 0555 "$BIN" "$OUT/build/bench_v4_residency"
install -m 0444 "$0" "$OUT/build/runner.sh"
git -C "$ROOT" rev-parse HEAD >"$OUT/source-head.txt"
nvidia-smi -q >"$OUT/nvidia-smi-q.txt"
sha256sum "$OUT/build"/* "$MODEL" "$CODEC" "$FIXTURE" "$REF1" "$REF2" "$BUNDLE" >"$OUT/pins.sha256"

exec 9>>"$LOCK"
flock -n 9 || die 'RF profile GPU lock is held'
idle=0
for _ in $(seq 1 60); do
  row=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  pids=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
  if [[ $row =~ ^([0-9]+),[[:space:]]*([0-9]+)$ ]] && ((BASH_REMATCH[1] <= 512 && BASH_REMATCH[2] <= 5)) && [[ ! $pids =~ [0-9] ]]; then idle=1; break; fi
  sleep 1
done
((idle == 1)) || die 'GPU did not become idle'

set +e
nsys profile --trace=vulkan,nvtx,osrt --vulkan-gpu-workload=true --sample=none \
  --cpuctxsw=none --wait=primary --force-overwrite=false --output "$OUT/rf-489-design" \
  /usr/bin/env -u CUDA_VISIBLE_DEVICES WGPU_BACKEND=vulkan XDG_CACHE_HOME="$OUT/xdg" \
  taskset -c 0-11 "$OUT/build/bench_v4_residency" --mode all-resident \
    --checkpoint "$MODEL" --codec-weights "$CODEC" --fixture "$FIXTURE" \
    --reference "$REF1" "$REF2" --requests 2 --warmups 1 --num-steps 40 --cfg-caption 4 \
    --designed --precision fp32 --allocator exclusive-pages --codec-residency decode-only \
    --load-strategy parallel --rf-checkpoint-loader indexed-file \
    --rf-weight-residency production-prepared --cubecl-cache-dir "$OUT/cubecl" \
    --cubecl-bundle-in "$BUNDLE" --output-json "$OUT/result.json" \
    >"$OUT/stdout.log" 2>"$OUT/stderr.log"
NSYS_STATUS=$?
set -e
printf 'nsys_profile_exit_status=%s\n' "$NSYS_STATUS" >"$OUT/nsys-status.txt"

REP="$OUT/rf-489-design.nsys-rep"
[[ -f $REP && -s $REP ]] || die 'missing Nsight report'
[[ -f $OUT/result.json && -s $OUT/result.json ]] || die 'profiled target did not write its result'
if ((NSYS_STATUS != 0 && NSYS_STATUS != 1)); then
  die "unexpected nsys profile exit status $NSYS_STATUS"
fi
nsys stats --report nvtx_pushpop_sum --format csv "$REP" >"$OUT/nvtx-pushpop.csv"
nsys stats --report vulkan_gpu_marker_sum --format csv "$REP" >"$OUT/vulkan-gpu-marker.csv"
nsys stats --report vulkan_api_sum --format csv "$REP" >"$OUT/vulkan-api.csv"
jq -e '.euler_evaluations==40 and .block_calls==480 and .rf_weight_residency=="production_prepared"' \
  "$OUT/result.json" >/dev/null || die 'profile work manifest gate failed'
COMPLETE=1
seal COMPLETE
printf 'complete: %s\n' "$OUT"
