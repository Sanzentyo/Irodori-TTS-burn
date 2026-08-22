#!/usr/bin/env bash
# Fresh diagnostic capture of allocation high-water marks inside each SDPA stage.

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
ACTIVE_MONITOR=
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
  printf 'status=%s\nautomatic_retries=0\ndiagnostic_stage_sync=true\n' "$status" >"$OUT/$status"
  (cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
}
on_exit() { local status=$?; if ((status != 0 && ! COMPLETE)); then set +e; seal FAILURE; fi; return "$status"; }
trap on_exit EXIT

for command in flock git jq nvidia-smi taskset; do command -v "$command" >/dev/null || die "missing $command"; done
for path in "$MODEL" "$CODEC" "$FIXTURE" "$REF1" "$REF2" "$BUNDLE" "$BIN"; do
  [[ -f $path && -s $path ]] || die "missing input: $path"
done
[[ -z $(git -C "$ROOT" status --short) ]] || die 'source tree must be clean'

mkdir -p "$OUT/build" "$OUT/cache" "$OUT/xdg"
install -m 0555 "$BIN" "$OUT/build/bench_v4_residency"
install -m 0444 "$0" "$OUT/build/runner.sh"
git -C "$ROOT" rev-parse HEAD >"$OUT/source-head.txt"
nvidia-smi -q >"$OUT/nvidia-smi-q.txt"
sha256sum "$OUT/build"/* "$MODEL" "$CODEC" "$FIXTURE" "$REF1" "$REF2" "$BUNDLE" >"$OUT/pins.sha256"

exec 9>>/tmp/irodori-v4-sdpa-internal-peak-gpu0.lock
flock -n 9 || die 'GPU lock is held'
nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits -lms 100 -f "$OUT/nvml.csv" &
ACTIVE_MONITOR=$!
set +e
env -u CUDA_VISIBLE_DEVICES WGPU_BACKEND=vulkan XDG_CACHE_HOME="$OUT/xdg" \
  IRODORI_RF_DETAIL_PROFILE=1 taskset -c 0-11 "$OUT/build/bench_v4_residency" \
    --mode all-resident --checkpoint "$MODEL" --codec-weights "$CODEC" \
    --fixture "$FIXTURE" --reference "$REF1" "$REF2" --requests 1 --warmups 0 \
    --num-steps 4 --cfg-caption 4 --designed --precision fp32 --allocator exclusive-pages \
    --codec-residency decode-only --load-strategy parallel --rf-checkpoint-loader indexed-file \
    --rf-weight-residency production-prepared --cubecl-cache-dir "$OUT/cache" \
    --cubecl-bundle-in "$BUNDLE" --output-json "$OUT/result.json" \
    >"$OUT/stdout.log" 2>"$OUT/stderr.log"
status=$?
set -e
stop_monitor
((status == 0)) || die 'profile target failed'
grep 'stage=sdpa_internal_peak' "$OUT/stderr.log" >"$OUT/sdpa-internal-peak.log"
[[ $(wc -l <"$OUT/sdpa-internal-peak.log") -eq 48 ]] || die 'expected one SDPA peak receipt per block call'
jq -e '.euler_evaluations==4 and .block_calls==48 and .rf_weight_residency=="production_prepared"' \
  "$OUT/result.json" >/dev/null || die 'work manifest gate failed'

awk '
  BEGIN { print "batch\tsequence\tpeak_delta_in_use_bytes\tpeak_delta_reserved_bytes\treservation_events" }
  {
    delete value
    for (i=1; i<=NF; i++) { split($i, field, "="); value[field[1]]=field[2] }
    print value["batch"] "\t" value["sequence"] "\t" value["peak_delta_in_use_bytes"] "\t" value["peak_delta_reserved_bytes"] "\t" value["reservation_events"]
  }
' "$OUT/sdpa-internal-peak.log" >"$OUT/sdpa-internal-peak.tsv"

COMPLETE=1
seal COMPLETE
printf 'complete: %s\n' "$OUT"
