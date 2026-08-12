#!/usr/bin/env bash
# Freeze the 45-frame fresh/restored/process-warm strict-FP32 accuracy states.

set -Eeuo pipefail
IFS=$'\n\t'

SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
CAMPAIGN_OUT=${1:-}
[[ -n $CAMPAIGN_OUT ]] || { printf 'usage: %s FRESH_OUTPUT_DIR\n' "$0" >&2; exit 2; }
[[ $CAMPAIGN_OUT == /* ]] || CAMPAIGN_OUT=$PWD/$CAMPAIGN_OUT
CAMPAIGN_OUT=$(realpath -m -- "$CAMPAIGN_OUT")
[[ ! -e $CAMPAIGN_OUT && ! -L $CAMPAIGN_OUT ]] || { printf 'error: output exists: %s\n' "$CAMPAIGN_OUT" >&2; exit 1; }

USER_HOME_DIR=$(getent passwd "$(id -u)" | cut -d: -f6)
MODEL=$USER_HOME_DIR/.cache/huggingface/hub/models--Aratako--Irodori-TTS-v4-Small/snapshots/e4aaac4df355ff560dcd35e0dae272c3a759317b/model.safetensors
CODEC=$SOURCE_ROOT/target/v4_dacvae_weights.safetensors
FIXTURE=$USER_HOME_DIR/benchmark-artifacts/irodori-v4-12gb-baseline-20260812-attempt1/accuracy-campaign/lengths/s1p8/oracle.safetensors
POLICY=$SOURCE_ROOT/docs/benchmarks/runtime-scenarios-12gb-2026-08-12/v4-autotune-accuracy-policy-0.21.json
MODEL_SHA=5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593
CODEC_SHA=4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1
FIXTURE_SHA=54022cffb74c0828793a989d77b664fcae305ac78c491db748ee43f5851740c8
GPU_NAME='NVIDIA GeForce RTX 5070 Ti Laptop GPU'
GPU_DRIVER=595.71.05
GPU_PCI=00000000:01:00.0
GPU_TOTAL_MIB=12227
LOCK=/tmp/irodori-v4-12gb-gpu0.lock
CURRENT_PHASE=preflight
COMPLETE=0
ACTIVE_MONITOR=

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
  [[ -d $CAMPAIGN_OUT && ! -L $CAMPAIGN_OUT ]] || return 0
  printf 'status=%s\nphase=%s\nautomatic_retries=0\nold_measurements_pooled=false\n' \
    "$status" "$CURRENT_PHASE" >"$CAMPAIGN_OUT/$status"
  (cd "$CAMPAIGN_OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS)
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

for input in "$MODEL" "$CODEC" "$FIXTURE" "$POLICY"; do
  [[ -f $input && -s $input ]] || die "missing input: $input"
done
[[ $(sha "$MODEL") == "$MODEL_SHA" ]] || die 'model SHA mismatch'
[[ $(sha "$CODEC") == "$CODEC_SHA" ]] || die 'codec SHA mismatch'
[[ $(sha "$FIXTURE") == "$FIXTURE_SHA" ]] || die 'fixture SHA mismatch'
jq -e '.schema_version == 2 and .required_cases[0].fixture_sha256 == "54022cffb74c0828793a989d77b664fcae305ac78c491db748ee43f5851740c8" and .required_cases[0].latent_frames == 45 and .required_cases[0].hard_gate.waveform.minimum_snr_db == 80 and .required_cases[0].target_waveform_snr_db == 85' "$POLICY" >/dev/null || die 'approval policy gate failed'

gpu_row=$(nvidia-smi -i 0 --query-gpu=name,driver_version,pci.bus_id,memory.total --format=csv,noheader,nounits)
[[ $gpu_row == "$GPU_NAME, $GPU_DRIVER, $GPU_PCI, $GPU_TOTAL_MIB" ]] || die "GPU identity mismatch: $gpu_row"

mkdir -p "$CAMPAIGN_OUT/build" "$CAMPAIGN_OUT/environment" "$CAMPAIGN_OUT/fresh-autotune" \
  "$CAMPAIGN_OUT/restored-autotune" "$CAMPAIGN_OUT/process-warm"
git -C "$SOURCE_ROOT" diff --binary >"$CAMPAIGN_OUT/build/source.diff"
git -C "$SOURCE_ROOT" status --short >"$CAMPAIGN_OUT/build/source-status.txt"
install -m 0444 "$POLICY" "$CAMPAIGN_OUT/build/approval-policy.json"

CURRENT_PHASE=build
cargo build --manifest-path "$SOURCE_ROOT/Cargo.toml" --release --features inference,codec,cli \
  --bin validate_v4_precision --bin approve_v4_autotune --bin probe_wgpu_adapter \
  >"$CAMPAIGN_OUT/build/cargo.stdout.log" 2>"$CAMPAIGN_OUT/build/cargo.stderr.log"
install -m 0555 "$SOURCE_ROOT/target/release/validate_v4_precision" "$CAMPAIGN_OUT/build/validate_v4_precision"
install -m 0555 "$SOURCE_ROOT/target/release/approve_v4_autotune" "$CAMPAIGN_OUT/build/approve_v4_autotune"
install -m 0555 "$SOURCE_ROOT/target/release/probe_wgpu_adapter" "$CAMPAIGN_OUT/build/probe_wgpu_adapter"

nvidia-smi -q >"$CAMPAIGN_OUT/environment/nvidia-smi-q.txt"
nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id,memory.total,memory.free --format=csv,noheader,nounits \
  >"$CAMPAIGN_OUT/environment/nvidia-smi.csv"
if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary >"$CAMPAIGN_OUT/environment/vulkan-summary.txt" 2>"$CAMPAIGN_OUT/environment/vulkan-summary.stderr.log"
else
  printf 'vulkaninfo unavailable; adapter identity is recorded by probe_wgpu_adapter\n' \
    >"$CAMPAIGN_OUT/environment/vulkan-summary.txt"
  : >"$CAMPAIGN_OUT/environment/vulkan-summary.stderr.log"
fi
env WGPU_BACKEND=vulkan "$CAMPAIGN_OUT/build/probe_wgpu_adapter" \
  >"$CAMPAIGN_OUT/environment/wgpu-adapter.json" 2>"$CAMPAIGN_OUT/environment/wgpu-adapter.stderr.log"
rustc --version --verbose >"$CAMPAIGN_OUT/environment/rustc.txt"
cargo --version >"$CAMPAIGN_OUT/environment/cargo.txt"
cargo tree --manifest-path "$SOURCE_ROOT/Cargo.toml" -d >"$CAMPAIGN_OUT/environment/cargo-tree-duplicates.txt"

jq -n \
  --arg burn '0.21.0-pre.3' --arg burn_cubecl '0.21.0-pre.3' --arg cubecl '0.10.0-pre.3' \
  --arg runtime 'CubeBackend<WgpuRuntime,f32,i32,u32>' --arg backend vulkan \
  --arg gpu "$GPU_NAME" --arg driver "$GPU_DRIVER" --arg pci "$GPU_PCI" \
  --arg allocator exclusive-pages --arg float f32 --arg int i32 \
  --arg bounds cubecl-default-checked \
  --arg model_revision e4aaac4df355ff560dcd35e0dae272c3a759317b --arg model_sha "$MODEL_SHA" \
  --arg codec_revision 47376ee24834d7a05a48ebabfe3cde29b3c5e214 --arg codec_sha "$CODEC_SHA" \
  --arg fixture_sha "$FIXTURE_SHA" \
  '{burn_version:$burn,burn_cubecl_version:$burn_cubecl,cubecl_version:$cubecl,runtime:$runtime,
    wgpu_backend:$backend,gpu_name:$gpu,driver_version:$driver,pci_bus_id:$pci,
    allocator_policy:$allocator,float_dtype:$float,int_dtype:$int,bounds_check_policy:$bounds,
    model_revision:$model_revision,model_sha256:$model_sha,codec_revision:$codec_revision,
    converted_codec_sha256:$codec_sha,fixture_sha256:$fixture_sha}' \
  >"$CAMPAIGN_OUT/environment/runtime-identity.json"

printf 'source_head=%s\nsource_diff_sha256=%s\nprecision=strict-fp32\ntf32=false\nautocast=false\nframes=45\nseconds=1.8\nmemory_config=exclusive-pages\ntasks_max=32\nautomatic_retries=0\nold_measurements_pooled=false\n' \
  "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" "$(sha "$CAMPAIGN_OUT/build/source.diff")" \
  >"$CAMPAIGN_OUT/protocol.txt"
sha256sum "$CAMPAIGN_OUT/build"/* "$MODEL" "$CODEC" "$FIXTURE" >"$CAMPAIGN_OUT/pins.sha256"

exec 9>>"$LOCK"
flock -n 9 || die 'GPU0 campaign lock is held'

wait_idle() {
  local stable=0 telemetry processes
  for _ in $(seq 1 30); do
    telemetry=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ $telemetry =~ ^([0-9]+),[[:space:]]*([0-9]+)$ ]] \
      && ((BASH_REMATCH[1] <= 128 && BASH_REMATCH[2] <= 5)) && [[ ! $processes =~ [0-9] ]]; then
      ((stable += 1))
      ((stable >= 2)) && return 0
    else
      stable=0
    fi
    sleep 1
  done
  die 'GPU did not settle'
}

run_monitored() {
  local directory=$1
  shift
  nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader,nounits -lms 100 -f "$directory/nvml.csv" &
  ACTIVE_MONITOR=$!
  local status
  if /usr/bin/time -o "$directory/wall.txt" -f 'exit_status=%x\nelapsed_seconds=%e\nmax_rss_kib=%M' \
    "$@" >"$directory/stdout.log" 2>"$directory/stderr.log"; then
    status=0
  else
    status=$?
  fi
  stop_monitor
  return "$status"
}

VALIDATOR_ARGS=(
  --execution wgsl --precision fp32 --fixture "$FIXTURE" --fixture-sha256 "$FIXTURE_SHA"
  --checkpoint "$MODEL" --codec-weights "$CODEC" --adapter-index 0 --tasks-max 32
  --memory-config exclusive-pages --enforce
  --latent-max-abs 0.0002 --latent-mean-abs 0.00001 --latent-rmse 0.00002
  --latent-min-snr-db 90 --latent-min-cosine 0.99999999
  --waveform-max-abs 0.00015 --waveform-mean-abs 0.000005 --waveform-rmse 0.00001
  --waveform-min-snr-db 80 --waveform-min-cosine 0.99999999
)

CURRENT_PHASE=fresh-autotune
fresh_table=$CAMPAIGN_OUT/fresh-autotune/results.tsv
: >"$fresh_table"
for session in 1 2 3 4 5; do
  session_dir=$CAMPAIGN_OUT/fresh-autotune/session-$session
  mkdir "$session_dir"
  wait_idle
  if run_monitored "$session_dir" env -u CUDA_VISIBLE_DEVICES WGPU_BACKEND=vulkan RUST_LOG=warn \
    "$CAMPAIGN_OUT/build/validate_v4_precision" "${VALIDATOR_ARGS[@]}" \
    --cubecl-cache-dir "$session_dir/cache" --repeats 1; then
    fresh_status=0
  else
    fresh_status=$?
  fi
  latent_max_abs=$(sed -n 's/^final_patched_latent\[1\].*max_abs=\([^ ]*\).*/\1/p' "$session_dir/stdout.log")
  latent_mean_abs=$(sed -n 's/^final_patched_latent\[1\].*mean_abs=\([^ ]*\).*/\1/p' "$session_dir/stdout.log")
  latent_rmse=$(sed -n 's/^final_patched_latent\[1\].*rmse=\([^ ]*\).*/\1/p' "$session_dir/stdout.log")
  fresh_latent_snr=$(sed -n 's/^final_patched_latent\[1\].*snr_db=\([^ ]*\).*/\1/p' "$session_dir/stdout.log")
  latent_cosine=$(sed -n 's/^final_patched_latent\[1\].*cosine=\([^ ]*\).*/\1/p' "$session_dir/stdout.log")
  waveform_max_abs=$(sed -n 's/^raw_decoded_waveform\[1\].*max_abs=\([^ ]*\).*/\1/p' "$session_dir/stdout.log")
  waveform_mean_abs=$(sed -n 's/^raw_decoded_waveform\[1\].*mean_abs=\([^ ]*\).*/\1/p' "$session_dir/stdout.log")
  waveform_rmse=$(sed -n 's/^raw_decoded_waveform\[1\].*rmse=\([^ ]*\).*/\1/p' "$session_dir/stdout.log")
  fresh_snr=$(sed -n 's/^raw_decoded_waveform\[1\].*snr_db=\([^ ]*\).*/\1/p' "$session_dir/stdout.log")
  waveform_cosine=$(sed -n 's/^raw_decoded_waveform\[1\].*cosine=\([^ ]*\).*/\1/p' "$session_dir/stdout.log")
  fresh_latent_hash=$(sed -n 's/^repeat_tensor_sha256 name=final_patched_latent repeat=1 .*sha256=\([0-9a-f]*\)$/\1/p' "$session_dir/stdout.log")
  fresh_waveform_hash=$(sed -n 's/^repeat_tensor_sha256 name=raw_decoded_waveform repeat=1 .*sha256=\([0-9a-f]*\)$/\1/p' "$session_dir/stdout.log")
  rf_device=$(sed -n 's/^rf_repeat=1\/1 sample_device_complete_s=\([^ ]*\).*/\1/p' "$session_dir/stdout.log")
  codec_device=$(sed -n 's/^codec_repeat=1\/1 decode_device_complete_s=\([^ ]*\).*/\1/p' "$session_dir/stdout.log")
  [[ -n $fresh_latent_snr && -n $fresh_snr && ${#fresh_latent_hash} == 64 \
    && ${#fresh_waveform_hash} == 64 && -n $rf_device && -n $codec_device ]] \
    || die "fresh session $session evidence is incomplete"
  total_device=$(awk -v rf="$rf_device" -v codec="$codec_device" 'BEGIN { printf "%.9f", rf + codec }')
  if [[ $fresh_status == 0 ]]; then
    awk -v snr="$fresh_snr" 'BEGIN { exit !(snr >= 80.0) }' || die "fresh session $session returned success below hard gate"
    status_label=PASS
    printf 'pass\n' >"$session_dir/PASS"
  elif [[ $fresh_status == 1 ]]; then
    status_label=EXPECTED_ACCURACY_FAILURE
    printf 'expected_accuracy_failure\n' >"$session_dir/EXPECTED_FAILURE"
  else
    die "fresh session $session failed outside numerical gate with status $fresh_status"
  fi
  if awk -v snr="$fresh_snr" 'BEGIN { exit !(snr >= 85.0) }'; then target_status=TARGET_PASS; else target_status=TARGET_WARNING; fi
  jq -n --argjson session "$session" --arg status "$status_label" --arg target_status "$target_status" \
    --argjson latent_max_abs "$latent_max_abs" --argjson latent_mean_abs "$latent_mean_abs" \
    --argjson latent_rmse "$latent_rmse" --argjson latent_snr "$fresh_latent_snr" --argjson latent_cosine "$latent_cosine" \
    --argjson waveform_max_abs "$waveform_max_abs" --argjson waveform_mean_abs "$waveform_mean_abs" \
    --argjson waveform_rmse "$waveform_rmse" --argjson waveform_snr "$fresh_snr" --argjson waveform_cosine "$waveform_cosine" \
    --argjson rf "$rf_device" --argjson codec "$codec_device" --argjson total "$total_device" \
    --arg latent_hash "$fresh_latent_hash" --arg waveform_hash "$fresh_waveform_hash" \
    '{session:$session,status:$status,target_status:$target_status,
      latent:{max_abs:$latent_max_abs,mean_abs:$latent_mean_abs,rmse:$latent_rmse,snr_db:$latent_snr,cosine:$latent_cosine},
      waveform:{max_abs:$waveform_max_abs,mean_abs:$waveform_mean_abs,rmse:$waveform_rmse,snr_db:$waveform_snr,cosine:$waveform_cosine},
      rf_device_s:$rf,codec_device_s:$codec,total_device_s:$total,
      latent_sha256:$latent_hash,waveform_sha256:$waveform_hash}' >"$session_dir/result.json"
  printf '%s\t%s\t%s\n' "$session" "$status_label" "$total_device" >>"$fresh_table"
done

chosen_session=$(awk -F '\t' '$2 == "PASS" {print $1, $3}' "$fresh_table" | sort -k2,2n | awk 'NR == 1 {print $1}')
[[ -n $chosen_session ]] || die 'none of the five predefined fresh sessions passed accuracy'
chosen_dir=$CAMPAIGN_OUT/fresh-autotune/session-$chosen_session
jq -s '.' "$CAMPAIGN_OUT"/fresh-autotune/session-*/result.json >"$CAMPAIGN_OUT/fresh-autotune/results.json"
printf '%s\n' "$chosen_session" >"$CAMPAIGN_OUT/fresh-autotune/selected-session.txt"

CURRENT_PHASE=approve-cache
cp -a "$chosen_dir/cache" "$CAMPAIGN_OUT/restored-autotune/cache"
jq --arg fixture "$FIXTURE_SHA" '. | {cases:[{fixture_sha256:$fixture,latent_frames:45,
  latent,waveform,latent_sha256,waveform_sha256}]}' \
  "$chosen_dir/result.json" >"$CAMPAIGN_OUT/restored-autotune/accuracy-evidence.json"
"$CAMPAIGN_OUT/build/approve_v4_autotune" seal \
  --policy "$CAMPAIGN_OUT/build/approval-policy.json" \
  --identity "$CAMPAIGN_OUT/environment/runtime-identity.json" \
  --evidence "$CAMPAIGN_OUT/restored-autotune/accuracy-evidence.json" \
  --cache-root "$CAMPAIGN_OUT/restored-autotune/cache" \
  --output-manifest "$CAMPAIGN_OUT/restored-autotune/approved-cache-manifest.json" \
  >"$CAMPAIGN_OUT/restored-autotune/approval.stdout.log" \
  2>"$CAMPAIGN_OUT/restored-autotune/approval.stderr.log"
"$CAMPAIGN_OUT/build/approve_v4_autotune" verify \
  --manifest "$CAMPAIGN_OUT/restored-autotune/approved-cache-manifest.json" \
  --identity "$CAMPAIGN_OUT/environment/runtime-identity.json" \
  --cache-root "$CAMPAIGN_OUT/restored-autotune/cache" \
  --receipt "$CAMPAIGN_OUT/restored-autotune/pre-run-verification.json" \
  >"$CAMPAIGN_OUT/restored-autotune/verify.stdout.log" \
  2>"$CAMPAIGN_OUT/restored-autotune/verify.stderr.log"

CURRENT_PHASE=restored-autotune
wait_idle
run_monitored "$CAMPAIGN_OUT/restored-autotune" env -u CUDA_VISIBLE_DEVICES WGPU_BACKEND=vulkan RUST_LOG=warn \
  "$CAMPAIGN_OUT/build/validate_v4_precision" "${VALIDATOR_ARGS[@]}" \
  --cubecl-cache-dir "$CAMPAIGN_OUT/restored-autotune/cache" --repeats 1 \
  || die 'restored autotune accuracy failed without retry'
restored_snr=$(sed -n 's/^raw_decoded_waveform\[1\].*snr_db=\([^ ]*\).*/\1/p' "$CAMPAIGN_OUT/restored-autotune/stdout.log")
restored_latent_hash=$(sed -n 's/^repeat_tensor_sha256 name=final_patched_latent repeat=1 .*sha256=\([0-9a-f]*\)$/\1/p' "$CAMPAIGN_OUT/restored-autotune/stdout.log")
restored_waveform_hash=$(sed -n 's/^repeat_tensor_sha256 name=raw_decoded_waveform repeat=1 .*sha256=\([0-9a-f]*\)$/\1/p' "$CAMPAIGN_OUT/restored-autotune/stdout.log")
[[ -n $restored_snr && ${#restored_latent_hash} == 64 && ${#restored_waveform_hash} == 64 ]] \
  || die 'restored accuracy evidence is incomplete'
awk -v snr="$restored_snr" 'BEGIN { exit !(snr >= 80.0) }' || die 'restored hard SNR gate failed'
approved_latent_hash=$(jq -r '.latent_sha256' "$chosen_dir/result.json")
approved_waveform_hash=$(jq -r '.waveform_sha256' "$chosen_dir/result.json")
[[ $restored_latent_hash == "$approved_latent_hash" ]] || die 'restored latent differs from the approved cache evidence'
[[ $restored_waveform_hash == "$approved_waveform_hash" ]] || die 'restored waveform differs from the approved cache evidence'
"$CAMPAIGN_OUT/build/approve_v4_autotune" verify \
  --manifest "$CAMPAIGN_OUT/restored-autotune/approved-cache-manifest.json" \
  --identity "$CAMPAIGN_OUT/environment/runtime-identity.json" \
  --cache-root "$CAMPAIGN_OUT/restored-autotune/cache" \
  --receipt "$CAMPAIGN_OUT/restored-autotune/post-run-verification.json" \
  >"$CAMPAIGN_OUT/restored-autotune/post-verify.stdout.log" \
  2>"$CAMPAIGN_OUT/restored-autotune/post-verify.stderr.log"
printf 'pass\n' >"$CAMPAIGN_OUT/restored-autotune/PASS"

CURRENT_PHASE=process-warm
cp -a "$CAMPAIGN_OUT/restored-autotune/cache" "$CAMPAIGN_OUT/process-warm/cache"
"$CAMPAIGN_OUT/build/approve_v4_autotune" verify \
  --manifest "$CAMPAIGN_OUT/restored-autotune/approved-cache-manifest.json" \
  --identity "$CAMPAIGN_OUT/environment/runtime-identity.json" \
  --cache-root "$CAMPAIGN_OUT/process-warm/cache" \
  --receipt "$CAMPAIGN_OUT/process-warm/pre-run-verification.json" \
  >"$CAMPAIGN_OUT/process-warm/verify.stdout.log" \
  2>"$CAMPAIGN_OUT/process-warm/verify.stderr.log"
wait_idle
run_monitored "$CAMPAIGN_OUT/process-warm" env -u CUDA_VISIBLE_DEVICES WGPU_BACKEND=vulkan RUST_LOG=warn \
  "$CAMPAIGN_OUT/build/validate_v4_precision" "${VALIDATOR_ARGS[@]}" \
  --cubecl-cache-dir "$CAMPAIGN_OUT/process-warm/cache" --repeats 2 \
  || die 'process-warm accuracy failed without retry'
[[ $(grep -Fc "sha256=$restored_latent_hash" "$CAMPAIGN_OUT/process-warm/stdout.log") == 2 ]] \
  || die 'process-warm latent determinism gate failed'
[[ $(grep -Fc "sha256=$restored_waveform_hash" "$CAMPAIGN_OUT/process-warm/stdout.log") == 2 ]] \
  || die 'process-warm waveform determinism gate failed'
printf 'pass\n' >"$CAMPAIGN_OUT/process-warm/PASS"

jq -n --slurpfile fresh "$CAMPAIGN_OUT/fresh-autotune/results.json" \
  --argjson chosen "$chosen_session" --argjson restored "$restored_snr" \
  --arg approved_latent "$restored_latent_hash" --arg approved_waveform "$restored_waveform_hash" \
  '{schema_version:2,status:"PASS",fresh_autotune:{sessions:$fresh[0],selected_session:$chosen},
    restored_autotune:{status:"PASS",waveform_snr_db:$restored},process_warm:{status:"PASS",repeats:2,deterministic:true},
    accuracy_gate:{waveform_hard_min_snr_db:80.0,waveform_target_snr_db:85.0},approved_hashes:{latent:$approved_latent,waveform:$approved_waveform},
    automatic_retries:0,old_measurements_pooled:false}' >"$CAMPAIGN_OUT/summary.json"

CURRENT_PHASE=complete
wait_idle
COMPLETE=1
seal COMPLETE
printf 'autotune_accuracy_tristate_complete=%s\n' "$CAMPAIGN_OUT"
