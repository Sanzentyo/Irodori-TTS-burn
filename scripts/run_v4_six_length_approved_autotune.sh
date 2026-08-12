#!/usr/bin/env bash
# Extend a 45-frame approved cache to six lengths and verify restoration.

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
FIXTURE_ROOT=$USER_HOME_DIR/benchmark-artifacts/irodori-v4-12gb-baseline-20260812-attempt1/accuracy-campaign/lengths
SOURCE_APPROVED=$USER_HOME_DIR/benchmark-artifacts/irodori-v4-autotune-accuracy-tristate-20260812-attempt6
SOURCE_APPROVED_SUMS_SHA=8a8847057756d1f8aa2ad01f936da94c5acad91a1bd5f3c0f6354d757f8bdd02
POLICY=$SOURCE_ROOT/docs/benchmarks/runtime-scenarios-12gb-2026-08-12/v4-autotune-accuracy-policy-six-lengths-0.21.json
MODEL_SHA=5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593
CODEC_SHA=4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1
SLUGS=(s1p8 s4p48 s10p2 s13p32 s19p56 s27p4)
FRAMES=(45 112 255 333 489 685)
FIXTURE_SHAS=(
  54022cffb74c0828793a989d77b664fcae305ac78c491db748ee43f5851740c8
  f90e785823da3a0ec05caddadfc3d337bf833ad003daa9ff968f42086043d032
  dbe7a09c74ba9c9b5da1fb861c59ac526514bbe28c4c9c09827e24302094afbb
  52e5524270c4885ecdb961429adf21c3e766c486f20f497f6ddc144b6714cb9d
  85e78a9748ec01f37ad2ee0f1692e76ef84a5fd48972d570a02b813208d12872
  c6066c1be1030d73daa1921571ea4050266a445242b57e2a31e4fa3113f8f9cd
)
GPU_NAME='NVIDIA GeForce RTX 5070 Ti Laptop GPU'
GPU_DRIVER=595.71.05
GPU_PCI=00000000:01:00.0
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
  if ((status != 0 && ! COMPLETE)); then set +e; seal FAILURE; fi
  return "$status"
}
trap on_exit EXIT

[[ -f $MODEL && $(sha "$MODEL") == "$MODEL_SHA" ]] || die 'model pin failed'
[[ -f $CODEC && $(sha "$CODEC") == "$CODEC_SHA" ]] || die 'codec pin failed'
[[ -f $POLICY ]] || die 'six-length policy missing'
[[ -f $SOURCE_APPROVED/COMPLETE ]] || die 'source approved campaign incomplete'
[[ $(sha "$SOURCE_APPROVED/SHA256SUMS") == "$SOURCE_APPROVED_SUMS_SHA" ]] || die 'source approved campaign SHA mismatch'
(cd "$SOURCE_APPROVED" && sha256sum -c SHA256SUMS >/dev/null) || die 'source approved campaign checksum failure'
for index in "${!SLUGS[@]}"; do
  fixture=$FIXTURE_ROOT/${SLUGS[$index]}/oracle.safetensors
  [[ -f $fixture && $(sha "$fixture") == "${FIXTURE_SHAS[$index]}" ]] || die "fixture pin failed: ${SLUGS[$index]}"
done
gpu_row=$(nvidia-smi -i 0 --query-gpu=name,driver_version,pci.bus_id,memory.total --format=csv,noheader,nounits)
[[ $gpu_row == "$GPU_NAME, $GPU_DRIVER, $GPU_PCI, 12227" ]] || die "GPU identity mismatch: $gpu_row"

mkdir -p "$CAMPAIGN_OUT/build" "$CAMPAIGN_OUT/environment" "$CAMPAIGN_OUT/fresh-shapes" \
  "$CAMPAIGN_OUT/restored" "$CAMPAIGN_OUT/approved" "$CAMPAIGN_OUT/candidate-seed"
git -C "$SOURCE_ROOT" diff --binary >"$CAMPAIGN_OUT/build/source.diff"
git -C "$SOURCE_ROOT" status --short >"$CAMPAIGN_OUT/build/source-status.txt"
install -m 0444 "$POLICY" "$CAMPAIGN_OUT/build/accuracy-policy.json"
CURRENT_PHASE=build
cargo build --manifest-path "$SOURCE_ROOT/Cargo.toml" --release --features inference,codec,cli \
  --bin validate_v4_precision --bin approve_v4_autotune --bin probe_wgpu_adapter \
  >"$CAMPAIGN_OUT/build/cargo.stdout.log" 2>"$CAMPAIGN_OUT/build/cargo.stderr.log"
for binary in validate_v4_precision approve_v4_autotune probe_wgpu_adapter; do
  install -m 0555 "$SOURCE_ROOT/target/release/$binary" "$CAMPAIGN_OUT/build/$binary"
done
nvidia-smi -q >"$CAMPAIGN_OUT/environment/nvidia-smi-q.txt"
nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id,memory.total,memory.free --format=csv,noheader,nounits \
  >"$CAMPAIGN_OUT/environment/nvidia-smi.csv"
env WGPU_BACKEND=vulkan "$CAMPAIGN_OUT/build/probe_wgpu_adapter" >"$CAMPAIGN_OUT/environment/wgpu-adapter.json"
rustc --version --verbose >"$CAMPAIGN_OUT/environment/rustc.txt"
cargo --version >"$CAMPAIGN_OUT/environment/cargo.txt"
cargo tree --manifest-path "$SOURCE_ROOT/Cargo.toml" -d >"$CAMPAIGN_OUT/environment/cargo-tree-duplicates.txt"
jq '.identity' "$POLICY" >"$CAMPAIGN_OUT/environment/runtime-identity.json"
printf 'source_head=%s\nsource_diff_sha256=%s\nsource_45frame_cache_campaign=%s\nsource_45frame_SHA256SUMS_sha256=%s\nprecision=strict-fp32\ntf32=false\nautocast=false\nframes=45,112,255,333,489,685\nmemory_config=exclusive-pages\nautomatic_retries=0\nold_measurements_pooled=false\n' \
  "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" "$(sha "$CAMPAIGN_OUT/build/source.diff")" "$SOURCE_APPROVED" "$SOURCE_APPROVED_SUMS_SHA" \
  >"$CAMPAIGN_OUT/protocol.txt"
sha256sum "$CAMPAIGN_OUT/build"/* "$MODEL" "$CODEC" "$POLICY" "$SOURCE_APPROVED/SHA256SUMS" >"$CAMPAIGN_OUT/pins.sha256"

exec 9>>"$LOCK"
flock -n 9 || die 'GPU0 campaign lock is held'
wait_idle() {
  local stable=0 telemetry processes
  for _ in $(seq 1 30); do
    telemetry=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    processes=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ $telemetry =~ ^([0-9]+),[[:space:]]*([0-9]+)$ ]] \
      && ((BASH_REMATCH[1] <= 128 && BASH_REMATCH[2] <= 5)) && [[ ! $processes =~ [0-9] ]]; then
      ((stable += 1)); ((stable >= 2)) && return 0
    else stable=0; fi
    sleep 1
  done
  die 'GPU did not settle'
}
run_monitored() {
  local directory=$1; shift
  nvidia-smi --query-gpu=timestamp,index,pci.bus_id,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw \
    --format=csv,noheader,nounits -lms 100 -f "$directory/nvml.csv" &
  ACTIVE_MONITOR=$!
  local status
  if /usr/bin/time -o "$directory/wall.txt" -f 'exit_status=%x\nelapsed_seconds=%e\nmax_rss_kib=%M' \
    "$@" >"$directory/stdout.log" 2>"$directory/stderr.log"; then status=0; else status=$?; fi
  stop_monitor
  return "$status"
}

VALIDATOR_COMMON=(
  --execution wgsl --precision fp32 --checkpoint "$MODEL" --codec-weights "$CODEC"
  --adapter-index 0 --tasks-max 32 --memory-config exclusive-pages --enforce
  --latent-max-abs 0.0002 --latent-mean-abs 0.00001 --latent-rmse 0.00002
  --latent-min-snr-db 90 --latent-min-cosine 0.99999999
  --waveform-max-abs 0.00015 --waveform-mean-abs 0.000005 --waveform-rmse 0.00001
  --waveform-min-snr-db 80 --waveform-min-cosine 0.99999999
)

CURRENT_PHASE=fresh-shapes
cp -a "$SOURCE_APPROVED/restored-autotune/cache" "$CAMPAIGN_OUT/candidate-seed/cache"
reduce_log=$(find "$CAMPAIGN_OUT/candidate-seed/cache" -type f -name 'burn_cubecl-kernel-reduce-tune-reduce-dim.json.log' -print -quit)
[[ -n $reduce_log ]] || die 'reduce autotune log missing from candidate seed'
jq -sc 'map(if .key.key.vector_size == 1024 and .key.key.vector_count == 8
  then .value.fastest_index = 1 else . end)[]' "$reduce_log" >"$reduce_log.next"
mv "$reduce_log.next" "$reduce_log"
jq -e 'select(.key.key.vector_size == 1024 and .key.key.vector_count == 8) | .value.fastest_index == 1' \
  "$reduce_log" >/dev/null || die 'failed to seed the cross-length reduction candidate'
printf 'source_vector=%s\nchanged_key=reduce(vector_size=1024,vector_count=8)\nchanged_fastest_index=1\nreason=all-six hard-gate candidate; 85dB remains a target, not a hard gate\n' \
  "$SOURCE_APPROVED" >"$CAMPAIGN_OUT/candidate-seed/provenance.txt"
current_cache=$CAMPAIGN_OUT/candidate-seed/cache
for index in "${!SLUGS[@]}"; do
  slug=${SLUGS[$index]}; frames=${FRAMES[$index]}; fixture=$FIXTURE_ROOT/$slug/oracle.safetensors
  fixture_sha=${FIXTURE_SHAS[$index]}; shape_dir=$CAMPAIGN_OUT/fresh-shapes/$slug
  mkdir "$shape_dir"; shape_table=$shape_dir/results.tsv; : >"$shape_table"
  for session in 1 2 3 4 5; do
    directory=$shape_dir/session-$session; mkdir "$directory"; cp -a "$current_cache" "$directory/cache"
    wait_idle
    if run_monitored "$directory" env -u CUDA_VISIBLE_DEVICES WGPU_BACKEND=vulkan RUST_LOG=warn \
      "$CAMPAIGN_OUT/build/validate_v4_precision" "${VALIDATOR_COMMON[@]}" \
      --fixture "$fixture" --fixture-sha256 "$fixture_sha" \
      --cubecl-cache-dir "$directory/cache" --repeats 1; then
      fresh_status=0
    else
      fresh_status=$?
    fi
    latent_metrics=$(sed -n 's/^final_patched_latent\[1\]: //p' "$directory/stdout.log")
    waveform_metrics=$(sed -n 's/^raw_decoded_waveform\[1\]: //p' "$directory/stdout.log")
    latent_max_abs=$(sed -n 's/^final_patched_latent\[1\].*max_abs=\([^ ]*\).*/\1/p' "$directory/stdout.log")
    latent_mean_abs=$(sed -n 's/^final_patched_latent\[1\].*mean_abs=\([^ ]*\).*/\1/p' "$directory/stdout.log")
    latent_rmse=$(sed -n 's/^final_patched_latent\[1\].*rmse=\([^ ]*\).*/\1/p' "$directory/stdout.log")
    latent_snr=$(sed -n 's/^final_patched_latent\[1\].*snr_db=\([^ ]*\).*/\1/p' "$directory/stdout.log")
    latent_cosine=$(sed -n 's/^final_patched_latent\[1\].*cosine=\([^ ]*\).*/\1/p' "$directory/stdout.log")
    waveform_max_abs=$(sed -n 's/^raw_decoded_waveform\[1\].*max_abs=\([^ ]*\).*/\1/p' "$directory/stdout.log")
    waveform_mean_abs=$(sed -n 's/^raw_decoded_waveform\[1\].*mean_abs=\([^ ]*\).*/\1/p' "$directory/stdout.log")
    waveform_rmse=$(sed -n 's/^raw_decoded_waveform\[1\].*rmse=\([^ ]*\).*/\1/p' "$directory/stdout.log")
    waveform_snr=$(sed -n 's/^raw_decoded_waveform\[1\].*snr_db=\([^ ]*\).*/\1/p' "$directory/stdout.log")
    waveform_cosine=$(sed -n 's/^raw_decoded_waveform\[1\].*cosine=\([^ ]*\).*/\1/p' "$directory/stdout.log")
    latent_hash=$(sed -n 's/^repeat_tensor_sha256 name=final_patched_latent repeat=1 .*sha256=\([0-9a-f]*\)$/\1/p' "$directory/stdout.log")
    waveform_hash=$(sed -n 's/^repeat_tensor_sha256 name=raw_decoded_waveform repeat=1 .*sha256=\([0-9a-f]*\)$/\1/p' "$directory/stdout.log")
    rf_device=$(sed -n 's/^rf_repeat=1\/1 sample_device_complete_s=\([^ ]*\).*/\1/p' "$directory/stdout.log")
    codec_device=$(sed -n 's/^codec_repeat=1\/1 decode_device_complete_s=\([^ ]*\).*/\1/p' "$directory/stdout.log")
    if [[ $fresh_status == 0 ]]; then
      [[ -n $latent_metrics && -n $waveform_metrics && -n $latent_snr && -n $waveform_snr \
        && ${#latent_hash} == 64 && ${#waveform_hash} == 64 \
        && -n $rf_device && -n $codec_device ]] || die "incomplete PASS evidence: $slug session $session"
      total_device=$(awk -v rf="$rf_device" -v codec="$codec_device" 'BEGIN { printf "%.9f", rf + codec }')
      status_label=PASS; printf 'pass\n' >"$directory/PASS"
    elif [[ $fresh_status == 1 ]] && grep -Eq '^Error: (final_patched_latent|raw_decoded_waveform)' "$directory/stderr.log"; then
      status_label=EXPECTED_ACCURACY_FAILURE; total_device=999999
      printf 'expected_accuracy_failure\n' >"$directory/EXPECTED_FAILURE"
    else
      die "fresh shape failed outside numerical gate: $slug session $session status $fresh_status"
    fi
    target_status=NOT_EVALUATED
    if [[ $fresh_status == 0 ]]; then
      if awk -v snr="$waveform_snr" 'BEGIN { exit !(snr >= 85.0) }'; then
        target_status=TARGET_PASS
      else
        target_status=TARGET_WARNING
      fi
    fi
    jq -n --argjson session "$session" --arg status "$status_label" --arg target_status "$target_status" \
      --arg latent_max_abs "${latent_max_abs:-}" --arg latent_mean_abs "${latent_mean_abs:-}" \
      --arg latent_rmse "${latent_rmse:-}" --arg latent_snr "${latent_snr:-}" --arg latent_cosine "${latent_cosine:-}" \
      --arg waveform_max_abs "${waveform_max_abs:-}" --arg waveform_mean_abs "${waveform_mean_abs:-}" \
      --arg waveform_rmse "${waveform_rmse:-}" --arg waveform_snr "${waveform_snr:-}" --arg waveform_cosine "${waveform_cosine:-}" \
      --arg latent_hash "${latent_hash:-}" --arg waveform_hash "${waveform_hash:-}" \
      --arg rf "${rf_device:-}" --arg codec "${codec_device:-}" --arg total "$total_device" \
      '{session:$session,status:$status,target_status:$target_status,
        latent:{max_abs:$latent_max_abs,mean_abs:$latent_mean_abs,rmse:$latent_rmse,snr_db:$latent_snr,cosine:$latent_cosine},
        waveform:{max_abs:$waveform_max_abs,mean_abs:$waveform_mean_abs,rmse:$waveform_rmse,snr_db:$waveform_snr,cosine:$waveform_cosine},
        latent_sha256:$latent_hash,waveform_sha256:$waveform_hash,rf_device_s:$rf,codec_device_s:$codec,total_device_s:$total}' \
      >"$directory/result.json"
    printf '%s\t%s\t%s\n' "$session" "$status_label" "$total_device" >>"$shape_table"
  done
  selected_session=$(awk -F '\t' '$2 == "PASS" {print $1, $3}' "$shape_table" | sort -k2,2n | awk 'NR == 1 {print $1}')
  [[ -n $selected_session ]] || die "no accuracy-approved fresh selection among five sessions: $slug"
  printf '%s\n' "$selected_session" >"$shape_dir/selected-session.txt"
  selected_dir=$shape_dir/session-$selected_session
  current_cache=$selected_dir/cache
  jq -n --arg slug "$slug" --arg fixture "$fixture_sha" --argjson frames "$frames" \
    --argjson latent "$(jq '.latent | with_entries(.value |= tonumber)' "$selected_dir/result.json")" \
    --argjson waveform "$(jq '.waveform | with_entries(.value |= tonumber)' "$selected_dir/result.json")" \
    --arg latent_hash "$(jq -r '.latent_sha256' "$selected_dir/result.json")" \
    --arg waveform_hash "$(jq -r '.waveform_sha256' "$selected_dir/result.json")" \
    '{slug:$slug,fixture_sha256:$fixture,latent_frames:$frames,latent:$latent,
      waveform:$waveform,latent_sha256:$latent_hash,waveform_sha256:$waveform_hash}' \
    >"$shape_dir/evidence.json"
  jq -s '.' "$shape_dir"/session-*/result.json >"$shape_dir/results.json"
done
jq -s '{cases:.}' "$CAMPAIGN_OUT"/fresh-shapes/s*/evidence.json >"$CAMPAIGN_OUT/approved/accuracy-evidence.json"

CURRENT_PHASE=seal-six-length-cache
cp -a "$current_cache" "$CAMPAIGN_OUT/approved/cache"
"$CAMPAIGN_OUT/build/approve_v4_autotune" seal \
  --policy "$CAMPAIGN_OUT/build/accuracy-policy.json" --identity "$CAMPAIGN_OUT/environment/runtime-identity.json" \
  --evidence "$CAMPAIGN_OUT/approved/accuracy-evidence.json" --cache-root "$CAMPAIGN_OUT/approved/cache" \
  --output-manifest "$CAMPAIGN_OUT/approved/cache-manifest.json" >"$CAMPAIGN_OUT/approved/seal.stdout.log" 2>"$CAMPAIGN_OUT/approved/seal.stderr.log"
"$CAMPAIGN_OUT/build/approve_v4_autotune" verify \
  --manifest "$CAMPAIGN_OUT/approved/cache-manifest.json" --identity "$CAMPAIGN_OUT/environment/runtime-identity.json" \
  --cache-root "$CAMPAIGN_OUT/approved/cache" --receipt "$CAMPAIGN_OUT/approved/verification.json" \
  >"$CAMPAIGN_OUT/approved/verify.stdout.log" 2>"$CAMPAIGN_OUT/approved/verify.stderr.log"

CURRENT_PHASE=restored-six-lengths
cp -a "$CAMPAIGN_OUT/approved/cache" "$CAMPAIGN_OUT/restored/cache"
for index in "${!SLUGS[@]}"; do
  slug=${SLUGS[$index]}; fixture=$FIXTURE_ROOT/$slug/oracle.safetensors; fixture_sha=${FIXTURE_SHAS[$index]}
  directory=$CAMPAIGN_OUT/restored/$slug; mkdir "$directory"; wait_idle
  run_monitored "$directory" env -u CUDA_VISIBLE_DEVICES WGPU_BACKEND=vulkan RUST_LOG=warn \
    "$CAMPAIGN_OUT/build/validate_v4_precision" "${VALIDATOR_COMMON[@]}" \
    --fixture "$fixture" --fixture-sha256 "$fixture_sha" --cubecl-cache-dir "$CAMPAIGN_OUT/restored/cache" --repeats 2 \
    || die "restored/process-warm accuracy failed without retry: $slug"
  approved_latent=$(jq -r '.latent_sha256' "$CAMPAIGN_OUT/fresh-shapes/$slug/evidence.json")
  approved_waveform=$(jq -r '.waveform_sha256' "$CAMPAIGN_OUT/fresh-shapes/$slug/evidence.json")
  [[ $(grep -Fc "sha256=$approved_latent" "$directory/stdout.log") == 2 ]] || die "restored latent determinism failed: $slug"
  [[ $(grep -Fc "sha256=$approved_waveform" "$directory/stdout.log") == 2 ]] || die "restored waveform determinism failed: $slug"
  printf 'pass\n' >"$directory/PASS"
done
"$CAMPAIGN_OUT/build/approve_v4_autotune" verify \
  --manifest "$CAMPAIGN_OUT/approved/cache-manifest.json" --identity "$CAMPAIGN_OUT/environment/runtime-identity.json" \
  --cache-root "$CAMPAIGN_OUT/restored/cache" --receipt "$CAMPAIGN_OUT/restored/post-run-verification.json" \
  >"$CAMPAIGN_OUT/restored/verify.stdout.log" 2>"$CAMPAIGN_OUT/restored/verify.stderr.log"

jq -n --slurpfile evidence "$CAMPAIGN_OUT/approved/accuracy-evidence.json" \
  --argjson selections "$(jq '.selections|length' "$CAMPAIGN_OUT/approved/cache-manifest.json")" \
  '{schema_version:2,status:"PASS",fresh_shapes:$evidence[0].cases,
    approved_selection_count:$selections,restored:{all_six_pass:true,repeats_per_length:2,deterministic:true},
    accuracy_gate:{waveform_hard_min_snr_db:80.0,waveform_target_snr_db:85.0},
    automatic_retries:0,old_measurements_pooled:false}' >"$CAMPAIGN_OUT/summary.json"
CURRENT_PHASE=complete
wait_idle
COMPLETE=1
seal COMPLETE
printf 'six_length_approved_autotune_complete=%s\n' "$CAMPAIGN_OUT"
