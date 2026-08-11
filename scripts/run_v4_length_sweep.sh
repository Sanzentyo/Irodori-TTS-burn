#!/usr/bin/env bash
# Strict-FP32 multi-length measurement campaign. Each length gets an A/B
# cross-process oracle and a complete fresh-process PyTorch/WGPU stage campaign.

set -Eeuo pipefail
IFS=$'\n\t'

readonly FORMAT="irodori-v4-length-sweep-v1"
readonly COMPLETE_FORMAT="irodori-v4-length-sweep-complete-v1"
readonly CPU_SET="6-11,18-23"
readonly CUDA_NVML_INDEX=1
readonly EXPECTED_GPU_PCI="00000000:07:00.0"
readonly MAX_IDLE_MEMORY_MIB=128
readonly GLOBAL_LOCK_PATH="/tmp/irodori-v4-post18-gpu1.lock"
LENGTHS=("0.5" "1" "2" "4" "8")
SLUGS=("s0p5" "s1" "s2" "s4" "s8")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
UPSTREAM_ROOT="$(cd -- "$REPOSITORY_ROOT/../Irodori-TTS" && pwd -P)"
EXPORTER="$SCRIPT_DIR/export_v4_precision_oracle.py"
STAGE_RUNNER="$SCRIPT_DIR/run_v4_same_precision_stage_ab.sh"
SOURCE_FIXTURE="/tmp/irodori-v4-e2e-oracle.safetensors"
OUTPUT_DIR="/tmp/irodori-v4-length-sweep-20260811"
LENGTHS_JSON=""
DRY_RUN=0
SELF_TEST=0
RUN_STARTED=0
RUN_COMPLETE=0
CURRENT_PHASE="preflight"

say() { printf '[length-sweep] %s\n' "$1"; }
die() { printf 'ERROR: %s\n' "$1" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: scripts/run_v4_length_sweep.sh [OPTIONS]
  --output-dir PATH  Fresh output root
  --lengths-json PATH
                     Validated dynamic length specification; defaults to
                     0.5, 1, 2, 4, and 8 seconds
  --dry-run          Print the protocol without build/model/GPU work
  --self-test        Run CPU-only manifest/aggregation tests
  -h, --help         Show this help

There is no retry, resume, overwrite, or performance-filtering mode.
EOF
}

while (($#)); do
    case "$1" in
        --output-dir) (($# >= 2)) || die "--output-dir requires a value"; OUTPUT_DIR="$2"; shift 2 ;;
        --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
        --lengths-json) (($# >= 2)) || die "--lengths-json requires a value"; LENGTHS_JSON="$2"; shift 2 ;;
        --lengths-json=*) LENGTHS_JSON="${1#*=}"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --self-test) SELF_TEST=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done
((DRY_RUN + SELF_TEST <= 1)) || die "--dry-run and --self-test are mutually exclusive"
[[ -n "$OUTPUT_DIR" ]] || die "--output-dir must not be empty"
[[ "$OUTPUT_DIR" == /* ]] || OUTPUT_DIR="$PWD/$OUTPUT_DIR"
OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"

sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }
require_absent() { [[ ! -e "$1" && ! -L "$1" ]] || die "refusing existing path: $1"; }
require_file() { [[ -f "$1" && ! -L "$1" && -s "$1" ]] || die "unsafe or empty file: $1"; }

load_length_spec() {
    local path="$1"
    require_file "$path"
    jq -e '
      .format == "irodori-v4-length-spec-v1" and
      (.lengths | type == "array" and length > 0 and length <= 8) and
      ([.lengths[].slug] | length == (unique | length)) and
      ([.lengths[].seconds] | length == (unique | length)) and
      (.lengths | all(.[];
        . as $row |
        ($row.slug | type == "string" and test("^[a-z0-9][a-z0-9_-]*$")) and
        ($row.seconds | type == "number") and $row.seconds > 0 and $row.seconds <= 30 and
        ($row.target_samples | type == "number") and
          $row.target_samples == ($row.seconds * 48000 | floor) and
        ($row.latent_steps | type == "number") and
          $row.latent_steps == (($row.target_samples + 1919) / 1920 | floor) and
        ($row.decoded_samples | type == "number") and
          $row.decoded_samples == ($row.latent_steps * 1920)))
    ' "$path" >/dev/null || die "invalid length specification: $path"
    mapfile -t LENGTHS < <(jq -er '.lengths[].seconds | tostring' "$path")
    mapfile -t SLUGS < <(jq -er '.lengths[].slug' "$path")
    ((${#LENGTHS[@]} == ${#SLUGS[@]} && ${#LENGTHS[@]} > 0)) || die "length specification table mismatch"
}

emit_source_inventory() {
    (
        cd "$REPOSITORY_ROOT"
        {
            printf '%s\0' Cargo.toml Cargo.lock
            find src -type f \( -name '*.rs' -o -name '*.wgsl' \) -print0
            printf '%s\0' \
                scripts/export_v4_precision_oracle.py \
                scripts/bench_python_e2e_precision.py \
                scripts/run_v4_same_precision_stage_ab.sh \
                scripts/run_v4_length_sweep.sh
        } | LC_ALL=C sort -z | xargs -0 sha256sum --
    )
}

verify_source_inventory() {
    local expected current
    expected="$(awk 'NR==1 {print $1}' "$OUTPUT_DIR/source-inventory.sha256")"
    [[ "$(sha256_file "$OUTPUT_DIR/source-sha256.txt")" == "$expected" ]] || die "source inventory record changed"
    current="$(emit_source_inventory | sha256sum | awk '{print $1}')"
    [[ "$current" == "$expected" ]] || die "source changed during length sweep"
}

gpu_idle_sample() {
    local row processes
    row="$(nvidia-smi --id="$CUDA_NVML_INDEX" --query-gpu=pci.bus_id,memory.used,utilization.gpu --format=csv,noheader,nounits)" || die "GPU query failed"
    [[ "$(grep -c . <<<"$row")" == 1 ]] || die "GPU identity query returned multiple rows"
    IFS=',' read -r pci memory utilization <<<"$row"
    pci="${pci//[[:space:]]/}"; memory="${memory//[[:space:]]/}"; utilization="${utilization//[[:space:]]/}"
    [[ "${pci^^}" == "$EXPECTED_GPU_PCI" ]] || die "GPU PCI mismatch: $pci"
    [[ "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || die "invalid GPU telemetry"
    ((memory <= MAX_IDLE_MEMORY_MIB && utilization == 0)) || return 1
    processes="$(nvidia-smi --id="$CUDA_NVML_INDEX" --query-compute-apps=pid,process_name --format=csv,noheader,nounits)" || die "compute-process query failed"
    [[ -z "$processes" ]] || return 1
}

await_gpu_idle() {
    local attempt quiet=0
    for attempt in {1..15}; do
        if gpu_idle_sample; then
            ((quiet += 1))
            if ((quiet >= 2)); then
                return 0
            fi
        else
            quiet=0
        fi
        sleep 1
    done
    die "GPU did not provide two consecutive idle samples within 15 seconds"
}

run_export() {
    local seconds="$1" slug="$2" label="$3"
    local root="$OUTPUT_DIR/oracles/$slug"
    local output="$root/oracle-$label.safetensors"
    local manifest="$root/oracle-$label.json" wav="$root/oracle-$label.wav" log="$root/oracle-$label.log"
    require_absent "$output"; require_absent "$manifest"; require_absent "$wav"; require_absent "$log"
    CURRENT_PHASE="oracle_${slug}_${label}"
    (
        exec 9>>"$GLOBAL_LOCK_PATH"
        flock -n 9 || die "GPU1 global lock is busy"
        await_gpu_idle
        env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$CUDA_NVML_INDEX" \
            UV_NO_PROGRESS=1 PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 \
            TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
            taskset -c "$CPU_SET" timeout --signal=TERM --kill-after=5s 600s \
            uv run --directory "$UPSTREAM_ROOT" --extra cu128 --frozen --no-sync \
            python "$EXPORTER" --precision fp32 --seconds "$seconds" \
            --upstream "$UPSTREAM_ROOT" --source-fixture "$SOURCE_FIXTURE" \
            --output "$output" --manifest-out "$manifest" --verification-wav "$wav" \
            --model-device cuda:0 --codec-device cuda:0 >"$log" 2>&1
    ) || die "oracle export failed without retry: $slug/$label"
    require_file "$output"; require_file "$manifest"; require_file "$wav"; require_file "$log"
}

gate_oracle_pair() {
    local seconds="$1" slug="$2"
    local root="$OUTPUT_DIR/oracles/$slug"
    local a="$root/oracle-A.safetensors" b="$root/oracle-B.safetensors"
    local ma="$root/oracle-A.json" mb="$root/oracle-B.json"
    [[ "$(sha256_file "$a")" == "$(sha256_file "$b")" ]] || die "oracle fixtures differ across fresh processes: $slug"
    [[ "$(sha256_file "$root/oracle-A.wav")" == "$(sha256_file "$root/oracle-B.wav")" ]] || die "oracle WAVs differ: $slug"
    jq -e --argjson seconds "$seconds" '
      .format == "irodori-v4-precision-oracle-export-manifest-v1" and
      .precision == "fp32" and .length.seconds == $seconds and
      .length.target_samples == (($seconds * 48000)|floor) and
      .length.latent_steps == ((.length.target_samples + 1919) / 1920 | floor) and
      .length.decoded_samples == (.length.latent_steps * 1920)' "$ma" >/dev/null || die "oracle A length contract failed: $slug"
    jq -e --slurpfile b "$mb" '
      .precision == $b[0].precision and .length == $b[0].length and
      .noise == $b[0].noise and .outputs == $b[0].outputs and
      .tensor_manifest == $b[0].tensor_manifest' "$ma" >/dev/null || die "oracle manifests differ semantically: $slug"
    sha256sum "$root"/oracle-* >"$root/SHA256SUMS"
    (cd "$root" && sha256sum --quiet --strict --check SHA256SUMS) || die "oracle manifest self-check failed: $slug"
}

run_length_campaign() {
    local slug="$1"
    local manifest="$OUTPUT_DIR/oracles/$slug/oracle-A.json"
    local campaign="$OUTPUT_DIR/campaigns/$slug"
    require_absent "$campaign"
    CURRENT_PHASE="stage_campaign_$slug"
    bash "$STAGE_RUNNER" --oracle-manifest "$manifest" --measure-only --output-dir "$campaign" || {
        die "length stage campaign failed without retry: $slug"
    }
    require_file "$campaign/COMPLETE"; require_file "$campaign/SHA256SUMS"; require_file "$campaign/summary.json"
}

write_summary() {
    local output="$OUTPUT_DIR/summary.json"
    jq -s --arg format "$FORMAT" '
      sort_by(.length_contract.seconds) as $rows |
      {format:$format,status:"measured",precision:"fp32",
       lengths_seconds:[$rows[].length_contract.seconds],
       process_contract:{fresh_processes_per_runtime_per_length:5,warmups_per_process:2,
                         measured_per_process:10,automatic_retries:0},
       timer_contract:{primary:"device complete; output readback excluded",
                       secondary:"owned contiguous float32 CPU readback complete"},
       every_length_performance_pass:all($rows[];
         .performance.acceptance.every_wgpu_rf_below_global_python_min and
         .performance.acceptance.every_wgpu_codec_below_global_python_min and
         .performance_readback_inclusive.acceptance.every_wgpu_rf_below_global_python_min and
         .performance_readback_inclusive.acceptance.every_wgpu_codec_below_global_python_min),
       results:[$rows[] | {length_contract,
         device_complete:{python:.performance.python,wgpu:.performance.wgpu,
                          speedup_median:.performance.speedup_median,acceptance:.performance.acceptance},
         cpu_readback_inclusive:{python:.performance_readback_inclusive.python,
                                wgpu:.performance_readback_inclusive.wgpu,
                                speedup_median:.performance_readback_inclusive.speedup_median,
                                acceptance:.performance_readback_inclusive.acceptance},
         graph_disclosure,accuracy_gates,pins}]}' \
      "$OUTPUT_DIR"/campaigns/*/summary.json >"$output"
    jq -e --slurpfile spec "$OUTPUT_DIR/lengths.json" '
      .status == "measured" and
      (.results|length) == ($spec[0].lengths|length) and
      .lengths_seconds == [$spec[0].lengths[].seconds]
    ' "$output" >/dev/null || die "length sweep aggregation failed"
}

seal_tree() {
    local terminal="$1" status="$2" manifest="$OUTPUT_DIR/SHA256SUMS"
    verify_source_inventory
    printf 'format=%s\nstatus=%s\nphase=%s\nautomatic_retries=0\n' "$terminal" "$status" "$CURRENT_PHASE" >"$OUTPUT_DIR/$terminal"
    (
        cd "$OUTPUT_DIR"
        find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum --
    ) >"$manifest"
    (cd "$OUTPUT_DIR" && sha256sum --quiet --strict --check SHA256SUMS) || die "final manifest self-check failed"
    find "$OUTPUT_DIR" -type f -exec chmod 0444 -- {} +
    find "$OUTPUT_DIR" -type d -exec chmod 0555 -- {} +
}

on_exit() {
    local status=$?
    if ((status != 0 && RUN_STARTED && !RUN_COMPLETE)) && [[ -d "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]]; then
        set +e
        seal_tree FAILURE failed
    fi
    return "$status"
}

run_self_test() {
    local temp="$1"
    jq -n '{format:"irodori-v4-length-spec-v1",lengths:[
      {slug:"s0p5",seconds:0.5,target_samples:24000,latent_steps:13,decoded_samples:24960},
      {slug:"predicted",seconds:1.8,target_samples:86400,latent_steps:45,decoded_samples:86400}]}' \
      >"$temp/lengths.json"
    load_length_spec "$temp/lengths.json"
    [[ "${LENGTHS[*]}" == $'0.5\n1.8' && "${SLUGS[*]}" == $'s0p5\npredicted' ]] || die "dynamic length table self-test failed"
    mkdir -p "$temp/oracles/s0p5"
    jq -n '{format:"irodori-v4-precision-oracle-export-manifest-v1",precision:"fp32",
      artifact:{path:"/tmp/a",sha256:("a"*64)},
      length:{seconds:0.5,sample_rate:48000,hop_length:1920,target_samples:24000,
              decoded_samples:24960,latent_steps:13,patched_steps:13},
      noise:{source_fp32_sha256:("b"*64),effective_sha256:("b"*64),derivation:"canonical_tile_alternating_sign_v1"},
      outputs:{final_patched_latent_sha256:("c"*64),target_waveform_sha256:("d"*64),full_waveform_sha256:("e"*64)},tensor_manifest:{}}' \
      >"$temp/oracles/s0p5/oracle-A.json"
    jq -e '.length.latent_steps == 13 and .length.decoded_samples == 24960' "$temp/oracles/s0p5/oracle-A.json" >/dev/null
    say "self-test=passed length rounding and manifest schema"
}

main() {
    trap on_exit EXIT
    for command in awk bash chmod find flock install jq mkdir nvidia-smi realpath sha256sum sleep sort taskset timeout uv xargs; do
        require_command "$command"
    done
    require_file "$EXPORTER"; require_file "$STAGE_RUNNER"; require_file "$SOURCE_FIXTURE"
    if ((SELF_TEST)); then
        local temp
        temp="$(mktemp -d /tmp/irodori-v4-length-sweep-selftest.XXXXXXXX)"
        trap "rm -rf -- '$temp'" EXIT
        run_self_test "$temp"
        return
    fi
    if ((DRY_RUN)); then
        if [[ -n "$LENGTHS_JSON" ]]; then
            load_length_spec "$LENGTHS_JSON"
        fi
        say "DRY RUN: output=$OUTPUT_DIR lengths=${LENGTHS[*]}"
        say "per length: two fresh FP32 oracle exports, then five fresh processes/runtime with 2 warmups + 10 measured"
        say "both device-complete and owned-contiguous-float32 CPU-readback boundaries are recorded; performance never filters evidence"
        return
    fi
    if [[ -n "$LENGTHS_JSON" ]]; then
        [[ "$LENGTHS_JSON" == /* ]] || LENGTHS_JSON="$PWD/$LENGTHS_JSON"
        LENGTHS_JSON="$(realpath -e -- "$LENGTHS_JSON")"
        load_length_spec "$LENGTHS_JSON"
    fi
    require_absent "$OUTPUT_DIR"
    RUN_STARTED=1
    mkdir -p "$OUTPUT_DIR/oracles" "$OUTPUT_DIR/campaigns"
    if [[ -n "$LENGTHS_JSON" ]]; then
        install -m 0444 -- "$LENGTHS_JSON" "$OUTPUT_DIR/lengths.json"
        load_length_spec "$OUTPUT_DIR/lengths.json"
    else
        jq -n --argjson lengths "$(
          for index in "${!LENGTHS[@]}"; do
            jq -n --arg slug "${SLUGS[$index]}" --argjson seconds "${LENGTHS[$index]}" \
              '{slug:$slug,seconds:$seconds,target_samples:($seconds*48000|floor),
                latent_steps:((($seconds*48000|floor)+1919)/1920|floor),
                decoded_samples:((((($seconds*48000|floor)+1919)/1920|floor))*1920)}'
          done | jq -s .
        )" '{format:"irodori-v4-length-spec-v1",lengths:$lengths}' >"$OUTPUT_DIR/lengths.json"
        load_length_spec "$OUTPUT_DIR/lengths.json"
    fi
    emit_source_inventory >"$OUTPUT_DIR/source-sha256.txt"
    sha256sum "$OUTPUT_DIR/source-sha256.txt" >"$OUTPUT_DIR/source-inventory.sha256"
    verify_source_inventory
    local index seconds slug
    for index in "${!LENGTHS[@]}"; do
        seconds="${LENGTHS[$index]}"; slug="${SLUGS[$index]}"
        mkdir -p "$OUTPUT_DIR/oracles/$slug"
        run_export "$seconds" "$slug" A
        run_export "$seconds" "$slug" B
        gate_oracle_pair "$seconds" "$slug"
        run_length_campaign "$slug"
    done
    CURRENT_PHASE="aggregate"
    write_summary
    CURRENT_PHASE="complete"
    seal_tree COMPLETE measured
    RUN_COMPLETE=1
    say "campaign=COMPLETE summary=$OUTPUT_DIR/summary.json"
}

main
