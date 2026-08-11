#!/usr/bin/env bash
# Measure the released duration predictor and feed its exact resolved lengths
# into the same strict-FP32 RF/codec campaign used by the fixed-length sweep.

set -Eeuo pipefail
IFS=$'\n\t'

readonly FORMAT="irodori-v4-predicted-length-sweep-v1"
readonly COMPLETE_FORMAT="irodori-v4-predicted-length-sweep-complete-v1"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
DURATION_RUNNER="$SCRIPT_DIR/run_v4_duration_sweep.sh"
LENGTH_RUNNER="$SCRIPT_DIR/run_v4_length_sweep.sh"
OUTPUT_DIR="/tmp/irodori-v4-predicted-length-sweep-20260812"
DURATION_ARTIFACT=""
DRY_RUN=0
SELF_TEST=0
RUN_STARTED=0
RUN_COMPLETE=0
CURRENT_PHASE="preflight"

say() { printf '[predicted-length-sweep] %s\n' "$1"; }
die() { printf 'ERROR: %s\n' "$1" >&2; exit 1; }
require_file() { [[ -f "$1" && ! -L "$1" && -s "$1" ]] || die "unsafe or empty file: $1"; }
require_absent() { [[ ! -e "$1" && ! -L "$1" ]] || die "refusing existing path: $1"; }
sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }

usage() {
    cat <<'EOF'
Usage: scripts/run_v4_predicted_length_sweep.sh [OPTIONS]
  --output-dir PATH  Fresh output root
  --duration-artifact PATH
                     Reuse one fully frozen, manifest-verified duration sweep
                     as a downstream prerequisite; duration is not remeasured
  --dry-run          Print the protocol without build/model/GPU work
  --self-test        Run CPU-only predictor-to-length aggregation tests
  -h, --help         Show this help

The campaign first measures the duration predictor, then exports independent
strict-FP32 oracles and measures RF/codec at the exact resolved audio lengths.
It never retries, resumes, overwrites, or filters measurements.
EOF
}

while (($#)); do
    case "$1" in
        --output-dir) (($# >= 2)) || die "--output-dir requires a value"; OUTPUT_DIR="$2"; shift 2 ;;
        --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
        --duration-artifact) (($# >= 2)) || die "--duration-artifact requires a value"; DURATION_ARTIFACT="$2"; shift 2 ;;
        --duration-artifact=*) DURATION_ARTIFACT="${1#*=}"; shift ;;
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

emit_source_inventory() {
    (
        cd "$REPOSITORY_ROOT"
        {
            printf '%s\0' Cargo.toml Cargo.lock
            find src -type f \( -name '*.rs' -o -name '*.wgsl' \) -print0
            printf '%s\0' \
                examples/bench_v4_duration.rs \
                scripts/bench_python_duration.py \
                scripts/bench_python_e2e_precision.py \
                scripts/export_v4_precision_oracle.py \
                scripts/run_v4_duration_sweep.sh \
                scripts/run_v4_length_sweep.sh \
                scripts/run_v4_predicted_length_sweep.sh \
                scripts/run_v4_same_precision_stage_ab.sh
        } | LC_ALL=C sort -z | xargs -0 sha256sum --
    )
}

verify_source_inventory() {
    local expected current
    expected="$(awk 'NR==1 {print $1}' "$OUTPUT_DIR/source-inventory.sha256")"
    [[ "$(sha256_file "$OUTPUT_DIR/source-sha256.txt")" == "$expected" ]] || die "source inventory record changed"
    current="$(emit_source_inventory | sha256sum | awk '{print $1}')"
    [[ "$current" == "$expected" ]] || die "source changed during predicted-length sweep"
}

verify_child_tree() {
    local root="$1" files manifest_rows
    [[ -d "$root" && ! -L "$root" ]] || die "unsafe child directory: $root"
    require_file "$root/COMPLETE"
    require_file "$root/SHA256SUMS"
    (cd "$root" && sha256sum --quiet --strict --check SHA256SUMS) || die "child manifest failed: $root"
    [[ -z "$(find "$root" -type l -print -quit)" ]] || die "child tree contains a symlink: $root"
    [[ -z "$(find "$root" -type f ! -perm 0444 -print -quit)" ]] || die "mutable child file: $root"
    [[ -z "$(find "$root" -type d ! -perm 0555 -print -quit)" ]] || die "mutable child directory: $root"
    files="$(find "$root" -type f ! -name SHA256SUMS -printf '.'"%P\n" | wc -l)"
    manifest_rows="$(wc -l <"$root/SHA256SUMS")"
    [[ "$files" == "$manifest_rows" ]] || die "child manifest coverage mismatch: $root"
    return 0
}

write_length_spec() {
    local duration_summary="$1" output="$2"
    require_file "$duration_summary"
    jq -e '
      .format == "irodori-v4-duration-sweep-v1" and
      (.cases | length) == 4 and
      all(.cases[]; .resolved_length_equal_across_runtimes) and
      [.cases[].resolved_length.latent_frames] == [45,112,333,685] and
      [.cases[].resolved_length.target_samples] == [86400,215040,639360,1315200] and
      [.cases[].resolved_length.seconds] == [1.8,4.48,13.32,27.4]
    ' "$duration_summary" >/dev/null || die "duration summary cannot drive generation lengths"
    jq '{format:"irodori-v4-length-spec-v1",source:"released_duration_predictor",
      lengths:[.cases[] | {
        slug:.name,
        seconds:.resolved_length.seconds,
        target_samples:.resolved_length.target_samples,
        latent_steps:.resolved_length.latent_frames,
        decoded_samples:.resolved_length.target_samples,
        predictor:{text:.text,text_valid_tokens:.text_valid_tokens,
                   predicted_frames:.predicted_frames}}]}' \
      "$duration_summary" >"$output"
    require_file "$output"
}

write_summary() {
    local duration="$OUTPUT_DIR/duration/summary.json"
    local generation="$OUTPUT_DIR/generation/summary.json"
    jq -n --arg format "$FORMAT" --slurpfile duration "$duration" --slurpfile generation "$generation" '
      {format:$format,status:"measured",precision:"fp32",
       measurement_boundaries:{
         duration_primary:"pre-sync to device complete; scalar readback excluded",
         duration_secondary:"owned contiguous float32 scalar CPU readback complete",
         rf_codec_primary:"pre-sync to device complete; output readback excluded",
         rf_codec_secondary:"owned contiguous float32 CPU readback complete",
         composition:"duration and RF/codec are separate matched component intervals, not one monolithic wall interval"},
       every_resolved_length_equal_across_runtimes:all($duration[0].cases[];.resolved_length_equal_across_runtimes),
       every_generation_stage_performance_pass:$generation[0].every_length_performance_pass,
       cases:[$duration[0].cases[] as $d |
         $generation[0].results[] |
         select(.length_contract.seconds == $d.resolved_length.seconds) |
         {name:$d.name,text:$d.text,text_valid_tokens:$d.text_valid_tokens,
          predicted_frames:$d.predicted_frames,resolved_length:$d.resolved_length,
          duration_performance:{python:$d.python,wgpu:$d.wgpu,all_point_wins:$d.all_point_wins},
          generation:{length_contract,device_complete,cpu_readback_inclusive,
                      accuracy_gates,graph_disclosure,pins}}]}' >"$OUTPUT_DIR/summary.json"
    jq -e '
      .status == "measured" and (.cases|length) == 4 and
      [.cases[].name] == ["short","medium","long","very_long"] and
      [.cases[].resolved_length.seconds] == [1.8,4.48,13.32,27.4] and
      all(.cases[];.generation.length_contract.target_samples == .resolved_length.target_samples)
    ' "$OUTPUT_DIR/summary.json" >/dev/null || die "predicted-length aggregation failed"
}

seal_tree() {
    local terminal="$1" status="$2"
    verify_source_inventory
    printf 'format=%s\nstatus=%s\nphase=%s\nautomatic_retries=0\n' \
        "$COMPLETE_FORMAT" "$status" "$CURRENT_PHASE" >"$OUTPUT_DIR/$terminal"
    (
        cd "$OUTPUT_DIR"
        find . -type f ! -name SHA256SUMS -print0 | LC_ALL=C sort -z | xargs -0 sha256sum --
    ) >"$OUTPUT_DIR/SHA256SUMS"
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
    mkdir -p "$temp"
    jq -n '{format:"irodori-v4-duration-sweep-v1",cases:[
      {name:"short",text:"a",text_valid_tokens:3,predicted_frames:45.38,resolved_length_equal_across_runtimes:true,resolved_length:{latent_frames:45,target_samples:86400,seconds:1.8}},
      {name:"medium",text:"b",text_valid_tokens:12,predicted_frames:111.60,resolved_length_equal_across_runtimes:true,resolved_length:{latent_frames:112,target_samples:215040,seconds:4.48}},
      {name:"long",text:"c",text_valid_tokens:28,predicted_frames:333.44,resolved_length_equal_across_runtimes:true,resolved_length:{latent_frames:333,target_samples:639360,seconds:13.32}},
      {name:"very_long",text:"d",text_valid_tokens:61,predicted_frames:685.14,resolved_length_equal_across_runtimes:true,resolved_length:{latent_frames:685,target_samples:1315200,seconds:27.4}}]}' >"$temp/duration.json"
    write_length_spec "$temp/duration.json" "$temp/lengths.json"
    jq -e '.format == "irodori-v4-length-spec-v1" and
      [.lengths[].latent_steps] == [45,112,333,685] and
      [.lengths[].decoded_samples] == [86400,215040,639360,1315200]' \
      "$temp/lengths.json" >/dev/null || die "predictor-to-length self-test failed"
    say "self-test=passed predictor rounding, sample geometry, and length-spec schema"
}

main() {
    trap on_exit EXIT
    local command
    for command in awk bash chmod cp find grep jq mkdir realpath sha256sum sort wc xargs; do
        command -v "$command" >/dev/null 2>&1 || die "missing command: $command"
    done
    require_file "$DURATION_RUNNER"
    require_file "$LENGTH_RUNNER"
    if [[ -n "$DURATION_ARTIFACT" ]]; then
        [[ "$DURATION_ARTIFACT" == /* ]] || DURATION_ARTIFACT="$PWD/$DURATION_ARTIFACT"
        DURATION_ARTIFACT="$(realpath -e -- "$DURATION_ARTIFACT")"
        verify_child_tree "$DURATION_ARTIFACT"
    fi
    if ((SELF_TEST)); then
        local temp
        temp="$(mktemp -d /tmp/irodori-v4-predicted-length-selftest.XXXXXXXX)"
        trap "rm -rf -- '$temp'" EXIT
        run_self_test "$temp"
        return
    fi
    if ((DRY_RUN)); then
        say "DRY RUN: output=$OUTPUT_DIR"
        if [[ -n "$DURATION_ARTIFACT" ]]; then
            say "phase 1: verify and copy frozen duration prerequisite=$DURATION_ARTIFACT (no duration remeasurement)"
        else
            say "phase 1: four texts, three fresh processes/runtime, duration predictor device/readback boundaries"
        fi
        say "phase 2: exact resolved lengths -> two FP32 oracles -> five fresh processes/runtime for RF/codec"
        say "RF/codec both record device-complete and owned-contiguous-f32 CPU-readback-complete boundaries"
        say "component intervals stay separate; no performance filtering, retry, resume, or overwrite"
        return
    fi
    require_absent "$OUTPUT_DIR"
    RUN_STARTED=1
    mkdir -- "$OUTPUT_DIR"
    emit_source_inventory >"$OUTPUT_DIR/source-sha256.txt"
    sha256sum "$OUTPUT_DIR/source-sha256.txt" >"$OUTPUT_DIR/source-inventory.sha256"
    verify_source_inventory

    CURRENT_PHASE="duration_predictor"
    if [[ -n "$DURATION_ARTIFACT" ]]; then
        verify_child_tree "$DURATION_ARTIFACT"
        cp -a -- "$DURATION_ARTIFACT" "$OUTPUT_DIR/duration"
    else
        bash "$DURATION_RUNNER" --output-dir "$OUTPUT_DIR/duration" || die "duration sweep failed without retry"
    fi
    verify_child_tree "$OUTPUT_DIR/duration"

    CURRENT_PHASE="resolve_lengths"
    write_length_spec "$OUTPUT_DIR/duration/summary.json" "$OUTPUT_DIR/predicted-lengths.json"

    CURRENT_PHASE="rf_codec_generation"
    bash "$LENGTH_RUNNER" --lengths-json "$OUTPUT_DIR/predicted-lengths.json" \
        --output-dir "$OUTPUT_DIR/generation" || die "predicted-length RF/codec sweep failed without retry"
    verify_child_tree "$OUTPUT_DIR/generation"

    CURRENT_PHASE="aggregate"
    write_summary
    CURRENT_PHASE="complete"
    seal_tree COMPLETE measured
    RUN_COMPLETE=1
    say "campaign=COMPLETE summary=$OUTPUT_DIR/summary.json"
}

main
