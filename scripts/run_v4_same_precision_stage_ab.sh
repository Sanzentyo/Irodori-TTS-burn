#!/usr/bin/env bash
# Final strict-FP32 stage A/B campaign: official PyTorch/CUDA versus the
# production Rust/WGPU validator.
#
# A campaign is intentionally all-or-nothing.  It builds and freezes the
# validator after the production source tree is finalized, then launches ten
# fresh processes in a balanced alternating order.  Each process performs two
# excluded warmups followed by ten measured repetitions.  Any failed command,
# protocol mismatch, device mismatch, numerical-gate failure, or performance
# failure stops the campaign; there is no retry, resume, or overwrite mode.

set -Eeuo pipefail
IFS=$'\n\t'

readonly FORMAT="irodori-v4-same-precision-stage-run-v2"
readonly SESSION_FORMAT="irodori-v4-same-precision-stage-session-v2"
readonly SUMMARY_FORMAT="irodori-v4-same-precision-stage-summary-v2"
readonly COMPLETE_FORMAT="irodori-v4-same-precision-stage-complete-v2"

readonly CPU_SET="6-11,18-23"
readonly CUDA_NVML_INDEX=1
readonly WGPU_ADAPTER_INDEX=0
readonly EXPECTED_GPU_PCI="00000000:07:00.0"
readonly EXPECTED_GPU_NAME="NVIDIA GeForce RTX 3060 Ti"
readonly EXPECTED_GPU_MEMORY_MIB=8192
readonly EXPECTED_CUDA_TOTAL_MEMORY_MIB=7838.25
readonly MAX_IDLE_MEMORY_MIB=128
readonly MAX_IDLE_UTILIZATION_PERCENT=5
readonly POST_RUN_SETTLE_TIMEOUT_SECONDS=15
readonly POST_RUN_SETTLE_INTERVAL_SECONDS=1
readonly POST_RUN_SETTLE_QUIET_SAMPLES=2
readonly MIN_FREE_BYTES=4294967296
readonly GLOBAL_LOCK_PATH="/tmp/irodori-v4-post18-gpu1.lock"
readonly WGPU_INIT_CVD_DIAGNOSTIC_DIR="/tmp/irodori-v4-wgpu-init-gdb-diagnostic-attempt2-20260810"
readonly WGPU_INIT_CVD_DIAGNOSTIC_MANIFEST_SHA256="236ec60c4c78cafb92684e40247f3d8367e3cf4bdba7d4132810fe29fabe5867"

readonly SESSIONS_PER_RUNTIME=5
readonly WARMUP_REPEATS=2
readonly MEASURED_REPEATS=10
readonly TOTAL_REPEATS=12
readonly TASKS_MAX=32
readonly MEMORY_CONFIG="sub-slices"
readonly EXPECTED_SCHEDULE_JSON='[1065336439,1061146329,1056947831,1048559223,0]'
readonly EXPECTED_BATCHES_JSON='[2,2,1,1]'
readonly EXPECTED_CFG_JSON='[true,true,false,false]'

readonly UPSTREAM_COMMIT="9f19d9a9048099a4b978a762d0509228fe624e3f"
readonly MODEL_SHA256="5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593"
readonly PYTHON_CODEC_SHA256="db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5"
readonly MODEL_SNAPSHOT_REVISION="e4aaac4df355ff560dcd35e0dae272c3a759317b"
readonly PYTHON_CODEC_SNAPSHOT_REVISION="47376ee24834d7a05a48ebabfe3cde29b3c5e214"
readonly CONVERTED_CODEC_SHA256="4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1"
readonly SOURCE_FIXTURE_SHA256="8022b2baeed05e68dd2d335bebb10392b5817d1251e006413294ff597d363fc8"
FP32_ORACLE_SHA256="4287eeea818a53e382b0c8b13fd25a373c33022d62be2d309570778074a1b047"
EXPECTED_NOISE_SHA256="4d90263e9b10a7cb0aac17167049a1a9a69f1a8667734d46daa44d0b833802c4"
EXPECTED_PYTHON_LATENT_SHA256="882f9cd910f69f917b4b6742efd7f15b160c66ba6b8b90b8a56ddd704555cde8"
EXPECTED_PYTHON_AUDIO_SHA256="f56c81fcae6e8ca47b1320166fac03e5d2d88712a23c75b717e6d3876affab19"
readonly UPSTREAM_PYPROJECT_SHA256="a67e3494530cd9c29817507c67a496bb299a9a81e2edd4df6ffb80cf330dae71"
readonly UPSTREAM_UV_LOCK_SHA256="8175adbb9ad7ae77d1f048344343a63876e57c333b659314bcc054230b5b3e6c"

# Ten explicit FP32 accuracy metrics: five for the RF latent and five for the
# decoded waveform.  validate_v4_precision applies them to every repetition.
readonly LATENT_MAX_ABS="0.0002"
readonly LATENT_MEAN_ABS="0.00001"
readonly LATENT_RMSE="0.00002"
readonly LATENT_MIN_SNR_DB="90"
readonly LATENT_MIN_COSINE="0.99999999"
readonly WAVEFORM_MAX_ABS="0.00015"
readonly WAVEFORM_MEAN_ABS="0.000005"
readonly WAVEFORM_RMSE="0.00001"
readonly WAVEFORM_MIN_SNR_DB="85"
readonly WAVEFORM_MIN_COSINE="0.99999999"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
UPSTREAM_ROOT="$(cd -- "$REPOSITORY_ROOT/../Irodori-TTS" && pwd -P)"
USER_HOME_DIR="$(getent passwd "$(id -u)" | cut -d: -f6)"
[[ -n "$USER_HOME_DIR" && "$USER_HOME_DIR" == /* ]] || {
    printf 'ERROR: unable to determine an absolute user home directory\n' >&2
    exit 1
}

HF_HUB_ROOT="$USER_HOME_DIR/.cache/huggingface/hub"
# The upstream loaders dispatch by filename suffix, so actual workloads must
# receive the extension-bearing snapshot entries.  Every preflight and final
# audit binds each snapshot symlink to its expected immutable blob and SHA.
MODEL_REPOSITORY_ROOT="$HF_HUB_ROOT/models--Aratako--Irodori-TTS-v4-Small"
PYTHON_CODEC_REPOSITORY_ROOT="$HF_HUB_ROOT/models--Aratako--Semantic-DACVAE-Japanese-32dim"
MODEL_PATH="$MODEL_REPOSITORY_ROOT/snapshots/$MODEL_SNAPSHOT_REVISION/model.safetensors"
MODEL_BLOB_PATH="$MODEL_REPOSITORY_ROOT/blobs/$MODEL_SHA256"
PYTHON_CODEC_PATH="$PYTHON_CODEC_REPOSITORY_ROOT/snapshots/$PYTHON_CODEC_SNAPSHOT_REVISION/weights.pth"
PYTHON_CODEC_BLOB_PATH="$PYTHON_CODEC_REPOSITORY_ROOT/blobs/$PYTHON_CODEC_SHA256"
CONVERTED_CODEC_PATH="$REPOSITORY_ROOT/target/v4_dacvae_weights.safetensors"
SOURCE_FIXTURE_PATH="/tmp/irodori-v4-e2e-oracle.safetensors"
FP32_ORACLE_PATH="/tmp/irodori-v4-post18-precision-20260810/oracles/fp32-A.safetensors"
ORACLE_MANIFEST_PATH=""
BENCH_SOURCE_PATH="$SOURCE_FIXTURE_PATH"
BENCH_SOURCE_SHA256="$SOURCE_FIXTURE_SHA256"
ORACLE_MANIFEST_SHA256="legacy-two-second-oracle"
AUDIO_SECONDS="2"
TARGET_SAMPLES=96000
DECODED_SAMPLES=96000
LATENT_STEPS=50
RF_READBACK_ELEMENTS=1600
PYTHON_REQUESTED_JOINT_AXIS=822
PYTHON_EXECUTED_JOINT_AXIS=820
WGPU_COMPACTED_JOINT_AXIS=53
PYTHON_BENCH_SCRIPT="$REPOSITORY_ROOT/scripts/bench_python_e2e_precision.py"
VALIDATOR_SOURCE="$REPOSITORY_ROOT/src/bin/validate_v4_precision.rs"
TARGET_VALIDATOR_BINARY="$REPOSITORY_ROOT/target/release/validate_v4_precision"

OUTPUT_DIR="/tmp/irodori-v4-same-precision-stage-ab-20260810"
DRY_RUN=0
PREFLIGHT_ONLY=0
SELF_TEST=0
MEASURE_ONLY=0
ACTIVE_NVML_PID=""
RUN_STARTED=0
RUN_COMPLETED=0
LAST_GPU_SNAPSHOT=""
RUN_ORDINAL=0
CURRENT_PHASE="preflight"
CAMPAIGN_SOURCE_INVENTORY_SHA256="synthetic-self-test"
CAMPAIGN_VALIDATOR_SHA256="synthetic-self-test"
AGGREGATE_REQUIRE_PERFORMANCE_PASS=1
AGGREGATE_SUMMARY_STATUS="passed"

usage() {
    cat <<'EOF'
Usage: scripts/run_v4_same_precision_stage_ab.sh [OPTIONS]

Options:
  --output-dir PATH  New artifact directory (must not already exist)
  --oracle-manifest PATH
                     Export manifest for a variable-length strict FP32 oracle
  --preflight-only   Validate pins/protocol/device without build or workload
  --dry-run          Print the exact protocol and current readiness
  --self-test        Run CPU-only synthetic fail-closed gate tests
  --measure-only     Complete structurally valid measurements even if WGPU loses
  -h, --help         Show this help

There is deliberately no retry, resume, force, or overwrite option.
EOF
}

say() {
    printf '[same-fp32-stage] %s\n' "$1"
}

die() {
    printf 'ERROR: %s\n' "$1" >&2
    exit 1
}

while (($# > 0)); do
    case "$1" in
        --output-dir)
            (($# >= 2)) || die "--output-dir requires a value"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --oracle-manifest)
            (($# >= 2)) || die "--oracle-manifest requires a value"
            ORACLE_MANIFEST_PATH="$2"
            shift 2
            ;;
        --oracle-manifest=*)
            ORACLE_MANIFEST_PATH="${1#*=}"
            shift
            ;;
        --preflight-only)
            PREFLIGHT_ONLY=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --self-test)
            SELF_TEST=1
            shift
            ;;
        --measure-only)
            MEASURE_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *) die "unknown argument: $1" ;;
    esac
done

((DRY_RUN + PREFLIGHT_ONLY + SELF_TEST <= 1)) || {
    die "--dry-run, --preflight-only, and --self-test are mutually exclusive"
}
if ((MEASURE_ONLY)); then
    AGGREGATE_REQUIRE_PERFORMANCE_PASS=0
    AGGREGATE_SUMMARY_STATUS="measured"
fi
[[ -n "$OUTPUT_DIR" ]] || die "--output-dir must not be empty"
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$PWD/$OUTPUT_DIR"
fi
OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"

configure_oracle_contract() {
    local manifest artifact
    [[ -n "$ORACLE_MANIFEST_PATH" ]] || return 0
    manifest="$(realpath -m -- "$ORACLE_MANIFEST_PATH")"
    require_nonempty_file "$manifest"
    [[ ! -L "$manifest" ]] || die "oracle manifest must not be a symlink"
    jq -e '
      .format == "irodori-v4-precision-oracle-export-manifest-v1" and
      .precision == "fp32" and
      (.artifact.path | type == "string" and startswith("/")) and
      (.artifact.sha256 | test("^[0-9a-f]{64}$")) and
      (.length.seconds | type == "number" and . > 0) and
      .length.sample_rate == 48000 and .length.hop_length == 1920 and
      .length.target_samples > 0 and
      .length.decoded_samples == (.length.latent_steps * .length.hop_length) and
      .length.decoded_samples >= .length.target_samples and
      .length.patched_steps == .length.latent_steps and
      (.noise.source_fp32_sha256 | test("^[0-9a-f]{64}$")) and
      (.outputs.final_patched_latent_sha256 | test("^[0-9a-f]{64}$")) and
      (.outputs.target_waveform_sha256 | test("^[0-9a-f]{64}$")) and
      (.outputs.full_waveform_sha256 | test("^[0-9a-f]{64}$"))' \
      "$manifest" >/dev/null || die "invalid variable-length oracle manifest: $manifest"
    artifact="$(jq -r '.artifact.path' "$manifest")"
    require_nonempty_file "$artifact"
    [[ ! -L "$artifact" ]] || die "oracle artifact must not be a symlink"
    FP32_ORACLE_PATH="$artifact"
    FP32_ORACLE_SHA256="$(jq -r '.artifact.sha256' "$manifest")"
    expect_file_sha256 "variable-length strict FP32 oracle" "$FP32_ORACLE_SHA256" "$FP32_ORACLE_PATH"
    AUDIO_SECONDS="$(jq -r '.length.seconds' "$manifest")"
    TARGET_SAMPLES="$(jq -r '.length.target_samples' "$manifest")"
    DECODED_SAMPLES="$(jq -r '.length.decoded_samples' "$manifest")"
    LATENT_STEPS="$(jq -r '.length.latent_steps' "$manifest")"
    RF_READBACK_ELEMENTS=$((LATENT_STEPS * 32))
    PYTHON_REQUESTED_JOINT_AXIS=$((LATENT_STEPS + 772))
    PYTHON_EXECUTED_JOINT_AXIS=$((LATENT_STEPS + 770))
    WGPU_COMPACTED_JOINT_AXIS=$((LATENT_STEPS + 3))
    EXPECTED_NOISE_SHA256="$(jq -r '.noise.source_fp32_sha256' "$manifest")"
    EXPECTED_PYTHON_LATENT_SHA256="$(jq -r '.outputs.final_patched_latent_sha256' "$manifest")"
    EXPECTED_PYTHON_AUDIO_SHA256="$(jq -r '.outputs.target_waveform_sha256' "$manifest")"
    BENCH_SOURCE_PATH="$FP32_ORACLE_PATH"
    BENCH_SOURCE_SHA256="$FP32_ORACLE_SHA256"
    ORACLE_MANIFEST_PATH="$manifest"
    ORACLE_MANIFEST_SHA256="$(sha256_file "$manifest")"
}

cleanup_nvml() {
    if [[ -n "$ACTIVE_NVML_PID" ]]; then
        kill "$ACTIVE_NVML_PID" 2>/dev/null || true
        wait "$ACTIVE_NVML_PID" 2>/dev/null || true
        ACTIVE_NVML_PID=""
    fi
}

seal_failed_campaign() {
    local exit_status="$1" failure manifest python_processes wgpu_processes
    [[ -d "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || return 1
    failure="$OUTPUT_DIR/FAILURE"
    manifest="$OUTPUT_DIR/SHA256SUMS"
    [[ ! -e "$failure" && ! -L "$failure" && ! -e "$manifest" && ! -L "$manifest" ]] || {
        return 1
    }
    [[ -z "$(find "$OUTPUT_DIR" -type l -print -quit)" ]] || return 1
    python_processes="$(find "$OUTPUT_DIR/python" -maxdepth 1 -type f -name '*-command-wall.json' | wc -l)"
    wgpu_processes="$(find "$OUTPUT_DIR/wgpu" -maxdepth 1 -type f -name '*-command-wall.json' | wc -l)"
    {
        printf 'format=irodori-v4-same-precision-stage-failure-v1\n'
        printf 'status=failed\n'
        printf 'failed_phase=%s\n' "$CURRENT_PHASE"
        printf 'exit_status=%s\n' "$exit_status"
        printf 'python_process_invocations=%s\n' "$python_processes"
        printf 'wgpu_process_invocations=%s\n' "$wgpu_processes"
        printf 'automatic_retries=0\n'
        printf 'resume_permitted=false\n'
        printf 'output_reuse_permitted=false\n'
        printf 'failed_runner_sha256=%s\n' "$(sha256_file "$SCRIPT_DIR/run_v4_same_precision_stage_ab.sh")"
    } >"$failure" || return 1
    (
        cd "$OUTPUT_DIR" || exit 1
        find . -type f ! -name SHA256SUMS ! -name COMPLETE -print0 |
            LC_ALL=C sort -z | xargs -0 sha256sum --
    ) >"$manifest" || return 1
    (cd "$OUTPUT_DIR" && sha256sum --quiet --strict --check SHA256SUMS) || return 1
    find "$OUTPUT_DIR" -type f -exec chmod 0444 -- {} + || return 1
    find "$OUTPUT_DIR" -type d -exec chmod 0555 -- {} + || return 1
}

on_exit() {
    local status=$?
    cleanup_nvml
    if ((status != 0 && RUN_STARTED && !RUN_COMPLETED)); then
        set +e
        if ! seal_failed_campaign "$status"; then
            printf 'ERROR: failed to seal terminal campaign evidence\n' >&2
        fi
        printf 'ERROR: campaign stopped permanently; no retry/resume is permitted and COMPLETE was not written\n' >&2
    fi
    return "$status"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command is unavailable: $1"
}

require_absent() {
    [[ ! -e "$1" && ! -L "$1" ]] || die "refusing to overwrite existing path: $1"
}

require_nonempty_file() {
    [[ -s "$1" && -f "$1" && ! -L "$1" ]] || die "expected a non-empty regular file: $1"
}

sha256_file() {
    local output
    output="$(sha256sum -- "$1")"
    printf '%s\n' "${output%% *}"
}

expect_file_sha256() {
    local label="$1" expected="$2" path="$3" actual
    require_nonempty_file "$path"
    actual="$(sha256_file "$path")"
    [[ "$actual" == "$expected" ]] || {
        die "$label SHA-256 mismatch: expected $expected, got $actual ($path)"
    }
}

expect_hf_snapshot_symlink() {
    local label="$1" expected="$2" snapshot_path="$3" blob_path="$4"
    local target_before target_after canonical_blob snapshot_sha blob_sha
    [[ -L "$snapshot_path" ]] || die "$label snapshot is not a symlink: $snapshot_path"
    require_nonempty_file "$blob_path"
    canonical_blob="$(realpath -e -- "$blob_path")" || die "$label blob cannot be resolved"
    target_before="$(realpath -e -- "$snapshot_path")" || die "$label snapshot cannot be resolved"
    [[ "$target_before" == "$canonical_blob" ]] || {
        die "$label snapshot target mismatch: expected $canonical_blob, got $target_before"
    }
    blob_sha="$(sha256_file "$blob_path")"
    snapshot_sha="$(sha256_file "$snapshot_path")"
    target_after="$(realpath -e -- "$snapshot_path")" || die "$label snapshot cannot be re-resolved"
    [[ "$target_after" == "$target_before" && "$target_after" == "$canonical_blob" ]] || {
        die "$label snapshot target changed during validation"
    }
    [[ "$blob_sha" == "$expected" && "$snapshot_sha" == "$expected" ]] || {
        die "$label SHA-256 mismatch: expected $expected, blob=$blob_sha, snapshot=$snapshot_sha"
    }
}

existing_parent() {
    local path="$1" parent
    while [[ ! -e "$path" ]]; do
        parent="$(dirname -- "$path")"
        [[ "$parent" != "$path" ]] || die "cannot locate an existing parent for $1"
        path="$parent"
    done
    printf '%s\n' "$path"
}

assert_free_space() {
    local existing available
    existing="$(existing_parent "$1")"
    available="$(df -PB1 -- "$existing" | awk 'NR == 2 { print $4 }')"
    [[ "$available" =~ ^[0-9]+$ ]] || die "could not determine free bytes for $existing"
    ((available >= MIN_FREE_BYTES)) || die "free space below 4 GiB at $existing"
}

validator_protocol_ready() {
    # These are intentional source-level admission gates.  The current
    # validator historically capped repeats at ten and timed codec readback.
    # Measurement cannot begin until the minimal validator patch exposes the
    # exact protocol consumed below.
    grep -Eq '1\.\.=12|MAX_[A-Z_]*REPEATS[^\n]*12|TOTAL_REPEATS[^\n]*12' "$VALIDATOR_SOURCE" &&
        grep -F 'codec_timing_manifest={}' "$VALIDATOR_SOURCE" >/dev/null &&
        grep -F 'decode_device_complete_s' "$VALIDATOR_SOURCE" >/dev/null &&
        grep -F 'primary_includes_waveform_readback' "$VALIDATOR_SOURCE" >/dev/null &&
        grep -F 'primary_metric: "decode_device_complete_s"' "$VALIDATOR_SOURCE" >/dev/null &&
        grep -F 'cpu_readback_dtype: "float32"' "$VALIDATOR_SOURCE" >/dev/null &&
        grep -F 'cpu_readback_owned: true' "$VALIDATOR_SOURCE" >/dev/null &&
        grep -F 'cpu_readback_contiguous: true' "$VALIDATOR_SOURCE" >/dev/null &&
        grep -F 'secondary_metric: "decode_and_readback_s"' "$VALIDATOR_SOURCE" >/dev/null
}

assert_wgpu_cuda_environment_contract() {
    local runner="$SCRIPT_DIR/run_v4_same_precision_stage_ab.sh" required forbidden
    required='env -u CUDA_VISIBLE_'
    required+='DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID'
    forbidden='env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_'
    forbidden+='DEVICES='
    grep -F "$required" "$runner" >/dev/null || {
        die "WGPU command must launch with CUDA_VISIBLE_DEVICES unset"
    }
    if grep -F "$forbidden" "$runner" >/dev/null; then
        die "WGPU command still assigns an empty CUDA_VISIBLE_DEVICES value"
    fi
}

verify_wgpu_init_cvd_diagnostic() {
    expect_file_sha256 \
        "WGPU init CVD diagnostic manifest" \
        "$WGPU_INIT_CVD_DIAGNOSTIC_MANIFEST_SHA256" \
        "$WGPU_INIT_CVD_DIAGNOSTIC_DIR/SHA256SUMS"
    (cd "$WGPU_INIT_CVD_DIAGNOSTIC_DIR" && sha256sum --quiet --strict --check SHA256SUMS) || {
        die "WGPU init CVD diagnostic manifest verification failed"
    }
    grep -Fx 'outcome=initialize_wgpu_returned_with_cuda_visible_devices_unset_before_model_load' \
        "$WGPU_INIT_CVD_DIAGNOSTIC_DIR/RESULT" >/dev/null || {
        die "WGPU init CVD diagnostic outcome mismatch"
    }
    grep -Fx 'sigsegv_count=0' "$WGPU_INIT_CVD_DIAGNOSTIC_DIR/RESULT" >/dev/null || {
        die "WGPU init CVD diagnostic did not prove a zero-SIGSEGV treatment"
    }
    grep -Fx 'adapter_identity_lines=1' "$WGPU_INIT_CVD_DIAGNOSTIC_DIR/RESULT" >/dev/null || {
        die "WGPU init CVD diagnostic adapter count mismatch"
    }
    grep -Fx 'run_backend_breakpoint_hits=0' "$WGPU_INIT_CVD_DIAGNOSTIC_DIR/RESULT" >/dev/null || {
        die "WGPU init CVD diagnostic entered run_backend"
    }
    grep -Fx 'model_load_lines=0' "$WGPU_INIT_CVD_DIAGNOSTIC_DIR/RESULT" >/dev/null || {
        die "WGPU init CVD diagnostic entered model load"
    }
}

print_protocol_blocker() {
    cat <<'EOF'
[same-fp32-stage] protocol_status=BLOCKED
[same-fp32-stage] Minimal required validator patch (do not weaken the runner):
  1. accept --repeats 12 (two excluded warmups plus ten measured repeats);
  2. before every codec timer, synchronize the WGPU device;
  3. launch decode, synchronize device completion, and record decode_device_complete_s before any waveform readback;
  4. then read back and record decode_and_readback_s as a secondary diagnostic;
  5. emit codec_timing_manifest JSON with schema_version=1, clock=std::time::Instant,
     pre_start_device_sync=true, primary_includes_waveform_readback=false,
     waveform_readback_elements=96000, cpu_readback_dtype=float32,
     cpu_readback_owned=true, cpu_readback_contiguous=true,
     secondary_stops_after_readback_sync=true,
     primary_metric=decode_device_complete_s, secondary_metric=decode_and_readback_s;
  6. add CPU parser/schema tests for the 12-repeat boundary and timing manifest.
EOF
}

static_preflight() {
    local command_name
    for command_name in \
        awk bash cargo cp cut date df dirname find flock getent git grep id jq \
        mkdir mktemp nvidia-smi realpath rustc sed sha256sum sort stat taskset \
        tee uv xargs; do
        require_command "$command_name"
    done
    [[ "$REPOSITORY_ROOT" == /* && "$UPSTREAM_ROOT" == /* ]] || die "repository paths must be absolute"
    [[ "$(git -C "$UPSTREAM_ROOT" rev-parse HEAD)" == "$UPSTREAM_COMMIT" ]] || {
        die "upstream HEAD is not pinned to $UPSTREAM_COMMIT"
    }
    [[ -z "$(git -C "$UPSTREAM_ROOT" status --short --untracked-files=no)" ]] || {
        die "upstream has tracked worktree changes"
    }
    expect_file_sha256 "upstream pyproject.toml" "$UPSTREAM_PYPROJECT_SHA256" "$UPSTREAM_ROOT/pyproject.toml"
    expect_file_sha256 "upstream uv.lock" "$UPSTREAM_UV_LOCK_SHA256" "$UPSTREAM_ROOT/uv.lock"
    expect_hf_snapshot_symlink \
        "official model" "$MODEL_SHA256" "$MODEL_PATH" "$MODEL_BLOB_PATH"
    expect_hf_snapshot_symlink \
        "official Python codec" "$PYTHON_CODEC_SHA256" "$PYTHON_CODEC_PATH" "$PYTHON_CODEC_BLOB_PATH"
    expect_file_sha256 "converted Rust codec" "$CONVERTED_CODEC_SHA256" "$CONVERTED_CODEC_PATH"
    expect_file_sha256 "source noise fixture" "$SOURCE_FIXTURE_SHA256" "$SOURCE_FIXTURE_PATH"
    expect_file_sha256 "strict FP32 oracle" "$FP32_ORACLE_SHA256" "$FP32_ORACLE_PATH"
    expect_file_sha256 "Python benchmark source fixture" "$BENCH_SOURCE_SHA256" "$BENCH_SOURCE_PATH"
    if [[ -n "$ORACLE_MANIFEST_PATH" ]]; then
        expect_file_sha256 "oracle export manifest" "$ORACLE_MANIFEST_SHA256" "$ORACLE_MANIFEST_PATH"
    fi
    require_nonempty_file "$PYTHON_BENCH_SCRIPT"
    require_nonempty_file "$VALIDATOR_SOURCE"
    assert_wgpu_cuda_environment_contract
    verify_wgpu_init_cvd_diagnostic
    taskset -c "$CPU_SET" true || die "CPU affinity set is unavailable: $CPU_SET"
    assert_free_space "$REPOSITORY_ROOT"
    assert_free_space "$OUTPUT_DIR"
}

trim() {
    sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' <<<"$1"
}

capture_gpu_snapshot() {
    local snapshot rows
    snapshot="$(nvidia-smi \
        --id="$CUDA_NVML_INDEX" \
        --query-gpu=index,pci.bus_id,name,memory.used,utilization.gpu,memory.total \
        --format=csv,noheader,nounits)" || die "nvidia-smi GPU query failed"
    rows="$(grep -c . <<<"$snapshot" || true)"
    [[ "$rows" == "1" ]] || die "expected one GPU snapshot row, got $rows"
    printf '%s\n' "$snapshot"
}

check_gpu_idle_sample() {
    local allow_transient="$1" snapshot gpu_index gpu_pci gpu_name memory_used utilization memory_total processes
    snapshot="$(capture_gpu_snapshot)"
    IFS=',' read -r gpu_index gpu_pci gpu_name memory_used utilization memory_total <<<"$snapshot"
    gpu_index="$(trim "$gpu_index")"
    gpu_pci="$(trim "$gpu_pci")"
    gpu_name="$(trim "$gpu_name")"
    memory_used="$(trim "$memory_used")"
    utilization="$(trim "$utilization")"
    memory_total="$(trim "$memory_total")"
    [[ "$gpu_index" == "$CUDA_NVML_INDEX" ]] || die "unexpected NVML index: $gpu_index"
    [[ "${gpu_pci^^}" == "$EXPECTED_GPU_PCI" ]] || die "unexpected GPU PCI identity: $gpu_pci"
    [[ "$gpu_name" == "$EXPECTED_GPU_NAME" ]] || die "unexpected GPU name: $gpu_name"
    [[ "$memory_total" == "$EXPECTED_GPU_MEMORY_MIB" ]] || die "unexpected GPU memory: $memory_total MiB"
    [[ "$memory_used" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || die "non-numeric GPU telemetry"
    ((memory_used <= MAX_IDLE_MEMORY_MIB)) || die "GPU $CUDA_NVML_INDEX is busy: memory.used=$memory_used MiB"
    processes="$(nvidia-smi \
        --id="$CUDA_NVML_INDEX" \
        --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits)" || die "nvidia-smi process query failed"
    [[ -z "$processes" ]] || die "GPU $CUDA_NVML_INDEX has compute processes: $processes"
    LAST_GPU_SNAPSHOT="$snapshot"
    if ((utilization > MAX_IDLE_UTILIZATION_PERCENT)); then
        ((allow_transient)) || die "GPU $CUDA_NVML_INDEX is busy: utilization=$utilization%"
        return 1
    fi
}

assert_gpu_idle() {
    local snapshot_path="${1:-}"
    check_gpu_idle_sample 0
    if [[ -n "$snapshot_path" ]]; then
        require_absent "$snapshot_path"
        printf '%s\n' "$LAST_GPU_SNAPSHOT" >"$snapshot_path"
    fi
}

await_post_run_gpu_settle() {
    local label="$1" log_path="$2" started=$SECONDS quiet=0 samples=0
    require_absent "$log_path"
    printf 'timestamp_jst,gpu_snapshot\n' >"$log_path"
    while :; do
        ((samples += 1))
        if check_gpu_idle_sample 1; then
            ((quiet += 1))
        else
            quiet=0
        fi
        printf '%s,%s\n' "$(TZ=Asia/Tokyo date --iso-8601=ns)" "$LAST_GPU_SNAPSHOT" >>"$log_path"
        if ((quiet >= POST_RUN_SETTLE_QUIET_SAMPLES)); then
            say "gpu_settled label='$label' samples=$samples retries=0"
            return
        fi
        ((SECONDS - started < POST_RUN_SETTLE_TIMEOUT_SECONDS)) || {
            die "GPU did not settle after $label; workload was not retried"
        }
        sleep "$POST_RUN_SETTLE_INTERVAL_SECONDS"
    done
}

assert_safe_lock_path() {
    if [[ -e "$GLOBAL_LOCK_PATH" || -L "$GLOBAL_LOCK_PATH" ]]; then
        [[ -f "$GLOBAL_LOCK_PATH" && ! -L "$GLOBAL_LOCK_PATH" ]] || {
            die "global lock must be a regular non-symlink file: $GLOBAL_LOCK_PATH"
        }
    fi
}

prepare_output_directory() {
    require_absent "$OUTPUT_DIR"
    mkdir -p \
        "$OUTPUT_DIR/build" "$OUTPUT_DIR/python" "$OUTPUT_DIR/wgpu" \
        "$OUTPUT_DIR/nvml" "$OUTPUT_DIR/sessions"
    local directory
    for directory in "$OUTPUT_DIR" "$OUTPUT_DIR/build" "$OUTPUT_DIR/python" \
        "$OUTPUT_DIR/wgpu" "$OUTPUT_DIR/nvml" "$OUTPUT_DIR/sessions"; do
        [[ -d "$directory" && ! -L "$directory" ]] || die "artifact directory is unsafe: $directory"
    done
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
                scripts/run_v4_length_sweep.sh \
                scripts/run_v4_same_precision_stage_ab.sh
        } | LC_ALL=C sort -z | xargs -0 sha256sum --
    )
}

write_source_inventory() {
    local inventory="$OUTPUT_DIR/build/source-sha256.txt"
    local record="$OUTPUT_DIR/build/source-inventory.sha256"
    require_absent "$inventory"
    require_absent "$record"
    emit_source_inventory >"$inventory"
    sha256sum -- "$inventory" >"$record"
}

verify_source_inventory_unchanged() {
    local inventory="$OUTPUT_DIR/build/source-sha256.txt"
    local record="$OUTPUT_DIR/build/source-inventory.sha256"
    local expected current
    require_nonempty_file "$inventory"
    require_nonempty_file "$record"
    expected="$(awk 'NR == 1 { print $1 }' "$record")"
    [[ "$(sha256_file "$inventory")" == "$expected" ]] || die "source inventory record is stale"
    current="$(emit_source_inventory | sha256sum | awk '{ print $1 }')"
    [[ "$current" == "$expected" ]] || die "source changed after final campaign build"
}

discover_libtorch_directory() {
    local candidates
    candidates="$(find "$REPOSITORY_ROOT/target/release/build" \
        -type d -path '*/torch-sys-*/out/libtorch/libtorch/lib' -print | sort)"
    [[ -n "$candidates" ]] || die "release libtorch directory was not found"
    [[ "$(grep -c . <<<"$candidates")" == "1" ]] || die "expected exactly one release libtorch directory"
    printf '%s\n' "$candidates"
}

emit_libtorch_inventory() {
    local directory="$1"
    [[ -d "$directory" && ! -L "$directory" ]] || die "libtorch directory is unsafe"
    [[ -z "$(find "$directory" \( -type l -o \( ! -type f ! -type d \) \) -print -quit)" ]] || {
        die "libtorch directory contains a symlink or special file"
    }
    (
        cd "$directory"
        find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum --
    )
}

write_libtorch_inventory() {
    local directory="$1"
    emit_libtorch_inventory "$directory" >"$OUTPUT_DIR/build/libtorch-sha256.txt"
    sha256sum -- "$OUTPUT_DIR/build/libtorch-sha256.txt" >"$OUTPUT_DIR/build/libtorch-inventory.sha256"
}

verify_libtorch_inventory_unchanged() {
    local directory expected current
    directory="$(<"$OUTPUT_DIR/build/libtorch-lib-dir.txt")"
    expected="$(awk 'NR == 1 { print $1 }' "$OUTPUT_DIR/build/libtorch-inventory.sha256")"
    [[ "$(sha256_file "$OUTPUT_DIR/build/libtorch-sha256.txt")" == "$expected" ]] || die "libtorch inventory record is stale"
    current="$(emit_libtorch_inventory "$directory" | sha256sum | awk '{ print $1 }')"
    [[ "$current" == "$expected" ]] || die "libtorch changed after final campaign build"
}

record_command() {
    local argument
    printf '# %s\n' "$(TZ=Asia/Tokyo date --iso-8601=ns)" >>"$OUTPUT_DIR/commands.sh"
    for argument in "$@"; do
        printf '%q ' "$argument" >>"$OUTPUT_DIR/commands.sh"
    done
    printf '\n' >>"$OUTPUT_DIR/commands.sh"
}

build_and_freeze_validator() {
    local build_log="$OUTPUT_DIR/build/cargo-build-release.log"
    local frozen="$OUTPUT_DIR/build/validate_v4_precision"
    local libtorch_dir
    write_source_inventory
    record_command cargo build --manifest-path "$REPOSITORY_ROOT/Cargo.toml" --release --locked \
        --features inference,codec,cli --bin validate_v4_precision
    cargo build \
        --manifest-path "$REPOSITORY_ROOT/Cargo.toml" \
        --release --locked --features inference,codec,cli \
        --bin validate_v4_precision >"$build_log" 2>&1 || {
        die "release validator build failed; no retry performed: $build_log"
    }
    require_nonempty_file "$TARGET_VALIDATOR_BINARY"
    cp -- "$TARGET_VALIDATOR_BINARY" "$frozen"
    chmod 0555 "$frozen"
    sha256sum -- "$frozen" >"$OUTPUT_DIR/build/validate_v4_precision.sha256"
    CAMPAIGN_SOURCE_INVENTORY_SHA256="$(sha256_file "$OUTPUT_DIR/build/source-sha256.txt")"
    CAMPAIGN_VALIDATOR_SHA256="$(sha256_file "$frozen")"
    libtorch_dir="$(discover_libtorch_directory)"
    printf '%s\n' "$libtorch_dir" >"$OUTPUT_DIR/build/libtorch-lib-dir.txt"
    write_libtorch_inventory "$libtorch_dir"
    env "LD_LIBRARY_PATH=$libtorch_dir" "$frozen" --help >"$OUTPUT_DIR/build/validator-help.txt"
    grep -F -- '--repeats' "$OUTPUT_DIR/build/validator-help.txt" >/dev/null || die "frozen validator lacks --repeats"
    verify_source_inventory_unchanged
    verify_libtorch_inventory_unchanged
}

verify_frozen_validator() {
    local binary="$OUTPUT_DIR/build/validate_v4_precision" expected
    require_nonempty_file "$binary"
    expected="$(awk 'NR == 1 { print $1 }' "$OUTPUT_DIR/build/validate_v4_precision.sha256")"
    [[ "$(sha256_file "$binary")" == "$expected" ]] || die "frozen validator binary changed"
}

run_with_nvml_and_wall() {
    local log_path="$1" telemetry_path="$2" timing_path="$3"
    shift 3
    [[ "$1" == "--" ]] || die "internal run wrapper error"
    shift
    local start_epoch end_epoch start_jst end_jst elapsed command_status monitor_status monitor_ready=0 attempt
    require_absent "$log_path"
    require_absent "$telemetry_path"
    require_absent "$timing_path"
    record_command "$@"
    printf '%s\n' 'timestamp,index,pci.bus_id,name,utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw' >"$telemetry_path"
    nvidia-smi \
        --id="$CUDA_NVML_INDEX" \
        --query-gpu=timestamp,index,pci.bus_id,name,utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw \
        --format=csv,noheader,nounits --loop-ms=100 >>"$telemetry_path" 2>&1 &
    ACTIVE_NVML_PID=$!
    for attempt in {1..40}; do
        if (($(grep -c . "$telemetry_path" || true) >= 2)); then
            monitor_ready=1
            break
        fi
        kill -0 "$ACTIVE_NVML_PID" 2>/dev/null || break
        sleep 0.05
    done
    ((monitor_ready)) || die "NVML sidecar did not produce a pre-command sample"
    start_jst="$(TZ=Asia/Tokyo date --iso-8601=ns)"
    start_epoch="$EPOCHREALTIME"
    set +e
    "$@" >"$log_path" 2>&1
    command_status=$?
    end_epoch="$EPOCHREALTIME"
    end_jst="$(TZ=Asia/Tokyo date --iso-8601=ns)"
    kill "$ACTIVE_NVML_PID" 2>/dev/null
    wait "$ACTIVE_NVML_PID" 2>/dev/null
    monitor_status=$?
    ACTIVE_NVML_PID=""
    set -e
    elapsed="$(awk -v start="$start_epoch" -v end="$end_epoch" 'BEGIN {
        value = end - start; if (value <= 0) exit 1; printf "%.9f", value
    }')" || die "invalid external command wall time"
    jq -n \
        --arg format "irodori-v4-external-command-wall-v1" \
        --arg start_jst "$start_jst" --arg end_jst "$end_jst" \
        --arg start_epoch "$start_epoch" --arg end_epoch "$end_epoch" \
        --argjson elapsed "$elapsed" --argjson status "$command_status" \
        '{format:$format,start_jst:$start_jst,end_jst:$end_jst,
          start_epoch_realtime:$start_epoch,end_epoch_realtime:$end_epoch,
          elapsed_seconds:$elapsed,exit_status:$status}' >"$timing_path"
    ((command_status == 0)) || die "command failed with exit $command_status; no retry: $log_path"
    ((monitor_status == 0 || monitor_status == 143)) || die "NVML sidecar failed: $telemetry_path"
}

gate_nvml_sidecar() {
    local telemetry_path="$1"
    require_nonempty_file "$telemetry_path"
    awk -F ',' \
        -v expected_index="$CUDA_NVML_INDEX" \
        -v expected_pci="$EXPECTED_GPU_PCI" \
        -v expected_name="$EXPECTED_GPU_NAME" '
        function trim(value) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", value); return value }
        NR == 1 {
            if ($0 != "timestamp,index,pci.bus_id,name,utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw") bad=1
            next
        }
        {
            rows += 1
            if (NF != 9 || trim($2) != expected_index || toupper(trim($3)) != expected_pci || trim($4) != expected_name) bad=1
            util=trim($5); memory=trim($6)
            if (util !~ /^[0-9]+([.][0-9]+)?$/ || memory !~ /^[0-9]+([.][0-9]+)?$/) { bad=1; next }
            if (util+0 > max_util) max_util=util+0
            if (memory+0 > max_memory) max_memory=memory+0
        }
        END { if (bad || rows < 1 || max_util < 1 || max_memory <= 128) exit 1 }
        ' "$telemetry_path" || die "NVML identity/activity gate failed: $telemetry_path"
}

write_nvml_summary() {
    local telemetry_path="$1" output_path="$2"
    local stats rows max_util max_memory
    stats="$(awk -F ',' '
        function trim(value) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", value); return value }
        NR == 1 { next }
        { rows += 1; util=trim($5)+0; memory=trim($6)+0;
          if (util > max_util) max_util=util; if (memory > max_memory) max_memory=memory }
        END { printf "%d %.6f %.6f", rows, max_util, max_memory }
        ' "$telemetry_path")"
    IFS=' ' read -r rows max_util max_memory <<<"$stats"
    jq -n \
        --argjson index "$CUDA_NVML_INDEX" --arg pci "$EXPECTED_GPU_PCI" \
        --arg name "$EXPECTED_GPU_NAME" --argjson rows "$rows" \
        --argjson max_util "$max_util" --argjson max_memory "$max_memory" \
        '{format:"irodori-v4-nvml-stage-session-v1",index:$index,pci_bus_id:$pci,
          name:$name,rows:$rows,activity_verified:true,
          max_utilization_percent:$max_util,max_memory_used_mib:$max_memory}' >"$output_path"
}

gate_python_work_report_filter='def work_ok:
  .schema_version == 1 and .num_steps == 4 and
  .schedule_f32_bits == [1065336439,1061146329,1056947831,1048559223,0] and
  .guidance_mode == "independent" and .enabled_cfg == ["text"] and
  .requested == {batch_rows:1,latent_sequence:$latent_steps,latent_dim:32,text_tokens:256,speaker_tokens:4,caption_tokens:512,joint_axis:$python_requested_axis} and
  .encoded == {batch_rows:1,latent_sequence:$latent_steps,latent_dim:32,text_tokens:256,speaker_tokens:2,caption_tokens:512,joint_axis:$python_executed_axis} and
  .encode_calls == 1 and
  (.context_kv_builds | length) == 2 and
  [.context_kv_builds[].ordinal] == [0,1] and
  [.context_kv_builds[].batch_rows] == [1,2] and
  all(.context_kv_builds[]; .text_tokens == 256 and .speaker_tokens == 2 and
      .caption_tokens == 512 and .context_tokens == 770 and .layers == 12) and
  .context_kv_forward_hits == 4 and .cond_mlp_batches == [2,2,1,1] and
  (.forwards | length) == 4 and [.forwards[].ordinal] == [0,1,2,3] and
  [.forwards[].batch_rows] == [2,2,1,1] and
  [.forwards[].timestep_shape] == [[2],[2],[1],[1]] and
  [.forwards[].timestep_f32_bits] == [1065336439,1061146329,1056947831,1048559223] and
  [.forwards[].cfg_active] == [true,true,false,false] and
  [.forwards[].context_kv_build_ordinal] == [1,1,0,0] and
  all(.forwards[]; .latent_sequence == $latent_steps and .latent_dim == 32 and
      .timestep_dtype == "float32" and .text_tokens == 256 and
      .speaker_tokens == 2 and .caption_tokens == 512 and .joint_axis == $python_executed_axis and
      .context_kv_layers == 12 and .output_shape == [.batch_rows,$latent_steps,32]) and
  .whole_model_forwards == 4 and .forward_batches == [2,2,1,1] and
  .effective_model_rows == 6 and .model_layers == 12 and .model_block_calls == 48;'

gate_wgpu_work_report_filter='def work_ok:
  .schema_version == 1 and .method == "euler" and .num_steps == 4 and
  .schedule_f32_bits == [1065336439,1061146329,1056947831,1048559223,0] and
  .guidance_mode == "independent" and .enabled_cfg == ["text"] and
  .requested == {batch_rows:1,latent_sequence:$latent_steps,latent_dim:32,text_tokens:256,speaker_tokens:4,caption_tokens:512,joint_axis:$python_requested_axis} and
  .compacted == {batch_rows:1,latent_sequence:$latent_steps,latent_dim:32,text_tokens:3,speaker_tokens:null,caption_tokens:null,joint_axis:$wgpu_compacted_axis} and
  .encoded == .compacted and .conditioned_text_mask_all_valid == true and
  .has_speaker_context == false and .has_caption_context == false and
  .context_kv == {enabled:true,derived_text_cfg_pair_used:true,conditional_layers:12,batched_cfg_layers:12} and
  .fixed_timestep_condition == {engine_cache_supplied:true,request_selected:true,lookup_attempts:4,lookup_hits:4,precomputed_forward_hits:4,ordinary_cond_forwards:0} and
  .model_layers == 12 and .whole_model_forwards == 4 and .model_block_calls == 48 and
  (.forwards | length) == 4 and [.forwards[].step_index] == [0,1,2,3] and
  [.forwards[].evaluation] == ["primary","primary","primary","primary"] and
  [.forwards[].timestep_f32_bits] == [1065336439,1061146329,1056947831,1048559223] and
  [.forwards[].cfg_active] == [true,true,false,false] and
  [.forwards[].lane] == ["batched_independent","batched_independent","conditional","conditional"] and
  [.forwards[].batch_rows] == [2,2,1,1] and ([.forwards[].batch_rows] | add) == 6 and
  all(.forwards[]; .latent_sequence == $latent_steps and .latent_dim == 32 and
      .text_tokens == 3 and .speaker_tokens == null and .caption_tokens == null and
      .joint_axis == $wgpu_compacted_axis and .context_kv_layers == 12 and
      .fixed_cond_lookup_attempted == true and .fixed_cond_lookup_hit == true and
      .precomputed_cond_forward_used == true);'

gate_python_result() {
    local json_path="$1"
    require_nonempty_file "$json_path"
    jq -e \
        --arg gpu "$EXPECTED_GPU_NAME" \
        --argjson memory "$EXPECTED_CUDA_TOTAL_MEMORY_MIB" \
        --arg noise "$EXPECTED_NOISE_SHA256" \
        --arg latent "$EXPECTED_PYTHON_LATENT_SHA256" \
        --arg audio "$EXPECTED_PYTHON_AUDIO_SHA256" \
        --argjson repeats "$TOTAL_REPEATS" \
        --argjson seconds "$AUDIO_SECONDS" \
        --argjson target_samples "$TARGET_SAMPLES" \
        --argjson decoded_samples "$DECODED_SAMPLES" \
        --argjson latent_steps "$LATENT_STEPS" \
        --argjson rf_readback_elements "$RF_READBACK_ELEMENTS" \
        --argjson python_requested_axis "$PYTHON_REQUESTED_JOINT_AXIS" \
        --argjson python_executed_axis "$PYTHON_EXECUTED_JOINT_AXIS" \
        --argjson wgpu_compacted_axis "$WGPU_COMPACTED_JOINT_AXIS" \
        "$gate_python_work_report_filter
        .format == \"irodori-v4-python-e2e-precision-benchmark-v1\" and
        .precision == \"fp32\" and .native_dtype == \"float32\" and
        .strict_math == true and .no_autocast == true and .repeats == \$repeats and
        .runtime_reused_across_repeats == true and
        .cuda_device_identity_verified_before_and_after == true and
        .environment.device == \$gpu and .environment.visible_device_count == 1 and
        .environment.device_index_after_visibility == 0 and
        .environment.total_memory_mib == \$memory and
        (.environment.pci_bus_id_after_visibility == 7 or
         .environment.pci_bus_id_after_visibility == \"0000:07:00.0\" or
         .environment.pci_bus_id_after_visibility == \"00000000:07:00.0\") and
        .environment.torch == \"2.10.0+cu128\" and .environment.cuda_runtime == \"12.8\" and
        .environment.effective_math.cuda_matmul_allow_tf32 == false and
        .environment.effective_math.cudnn_allow_tf32 == false and
        .environment.effective_math.float32_matmul_precision == \"highest\" and
        .environment.effective_math.autocast == false and
        .parameters.model_precision == \"fp32\" and .parameters.codec_precision == \"fp32\" and
        .parameters.seconds == \$seconds and
        .length_contract == {seconds:\$seconds,target_samples:\$target_samples,
                             latent_steps:\$latent_steps,decoded_samples:\$decoded_samples} and
        .parameters.num_steps == 4 and .parameters.cfg_guidance_mode == \"independent\" and
        .parameters.cfg_effective == {text:3,caption:3,speaker:0} and
        .parameters.context_kv_cache == true and .parameters.watermark == false and
        .noise_contract.source_tensor_sha256 == \$noise and
        .noise_contract.effective_tensor_sha256 == \$noise and
        .noise_contract.cast_count == 1 and .noise_contract.same_effective_tensor_reused == true and
        .noise_contract.total_sampler_randn_interceptions == \$repeats and
        (.repeat_results | length) == \$repeats and
        [.repeat_results[].repeat] == [range(1; \$repeats + 1)] and
        .repeat_results[0].cold == true and all(.repeat_results[1:][]; .cold == false) and
        all(.repeat_results[];
          .final_latent_dtype == \"float32\" and .final_latent_shape == [1,\$latent_steps,32] and
          .final_latent_native_sha256 == \$latent and .final_latent_f32_sha256 == \$latent and
          .audio_dtype == \"float32\" and .audio_shape == [1,\$target_samples] and
          .audio_native_sha256 == \$audio and .audio_f32_sha256 == \$audio and
          .effective_noise_native_sha256 == \$noise and
          .global_cpu_rng_unchanged == true and .global_cuda_rng_unchanged == true and
          .sampler_randn_interceptions == 1 and
          .sample_rf_probe.pre_start_device_sync == true and
          .sample_rf_probe.stop_after_cuda_event_sync == true and
          .sample_rf_probe.final_latent_readback_included == false and
          .sample_rf_probe.cpu_readback_elements == \$rf_readback_elements and
          .sample_rf_probe.cpu_readback_dtype == \"float32\" and
          .sample_rf_probe.cpu_readback_owned == true and
          .sample_rf_probe.cpu_readback_contiguous == true and
          .sample_rf_probe.secondary_includes_cpu_readback == true and
          .sample_rf_probe.secondary_stops_after_cpu_readback == true and
          .sample_rf_probe.secondary_metric == \"synchronized_wall_with_readback_seconds\" and
          .sample_rf_probe.work_report_inside_timed_region == true and
          .sample_rf_probe.primary_metric == \"synchronized_wall_seconds\" and
          .sample_rf_probe.synchronized_wall_seconds > 0 and
          .sample_rf_probe.synchronized_wall_with_readback_seconds >= .sample_rf_probe.synchronized_wall_seconds and
          .decode_latent_probe.pre_start_device_sync == true and
          .decode_latent_probe.stop_after_cuda_event_sync == true and
          .decode_latent_probe.final_latent_readback_included == false and
          .decode_latent_probe.cpu_readback_elements == \$decoded_samples and
          .decode_latent_probe.cpu_readback_dtype == \"float32\" and
          .decode_latent_probe.cpu_readback_owned == true and
          .decode_latent_probe.cpu_readback_contiguous == true and
          .decode_latent_probe.secondary_includes_cpu_readback == true and
          .decode_latent_probe.secondary_stops_after_cpu_readback == true and
          .decode_latent_probe.secondary_metric == \"synchronized_wall_with_readback_seconds\" and
          .decode_latent_probe.work_report_inside_timed_region == false and
          .decode_latent_probe.primary_metric == \"synchronized_wall_seconds\" and
          .decode_latent_probe.synchronized_wall_seconds > 0 and
          .decode_latent_probe.synchronized_wall_with_readback_seconds >= .decode_latent_probe.synchronized_wall_seconds and
          (.sampler_work_report | work_ok)) and
        .summary.all_audio_native_hashes_equal == true and
        .summary.all_audio_f32_hashes_equal == true and
        .summary.all_latent_native_hashes_equal == true and
        .summary.all_latent_f32_hashes_equal == true" \
        "$json_path" >/dev/null || die "Python session structural/work/timing gate failed: $json_path"
}

write_python_session_record() {
    local session="$1" ordinal="$2" result="$3" wall="$4" nvml="$5" output="$6"
    jq -n \
        --slurpfile result "$result" --slurpfile wall "$wall" --slurpfile nvml "$nvml" \
        --arg format "$SESSION_FORMAT" --arg runtime "python-pytorch-cuda-fp32" \
        --argjson session "$session" --argjson ordinal "$ordinal" \
        --argjson warmups "$WARMUP_REPEATS" --argjson measured "$MEASURED_REPEATS" '
        {format:$format,runtime:$runtime,session:$session,schedule_ordinal:$ordinal,
         fresh_process:true,precision:"fp32",warmup_repeats:$warmups,measured_repeats:$measured,
         primary_timers:{rf:"sample_rf_probe.synchronized_wall_seconds",
                         codec:"decode_latent_probe.synchronized_wall_seconds"},
         timer_contract:{pre_start_device_sync:true,stop_at_device_complete_sync:true,
                         primary_excludes_final_output_readback:true,
                         secondary_stops_after_cpu_readback:true},
         rf_warmup_seconds:[$result[0].repeat_results[:$warmups][].sample_rf_probe.synchronized_wall_seconds],
         codec_warmup_seconds:[$result[0].repeat_results[:$warmups][].decode_latent_probe.synchronized_wall_seconds],
         rf_measured_seconds:[$result[0].repeat_results[$warmups:][].sample_rf_probe.synchronized_wall_seconds],
         codec_measured_seconds:[$result[0].repeat_results[$warmups:][].decode_latent_probe.synchronized_wall_seconds],
         rf_warmup_readback_seconds:[$result[0].repeat_results[:$warmups][].sample_rf_probe.synchronized_wall_with_readback_seconds],
         codec_warmup_readback_seconds:[$result[0].repeat_results[:$warmups][].decode_latent_probe.synchronized_wall_with_readback_seconds],
         rf_measured_readback_seconds:[$result[0].repeat_results[$warmups:][].sample_rf_probe.synchronized_wall_with_readback_seconds],
         codec_measured_readback_seconds:[$result[0].repeat_results[$warmups:][].decode_latent_probe.synchronized_wall_with_readback_seconds],
         model_load_build_seconds:$result[0].load_wall_seconds,
         codec_load_seconds:null,
         external_process_wall_seconds:$wall[0].elapsed_seconds,
         sampler_work_report:$result[0].repeat_results[0].sampler_work_report,
         output_hashes:{latent:$result[0].summary.final_latent_f32_sha256,
                        waveform:$result[0].summary.audio_f32_sha256},
         device:$result[0].environment,nvml:$nvml[0]}' >"$output"
}

run_python_session() {
    local session="$1" ordinal="$2"
    local stem="python-s${session}"
    local result="$OUTPUT_DIR/python/$stem.json" log="$OUTPUT_DIR/python/$stem.log"
    local wall="$OUTPUT_DIR/python/$stem-command-wall.json" telemetry="$OUTPUT_DIR/nvml/$stem.csv"
    local nvml="$OUTPUT_DIR/nvml/$stem-summary.json" pre="$OUTPUT_DIR/nvml/$stem-pre.txt"
    local settle="$OUTPUT_DIR/nvml/$stem-settle.csv" record="$OUTPUT_DIR/sessions/$stem.json"
    verify_source_inventory_unchanged
    verify_frozen_validator
    verify_libtorch_inventory_unchanged
    assert_gpu_idle "$pre"
    CURRENT_PHASE="python_session_${session}_workload"
    run_with_nvml_and_wall "$log" "$telemetry" "$wall" -- \
        env -u CUBECL_WGPU_MAX_TASKS -u LD_LIBRARY_PATH \
        CUDA_DEVICE_ORDER=PCI_BUS_ID "CUDA_VISIBLE_DEVICES=$CUDA_NVML_INDEX" \
        UV_NO_PROGRESS=1 PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        HF_HUB_DISABLE_TELEMETRY=1 \
        taskset -c "$CPU_SET" \
        uv run --directory "$UPSTREAM_ROOT" --extra cu128 --frozen --no-sync \
        python "$PYTHON_BENCH_SCRIPT" \
        --precision fp32 --upstream "$UPSTREAM_ROOT" \
        --checkpoint "$MODEL_PATH" --codec "$PYTHON_CODEC_PATH" \
        --source-fixture "$BENCH_SOURCE_PATH" \
        --source-fixture-sha256 "$BENCH_SOURCE_SHA256" \
        --seconds "$AUDIO_SECONDS" \
        --model-device cuda:0 --codec-device cuda:0 \
        --repeats "$TOTAL_REPEATS" --json-out "$result"
    CURRENT_PHASE="python_session_${session}_postprocess_gate"
    gate_python_result "$result"
    gate_nvml_sidecar "$telemetry"
    write_nvml_summary "$telemetry" "$nvml"
    CURRENT_PHASE="python_session_${session}_gpu_settle"
    await_post_run_gpu_settle "Python session $session" "$settle"
    CURRENT_PHASE="python_session_${session}_record"
    write_python_session_record "$session" "$ordinal" "$result" "$wall" "$nvml" "$record"
    say "Python session=$session ordinal=$ordinal passed"
}

extract_unique_scalar() {
    local pattern="$1" path="$2" values count
    values="$(sed -n "$pattern" "$path")"
    count="$(grep -c . <<<"$values" || true)"
    [[ "$count" == "1" ]] || die "expected one scalar in $path, got $count"
    printf '%s\n' "$values"
}

gate_wgpu_timing_streams() {
    local rf_json="$1" codec_json="$2"
    jq -e \
        --argjson repeats "$TOTAL_REPEATS" \
        --argjson rf_readback_elements "$RF_READBACK_ELEMENTS" '
        length == $repeats and all(.[];
          .schema_version == 1 and .clock == "std::time::Instant" and
          .pre_start_device_sync == true and .work_report_inside_timed_region == true and
          .enqueue_return_s >= 0 and .enqueue_return_s <= .sample_device_complete_s and
          .sample_device_complete_s > 0 and
          .primary_includes_final_latent_readback == false and
          .final_latent_readback_elements == $rf_readback_elements and
          .cpu_readback_dtype == "float32" and
          .cpu_readback_owned == true and
          .cpu_readback_contiguous == true and
          .sample_and_readback_s >= .sample_device_complete_s and
          .secondary_stops_after_readback_sync == true and
          .secondary_metric == "sample_and_readback_s" and
          .primary_metric == "sample_device_complete_s")' "$rf_json" >/dev/null || {
        die "WGPU RF timing-boundary gate failed: $rf_json"
    }
    jq -e \
        --argjson repeats "$TOTAL_REPEATS" \
        --argjson decoded_samples "$DECODED_SAMPLES" '
        length == $repeats and all(.[];
          .schema_version == 1 and .clock == "std::time::Instant" and
          .pre_start_device_sync == true and .decode_device_complete_s > 0 and
          .enqueue_return_s >= 0 and .enqueue_return_s <= .decode_device_complete_s and
          .primary_includes_waveform_readback == false and
          .waveform_readback_elements == $decoded_samples and
          .cpu_readback_dtype == "float32" and
          .cpu_readback_owned == true and
          .cpu_readback_contiguous == true and
          .decode_and_readback_s >= .decode_device_complete_s and
          .secondary_stops_after_readback_sync == true and
          .secondary_metric == "decode_and_readback_s" and
          .primary_metric == "decode_device_complete_s")' "$codec_json" >/dev/null || {
        die "WGPU codec timing-boundary gate failed: $codec_json"
    }
}

gate_repeated_tensor_hashes() {
    local log_path="$1" tensor_name="$2"
    awk -v tensor="$tensor_name" -v repeats="$TOTAL_REPEATS" '
      $1 == "repeat_tensor_sha256" && $2 == "name=" tensor {
        count += 1
        if ($3 != "repeat=" count || $4 != "encoding=ieee754-f32-le" || $5 !~ /^sha256=[0-9a-f]{64}$/) bad=1
        value=substr($5,8); if (count == 1) first=value; else if (value != first) bad=1
      }
      END { if (bad || count != repeats || first == "") exit 1; print first }
    ' "$log_path"
}

gate_wgpu_log_and_extract() {
    local log_path="$1" work_json="$2" rf_json="$3" codec_json="$4"
    local count latent_hash waveform_hash
    require_nonempty_file "$log_path"
    grep -F "wgpu_adapter: index=$WGPU_ADAPTER_INDEX name=\"$EXPECTED_GPU_NAME\" backend=Vulkan device_type=DiscreteGpu tasks_max=$TASKS_MAX memory_config=$MEMORY_CONFIG" "$log_path" >/dev/null || {
        die "WGPU adapter policy gate failed: $log_path"
    }
    grep -F "execution=wgsl precision=fp32 repeats=$TOTAL_REPEATS" "$log_path" >/dev/null || {
        die "WGPU execution/repeat gate failed: $log_path"
    }
    grep -F 'acceptance_mode=enforce numerical_drift_gates=explicit structural_failures=fail-closed' "$log_path" >/dev/null || {
        die "WGPU explicit ten-metric enforcement is missing"
    }
    for prefix in rf_repeat rf_work_manifest rf_timing_manifest codec_repeat codec_timing_manifest; do
        count="$(grep -c "^${prefix}=" "$log_path" || true)"
        [[ "$count" == "$TOTAL_REPEATS" ]] || die "expected $TOTAL_REPEATS $prefix rows, got $count"
    done
    [[ "$(grep -c '^final_patched_latent\[' "$log_path" || true)" == "$TOTAL_REPEATS" ]] || die "latent metric row count mismatch"
    [[ "$(grep -c '^raw_decoded_waveform\[' "$log_path" || true)" == "$TOTAL_REPEATS" ]] || die "waveform metric row count mismatch"
    if grep -E 'Validation Error|DeviceLost|OutOfMemory|panicked at|Fatal:|ERROR' "$log_path" >/dev/null; then
        die "WGPU runtime failure text found in $log_path"
    fi
    sed -n 's/^rf_work_manifest=//p' "$log_path" | jq -s . >"$work_json"
    sed -n 's/^rf_timing_manifest=//p' "$log_path" | jq -s . >"$rf_json"
    sed -n 's/^codec_timing_manifest=//p' "$log_path" | jq -s . >"$codec_json"
    jq -e --argjson repeats "$TOTAL_REPEATS" \
        --argjson latent_steps "$LATENT_STEPS" \
        --argjson python_requested_axis "$PYTHON_REQUESTED_JOINT_AXIS" \
        --argjson python_executed_axis "$PYTHON_EXECUTED_JOINT_AXIS" \
        --argjson wgpu_compacted_axis "$WGPU_COMPACTED_JOINT_AXIS" \
        "$gate_wgpu_work_report_filter
         .[0] as \$first |
         length == \$repeats and all(.[]; work_ok) and all(.[]; . == \$first)" \
        "$work_json" >/dev/null || die "WGPU RF work-report gate failed: $work_json"
    gate_wgpu_timing_streams "$rf_json" "$codec_json"
    latent_hash="$(gate_repeated_tensor_hashes "$log_path" final_patched_latent)" || die "WGPU latent hash gate failed"
    waveform_hash="$(gate_repeated_tensor_hashes "$log_path" raw_decoded_waveform)" || die "WGPU waveform hash gate failed"
    printf '%s\t%s\n' "$latent_hash" "$waveform_hash"
}

write_wgpu_session_record() {
    local session="$1" ordinal="$2" wall="$3" nvml="$4" work="$5" rf="$6" codec="$7"
    local model_load="$8" codec_load="$9" latent_hash="${10}" waveform_hash="${11}" output="${12}"
    jq -n \
        --slurpfile wall "$wall" --slurpfile nvml "$nvml" --slurpfile work "$work" \
        --slurpfile rf "$rf" --slurpfile codec "$codec" \
        --arg format "$SESSION_FORMAT" --arg runtime "rust-wgpu-wgsl-fp32" \
        --arg latent "$latent_hash" --arg waveform "$waveform_hash" \
        --argjson session "$session" --argjson ordinal "$ordinal" \
        --argjson warmups "$WARMUP_REPEATS" --argjson measured "$MEASURED_REPEATS" \
        --argjson model_load "$model_load" --argjson codec_load "$codec_load" '
        {format:$format,runtime:$runtime,session:$session,schedule_ordinal:$ordinal,
         fresh_process:true,precision:"fp32",warmup_repeats:$warmups,measured_repeats:$measured,
         primary_timers:{rf:"sample_device_complete_s",codec:"decode_device_complete_s"},
         timer_contract:{pre_start_device_sync:true,stop_at_device_complete_sync:true,
                         primary_excludes_final_output_readback:true,
                         secondary_stops_after_cpu_readback:true},
         rf_warmup_seconds:[$rf[0][:$warmups][].sample_device_complete_s],
         codec_warmup_seconds:[$codec[0][:$warmups][].decode_device_complete_s],
         rf_measured_seconds:[$rf[0][$warmups:][].sample_device_complete_s],
         codec_measured_seconds:[$codec[0][$warmups:][].decode_device_complete_s],
         rf_warmup_readback_seconds:[$rf[0][:$warmups][].sample_and_readback_s],
         codec_warmup_readback_seconds:[$codec[0][:$warmups][].decode_and_readback_s],
         rf_measured_readback_seconds:[$rf[0][$warmups:][].sample_and_readback_s],
         codec_measured_readback_seconds:[$codec[0][$warmups:][].decode_and_readback_s],
         model_load_build_seconds:$model_load,codec_load_seconds:$codec_load,
         external_process_wall_seconds:$wall[0].elapsed_seconds,
         sampler_work_report:$work[0][0],output_hashes:{latent:$latent,waveform:$waveform},
         device:{adapter_index:0,nvml_index:1,pci_bus_id:"00000000:07:00.0",
                 name:"NVIDIA GeForce RTX 3060 Ti",backend:"Vulkan"},nvml:$nvml[0]}' >"$output"
}

run_wgpu_session() {
    local session="$1" ordinal="$2"
    local stem="wgpu-s${session}"
    local log="$OUTPUT_DIR/wgpu/$stem.log" wall="$OUTPUT_DIR/wgpu/$stem-command-wall.json"
    local telemetry="$OUTPUT_DIR/nvml/$stem.csv" nvml="$OUTPUT_DIR/nvml/$stem-summary.json"
    local pre="$OUTPUT_DIR/nvml/$stem-pre.txt" settle="$OUTPUT_DIR/nvml/$stem-settle.csv"
    local work="$OUTPUT_DIR/wgpu/$stem-work.json" rf="$OUTPUT_DIR/wgpu/$stem-rf-timing.json"
    local codec="$OUTPUT_DIR/wgpu/$stem-codec-timing.json" record="$OUTPUT_DIR/sessions/$stem.json"
    local binary="$OUTPUT_DIR/build/validate_v4_precision" libtorch_dir hashes latent_hash waveform_hash
    local model_load codec_load
    verify_source_inventory_unchanged
    verify_frozen_validator
    verify_libtorch_inventory_unchanged
    libtorch_dir="$(<"$OUTPUT_DIR/build/libtorch-lib-dir.txt")"
    assert_gpu_idle "$pre"
    CURRENT_PHASE="wgpu_session_${session}_workload"
    run_with_nvml_and_wall "$log" "$telemetry" "$wall" -- \
        env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
        CUBECL_WGPU_MAX_TASKS="$TASKS_MAX" \
        "LD_LIBRARY_PATH=$libtorch_dir" \
        taskset -c "$CPU_SET" "$binary" \
        --execution wgsl --precision fp32 \
        --fixture "$FP32_ORACLE_PATH" --fixture-sha256 "$FP32_ORACLE_SHA256" \
        --checkpoint "$MODEL_PATH" --codec-weights "$CONVERTED_CODEC_PATH" \
        --adapter-index "$WGPU_ADAPTER_INDEX" --tasks-max "$TASKS_MAX" \
        --memory-config "$MEMORY_CONFIG" --repeats "$TOTAL_REPEATS" --enforce \
        --latent-max-abs "$LATENT_MAX_ABS" --latent-mean-abs "$LATENT_MEAN_ABS" \
        --latent-rmse "$LATENT_RMSE" --latent-min-snr-db "$LATENT_MIN_SNR_DB" \
        --latent-min-cosine "$LATENT_MIN_COSINE" \
        --waveform-max-abs "$WAVEFORM_MAX_ABS" --waveform-mean-abs "$WAVEFORM_MEAN_ABS" \
        --waveform-rmse "$WAVEFORM_RMSE" --waveform-min-snr-db "$WAVEFORM_MIN_SNR_DB" \
        --waveform-min-cosine "$WAVEFORM_MIN_COSINE"
    CURRENT_PHASE="wgpu_session_${session}_postprocess_gate"
    hashes="$(gate_wgpu_log_and_extract "$log" "$work" "$rf" "$codec")"
    IFS=$'\t' read -r latent_hash waveform_hash <<<"$hashes"
    model_load="$(extract_unique_scalar 's/^model_load_build_s=\([0-9][0-9.]*\) .*/\1/p' "$log")"
    codec_load="$(extract_unique_scalar 's/^codec_load_s=\([0-9][0-9.]*\)$/\1/p' "$log")"
    gate_nvml_sidecar "$telemetry"
    write_nvml_summary "$telemetry" "$nvml"
    CURRENT_PHASE="wgpu_session_${session}_gpu_settle"
    await_post_run_gpu_settle "WGPU session $session" "$settle"
    CURRENT_PHASE="wgpu_session_${session}_record"
    write_wgpu_session_record "$session" "$ordinal" "$wall" "$nvml" "$work" "$rf" "$codec" \
        "$model_load" "$codec_load" "$latent_hash" "$waveform_hash" "$record"
    say "WGPU session=$session ordinal=$ordinal passed"
}

run_balanced_sessions() {
    local pair
    for pair in 1 2 3 4 5; do
        if ((pair % 2 == 1)); then
            ((RUN_ORDINAL += 1)); run_python_session "$pair" "$RUN_ORDINAL"
            ((RUN_ORDINAL += 1)); run_wgpu_session "$pair" "$RUN_ORDINAL"
        else
            ((RUN_ORDINAL += 1)); run_wgpu_session "$pair" "$RUN_ORDINAL"
            ((RUN_ORDINAL += 1)); run_python_session "$pair" "$RUN_ORDINAL"
        fi
    done
    ((RUN_ORDINAL == 10)) || die "internal session schedule count mismatch"
}

aggregate_sessions() {
    local output="$1"
    shift
    jq -s \
        --arg format "$SUMMARY_FORMAT" \
        --argjson warmups "$WARMUP_REPEATS" --argjson measured "$MEASURED_REPEATS" \
        --argjson sessions_per_runtime "$SESSIONS_PER_RUNTIME" \
        --argjson schedule "$EXPECTED_SCHEDULE_JSON" --argjson batches "$EXPECTED_BATCHES_JSON" \
        --argjson cfg "$EXPECTED_CFG_JSON" \
        --arg latent_max "$LATENT_MAX_ABS" --arg latent_mean "$LATENT_MEAN_ABS" \
        --arg latent_rmse "$LATENT_RMSE" --arg latent_snr "$LATENT_MIN_SNR_DB" \
        --arg latent_cos "$LATENT_MIN_COSINE" --arg wave_max "$WAVEFORM_MAX_ABS" \
        --arg wave_mean "$WAVEFORM_MEAN_ABS" --arg wave_rmse "$WAVEFORM_RMSE" \
        --arg wave_snr "$WAVEFORM_MIN_SNR_DB" --arg wave_cos "$WAVEFORM_MIN_COSINE" \
        --arg upstream_commit "$UPSTREAM_COMMIT" --arg model_sha "$MODEL_SHA256" \
        --arg python_codec_sha "$PYTHON_CODEC_SHA256" --arg rust_codec_sha "$CONVERTED_CODEC_SHA256" \
        --arg source_fixture_sha "$BENCH_SOURCE_SHA256" --arg oracle_sha "$FP32_ORACLE_SHA256" \
        --arg oracle_manifest_sha "$ORACLE_MANIFEST_SHA256" \
        --argjson seconds "$AUDIO_SECONDS" --argjson target_samples "$TARGET_SAMPLES" \
        --argjson decoded_samples "$DECODED_SAMPLES" --argjson latent_steps "$LATENT_STEPS" \
        --argjson rf_readback_elements "$RF_READBACK_ELEMENTS" \
        --argjson python_requested_axis "$PYTHON_REQUESTED_JOINT_AXIS" \
        --argjson python_executed_axis "$PYTHON_EXECUTED_JOINT_AXIS" \
        --argjson wgpu_compacted_axis "$WGPU_COMPACTED_JOINT_AXIS" \
        --arg source_inventory_sha "$CAMPAIGN_SOURCE_INVENTORY_SHA256" \
        --arg validator_sha "$CAMPAIGN_VALIDATOR_SHA256" \
        --arg summary_status "$AGGREGATE_SUMMARY_STATUS" \
        --argjson require_performance_pass "$AGGREGATE_REQUIRE_PERFORMANCE_PASS" \
        --arg wgpu_init_cvd_diagnostic_sha "$WGPU_INIT_CVD_DIAGNOSTIC_MANIFEST_SHA256" \
        --arg runner_sha "$(sha256_file "$SCRIPT_DIR/run_v4_same_precision_stage_ab.sh")" \
        --arg python_bench_sha "$(sha256_file "$PYTHON_BENCH_SCRIPT")" '
        def median:
          sort as $s | ($s|length) as $n |
          if $n == 0 then null elif ($n % 2) == 1 then $s[($n/2)|floor]
          else (($s[$n/2-1] + $s[$n/2]) / 2) end;
        sort_by(.schedule_ordinal) as $sessions |
        [$sessions[] | select(.runtime == "python-pytorch-cuda-fp32")] as $p |
        [$sessions[] | select(.runtime == "rust-wgpu-wgsl-fp32")] as $w |
        [$p[].rf_measured_seconds[]] as $prf |
        [$p[].codec_measured_seconds[]] as $pcodec |
        [$w[].rf_measured_seconds[]] as $wrf |
        [$w[].codec_measured_seconds[]] as $wcodec |
        [$p[].rf_measured_readback_seconds[]] as $prf_readback |
        [$p[].codec_measured_readback_seconds[]] as $pcodec_readback |
        [$w[].rf_measured_readback_seconds[]] as $wrf_readback |
        [$w[].codec_measured_readback_seconds[]] as $wcodec_readback |
        ($prf|min) as $prf_min | ($pcodec|min) as $pcodec_min |
        ($prf_readback|min) as $prf_readback_min |
        ($pcodec_readback|min) as $pcodec_readback_min |
        {
          format:$format,status:$summary_status,precision:"fp32",
          comparison_claim:"same precision and equal active RF semantic projection; optimized graphs differ",
          pins:{upstream_commit:$upstream_commit,model_sha256:$model_sha,
            python_codec_sha256:$python_codec_sha,rust_codec_sha256:$rust_codec_sha,
            source_fixture_sha256:$source_fixture_sha,fp32_oracle_sha256:$oracle_sha,
            oracle_export_manifest_sha256:$oracle_manifest_sha,
            source_inventory_sha256:$source_inventory_sha,validator_binary_sha256:$validator_sha,
            wgpu_init_cvd_diagnostic_manifest_sha256:$wgpu_init_cvd_diagnostic_sha,
            runner_sha256:$runner_sha,python_bench_sha256:$python_bench_sha},
          runtime_environment:{python_cuda_visible_devices:"1",wgpu_cuda_visible_devices:"unset",
            wgpu_cuda_device_order:"PCI_BUS_ID",
            rationale:"Paired init-only control: empty CUDA_VISIBLE_DEVICES reproduced a NVIDIA vkCreateDevice/libcuda SIGSEGV; unset returned before model load."},
          process_design:{sessions_per_runtime:$sessions_per_runtime,total_fresh_processes:10,
            schedule:["python","wgpu","wgpu","python","python","wgpu","wgpu","python","python","wgpu"],
            warmups_per_process:$warmups,measured_per_process:$measured,automatic_retries:0,
            aggregation:"50 measured samples per runtime/stage; no warmup pooling"},
          length_contract:{seconds:$seconds,target_samples:$target_samples,
            decoded_samples:$decoded_samples,latent_steps:$latent_steps},
          timer_contract:{primary:{rf:"pre-sync -> sampler -> device-complete sync; final latent readback excluded",
            codec:"pre-sync -> codec launch -> device-complete sync; waveform readback excluded"},
            readback_inclusive:{rf:("same pre-sync start -> " + ($rf_readback_elements|tostring) + "-element CPU tensor materialization complete"),
              codec:("same pre-sync start -> " + ($decoded_samples|tostring) + "-element CPU tensor materialization complete")},
            model_load_build:"reported separately",external_process_wall:"reported separately"},
          active_rf_work:{num_steps:4,schedule_f32_bits:$schedule,guidance_mode:"independent",
            enabled_cfg:["text"],whole_model_forwards:4,forward_batches:$batches,
            effective_model_rows:6,model_layers:12,model_block_calls:48,cfg_active:$cfg,
            equal_between_runtimes:true},
          graph_disclosure:{same_graph:false,
            python:{requested_joint_axis:$python_requested_axis,encoded_joint_axis:$python_executed_axis,forward_joint_axis:$python_executed_axis,
                    context_kv_builds:2,context_kv_build_ordinals:[1,1,0,0]},
            wgpu:{requested_joint_axis:$python_requested_axis,compacted_joint_axis:$wgpu_compacted_axis,encoded_joint_axis:$wgpu_compacted_axis,
                  forward_joint_axis:$wgpu_compacted_axis,derived_context_kv_pair:true,fixed_condition_hits:4},
            interpretation:"Only the active semantic work projection is equal; this is not a same-graph benchmark."},
          accuracy_gates:{unique_metric_count:10,applied_to_all_wgpu_repetitions:true,
            latent:{max_abs_lte:($latent_max|tonumber),mean_abs_lte:($latent_mean|tonumber),
                    rmse_lte:($latent_rmse|tonumber),snr_db_gte:($latent_snr|tonumber),
                    cosine_gte:($latent_cos|tonumber)},
            waveform:{max_abs_lte:($wave_max|tonumber),mean_abs_lte:($wave_mean|tonumber),
                      rmse_lte:($wave_rmse|tonumber),snr_db_gte:($wave_snr|tonumber),
                      cosine_gte:($wave_cos|tonumber)}},
          performance:{
            python:{rf:{samples_seconds:$prf,min_seconds:($prf|min),median_seconds:($prf|median),max_seconds:($prf|max)},
                    codec:{samples_seconds:$pcodec,min_seconds:($pcodec|min),median_seconds:($pcodec|median),max_seconds:($pcodec|max)}},
            wgpu:{rf:{samples_seconds:$wrf,min_seconds:($wrf|min),median_seconds:($wrf|median),max_seconds:($wrf|max)},
                  codec:{samples_seconds:$wcodec,min_seconds:($wcodec|min),median_seconds:($wcodec|median),max_seconds:($wcodec|max)}},
            speedup_median:{rf:(($prf|median)/($wrf|median)),codec:(($pcodec|median)/($wcodec|median))},
            acceptance:{every_wgpu_rf_below_global_python_min:all($wrf[]; . < $prf_min),
                        every_wgpu_codec_below_global_python_min:all($wcodec[]; . < $pcodec_min)}},
          performance_readback_inclusive:{
            python:{rf:{samples_seconds:$prf_readback,min_seconds:($prf_readback|min),
                        median_seconds:($prf_readback|median),max_seconds:($prf_readback|max)},
                    codec:{samples_seconds:$pcodec_readback,min_seconds:($pcodec_readback|min),
                           median_seconds:($pcodec_readback|median),max_seconds:($pcodec_readback|max)}},
            wgpu:{rf:{samples_seconds:$wrf_readback,min_seconds:($wrf_readback|min),
                      median_seconds:($wrf_readback|median),max_seconds:($wrf_readback|max)},
                  codec:{samples_seconds:$wcodec_readback,min_seconds:($wcodec_readback|min),
                         median_seconds:($wcodec_readback|median),max_seconds:($wcodec_readback|max)}},
            speedup_median:{rf:(($prf_readback|median)/($wrf_readback|median)),
                            codec:(($pcodec_readback|median)/($wcodec_readback|median))},
            acceptance:{
              every_wgpu_rf_below_global_python_min:all($wrf_readback[]; . < $prf_readback_min),
              every_wgpu_codec_below_global_python_min:all($wcodec_readback[]; . < $pcodec_readback_min)}},
          diagnostics:{python_model_load_build_seconds:[$p[].model_load_build_seconds],
            wgpu_model_load_build_seconds:[$w[].model_load_build_seconds],
            wgpu_codec_load_seconds:[$w[].codec_load_seconds],
            python_external_process_wall_seconds:[$p[].external_process_wall_seconds],
            wgpu_external_process_wall_seconds:[$w[].external_process_wall_seconds]},
          output_determinism:{
            python_latent_hashes:([$p[].output_hashes.latent]|unique),
            python_waveform_hashes:([$p[].output_hashes.waveform]|unique),
            wgpu_latent_hashes:([$w[].output_hashes.latent]|unique),
            wgpu_waveform_hashes:([$w[].output_hashes.waveform]|unique),
            all_cross_session_hash_sets_singleton:true},
          sessions:$sessions,
          report_validation:"structural_only; browser rendering unavailable in campaign environment"
        }
        | select(($p|length) == $sessions_per_runtime and ($w|length) == $sessions_per_runtime)
        | select(($prf|length) == ($sessions_per_runtime*$measured) and
                 ($pcodec|length) == ($sessions_per_runtime*$measured) and
                 ($wrf|length) == ($sessions_per_runtime*$measured) and
                 ($wcodec|length) == ($sessions_per_runtime*$measured) and
                 ($prf_readback|length) == ($sessions_per_runtime*$measured) and
                 ($pcodec_readback|length) == ($sessions_per_runtime*$measured) and
                 ($wrf_readback|length) == ($sessions_per_runtime*$measured) and
                 ($wcodec_readback|length) == ($sessions_per_runtime*$measured))
        | select(.sessions | map(.schedule_ordinal) == [1,2,3,4,5,6,7,8,9,10])
        | select(.sessions | map(.runtime) == [
            "python-pytorch-cuda-fp32","rust-wgpu-wgsl-fp32","rust-wgpu-wgsl-fp32",
            "python-pytorch-cuda-fp32","python-pytorch-cuda-fp32","rust-wgpu-wgsl-fp32",
            "rust-wgpu-wgsl-fp32","python-pytorch-cuda-fp32","python-pytorch-cuda-fp32",
            "rust-wgpu-wgsl-fp32"])
        | select(($p | map(.session)) == [1,2,3,4,5] and ($w | map(.session)) == [1,2,3,4,5])
        | select(all(.sessions[]; . as $session |
                     .format == "irodori-v4-same-precision-stage-session-v2" and
                     .fresh_process == true and .precision == "fp32" and
                     .warmup_repeats == $warmups and .measured_repeats == $measured and
                     .timer_contract.pre_start_device_sync == true and
                     .timer_contract.stop_at_device_complete_sync == true and
                     .timer_contract.primary_excludes_final_output_readback == true and
                     .timer_contract.secondary_stops_after_cpu_readback == true and
                     (.rf_warmup_seconds|length) == $warmups and
                     (.codec_warmup_seconds|length) == $warmups and
                     (.rf_measured_seconds|length) == $measured and
                     (.codec_measured_seconds|length) == $measured and
                     (.rf_warmup_readback_seconds|length) == $warmups and
                     (.codec_warmup_readback_seconds|length) == $warmups and
                     (.rf_measured_readback_seconds|length) == $measured and
                     (.codec_measured_readback_seconds|length) == $measured and
                     all(.rf_warmup_seconds[],.codec_warmup_seconds[],
                         .rf_measured_seconds[],.codec_measured_seconds[],
                         .rf_warmup_readback_seconds[],.codec_warmup_readback_seconds[],
                         .rf_measured_readback_seconds[],.codec_measured_readback_seconds[]; . > 0) and
                     all(range(0;$warmups); . as $i |
                       $session.rf_warmup_readback_seconds[$i] >= $session.rf_warmup_seconds[$i] and
                       $session.codec_warmup_readback_seconds[$i] >= $session.codec_warmup_seconds[$i]) and
                     all(range(0;$measured); . as $i |
                       $session.rf_measured_readback_seconds[$i] >= $session.rf_measured_seconds[$i] and
                       $session.codec_measured_readback_seconds[$i] >= $session.codec_measured_seconds[$i]) and
                     .model_load_build_seconds > 0 and .external_process_wall_seconds > 0 and
                     (.output_hashes.latent | test("^[0-9a-f]{64}$")) and
                     (.output_hashes.waveform | test("^[0-9a-f]{64}$"))))
        | select((.output_determinism.python_latent_hashes|length) == 1 and
                 (.output_determinism.python_waveform_hashes|length) == 1 and
                 (.output_determinism.wgpu_latent_hashes|length) == 1 and
                 (.output_determinism.wgpu_waveform_hashes|length) == 1)
        | select(($require_performance_pass == 0) or
                 (.performance.acceptance.every_wgpu_rf_below_global_python_min == true and
                  .performance.acceptance.every_wgpu_codec_below_global_python_min == true and
                  .performance.speedup_median.rf > 1 and .performance.speedup_median.codec > 1 and
                  .performance_readback_inclusive.acceptance.every_wgpu_rf_below_global_python_min == true and
                  .performance_readback_inclusive.acceptance.every_wgpu_codec_below_global_python_min == true and
                  .performance_readback_inclusive.speedup_median.rf > 1 and
                  .performance_readback_inclusive.speedup_median.codec > 1))
        ' "$@" >"$output"
    [[ -s "$output" && -f "$output" && ! -L "$output" ]]
}

write_html_report() {
    local summary="$1" output="$2"
    local generated rows readback_rows headline
    generated="$(TZ=Asia/Tokyo date --iso-8601=seconds)"
    rows="$(jq -r '
      [.performance.python.rf,.performance.wgpu.rf,.performance.python.codec,.performance.wgpu.codec]
      | ["PyTorch CUDA","WGPU","PyTorch CUDA","WGPU"] as $runtime
      | ["RF","RF","Codec","Codec"] as $stage
      | to_entries[] | "<tr><td>\($stage[.key])</td><td>\($runtime[.key])</td><td>\(.value.min_seconds*1000)</td><td>\(.value.median_seconds*1000)</td><td>\(.value.max_seconds*1000)</td></tr>"
      ' "$summary")"
    readback_rows="$(jq -r '
      [.performance_readback_inclusive.python.rf,.performance_readback_inclusive.wgpu.rf,
       .performance_readback_inclusive.python.codec,.performance_readback_inclusive.wgpu.codec]
      | ["PyTorch CUDA","WGPU","PyTorch CUDA","WGPU"] as $runtime
      | ["RF","RF","Codec","Codec"] as $stage
      | to_entries[] | "<tr><td>\($stage[.key])</td><td>\($runtime[.key])</td><td>\(.value.min_seconds*1000)</td><td>\(.value.median_seconds*1000)</td><td>\(.value.max_seconds*1000)</td></tr>"
      ' "$summary")"
    if ((AGGREGATE_REQUIRE_PERFORMANCE_PASS)); then
        headline='PASS: every measured WGPU RF and codec sample is below the corresponding global PyTorch minimum at both device-complete and CPU-readback boundaries.'
    else
        headline='MEASURED: structural, work-count, timing-boundary, accuracy, and determinism gates passed; performance results are reported without requiring WGPU to win.'
    fi
    require_absent "$output"
    {
        printf '%s\n' '<!doctype html><html lang="ja"><head><meta charset="utf-8">'
        printf '%s\n' '<meta name="viewport" content="width=device-width,initial-scale=1">'
        printf '%s\n' '<title>Irodori v4 FP32 Stage A/B</title><style>body{font:16px system-ui;max-width:980px;margin:2rem auto;padding:0 1rem;color:#17202a}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:.45rem;text-align:right}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}.pass{color:#087830;font-weight:700}code{background:#eee;padding:.1rem .25rem}</style></head><body>'
        printf '<h1>Irodori-TTS v4 FP32 stage A/B</h1><p class="pass">%s</p>\n' "$headline"
        printf '<p>Generated: %s. Validation: structural only (browser unavailable in the campaign environment).</p>\n' "$generated"
        printf '%s\n' '<table><thead><tr><th>Stage</th><th>Runtime</th><th>min ms</th><th>median ms</th><th>max ms</th></tr></thead><tbody>'
        printf '%s\n' "$rows"
        printf '%s\n' '</tbody></table>'
        printf '%s\n' '<h2>CPU readback inclusive</h2><table><thead><tr><th>Stage</th><th>Runtime</th><th>min ms</th><th>median ms</th><th>max ms</th></tr></thead><tbody>'
        printf '%s\n' "$readback_rows"
        printf '%s\n' '</tbody></table>'
        printf '<p>Both sides use FP32 at %.9g seconds (%d latent frames). RF active work is 4 Euler forwards, batches [2,2,1,1], 6 effective rows, 12 layers, and 48 block calls. Both request joint axis %d. The graphs then intentionally differ: Python encodes/forwards axis %d, while WGPU compacts/encodes/forwards axis %d.</p>\n' "$AUDIO_SECONDS" "$LATENT_STEPS" "$PYTHON_REQUESTED_JOINT_AXIS" "$PYTHON_EXECUTED_JOINT_AXIS" "$WGPU_COMPACTED_JOINT_AXIS"
        printf '<p>Primary timers stop at device completion and preserve the zero-copy production boundary. Secondary timers share the same pre-sync start and stop after CPU materialization of %d RF or %d codec float32 elements. Two warmups per fresh process are excluded; ten repetitions are measured in each of five processes per runtime.</p>\n' "$RF_READBACK_ELEMENTS" "$DECODED_SAMPLES"
        printf '%s\n' '</body></html>'
    } >"$output"
    grep -F '<!doctype html>' "$output" >/dev/null || die "HTML structural check failed"
    grep -F "Python encodes/forwards axis $PYTHON_EXECUTED_JOINT_AXIS" "$output" >/dev/null || die "HTML graph disclosure is missing"
}

write_summary() {
    local summary="$OUTPUT_DIR/summary.json"
    aggregate_sessions "$summary" "$OUTPUT_DIR/sessions/"*.json || {
        die "stage aggregation failed; a structural, timing, determinism, accuracy, or required performance gate failed"
    }
    write_html_report "$summary" "$OUTPUT_DIR/report.html"
}

write_run_metadata() {
    local metadata="$OUTPUT_DIR/run-metadata.txt"
    require_absent "$metadata"
    {
        printf 'format=%s\n' "$FORMAT"
        printf 'started_jst=%s\n' "$(TZ=Asia/Tokyo date --iso-8601=seconds)"
        printf 'repository_root=%s\n' "$REPOSITORY_ROOT"
        printf 'upstream_root=%s\n' "$UPSTREAM_ROOT"
        printf 'output_dir=%s\n' "$OUTPUT_DIR"
        printf 'cpu_set=%s\n' "$CPU_SET"
        printf 'python_cuda_visible_devices=%s\n' "$CUDA_NVML_INDEX"
        printf 'wgpu_cuda_visible_devices=unset\n'
        printf 'wgpu_cuda_device_order=PCI_BUS_ID\n'
        printf 'wgpu_init_cvd_diagnostic_manifest_sha256=%s\n' "$WGPU_INIT_CVD_DIAGNOSTIC_MANIFEST_SHA256"
        printf 'expected_gpu_pci=%s\n' "$EXPECTED_GPU_PCI"
        printf 'wgpu_adapter_index=%s\n' "$WGPU_ADAPTER_INDEX"
        printf 'precision=fp32\n'
        printf 'tf32=false\n'
        printf 'warmups_per_session=%s\n' "$WARMUP_REPEATS"
        printf 'measured_per_session=%s\n' "$MEASURED_REPEATS"
        printf 'sessions_per_runtime=%s\n' "$SESSIONS_PER_RUNTIME"
        printf 'schedule=P,W,W,P,P,W,W,P,P,W\n'
        printf 'automatic_retries=0\n'
        printf 'rf_primary=pre-sync_to_device-complete;readback_excluded\n'
        printf 'codec_primary=pre-sync_to_device-complete;readback_excluded\n'
        printf 'audio_seconds=%s\n' "$AUDIO_SECONDS"
        printf 'target_samples=%s\n' "$TARGET_SAMPLES"
        printf 'decoded_samples=%s\n' "$DECODED_SAMPLES"
        printf 'latent_steps=%s\n' "$LATENT_STEPS"
        printf 'rf_secondary=same_pre-sync_to_cpu_readback_complete;elements=%s\n' "$RF_READBACK_ELEMENTS"
        printf 'codec_secondary=same_pre-sync_to_cpu_readback_complete;elements=%s\n' "$DECODED_SAMPLES"
        printf 'graph_claim=same_active_semantics_not_same_graph\n'
        printf 'python_requested_joint_axis=%s\n' "$PYTHON_REQUESTED_JOINT_AXIS"
        printf 'python_encoded_joint_axis=%s\n' "$PYTHON_EXECUTED_JOINT_AXIS"
        printf 'python_forward_joint_axis=%s\n' "$PYTHON_EXECUTED_JOINT_AXIS"
        printf 'wgpu_requested_joint_axis=%s\n' "$PYTHON_REQUESTED_JOINT_AXIS"
        printf 'wgpu_compacted_joint_axis=%s\n' "$WGPU_COMPACTED_JOINT_AXIS"
        printf 'oracle_manifest_sha256=%s\n' "$ORACLE_MANIFEST_SHA256"
        printf 'performance_gate_required=%s\n' "$AGGREGATE_REQUIRE_PERFORMANCE_PASS"
        printf 'runner_sha256=%s\n' "$(sha256_file "$SCRIPT_DIR/run_v4_same_precision_stage_ab.sh")"
        printf 'source_inventory_sha256=%s\n' "$(sha256_file "$OUTPUT_DIR/build/source-sha256.txt")"
        printf 'validator_sha256=%s\n' "$(sha256_file "$OUTPUT_DIR/build/validate_v4_precision")"
        printf 'rustc=%s\n' "$(rustc --version)"
        printf 'cargo=%s\n' "$(cargo --version)"
        printf 'uv=%s\n' "$(uv --version)"
    } >"$metadata"
}

final_audit_and_complete() {
    local manifest="$OUTPUT_DIR/SHA256SUMS" complete="$OUTPUT_DIR/COMPLETE"
    verify_source_inventory_unchanged
    verify_frozen_validator
    verify_libtorch_inventory_unchanged
    expect_hf_snapshot_symlink \
        "official model" "$MODEL_SHA256" "$MODEL_PATH" "$MODEL_BLOB_PATH"
    expect_hf_snapshot_symlink \
        "official Python codec" "$PYTHON_CODEC_SHA256" "$PYTHON_CODEC_PATH" "$PYTHON_CODEC_BLOB_PATH"
    expect_file_sha256 "converted Rust codec" "$CONVERTED_CODEC_SHA256" "$CONVERTED_CODEC_PATH"
    expect_file_sha256 "source fixture" "$SOURCE_FIXTURE_SHA256" "$SOURCE_FIXTURE_PATH"
    expect_file_sha256 "FP32 oracle" "$FP32_ORACLE_SHA256" "$FP32_ORACLE_PATH"
    expect_file_sha256 "Python benchmark source fixture" "$BENCH_SOURCE_SHA256" "$BENCH_SOURCE_PATH"
    if [[ -n "$ORACLE_MANIFEST_PATH" ]]; then
        expect_file_sha256 "oracle export manifest" "$ORACLE_MANIFEST_SHA256" "$ORACLE_MANIFEST_PATH"
    fi
    assert_wgpu_cuda_environment_contract
    verify_wgpu_init_cvd_diagnostic
    jq -e \
      --arg source_inventory "$CAMPAIGN_SOURCE_INVENTORY_SHA256" \
      --arg validator "$CAMPAIGN_VALIDATOR_SHA256" \
      --arg expected_status "$AGGREGATE_SUMMARY_STATUS" \
      --argjson require_performance "$AGGREGATE_REQUIRE_PERFORMANCE_PASS" '
      .format == "irodori-v4-same-precision-stage-summary-v2" and .status == $expected_status and
      .pins.source_inventory_sha256 == $source_inventory and
      .pins.validator_binary_sha256 == $validator and
      (($require_performance == 0) or
       (.performance.acceptance.every_wgpu_rf_below_global_python_min == true and
        .performance.acceptance.every_wgpu_codec_below_global_python_min == true and
        .performance_readback_inclusive.acceptance.every_wgpu_rf_below_global_python_min == true and
        .performance_readback_inclusive.acceptance.every_wgpu_codec_below_global_python_min == true)) and
      .accuracy_gates.unique_metric_count == 10 and .active_rf_work.equal_between_runtimes == true and
      .graph_disclosure.same_graph == false and (.sessions|length) == 10
      ' "$OUTPUT_DIR/summary.json" >/dev/null || die "final summary gate failed"
    require_absent "$manifest"
    require_absent "$complete"
    (
        cd "$OUTPUT_DIR"
        find . -type f ! -name SHA256SUMS ! -name COMPLETE -print0 |
            LC_ALL=C sort -z | xargs -0 sha256sum --
    ) >"$manifest"
    (cd "$OUTPUT_DIR" && sha256sum --quiet --strict --check SHA256SUMS) || die "SHA256SUMS self-check failed"
    {
        printf 'format=%s\n' "$COMPLETE_FORMAT"
        printf 'completed_jst=%s\n' "$(TZ=Asia/Tokyo date --iso-8601=seconds)"
        printf 'status=%s\n' "$AGGREGATE_SUMMARY_STATUS"
        printf 'precision=fp32\n'
        printf 'sessions_per_runtime=%s\n' "$SESSIONS_PER_RUNTIME"
        printf 'warmups_per_session=%s\n' "$WARMUP_REPEATS"
        printf 'measured_per_session=%s\n' "$MEASURED_REPEATS"
        printf 'automatic_retries=0\n'
        printf 'summary_sha256=%s\n' "$(sha256_file "$OUTPUT_DIR/summary.json")"
        printf 'report_sha256=%s\n' "$(sha256_file "$OUTPUT_DIR/report.html")"
        printf 'manifest_sha256=%s\n' "$(sha256_file "$manifest")"
    } >"$complete"
    [[ -z "$(find "$OUTPUT_DIR" -type l -print -quit)" ]] || die "success artifact contains a symlink"
    find "$OUTPUT_DIR" -type f -exec chmod 0444 -- {} + || die "failed to freeze success files"
    find "$OUTPUT_DIR" -type d -exec chmod 0555 -- {} + || die "failed to freeze success directories"
    RUN_COMPLETED=1
}

run_self_test() (
    local temporary good bad summary good_work bad_work python_rf_readback_min failure_output
    temporary="$(mktemp -d /tmp/irodori-v4-stage-selftest.XXXXXXXX)"
    [[ "$temporary" == /tmp/irodori-v4-stage-selftest.* ]] || die "unsafe self-test directory"
    trap 'rm -rf -- "$temporary"' EXIT
    assert_wgpu_cuda_environment_contract
    good="$temporary/good"
    mkdir -p "$good"
    local ordinal runtime session rf codec
    for ordinal in {1..10}; do
        case "$ordinal" in
            1|4|5|8|9) runtime="python-pytorch-cuda-fp32"; rf=0.140; codec=0.050 ;;
            *) runtime="rust-wgpu-wgsl-fp32"; rf=0.120; codec=0.040 ;;
        esac
        session=$(((ordinal + 1) / 2))
        jq -n --arg format "$SESSION_FORMAT" --arg runtime "$runtime" \
            --argjson ordinal "$ordinal" --argjson session "$session" \
            --argjson rf "$rf" --argjson codec "$codec" '
            {format:$format,runtime:$runtime,session:$session,schedule_ordinal:$ordinal,
             fresh_process:true,precision:"fp32",warmup_repeats:2,measured_repeats:10,
             primary_timers:{rf:"synthetic",codec:"synthetic"},
             timer_contract:{pre_start_device_sync:true,stop_at_device_complete_sync:true,
                             primary_excludes_final_output_readback:true,
                             secondary_stops_after_cpu_readback:true},
             rf_warmup_seconds:[$rf,$rf],codec_warmup_seconds:[$codec,$codec],
             rf_measured_seconds:[range(0;10)|$rf],codec_measured_seconds:[range(0;10)|$codec],
             rf_warmup_readback_seconds:[($rf+0.001),($rf+0.001)],
             codec_warmup_readback_seconds:[($codec+0.001),($codec+0.001)],
             rf_measured_readback_seconds:[range(0;10)|($rf+0.001)],
             codec_measured_readback_seconds:[range(0;10)|($codec+0.001)],
             model_load_build_seconds:1,codec_load_seconds:null,external_process_wall_seconds:2,
             output_hashes:{latent:"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                            waveform:"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}}' \
            >"$good/$ordinal.json"
    done
    summary="$temporary/good-summary.json"
    aggregate_sessions "$summary" "$good/"*.json
    jq -e '(.performance.speedup_median.rf > 1 and .performance.speedup_median.codec > 1 and
            .performance_readback_inclusive.speedup_median.rf > 1 and
            .performance_readback_inclusive.speedup_median.codec > 1)' "$summary" >/dev/null || {
        die "self-test positive aggregation failed"
    }
    AGGREGATE_REQUIRE_PERFORMANCE_PASS=0
    AGGREGATE_SUMMARY_STATUS="measured"
    aggregate_sessions "$temporary/measure-only-summary.json" "$good/"*.json
    jq -e '.status == "measured"' "$temporary/measure-only-summary.json" >/dev/null || {
        die "self-test measure-only aggregation failed"
    }
    AGGREGATE_REQUIRE_PERFORMANCE_PASS=1
    AGGREGATE_SUMMARY_STATUS="passed"

    good_work="$temporary/good-work.json"
    jq -n '{
      schema_version:1,num_steps:4,
      schedule_f32_bits:[1065336439,1061146329,1056947831,1048559223,0],
      guidance_mode:"independent",enabled_cfg:["text"],
      requested:{batch_rows:1,latent_sequence:50,latent_dim:32,text_tokens:256,
                 speaker_tokens:4,caption_tokens:512,joint_axis:822},
      encoded:{batch_rows:1,latent_sequence:50,latent_dim:32,text_tokens:256,
               speaker_tokens:2,caption_tokens:512,joint_axis:820},
      encode_calls:1,
      context_kv_builds:[
        {ordinal:0,batch_rows:1,text_tokens:256,speaker_tokens:2,caption_tokens:512,
         context_tokens:770,layers:12},
        {ordinal:1,batch_rows:2,text_tokens:256,speaker_tokens:2,caption_tokens:512,
         context_tokens:770,layers:12}],
      context_kv_forward_hits:4,cond_mlp_batches:[2,2,1,1],
      forwards:[
        {ordinal:0,batch_rows:2,timestep_shape:[2],timestep_f32_bits:1065336439,
         cfg_active:true,context_kv_build_ordinal:1,latent_sequence:50,latent_dim:32,
         timestep_dtype:"float32",text_tokens:256,speaker_tokens:2,caption_tokens:512,
         joint_axis:820,context_kv_layers:12,output_shape:[2,50,32]},
        {ordinal:1,batch_rows:2,timestep_shape:[2],timestep_f32_bits:1061146329,
         cfg_active:true,context_kv_build_ordinal:1,latent_sequence:50,latent_dim:32,
         timestep_dtype:"float32",text_tokens:256,speaker_tokens:2,caption_tokens:512,
         joint_axis:820,context_kv_layers:12,output_shape:[2,50,32]},
        {ordinal:2,batch_rows:1,timestep_shape:[1],timestep_f32_bits:1056947831,
         cfg_active:false,context_kv_build_ordinal:0,latent_sequence:50,latent_dim:32,
         timestep_dtype:"float32",text_tokens:256,speaker_tokens:2,caption_tokens:512,
         joint_axis:820,context_kv_layers:12,output_shape:[1,50,32]},
        {ordinal:3,batch_rows:1,timestep_shape:[1],timestep_f32_bits:1048559223,
         cfg_active:false,context_kv_build_ordinal:0,latent_sequence:50,latent_dim:32,
         timestep_dtype:"float32",text_tokens:256,speaker_tokens:2,caption_tokens:512,
         joint_axis:820,context_kv_layers:12,output_shape:[1,50,32]}],
      whole_model_forwards:4,forward_batches:[2,2,1,1],effective_model_rows:6,
      model_layers:12,model_block_calls:48}' >"$good_work"
    jq -e \
        --argjson latent_steps "$LATENT_STEPS" \
        --argjson python_requested_axis "$PYTHON_REQUESTED_JOINT_AXIS" \
        --argjson python_executed_axis "$PYTHON_EXECUTED_JOINT_AXIS" \
        "$gate_python_work_report_filter work_ok" "$good_work" >/dev/null || {
        die "self-test rejected the canonical Python requested/encoded work contract"
    }
    bad_work="$temporary/bad-requested-work.json"
    jq '.requested.speaker_tokens = 2 | .requested.joint_axis = 820' "$good_work" >"$bad_work"
    if jq -e --argjson latent_steps "$LATENT_STEPS" \
        --argjson python_requested_axis "$PYTHON_REQUESTED_JOINT_AXIS" \
        --argjson python_executed_axis "$PYTHON_EXECUTED_JOINT_AXIS" \
        "$gate_python_work_report_filter work_ok" "$bad_work" >/dev/null 2>&1; then
        die "self-test accepted encoded geometry in the Python requested-work fields"
    fi
    bad_work="$temporary/bad-encoded-work.json"
    jq '.encoded = .requested' "$good_work" >"$bad_work"
    if jq -e --argjson latent_steps "$LATENT_STEPS" \
        --argjson python_requested_axis "$PYTHON_REQUESTED_JOINT_AXIS" \
        --argjson python_executed_axis "$PYTHON_EXECUTED_JOINT_AXIS" \
        "$gate_python_work_report_filter work_ok" "$bad_work" >/dev/null 2>&1; then
        die "self-test accepted requested geometry in the Python encoded-work fields"
    fi

    bad="$temporary/bad"
    mkdir -p "$bad"
    cp -- "$good/"*.json "$bad/"
    jq '.rf_measured_seconds[0] = 0.140 | .rf_measured_readback_seconds[0] = 0.141' \
        "$bad/2.json" >"$bad/2.mutated"
    mv -- "$bad/2.mutated" "$bad/2.json"
    if aggregate_sessions "$temporary/bad-summary.json" "$bad/"*.json 2>/dev/null; then
        die "self-test accepted a WGPU RF sample equal to the Python minimum"
    fi
    AGGREGATE_REQUIRE_PERFORMANCE_PASS=0
    AGGREGATE_SUMMARY_STATUS="measured"
    aggregate_sessions "$temporary/measure-only-loser-summary.json" "$bad/"*.json || {
        die "self-test rejected structurally valid losing data in measure-only mode"
    }
    jq -e '.status == "measured" and
      .performance.acceptance.every_wgpu_rf_below_global_python_min == false' \
      "$temporary/measure-only-loser-summary.json" >/dev/null || {
        die "self-test measure-only losing-data disclosure failed"
    }
    AGGREGATE_REQUIRE_PERFORMANCE_PASS=1
    AGGREGATE_SUMMARY_STATUS="passed"

    rm -rf -- "$bad"
    mkdir -p "$bad"
    cp -- "$good/"*.json "$bad/"
    python_rf_readback_min="$(
        jq -s '[.[] | select(.runtime == "python-pytorch-cuda-fp32") |
          .rf_measured_readback_seconds[]] | min' "$good/"*.json
    )"
    jq --argjson threshold "$python_rf_readback_min" \
        '.rf_measured_readback_seconds[0] = $threshold' \
        "$bad/2.json" >"$bad/2.mutated"
    mv -- "$bad/2.mutated" "$bad/2.json"
    if aggregate_sessions "$temporary/bad-readback-summary.json" "$bad/"*.json 2>/dev/null; then
        die "self-test accepted a WGPU RF readback sample equal to the Python minimum"
    fi

    jq -n '{schema_version:1,num_steps:5}' >"$temporary/bad-work.json"
    if jq -e --argjson latent_steps "$LATENT_STEPS" \
        --argjson python_requested_axis "$PYTHON_REQUESTED_JOINT_AXIS" \
        --argjson python_executed_axis "$PYTHON_EXECUTED_JOINT_AXIS" \
        "$gate_python_work_report_filter work_ok" "$temporary/bad-work.json" >/dev/null 2>&1; then
        die "self-test accepted a mutated RF work count"
    fi

    jq -n '[{schema_version:1,clock:"std::time::Instant",pre_start_device_sync:true,
      decode_device_complete_s:0.04,primary_includes_waveform_readback:true,
      waveform_readback_elements:96000,decode_and_readback_s:0.05,
      secondary_stops_after_readback_sync:true,primary_metric:"decode_device_complete_s"}]' \
      >"$temporary/bad-codec-timing.json"
    if jq -e 'all(.[]; .primary_includes_waveform_readback == false)' "$temporary/bad-codec-timing.json" >/dev/null 2>&1; then
        die "self-test accepted a codec timer containing readback"
    fi

    failure_output="$temporary/failure-output"
    mkdir -p "$failure_output/python" "$failure_output/wgpu"
    printf 'synthetic\n' >"$failure_output/python/input.txt"
    OUTPUT_DIR="$failure_output"
    CURRENT_PHASE="synthetic_failure"
    seal_failed_campaign 77 || die "self-test could not seal failure evidence"
    [[ -f "$failure_output/FAILURE" && -f "$failure_output/SHA256SUMS" ]] || {
        die "self-test failure evidence is incomplete"
    }
    (cd "$failure_output" && sha256sum --quiet --strict --check SHA256SUMS) || {
        die "self-test failure manifest did not verify"
    }
    [[ "$(stat -c '%a' "$failure_output/FAILURE")" == "444" ]] || {
        die "self-test failure evidence was not frozen"
    }
    chmod -R u+w "$failure_output"
    write_html_report "$summary" "$temporary/report.html"
    say "self-test=passed positive aggregation plus RF equality/work-count/readback adversaries"
)

print_dry_run_plan() {
    cat <<EOF
[same-fp32-stage] DRY RUN: no build, model, GPU workload, or artifact write.
[same-fp32-stage] output_dir=$OUTPUT_DIR (must be absent for an actual run)
[same-fp32-stage] GPU: CUDA/NVML index $CUDA_NVML_INDEX PCI $EXPECTED_GPU_PCI; WGPU adapter $WGPU_ADAPTER_INDEX
[same-fp32-stage] GPU environments: Python CUDA_VISIBLE_DEVICES=$CUDA_NVML_INDEX; WGPU CUDA_VISIBLE_DEVICES unset (paired init-only CVD A/B manifest $WGPU_INIT_CVD_DIAGNOSTIC_MANIFEST_SHA256)
[same-fp32-stage] CPU affinity: $CPU_SET
[same-fp32-stage] precision: strict FP32 on both sides; PyTorch TF32/autocast disabled
[same-fp32-stage] process order: P,W,W,P,P,W,W,P,P,W (five fresh processes/runtime)
[same-fp32-stage] each process: $WARMUP_REPEATS excluded warmups + $MEASURED_REPEATS measured repeats
[same-fp32-stage] length: seconds=$AUDIO_SECONDS latent_steps=$LATENT_STEPS target_samples=$TARGET_SAMPLES decoded_samples=$DECODED_SAMPLES
[same-fp32-stage] RF active work: Euler 4; schedule=$EXPECTED_SCHEDULE_JSON; batches=$EXPECTED_BATCHES_JSON; rows=6; layers=12; blocks=48; cfg=$EXPECTED_CFG_JSON
[same-fp32-stage] graph disclosure: both request axis$PYTHON_REQUESTED_JOINT_AXIS; Python encoded/forward$PYTHON_EXECUTED_JOINT_AXIS versus WGPU compacted/encoded/forward$WGPU_COMPACTED_JOINT_AXIS; active semantic projection only
[same-fp32-stage] performance_gate_required=$AGGREGATE_REQUIRE_PERFORMANCE_PASS; both device-complete and full-float32-CPU-readback values are always reported
[same-fp32-stage] final artifacts: per-session logs/JSON/NVML, summary.json, report.html, SHA256SUMS, then COMPLETE
EOF
    if validator_protocol_ready; then
        say "protocol_status=READY"
    else
        print_protocol_blocker
    fi
}

main() {
    trap on_exit EXIT
    trap 'cleanup_nvml; exit 130' INT
    trap 'cleanup_nvml; exit 143' TERM
    configure_oracle_contract
    if ((SELF_TEST)); then
        for command_name in awk bash chmod cp find jq mkdir mktemp mv rm sha256sum sort stat wc xargs; do
            require_command "$command_name"
        done
        run_self_test
        return
    fi
    if ((DRY_RUN)); then
        print_dry_run_plan
        return
    fi
    static_preflight
    if ! validator_protocol_ready; then
        print_protocol_blocker
        die "validator protocol is not yet capable of the required 2+10/readback-excluded comparison"
    fi
    assert_safe_lock_path
    exec 9>>"$GLOBAL_LOCK_PATH"
    flock -n 9 || die "another GPU1 campaign holds the global lock"
    assert_gpu_idle ""
    if ((PREFLIGHT_ONLY)); then
        say "preflight=passed; no build, model, GPU workload, or artifact write occurred"
        return
    fi
    RUN_STARTED=1
    CURRENT_PHASE="prepare_output_directory"
    prepare_output_directory
    CURRENT_PHASE="build_and_freeze_validator"
    build_and_freeze_validator
    CURRENT_PHASE="write_run_metadata"
    write_run_metadata
    CURRENT_PHASE="balanced_sessions"
    run_balanced_sessions
    CURRENT_PHASE="aggregate_and_acceptance"
    write_summary
    CURRENT_PHASE="final_audit_and_complete"
    final_audit_and_complete
    say "campaign=COMPLETE summary=$OUTPUT_DIR/summary.json report=$OUTPUT_DIR/report.html"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main
fi
