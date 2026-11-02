#!/usr/bin/env bash
#
# Repeated-rollout batch runs for RL pre-rollout data.
#
# Expands each Harbor task into ROLLOUTS independent copies (RL needs several
# samples per prompt, and Nexus schedules one run per task directory), then runs
# them sharded so an interrupted batch can resume.
#
# Credentials come from the launching shell; none are baked in. Export them there
# or inject them from your scheduler's secret store, never into task data:
#
#   OPENAI_API_KEY       inference
#   JUDGE_API_KEY        judge models, via the task's [verifier.env]
#   GONGFENG_TOKEN       sandbox provider (or GONGFENG_RUNTIME_TOKEN)
#
# Run it:
#
#   bash nexus/extensions/tasks/artifact/run_harbor_rollouts.sh                  # default pool
#   bash nexus/extensions/tasks/artifact/run_harbor_rollouts.sh /path/to/tasks   # another pool
#   bash nexus/extensions/tasks/artifact/run_harbor_rollouts.sh --resume <run-dir>
#   bash nexus/extensions/tasks/artifact/run_harbor_rollouts.sh --check          # compile only
#
# Everything model- and timeout-related lives in the profiles, not in here:
#   AGENT_PROFILE  configs/agents/hycode-rl-0922.yaml
#   LLM_PROFILE    configs/llm/hy4-preview-offline-taiji.yaml
#   BATCH_PROFILE  configs/rollout-batch.yaml   (chat_tracker, runner/provider timeouts)
# To try another model, point LLM_PROFILE at a different YAML. Do not add --set
# flags here; put them in a profile so a run is reproducible from config alone.
#
# Other environment overrides:
#   ROLLOUTS           copies per task (4)
#   CONCURRENT         total active task slots (1024)
#   WORKER_PROCESSES   local batch processes and SQLite shard count (64)
#   SOURCE_LIMIT       expand only the first N source tasks
#   RUN_NAME           output subdirectory (default: timestamp)
#   OUT_ROOT           where runs land
#   WORKER_VENV_SOURCE nexus source uploaded into worker environments
#
# WORKER_PROCESSES shards the batch DB; it does not allocate GPUs. Concurrency is
# capped by CONCURRENT across all shards. Resume must reuse the same run
# directory AND the same worker count, or the shard layout will not match.
#
# Inference and evaluation run inside the same Harbor task. Resume re-runs an
# unfinished task from the start; there is no workspace checkpoint between phases.
set +x
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
NEXUS_REPO=${NEXUS_REPO:-$(cd "${HERE}/../../../.." && pwd -P)}
HB=${NEXUS_REPO}/nexus/extensions/harbor_bridge

# The RL pool: same requirements as the source tasks plus an instruction-level
# ban on reading images, which a text-only policy model needs to run at all.
DEFAULT_SOURCE_TASKS=${DEFAULT_SOURCE_TASKS:-/data/t3-data/zwfy10_mghaoyang/data/rl_data_bus/0917/CodeAgent/Frontend/variants/textonly/tasks}
OUT_ROOT=${OUT_ROOT:-/taijifs_zw35/t3-h20-all/mghaoyang/data/rl_pre_rollout/0917}
ROLLOUTS=${ROLLOUTS:-4}
CONCURRENT=${CONCURRENT:-1024}
WORKER_PROCESSES=${WORKER_PROCESSES:-64}
EXPAND_JOBS=${EXPAND_JOBS:-32}
SOURCE_LIMIT=${SOURCE_LIMIT:-}
CHECK_ONLY=false

fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    sed -n '3,44p' -- "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
fi

for spec in "ROLLOUTS:${ROLLOUTS}" "CONCURRENT:${CONCURRENT}" "WORKER_PROCESSES:${WORKER_PROCESSES}" "EXPAND_JOBS:${EXPAND_JOBS}"; do
    [[ ${spec#*:} =~ ^[1-9][0-9]*$ ]] || fail "${spec%%:*} must be a positive integer"
done
(( WORKER_PROCESSES <= CONCURRENT )) || fail 'WORKER_PROCESSES must not exceed CONCURRENT'
if [[ -n ${SOURCE_LIMIT} ]]; then
    [[ ${SOURCE_LIMIT} =~ ^[1-9][0-9]*$ ]] || fail 'SOURCE_LIMIT must be a positive integer'
fi

AGENT_PROFILE=${AGENT_PROFILE:-${HERE}/configs/agents/hycode-rl-0922.yaml}
LLM_PROFILE=${LLM_PROFILE:-${HERE}/configs/llm/hy4-preview-offline-taiji.yaml}
BATCH_PROFILE=${BATCH_PROFILE:-${HERE}/configs/rollout-batch.yaml}
RUNTIME_PROFILE=${RUNTIME_PROFILE:-${HB}/configs/runtime/gongfeng-prod.yaml}
for profile in "${AGENT_PROFILE}" "${LLM_PROFILE}" "${BATCH_PROFILE}" "${RUNTIME_PROFILE}"; do
    [[ -f ${profile} ]] || fail "profile not found: ${profile}"
done

export GONGFENG_RUNTIME_TOKEN=${GONGFENG_RUNTIME_TOKEN:-${GONGFENG_TOKEN:-}}
[[ -n ${OPENAI_API_KEY:-} ]] || fail 'OPENAI_API_KEY is required (inference credential)'
[[ -n ${GONGFENG_RUNTIME_TOKEN:-} ]] || fail 'GONGFENG_RUNTIME_TOKEN or GONGFENG_TOKEN is required'
# Referenced by the task's [verifier.env]; a missing value only surfaces once the
# verifier starts, after the whole inference phase has already been paid for.
[[ -n ${JUDGE_API_KEY:-} ]] || fail 'JUDGE_API_KEY is required (judge credential)'
command -v uv >/dev/null || fail 'uv is required'

mkdir -p -- "${OUT_ROOT}"
OUT_ROOT=$(cd -- "${OUT_ROOT}" && pwd -P)

# ---------------------------------------------------------------- expansion

# Give one rollout copy its own identity. The rollout keeps the source ID in
# metadata so results can be grouped back per prompt, and T3_INSTANCE_ID must
# differ per rollout or the judge cache would be shared across samples of the
# same prompt and pollute the measured variance.
#
# task.toml is replaced through a temp file rather than edited in place: the tree
# is hard-linked from the source pool, so writing through the link would modify
# the source task.
expand_one() {
    local source=$1 out=$2 index=$3
    local name rollout_id destination toml tmp
    name=$(basename -- "${source}")
    rollout_id="${name}__r${index}"
    destination="${out}/${rollout_id}"

    rm -rf -- "${destination}"
    # Trailing /. copies the contents, not the directory itself: cp would nest
    # the tree one level deeper once the destination exists.
    mkdir -p -- "${destination}"
    cp -rl -- "${source}/." "${destination}/" 2>/dev/null \
        || cp -r -- "${source}/." "${destination}/"

    toml="${destination}/task.toml"
    tmp=$(mktemp -- "${destination}/.task.toml.XXXXXX")
    ROLLOUT_ID="${rollout_id}" SOURCE_ID="${name}" ROLLOUT_INDEX="${index}" \
        awk '
        BEGIN { rid = ENVIRON["ROLLOUT_ID"]; sid = ENVIRON["SOURCE_ID"]; idx = ENVIRON["ROLLOUT_INDEX"] }
        function add_render_token() {
            print "GONGFENG_RUNTIME_TOKEN = \"${GONGFENG_RUNTIME_TOKEN}\""
            render_token_added = 1
        }
        /^[[:space:]]*\[/ {
            if (in_meta && !added) {
                print "instance_id = \"" rid "\""
                print "source_instance_id = \"" sid "\""
                print "rollout_index = " idx
                added = 1
            }
            if (in_env && !render_token_added) add_render_token()
            in_meta = ($0 ~ /^[[:space:]]*\[metadata\]/)
            in_env  = ($0 ~ /^[[:space:]]*\[verifier\.env\]/)
            if (in_meta) seen_meta = 1
            if (in_env) seen_env = 1
            print; next
        }
        in_meta && /^[[:space:]]*(instance_id|source_instance_id|rollout_index)[[:space:]]*=/ { next }
        in_env && /^[[:space:]]*T3_INSTANCE_ID[[:space:]]*=/ { print "T3_INSTANCE_ID = \"" rid "\""; next }
        in_env && /^[[:space:]]*JUDGE_CACHE_TASK_ID[[:space:]]*=/ { print "JUDGE_CACHE_TASK_ID = \"" rid "\""; next }
        in_env && /^[[:space:]]*GONGFENG_(RUNTIME_)?TOKEN[[:space:]]*=/ {
            if (!render_token_added) add_render_token()
            next
        }
        { print }
        END {
            if (in_env && !render_token_added) add_render_token()
            if (in_meta && !added) {
                print "instance_id = \"" rid "\""
                print "source_instance_id = \"" sid "\""
                print "rollout_index = " idx
            } else if (!seen_meta) {
                print ""
                print "[metadata]"
                print "instance_id = \"" rid "\""
                print "source_instance_id = \"" sid "\""
                print "rollout_index = " idx
            }
            if (!seen_env) {
                print "task has no [verifier.env] table: " FILENAME > "/dev/stderr"
                exit 3
            }
        }' "${toml}" > "${tmp}"
    chmod --reference="${toml}" -- "${tmp}" 2>/dev/null || true
    mv -f -- "${tmp}" "${toml}"
}
export -f expand_one

# ---------------------------------------------------------------- arguments

if [[ ${1:-} == --resume ]]; then
    [[ $# -ge 2 ]] || fail '--resume requires an existing run directory'
    RUN_DIR=$(cd -- "$2" && pwd -P) || fail "run directory not found: $2"
    shift 2
    TASKS_DIR=${RUN_DIR}/tasks
    DB=${RUN_DIR}/batch.db
    [[ -d ${TASKS_DIR} ]] || fail "expanded task directory not found: ${TASKS_DIR}"
    [[ -e ${DB} || -d ${DB}.shards ]] || fail "batch database not found: ${DB}"
else
    SOURCE_TASKS=${DEFAULT_SOURCE_TASKS}
    if [[ ${1:-} == --check ]]; then
        CHECK_ONLY=true
        shift
    elif [[ $# -gt 0 && $1 != -* ]]; then
        SOURCE_TASKS=$1
        shift
        if [[ ${1:-} == --check ]]; then
            CHECK_ONLY=true
            shift
        fi
    fi
    [[ -d ${SOURCE_TASKS} ]] || fail "source task directory not found: ${SOURCE_TASKS}"
    SOURCE_TASKS=$(cd -- "${SOURCE_TASKS}" && pwd -P)

    RUN_NAME=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}
    [[ ${RUN_NAME} =~ ^[A-Za-z0-9_.-]+$ ]] || fail 'RUN_NAME may contain only letters, digits, dot, underscore, hyphen'
    RUN_DIR=${OUT_ROOT}/${RUN_NAME}
    [[ ! -e ${RUN_DIR} ]] || fail "run directory already exists: ${RUN_DIR}"
    TASKS_DIR=${RUN_DIR}/tasks
    DB=${RUN_DIR}/batch.db
    mkdir -p -- "${TASKS_DIR}"

    # Copying thousands of small trees onto a network filesystem is latency
    # bound, so fan the copies out; serial expansion of 10k rollouts takes
    # over an hour.
    mapfile -t SOURCES < <(find "${SOURCE_TASKS}" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/task.toml' \; -print | sort)
    [[ ${#SOURCES[@]} -gt 0 ]] || fail "no Harbor tasks found under ${SOURCE_TASKS}"
    if [[ -n ${SOURCE_LIMIT} && ${SOURCE_LIMIT} -lt ${#SOURCES[@]} ]]; then
        SOURCES=("${SOURCES[@]:0:${SOURCE_LIMIT}}")
    fi

    printf 'Expanding %d task(s) x %d rollout(s) with %d job(s)...\n' \
        "${#SOURCES[@]}" "${ROLLOUTS}" "${EXPAND_JOBS}"
    for (( index = 0; index < ROLLOUTS; index++ )); do
        printf '%s\0' "${SOURCES[@]}" \
            | xargs -0 -P "${EXPAND_JOBS}" -I {} bash -euo pipefail -c \
                'expand_one "$1" "$2" "$3"' _ {} "${TASKS_DIR}" "${index}"
    done
    printf 'Prepared %d rollout task(s) -> %s\n' "$(( ${#SOURCES[@]} * ROLLOUTS ))" "${TASKS_DIR}"
fi

# ---------------------------------------------------------------- run

WORKER_VENV_SOURCE=${WORKER_VENV_SOURCE:-${NEXUS_REPO}}
[[ -f "${WORKER_VENV_SOURCE}/pyproject.toml" ]] || fail "invalid WORKER_VENV_SOURCE: ${WORKER_VENV_SOURCE}"
WORKER_VENV_SOURCE=$(cd -- "${WORKER_VENV_SOURCE}" && pwd -P)

# New expansions receive this mapping above; the check also protects resumes made
# by older script versions, which would otherwise finish inference before Render
# reports the missing verifier credential.
(
    cd -- "${NEXUS_REPO}"
    uv run --with 'harbor==0.20.0' python - "${TASKS_DIR}" <<'PY'
import sys
import tomllib

from pathlib import Path

from nexus.extensions.harbor_bridge.compiler import resolve_env_mapping


missing = []
for task_file in sorted(Path(sys.argv[1]).glob('*/task.toml')):
    with task_file.open('rb') as handle:
        task = tomllib.load(handle)
    declared = task.get('verifier', {}).get('env', {})
    try:
        resolved = resolve_env_mapping(declared)
    except ValueError as exc:
        raise SystemExit(f'Verifier environment preflight failed for {task_file}: {exc}') from None
    token = (
        resolved.get('GONGFENG_RUNTIME_TOKEN', '').strip()
        or resolved.get('GONGFENG_TOKEN', '').strip()
    )
    if not token:
        missing.append(str(task_file))

if missing:
    raise SystemExit(
        'Render credential preflight failed; map GONGFENG_RUNTIME_TOKEN or GONGFENG_TOKEN '
        'in [verifier.env]: ' + ', '.join(missing[:8])
    )
PY
)

printf '%s\n' \
    "Run directory: ${RUN_DIR}" \
    "Tasks: ${TASKS_DIR}" \
    "Database: ${DB}" \
    "Agent: $(basename -- "${AGENT_PROFILE}") + $(basename -- "${LLM_PROFILE}")" \
    "Total concurrency: ${CONCURRENT}" \
    "Worker processes: ${WORKER_PROCESSES}"

if ${CHECK_ONLY}; then
    # run_harbor_test.sh owns the judge/secret preflight; reuse it rather than
    # keeping a second copy of that logic here.
    exec env AGENT_PROFILE="${AGENT_PROFILE}" LLM_PROFILE="${LLM_PROFILE}" \
        RUNTIME_PROFILE="${RUNTIME_PROFILE}" \
        bash "${HERE}/run_harbor_test.sh" "${TASKS_DIR}" --check "$@"
fi

# Nexus is invoked directly rather than through run_harbor_test.sh because that
# script deliberately refuses an existing DB. Reusing the same parent DB is
# exactly what makes sharded resume deterministic.
cd -- "${NEXUS_REPO}"
exec uv run --with 'harbor==0.20.0' nexus run batch \
    --data "t3-harbor://${TASKS_DIR}?flat=true&workers=${WORKER_PROCESSES}" \
    --use harbor_bridge:mode=agent \
    --use "${RUNTIME_PROFILE}" \
    --use "${AGENT_PROFILE}" \
    --use "${LLM_PROFILE}" \
    --use "${BATCH_PROFILE}" \
    --set "runtime_configs.main.worker_venv.source=${WORKER_VENV_SOURCE}" \
    --set "runtime_configs.verifier.worker_venv.source=${WORKER_VENV_SOURCE}" \
    --max-concurrent "${CONCURRENT}" \
    --worker-processes "${WORKER_PROCESSES}" \
    -d "${DB}" \
    "$@"
