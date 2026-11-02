#!/usr/bin/env bash
# Run Harbor Task directories through Dataset + Compose; no artifact_task generator.
set +x
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: run_harbor_test.sh <task-dir|tasks-dir> [--check] [nexus run batch options]

  bash run_harbor_test.sh /path/to/tasks/parrot_bike_svg
  LIMIT=20 CONCURRENT=4 bash run_harbor_test.sh /path/to/tasks
  bash run_harbor_test.sh /path/to/tasks/parrot_bike_svg --check

--check compiles tasks/profiles and validates Judge Profiles and secret mappings.
It does not run agents/verifiers or check upstream authentication.
No .env file is sourced implicitly. Export secrets in the launching shell or
inject them through your scheduler's secret environment, never into task data.

Task configuration in task.toml [verifier.env]:
  T3_JUDGE_NAME        Required strategy name; models are bound inside the strategy
  T3_JUDGE_MODELS      Optional role-to-YAML JSON overrides for controlled comparisons
  JUDGE_API_KEY        "${JUDGE_API_KEY}" forwards the host secret into the verifier

Single-model YAMLs live in artifact/configs/llm/*-aihub-2.0.yaml and use the same
body/imports/agent_config/openai_clients.main/merge/source format as inference.
Each Judge binds its own models to agent (tool-using agent loop) and/or vlm
(direct model judgments, with per-check calls and retries allowed).
Web uses agent for flow2code and vlm for functional/aesthetic judgments.
Use agent/vlm keys in T3_JUDGE_MODELS; old flow2code/scoring keys are rejected.
Missing bindings fail; there is no silent fallback to inference main.
Cache IDs default to the data instance_id, shared by agent/vlm across turns,
retries and repeated evaluations of the same data. No random run ID is generated.
No separate cache ID is required in task.toml or model YAMLs.
LiteLLM appends/replaces cache_task_id in the API-key query, preserving other fields.
For experiments only, a non-empty JUDGE_CACHE_TASK_ID already in the verifier
environment overrides instance_id; empty values still use the data instance_id.
This routing hint does not enable cache_control or prove upstream cache hits.
No clients JSON or JUDGE_CLIENTS_FILE is required. Legacy NEXUS_OPENAI_CLIENTS
and T3_JUDGE_PROFILE are rejected. The host preflight parses model YAMLs in a
fresh process containing only the task's declared verifier environment.
Missing declared secrets and unresolved placeholders fail before batch execution.
OPENAI_API_KEY configures the inference profile, not the judge clients;
GONGFENG_RUNTIME_TOKEN configures the runtime provider.

Options via environment:
  LIMIT                Positive task limit; unset means all discovered tasks
  CONCURRENT / WORKERS Run concurrency / directory scan workers (1 / 8)
  FLAT                 true: direct children; false: recursive scan (default true)
  AGENT_PROFILE        configs/agents/claude-code.yaml
  LLM_PROFILE          configs/llm/opus5-aihub-2.0.yaml
  RUNTIME_PROFILE      configs/runtime/gongfeng-prod.yaml
  WORKER_VENV_SOURCE   Nexus source installed in task runtimes (default this repo)
  OUT_ROOT / TAG       Output directory / unique run name; existing DBs are rejected

Profiles accept absolute paths, repo-relative paths, or Harbor configs/... paths.
Additional batch options are forwarded unchanged. Use a new TAG for each comparison.
HELP
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    usage
    exit 0
fi
if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }
[[ -d "$1" ]] || fail "task directory not found: $1"
TASKS_DIR=$(cd -- "$1" && pwd -P)
shift
CHECK_ONLY=false
if [[ ${1:-} == --check ]]; then
    CHECK_ONLY=true
    shift
fi

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
NEXUS_REPO=${NEXUS_REPO:-$(cd "${HERE}/../../../.." && pwd -P)}
cd -- "${NEXUS_REPO}"
NEXUS_REPO=$PWD
HB=${NEXUS_REPO}/nexus/extensions/harbor_bridge
WORKER_VENV_SOURCE=${WORKER_VENV_SOURCE:-${NEXUS_REPO}}
[[ -f "${WORKER_VENV_SOURCE}/pyproject.toml" ]] || fail "invalid WORKER_VENV_SOURCE"
WORKER_VENV_SOURCE=$(cd -- "${WORKER_VENV_SOURCE}" && pwd -P)
command -v uv >/dev/null || fail 'uv is required'

resolve_profile() {
    local path=$1
    [[ ${path} != configs/* ]] || path="${HB}/${path}"
    [[ -f ${path} ]] || fail "profile not found: ${path}"
    printf '%s/%s\n' "$(cd -- "$(dirname -- "${path}")" && pwd -P)" "$(basename -- "${path}")"
}

AGENT_PROFILE=$(resolve_profile "${AGENT_PROFILE:-configs/agents/hycode_plan_askuser.yaml}")
LLM_PROFILE=$(resolve_profile "${LLM_PROFILE:-configs/llm/opus5-aihub-2.0.yaml}")
RUNTIME_PROFILE=$(resolve_profile "${RUNTIME_PROFILE:-configs/runtime/gongfeng-prod.yaml}")
WORKERS=${WORKERS:-8}
CONCURRENT=${CONCURRENT:-1}
FLAT=${FLAT:-true}
for var in WORKERS CONCURRENT; do
    [[ ${!var} =~ ^[1-9][0-9]*$ ]] || fail "${var} must be a positive integer"
done
[[ ${FLAT} == true || ${FLAT} == false ]] || fail 'FLAT must be true or false'
LIMIT_ARG=()
if [[ -n ${LIMIT:-} ]]; then
    [[ ${LIMIT} =~ ^[1-9][0-9]*$ ]] || fail 'LIMIT must be positive; unset it to run all tasks'
    LIMIT_ARG=(--limit "${LIMIT}")
fi
export GONGFENG_RUNTIME_TOKEN=${GONGFENG_RUNTIME_TOKEN:-${GONGFENG_TOKEN:-}}
[[ -n ${GONGFENG_RUNTIME_TOKEN} ]] || fail 'GONGFENG_RUNTIME_TOKEN or GONGFENG_TOKEN is required'

UV=(uv run --with 'harbor==0.20.0')
TASKS_URI=$("${UV[@]}" python -c 'import sys; from urllib.parse import quote; print("t3-harbor://" + quote(sys.argv[1], safe="/"))' "${TASKS_DIR}")
ARGS=(
    --data "${TASKS_URI}?flat=${FLAT}&workers=${WORKERS}"
    --use harbor_bridge:mode=agent
    --use "${RUNTIME_PROFILE}"
    --use "${AGENT_PROFILE}"
    --use "${LLM_PROFILE}"
    --set "runtime_configs.main.worker_venv.source=${WORKER_VENV_SOURCE}"
    --set "runtime_configs.verifier.worker_venv.source=${WORKER_VENV_SOURCE}"
    "${LIMIT_ARG[@]}"
    --max-concurrent "${CONCURRENT}"
)

"${UV[@]}" python - "${ARGS[@]}" "$@" <<'PY'
import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from nexus.extensions.harbor_bridge.compiler import resolve_env_mapping
from nexus.extensions.tasks.artifact.judges.registry import get_judge
from nexus.runner.compose import compile_batch, parse_ordered_parts

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--limit', type=int)
parser.add_argument('--split')
args, _ = parser.parse_known_args(sys.argv[1:])
items = compile_batch(parse_ordered_parts(sys.argv[1:]), limit=args.limit, split=args.split)
if not items:
    raise SystemExit('No Harbor tasks selected')
for item in items:
    cfg = item.config
    if cfg.task_files is None or cfg.task_files.type != 'directory':
        raise SystemExit('Harbor preflight requires a local task directory')
    with (Path(cfg.task_files.path) / 'task.toml').open('rb') as handle:
        task = tomllib.load(handle)
    declared_env = task.get('verifier', {}).get('env', {})
    if any(k in declared_env for k in ('T3_JUDGE_NAME', 'T3_JUDGE_MODELS', 'T3_JUDGE_PROFILE', 'NEXUS_OPENAI_CLIENTS')):
        try:
            verifier_env = resolve_env_mapping(declared_env)
        except ValueError as exc:
            raise SystemExit(f'Judge environment preflight failed: {exc}') from None
        judge_name = verifier_env.get('T3_JUDGE_NAME', '').strip()
        if judge_name:
            try:
                judge_spec = get_judge(judge_name)
            except Exception as exc:
                raise SystemExit(f'Judge strategy preflight failed: {type(exc).__name__}: {exc}') from None
            if set(judge_spec.capabilities) & {'deploy', 'screenshot'}:
                render_token = (
                    verifier_env.get('GONGFENG_TOKEN', '').strip()
                    or verifier_env.get('GONGFENG_RUNTIME_TOKEN', '').strip()
                )
                if not render_token:
                    raise SystemExit(
                        'Render credential preflight failed: task [verifier.env] must map '
                        'GONGFENG_RUNTIME_TOKEN or GONGFENG_TOKEN'
                    )
        child_env = {'PATH': os.environ.get('PATH', ''), 'PYTHONDONTWRITEBYTECODE': '1',
                     'NEXUS_AUTO_INSTALL': '0', **verifier_env}
        checked = subprocess.run(
            [sys.executable, '-m', 'nexus.extensions.tasks.artifact.judges.profiles'],
            env=child_env, text=True, capture_output=True, timeout=60,
        )
        if checked.returncode:
            raise SystemExit('Judge model preflight failed: ' + checked.stderr.strip())
        print('Judge models: ' + checked.stdout.strip())
    print(f'{item.run_id}: task={cfg.task_name}, mode={cfg.task_config.get("mode")}, '
          f'agent_timeout={cfg.task_config.get("agent_timeout")}, '
          f'verifier_timeout={cfg.task_config.get("test_timeout")}')
print(f'Validated {len(items)} task(s); no runtime or model was started. Upstream authentication was not checked.')
PY
if ${CHECK_ONLY}; then
    exit 0
fi

TAG=${TAG:-$(date +%Y%m%d_%H%M%S)_$$}
[[ ${TAG} =~ ^[A-Za-z0-9_.-]+$ ]] || fail 'TAG may contain only letters, digits, dot, underscore, and hyphen'
OUT_ROOT=${OUT_ROOT:-${NEXUS_REPO}/outputs/harbor_test}
mkdir -p -- "${OUT_ROOT}"
OUT_ROOT=$(cd -- "${OUT_ROOT}" && pwd -P)
DB=${OUT_ROOT}/${TAG}.db
LOG=${OUT_ROOT}/${TAG}.log
[[ ! -e ${DB} && ! -e ${LOG} ]] || fail 'output already exists; choose a new TAG (resume with nexus CLI explicitly)'

printf 'Tasks: %s\nAgent: %s + %s\nWorker source: %s\nDatabase: %s\nLog: %s\n' \
    "${TASKS_DIR}" "$(basename -- "${AGENT_PROFILE}")" "$(basename -- "${LLM_PROFILE}")" \
    "${WORKER_VENV_SOURCE}" "${DB}" "${LOG}"

# Preserve batch/tee failures while still showing any recorded results.
set +e
"${UV[@]}" nexus run batch "${ARGS[@]}" -d "${DB}" "$@" 2>&1 | tee "${LOG}"
PIPE_STATUS=("${PIPESTATUS[@]}")
set -e
RUN_STATUS=${PIPE_STATUS[0]}
[[ ${RUN_STATUS} -ne 0 ]] || RUN_STATUS=${PIPE_STATUS[1]}
[[ -f ${DB} ]] || exit "${RUN_STATUS}"

printf '\nResults:\n'
set +e
"${UV[@]}" python - "${DB}" <<'PY'
import json
import sqlite3
import sys
from pathlib import Path

db = sqlite3.connect(Path(sys.argv[1]).resolve().as_uri() + '?mode=ro', uri=True)
for run_id, status, out, err in db.execute('select run_id, status, task_output, error from runs order by run_id'):
    payload = {}
    if out:
        try:
            payload = json.loads(out)
        except Exception:
            payload = {}
    rm = payload.get('reward_metadata') or {}
    jm = rm.get('judge_metadata') or {}
    s = jm.get('summary') or {}
    c = jm.get('collection') or {}
    line = [f'{run_id[:56]:56}', f'{status:8}']
    if payload.get('score') is not None:
        line.append(f'score={payload["score"]:.3f}')
    for label, value in (('func', s.get('func_score')), ('aes', s.get('aesthetic_score'))):
        if value is not None:
            line.append(f'{label}={float(value):.3f}')
    counts = s.get('counts') or {}
    if counts:
        line.append(f'p/f/inc={counts.get("pass", 0)}/{counts.get("fail", 0)}/{counts.get("inconclusive", 0)}')
    if c.get('shot_calls') is not None:
        line.append(f'shots={c.get("shot_calls")}/{c.get("shot_successes")}')
    if rm.get('failure_kind'):
        line.append(f'[{rm["failure_kind"]}]')
    if jm.get('zero_reason'):
        line.append(f'[{jm["zero_reason"]}]')
    archive = rm.get('evidence_archive') or {}
    if archive.get('url'):
        line.append(f'evidence={archive["url"]}')
    print('  ' + ' '.join(line))
    if status != 'success' and err:
        print(f'    error: {str(err)[:200]}')
db.close()
PY
SUMMARY_STATUS=$?
set -e
if [[ ${RUN_STATUS} -eq 0 && ${SUMMARY_STATUS} -ne 0 ]]; then
    RUN_STATUS=${SUMMARY_STATUS}
fi
exit "${RUN_STATUS}"
