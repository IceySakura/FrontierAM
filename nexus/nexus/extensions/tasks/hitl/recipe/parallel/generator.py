"""Batch generator for the parallel task: several sessions, checks named by the run.

What is being tested is stated in the launch script rather than read from a dataset:

    HITL_PARALLEL_SESSIONS  the session names to open, comma separated
    HITL_PARALLEL_SCRIPT    the steps to run, a JSON list or a path to one
    HITL_PARALLEL_CHECKS    what to look for afterwards, likewise

There is no ``live`` mode. The point of the run is what happens between sessions, so
which of them is addressed cannot be left to a model deciding turn by turn -- the
script names its recipients and the same input runs every time.

What has to agree with something outside this file:

``runtime_configs['main']['worker_venv']``
    The worker manager invokes this package's sandbox code in ``main``, where the venv
    installs it.

``agent_config['working_dir']``
    Sent here because this task has no dataset to read it from, and the agent requires
    it. Each session works in a directory under it unless the run shares one.
"""

import json
import logging
import os

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus.extensions.tasks.hitl.actor.factory import backends
from nexus.extensions.tasks.hitl.model import known_profiles
from nexus.extensions.tasks.hitl.recipe.parallel.agent import (
    AGENT_NAME,
    DEFAULT_SESSIONS,
    ParallelAgentInput,
    ParallelStep,
    check_script,
    check_sessions,
)
from nexus.extensions.tasks.hitl.recipe.parallel.task import TASK_NAME, ParallelCheck
from nexus.extensions.tasks.hitl.utils.generator_util import (
    _build_openai_clients,
    _build_runtime_provider_config,
    _build_worker_config,
    _env_bool,
    _env_choice,
    _env_int,
    _env_list,
    _new_session_id,
    _require_env,
    client_imports,
    with_worker_skills,
    worker_usage_profile,
)
from nexus.runner.batch.batch_config import BatchItem
from nexus.runner.single import RunInput


logger = logging.getLogger(__name__)


TASK_IMPORT = 'nexus.extensions.tasks.hitl.recipe.parallel.task'
AGENT_IMPORT = 'nexus.extensions.tasks.hitl.recipe.parallel.agent'

PROCESSOR_IMPORTS = (
    'nexus.extensions.tasks.hitl.trace.chat',
    'nexus.extensions.tasks.hitl.trace.usage',
    'nexus.extensions.tasks.hitl.trace.langfuse',
)

DEFAULT_WORKER_BACKEND = 'print_claude_code'
DEFAULT_WORKING_DIR = '/workspace'
DEFAULT_TIMEOUT_MULTIPLIER = 30

WORKER_CLIENT = 'worker'

# ``branch_last`` keeps a per-request branch, which is what carries the trace label;
# ``global_first`` exports one flat list with nowhere to put it.
CHAT_PROCESSOR: dict[str, Any] = {'chat': {'name': 'chat_tracker', 'args': {'tools_mode': 'branch_last'}}}

# Variables another recipe exports that mean nothing here, mapped to what to use
# instead. Not read by this generator, so the run would take the defaults instead.
FOREIGN_ENV: dict[str, str] = {
    'HITL_SMOKE_MESSAGE': 'HITL_PARALLEL_MESSAGE',
    'HITL_SMOKE_CHECKS': 'HITL_PARALLEL_CHECKS',
    'HITL_SMOKE_SCRIPT': 'HITL_PARALLEL_SCRIPT, whose steps address sessions by name',
    'HITL_SMOKE_MODE': 'nothing -- a parallel run is always driven from its script',
    'HITL_SMOKE_INJECTIONS': 'nothing -- as above',
    'HITL_SMOKE_IMAGE': 'HITL_PARALLEL_IMAGE',
    'SWE_DATASET_PATH': 'nothing -- a parallel run states its own script',
}

ENVIRONMENT: dict[str, str] = {
    # -- what this run tests --
    'HITL_PARALLEL_SESSIONS': f'session names, comma separated. Default {",".join(DEFAULT_SESSIONS)!r}.',
    'HITL_PARALLEL_SCRIPT': 'required. A JSON list of {parallel:[{to,message}]}, or a path to one.',
    'HITL_PARALLEL_CHECKS': 'required. A JSON list of checks, or a path to a file holding one.',
    'HITL_PARALLEL_MESSAGE': 'context recorded on the run. The script is what drives it.',
    'HITL_PARALLEL_SHARED_WORKSPACE': 'put every session in working_dir instead of one directory each.',
    'HITL_PARALLEL_NAME': "the run id. Default 'parallel'.",
    'HITL_PARALLEL_IMAGE': 'required. The container image.',
    'HITL_PARALLEL_WORKING_DIR': f'where the sessions work. Default {DEFAULT_WORKING_DIR!r}.',
    'HITL_PARALLEL_REPEAT': 'run the same configuration this many times. Default 1.',
    'HITL_PARALLEL_COMMAND_TIMEOUT': "a command check's wall clock, seconds. Default 60.",
    # -- the run --
    'TIMEOUT_MULTIPLIER': f'the run ceiling = 60s * this. Default {DEFAULT_TIMEOUT_MULTIPLIER}.',
    # -- both clients --
    'OPENAI_API_KEY': 'fallback key when the per-role one is unset.',
    'OPENAI_BASE_URL': 'fallback endpoint.',
    'OPENAI_MODEL': 'fallback model.',
    'HITL_LLM_TIMEOUT': 'per-request timeout for any role without its own. Default 900.',
    # -- the workers --
    'HITL_WORKER_BACKEND': f'print_claude_code or pty_claude_code. Default {DEFAULT_WORKER_BACKEND!r}.',
    'HITL_WORKER_VERSION': 'harness release to install, whichever harness the backend names. Each has its own default.',
    'HITL_WORKER_TURN_TIMEOUT': 'one worker turn, seconds. Default 7200.',
    'HITL_WORKER_CONTEXT_WINDOW': (
        'the context window auto-compaction is sized against, tokens. Claude Code compacts at '
        'HITL_CC_AUTOCOMPACT_PCT of it (90% by default), Codex at min(that limit, this * 90%).'
    ),
    'HITL_WORKER_CLIENT_PROFILE': "the workers' provider protocol and usage profile. Default 'default'.",
    'HITL_WORKER_MODEL': "the workers' model. Falls back to OPENAI_MODEL.",
    'HITL_WORKER_BASE_URL': 'its endpoint. Falls back to OPENAI_BASE_URL.',
    'HITL_WORKER_API_KEY': 'its key. Falls back to OPENAI_API_KEY.',
    'HITL_WORKER_LLM_TIMEOUT': 'its per-request timeout, seconds.',
    'HITL_WORKER_THINKING_ENABLED': "thinking on. Default true; 0 stamps the profile's no-think args.",
    'HITL_WORKER_TOOL_SEARCH': 'progressive tool disclosure, which is how Grep/Glob reach a worker.',
    'HITL_WORKER_MEMORY_FILE': 'path to the CLAUDE.md placed for the workers.',
    'HITL_WORKER_SKILL_DIRS': 'directories packed into the sandbox as worker skills.',
    'HITL_WORKER_BINARY_URL': 'Claude Code binary URL.',
    'HITL_CC_BUILTIN_COMMANDS': "the harness's own commands to offer, comma-separated. None by default.",
    'HITL_WORKER_MCP_SERVERS': 'path to a JSON list of MCP servers registered for the workers.',
    'HITL_WORKER_MCP_SEARCH': 'register the built-in search MCP server for the workers.',
    'SEARCH_API_KEY': 'required when HITL_WORKER_MCP_SEARCH=1.',
    # -- the sandbox --
    'SWE_RUNTIME_PROVIDER_NAME': "runtime provider. Default 'gongfeng'.",
    'SWE_RUNTIME_BASE_URL': 'its API base URL.',
    'SWE_RUNTIME_TOKEN': 'its token.',
    'SWE_RUNTIME_CPU': 'cpu per sandbox. Default 2.',
    'SWE_RUNTIME_MEMORY': 'memory per sandbox, GB. Default 10.',
    'SWE_RUNTIME_TTL': "sandbox lifetime as a Go duration ('12h', '1h30m'), max 168h. Backend default when unset.",
    'SWE_RUNTIME_PURPOSE': "sandbox lifecycle tag. Default 'test'.",
    'SWE_WORKER_VENV': 'install the nexus wheel in the sandboxes. Default true, and the workers need it.',
    # -- tracing --
    'LANGFUSE_PUBLIC_KEY': 'Langfuse upload is configured only when this and the secret are both set.',
    'LANGFUSE_SECRET_KEY': 'see above.',
    'LANGFUSE_BASE_URL': 'Langfuse endpoint.',
    'HITL_DISABLE_LANGFUSE': 'skip Langfuse upload even when credentials are present.',
}


def generate_configs() -> Iterator[BatchItem]:
    """One ``BatchItem`` per repeat of the single configuration this run declares."""
    _check_foreign_env()

    sessions = check_sessions(_env_list('HITL_PARALLEL_SESSIONS') or list(DEFAULT_SESSIONS))
    steps = _load_script()
    check_script(steps, sessions=sessions)
    checks = _load_checks()

    image = _require_env('HITL_PARALLEL_IMAGE', 'HITL_PARALLEL_IMAGE required (the container to run in)')
    name = os.environ.get('HITL_PARALLEL_NAME', '').strip() or 'parallel'
    message = os.environ.get('HITL_PARALLEL_MESSAGE', '').strip()
    working_dir = os.environ.get('HITL_PARALLEL_WORKING_DIR', '').strip() or DEFAULT_WORKING_DIR
    shared_workspace = _env_bool('HITL_PARALLEL_SHARED_WORKSPACE')
    repeat = _env_int('HITL_PARALLEL_REPEAT', default=1)
    command_timeout = _env_int('HITL_PARALLEL_COMMAND_TIMEOUT', default=60)

    runtime_provider_name = os.environ.get('SWE_RUNTIME_PROVIDER_NAME', '').strip() or 'gongfeng'
    timeout_seconds = 60 * _env_int('TIMEOUT_MULTIPLIER', default=DEFAULT_TIMEOUT_MULTIPLIER)
    worker_venv = _env_bool('SWE_WORKER_VENV', default=True)

    worker_profile = _env_choice('HITL_WORKER_CLIENT_PROFILE', 'default', known_profiles())
    worker_backend = _env_choice('HITL_WORKER_BACKEND', DEFAULT_WORKER_BACKEND, backends())

    session_id = _new_session_id()
    openai_clients = _build_openai_clients([], worker_profile, session_id=session_id)
    worker_config = _build_worker_config(worker_profile, session_id=session_id)
    worker_skill_dirs = _env_list('HITL_WORKER_SKILL_DIRS')
    if worker_skill_dirs:
        worker_config = with_worker_skills(worker_config, worker_skill_dirs)
    worker_config = {**worker_config, 'working_dir': working_dir}
    runtime_provider_config = _build_runtime_provider_config()
    imports = [TASK_IMPORT, AGENT_IMPORT, *PROCESSOR_IMPORTS, *client_imports(worker_profile)]

    agent_config: dict[str, Any] = {
        'working_dir': working_dir,
        'worker_client_name': WORKER_CLIENT,
        'worker_backend': worker_backend,
        'worker_config': worker_config,
        'sessions': sessions,
        'script': [step.model_dump() for step in steps],
        'shared_workspace': shared_workspace,
        # The agent's own ceiling. Sent here rather than as a task key: the runner
        # builds a task input from the fields the task declares and drops the rest, so
        # a timeout named there would be silently ignored.
        'timeout': float(timeout_seconds),
    }
    _check_agent_config(agent_config)

    logger.info(
        '[hitl/parallel/generator] %s: %d session(s) %s, %d step(s), %d check(s), %d run(s), backend=%s',
        name,
        len(sessions),
        sessions,
        len(steps),
        len(checks),
        repeat,
        worker_backend,
    )
    for check in checks:
        logger.info('[hitl/parallel/generator]   check %s (%s)', check.name, check.kind())

    for attempt in range(repeat):
        yield BatchItem(
            config=RunInput(
                imports=imports,
                task_name=TASK_NAME,
                task_config={
                    'message': message,
                    'checks': [check.model_dump() for check in checks],
                    'working_dir': working_dir,
                    'command_timeout': command_timeout,
                },
                agent_name=AGENT_NAME,
                agent_config=agent_config,
                runtime_provider_name=runtime_provider_name,
                runtime_provider_config={
                    **runtime_provider_config,
                    'image': image,
                    'working_dir': working_dir,
                },
                runtime_configs={'main': {'worker_venv': worker_venv}},
                openai_clients=openai_clients,
                processors=_processors(worker_profile, name),
            ),
            run_id=name if repeat == 1 else f'{name}__try{attempt + 1}',
            metadata={
                'parallel': name,
                'attempt': attempt + 1,
                'sessions': sessions,
                'checks': [check.name for check in checks],
                'worker_backend': worker_backend,
            },
        )


def _load_json_list(key: str, *, what: str, required: bool) -> list[Any]:
    """The JSON list ``key`` holds, or the one in the file it names."""
    raw = os.environ.get(key, '').strip()
    if not raw:
        if required:
            raise ValueError(f'{key} required ({what})')
        return []
    text = raw
    if not raw.lstrip().startswith('['):
        candidate = Path(raw).expanduser()
        if not candidate.is_file():
            raise ValueError(f'{key} is neither a JSON list nor a file: {raw}')
        text = candidate.read_text(encoding='utf-8')
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f'{key} is not valid JSON: {exc}') from exc
    if not isinstance(loaded, list):
        raise ValueError(f'{key} must be a JSON list, got {type(loaded).__name__}')
    return loaded


def _load_checks() -> list[ParallelCheck]:
    raw = _load_json_list('HITL_PARALLEL_CHECKS', what='what this run expects to observe', required=True)
    checks = [ParallelCheck.model_validate(item) for item in raw]
    if not checks:
        raise ValueError('HITL_PARALLEL_CHECKS is empty; a run with no checks reports nothing')
    for check in checks:
        check.kind()
    return checks


def _load_script() -> list[ParallelStep]:
    raw = _load_json_list('HITL_PARALLEL_SCRIPT', what='the steps to run', required=True)
    return [ParallelStep.model_validate(item) for item in raw]


def _check_agent_config(agent_config: dict[str, Any]) -> None:
    """Refuse a key the agent does not declare, which pydantic would drop in silence."""
    unknown = sorted(set(agent_config) - set(ParallelAgentInput.model_fields))
    if unknown:
        raise ValueError(f'agent_config carries {unknown}, which {AGENT_NAME} does not declare')


def _check_foreign_env() -> None:
    """Refuse a variable from another recipe that means nothing here."""
    offenders = {key: instead for key, instead in FOREIGN_ENV.items() if os.environ.get(key, '').strip()}
    if offenders:
        listed = '\n'.join(f'  {key} is not read here; use {instead}' for key, instead in sorted(offenders.items()))
        raise ValueError(f'this generator does not read these:\n{listed}')


def _processors(worker_profile: str, name: str) -> dict[str, Any]:
    """The chat tracker, the usage counters, and Langfuse when it is configured."""
    processors: dict[str, Any] = {
        **CHAT_PROCESSOR,
        'usage': {
            'name': 'hitl_usage',
            'args': {'default_profile': worker_usage_profile(worker_profile)},
        },
    }
    langfuse = _langfuse_processor(name)
    if langfuse:
        processors['langfuse'] = langfuse
    return processors


def _langfuse_processor(name: str) -> dict[str, Any] | None:
    """The Langfuse uploader, or None when this run is not uploading."""
    public_key = os.environ.get('LANGFUSE_PUBLIC_KEY', '').strip()
    secret_key = os.environ.get('LANGFUSE_SECRET_KEY', '').strip()
    if not public_key or not secret_key or _env_bool('HITL_DISABLE_LANGFUSE'):
        return None
    return {
        'name': 'multiagent_langfuse',
        'args': {
            'public_key': public_key,
            'secret_key': secret_key,
            'base_url': os.environ.get('LANGFUSE_BASE_URL', '').strip(),
            'session_id': f'parallel-{name}',
        },
    }


__all__ = ['ENVIRONMENT', 'generate_configs']
