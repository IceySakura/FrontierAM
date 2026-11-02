"""Batch generator: the user simulator in front of the stock ``terminal_task``.

One terminal-bench instance per run, scored by the tests in its own container, driven
by ``hitl/agent_with_user`` rather than by handing the instruction to the worker.

The instruction becomes the single ``user_prompt``, so a run has one requirement and
the user decides across as many turns as it takes when to ``accept``.

``agent_config['working_dir']`` is required here: ``terminal_task`` sends only
``user_prompt``, and ``agent_config`` merges under the task's own kwargs.
``runtime_configs['main']['worker_venv']`` installs this module's sandbox code in
``main``, the container the task scores.

``TASK_BASE_DIR`` holds one directory per task; ``TASK2IMAGE_FILE`` maps a task name to
its image. A mapping entry with no task directory is skipped. Each task's working
directory comes from its own Dockerfile.

Two timeouts bound a run: ``TIMEOUT_MULTIPLIER`` sets the whole run's ceiling, which
the task enforces around the agent, and ``HITL_CC_TURN_TIMEOUT`` bounds one worker turn
(default 7200s). A multi-turn run wants the per-turn value well below the ceiling.
"""

import json
import logging
import os

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from nexus.extensions.tasks.hitl.actor.factory import backends
from nexus.extensions.tasks.hitl.agents.agent_with_user import AgentWithUserInput
from nexus.extensions.tasks.hitl.model import known_profiles
from nexus.extensions.tasks.hitl.user_agent import mcp_config
from nexus.extensions.tasks.hitl.user_agent.compact import DEFAULT_COMPACT_PCT
from nexus.extensions.tasks.hitl.user_agent.personas import load_persona_dirs, registered_personas
from nexus.extensions.tasks.hitl.user_agent.reminders import HEAVY_EVERY, LIGHT_EVERY
from nexus.extensions.tasks.hitl.utils.generator_util import (
    _build_openai_clients,
    _build_runtime_provider_config,
    _env_bool,
    _env_choice,
    _env_int,
    _env_list,
    _new_session_id,
    _require_env,
    client_imports,
    inject_cache_task_ids,
    with_worker_skills,
    worker_config_for,
    worker_usage_profile,
)
from nexus.runner.batch.batch_config import BatchItem
from nexus.runner.single import RunInput


logger = logging.getLogger(__name__)


AGENT_NAME = 'hitl/agent_with_user'
TASK_NAME = 'terminal_task'

# The task's own module, where ``terminal_task`` is registered.
TASK_IMPORT = 'nexus.extensions.tasks.terminal'

AGENT_IMPORT = 'nexus.extensions.tasks.hitl.agents.agent_with_user'

# Where each processor this generator names is registered. ``chat_tracker`` is
# upstream's own and resolves lazily, so it has no entry here.
PROCESSOR_IMPORTS = (
    'nexus.extensions.tasks.hitl.trace.usage',
    'nexus.extensions.tasks.hitl.trace.langfuse',
)

DEFAULT_WORKER_BACKEND = 'print_claude_code'
DEFAULT_PERSONA = 'default'

# The agent's timeout, in units of 60s.
DEFAULT_TIMEOUT_MULTIPLIER = 240

# The client names this generator builds, and the names the agent is told to ask for.
HUMAN_CLIENT = 'human'
WORKER_CLIENT = 'worker'

# ``branch_last`` keeps the per-request branch that carries the trace label.
CHAT_PROCESSOR: dict[str, Any] = {'chat': {'name': 'chat_tracker', 'args': {'tools_mode': 'branch_last'}}}

# Fields the agent declares but never reads. A turn's wall clock is
# ``worker_config['timeout']``.
DEAD_AGENT_FIELDS = frozenset({'timeout', 'npm_registry'})

# Variables a chain run script exports that this generator does not read, mapped to
# what to use instead. Refused at generation time.
CHAIN_ONLY_ENV: dict[str, str] = {
    'HITL_MAX_AGENT_ITERATIONS': 'HITL_MAX_TURNS (the generic agent calls the field max_turns)',
    'HITL_EVAL_TYPES': 'nothing -- terminal_task always scores by the tests in the task directory',
    'HITL_SCORE_POLICY': 'nothing -- terminal_task has one score',
    'HITL_UNIT_SCORE_POLICY': 'nothing -- terminal_task decides this itself',
    'HITL_SKIP_EVAL': 'nothing -- terminal_task always runs its verifier',
    'HITL_USER_COMPACT_AT': 'HITL_USER_CONTEXT_WINDOW with HITL_USER_COMPACT_PCT, which give the same threshold',
    'HITL_RUBRIC_EVAL_MODEL': 'nothing -- no rubric eval here',
    'SWE_DATASET_PATH': 'TASK_BASE_DIR with TASK2IMAGE_FILE',
}

# Every variable a run reads: this module, the helpers it calls, and the sandbox side
# of the worker.
ENVIRONMENT: dict[str, str] = {
    # -- dataset and selection --
    'TASK_BASE_DIR': 'required. Directory holding one subdirectory per terminal-bench task.',
    'TASK2IMAGE_FILE': 'required. JSON mapping task name -> container image.',
    'TASK_NAMES': 'comma-separated allowlist of task names.',
    'TASK_LIMIT': 'stop after this many tasks.',
    'EVAL_TIMES': 'how many times to run each task. Default 1.',
    # -- the run --
    'HITL_MAX_TURNS': "the user's turn cap. Default 120.",
    'TIMEOUT_MULTIPLIER': f'agent/test/docker timeout = 60s * this. Default {DEFAULT_TIMEOUT_MULTIPLIER}.',
    'RUN_ORACLE': 'run the task-provided solution instead of the agent. Default false.',
    # -- the user agent --
    'HITL_USER_PERSONA': f'persona name; refused unless registered. Default {DEFAULT_PERSONA!r}.',
    'HITL_USER_PERSONA_DIRS': 'directories holding persona .md files; one may replace a built-in by name.',
    'HITL_USER_SKILL_DIRS': 'extra skill directories to load.',
    'HITL_USER_EXTRA_TOOLS': "tools to hold beyond the persona's.",
    'HITL_USER_MCP_SERVERS': 'path to a JSON list of MCP servers the user simulator may call.',
    'HITL_USER_REMINDER_LIGHT': 'the one-line persona reminder. Blank stays quiet.',
    'HITL_USER_REMINDER_HEAVY_EVERY': f'turns between full persona reminders. Default {HEAVY_EVERY}.',
    'HITL_USER_REMINDER_LIGHT_EVERY': f'quiet turns before the light reminder fires. Default {LIGHT_EVERY}.',
    'HITL_USER_CONTEXT_WINDOW': "the user model's context size, tokens. 0 leaves the history uncompacted.",
    'HITL_USER_COMPACT_PCT': (
        f'share of that window the history may fill before it is summarised. Default {DEFAULT_COMPACT_PCT}.'
    ),
    # -- the user's client --
    'HITL_HUMAN_MODEL': 'model for the user simulator.',
    'HITL_HUMAN_BASE_URL': 'its endpoint.',
    'HITL_HUMAN_API_KEY': 'its key.',
    'HITL_HUMAN_LLM_TIMEOUT': 'its per-request timeout, seconds.',
    'HITL_HUMAN_CLIENT_PROFILE': (
        "the user client's protocol, which decides whether its requests carry prompt-cache markers. "
        "Default 'openai_basic', which carries none -- right for a gateway that caches server-side. "
        "Name 'bedrock_claude' or 'gcp_claude' for an Anthropic endpoint, where an unmarked request "
        'caches nothing.'
    ),
    'HITL_HUMAN_REASONING_EFFORT': (
        "the effort the user's requests name. Unset sends none, which every endpoint reached so far "
        "accepts; aihub's /standard/v1 refuses a request carrying function tools without it."
    ),
    # -- both clients --
    'OPENAI_API_KEY': 'fallback key when the per-role one is unset.',
    'OPENAI_BASE_URL': 'fallback endpoint.',
    'OPENAI_MODEL': 'fallback model.',
    'HITL_LLM_TIMEOUT': 'per-request timeout for any role without its own. Default 900.',
    # -- the worker --
    'HITL_WORKER_BACKEND': f'print_claude_code or pty_claude_code. Default {DEFAULT_WORKER_BACKEND!r}.',
    'HITL_WORKER_VERSION': 'harness release to install, whichever harness the backend names. Each has its own default.',
    'HITL_WORKER_TURN_TIMEOUT': 'one worker turn, seconds. Default 7200.',
    'HITL_WORKER_CONTEXT_WINDOW': (
        'the context window auto-compaction is sized against, tokens. Claude Code compacts at '
        'HITL_CC_AUTOCOMPACT_PCT of it (90% by default), Codex at min(that limit, this * 90%).'
    ),
    'HITL_WORKER_CLIENT_PROFILE': "the worker's provider protocol and usage profile. Default 'default'.",
    'HITL_WORKER_MODEL': "the worker's model. Falls back to OPENAI_MODEL.",
    'HITL_WORKER_BASE_URL': 'its endpoint. Falls back to OPENAI_BASE_URL.',
    'HITL_WORKER_API_KEY': 'its key. Falls back to OPENAI_API_KEY.',
    'HITL_WORKER_LLM_TIMEOUT': 'its per-request timeout, seconds.',
    'HITL_WORKER_THINKING_ENABLED': "thinking on. Default true; 0 stamps the profile's no-think args.",
    'HITL_WORKER_THINKING_PRESET': "proxy thinking preset. Overrides the profile's deploy default.",
    'HITL_WORKER_THINKING_TYPE': 'thinking type passed as a request arg.',
    'HITL_WORKER_REASONING_EFFORT': 'reasoning effort passed as a request arg.',
    'HITL_WORKER_REASONING_MODE': (
        "the Responses reasoning.mode a gpt worker sends. Default 'standard'; empty omits the field. "
        'Only the gpt_responses profile reads it.'
    ),
    'HITL_WORKER_CACHE_STRATEGY': 'prompt-cache strategy stamped onto the worker client.',
    'HITL_WORKER_TOOL_SEARCH': 'progressive tool disclosure, which is how Grep/Glob reach the worker. Default true.',
    'HITL_WORKER_TOOL_SEARCH_MODE': "value for ENABLE_TOOL_SEARCH. Default 'true'.",
    'HITL_WORKER_MEMORY_FILE': 'path to the CLAUDE.md placed for the worker.',
    'HITL_WORKER_SKILL_DIRS': (
        "skill directories packed into the sandbox at the backend's skills_path, and offered to the user by name."
    ),
    'HITL_WORKER_DISABLE_AGENT': 'also disallow the Agent tool.',
    'HITL_WORKER_APPEND_SYSTEM_PROMPT': "appended to the worker CLI's system prompt.",
    'HITL_WORKER_BINARY_URL': 'Claude Code binary URL.',
    'HITL_WORKER_CC_IDENTIFIER': 'the model name CC shows itself. Cosmetic.',
    'HITL_WORKER_HAIKU_MODEL': 'haiku-class model for CC internal calls.',
    'HITL_WORKER_NO_THINKING_MODEL': 'model for no-think requests.',
    'HITL_CC_BUILTIN_COMMANDS': "the harness's own commands to offer, comma-separated. None by default.",
    'HITL_CC_AUTOCOMPACT_PCT': 'auto-compact threshold percentage override.',
    'CLAUDE_CODE_MAX_OUTPUT_TOKENS': "one CC turn's output cap, incl. thinking.",
    'HITL_WORKER_MCP_SERVERS': 'path to a JSON list of MCP servers registered for the worker.',
    'HITL_WORKER_MCP_SEARCH': 'register the built-in search MCP server for the worker.',
    'SEARCH_API_KEY': 'required when HITL_WORKER_MCP_SEARCH=1.',
    'SEARCH_API_URL': 'endpoint for that search server.',
    # -- the sandbox --
    'SWE_RUNTIME_PROVIDER_NAME': "runtime provider. Default 'gongfeng'.",
    'SWE_RUNTIME_BASE_URL': 'its API base URL.',
    'SWE_RUNTIME_TOKEN': 'its token.',
    'SWE_RUNTIME_IMAGE': "the provider's own runtime image. Not the task's container image.",
    'SWE_RUNTIME_TIMEOUT': 'provider request timeout, seconds.',
    'SWE_RUNTIME_TTL': "sandbox lifetime as a Go duration ('12h', '1h30m'), max 168h. Backend default when unset.",
    'SWE_RUNTIME_CPU': 'cpu per sandbox. Default 2.',
    'SWE_RUNTIME_MEMORY': 'memory per sandbox, GB. Default 10.',
    'SWE_RPC_PROTOCOL': "http or ws. Default 'http'.",
    'SWE_RUNTIME_PURPOSE': "sandbox lifecycle tag: prod/eval/test. Default 'test'.",
    'GONGFENG_QUEUE_WAIT_MAX': 'client-side queue wait budget, seconds. Default 1200.',
    'GONGFENG_STARTUP_JITTER_MAX': 'random pre-acquire sleep, seconds. Default 12.',
    'SWE_WORKER_VENV': 'install the nexus wheel in the sandboxes. Default true, and the worker needs it.',
    # -- caching --
    'HITL_CACHE_TASK_ID_ENABLED': 'inject per-run cache task ids into the clients.',
    'HITL_CACHE_TASK_ID_ROLES': 'which client roles get one.',
    'APP_ID': 'mixed into the cache task id seed. Optional.',
    # -- tracing --
    'LANGFUSE_PUBLIC_KEY': 'Langfuse upload is configured only when this and the secret are both set.',
    'LANGFUSE_SECRET_KEY': 'see above.',
    'LANGFUSE_BASE_URL': 'Langfuse endpoint. Also read as LANGFUSE_HOST by the processor.',
    'HITL_DISABLE_LANGFUSE': 'skip Langfuse upload even when credentials are present.',
}


def generate_configs() -> Iterator[BatchItem]:
    """One ``BatchItem`` per selected task, repeated ``EVAL_TIMES`` times."""
    task_base_dir = Path(
        _require_env('TASK_BASE_DIR', 'TASK_BASE_DIR required (directory of terminal-bench task folders)')
    )
    _check_chain_only_env()

    task2image = _load_task2image()
    available = _available_tasks(task_base_dir, task2image)
    if not available:
        raise ValueError(f'no task under {task_base_dir} has both a task2image entry and an instruction.md')

    name_filter = set(_env_list('TASK_NAMES'))
    unknown = sorted(name_filter - set(available))
    if unknown:
        raise ValueError(f'TASK_NAMES lists {unknown}, which are not available. Available: {len(available)} tasks')
    selected = [name for name in available if not name_filter or name in name_filter]
    limit = _env_int('TASK_LIMIT') or None
    eval_times = _env_int('EVAL_TIMES', default=1)

    runtime_provider_name = os.environ.get('SWE_RUNTIME_PROVIDER_NAME', '').strip() or 'gongfeng'
    max_turns = _env_int('HITL_MAX_TURNS', default=120)
    timeout_seconds = 60 * _env_int('TIMEOUT_MULTIPLIER', default=DEFAULT_TIMEOUT_MULTIPLIER)
    run_oracle = _env_bool('RUN_ORACLE')

    worker_venv = _env_bool('SWE_WORKER_VENV', default=True)
    worker_profile = _env_choice('HITL_WORKER_CLIENT_PROFILE', 'default', known_profiles())
    worker_backend = _env_choice('HITL_WORKER_BACKEND', DEFAULT_WORKER_BACKEND, backends())

    persona_dirs = _env_list('HITL_USER_PERSONA_DIRS')
    load_persona_dirs(persona_dirs)
    persona = _env_choice('HITL_USER_PERSONA', DEFAULT_PERSONA, registered_personas())
    skill_dirs = _env_list('HITL_USER_SKILL_DIRS')
    worker_skill_dirs = _env_list('HITL_WORKER_SKILL_DIRS')
    extra_tools = _env_list('HITL_USER_EXTRA_TOOLS')
    user_mcp_servers = mcp_config.user_servers()
    reminder_light = os.environ.get('HITL_USER_REMINDER_LIGHT', '').strip()
    reminder_heavy_every = _env_int('HITL_USER_REMINDER_HEAVY_EVERY')
    reminder_light_every = _env_int('HITL_USER_REMINDER_LIGHT_EVERY')
    context_window = _env_int('HITL_USER_CONTEXT_WINDOW')
    compact_pct = _env_int('HITL_USER_COMPACT_PCT', default=DEFAULT_COMPACT_PCT)

    session_id = _new_session_id()
    # No rubric-eval client: this task scores by the tests in its own directory.
    openai_clients = _build_openai_clients(['unit_test'], worker_profile, session_id=session_id)
    base_worker_config = worker_config_for(worker_backend, worker_profile, session_id=session_id)
    if worker_skill_dirs:
        base_worker_config = with_worker_skills(base_worker_config, worker_skill_dirs)
    runtime_provider_config = _build_runtime_provider_config()
    _check_runtime_provider(runtime_provider_name, runtime_provider_config)
    _check_processors_importable(PROCESSOR_IMPORTS)
    imports = [TASK_IMPORT, AGENT_IMPORT, *PROCESSOR_IMPORTS, *client_imports(worker_profile)]

    cache_task_id_enabled = _env_bool('HITL_CACHE_TASK_ID_ENABLED')
    cache_task_id_roles = _env_list('HITL_CACHE_TASK_ID_ROLES') or None

    logger.info('[hitl/terminal/generator] %d task(s) selected from %s', len(selected), task_base_dir)

    count = 0
    for task_name in selected:
        task_folder = task_base_dir / task_name
        instruction = _read_instruction(task_folder)
        working_dir = _task_workdir(task_folder)

        for attempt in range(eval_times):
            run_clients = (
                inject_cache_task_ids(openai_clients, roles=cache_task_id_roles)
                if cache_task_id_enabled
                else openai_clients
            )

            # ``working_dir`` reaches the agent here because the task does not send
            # it; see the module docstring. It also goes to the provider, which is
            # where the container's own cwd is set.
            worker_config = {**base_worker_config, 'working_dir': working_dir}
            agent_config: dict[str, Any] = {
                'working_dir': working_dir,
                'user_client_name': HUMAN_CLIENT,
                'worker_client_name': WORKER_CLIENT,
                'worker_backend': worker_backend,
                'worker_config': worker_config,
                'persona': persona,
                'max_turns': max_turns,
            }
            if skill_dirs:
                agent_config['skill_dirs'] = skill_dirs
            if persona_dirs:
                agent_config['persona_dirs'] = persona_dirs
            if extra_tools:
                agent_config['extra_tools'] = extra_tools
            if user_mcp_servers:
                agent_config['mcp_servers'] = user_mcp_servers
            if reminder_light:
                agent_config['reminder_light'] = reminder_light
            if reminder_heavy_every:
                agent_config['reminder_heavy_every'] = reminder_heavy_every
            if reminder_light_every:
                agent_config['reminder_light_every'] = reminder_light_every
            if context_window:
                agent_config['context_window'] = context_window
                agent_config['compact_pct'] = compact_pct
            _check_agent_config(agent_config)
            _check_clients(agent_config, run_clients)

            run_id = task_name if eval_times == 1 else f'{task_name}__try{attempt + 1}'
            yield BatchItem(
                config=RunInput(
                    imports=imports,
                    task_name=TASK_NAME,
                    task_config={
                        'message': instruction,
                        'task_dir': str(task_folder),
                        'run_oracle': run_oracle,
                        'agent_timeout': timeout_seconds,
                        'test_timeout': timeout_seconds,
                        'docker_timeout': timeout_seconds,
                    },
                    agent_name=AGENT_NAME,
                    agent_config=agent_config,
                    runtime_provider_name=runtime_provider_name,
                    runtime_provider_config={
                        **runtime_provider_config,
                        'image': task2image[task_name],
                        'working_dir': working_dir,
                    },
                    runtime_configs={'main': {'worker_venv': worker_venv}},
                    openai_clients=run_clients,
                    processors=_processors(worker_profile, task_name),
                ),
                run_id=run_id,
                metadata={
                    'task': task_name,
                    'attempt': attempt + 1,
                    'image': task2image[task_name],
                    'working_dir': working_dir,
                    'persona': persona,
                    'worker_backend': worker_backend,
                },
            )

            count += 1
            if limit and count >= limit:
                return


def _load_task2image() -> dict[str, str]:
    """The task name -> image mapping this run scores against."""
    path = _require_env('TASK2IMAGE_FILE', 'TASK2IMAGE_FILE required (JSON mapping task name -> image)')
    with open(path, encoding='utf-8') as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f'{path} must hold a non-empty JSON object mapping task name to image')
    logger.info('[hitl/terminal/generator] loaded %d task2image entries from %s', len(mapping), path)
    return {str(name): str(image) for name, image in mapping.items()}


def _available_tasks(task_base_dir: Path, task2image: Mapping[str, str]) -> list[str]:
    """Task names that have both an image and a directory holding an instruction."""
    return sorted(name for name in task2image if (task_base_dir / name / 'instruction.md').exists())


def _read_instruction(task_folder: Path) -> str:
    """The task's instruction, which becomes the user's one requirement."""
    text = (task_folder / 'instruction.md').read_text(encoding='utf-8').strip()
    if not text:
        raise ValueError(f'{task_folder / "instruction.md"} is empty; there would be nothing to ask for')
    return text


def _task_workdir(task_folder: Path) -> str:
    """The last ``WORKDIR`` in the task's Dockerfile, or ``/app``.

    Not uniform across terminal-bench: most tasks end on ``/app``, a few elsewhere.
    """
    dockerfile = task_folder / 'environment/Dockerfile'
    workdir = '/app'
    if dockerfile.exists():
        for line in dockerfile.read_text(encoding='utf-8').splitlines():
            stripped = line.strip()
            if stripped.startswith('WORKDIR '):
                workdir = stripped[len('WORKDIR ') :].strip()
    return workdir


def _check_runtime_provider(provider_name: str, config: Mapping[str, Any]) -> None:
    """Refuse a gongfeng config with no endpoint or token.

    Both come from ``SWE_RUNTIME_*``, not the ``GONGFENG_*`` names the terminal run
    script uses.
    """
    if provider_name != 'gongfeng':
        return
    missing = [
        env
        for env, key in (('SWE_RUNTIME_BASE_URL', 'api_base_url'), ('SWE_RUNTIME_TOKEN', 'token'))
        if not config.get(key)
    ]
    if missing:
        raise ValueError(
            f'the gongfeng runtime provider needs {missing}; without them every sandbox '
            'acquisition fails after the batch has started'
        )


def _check_processors_importable(processor_imports: Sequence[str]) -> None:
    """Refuse a run whose named processors are not in the registry."""
    import importlib

    from nexus.trace.processors.base import span_processor_registry

    for module in processor_imports:
        try:
            importlib.import_module(module)
        except ImportError as exc:
            raise ValueError(f'imports lists {module!r}, which this host cannot import: {exc}') from exc

    wanted = sorted({config['name'] for config in _processors('default', 'probe').values()})
    for name in wanted:
        try:
            span_processor_registry.get(name)
        except Exception as exc:  # noqa: BLE001 - re-raised with what to do about it
            raise ValueError(
                f'processor {name!r} is named by this generator but not in the registry after '
                f'importing {list(processor_imports)}: {type(exc).__name__}. '
                'Add the module that registers it.'
            ) from exc


def _processors(worker_profile: str, task_name: str) -> dict[str, Any]:
    """The run-level processors: the chat tracker, token accounting, and Langfuse.

    All three read ``openai_completion`` spans and group by the ``_chat_phase`` label.
    The usage profile is per client, keyed by the name the agent asks for; the user's
    client stays on ``default``.
    """
    processors: dict[str, Any] = dict(CHAT_PROCESSOR)
    processors['usage'] = {
        'name': 'hitl_usage',
        'args': {
            'profiles': {WORKER_CLIENT: worker_usage_profile(worker_profile)},
            'default_profile': 'default',
        },
    }
    langfuse = _langfuse_processor(task_name)
    if langfuse is not None:
        processors['langfuse'] = langfuse
    return processors


def _langfuse_processor(task_name: str) -> dict[str, Any] | None:
    """The Langfuse processor config, or None when the run should not upload."""
    if _env_bool('HITL_DISABLE_LANGFUSE'):
        return None
    if not (os.environ.get('LANGFUSE_PUBLIC_KEY') and os.environ.get('LANGFUSE_SECRET_KEY')):
        return None
    return {
        'name': 'multiagent_langfuse',
        'args': {
            'session_id': task_name,
            'trace_name': task_name,
            'tags': ['hitl', 'terminal_user_sim'],
            'trace_metadata': {'task': task_name},
        },
    }


def _check_chain_only_env(environ: Mapping[str, str] | None = None) -> None:
    """Refuse a variable from another recipe that this generator does not read."""
    env = os.environ if environ is None else environ
    offenders = {key: instead for key, instead in CHAIN_ONLY_ENV.items() if env.get(key, '').strip()}
    if offenders:
        lines = '\n'.join(f'  {key} -> use {instead}' for key, instead in sorted(offenders.items()))
        raise ValueError(f'these are set but mean nothing to this generator:\n{lines}')


def _check_agent_config(agent_config: dict[str, Any]) -> None:
    """Refuse a key the agent would drop, and one it declares but never reads.

    Checked against the agent's own ``model_fields``.
    """
    declared = set(AgentWithUserInput.model_fields)
    unknown = sorted(set(agent_config) - declared)
    if unknown:
        raise ValueError(
            f'{AGENT_NAME} does not declare {unknown}, so _build_input would drop them silently '
            f'and the run would report a number anyway. It declares: {", ".join(sorted(declared))}'
        )
    dead = sorted(DEAD_AGENT_FIELDS & set(agent_config))
    if dead:
        raise ValueError(
            f'{AGENT_NAME} declares {dead} but never reads them; '
            "set worker_config['timeout'] for a turn's wall clock instead"
        )
    for required in ('working_dir', 'user_client_name', 'worker_client_name'):
        if not agent_config.get(required):
            raise ValueError(
                f'{required!r} must be in agent_config: terminal_task sends only user_prompt, '
                'so anything else the agent requires has to come from here'
            )


def _check_clients(agent_config: dict[str, Any], openai_clients: dict[str, Any]) -> None:
    """Refuse a client name the agent asks for and this run does not build."""
    asked = {agent_config[key] for key in ('user_client_name', 'worker_client_name') if key in agent_config}
    missing = sorted(asked - set(openai_clients))
    if missing:
        raise ValueError(
            f'{AGENT_NAME} is configured to use clients {missing}, which this run does not build. '
            f'Built: {", ".join(sorted(openai_clients))}'
        )


def get_batch_metadata() -> dict[str, Any]:
    return {
        'task_type': 'hitl/terminal_user_sim',
        'task_name': TASK_NAME,
        'agent_name': AGENT_NAME,
        'task_base_dir': os.environ.get('TASK_BASE_DIR', ''),
        'persona': os.environ.get('HITL_USER_PERSONA', DEFAULT_PERSONA),
        'worker_backend': os.environ.get('HITL_WORKER_BACKEND', DEFAULT_WORKER_BACKEND),
        'worker_profile': os.environ.get('HITL_WORKER_CLIENT_PROFILE', '').strip() or 'default',
    }


__all__ = ['AGENT_NAME', 'ENVIRONMENT', 'TASK_NAME', 'generate_configs', 'get_batch_metadata']
