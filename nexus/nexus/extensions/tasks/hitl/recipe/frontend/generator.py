"""Batch generator: the user simulator in front of the stock ``frontend_task``.

One greenfield web build per dataset row, driven by ``hitl/agent_with_user`` rather
than by handing the specification straight to the worker.

Each row becomes a one-requirement run: the specification is the single
``user_prompt``, and the user decides across as many turns as it takes when to
``accept``.

``FRONTEND_IMAGE`` carries the Node toolchain; the task installs none of it.
``runtime_configs['main']['worker_venv']`` installs this package's sandbox code in the
runtime the worker manager uses. ``working_dir`` stays out of ``agent_config`` --
``frontend_task`` sends it to ``run_agent``.

``FRONTEND_WORKER_SKILL`` prefixes the specification with ``/<name>``;
``HITL_WORKER_SKILL_DIRS`` packs skill directories into the sandbox. Both are off
by default.

Scoring runs only when ``FRONTEND_TASK_DIR`` names a directory holding ``tests/``.

The ``NEXUS_STOP_HOOK_*`` variables are refused: the turns come from the user
simulator here.
"""

import json
import logging
import os

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from nexus.extensions.tasks.hitl.actor.factory import backends
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
TASK_NAME = 'frontend_task'

# The task's own module, where ``frontend_task`` is registered.
TASK_IMPORT = 'nexus.extensions.tasks.frontend.task'

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

# The fields a row's specification is read from, in order.
PROMPT_FIELDS = ('problem_statement', 'prompt', 'text', 'query', 'question')

# Variables another recipe exports that this one does not read, mapped to what to use
# instead. Refused at generation time.
FOREIGN_ENV: dict[str, str] = {
    # -- the stop-hook turn loop --
    'NEXUS_STOP_HOOK_ENABLE': 'nothing -- the user simulator drives the turns here',
    'NEXUS_STOP_HOOK_PERSONA': 'HITL_USER_PERSONA',
    'NEXUS_STOP_HOOK_ITERATIVE_TURNS': 'HITL_MAX_TURNS',
    'NEXUS_STOP_HOOK_PERSONAS_DIR': 'HITL_USER_PERSONA_DIRS',
    'NEXUS_STOP_HOOK_SUB_CC_MEMORY': 'nothing -- the user keeps one session across turns',
    'FRONTEND_LLM_ROUTER_ENABLE': 'nothing -- one worker client, named by worker_client_name',
    # -- the task recipe's own selection and runtime knobs --    'FRONTEND_DATASET_PATH': 'FRONTEND_DATASET',
    'FRONTEND_LIMIT': 'FRONTEND_TASK_LIMIT',
    'FRONTEND_SKIP_EVAL': 'nothing -- scoring here follows FRONTEND_TASK_DIR',
    'FRONTEND_RUNTIME_CONTAINER_IMAGE': 'FRONTEND_IMAGE',
    'FRONTEND_RUNTIME_PROVIDER_NAME': 'SWE_RUNTIME_PROVIDER_NAME',
    'FRONTEND_RUNTIME_API_BASE_URL': 'SWE_RUNTIME_BASE_URL',
    'FRONTEND_RUNTIME_TOKEN': 'SWE_RUNTIME_TOKEN',
    'FRONTEND_WORKING_DIR': 'FRONTEND_WORK_DIR',
    'SKILL_NAME': 'FRONTEND_WORKER_SKILL',
    'FRONTEND_SKILLS_DIR': 'HITL_WORKER_SKILL_DIRS',
    'FRONTEND_WORKER_SKILL_DIRS': 'HITL_WORKER_SKILL_DIRS',
    # -- chain-recipe leftovers --
    'HITL_MAX_AGENT_ITERATIONS': 'HITL_MAX_TURNS (the generic agent calls the field max_turns)',
    'HITL_EVAL_TYPES': 'nothing -- frontend_task scores by the tests in a task directory, or not at all',
    'HITL_SCORE_POLICY': 'nothing -- frontend_task has one score',
    'HITL_SKIP_EVAL': 'nothing -- scoring here follows FRONTEND_TASK_DIR',
    'HITL_USER_COMPACT_AT': 'HITL_USER_CONTEXT_WINDOW with HITL_USER_COMPACT_PCT, which give the same threshold',
    'SWE_DATASET_PATH': 'FRONTEND_DATASET',
    'TASK_BASE_DIR': 'FRONTEND_DATASET',
}

# Every variable a run reads: this module, the helpers it calls, and the sandbox side
# of the worker.
ENVIRONMENT: dict[str, str] = {
    # -- dataset and selection --
    'FRONTEND_DATASET': 'required. JSONL file, one specification per line.',
    'FRONTEND_INSTANCE_IDS': 'comma-separated allowlist of instance ids.',
    'FRONTEND_TASK_LIMIT': 'stop after this many rows.',
    'EVAL_TIMES': 'how many times to run each row. Default 1.',
    # -- the container --
    'FRONTEND_IMAGE': 'required. Image carrying the Node toolchain the build needs.',
    'FRONTEND_WORK_DIR': "where the worker builds. Default '/workspace'.",
    # -- scoring --
    'FRONTEND_TASK_DIR': 'directory holding tests/test.sh. Unset means trajectories and no score.',
    'FRONTEND_TEST_TIMEOUT': "the verifier's wall clock, seconds. Default 600.",
    # -- the run --
    'HITL_MAX_TURNS': "the user's turn cap. Default 120.",
    'TIMEOUT_MULTIPLIER': f'agent timeout = 60s * this. Default {DEFAULT_TIMEOUT_MULTIPLIER}.',
    # -- the worker's skill, optional --
    'FRONTEND_WORKER_SKILL': "prefix the specification with '/<name>' to trigger that skill.",
    'HITL_WORKER_SKILL_DIRS': (
        "skill directories packed into the sandbox at the backend's skills_path, and offered to the user by name."
    ),
    # -- the user agent --
    'HITL_USER_PERSONA': f'persona name; refused unless registered. Default {DEFAULT_PERSONA!r}.',
    'HITL_USER_PERSONA_DIRS': 'directories holding persona .md files; one may replace a built-in by name.',
    'HITL_USER_SKILL_DIRS': 'extra skill directories for the user, not the worker.',
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
    'SWE_RUNTIME_IMAGE': "the provider's own runtime image.",
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
    """One ``BatchItem`` per selected dataset row, repeated ``EVAL_TIMES`` times."""
    dataset_path = Path(_require_env('FRONTEND_DATASET', 'FRONTEND_DATASET required (JSONL of specifications)'))
    image = _require_env('FRONTEND_IMAGE', 'FRONTEND_IMAGE required (the build needs the Node toolchain)')
    _check_foreign_env()

    rows = _load_dataset(dataset_path)
    id_filter = set(_env_list('FRONTEND_INSTANCE_IDS'))
    unknown = sorted(id_filter - {instance_id for instance_id, _ in rows})
    if unknown:
        raise ValueError(f'FRONTEND_INSTANCE_IDS lists {unknown}, which {dataset_path} does not hold')
    selected = [row for row in rows if not id_filter or row[0] in id_filter]
    limit = _env_int('FRONTEND_TASK_LIMIT') or None
    eval_times = _env_int('EVAL_TIMES', default=1)

    runtime_provider_name = os.environ.get('SWE_RUNTIME_PROVIDER_NAME', '').strip() or 'gongfeng'
    working_dir = os.environ.get('FRONTEND_WORK_DIR', '').strip() or '/workspace'
    max_turns = _env_int('HITL_MAX_TURNS', default=120)
    timeout_seconds = 60 * _env_int('TIMEOUT_MULTIPLIER', default=DEFAULT_TIMEOUT_MULTIPLIER)

    task_dir = os.environ.get('FRONTEND_TASK_DIR', '')
    test_timeout = _env_int('FRONTEND_TEST_TIMEOUT', default=600)
    worker_skill = os.environ.get('FRONTEND_WORKER_SKILL', '').strip().lstrip('/')
    worker_skill_dirs = _env_list('HITL_WORKER_SKILL_DIRS')

    worker_venv = _env_bool('SWE_WORKER_VENV', default=True)
    worker_profile = _env_choice('HITL_WORKER_CLIENT_PROFILE', 'default', known_profiles())
    worker_backend = _env_choice('HITL_WORKER_BACKEND', DEFAULT_WORKER_BACKEND, backends())

    persona_dirs = _env_list('HITL_USER_PERSONA_DIRS')
    load_persona_dirs(persona_dirs)
    persona = _env_choice('HITL_USER_PERSONA', DEFAULT_PERSONA, registered_personas())
    skill_dirs = _env_list('HITL_USER_SKILL_DIRS')
    extra_tools = _env_list('HITL_USER_EXTRA_TOOLS')
    user_mcp_servers = mcp_config.user_servers()
    reminder_light = os.environ.get('HITL_USER_REMINDER_LIGHT', '').strip()
    reminder_heavy_every = _env_int('HITL_USER_REMINDER_HEAVY_EVERY')
    reminder_light_every = _env_int('HITL_USER_REMINDER_LIGHT_EVERY')
    context_window = _env_int('HITL_USER_CONTEXT_WINDOW')
    compact_pct = _env_int('HITL_USER_COMPACT_PCT', default=DEFAULT_COMPACT_PCT)

    session_id = _new_session_id()
    # No rubric-eval client: this task either scores by its own tests or not at all.
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

    logger.info('[hitl/frontend/generator] %d row(s) selected from %s', len(selected), dataset_path)

    count = 0
    for instance_id, specification in selected:
        prompt = f'/{worker_skill} {specification}' if worker_skill else specification

        for attempt in range(eval_times):
            run_clients = (
                inject_cache_task_ids(openai_clients, roles=cache_task_id_roles)
                if cache_task_id_enabled
                else openai_clients
            )

            worker_config = {**base_worker_config, 'working_dir': working_dir}
            # No ``working_dir``: ``frontend_task`` sends it to ``run_agent`` itself.
            agent_config: dict[str, Any] = {
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

            task_config: dict[str, Any] = {
                'instance_id': instance_id,
                'prompt': prompt,
                'working_dir': working_dir,
                'image_override': image,
                'timeout_seconds': timeout_seconds,
            }
            if task_dir:
                task_config['task_dir'] = task_dir
                task_config['test_timeout'] = test_timeout
            else:
                # The task defaults ``run_eval`` on; the demo datasets carry no tests/.
                task_config['run_eval'] = False

            run_id = instance_id if eval_times == 1 else f'{instance_id}__try{attempt + 1}'
            yield BatchItem(
                config=RunInput(
                    imports=imports,
                    task_name=TASK_NAME,
                    task_config=task_config,
                    agent_name=AGENT_NAME,
                    agent_config=agent_config,
                    runtime_provider_name=runtime_provider_name,
                    runtime_provider_config={
                        **runtime_provider_config,
                        'image': image,
                        'working_dir': working_dir,
                    },
                    runtime_configs={'main': {'worker_venv': worker_venv}},
                    openai_clients=run_clients,
                    processors=_processors(worker_profile, instance_id),
                ),
                run_id=run_id,
                metadata={
                    'instance_id': instance_id,
                    'attempt': attempt + 1,
                    'image': image,
                    'working_dir': working_dir,
                    'persona': persona,
                    'worker_backend': worker_backend,
                    'worker_skill': worker_skill,
                    'scored': bool(task_dir),
                },
            )

            count += 1
            if limit and count >= limit:
                return


def _load_dataset(dataset_path: Path) -> list[tuple[str, str]]:
    """The ``(instance_id, specification)`` pairs this run drives.

    A row with no specification, and a repeated id, are both refused.
    """
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    with open(dataset_path, encoding='utf-8', errors='replace') as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{dataset_path}:{number} is not JSON: {exc}') from exc
            specification = next((str(record[field]) for field in PROMPT_FIELDS if record.get(field)), '')
            if not specification:
                raise ValueError(f'{dataset_path}:{number} has none of {list(PROMPT_FIELDS)}; nothing to ask for')
            instance_id = str(record.get('instance_id') or '') or f'row{number}'
            if instance_id in seen:
                raise ValueError(f'{dataset_path}:{number} repeats instance_id {instance_id!r}, which names the run')
            seen.add(instance_id)
            rows.append((instance_id, specification))
    if not rows:
        raise ValueError(f'{dataset_path} holds no rows')
    logger.info('[hitl/frontend/generator] loaded %d row(s) from %s', len(rows), dataset_path)
    return rows


def _check_runtime_provider(provider_name: str, config: Mapping[str, Any]) -> None:
    """Refuse a gongfeng config with no endpoint or token.

    Both come from ``SWE_RUNTIME_*``, not the ``FRONTEND_RUNTIME_*`` names the task's
    own run script uses.
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


def _processors(worker_profile: str, instance_id: str) -> dict[str, Any]:
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
    langfuse = _langfuse_processor(instance_id)
    if langfuse is not None:
        processors['langfuse'] = langfuse
    return processors


def _langfuse_processor(instance_id: str) -> dict[str, Any] | None:
    """The Langfuse processor config, or None when the run should not upload."""
    if _env_bool('HITL_DISABLE_LANGFUSE'):
        return None
    if not (os.environ.get('LANGFUSE_PUBLIC_KEY') and os.environ.get('LANGFUSE_SECRET_KEY')):
        return None
    return {
        'name': 'multiagent_langfuse',
        'args': {
            'session_id': instance_id,
            'trace_name': instance_id,
            'tags': ['hitl', 'frontend_user_sim'],
            'trace_metadata': {'instance_id': instance_id},
        },
    }


def _check_foreign_env(environ: Mapping[str, str] | None = None) -> None:
    """Refuse a variable from another recipe that this generator does not read."""
    env = os.environ if environ is None else environ
    offenders = {key: instead for key, instead in FOREIGN_ENV.items() if env.get(key, '').strip()}
    if offenders:
        lines = '\n'.join(f'  {key} -> use {instead}' for key, instead in sorted(offenders.items()))
        raise ValueError(f'these are set but mean nothing to this generator:\n{lines}')


def _check_agent_config(agent_config: dict[str, Any]) -> None:
    """Refuse a key the agent would drop, and one it declares but never reads.

    Checked against the agent's own ``model_fields``.
    """
    from nexus.extensions.tasks.hitl.agents.agent_with_user import AgentWithUserInput

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
    for required in ('user_client_name', 'worker_client_name'):
        if not agent_config.get(required):
            raise ValueError(f'{required!r} must be in agent_config: the agent defaults it to a client nobody builds')


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
        'task_type': 'hitl/frontend_user_sim',
        'task_name': TASK_NAME,
        'agent_name': AGENT_NAME,
        'dataset': os.environ.get('FRONTEND_DATASET', ''),
        'image': os.environ.get('FRONTEND_IMAGE', ''),
        'persona': os.environ.get('HITL_USER_PERSONA', DEFAULT_PERSONA),
        'worker_backend': os.environ.get('HITL_WORKER_BACKEND', DEFAULT_WORKER_BACKEND),
        'worker_profile': os.environ.get('HITL_WORKER_CLIENT_PROFILE', '').strip() or 'default',
        'worker_skill': os.environ.get('FRONTEND_WORKER_SKILL', ''),
    }


__all__ = ['AGENT_NAME', 'ENVIRONMENT', 'TASK_NAME', 'generate_configs', 'get_batch_metadata']
