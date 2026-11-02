"""Batch generator: the user simulator in front of the stock ``swe_task``.

One SWE-bench instance per run, scored by its own unit tests, driven by
``hitl/agent_with_user`` rather than by handing the statement to the worker.

The dataset is raw SWE data: no ``problem_statements`` chain and no ``rubrics_yaml``.
The statement becomes the single ``user_prompt``, so a run has one requirement and the
user decides when it is done.

Settings this generator supplies:

``agent_config`` client names
    ``human`` and ``worker``; the agent defaults both to ``main``.

``runtime_configs['main']['worker_venv']``
    Installs hitl's sandbox modules in ``main``, the runtime the worker manager uses.

``max_turns``
    The agent's own field name for the user's turn cap. The chain agent calls it
    ``max_agent_iterations``.

``worker_config``
    Everything about the worker: ``swe_task`` sends only ``user_prompt`` and
    ``working_dir``.
"""

import logging
import os
import random

from collections.abc import Iterator, Mapping, Sequence
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
    _load_dataset,
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
TASK_NAME = 'swe_task'

# The task's own package.
TASK_IMPORT = 'nexus.extensions.tasks.swe'

AGENT_IMPORT = 'nexus.extensions.tasks.hitl.agents.agent_with_user'

# Where each processor this generator names is registered. ``chat_tracker`` is
# upstream's own and resolves lazily, so it has no entry here.
PROCESSOR_IMPORTS = (
    'nexus.extensions.tasks.hitl.trace.usage',
    'nexus.extensions.tasks.hitl.trace.langfuse',
)

DEFAULT_WORKER_BACKEND = 'print_claude_code'
DEFAULT_PERSONA = 'default'

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
    'HITL_AGENT_TIMEOUT': 'nothing -- swe_task does not wrap run_agent in a timeout; bound the run with '
    "HITL_MAX_TURNS and worker_config['timeout']",
    'HITL_EVAL_TYPES': 'nothing -- swe_task always scores by the instance unit tests',
    'HITL_SKIP_EVAL': 'HITL_SKIP_EMPTY_PATCH_EVAL, or task_config skip_infer_patch_eval',
    'HITL_SCORE_POLICY': 'nothing -- swe_task has one score',
    'HITL_UNIT_SCORE_POLICY': 'nothing -- swe_task decides this itself (use_partial_score)',
    'HITL_RUBRIC_EVAL_MODEL': 'nothing -- no rubric eval here',
    'HITL_RUBRIC_EVAL_BASE_URL': 'nothing -- no rubric eval here',
    'HITL_RUBRIC_EVAL_API_KEY': 'nothing -- no rubric eval here',
}


# Every variable a run reads: this module, the shared helpers in
# ``utils/generator_util.py``, and the sandbox side of the worker.
ENVIRONMENT: dict[str, str] = {
    # -- dataset and selection --
    'SWE_DATASET_PATH': 'required. Raw SWE dataset JSONL path.',
    'SWE_INSTANCE_IDS': 'comma-separated allowlist of instance ids.',
    'SWE_LIMIT': 'stop after this many instances.',
    'HITL_SKIP_FIRST_N': 'drop this many instances from the front.',
    'HITL_RANDOMIZE_INSTANCES': 'shuffle before limiting.',
    'HITL_RANDOM_SEED': 'seed for that shuffle. Default 0.',
    # -- the run --
    'HITL_MAX_TURNS': "the user's turn cap. Default 120.",
    'HITL_EVAL_TIMEOUT': "the task's eval timeout, seconds. Default 3600.",
    'HITL_SKIP_EMPTY_PATCH_EVAL': 'skip scoring a run that changed nothing. Default true.',
    'SWE_EVAL_MAX_ATTEMPTS': 'retries for the eval step. Default 1.',
    # -- the user agent --
    'HITL_USER_PERSONA': f'persona name; refused unless registered. Default {DEFAULT_PERSONA!r}.',
    'HITL_USER_PERSONA_DIRS': 'directories holding persona .md files; one may replace a built-in by name.',
    'HITL_USER_SKILL_DIRS': 'extra skill directories to load.',
    'HITL_USER_EXTRA_TOOLS': "tools to hold beyond the persona's, from toolset.CATALOGUE.",
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
    'CLAUDE_CODE_MAX_OUTPUT_TOKENS': "one CC turn's output cap, incl. thinking. Passed through by envs._PASSTHROUGH.",
    'HITL_WORKER_MCP_SERVERS': 'path to a JSON list of MCP servers registered for the worker.',
    'HITL_WORKER_MCP_SEARCH': 'register the built-in search MCP server for the worker.',
    'SEARCH_API_KEY': 'required when HITL_WORKER_MCP_SEARCH=1.',
    'SEARCH_API_URL': 'endpoint for that search server.',
    # -- the sandbox --
    'SWE_RUNTIME_PROVIDER_NAME': "runtime provider. Default 'gongfeng'.",
    'SWE_RUNTIME_BASE_URL': 'its API base URL.',
    'SWE_RUNTIME_TOKEN': 'its token.',
    'SWE_RUNTIME_IMAGE': "the provider's own runtime image. Not the instance image.",
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
    """One ``BatchItem`` per selected instance."""
    dataset_path = _require_env('SWE_DATASET_PATH', 'SWE_DATASET_PATH required (raw SWE dataset JSONL path)')
    _check_chain_only_env()

    instance_filter = set(_env_list('SWE_INSTANCE_IDS'))
    limit = _env_int('SWE_LIMIT') or None
    skip_first_n = _env_int('HITL_SKIP_FIRST_N')
    randomize = _env_bool('HITL_RANDOMIZE_INSTANCES')
    random_seed = _env_int('HITL_RANDOM_SEED', default=0)

    runtime_provider_name = os.environ.get('SWE_RUNTIME_PROVIDER_NAME', '').strip() or 'gongfeng'
    max_turns = _env_int('HITL_MAX_TURNS', default=120)
    eval_timeout = _env_int('HITL_EVAL_TIMEOUT', default=3600)
    eval_max_attempts = _env_int('SWE_EVAL_MAX_ATTEMPTS', default=1)
    skip_empty_patch_eval = _env_bool('HITL_SKIP_EMPTY_PATCH_EVAL', default=True)

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
    # No rubric-eval client: this task scores by the instance's own tests.
    openai_clients = _build_openai_clients(['unit_test'], worker_profile, session_id=session_id)
    worker_config = worker_config_for(worker_backend, worker_profile, session_id=session_id)
    if worker_skill_dirs:
        worker_config = with_worker_skills(worker_config, worker_skill_dirs)
    runtime_provider_config = _build_runtime_provider_config()
    _check_runtime_provider(runtime_provider_name, runtime_provider_config)
    processor_imports = list(PROCESSOR_IMPORTS)
    _check_processors_importable(processor_imports)
    imports = [TASK_IMPORT, AGENT_IMPORT, *processor_imports, *client_imports(worker_profile)]

    cache_task_id_enabled = _env_bool('HITL_CACHE_TASK_ID_ENABLED')
    cache_task_id_roles = _env_list('HITL_CACHE_TASK_ID_ROLES') or None

    # No ``working_dir`` or ``user_prompt``: ``swe_task`` sends both to ``run_agent``
    # itself, and the task's kwargs outrank ``agent_config``.
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
    _check_clients(agent_config, openai_clients)

    instances = _load_dataset(dataset_path)
    logger.info('[hitl/swe_user_sim/generator] Loaded %d instances from %s', len(instances), dataset_path)

    selected = [
        inst
        for inst in instances
        if inst.get('instance_id') and (not instance_filter or inst.get('instance_id') in instance_filter)
    ]
    if randomize:
        random.Random(random_seed).shuffle(selected)
        logger.info('[hitl/swe_user_sim/generator] Shuffled with seed=%d', random_seed)
    if skip_first_n:
        selected = selected[skip_first_n:]
        logger.info('[hitl/swe_user_sim/generator] Skipped first %d instances', skip_first_n)

    count = 0
    for inst in selected:
        instance_id = inst['instance_id']

        run_clients = (
            inject_cache_task_ids(openai_clients, roles=cache_task_id_roles)
            if cache_task_id_enabled
            else openai_clients
        )

        # The task reads the row back out of the file itself
        # (``load_instance_from_dataset``), so only the path and the id go here. The
        # instance's image and working directory come from the row.
        task_config: dict[str, Any] = {
            'dataset_path': dataset_path,
            'instance_id': instance_id,
            'eval_timeout_seconds': eval_timeout,
            'eval_max_attempts': eval_max_attempts,
            'skip_empty_patch_eval': skip_empty_patch_eval,
        }

        yield BatchItem(
            config=RunInput(
                imports=imports,
                task_name=TASK_NAME,
                task_config=task_config,
                agent_name=AGENT_NAME,
                agent_config=agent_config,
                runtime_provider_name=runtime_provider_name,
                runtime_provider_config=runtime_provider_config,
                runtime_configs={
                    'main': {'worker_venv': worker_venv},
                    'eval': {'worker_venv': worker_venv},
                },
                openai_clients=run_clients,
                processors=_processors(worker_profile, instance_id),
            ),
            run_id=instance_id,
            metadata={
                'instance_id': instance_id,
                'dataset_path': dataset_path,
                'repo': inst.get('repo', ''),
                'programming_language': inst.get('language', ''),
                'dataset_type': inst.get('dataset_type', ''),
                'persona': persona,
                'worker_backend': worker_backend,
            },
        )

        count += 1  # noqa: SIM113 - counts yielded items, not iterations
        if limit and count >= limit:
            return


def _check_runtime_provider(provider_name: str, config: Mapping[str, Any]) -> None:
    """Refuse a gongfeng config with no endpoint or token.

    Both come from ``SWE_RUNTIME_*``, not the ``GONGFENG_*`` names a run script may set.
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
            'tags': ['hitl', 'swe_user_sim'],
            'trace_metadata': {'instance_id': instance_id},
        },
    }


def _check_chain_only_env(environ: Mapping[str, str] | None = None) -> None:
    """Refuse a chain-recipe variable that this generator does not read."""
    env = os.environ if environ is None else environ
    stale = sorted(key for key in CHAIN_ONLY_ENV if env.get(key, '').strip())
    if stale:
        detail = '; '.join(f'{key} -> use {CHAIN_ONLY_ENV[key]}' for key in stale)
        raise ValueError(
            f'these are chain-recipe variables and are not read by this generator: {detail}. '
            'Unset them, or run the chain generator instead.'
        )


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
        'task_type': 'hitl/swe_user_sim',
        'task_name': TASK_NAME,
        'agent_name': AGENT_NAME,
        'dataset_path': os.environ.get('SWE_DATASET_PATH', ''),
        'persona': os.environ.get('HITL_USER_PERSONA', DEFAULT_PERSONA),
        'worker_backend': os.environ.get('HITL_WORKER_BACKEND', DEFAULT_WORKER_BACKEND),
        'worker_profile': os.environ.get('HITL_WORKER_CLIENT_PROFILE', '').strip() or 'default',
    }


__all__ = ['AGENT_NAME', 'ENVIRONMENT', 'TASK_NAME', 'generate_configs', 'get_batch_metadata']
