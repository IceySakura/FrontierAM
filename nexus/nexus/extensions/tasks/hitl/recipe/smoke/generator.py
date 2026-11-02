"""Batch generator for the smoke task: one conversation, checks named by the run.

Unlike the other recipes, the thing being tested is stated in the launch script rather
than read from a dataset. A run declares what to ask for and what to look for:

    HITL_SMOKE_MESSAGE   what the user asks the worker for
    HITL_SMOKE_CHECKS    a JSON list of checks, or a path to a file holding one

so a new mechanism gets a new script, not a new generator. The checks are
:class:`~nexus.extensions.tasks.hitl.recipe.smoke.task.SmokeCheck` objects; that class
documents the three kinds.

``HITL_SMOKE_MODE`` picks how the user is driven, and the two modes test different
halves (see :mod:`~nexus.extensions.tasks.hitl.recipe.smoke.agent`):

``live``
    The user simulator runs for real. ``HITL_SMOKE_INJECTIONS`` puts a line into its
    history at a named turn, which is how a run asks for a particular behaviour without
    waiting for the persona to think of it.
``replay``
    No user-side model call. ``HITL_SMOKE_SCRIPT`` lists the tool calls to make, so the
    same input runs every time -- what a worker-side bug needs.

The worker is a real Claude Code in a real sandbox either way.

``HITL_SMOKE_IMAGE`` is the container. Any image works -- nothing here needs a
repository -- but Claude Code is installed into it at setup, so an image without node
spends the install on every run. ``HITL_SMOKE_REPEAT`` runs the same configuration
several times, which is how a flaky mechanism is told from a broken one.

What has to agree with something outside this file:

``runtime_configs['main']['worker_venv']``
    The worker manager invokes this package's sandbox code in ``main``, where the venv
    installs it. Also what puts ``mcp`` and ``httpx`` in reach of a worker MCP server.

``agent_config['working_dir']``
    Sent here because the smoke task has no dataset to read it from, and the agent
    requires it.
"""

import json
import logging
import os

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nexus.extensions.tasks.hitl.actor.factory import backends
from nexus.extensions.tasks.hitl.model import known_profiles
from nexus.extensions.tasks.hitl.recipe.smoke.agent import (
    AGENT_NAME,
    Injection,
    ScriptedTurn,
    SmokeAgentInput,
    check_injections,
)
from nexus.extensions.tasks.hitl.recipe.smoke.task import TASK_NAME, SmokeCheck
from nexus.extensions.tasks.hitl.user_agent import mcp_config
from nexus.extensions.tasks.hitl.user_agent.compact import DEFAULT_COMPACT_PCT
from nexus.extensions.tasks.hitl.user_agent.personas import load_persona_dirs, registered_personas
from nexus.extensions.tasks.hitl.utils.generator_util import (
    CATALOG_DIR,
    _build_openai_clients,
    _build_runtime_provider_config,
    _env_bool,
    _env_choice,
    _env_int,
    _env_list,
    _new_session_id,
    _require_env,
    client_imports,
    with_worker_skills,
    worker_config_for,
    worker_usage_profile,
)
from nexus.runner.batch.batch_config import BatchItem
from nexus.runner.single import RunInput


logger = logging.getLogger(__name__)


TASK_IMPORT = 'nexus.extensions.tasks.hitl.recipe.smoke.task'
AGENT_IMPORT = 'nexus.extensions.tasks.hitl.recipe.smoke.agent'

MODES = ('live', 'replay')

PROCESSOR_IMPORTS = (
    'nexus.extensions.tasks.hitl.trace.chat',
    'nexus.extensions.tasks.hitl.trace.usage',
    'nexus.extensions.tasks.hitl.trace.langfuse',
)

DEFAULT_WORKER_BACKEND = 'print_claude_code'
DEFAULT_PERSONA = 'default'
DEFAULT_WORKING_DIR = '/workspace'

# A smoke run is a handful of turns, so the caps are far below the other recipes'.
DEFAULT_MAX_TURNS = 8
DEFAULT_TIMEOUT_MULTIPLIER = 30

HUMAN_CLIENT = 'human'
WORKER_CLIENT = 'worker'

# ``branch_last`` keeps a per-request branch, which is what carries the trace label;
# ``global_first`` exports one flat list with nowhere to put it.
CHAT_PROCESSOR: dict[str, Any] = {'chat': {'name': 'chat_tracker', 'args': {'tools_mode': 'branch_last'}}}

# Variables another recipe exports that mean nothing here, mapped to what to use
# instead. Not read by this generator, so the run would take the defaults instead.
FOREIGN_ENV: dict[str, str] = {
    'HITL_MAX_AGENT_ITERATIONS': 'HITL_MAX_TURNS',
    'SWE_DATASET_PATH': 'nothing -- a smoke run states its own message',
    'TASK_BASE_DIR': 'nothing -- as above',
    'HITL_EVAL_TYPES': 'HITL_SMOKE_CHECKS, which is what this task scores',
    'HITL_SCORE_POLICY': 'nothing -- the score is the share of checks that passed',
    'HITL_SKIP_EVAL': 'nothing -- the checks are the evaluation',
}

ENVIRONMENT: dict[str, str] = {
    # -- what this run tests --
    'HITL_SMOKE_MESSAGE': 'required. What the user asks the worker for.',
    'HITL_SMOKE_CHECKS': 'required. A JSON list of checks, or a path to a file holding one.',
    'HITL_SMOKE_MODE': f'how the user is driven: {" or ".join(MODES)}. Default {MODES[0]!r}.',
    'HITL_SMOKE_INJECTIONS': 'live only. A JSON list of {turn, text}, or a path to one.',
    'HITL_SMOKE_SCRIPT': 'replay only, and required there. A JSON list of {tool, args}, or a path to one.',
    'HITL_SMOKE_NAME': "the run id. Default 'smoke'.",
    'HITL_SMOKE_IMAGE': 'required. The container image.',
    'HITL_SMOKE_WORKING_DIR': f'where the worker starts. Default {DEFAULT_WORKING_DIR!r}.',
    'HITL_SMOKE_REPEAT': 'run the same configuration this many times. Default 1.',
    'HITL_SMOKE_COMMAND_TIMEOUT': "a command check's wall clock, seconds. Default 60.",
    # -- the run --
    'HITL_MAX_TURNS': f"the user's turn cap. Default {DEFAULT_MAX_TURNS}.",
    'TIMEOUT_MULTIPLIER': f'the run ceiling = 60s * this. Default {DEFAULT_TIMEOUT_MULTIPLIER}.',
    # -- the user agent --
    'HITL_USER_PERSONA': f'persona name; refused unless registered. Default {DEFAULT_PERSONA!r}.',
    'HITL_USER_PERSONA_DIRS': 'directories holding persona files, read before the name is resolved.',
    'HITL_USER_SKILL_DIRS': 'extra skill directories to load.',
    'HITL_USER_EXTRA_TOOLS': "tools to hold beyond the persona's.",
    'HITL_USER_MCP_SERVERS': 'path to a JSON list of MCP servers for the user simulator.',
    'HITL_USER_CONTEXT_WINDOW': "the user model's context size, tokens. 0 leaves the history uncompacted.",
    'HITL_USER_COMPACT_PCT': f'share of that window before it is summarised. Default {DEFAULT_COMPACT_PCT}.',
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
    'HITL_WORKER_BACKEND': (
        f'print_claude_code, pty_claude_code or print_codex. Default {DEFAULT_WORKER_BACKEND!r}. '
        'The HITL_CC_* and HITL_CODEX_* variables below apply to whichever it names.'
    ),
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
    'HITL_WORKER_TOOL_SEARCH': 'progressive tool disclosure, which is how Grep/Glob reach the worker.',
    'HITL_WORKER_MEMORY_FILE': 'path to the CLAUDE.md placed for the worker.',
    'HITL_WORKER_SKILL_DIRS': 'directories packed into the sandbox as worker skills.',
    'HITL_WORKER_BINARY_URL': 'Claude Code binary URL.',
    'HITL_CC_BUILTIN_COMMANDS': "the harness's own commands to offer, comma-separated. None by default.",
    # -- the worker, Codex backends only --
    'HITL_CODEX_MODEL': 'the model Codex names in its own config.toml. The proxy substitutes the real one.',
    'HITL_CODEX_MODEL_CATALOG': (
        f'the model catalog Codex runs under, a name in {CATALOG_DIR} or a path of its own. '
        'Unset leaves the catalog Codex bundles in effect.'
    ),
    'HITL_CODEX_RESPONSES_LITE': 'override use_responses_lite on every catalog entry. Unset follows the file.',
    'HITL_CODEX_AZURE_PROVIDER': (
        'write provider name "azure", which keeps store=true and re-attaches reasoning item ids. '
        'Default true; 0 for a model whose reasoning is plaintext.'
    ),
    'HITL_CODEX_CONFIG_OVERRIDES': '``key.path=value`` entries passed as Codex ``-c``, comma-separated.',
    'HITL_WORKER_NPM_REGISTRY': 'npm registry for the Codex install.',
    # -- the worker, either harness --
    'HITL_WORKER_MCP_SERVERS': 'path to a JSON list of MCP servers registered for the worker.',
    'HITL_WORKER_MCP_SEARCH': 'register the built-in search MCP server for the worker.',
    'SEARCH_API_KEY': 'required when HITL_WORKER_MCP_SEARCH=1.',
    'SEARCH_API_URL': 'endpoint for that search server.',
    # -- the sandbox --
    'SWE_RUNTIME_PROVIDER_NAME': "runtime provider. Default 'gongfeng'.",
    'SWE_RUNTIME_BASE_URL': 'its API base URL.',
    'SWE_RUNTIME_TOKEN': 'its token.',
    'SWE_RUNTIME_CPU': 'cpu per sandbox. Default 2.',
    'SWE_RUNTIME_MEMORY': 'memory per sandbox, GB. Default 10.',
    'SWE_RUNTIME_TTL': "sandbox lifetime as a Go duration ('12h', '1h30m'), max 168h. Backend default when unset.",
    'SWE_RUNTIME_PURPOSE': "sandbox lifecycle tag. Default 'test'.",
    'SWE_WORKER_VENV': 'install the nexus wheel in the sandboxes. Default true, and the worker needs it.',
    # -- tracing --
    'LANGFUSE_PUBLIC_KEY': 'Langfuse upload is configured only when this and the secret are both set.',
    'LANGFUSE_SECRET_KEY': 'see above.',
    'LANGFUSE_BASE_URL': 'Langfuse endpoint.',
    'HITL_DISABLE_LANGFUSE': 'skip Langfuse upload even when credentials are present.',
}


def generate_configs() -> Iterator[BatchItem]:
    """One ``BatchItem`` per repeat of the single configuration this run declares."""
    _check_foreign_env()

    message = _require_env('HITL_SMOKE_MESSAGE', 'HITL_SMOKE_MESSAGE required (what to ask the worker for)')
    checks = _load_checks()
    mode = _env_choice('HITL_SMOKE_MODE', MODES[0], MODES)
    image = _require_env('HITL_SMOKE_IMAGE', 'HITL_SMOKE_IMAGE required (the container to run in)')
    name = os.environ.get('HITL_SMOKE_NAME', '').strip() or 'smoke'
    working_dir = os.environ.get('HITL_SMOKE_WORKING_DIR', '').strip() or DEFAULT_WORKING_DIR
    repeat = _env_int('HITL_SMOKE_REPEAT', default=1)
    command_timeout = _env_int('HITL_SMOKE_COMMAND_TIMEOUT', default=60)

    runtime_provider_name = os.environ.get('SWE_RUNTIME_PROVIDER_NAME', '').strip() or 'gongfeng'
    max_turns = _env_int('HITL_MAX_TURNS', default=DEFAULT_MAX_TURNS)
    timeout_seconds = 60 * _env_int('TIMEOUT_MULTIPLIER', default=DEFAULT_TIMEOUT_MULTIPLIER)
    worker_venv = _env_bool('SWE_WORKER_VENV', default=True)

    worker_profile = _env_choice('HITL_WORKER_CLIENT_PROFILE', 'default', known_profiles())
    worker_backend = _env_choice('HITL_WORKER_BACKEND', DEFAULT_WORKER_BACKEND, backends())

    persona_dirs = _env_list('HITL_USER_PERSONA_DIRS')
    load_persona_dirs(persona_dirs)
    persona = _env_choice('HITL_USER_PERSONA', DEFAULT_PERSONA, registered_personas())
    skill_dirs = _env_list('HITL_USER_SKILL_DIRS')
    extra_tools = _env_list('HITL_USER_EXTRA_TOOLS')
    # The user's own servers, which reach it through ``load_mcp_tools`` rather than
    # through Claude Code. A different shape and a different mechanism from the
    # worker's; see ``user_agent/mcp_config.py``.
    user_mcp_servers = mcp_config.user_servers()
    context_window = _env_int('HITL_USER_CONTEXT_WINDOW')
    compact_pct = _env_int('HITL_USER_COMPACT_PCT', default=DEFAULT_COMPACT_PCT)

    session_id = _new_session_id()
    openai_clients = _build_openai_clients([], worker_profile, session_id=session_id)
    worker_config = worker_config_for(worker_backend, worker_profile, session_id=session_id)
    worker_skill_dirs = _env_list('HITL_WORKER_SKILL_DIRS')
    if worker_skill_dirs:
        worker_config = with_worker_skills(worker_config, worker_skill_dirs)
    worker_config = {**worker_config, 'working_dir': working_dir}
    runtime_provider_config = _build_runtime_provider_config()
    imports = [TASK_IMPORT, AGENT_IMPORT, *PROCESSOR_IMPORTS, *client_imports(worker_profile)]

    agent_config: dict[str, Any] = {
        'working_dir': working_dir,
        'user_client_name': HUMAN_CLIENT,
        'worker_client_name': WORKER_CLIENT,
        'worker_backend': worker_backend,
        'worker_config': worker_config,
        'persona': persona,
        'max_turns': max_turns,
        'mode': mode,
    }
    if mode == 'replay':
        agent_config['script'] = [step.model_dump() for step in _load_script()]
    else:
        injections = _load_injections(cap=max_turns)
        if injections:
            agent_config['injections'] = [injection.model_dump() for injection in injections]
    if persona_dirs:
        agent_config['persona_dirs'] = persona_dirs
    if skill_dirs:
        agent_config['skill_dirs'] = skill_dirs
    if extra_tools:
        agent_config['extra_tools'] = extra_tools
    if user_mcp_servers:
        agent_config['mcp_servers'] = user_mcp_servers
    if context_window:
        agent_config['context_window'] = context_window
        agent_config['compact_pct'] = compact_pct
    _check_agent_config(agent_config)

    logger.info(
        '[hitl/smoke/generator] %s: mode=%s %d check(s), %d run(s), backend=%s, user_mcp=%d worker_mcp=%d',
        name,
        mode,
        len(checks),
        repeat,
        worker_backend,
        len(user_mcp_servers),
        len(worker_config.get('mcp_servers') or []),
    )
    for check in checks:
        logger.info('[hitl/smoke/generator]   check %s (%s)', check.name, check.kind())

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
                    'agent_timeout': timeout_seconds,
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
                'smoke': name,
                'mode': mode,
                'attempt': attempt + 1,
                'checks': [check.name for check in checks],
                'worker_backend': worker_backend,
                'persona': persona,
            },
        )


def _load_json_list(key: str, *, what: str, required: bool) -> list[Any]:
    """The JSON list ``key`` holds, or the one in the file it names.

    Both forms are accepted because a launch script writes whichever is less painful:
    a one-entry list inline, a several-entry one in a file where the quoting cannot go
    wrong. Returns empty for an unset optional key.
    """
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
        raise ValueError(f'{key} must hold a list, got {type(loaded).__name__}')
    if required and not loaded:
        raise ValueError(f'{key} must hold a non-empty list')
    return loaded


def _load_checks() -> list[SmokeCheck]:
    """The declared checks.

    A malformed one is refused here rather than in the sandbox: it would otherwise pay
    for a conversation and then report nothing about it.
    """
    entries = _load_json_list('HITL_SMOKE_CHECKS', what='a JSON list of checks, or a path to one', required=True)
    checks = [SmokeCheck.model_validate(entry) for entry in entries]
    # Calls kind() for its refusal: a check setting none of the three fields would
    # otherwise raise inside the sandbox, after the conversation was paid for.
    for check in checks:
        check.kind()
    names = [check.name for check in checks]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f'HITL_SMOKE_CHECKS names a check more than once: {repeated}')
    return checks


def _load_script() -> list[ScriptedTurn]:
    """The replay's tool calls, in the order they are made."""
    entries = _load_json_list(
        'HITL_SMOKE_SCRIPT',
        what='a JSON list of {tool, args}, or a path to one; required in replay mode',
        required=True,
    )
    return [ScriptedTurn.model_validate(entry) for entry in entries]


def _load_injections(*, cap: int) -> list[Injection]:
    """The lines to put into a live user's history, checked against the turn cap.

    Checked here as well as in the agent: an injection past the cap never fires, and a
    check looking for what it asked for then fails for the wrong reason.
    """
    entries = _load_json_list(
        'HITL_SMOKE_INJECTIONS', what='a JSON list of {turn, text}, or a path to one', required=False
    )
    injections = [Injection.model_validate(entry) for entry in entries]
    check_injections(injections, cap=cap)
    return injections


def _check_agent_config(agent_config: dict[str, Any]) -> None:
    """Refuse a key the agent input would drop.

    An agent input silently drops what it does not declare, so a misspelled key
    reaches nothing and the run takes the default.
    """
    unknown = sorted(set(agent_config) - set(SmokeAgentInput.model_fields))
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
            'session_id': f'smoke-{name}',
        },
    }


__all__ = ['ENVIRONMENT', 'generate_configs']
