"""Batch generator for the chained SWE recipe on the user_agent components.

The environment variables are the legacy chain generator's, so an existing
pipeline script runs this recipe by changing the recipe import and nothing else.
One addition, about the user side: ``HITL_USER_PERSONA``. :data:`ENVIRONMENT`
lists every variable a run is configured by, including the ones the shared helpers
read.

Three settings have to agree with something outside this file. Each is silent
when it does not, which is why they are listed rather than left to be discovered:

``HITL_WORKER_CLIENT_PROFILE`` -> the usage profile
    The profile names the provider protocol (``gcp_claude`` -> Vertex,
    ``bedrock_claude`` -> Bedrock, ``default`` -> plain OpenAI-compatible). It
    decides both how the worker's client is built and how its usage block is read:
    Vertex and Bedrock report cache creation separately from cache reads, and the
    default profile does not look for it, so the creation counter reads as zero.
    :func:`_processors` keys the profile by :data:`WORKER_CLIENT`; that key must
    equal the agent's ``worker_client_name``, or the profile applies to no client at
    all. The user's client is always a plain OpenAI-compatible endpoint, so it stays
    on ``default``.

``tools_mode='branch_last'`` on the chat tracker
    Labels ride on spans and ``global_first`` exports one flat list with nowhere to
    put them, so every conversation would merge into a single trajectory. The agent
    asserts this at startup (``labels.require_labelling_tracker``) because the damage is only
    visible when the export is read, after the run.

worker backend -> trace attribution
    ``print_claude_code`` and ``pty_claude_code`` attribute traces by per-request
    label. A backend that instead registers a processor per session cannot share
    the one run-level tracker this generator configures.
"""

import logging
import os
import random

from collections.abc import Iterator
from typing import Any, get_args

from nexus.extensions.tasks.hitl.actor.factory import backend_class, backends
from nexus.extensions.tasks.hitl.eval.aggregate import SWEScorePolicy, SWEUnitScorePolicy
from nexus.extensions.tasks.hitl.model import known_profiles, resolve_profile
from nexus.extensions.tasks.hitl.recipe.swe_chain_v2.requirement_groups import (
    DEFAULT_GROUP_MAX_CHARS,
    DEFAULT_GROUP_SIZE,
)
from nexus.extensions.tasks.hitl.user_agent import mcp_config
from nexus.extensions.tasks.hitl.user_agent.compact import DEFAULT_COMPACT_PCT
from nexus.extensions.tasks.hitl.user_agent.personas import load_persona_dirs, registered_personas
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
    _worker_profile_kwargs,
    _worker_reasoning_effort,
    client_imports,
    inject_cache_task_ids,
    with_worker_skills,
    worker_config_for,
    worker_usage_profile,
)
from nexus.extensions.tasks.hitl.utils.generator_util.codex import CATALOG_DIR
from nexus.runner.batch.batch_config import BatchItem
from nexus.runner.single import RunInput


logger = logging.getLogger(__name__)

RECIPE_IMPORT = 'nexus.extensions.tasks.hitl.recipe.swe_chain_v2'
AGENT_NAME = 'hitl/agent_with_user'
DEFAULT_WORKER_BACKEND = 'print_claude_code'
DEFAULT_PERSONA = 'default'


# Every variable a run is configured by, and what it decides. Read here or by the
# helpers in ``utils/generator_util.py``, which is
# why grepping this file alone does not find them all.
#
# `test_recipe_wiring` asserts this covers what the module reads, so a variable
# added without an entry fails rather than becoming undocumented.
ENVIRONMENT: dict[str, str] = {
    # -- dataset and selection --
    'SWE_DATASET_PATH': 'required. Chain dataset JSONL path.',
    'SWE_INSTANCE_IDS': 'comma-separated allowlist of instance ids.',
    'SWE_LIMIT': 'stop after this many instances.',
    'HITL_SKIP_FIRST_N': 'drop this many instances from the front.',
    'HITL_RANDOMIZE_INSTANCES': 'shuffle before limiting.',
    'HITL_RANDOM_SEED': 'seed for that shuffle. Default 0.',
    'HITL_DATASET_PRESET': 'dataset preset name passed to the task.',
    # -- the run --
    'HITL_MAX_AGENT_ITERATIONS': "the user's turn cap. Default 120.",
    'HITL_AGENT_TIMEOUT': "wall clock for one instance's agent, seconds. Default 7200.",
    'HITL_REQUIREMENT_GROUP_SIZE': (
        f'dataset entries per released requirement. Default {DEFAULT_GROUP_SIZE}, which '
        'releases the chain as the dataset holds it.'
    ),
    'HITL_REQUIREMENT_GROUP_MAX_CHARS': (
        f'characters a group may reach before the next entry starts a new one. Default {DEFAULT_GROUP_MAX_CHARS}.'
    ),
    # -- the user agent --
    'HITL_USER_PERSONA': f'persona name; refused unless registered. Default {DEFAULT_PERSONA!r}.',
    'HITL_USER_PERSONA_DIRS': 'directories holding persona .md files; one may replace a built-in by name.',
    'HITL_USER_SKILL_DIRS': 'extra skill directories to load.',
    'HITL_USER_EXTRA_TOOLS': "tools to hold beyond the persona's, from toolset.CATALOGUE.",
    'HITL_USER_MCP_SERVERS': 'path to a JSON list of MCP servers for the user simulator.',
    'HITL_USER_CONTEXT_WINDOW': "the user model's context size, tokens. 0 leaves the history uncompacted.",
    'HITL_USER_COMPACT_PCT': f'share of that window the history may fill before it is summarised. Default {DEFAULT_COMPACT_PCT}.',
    # -- the user's client (plain OpenAI-compatible; usage profile stays default) --
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
    'OPENAI_MODEL': 'fallback model when a role sets none of its own.',
    'OPENAI_BASE_URL': 'fallback endpoint.',
    'OPENAI_API_KEY': 'fallback key.',
    # -- the worker --
    'HITL_WORKER_BACKEND': f'worker backend; refused unless registered. Default {DEFAULT_WORKER_BACKEND!r}.',
    'HITL_WORKER_VERSION': 'harness release to install, whichever harness the backend names. Each has its own default.',
    'HITL_WORKER_TURN_TIMEOUT': 'one worker turn, seconds. Default 7200.',
    'HITL_WORKER_CONTEXT_WINDOW': (
        'the context window auto-compaction is sized against, tokens. Claude Code compacts at '
        'HITL_CC_AUTOCOMPACT_PCT of it (90% by default), Codex at min(that limit, this * 90%).'
    ),
    'HITL_WORKER_CLIENT_PROFILE': 'model profile, one of the known ones. Also selects the usage profile; see the module docstring.',
    'HITL_WORKER_MODEL': "the worker's model.",
    'HITL_WORKER_BASE_URL': 'its endpoint.',
    'HITL_WORKER_API_KEY': 'its key.',
    'HITL_WORKER_LLM_TIMEOUT': 'its per-request timeout, seconds.',
    'HITL_WORKER_THINKING_ENABLED': (
        "which of the profile's two request-arg sets the worker client carries. Default on, "
        'which is the thinking set. Not a thinking switch: to stop the model thinking, name an '
        "effort that says so (gpt_responses: 'none') or a thinking_type that does "
        "(bedrock_claude: 'disabled'). =0 on bedrock_claude stamps an empty set, dropping the "
        'effort along with the thinking block; on print_codex it does nothing at all, since '
        'Codex builds its own requests. What switches per request is the harness: Claude Code '
        'sends some calls without a thinking block, and the host answers those from the '
        'no-thinking set (HITL_WORKER_NO_THINKING_MODEL).'
    ),
    'HITL_WORKER_REASONING_EFFORT': (
        "reasoning effort in the profile's own terms, for the worker client, the Codex config "
        '(model_reasoning_effort) and the run metadata alike. Its default applies when unset: '
        "the profile's for the client, the catalog entry's for Codex."
    ),
    'HITL_WORKER_THINKING_PRESET': 'thinking preset for the CC harness.',
    'HITL_WORKER_THINKING_TYPE': 'thinking type passed as a request arg.',
    'HITL_WORKER_CACHE_STRATEGY': 'prompt cache strategy.',
    'HITL_WORKER_NO_THINKING_MODEL': 'model for the no-think path.',
    'HITL_WORKER_HAIKU_MODEL': "model for the harness's small-model calls.",
    'HITL_WORKER_CC_IDENTIFIER': 'Claude Code build identifier.',
    'HITL_WORKER_BINARY_URL': 'Claude Code binary URL.',
    'HITL_WORKER_APPEND_SYSTEM_PROMPT': "text appended to the worker's system prompt.",
    'HITL_WORKER_SKILL_DIRS': (
        "skill directories packed into the sandbox at the backend's skills_path, where the directory "
        'name is the command the user may send as /<name>. Only reaches the user on a backend that '
        'declares a command_prefix.'
    ),
    'HITL_WORKER_MEMORY_FILE': (
        'path to a markdown file placed where the worker manager declares its harness reads user '
        "memory -- for Claude Code the sandbox's ~/.claude/CLAUDE.md, which the worker and its "
        'subagents read. Any path; empty means none. A path naming no file is refused.'
    ),
    'HITL_WORKER_DISABLE_AGENT': 'disable the harness Agent tool.',
    'HITL_WORKER_MCP_SEARCH': 'give the worker the built-in search MCP server.',
    'HITL_WORKER_MCP_SERVERS': 'path to a JSON list of MCP servers for the worker.',
    'HITL_WORKER_TOOL_SEARCH': 'progressive tool disclosure, which is how Grep and Glob reach the worker.',
    'HITL_WORKER_TOOL_SEARCH_MODE': "value for ENABLE_TOOL_SEARCH. Default 'true'.",
    'HITL_CC_AUTOCOMPACT_PCT': 'the fraction of that window it compacts at.',
    'HITL_CC_BUILTIN_COMMANDS': "the harness's own commands to offer, comma-separated. None by default.",
    # -- the Codex backends, whose knobs the CC ones above do not reach --
    'HITL_CODEX_MODEL': "the model slug in the session's config.toml, which the catalog must carry.",
    'HITL_CODEX_MODEL_CATALOG': (
        f'the model catalog Codex runs under, a name in {CATALOG_DIR} or a path of its own. '
        'Unset leaves the catalog Codex bundles in effect.'
    ),
    'HITL_CODEX_RESPONSES_LITE': 'override use_responses_lite on every catalog entry. Unset follows the file.',
    'HITL_CODEX_AZURE_PROVIDER': 'name the provider "azure", which keeps store=true and replayed reasoning ids.',
    'HITL_CODEX_CONFIG_OVERRIDES': '``key.path=value`` entries passed as Codex ``-c``, comma-separated.',
    'HITL_WORKER_NPM_REGISTRY': 'npm registry the harness installs from.',
    'HITL_WORKER_REASONING_MODE': 'reasoning.mode on the request. Empty omits the field.',
    'HITL_LLM_TIMEOUT': 'default per-request timeout for a role that sets none.',
    # -- caching --
    'HITL_CACHE_TASK_ID_ENABLED': 'route one task to one backend resource to keep cache hits.',
    'HITL_CACHE_TASK_ID_ROLES': 'restrict that to these roles.',
    # -- evaluation --
    'HITL_EVAL_TYPES': "comma-separated. Default 'unit_test'.",
    'HITL_EVAL_TIMEOUT': 'evaluation timeout, seconds. Default 3600.',
    'HITL_SKIP_EVAL': 'produce the patch and stop.',
    'HITL_SKIP_EMPTY_PATCH_EVAL': 'skip evaluation when the patch is empty.',
    'HITL_SCORE_POLICY': "one of unit_test / patch_rubrics / mean. Default 'mean'.",
    'HITL_UNIT_SCORE_POLICY': "unit-test scoring: official or fraction. Default 'official'.",
    'HITL_RUBRIC_EVAL_MODEL': 'model for rubric evaluation. Falls back to the user model.',
    'HITL_RUBRIC_EVAL_BASE_URL': 'its endpoint.',
    'HITL_RUBRIC_EVAL_API_KEY': 'its key.',
    # -- sandbox --
    'SWE_RUNTIME_PROVIDER_NAME': "runtime provider. Default 'gongfeng'.",
    'SWE_RUNTIME_IMAGE': 'sandbox image.',
    'SWE_RUNTIME_CPU': 'sandbox CPU.',
    'SWE_RUNTIME_MEMORY': 'sandbox memory.',
    'SWE_RUNTIME_TIMEOUT': 'per-request timeout to the provider, seconds.',
    'SWE_RUNTIME_TTL': "sandbox lifetime as a Go duration ('12h', '1h30m'), max 168h. Backend default when unset.",
    'SWE_RUNTIME_BASE_URL': 'provider endpoint.',
    'SWE_RUNTIME_TOKEN': 'provider token.',
    'SWE_RUNTIME_PURPOSE': 'provider purpose tag.',
    'SWE_RPC_PROTOCOL': 'runtime RPC protocol.',
    'SWE_WORKER_VENV': "run the sandbox worker in a venv. Default 'true'.",
    'GONGFENG_QUEUE_WAIT_MAX': 'max queue wait before giving up, seconds.',
    'GONGFENG_STARTUP_JITTER_MAX': 'random startup delay, to spread sandbox creation.',
    'SEARCH_API_URL': 'search MCP endpoint, when HITL_WORKER_MCP_SEARCH is on.',
    'SEARCH_API_KEY': 'its key.',
    # -- tracing --
    'LANGFUSE_PUBLIC_KEY': 'Langfuse upload is configured only when this and the secret are both set.',
    'LANGFUSE_SECRET_KEY': 'see above.',
    'LANGFUSE_BASE_URL': 'Langfuse endpoint. Also read as LANGFUSE_HOST by the processor.',
    'HITL_DISABLE_LANGFUSE': 'skip Langfuse upload even when credentials are present.',
}

# Labels live on spans, so the export has to keep a per-request branch. See the
# module docstring.
CHAT_PROCESSOR: dict[str, Any] = {'chat': {'name': 'chat_tracker', 'args': {'tools_mode': 'branch_last'}}}

# The client the worker talks through. Must equal the agent's
# ``worker_client_name`` default: the usage profile is keyed by it, and a name that
# matches no client applies the profile to nothing.
WORKER_CLIENT = 'worker'

# The client the user simulator talks through, named in the same mapping.
HUMAN_CLIENT = 'human'


def _processors(worker_profile: str, instance_id: str = '') -> dict[str, Any]:
    """The run-level processors: the chat tracker, token accounting, and Langfuse.

    Usage is counted from spans rather than read off the worker manager, so the
    user's own turns and its compaction summaries are counted too, and a run whose
    agent never returns still reports what it spent.

    The usage profile is per client, keyed by the client name the agent asks for.
    A profile decides how one provider's usage block is read, and a run has at
    least two providers: the worker's, from ``worker_profile``, and the user's,
    which is a plain OpenAI-compatible endpoint. Reading Vertex or Bedrock counters
    with the default profile silently drops cache creation, which is priced
    separately from cache reads.

    All three read the same completion spans and group by the same trace label, so
    ``langfuse['by_label']`` and ``usage['by_label']`` can be compared: they count
    the same calls through independent code.
    """
    processors: dict[str, Any] = {
        **CHAT_PROCESSOR,
        'usage': {
            'name': 'hitl_usage',
            'args': {
                'profiles': {WORKER_CLIENT: worker_usage_profile(worker_profile)},
                'default_profile': 'default',
            },
        },
    }
    langfuse = _langfuse_processor(instance_id)
    if langfuse is not None:
        processors['langfuse'] = langfuse
    return processors


def _langfuse_processor(instance_id: str) -> dict[str, Any] | None:
    """The Langfuse processor config, or None when the run should not upload.

    Omitted rather than disabled-at-runtime: a processor the generator does not
    name is never constructed, so a run without credentials does not log a warning
    per instance about a tracker nobody asked for.

    ``session_id`` is the instance id, which is what groups this run's spans in the
    Langfuse UI. One trace holds the whole run, so the session collects the reruns
    of one instance rather than one run's several agents.
    """
    if _env_bool('HITL_DISABLE_LANGFUSE'):
        return None
    if not (os.environ.get('LANGFUSE_PUBLIC_KEY') and os.environ.get('LANGFUSE_SECRET_KEY')):
        return None
    return {
        'name': 'multiagent_langfuse',
        'args': {
            'session_id': instance_id,
            'trace_name': instance_id,
            'tags': ['hitl', 'swe_chain_v2'],
            'trace_metadata': {'instance_id': instance_id},
        },
    }


def generate_configs() -> Iterator[BatchItem]:
    """Generate batch configs for the chained SWE recipe."""

    dataset_path = _require_env('SWE_DATASET_PATH', 'SWE_DATASET_PATH required (chain dataset JSONL path)')

    instance_filter = set(_env_list('SWE_INSTANCE_IDS'))
    limit = _env_int('SWE_LIMIT') or None
    skip_first_n = _env_int('HITL_SKIP_FIRST_N')
    randomize = _env_bool('HITL_RANDOMIZE_INSTANCES')
    random_seed = _env_int('HITL_RANDOM_SEED', default=0)

    runtime_provider_name = os.environ.get('SWE_RUNTIME_PROVIDER_NAME', '').strip() or 'gongfeng'
    max_agent_iterations = _env_int('HITL_MAX_AGENT_ITERATIONS', default=120)
    agent_timeout = _env_int('HITL_AGENT_TIMEOUT', default=7200)
    requirement_group_size = _env_int('HITL_REQUIREMENT_GROUP_SIZE', default=DEFAULT_GROUP_SIZE)
    requirement_group_max_chars = _env_int('HITL_REQUIREMENT_GROUP_MAX_CHARS', default=DEFAULT_GROUP_MAX_CHARS)
    if requirement_group_size < 1:
        raise ValueError(
            f'HITL_REQUIREMENT_GROUP_SIZE must be at least 1, got {requirement_group_size}. '
            'Set 1 to release the chain as the dataset holds it.'
        )

    eval_types = _env_list('HITL_EVAL_TYPES') or ['unit_test']
    eval_timeout = _env_int('HITL_EVAL_TIMEOUT', default=3600)
    skip_eval = _env_bool('HITL_SKIP_EVAL')
    skip_empty_patch_eval = _env_bool('HITL_SKIP_EMPTY_PATCH_EVAL')
    score_policy = _env_choice('HITL_SCORE_POLICY', 'mean', get_args(SWEScorePolicy))
    unit_score_policy = _env_choice('HITL_UNIT_SCORE_POLICY', 'official', get_args(SWEUnitScorePolicy))

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
    context_window = _env_int('HITL_USER_CONTEXT_WINDOW')
    compact_pct = _env_int('HITL_USER_COMPACT_PCT', default=DEFAULT_COMPACT_PCT)

    session_id = _new_session_id()
    openai_clients = _build_openai_clients(eval_types, worker_profile, session_id=session_id)
    worker_config = worker_config_for(worker_backend, worker_profile, session_id=session_id)
    if worker_skill_dirs:
        worker_config = with_worker_skills(worker_config, worker_skill_dirs)
    worker_meta = _worker_meta(worker_backend, worker_config, worker_profile)
    runtime_provider_config = _build_runtime_provider_config()
    imports = [RECIPE_IMPORT, *client_imports(worker_profile)]

    cache_task_id_enabled = _env_bool('HITL_CACHE_TASK_ID_ENABLED')
    cache_task_id_roles = _env_list('HITL_CACHE_TASK_ID_ROLES') or None

    instances = _load_dataset(dataset_path)
    logger.info('[hitl/swe_chain_v2/generator] Loaded %d instances from %s', len(instances), dataset_path)

    selected = [
        inst
        for inst in instances
        if inst.get('instance_id') and (not instance_filter or inst.get('instance_id') in instance_filter)
    ]
    if randomize:
        random.Random(random_seed).shuffle(selected)
        logger.info('[hitl/swe_chain_v2/generator] Shuffled with seed=%d', random_seed)
    if skip_first_n:
        selected = selected[skip_first_n:]
        logger.info('[hitl/swe_chain_v2/generator] Skipped first %d instances', skip_first_n)

    count = 0
    for inst in selected:
        instance_id = inst['instance_id']
        problem_statements = _string_list(inst.get('problem_statements'))
        chain_patches = _string_list(inst.get('patch'))

        run_clients = (
            inject_cache_task_ids(openai_clients, roles=cache_task_id_roles)
            if cache_task_id_enabled
            else openai_clients
        )

        task_config: dict[str, Any] = {
            'instance_id': instance_id,
            # The record itself, so the task opens no file. Sending the path instead
            # had every instance re-scan the dataset this loop already read.
            'instance': inst,
            'agent_timeout_seconds': agent_timeout,
            'eval_types': eval_types,
            'eval_timeout_seconds': eval_timeout,
            'skip_eval': skip_eval,
            'skip_empty_patch_eval': skip_empty_patch_eval,
            'score_policy': score_policy,
            'unit_score_policy': unit_score_policy,
            'requirement_group_size': requirement_group_size,
            'requirement_group_max_chars': requirement_group_max_chars,
        }
        if problem_statements:
            task_config['chain_problem_statements'] = problem_statements
        if chain_patches:
            task_config['chain_patches'] = chain_patches
        rubrics_yaml = inst.get('rubrics_yaml', '')
        if rubrics_yaml:
            task_config['rubrics_yaml'] = rubrics_yaml

        # Every setting the conversation runs under. The task sends only what it
        # computes at run time -- the requirements, the workspace and its hook --
        # and those win on conflict (runner/single.py:538-540).
        agent_config: dict[str, Any] = {
            # the user simulator
            'persona': persona,
            'user_client_name': HUMAN_CLIENT,
            # the worker
            'worker_backend': worker_backend,
            'worker_client_name': WORKER_CLIENT,
            'worker_config': worker_config,
            # the loop
            'max_turns': max_agent_iterations,
        }
        if context_window:
            agent_config['context_window'] = context_window
            agent_config['compact_pct'] = compact_pct
        if skill_dirs:
            agent_config['skill_dirs'] = skill_dirs
        if persona_dirs:
            agent_config['persona_dirs'] = persona_dirs
        if extra_tools:
            agent_config['extra_tools'] = extra_tools
        if user_mcp_servers:
            agent_config['mcp_servers'] = user_mcp_servers

        yield BatchItem(
            config=RunInput(
                imports=imports,
                task_name='hitl/swe_chain_v2_task',
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
                'programming_language': inst.get('language', ''),
                'chain_steps': len(problem_statements),
                'source_dataset_type': inst.get('source_dataset_type', ''),
                'persona': persona,
                'worker_backend': worker_backend,
                'worker': worker_meta,
                'user': _user_meta(),
            },
        )

        count += 1  # noqa: SIM113 - counts yielded items, not iterations
        if limit and count >= limit:
            return


def get_batch_metadata() -> dict[str, Any]:
    backend = _env_choice('HITL_WORKER_BACKEND', DEFAULT_WORKER_BACKEND, backends())
    profile = _env_choice('HITL_WORKER_CLIENT_PROFILE', 'default', known_profiles())
    return {
        'task_type': 'hitl/swe_chain_v2',
        'dataset_path': os.environ.get('SWE_DATASET_PATH', ''),
        'runtime_provider': os.environ.get('SWE_RUNTIME_PROVIDER_NAME', '').strip() or 'gongfeng',
        'max_agent_iterations': _env_int('HITL_MAX_AGENT_ITERATIONS', default=120),
        'eval_types': os.environ.get('HITL_EVAL_TYPES', '').strip() or 'unit_test',
        'randomize': _env_bool('HITL_RANDOMIZE_INSTANCES'),
        'random_seed': _env_int('HITL_RANDOM_SEED'),
        'persona': _persona(),
        'worker_backend': backend,
        'worker': _worker_meta(backend, worker_config_for(backend, profile), profile),
        'user': _user_meta(),
        'requirement_group_size': _env_int('HITL_REQUIREMENT_GROUP_SIZE', default=DEFAULT_GROUP_SIZE),
    }


def _harness_version(worker_config: dict[str, Any], backend: str) -> str:
    """The harness release this run installs, as its ``worker_config`` names it.

    Raises when the config names none: every backend's builder sets it, so a
    missing one means the run installs whatever the manager defaults to and the
    batch cannot say what it ran.
    """
    version = str(worker_config.get('version') or '').strip()
    if not version:
        raise ValueError(
            f'the worker_config for {backend!r} names no version. Set one in its '
            'generator_util builder so a batch records what it installed.'
        )
    return version


def _worker_meta(backend: str, worker_config: dict[str, Any], profile: str) -> dict[str, Any]:
    """Which harness, which release and which model the worker ran under.

    ``model`` is the string the run configured, passed through rather than
    normalised: the same model is named differently per gateway, and the mapping
    belongs to whoever compares batches.

    ``thinking_enabled`` is the client's main mode: which of the profile's two
    request-arg sets it carries, and whether that set asks the model to think.
    A harness may still ask the host for the other set per request.

    ``reasoning_effort`` is omitted when the run named none, in which case the
    profile's own default applies and ``profile`` is what says which.
    """
    protocol = resolve_profile(profile)(**_worker_profile_kwargs())
    use_thinking = _env_bool('HITL_WORKER_THINKING_ENABLED', default=True)
    meta: dict[str, Any] = {
        'backend': backend,
        'harness': backend_class(backend).product,
        'harness_version': _harness_version(worker_config, backend),
        'profile': profile,
        'model': os.environ.get('HITL_WORKER_MODEL', '').strip() or os.environ.get('OPENAI_MODEL', '').strip(),
        'thinking_enabled': use_thinking and protocol.thinking_on_think_args,
    }
    effort = _worker_reasoning_effort()
    if effort:
        meta['reasoning_effort'] = effort
    return meta


def _user_meta() -> dict[str, Any]:
    """Which model the user simulator ran under."""
    meta: dict[str, Any] = {
        'model': os.environ.get('HITL_HUMAN_MODEL', '').strip() or os.environ.get('OPENAI_MODEL', '').strip(),
    }
    effort = os.environ.get('HITL_HUMAN_REASONING_EFFORT', '').strip()
    if effort:
        meta['reasoning_effort'] = effort
    return meta


def _persona() -> str:
    """The configured persona, refused unless a built-in or a run's own directory has it.

    The directories are read here and named in ``task_config`` for the worker process,
    which resolves the name again at startup.
    """
    load_persona_dirs(_env_list('HITL_USER_PERSONA_DIRS'))
    return _env_choice('HITL_USER_PERSONA', DEFAULT_PERSONA, registered_personas())


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [item for item in (str(item) for item in value) if item]
    if isinstance(value, str) and value:
        return [value]
    return []


__all__ = [
    'AGENT_NAME',
    'CHAT_PROCESSOR',
    'DEFAULT_PERSONA',
    'DEFAULT_WORKER_BACKEND',
    'generate_configs',
    'get_batch_metadata',
]
