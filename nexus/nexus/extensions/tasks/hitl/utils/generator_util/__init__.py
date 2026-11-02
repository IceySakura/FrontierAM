"""Generator-side helpers shared across HITL recipes.

Holds what a recipe generator needs regardless of which one it is, in four parts:

- :mod:`env` reads the environment and loads a dataset.
- :mod:`clients` builds the LLM client each role gets, and its cache_task_id.
- :mod:`worker` holds the worker configuration that holds for any harness: the user
  memory and skill directories a run supplies, and its MCP servers.
- :mod:`claude_code` builds the ``worker_config`` only a Claude Code backend accepts.
- :mod:`codex` builds the ``worker_config`` only a Codex backend accepts.
- :mod:`backend` picks between those two by the backend a run names.
- :mod:`runtime_provider` builds the sandbox provider's config.

A generator driving a harness with no module here takes the first three and
assembles its own worker config.
"""

from nexus.extensions.tasks.hitl.utils.generator_util.backend import worker_config_for
from nexus.extensions.tasks.hitl.utils.generator_util.claude_code import (
    DEFAULT_CC_TURN_TIMEOUT_SECONDS,
    _build_worker_config,
)
from nexus.extensions.tasks.hitl.utils.generator_util.clients import (
    LITELLM_IMPORT,
    _build_openai_clients,
    _client_from_env,
    _new_session_id,
    client_imports,
    inject_cache_task_ids,
    make_cache_task_id,
    worker_usage_profile,
)
from nexus.extensions.tasks.hitl.utils.generator_util.codex import (
    CATALOG_DIR,
    DEFAULT_CODEX_VERSION,
    _build_codex_worker_config,
)
from nexus.extensions.tasks.hitl.utils.generator_util.env import (
    FALSE_VALUES,
    TRUE_VALUES,
    EnvValueError,
    _env_bool,
    _env_choice,
    _env_int,
    _env_list,
    _load_dataset,
    _require_env,
)
from nexus.extensions.tasks.hitl.utils.generator_util.runtime_provider import _build_runtime_provider_config
from nexus.extensions.tasks.hitl.utils.generator_util.worker import (
    _read_user_memory,
    _search_server,
    _worker_profile_kwargs,
    _worker_reasoning_effort,
    with_worker_skills,
    worker_mcp_servers,
)


__all__ = [
    'CATALOG_DIR',
    'DEFAULT_CC_TURN_TIMEOUT_SECONDS',
    'DEFAULT_CODEX_VERSION',
    'FALSE_VALUES',
    'LITELLM_IMPORT',
    'TRUE_VALUES',
    'EnvValueError',
    '_build_codex_worker_config',
    '_build_openai_clients',
    '_build_runtime_provider_config',
    '_build_worker_config',
    '_client_from_env',
    '_env_bool',
    '_env_choice',
    '_env_int',
    '_env_list',
    '_load_dataset',
    '_new_session_id',
    '_read_user_memory',
    '_require_env',
    '_search_server',
    '_worker_profile_kwargs',
    '_worker_reasoning_effort',
    'client_imports',
    'inject_cache_task_ids',
    'make_cache_task_id',
    'with_worker_skills',
    'worker_config_for',
    'worker_mcp_servers',
    'worker_usage_profile',
]
