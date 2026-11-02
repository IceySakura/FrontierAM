"""The LLM clients a run gives its roles, and their cache_task_id.

Some LLM gateways (the trpc-gpt-eval family) accept ``&cache_task_id=<id>`` appended
to the api_key and route requests sharing an id to one backing resource. The id is a
32-character md5, one per client. An empty api_key is skipped.
"""

import copy
import hashlib
import logging
import os
import time
import uuid

from collections.abc import Iterable
from typing import Any

from nexus.extensions.tasks.hitl.model import (
    apply_protocol_to_client,
    build_llm_client,
    known_profiles,
    resolve_profile,
)
from nexus.extensions.tasks.hitl.utils.generator_util.env import _env_bool, _env_choice, _env_int
from nexus.extensions.tasks.hitl.utils.generator_util.worker import _worker_profile_kwargs


logger = logging.getLogger(__name__)


LITELLM_IMPORT = 'nexus.extensions.openai_clients.litellm'

_DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS = 900

_CLIENT_IMPORTS = {
    'litellm': LITELLM_IMPORT,
    'vertex_anthropic': 'nexus.extensions.openai_clients.vertex_anthropic',
    'openai_responses': 'nexus.extensions.openai_clients.openai_responses',
}


def make_cache_task_id(*, salt: str = '') -> str:
    """A 32-character md5 cache_task_id, unique per task.

    ``md5(time + app_id + uuid + salt)``. ``salt`` separates clients within one task,
    giving each its own id.
    """
    seed = f'{time.time()}:{os.environ.get("APP_ID", "")}:{uuid.uuid4().hex}:{salt}'
    return hashlib.md5(seed.encode('utf-8')).hexdigest()  # noqa: S324 - an id, not a security digest


def _append_cache_task_id(api_key: str, cache_task_id: str) -> str:
    """Append ``&cache_task_id=<id>`` to the api_key's query, once."""
    if not api_key or 'cache_task_id=' in api_key:
        return api_key
    sep = '&' if '?' in api_key else '?'
    return f'{api_key}{sep}cache_task_id={cache_task_id}'


def inject_cache_task_ids(
    clients: dict[str, Any],
    *,
    roles: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Deep-copy ``clients``, giving each selected one its own cache_task_id.

    ``roles=None`` covers every client; otherwise only those named. A client with an
    empty api_key is skipped.

    Returns a new dict; the argument is not modified.
    """
    role_filter = set(roles) if roles is not None else None
    out = copy.deepcopy(clients)
    for name, client in out.items():
        if role_filter is not None and name not in role_filter:
            continue
        client_args = client.get('config', {}).get('client_args')
        if not isinstance(client_args, dict):
            continue
        api_key = client_args.get('api_key', '')
        if not api_key:
            continue
        client_args['api_key'] = _append_cache_task_id(api_key, make_cache_task_id(salt=name))
    return out


def client_imports(worker_profile: str) -> list[str]:
    """Import paths to register every openai client a run needs.

    Always includes litellm (the human / rubric-eval clients use it); adds the worker's
    client backend when it differs, e.g. vertex_anthropic for gcp_claude. The worker
    client comes from the resolved profile's protocol.client_name.
    """
    imports = [LITELLM_IMPORT]
    worker_client = resolve_profile(worker_profile)(**_worker_profile_kwargs()).client_name
    worker_import = _CLIENT_IMPORTS.get(worker_client)
    if worker_import and worker_import not in imports:
        imports.append(worker_import)
    return imports


def worker_usage_profile(worker_profile: str) -> str:
    """The usage parser for a worker on ``worker_profile``, for the run's processors.

    Which counters a provider reports differs by protocol: Vertex and Bedrock report
    cache creation separately from cache reads, and ``default`` does not read it.
    """
    return resolve_profile(worker_profile)(**_worker_profile_kwargs()).usage_profile


def _client_from_env(prefix: str, *, client_name: str = 'litellm') -> dict[str, Any]:
    """Build a client config from {PREFIX}_API_KEY/_BASE_URL/_MODEL, OPENAI_* fallback."""
    api_key = os.environ.get(f'{prefix}_API_KEY', '') if prefix else ''
    base_url = os.environ.get(f'{prefix}_BASE_URL', '') if prefix else ''
    model = os.environ.get(f'{prefix}_MODEL', '') if prefix else ''
    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY', '')
    if not base_url:
        base_url = os.environ.get('OPENAI_BASE_URL', '')
    if not model:
        model = os.environ.get('OPENAI_MODEL', '')
    if not api_key:
        raise ValueError(f'No API key found for {prefix or "OPENAI"}_API_KEY')
    # Per-request client-side timeout, aborting a hung request. Per role via
    # {PREFIX}_LLM_TIMEOUT, else the global HITL_LLM_TIMEOUT, else the default.
    timeout = _env_int(f'{prefix}_LLM_TIMEOUT') if prefix else 0
    if not timeout:
        timeout = _env_int('HITL_LLM_TIMEOUT', default=_DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS)
    return build_llm_client(
        api_key=api_key, base_url=base_url, model=model, client_name=client_name, timeout=float(timeout)
    )


def _new_session_id() -> str:
    """One routing SessionID per run (consistent-hash cache affinity for multiturn)."""
    return uuid.uuid4().hex


def _with_human_reasoning_effort(client: dict[str, Any]) -> None:
    """Put ``HITL_HUMAN_REASONING_EFFORT`` on the user's requests when it is set.

    Some gateways make the setting mandatory rather than optional: aihub's
    ``/standard/v1`` refuses a request that carries function tools unless the
    request names an effort, and the user simulator sends tools on every turn.
    Unset leaves the request as it was, which is what every endpoint reached so far
    accepts.
    """
    effort = os.environ.get('HITL_HUMAN_REASONING_EFFORT', '').strip()
    if not effort:
        return
    client['config'].setdefault('request_args', {})['reasoning_effort'] = effort


def _with_human_cache_control(client: dict[str, Any]) -> None:
    """Give the user's client the cache-control strategy its profile declares.

    Only an Anthropic-protocol endpoint caches by request marker, and only what the
    request marks. The markers are not inert elsewhere: tagging rewrites a string
    ``content`` into a text-block list, a shape another provider need not accept. So
    the profile decides -- ``bedrock_claude`` and ``gcp_claude`` declare a strategy,
    ``openai_basic`` and ``naci_glm`` declare none and are left alone.

    The name goes through ``_env_choice`` because ``resolve_profile`` answers an
    unknown one with the fallback protocol, whose strategy is empty: a typo would
    otherwise read as "this endpoint needs no markers".
    """
    profile_name = _env_choice('HITL_HUMAN_CLIENT_PROFILE', 'openai_basic', known_profiles())
    protocol = resolve_profile(profile_name)()
    if not protocol.cache_control_strategy:
        return
    client_args = client['config'].setdefault('client_args', {})
    client_args['cache_control_enabled'] = True
    client_args['cache_control_strategy'] = protocol.cache_control_strategy


def _with_responses_worker_args(client: dict[str, Any], *, session_id: str) -> None:
    """Add the Responses args that come from the run rather than the profile.

    Only for ``openai_responses``. The other clients hand an arg they do not claim to
    ``openai.AsyncOpenAI``, which raises on one it does not know.
    """
    if client['name'] != 'openai_responses':
        return
    config = client['config']

    # Puts the raw Responses payloads on provider_call, where a reasoning item's
    # encrypted content can be read back.
    config['client_args']['capture_raw_io'] = _env_bool('HITL_WORKER_CAPTURE_RAW_IO', default=True)

    request_args = config.setdefault('request_args', {})
    # One key per run: the gateway caches its prefix under this.
    request_args['prompt_cache_key'] = session_id or _new_session_id()
    request_args['stream'] = False


def _build_openai_clients(eval_types: list[str], worker_profile: str, *, session_id: str = '') -> dict[str, Any]:
    """Assemble the {name: client_config} mapping for a HITL run."""
    protocol = resolve_profile(worker_profile)(**_worker_profile_kwargs(session_id=session_id))

    human_client = _client_from_env('HITL_HUMAN')
    _with_human_cache_control(human_client)
    _with_human_reasoning_effort(human_client)
    openai_clients: dict[str, Any] = {'human': human_client}

    try:
        worker_client = _client_from_env('HITL_WORKER', client_name=protocol.client_name)
    except ValueError:
        worker_client = _client_from_env('', client_name=protocol.client_name)
    apply_protocol_to_client(
        worker_client,
        protocol,
        cache_strategy=os.environ.get('HITL_WORKER_CACHE_STRATEGY', '').strip(),
        # Thinking is on by default. HITL_WORKER_THINKING_ENABLED=0 stamps the
        # profile's non-thinking request_args onto the client base instead.
        use_thinking=_env_bool('HITL_WORKER_THINKING_ENABLED', default=True),
    )
    _with_responses_worker_args(worker_client, session_id=session_id)
    openai_clients['worker'] = worker_client

    if 'patch_rubrics' in eval_types:
        try:
            openai_clients['rubric_eval'] = _client_from_env('HITL_RUBRIC_EVAL')
        except ValueError:
            openai_clients['rubric_eval'] = human_client

    return openai_clients


__all__ = [
    'LITELLM_IMPORT',
    '_build_openai_clients',
    '_client_from_env',
    '_new_session_id',
    'client_imports',
    'inject_cache_task_ids',
    'make_cache_task_id',
    'worker_usage_profile',
]
