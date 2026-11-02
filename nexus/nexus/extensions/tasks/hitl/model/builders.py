"""Assemble an OpenAI worker client from a resolved model protocol.

Pure functions: they read NO environment. The generator owns all env access and
dynamic values, resolves the protocol, and calls these to stamp the protocol onto
a client config dict. Worker (Claude Code harness) config — model strings, haiku,
no-think model, thinking_preset — is assembled by the generator, not here.
"""

from typing import Any

# Ensure profile modules are imported so the registry is populated.
import nexus.extensions.tasks.hitl.model.profiles  # noqa: F401

from nexus.extensions.tasks.hitl.model.base import ModelProtocol, base_request_args


def build_llm_client(
    *, api_key: str, base_url: str, model: str, client_name: str = 'litellm', timeout: float | None = None
) -> dict[str, Any]:
    """Build an OpenAI-compatible client config dict from resolved connection info.

    ``client_name`` selects the registered openai client (e.g. 'litellm' or
    'vertex_anthropic'); it is a deployment choice the generator supplies.

    ``timeout`` (seconds) is a per-request client-side deadline, aborting a request
    that hangs this long. It goes into ``client_args``, which both backends read:
    litellm passes it through to ``acompletion(timeout=...)`` and vertex_anthropic
    reads ``client_args['timeout']`` for its aiohttp deadline. It is independent of
    the timeout passed through to the upstream gateway in the api_key query, which is
    the provider returning an error rather than this side stopping the request.

    ``num_retries=0`` turns off litellm's *internal* retries. Its default is ~3
    inner attempts on a hang or a transient error, each waiting out the full timeout,
    which makes one hung request take ``timeout x 3`` before the outer
    ``configurable_retry`` multiplies it further. With it off, a single timeout is
    exactly ``timeout`` and retries are the outer layer's job
    (``client_args['max_retries']``, exponential backoff). It is written only for the
    litellm backend: ``openai_responses`` forwards every client_arg it does not claim
    straight to ``openai.AsyncOpenAI``, which raises on one it does not know.
    """
    if not api_key:
        raise ValueError('build_llm_client requires a non-empty api_key')
    client_args: dict[str, Any] = {
        'api_key': api_key,
        'base_url': base_url or None,
        'max_retries': 10,
    }
    if client_name == 'litellm':
        client_args['num_retries'] = 0
    if timeout is not None:
        client_args['timeout'] = timeout
    return {
        'name': client_name,
        'config': {
            'client_args': client_args,
            'request_args': {
                'model': model,
            },
        },
    }


def apply_protocol_to_client(
    client: dict[str, Any],
    protocol: ModelProtocol,
    *,
    cache_strategy: str = '',
    use_thinking: bool = True,
) -> None:
    """Stamp a model protocol onto a worker client config.

    The client's base request_args is the thinking path by default; pass
    ``use_thinking=False`` to stamp the non-thinking path instead (the generator
    decides from env). The runtime may still apply the OTHER path per-request as
    an override (carried in the worker config) — this only sets the client's base.

    ``cache_strategy`` (env) takes precedence over the protocol's default; an
    empty resolved strategy leaves caching off.

    ``log_llm_usage`` asks the host to log one usage line per call. It is written only
    for the litellm backend: ``openai_responses`` forwards every client_arg it does not
    claim straight to ``openai.AsyncOpenAI``, which raises on one it does not know. A
    run on that backend reads its counters from the usage export instead.
    """
    client_args = client['config'].setdefault('client_args', {})
    if protocol.client_name == 'litellm':
        client_args['log_llm_usage'] = True

    strategy = cache_strategy or protocol.cache_control_strategy
    if strategy:
        client_args['cache_control_enabled'] = True
        client_args['cache_control_strategy'] = strategy

    request_args = client['config'].setdefault('request_args', {})
    request_args.update(base_request_args(protocol, use_thinking=use_thinking))
