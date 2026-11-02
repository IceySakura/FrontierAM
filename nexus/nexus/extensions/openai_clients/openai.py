# /// script
# dependencies = []
#
# [project]
# name = "openai"
# authors = [
#     { name = "tarschen" },
# ]
#
# [tool.nexus]
# openai_clients = { openai = "nexus.extensions.openai_clients.openai" }
# run_parts = { openai = "nexus.extensions.openai_clients.openai" }
# ///

from __future__ import annotations

import copy

from typing import Any

import httpx
import openai

from nexus.core import ContextLengthExceededError, OpenAICompatibleClient, OpenAIConfig, openai_client_registry
from nexus.core.types import ExtendedChatCompletion
from nexus.extensions.openai_clients.cache_task_id import prepare_cache_task_id
from nexus.runner.compose import run_part
from nexus.utils.common import configurable_retry, with_call_deadline


@run_part('openai')
def openai_run_part(params: dict[str, Any]) -> dict[str, Any]:
    if params:
        raise ValueError(f'Run part openai does not accept parameters: {sorted(params)}')
    return {
        'openai_clients': {
            'main': {
                'name': 'openai',
                'config': {
                    'client_args': {'api_key': '${oc.env:OPENAI_API_KEY,""}'},
                    'request_args': {'model': '${oc.env:OPENAI_MODEL,""}'},
                },
            },
        },
    }


@openai_client_registry.register('openai')
class OpenAIClient(OpenAICompatibleClient):
    """OpenAI client with optional Anthropic prompt caching support.

    The gateway does not auto-cache: prompt caching only kicks in when messages/tools
    carry explicit ``cache_control`` breakpoints. Enable via client_args (same keys as
    the litellm client):
        - cache_control_enabled: bool = False
        - cache_control_strategy: str = "default" ("claude_code" | "<module>::<func>" | "default")
        - cache_control_last_n_messages: int = 2
        - cache_control_tagged_roles: list[str] = ["user", "tool"]
    """

    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        # cache_control_* are internal knobs, not AsyncOpenAI constructor args — strip them.
        client_args = {
            k: v
            for k, v in prepare_cache_task_id(config.client_args).items()
            if k != 'max_retries' and not k.startswith('cache_control_')
        }
        # Keep structured timeouts serializable until the SDK client is constructed.
        if isinstance(client_args.get('timeout'), dict):
            client_args['timeout'] = httpx.Timeout(**client_args['timeout'])
        if 'sdk_max_retries' in client_args:
            client_args['max_retries'] = client_args.pop('sdk_max_retries')
        self.client = openai.AsyncOpenAI(**client_args)

    async def _create_completion(self, messages, model=None, **kwargs):
        max_retries = self.config.client_args.get('max_retries', 15)

        if self.config.client_args.get('cache_control_enabled'):
            strategy = self.config.client_args.get('cache_control_strategy', 'default')
            cache_control_kwargs = {
                k.removeprefix('cache_control_'): v
                for k, v in self.config.client_args.items()
                if k.startswith('cache_control_')
                and k not in {'cache_control_enabled', 'cache_control_strategy'}
                and v is not None
            }
            if strategy == 'claude_code':
                messages = apply_cache_control_claude_code(messages)
                if 'tools' in kwargs:
                    kwargs['tools'] = apply_tools_cache_control(kwargs['tools'])
            elif '::' in strategy:
                from nexus.core.registry import import_from_path

                messages = import_from_path(strategy)(messages, **cache_control_kwargs)
            else:
                messages = apply_cache_control(messages, **cache_control_kwargs)

        request_args_override = kwargs.pop('_request_args_override', None)
        request_args = self._apply_enforced_request_args({
            'model': model,
            **(request_args_override if request_args_override is not None else self.config.request_args),
            **kwargs,
            'messages': messages,
        })

        try:
            # Apply the per-call deadline inside the retry layer so timed-out requests can be retried.
            _create = with_call_deadline()(self.client.chat.completions.create)

            # Reject responses without choices inside the retry layer so they receive the same retry handling.
            async def _create_checked(**kw):
                r = await _create(**kw)
                if not getattr(r, 'choices', None):
                    raise openai.APIError(
                        f'upstream returned empty choices (choices={getattr(r, "choices", None)!r}); '
                        'retrying — see comment at openai.py:_create_completion',
                        request=httpx.Request('POST', self.client.base_url.join('chat/completions')),
                        body=None,
                    )
                return r

            response = await configurable_retry(max_retries)(_create_checked)(**request_args)
            return ExtendedChatCompletion.from_openai(response)
        except openai.BadRequestError as e:
            # Check if this is a context length error
            error_message = str(e)
            if 'context_length_exceeded' in error_message.lower() or 'maximum context length' in error_message.lower():
                raise ContextLengthExceededError(
                    f'OpenAI context length exceeded: {error_message}', original_error=e
                ) from e
            raise


# NOTE: These cache-control helpers are intentionally a private copy of the same
# functions in litellm.py. They are duplicated (not shared) so the litellm client —
# used by other tasks — stays untouched, and so importing this module never pulls in
# the litellm package. Keep the two copies in sync if the caching logic changes.
def apply_cache_control(
    messages: list[dict[str, Any]],
    last_n_messages: int = 2,
    tagged_roles: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Apply cache control marks to messages for Anthropic Claude prompt caching."""
    if tagged_roles is None:
        tagged_roles = ['user', 'tool']

    new_messages = copy.deepcopy(messages)
    n_tagged = 0

    for entry in reversed(new_messages):
        if n_tagged < last_n_messages and entry.get('role') in tagged_roles:
            content = entry.get('content')
            if isinstance(content, str):
                entry['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
            elif isinstance(content, list) and content:
                content[-1]['cache_control'] = {'type': 'ephemeral'}
            n_tagged += 1

    return new_messages


def _tag_message_cache_control(entry: dict[str, Any]) -> bool:
    """Tag a single message with cache_control. Returns True if successfully tagged."""
    content = entry.get('content')
    if content is None:
        return False

    if isinstance(content, str) and content.strip():
        entry['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        return True

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                text = block.get('text', '')
                if text.strip():
                    block['cache_control'] = {'type': 'ephemeral'}
                    return True

    return False


def apply_cache_control_claude_code(
    messages: list[dict[str, Any]],
    tagged_roles: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Apply cache control for Claude Code proxy requests.

    Tags only the last eligible user/tool message with content validation.
    """
    if tagged_roles is None:
        tagged_roles = ['user', 'tool']

    new_messages = copy.deepcopy(messages)

    for entry in reversed(new_messages):
        if entry.get('role') in tagged_roles and _tag_message_cache_control(entry):
            break

    return new_messages


def apply_tools_cache_control(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tag the last tool definition with cache_control.

    Cache prefix order is system → tools → messages. Tagging the last tool
    caches the entire system + tools prefix as one unit.
    """
    if not tools:
        return tools

    new_tools = copy.deepcopy(tools)
    new_tools[-1]['cache_control'] = {'type': 'ephemeral'}
    return new_tools
