# /// script
# dependencies = []
#
# [project]
# name = "vertex_anthropic"
# authors = [
#     { name = "kechizhang" },
# ]
#
# [tool.nexus]
# openai_clients = { vertex_anthropic = "nexus.extensions.openai_clients.vertex_anthropic" }
# ///

from __future__ import annotations

import warnings

from nexus.core import OpenAIConfig, openai_client_registry
from nexus.extensions.openai_clients.vertex import (
    VertexAnthropicHTTPError,
    VertexClient,
    _StreamAccumulator,
    _build_payload,
    _content_text,
    _convert_assistant_message,
    _convert_image_url,
    _convert_response,
    _convert_text_part,
    _convert_tool_choice,
    _convert_tool_result_content,
    _convert_user_content,
    _default_max_tokens_for_model,
    _normalize_output_config,
    _normalize_thinking,
    _parse_sse_event,
    _parse_tool_arguments,
    _read_stream_response,
    _safe_json_loads,
    _stop_reason_to_finish_reason,
    _tag_message_cache_control,
    _thinking_enabled,
    _tool_call_to_dict,
    apply_cache_control_claude_code,
    apply_tools_cache_control,
    convert_messages_to_anthropic,
    convert_tools_to_anthropic,
)


@openai_client_registry.register('vertex_anthropic')
class VertexAnthropicClient(VertexClient):
    """Deprecated compatibility alias for ``vertex`` with the Anthropic protocol."""

    def __init__(self, config: OpenAIConfig):
        warnings.warn(
            'vertex_anthropic is deprecated; use vertex with protocol=anthropic',
            DeprecationWarning,
            stacklevel=2,
        )
        client_args = {**config.client_args, 'protocol': 'anthropic'}
        super().__init__(
            OpenAIConfig(
                client_args=client_args,
                request_args=dict(config.request_args),
                enforced_request_args=dict(config.enforced_request_args),
            )
        )


__all__ = [
    'VertexAnthropicClient',
    'VertexAnthropicHTTPError',
    'apply_cache_control_claude_code',
    'apply_tools_cache_control',
    'convert_messages_to_anthropic',
    'convert_tools_to_anthropic',
    '_StreamAccumulator',
    '_build_payload',
    '_content_text',
    '_convert_assistant_message',
    '_convert_image_url',
    '_convert_response',
    '_convert_text_part',
    '_convert_tool_choice',
    '_convert_tool_result_content',
    '_convert_user_content',
    '_default_max_tokens_for_model',
    '_normalize_output_config',
    '_normalize_thinking',
    '_parse_sse_event',
    '_parse_tool_arguments',
    '_read_stream_response',
    '_safe_json_loads',
    '_stop_reason_to_finish_reason',
    '_tag_message_cache_control',
    '_thinking_enabled',
    '_tool_call_to_dict',
]
