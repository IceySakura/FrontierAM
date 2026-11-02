"""Benchmark-local OpenAI clients for stem_mas evaluation runs."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import urllib.error
import urllib.request
import uuid

from copy import deepcopy
from typing import Any, ClassVar
from urllib.parse import urlsplit

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nexus.core import OpenAICompatibleClient, OpenAIConfig, openai_client_registry
from nexus.core.types import ExtendedChatCompletion
from nexus.extensions.tasks.stem_mas.benchmark.eval.providers import HyaOpenAPIContract


logger = logging.getLogger(__name__)
THINK_BLOCK_RE = re.compile(r'<think(?:ing)?>(.*?)</think(?:ing)?>', re.DOTALL | re.IGNORECASE)


def normalize_hya_chat_messages(raw_messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    system_parts: list[str] = []
    body: list[dict[str, str]] = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        role = _chat_role(str(raw.get('role') or 'user'))
        content = _content_to_text(raw.get('content', ''))
        if role == 'system':
            if content:
                system_parts.append(content)
            continue
        if role == 'tool':
            role = 'user'
            tool_call_id = str(raw.get('tool_call_id') or '').strip()
            prefix = f'Tool output for {tool_call_id}:' if tool_call_id else 'Tool output:'
            content = f'{prefix}\n{content}'
        elif role == 'assistant':
            tool_calls = _tool_calls_to_text(raw.get('tool_calls'))
            if tool_calls:
                content = '\n\n'.join(part for part in (content, tool_calls) if part)
        _append_normalized_message(body, role, content)

    messages: list[dict[str, str]] = []
    if system_parts:
        messages.append({'role': 'system', 'content': '\n\n'.join(system_parts)})
    if not body or body[0]['role'] != 'user':
        body.insert(0, {'role': 'user', 'content': ''})
    messages.extend(body)
    if len(messages) == 1:
        messages.insert(0, {'role': 'system', 'content': ''})
    return messages


def ensure_chat_response(data: Any, *, model: str) -> dict[str, Any]:
    if isinstance(data, dict) and isinstance(data.get('choices'), list):
        response = deepcopy(data)
        _normalize_chat_tool_call_response(response)
        if not isinstance(response.get('usage'), dict):
            response['usage'] = {}
        return response

    text = _content_to_text(data)
    return {
        'id': f'chatcmpl-{uuid.uuid4().hex}',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': model,
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
        'usage': {},
    }


def _with_tool_call_instruction(messages: Any, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = list(messages) if isinstance(messages, list) else [{'role': 'user', 'content': ''}]
    instruction = _tool_call_instruction(tools)
    if not instruction:
        return normalized
    return [{'role': 'system', 'content': instruction}, *normalized]


def _tool_call_instruction(tools: list[dict[str, Any]]) -> str:
    entries: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function_value = tool.get('function')
        function: dict[str, Any] = function_value if isinstance(function_value, dict) else {}
        name = str(function.get('name') or tool.get('name') or '').strip()
        if not name:
            continue
        description = str(function.get('description') or tool.get('description') or '').strip()
        parameters_value = function.get('parameters')
        parameters = parameters_value if isinstance(parameters_value, dict) else {}
        entries.append(f'- {name}: {description}\n  parameters: {json.dumps(parameters, ensure_ascii=False)}')
    if not entries:
        return ''
    return (
        'You can call tools using the Hunyuan XML tool-call format. '
        'When a tool is needed, output only:\n'
        '<tool_calls><tool_call>tool_name<tool_sep>'
        '<arg_key>argument_name</arg_key><arg_value>argument_value</arg_value>'
        '</tool_call></tool_calls>\n'
        'Available tools:\n' + '\n'.join(entries)
    )


def _normalize_chat_tool_call_response(response: dict[str, Any]) -> None:
    choices = response.get('choices')
    if not isinstance(choices, list) or not choices:
        return
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get('message') if isinstance(choice.get('message'), dict) else None
    if not isinstance(message, dict):
        return
    _normalize_chat_reasoning_response(message)
    if isinstance(message.get('tool_calls'), list) and message['tool_calls']:
        return
    tool_calls = _parse_hunyuan_xml_tool_calls(_content_to_text(message.get('content', '')))
    if not tool_calls:
        return
    message['tool_calls'] = tool_calls
    message['content'] = ''
    choice['finish_reason'] = 'tool_calls'


def _parse_hunyuan_xml_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in re.finditer(r'<tool_call>(.*?)</tool_call>', text, flags=re.DOTALL):
        body = match.group(1)
        name, sep, arg_text = body.partition('<tool_sep>')
        if not sep:
            continue
        name = name.strip()
        if not name:
            continue
        args: dict[str, str] = {}
        for arg_match in re.finditer(
            r'<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>',
            arg_text,
            flags=re.DOTALL,
        ):
            key = arg_match.group(1).strip()
            if key:
                args[key] = arg_match.group(2).strip()
        calls.append({
            'id': f'call_{uuid.uuid4().hex}',
            'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)},
        })
    return calls


def _normalize_chat_reasoning_response(message: dict[str, Any]) -> None:
    content = _content_to_text(message.get('content', ''))
    if not content:
        return
    reasoning = _reasoning_text_from_content(content)
    if reasoning and not _content_to_text(message.get('reasoning_content', '')):
        message['reasoning_content'] = reasoning
    stripped = _visible_text(content)
    if stripped != content:
        message['content'] = stripped


def _reasoning_text_from_content(text: str) -> str:
    parts = [match.group(1).strip() for match in THINK_BLOCK_RE.finditer(text)]
    return '\n\n'.join(part for part in parts if part)


def _visible_text(text: str) -> str:
    return THINK_BLOCK_RE.sub('', text).strip()


def _content_to_text(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get('type')
                if item_type in {'input_text', 'output_text', 'text'}:
                    parts.append(str(item.get('text') or ''))
                elif item_type == 'input_image':
                    parts.append('[image input omitted]')
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return '\n'.join(part for part in parts if part)
    return json.dumps(content, ensure_ascii=False)


def _chat_role(role: str) -> str:
    if role == 'developer':
        return 'system'
    if role in {'system', 'user', 'assistant', 'tool'}:
        return role
    return 'user'


def _append_normalized_message(messages: list[dict[str, str]], role: str, content: str) -> None:
    if messages and messages[-1]['role'] == role:
        messages[-1]['content'] = '\n\n'.join(part for part in (messages[-1]['content'], content) if part)
        return
    messages.append({'role': role, 'content': content})


def _tool_calls_to_text(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list):
        return ''
    rendered: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get('function') if isinstance(tool_call.get('function'), dict) else {}
        name = str(function.get('name') or '')
        arguments = function.get('arguments', '')
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        call_id = str(tool_call.get('id') or '').strip()
        label = f'Tool call {call_id}' if call_id else 'Tool call'
        rendered.append(f'{label}: {name}({arguments})')
    return '\n'.join(rendered)


class HyaClientArgs(BaseModel):
    """Validated client_args for stem_mas_hya_openapi."""

    model_config = ConfigDict(extra='forbid')

    model: str
    service: str
    authorization: str
    wsid: str
    hya_openapi_contract: HyaOpenAPIContract
    timeout: float = Field(default=900.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0)

    @field_validator('model', 'service', 'authorization', 'wsid', mode='before')
    @classmethod
    def require_text(cls, value: Any, info: Any) -> str:
        text = str(value or '').strip()
        if not text:
            raise ValueError(f'client_args.{info.field_name} is required')
        return text

    @property
    def chat_completions_url(self) -> str:
        path = urlsplit(self.service).path.rstrip('/')
        if path.endswith('/v1'):
            return self.service.rstrip('/') + '/chat/completions'
        raise ValueError('client_args.service must be the one-stop OpenAI-compatible base URL ending with /v1')

    @property
    def request_headers(self) -> dict[str, str]:
        return {'content-type': 'application/json', 'authorization': self.authorization, 'wsid': self.wsid}


class HyaChatRequest(BaseModel):
    """Validated chat-completion request payload before HYA OpenAPI conversion."""

    model_config = ConfigDict(extra='forbid')
    harness_tool_fields: ClassVar[frozenset[str]] = frozenset({'parallel_tool_calls', 'tool_choice', 'tools'})

    model: str
    messages: list[dict[str, Any]]
    temperature: float
    top_p: float
    max_tokens: int
    top_k: int | None = Field(default=None, gt=0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    extra_body: dict[str, Any] | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: Any = None
    parallel_tool_calls: bool | None = None

    @model_validator(mode='before')
    @classmethod
    def reject_unsupported_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            raise ValueError('request_args must be a dict')
        unknown = sorted(key for key in value if key not in cls.model_fields)
        if unknown:
            raise ValueError(f'request_args contains unsupported fields for HYA OpenAPI: {unknown}')
        return value

    @field_validator('tools', mode='before')
    @classmethod
    def validate_tools(cls, value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list) or not all(isinstance(tool, dict) for tool in value):
            raise ValueError('request_args.tools must be a list of tool dictionaries')
        return value

    @field_validator('tool_choice')
    @classmethod
    def validate_tool_choice(cls, value: Any) -> Any:
        if value not in (None, 'none', 'auto'):
            raise ValueError('HYA OpenAPI XML tool adapter only accepts tool_choice auto/none')
        return value

    @field_validator('parallel_tool_calls', mode='before')
    @classmethod
    def validate_parallel_tool_calls(cls, value: Any) -> bool | None:
        if value not in (None, False, True):
            raise ValueError('request_args.parallel_tool_calls must be a boolean')
        return value

    @field_validator('model', mode='before')
    @classmethod
    def require_model(cls, value: Any) -> str:
        text = str(value or '').strip()
        if not text:
            raise ValueError('request_args.model is required')
        return text

    @field_validator('max_tokens', mode='before')
    @classmethod
    def validate_max_tokens(cls, value: Any) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError('request_args.max_tokens must be an integer')
        if value <= 0:
            raise ValueError('request_args.max_tokens must be positive')
        return value

    def to_hya_payload(self, contract: HyaOpenAPIContract) -> dict[str, Any]:
        if contract.max_tokens_cap is not None and self.max_tokens > contract.max_tokens_cap:
            raise ValueError(
                f'request_args.max_tokens={self.max_tokens} exceeds provider cap {contract.max_tokens_cap}'
            )

        payload = self.model_dump(exclude_none=True)
        payload['messages'] = self.to_hya_messages()
        self.merge_default_extra_body(payload, contract.default_extra_body)
        self.apply_reasoning_config(payload, contract)
        self.apply_tool_config(payload)
        return payload

    def to_hya_messages(self) -> list[dict[str, str]]:
        for raw in self.messages:
            self.validate_chat_message(raw)
        messages = normalize_hya_chat_messages(self.messages)
        if not messages:
            raise ValueError('HYA OpenAPI messages must not be empty')
        non_system = [message for message in messages if message['role'] != 'system']
        if not non_system:
            raise ValueError('HYA OpenAPI messages must include at least one non-system message')
        if non_system[0]['role'] != 'user':
            raise ValueError('HYA OpenAPI messages must start with a user message after optional system messages')
        return messages

    @staticmethod
    def merge_default_extra_body(payload: dict[str, Any], default_extra_body: dict[str, Any]) -> None:
        extra_body = payload.get('extra_body')
        if extra_body is None:
            payload['extra_body'] = dict(default_extra_body)
            return
        if not isinstance(extra_body, dict):
            raise ValueError('request_args.extra_body must be a dict')
        payload['extra_body'] = {**default_extra_body, **extra_body}

    @staticmethod
    def apply_reasoning_config(payload: dict[str, Any], contract: HyaOpenAPIContract) -> None:
        payload['chat_template_kwargs'] = {'enable_thinking': contract.reasoning.enabled}
        if contract.reasoning.enabled:
            payload['reasoning_effort'] = contract.reasoning.effort

    def apply_tool_config(self, payload: dict[str, Any]) -> None:
        tools = payload.pop('tools', [])
        payload.pop('tool_choice', None)
        payload.pop('parallel_tool_calls', None)
        if not tools:
            return
        payload['messages'] = _with_tool_call_instruction(payload['messages'], tools)
        payload.setdefault('extra_body', {})
        payload['extra_body']['nexus_tool_call_format'] = 'hunyuan_xml'

    @classmethod
    def validate_chat_message(cls, raw: Any) -> None:
        if not isinstance(raw, dict):
            raise ValueError('HYA OpenAPI messages entries must be dicts')
        role = cls.chat_role(raw.get('role'))
        tool_calls = raw.get('tool_calls')
        if tool_calls is not None:
            if role != 'assistant':
                raise ValueError('HYA OpenAPI tool_calls are only valid on assistant messages')
            if not isinstance(tool_calls, list):
                raise ValueError('HYA OpenAPI assistant tool_calls must be a list')
        tool_call_id = raw.get('tool_call_id')
        if tool_call_id is not None and role != 'tool':
            raise ValueError('HYA OpenAPI tool_call_id is only valid on tool messages')

    @staticmethod
    def chat_role(raw_role: Any) -> str:
        role = str(raw_role or '').strip()
        if role in {'system', 'user', 'assistant', 'tool'}:
            return role
        raise ValueError(f'Unsupported chat message role for HYA OpenAPI: {role!r}')

    @staticmethod
    def content_to_text(content: Any) -> str:
        if content is None:
            return ''
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get('text') or part.get('content') or ''))
                else:
                    parts.append(str(part))
            return ''.join(parts)
        return str(content)


@openai_client_registry.register('stem_mas_hya_openapi')
class StemMasHyaOpenAIClient(OpenAICompatibleClient):
    """Use benchmark-local HYA OpenAPI mapping without starting a gateway process."""

    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        client_args = HyaClientArgs.model_validate(config.client_args)
        if 'reasoning' in self.get_effective_request_args():
            raise ValueError('request_args.reasoning is not supported; set model_provider.generation.reasoning')

        self.contract = client_args.hya_openapi_contract
        self.model = client_args.model
        self.chat_url = client_args.chat_completions_url
        self.request_headers = client_args.request_headers
        self.timeout = client_args.timeout
        self.max_retries = client_args.max_retries

    async def _create_completion(
        self,
        messages: list[ChatCompletionMessageParam | dict],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        request = {
            'model': model or self.model,
            **kwargs,
            **self.config.request_args,
            'messages': messages,
        }
        if self.config.enforced_request_args:
            request = self._resolve_effective_request_args(request)
        extra_headers = request.pop('extra_headers', {})
        if extra_headers is not None and not isinstance(extra_headers, dict):
            raise ValueError('request_args.extra_headers must be a dict')
        model_name = str(request.get('model') or self.model)
        payload = HyaChatRequest.model_validate(request).to_hya_payload(self.contract)
        request_headers = dict(self.request_headers)
        request_headers.update({str(key): str(value) for key, value in (extra_headers or {}).items()})

        response_data = await asyncio.to_thread(self._post_json_with_retries, self.chat_url, payload, request_headers)
        if isinstance(response_data, dict) and response_data.get('error'):
            raise RuntimeError(f'HYA OpenAPI error: {json.dumps(response_data["error"], ensure_ascii=False)[:500]}')
        chat_response = self._ensure_chat_response(response_data, model=model_name)
        logger.debug(
            'HYA OpenAPI response shape: type=%s keys=%s',
            type(response_data).__name__,
            sorted(response_data.keys()) if isinstance(response_data, dict) else [],
        )
        return ExtendedChatCompletion.model_validate(chat_response)

    def _post_json_with_retries(self, url: str, body: dict[str, Any], request_headers: dict[str, str]) -> Any:
        last_error: Exception | None = None
        for _ in range(max(self.max_retries, 1)):
            try:
                return self._post_json(url, body, request_headers)
            except Exception as exc:  # pragma: no cover - exercised in platform smoke.
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError('HYA OpenAPI request failed')

    def _post_json(self, url: str, body: dict[str, Any], request_headers: dict[str, str]) -> Any:
        data = json.dumps(body, ensure_ascii=False).encode('utf-8')
        request = urllib.request.Request(url, data=data, headers=request_headers, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode('utf-8', errors='replace')
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode('utf-8', errors='replace')
            raise RuntimeError(f'HYA OpenAPI HTTP {exc.code}: {raw[:500]}') from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f'HYA OpenAPI response must be JSON: {raw[:500]}') from exc

    def _ensure_chat_response(self, data: Any, *, model: str) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise RuntimeError(f'HYA OpenAPI response must be a dict, got {type(data).__name__}')
        if not isinstance(data.get('choices'), list):
            raise RuntimeError('HYA OpenAPI response must include choices list')

        response = ensure_chat_response(data, model=model)
        for choice in response.get('choices', []):
            if not isinstance(choice, dict):
                raise RuntimeError('HYA OpenAPI response choices entries must be dicts')
            message = choice.get('message')
            if not isinstance(message, dict):
                raise RuntimeError('HYA OpenAPI response choices entries must include message dicts')
            self._normalize_response_message(message)
        response.setdefault('id', f'chatcmpl-{uuid.uuid4().hex}')
        response.setdefault('object', 'chat.completion')
        response.setdefault('created', int(time.time()))
        response.setdefault('model', model)
        return response

    def _normalize_response_message(self, message: dict[str, Any]) -> None:
        content = HyaChatRequest.content_to_text(message.get('content')).strip()
        reasoning_text = HyaChatRequest.content_to_text(message.get(self.contract.response_reasoning_field)).strip()

        message['content'] = content
        if reasoning_text and not message.get(self.contract.normalize_reasoning_to):
            message[self.contract.normalize_reasoning_to] = reasoning_text

        if self.contract.reasoning.enabled and self.contract.require_final_content and not content:
            raise RuntimeError(
                f'Provider returned {self.contract.response_reasoning_field} but no final content; '
                'disable require_final_content only for raw reasoning probes'
            )


__all__ = [
    'HyaChatRequest',
    'HyaClientArgs',
    'HyaOpenAPIContract',
    'StemMasHyaOpenAIClient',
]
