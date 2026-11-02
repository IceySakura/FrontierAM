"""Taiji OpenAI-compatible client with Polaris service discovery.

This client uses Tencent-internal Polaris service discovery to access Taiji LLM services.
It wraps the official OpenAI client and supports standard OpenAI APIs, including tool calling.

Dependency note:
    This client depends on `polaris-cpp-py` for service discovery. The package is not in `pyproject.toml`.
    Install it manually:
        uv pip install /path/to/polaris-cpp-py-*.tar.gz

Config example:
    openai_clients:
      default:
        type: taiji
        client_args:
          model: alex-oss                    # Model name used in /chat/completions
          namespace: Production              # Polaris namespace (default: Production)
          service: taiji-stream-server-online-openapi-prod  # Polaris service name (default provided)
          authorization: Bearer xxx          # Auth token (default provided)
          wsid: '11644'                      # Workspace ID (default provided)
          enable_thinking: true              # Enable thinking mode (default: false)
          timeout: 120                       # Request timeout seconds (default: 120)
          max_retries: 6                     # Max retries (default: 6)
          url_pattern: openapi               # URL pattern: openapi or v1 (default: openapi)
          rate_limit: 100                    # Single process concurrency limit (default: 100)
        request_args:
          temperature: 0.7
          top_p: 1.0
"""

# /// script
# dependencies = []
#
# [project]
# name = "taiji"
# authors = [
#     { name = "adamwzhang" },
#     { name = "felixjjiang" },
# ]
#
# [optional-dependencies]
# host = ["polaris-cpp-py"]
#
# [tool.nexus]
# openai_clients = { taiji = "nexus.extensions.openai_clients.taiji" }
# ///

from __future__ import annotations

import logging
import uuid

from functools import cache
from typing import Any, cast

import openai

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from nexus.core import ContextLengthExceededError, OpenAICompatibleClient, OpenAIConfig, openai_client_registry
from nexus.core.types import ExtendedChatCompletion, ExtendedChoice, ExtendedMessage, ExtendedUsage
from nexus.utils.common import configurable_retry, with_call_deadline


logger = logging.getLogger(__name__)


@cache
def _get_consumer_api():
    """Get or create a Polaris Consumer API instance (cached)."""
    try:
        from polaris.api.consumer import create_consumer_by_config

        return create_consumer_by_config('')
    except ImportError as e:
        raise ImportError(
            'polaris-cpp-py package is required for TaijiClient. '
            'Install it with: uv pip install /path/to/polaris-cpp-py-*.tar.gz'
        ) from e


def _discover_service(namespace: str, service: str) -> tuple[str, int]:
    """Discover a service address via Polaris.

    Note: Each call triggers service discovery (no caching) to tolerate dynamic instance changes.

    Args:
        namespace: Polaris namespace.
        service: Polaris service name.

    Returns:
        (host, port) tuple.
    """
    try:
        from polaris.pkg.model.service import GetOneInstanceRequest

        consumer_api = _get_consumer_api()
        request = GetOneInstanceRequest(namespace=namespace, service=service)
        response = consumer_api.get_one_instance_resp(request)

        if response.is_service_data_expired():
            logger.warning(f'Polaris service {namespace}/{service} data is expired')

        instance = response[0]
        host = instance.get_host()
        port = instance.get_port()
        logger.debug(f'Discovered Polaris service at {host}:{port}')
        return host, port
    except Exception as e:
        logger.error(f'Failed to discover Polaris service: {e}')
        raise RuntimeError(f'Polaris service discovery failed: {e}') from e


def _format_taiji_error(response: ChatCompletion) -> str | None:
    """Extract a readable error from Taiji's OpenAI-shaped error payload."""
    model_extra = getattr(response, 'model_extra', None) or {}
    error = model_extra.get('error')
    if not isinstance(error, dict):
        return None

    err_type = error.get('type') or 'TaijiError'
    message = error.get('message') or error.get('code') or 'unknown error'
    code = error.get('code')
    ret_code = error.get('ret_code')

    details = [f'{err_type}: {message}']
    if code and code != message:
        details.append(f'code={code}')
    if ret_code is not None:
        details.append(f'ret_code={ret_code}')
    return ' | '.join(details)


@openai_client_registry.register('taiji')
class TaijiClient(OpenAICompatibleClient):
    """OpenAI-compatible client for Taiji models via Polaris service discovery.

    Uses Tencent-internal Polaris service discovery to access Taiji LLM services.
    Wraps the official OpenAI client. Supports standard OpenAI APIs, including tool calling and reasoning_content.
    """

    # Defaults
    DEFAULT_NAMESPACE = 'Production'
    DEFAULT_SERVICE = 'taiji-stream-server-online-openapi-prod'
    DEFAULT_AUTHORIZATION = 'Bearer 7auGXNATFSKl7dF'
    DEFAULT_WSID = '11644'
    DEFAULT_TIMEOUT = 120
    DEFAULT_MAX_RETRIES = 6
    DEFAULT_URL_PATTERN = 'openapi'  # 'openapi' or 'v1'
    DEFAULT_RATE_LIMIT = 100

    def __init__(self, config: OpenAIConfig):
        super().__init__(config)

        # Extract config from client_args
        self.model = config.client_args.get('model')
        self.namespace = config.client_args.get('namespace', self.DEFAULT_NAMESPACE)
        self.service = config.client_args.get('service', self.DEFAULT_SERVICE)
        self.authorization = config.client_args.get('authorization', self.DEFAULT_AUTHORIZATION)
        self.wsid = config.client_args.get('wsid', self.DEFAULT_WSID)
        self.enable_thinking = config.client_args.get('enable_thinking', False)
        self.chat_template_kwargs = config.client_args.get('chat_template_kwargs', {})
        self.timeout = float(config.client_args.get('timeout', self.DEFAULT_TIMEOUT))
        self.max_retries = int(config.client_args.get('max_retries', self.DEFAULT_MAX_RETRIES))
        self.url_pattern = config.client_args.get('url_pattern', self.DEFAULT_URL_PATTERN)

        # 从 request_args 提取默认请求参数（仅标准 OpenAI 顶层参数）
        self.default_request_args = {
            'temperature': config.request_args.get('temperature', 0.7),
            'top_p': config.request_args.get('top_p', 1.0),
        }
        if 'max_tokens' in config.request_args:
            self.default_request_args['max_tokens'] = config.request_args['max_tokens']

        # 非标准参数只能通过 extra_body 透传，不能作为顶层参数传给 OpenAI SDK
        self.default_extra_body_args: dict[str, Any] = {}
        for key in ['max_input_seq_len', 'output_seq_len', 'top_k']:
            if key in config.request_args:
                self.default_extra_body_args[key] = config.request_args[key]

        self._logged_kwargs = False

        logger.debug(f'TaijiClient initialized with namespace={self.namespace}, service={self.service}')

    def _create_openai_client(self, host: str, port: int) -> openai.AsyncOpenAI:
        """Create a configured OpenAI client.

        base_url varies by url_pattern:
        - 'openapi': http://{host}:81/openapi (fixed port 81 for taiji-stream-server)
        - 'v1': http://{host}:{port}/v1 (uses port returned by Polaris)
        """
        base_url = f'http://{host}:81/openapi' if self.url_pattern == 'openapi' else f'http://{host}:{port}/v1'

        return openai.AsyncOpenAI(
            base_url=base_url,
            api_key='dummy',  # Taiji does not require a real api_key.
            timeout=self.timeout,
            default_headers={
                'Authorization': self.authorization,
                'wsid': self.wsid,
            },
        )

    async def _create_completion(
        self,
        messages: list[ChatCompletionMessageParam | dict],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        """Create a chat completion using Taiji service.

        Args:
            messages: Chat messages.
            model: Model name (overrides `client_args.model`).
            **kwargs: Additional arguments.
                - tools: Tool definitions
                - tool_choice: Tool selection strategy
                - enable_thinking: Whether to enable thinking mode (overrides config)
                - max_retries: Max retries (overrides config, pass 1 to disable retries)

        Returns:
            ExtendedChatCompletion response.
        """
        # Service discovery
        host, port = _discover_service(self.namespace, self.service)

        # Create OpenAI client
        client = self._create_openai_client(host, port)

        # Resolve model and request args
        actual_model = model or self.model

        enable_thinking = kwargs.pop('enable_thinking', self.enable_thinking)
        max_retries = kwargs.pop('max_retries', self.max_retries)

        # Build extra_body for non-standard parameters
        extra_body: dict[str, Any] = {
            'query_id': str(uuid.uuid4()),
            'openai_infer': True,
            'stream': False,  # Force non-streaming responses.
            'add_generation_prompt': True,
            'chat_template_kwargs': {
                'enable_thinking': enable_thinking,
                **self.chat_template_kwargs,
            },
        }

        # Log chat_template_kwargs once to verify thinking/reasoning settings
        if not self._logged_kwargs:
            logger.debug(f'TaijiClient chat_template_kwargs: {extra_body["chat_template_kwargs"]}')
            self._logged_kwargs = True

        # Log message count and tool presence for multi-turn debugging
        logger.debug(f'TaijiClient request: {len(messages)} messages, tools={"yes" if kwargs.get("tools") else "no"}')

        # Add optional extra parameters to extra_body
        for key in ['max_input_seq_len', 'output_seq_len', 'top_k']:
            if key in kwargs:
                extra_body[key] = kwargs.pop(key)
            elif key in self.default_extra_body_args:
                extra_body[key] = self.default_extra_body_args[key]

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            'model': actual_model,
            'messages': messages,
            'extra_body': extra_body,
            **self.default_request_args,
        }

        # Add standard OpenAI parameters
        for key in ['tools', 'tool_choice', 'temperature', 'top_p', 'max_tokens']:
            if key in kwargs:
                request_kwargs[key] = kwargs[key]
        request_kwargs = self._apply_enforced_request_args(request_kwargs)
        effective_request_args = (
            self._resolve_effective_request_args(request_kwargs)
            if self.config.enforced_request_args
            else request_kwargs
        )
        if not effective_request_args.get('model'):
            raise ValueError('Model must be specified in client_args, enforced_request_args, or as parameter')

        try:
            # Apply the per-call deadline inside the retry layer.
            _create = with_call_deadline()(client.chat.completions.create)
            response = await configurable_retry(max_retries)(_create)(**request_kwargs)

            logger.debug(f'Successfully called Taiji service at {host}:{port}')

            service_error = _format_taiji_error(response)
            if service_error is not None:
                raise RuntimeError(service_error)

            # Convert response and extract reasoning_content
            return self._convert_response(response)

        except openai.BadRequestError as e:
            # Detect context length errors
            error_message = str(e)
            if 'context_length_exceeded' in error_message.lower() or 'maximum context length' in error_message.lower():
                raise ContextLengthExceededError(
                    f'Taiji context length exceeded: {error_message}', original_error=e
                ) from e
            raise
        except Exception as e:
            logger.error(f'Error calling Taiji service at {host}:{port}: {e}')
            raise

    def _convert_response(self, response: ChatCompletion) -> ExtendedChatCompletion:
        """Convert OpenAI response to ExtendedChatCompletion and extract reasoning_content.

        Args:
            response: Response returned by the official OpenAI client.

        Returns:
            ExtendedChatCompletion
        """
        service_error = _format_taiji_error(response)
        if service_error is not None:
            raise RuntimeError(service_error)

        if response.choices is None:
            raise RuntimeError('Taiji returned no choices in chat completion response')

        # Extract reasoning_content from model_extra (if present).
        choices = []
        for choice in response.choices:
            message = choice.message
            reasoning_content = None

            # Try to read reasoning_content from message.model_extra.
            if hasattr(message, 'model_extra') and message.model_extra:
                reasoning_content = message.model_extra.get('reasoning_content')

            ext_message = ExtendedMessage(
                role=message.role,
                content=message.content,
                tool_calls=cast(list[ChatCompletionMessageToolCall], list(message.tool_calls))
                if message.tool_calls
                else None,
                reasoning_content=reasoning_content,
            )

            ext_choice = ExtendedChoice(
                index=choice.index,
                message=ext_message,
                finish_reason=choice.finish_reason,
                logprobs=choice.logprobs,
            )
            choices.append(ext_choice)

        return ExtendedChatCompletion(
            id=response.id,
            model=response.model,
            object=response.object,
            created=response.created,
            choices=choices,
            usage=ExtendedUsage.from_openai(response.usage),
        )
