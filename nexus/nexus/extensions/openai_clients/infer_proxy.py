# /// script
# dependencies = []
#
# [project]
# name = "infer_proxy"
# authors = [
#     { name = "mannaliang" },
# ]
#
# [tool.nexus]
# openai_clients = { infer_proxy = "nexus.extensions.openai_clients.infer_proxy" }
# ///

from __future__ import annotations

import uuid

import openai

from nexus.core import ContextLengthExceededError, OpenAICompatibleClient, OpenAIConfig, openai_client_registry
from nexus.core.request_admission import BoundRequestAdmission, current_request_admission
from nexus.core.types import ExtendedChatCompletion
from nexus.extensions.openai_clients.cache_task_id import prepare_cache_task_id
from nexus.utils.common import configurable_retry


@openai_client_registry.register('infer_proxy')
class InferProxyClient(OpenAICompatibleClient):
    """OpenAI-compatible client for Hunyuan Infer Proxy with reasoning_content support."""

    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        client_args = {k: v for k, v in prepare_cache_task_id(config.client_args).items() if k != 'max_retries'}
        if client_args.pop('tide_qpm', None) is not None:
            raise ValueError('tide_qpm cannot be combined with platform request_policy budget')
        self._client_id = f'infer_proxy_{uuid.uuid4().hex[:12]}'
        self._bound_admission = current_request_admission()
        self._local_policy_config = client_args.pop('local_request_policy', None)
        self._local_admission: BoundRequestAdmission | None = None
        if self._bound_admission is not None:
            # The outer retry reacquires a permit for every actual send attempt.
            client_args['max_retries'] = 0
        client_args.setdefault('timeout', 600)
        self.timeout = client_args['timeout']
        self.client = openai.AsyncOpenAI(**client_args)

    async def _create_completion(self, messages, model=None, **kwargs):
        max_retries = self.config.client_args.get('max_retries', 15)

        # Merge extra_body from request_args and runtime kwargs
        extra_body = {
            **self.config.request_args.get('extra_body', {}),
            **kwargs.pop('extra_body', {}),
        }

        # Build final request parameters
        request_params = {
            'model': model,
            **{k: v for k, v in self.config.request_args.items() if k != 'extra_body'},
            **kwargs,
            'messages': messages,
        }
        if extra_body:
            request_params['extra_body'] = extra_body
        request_params = self._apply_enforced_request_args(request_params)

        if self._local_policy_config is not None and not (current_request_admission() or self._bound_admission):
            from nexus.extensions.openai_clients.infer_proxy_local_policy import local_request_admission

            configured_model = self.get_effective_request_args().get('model')
            if self._resolve_effective_request_args(request_params).get('model') != configured_model:
                raise ValueError('local_request_policy requests must use the configured model')
            if self._local_admission is None:
                self._local_admission = local_request_admission(
                    model=configured_model,
                    api_key=self.client.api_key,
                    config=self._local_policy_config,
                    client_id=self._client_id,
                )

        async def send_completion():
            active_bound = current_request_admission() or self._bound_admission or self._local_admission
            if active_bound is not None:
                permit = await active_bound.acquire(client_id=self._client_id)
                permit.ensure_valid()
                target_client = self.client if self.client.max_retries == 0 else self.client.with_options(max_retries=0)
                return await target_client.chat.completions.create(**request_params)
            return await self.client.chat.completions.create(**request_params)

        try:
            response = await configurable_retry(max_retries)(send_completion)()
            result = ExtendedChatCompletion.from_openai(response)
            # Enrich with reasoning_content (not covered by standard from_openai)
            for orig, ext in zip(response.choices, result.choices, strict=False):
                ext.message.reasoning_content = getattr(orig.message, 'reasoning_content', None)
            return result
        except openai.APITimeoutError as e:
            raise TimeoutError(f'Infer Proxy request timed out after {self.timeout}s: {e}') from e
        except openai.BadRequestError as e:
            error_message = str(e)
            if 'context_length_exceeded' in error_message.lower() or 'maximum context length' in error_message.lower():
                raise ContextLengthExceededError(
                    f'Infer Proxy context length exceeded: {error_message}', original_error=e
                ) from e
            raise
