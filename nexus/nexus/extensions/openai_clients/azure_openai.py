# /// script
# dependencies = []
#
# [project]
# name = "azure_openai"
# authors = [
#     { name = "tarschen" },
# ]
#
# [tool.nexus]
# openai_clients = { azure_openai = "nexus.extensions.openai_clients.azure_openai" }
# ///

from __future__ import annotations

import openai

from nexus.core import ContextLengthExceededError, OpenAICompatibleClient, OpenAIConfig, openai_client_registry
from nexus.core.types import ExtendedChatCompletion
from nexus.extensions.openai_clients.cache_task_id import prepare_cache_task_id
from nexus.utils.common import configurable_retry


@openai_client_registry.register('azure_openai')
class AzureOpenAIClient(OpenAICompatibleClient):
    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        client_args = {k: v for k, v in prepare_cache_task_id(config.client_args).items() if k != 'max_retries'}
        self.client = openai.AsyncAzureOpenAI(**client_args)

    async def _create_completion(self, messages, model=None, **kwargs):
        max_retries = self.config.client_args.get('max_retries', 15)
        request_args = self._apply_enforced_request_args({
            'model': model,
            **self.config.request_args,
            **kwargs,
            'messages': messages,
        })

        try:
            response = await configurable_retry(max_retries)(self.client.chat.completions.create)(**request_args)
            return ExtendedChatCompletion.from_openai(response)
        except openai.BadRequestError as e:
            # Check if this is a context length error
            error_message = str(e)
            if 'context_length_exceeded' in error_message.lower() or 'maximum context length' in error_message.lower():
                raise ContextLengthExceededError(
                    f'Azure OpenAI context length exceeded: {error_message}', original_error=e
                ) from e
            raise
