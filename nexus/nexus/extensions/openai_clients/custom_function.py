# /// script
# dependencies = []
#
# [project]
# name = "custom_function"
# authors = [
#     { name = "tarschen" },
# ]
#
# [tool.nexus]
# openai_clients = { custom_function = "nexus.extensions.openai_clients.custom_function" }
# ///

from __future__ import annotations

from openai.types.chat import ChatCompletion

from nexus.core import OpenAICompatibleClient, OpenAIConfig, openai_client_registry
from nexus.core.types import ExtendedChatCompletion
from nexus.utils.common import configurable_retry


@openai_client_registry.register('custom_function')
class CustomFunctionClient(OpenAICompatibleClient):
    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        self.client = config.client_args['function']

    async def _create_completion(self, messages, model=None, **kwargs):
        max_retries = self.config.client_args.get('max_retries', 15)
        request_args = self._apply_enforced_request_args({
            'model': model,
            **self.config.request_args,
            **kwargs,
            'messages': messages,
        })
        response = await configurable_retry(max_retries)(self.client)(**request_args)
        # Custom function may return ChatCompletion or ExtendedChatCompletion
        if isinstance(response, ExtendedChatCompletion):
            return response
        if isinstance(response, ChatCompletion):
            return ExtendedChatCompletion.from_openai(response)
        # Assume dict-like response, try to construct ExtendedChatCompletion
        return ExtendedChatCompletion.model_validate(response)
