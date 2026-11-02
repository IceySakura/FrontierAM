"""BFCL handler bridging BFCL's BaseHandler with a Nexus OpenAI-compatible client.

This handler subclasses OpenAICompletionsHandler and replaces the OpenAI client
with a sync wrapper around the Nexus async OpenAI client. This allows BFCL's
synchronous inference pipeline to use Nexus's client configuration.
"""

from __future__ import annotations

import json
import logging
import os
import time

from typing import Any

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import (
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from openai import OpenAI, RateLimitError


logger = logging.getLogger(__name__)


def _get_rss_mb() -> float:
    """Get current process RSS in MB."""
    try:
        with open(f'/proc/{os.getpid()}/status') as f:
            for line in f:
                if line.startswith('VmRSS'):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


class NexusOpenAIHandler(BaseHandler):
    """BFCL handler using a pre-configured OpenAI sync client.

    Instead of reading API credentials from environment variables,
    this handler accepts a pre-built OpenAI sync client instance,
    allowing it to use whatever Nexus client configuration was set up.
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool,
        extra_create_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        self.client = client
        self.extra_create_kwargs = extra_create_kwargs or {}

    # ======================================================================
    # Decode methods
    # ======================================================================

    def decode_ast(self, result, language, has_tool_call_tag):
        if self.is_fc_model:
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                decoded_output.append({name: params})
            return decoded_output
        return default_decode_ast_prompting(result, language, has_tool_call_tag)

    def decode_execute(self, result, has_tool_call_tag):
        if self.is_fc_model:
            return convert_to_function_call(result)
        return default_decode_execute_prompting(result)

    # ======================================================================
    # API call with retry
    # ======================================================================

    @retry_with_backoff(error_type=[RateLimitError, TimeoutError, ConnectionError])
    def generate_with_backoff(self, **kwargs):
        if self.extra_create_kwargs:
            kwargs = {**kwargs, **self.extra_create_kwargs}
        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()
        return api_response, end_time - start_time

    # ======================================================================
    # FC (Function Calling) methods
    # ======================================================================

    def _query_FC(self, inference_data: dict):
        message: list[dict] = inference_data['message']
        tools = inference_data['tools']
        inference_data['inference_input_log'] = {'message': repr(message), 'tools': tools}

        kwargs = {
            'messages': message,
            'model': self.model_name,
            'temperature': self.temperature,
        }

        if len(tools) > 0:
            kwargs['tools'] = tools

        return self.generate_with_backoff(**kwargs)

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data['message'] = []
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry['function']
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)
        inference_data['tools'] = tools
        return inference_data

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        # Log response size and RSS for debugging memory issues
        try:
            resp_content = api_response.choices[0].message.content or ''
            tool_calls = api_response.choices[0].message.tool_calls
            n_tool_calls = len(tool_calls) if tool_calls else 0
            content_len = len(resp_content)
            args_len = sum(len(tc.function.arguments or '') for tc in tool_calls) if tool_calls else 0
            rss = _get_rss_mb()

            if content_len > 100_000 or args_len > 100_000:
                logger.warning(
                    f'[handler] LARGE API response: content={content_len}, args_total={args_len}, '
                    f'tool_calls={n_tool_calls}, RSS={rss:.0f}MB'
                )
            else:
                logger.debug(
                    f'[handler] API response: content={content_len}, args_total={args_len}, '
                    f'tool_calls={n_tool_calls}, RSS={rss:.0f}MB'
                )

            # Check for reasoning_content size
            msg = api_response.choices[0].message
            if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                rc_len = len(msg.reasoning_content)
                if rc_len > 100_000:
                    logger.warning(f'[handler] LARGE reasoning_content: {rc_len} chars, RSS={rss:.0f}MB')
        except Exception:
            pass

        try:
            model_responses = [
                {func_call.function.name: func_call.function.arguments}
                for func_call in api_response.choices[0].message.tool_calls
            ]
            tool_call_ids = [func_call.id for func_call in api_response.choices[0].message.tool_calls]
        except Exception:
            model_responses = api_response.choices[0].message.content
            tool_call_ids = []

        model_responses_message_for_chat_history = api_response.choices[0].message

        response_data = {
            'model_responses': model_responses,
            'model_responses_message_for_chat_history': model_responses_message_for_chat_history,
            'tool_call_ids': tool_call_ids,
            'input_token': api_response.usage.prompt_tokens,
            'output_token': api_response.usage.completion_tokens,
        }

        # Strip reasoning_content for DeepSeek models
        self._add_reasoning_content_if_available_FC(api_response, response_data)

        return response_data

    def add_first_turn_message_FC(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data['message'].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_FC(self, inference_data: dict, user_message: list[dict]) -> dict:
        inference_data['message'].extend(user_message)
        return inference_data

    def _add_assistant_message_FC(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data['message'].append(model_response_data['model_responses_message_for_chat_history'])
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        for execution_result, tool_call_id in zip(execution_results, model_response_data['tool_call_ids']):
            tool_message = {
                'role': 'tool',
                'content': execution_result,
                'tool_call_id': tool_call_id,
            }
            inference_data['message'].append(tool_message)
        return inference_data

    def _add_reasoning_content_if_available_FC(self, api_response: Any, response_data: dict) -> None:
        """Strip reasoning_content from chat history for DeepSeek-style models.

        Saves reasoning_content to response_data for logging but removes it
        from the message used in chat history to avoid API errors.
        """
        message = api_response.choices[0].message

        # Preserve tool_call information but strip reasoning_content
        if getattr(message, 'tool_calls', None):
            assistant_message = {
                'role': 'assistant',
                'content': message.content,
                'tool_calls': [
                    {
                        'id': tool_call.id,
                        'type': tool_call.type,
                        'function': {
                            'name': tool_call.function.name,
                            'arguments': tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            }
            response_data['model_responses_message_for_chat_history'] = assistant_message
        elif hasattr(message, 'reasoning_content'):
            response_data['model_responses_message_for_chat_history'] = {
                'role': 'assistant',
                'content': message.content,
            }

        if hasattr(message, 'reasoning_content'):
            response_data['reasoning_content'] = message.reasoning_content

    # ======================================================================
    # Prompting methods
    # ======================================================================

    def _query_prompting(self, inference_data: dict):
        inference_data['inference_input_log'] = {'message': repr(inference_data['message'])}

        return self.generate_with_backoff(
            messages=inference_data['message'],
            model=self.model_name,
            temperature=self.temperature,
        )

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry['function']
        test_entry_id: str = test_entry['id']

        test_entry['question'][0] = system_prompt_pre_processing_chat_model(
            test_entry['question'][0], functions, test_entry_id
        )

        return {'message': []}

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        response_data = {
            'model_responses': api_response.choices[0].message.content,
            'model_responses_message_for_chat_history': api_response.choices[0].message,
            'input_token': api_response.usage.prompt_tokens,
            'output_token': api_response.usage.completion_tokens,
        }

        # Strip reasoning_content for DeepSeek-style models
        self._add_reasoning_content_if_available_prompting(api_response, response_data)

        return response_data

    def add_first_turn_message_prompting(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data['message'].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(self, inference_data: dict, user_message: list[dict]) -> dict:
        inference_data['message'].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data['message'].append(model_response_data['model_responses_message_for_chat_history'])
        return inference_data

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        inference_data['message'].append({'role': 'user', 'content': formatted_results_message})
        return inference_data

    def _add_reasoning_content_if_available_prompting(self, api_response: Any, response_data: dict) -> None:
        """Strip reasoning_content from prompting chat history for DeepSeek-style models."""
        message = api_response.choices[0].message
        if hasattr(message, 'reasoning_content'):
            response_data['reasoning_content'] = message.reasoning_content
            response_data['model_responses_message_for_chat_history'] = {
                'role': 'assistant',
                'content': str(response_data['model_responses']),
            }
