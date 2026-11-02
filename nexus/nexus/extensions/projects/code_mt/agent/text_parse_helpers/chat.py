"""Chat session management and LLM utilities for Terminus2 agent."""

from dataclasses import dataclass
from typing import Any

from nexus.core import require_context
from nexus.utils.timeout import TimeBudget, await_completion_with_time_budget


@dataclass
class LLMResponse:
    content: str
    reasoning_content: str | None = None
    usage: dict | None = None
    raw_message: dict | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    thinking_blocks: list[dict] | None = None
    signature: str | None = None


class OutputLengthExceededError(Exception):
    """Raised when the LLM response was truncated due to max_tokens limit."""

    def __init__(
        self,
        message: str,
        truncated_response: str | None = None,
        raw_message: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.truncated_response = truncated_response
        self.raw_message = raw_message


async def create_completion_with_budget(
    client: Any,
    messages: list[dict[str, Any]],
    inference_budget: TimeBudget | None,
):
    if inference_budget is None:
        return await client.create_completion(messages=messages)
    return await await_completion_with_time_budget(
        inference_budget,
        lambda: client.create_completion(messages=messages),
    )


async def lite_llm_call(
    prompt: str,
    message_history: list[dict[str, Any]] | None = None,
    *,
    inference_budget: TimeBudget | None = None,
):
    if message_history is None:
        message_history = []
    ctx = require_context()
    client = ctx.openai_client
    messages = message_history + [{'role': 'user', 'content': prompt}]
    response = await create_completion_with_budget(client, messages, inference_budget)
    choice = response.choices[0]
    message = choice.message
    raw_message = message.model_dump(exclude_none=True)
    content = message.content
    reasoning_content = message.reasoning_content if message.reasoning_content else None

    # Some reasoning parsers return the usable response in reasoning_content
    # with content=None. Use it for text_parse parsing, but keep raw_message
    # byte-for-byte equivalent to the completion cached by HyTuner.
    effective_content = reasoning_content if reasoning_content and content is None else content

    # Sometimes the LLM returns a response with a finish reason of "length"
    # This typically means we hit the max_tokens limit, not the context window
    if choice.finish_reason == 'length':
        # Preserve the complete assistant message so output-length retries do
        # not rewrite the history cached by HyTuner.
        exc = OutputLengthExceededError(
            'Model hit max_tokens limit. Response was truncated. Consider increasing max_tokens if possible.',
            truncated_response=effective_content,
            raw_message=raw_message,
        )
        raise exc

    return LLMResponse(
        content=effective_content,
        reasoning_content=reasoning_content,
        usage=getattr(response, 'usage', None),
        raw_message=raw_message,
        signature=None,
        thinking_blocks=None,
    )


class Chat:
    """Chat session management for Terminus2 agent."""

    def __init__(self, *, inference_budget: TimeBudget | None = None):
        self._messages = []
        self.inference_budget = inference_budget
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0
        self._cumulative_cache_tokens = 0
        self._cumulative_cost = 0.0
        self._prompt_token_ids_list: list[list[int]] = []
        self._completion_token_ids_list: list[list[int]] = []
        self._logprobs_list: list[list[float]] = []

    @property
    def total_input_tokens(self) -> int:
        return self._cumulative_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._cumulative_output_tokens

    @property
    def total_cache_tokens(self) -> int:
        return self._cumulative_cache_tokens

    @property
    def total_cost(self) -> float:
        return self._cumulative_cost

    @property
    def messages(self) -> list:
        return self._messages

    async def chat(
        self,
        prompt: str,
    ) -> LLMResponse:
        llm_response: LLMResponse = await lite_llm_call(
            prompt=prompt,
            message_history=self._messages,
            inference_budget=self.inference_budget,
        )

        usage = llm_response.usage
        if usage is not None:
            self._cumulative_input_tokens += getattr(usage, 'prompt_tokens', 0) or 0
            self._cumulative_output_tokens += getattr(usage, 'completion_tokens', 0) or 0
            cached = None
            if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                cached = getattr(usage.prompt_tokens_details, 'cached_tokens', None)
            self._cumulative_cache_tokens += cached or 0
            self._cumulative_cost += getattr(usage, 'cost', 0) or 0

        self._messages.append({'role': 'user', 'content': prompt})
        if llm_response.raw_message:
            self._messages.append(llm_response.raw_message)
        else:
            self._messages.append({'role': 'assistant', 'content': llm_response.content})
        return llm_response


def count_total_tokens(chat: Chat, model_name: str) -> int:
    """Count total tokens across all messages in the chat."""
    import litellm

    litellm.__dict__['suppress_debug_info'] = True
    from litellm.utils import token_counter

    return token_counter(model=model_name, messages=chat.messages)
