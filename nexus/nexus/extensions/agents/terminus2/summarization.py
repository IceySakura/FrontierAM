"""Context summarization for Terminus2 agent.

This module provides functions for summarizing conversation context when
approaching token limits, using a three-step handoff process.
"""

import logging

from typing import TYPE_CHECKING

from nexus.core import require_context
from nexus.extensions.agents.terminus2.chat import lite_llm_call
from nexus.extensions.agents.terminus2.prompts import (
    ANSWER_REQUEST_TEMPLATE,
    HANDOFF_TEMPLATE,
    QUESTION_PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
)


if TYPE_CHECKING:
    from nexus.extensions.agents.terminus2.agent import Chat

logger = logging.getLogger(__name__)


def get_model_context_limit(model_name: str) -> int:
    """Get the context limit (max input tokens) for the current model.

    Args:
        model_name: The model name to look up

    Returns:
        int: The maximum input tokens the model can accept, or a fallback value if unavailable.
    """
    fallback_context_limit = 200000
    try:
        import litellm

        litellm.__dict__['suppress_debug_info'] = True
        from litellm import get_model_info

        model_info = get_model_info(model_name)
        max_input_tokens = model_info.get('max_input_tokens')

        # Fallback to max_tokens if max_input_tokens not available
        if max_input_tokens is None:
            max_input_tokens = model_info.get('max_tokens')

        return max_input_tokens or fallback_context_limit
    except Exception:
        return fallback_context_limit


def get_model_output_limit(model_name: str) -> int | None:
    """Get the output limit (max output tokens) for the current model.

    Args:
        model_name: The model name to look up

    Returns:
        int or None: The maximum output tokens, or None if unavailable.
    """
    try:
        import litellm

        litellm.__dict__['suppress_debug_info'] = True
        from litellm import get_model_info

        model_info = get_model_info(model_name)
        return model_info.get('max_output_tokens')
    except Exception:
        return None


def unwind_messages_to_free_tokens(
    history: list[dict],
    current_total_tokens: int,
    context_limit: int,
    target_free_tokens: int = 4000,
) -> list[dict]:
    """Remove recent messages until we have enough free tokens.

    Uses actual token count from LLM response rather than estimation.

    Args:
        history: Message history list
        current_total_tokens: Current total tokens from last LLM response usage
        context_limit: Model's context limit in tokens
        target_free_tokens: Target number of free tokens to achieve

    Returns:
        Modified history with older messages removed
    """
    free_tokens = context_limit - current_total_tokens

    while len(history) > 1 and free_tokens < target_free_tokens:
        # Remove the most recent pair of messages (user + assistant)
        if len(history) >= 2:
            # Estimate tokens freed (rough estimate: ~500 tokens per message pair)
            history = history[:-2]
            free_tokens += 1000  # Rough estimate for removed pair
        else:
            break

    logger.debug(f'Unwound messages. Remaining messages: {len(history)}')
    return history


async def summarize_context(
    client,
    history: list[dict],
    original_instruction: str,
    current_screen: str,
) -> tuple[str, list[dict]]:
    """Create a summary of the agent's work to pass to a new agent instance.

    This implements a three-step context summarization process:

    **Step 1: Summary Generation**
    - Generate comprehensive summary of all work completed so far

    **Step 2: Question Asking**
    - Given the summary and current terminal, generate questions about missing info

    **Step 3: Answer Providing**
    - Answer the questions based on full conversation history

    Args:
        client: OpenAI-compatible client
        history: Current message history
        original_instruction: The original task instruction
        current_screen: Current terminal screen content

    Returns:
        tuple: (handoff_prompt, new_history)
            - handoff_prompt: The prompt to continue with (includes answers + instructions)
            - new_history: Compressed history for the next iteration
    """
    if len(history) == 0:
        return original_instruction, history

    logger.debug('Starting context summarization...')

    # ===== Step 1: Summary Generation =====
    summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(original_instruction=original_instruction)

    # Call LLM with current history + summary request
    summary_messages = history + [{'role': 'user', 'content': summary_prompt}]
    summary_response = await client.create_completion(messages=summary_messages)
    summary_content = summary_response.choices[0].message.content or ''

    logger.debug(f'Summary generated: {summary_content[:200]}...')

    # ===== Step 2: Question Asking =====
    question_prompt = QUESTION_PROMPT_TEMPLATE.format(
        original_instruction=original_instruction,
        summary=summary_content,
        current_screen=current_screen,
    )

    # Fresh context - just ask questions based on summary
    question_messages = [{'role': 'user', 'content': question_prompt}]
    questions_response = await client.create_completion(messages=question_messages)
    questions_content = questions_response.choices[0].message.content or ''

    logger.debug(f'Questions generated: {questions_content[:200]}...')

    # ===== Step 3: Answer Providing =====
    answer_request = ANSWER_REQUEST_TEMPLATE.format(questions=questions_content)

    # Answer with original history context only (matching Harbor implementation)
    answer_messages = history + [{'role': 'user', 'content': answer_request}]
    answers_response = await client.create_completion(messages=answer_messages)
    answers_content = answers_response.choices[0].message.content or ''

    logger.debug(f'Answers generated: {answers_content[:200]}...')

    # ===== Build new compressed history =====
    # Keep first message (system/initial prompt) + question exchange
    new_history = [
        history[0],  # Keep initial prompt
        {'role': 'user', 'content': question_prompt},
        {'role': 'assistant', 'content': questions_content},
    ]

    # Build handoff prompt
    handoff_prompt = HANDOFF_TEMPLATE.format(answers=answers_content)

    logger.info('Context summarization completed')

    return handoff_prompt, new_history


def is_context_length_exceeded_error(error: Exception) -> bool:
    """Check if an exception is a context length exceeded error.

    Checks for:
    1. LiteLLM's ContextWindowExceededError
    2. LiteLLM's BadRequestError with context length related messages
    3. String patterns in error message for other providers
    """
    # Check for LiteLLM specific exceptions
    try:
        import litellm

        litellm.__dict__['suppress_debug_info'] = True
        from litellm.exceptions import BadRequestError as LiteLLMBadRequestError
        from litellm.exceptions import ContextWindowExceededError as LiteLLMContextWindowExceededError

        if isinstance(error, LiteLLMContextWindowExceededError):
            return True

        if isinstance(error, LiteLLMBadRequestError):
            # Check error message for context length patterns
            parts = [
                str(error),
                str(getattr(error, 'body', '')),
                str(getattr(error, 'message', '')),
                str(getattr(error, 'error', '')),
            ]
            combined = ' '.join(part.lower() for part in parts if part)
            phrases = (
                'context length exceeded',
                'context_length_exceeded',
                'maximum context length',
                '`inputs` tokens + `max_new_tokens`',
            )
            if any(phrase in combined for phrase in phrases):
                return True
    except ImportError:
        pass

    # Fallback: check string patterns for other providers
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    patterns = [
        'context_length_exceeded',
        'context length',
        'maximum context',
        'contextwindowexceeded',
    ]

    return any(pattern in error_str or pattern in error_type for pattern in patterns)


async def summarize(chat: 'Chat', original_instruction: str, session_id: str) -> tuple[str, list]:
    """Create a summary of the agent's work to pass to a new agent instance.

    This implements the three-step context summarization process from Harbor:

    **Step 1: Summary Generation (SUBAGENT 1)**
    - Message history: chat.messages (unwound)
    - Prompt: Generate comprehensive summary of all work completed so far
    - Response: summary_response (detailed narrative of progress)

    **Step 2: Question Asking (SUBAGENT 2)**
    - Message history: [] (fresh start, no conversation context)
    - Prompt: Given the original task, summary from Step 1, and current terminal screen,
              generate questions about information missing from the summary.
    - Response: model_questions (list of clarifying questions)

    **Step 3: Answer Providing (SUBAGENT 3)**
    - Message history: chat.messages + [summary_prompt, summary_response] (extended context)
    - Prompt: Given the questions from Step 2, answer the questions based on full conversation history.
    - Response: answers_content (detailed answers)

    **Final handoff:**
    - Chat history replaced with: [first_message, question_prompt, model_questions]
    - Result: Compressed context that preserves critical task-specific information

    Args:
        chat: Chat object containing conversation history (may be unwound)
        original_instruction: The original task instruction from user
        session_id: Terminal session ID for capturing current terminal state

    Returns:
        tuple: (handoff_prompt, subagent_trajectory_refs)
            - handoff_prompt: The prompt to continue with (includes answers + instructions)
            - subagent_trajectory_refs: Empty list (trajectory logging not needed in nexus)
    """
    if len(chat.messages) == 0:
        return original_instruction, []

    logger.debug('Starting context summarization (three-step process)...')

    # Get runtime for terminal capture
    runtime = await require_context().get_runtime()

    # ===== SUBAGENT 1: Summary Generation =====
    logger.debug('SUMMARIZATION Step 1: Summary Generation')
    summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(original_instruction=original_instruction)

    # Call LLM with current chat history as message_history
    summary_response = await lite_llm_call(summary_prompt, chat.messages)
    summary_content = summary_response.content
    logger.debug(f'Summary generated: {summary_content[:200]}...')

    # ===== SUBAGENT 2: Question Asking =====
    logger.debug('SUMMARIZATION Step 2: Question Asking')

    # Get current terminal screen
    current_screen = (await runtime.capture_terminal_pane(session_id=session_id, capture_entire=False)).content

    question_prompt = QUESTION_PROMPT_TEMPLATE.format(
        original_instruction=original_instruction,
        summary=summary_content,
        current_screen=current_screen,
    )

    # Fresh context - just ask questions based on summary (empty message history)
    model_questions = await lite_llm_call(question_prompt, [])
    model_questions = model_questions.content
    logger.debug(f'Questions generated: {model_questions[:200]}...')

    # ===== SUBAGENT 3: Answer Providing =====
    logger.debug('SUMMARIZATION Step 3: Answer Providing')
    answer_request_prompt = ANSWER_REQUEST_TEMPLATE.format(questions=model_questions)

    # The answer subagent sees: chat.messages + [summary_prompt, summary_response]
    # This matches Harbor implementation exactly (line 738-741 in terminus_2.py)
    answers_message_history = chat.messages + [
        {'role': 'user', 'content': summary_prompt},
        {'role': 'assistant', 'content': summary_content},
    ]

    answers_response = await lite_llm_call(answer_request_prompt, answers_message_history)
    answers_content = answers_response.content
    logger.debug(f'Answers generated: {answers_content[:200]}...')

    # ===== Update chat history =====
    # Replace chat history with compressed version:
    # [first_message, question_prompt, model_questions]
    # This matches Harbor implementation exactly (line 758-762 in terminus_2.py)
    chat._messages = [
        chat.messages[0],  # Keep first message (system/initial prompt)
        {'role': 'user', 'content': question_prompt},
        {'role': 'assistant', 'content': model_questions},
    ]

    # ===== Build handoff prompt =====
    # This matches Harbor implementation exactly (line 764-771 in terminus_2.py)
    handoff_prompt = HANDOFF_TEMPLATE.format(answers=answers_content)

    logger.info('Context summarization completed')

    # Return empty list for subagent_trajectory_refs since trajectory logging is not needed
    return handoff_prompt, []
