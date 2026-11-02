"""Search agent using ToolCallAgent infrastructure.

This agent demonstrates:
- Extending ToolCallAgent for domain-specific functionality
- Data-driven configuration over complex control flow
- Clean composition: configure, create, run, return
"""

import re

from json import JSONDecodeError
from typing import Any

from pydantic import Field

from nexus.core import ContextLengthExceededError, ToolExecutor, agent_registry, tool_registry
from nexus.core.registry import EntryNotFoundError
from nexus.core.types import ExtendedMessage
from nexus.extensions.agents.base.errors import is_non_retryable_provider_exception
from nexus.extensions.agents.base.logger import logger
from nexus.extensions.agents.base.schema import (
    MessageContent,
    TerminationReason,
    ToolCallAgentInput,
    ToolCallAgentOutput,
    ToolChoice,
)
from nexus.extensions.agents.base.toolcall import ToolCallAgent
from nexus.extensions.agents.base.utils import extract_text_from_content
from nexus.extensions.agents.search_agent.prompt import get_default_system_prompt
from nexus.trace import start_span
from nexus.trace.types import Status


TOOLSET_REQUIRED_TOOLS: dict[str, list[str]] = {
    'v1': ['google_serper_search', 'base/finish', 'visit'],
    'v2': ['google_serper_search', 'base/finish', 'visit', 'base/write_todos', 'execute_python'],
}

TOOLCALL_MARKUP_PATTERNS = [
    re.compile(r'<tool_call\b', re.IGNORECASE),
    re.compile(r'</tool_call>', re.IGNORECASE),
    re.compile(r'<function=[^>]+>', re.IGNORECASE),
    re.compile(r'</function>', re.IGNORECASE),
    re.compile(r'<parameter=[^>]+>', re.IGNORECASE),
    re.compile(r'</parameter>', re.IGNORECASE),
]

_FORCE_FINISH_WITH_TOOL = 'You have now reached the maximum context length you can handle. Based only on the information above, think carefully and call the `finish` tool now with your best answer.'
_FORCE_FINISH_WITHOUT_TOOL = 'You have now reached the maximum context length you can handle. Based only on the information above, think carefully and give your best answer now.'
OBSERVATION_FOLDED_PLACEHOLDER = 'Tool result is omitted to save tokens.'
AZURE_REFUSAL_RETRY_RESPONSE = "I'm sorry, but I cannot assist with that request."

COMPACTION_PROMPT = (
    'You have been working on the search task described at the start of this conversation. '
    'Please write a summary of all progress so far. This summary will REPLACE the conversation '
    'history above, so include everything needed to continue the work.\n\n'
    'Your summary should include the following sections:\n'
    '1. **Search Results & Findings**: Key information gathered from each search and visit. These cannot be recovered after compaction.\n'
    "2. **Dead Ends**: Searches or visits that were unhelpful and why (so you don't repeat them).\n"
    '3. **Current Status**: What has been worked on.\n'
    '4. **Next Step**: What to search or investigate next, you must keep the entire information of next step.\n\n'
)

COMPACTION_CONTINUE_PROMPT = (
    '[Summary of Prior Work]\n\n'
    '{summary}\n\n'
    'The above is a summary of your prior work. Continue working on the original task '
    'from where you left off. Use the available tools to make progress.'
)


def contains_toolcall_markup(text: str) -> bool:
    return any(pattern.search(text) for pattern in TOOLCALL_MARKUP_PATTERNS)


def get_llm_retry_reason(message: Any) -> str | None:
    content_text = extract_text_from_content(getattr(message, 'content', None) or '')
    reasoning_text = getattr(message, 'reasoning_content', None) or ''
    tool_calls = getattr(message, 'tool_calls', None) or []
    has_content = bool(content_text.strip())
    has_reasoning = bool(reasoning_text.strip())

    if content_text.strip() == AZURE_REFUSAL_RETRY_RESPONSE:
        return 'refusal response'
    if contains_toolcall_markup(content_text) or contains_toolcall_markup(reasoning_text):
        return 'raw tool-call markup in output'
    if not has_content and not tool_calls:
        if has_reasoning:
            return 'reasoning only, no content and no tool_calls'
        return 'empty response, no content and no tool_calls'
    return None


def _resolve_tools(toolset_version: str = 'v2', *, include_finish: bool = True) -> list:
    """Resolve and validate all required tools for the given toolset. Returns tool instances (not names)."""
    required_tool_names = TOOLSET_REQUIRED_TOOLS.get(toolset_version)
    if required_tool_names is None:
        raise ValueError(f'Unknown toolset_version: {toolset_version!r}. Available: {list(TOOLSET_REQUIRED_TOOLS)}')
    return _resolve_tools_by_names(required_tool_names, include_finish=include_finish)


def _resolve_tools_by_names(tool_names: list[str], *, include_finish: bool = True) -> list:
    """Resolve tool instances from an explicit list of registered tool names."""
    if not include_finish:
        tool_names = [n for n in tool_names if n != 'base/finish']

    tools: list = []
    missing: list[str] = []
    for name in tool_names:
        try:
            tools.append(tool_registry.get(name))
        except EntryNotFoundError:
            missing.append(name)
    if missing:
        raise ValueError(f'Tools not found in registry: {missing}')

    return tools


class SearchAgentInput(ToolCallAgentInput):
    """Search agent input - inherits ToolCallAgent configuration with sensible defaults."""

    system_prompt: str | None = None
    is_training: bool = True
    toolset_version: str = Field(default='v2', description='Toolset version to use. Available: v1, v2')
    max_iterations: int = 10
    finish_tool_names: list[str] = Field(default_factory=lambda: ['finish'])
    max_llm_retries: int = Field(default=0, description='Max LLM call retries on bad output (0 to disable)')
    keep_recent_k: int | None = Field(
        default=None,
        description='Keep-recent-k strategy: fold tool observations older than the most recent k rounds. '
        'Folded observations are replaced with a placeholder to save tokens. '
        'None disables this strategy. Recommended value: 5.',
    )
    max_context_tokens: int | None = Field(
        default=None,
        description='Maximum context token budget (e.g. 32000). When the total tokens (prompt + completion) '
        'reported by the API exceeds this threshold, all old assistant/tool messages are discarded '
        'and the agent continues in a fresh context with only the system prompt, original question, '
        'a context-reset notice, and the most recent keep_recent_k rounds. '
        'Token counts are taken directly from the API usage response (no local tokenizer needed). '
        'None disables this strategy.',
    )
    enable_compaction: bool = Field(
        default=False,
        description='When True and max_context_tokens is set, the agent will first ask the LLM to '
        'summarize the conversation before discarding old messages (compaction). '
        'When False, old messages are discarded without summarization (hard reset). ',
    )


class SearchAgentOutput(ToolCallAgentOutput):
    """Search agent output with trajectory and token statistics.

    Trajectory stats (num_steps, num_tool_calls) are derived from main_trajectory
    at output time.  Token stats are accumulated from API responses during the run.
    """

    # Trajectory statistics (derived from main_trajectory)
    num_steps: int = Field(default=0, description='Number of agent steps (assistant turns)')
    num_tool_calls: int = Field(default=0, description='Total tool calls across all steps')

    # Token usage (from LLM API responses).
    total_prompt_tokens: int = Field(default=0, description='Prompt tokens of the last LLM call (trajectory peak)')
    total_completion_tokens: int = Field(default=0, description='Cumulative completion tokens across all LLM calls')

    # Context management
    context_reset_count: int = Field(default=0, description='Number of context resets due to token budget')


class SearchAgent(ToolCallAgent):
    """Search agent - thin wrapper over ToolCallAgent."""

    name: str = 'search_agent'
    description: str = 'An agent that searches the web for information.'
    max_llm_retries: int = 0
    keep_recent_k: int | None = None
    max_context_tokens: int | None = Field(
        default=None,
        description='Token budget for context reset. When exceeded, old messages are discarded.',
    )
    enable_compaction: bool = Field(
        default=False,
        description='When True, summarize context via LLM before discarding old messages.',
    )

    # --- Internal state (not user-configurable) ---
    _context_token_count: int = 0
    _context_reset_count: int = 0
    _force_finishing: bool = False
    _total_prompt_tokens: int = 0
    _total_completion_tokens: int = 0
    _compaction_count: int = 0

    def _record_token_usage(self, usage: Any) -> None:
        """Record token usage for context control and trajectory stats."""
        prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
        completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
        self._context_token_count = prompt_tokens + completion_tokens
        self._total_prompt_tokens = prompt_tokens
        self._total_completion_tokens += completion_tokens
        logger.debug(
            f'📊 [{self.name}] Tokens: prompt={prompt_tokens}, completion={completion_tokens}, '
            f'total_completion={self._total_completion_tokens}'
        )

    def _should_reset_context(self) -> bool:
        """Check whether context token count exceeds the budget."""
        if self.max_context_tokens is None:
            return False
        if self._force_finishing:
            return False
        return self._context_token_count >= self.max_context_tokens

    def _reset_context(self) -> None:
        """Discard all assistant/tool messages, keeping only
        the prefix up to and including the first user message (the original question).

        If compaction was performed beforehand, a summary message is injected after the
        original question so the agent retains key context.

        Resets _context_token_count — it will be re-measured by the next API call.
        """
        first_user_idx = next((i for i, m in enumerate(self.messages) if m.get('role') == 'user'), None)
        if first_user_idx is None:
            return

        self.messages = self.messages[: first_user_idx + 1]
        self._context_reset_count += 1

        # Reset token count — it will be re-measured by the next API call
        self._context_token_count = 0

    async def _compact_context(self, **kwargs) -> None:
        """Summarize conversation via LLM, then reset context with summary injected.

        After reset the memory becomes:
          [user: original question] → [user: summary + continue prompt]
        The summary is provided as a user message because it is context being fed TO
        the model, not something the model actually said.  This avoids fabricating an
        assistant turn and makes the model treat it as actionable instructions.
        """
        logger.warning(f'📝 [{self.name}] Compacting context (compaction #{self._compaction_count + 1})...')

        # Build a one-shot summarization request using current conversation
        summary_messages = self.build_messages()
        summary_messages.append({'role': 'user', 'content': COMPACTION_PROMPT})
        summary_kwargs = {k: v for k, v in kwargs.items() if k not in {'tools', 'tool_choice'}}

        try:
            response = await self.client.create_completion(
                messages=summary_messages,
                tools=[],
                tool_choice=ToolChoice.NONE,
                **summary_kwargs,
            )
            summary_text = response.choices[0].message.content or ''
            logger.debug(f'📝 [{self.name}] Compaction summary ({len(summary_text)} chars): {summary_text[:200]}...')
        except Exception as e:
            logger.error(f'📝 [{self.name}] Compaction summarization failed: {e}, falling back to hard reset')
            self._reset_context()
            return

        # Hard reset first
        self._reset_context()
        self._compaction_count += 1

        self.add_message('user', COMPACTION_CONTINUE_PROMPT.format(summary=summary_text))

    async def ask_llm(self, **kwargs: Any) -> ExtendedMessage:
        """Query LLM, capture token usage from API response, and handle retries.

        Overrides ToolCallAgent.ask_llm to:
        1. Capture response.usage for token tracking (used by context reset logic)
        2. Perform context reset when token budget is exceeded (before the next call)
        3. Retain inference-mode retry logic for malformed tool-call output
        """
        # --- Context reset check (before sending to LLM) ---
        if self._should_reset_context():
            logger.warning(
                f'🔄 [{self.name}] Context tokens ({self._context_token_count}) >= '
                f'budget ({self.max_context_tokens}), '
                f'{"compacting" if self.enable_compaction else "resetting"} context'
            )
            if self.enable_compaction:
                await self._compact_context(**kwargs)
            else:
                self._reset_context()

        if self.is_training:
            return await self._ask_llm_with_usage(**kwargs)

        max_retries = max(self.max_llm_retries, 0)
        for attempt in range(max_retries + 1):
            message = await self._ask_llm_with_usage(**kwargs)
            if self.exit_status is not None:
                return message

            retry_reason = get_llm_retry_reason(message)
            if retry_reason is None or attempt >= max_retries:
                return message

            logger.warning(
                f'🎯 [{self.name}] Tool-call parse failure ({retry_reason}), retrying ({attempt + 1}/{max_retries})'
            )

        raise RuntimeError('LLM retry loop exhausted unexpectedly')

    async def _ask_llm_with_usage(self, **kwargs: Any) -> ExtendedMessage:
        """Call LLM and extract token usage from the response.

        Wraps the parent ask_llm call but also reads response.usage to update the token counter.
        """
        tools = self.executor.get_tool_schemas()
        messages = self.build_messages()

        try:
            response = await self.client.create_completion(
                messages=messages,
                tools=tools,
                tool_choice=self.tool_choice,
                **kwargs,
            )
            usage = getattr(response, 'usage', None)
            if usage is not None:
                self._record_token_usage(usage)

            return response.choices[0].message
        except ContextLengthExceededError as e:
            logger.error(f'🎯 [{self.name}] Context length exceeded: {e}')
            self.exit_status = TerminationReason.TRUNCATION
            return ExtendedMessage(content=None)
        except (JSONDecodeError, KeyError) as e:
            logger.error(f'🎯 [{self.name}] Tool format error: {e}')
            self.tool_error_count += 1
            self.exit_status = TerminationReason.FORMAT_ERROR
            return ExtendedMessage(content=None)
        except Exception as e:
            if is_non_retryable_provider_exception(e):
                logger.error(f'🎯 [{self.name}] Provider error: {e}')
                self.tool_error_count += 1
                self.exit_status = TerminationReason.PROVIDER_ERROR
                return ExtendedMessage(content=None)
            logger.error(f'🎯 [{self.name}] LLM client error: {e}')
            self.tool_error_count += 1
            self.exit_status = TerminationReason.UNHANDLED_ERROR
            return ExtendedMessage(content=None)

    def build_messages(self) -> list[dict[str, Any]]:
        """Build messages for LLM, applying Keep-recent-k strategy if configured.

        For the most recent k rounds, observations are kept as-is.
        For older rounds (i = 1, …, n-k), tool observations are folded:
            oᵢ ← OBSERVATION_FOLDED_PLACEHOLDER
        The initial user message (question) is always preserved in full.
        self.messages is NOT modified — this only affects the view sent to LLM.
        """
        if self.keep_recent_k is None:
            return super().build_messages()

        messages = self.messages

        # Split into: prefix (up to and including first user msg) + remainder
        first_user_index = next((i for i, m in enumerate(messages) if m.get('role') == 'user'), None)
        if first_user_index is None:
            return super().build_messages()

        prefix = messages[: first_user_index + 1]
        remainder = messages[first_user_index + 1 :]

        rounds: list[list[dict]] = []
        current_round: list[dict] = []
        for msg in remainder:
            if msg.get('role') == 'assistant':
                if current_round:
                    rounds.append(current_round)
                current_round = [msg]
            else:
                current_round.append(msg)
        if current_round:
            rounds.append(current_round)

        # Determine which rounds need folding: all except the most recent k
        fold_before = max(0, len(rounds) - self.keep_recent_k)

        result_messages: list[dict] = []

        # System prompt
        if self.system_prompt:
            result_messages.append({'role': 'system', 'content': self.system_prompt})

        # Initial user message (question) — always kept in full
        result_messages.extend(dict(m) for m in prefix)

        for i, round_msgs in enumerate(rounds):
            for msg in round_msgs:
                if i < fold_before and msg.get('role') == 'tool':
                    result_messages.append({**msg, 'content': OBSERVATION_FOLDED_PLACEHOLDER})
                else:
                    result_messages.append(dict(msg))

        return result_messages

    def get_output(self) -> 'SearchAgentOutput':
        trajectory = self.main_trajectory
        num_steps = sum(1 for m in trajectory if m.get('role') == 'assistant')
        num_tool_calls = sum(len(m.get('tool_calls') or []) for m in trajectory if m.get('role') == 'assistant')
        return SearchAgentOutput(
            main_trajectory=trajectory,
            final_answer=self.get_final_answer(),
            exit_status=self.exit_status or TerminationReason.SUCCESS,
            num_steps=num_steps,
            num_tool_calls=num_tool_calls,
            total_prompt_tokens=self._total_prompt_tokens,
            total_completion_tokens=self._total_completion_tokens,
            context_reset_count=self._context_reset_count,
        )

    async def run(self, request: MessageContent | None = None) -> MessageContent | None:
        """Run agent and enforce finish-tool based termination at max iterations."""
        if self.exit_status is not None:
            raise RuntimeError(f'Cannot run agent: already exited with {self.exit_status}')

        self.current_iteration = 0
        if request is not None:
            self.add_message('user', request)

        try:
            while self.current_iteration < self.max_iterations and self.exit_status is None:
                self.current_iteration += 1
                logger.debug(f'Executing step {self.current_iteration}/{self.max_iterations}')
                iter_span = start_span(
                    'agent.iteration',
                    attributes={
                        'type': 'agent_iteration',
                        'iteration': self.current_iteration,
                    },
                )
                try:
                    await self.step()
                    iter_span.update(status=Status(code='OK'))
                except Exception as e:
                    iter_span.update(status=Status(code='ERROR', message=str(e)))
                    raise
                finally:
                    iter_span.finish()

            if self.exit_status == TerminationReason.TRUNCATION:
                logger.warning(f'[{self.name}] Context length exceeded, attempting forced finish finalization')
                finalized = await self.force_finish_once()
                if not finalized:
                    self.exit_status = TerminationReason.TRUNCATION
                    logger.warning(f'[{self.name}] Forced finish failed after context length exceeded')

            if self.exit_status is None and self.current_iteration >= self.max_iterations:
                should_force_finish = not self.is_training
                if should_force_finish:
                    logger.warning(
                        f'[{self.name}] Reached max iterations ({self.max_iterations}), forcing finish tool finalization'
                    )
                    finalized = await self.force_finish_once()
                    if not finalized:
                        self.exit_status = TerminationReason.MAX_ITERATIONS
                        logger.warning(f'[{self.name}] Forced finish failed, terminated as max_iterations')
                else:
                    self.exit_status = TerminationReason.MAX_ITERATIONS

        except Exception as e:
            self.exit_status = TerminationReason.UNHANDLED_ERROR
            logger.exception(f'Agent failed with unexpected error: {e}')

        return self.get_final_answer()

    async def force_finish_once(self, max_retries: int = 10) -> bool:
        """Try one final constrained step to get the agent's best answer.

        With finish tools: constrains to only finish tool and prompts to call it.
        Without finish tools: hides all tools so the model outputs text directly.
        """
        prompt = _FORCE_FINISH_WITH_TOOL if self.finish_tool_names else _FORCE_FINISH_WITHOUT_TOOL
        self.add_message('user', prompt)
        force_finish_message = self.messages[-1]
        previous_exit_status = self.exit_status
        original_allowed = self.executor._allowed_tool_names
        original_retries = self.max_llm_retries
        original_tool_choice = self.tool_choice
        self.exit_status = None
        self.executor._allowed_tool_names = ['base/finish'] if self.finish_tool_names else []
        self.tool_choice = ToolChoice.AUTO if self.finish_tool_names else ToolChoice.NONE
        self.max_llm_retries = 0
        self._force_finishing = True
        msg_checkpoint = len(self.messages)
        try:
            for attempt in range(max_retries + 1):
                await self.step()
                if self.exit_status == TerminationReason.SUCCESS:
                    break
                if self.exit_status is not None:
                    break
                del self.messages[msg_checkpoint:]
                if attempt < max_retries:
                    logger.warning(
                        f'[{self.name}] Force finish attempt {attempt + 1}/{max_retries + 1} did not succeed, retrying'
                    )
        except Exception as e:
            logger.error(f'[{self.name}] Unexpected error during force finish: {e}')
        finally:
            self._force_finishing = False
            self.executor._allowed_tool_names = original_allowed
            self.max_llm_retries = original_retries
            self.tool_choice = original_tool_choice
            for index in range(len(self.messages) - 1, -1, -1):
                if self.messages[index] is force_finish_message:
                    del self.messages[index]
                    break

        if self.exit_status == TerminationReason.SUCCESS:
            return True

        if previous_exit_status is not None:
            self.exit_status = previous_exit_status
        logger.warning(f'[{self.name}] Forced finish step did not reach SUCCESS')
        return False


@agent_registry.register('search_agent')
async def run_search_agent(agent_input: SearchAgentInput) -> SearchAgentOutput:
    """Search agent entry point."""
    include_finish = bool(agent_input.finish_tool_names)
    if not agent_input.system_prompt:
        agent_input.system_prompt = get_default_system_prompt(
            agent_input.is_training, agent_input.toolset_version, include_finish=include_finish
        )
    if agent_input.tools is not None:
        resolved_tools = _resolve_tools_by_names(agent_input.tools, include_finish=include_finish)
    else:
        resolved_tools = _resolve_tools(agent_input.toolset_version, include_finish=include_finish)
    executor = ToolExecutor(resolved_tools)
    agent = SearchAgent.create(agent_input, executor=executor)
    available_tool_names = agent.executor._all_canonical_tool_names()
    tools = [schema['function']['name'] for schema in agent.executor.get_tool_schemas()]

    logger.warning(
        f'[{agent.name}] Agent config:\n'
        f'  toolset_version={agent_input.toolset_version}\n'
        f'  is_training={agent.is_training}\n'
        f'  max_iterations={agent.max_iterations}\n'
        f'  max_llm_retries={agent.max_llm_retries}\n'
        f'  keep_recent_k={agent.keep_recent_k}\n'
        f'  max_context_tokens={agent.max_context_tokens}\n'
        f'  enable_compaction={agent.enable_compaction}\n'
        f'  finish_tool_names={agent.finish_tool_names}\n'
        f'  available_tool_names={available_tool_names}\n'
        f'  tools={tools}'
    )
    logger.warning(f'[{agent.name}] User prompt:\n  {agent_input.user_prompt}')
    logger.warning(f'[{agent.name}] System prompt:\n  {agent.system_prompt}')

    await agent.run(agent_input.user_prompt)
    return agent.get_output()
