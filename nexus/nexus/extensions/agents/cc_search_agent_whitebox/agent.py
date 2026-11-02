"""White-box Claude Code style search agent.

This agent reproduces the *behavior* of the packaged Claude Code
``search`` subagent (see ``cc_search_agent_whitebox/search.md``) —
same prompt, same logical tool surface (``Bash``, ``WebSearch``,
``WebFetch``) — but drives the LLM and tool-call loop entirely
inside Nexus. Concretely it delegates to ``main_agent.run_main_agent``
which gives us a full white-box view of every inference, every tool
call, and every observation through the existing span / processor
plumbing.

Key differences from the black-box CC search agent:

- No ``claude`` CLI is launched; we run our own LLM via the configured
  ``OpenAICompatibleClient`` from the ExecutionContext.
- The CLI's first-party ``Bash`` / ``WebSearch`` / ``WebFetch`` tools are
  re-implemented as Nexus tools in :mod:`tools` so they can be called
  through the standard tool registry.

Example::

    from nexus.extensions.agents.cc_search_agent_whitebox import (
        CCSearchAgentWhiteboxInput,
        run_cc_search_agent_whitebox,
    )

    out = await run_cc_search_agent_whitebox(
        CCSearchAgentWhiteboxInput(user_prompt='Who won the Turing award in 2024?'),
    )
    print(out.final_answer)
"""

import contextvars
import logging
import time

from typing import Any

from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam
from pydantic import Field

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    ExtendedChatCompletion,
    OpenAICompatibleClient,
    WithMainTrajectory,
    WithUserPrompt,
    agent_registry,
)
from nexus.core.context import require_context

# Importing this module registers the ``search/self_summarize`` compaction
# strategy.  It must run before ``_install_compaction_manager`` resolves the
# strategy by name, which is guaranteed because that helper lives in this module.
from nexus.extensions.agents.cc_search_agent_whitebox import compaction as _compaction  # noqa: F401
from nexus.extensions.agents.cc_search_agent_whitebox.prompt import get_default_system_prompt
from nexus.extensions.agents.cc_search_agent_whitebox.tools import (
    BASH_TOOL_REGISTRY_NAME,
    EXECUTE_CODE_TOOL_REGISTRY_NAME,
    WEB_FETCH_TOOL_REGISTRY_NAME,
    WEB_SEARCH_TOOL_REGISTRY_NAME,
)
from nexus.extensions.agents.main.agent import (
    MainAgentInput,
    MainAgentToolOverride,
    run_main_agent,
)


logger = logging.getLogger(__name__)

# Per-run iteration counter, scoped via contextvars so concurrent runs don't interfere.
_iter_counter: contextvars.ContextVar[int] = contextvars.ContextVar('_cc_search_iter', default=0)

_ITERATION_HOOK_PATH = 'nexus.extensions.agents.cc_search_agent_whitebox.agent::iteration_logging_hook'


class _LatestReminderClient(OpenAICompatibleClient):
    """Wrap the configured client and append a latest_reminder per request."""

    def __init__(
        self,
        inner: OpenAICompatibleClient,
        latest_reminder: str | list[ChatCompletionContentPartParam],
    ) -> None:
        super().__init__(inner.config)
        self._inner = inner
        self._latest_reminder = latest_reminder

    async def create_completion(
        self,
        messages: list[ChatCompletionMessageParam | dict],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        return await self._inner.create_completion(
            messages=[*messages, {'role': 'latest_reminder', 'content': self._latest_reminder}],
            model=model,
            **kwargs,
        )

    async def _create_completion(
        self,
        messages: list[ChatCompletionMessageParam | dict],
        model: str | Any = None,
        **kwargs: Any,
    ) -> ExtendedChatCompletion:
        return await self.create_completion(messages, model=model, **kwargs)


async def iteration_logging_hook(ctx: Any, hook_context: Any) -> None:
    """Post-iteration hook: logs tool responses aligned with ToolCallAgent style (see toolcall.py:264)."""
    n = _iter_counter.get() + 1
    _iter_counter.set(n)
    msgs = hook_context.tool_messages or []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        content = m.get('content', '')
        logger.debug(f'🎯 Tool response: {str(content)}')


# ---------------------------------------------------------------------------
# Defaults: tools mirror the Claude Code ``search`` subagent (Bash, WebSearch,
# WebFetch) plus ``ExecuteCode`` for numerical work. ``WebSearch`` /
# ``WebFetch`` are LOCAL Nexus tools registered in
# ``cc_search_agent_whitebox.tools``; ``Bash`` and ``ExecuteCode`` are
# RUNTIME tools (executed inside the runtime sandbox).
# ---------------------------------------------------------------------------

DEFAULT_REQUIRED_TOOLS: list[str] = [
    WEB_SEARCH_TOOL_REGISTRY_NAME,
    WEB_FETCH_TOOL_REGISTRY_NAME,
    BASH_TOOL_REGISTRY_NAME,
    EXECUTE_CODE_TOOL_REGISTRY_NAME,
]


# ---------------------------------------------------------------------------
# Input / Output schema (main_agent style)
# ---------------------------------------------------------------------------


class CCSearchAgentWhiteboxInput(AgentRunInput, WithUserPrompt):
    """White-box CC search agent input — main_agent shaped."""

    system_prompt: str | list[ChatCompletionContentPartParam] | None = Field(
        default=None,
        description="System prompt; defaults to the packaged CC search prompt with today's date filled in.",
    )
    max_iterations: int = Field(
        default=30,
        description='Maximum LLM <-> tool iterations before the agent is forced to stop.',
    )

    # Latest reminder：每轮注入、贴在消息末尾、不写入 history、不累积。
    # 由本 agent 临时包装 openai_client 实现，main_agent 对 reminder 概念无感知。
    latest_reminder: str | list[ChatCompletionContentPartParam] | None = Field(
        default=None,
        description='Per-turn reminder injected as a latest_reminder message at the end of the '
        'messages sent to the LLM (not written into history).',
    )

    # Tool surface
    tools: list[str] | None = Field(
        default=None,
        description='Registered Nexus tool names to expose. Defaults to the CC search tool set '
        '(WebSearch, WebFetch, Bash, ExecuteCode).',
    )
    mcp_servers: list[dict[str, Any]] | None = None
    parallel_tool_calls: bool = False
    invalid_tool_args_retries: int = 0
    tool_call_error_retries: int = 0
    tool_output_truncation: int | None = None
    tool_overrides: list[MainAgentToolOverride] | None = None
    tool_schema_overrides: list[dict[str, Any]] | str | None = None
    multimodal_tool_output: bool = False  # CC search tools only return text.
    tool_error_continue: bool = True

    # Hooks / timeouts (forwarded to main_agent verbatim)
    post_iteration_hook: str | None = None
    observation_time_timeout: float | None = None
    inference_time_timeout: float | None = None

    # ------------------------------------------------------------------
    # Context management (mirrors search_agent semantics)
    # ------------------------------------------------------------------
    # When ``max_context_tokens`` is set we install a ``CompactionManager`` on
    # the current ``ExecutionContext``.  ``main_agent`` reads this manager every
    # iteration (see ``main/agent.py`` line ~371) and compacts ``history`` when
    # the configured condition fires.  Leaving ``max_context_tokens`` unset
    # disables compaction entirely (the agent runs unmodified).
    max_context_tokens: int | None = Field(
        default=None,
        description='Token budget that triggers context compaction. When the running prompt token '
        'count meets or exceeds this value, the configured strategy is applied. None disables '
        'context management; the agent will fail with ContextLengthExceededError if the model '
        'context is exhausted.',
    )
    compaction_strategy: str = Field(
        default='discard_all',
        description='Registered compaction strategy name. "discard_all" keeps only the '
        'system prompt and the original user task (search_agent-style hard reset, no LLM call). '
        '"self_summarize" asks the main openai_client to summarize the middle of the conversation '
        'and replaces it with a single assistant summary message.',
    )
    summary_prompt: str | None = Field(
        default=None,
        description='Optional override for the summary prompt template used by self_summarize. '
        'Must contain a "{text}" placeholder which is replaced with the flattened middle '
        'messages. When None, the strategy default is used.',
    )


class CCSearchAgentWhiteboxOutput(AgentRunOutput, WithMainTrajectory):
    """White-box CC search agent output — main_agent shaped."""

    exit_status: str
    iterations: int = 0
    final_answer: str | None = None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_main_input(agent_input: CCSearchAgentWhiteboxInput) -> MainAgentInput:
    """Build a :class:`MainAgentInput` from the CC-search-shaped input."""
    system_prompt: str | list[ChatCompletionContentPartParam] = (
        agent_input.system_prompt if agent_input.system_prompt else get_default_system_prompt()
    )
    tools = agent_input.tools or list(DEFAULT_REQUIRED_TOOLS)
    iteration_hook = agent_input.post_iteration_hook or _ITERATION_HOOK_PATH
    return MainAgentInput(
        user_prompt=agent_input.user_prompt,
        system_prompt=system_prompt,
        max_iterations=agent_input.max_iterations,
        tools=tools,
        mcp_servers=agent_input.mcp_servers,
        parallel_tool_calls=agent_input.parallel_tool_calls,
        invalid_tool_args_retries=agent_input.invalid_tool_args_retries,
        tool_call_error_retries=agent_input.tool_call_error_retries,
        tool_output_truncation=agent_input.tool_output_truncation,
        tool_overrides=agent_input.tool_overrides,
        tool_schema_overrides=agent_input.tool_schema_overrides,
        multimodal_tool_output=agent_input.multimodal_tool_output,
        tool_error_continue=agent_input.tool_error_continue,
        post_iteration_hook=iteration_hook,
        observation_time_timeout=agent_input.observation_time_timeout,
        inference_time_timeout=agent_input.inference_time_timeout,
    )


def _install_compaction_manager(agent_input: CCSearchAgentWhiteboxInput) -> bool:
    """Build a CompactionManager from the agent input and attach it to the
    current ExecutionContext.

    Returns True if a manager was installed, False otherwise (either the input
    disabled compaction, or one was already installed by an outer layer — in
    which case we leave it untouched).
    """
    if agent_input.max_context_tokens is None:
        return False

    ctx = require_context()
    if ctx.compaction_manager is not None:
        # Outer layer (e.g. hytuner) already configured one — defer to it.
        logger.debug('[cc_search_whitebox] ExecutionContext already has compaction_manager; skipping in-agent install.')
        return False

    # Lazy imports keep the agent import-light when compaction is disabled.
    from nexus.extensions.compaction import CompactionManager

    try:
        import tiktoken

        # cl100k_base is the GPT-4 / GPT-3.5-turbo encoding. For non-OpenAI
        # models (e.g. Claude) it under-estimates by 10–25%, which is fine for
        # the compaction trigger as long as max_context_tokens leaves a buffer.
        enc = tiktoken.get_encoding('cl100k_base')
    except Exception as e:  # pragma: no cover
        logger.warning(
            '[cc_search_whitebox] tiktoken unavailable (%s); falling back to char/4 heuristic',
            e,
        )
        enc = None

    async def _token_count_fn(messages: list[dict], tools: Any) -> int:
        total = 0
        for m in messages:
            content = m.get('content')
            if isinstance(content, str):
                total += len(enc.encode(content)) if enc else len(content) // 4
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get('text'), str):
                        text = part['text']
                        total += len(enc.encode(text)) if enc else len(text) // 4
            for tc in m.get('tool_calls') or []:
                args = tc.get('function', {}).get('arguments', '')
                if isinstance(args, str):
                    total += len(enc.encode(args)) if enc else len(args) // 4
        return total

    # Per-strategy passthrough kwargs. ``self_summary_prompt`` is consumed by
    # both ``self_summarize`` and ``search/self_summarize``; other strategies
    # ignore unknown kwargs.
    strategy_kwargs: dict[str, Any] = {}
    if agent_input.summary_prompt is not None:
        strategy_kwargs['self_summary_prompt'] = agent_input.summary_prompt

    ctx.compaction_manager = CompactionManager(
        strategy_name=agent_input.compaction_strategy,
        condition_name='remaining_tokens_exhausted',
        target_tokens=0,
        token_count_fn=_token_count_fn,
        condition_kwargs={'max_generation_length': agent_input.max_context_tokens},
        **strategy_kwargs,
    )
    logger.debug(
        '[cc_search_whitebox] CompactionManager installed: strategy=%s threshold=%d summary_prompt_overridden=%s',
        agent_input.compaction_strategy,
        agent_input.max_context_tokens,
        agent_input.summary_prompt is not None,
    )
    return True


@agent_registry.register('cc_search_agent_whitebox')
async def run_cc_search_agent_whitebox(agent_input: CCSearchAgentWhiteboxInput) -> CCSearchAgentWhiteboxOutput:
    """White-box CC search agent entry point.

    Builds a :class:`MainAgentInput` with CC search defaults (prompt, tool set)
    and delegates to :func:`run_main_agent`. The trajectory, exit status and
    final answer are forwarded back to the caller without modification.
    """
    _iter_counter.set(0)
    t0 = time.monotonic()

    _install_compaction_manager(agent_input)

    main_input = _build_main_input(agent_input)
    logger.debug(
        '[cc_search_whitebox] start | max_iter=%d tools=%s prompt=%.80s',
        main_input.max_iterations,
        [t.rsplit('/', 1)[-1] for t in main_input.tools],
        agent_input.user_prompt,
    )

    ctx = require_context()
    original_client = ctx.openai_clients.get('main')
    if agent_input.latest_reminder and original_client is not None:
        ctx.openai_clients['main'] = _LatestReminderClient(original_client, agent_input.latest_reminder)
    try:
        main_output = await run_main_agent(main_input)
    finally:
        if agent_input.latest_reminder and original_client is not None:
            ctx.openai_clients['main'] = original_client

    elapsed = time.monotonic() - t0
    logger.info(
        '[cc_search_whitebox] done | exit=%s iter=%d elapsed=%.1fs answer_len=%d',
        main_output.exit_status,
        main_output.iterations,
        elapsed,
        len(main_output.final_answer) if main_output.final_answer else 0,
    )
    return CCSearchAgentWhiteboxOutput(
        main_trajectory=main_output.main_trajectory,
        exit_status=main_output.exit_status,
        iterations=main_output.iterations,
        final_answer=main_output.final_answer,
    )


__all__ = [
    'CCSearchAgentWhiteboxInput',
    'CCSearchAgentWhiteboxOutput',
    'DEFAULT_REQUIRED_TOOLS',
    'iteration_logging_hook',
    'run_cc_search_agent_whitebox',
]
