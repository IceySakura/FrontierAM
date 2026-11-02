"""BrowserGym agent implementation for web automation.

This agent implements the DemoAgent architecture from BrowserGym,
adapted to work with Nexus's tool system and execution context.
"""

import asyncio
import base64
import json
import logging
import os


# 已注释掉任务过滤模块
# from nexus.extensions.agents.browser_gym.task_aware_filter import (
#     TaskAwareFilter,
#     FilterStats,
# )
# from nexus.extensions.agents.browser_gym.site_aware_filter import (
#     detect_site_from_url,
#     detect_site_from_task,
#     get_site_filter_config,
#     create_site_aware_filter,
# )
# Try to import tiktoken for accurate token counting
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    ToolExecutor,
    WithMainTrajectory,
    WithUserPrompt,
    agent_registry,
    require_context,
    tool_registry,
)
from nexus.extensions.agents.browser_gym.delta_observation import (
    ObservationTracker,
)
from nexus.extensions.agents.browser_gym.dom_compression import (
    CompressionStats,
    compress_observation,
    get_recommended_compression_level,
)
from nexus.extensions.agents.browser_gym.dom_compression import (
    count_tokens as compress_count_tokens,
)
from nexus.extensions.agents.browser_gym.token_counting import (
    count_tokens_with_fallback,
)
from nexus.extensions.agents.browser_gym.tool_config import ToolSubset, get_tools_for_subset


logger = logging.getLogger(__name__)

# Default max context length in tokens (180K tokens for the model, leave buffer for output)
DEFAULT_MAX_CONTEXT_TOKENS = 180 * 1024  # 170K tokens


@dataclass
class TruncationInfo:
    """Information about a single truncation event."""

    iteration: int  # Which iteration triggered the truncation
    iterations_removed: int  # How many iterations were removed
    original_tokens: int  # Token count before truncation
    truncated_tokens: int  # Token count after truncation
    original_messages: int  # Number of messages before truncation
    truncated_messages: int  # Number of messages after truncation


@dataclass
class TruncationStats:
    """Aggregate statistics about all truncations during a run."""

    total_truncations: int = 0  # How many times truncation was triggered
    total_iterations_removed: int = 0  # Total iterations removed across all truncations
    truncation_events: list[TruncationInfo] = field(default_factory=list)  # Detailed info per truncation

    def add_event(self, event: TruncationInfo) -> None:
        """Add a truncation event."""
        self.total_truncations += 1
        self.total_iterations_removed += event.iterations_removed
        self.truncation_events.append(event)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_truncations': self.total_truncations,
            'total_iterations_removed': self.total_iterations_removed,
            'truncation_events': [
                {
                    'iteration': e.iteration,
                    'iterations_removed': e.iterations_removed,
                    'original_tokens': e.original_tokens,
                    'truncated_tokens': e.truncated_tokens,
                    'original_messages': e.original_messages,
                    'truncated_messages': e.truncated_messages,
                }
                for e in self.truncation_events
            ],
        }


def count_tokens(text: str) -> int:
    """Count tokens in a text string using tiktoken.

    Args:
        text: Text to count tokens for

    Returns:
        Number of tokens
    """
    return count_tokens_with_fallback(text, chars_per_token=2)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Estimate the token count of a message.

    Args:
        message: A message dict with 'role', 'content', 'tool_calls', etc.

    Returns:
        Estimated token count
    """
    tokens = 0

    # Role tokens (roughly 1-2 tokens)
    tokens += 2

    # Content
    if 'content' in message and message['content']:
        tokens += count_tokens(str(message['content']))

    # Tool calls (assistant messages)
    if 'tool_calls' in message and message['tool_calls']:
        for tc in message['tool_calls']:
            if isinstance(tc, dict):
                tokens += count_tokens(tc.get('function', {}).get('name', ''))
                tokens += count_tokens(tc.get('function', {}).get('arguments', ''))
            else:
                # Object with attributes
                tokens += count_tokens(getattr(tc.function, 'name', ''))
                tokens += count_tokens(getattr(tc.function, 'arguments', ''))
            # Tool call overhead
            tokens += 10

    # Message overhead (separators, etc.)
    tokens += 4

    return tokens


def estimate_total_tokens(history: list[dict[str, Any]]) -> int:
    """Estimate total token count of conversation history.

    Args:
        history: List of message dicts

    Returns:
        Total estimated token count
    """
    return sum(estimate_message_tokens(msg) for msg in history)


def find_iteration_boundaries(history: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Find the start and end indices of each iteration in history.

    An iteration typically consists of:
    - user message (DOM observation)
    - assistant message (with tool_calls)
    - tool response message(s)

    Args:
        history: Conversation history

    Returns:
        List of (start_idx, end_idx) tuples for each iteration.
        The first two messages (system + task goal) are never included.
    """
    boundaries = []
    i = 2  # Skip system message and task goal

    while i < len(history):
        start_idx = i

        # Find user message (observation)
        if i < len(history) and history[i].get('role') == 'user':
            i += 1
        else:
            # Not a standard iteration, skip this message
            i += 1
            continue

        # Find assistant message
        if i < len(history) and history[i].get('role') == 'assistant':
            i += 1
        else:
            # Incomplete iteration
            boundaries.append((start_idx, i))
            continue

        # Find all tool responses for this iteration
        while i < len(history) and history[i].get('role') == 'tool':
            i += 1

        end_idx = i
        boundaries.append((start_idx, end_idx))

    return boundaries


def truncate_history_if_needed(
    history: list[dict[str, Any]],
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    current_iteration: int = 0,
    min_keep_iterations: int = 3,
) -> tuple[list[dict[str, Any]], TruncationInfo | None]:
    """Truncate history if it exceeds the maximum token limit.

    This function removes complete iterations from the beginning of the history
    (after system message and task goal) until the total token count is under the limit.

    Args:
        history: Conversation history to potentially truncate
        max_tokens: Maximum allowed token count (default: 170K tokens)
        current_iteration: Current iteration number (for logging)
        min_keep_iterations: Minimum number of recent iterations to keep (default: 3)

    Returns:
        Tuple of (truncated_history, truncation_info).
        truncation_info is None if no truncation was needed.
    """
    original_tokens = estimate_total_tokens(history)
    original_messages = len(history)

    if original_tokens <= max_tokens:
        return history, None

    logger.warning(
        f'✂️ [browser_gym] History too long ({original_tokens:,} tokens > {max_tokens:,} tokens), truncating...'
    )

    # Find iteration boundaries
    boundaries = find_iteration_boundaries(history)

    if not boundaries:
        logger.warning('✂️ [browser_gym] No iterations found to truncate')
        return history, None

    # Calculate how many iterations we can remove (keep at least min_keep_iterations)
    max_removable = max(0, len(boundaries) - min_keep_iterations)

    if max_removable == 0:
        logger.warning(
            f'✂️ [browser_gym] Only {len(boundaries)} iterations, keeping all (min_keep={min_keep_iterations})'
        )
        return history, None

    # Keep removing iterations from the beginning until under limit
    iterations_removed = 0
    truncated_history = history.copy()

    while boundaries and iterations_removed < max_removable and estimate_total_tokens(truncated_history) > max_tokens:
        # Remove the first (oldest) iteration
        start_idx, end_idx = boundaries[0]

        # Always start from index 2 (after system + task goal)
        # Remove the iteration
        messages_to_remove = boundaries[0][1] - boundaries[0][0]
        del truncated_history[2 : 2 + messages_to_remove]

        iterations_removed += 1
        boundaries.pop(0)

        # Recalculate boundaries for remaining history
        boundaries = find_iteration_boundaries(truncated_history)

        current_tokens = estimate_total_tokens(truncated_history)
        logger.debug(
            f'✂️ [browser_gym] Removed iteration {iterations_removed}, '
            f'new length: {current_tokens:,} tokens, remaining iterations: {len(boundaries)}'
        )

    # Add a note about truncation
    if iterations_removed > 0:
        # Insert a note after task goal
        truncation_note = {
            'role': 'user',
            'content': f'[Note: {iterations_removed} earlier iteration(s) were truncated due to context length limits. '
            f'The conversation continues from a later point.]',
        }
        truncated_history.insert(2, truncation_note)

        truncated_tokens = estimate_total_tokens(truncated_history)
        truncated_messages = len(truncated_history)

        logger.warning(
            f'✂️ [browser_gym] Truncated {iterations_removed} iteration(s), '
            f'final length: {truncated_tokens:,} tokens, kept {len(boundaries)} iterations'
        )

        # Create truncation info
        truncation_info = TruncationInfo(
            iteration=current_iteration,
            iterations_removed=iterations_removed,
            original_tokens=original_tokens,
            truncated_tokens=truncated_tokens,
            original_messages=original_messages,
            truncated_messages=truncated_messages,
        )

        return truncated_history, truncation_info

    return truncated_history, None


# Browser observation mode
ObservationMode = Literal['axtree', 'html', 'both']


class ActionDeduplicator:
    """Detects and warns about repeated actions to prevent infinite loops.

    This class tracks action history and generates warnings when:
    1. The same action is repeated consecutively multiple times
    2. A single action dominates the recent action window
    3. The agent appears to be stuck in a loop pattern
    """

    def __init__(
        self,
        max_consecutive_repeat: int = 3,
        window_size: int = 10,
        loop_detection_threshold: float = 0.6,
    ):
        """
        Args:
            max_consecutive_repeat: Max times same action can repeat consecutively
            window_size: Number of recent actions to analyze
            loop_detection_threshold: Ratio threshold for detecting action loops
        """
        self.action_history: list[str] = []
        self.max_consecutive_repeat = max_consecutive_repeat
        self.window_size = window_size
        self.loop_detection_threshold = loop_detection_threshold
        self.warning_count: int = 0

    def _normalize_action(self, action: str) -> str:
        """Normalize action for comparison (ignoring minor variations)."""
        # Extract tool name and key parameters
        import json

        try:
            # Parse action like 'tool_name({"param": "value"})'
            if '(' in action and action.endswith(')'):
                tool_name = action.split('(')[0]
                args_str = action[len(tool_name) + 1 : -1]
                args = json.loads(args_str)

                # Create normalized representation
                # For click/fill actions, the bid is the key identifier
                if 'bid' in args:
                    return f'{tool_name}(bid={args["bid"]})'
                if 'text' in args:
                    # For fill, include truncated text
                    return f'{tool_name}(text={args["text"][:20]}...)'
                return f'{tool_name}({list(args.keys())})'
        except (json.JSONDecodeError, IndexError):
            pass
        return action

    def check_and_warn(self, action: str) -> str | None:
        """Check for repeated actions and return warning if detected.

        Args:
            action: The action string to check

        Returns:
            Warning message if problematic pattern detected, None otherwise
        """
        normalized = self._normalize_action(action)
        self.action_history.append(normalized)

        warnings = []

        # Check 1: Consecutive repetition
        recent = self.action_history[-self.max_consecutive_repeat :]
        if len(recent) >= self.max_consecutive_repeat and len(set(recent)) == 1:
            warnings.append(
                f"⚠️ REPEATED ACTION: '{normalized}' has been executed {self.max_consecutive_repeat} times consecutively. "
                f'This approach may not be working - consider trying a different element or strategy.'
            )

        # Check 2: Single action dominance in window
        window = self.action_history[-self.window_size :]
        if len(window) >= self.window_size // 2:
            action_counts = Counter(window)
            most_common_action, most_common_count = action_counts.most_common(1)[0]
            ratio = most_common_count / len(window)

            if ratio >= self.loop_detection_threshold and most_common_count >= 4:
                warnings.append(
                    f"⚠️ ACTION LOOP DETECTED: '{most_common_action}' accounts for {ratio:.0%} of recent actions. "
                    f'You appear to be stuck. Try a completely different approach.'
                )

        # Check 3: Oscillation pattern (A-B-A-B)
        if len(self.action_history) >= 6:
            recent_6 = self.action_history[-6:]
            if (
                recent_6[0] == recent_6[2] == recent_6[4]
                and recent_6[1] == recent_6[3] == recent_6[5]
                and recent_6[0] != recent_6[1]
            ):
                warnings.append(
                    f"⚠️ OSCILLATION DETECTED: Alternating between '{recent_6[0]}' and '{recent_6[1]}'. "
                    f'This pattern indicates the actions are not achieving the goal. Try a new strategy.'
                )

        if warnings:
            self.warning_count += 1
            return '\n\n' + '\n'.join(warnings)

        return None

    def get_summary(self) -> dict:
        """Get summary of action deduplication analysis."""
        action_counts = Counter(self.action_history)
        return {
            'total_actions': len(self.action_history),
            'unique_actions': len(action_counts),
            'warnings_generated': self.warning_count,
            'most_common_actions': action_counts.most_common(5),
        }


class BrowserGymAgentInput(AgentRunInput, WithUserPrompt):
    """BrowserGym agent input configuration.

    Attributes:
        user_prompt: The task goal/instruction
        tool_subset: Which tool subset to use (e.g., 'webarena', 'full')
        custom_tools: Additional custom tool identifiers to include
        observation_mode: What page representation to use ('axtree', 'html', 'both')
        max_iterations: Maximum number of agent steps
        include_screenshot: Whether to include screenshots in observations
        max_actions_history: Maximum number of past actions to include in context
        max_context_tokens: Maximum context length in tokens before truncation
        min_keep_iterations: Minimum number of recent iterations to keep during truncation
        enable_dom_compression: Whether to compress DOM trees to reduce token usage
        dom_compression_threshold: Token threshold above which to compress DOM trees
        enable_delta_observation: Whether to use incremental observations
        enable_task_filter: Whether to filter DOM based on task relevance
        task_filter_threshold: Relevance threshold for task-aware filtering
        browser: Browser declaration used to open a session when one is not already available
    """

    browser: dict[str, Any] | None = Field(
        default=None,
        description=(
            'Browser declaration (start_url, headless, viewport_width, viewport_height, '
            'nav_timeout_ms). Used only when ctx.tool_state has no browser_session_id.'
        ),
    )
    tool_subset: ToolSubset = Field(default='webarena', description='Tool subset to use (webarena, full, or custom)')
    custom_tools: list[str] = Field(default_factory=list, description='Additional tool identifiers to include')
    observation_mode: ObservationMode = Field(
        default='axtree', description="Page observation mode: 'axtree', 'html', or 'both'"
    )
    max_iterations: int = Field(default=30, description='Maximum agent steps (WebArena default: 30)')
    include_screenshot: bool = Field(default=False, description='Include screenshots in observations')
    max_actions_history: int = Field(default=20, description='Maximum actions to keep in history')
    max_context_tokens: int = Field(
        default=DEFAULT_MAX_CONTEXT_TOKENS,
        description='Maximum context length in tokens before truncation (default: 170K tokens)',
    )
    min_keep_iterations: int = Field(
        default=3, description='Minimum number of recent iterations to keep during truncation'
    )
    enable_dom_compression: bool = Field(default=True, description='Enable DOM tree compression to reduce token usage')
    dom_compression_threshold: int = Field(
        default=15000, description='Token threshold above which to compress DOM trees (default: 15K tokens)'
    )
    enable_delta_observation: bool = Field(
        default=True, description='Enable incremental observations to reduce redundant content'
    )
    enable_task_filter: bool = Field(
        default=True,
        description='Enable site-aware task filtering. Filter strength automatically adjusts based on website type (disabled for Reddit, minimal for Shopping Admin, standard for others).',
    )
    task_filter_threshold: float = Field(
        default=0.05,
        description='Base relevance threshold for task-aware filtering. Actual threshold is adjusted per site.',
    )


class BrowserGymAgentOutput(AgentRunOutput, WithMainTrajectory):
    """BrowserGym agent output with trajectory.

    Attributes:
        main_trajectory: Full conversation history (untruncated)
        truncation_stats: Statistics about context truncation during the run
        exit_status: Why the BrowserGym loop stopped
    """

    exit_status: Literal['submitted', 'max_iterations', 'no_tool_calls_incomplete', 'format_error'] = Field(
        description='Why the BrowserGym loop stopped'
    )
    truncation_stats: dict[str, Any] | None = Field(
        default=None, description='Statistics about context truncation (if any occurred)'
    )


def build_system_prompt(webarena_urls: dict[str, str] | None = None) -> str:
    """Build system prompt with BrowserGym instructions.

    Args:
        webarena_urls: Optional mapping of site names to their sandbox URLs.
                       When provided, a site directory section is appended to help
                       the model understand which URL corresponds to which service.
    """
    prompt = """\
# Instructions

You are an expert web automation agent. Your goal is to complete tasks efficiently by interacting with web pages.

## Core Principles

1. **Plan before acting**: Think about what steps are needed to accomplish the goal
2. **Verify after acting**: Check if your action had the expected effect before proceeding
3. **Adapt when stuck**: If an approach isn't working after 2-3 tries, switch to a different strategy
4. **Be precise with answers**: Extract exact values, not descriptions or explanations

## Action Guidelines

- Each action is executed as a tool call with specific parameters
- Use browser IDs (bid) from the accessibility tree to identify elements
- When the task is complete, call submit_result with your final answer
- If the task is infeasible after multiple attempts, call submit_result with "N/A"

## CRITICAL: Answer Format

When calling submit_result, provide ONLY the direct answer without any explanation or context:

✅ CORRECT Examples:
- For "What is the price of X?" → submit "$19.99"
- For "What is the name of Y?" → submit "Product Name"
- For "How many items?" → submit "42"
- For yes/no questions → submit "yes" or "no"
- For impossible tasks → submit "N/A"

❌ WRONG Examples (DO NOT do this):
- "The price of X is $19.99" → Should be just "$19.99"
- "Based on my search, I found that..." → Just provide the answer
- "The answer is 42" → Should be just "42"

## Interaction Tips

1. **If an element is not visible**: Try scrolling the page first (scroll down/up)
2. **If a click doesn't work**: The element might be outside the viewport - scroll to make it visible
3. **If a form doesn't submit**: Check for validation errors or required fields
4. **If search returns no results**: Try alternative search terms or check spelling
5. **For dropdowns/menus**: You may need to click to open them before selecting an option

## Common Pitfalls to Avoid

⚠️ **Repetition Loop**: If you find yourself repeating the same action 3+ times, STOP and try a different approach
⚠️ **Missing Verification**: Always verify that your action succeeded before moving to the next step
⚠️ **Wrong Element**: Double-check the bid before clicking - similar elements may have different bids
⚠️ **Premature Submission**: Make sure you have found the actual answer before submitting

## Available Information

You will receive:
1. Task goal - what you need to accomplish
2. Current page state - accessibility tree showing the page structure with element IDs
3. Open tabs - list of currently open browser tabs
4. Action history - your previous actions and their results
5. Available tools - the actions you can take

Use all available information to make informed decisions about the next action.
"""

    # Append site directory if webarena_urls are available
    if webarena_urls:
        site_descriptions = {
            'shopping': 'E-commerce website (online store with products, orders, reviews)',
            'shopping_admin': 'E-commerce admin panel (Magento backend for managing the store)',
            'reddit': 'Social forum website (Reddit-like community with posts and comments)',
            'gitlab': 'Code hosting platform (GitLab instance with repositories and issues)',
            'wikipedia': 'Online encyclopedia (Wikipedia with articles and references)',
            'map': 'Map service (OpenStreetMap-based tile server)',
        }

        # Filter out placeholder URLs (e.g. __GITLAB__) that haven't been resolved
        resolved_urls = {
            name: url for name, url in webarena_urls.items() if not (url.startswith('__') and url.endswith('__'))
        }

        if resolved_urls:
            lines = ['\n## Sandbox Site Directory\n']
            lines.append('The following URLs correspond to different web services in the sandbox environment:\n')
            for site_name, url in resolved_urls.items():
                desc = site_descriptions.get(site_name, '')
                if desc:
                    lines.append(f'- **{site_name}**: `{url}` — {desc}')
                else:
                    lines.append(f'- **{site_name}**: `{url}`')
            lines.append('')
            prompt += '\n'.join(lines)

    return prompt


async def get_observation(
    session_id: str,
    include_screenshot: bool = False,
    enable_compression: bool = False,
    compression_threshold: int = 15000,
) -> tuple[str, CompressionStats | None]:
    """Get current page observation using runtime.

    Args:
        session_id: Browser session ID
        include_screenshot: Whether to include screenshot in the observation text
        enable_compression: Whether to enable DOM tree compression
        compression_threshold: Token threshold above which to compress DOM trees

    Returns:
        Tuple of (formatted_observation_text, compression_stats)
    """
    runtime = await require_context().get_runtime()

    # Always capture screenshot for saving to local file
    obs_response = await runtime.browser_get_observation(session_id=session_id, include_screenshot=True)

    # Save screenshot to local file if available
    if obs_response.screenshot:
        try:
            # Create screenshots directory if not exists
            screenshots_dir = os.path.join(os.getcwd(), 'workspace/screenshots')
            os.makedirs(screenshots_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f'screenshot_{session_id}_{timestamp}.png'
            filepath = os.path.join(screenshots_dir, filename)

            # Decode base64 and save
            screenshot_data = base64.b64decode(obs_response.screenshot)
            with open(filepath, 'wb') as f:
                f.write(screenshot_data)

            logger.debug(f'📸 [browser_gym] Screenshot saved: {filepath}')
        except Exception as e:
            logger.warning(f'Failed to save screenshot: {e}')

    # Format observation
    parts = [
        '# Current Page Observation',
        '',
        f'URL: {obs_response.url}',
        f'Title: {obs_response.title}',
        '',
        '## Accessibility Tree',
        '',
        obs_response.axtree,
    ]

    if include_screenshot and obs_response.screenshot:
        parts.append('')
        parts.append('[Screenshot included]')

    observation_text = '\n'.join(parts)

    # Apply DOM compression if enabled and observation is large enough
    compression_stats = None
    if enable_compression:
        obs_tokens = compress_count_tokens(observation_text)
        if obs_tokens > compression_threshold:
            # Determine compression level based on how much over threshold
            compression_level = get_recommended_compression_level(obs_tokens, compression_threshold)

            compressed_obs, compression_stats = compress_observation(
                observation_text,
                max_tokens=compression_threshold,
                compression_level=compression_level,
            )

            if compression_stats:
                logger.debug(
                    f'🗜️ [browser_gym] DOM compressed: '
                    f'{compression_stats.original_tokens:,} → {compression_stats.compressed_tokens:,} tokens '
                    f'({compression_stats.compression_ratio:.1%} reduction)'
                )
                observation_text = compressed_obs

    return observation_text, compression_stats


async def _open_declared_browser_session(ctx: Any, browser: dict[str, Any] | None) -> str | None:
    """Create and navigate the browser declared by a Harbor task."""
    if not browser:
        return None

    start_url = str(browser.get('start_url') or '').strip()
    if not start_url:
        raise ValueError('browser declaration requires start_url')

    runtime = await ctx.get_runtime(browser.get('runtime_name') or 'main')
    nav_timeout_ms = int(browser.get('nav_timeout_ms', 10000))
    created = await runtime.create_browser_session(
        headless=bool(browser.get('headless', True)),
        viewport_width=int(browser.get('viewport_width', 1280)),
        viewport_height=int(browser.get('viewport_height', 720)),
        timeout=nav_timeout_ms,
    )
    session_id = str(created.session_id or '').strip()
    if not session_id:
        raise RuntimeError('Runtime returned an empty browser session ID')

    try:
        navigation = await runtime.browser_goto(session_id, start_url, timeout=nav_timeout_ms)
        if not navigation.success:
            raise RuntimeError(f'Initial browser navigation to {start_url!r} failed; current URL: {navigation.url!r}')
    except BaseException:
        try:
            await asyncio.shield(runtime.destroy_browser_session(session_id))
        except BaseException:
            logger.exception('Failed to destroy browser session after initial navigation failure')
        raise

    # Publish only after navigation succeeds so the Bridge never hands an
    # uninitialized live resource to the Verifier.
    ctx.tool_state['browser_session_id'] = session_id
    logger.debug(f'🌐 [browser_gym] Opened browser session {session_id} at {start_url}')
    return session_id


@agent_registry.register('browser_gym_agent')
async def run_browser_gym_agent(agent_input: BrowserGymAgentInput) -> BrowserGymAgentOutput:
    """Run BrowserGym agent with the given input.

    This is a simple implementation following demo_agent pattern:
    - Direct loop over iterations
    - Get observation before each step
    - Call LLM with tools
    - Execute tool calls
    - Track action history
    - Stop when submit_result is called or max iterations reached

    Args:
        agent_input: Agent configuration including tool subset, task goal, etc.

    Returns:
        Agent output with trajectory and final answer
    """
    ctx = require_context()
    client = ctx.openai_client

    # Reuse a session created by a native task, or create one from a Harbor declaration.
    session_id = ctx.tool_state.get('browser_session_id')
    if not session_id:
        session_id = await _open_declared_browser_session(ctx, agent_input.browser)
    if not session_id:
        raise ValueError('No browser session available. Make sure browser session is initialized in tool_state.')

    # Get tools for the specified subset
    tool_ids = get_tools_for_subset(agent_input.tool_subset)

    # Add any custom tools
    if agent_input.custom_tools:
        tool_ids.extend(agent_input.custom_tools)

    # Create executor
    tools = [tool_registry.get(tool_id) for tool_id in tool_ids]
    executor = ToolExecutor(tools)

    # Get tool schemas for LLM
    tool_schemas = executor.get_tool_schemas()

    # Initialize conversation history (include sandbox site URLs if available)
    webarena_urls = ctx.tool_state.get('webarena_urls')
    system_msg = {
        'role': 'system',
        'content': build_system_prompt(webarena_urls=webarena_urls),
    }
    history = [
        system_msg,
        {'role': 'user', 'content': f'# Task Goal\n\n{agent_input.user_prompt}'},
    ]

    # Track action history
    action_history: list[str] = []

    # Track truncation statistics
    truncation_stats = TruncationStats()

    # Track compression statistics
    total_compression_saved_tokens = 0

    # Initialize observation optimization modules
    observation_tracker: ObservationTracker | None = None
    # 已注释掉任务过滤模块
    task_filter = None  # 禁用任务过滤
    action_deduplicator = ActionDeduplicator(
        max_consecutive_repeat=3,
        window_size=10,
        loop_detection_threshold=0.6,
    )

    if agent_input.enable_delta_observation:
        observation_tracker = ObservationTracker(
            enable_delta=True,
            min_similarity_for_delta=0.3,
        )

    # 已注释掉网站特定过滤模块初始化
    # if agent_input.enable_task_filter:
    #     current_site = detect_site_from_task(agent_input.user_prompt)
    #     site_config = get_site_filter_config(current_site)
    #
    #     if site_config['enable_filter']:
    #         task_filter = TaskAwareFilter(
    #             task_goal=agent_input.user_prompt,
    #             enable_filter=True,
    #             relevance_threshold=site_config['relevance_threshold'],
    #             min_elements=site_config['min_elements'],
    #             adaptive_threshold=True,
    #         )
    #         task_filter.update_task_context(site_config['additional_keywords'])
    #         logger.info(
    #             f'🎯 [browser_gym] Site-aware filter enabled for "{current_site}": '
    #             f'threshold={site_config["relevance_threshold"]}, min_elements={site_config["min_elements"]}'
    #         )
    #     else:
    #         logger.info(f'🎯 [browser_gym] Task filter disabled for site "{current_site}" (filtering hurts performance)')

    # Main agent loop
    exit_status: Literal['submitted', 'max_iterations', 'no_tool_calls_incomplete', 'format_error'] = 'max_iterations'
    for iteration in range(agent_input.max_iterations):
        logger.debug(f'🌐 [browser_gym] Step {iteration + 1}/{agent_input.max_iterations}')

        # Get current page observation (with optional DOM compression)
        try:
            obs_text, compression_stats = await get_observation(
                session_id,
                agent_input.include_screenshot,
                enable_compression=agent_input.enable_dom_compression,
                compression_threshold=agent_input.dom_compression_threshold,
            )

            # Track compression savings
            if compression_stats:
                total_compression_saved_tokens += (
                    compression_stats.original_tokens - compression_stats.compressed_tokens
                )

            # Apply task-aware filtering (before delta to filter irrelevant elements first)
            if task_filter:
                obs_text, filter_stats = task_filter.filter(obs_text)
                if filter_stats:
                    total_compression_saved_tokens += filter_stats.original_tokens - filter_stats.filtered_tokens

            # Apply delta observation (compute incremental changes)
            if observation_tracker:
                obs_text, delta_stats = observation_tracker.update(obs_text)
                if delta_stats:
                    total_compression_saved_tokens += delta_stats.original_tokens - delta_stats.delta_tokens

            # Add action history if available
            if action_history:
                history_text = '\n'.join(
                    f'Step {i + 1}: {action}'
                    for i, action in enumerate(action_history[-agent_input.max_actions_history :])
                )
                obs_text += f'\n\n## Action History\n\n{history_text}'

            # Add observation to history
            history.append({'role': 'user', 'content': obs_text})

        except Exception as e:
            logger.error(f'Failed to get observation: {e}')
            history.append({'role': 'user', 'content': f'# Observation Error\n\nFailed to get page state: {e}'})

        # Truncate history if needed before calling LLM
        messages_to_send, truncation_info = truncate_history_if_needed(
            history,
            max_tokens=agent_input.max_context_tokens,
            current_iteration=iteration + 1,
            min_keep_iterations=agent_input.min_keep_iterations,
        )

        # Record truncation event if it occurred
        if truncation_info:
            truncation_stats.add_event(truncation_info)

        # Call LLM with potentially truncated history
        response = await client.create_completion(messages=messages_to_send, tools=tool_schemas or None)
        message = response.choices[0].message
        history.append(message.model_dump(exclude_none=True))

        if message.content:
            logger.debug(f'💬 [browser_gym] Content: {message.content}')

        # Check if there are tool calls
        if not message.tool_calls:
            logger.info('🏁 [browser_gym] No more tool calls before submit_result')
            exit_status = 'no_tool_calls_incomplete'
            break

        # Execute tool calls
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.error('BrowserGym tool arguments are invalid JSON for %s: %s', tool_name, exc)
                exit_status = 'format_error'
                break

            logger.debug(f'🛠️ [browser_gym] Tool: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})')

            # Record action
            action_str = f'{tool_name}({json.dumps(tool_args)})'
            action_history.append(action_str)

            # Check for repeated actions and generate warning
            action_warning = action_deduplicator.check_and_warn(action_str)

            # Execute tool
            try:
                tool_response = await executor.call(tool_name, tool_args)
                logger.debug(f'🎯 [browser_gym] Result: {tool_response.output}')

                # Prepare response content (tool_response.output is a list)
                response_content = tool_response.output

                # Append action warning to tool response if detected
                if action_warning:
                    logger.warning('🔄 [browser_gym] Action repetition detected')
                    # Append warning as a new text part to the list
                    response_content = list(response_content)  # Make a copy
                    response_content.append({'type': 'text', 'text': action_warning})

                # Add tool response to history
                history.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': response_content,
                })

                # Check if this is submit_result (completion signal)
                if tool_name == 'submit_result':
                    logger.info('🏁 [browser_gym] Task completed via submit_result')
                    history.append({'role': 'assistant', 'content': tool_args['message']})
                    exit_status = 'submitted'
                    break

            except Exception as e:
                error_msg = f'Error: {tool_name} failed: {e}'
                logger.error(f'🎯 [browser_gym] {error_msg}')
                history.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': error_msg,
                })

        # Stop if completed
        if exit_status in {'submitted', 'format_error'}:
            break

    else:
        logger.warning(f'⚠️ [browser_gym] Reached max_iterations ({agent_input.max_iterations})')

    # Log truncation summary if any occurred
    if truncation_stats.total_truncations > 0:
        logger.info(
            f'📊 [browser_gym] Truncation summary: {truncation_stats.total_truncations} truncation(s), '
            f'{truncation_stats.total_iterations_removed} total iteration(s) removed'
        )

    # Log observation optimization summary
    if total_compression_saved_tokens > 0:
        logger.info(
            f'🗜️ [browser_gym] Observation optimization summary: saved {total_compression_saved_tokens:,} tokens total'
        )

    # 已注释掉任务过滤日志
    # if task_filter:
    #     filter_summary = task_filter.get_summary()
    #     logger.info(
    #         f'🎯 [browser_gym] Site-aware filter summary (site={current_site}): '
    #         f'{filter_summary["filter_count"]} filter(s), '
    #         f'threshold={filter_summary["final_threshold"]:.2f}, '
    #         f'keywords={filter_summary["task_keywords"][:5]}'
    #     )

    # Log delta observation summary if enabled
    if observation_tracker:
        delta_summary = observation_tracker.get_summary()
        if delta_summary['delta_count'] > 0:
            logger.info(
                f'📊 [browser_gym] Delta observation summary: '
                f'{delta_summary["delta_count"]} delta(s), '
                f'saved {delta_summary["total_tokens_saved"]:,} tokens'
            )

    # Log action deduplication summary
    dedup_summary = action_deduplicator.get_summary()
    if dedup_summary['warnings_generated'] > 0:
        logger.warning(
            f'🔄 [browser_gym] Action deduplication summary: '
            f'{dedup_summary["warnings_generated"]} warning(s) generated, '
            f'{dedup_summary["unique_actions"]}/{dedup_summary["total_actions"]} unique actions'
        )

    return BrowserGymAgentOutput(
        main_trajectory=history,
        exit_status=exit_status,
        truncation_stats=truncation_stats.to_dict() if truncation_stats.total_truncations > 0 else None,
    )
