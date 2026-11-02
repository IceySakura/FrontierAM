"""Browser game task - open a game in the sandbox browser and let the agent play it.

Workflow:
1. Acquire runtime
2. Start an HTTP server in the sandbox to serve a specified directory
3. Create browser session and navigate to the given URL
4. Run main_agent with visual tools + post_iteration_hook for auto-screenshot & score
5. Cleanup browser session
"""

import logging
import math

from typing import Any

from pydantic import BaseModel, Field

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.agents.main.agent import PostIterationHookResult


logger = logging.getLogger(__name__)

# Namespace for score_expr eval — exposes math functions + builtins for convenience.
# Usage: "score / 2048", "min(1, log2(max(score, 1)) / log2(2048))", "sqrt(score)"
_SCORE_EXPR_NS: dict[str, Any] = {
    'min': min,
    'max': max,
    'abs': abs,
    'round': round,
    'pow': pow,
    'math': math,
    'log': math.log,
    'log2': math.log2,
    'log10': math.log10,
    'sqrt': math.sqrt,
    'ceil': math.ceil,
    'floor': math.floor,
    'exp': math.exp,
}


def _eval_score_expr(expr: str, raw_score: float) -> float:
    """Evaluate a score expression.

    Single expression (e.g. ``score / 2048``) is eval'd directly.
    Multi-statement code must assign to ``result``::

        x = log2(max(score, 1)) / log2(2048)
        result = min(1.0, x)
    """
    ns = _SCORE_EXPR_NS | {'score': raw_score}
    try:
        return float(eval(expr, {'__builtins__': {}}, ns))  # noqa: S307
    except SyntaxError:
        exec(expr, {'__builtins__': {}}, ns)  # noqa: S102
        if 'result' not in ns:
            raise ValueError('Multi-statement score_expr must assign to `result`') from None
        return float(ns['result'])


# ============================================================================
# Helper Models
# ============================================================================


class StartHttpServerResult(BaseModel):
    """Result from starting the HTTP server in the sandbox."""

    success: bool
    port: int
    error_message: str | None = None


class ScreenshotResult(BaseModel):
    """Result from taking a screenshot in the sandbox."""

    screenshot: str | None = None


class ScoreResult(BaseModel):
    """Result from extracting score in the sandbox."""

    text: str | None = None


# ============================================================================
# Data Models
# ============================================================================


class BrowserGameTaskInput(TaskRunInput, WithAgentRunner):
    """Browser game task input."""

    user_prompt: str = Field(..., description='Instructions for the agent')
    url: str = Field(..., description='URL to open in the browser')
    serve_dir: str = Field(default='/workspace', description='Directory to serve via HTTP in the sandbox')
    serve_port: int = Field(default=8080, description='Port for the HTTP server')
    timeout: int = 600
    score_selector: str | None = Field(default=None, description='CSS selector to extract score text')
    score_expr: str | None = Field(
        default=None,
        description=(
            'Python expression/code to normalize the raw score. '
            'Variable `score` holds the raw value. '
            'Available: min, max, abs, round, pow, log, log2, log10, sqrt, ceil, floor, exp, math. '
            'Single expression: "score / 2048" or "min(1, log2(max(score, 1)) / log2(2048))". '
            'Multi-statement (assign to `result`): "x = log2(max(score, 1)) / log2(2048)\\nresult = min(1.0, x)".'
        ),
    )
    show_score_to_agent: bool = Field(default=False, description='Whether to show extracted score to the agent')


class BrowserGameTaskOutput(TaskRunOutput, WithScore):
    """Browser game task output.

    Attributes:
        score: Final normalized score (from WithScore). 0.0 if no scores collected.
        scores: Normalized score per iteration (= raw if no score_expr).
        raw_scores: Raw scores extracted from the page per iteration.
        url: URL that was opened.
        error_message: Error message if task failed.
    """

    scores: list[float] = []
    raw_scores: list[float] = []
    url: str | None = None
    error_message: str | None = None


# ============================================================================
# Task Implementation
# ============================================================================


@task_registry.register('browser_game')
async def run_browser_game_task(task_input: BrowserGameTaskInput) -> BrowserGameTaskOutput:
    """Run a browser game task."""
    ctx = require_context()
    assert ctx.runtime_provider is not None, 'browser_game requires runtime provider'

    runtime = await ctx.get_runtime()
    browser_session_id = None

    try:
        # Step 1: Start HTTP server
        logger.debug('Starting HTTP server on port %d serving %s...', task_input.serve_port, task_input.serve_dir)
        server_result = await runtime.invoke(
            _start_http_server,
            port=task_input.serve_port,
            serve_dir=task_input.serve_dir,
        )
        if not server_result.success:
            raise RuntimeError(f'Failed to start HTTP server: {server_result.error_message}')

        # Step 2: Create browser session
        create_result = await runtime.create_browser_session(
            headless=True,
            viewport_width=1280,
            viewport_height=720,
            timeout=10000,
        )
        browser_session_id = create_result.session_id
        logger.debug('Browser session created: %s', browser_session_id)

        # Step 3: Set up shared state for hook
        ctx.tool_state['browser_session_id'] = browser_session_id
        ctx.tool_state['score_selector'] = task_input.score_selector
        ctx.tool_state['score_expr'] = task_input.score_expr
        ctx.tool_state['show_score_to_agent'] = task_input.show_score_to_agent
        ctx.tool_state['raw_scores'] = []
        ctx.tool_state['scores'] = []

        # Step 4: Navigate to URL
        logger.debug('Navigating to: %s', task_input.url)
        await runtime.invoke(
            _navigate_to_url,
            session_id=browser_session_id,
            url=task_input.url,
        )

        # Step 5: Run agent
        logger.debug('Running agent...')
        await task_input.run_agent(
            user_prompt=task_input.user_prompt,
            timeout=task_input.timeout,
        )

        scores = ctx.tool_state.get('scores', [])
        raw_scores = ctx.tool_state.get('raw_scores', [])
        return BrowserGameTaskOutput(
            url=task_input.url,
            score=scores[-1] if scores else 0.0,
            scores=scores,
            raw_scores=raw_scores,
        )

    except Exception as e:
        logger.error('Browser game task failed: %s', e, exc_info=True)
        scores = ctx.tool_state.get('scores', [])
        raw_scores = ctx.tool_state.get('raw_scores', [])
        return BrowserGameTaskOutput(
            error_message=str(e),
            score=scores[-1] if scores else 0.0,
            scores=scores,
            raw_scores=raw_scores,
        )

    finally:
        if browser_session_id:
            try:
                await runtime.destroy_browser_session(session_id=browser_session_id)
            except Exception as e:
                logger.error('Failed to destroy browser session: %s', e)


# ============================================================================
# Post-iteration hook
# ============================================================================


async def game_observation(ctx: Any, _hook_context: Any) -> PostIterationHookResult | None:
    """Post-iteration hook: auto-screenshot + optional score extraction.

    Reads configuration from ctx.tool_state:
        - browser_session_id: active browser session
        - score_selector: CSS selector for score element (optional)
        - score_expr: Python expression to normalize raw score (optional)
        - show_score_to_agent: whether to include score in agent messages
        - raw_scores: list to append raw extracted scores to
        - scores: list to append normalized scores to
    """
    session_id = ctx.tool_state.get('browser_session_id')
    if not session_id:
        return None

    runtime = await ctx.get_runtime()

    extra_messages: list[dict[str, Any]] = []

    # Auto-screenshot
    try:
        result = await runtime.invoke(_take_screenshot, session_id=session_id)
        if result.screenshot:
            extra_messages.append({
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{result.screenshot}'}},
                    {'type': 'text', 'text': 'Current game screen.'},
                ],
            })
    except Exception:
        logger.warning('Hook: screenshot failed', exc_info=True)

    # Score extraction + normalization
    score_selector = ctx.tool_state.get('score_selector')
    if score_selector:
        try:
            result = await runtime.invoke(_extract_score, session_id=session_id, selector=score_selector)
            if result.text:
                raw_score = float(result.text.strip())
                ctx.tool_state['raw_scores'].append(raw_score)

                # Normalize via score_expr or use raw
                score_expr = ctx.tool_state.get('score_expr')
                normalized = _eval_score_expr(score_expr, raw_score) if score_expr else raw_score
                ctx.tool_state['scores'].append(normalized)

                logger.info('Hook: raw_score=%.2f score=%.4f', raw_score, normalized)
                if ctx.tool_state.get('show_score_to_agent', False):
                    extra_messages.append({
                        'role': 'user',
                        'content': f'Current score: {raw_score}',
                    })
        except (ValueError, TypeError):
            logger.debug('Hook: could not parse score text')
        except Exception:
            logger.warning('Hook: score extraction failed', exc_info=True)

    return PostIterationHookResult(extra_messages=extra_messages) if extra_messages else None


# ============================================================================
# Sandbox-side helper functions (executed via runtime.invoke)
# ============================================================================


async def _start_http_server(port: int, serve_dir: str) -> StartHttpServerResult:
    """Start a simple HTTP server in the sandbox."""
    import asyncio
    import sys

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            '-m',
            'http.server',
            str(port),
            cwd=serve_dir,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.sleep(1)
        if process.returncode is not None:
            return StartHttpServerResult(
                success=False, port=port, error_message='HTTP server process exited immediately'
            )
        return StartHttpServerResult(success=True, port=port)
    except Exception as e:
        return StartHttpServerResult(success=False, port=port, error_message=str(e))


async def _navigate_to_url(session_id: str, url: str) -> None:
    """Navigate the browser to the given URL."""
    from nexus.runtime.executors.browser import browser_goto
    from nexus.runtime.executors.models.browser import GotoRequest

    await browser_goto(GotoRequest(session_id=session_id, url=url))


async def _take_screenshot(session_id: str) -> ScreenshotResult:
    """Take a screenshot of the current browser page. Returns base64 PNG."""
    from nexus.runtime.executors.browser import browser_get_observation
    from nexus.runtime.executors.models.browser import GetObservationRequest

    result = await browser_get_observation(GetObservationRequest(session_id=session_id, include_screenshot=True))
    return ScreenshotResult(screenshot=result.screenshot)


async def _extract_score(session_id: str, selector: str) -> ScoreResult:
    """Extract text content from a CSS selector on the page."""
    from nexus.runtime.executors.browser import browser_eval_js
    from nexus.runtime.executors.models.browser import EvalJsRequest

    js = f"document.querySelector('{selector}')?.textContent || ''"
    result = await browser_eval_js(EvalJsRequest(session_id=session_id, expression=js))
    return ScoreResult(text=str(result.value) if result.value else None)
