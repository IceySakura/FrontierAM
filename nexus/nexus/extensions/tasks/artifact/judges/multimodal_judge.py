"""Multimodal judge: screenshot the running service, call a vision LLM to score it.

The judge:
1. Takes a screenshot of the target URL inside the eval runtime using Playwright.
2. Sends the screenshot (base64) + a scoring prompt to a multimodal OpenAI-compatible client.
3. Extracts a score from the LLM's response via a configurable regex.

Requirements:
  - The eval runtime must have Playwright installed (it's included in nexus-runtime).
  - The target service must be reachable from inside the runtime container.
  - The openai_client must support vision (image_url content parts).
"""

from __future__ import annotations

import logging
import re

from typing import TYPE_CHECKING, Any, Literal

from nexus.extensions.tasks.artifact.judges.base import BaseJudge, JudgeConfig, JudgeResult


if TYPE_CHECKING:
    from nexus.core.openai_client import OpenAICompatibleClient


logger = logging.getLogger(__name__)


class MultimodalJudgeConfig(JudgeConfig):
    """Config for the multimodal judge."""

    name: Literal['multimodal'] = 'multimodal'

    prompt: str
    """Scoring prompt sent to the vision LLM along with the screenshot.

    Example:
        'Evaluate the visual aesthetics of this web page on a scale from 0 to 10.
         Reply with ONLY a JSON object: {"score": <number>, "reason": "<one sentence>"}.'
    """

    target_url: str = 'http://localhost:8080'
    """URL to screenshot inside the runtime container."""

    screenshot_timeout_ms: int = 30_000
    """Playwright navigation timeout in milliseconds."""

    score_pattern: str = r'"score"\s*:\s*(\d+(?:\.\d+)?)'
    """Regex to extract the numeric score from the LLM response.

    The first capture group must be the score value.
    Default pattern matches: {"score": 7.5, ...}
    """

    score_max: float = 10.0
    """Maximum possible score (used to normalise to 0–1)."""

    pass_threshold: float = 0.6
    """Normalised score (0–1) above which passed=True."""

    wait_before_screenshot_ms: int = 2000
    """Extra wait time after page load before taking screenshot (ms)."""

    full_page: bool = True
    """Whether to capture a full-page screenshot."""

    openai_client_name: str = 'main'
    """Named client key in ExecutionContext.openai_clients to use for vision LLM calls."""


def _take_screenshot(
    url: str,
    timeout_ms: int = 30_000,
    wait_after_ms: int = 2000,
    full_page: bool = True,
) -> str:
    """Take a screenshot inside the runtime and return base64-encoded PNG.

    Executed via runtime.invoke (runs inside the sandbox).
    """
    import asyncio
    import base64

    from playwright.async_api import async_playwright

    async def _capture() -> bytes:
        async with async_playwright() as p:
            browser = await p.chromium.launch(args=['--no-sandbox', '--disable-dev-shm-usage'])
            try:
                page = await browser.new_page()
                await page.goto(url, timeout=timeout_ms, wait_until='networkidle')
                if wait_after_ms > 0:
                    await asyncio.sleep(wait_after_ms / 1000)
                return await page.screenshot(full_page=full_page)
            finally:
                await browser.close()

    png_bytes = asyncio.run(_capture())
    return base64.b64encode(png_bytes).decode('ascii')


class MultimodalJudge(BaseJudge):
    """Screenshot the artifact and ask a vision LLM to score it."""

    async def run(
        self,
        config: JudgeConfig,
        image: str,
        patch_text: str,
        working_dir: str,
        openai_client: OpenAICompatibleClient | None = None,
        **kwargs: Any,
    ) -> JudgeResult:
        assert isinstance(config, MultimodalJudgeConfig)
        import uuid

        from nexus.core.context import require_context

        # Resolve openai client: prefer named client from context, fall back to passed-in
        resolved_client = openai_client
        try:
            ctx = require_context()
            resolved_client = ctx.get_openai_client(config.openai_client_name)
        except Exception:
            pass

        if resolved_client is None:
            return JudgeResult(
                judge_name='multimodal',
                score=0.0,
                error='MultimodalJudge requires an openai_client but none was provided',
            )

        ctx = require_context()
        runtime_name = f'eval_multimodal_{uuid.uuid4().hex[:8]}'
        if image and ctx.runtime_provider:
            ctx.runtime_provider.configure(runtime_name, image=image)
        runtime = await ctx.get_runtime(runtime_name)

        try:
            if patch_text:
                from nexus.extensions.tasks.artifact.judges.pytest_judge import _apply_patch

                await _apply_patch(runtime, patch_text, working_dir)

            # 1. Take screenshot inside the runtime
            try:
                screenshot_b64: str = await runtime.invoke(
                    _take_screenshot,
                    config.target_url,
                    config.screenshot_timeout_ms,
                    config.wait_before_screenshot_ms,
                    config.full_page,
                )
            except Exception as e:
                return JudgeResult(
                    judge_name='multimodal',
                    score=0.0,
                    error=f'Screenshot failed: {type(e).__name__}: {e}',
                )
        finally:
            try:
                if ctx.runtime_provider:
                    await ctx.runtime_provider.release(name=runtime_name)
            except Exception as rel_err:
                logger.warning('MultimodalJudge: failed to release runtime %s: %s', runtime_name, rel_err)

        # 2. Call vision LLM on the host side
        try:
            response = await resolved_client.create_completion(
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': config.prompt},
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': f'data:image/png;base64,{screenshot_b64}',
                                    'detail': 'high',
                                },
                            },
                        ],
                    }
                ],
            )
            llm_output: str = response.choices[0].message.content or ''
        except Exception as e:
            return JudgeResult(
                judge_name='multimodal',
                score=0.0,
                error=f'Vision LLM call failed: {type(e).__name__}: {e}',
            )

        # 3. Extract score from LLM response
        score_normalized, evidence = self._extract_score(llm_output, config.score_pattern, config.score_max)
        passed = score_normalized >= config.pass_threshold

        return JudgeResult(
            judge_name='multimodal',
            score=round(score_normalized, 4),
            passed=passed,
            evidence=evidence,
            raw_output=llm_output,
        )

    @staticmethod
    def _extract_score(text: str, pattern: str, score_max: float) -> tuple[float, str]:
        """Extract and normalise score from LLM text.

        Returns:
            (normalised_score 0-1, evidence_string)
        """
        try:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                raw_score = float(m.group(1))
                normalised = min(raw_score / score_max, 1.0)
                return normalised, f'raw_score={raw_score}/{score_max}'
        except (ValueError, IndexError) as e:
            logger.warning('Could not parse score from LLM output: %s', e)
        return 0.0, 'score_not_found'


__all__ = ['MultimodalJudgeConfig', 'MultimodalJudge']
