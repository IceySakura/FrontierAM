"""macOSWorld's raw-command GPT-4o harness, using Nexus model/runtime clients."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import shlex

from pathlib import Path
from typing import Any, cast

from PIL import Image
from pydantic import Field

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    WithMainTrajectory,
    WithUserPrompt,
    agent_registry,
    require_context,
)
from nexus.extensions.agents.cua.macosworld_vlm.upstream import GPT_SYSTEM_PROMPT, parse_agent_output
from nexus.runtime.providers.macosworld_devcloud.gui import mac_multi_click
from nexus.utils.timeout import async_timeout


logger = logging.getLogger(__name__)


class MacOSWorldVLMInput(AgentRunInput, WithUserPrompt):
    runtime_name: str = 'main'
    max_iterations: int = Field(default=30, ge=1)
    max_screenshots: int = Field(default=15, ge=1)
    recent_screenshots: int = Field(default=3, ge=1)
    model_display_width: int = Field(default=1024, ge=1)
    model_display_height: int = Field(default=768, ge=1)
    action_interval: float = Field(default=1, ge=0)
    action_timeout: float = Field(default=120, gt=0)
    step_timeout: float = Field(default=120, gt=0)
    trajectory_dir: str = ''
    home_dir: str | None = None
    client_password: str = 'password'


class MacOSWorldVLMOutput(AgentRunOutput, WithMainTrajectory):
    exit_status: str
    iterations: int
    screenshots_sent: int
    final_answer: str | None = None
    action_results: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


def model_screenshot(image: str, width: int, height: int) -> dict[str, Any]:
    """Resize observation only; normalized commands map to actual logical desktop dimensions."""
    with Image.open(io.BytesIO(base64.b64decode(image))) as screenshot:
        resized = screenshot.convert('RGB').resize((width, height), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        resized.save(output, format='PNG')
    return {
        'type': 'image_url',
        'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode(),
            'detail': 'high',
        },
    }


def build_messages(instruction: str, images: list[dict[str, Any]], system_prompt: str) -> list[dict[str, Any]]:
    label = 'Screenshot: ' if len(images) == 1 else 'Rolling window of historical screenshots in chronological order: '
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': [{'type': 'text', 'text': f'Task: {instruction}\n{label}'}, *images]},
    ]


def action_code(action: dict[str, Any], width: int, height: int) -> str:
    """Translate parsed action data, never arbitrary model Python, to GUI operations."""
    name = action['action']
    if name == 'move_to':
        x, y = float(action['x']), float(action['y'])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('Non-finite coordinates')
        return f'pyautogui.moveTo({max(0, min(width - 1, round(x * (width - 1))))}, {max(0, min(height - 1, round(y * (height - 1))))})'
    if name in {'left_click', 'middle_click', 'right_click'}:
        return f'pyautogui.click(button={name.split("_")[0]!r})'
    if name == 'double_click':
        return mac_multi_click(None, None, 'left', 2)
    if name in {'scroll_up', 'scroll_down'}:
        amount = float(action['amount'])
        if not math.isfinite(amount):
            raise ValueError('Non-finite scroll amount')
        clicks = max(0, min(height, round(amount * height)))
        return f'pyautogui.scroll({clicks if name == "scroll_up" else -clicks})'
    if name == 'type_text':
        text = ''.join(c for c in action['text'] if c.isascii())
        return f'pyautogui.write({text!r}, interval=0.1)'
    if name == 'key_press':
        aliases = {'del': 'delete', 'ctrl': 'ctrl', 'alt': 'option', 'return': 'enter'}
        keys = [aliases.get(k, k) for k in action['key'].split('-')]
        return f'pyautogui.hotkey(*{keys!r})'
    raise ValueError(f'Unsupported action: {name}')


@agent_registry.register('macosworld_vlm')
async def run_macosworld_vlm(agent_input: MacOSWorldVLMInput) -> MacOSWorldVLMOutput:
    ctx = require_context()
    runtime = await ctx.get_runtime(agent_input.runtime_name)
    client = ctx.get_openai_client('main')
    prompt = agent_input.user_prompt
    parts = [] if isinstance(prompt, str) else cast(list[dict[str, Any]], prompt)
    instruction = (
        prompt
        if isinstance(prompt, str)
        else '\n'.join(p['text'] for p in parts if isinstance(p, dict) and p.get('type') == 'text')
    )
    initial_image = (
        None
        if isinstance(prompt, str)
        else next(
            (
                p['image_url']['url'].split(',', 1)[1]
                for p in parts
                if isinstance(p, dict) and p.get('type') == 'image_url'
            ),
            None,
        )
    )
    username = (agent_input.home_dir or '/Users/ec2-user').rstrip('/').rsplit('/', 1)[-1]
    system_prompt = GPT_SYSTEM_PROMPT.replace('"ec2-user"', f'"{username}"').replace(
        '"000000"', f'"{agent_input.client_password}"'
    )
    trajectory_dir = Path(agent_input.trajectory_dir) if agent_input.trajectory_dir else None
    if trajectory_dir is not None:
        trajectory_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    iterations = 0
    screenshots = 0
    status = 'max_iterations'
    error = None
    final_answer = None
    try:
        screen = await runtime.gui_get_screen_size()
        if not screen.success or not screen.width or not screen.height:
            raise RuntimeError('Cannot determine logical desktop size')
        for i in range(min(agent_input.max_iterations, agent_input.max_screenshots)):
            async with async_timeout(agent_input.step_timeout):
                if i == 0 and initial_image:
                    image = initial_image
                else:
                    shot = await runtime.gui_screenshot()
                    if not shot.success or not shot.image:
                        raise RuntimeError(f'Screenshot failed: {shot.error}')
                    image = shot.image
                if trajectory_dir is not None:
                    (trajectory_dir / f'step_{i + 1:03d}.png').write_bytes(base64.b64decode(image))
                images.append(
                    model_screenshot(image, agent_input.model_display_width, agent_input.model_display_height)
                )
                images = images[-agent_input.recent_screenshots :]
                messages = build_messages(instruction, images, system_prompt)
                screenshots += 1
                iterations += 1
                try:
                    response = await client.create_completion(messages, temperature=1.0, top_p=0.9)
                    final_answer = response.choices[0].message.content
                    if not isinstance(final_answer, str) or not final_answer.strip():
                        raise ValueError('Model returned no text commands')
                except Exception as exc:
                    status, error = 'inference_error', str(exc)
                    break
                history.extend([*messages, {'role': 'assistant', 'content': final_answer}])
                actions = parse_agent_output(final_answer)
                logger.info(
                    'macOSWorld VLM turn %s: %s screenshots, %s parsed actions', iterations, screenshots, len(actions)
                )
                if trajectory_dir is not None:
                    (trajectory_dir / f'step_{i + 1:03d}.json').write_text(
                        json.dumps({'response': final_answer, 'actions': actions}, ensure_ascii=False, indent=2)
                    )
                for action in actions:
                    await asyncio.sleep(agent_input.action_interval)
                    name = action['action']
                    if name in {'done', 'fail'}:
                        status = 'finished' if name == 'done' else 'infeasible'
                        results.append({'iteration': iterations, 'action': action, 'success': True})
                        break
                    if name == 'wait':
                        seconds = float(action['seconds'])
                        if not math.isfinite(seconds) or not 0 <= seconds <= agent_input.action_timeout:
                            raise ValueError('Invalid wait duration')
                        await asyncio.sleep(seconds)
                        results.append({'iteration': iterations, 'action': action, 'success': True})
                        continue
                    code = action_code(action, screen.width, screen.height)
                    command = 'python3 -c ' + shlex.quote('import pyautogui, time\n' + code)
                    outcome = await runtime.run_command(command=command, timeout=agent_input.action_timeout)
                    success = outcome.return_code == 0
                    results.append({
                        'iteration': iterations,
                        'action': action,
                        'success': success,
                        'error': None if success else (outcome.stderr or outcome.stdout or 'Action failed'),
                    })
                    if trajectory_dir is not None:
                        (trajectory_dir / 'action_results.json').write_text(
                            json.dumps(results, ensure_ascii=False, indent=2)
                        )
                    if not success:
                        raise RuntimeError(results[-1]['error'])
                if status in {'finished', 'infeasible'}:
                    break
    except Exception as exc:
        status, error = 'runtime_error', str(exc)
    return MacOSWorldVLMOutput(
        exit_status=status,
        iterations=iterations,
        screenshots_sent=screenshots,
        final_answer=final_answer,
        main_trajectory=history,
        action_results=results,
        error=error,
    )
