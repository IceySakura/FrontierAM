from __future__ import annotations

import json as _json
import math

from typing import Any, Literal, cast

from pydantic import Field

from nexus.core import agent_registry
from nexus.extensions.agents.cua.cua_anthropic_agent.agent import (
    COMPUTER_USE_BETA_FLAG_20250124,
    COMPUTER_USE_TYPE_20250124,
    SYSTEM_PROMPT_TEMPLATE,
    SYSTEM_PROMPT_WINDOWS_TEMPLATE,
    ClaudeRuntimeProfile,
    CuaAnthropicAdapter,
    CuaAnthropicAgentInput,
    CuaAnthropicAgentOutput,
    _build_pyautogui_run_command,
    _format_prompt_date,
    _get,
    logger,
    run_cua_anthropic_with_adapter,
    text_input_code,
)
from nexus.extensions.agents.cua.cua_anthropic_agent.agent import (
    get_claude_runtime_profile as _default_profile,
)
from nexus.runtime.providers.macosworld_devcloud.gui import mac_multi_click


CLAUDE_37_PROFILE = ClaudeRuntimeProfile(
    label='Claude 3.7 Sonnet computer-use implementation',
    computer_use_beta_flag=COMPUTER_USE_BETA_FLAG_20250124,
    computer_use_type=COMPUTER_USE_TYPE_20250124,
    thinking_mode='legacy',
    default_max_tokens=4096,
)


def get_claude_runtime_profile(model_name: str) -> ClaudeRuntimeProfile:
    normalized = (model_name or '').lower().replace('_', '-')
    if '3-7' in normalized or '3.7' in normalized:
        return CLAUDE_37_PROFILE
    return _default_profile(model_name)


SYSTEM_PROMPT_MACOS_TEMPLATE = """<SYSTEM_CAPABILITY>
* You are operating a macOS desktop with internet access through screenshots, mouse and keyboard actions.
* Use the application's GUI to complete the task. Open applications with Spotlight (Command+Space) or Finder.
* Use Command for macOS shortcuts such as copy, paste, select all and save. Control is a different key.
* Prefer the application or browser already open for the task. Use the macOS menu bar when needed.
* The home directory is {HOME_DIR}.
* Screenshots may contain a different UI language from the task instruction; follow the visible interface.
* Verify the visible result before declaring completion. Output [DONE] when finished.
* If the task cannot be completed, explain the blocker and output [INFEASIBLE].
* When asking the user a question, do not perform actions.
* The current date is {CURRENT_DATE}.
</SYSTEM_CAPABILITY>"""


def _is_macos(platform_name: str) -> bool:
    return ''.join(c for c in platform_name.lower() if c.isalnum()) in {'mac', 'macos', 'osx', 'darwin'}


def build_system_prompt(
    platform: str,
    task_current_date: Any = None,
    system_prompt_suffix: str = '',
    *,
    home_dir: str | None = None,
) -> str:
    if _is_macos(platform):
        template = SYSTEM_PROMPT_MACOS_TEMPLATE.replace('{HOME_DIR}', home_dir or 'the current user home directory')
    else:
        template = SYSTEM_PROMPT_WINDOWS_TEMPLATE if platform == 'Windows' else SYSTEM_PROMPT_TEMPLATE
    system_prompt = template.replace('{CURRENT_DATE}', _format_prompt_date(task_current_date))
    return f'{system_prompt}{" " + system_prompt_suffix if system_prompt_suffix else ""}'


def _trim_image_window(messages: list[dict[str, Any]], keep: int) -> None:
    """Trim all observation images, including the initial image, preserving tool IDs/text."""
    images: list[tuple[list, dict]] = []

    def visit(blocks: list) -> None:
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'image':
                images.append((blocks, block))
            elif isinstance(block.get('content'), list):
                visit(block['content'])

    for message in messages:
        if isinstance(message.get('content'), list):
            visit(message['content'])
    for blocks, block in images[: max(0, len(images) - keep)]:
        blocks.remove(block)


def _computer_key(key: str, platform_name: str) -> str:
    key = key.strip().lower()
    if _is_macos(platform_name):
        key = {
            'cmd': 'command',
            'command_l': 'command',
            'command_r': 'command',
            'super': 'command',
            'super_l': 'command',
            'super_r': 'command',
            'meta': 'command',
            'meta_l': 'command',
            'meta_r': 'command',
            'option': 'alt',
            'option_l': 'alt',
            'option_r': 'alt',
            'control': 'ctrl',
            'control_l': 'ctrl',
            'control_r': 'ctrl',
            'return': 'enter',
            'shift_l': 'shift',
            'shift_r': 'shift',
        }.get(key, key)
    return {'page_down': 'pagedown', 'page_up': 'pageup', 'super_l': 'win', 'super': 'command', 'escape': 'esc'}.get(
        key, key
    )


def _computer_coordinate(value: Any, field: str) -> tuple[int, int] | None:
    # Some native toolset responses encode structured fields as JSON strings.
    if value is None:
        return None
    if isinstance(value, str):
        value = _json.loads(value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f'{field} must be a pair of numbers')
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value):
        raise ValueError(f'{field} must contain finite numbers')
    return int(value[0]), int(value[1])


def _parse_actions_from_tool_call(
    tool_call: dict[str, Any],
    resize_factor: tuple[float, float] | None = None,
    *,
    platform_name: str = 'ubuntu',
) -> str:
    """OSWorld action port with shared text input. Returns code or 'FAIL' / 'DONE'.

    ``resize_factor`` — multiply model-emitted (x, y) coords by (sx, sy) to
    map them from model-display space (e.g. 1280x720) back to real sandbox
    pixels (e.g. 1920x1080). Matches OSWorld anthropic/main.py:181-191.
    """
    tool_name = tool_call.get('name', 'computer')
    if tool_name != 'computer':
        raise ValueError(f'Unknown tool: {tool_name}')

    result = ''
    function_args = tool_call.get('input') or {}

    # Batched actions: recursive call.
    batched_actions = function_args.get('actions')
    if batched_actions is not None:
        if not isinstance(batched_actions, list):
            raise ValueError(f'{batched_actions} must be a list')
        for batched_action in batched_actions:
            if not isinstance(batched_action, dict):
                raise ValueError(f'{batched_action} must be a dict')
            result += _parse_actions_from_tool_call(
                {'input': batched_action}, resize_factor, platform_name=platform_name
            )
        return result

    action = function_args.get('action')
    if not action:
        return result
    action_conversion = {'left click': 'click', 'right click': 'right_click'}
    action = action_conversion.get(action, action)

    text = function_args.get('text')
    coordinate = _computer_coordinate(function_args.get('coordinate'), 'coordinate')
    start_coordinate = _computer_coordinate(function_args.get('start_coordinate'), 'start_coordinate')
    scroll_direction = function_args.get('scroll_direction')
    scroll_amount = int(function_args.get('scroll_amount') or 0)
    raw_duration = function_args.get('duration')
    duration = float(raw_duration) if raw_duration is not None else None
    if duration is not None and (not math.isfinite(duration) or duration < 0):
        raise ValueError('duration must be a finite non-negative number')
    repeat = int(function_args.get('repeat') or 1)

    # Reverse-scale coordinates: model outputs in 1280x720 space -> real px.
    if resize_factor is not None:
        if coordinate and isinstance(coordinate, (list, tuple)) and len(coordinate) == 2:
            coordinate = (
                int(coordinate[0] * resize_factor[0]),
                int(coordinate[1] * resize_factor[1]),
            )
        if start_coordinate and isinstance(start_coordinate, (list, tuple)) and len(start_coordinate) == 2:
            start_coordinate = (
                int(start_coordinate[0] * resize_factor[0]),
                int(start_coordinate[1] * resize_factor[1]),
            )

    if action == 'left_mouse_down':
        result += 'pyautogui.mouseDown()\n'
    elif action == 'left_mouse_up':
        result += 'pyautogui.mouseUp()\n'
    elif action == 'hold_key':
        if not isinstance(text, str):
            raise ValueError(f'{text} must be a string')
        keys = text.split('+')
        for key in keys:
            key = _computer_key(key, platform_name)
            result += f'pyautogui.keyDown({key!r})\n'
    elif action in ('mouse_move', 'left_click_drag'):
        if coordinate is None:
            raise ValueError(f'coordinate is required for {action}')
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
            raise ValueError(f'{coordinate} must be a tuple of length 2')
        if not all(isinstance(i, int) for i in coordinate):
            raise ValueError(f'{coordinate} must be a tuple of ints')
        x, y = coordinate[0], coordinate[1]
        if action == 'mouse_move':
            result += f'pyautogui.moveTo({x}, {y}, duration={duration or 0.5})\n'
        elif action == 'left_click_drag':
            if start_coordinate:
                if not isinstance(start_coordinate, (list, tuple)) or len(start_coordinate) != 2:
                    raise ValueError(f'{start_coordinate} must be a tuple of length 2')
                start_x, start_y = start_coordinate[0], start_coordinate[1]
                result += f'pyautogui.moveTo({start_x}, {start_y}, duration={duration or 0.5})\n'
            result += f'pyautogui.dragTo({x}, {y}, duration={duration or 0.5}, button="left")\n'
    elif action in ('key', 'type'):
        if text is None:
            raise ValueError(f'text is required for {action}')
        if not isinstance(text, str):
            raise ValueError(f'{text} must be a string')
        if action == 'key':
            keys = text.split('+')
            for _ in range(repeat):
                for key in keys:
                    key = _computer_key(key, platform_name)
                    result += f'pyautogui.keyDown({key!r})\n'
                for key in reversed(keys):
                    key = _computer_key(key, platform_name)
                    result += f'pyautogui.keyUp({key!r})\n'
        elif action == 'type':
            result += text_input_code(text)
    elif action == 'scroll':
        if text is not None:
            result += f'pyautogui.keyDown({_computer_key(text, platform_name)!r})\n'
        if coordinate is None:
            if scroll_direction in ('up', 'down'):
                amount = scroll_amount if scroll_direction == 'up' else -scroll_amount
                result += f'pyautogui.scroll({amount})\n'
            elif scroll_direction in ('left', 'right'):
                amount = scroll_amount if scroll_direction == 'right' else -scroll_amount
                result += f'pyautogui.hscroll({amount})\n'
        else:
            x, y = coordinate[0], coordinate[1]
            if scroll_direction in ('up', 'down'):
                amount = scroll_amount if scroll_direction == 'up' else -scroll_amount
                result += f'pyautogui.scroll({amount}, {x}, {y})\n'
            elif scroll_direction in ('left', 'right'):
                amount = scroll_amount if scroll_direction == 'right' else -scroll_amount
                result += f'pyautogui.hscroll({amount}, {x}, {y})\n'
        if text is not None:
            result += f'pyautogui.keyUp({_computer_key(text, platform_name)!r})\n'
    elif action in ('left_click', 'right_click', 'double_click', 'middle_click', 'left_press', 'triple_click'):
        if text:
            keys = text.split('+')
            for key in keys:
                key = _computer_key(key, platform_name)
                result += f'pyautogui.keyDown({key!r})\n'
        if coordinate is not None:
            x, y = coordinate
            if action == 'left_click':
                result += f'pyautogui.click({x}, {y})\n'
            elif action == 'right_click':
                result += f'pyautogui.rightClick({x}, {y})\n'
            elif action == 'double_click':
                result += (
                    (mac_multi_click(x, y, 'left', 2) + '\n')
                    if _is_macos(platform_name)
                    else f'pyautogui.doubleClick({x}, {y})\n'
                )
            elif action == 'middle_click':
                result += f'pyautogui.middleClick({x}, {y})\n'
            elif action == 'left_press':
                result += f'pyautogui.mouseDown({x}, {y})\n'
                result += 'time.sleep(1)\n'
                result += f'pyautogui.mouseUp({x}, {y})\n'
            elif action == 'triple_click':
                result += (
                    (mac_multi_click(x, y, 'left', 3) + '\n')
                    if _is_macos(platform_name)
                    else f'pyautogui.tripleClick({x}, {y})\n'
                )
        else:
            if action == 'left_click':
                result += 'pyautogui.click()\n'
            elif action == 'right_click':
                result += 'pyautogui.rightClick()\n'
            elif action == 'double_click':
                result += (
                    (mac_multi_click(None, None, 'left', 2) + '\n')
                    if _is_macos(platform_name)
                    else 'pyautogui.doubleClick()\n'
                )
            elif action == 'middle_click':
                result += 'pyautogui.middleClick()\n'
            elif action == 'left_press':
                result += 'pyautogui.mouseDown()\n'
                result += 'time.sleep(1)\n'
                result += 'pyautogui.mouseUp()\n'
            elif action == 'triple_click':
                result += (
                    (mac_multi_click(None, None, 'left', 3) + '\n')
                    if _is_macos(platform_name)
                    else 'pyautogui.tripleClick()\n'
                )
        if text:
            keys = text.split('+')
            for key in reversed(keys):
                key = _computer_key(key, platform_name)
                result += f'pyautogui.keyUp({key!r})\n'
    elif action == 'wait':
        result += f'pyautogui.sleep({duration or 0.5})\n'
    elif action == 'fail':
        result += 'FAIL'
    elif action == 'done':
        result += 'DONE'
    elif action == 'screenshot':
        result += 'pyautogui.sleep(0.1)\n'
    elif action in ('cursor_position', 'zoom'):
        raise ValueError(f'{action} is not supported by this executor; use the full screenshot coordinate space')
    else:
        raise ValueError(f'Invalid action: {action}')

    return result


async def _execute_computer_code(
    runtime: Any,
    code: str,
    *,
    execution_mode: str,
    timeout: float,
) -> tuple[bool, str | None]:
    if not code or code in ('FAIL', 'DONE'):
        return False, None
    exec_code = f'import pyautogui\nimport time\n{code}'
    try:
        if execution_mode == 'run_command':
            wrapped = _build_pyautogui_run_command(exec_code)
            response = await runtime.run_command(command=wrapped, timeout=timeout)
            status = _get(response, 'status')
            status_value = getattr(status, 'value', status)
            return_code = _get(response, 'return_code')
            ok = status_value == 'Finished' and return_code == 0
            if ok:
                return True, None
            detail = _get(response, 'stderr') or _get(response, 'error') or ''
            return False, f'run_command status={status_value} rc={return_code}: {str(detail)[-2000:]}'
        response = await runtime.gui_pyautogui(exec_code, timeout=timeout)
        ok = bool(_get(response, 'success', False))
        return ok, _get(response, 'error')
    except Exception as exc:  # noqa: BLE001
        logger.exception('pyautogui execution failed')
        return False, str(exc)


class MacOSWorldAnthropicInput(CuaAnthropicAgentInput):
    strict_image_window: bool = False
    max_screenshots: int | None = Field(default=None, ge=1)
    home_dir: str | None = None
    computer_only: bool = False
    computer_tool_schema: Literal['auto', 'computer_toolset_20260801'] = 'auto'
    auto_screen_size: bool = False
    model_display_width: int = Field(default=1280, gt=0)
    model_display_height: int = Field(default=720, gt=0)


class MacOSWorldAnthropicOutput(CuaAnthropicAgentOutput):
    screenshots: int = 0


class MacOSWorldAnthropicAdapter(CuaAnthropicAdapter):
    def __init__(self) -> None:
        self.screenshots = 1
        self.execution_results = []

    output_type = MacOSWorldAnthropicOutput

    def profile(self, model: str) -> ClaudeRuntimeProfile:
        return get_claude_runtime_profile(model)

    def prompt(self, agent_input: Any, suffix: str) -> str:
        return build_system_prompt(
            agent_input.platform_name, agent_input.task_current_date, suffix, home_dir=agent_input.home_dir
        )

    def parse_action(self, block: dict[str, Any], resize_factor: tuple[float, float], agent_input: Any) -> str:
        return _parse_actions_from_tool_call(block, resize_factor, platform_name=agent_input.platform_name)

    async def execute_action(
        self, runtime: Any, action: dict[str, Any], *, execution_mode: str, timeout: float
    ) -> tuple[bool, str | None]:
        if action.get('parse_error'):
            return False, f'Invalid computer action: {action["parse_error"]}'
        return await _execute_computer_code(
            runtime, action.get('command') or '', execution_mode=execution_mode, timeout=timeout
        )

    def termination_text(self, raw_response: str | None, reasonings: list[str], actions: list[dict[str, Any]]) -> str:
        return '\n'.join(reasonings) if not actions else ''

    async def screen_size(self, runtime: Any, agent_input: Any) -> tuple[int, int]:
        if not agent_input.auto_screen_size:
            return await super().screen_size(runtime, agent_input)
        screen = await runtime.gui_get_screen_size()
        width, height = _get(screen, 'width'), _get(screen, 'height')
        if not _get(screen, 'success', False) or not all(
            isinstance(value, int) and value > 0 for value in (width, height)
        ):
            raise RuntimeError('CUA could not determine the runtime desktop size')
        return width, height

    def configure_tools(
        self, wire_style: str, profile: ClaudeRuntimeProfile, agent_input: Any
    ) -> tuple[list[dict[str, Any]], bool]:
        if agent_input.computer_tool_schema == 'computer_toolset_20260801':
            wire_style = 'bedrock_invoke'
        tools, use_toolset = super().configure_tools(wire_style, profile, agent_input)
        if agent_input.computer_only:
            tools = [
                tool for tool in tools if tool.get('name') not in {'bash', 'text_editor', 'str_replace_based_edit_tool'}
            ]
        return tools, use_toolset

    def trim_images(self, messages: list[dict[str, Any]], agent_input: Any) -> None:
        if agent_input.strict_image_window:
            _trim_image_window(messages, agent_input.only_n_most_recent_images)

    def action_fields(self, parse_error: str | None) -> dict[str, Any]:
        return {'parse_error': parse_error}

    def allow_actions(self, actions: list[dict[str, Any]], agent_input: Any) -> bool:
        self.execution_results = []
        # Reserve the whole result group; never execute a partial group at the budget boundary.
        new_images = sum(
            act.get('name', 'computer') in (None, 'computer')
            and (act.get('input', {}).get('action') == 'screenshot' or ai == len(actions) - 1)
            for ai, act in enumerate(actions)
        )
        budget = agent_input.max_screenshots
        if budget is not None and self.screenshots + new_images > budget:
            logger.info('Screenshot budget exhausted: %d/%d', self.screenshots, budget)
            return False
        return True

    def validate_tool(self, tool_name: str, agent_input: Any) -> None:
        if agent_input.computer_only and tool_name in {'bash', 'text_editor', 'str_replace_based_edit_tool'}:
            raise RuntimeError(f'{tool_name} is disabled for this computer-only evaluation')

    def record_execution(self, action: dict[str, Any], success: bool, error: str | None) -> None:
        self.execution_results.append({
            'tool_use_id': action.get('id'),
            'action': action.get('input', {}).get('action'),
            'success': success,
            'error': error,
        })

    def result_fields(self, success: bool, error: str | None) -> dict[str, Any]:
        failures = [str(r['error'] or 'failed') for r in self.execution_results if not r['success']]
        return {
            'action_result': '; '.join(failures) if failures else 'Success',
            'action_results': list(self.execution_results),
        }

    def image_added(self) -> None:
        self.screenshots += 1

    def output_fields(self) -> dict[str, Any]:
        return {'screenshots': self.screenshots}


@agent_registry.register('macosworld_cua_anthropic')
async def run_macosworld_cua_anthropic(agent_input: MacOSWorldAnthropicInput) -> MacOSWorldAnthropicOutput:
    return cast(
        MacOSWorldAnthropicOutput,
        await run_cua_anthropic_with_adapter(agent_input, adapter=MacOSWorldAnthropicAdapter()),
    )
