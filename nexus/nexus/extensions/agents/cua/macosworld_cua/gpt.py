from __future__ import annotations

import copy

from typing import Any, Literal, cast

from pydantic import Field

from nexus.core import agent_registry
from nexus.extensions.agents.cua.cua_gpt_agent.agent import (
    OPERATOR_PROMPT,
    CuaGptAdapter,
    CuaGptAgentInput,
    CuaGptAgentOutput,
    _build_pyautogui_run_command,
    _current_date_text,
    _get,
    run_cua_gpt_with_adapter,
)
from nexus.extensions.agents.cua.cua_gpt_agent.agent import (
    _convert_action_to_pyautogui as _default_action,
)


MACOSWORLD_PAPER_PROMPT = """You are using a macOS computer to complete a user-given task. Additional Notes:
* Available xdotool keys: ctrl, command, option, backspace, tab, enter, esc, del, left, up, right, down, and single ASCII characters.
* When you think the task can not be done, say ```FAIL```, don't easily say ```FAIL```, try your best to do the task. When you think the task is completed, say ```DONE```. Include the three backticks. If the task is not completed, don't raise any of these two flags.
* You may need my username and password. My username is `{USERNAME}` and password is `{CLIENT_PASSWORD}`.
"""  # noqa: E501

_BLACK_SCREENSHOT = (
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAIAAAACDbGyAAAADElEQVR4nGNgoC4AAABQAAHzMPcqAAAAAElFTkSuQmCC'
)


def _is_macos(platform_name: str) -> bool:
    normalized = ''.join(ch for ch in platform_name.lower() if ch.isalnum())
    return normalized.startswith(('macos', 'darwin', 'osx')) or normalized == 'mac'


def _convert_action_to_pyautogui(
    action_type: str | None, args: dict[str, Any], *, platform_name: str = 'ubuntu'
) -> str | None:
    if _is_macos(platform_name) and action_type == 'click' and args.get('button') in ('back', 'forward'):
        x, y = args.get('x'), args.get('y')
        if x is None or y is None:
            return None
        key = '[' if args['button'] == 'back' else ']'
        return f"pyautogui.moveTo({x}, {y})\npyautogui.hotkey('command', '{key}')"
    return _default_action(action_type, args)


def _screenshot_parts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        if value.get('type') in {'input_image', 'computer_screenshot'} and isinstance(value.get('image_url'), str):
            return [value]
        return [part for child in value.values() for part in _screenshot_parts(child)]
    if isinstance(value, list):
        return [part for child in value for part in _screenshot_parts(child)]
    return []


def _limited_screenshot_history(history: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    result = copy.deepcopy(history)
    for part in _screenshot_parts(result)[:-keep]:
        part['image_url'] = _BLACK_SCREENSHOT
        part.pop('detail', None)
    return result


def _native_history_output(message: Any) -> list[dict[str, Any]]:
    fields = _get(message, 'provider_specific_fields') or {}
    items = _get(fields, 'responses_output_items')
    if not isinstance(items, list) or not items:
        raise ValueError('Screenshot-window mode requires complete native Responses output items')
    for item in items:
        if not isinstance(item, dict) or item.get('type') == 'item_reference':
            raise ValueError('Screenshot-window mode cannot replay bare server-side item references')
        if item.get('type') in {'reasoning', 'compaction'} and not item.get('encrypted_content'):
            raise ValueError('Screenshot-window mode requires encrypted reasoning for lossless replay')
    return copy.deepcopy(items)


async def _execute_pyautogui_code(
    runtime: Any,
    code: str,
    *,
    mode: Literal['gui_pyautogui', 'run_command'],
    timeout: float,
) -> dict[str, Any]:
    if mode == 'gui_pyautogui':
        resp = await runtime.gui_pyautogui(code, timeout=timeout)
        return {
            'success': bool(_get(resp, 'success', False)),
            'error': _get(resp, 'error'),
            'stdout': _get(resp, 'stdout'),
            'stderr': _get(resp, 'stderr'),
        }

    if mode == 'run_command':
        resp = await runtime.run_command(command=_build_pyautogui_run_command(code), timeout=timeout)
        status = _get(resp, 'status')
        status_value = getattr(status, 'value', status)
        return_code = _get(resp, 'return_code')
        success = status_value == 'Finished' and return_code == 0
        error = None if success else f'run_command status={status_value} return_code={return_code}'
        return {
            'success': success,
            'error': error,
            'command_status': status_value,
            'return_code': return_code,
            'stdout': _get(resp, 'stdout'),
            'stderr': _get(resp, 'stderr'),
        }

    raise ValueError(f'Unsupported pyautogui_execution_mode: {mode}')


class MacOSWorldGptInput(CuaGptAgentInput):
    max_screenshots: int | None = Field(default=None, ge=1)
    max_recent_screenshots: int | None = Field(default=None, ge=1)
    prompt_style: Literal['operator', 'macosworld_paper'] = 'operator'
    home_dir: str | None = None


class MacOSWorldGptOutput(CuaGptAgentOutput):
    screenshots_sent: int = 0


class MacOSWorldGptAdapter(CuaGptAdapter):
    output_type = MacOSWorldGptOutput

    def instructions(self, agent_input: Any) -> str:
        home_dir = agent_input.home_dir or '/Users/ec2-user'
        if agent_input.prompt_style == 'macosworld_paper':
            if not _is_macos(agent_input.platform_name):
                raise ValueError('macosworld_paper prompt requires platform_name=macos')
            return MACOSWORLD_PAPER_PROMPT.format(
                USERNAME=home_dir.rstrip('/').rsplit('/', 1)[-1],
                CLIENT_PASSWORD=agent_input.client_password,
            )
        return OPERATOR_PROMPT.format(
            PLATFORM=agent_input.platform_name,
            CLIENT_PASSWORD=agent_input.client_password,
            CURRENT_DATE=_current_date_text(agent_input.task_current_date),
            HOME_DIR=home_dir,
        )

    async def execute_code(
        self, runtime: Any, code: str, *, mode: Literal['gui_pyautogui', 'run_command'], timeout: float
    ) -> dict[str, Any]:
        return await _execute_pyautogui_code(runtime, code, mode=mode, timeout=timeout)

    def parse_action(self, action_type: str | None, args: dict[str, Any], agent_input: Any) -> str | None:
        return _convert_action_to_pyautogui(action_type, args, platform_name=agent_input.platform_name)

    def action_diagnostics(self, response: dict[str, Any]) -> dict[str, Any]:
        diagnostics = {}
        if response.get('command_status') in {'Finished', 'Error', 'TimeLimitExceeded'}:
            diagnostics['command_status'] = response['command_status']
        if type(response.get('return_code')) is int:
            diagnostics['return_code'] = response['return_code']
        exception_type = response.get('exception_type')
        if isinstance(exception_type, str) and exception_type.isidentifier():
            diagnostics['exception_type'] = exception_type
        return diagnostics

    def initialize(self, agent_input: Any, client: Any) -> None:
        self.settings = agent_input
        self.history = []
        self.screenshots_sent = 0
        if agent_input.max_recent_screenshots is not None and getattr(client, 'stateful_previous_response', False):
            raise ValueError('Screenshot-window mode requires client_args.stateful_previous_response=false')

    def prepare_request(
        self, request_input: list[dict[str, Any]], request: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str | None]:
        # Count only new observations; replayed history does not consume budget.
        new_screenshots = len(_screenshot_parts(request_input))
        budget = self.settings.max_screenshots
        if budget is not None and self.screenshots_sent + new_screenshots > budget:
            return request_input, f'Screenshot budget exhausted ({self.screenshots_sent}/{budget}).'
        self.screenshots_sent += new_screenshots
        if self.settings.max_recent_screenshots is not None:
            self.history.extend(request_input)
            request_input = _limited_screenshot_history(self.history, self.settings.max_recent_screenshots)
            request.pop('previous_response_id', None)
            request['include'] = ['reasoning.encrypted_content']
            request['store'] = False
        return request_input, None

    def observe_response(self, message: Any) -> None:
        if self.settings.max_recent_screenshots is not None:
            self.history.extend(_native_history_output(message))

    def output_fields(self) -> dict[str, Any]:
        return {'screenshots_sent': self.screenshots_sent}

    def is_terminal_failure(self, assistant_text: str) -> bool:
        return self.settings.prompt_style == 'macosworld_paper' and '```FAIL```' in assistant_text

    def execution_exception(self, error: Exception) -> dict[str, Any]:
        return {'success': False, 'exception_type': type(error).__name__}


@agent_registry.register('macosworld_cua_gpt')
async def run_macosworld_cua_gpt(agent_input: MacOSWorldGptInput) -> MacOSWorldGptOutput:
    return cast(MacOSWorldGptOutput, await run_cua_gpt_with_adapter(agent_input, adapter=MacOSWorldGptAdapter()))
