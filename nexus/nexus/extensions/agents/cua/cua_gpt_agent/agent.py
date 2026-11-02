"""GPT computer-use agent for Nexus (OpenAI Responses API + native `computer` tool).

Port of OSWorld's ``mm_agents/gpt54_agent.py`` (GPT54Agent) into a Nexus agent.

Key differences from the generic ``main_agent`` (which uses chat.completions +
function tools):

- Uses the OpenAI **Responses API** (``client.responses.create``) with the native
  ``{"type": "computer"}`` tool. The model emits structured ``computer_call`` items
  (click/type/scroll/...) rather than free-form pyautogui.
- Routes requests through Nexus' configured ``openai_responses`` client so retries,
  tracing, chat tracking, provider-call capture, and redaction use the shared client
  lifecycle.
- Threads conversation state server-side via ``previous_response_id`` — each turn
  only sends the new screenshot back as a ``computer_call_output``.
- Converts each ``computer_call`` action to a pyautogui snippet and executes it via
  ``runtime.gui_pyautogui`` by default. Experimental runtimes can opt into
  executing the same snippet through ``runtime.run_command``. Screenshots come
  from ``runtime.gui_screenshot``.

Snippets emitted here follow the OSWorld GPT54Agent style: they may prepend
``import pyautogui`` / ``pyperclip`` / ``base64`` and use clipboard paste for
non-ASCII text. This assumes ``nexus.runtime.pyautogui_guard`` is running in
its default (non-strict) mode. If ``NEXUS_PYAUTOGUI_STRICT=1`` is set, the
``type`` action for non-ASCII (or any snippet that begins with ``import``)
will be rejected. click / drag / keypress / scroll / wait continue to emit
bare calls and remain compatible with strict mode.
"""

import asyncio
import base64
import json
import logging
import math

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, Literal

from nexus.core import AgentRunInput, AgentRunOutput, UserInputRequest, agent_registry, require_context
from nexus.core.agent import WithInputProvider, WithMainTrajectory, WithUserPrompt
from nexus.extensions.agents.cua.cua_support import (
    _append_jsonl,
    _init_traj_dir,
    _save_screenshot,
    _traj_dir,
)
from nexus.extensions.agents.cua.cua_text_input import text_input_code


logger = logging.getLogger(__name__)


# Derived from OSWorld mm_agents/gpt54_agent.py OPERATOR_PROMPT. Every line except the
# `[INFEASIBLE]` one is still byte-for-byte identical to the official prompt.
#
# ★ That one line means scores from this agent are NO LONGER directly comparable to the
# official OSWorld leaderboard, or to our own runs before 2026-09-18 (the archived
# osworld_369_gptcu_gpt6astra_nativecu_20260918 and gpt56sol_cua_20260727 batches).
# Record which prompt a batch used when archiving.
#
# Why it was changed: the official wording ("infeasible because of missing apps,
# permissions, contradictory requirements, or other hard blockers") invites a verdict
# that can be reached from one screenshot, and never asks the model to try anything
# first. Measured on osworld_369_gptcu_gpt6astra_nativecu_20260918: 12 tasks whose
# evaluator was a real metric (not the `infeasible` sentinel — those 21 were correct
# answers) were abandoned with a real `[INFEASIBLE]` on an action-free turn, several
# after a single screenshot. The gimp one declared "missing RAW loader" at iteration 2
# of a 60-iteration budget having only taken one screenshot; a chrome one gave up on
# turn 3 because a page had not finished loading.
#
# `genuinely` and the explicit "try one concrete path first" mirror the wording the CC
# guides have always used (native computer-use guides), so the three
# agent families now ask for the same standard of evidence before giving up.
OPERATOR_PROMPT = """

Here are some helpful tips:
- You are operating an {PLATFORM} desktop with internet access.
- My computer password is "{CLIENT_PASSWORD}" when sudo is needed.
- The current date is {CURRENT_DATE}.
- The home directory is "{HOME_DIR}".
- Stick to the website or application already opened for the task when possible.
- Prefer Chrome over Firefox/Chromium unless the task says otherwise.
- You can act without asking for confirmation.
- If content may be off-screen, scroll or zoom out before deciding it is unavailable.
- When possible, bundle multiple GUI actions into one computer-use turn.
- If the task is genuinely impossible (a required application is missing, the
  requirements contradict each other, or the feature does not exist), output
  exactly "[INFEASIBLE]". Do not declare this from a single screenshot — try at
  least one concrete path first, and say what you tried.
"""


# Match the explicit marker only in the final assistant turn. Ordinary words
# about feasibility in intermediate reasoning must not terminate action execution.
_INFEASIBLE_MARKER = '[infeasible]'


# Map Responses `computer` tool key names to pyautogui key names.
_KEY_MAPPING = {
    'alt': 'alt',
    'arrowdown': 'down',
    'arrowleft': 'left',
    'arrowright': 'right',
    'arrowup': 'up',
    'backspace': 'backspace',
    'capslock': 'capslock',
    'cmd': 'command',
    'command': 'command',
    'ctrl': 'ctrl',
    'control': 'ctrl',
    'delete': 'delete',
    'end': 'end',
    'enter': 'enter',
    'return': 'enter',
    'esc': 'esc',
    'escape': 'esc',
    'home': 'home',
    'insert': 'insert',
    'option': 'option',
    'pagedown': 'pagedown',
    'pageup': 'pageup',
    'shift': 'shift',
    'space': 'space',
    'super': 'super',
    'tab': 'tab',
    'win': 'win',
}
_SUPPORTED_ACTION_TYPES = {
    'click',
    'double_click',
    'drag',
    'keypress',
    'mouse_move',
    'move',
    'screenshot',
    'scroll',
    'type',
    'wait',
}
_SCROLL_NUMBER_FIELDS = (
    'x',
    'y',
    'scroll_x',
    'scroll_y',
    'delta_x',
    'delta_y',
    'deltaX',
    'deltaY',
)
_CLICK_BUTTON_MAP = {
    'left': 'left',
    'middle': 'middle',
    'right': 'right',
    'wheel': 'middle',
}
_CLICK_NAVIGATION_KEYS = {
    'back': 'left',
    'forward': 'right',
}
_SAFE_INCOMPLETE_RESPONSE = 'Model response was incomplete.'
_SAFE_ACTION_FAILURE = 'Computer action execution failed.'
_SUCCESS_FINISH_REASONS = {'stop', 'tool_calls'}


class CuaGptAgentInput(AgentRunInput, WithUserPrompt, WithInputProvider):
    """Input for the GPT computer-use agent.

    The os_world task passes system_prompt / user_prompt / tools / post_iteration_hook
    via run_agent(). We only consume user_prompt (the instruction + initial
    screenshot) and ignore the rest — _build_input filters to declared fields, so the
    extra kwargs are dropped safely.
    """

    max_iterations: int = 50
    model: str | None = None
    # Aligned with OSWorld GPT54Agent default: reasoning effort 'xhigh'.
    reasoning_effort: str = 'xhigh'
    # Optional cap on Responses API output tokens; None = no cap (aligned with
    # GPT54Agent, which passes max_output_tokens only when explicitly set).
    max_output_tokens: int | None = None
    runtime_name: str = 'main'
    platform_name: str = 'ubuntu'
    client_password: str = 'password'
    sleep_after_execution: float = 0.5
    step_pause: float = 1.0
    pyautogui_execution_mode: Literal['gui_pyautogui', 'run_command'] = 'gui_pyautogui'
    pyautogui_timeout: float = 30.0
    osworld_task_id: str = ''
    osworld_domain: str = ''
    # Explicit per-run trajectory dump directory. Unset kwargs are dropped by
    # _build_input, so without this field the task's trajectory_dir never arrives.
    trajectory_dir: str = ''
    get_user_input: Callable[[UserInputRequest | str | None], Awaitable[str | None]] | None = None
    task_current_date: str | None = None
    # OSWorld-style optional overrides (accepted for config parity; may be unused)
    observation_type: str = 'screenshot'
    action_space: str = 'pyautogui'
    # Model-space display dimensions the agent maps GUI actions against:
    # screenshots are presented to the model at this size and action coordinates
    # are scaled back to the real screen. These are NOT part of the tool schema —
    # gpt-5.6-sol's native computer tool is sent as bare `{type: computer}`
    # (verified: returns 200). The `computer_use_preview` schema carrying
    # display_width/height/environment is rejected by gpt-5.6-sol via this
    # gateway (400 "Unknown parameter: 'tools[0].display_width'").
    model_display_width: int = 1280
    model_display_height: int = 720


class CuaGptAgentOutput(AgentRunOutput, WithMainTrajectory):
    exit_status: str
    iterations: int
    final_answer: str | None = None


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _get(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)


def _model_dump(value: Any) -> Any:
    if hasattr(value, 'model_dump'):
        return value.model_dump()
    if isinstance(value, list):
        return [_model_dump(v) for v in value]
    if isinstance(value, dict):
        return {k: _model_dump(v) for k, v in value.items()}
    return value


def _instruction_text(user_prompt: Any) -> str:
    """Extract the plain instruction text from a str or content-part list prompt."""
    if isinstance(user_prompt, str):
        return user_prompt
    if isinstance(user_prompt, list):
        parts = []
        for part in user_prompt:
            if _get(part, 'type') == 'text':
                parts.append(_get(part, 'text', ''))
        return '\n'.join(p for p in parts if p)
    return str(user_prompt)


def _current_date_text(task_current_date: str | None) -> str:
    if not task_current_date:
        return datetime.now().strftime('%A, %B %d, %Y')
    try:
        return datetime.fromisoformat(task_current_date).strftime('%A, %B %d, %Y')
    except ValueError:
        return task_current_date


def _point(p: Any) -> tuple[Any, Any]:
    if isinstance(p, (list, tuple)) and len(p) == 2:
        return p[0], p[1]
    if isinstance(p, dict):
        return p.get('x'), p.get('y')
    return _get(p, 'x'), _get(p, 'y')


def _convert_action_to_pyautogui(action_type: str | None, args: dict[str, Any]) -> str | None:
    """Convert a Responses `computer` action to a bare-call pyautogui snippet.

    Pointer/key actions emit bare calls. Text input uses a self-contained
    focus-aware clipboard helper and requires the non-strict desktop executor.
    """
    if not action_type:
        return None
    try:
        if action_type == 'click':
            x, y = args.get('x'), args.get('y')
            button = args.get('button', 'left')
            if x is None or y is None:
                return None
            navigation_key = _CLICK_NAVIGATION_KEYS.get(button)
            if navigation_key is not None:
                return f"pyautogui.moveTo({x}, {y})\npyautogui.hotkey('alt', '{navigation_key}')"
            mapped_button = _CLICK_BUTTON_MAP.get(button)
            if mapped_button is None:
                return None
            return f"pyautogui.moveTo({x}, {y})\npyautogui.click(button='{mapped_button}')"

        if action_type == 'double_click':
            x, y = args.get('x'), args.get('y')
            if x is None or y is None:
                return None
            mapped_button = _CLICK_BUTTON_MAP.get(args.get('button', 'left'))
            if mapped_button is None:
                return None
            return f"pyautogui.moveTo({x}, {y})\npyautogui.doubleClick(button='{mapped_button}')"

        if action_type in ('move', 'mouse_move'):
            x, y = args.get('x'), args.get('y')
            if x is None or y is None:
                return None
            return f'pyautogui.moveTo({x}, {y})'

        if action_type == 'drag':
            path = args.get('path')
            if not path and args.get('from') and args.get('to'):
                path = [args['from'], args['to']]
            if not path or len(path) < 2:
                return None
            fx, fy = _point(path[0])
            if fx is None or fy is None:
                return None
            lines = [f'pyautogui.moveTo({fx}, {fy})']
            for pt in path[1:]:
                x, y = _point(pt)
                if x is None or y is None:
                    return None
                lines.append(f"pyautogui.dragTo({x}, {y}, duration=0.2, button='left')")
            return '\n'.join(lines)

        if action_type == 'type':
            return text_input_code(args.get('text', '')).rstrip()

        if action_type == 'keypress':
            keys = args.get('keys')
            if not keys and args.get('key'):
                keys = [args.get('key')]
            if not keys:
                return None
            if not isinstance(keys, (list, tuple)):
                keys = [keys]
            mapped = [_KEY_MAPPING.get(str(k).lower(), str(k).lower()) for k in keys]
            keys_str = ', '.join(repr(k) for k in mapped)
            return f'pyautogui.hotkey({keys_str})'

        if action_type == 'scroll':
            x, y = args.get('x'), args.get('y')
            scroll_y = int(args.get('scroll_y') or args.get('delta_y') or args.get('deltaY') or 0)
            scroll_x = int(args.get('scroll_x') or args.get('delta_x') or args.get('deltaX') or 0)
            pos = f', x={x}, y={y}' if x is not None and y is not None else ''
            if scroll_y:
                return f'pyautogui.scroll({scroll_y * -1}{pos})'
            if scroll_x:
                return f'pyautogui.hscroll({scroll_x * -1}{pos})'
            return None

        if action_type == 'wait':
            secs = max(0.1, float(args.get('ms', 1000)) / 1000.0)
            return f'time.sleep({secs})'

        if action_type == 'screenshot':
            return 'time.sleep(0.1)'
    except Exception:
        logger.exception('Failed to convert computer action: %s', action_type)
        return None

    logger.warning('Unsupported computer action: %s', action_type)
    return None


def _action_to_dict(action: Any) -> dict[str, Any]:
    raw = _model_dump(action)
    if isinstance(raw, dict):
        return {'type': raw.get('type'), 'args': {k: v for k, v in raw.items() if k != 'type'}}
    return {'type': _get(action, 'type'), 'args': {}}


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def _validate_number_field(
    args: dict[str, Any],
    field: str,
    *,
    required: bool = False,
    minimum: float | None = None,
) -> None:
    if field not in args:
        if required:
            raise ValueError('computer action is missing a required numeric field')
        return
    value = args[field]
    if not _is_finite_number(value) or (minimum is not None and value < minimum):
        raise ValueError('computer action contains an invalid numeric field')


def _validate_drag_point(point: Any) -> None:
    if isinstance(point, list) and len(point) == 2:
        coordinates = point
    elif isinstance(point, dict) and 'x' in point and 'y' in point:
        coordinates = [point['x'], point['y']]
    else:
        raise ValueError('computer drag action contains an invalid point')
    if any(not _is_finite_number(coordinate) for coordinate in coordinates):
        raise ValueError('computer drag action contains an invalid coordinate')


def _validate_computer_action(action_type: Any, args: dict[str, Any]) -> None:
    """Validate every value interpolated into pyautogui code before the turn executes."""

    if not isinstance(action_type, str) or action_type not in _SUPPORTED_ACTION_TYPES:
        raise ValueError('unsupported computer action')

    if action_type in {'click', 'double_click', 'move', 'mouse_move'}:
        _validate_number_field(args, 'x', required=True)
        _validate_number_field(args, 'y', required=True)
        if (
            action_type == 'click'
            and 'button' in args
            and args['button']
            not in {
                *_CLICK_BUTTON_MAP,
                *_CLICK_NAVIGATION_KEYS,
            }
        ):
            raise ValueError('computer click action contains an invalid button')
        if action_type == 'double_click' and 'button' in args and args['button'] not in _CLICK_BUTTON_MAP:
            raise ValueError('computer double-click action contains an invalid button')
        return

    if action_type == 'drag':
        if 'path' in args:
            path = args['path']
            if not isinstance(path, list) or len(path) < 2:
                raise ValueError('computer drag action contains an invalid path')
            for point in path:
                _validate_drag_point(point)
            return
        if 'from' not in args or 'to' not in args:
            raise ValueError('computer drag action is missing a path')
        _validate_drag_point(args['from'])
        _validate_drag_point(args['to'])
        return

    if action_type == 'type':
        if 'text' not in args or not isinstance(args['text'], str):
            raise ValueError('computer type action contains invalid text')
        return

    if action_type == 'keypress':
        has_keys = 'keys' in args
        has_key = 'key' in args
        if has_keys == has_key:
            raise ValueError('computer keypress action must provide exactly one key representation')
        if has_keys:
            keys = args['keys']
            if not isinstance(keys, list) or not keys or any(not isinstance(key, str) or not key for key in keys):
                raise ValueError('computer keypress action contains invalid keys')
        elif not isinstance(args['key'], str) or not args['key']:
            raise ValueError('computer keypress action contains an invalid key')
        return

    if action_type == 'scroll':
        for field in _SCROLL_NUMBER_FIELDS:
            _validate_number_field(args, field)
        if not any(field in args for field in _SCROLL_NUMBER_FIELDS[2:]):
            raise ValueError('computer scroll action is missing a scroll delta')
        return

    if action_type == 'wait':
        _validate_number_field(args, 'ms', minimum=0)


def _response_dump(completion: Any) -> dict[str, Any]:
    """Reconstruct a safe Responses-shaped record from normalized client fields."""

    provider_fields = _get(completion, 'provider_specific_fields', {}) or {}
    response_state = _model_dump(_get(provider_fields, 'response_state'))
    response_output = _model_dump(_get(provider_fields, 'response_output', []))
    if isinstance(response_state, dict) and isinstance(response_output, list):
        return {**response_state, 'output': response_output}

    dumped = _model_dump(completion)
    return dumped if isinstance(dumped, dict) else {'response': dumped}


def _completion_is_incomplete(completion: Any) -> bool:
    provider_fields = _get(completion, 'provider_specific_fields', {}) or {}
    response_status = _get(provider_fields, 'response_status')
    if response_status is None:
        response_status = _get(_get(provider_fields, 'response_state', {}) or {}, 'status')
    if isinstance(response_status, str) and response_status.lower() == 'incomplete':
        return True

    choices = _get(completion, 'choices', []) or []
    if not choices:
        return False
    finish_reason = _get(choices[0], 'finish_reason')
    return isinstance(finish_reason, str) and finish_reason not in _SUCCESS_FINISH_REASONS


def _safe_call_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return '<missing>'
    safe = ''.join(
        character for character in value.strip() if character.isascii() and (character.isalnum() or character in '._-')
    )
    return safe[:128] or '<invalid>'


class _ComputerToolError(ValueError):
    def __init__(self, call_id: Any):
        self.safe_call_id = _safe_call_id(call_id)
        super().__init__(f'Invalid computer tool call: {self.safe_call_id}')


def _parse_computer_tool_call(tool_call: Any) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    call_id = _get(tool_call, 'id')
    if not isinstance(call_id, str) or not call_id.strip():
        raise _ComputerToolError(call_id)

    function = _get(tool_call, 'function')
    arguments = _get(function, 'arguments')
    if not isinstance(arguments, str):
        raise _ComputerToolError(call_id)
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError) as error:
        raise _ComputerToolError(call_id) from error
    if not isinstance(parsed, dict):
        raise _ComputerToolError(call_id)

    actions = parsed.get('actions')
    safety = parsed.get('pending_safety_checks')
    if (
        not isinstance(actions, list)
        or not actions
        or any(not isinstance(action, dict) for action in actions)
        or not isinstance(safety, list)
        or any(not isinstance(check, dict) for check in safety)
    ):
        raise _ComputerToolError(call_id)
    return call_id, actions, safety


async def _screenshot_b64(runtime: Any) -> str:
    resp = await runtime.gui_screenshot()
    image = _get(resp, 'image')
    if not image:
        raise RuntimeError(f'gui_screenshot returned no image: {_get(resp, "error")}')
    return image


def _build_pyautogui_run_command(code: str) -> str:
    encoded_code = base64.b64encode(code.encode('utf-8')).decode('ascii')
    runner = f"""\
import base64
import contextlib
import platform
import sys
import time
import types

def _load_pyautogui():
    try:
        import pyautogui as module
    except SystemExit as exc:
        if 'MouseInfo' not in str(exc) and 'tkinter' not in str(exc):
            raise
        sys.modules.pop('pyautogui', None)
        mouseinfo_stub = types.ModuleType('mouseinfo')
        def _mouse_info_window():
            raise RuntimeError('MouseInfo is unavailable in the runtime GUI environment')
        mouseinfo_stub.MouseInfoWindow = _mouse_info_window
        sys.modules['mouseinfo'] = mouseinfo_stub
        import pyautogui as module
    module.FAILSAFE = False
    _osworld_shift_chars = '~!@#$%^&*()_+' + chr(123) + chr(125) + '|:"<>?'
    _osworld_linux_shift_chars = '~!@#$%^&*()_+' + chr(123) + chr(125) + '|:">?'
    module.isShiftCharacter = lambda character: character.isupper() or character in (
        _osworld_linux_shift_chars if platform.system() == 'Linux' else _osworld_shift_chars
    )
    return module

with contextlib.redirect_stdout(sys.stderr):
    pyautogui = _load_pyautogui()

code = base64.b64decode({encoded_code!r}).decode('utf-8')
globals_dict = {{
    'pyautogui': pyautogui,
    'time': time,
    'platform': platform,
}}
exec(compile(code, '<cua_gpt_agent_pyautogui>', 'exec'), globals_dict, globals_dict)
"""
    return (
        'if command -v python3 >/dev/null 2>&1; then _nexus_python=python3; else _nexus_python=python; fi\n'
        '"$_nexus_python" - <<\'PY\'\n' + runner + '\nPY'
    )


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
            'stdout': _get(resp, 'stdout'),
            'stderr': _get(resp, 'stderr'),
        }

    raise ValueError(f'Unsupported pyautogui_execution_mode: {mode}')


# --------------------------------------------------------------------------- #
# Agent                                                                        #
# --------------------------------------------------------------------------- #


class CuaGptAdapter:
    """Per-run policy, never selected from an OS name or process-global state.

    initialize runs before request hooks; mutable history belongs to this instance.
    Diagnostic/output hooks return additional fields, without changing shared schemas.
    """

    output_type = CuaGptAgentOutput

    def instructions(self, agent_input: CuaGptAgentInput) -> str:
        home_dir = '/home/user' if not agent_input.platform_name.lower().startswith('win') else 'C:\\Users\\user'
        return OPERATOR_PROMPT.format(
            PLATFORM=agent_input.platform_name,
            CLIENT_PASSWORD=agent_input.client_password,
            CURRENT_DATE=_current_date_text(agent_input.task_current_date),
            HOME_DIR=home_dir,
        )

    def parse_action(self, action_type: str | None, args: dict[str, Any], agent_input: CuaGptAgentInput) -> str | None:
        return _convert_action_to_pyautogui(action_type, args)

    async def execute_code(
        self, runtime: Any, code: str, *, mode: Literal['gui_pyautogui', 'run_command'], timeout: float
    ) -> dict[str, Any]:
        return await _execute_pyautogui_code(runtime, code, mode=mode, timeout=timeout)

    def action_diagnostics(self, response: dict[str, Any]) -> dict[str, Any]:
        return {}

    def initialize(self, agent_input: CuaGptAgentInput, client: Any) -> None:
        pass

    def prepare_request(
        self, request_input: list[dict[str, Any]], request: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Return request items and an optional stop reason; request options may be updated."""
        return request_input, None

    def observe_response(self, message: Any) -> None:
        pass

    def output_fields(self) -> dict[str, Any]:
        return {}

    def is_terminal_failure(self, assistant_text: str) -> bool:
        return False

    def execution_exception(self, error: Exception) -> dict[str, Any]:
        return {'success': False}


@agent_registry.register('cua_gpt_agent')
async def run_cua_gpt_agent(agent_input: CuaGptAgentInput) -> CuaGptAgentOutput:
    return await run_cua_gpt_with_adapter(agent_input, adapter=CuaGptAdapter())


async def run_cua_gpt_with_adapter(agent_input: CuaGptAgentInput, *, adapter: CuaGptAdapter) -> CuaGptAgentOutput:
    """Run the GPT computer-use loop against a nexus GUI runtime."""

    ctx = require_context()
    runtime = await ctx.get_runtime(agent_input.runtime_name)
    client = ctx.get_openai_client('main')
    adapter.initialize(agent_input, client)

    instruction = _instruction_text(agent_input.user_prompt)
    instructions = adapter.instructions(agent_input)

    # OpenAI Responses API native `computer` tool. Historical successful runs
    # (osworld/gpt56sol_cua_20260727_234044) use bare `{type: computer}` with
    # gpt-5.6-sol; the newer `computer_use_preview` schema is NOT supported by
    # gpt-5.6-sol via this gateway (returns "Tool 'computer_use_preview' is not
    # supported with gpt-5.6-sol").
    tools = [{'type': 'computer'}]
    previous_response_id: str | None = None
    pending_input: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    exit_status = 'max_iterations'
    final_answer: str | None = None
    # The FINAL turn's assistant content, for the post-loop `[INFEASIBLE]` check.
    # Strictly the last turn, not the last non-empty one: a model that muses
    # "this looks impossible" early and then works for 50 more turns must not be
    # classified by that stale sentence — which is the bug this replaces.
    last_turn_assistant_text = ''

    traj_dir = _traj_dir(agent_input)
    if traj_dir:
        logger.info('cua_gpt_agent: dumping trajectory to %s', traj_dir)
        _init_traj_dir(traj_dir)

    # Initial screenshot for turn 1.
    screenshot_b64 = await _screenshot_b64(runtime)
    _save_screenshot(traj_dir, 'obs_000.png', base64.b64decode(screenshot_b64))

    for iteration in range(1, agent_input.max_iterations + 1):
        # Build request input for this turn.
        if not previous_response_id:
            request_input: list[dict[str, Any]] = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'input_text', 'text': instruction},
                        {
                            'type': 'input_image',
                            'image_url': f'data:image/png;base64,{screenshot_b64}',
                            'detail': 'original',
                        },
                    ],
                }
            ]
        else:
            request_input = list(pending_input)
            if not request_input:
                request_input = [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'input_text', 'text': 'Continue from the latest screenshot.'},
                            {
                                'type': 'input_image',
                                'image_url': f'data:image/png;base64,{screenshot_b64}',
                                'detail': 'original',
                            },
                        ],
                    }
                ]

        # Field-by-field aligned with historical successful gpt-5.6-sol runs
        # (osworld/gpt56sol_cua_20260727_234044):
        #   tool_choice='auto', temperature=1.0, top_p=0.98,
        #   reasoning={context, effort, summary, mode}
        request: dict[str, Any] = {
            'instructions': instructions,
            'tools': tools,
            'tool_choice': 'auto',
            'parallel_tool_calls': False,
            'reasoning': {
                'context': 'all_turns',
                'effort': agent_input.reasoning_effort,
                'summary': 'concise',
                'mode': 'standard',
            },
            'temperature': 1.0,
            'top_p': 0.98,
            'truncation': 'auto',
        }
        if agent_input.max_output_tokens is not None:
            request['max_output_tokens'] = agent_input.max_output_tokens
        if previous_response_id:
            request['previous_response_id'] = previous_response_id
        request_input, stop_reason = adapter.prepare_request(request_input, request)
        if stop_reason is not None:
            final_answer = stop_reason
            break

        completion = await client.create_completion(
            messages=request_input,
            model=agent_input.model,
            **request,
        )
        previous_response_id = completion.id
        pending_input = []
        _append_jsonl(traj_dir, 'full_responses.jsonl', _response_dump(completion))

        message = completion.choices[0].message if completion.choices else None
        actions: list[dict[str, Any]] = []
        responses_text: list[str] = []
        assistant_texts: list[str] = []
        _step_summary = str(_get(message, 'reasoning_content') or '')
        _step_content = str(_get(message, 'content') or '')
        _step_raw_actions: list[Any] = []
        if _step_summary:
            responses_text.append(_step_summary)
        if _step_content:
            responses_text.append(_step_content)
            assistant_texts.append(_step_content)

        if _completion_is_incomplete(completion):
            turn_text = '\n'.join(text for text in responses_text if text)
            trajectory.append({
                'iteration': iteration,
                'response_id': previous_response_id,
                'text': turn_text,
                'num_actions': 0,
                'actions': [],
                'incomplete': True,
            })
            logger.warning('[iter %d] model response was incomplete; no actions executed', iteration)
            exit_status = 'incomplete'
            final_answer = _SAFE_INCOMPLETE_RESPONSE
            ctx.tool_state['os_world_last_action'] = {'action_type': 'FAIL'}
            break

        adapter.observe_response(message)

        tool_error: _ComputerToolError | None = None
        parsed_calls: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = []
        try:
            for tool_call in _get(message, 'tool_calls', []) or []:
                if _get(_get(tool_call, 'function'), 'name') != 'computer':
                    raise _ComputerToolError(_get(tool_call, 'id'))
                parsed_calls.append(_parse_computer_tool_call(tool_call))

            for call_index, (call_id, raw_actions, safety) in enumerate(parsed_calls):
                _step_raw_actions.extend(_model_dump(raw_action) for raw_action in raw_actions)
                for action_index, raw_action in enumerate(raw_actions):
                    info = _action_to_dict(raw_action)
                    try:
                        _validate_computer_action(info['type'], info['args'])
                    except (TypeError, ValueError) as error:
                        raise _ComputerToolError(call_id) from error
                    code = adapter.parse_action(info['type'], info['args'], agent_input)
                    if not code:
                        raise _ComputerToolError(call_id)
                    actions.append({
                        'code': code,
                        'action_type': info['type'],
                        'call_id': call_id,
                        'safety': safety,
                        'batch_last': action_index == len(raw_actions) - 1,
                        'final_call': call_index == len(parsed_calls) - 1,
                    })
        except _ComputerToolError as error:
            tool_error = error
            actions = []

        turn_text = '\n'.join(t for t in responses_text if t)
        assistant_text = '\n'.join(t for t in assistant_texts if t)
        last_turn_assistant_text = assistant_text
        trajectory_item: dict[str, Any] = {
            'iteration': iteration,
            'response_id': previous_response_id,
            'text': turn_text,
            'num_actions': len(actions),
            'actions': [action['action_type'] for action in actions],
            **adapter.output_fields(),
        }
        if tool_error is not None:
            trajectory_item['tool_error'] = str(tool_error)
        trajectory.append(trajectory_item)
        logger.info('[iter %d] text=%r actions=%s', iteration, turn_text[:200], [a['action_type'] for a in actions])

        if tool_error is not None:
            exit_status = 'tool_error'
            final_answer = str(tool_error)
            ctx.tool_state['os_world_last_action'] = {'action_type': 'FAIL'}
            break

        if not actions:
            if adapter.is_terminal_failure(assistant_text):
                exit_status = 'infeasible'
                final_answer = assistant_text
                ctx.tool_state['os_world_last_action'] = {'action_type': 'FAIL'}
                break
            if '[DONE]' in turn_text:
                exit_status = 'finished'
                final_answer = turn_text
                ctx.tool_state['os_world_last_action'] = {'action_type': 'DONE'}
                break
            # `[INFEASIBLE]` is a terminal declaration, so it is honored on the same
            # footing as `[DONE]`: only when the model ALSO stopped acting. A turn that
            # says the word but still drives the GUI is still working — let it work.
            if _INFEASIBLE_MARKER in turn_text.lower():
                exit_status = 'infeasible'
                final_answer = turn_text or '[INFEASIBLE]'
                ctx.tool_state['os_world_last_action'] = {'action_type': 'FAIL'}
                break
            if agent_input.get_user_input is not None:
                if not assistant_text.strip():
                    exit_status = 'failed'
                    final_answer = turn_text or 'No explicit user question was produced.'
                    ctx.tool_state['os_world_last_action'] = {'action_type': 'FAIL'}
                    break
                user_request = UserInputRequest(
                    hint=assistant_text,
                    metadata={
                        'osworld_task_id': agent_input.osworld_task_id,
                        'iteration': iteration,
                    },
                )
                answer = await agent_input.get_user_input(user_request)
                trajectory[-1].update({
                    'action': 'ASK_USER',
                    'question': assistant_text,
                    'user_answer': answer,
                })
                if traj_dir:
                    _append_jsonl(
                        traj_dir,
                        'trajectory.jsonl',
                        {
                            'step': iteration,
                            'action': 'ASK_USER',
                            'question': assistant_text,
                            'user_answer': answer,
                            'screenshot_file': None,
                        },
                    )
                if answer is None or not answer.strip():
                    exit_status = 'failed'
                    final_answer = turn_text
                    ctx.tool_state['os_world_last_action'] = {'action_type': 'FAIL'}
                    break
                pending_input = [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'input_text', 'text': f'User response: {answer}'},
                        ],
                    }
                ]
                ctx.tool_state['os_world_last_action'] = {'action_type': 'ASK_USER'}
                continue

            # No callback keeps the existing terminal behavior.
            exit_status = 'finished'
            final_answer = turn_text
            ctx.tool_state['os_world_last_action'] = {'action_type': 'DONE'}
            break

        # Execute actions, then feed the resulting screenshot back to the model.
        execution_results = [
            {
                'action_type': action['action_type'],
                'code': action['code'],
                'status': 'not_executed',
            }
            for action in actions
        ]
        action_failure = False
        for action_index, action in enumerate(actions):
            try:
                resp = await adapter.execute_code(
                    runtime,
                    action['code'],
                    mode=agent_input.pyautogui_execution_mode,
                    timeout=agent_input.pyautogui_timeout,
                )
            except Exception as exc:  # noqa: BLE001 - record only structured diagnostics, never raw command output
                resp = adapter.execution_exception(exc)
            if not resp['success']:
                execution_results[action_index].update({
                    'status': 'failed',
                    'error': _SAFE_ACTION_FAILURE,
                })
                diagnostics = adapter.action_diagnostics(resp)
                if diagnostics:
                    execution_results[action_index]['diagnostics'] = diagnostics
                action_failure = True
                ctx.tool_state['os_world_last_action'] = {'action_type': 'FAIL'}
                logger.warning('pyautogui action failed; stopping the current computer turn')
                break

            execution_results[action_index]['status'] = 'success'
            ctx.tool_state['os_world_last_action'] = {'action_type': action['action_type']}
            if agent_input.sleep_after_execution > 0:
                await asyncio.sleep(agent_input.sleep_after_execution)

            if action['batch_last']:
                if agent_input.step_pause > 0:
                    await asyncio.sleep(agent_input.step_pause)
                screenshot_b64 = await _screenshot_b64(runtime)
                if action['final_call']:
                    _save_screenshot(traj_dir, f'obs_{iteration:03d}.png', base64.b64decode(screenshot_b64))
                output_item: dict[str, Any] = {
                    'type': 'computer_call_output',
                    'call_id': action['call_id'],
                    'output': {
                        'type': 'computer_screenshot',
                        'image_url': f'data:image/png;base64,{screenshot_b64}',
                        'detail': 'original',
                    },
                }
                if action.get('safety'):
                    output_item['acknowledged_safety_checks'] = action['safety']
                pending_input.append(output_item)

        from datetime import datetime as _dt
        from datetime import timezone as _tz

        _append_jsonl(
            traj_dir,
            'trajectory.jsonl',
            {
                'step': iteration,
                'observation': f'screenshots/obs_{iteration - 1:03d}.png',
                'summary': _step_summary,
                'reasoning_content': '',
                'content': _step_content,
                'raw_actions': _step_raw_actions,
                'actions': execution_results,
                'tool_name': 'computer',
                'action_result': 'Failed' if action_failure else 'Success',
                **({'action_error': _SAFE_ACTION_FAILURE} if action_failure else {}),
                'result_observation': (
                    'screenshots/obs_final.png' if action_failure else f'screenshots/obs_{iteration:03d}.png'
                ),
                'timestamp': _dt.now(_tz.utc).isoformat(),
            },
        )

        trajectory_item['execution_results'] = execution_results
        trajectory_item['action_result'] = 'Failed' if action_failure else 'Success'
        if action_failure:
            trajectory_item['action_error'] = _SAFE_ACTION_FAILURE
            exit_status = 'action_error'
            final_answer = _SAFE_ACTION_FAILURE
            break

    # Post-loop classification, for the path where the loop ran to exhaustion. The
    # in-loop check above only fires when the model stopped issuing actions; a model
    # that declares `[INFEASIBLE]` on its very last budgeted turn while still acting
    # would otherwise land on `max_iterations` and read as "ran out of time".
    if exit_status == 'max_iterations' and _INFEASIBLE_MARKER in last_turn_assistant_text.lower():
        exit_status = 'infeasible'
        final_answer = last_turn_assistant_text
        ctx.tool_state['os_world_last_action'] = {'action_type': 'FAIL'}

    if traj_dir:
        final_b64 = await _screenshot_b64(runtime)
        _save_screenshot(traj_dir, 'obs_final.png', base64.b64decode(final_b64))

    return adapter.output_type(
        main_trajectory=trajectory,
        exit_status=exit_status,
        iterations=len(trajectory),
        final_answer=final_answer,
        **adapter.output_fields(),
    )


__all__ = ['CuaGptAgentInput', 'CuaGptAgentOutput', 'run_cua_gpt_agent']
