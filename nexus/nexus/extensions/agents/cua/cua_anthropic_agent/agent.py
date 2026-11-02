"""Anthropic Claude computer-use agent for Nexus.

Model profiles select the computer tool version, prompt additions and fixed-budget
or adaptive thinking. System/user content supports prompt-cache annotations and
bounded image history, with emergency truncation on request_too_large.

Injected Nexus clients own routing, credentials and retries. Direct HTTP transport
supports the configured Messages or Bedrock invoke request style.

Computer tool inputs execute through the runtime GUI API. Text input uses safe
ASCII typing or focus-aware clipboard paste; multiline input does not imply Enter.
The sandbox must permit the clipboard operations used by the selected input mode.

Termination uses explicit DONE/INFEASIBLE markers. Task-managed trajectory output
contains per-step screenshots, structured trajectory entries and raw responses.
"""

import asyncio
import base64
import io
import json as _json
import logging
import os
import re

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal, cast

import httpx

from PIL import Image
from pydantic import Field

from nexus.core import AgentRunInput, AgentRunOutput, ContextLengthExceededError, agent_registry, require_context
from nexus.core.agent import WithMainTrajectory, WithUserPrompt
from nexus.extensions.agents.cua.cua_support import (
    _append_jsonl,
    _b64,
    _get,
    _init_traj_dir,
    _instruction_text,
    _save_screenshot,
    _screenshot_bytes,
    _traj_dir,
)
from nexus.extensions.agents.cua.cua_text_input import text_input_code


logger = logging.getLogger(__name__)


API_RETRY_INTERVAL = 5
COMPUTER_TOOLSET_TYPE = 'computer_toolset_20260801'
TOOL_CONFIG_ANCHOR_NAME = 'nexus_tool_config_anchor'
NON_COMPUTER_TOOL_NAMES = frozenset({'bash', 'text_editor', 'str_replace_based_edit_tool', TOOL_CONFIG_ANCHOR_NAME})
_TRANSIENT_CLIENT_STATUS_CODES = {408, 409, 429}
_HTTP_STATUS_IN_MESSAGE = re.compile(r'\bHTTP\s+(\d{3})\b')


# ─── Constants (verbatim from utils.py) ────────────────────────────────────
COMPUTER_USE_BETA_FLAG_20250124 = 'computer-use-2025-01-24'
COMPUTER_USE_BETA_FLAG_20251124 = 'computer-use-2025-11-24'
PROMPT_CACHING_BETA_FLAG = 'prompt-caching-2024-07-31'
COMPUTER_USE_TYPE_20250124 = 'computer_20250124'
COMPUTER_USE_TYPE_20251124 = 'computer_20251124'


# ─── ClaudeRuntimeProfile (verbatim from utils.py) ────────────────────────
@dataclass(frozen=True)
class ClaudeRuntimeProfile:
    label: str
    computer_use_beta_flag: str
    computer_use_type: str
    thinking_mode: str
    default_max_tokens: int
    use_claude_47_prompt: bool = False


CLAUDE_45_PROFILE = ClaudeRuntimeProfile(
    label='Claude 4.5 legacy implementation',
    computer_use_beta_flag=COMPUTER_USE_BETA_FLAG_20250124,
    computer_use_type=COMPUTER_USE_TYPE_20250124,
    thinking_mode='legacy',
    default_max_tokens=3000,
)
CLAUDE_46_PROFILE = ClaudeRuntimeProfile(
    label='Claude 4.6 legacy implementation',
    computer_use_beta_flag=COMPUTER_USE_BETA_FLAG_20251124,
    computer_use_type=COMPUTER_USE_TYPE_20251124,
    thinking_mode='legacy',
    default_max_tokens=3000,
)
CLAUDE_OPUS_46_PROFILE = ClaudeRuntimeProfile(
    label='Claude Opus 4.6 adaptive-effort implementation',
    computer_use_beta_flag=COMPUTER_USE_BETA_FLAG_20251124,
    computer_use_type=COMPUTER_USE_TYPE_20251124,
    thinking_mode='adaptive',
    default_max_tokens=16000,
)
CLAUDE_SONNET_46_PROFILE = ClaudeRuntimeProfile(
    label='Claude Sonnet 4.6 adaptive-effort implementation',
    computer_use_beta_flag=COMPUTER_USE_BETA_FLAG_20251124,
    computer_use_type=COMPUTER_USE_TYPE_20251124,
    thinking_mode='adaptive',
    default_max_tokens=16000,
)
CLAUDE_47_PROFILE = ClaudeRuntimeProfile(
    label='Claude 4.7 adaptive-effort implementation',
    computer_use_beta_flag=COMPUTER_USE_BETA_FLAG_20251124,
    computer_use_type=COMPUTER_USE_TYPE_20251124,
    thinking_mode='adaptive',
    default_max_tokens=16000,
    use_claude_47_prompt=True,
)
CLAUDE_OPUS_48_PROFILE = ClaudeRuntimeProfile(
    label='Claude Opus 4.8 adaptive-effort implementation',
    computer_use_beta_flag=COMPUTER_USE_BETA_FLAG_20251124,
    computer_use_type=COMPUTER_USE_TYPE_20251124,
    thinking_mode='adaptive',
    default_max_tokens=16000,
    use_claude_47_prompt=True,
)
CLAUDE_OPUS_5_PROFILE = ClaudeRuntimeProfile(
    label='Claude Opus 5 adaptive-effort implementation',
    computer_use_beta_flag=COMPUTER_USE_BETA_FLAG_20251124,
    computer_use_type=COMPUTER_USE_TYPE_20251124,
    thinking_mode='adaptive',
    default_max_tokens=16000,
    use_claude_47_prompt=True,
)
CLAUDE_SONNET_5_PROFILE = ClaudeRuntimeProfile(
    label='Claude Sonnet 5 adaptive-effort implementation',
    computer_use_beta_flag=COMPUTER_USE_BETA_FLAG_20251124,
    computer_use_type=COMPUTER_USE_TYPE_20251124,
    thinking_mode='adaptive',
    default_max_tokens=16000,
    use_claude_47_prompt=True,
)
DEFAULT_CLAUDE_PROFILE = CLAUDE_46_PROFILE


def get_claude_runtime_profile(model_name: str) -> ClaudeRuntimeProfile:
    """Verbatim port of utils.py::get_claude_runtime_profile."""
    normalized = (model_name or '').lower().replace('_', '-')
    # Claude 5 family (opus-5 / sonnet-5). Match before 4.x because model id
    # 'claude-opus-5' has no minor version segment.
    if 'opus-5' in normalized or 'opus5' in normalized:
        return CLAUDE_OPUS_5_PROFILE
    if 'sonnet-5' in normalized or 'sonnet5' in normalized:
        return CLAUDE_SONNET_5_PROFILE
    if '4-8' in normalized or '4.8' in normalized:
        return CLAUDE_SONNET_46_PROFILE if 'sonnet' in normalized else CLAUDE_OPUS_48_PROFILE
    if '4-7' in normalized or '4.7' in normalized:
        return CLAUDE_47_PROFILE
    if '4-6' in normalized or '4.6' in normalized:
        if 'sonnet' in normalized:
            return CLAUDE_SONNET_46_PROFILE
        if 'opus' in normalized:
            return CLAUDE_OPUS_46_PROFILE
        return CLAUDE_46_PROFILE
    if '4-5' in normalized or '4.5' in normalized:
        return CLAUDE_45_PROFILE
    return DEFAULT_CLAUDE_PROFILE


# ─── Date helper (verbatim from utils.py) ─────────────────────────────────
def _format_prompt_date(value: Any = None) -> str:
    if value is None:
        return datetime.today().strftime('%A, %B %d, %Y')
    if isinstance(value, datetime):
        return value.strftime('%A, %B %d, %Y')
    if isinstance(value, date):
        return value.strftime('%A, %B %d, %Y')
    if isinstance(value, str):
        return value
    return str(value)


# ─── Prompts (verbatim from utils.py) ─────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using x86_64 architecture with internet access.
* You can feel free to install Ubuntu applications with your bash tool. Use curl instead of wget.
* To open browser, please just click on the Chrome icon.  Note, Chrome is what is installed on your system.
* Using bash tool you can start GUI applications, but you need to set export DISPLAY=:1 and use a subshell. For example "(DISPLAY=:1 xterm &)". GUI apps run with bash tool will appear within your desktop environment, but they may take some time to appear. Take a screenshot to confirm it did.
* When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_editor or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* The current date is {CURRENT_DATE}.
* Home directory of this Ubuntu system is '/home/user'.
* If you need a password for sudo, the password of the computer is 'osworld-public-evaluation'.
* When you asking user questions, do not perform any action.
* When you think you have completed the task, output exactly "[DONE]" (including the square brackets) anywhere in your response to trigger the done action.
The system will automatically detect this pattern and terminate the task appropriately.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your StrReplaceEditTool.
</IMPORTANT>"""


SYSTEM_PROMPT_WINDOWS_TEMPLATE = """<SYSTEM_CAPABILITY>
* You are utilising a Windows virtual machine using x86_64 architecture with internet access.
* To open browser, please just click on the Chrome icon.  Note, Chrome is what is installed on your system.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* The current date is {CURRENT_DATE}.
* Home directory of this Windows system is 'C:\\Users\\user'.
* When you want to open some applications on Windows, please use Double Click on it instead of clicking once.
* If you need a password for sudo, The password of the computer is 'osworld-public-evaluation'.
* When you asking user questions, do not perform any action.
</SYSTEM_CAPABILITY>"""


def build_system_prompt(
    platform: str,
    task_current_date: Any = None,
    system_prompt_suffix: str = '',
) -> str:
    template = SYSTEM_PROMPT_WINDOWS_TEMPLATE if platform == 'Windows' else SYSTEM_PROMPT_TEMPLATE
    system_prompt = template.replace('{CURRENT_DATE}', _format_prompt_date(task_current_date))
    return f'{system_prompt}{" " + system_prompt_suffix if system_prompt_suffix else ""}'


CLAUDE_47_PROMPT_ADDITIONS = """<CLAUDE_4_7_NOTES>
* DO NOT use LibreOffice macros or GIMP Script-Fu to complete tasks. Always use the GUI interface directly with mouse and keyboard actions. Macros and scripting cause reliability issues and task failures.
* For GIMP tasks, do NOT save or export files unless the instruction explicitly asks you to. Note that existing tasks that require file output will ask you to "export", not "save". Most GIMP tasks are evaluated automatically without requiring you to save.
* Before starting a task, consider whether it is achievable with the designated application's native GUI features. If the app fundamentally lacks the requested capability, declare it infeasible instead of using CLI tools, Python scripts, or other applications as workarounds.
* After completing a task, verify the visible or functional result. If your actions had no real effect, reconsider whether the task is feasible.
</CLAUDE_4_7_NOTES>"""


BATCHED_ACTION_PROMPT = """CRITICAL - ALWAYS BATCH ACTIONS: You MUST combine as many actions as possible into a SINGLE tool call.
Each tool call has significant latency overhead, so batching is essential for efficiency.
Do NOT make separate tool calls for actions you can predict the outcome of.

You do NOT need a screenshot after every action. Only take a screenshot when you genuinely
need to observe the result to decide what to do next. Many action sequences are predictable.

Example of GOOD batching (1 tool call instead of 6):
  actions=[
    {"action": "left_click", "coordinate": [300, 200]},
    {"action": "wait", "duration": 1},
    {"action": "triple_click", "coordinate": [400, 150]},
    {"action": "type", "text": "new value"},
    {"action": "key", "text": "Tab"},
    {"action": "type", "text": "another value"}
  ]

WRONG approach: making a separate tool call for each individual click, type, or keypress.
This wastes round trips and slows down task completion dramatically.

Think of each tool call as expensive. Ask yourself: "Can I predict what will happen?"
If yes, batch the action with subsequent actions. Only screenshot when the next step
depends on seeing the screen.

For form filling, menu navigation, and keyboard shortcuts, you almost never need a
screenshot between individual actions - batch them all together."""


# ─── _inject_prompt_caching (verbatim from utils.py) ──────────────────────
def _inject_prompt_caching(messages: list[dict[str, Any]]) -> None:
    """Set cache breakpoints for the 3 most recent turns; one breakpoint reserved
    for tools/system.
    """
    breakpoints_remaining = 2
    messages_processed = 0
    for message in reversed(messages):
        if message.get('role') != 'user' or not isinstance(message.get('content'), list):
            continue
        content = message['content']
        messages_processed += 1
        if breakpoints_remaining >= len(content):
            breakpoints_remaining -= len(content)
            content[-1]['cache_control'] = {'type': 'ephemeral'}  # type: ignore
        else:
            is_first_message = messages_processed == len([m for m in messages if m.get('role') == 'user'])
            if not is_first_message and isinstance(content[-1], dict):
                content[-1].pop('cache_control', None)


# ─── _maybe_filter_to_n_most_recent_images (verbatim from utils.py) ───────
def _maybe_filter_to_n_most_recent_images(
    messages: list[dict[str, Any]],
    images_to_keep: int | None,
    min_removal_threshold: int,
) -> None:
    if images_to_keep is None:
        return

    tool_result_blocks = [
        item
        for message in messages
        for item in (message['content'] if isinstance(message.get('content'), list) else [])
        if isinstance(item, dict) and item.get('type') == 'tool_result'
    ]

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in (tool_result.get('content') or [])
        if isinstance(content, dict) and content.get('type') == 'image'
    )

    images_to_remove = total_images - images_to_keep
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get('content'), list):
            new_content = []
            for content in tool_result.get('content', []):
                if isinstance(content, dict) and content.get('type') == 'image' and images_to_remove > 0:
                    images_to_remove -= 1
                    continue
                new_content.append(content)
            tool_result['content'] = new_content


# ─── Input / Output ───────────────────────────────────────────────────────
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
            return ok, None if ok else f'run_command status={status_value} rc={return_code}'
        response = await runtime.gui_pyautogui(exec_code, timeout=timeout)
        ok = bool(_get(response, 'success', False))
        return ok, _get(response, 'error')
    except Exception as exc:  # noqa: BLE001
        logger.exception('pyautogui execution failed')
        return False, str(exc)


class CuaAnthropicAgentInput(AgentRunInput, WithUserPrompt):
    max_iterations: int = 50
    model: str | None = None
    max_tokens: int = 4096
    only_n_most_recent_images: int = 10
    system_prompt_suffix: str = ''
    task_current_date: str | None = None
    no_thinking: bool = False
    effort: str = 'high'
    use_isp: bool = False
    # Sampling params. Both are None by default => omit from request
    # (Anthropic default).
    temperature: float | None = None
    top_p: float | None = None
    runtime_name: str = 'main'
    # auto uses the named client when configured, otherwise direct HTTP.
    # Task-domain env overrides belong to the generator, never this runtime agent.
    transport: Literal['auto', 'client', 'httpx'] = 'auto'
    client_name: str = 'main'
    api_style: Literal['messages', 'bedrock_invoke'] = 'messages'
    bedrock_anthropic_version: str = 'bedrock-2023-05-31'
    platform_name: str = 'ubuntu'
    client_password: str = 'password'
    # Actual sandbox screen size. Screenshots are resized to
    # (model_display_width, model_display_height) before being sent to the
    # model; model-emitted coordinates are then rescaled back by
    # resize_factor = (screen_width/model_display_width, screen_height/model_display_height).
    screen_width: int = 1920
    screen_height: int = 1080
    # Fixed at 1280x720 by the official OSWorld anthropic agent (Anthropic API's
    # long-edge <=1568 px limit + ~1600-token budget). Do not change unless the
    # gateway/model policy changes.
    model_display_width: int = 1280
    model_display_height: int = 720
    sleep_after_execution: float = 0.5
    step_pause: float = 1.0
    observation_type: str = 'screenshot'
    action_space: str = 'claude_computer_use'
    # Pyautogui execution channel:
    #   - 'gui_pyautogui': call runtime.gui_pyautogui(code) directly. Works
    #     on gongfeng, but AGS nexus interface REJECTS this executor.
    #   - 'run_command': wrap the pyautogui snippet as a python -c string and
    #     execute it via runtime.run_command(). Works on both gongfeng and
    #     AGS nexus interface. Set this whenever you're on AGS.
    pyautogui_execution_mode: Literal['gui_pyautogui', 'run_command'] = 'gui_pyautogui'
    pyautogui_timeout: float = 30.0
    osworld_task_id: str = ''
    osworld_domain: str = ''
    # Internal per-attempt directory supplied by the task's artifact collector.
    trajectory_dir: str = ''
    # Retries for one gateway request (429 / 5xx / request_too_large truncation).
    api_retry_times: int = Field(default=5, ge=1)


class CuaAnthropicAgentOutput(AgentRunOutput, WithMainTrajectory):
    exit_status: str
    iterations: int
    final_answer: str | None = None


def _resolve_llm_config(model_override: str | None) -> tuple[str, str, str]:
    api_key = (
        os.getenv('GPT_CUA_API_KEY')
        or os.getenv('OPENAI_API_KEY')
        or (
            f'{os.getenv("MODEL_SERVICE_API_APP_ID", "")}:{os.getenv("MODEL_SERVICE_API_APP_KEY", "")}'
            if os.getenv('MODEL_SERVICE_API_APP_ID')
            else ''
        )
    )
    if not api_key:
        raise RuntimeError('cua_anthropic_agent requires an API key.')
    base_url = (
        os.getenv('GPT_CUA_BASE_URL')
        or os.getenv('MODEL_SERVICE_API_BASE_URL')
        or 'http://trpc-gpt-eval.production.polaris:8080'
    )
    model = model_override or os.getenv('GPT_CUA_MODEL') or os.getenv('NEXUS_MODEL') or 'claude-sonnet-4-5-20250929'
    if '?' not in api_key:
        api_key = f'{api_key}?provider=anthropic&model={model}&timeout=120'
    return api_key, base_url, model


def _dispatched_action_names(tool_call: dict[str, Any]) -> list[str]:
    """Describe dispatch without mistaking action-less shell tools for batches."""
    name = tool_call.get('name') or 'computer'
    if name != 'computer':
        return [str(name)]
    tool_input = tool_call.get('input') or {}
    if not isinstance(tool_input, dict):
        return ['invalid_computer_input']
    children = tool_input.get('actions')
    if children is not None:
        if not isinstance(children, list):
            return ['invalid_computer_batch']
        if not children:
            return ['empty_computer_batch']
        return [label for child in children for label in _dispatched_action_names({'name': 'computer', 'input': child})]
    action = tool_input.get('action')
    return [action if isinstance(action, str) and action else 'unknown_computer_action']


# ─── parse_actions_from_tool_call ─────────────────


def _parse_actions_from_tool_call(
    tool_call: dict[str, Any],
    resize_factor: tuple[float, float] | None = None,
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
            result += _parse_actions_from_tool_call({'input': batched_action}, resize_factor)
        return result

    action = function_args.get('action')
    if not action:
        return result
    action_conversion = {'left click': 'click', 'right click': 'right_click'}
    action = action_conversion.get(action, action)

    text = function_args.get('text')
    coordinate = function_args.get('coordinate')
    start_coordinate = function_args.get('start_coordinate')
    scroll_direction = function_args.get('scroll_direction')
    scroll_amount = int(function_args.get('scroll_amount') or 0)
    duration = function_args.get('duration')
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
            key = key.strip().lower()
            result += f"pyautogui.keyDown('{key}')\n"
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
            result += f'pyautogui.dragTo({x}, {y}, duration={duration or 0.5})\n'
    elif action in ('key', 'type'):
        if text is None:
            raise ValueError(f'text is required for {action}')
        if not isinstance(text, str):
            raise ValueError(f'{text} must be a string')
        if action == 'key':
            key_conversion = {
                'page_down': 'pagedown',
                'page_up': 'pageup',
                'super_l': 'win',
                'super': 'command',
                'escape': 'esc',
            }
            keys = text.split('+')
            for _ in range(repeat):
                for key in keys:
                    key = key.strip().lower()
                    key = key_conversion.get(key, key)
                    result += f"pyautogui.keyDown('{key}')\n"
                for key in reversed(keys):
                    key = key.strip().lower()
                    key = key_conversion.get(key, key)
                    result += f"pyautogui.keyUp('{key}')\n"
        elif action == 'type':
            result += text_input_code(text)
    elif action == 'scroll':
        if text is not None:
            result += f"pyautogui.keyDown('{text.lower()}')\n"
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
            result += f"pyautogui.keyUp('{text.lower()}')\n"
    elif action in ('left_click', 'right_click', 'double_click', 'middle_click', 'left_press', 'triple_click'):
        if text:
            keys = text.split('+')
            for key in keys:
                key = key.strip().lower()
                result += f"pyautogui.keyDown('{key}')\n"
        if coordinate is not None:
            x, y = coordinate
            if action == 'left_click':
                result += f'pyautogui.click({x}, {y})\n'
            elif action == 'right_click':
                result += f'pyautogui.rightClick({x}, {y})\n'
            elif action == 'double_click':
                result += f'pyautogui.doubleClick({x}, {y})\n'
            elif action == 'middle_click':
                result += f'pyautogui.middleClick({x}, {y})\n'
            elif action == 'left_press':
                result += f'pyautogui.mouseDown({x}, {y})\n'
                result += 'time.sleep(1)\n'
                result += f'pyautogui.mouseUp({x}, {y})\n'
            elif action == 'triple_click':
                result += f'pyautogui.tripleClick({x}, {y})\n'
        else:
            if action == 'left_click':
                result += 'pyautogui.click()\n'
            elif action == 'right_click':
                result += 'pyautogui.rightClick()\n'
            elif action == 'double_click':
                result += 'pyautogui.doubleClick()\n'
            elif action == 'middle_click':
                result += 'pyautogui.middleClick()\n'
            elif action == 'left_press':
                result += 'pyautogui.mouseDown()\n'
                result += 'time.sleep(1)\n'
                result += 'pyautogui.mouseUp()\n'
            elif action == 'triple_click':
                result += 'pyautogui.tripleClick()\n'
        if text:
            keys = text.split('+')
            for key in reversed(keys):
                key = key.strip().lower()
                result += f"pyautogui.keyUp('{key}')\n"
    elif action == 'wait':
        result += f'pyautogui.sleep({duration or 0.5})\n'
    elif action == 'fail':
        result += 'FAIL'
    elif action == 'done':
        result += 'DONE'
    elif action == 'screenshot':
        result += 'pyautogui.sleep(0.1)\n'
    elif action in ('cursor_position',):
        # Not directly executable — treat as a tiny sleep so pipeline continues.
        result += 'pyautogui.sleep(0.05)\n'
    elif action == 'zoom':
        # zoom action not directly translatable to pyautogui — no-op sleep.
        result += 'pyautogui.sleep(0.1)\n'
    else:
        raise ValueError(f'Invalid action: {action}')

    return result


def _extract_raw_response_string(content_blocks: list[dict[str, Any]]) -> str:
    """Verbatim port of _extract_raw_response_string but for dict blocks."""
    out = ''
    for block in content_blocks:
        t = block.get('type')
        if t == 'text' and block.get('text'):
            out += f'[TEXT] {block["text"]}\n'
        elif t == 'thinking' and block.get('thinking'):
            out += f'[THINKING] {block["thinking"]}\n'
        elif t == 'tool_use':
            out += f'[TOOL_USE] {block.get("name", "?")}: {block.get("input")}\n'
        else:
            out += f'[OTHER] {block}\n'
    return out.strip()


def _is_request_too_large(error: str | BaseException) -> bool:
    # Injected clients redact their public message and preserve the provider
    # error as __cause__. Classify it without logging the raw request/credential.
    if isinstance(error, ContextLengthExceededError):
        return True
    if isinstance(error, BaseException):
        seen = set()
        while id(error) not in seen:
            seen.add(id(error))
            if _is_request_too_large(str(error)):
                return True
            if error.__cause__ is None:
                return False
            error = error.__cause__
        return False
    low = (error or '').lower()
    return (
        'request too large' in low
        or 'request_too_large' in low
        or 'prompt is too long' in low
        or 'prompt_too_long' in low
    )


def _screen_geometry_prompt(width: int, height: int) -> str:
    # Bedrock toolsets reject display_* fields; without a prompt the model guesses
    # the coordinate space and clicks are rescaled into the wrong positions.
    return (
        f'The screenshots you receive are {width}x{height} pixels; '
        f'give every coordinate in that same {width}x{height} space.'
    )


def _client_tools(profile: ClaudeRuntimeProfile, width: int, height: int) -> list[dict[str, Any]]:
    """The LiteLLM/Converse shape, including its toolConfig anchor.

    Computer tools must be function-wrapped; bash must not be. A regular dummy
    function forces Converse to emit toolConfig on subsequent tool-result turns.
    Do not advertise enable_zoom: zoom-region execution is not supported.
    """
    return [
        {
            'type': profile.computer_use_type,
            'function': {
                'name': 'computer',
                'parameters': {'display_width_px': width, 'display_height_px': height, 'display_number': 1},
            },
        },
        {'type': 'bash_20250124', 'name': 'bash'},
        {
            'type': 'function',
            'function': {
                'name': TOOL_CONFIG_ANCHOR_NAME,
                'description': 'Internal no-op placeholder. Never call this tool.',
                'parameters': {'type': 'object', 'properties': {}, 'required': []},
            },
        },
    ]


def _native_tools(profile: ClaudeRuntimeProfile, width: int, height: int) -> list[dict[str, Any]]:
    return [
        {
            'name': 'computer',
            'type': profile.computer_use_type,
            'display_width_px': width,
            'display_height_px': height,
            'display_number': 1,
        },
        {'name': 'bash', 'type': 'bash_20250124'},
    ]


def _bedrock_toolset_tools() -> list[dict[str, Any]]:
    # The dated toolset fixes its member names. Bedrock rejects name/display_*
    # fields here, and needs no Converse toolConfig anchor on this native wire.
    return [{'type': COMPUTER_TOOLSET_TYPE}, {'name': 'bash', 'type': 'bash_20250124'}]


def _transport_wire_style(client: Any, httpx_style: str = 'messages') -> str:
    """Select the schema by the wire, not by the presence of an injected client."""
    if client is None:
        return httpx_style
    return str(getattr(client, 'api_style', '') or '').strip().lower()


def _to_chat_content(blocks: list[dict[str, Any]] | str) -> list[dict[str, Any]]:
    if isinstance(blocks, str):
        return [{'type': 'text', 'text': blocks}]
    parts = []
    for block in blocks:
        if block.get('type') == 'text':
            parts.append({'type': 'text', 'text': block.get('text') or ''})
        elif block.get('type') == 'image':
            source = block.get('source') or {}
            media = source.get('media_type') or 'image/png'
            parts.append({
                'type': 'image_url',
                'image_url': {'url': f'data:{media};base64,{source.get("data", "")}'},
            })
    return parts


def _to_chat_messages(messages: list[dict[str, Any]], system_text: str) -> list[dict[str, Any]]:
    """Convert content blocks to Chat messages; caching is owned by the client."""
    out: list[dict[str, Any]] = [{'role': 'system', 'content': system_text}]
    for message in messages:
        role, blocks = message.get('role'), message.get('content') or []
        if isinstance(blocks, str):
            out.append({'role': role, 'content': blocks})
            continue
        if role == 'assistant':
            assistant: dict[str, Any] = {
                'role': 'assistant',
                'content': ''.join(b.get('text', '') for b in blocks if b.get('type') == 'text'),
            }
            tools = [
                {
                    'id': b.get('id'),
                    'type': 'function',
                    'function': {
                        'name': b.get('name') or 'computer',
                        'arguments': _json.dumps(b.get('input') or {}, ensure_ascii=False),
                    },
                }
                for b in blocks
                if b.get('type') == 'tool_use'
            ]
            thinking = [b for b in blocks if b.get('type') in {'thinking', 'redacted_thinking'}]
            if tools:
                assistant['tool_calls'] = tools
            if thinking:
                assistant['thinking_blocks'] = thinking
            out.append(assistant)
            continue
        pending = []
        for block in blocks:
            if block.get('type') == 'tool_result':
                out.append({
                    'role': 'tool',
                    'tool_call_id': block.get('tool_use_id'),
                    'content': _to_chat_content(block.get('content') or []),
                })
            else:
                pending.extend(_to_chat_content([block]))
        if pending:
            out.append({'role': 'user', 'content': pending})
    return out


def _chat_response_to_anthropic(response: Any) -> dict[str, Any]:
    choice = response.choices[0]
    message = choice.message
    content = [
        b.model_dump() if callable(getattr(b, 'model_dump', None)) else dict(b)
        for b in getattr(message, 'thinking_blocks', None) or []
    ]
    if message.content:
        content.append({'type': 'text', 'text': message.content})
    for tool in message.tool_calls or []:
        raw = tool.function.arguments
        if isinstance(raw, str):
            try:
                args = _json.loads(raw or '{}')
            except ValueError:
                args = {'raw': raw}
        else:
            args = raw or {}
        content.append({
            'type': 'tool_use',
            'id': tool.id,
            'name': tool.function.name,
            'input': args if isinstance(args, dict) else {'raw': args},
        })
    finish = choice.finish_reason
    return {
        'content': content,
        'stop_reason': {'tool_calls': 'tool_use', 'stop': 'end_turn', 'length': 'max_tokens'}.get(finish, finish),
    }


async def _dispatch_model_call(
    *,
    client: Any,
    client_kwargs: dict[str, Any] | None,
    http: Any,
    endpoint: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    messages: list[dict[str, Any]],
    system_text: str,
) -> dict[str, Any]:
    # Read messages fresh: emergency image pruning mutates the same history and
    # must affect retries on both transports, not only the HTTP payload.
    if client_kwargs is not None:
        response = await client.create_completion(
            messages=_to_chat_messages(messages, system_text),
            **client_kwargs,
        )
        return _chat_response_to_anthropic(response)
    payload['messages'] = messages
    response = await http.post(endpoint, headers=headers, content=_json.dumps(payload))
    if response.status_code != 200:
        raise RuntimeError(f'HTTP {response.status_code}: {response.text[:400]}')
    result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError('Unexpected Anthropic response shape')
    if result.get('type') == 'error' or result.get('error'):
        raise RuntimeError(str(result.get('error') or result))
    return result


def _error_status(error: BaseException) -> int | None:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for status in (
            getattr(current, 'status', None),
            getattr(current, 'status_code', None),
            getattr(getattr(current, 'response', None), 'status_code', None),
        ):
            if isinstance(status, int):
                return status
        match = _HTTP_STATUS_IN_MESSAGE.search(str(current))
        if match:
            return int(match.group(1))
        current = current.__cause__
    return None


def _is_hard_client_error(error: BaseException) -> bool:
    status = _error_status(error)
    return status is not None and 400 <= status < 500 and status not in _TRANSIENT_CLIENT_STATUS_CODES


# ─── Agent ─────────────────────────────────────────────────────────────────
class CuaAnthropicAdapter:
    """Per-run hooks preserving shared behavior unless explicitly overridden.

    parse_action returns code; execute_action returns (success, error).
    allow_actions checks the whole action group before any action is executed.
    Diagnostic/output hooks add fields; mutable counters belong to this instance.
    """

    output_type = CuaAnthropicAgentOutput

    def profile(self, model: str) -> ClaudeRuntimeProfile:
        return get_claude_runtime_profile(model)

    def prompt(self, agent_input: CuaAnthropicAgentInput, suffix: str) -> str:
        return build_system_prompt(agent_input.platform_name, agent_input.task_current_date, suffix)

    def parse_action(
        self, block: dict[str, Any], resize_factor: tuple[float, float], agent_input: CuaAnthropicAgentInput
    ) -> str:
        return _parse_actions_from_tool_call(block, resize_factor)

    async def execute_action(
        self, runtime: Any, action: dict[str, Any], *, execution_mode: str, timeout: float
    ) -> tuple[bool, str | None]:
        return await _execute_computer_code(
            runtime, action.get('command') or '', execution_mode=execution_mode, timeout=timeout
        )

    def termination_text(self, raw_response: str | None, reasonings: list[str], actions: list[dict[str, Any]]) -> str:
        return raw_response or ''

    async def screen_size(self, runtime: Any, agent_input: CuaAnthropicAgentInput) -> tuple[int, int]:
        return agent_input.screen_width, agent_input.screen_height

    def configure_tools(
        self, wire_style: str, profile: ClaudeRuntimeProfile, agent_input: CuaAnthropicAgentInput
    ) -> tuple[list[dict[str, Any]], bool]:
        use_toolset = wire_style == 'bedrock_invoke'
        if use_toolset:
            tools = _bedrock_toolset_tools()
        elif wire_style == 'messages':
            tools = _native_tools(profile, agent_input.model_display_width, agent_input.model_display_height)
        else:
            tools = _client_tools(profile, agent_input.model_display_width, agent_input.model_display_height)
        return tools, use_toolset

    def trim_images(self, messages: list[dict[str, Any]], agent_input: CuaAnthropicAgentInput) -> None:
        pass

    def action_fields(self, parse_error: str | None) -> dict[str, Any]:
        return {}

    def allow_actions(self, actions: list[dict[str, Any]], agent_input: CuaAnthropicAgentInput) -> bool:
        return True

    def validate_tool(self, tool_name: str, agent_input: CuaAnthropicAgentInput) -> None:
        pass

    def record_execution(self, action: dict[str, Any], success: bool, error: str | None) -> None:
        pass

    def result_fields(self, success: bool, error: str | None) -> dict[str, Any]:
        return {'action_result': 'Success' if success else str(error or 'failed')}

    def image_added(self) -> None:
        pass

    def output_fields(self) -> dict[str, Any]:
        return {}


@agent_registry.register('cua_anthropic_agent')
async def run_cua_anthropic_agent(agent_input: CuaAnthropicAgentInput) -> CuaAnthropicAgentOutput:
    return await run_cua_anthropic_with_adapter(agent_input, adapter=CuaAnthropicAdapter())


async def run_cua_anthropic_with_adapter(
    agent_input: CuaAnthropicAgentInput, *, adapter: CuaAnthropicAdapter
) -> CuaAnthropicAgentOutput:
    ctx = require_context()
    runtime = await ctx.get_runtime(agent_input.runtime_name)

    client = None
    if agent_input.transport in ('auto', 'client'):
        try:
            client = ctx.get_openai_client(agent_input.client_name)
        except KeyError:
            if agent_input.transport == 'client':
                raise RuntimeError(
                    f'transport=client requires an injected openai client named {agent_input.client_name!r}'
                ) from None
    if client is not None:
        # The client owns routing and credentials. No host env is consulted here.
        api_key = ''
        model = agent_input.model or client.get_effective_request_args().get('model') or ''
        endpoint = f'openai_client:{agent_input.client_name}'
    else:
        api_key, base_url, model = _resolve_llm_config(agent_input.model)
        if agent_input.osworld_task_id:
            import hashlib

            cache_id = hashlib.md5(agent_input.osworld_task_id.encode()).hexdigest()
            sep = '&' if '?' in api_key else '?'
            api_key = f'{api_key}{sep}cache_task_id={cache_id}'
        # Avoid /v1/v1/messages when the base URL already includes /v1.
        _base = base_url.rstrip('/')
        if _base.endswith('/v1'):
            _base = _base[: -len('/v1')]
        endpoint = (
            f'{_base}/model/{model}/invoke' if agent_input.api_style == 'bedrock_invoke' else f'{_base}/v1/messages'
        )
    profile = adapter.profile(model)
    logger.info('cua_anthropic_agent model=%s profile=%s endpoint=%s', model, profile.label, endpoint)

    instruction = _instruction_text(agent_input.user_prompt)

    traj_dir = _traj_dir(agent_input)
    if traj_dir:
        logger.info('cua_anthropic_agent: dumping trajectory to %s', traj_dir)
        _init_traj_dir(traj_dir)

    # Capture initial screenshot.
    initial_bytes = await _screenshot_bytes(runtime)
    _save_screenshot(traj_dir, 'obs_000.png', initial_bytes)

    # OSWorld resizes to model_display_(width|height) — 1280x720 by default —
    # for the model input. Sandbox is at screen_(width|height) = 1920x1080.
    def _resize_for_model(png_bytes: bytes) -> bytes:
        img = Image.open(io.BytesIO(png_bytes))
        resized = img.resize(
            (agent_input.model_display_width, agent_input.model_display_height),
            Image.Resampling.LANCZOS,
        )
        buf = io.BytesIO()
        resized.save(buf, format='PNG')
        return buf.getvalue()

    screen_width, screen_height = await adapter.screen_size(runtime, agent_input)
    resize_factor = (
        screen_width / agent_input.model_display_width,
        screen_height / agent_input.model_display_height,
    )
    initial_resized = _resize_for_model(initial_bytes)

    # Message state (mirrors self.messages in official).
    messages: list[dict[str, Any]] = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'source': {'type': 'base64', 'media_type': 'image/png', 'data': _b64(initial_resized)},
                },
                {'type': 'text', 'text': instruction},
            ],
        }
    ]

    request_too_large_seen = False
    trajectory: list[dict[str, Any]] = []
    exit_status = 'max_iterations'
    final_answer: str | None = None
    latest_screenshot_bytes = initial_bytes  # for final dump

    wire_style = _transport_wire_style(client, agent_input.api_style)
    tools, use_toolset = adapter.configure_tools(wire_style, profile, agent_input)
    prompt_suffix = agent_input.system_prompt_suffix
    if use_toolset:
        geometry = _screen_geometry_prompt(agent_input.model_display_width, agent_input.model_display_height)
        prompt_suffix = f'{prompt_suffix} {geometry}' if prompt_suffix else geometry

    async with httpx.AsyncClient(timeout=180.0) as http:
        for iteration in range(1, agent_input.max_iterations + 1):
            # Build system prompt exactly as official predict() does.
            system_text = adapter.prompt(agent_input, prompt_suffix)
            if profile.use_claude_47_prompt:
                system_text += f'\n\n{CLAUDE_47_PROMPT_ADDITIONS}'
                system_text += (
                    f'\n\n<STEP_LIMIT>You have a maximum of {agent_input.max_iterations} steps '
                    f'to complete this task. Use your steps wisely.</STEP_LIMIT>'
                )
                system_text += f'\n\n{BATCHED_ACTION_PROMPT}'

            system_block: dict[str, Any] = {'type': 'text', 'text': system_text}

            enable_prompt_caching = True  # gateway supports Anthropic proto — official ANTHROPIC path.

            betas = [profile.computer_use_beta_flag]
            if agent_input.use_isp and profile.thinking_mode == 'legacy':
                betas.append('interleaved-thinking-2025-05-14')

            image_truncation_threshold = 10
            if enable_prompt_caching:
                betas.append(PROMPT_CACHING_BETA_FLAG)
                _inject_prompt_caching(messages)
                image_truncation_threshold = 10 if request_too_large_seen else 20
                system_block['cache_control'] = {'type': 'ephemeral'}

            _maybe_filter_to_n_most_recent_images(
                messages,
                agent_input.only_n_most_recent_images,
                min_removal_threshold=image_truncation_threshold,
            )

            adapter.trim_images(messages, agent_input)

            # Configure thinking / max_tokens per profile.
            if profile.thinking_mode == 'adaptive':
                if agent_input.no_thinking:
                    extra_body: dict[str, Any] = {}
                    actual_max_tokens = agent_input.max_tokens
                else:
                    extra_body = {
                        'thinking': {'type': 'adaptive'},
                        'output_config': {'effort': agent_input.effort},
                    }
                    actual_max_tokens = max(agent_input.max_tokens, profile.default_max_tokens)
            else:
                # Direct HTTP mode.
                if agent_input.no_thinking:
                    extra_body = {}
                    actual_max_tokens = agent_input.max_tokens
                else:
                    budget_tokens = 2048
                    if agent_input.max_tokens <= budget_tokens:
                        actual_max_tokens = budget_tokens + 500
                    else:
                        actual_max_tokens = agent_input.max_tokens
                    extra_body = {'thinking': {'type': 'enabled', 'budget_tokens': budget_tokens}}

            # Build request payload.
            payload = {
                'model': model,
                'max_tokens': actual_max_tokens,
                'system': [system_block],
                'messages': messages,
                'tools': tools,
                **extra_body,
            }
            # Optional sampling params (only include when explicitly set —
            # matches official main.py:_get_sampling_params behavior).
            if agent_input.temperature is not None:
                payload['temperature'] = agent_input.temperature
            if agent_input.top_p is not None:
                payload['top_p'] = agent_input.top_p
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}',
                'anthropic-beta': ','.join(betas),
            }
            client_kwargs = None
            if client is not None:
                client_kwargs = {
                    'max_tokens': actual_max_tokens,
                    'tools': tools,
                    'extra_headers': {'anthropic-beta': ','.join(betas)},
                    **extra_body,
                }
                if model:
                    client_kwargs['model'] = model
                for name in ('temperature', 'top_p'):
                    value = getattr(agent_input, name)
                    if value is not None:
                        client_kwargs[name] = value
            elif wire_style == 'bedrock_invoke':
                payload.pop('model')
                payload['anthropic_version'] = agent_input.bedrock_anthropic_version
                headers.pop('anthropic-beta')

            async def _call(
                _kwargs: dict[str, Any] | None = client_kwargs,
                _headers: dict[str, str] = headers,
                _payload: dict[str, Any] = payload,
                _system: str = system_text,
            ) -> dict[str, Any]:
                return await _dispatch_model_call(
                    client=client,
                    client_kwargs=_kwargs,
                    http=http,
                    endpoint=endpoint,
                    headers=_headers,
                    payload=_payload,
                    messages=messages,
                    system_text=_system,
                )

            # Request with retries + emergency truncation on request_too_large.
            resp_data: dict[str, Any] | None = None
            last_err_msg = ''
            for attempt in range(agent_input.api_retry_times):
                try:
                    resp_data = await _call()
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err_msg = str(exc)
                    logger.warning(
                        'anthropic call error (attempt %d/%d): %s',
                        attempt + 1,
                        agent_input.api_retry_times,
                        last_err_msg[:400],
                    )
                    if _is_request_too_large(exc):
                        # Emergency truncation loop (mirrors official).
                        if not request_too_large_seen:
                            logger.warning('request_too_large seen — dropping to threshold=10')
                        request_too_large_seen = True
                        emergency_threshold = min(10, image_truncation_threshold)
                        while True:
                            _maybe_filter_to_n_most_recent_images(
                                messages,
                                agent_input.only_n_most_recent_images,
                                min_removal_threshold=emergency_threshold,
                            )
                            payload['messages'] = messages
                            try:
                                resp_data = await _call()
                                break
                            except Exception as inner_exc:  # noqa: BLE001
                                if not _is_request_too_large(inner_exc):
                                    raise
                                if emergency_threshold == 1:
                                    logger.error('request_too_large persists at threshold=1')
                                    resp_data = None
                                    break
                                emergency_threshold = max(1, emergency_threshold // 2)
                        break
                    if _is_hard_client_error(exc):
                        # Terminal client errors must reach the task as failures.
                        raise
                    if attempt < agent_input.api_retry_times - 1:
                        await asyncio.sleep(API_RETRY_INTERVAL)

            if resp_data is None:
                raise RuntimeError(f'Anthropic API failed too many times: {last_err_msg[:200]}')

            _append_jsonl(traj_dir, 'full_responses.jsonl', resp_data)

            content_blocks = resp_data.get('content') or []
            raw_response_str = _extract_raw_response_string(content_blocks)
            stop_reason = resp_data.get('stop_reason')

            # Store response in message history (as official does).
            # Convert response blocks to param format (drop internal fields).
            response_params: list[dict[str, Any]] = []
            for blk in content_blocks:
                t = blk.get('type')
                if t == 'thinking':
                    response_params.append(blk)  # keep signature
                elif t == 'text' and blk.get('text'):
                    response_params.append({'type': 'text', 'text': blk['text']})
                elif t == 'tool_use':
                    response_params.append({
                        'type': 'tool_use',
                        'id': blk.get('id'),
                        'name': blk.get('name'),
                        'input': blk.get('input') or {},
                    })
                else:
                    response_params.append(blk)
            messages.append({'role': 'assistant', 'content': response_params})

            # Parse actions.
            actions: list[dict[str, Any]] = []
            reasonings: list[str] = []
            for blk in response_params:
                t = blk.get('type')
                if t == 'tool_use':
                    # Bedrock toolsets put the member action in the tool name.
                    # Keep the original history for the provider; normalize only
                    # the executor's view, preserving the tool_use id.
                    if use_toolset and blk.get('name') not in (
                        None,
                        'computer',
                        *NON_COMPUTER_TOOL_NAMES,
                    ):
                        blk = {
                            **blk,
                            'name': 'computer',
                            'input': {'action': blk.get('name'), **(blk.get('input') or {})},
                        }
                    pyautogui_code = ''
                    parse_error = None
                    if blk.get('name') in (None, 'computer'):
                        try:
                            pyautogui_code = adapter.parse_action(blk, resize_factor, agent_input)
                        except Exception as parse_exc:  # noqa: BLE001
                            parse_error = str(parse_exc)
                            logger.warning('parse_actions failed: %s', parse_exc)
                    actions.append({
                        'name': blk.get('name'),
                        'input': cast(dict[str, Any], blk.get('input') or {}),
                        'id': blk.get('id'),
                        'action_type': 'tool_use',
                        'command': pyautogui_code,
                        **adapter.action_fields(parse_error),
                        'raw_response': raw_response_str,
                    })
                elif t == 'text' and blk.get('text'):
                    reasonings.append(blk['text'])

            reasoning_text = reasonings[0] if reasonings else ''

            termination_text = adapter.termination_text(raw_response_str, reasonings, actions)
            if '[INFEASIBLE]' in termination_text:
                logger.info('detected [INFEASIBLE] — FAIL')
                actions = [{'action_type': 'FAIL', 'raw_response': raw_response_str}]
                exit_status = 'infeasible'
                final_answer = raw_response_str[:400]
                ctx.tool_state['os_world_last_action'] = {'action_type': 'FAIL'}
                trajectory.append({
                    'iteration': iteration,
                    'stop_reason': stop_reason,
                    'text': reasoning_text[:400],
                    'actions': ['FAIL'],
                })
                break
            if '[DONE]' in termination_text:
                logger.info('detected [DONE] — done')
                actions = [{'action_type': 'DONE', 'raw_response': raw_response_str}]
                exit_status = 'finished'
                final_answer = reasoning_text[:400] or 'DONE'
                ctx.tool_state['os_world_last_action'] = {'action_type': 'DONE'}
                trajectory.append({
                    'iteration': iteration,
                    'stop_reason': stop_reason,
                    'text': reasoning_text[:400],
                    'actions': ['DONE'],
                })
                break

            dispatched_actions = [label for act in actions for label in _dispatched_action_names(act)]
            trajectory.append({
                'iteration': iteration,
                'stop_reason': stop_reason,
                'text': reasoning_text[:400],
                'num_tool_uses': len(actions),
                'actions': dispatched_actions,
            })
            logger.info(
                '[iter %d] stop=%s tools=%d actions=%s',
                iteration,
                stop_reason,
                len(actions),
                dispatched_actions,
            )

            # No actions & stop_reason=end_turn → task is done.
            if not actions and stop_reason == 'end_turn':
                exit_status = 'finished'
                final_answer = reasoning_text
                ctx.tool_state['os_world_last_action'] = {'action_type': 'DONE'}
                break
            if not actions:
                # Model asking user question without tool_use — treat as done.
                exit_status = 'finished'
                final_answer = reasoning_text or 'no-tool-use'
                ctx.tool_state['os_world_last_action'] = {'action_type': 'DONE'}
                break

            if not adapter.allow_actions(actions, agent_input):
                break

            # Execute tool calls + build tool_result blocks.
            tool_results_content: list[dict[str, Any]] = []
            for ai, act in enumerate(actions):
                is_last_tool = ai == len(actions) - 1
                tool_name = act.get('name', 'computer')
                inner_action = act.get('input', {}).get('action')
                tool_input = act.get('input', {})
                if tool_name == TOOL_CONFIG_ANCHOR_NAME:
                    tool_results_content.append({
                        'type': 'tool_result',
                        'tool_use_id': act.get('id'),
                        'content': [{'type': 'text', 'text': 'No-op placeholder acknowledged.'}],
                    })
                    continue

                adapter.validate_tool(tool_name, agent_input)

                # --- bash / text_editor tools ---
                if tool_name in ('bash', 'text_editor', 'str_replace_based_edit_tool'):
                    bash_cmd = tool_input.get('command', '') if tool_name == 'bash' else ''
                    te_command = (
                        tool_input.get('command', '')
                        if tool_name in ('text_editor', 'str_replace_based_edit_tool')
                        else ''
                    )
                    te_path = tool_input.get('path', '')
                    te_content = tool_input.get('file_text', '') or tool_input.get('new_str', '')

                    if tool_name == 'bash' and bash_cmd:
                        try:
                            resp = await runtime.run_command(command=bash_cmd, timeout=120)
                            stdout = _get(resp, 'stdout') or _get(resp, 'output') or ''
                            stderr = _get(resp, 'stderr') or _get(resp, 'error') or ''
                            rc = _get(resp, 'return_code') or _get(resp, 'returncode')
                            output_text = stdout or stderr or f'(exit code {rc})'
                        except Exception as exc:
                            output_text = f'error: {exc}'
                    elif tool_name in ('text_editor', 'str_replace_based_edit_tool'):
                        if te_command == 'view' and te_path:
                            try:
                                resp = await runtime.run_command(command=f'cat {te_path}', timeout=30)
                                output_text = _get(resp, 'stdout') or _get(resp, 'output') or ''
                            except Exception as exc:
                                output_text = f'error: {exc}'
                        elif te_command == 'create' and te_path and te_content:
                            try:
                                resp = await runtime.run_command(
                                    command=f"cat > {te_path} << 'NEXUS_EOF'\n{te_content}\nNEXUS_EOF",
                                    timeout=30,
                                )
                                output_text = 'File created successfully.'
                            except Exception as exc:
                                output_text = f'error: {exc}'
                        elif te_command == 'str_replace' and te_path:
                            old_str = tool_input.get('old_str', '')
                            new_str = tool_input.get('new_str', '')
                            try:
                                sed_old = old_str.replace('/', '\\/').replace('\n', '\\n')
                                sed_new = new_str.replace('/', '\\/').replace('\n', '\\n')
                                resp = await runtime.run_command(
                                    command=f"sed -i 's/{sed_old}/{sed_new}/g' {te_path}",
                                    timeout=30,
                                )
                                output_text = 'Replacement done.'
                            except Exception as exc:
                                output_text = f'error: {exc}'
                        else:
                            output_text = f'text_editor command={te_command} executed.'
                    else:
                        output_text = 'OK'

                    ctx.tool_state['os_world_last_action'] = {'action_type': tool_name}
                    tool_results_content.append({
                        'type': 'tool_result',
                        'tool_use_id': act.get('id'),
                        'content': [{'type': 'text', 'text': output_text[:10000]}],
                    })
                    continue

                # --- computer tool (GUI actions) ---
                pg_ok, pg_err = await adapter.execute_action(
                    runtime,
                    act,
                    execution_mode=agent_input.pyautogui_execution_mode,
                    timeout=agent_input.pyautogui_timeout,
                )
                adapter.record_execution(act, pg_ok, pg_err)

                ctx.tool_state['os_world_last_action'] = {'action_type': inner_action or 'unknown'}

                if agent_input.sleep_after_execution > 0:
                    await asyncio.sleep(agent_input.sleep_after_execution)
                if agent_input.step_pause > 0 and is_last_tool:
                    await asyncio.sleep(agent_input.step_pause)

                # Take a fresh screenshot after this action.
                latest_screenshot_bytes = await _screenshot_bytes(runtime)
                if is_last_tool:
                    _save_screenshot(traj_dir, f'obs_{iteration:03d}.png', latest_screenshot_bytes)
                    from datetime import datetime as _dt
                    from datetime import timezone as _tz

                    _step_reasoning = ''
                    _step_content_text = ''
                    for _blk in content_blocks:
                        if _blk.get('type') == 'thinking':
                            _step_reasoning = _blk.get('thinking', '')
                        elif _blk.get('type') == 'text':
                            _step_content_text = _blk.get('text', '')
                    _step_raw_actions = []
                    _step_exec_actions = []
                    for _act in actions:
                        _a_input = _act.get('input', {})
                        _a_name = _act.get('name', 'computer')
                        _a_code = _act.get('command', '')
                        _step_raw_actions.append({'tool_name': _a_name, **_a_input})
                        if _a_name == 'bash':
                            _step_exec_actions.append({'type': 'bash', 'command': _a_input.get('command', '')})
                        else:
                            _step_exec_actions.append({'action_type': _a_input.get('action', ''), 'code': _a_code})
                    _append_jsonl(
                        traj_dir,
                        'trajectory.jsonl',
                        {
                            'step': iteration,
                            'observation': f'screenshots/obs_{iteration - 1:03d}.png',
                            'summary': '',
                            'reasoning_content': _step_reasoning,
                            'content': _step_content_text,
                            'raw_actions': _step_raw_actions,
                            'actions': _step_exec_actions,
                            'tool_name': actions[-1].get('name', 'computer') if actions else 'computer',
                            **adapter.result_fields(pg_ok, pg_err),
                            'result_observation': f'screenshots/obs_{iteration:03d}.png',
                            'timestamp': _dt.now(_tz.utc).isoformat(),
                        },
                    )

                include_screenshot = (inner_action == 'screenshot') or is_last_tool
                tr_content: list[dict[str, Any]] = [
                    {'type': 'text', 'text': 'Success' if pg_ok else f'error: {pg_err}'}
                ]
                if include_screenshot and latest_screenshot_bytes:
                    adapter.image_added()
                    resized = _resize_for_model(latest_screenshot_bytes)
                    tr_content.append({
                        'type': 'image',
                        'source': {'type': 'base64', 'media_type': 'image/png', 'data': _b64(resized)},
                    })
                # NOTE: do NOT set is_error. The Anthropic API rejects the
                # request if is_error=True while content contains an image
                # block ("all content must be type `text` if `is_error` is
                # true"). Official add_tool_result never sets is_error —
                # errors are conveyed via the text of the tool_result and
                # the model reads them like any other observation.
                tool_results_content.append({
                    'type': 'tool_result',
                    'tool_use_id': act.get('id'),
                    'content': tr_content,
                })

            messages.append({'role': 'user', 'content': tool_results_content})

            # Claude 4.7 gets an explicit step counter appended after every
            # tool_result so the model knows how much budget remains. Matches
            # OSWorld-V2 main.py:494-498. Only applied when the profile opts in
            # (use_claude_47_prompt=True); other prompt variants omit it.
            if profile.use_claude_47_prompt:
                messages[-1]['content'].append({
                    'type': 'text',
                    'text': f'[Current step: {iteration}/{agent_input.max_iterations}]',
                })

    if traj_dir:
        try:
            final_bytes = await _screenshot_bytes(runtime)
            _save_screenshot(traj_dir, 'obs_final.png', final_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning('final screenshot failed: %s', exc)

    return adapter.output_type(
        main_trajectory=trajectory,
        exit_status=exit_status,
        iterations=len(trajectory),
        **adapter.output_fields(),
        final_answer=final_answer,
    )


__all__ = ['CuaAnthropicAgentInput', 'CuaAnthropicAgentOutput', 'run_cua_anthropic_agent']
