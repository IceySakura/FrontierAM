"""Protocol helpers for the ``yb_vlm_oneagent`` task.

This module owns the task-facing protocol glue:

* render mm_agent-style task input into the initial ``run_agent`` payload;
* convert managed image parts into base64 data URIs after context sizing;
* seed ``ctx.tool_state`` keys consumed by YB search tools and hooks;
* validate the tool set declared by ``agent_config``.

The task entry point should stay focused on orchestration, reward handling,
and abnormal-exit behavior.
"""

import base64
import os

from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any

from PIL import Image

from nexus.extensions.projects.yb_vlm_oneagent.context_manage import ContextBudget, TokenCounter
from nexus.extensions.projects.yb_vlm_oneagent.context_manage.image_utils import load_image_bytes
from nexus.extensions.projects.yb_vlm_oneagent.hooks import ENABLED_KEY
from nexus.extensions.projects.yb_vlm_oneagent.prompts import apply_oneagent_to_messages
from nexus.extensions.tasks.mm_agent.task import SearchTaskInput as MMSearchTaskInput


if TYPE_CHECKING:
    from nexus.extensions.projects.yb_vlm_oneagent.task import YbVlmOneAgentTaskInput
else:
    YbVlmOneAgentTaskInput = MMSearchTaskInput


HOOK_PATH = 'nexus.extensions.projects.yb_vlm_oneagent.hooks::budget_and_finish_hook'
PINNED_TOOLS: tuple[str, str] = ('yb_web_search', 'yb_vision_search')
_EXPECTED_TOOLS: frozenset[str] = frozenset(PINNED_TOOLS)

# Search-tool consumption.
_USER_QUESTION_KEY = 'yb_search.user_question'
_USER_IMAGES_KEY = 'yb_search.images'
_NEXT_IMAGE_IDX_KEY = 'yb_search.next_image_idx'

# Hook-side budget accounting.
_TOOL_IMG_MIN_KEY = 'yb_search.tool_image_min_pixels'
_TOOL_IMG_MAX_KEY = 'yb_search.tool_image_max_pixels'
_TOOL_ROUND_MAX_TOKENS_KEY = 'yb_search.tool_round_max_tokens'
_TOOL_ROUND_SPECIAL_RESERVE_KEY = 'yb_search.tool_round_special_reserve_tokens'


@dataclass(frozen=True)
class OneAgentInitialPayload:
    """Initial payload passed to ``run_agent`` after protocol rendering."""

    system_prompt: str
    history: list[dict[str, Any]]
    user_prompt: list[dict[str, Any]]
    user_images: list[Any]
    tool_image_min_pixels: int
    tool_image_max_pixels: int
    budget: ContextBudget


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == '':
        return default
    return int(raw)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == '':
        return default
    return raw.strip().lower() in {'1', 'true', 'yes', 'on'}


def context_budget_from_env() -> ContextBudget:
    """Return context budget with stable code defaults and optional env overrides."""
    default = ContextBudget()
    return ContextBudget(
        system_max_tokens=env_int('YB_ONEAGENT_SYSTEM_MAX_TOKENS', default.system_max_tokens),
        history_user_max_tokens=env_int('YB_ONEAGENT_HISTORY_USER_MAX_TOKENS', default.history_user_max_tokens),
        history_user_text_max_tokens=env_int(
            'YB_ONEAGENT_HISTORY_USER_TEXT_MAX_TOKENS',
            default.history_user_text_max_tokens,
        ),
        history_answer_max_tokens=env_int('YB_ONEAGENT_HISTORY_ANSWER_MAX_TOKENS', default.history_answer_max_tokens),
        tool_round_max_tokens=env_int('YB_ONEAGENT_TOOL_ROUND_MAX_TOKENS', default.tool_round_max_tokens),
        tool_round_special_reserve_tokens=env_int(
            'YB_ONEAGENT_TOOL_ROUND_SPECIAL_RESERVE_TOKENS',
            default.tool_round_special_reserve_tokens,
        ),
        factor=env_int('YB_ONEAGENT_RESIZE_FACTOR', default.factor),
        min_pixels=env_int('YB_ONEAGENT_MIN_PIXELS', default.min_pixels),
        base_max_pixels=env_int('YB_ONEAGENT_BASE_MAX_PIXELS', default.base_max_pixels),
        tool_max_pixels=env_int('YB_ONEAGENT_TOOL_MAX_PIXELS', default.tool_max_pixels),
        multi_min_pixels=env_int('YB_ONEAGENT_MULTI_MIN_PIXELS', default.multi_min_pixels),
        interleave_max_size=env_int('YB_ONEAGENT_INTERLEAVE_MAX_SIZE', default.interleave_max_size),
        multi_nums_limit=env_int('YB_ONEAGENT_MULTI_NUMS_LIMIT', default.multi_nums_limit),
        max_ratio=env_int('YB_ONEAGENT_MAX_RATIO', default.max_ratio),
    )


def extract_agent_config_from_run_agent(run_agent_fn: Any) -> dict[str, Any] | None:
    """Best-effort recovery of ``run_input.agent_config`` from ``run_agent`` closure."""
    closure = getattr(run_agent_fn, '__closure__', None) or ()
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        agent_cfg = getattr(value, 'agent_config', None)
        if isinstance(agent_cfg, dict):
            return agent_cfg
    return None


def validate_pinned_tools(agent_config: dict[str, Any] | None) -> None:
    """Fail loudly if run config declares a non-oneagent tool set."""
    if agent_config is None:
        return

    declared_tools = agent_config.get('tools')
    if declared_tools is None or frozenset(declared_tools) == _EXPECTED_TOOLS:
        return

    raise ValueError(
        'yb_vlm_oneagent: agent_config.tools must be exactly '
        f'{sorted(_EXPECTED_TOOLS)} (got {sorted(declared_tools)}). '
        'This task pins its tool set because the oneagent SP, the '
        'budget_and_finish hook, and the ctx.tool_state schema '
        'are all calibrated for these two tools — silently '
        'swapping them would lead to undefined rollout behaviour.'
    )


def _resize_image_to_base64_data_uri(image_source: Any, new_size: Any) -> str:
    image_bytes = load_image_bytes(image_source)
    with Image.open(BytesIO(image_bytes)) as img:
        if new_size is not None:
            width, height = new_size
            target_size = (int(width), int(height))
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=95, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f'data:image/jpeg;base64,{encoded}'


def _build_raw_user_content(question: str, images: list[str] | None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image in images or []:
        content.append({'type': 'image_url', 'image_url': {'url': image}})
    content.append({'type': 'text', 'text': question})
    return content


def _finalize_history_images(history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for message in history or []:
        content = message.get('content')
        if not isinstance(content, list):
            finalized.append(message)
            continue

        new_content: list[Any] = []
        changed = False
        for part in content:
            if not isinstance(part, dict) or part.get('type') != 'image_url':
                new_content.append(part)
                continue

            image_url = part.get('image_url')
            if not isinstance(image_url, dict) or '_resize_plan' not in image_url:
                new_content.append(part)
                continue

            source = image_url.get('_image_source') or image_url.get('url')
            resize_plan = image_url.get('_resize_plan') or {}
            if source is None or source == '':
                new_content.append(part)
                continue

            new_content.append({
                'type': 'image_url',
                'image_url': {
                    'url': _resize_image_to_base64_data_uri(source, resize_plan.get('new_size')),
                    'detail': image_url.get('detail', 'high'),
                },
            })
            changed = True

        finalized.append({**message, 'content': new_content} if changed else message)
    return finalized


def _extract_image_sources(messages: list[dict[str, Any]]) -> list[Any]:
    """Return original image paths from managed messages in conversation order."""
    images: list[Any] = []
    for message in messages:
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get('type') != 'image_url':
                continue
            image_url = part.get('image_url')
            if not isinstance(image_url, dict):
                continue
            source = image_url.get('_image_source') or image_url.get('url')
            if source is not None and source != '':
                images.append(source)
    return images


def _build_raw_messages(task_input: YbVlmOneAgentTaskInput) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if task_input.system_prompt:
        messages.append({'role': 'system', 'content': task_input.system_prompt})
    messages.extend(task_input.history or [])
    messages.append({
        'role': 'user',
        'content': _build_raw_user_content(task_input.question, task_input.images),
    })
    return messages


def build_initial_payload(
    task_input: YbVlmOneAgentTaskInput,
    token_counter: TokenCounter,
) -> OneAgentInitialPayload:
    """Build ``run_agent`` initial inputs from mm_agent-style task input."""
    budget = context_budget_from_env()
    managed_messages, tool_min_px, tool_max_px, _ = apply_oneagent_to_messages(
        _build_raw_messages(task_input),
        template_name=os.environ.get('YB_VLM_SP_TEMPLATE_NAME', 'sp_v8'),
        location_info=task_input.location_info or '',
        budget=budget,
        token_counter=token_counter,
    )

    system_prompt = ''
    if managed_messages and managed_messages[0].get('role') == 'system':
        content = managed_messages[0].get('content')
        if isinstance(content, str):
            system_prompt = content

    history = managed_messages[1:-1] if len(managed_messages) > 2 else []
    user_prompt = managed_messages[-1].get('content') if managed_messages else []
    if not isinstance(user_prompt, list):
        user_prompt = [{'type': 'text', 'text': str(user_prompt or '')}]

    return OneAgentInitialPayload(
        system_prompt=system_prompt,
        history=_finalize_history_images(history),
        user_prompt=_finalize_history_images([{'role': 'user', 'content': user_prompt}])[0]['content'],
        user_images=_extract_image_sources(managed_messages),
        tool_image_min_pixels=int(tool_min_px),
        tool_image_max_pixels=int(tool_max_px),
        budget=budget,
    )


def seed_tool_state(
    ctx: Any,
    *,
    question: str,
    payload: OneAgentInitialPayload,
    token_counter: TokenCounter,
    enable_force_finish: bool,
) -> None:
    """Seed ``ctx.tool_state`` keys consumed by tools and the post-iteration hook."""
    state = ctx.tool_state
    state['token_counter'] = token_counter
    state[_USER_QUESTION_KEY] = question
    state[_USER_IMAGES_KEY] = payload.user_images
    state[_NEXT_IMAGE_IDX_KEY] = len(payload.user_images) + 1
    state[_TOOL_IMG_MIN_KEY] = payload.tool_image_min_pixels
    state[_TOOL_IMG_MAX_KEY] = payload.tool_image_max_pixels
    state[_TOOL_ROUND_MAX_TOKENS_KEY] = payload.budget.tool_round_max_tokens
    state[_TOOL_ROUND_SPECIAL_RESERVE_KEY] = payload.budget.tool_round_special_reserve_tokens
    state[ENABLED_KEY] = enable_force_finish
