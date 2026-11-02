"""Fill a scenario card into the adjacent simulator.md template, without model calls.

The scenario data pipeline renders its ``prompt/<script_id>.md`` with the same template and slot
filling, so a card renders byte-for-byte alike on both sides; per-turn conversation, state and
timing are supplied by ``simulation_state``.
"""

from __future__ import annotations

import math
import re

from pathlib import Path
from typing import Any


TEMPLATE = Path(__file__).with_name('simulator.md')
PROMPT_VERSION = 'scenario-persona/2026-10-15'

DEFAULT_EVENTS = {
    'question': '知道的就答，可以只答被问到的；还没想好就说没想好或让对方定。',
    'permission': (
        '你能接受的操作，允许这一次；同类请求问多了，就让它这类以后直接做；'
        '碰到你不能接受的就拒绝；看不懂要做什么，先问清楚。'
    ),
    'failure': '先问一句怎么回事，让它再试一次；还不行就接受当时能拿到的，或者这件不做了。',
}
STYLES = {
    'A': '第一次开口就说清主要目标和关键要求；话少一句说不完的，紧接着一两句把要求说完，不等它问。',
    'B': (
        '第一次开口只说个大方向（{disclosure}）。助手问到时如实回答；'
        '它没问就开始做了，你在下一句主动补上没说的初始需求，不等到交付后才责怪它。'
    ),
}
MOMENTS = {
    '开场': '对话开始时',
    '在做时': '助手开始处理、还没说做完的时候',
    '等久了': '助手在处理、而且好一阵没有新消息的时候',
    '做完后': '助手说做完之后',
    '停了以后': '助手确认停下之后',
    '空档': '没有在等你回应的时候',
}
EVENTS = {'question': '助手问你问题时', 'permission': '助手请你授权时', 'failure': '助手说没做成时'}
_SLOTS = re.compile(r'<<(\w+)>>')


def _moment(when: str) -> str:
    head, separator, extra = str(when).partition('：')
    if not separator:
        head, _, extra = str(when).partition(':')
    return MOMENTS.get(head.strip(), head.strip()) + (f'（且{extra.strip()}）' if extra.strip() else '')


def _listed(items: list[dict[str, Any]]) -> str:
    return '\n'.join(
        f'[{index}] {_moment(item.get("when", "空档"))}：{item["action"]}' for index, item in enumerate(items)
    )


def _materials(items: list[Any]) -> str:
    if not items:
        return ''
    lines = ['你手上有这些，需要时原样提供；只有实际发出的片段才算助手收到：']
    for index, item in enumerate(items):
        material = {'name': f'材料{index + 1}', 'content': item} if isinstance(item, str) else item
        if material.get('url'):
            detail = f'{material["url"]}（你只知道链接，不能声称读过网页）'
        elif material.get('via') == '文件' or material.get('path'):
            detail = '文件材料；实际上传成功后再使用 material_chunks 给出的路径。'
            if material.get('content'):
                detail += f'正文：{material["content"]}'
        else:
            detail = material.get('content') or material.get('note') or ''
        lines.append(f'- {material.get("name", f"材料{index + 1}")}：{detail}')
    return '\n'.join(lines)


def fields(
    script: dict[str, Any], *, decision_interval_s: float = 30, progress_interval_s: float = 90
) -> dict[str, str]:
    """Return template slots without exposing metadata tags as user instructions."""
    for interval in (decision_interval_s, progress_interval_s):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('decision/progress intervals must be finite positive seconds')
    user, plan = script['user'], script.get('event_plan') or {}
    insertions = script.get('optional_insertions') or []
    materials = (script.get('context') or {}).get('materials') or []
    constraints = script.get('constraints') or []
    habits = user.get('habits') or []
    ending = script['ending']
    disclosure = (user.get('disclosure') or '首轮不把具体要求全说完').rstrip('。')
    style = STYLES[user.get('style', 'A')].format(disclosure=disclosure)
    if user.get('style', 'A') == 'A' and user.get('disclosure'):
        style += f'\n材料披露方式：{user["disclosure"]}'
    return {
        'background': f'你的情况：{user["background"]}',
        'habits': f'你的说话习惯：{"；".join(habits)}。' if habits else '',
        'goal': f'你想要的是：{script["goal"]}',
        'style': style,
        'materials': _materials(materials),
        'constraints': ('你不能接受：' + ''.join(f'\n- {constraint}' for constraint in constraints))
        if constraints
        else '',
        'nodes': _listed(script.get('required_nodes') or []) or '没有额外的顺序事项，围绕目标自然交流。',
        'insertions': ('# 有空时可以顺口问的（可以不问，每件最多一次，问完回到正事）\n' + _listed(insertions) + '\n')
        if insertions
        else '',
        'events': '\n'.join(
            f'- {label}：{(plan.get(key) or {}).get("response") or DEFAULT_EVENTS[key]}'
            for key, label in EVENTS.items()
        ),
        'endings': '\n'.join(
            [f'- 正常结束：{ending["normal"]}'] + ([f'- 提前结束：{ending["early"]}'] if ending.get('early') else [])
        ),
        'timing': (
            f'静默时每约{decision_interval_s:g}秒获得一次决策机会，新发言可提前唤醒，不需要每次回复。'
            f'普通进度追问至少间隔{progress_interval_s:g}秒，以 timing.progress_query_allowed 为准；'
            '这是最小间隔，不是到点就催。没有新情况就等着。\n'
            '澄清、授权、失败回应和做完后询问结果质量不是催进度。'
            '不把内部计时和未来计划说给助手；业务日期和截止时间可正常说。'
        ),
    }


def system_prompt(
    script: dict[str, Any],
    *,
    decision_interval_s: float = 30,
    progress_interval_s: float = 90,
) -> str:
    """Render one system message. Scenario text is literal data, never another template."""
    text = TEMPLATE.read_text(encoding='utf-8')
    values = fields(script, decision_interval_s=decision_interval_s, progress_interval_s=progress_interval_s)
    unknown = set(_SLOTS.findall(text)) - values.keys()
    if unknown:
        raise ValueError(f'template slots not filled: {sorted(unknown)}')
    rendered = _SLOTS.sub(lambda match: values[match.group(1)], text)
    return re.sub(r'\n{3,}', '\n\n', rendered)
