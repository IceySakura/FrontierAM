"""Miscellaneous expected/result getters."""

from __future__ import annotations

import contextlib
import copy
import datetime as dt
import json

from typing import Any, TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from nexus.extensions.tasks.os_world.evaluation.registry import getter_registry
from nexus.extensions.tasks.os_world.getters.runtime import response_stdout, run_python


R = TypeVar('R')

_DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
_MONTH = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
_MONTH_FULL = [
    '',
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December',
]


@getter_registry.register()
async def get_rule(runtime: Any, config: dict[str, R]) -> R:
    return config['rules']


@getter_registry.register()
async def get_accessibility_tree(runtime: Any, config: dict[str, Any] | None = None) -> str:
    if hasattr(runtime, 'accessibility_tree'):
        tree = await runtime.accessibility_tree()
        if isinstance(tree, str):
            return tree
        if isinstance(tree, dict):
            return str(tree.get('tree') or tree.get('xml') or json.dumps(tree))
        return str(tree)
    return ''


@getter_registry.register('rule_relativeTime')
async def get_rule_relative_time(runtime: Any, config: dict[str, Any]) -> dict[str, Any]:
    rules = copy.deepcopy(config['rules'])
    now = await _vm_now(runtime)
    if tz_str := rules.get('timezone'):
        with contextlib.suppress(ZoneInfoNotFoundError, KeyError):
            now = now.astimezone(ZoneInfo(tz_str))
    relative = rules['relativeTime']
    expected = rules['expected']

    if 'to' not in relative:
        when = _resolve_relative_day(now, str(relative['from']))
        value = expected.get('time')
        expected['time'] = (
            [_format_time(item, when) for item in value] if isinstance(value, list) else _format_time(value, when)
        )
        return rules

    from_day = _resolve_relative_day(now, str(relative['from']))
    to_day = _resolve_relative_day(now, str(relative['to']), from_time=str(relative['from']), from_day=from_day)

    if _is_split_label(relative['from']):
        expected['puDay'] = _format_time(expected['puDay'], from_day)
        expected['puMonth'] = _format_time(expected['puMonth'], from_day)
        expected['puYear'] = _format_time(expected['puYear'], from_day)
    else:
        expected['from'] = _format_time(expected['from'], from_day)

    if _is_split_label(relative['to']):
        expected['doDay'] = _format_time(expected['doDay'], to_day)
        expected['doMonth'] = _format_time(expected['doMonth'], to_day)
        expected['doYear'] = _format_time(expected['doYear'], to_day)
    else:
        expected['to'] = _format_time(expected['to'], to_day)
    return rules


@getter_registry.register()
async def get_time_diff_range(runtime: Any, config: dict[str, Any]) -> str:
    return str(config.get('diff_range', config.get('rules', '')))


async def _vm_now(runtime: Any) -> dt.datetime:
    try:
        result = await run_python(
            runtime, 'from datetime import datetime; print(datetime.now().astimezone().isoformat())'
        )
        output = response_stdout(result).strip()
        if output:
            return dt.datetime.fromisoformat(output)
    except Exception:
        pass
    return dt.datetime.now().astimezone()


def _next_month_day(now: dt.datetime, day: int) -> dt.datetime:
    month = now.month + 1
    year = now.year
    if month == 13:
        month = 1
        year += 1
    return now.replace(year=year, month=month, day=day)


def _next_weekday(now: dt.datetime, weekday: int, *, include_this_week: bool = False) -> dt.datetime:
    delta = weekday - now.weekday()
    if delta < 0 or (delta == 0 and not include_this_week):
        delta += 7
    return now + dt.timedelta(days=delta)


def _week_after_next_weekday(now: dt.datetime, weekday: int) -> dt.datetime:
    return now + dt.timedelta(days=(7 - now.weekday()) + weekday + 7)


def _resolve_relative_day(
    now: dt.datetime,
    label: str,
    *,
    from_time: str | None = None,
    from_day: dt.datetime | None = None,
) -> dt.datetime:
    if label == 'tomorrow':
        return now + dt.timedelta(days=1)
    if label == 'this month':
        return now
    if label == '5th next month':
        return _next_month_day(now, 5)
    if label == '10th next month':
        return _next_month_day(now, 10)
    if label == '11th next month':
        return _next_month_day(now, 11)
    if label in {'next Monday', 'next Monday split'}:
        return _next_weekday(now, 0)
    if label in {'next Friday', 'next Friday split'}:
        if from_time in {'next Monday', 'next Monday split'} and from_day is not None:
            return from_day + dt.timedelta(days=4)
        return _next_weekday(now, 4)
    if label == 'next Saturday':
        return _next_weekday(now, 5)
    if label == 'next Sunday':
        if from_time in {'next Friday', 'next Saturday'} and from_day is not None:
            return from_day + dt.timedelta(days=6 - from_day.weekday())
        return _next_weekday(now, 6)
    if label == 'next week Friday':
        return now + dt.timedelta(days=(7 - now.weekday()) + 4)
    if label == 'next week Saturday':
        return now + dt.timedelta(days=(7 - now.weekday()) + 5)
    if label == 'next week Sunday':
        if from_time in {'next week Friday', 'next week Saturday'} and from_day is not None:
            return from_day + dt.timedelta(days=6 - from_day.weekday())
        return now + dt.timedelta(days=(7 - now.weekday()) + 6)
    if label == 'week after next Friday':
        return _week_after_next_weekday(now, 4)
    if label == 'week after next Saturday':
        return _week_after_next_weekday(now, 5)
    if label == 'week after next Sunday':
        if from_time in {'week after next Friday', 'week after next Saturday'} and from_day is not None:
            return from_day + dt.timedelta(days=6 - from_day.weekday())
        return _week_after_next_weekday(now, 6)
    if label == 'week after next Monday split':
        return _week_after_next_weekday(now, 0)
    if label == 'week after next Friday split':
        if from_time in {'week after next Monday split'} and from_day is not None:
            return from_day + dt.timedelta(days=4)
        return _week_after_next_weekday(now, 4)
    if label == 'first monday eight months later':
        return _first_weekday_months_later(now, months=8, weekday=0)
    if label == 'first monday four months later':
        return _first_weekday_months_later(now, months=4, weekday=0)
    return now


def _first_weekday_months_later(now: dt.datetime, *, months: int, weekday: int) -> dt.datetime:
    month_index = now.month - 1 + months
    year = now.year + month_index // 12
    month = month_index % 12 + 1
    first = now.replace(year=year, month=month, day=1)
    return first + dt.timedelta(days=(weekday - first.weekday()) % 7)


def _is_split_label(label: Any) -> bool:
    """Return True for relative-time labels that carry the ``split`` contract.

    ``split`` labels tell the evaluator that the resolved day is written back as
    separate ``pu*``/``do*`` components instead of a single ``from``/``to`` value.
    Match the whole suffix word so every phrasing is handled, not just the two
    literals the task set happened to use first.
    """
    return str(label).split()[-1:] == ['split']


def _format_time(template: str, value: dt.datetime) -> str:
    return (
        str(template)
        .replace('{DoW}', _DOW[value.weekday()], 1)
        .replace('{Month}', _MONTH[value.month], 1)
        .replace('{DayD}', str(value.day), 1)
        .replace('{Year}', str(value.year), 1)
        .replace('{Month0D}', f'{value.month:02d}', 1)
        .replace('{month}', _MONTH_FULL[value.month].lower(), 1)
        .replace('{MonthFull}', _MONTH_FULL[value.month], 1)
        .replace('{Day0D}', f'{value.day:02d}', 1)
        .replace('{MonthD}', str(value.month), 1)
    )
