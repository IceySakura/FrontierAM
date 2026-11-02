"""Miscellaneous getters for WAA evaluation.

These getters handle expected value resolution (rule, rule_relativeTime).

All functions have the unified async signature:
    async def get_xxx(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any
"""

from __future__ import annotations

import logging

from datetime import datetime, timedelta
from typing import Any

from nexus.extensions.tasks.waa.registry import getter_registry
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime


logger = logging.getLogger('nexus.waa.getters.misc')

# ---------------------------------------------------------------------------
# Time mappings (from original WAA misc.py)
# ---------------------------------------------------------------------------

day_of_week_mapping = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

month_mapping = {
    1: 'Jan',
    2: 'Feb',
    3: 'Mar',
    4: 'Apr',
    5: 'May',
    6: 'Jun',
    7: 'Jul',
    8: 'Aug',
    9: 'Sep',
    10: 'Oct',
    11: 'Nov',
    12: 'Dec',
}

month_mapping_full = {
    1: 'january',
    2: 'february',
    3: 'march',
    4: 'april',
    5: 'may',
    6: 'june',
    7: 'july',
    8: 'august',
    9: 'september',
    10: 'october',
    11: 'november',
    12: 'december',
}

Month_Mapping_Full = {  # noqa: N816
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December',
}

relativeTime_to_IntDay = {  # noqa: N816
    'tomorrow': 1,
    '5th next month': 'special',
    '10th next month': 'special',
    '11th next month': 'special',
    'this month': 'special',
    'this Saturday': 'special',
    'this Sunday': 'special',
    'next Monday': 'special',
    'next Friday': 'special',
    'first monday four months later': 'special',
}


def _apply_rules_to_time_format(time_format: str, absolute_day: datetime) -> str:
    """Replace time format placeholders with actual date values."""
    result = time_format.replace('{DoW}', day_of_week_mapping[absolute_day.weekday()], 1)
    result = result.replace('{Month}', month_mapping[absolute_day.month], 1)
    result = result.replace('{DayD}', str(absolute_day.day), 1)
    result = result.replace('{Year}', str(absolute_day.year), 1)
    result = result.replace(
        '{Month0D}', f'0{absolute_day.month}' if absolute_day.month < 10 else str(absolute_day.month), 1
    )
    result = result.replace('{month}', month_mapping_full[absolute_day.month], 1)
    result = result.replace('{MonthFull}', Month_Mapping_Full[absolute_day.month], 1)
    return result.replace('{Day0D}', f'0{absolute_day.day}' if absolute_day.day < 10 else str(absolute_day.day), 1)


def _resolve_relative_time(relative_time_str: str, now: datetime) -> datetime:
    """Resolve a relative time string to an absolute datetime."""
    int_day = relativeTime_to_IntDay.get(relative_time_str)
    if int_day is None:
        raise ValueError(f'Unknown relative time: {relative_time_str}')

    if int_day != 'special':
        return now + timedelta(days=int(int_day))

    # Special cases
    if relative_time_str == '5th next month':
        next_year = now.year + 1 if now.month == 12 else now.year
        next_month = now.month + 1 if now.month < 12 else 1
        return datetime(next_year, next_month, 5)
    if relative_time_str == '10th next month':
        next_year = now.year + 1 if now.month == 12 else now.year
        next_month = now.month + 1 if now.month < 12 else 1
        return datetime(next_year, next_month, 10)
    if relative_time_str == '11th next month':
        next_year = now.year + 1 if now.month == 12 else now.year
        next_month = now.month + 1 if now.month < 12 else 1
        return datetime(next_year, next_month, 11)
    if relative_time_str == 'this month':
        return now
    if relative_time_str == 'next Monday':
        return now + timedelta(days=((6 - now.weekday()) + 1))
    if relative_time_str == 'this Saturday':
        return now + timedelta(days=(5 - now.weekday()))
    if relative_time_str == 'this Sunday':
        return now + timedelta(days=(6 - now.weekday()))
    if relative_time_str == 'first monday four months later':
        next_year = now.year + 1 if now.month >= 9 else now.year
        next_month = (now.month + 4) % 12
        temp_date = datetime(next_year, next_month, 1)
        return temp_date + timedelta(days=((6 - temp_date.weekday()) + 1) % 7)
    raise ValueError(f'Unhandled special relative time: {relative_time_str}')


# ---------------------------------------------------------------------------
# Registered getters
# ---------------------------------------------------------------------------


@getter_registry.register()
async def get_rule(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:
    """Returns the rule as-is. Used for inline expected values."""
    return config['rules']


@getter_registry.register()
async def get_rule_relativeTime(runtime: AGSRawWindowsAgentArenaRuntime, config: dict) -> Any:  # noqa: N802
    """Convert relative time references to absolute dates, then return rules.

    Handles time placeholders like {DoW}, {Month}, {DayD}, {Year}, etc.
    """
    rules = config['rules']
    relative_time = rules['relativeTime']
    now = datetime.now()

    if 'to' not in relative_time:
        # Single time reference
        start_relative = relative_time['from']
        absolute_day = _resolve_relative_time(start_relative, now)
        rules['expected']['time'] = _apply_rules_to_time_format(rules['expected']['time'], absolute_day)
    else:
        # From-to time range
        from_time_str = relative_time['from']
        to_time_str = relative_time['to']

        from_absolute = _resolve_relative_time(from_time_str, now)
        rules['expected']['from'] = _apply_rules_to_time_format(rules['expected']['from'], from_absolute)

        # Handle 'next Friday' special case with context from 'from' time
        if to_time_str == 'next Friday' and from_time_str == 'next Monday':
            if now.weekday() < 4:
                to_absolute = now + timedelta(days=((4 - now.weekday()) + 7))
            else:
                to_absolute = now + timedelta(
                    days=((4 - now.weekday()) if now.weekday() < 4 else (6 - now.weekday()) + 5)
                )
        else:
            to_absolute = _resolve_relative_time(to_time_str, now)

        rules['expected']['to'] = _apply_rules_to_time_format(rules['expected']['to'], to_absolute)

    return rules
