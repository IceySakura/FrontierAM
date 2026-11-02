"""Lightweight metrics used by smoke tests and common OSWorld rules."""

from __future__ import annotations

import csv
import difflib
import json
import re
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from nexus.extensions.tasks.os_world.evaluation.registry import metric_registry


@metric_registry.register()
def exact_match(result: Any, rules: dict[str, Any]) -> float:
    return 1.0 if result == rules['expected'] else 0.0


@metric_registry.register()
def match_in_list(result: Any, rules: dict[str, Any]) -> float:
    return 1.0 if result in rules['expected'] else 0.0


@metric_registry.register()
def is_in_list(result: Any, rules: dict[str, Any]) -> float:
    return 1.0 if rules['expected'] in result else 0.0


@metric_registry.register()
def check_include_exclude(result: str | None, rules: dict[str, Any]) -> float:
    if result is None:
        return 0.0
    include = rules.get('include', [])
    exclude = rules.get('exclude', [])
    return 1.0 if all(item in result for item in include) and all(item not in result for item in exclude) else 0.0


@metric_registry.register()
def check_direct_json_object(result: Any, rules: dict[str, Any]) -> float:
    if isinstance(result, str):
        try:
            result = json.loads(result.strip().replace("'", '"'))
        except json.JSONDecodeError:
            return 0.0
    if result is None:
        return 0.0

    expected = rules.get('expected', {})
    if any(value == '__EVALUATION_FAILED__' for value in expected.values()):
        return 0.0

    if not rules.get('expect_in_result', False):
        for key, expected_value in expected.items():
            actual_value = result.get(key)
            if (
                rules.get('ignore_list_order', False)
                and isinstance(expected_value, list)
                and isinstance(actual_value, list)
            ):
                if sorted(expected_value) != sorted(actual_value):
                    return 0.0
            elif expected_value != actual_value:
                return 0.0
        return 1.0

    for key, expected_value in expected.items():
        actual_value = result.get(key)
        if isinstance(expected_value, list):
            if isinstance(actual_value, list) and any(item in actual_value for item in expected_value):
                continue
            if isinstance(actual_value, str) and any(item == actual_value for item in expected_value):
                continue
            return 0.0
        if isinstance(expected_value, str):
            if not isinstance(actual_value, str) or expected_value not in actual_value:
                return 0.0
            continue
        return 0.0
    return 1.0


@metric_registry.register()
def check_list(result: str | None, rules: dict[str, Any]) -> float:
    if result is None:
        return 0.0
    try:
        lines = Path(result).read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return 0.0
    expect = [re.compile(pattern) for pattern in rules.get('expect', [])]
    unexpect = [re.compile(pattern) for pattern in rules.get('unexpect', [])]
    expect_ok = all(any(pattern.search(line) for line in lines) for pattern in expect)
    unexpect_ok = all(not any(pattern.search(line) for line in lines) for pattern in unexpect)
    return 1.0 if expect_ok and unexpect_ok else 0.0


@metric_registry.register()
def check_csv(result: str | None, rules: dict[str, Any]) -> float:
    if result is None:
        return 0.0
    try:
        with open(result, newline='', encoding='utf-8', errors='replace') as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return 0.0
    expect_ok = all(any(_record_matches(row, rule) for row in rows) for rule in rules.get('expect', []))
    unexpect_ok = all(not any(_record_matches(row, rule) for row in rows) for rule in rules.get('unexpect', []))
    return 1.0 if expect_ok and unexpect_ok else 0.0


def _record_matches(row: dict[str, Any], rule: dict[str, Any]) -> bool:
    for key, expected in rule.items():
        actual = row.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


@metric_registry.register()
def diff_text_file(result: str | None, expected: str | None) -> float:
    if not result or not expected:
        return 0.0
    try:
        result_lines = Path(result).read_text(encoding='utf-8', errors='replace').splitlines()
        expected_lines = Path(expected).read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return 0.0
    return difflib.SequenceMatcher(a=result_lines, b=expected_lines).ratio()


@metric_registry.register()
def fuzzy_match(result: str, rules: dict[str, Any]) -> float:
    return fuzz.ratio(result, rules['expected']) / 100


@metric_registry.register()
def literal_match(result: Any, expected: Any, **options: Any) -> float:
    literal_type = options.get('type', 'str')
    ignore_case = bool(options.get('ignore_case', False))
    if literal_type == 'str':
        left, right = str(result), str(expected)
        return float(left.lower() == right.lower() if ignore_case else left == right)
    if literal_type == 'list':
        if (
            not isinstance(result, (list, tuple))
            or not isinstance(expected, (list, tuple))
            or len(result) != len(expected)
        ):
            return 0.0
        left = [str(item).lower() if ignore_case else str(item) for item in result]
        right = [str(item).lower() if ignore_case else str(item) for item in expected]
        return float(left == right)
    raise NotImplementedError(f'Unsupported literal_match type: {literal_type}')


@metric_registry.register()
def file_contains(file_path: str | None, config: dict[str, Any]) -> float:
    if not file_path:
        return 0.0
    try:
        text = Path(file_path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return 0.0
    return 1.0 if all(item in text for item in config['expected']) else 0.0


@metric_registry.register()
def check_line_number(file_path: str | None, line_number: dict[str, Any]) -> float:
    if not file_path:
        return 0.0
    try:
        text = Path(file_path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return 0.0
    count = len(re.findall(r'([01]\d|2[0-3]):[0-5]\d:([0-5]\d|60)', text))
    return 1.0 if count == int(line_number['expected']) else 0.0


@metric_registry.register()
def compare_python_pure_text(py_file_path: str | None, gold_file_path: str | None) -> float:
    if not py_file_path or not gold_file_path:
        return 0.0
    try:
        left = _normalize_python_text(Path(py_file_path).read_text(encoding='utf-8'))
        right = _normalize_python_text(Path(gold_file_path).read_text(encoding='utf-8'))
    except OSError:
        return 0.0
    return 1.0 if left == right else 0.0


def _normalize_python_text(text: str) -> str:
    lines = []
    for idx, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if idx < 3 and (
            stripped.startswith('#!') or stripped.startswith('# -*- coding:') or stripped.startswith('# coding')
        ):
            continue
        if stripped:
            lines.append(line.expandtabs(4).rstrip())
    return '\n'.join(lines)


@metric_registry.register()
def compare_terminal_and_txt(txt_file_path: str | None, terminal_output: str | None) -> float:
    if not txt_file_path or terminal_output is None:
        return 0.0
    try:
        return 1.0 if Path(txt_file_path).read_text(encoding='utf-8', errors='replace') == terminal_output else 0.0
    except OSError:
        return 0.0


@metric_registry.register()
def check_url_and_content_include(result: dict[str, str], rules: dict[str, Any]) -> float:
    if not isinstance(result, dict):
        return 0.0
    url = result.get('url', '') or ''
    body = result.get('body', '') or result.get('content', '') or ''
    if not all(item in url for item in rules.get('url_must_include', [])):
        return 0.0
    if not all(item in body for item in rules.get('body_must_include', [])):
        return 0.0
    if any(item in body for item in rules.get('body_must_exclude', [])):
        return 0.0
    return 1.0


@metric_registry.register()
def is_extension_installed(actual: str | None, rules: dict[str, Any], **options: Any) -> float:
    if actual is None:
        actual = ''

    expected = str(rules['expected'])
    rule_type = rules.get('type')
    if rule_type == 'contain':
        return 1.0 if expected in actual else 0.0
    if rule_type == 'not_contain':
        return 1.0 if expected not in actual else 0.0
    raise NotImplementedError(f'Unsupported is_extension_installed rule type: {rule_type}')


@metric_registry.register()
def check_config_status(actual_config_path: str | None, rule: dict[str, Any]) -> float:
    if actual_config_path is None:
        return 0.0

    try:
        lines = Path(actual_config_path).read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return 0.0

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        items = line.lstrip('(').rstrip(')').split()
        if not items:
            continue

        key = rule.get('key')
        if isinstance(key, str):
            if items[0] == key and items[-1] == rule.get('value'):
                return 1.0
        elif (
            isinstance(key, list)
            and len(key) == 2
            and len(items) >= 3
            and items[0] == key[0]
            and items[1] == key[1]
            and items[-1] == rule.get('value')
        ):
            return 1.0
    return 0.0


@metric_registry.register()
def check_qt_bgcone(actual_config_path: str | None, rule: dict[str, Any]) -> float:
    if actual_config_path is None:
        return 0.0

    try:
        config_file = Path(actual_config_path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return 0.0

    expected_qt_bgcone = str(rule['expected_qt_bgcone'])
    qt_bgcone = '1'
    for line in config_file.splitlines():
        if 'qt-bgcone=' in line:
            qt_bgcone = line.split('=')[-1].strip()
    return 1.0 if qt_bgcone == expected_qt_bgcone else 0.0


@metric_registry.register()
def check_auto_saving_time(config_path: str | None, rules: dict[str, Any]) -> float:
    if config_path is None:
        return 0.0

    try:
        root = ET.parse(config_path).getroot()
    except (OSError, ET.ParseError):
        return 0.0

    for item in root.findall('.//item'):
        path = item.get('{http://openoffice.org/2001/registry}path')
        if path != '/org.openoffice.Office.Common/Save/Document':
            continue
        for prop in item.findall('.//prop'):
            name = prop.get('{http://openoffice.org/2001/registry}name')
            if name != 'AutoSaveTimeIntervall':
                continue
            value = prop.find('.//value')
            if value is None or value.text is None:
                return 0.0
            try:
                return 1.0 if int(value.text) == int(rules['minutes']) else 0.0
            except ValueError:
                return 0.0
    return 0.0
