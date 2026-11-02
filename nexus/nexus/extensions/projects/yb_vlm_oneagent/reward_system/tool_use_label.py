"""Tool-use label reward for Yb VLM oneagent."""

from __future__ import annotations

import json

from typing import cast


JsonDict = dict[str, object]

_TEXT_SEARCH_TOOL = 'web_search'
_IMAGE_SEARCH_TOOL = 'vision_search'
_OPTIONAL_LABEL = '可搜可不搜'
_LABELS_MAP: dict[object, int] = {'是': 1, '否': 0, '1': 1, '0': 0, 1: 1, 0: 0, '': 0}


def _load_label(target_answer: str) -> JsonDict:
    if not target_answer.strip():
        return {}
    try:
        parsed = cast(object, json.loads(target_answer))
    except json.JSONDecodeError:
        return {}
    return cast(JsonDict, parsed) if isinstance(parsed, dict) else {}


def _extract_tool_names(trajectory: list[JsonDict] | None) -> list[str]:
    names: list[str] = []
    if not isinstance(trajectory, list):
        return names

    for msg in trajectory:
        if msg.get('role') != 'assistant':
            continue
        raw_tool_calls = msg.get('tool_calls')
        if not isinstance(raw_tool_calls, list):
            continue
        tool_calls = cast(list[object], raw_tool_calls)
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call_dict = cast(JsonDict, tool_call)
            raw_function = tool_call_dict.get('function')
            function_dict = cast(JsonDict, raw_function) if isinstance(raw_function, dict) else {}
            name = function_dict.get('name')
            if isinstance(name, str) and name:
                names.append(name)
    return names


def _score_one_tool(label: object, called: bool) -> float:
    if label == _OPTIONAL_LABEL:
        return 0.5
    expected_called = _LABELS_MAP[label] == 1
    return 0.5 if called == expected_called else 0.0


def _require_tool_label(labels: JsonDict, key: str) -> object:
    if key not in labels:
        raise ValueError(f'missing required tool-use label: {key}')

    label = labels[key]
    if label != _OPTIONAL_LABEL and label not in _LABELS_MAP:
        raise ValueError(f'invalid tool-use label for {key}: {label!r}')
    return label


def _tool_pr_counts(label: object, called: bool) -> dict[str, float]:
    gt = _LABELS_MAP.get(label)
    if gt is None:
        return {'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'tn': 0.0, 'total': 0.0}
    pred = 1 if called else 0
    return {
        'tp': float(gt == 1 and pred == 1),
        'fp': float(gt == 0 and pred == 1),
        'fn': float(gt == 1 and pred == 0),
        'tn': float(gt == 0 and pred == 0),
        'total': 1.0,
    }


async def tool_use_label_reward(
    question: str,
    response: str,
    target_answer: str,
    trajectory: list[JsonDict] | None = None,
    **kwargs: object,
) -> tuple[float, dict[str, object]]:
    """Score whether the trajectory used the labeled search tools."""
    _ = question, response, kwargs
    labels = _load_label(target_answer)
    tool_names = _extract_tool_names(trajectory)
    called_tools = set(tool_names)

    txt_label = _require_tool_label(labels, 'txt2txt_search')
    img_label = _require_tool_label(labels, 'img2img_search')
    text_search_called = _TEXT_SEARCH_TOOL in called_tools
    image_search_called = _IMAGE_SEARCH_TOOL in called_tools

    text_score = _score_one_tool(txt_label, text_search_called)
    image_score = _score_one_tool(img_label, image_search_called)
    score = text_score + image_score
    text_pr_counts = _tool_pr_counts(txt_label, text_search_called)
    image_pr_counts = _tool_pr_counts(img_label, image_search_called)

    details: dict[str, object] = {
        'reward_func': 'tool_use_label',
        'score': score,
        'text_score': text_score,
        'image_score': image_score,
        'txt2txt_search': txt_label,
        'img2img_search': img_label,
        'text_search_tool': _TEXT_SEARCH_TOOL,
        'image_search_tool': _IMAGE_SEARCH_TOOL,
        'tool_names': tool_names,
    }
    for key, value in text_pr_counts.items():
        details[f'toolcall_prf/web_search/{key}'] = value
    for key, value in image_pr_counts.items():
        details[f'toolcall_prf/vision_search/{key}'] = value
    return score, details


__all__ = ['tool_use_label_reward']
