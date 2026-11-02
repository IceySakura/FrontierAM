"""
状态变更验证器

验证系统状态/数据字段的变更是否符合预期。
使用 LLM 进行语义比较，判断状态变更是否满足任务要求。
"""

import json
import logging

from typing import Any

from .base import BaseVerifier, VerificationResult


logger = logging.getLogger(__name__)


async def llm_state_match(
    expected_states: dict[str, Any],
    actual_changes: dict[str, Any],
    question: str = '',
    mode: str = 'include',
) -> dict[str, Any]:
    """使用 LLM 判断状态变更是否符合预期。

    Args:
        expected_states: 期望状态变更
        actual_changes: 实际状态变更
        question: 任务题目
        mode: 验证模式 ('include' 或 'exact')

    Returns:
        dict 包含 equivalent, mismatched_fields, reason
    """
    from nexus.core import require_context

    expected_str = json.dumps(expected_states, ensure_ascii=False, indent=2)
    actual_str = json.dumps(actual_changes, ensure_ascii=False, indent=2) if actual_changes else '（无状态变更）'

    if mode == 'include' or mode == 'exact':
        pass

    message = (
        '你是一个专业的状态验证助手。请判断实际状态变更是否满足题目要求。\n\n'
        '## 重要概念说明\n'
        '- **预期关键状态**：表示操作完成后，该字段应该保留/存在的最终关键内容，可能省略某些子字段（有时也可能会用自然语言描述）。\n'
        '- **实际状态变更**：记录了实际发生的变更操作，包含 change_type（added/removed/modified）表示变更类型。\n\n'
        '## 验证规则\n'
        '1. 核心标准：实际状态变更是否完成了题目要求的操作。\n'
        '2. **关键理解**：预期关键状态给出的是操作后应该**保留**的内容。如果题目要求"删除/移除/取消"某项，那么预期关键状态会列出**剩余**的项目，而实际状态变更中应该有对应项目的 "removed" 记录。\n'
        '   - 例如：题目要求"取消收藏商品A"，预期关键状态列出保留的商品B和C，实际状态变更显示商品A被removed，这表示操作正确。\n'
        '3. 实际状态中可能有额外的字段或额外的内容（如系统自动添加的前缀、ID、时间戳、记录等），只要核心内容正确，额外信息不影响判断。\n'
        '4. 对于列表/数组，关注元素是否匹配，顺序可以不同。\n'
        '5. 如果预期关键状态为空，表示应该没有变化，此时如果实际状态变更不为空，则算错。\n'
        '6. 判断时需要结合题目语义：如果题目要求删除/移除某项，实际变更中该项被 removed 就是正确的；如果题目要求添加某项，实际变更中该项被 added 就是正确的。\n\n'
        f'## 题目\n{question or "（未提供题目）"}\n\n'
        f'## 预期关键状态（操作后应保留的内容）\n{expected_str}\n\n'
        f'## 实际状态变更（发生的变更操作记录）\n{actual_str}\n\n'
        '请以 JSON 格式输出：{"equivalent": true/false, "mismatched_fields": [], "reason": "判断理由"}'
    )

    messages = [
        {'role': 'system', 'content': '你是一个精确的状态验证助手，只输出 JSON。'},
        {'role': 'user', 'content': message},
    ]

    try:
        ctx = require_context()
        client = ctx.get_openai_client('judge')
        response = await client.create_completion(messages=messages, temperature=0.0)
        result_text = response.choices[0].message.content

        if result_text:
            # Remove markdown code block markers
            text = result_text.strip()
            if text.startswith('```JSON') or text.startswith('```json'):
                text = text[7:]
            elif text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            result = json.loads(text.strip())
            return {
                'equivalent': result.get('equivalent', False),
                'mismatched_fields': result.get('mismatched_fields', []),
                'reason': result.get('reason', ''),
            }
    except json.JSONDecodeError as e:
        logger.warning(f'LLM 响应解析失败: {e}')
    except Exception as e:
        logger.warning(f'LLM 调用失败: {e}')

    return {'equivalent': False, 'mismatched_fields': [], 'reason': '比较失败'}


def _get_list_item_id(item: Any) -> str | None:
    """尝试获取列表元素的唯一标识符"""
    if isinstance(item, dict):
        for id_field in ['id', 'ID', '_id', 'orderId', 'orderNo', 'productId', 'userId']:
            if id_field in item:
                return str(item[id_field])
    return None


def _compute_list_diff(initial: list, current: list, path: str) -> dict[str, Any]:
    """计算两个列表之间的差异"""
    diff: dict[str, Any] = {}

    initial_by_id: dict[str, tuple[int, Any]] = {}
    current_by_id: dict[str, tuple[int, Any]] = {}
    initial_no_id: list[tuple[int, Any]] = []
    current_no_id: list[tuple[int, Any]] = []

    for idx, item in enumerate(initial):
        item_id = _get_list_item_id(item)
        if item_id:
            initial_by_id[item_id] = (idx, item)
        else:
            initial_no_id.append((idx, item))

    for idx, item in enumerate(current):
        item_id = _get_list_item_id(item)
        if item_id:
            current_by_id[item_id] = (idx, item)
        else:
            current_no_id.append((idx, item))

    if len(initial_by_id) == 0 and len(current_by_id) == 0:
        if initial != current:
            diff[path] = {'change_type': 'modified', 'from': initial, 'to': current}
        return diff

    # 新增的元素
    added_ids = set(current_by_id.keys()) - set(initial_by_id.keys())
    for item_id in added_ids:
        idx, item = current_by_id[item_id]
        diff[f'{path}[{idx}]'] = {'change_type': 'added', 'item': item}

    # 删除的元素
    removed_ids = set(initial_by_id.keys()) - set(current_by_id.keys())
    for item_id in removed_ids:
        idx, item = initial_by_id[item_id]
        diff[f'{path}[{idx}]'] = {'change_type': 'removed', 'item': item}

    # 修改的元素
    common_ids = set(initial_by_id.keys()) & set(current_by_id.keys())
    for item_id in common_ids:
        _, old_item = initial_by_id[item_id]
        new_idx, new_item = current_by_id[item_id]
        if old_item != new_item and isinstance(old_item, dict) and isinstance(new_item, dict):
            item_diff = _compute_state_diff(old_item, new_item, f'{path}[{new_idx}]')
            diff.update(item_diff)

    return diff


def _compute_state_diff(initial: Any, current: Any, path: str = '') -> dict[str, Any]:
    """计算两个状态之间的差异"""
    diff: dict[str, Any] = {}

    if initial is None:
        initial = {}
    if current is None:
        current = {}

    if type(initial) is not type(current):
        field_key = path if path else '_root'
        diff[field_key] = {'change_type': 'type_changed', 'from': type(initial).__name__, 'to': type(current).__name__}
        return diff

    if isinstance(initial, dict) and isinstance(current, dict):
        all_keys = set(initial.keys()) | set(current.keys())

        for key in all_keys:
            field_path = f'{path}.{key}' if path else key
            old_val = initial.get(key)
            new_val = current.get(key)

            if key not in initial:
                diff[field_path] = {'change_type': 'added', 'value': new_val}
            elif key not in current:
                diff[field_path] = {'change_type': 'removed', 'value': old_val}
            elif old_val != new_val:
                if isinstance(old_val, dict) and isinstance(new_val, dict):
                    nested_diff = _compute_state_diff(old_val, new_val, field_path)
                    diff.update(nested_diff)
                elif isinstance(old_val, list) and isinstance(new_val, list):
                    list_diff = _compute_list_diff(old_val, new_val, field_path)
                    diff.update(list_diff)
                else:
                    diff[field_path] = {'change_type': 'modified', 'from': old_val, 'to': new_val}

    elif isinstance(initial, list) and isinstance(current, list):
        if initial != current:
            field_key = path if path else '_root'
            list_diff = _compute_list_diff(initial, current, field_key)
            diff.update(list_diff)

    elif initial != current:
        field_key = path if path else '_root'
        diff[field_key] = {'change_type': 'modified', 'from': initial, 'to': current}

    return diff


class StateVerifier(BaseVerifier):
    """
    状态变更验证器

    使用 LLM 进行语义比较，验证状态变更是否正确。

    Groundtruth 格式:
    {
        "expected_states": {"cart": [{"name": "幽兰拿铁", "quantity": 1}]},
        "options": {"mode": "include", "similarity_threshold": 0.8}
    }

    Response 格式:
    {
        "executed_states": {"initial_state": {...}, "current_state": {...}},
        "question": "任务题目描述"
    }
    """

    def __init__(self, name: str | None = None):
        """初始化状态验证器"""
        super().__init__(name or 'StateVerifier')

    async def verify(
        self,
        groundtruth: dict[str, Any],
        response: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """执行状态验证"""
        options = options or {}
        mode = options.get('mode', 'include')

        expected_states = groundtruth.get('expected_states', {})
        executed_states = response.get('executed_states', {})
        question = response.get('question', '')

        if not expected_states:
            return VerificationResult.failed('groundtruth 缺少 expected_states', details={'groundtruth': groundtruth})

        if executed_states is None:
            return VerificationResult.failed(
                'response 缺少 executed_states', details={'expected_states': expected_states, 'response': response}
            )

        initial_state = executed_states.get('initial_state', {})
        current_state = executed_states.get('current_state', executed_states)

        logger.debug(f'StateVerifier: initial_state type: {type(initial_state).__name__}')
        logger.debug(f'StateVerifier: current_state type: {type(current_state).__name__}')
        logger.debug(f'StateVerifier: states are identical: {initial_state == current_state}')

        # 计算实际状态变更
        actual_changes = _compute_state_diff(initial_state, current_state)

        logger.debug(f'StateVerifier: computed {len(actual_changes)} state changes')
        if actual_changes:
            logger.debug(f'StateVerifier: changed fields: {list(actual_changes.keys())[:10]}')

        # 使用 LLM 进行状态比较
        compare_result = await llm_state_match(expected_states, actual_changes, question, mode)

        is_equivalent = compare_result.get('equivalent', False)

        details = {
            'expected_changes': expected_states,
            'actual_changes': actual_changes,
            'question': question,
            'mode': mode,
            'llm_comparison': compare_result,
        }

        if is_equivalent:
            return VerificationResult.passed('状态验证通过', details=details)
        return VerificationResult.failed(
            f'状态不匹配: {compare_result.get("reason", "未知原因")}', score=0.0, details=details
        )
