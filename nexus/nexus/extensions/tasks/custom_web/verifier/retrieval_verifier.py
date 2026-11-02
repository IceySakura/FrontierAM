"""
信息检索验证器

验证检索到的信息是否与预期语义等价。
使用 LLM 进行语义比较，支持文本、结构化数据的验证。
"""

import json
import logging

from typing import Any

from .base import BaseVerifier, VerificationResult


logger = logging.getLogger(__name__)


async def llm_semantic_match(pred: str, reference: str, question: str = '') -> dict[str, Any]:
    """使用 LLM 判断两段内容是否语义等价。

    参考 webarena 的 llm_fuzzy_match 实现。

    Args:
        pred: 实际内容
        reference: 期望内容
        question: 任务题目，提供判断上下文

    Returns:
        dict 包含 equivalent, reason
    """
    from nexus.core import require_context

    message = (
        '你是一个专业的答案判断助手。请判断实际回复是否正确回答了题目的问题。\n\n'
        '## 核心原则\n'
        '**以题目为准**：只判断实际回复是否回答了题目直接提出的问题。期望回复仅作为参考答案，用于验证核心信息的正确性。\n\n'
        '## 重要背景\n'
        '实际回复来自一个被要求**简洁作答**的自动化Agent，该Agent被指示：\n'
        '- 只提供直接答案，不加任何解释或上下文\n'
        '- 例如问"X的价格是多少？"→ 回复"$19.99"（而不是"X的价格是$19.99"）\n'
        '- 例如问"Y的名称是什么？"→ 回复"产品名"（而不是"名称是产品名"）\n'
        '- 是否类问题 → 回复"yes"或"no"\n'
        '因此，实际回复通常非常简洁，只包含核心答案值。\n\n'
        '## 判断规则\n'
        '1. **提取题目问题**：首先明确题目到底在问什么（可能是数量、价格、名称、状态、列表等）。\n'
        '2. **核心信息匹配**：实际回复中的核心答案是否与期望回复中的对应信息一致。\n'
        '3. **简洁回复等价**：简洁回复与完整表述等价，例如：\n'
        '   - "$19.99" = "价格是$19.99" = "19.99元"\n'
        '   - "产品A" = "名称是产品A" = "产品A商品"\n'
        '   - "3" = "共3个" = "数量为3"\n'
        '   - "¥18" = "18元" = "价格18元" = "幽兰拿铁价格: 18元"\n'
        '4. **格式宽容**：数值等价即可（如 ¥328 = 328元 = 328），格式、符号、单位表达可以不同。\n'
        '5. **回复完整性**：实际回复只需包含题目问的核心信息，不要求包含期望回复中的额外描述（如背景、来源、补充说明等）。\n'
        '6. **信息充分性**：如果题目问了多个问题，实际回复需要回答所有问题；如果只问了一个问题，回答该问题即可。\n\n'
    )
    if question:
        message += f'## 题目\n{question}\n\n'
    message += f'期望回复：{reference}\n\n'
    message += f'实际回复：{pred}\n\n'
    message += '请以 JSON 格式输出：{"equivalent": true/false, "reason": "判断理由"}'

    messages = [
        {'role': 'system', 'content': '你是一个精确的语义比较助手，只输出规定的JSON内容。'},
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
                'reason': result.get('reason', ''),
            }
    except json.JSONDecodeError as e:
        logger.warning(f'LLM 响应解析失败: {e}')
    except Exception as e:
        logger.warning(f'LLM 调用失败: {e}')

    return {'equivalent': False, 'reason': '比较失败'}


class RetrievalVerifier(BaseVerifier):
    """
    信息检索验证器

    使用 LLM 进行语义比较，验证检索结果是否正确。

    Groundtruth 格式:
    {
        "expected_content": "期望的内容",
        "key_points": ["关键点1", "关键点2"]  # 可选
    }

    Response 格式:
    {
        "executed_content": "实际检索到的内容",
        "question": "任务题目描述"
    }

    Options:
    {
        "similarity_threshold": 0.8,
        "check_key_points": true
    }
    """

    def __init__(self, name: str | None = None):
        """初始化检索验证器"""
        super().__init__(name or 'RetrievalVerifier')

    async def verify(
        self,
        groundtruth: dict[str, Any],
        response: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """执行检索验证"""
        options = options or {}
        threshold = options.get('similarity_threshold', 0.8)
        check_key_points = options.get('check_key_points', True)

        expected_content = groundtruth.get('expected_content')
        executed_content = response.get('executed_content')
        question = response.get('question', '')

        if expected_content is None:
            return VerificationResult.failed('groundtruth 缺少 expected_content', details={'groundtruth': groundtruth})

        if executed_content is None:
            return VerificationResult.failed('response 缺少 executed_content', details={'response': response})

        # 转换内容为字符串
        expected_str = self._content_to_string(expected_content)
        actual_str = self._content_to_string(executed_content)

        # 主内容语义比较
        compare_result = await llm_semantic_match(actual_str, expected_str, question)

        is_equivalent = compare_result.get('equivalent', False)

        # 检查关键点
        key_points = groundtruth.get('key_points', [])
        key_points_result = None
        key_points_score = 1.0

        if check_key_points and key_points:
            key_points_result = await self._check_key_points(key_points, actual_str, question)
            key_points_score = key_points_result['score']

        # 构建详情
        details = {
            'expected_content': expected_str[:500] + '...' if len(expected_str) > 500 else expected_str,
            'executed_content': actual_str[:500] + '...' if len(actual_str) > 500 else actual_str,
            'semantic_comparison': compare_result,
        }

        if key_points_result:
            details['key_points_check'] = key_points_result

        # 判断是否通过：只根据 equivalent 判断
        if is_equivalent:
            # 如果有 key_points，结合 key_points 得分
            if key_points and key_points_score >= threshold:
                return VerificationResult.passed(
                    f'检索内容语义匹配，关键点得分: {key_points_score:.2f}', details=details
                )
            if key_points and key_points_score < threshold:
                return VerificationResult.partial(
                    key_points_score, f'检索内容语义匹配，但关键点得分不足: {key_points_score:.2f}', details=details
                )
            return VerificationResult.passed('检索内容语义匹配', details=details)
        return VerificationResult.failed(
            f'检索内容不匹配: {compare_result.get("reason", "")}', score=0.0, details=details
        )

    def _content_to_string(self, content: Any) -> str:
        """将内容转换为字符串"""
        if isinstance(content, str):
            return content
        if isinstance(content, (dict, list)):
            return json.dumps(content, ensure_ascii=False, indent=2)
        return str(content)

    async def _check_key_points(self, key_points: list, content: str, question: str = '') -> dict[str, Any]:
        """检查关键点是否在内容中出现"""
        matched = []
        missing = []

        for point in key_points:
            result = await llm_semantic_match(content, point, question)

            if result.get('equivalent', False):
                matched.append(point)
            else:
                missing.append(point)

        score = len(matched) / len(key_points) if key_points else 1.0

        return {
            'total': len(key_points),
            'matched': matched,
            'missing': missing,
            'score': score,
        }
