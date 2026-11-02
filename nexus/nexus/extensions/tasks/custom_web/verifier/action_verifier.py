"""
动作触发验证器

验证动作序列是否正确执行。
支持三种匹配模式：final（最终动作）、contains（包含）、exact（精确匹配）。
"""

from typing import Any

from .base import BaseVerifier, VerificationResult


class ActionVerifier(BaseVerifier):
    """
    动作触发验证器

    验证动作序列是否按预期执行。

    Groundtruth 格式:
    {
        "expected_actions": ["click_product", "click_add_to_cart", "click_checkout"],
        "options": {"mode": "contains"}
    }

    Response 格式:
    {
        "executed_actions": ["click_product", "scroll_down", "click_add_to_cart", "click_cart", "click_checkout"]
    }

    Options:
    {
        "mode": "final" | "contains" | "exact"  # 匹配模式，默认 final
    }

    验证模式说明:
    - final: 只比较末元素，验证最后一个动作
    - contains: expected_actions 是 executed_actions 的有序子序列
    - exact: 两个列表完全相等
    """

    def __init__(self, name: str | None = None):
        super().__init__(name or 'ActionVerifier')

    async def verify(
        self,
        groundtruth: dict[str, Any],
        response: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """执行动作验证"""
        options = options or {}
        mode = options.get('mode', 'final')

        expected_actions = groundtruth.get('expected_actions', [])
        executed_actions = response.get('executed_actions', [])

        if not expected_actions:
            return VerificationResult.failed('groundtruth 缺少 expected_actions', details={'groundtruth': groundtruth})

        if not executed_actions:
            return VerificationResult.failed('没有执行任何动作', details={'expected_actions': expected_actions})

        # 标准化动作名称（转小写）
        expected_normalized = [a.lower().strip() for a in expected_actions]
        executed_normalized = [a.lower().strip() for a in executed_actions]

        # 根据模式执行验证
        if mode == 'final':
            return self._verify_final(expected_normalized, executed_normalized)
        if mode == 'exact':
            return self._verify_exact(expected_normalized, executed_normalized)
        # contains
        return self._verify_contains(expected_normalized, executed_normalized)

    def _verify_final(
        self,
        expected: list[str],
        executed: list[str],
    ) -> VerificationResult:
        """验证最终动作（只比较末元素）"""
        expected_final = expected[-1]
        executed_final = executed[-1]

        if expected_final == executed_final:
            return VerificationResult.passed(
                f'最终动作匹配: {expected_final}',
                details={
                    'expected_final': expected_final,
                    'executed_final': executed_final,
                },
            )
        return VerificationResult.failed(
            f'最终动作不匹配，期望: {expected_final}，实际: {executed_final}',
            details={
                'expected_final': expected_final,
                'executed_final': executed_final,
            },
        )

    def _verify_exact(
        self,
        expected: list[str],
        executed: list[str],
    ) -> VerificationResult:
        """严格序列匹配"""
        if expected == executed:
            return VerificationResult.passed(
                '动作序列完全匹配',
                details={
                    'expected_actions': expected,
                    'executed_actions': executed,
                },
            )

        # 计算部分匹配得分
        match_count = 0
        for i, action in enumerate(expected):
            if i < len(executed) and executed[i] == action:
                match_count += 1

        score = match_count / len(expected) if expected else 0.0

        return VerificationResult.partial(
            score,
            f'动作序列部分匹配: {match_count}/{len(expected)}',
            details={
                'expected_actions': expected,
                'executed_actions': executed,
                'match_count': match_count,
            },
        )

    def _verify_contains(
        self,
        expected: list[str],
        executed: list[str],
    ) -> VerificationResult:
        """包含模式匹配 - 检查关键动作是否按顺序出现"""
        if not expected:
            return VerificationResult.passed('空动作序列，验证通过')

        # 使用双指针检查子序列
        expected_idx = 0
        matched_actions = []

        for executed_action in executed:
            if expected_idx < len(expected) and executed_action == expected[expected_idx]:
                matched_actions.append(executed_action)
                expected_idx += 1

        if expected_idx == len(expected):
            return VerificationResult.passed(
                '动作序列包含所有关键动作',
                details={
                    'expected_actions': expected,
                    'executed_actions': executed,
                    'matched_actions': matched_actions,
                },
            )

        # 计算部分匹配得分
        score = expected_idx / len(expected)
        missing_actions = expected[expected_idx:]

        return VerificationResult.partial(
            score,
            f'动作序列缺少: {missing_actions}',
            details={
                'expected_actions': expected,
                'executed_actions': executed,
                'matched_actions': matched_actions,
                'missing_actions': missing_actions,
            },
        )
