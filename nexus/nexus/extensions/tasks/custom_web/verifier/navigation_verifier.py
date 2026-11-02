"""
页面导航验证器

验证页面导航是否到达目标页面或按预期路径序列执行。

支持三种匹配模式：
- final: 只验证最终页面（默认）
- contains: 路径包含关键节点
- exact: 严格路径序列匹配
"""

from typing import Any

from .base import (
    BaseVerifier,
    VerificationResult,
    extract_path_from_url,
)


class NavigationVerifier(BaseVerifier):
    """
    页面导航验证器

    验证用户是否正确导航到目标页面，支持多种验证模式。

    Groundtruth 格式:
    {
        "expected_pages": ["/home", "/category", "/merchant"],  # 期望的页面路径列表
        "options": {"mode": "final"}                            # 验证模式
    }

    Response 格式:
    {
        "executed_pages": ["/home", "/search", "/category", "/merchant"]  # 实际导航路径
    }

    Options:
    {
        "mode": "final" | "contains" | "exact"  # 匹配模式，默认 final
    }

    验证模式说明:
    - final: 只比较末元素，验证最终页面
    - contains: expected_pages 是 executed_pages 的有序子序列
    - exact: 两个列表完全相等
    """

    def __init__(self, name: str | None = None):
        super().__init__(name or 'NavigationVerifier')

    async def verify(
        self,
        groundtruth: dict[str, Any],
        response: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """执行导航验证"""
        options = options or {}
        mode = options.get('mode', 'final')

        expected_pages = groundtruth.get('expected_pages', [])
        executed_pages = response.get('executed_pages', [])

        if not expected_pages:
            return VerificationResult.failed('groundtruth 缺少 expected_pages', details={'groundtruth': groundtruth})

        if not executed_pages:
            return VerificationResult.failed(
                'response 缺少 executed_pages', details={'expected_pages': expected_pages, 'response': response}
            )

        # 标准化路径
        expected_normalized = [extract_path_from_url(p) for p in expected_pages]
        executed_normalized = [extract_path_from_url(p) for p in executed_pages]

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
        """验证最终页面（只比较末元素）"""
        expected_final = expected[-1]
        executed_final = executed[-1]

        if expected_final == executed_final:
            return VerificationResult.passed(
                f'成功到达最终页面: {expected_final}',
                details={
                    'expected_final': expected_final,
                    'executed_final': executed_final,
                },
            )
        return VerificationResult.failed(
            f'最终页面不匹配，期望: {expected_final}，实际: {executed_final}',
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
                '路径序列完全匹配',
                details={
                    'expected_pages': expected,
                    'executed_pages': executed,
                },
            )

        # 计算部分匹配得分
        match_count = 0
        for i, path in enumerate(expected):
            if i < len(executed) and executed[i] == path:
                match_count += 1

        score = match_count / len(expected) if expected else 0.0

        return VerificationResult.partial(
            score,
            f'路径序列部分匹配: {match_count}/{len(expected)}',
            details={
                'expected_pages': expected,
                'executed_pages': executed,
                'match_count': match_count,
            },
        )

    def _verify_contains(
        self,
        expected: list[str],
        executed: list[str],
    ) -> VerificationResult:
        """包含模式匹配 - 检查关键节点是否按顺序出现"""
        if not expected:
            return VerificationResult.passed('空路径序列，验证通过')

        # 使用双指针检查子序列
        expected_idx = 0
        matched_paths = []

        for executed_path in executed:
            if expected_idx < len(expected) and executed_path == expected[expected_idx]:
                matched_paths.append(executed_path)
                expected_idx += 1

        if expected_idx == len(expected):
            return VerificationResult.passed(
                '路径包含所有关键节点',
                details={
                    'expected_pages': expected,
                    'executed_pages': executed,
                    'matched_paths': matched_paths,
                },
            )

        # 计算部分匹配得分
        score = expected_idx / len(expected)
        missing_paths = expected[expected_idx:]

        return VerificationResult.partial(
            score,
            f'路径缺少关键节点: {missing_paths}',
            details={
                'expected_pages': expected,
                'executed_pages': executed,
                'matched_paths': matched_paths,
                'missing_paths': missing_paths,
            },
        )
