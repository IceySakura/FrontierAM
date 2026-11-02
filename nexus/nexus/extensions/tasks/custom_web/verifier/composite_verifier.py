"""
复合任务验证器

组合多个基础验证器，用于验证复杂的多步骤任务。
按 type 字段匹配 groundtruth 和 response，不依赖顺序。
"""

from typing import Any

from .action_verifier import ActionVerifier
from .base import BaseVerifier, VerificationResult
from .navigation_verifier import NavigationVerifier
from .retrieval_verifier import RetrievalVerifier
from .state_verifier import StateVerifier


# 验证器类型映射
VERIFIER_MAP: dict[str, type[BaseVerifier]] = {
    'navigation': NavigationVerifier,
    'retrieval': RetrievalVerifier,
    'state': StateVerifier,
    'action': ActionVerifier,
}


class CompositeVerifier(BaseVerifier):
    """
    复合任务验证器

    组合多个验证步骤，按 type 字段匹配 groundtruth 和 response。

    Groundtruth 格式:
    {
        "groundtruth": [
            {
                "type": "navigation",
                "expected_pages": ["/home", "/merchant/chayanyuese"],
                "options": {"mode": "contains"}
            },
            {
                "type": "state",
                "expected_states": {
                    "cart": [{"name": "幽兰拿铁", "quantity": 1}],
                    "temp_data": null
                },
                "options": {"mode": "include"}
            }
        ]
    }
    """

    def __init__(self, name: str | None = None):
        """初始化复合验证器"""
        super().__init__(name or 'CompositeVerifier')
        self._verifiers: dict[str, BaseVerifier] = {}

    def _get_verifier(self, verifier_type: str) -> BaseVerifier:
        """获取或创建指定类型的验证器"""
        if verifier_type not in self._verifiers:
            verifier_class = VERIFIER_MAP.get(verifier_type)
            if verifier_class is None:
                raise ValueError(f'未知的验证器类型: {verifier_type}')
            self._verifiers[verifier_type] = verifier_class()

        return self._verifiers[verifier_type]

    def _build_response_map(self, response_list: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """构建按 type 索引的响应映射"""
        response_map = {}
        for item in response_list:
            item_type = item.get('type')
            if item_type:
                response_map[item_type] = item
        return response_map

    def _extract_groundtruth_data(self, item: dict[str, Any]) -> dict[str, Any]:
        """从扁平化的 groundtruth 项中提取验证数据"""
        return {k: v for k, v in item.items() if k not in ('type', 'options')}

    def _extract_response_data(self, item: dict[str, Any]) -> dict[str, Any]:
        """从扁平化的 response 项中提取响应数据"""
        return {k: v for k, v in item.items() if k != 'type'}

    async def verify(
        self,
        groundtruth: dict[str, Any],
        response: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """执行复合验证"""
        options = options or {}
        stop_on_failure = options.get('stop_on_failure', False)

        gt_list = groundtruth.get('groundtruth', [])
        resp_list = response.get('response', [])

        if not gt_list:
            return VerificationResult.failed('groundtruth 列表为空', details={'groundtruth': groundtruth})

        response_map = self._build_response_map(resp_list)

        results = []
        failed_types = []
        passed_types = []

        for item in gt_list:
            item_type = item.get('type')
            item_options = item.get('options', {})

            if not item_type:
                result = VerificationResult.failed('验证项缺少 type 字段')
                results.append({'type': None, 'result': result.to_dict()})
                failed_types.append('unknown')
                continue

            if item_type not in VERIFIER_MAP:
                result = VerificationResult.failed(f'未知的验证类型: {item_type}')
                results.append({'type': item_type, 'result': result.to_dict()})
                failed_types.append(item_type)
                continue

            resp_item = response_map.get(item_type, {})
            gt_data = self._extract_groundtruth_data(item)
            resp_data = self._extract_response_data(resp_item)

            try:
                verifier = self._get_verifier(item_type)
                result = await verifier.verify(gt_data, resp_data, item_options)
            except Exception as e:
                result = VerificationResult.failed(f'验证异常: {str(e)}')

            results.append({
                'type': item_type,
                'result': result.to_dict(),
            })

            if result.success:
                passed_types.append(item_type)
            else:
                failed_types.append(item_type)
                if stop_on_failure:
                    break

        total_count = len(gt_list)
        all_passed = len(failed_types) == 0 and len(passed_types) == total_count
        final_score = 1.0 if all_passed else 0.0

        details = {
            'total_count': total_count,
            'passed_count': len(passed_types),
            'failed_count': len(failed_types),
            'passed_types': passed_types,
            'failed_types': failed_types,
            'results': results,
        }

        if all_passed:
            return VerificationResult.passed(f'所有 {total_count} 项验证通过', details=details)

        return VerificationResult.failed(
            f'验证失败: {len(passed_types)}/{total_count} 项通过', score=final_score, details=details
        )

    async def verify_single(
        self,
        item_type: str,
        groundtruth: dict[str, Any],
        response: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """验证单个项（便捷方法）"""
        if item_type not in VERIFIER_MAP:
            return VerificationResult.failed(f'未知的验证类型: {item_type}')

        verifier = self._get_verifier(item_type)
        return await verifier.verify(groundtruth, response, options)
