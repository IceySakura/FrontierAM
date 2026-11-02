"""
Verifier Module - 网站操作任务验证器

提供四种基础验证类型：
1. NavigationVerifier - 页面导航验证
2. RetrievalVerifier - 信息检索验证
3. StateVerifier - 状态变更验证
4. ActionVerifier - 动作触发验证

以及组合验证器：
- CompositeVerifier - 复合任务验证
"""

from .action_verifier import ActionVerifier
from .base import BaseVerifier, VerificationResult
from .composite_verifier import CompositeVerifier
from .navigation_verifier import NavigationVerifier
from .retrieval_verifier import RetrievalVerifier
from .state_verifier import StateVerifier


__all__ = [
    'VerificationResult',
    'BaseVerifier',
    # 验证器
    'NavigationVerifier',
    'RetrievalVerifier',
    'StateVerifier',
    'ActionVerifier',
    'CompositeVerifier',
]
