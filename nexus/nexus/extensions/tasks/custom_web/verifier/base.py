"""
基础类和工具函数
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerificationResult:
    """
    验证结果数据类

    Attributes:
        success: 验证是否通过
        score: 验证得分 (0.0 - 1.0)
        message: 结果描述信息
        details: 详细信息字典
    """

    success: bool
    score: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """确保 score 在有效范围内"""
        self.score = max(0.0, min(1.0, self.score))

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'score': self.score,
            'message': self.message,
            'details': self.details,
        }

    @classmethod
    def passed(cls, message: str = '验证通过', details: dict | None = None) -> 'VerificationResult':
        """创建通过结果"""
        return cls(success=True, score=1.0, message=message, details=details or {})

    @classmethod
    def failed(cls, message: str, score: float = 0.0, details: dict | None = None) -> 'VerificationResult':
        """创建失败结果"""
        return cls(success=False, score=score, message=message, details=details or {})

    @classmethod
    def partial(cls, score: float, message: str, details: dict | None = None) -> 'VerificationResult':
        """创建部分通过结果 - 只有 score=1.0 才算 success"""
        return cls(success=score >= 1.0, score=score, message=message, details=details or {})


class BaseVerifier(ABC):
    """
    验证器基类

    所有具体验证器必须继承此类并实现 verify 方法
    """

    def __init__(self, name: str | None = None):
        """
        初始化验证器

        Args:
            name: 验证器名称，默认使用类名
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def verify(
        self, groundtruth: dict[str, Any], response: dict[str, Any], options: dict[str, Any] | None = None
    ) -> VerificationResult:
        """
        执行验证

        Args:
            groundtruth: 预期结果/标准答案
            response: 实际响应/执行结果
            options: 可选的验证配置参数

        Returns:
            VerificationResult: 验证结果
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


def normalize_path(path: str) -> str:
    """
    标准化路径字符串

    - 移除首尾空白
    - 统一为小写
    - 移除尾部斜杠
    - 移除查询参数和锚点

    Args:
        path: 原始路径

    Returns:
        标准化后的路径
    """
    path = path.strip().lower()
    # 移除查询参数和锚点
    if '?' in path:
        path = path.split('?')[0]
    if '#' in path:
        path = path.split('#')[0]
    # 移除尾部斜杠
    return path.rstrip('/')


def extract_path_from_url(url: str) -> str:
    """
    从完整 URL 中提取路径部分

    Args:
        url: 完整 URL 或路径

    Returns:
        路径部分
    """
    # 移除协议和域名
    if '://' in url:
        url = url.split('://', 1)[1]
        url = '/' + url.split('/', 1)[1] if '/' in url else '/'
    return normalize_path(url)


def deep_get(obj: dict[str, Any], path: str, default: Any = None) -> Any:
    """
    通过点分路径获取嵌套字典中的值

    Args:
        obj: 目标字典
        path: 点分路径，如 "user.profile.name"
        default: 默认值

    Returns:
        获取到的值或默认值
    """
    keys = path.split('.')
    current = obj
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list):
            try:
                idx = int(key)
                current = current[idx]
            except (ValueError, IndexError):
                return default
        else:
            return default
    return current


def compare_values(expected: Any, actual: Any, strict: bool = False) -> bool:
    """
    比较两个值是否相等

    Args:
        expected: 期望值
        actual: 实际值
        strict: 是否严格比较（类型敏感）

    Returns:
        是否相等
    """
    if expected is None:
        return actual is None

    if not strict:
        # 宽松比较：字符串不区分大小写，数字允许类型转换
        if isinstance(expected, str) and isinstance(actual, str):
            return expected.lower().strip() == actual.lower().strip()
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return float(expected) == float(actual)
        if isinstance(expected, bool) or isinstance(actual, bool):
            return bool(expected) == bool(actual)

    return expected == actual
