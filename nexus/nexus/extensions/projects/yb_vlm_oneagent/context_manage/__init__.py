"""yb_vlm_agent oneagent 协议的上下文管理工具。

包含三个模块：
  - ``image_resizer``: 单图 / 多图 场景下的像素预算与 factor 对齐。
  - ``token_counter``: 复用训练侧 tokenizer 的文本/图片 token 统计。
  - ``context_manager``: 按 oneagent 协议 按 role 分配 token 预算、构造 messages、
    按 doc 粒度截断工具返回。
主要供训练数据预处理（dataset）与 agent rollout（yb_vlm_agent）使用。
"""

from .context_manager import (  # noqa: F401
    ContextBudget,
    apply_tool_round_budget,
    build_managed_messages,
    split_web_doc_segments,
)
from .image_resizer import (  # noqa: F401
    interleave_resize,
    resize_smart,
)
from .token_counter import (  # noqa: F401
    TokenCounter,
    get_image_size,
)


# Explicit re-export list. Without ``__all__``, strict type checkers
# (basedpyright / pyright / mypy --strict) treat the names above as
# package-private and warn callers with
# ``"X" is not exported from module ...``. Listing them here makes the
# public surface of the package explicit and import-friendly.
__all__ = [
    # image_resizer
    'resize_smart',
    'interleave_resize',
    # token_counter
    'TokenCounter',
    'get_image_size',
    # context_manager
    'ContextBudget',
    'build_managed_messages',
    'apply_tool_round_budget',
    'split_web_doc_segments',
]
