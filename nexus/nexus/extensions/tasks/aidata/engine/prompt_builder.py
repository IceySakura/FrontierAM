"""兼容 shim ： engine.prompt_builder → aidata.runtime.prompt_builder 。

新代码请从 `nexus.extensions.tasks.aidata.runtime.prompt_builder` 导入。
"""

from ..runtime.prompt_builder import *  # noqa: F401,F403
from ..runtime.prompt_builder import (  # noqa: F401
    _FRONTMATTER_RE,
    _SANDBOX_PLATFORM,
    _SKILL_DESC_MAX_CHARS,
    ProjectContext,
    _parse_skill_frontmatter,
    _parse_skill_frontmatter_for_tests,
    discover_project_context,
    logger,
)
