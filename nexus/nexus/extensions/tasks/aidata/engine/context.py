"""兼容性 shim：engine.context → aidata.runtime.context。

新代码请从 `nexus.extensions.tasks.aidata.runtime.context` 导入。
"""

from ..runtime.context import *  # noqa: F401,F403
from ..runtime.context import (  # noqa: F401
    build_context_manifest,
    render_manifest,
)
