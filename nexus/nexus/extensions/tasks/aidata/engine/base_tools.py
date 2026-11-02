"""兼容性 shim：engine.base_tools → aidata.runtime.base_tools。

新代码请从 `nexus.extensions.tasks.aidata.runtime.base_tools` 导入。
"""

from ..runtime.base_tools import *  # noqa: F401,F403
from ..runtime.base_tools import (  # noqa: F401
    AskUserInput,
    _call_ask_user_script,
    ask_user_tool,
)
