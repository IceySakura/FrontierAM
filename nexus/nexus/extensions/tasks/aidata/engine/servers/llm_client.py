"""兼容 shim：engine.servers.llm_client → aidata.mock_servers.llm_client。

新代码请从 `nexus.extensions.tasks.aidata.mock_servers.llm_client` 导入。

使用绝对导入，以便本文件在作为顶级模块加载时也能正常工作。
"""

from nexus.extensions.tasks.aidata.mock_servers.llm_client import *  # noqa: F401,F403
from nexus.extensions.tasks.aidata.mock_servers.llm_client import (  # noqa: F401
    _BUDGET_LOCK,
    _BUDGET_USED,
    LLMBudgetExceededError,
    LLMUnavailableError,
    _budget_limit,
    _consume_budget,
    extract_json,
    llm_available,
    llm_call_budget_status,
    llm_chat,
    llm_chat_json,
)
