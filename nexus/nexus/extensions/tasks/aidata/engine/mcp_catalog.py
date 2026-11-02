"""兼容 shim：engine.mcp_catalog → aidata.runtime.mcp_catalog。

新代码请从 `nexus.extensions.tasks.aidata.runtime.mcp_catalog` 导入。
"""

from ..runtime.mcp_catalog import *  # noqa: F401,F403
from ..runtime.mcp_catalog import (  # noqa: F401
    _expand_wx_agent_hub_specs,
    build_specs,
    fake_now_epoch,
    mock_env,
    normalize_host_env,
    resolve_judge_creds,
    resolve_judge_model,
    resolve_mock_llm_creds,
)
