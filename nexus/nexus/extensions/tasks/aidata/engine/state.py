"""兼容性 shim： engine.state → aidata.runtime.state。

新代码请从 `nexus.extensions.tasks.aidata.runtime.state` 导入。
"""

from ..runtime.state import *  # noqa: F401,F403
from ..runtime.state import (  # noqa: F401
    _build_tool_call_id_map,
    _call_dump_state_with_retries,
    _collect_state_via_mcp,
    _dump_state_call_args,
    _dump_state_error_info,
    _dump_state_max_attempts,
    _dump_state_retry_backoff_s,
    _dump_state_text_from_response,
    _merge_dump_state,
    _namespace_from_tool_name,
    build_summary,
    collect_crontab_state,
    download_state,
    extract_final_answer,
    logger,
)
