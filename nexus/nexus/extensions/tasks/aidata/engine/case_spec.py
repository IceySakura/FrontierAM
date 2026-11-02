"""兼容 shim：engine.case_spec → aidata.case.spec。

新代码请从 `nexus.extensions.tasks.aidata.case.spec` （或 `...aidata.case`） 导入。 此
shim 直接委托给 `case.spec` （而非 `case` 包的 init），继续兼容其余旧的
私有 helper 导入。
"""

from ..case.spec import *  # noqa: F401,F403
from ..case.spec import (  # noqa: F401
    _ALLOWED_AGENT_TOOLS,
    _METADATA_DATE_RE,
    _parse_agent_tools_spec,
    _parse_ask_user_config,
    _parse_case_metadata,
    _parse_dump_state_spec,
    _parse_initial_trajectory,
    _parse_runtime_main_image,
    _parse_service_spec,
    _parse_wx_agent_hub_spec,
    _resolve_case_path,
    case_manifest_path,
    case_rubric_path,
    load_case_rubric,
    logger,
    normalize_extra_spec,
    resolve_case,
)
