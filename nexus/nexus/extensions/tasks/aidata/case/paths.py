"""兼容 shim： aidata.case.paths → aidata.engine.paths。

路径常量（``CASES_DIR`` / ``CASE_DIR_IN`` / ……）仍位于
``engine.paths``； 该 shim 将其重新导出，使 ``case.spec`` 可通过
本地的 ``.paths`` 名称导入而无需强制重命名。
"""

from ..engine.paths import *  # noqa: F401,F403
from ..engine.paths import (  # noqa: F401
    AGENT_DESKTOP,
    AGENT_INPUTS_DIR,
    AGENT_OUTPUT_DIR,
    CASES_DIR,
    CASE_DIR_IN,
    CONFINE_DENY_DEFAULT,
    CONFINE_DIR,
    CONFINE_DIR_IN,
    CONFINE_NET_DENY_DEFAULT,
    CONFINE_SH_IN,
    CONFINE_SO_IN,
    INTERNAL_ROOT,
    JUNIT_REPORT_PATH_IN,
    MCP_PY,
    MOCK_SERVER_DATA_DIR_IN,
    MOCK_SERVER_ROOT_IN,
    PDF_TOOLS_BIN,
    SERVERS_DIR,
    SERVERS_DIR_IN,
    SHELL_PATH_IN,
    SOLUTION_DIR_IN,
    STATE_DIR_IN,
    TASK_ROOT,
    TESTS_DIR_IN,
    WORKSPACE_ROOT,
    resolve_image_paths,
)
