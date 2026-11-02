"""兼容性 shim：engine.servers.shell_server → aidata.mock_servers.shell_server。

新代码请从 `nexus.extensions.tasks.aidata.mock_servers.shell_server` 导入。

使用绝对导入，以便在测试执行
``sys.path.insert(0, engine/servers/)`` + ``from shell_server import ...``
（作为无父包的顶层模块加载）时也能正常工作。
"""

from nexus.extensions.tasks.aidata.mock_servers.shell_server import *  # noqa: F401,F403
from nexus.extensions.tasks.aidata.mock_servers.shell_server import (  # noqa: F401
    ALLOWED_DIR,
    CONFINE_DENY,
    CONFINE_MODE,
    CONFINE_NET_DENY,
    CONFINE_SO,
    DEFAULT_TIMEOUT,
    MAX_OUTPUT_CHARS,
    _ENV_ASSIGN_RE,
    _RUN_COMMAND_DESC,
    _apply_net_deny_no_proxy,
    _blocking_run,
    _coerce_stream,
    _collect_warnings,
    _confine_envp,
    _extract_first_command,
    _run_bash,
    _truncate,
    handle_call_tool,
    handle_list_tools,
    main,
    server,
)
