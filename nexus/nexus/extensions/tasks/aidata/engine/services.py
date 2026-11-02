"""兼容 shim： engine.services → aidata.runtime.services。

新代码请从 `nexus.extensions.tasks.aidata.runtime.services` 导入。
"""

from ..runtime.services import *  # noqa: F401,F403
from ..runtime.services import (  # noqa: F401
    _SCRIPT_INTERPRETERS,
    ServiceReplacements,
    _resolve_connection_values,
    _run_in_service,
    _runtime_config_for,
    _upload_script,
    _upload_service_files,
    _wait_for_connection_port_mappings,
    _wait_ready,
    apply_text_replacements,
    logger,
    setup_services,
)
