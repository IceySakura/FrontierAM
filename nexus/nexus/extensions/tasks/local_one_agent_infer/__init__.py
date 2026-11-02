"""local_one_agent_infer — 本地 / AGS 模式推理/评测 task 包。

- ``task``：本地模式（``local_one_agent_infer``）—— ``shutil.copy`` 铺文件，无强制
  runtime。
- ``ags_task``：AGS 模式（``local_one_agent_infer_ags``）—— ``get_runtime('main')`` +
  upload 到远端 VM；当配置 ``runtime_provider_name: ags`` 时由 generator 路由至此。

``<media_info>`` 抽取策略可按 ``LOCAL_PROMPT_VERSION`` 路由（见 ``task._media_extract_mode``）。
"""

from __future__ import annotations

from nexus.extensions.tasks.local_one_agent_infer import (
    ags_task,  # noqa: F401
    task,  # noqa: F401
)
