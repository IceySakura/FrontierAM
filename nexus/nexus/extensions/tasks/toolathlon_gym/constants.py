"""Constants for the toolathlon_gym task extension.

Re-exports relevant constants from mcp_benchmark and defines gym-specific ones.
"""

from __future__ import annotations

from nexus.extensions.tasks.mcp_benchmark.constants import (
    BENCHMARK_IMAGE_HELPERS as BENCHMARK_IMAGE_HELPERS,
)
from nexus.extensions.tasks.mcp_benchmark.constants import (
    BENCHMARK_LOCAL_SERVERS as BENCHMARK_LOCAL_SERVERS,
)
from nexus.extensions.tasks.mcp_benchmark.constants import (
    BENCHMARK_WORK_ROOT as BENCHMARK_WORK_ROOT,
)
from nexus.extensions.tasks.mcp_benchmark.constants import (
    TOOLATHLON_IMAGE_ROOT as TOOLATHLON_IMAGE_ROOT,
)
from nexus.extensions.tasks.mcp_benchmark.constants import (
    TOOLATHLON_IMAGE_TASKS_ROOT as TOOLATHLON_IMAGE_TASKS_ROOT,
)
from nexus.extensions.tasks.mcp_benchmark.constants import (
    TOOLATHLON_IMAGE_VENV_BIN as TOOLATHLON_IMAGE_VENV_BIN,
)


TOOLATHLON_GYM_WORK_ROOT = f'{BENCHMARK_WORK_ROOT}/toolathlon_gym'
