"""兼容 shim ： engine.check_map → aidata.rubric.check_map 。

新代码请从 `nexus.extensions.tasks.aidata.rubric.check_map` 导入。
"""

from ..rubric.check_map import *  # noqa: F401,F403
from ..rubric.check_map import (  # noqa: F401
    CHECK_MAP_FILENAME,
    _pascal_case,
    derive_candidates,
    load_check_map,
    logger,
    resolve_testcase,
    summarise_mapping,
)
