"""兼容 shim： engine.junit → aidata.rubric.junit。

新代码请从 `nexus.extensions.tasks.aidata.rubric.junit` 导入。
"""

from ..rubric.junit import *  # noqa: F401,F403
from ..rubric.junit import (  # noqa: F401
    JunitParseError,
    JunitReport,
    TestCaseResult,
    _parse_testcase,
    logger,
    parse_junit_file,
    parse_junit_xml,
)
