"""兼容 shim ： aidata.runtime.models → aidata.engine.models。

类型化 I/O 模型仍位于 ``engine.models`` 中； 该 shim 让
``runtime.services`` 及其他运行时辅助模块可以通过本地
``.models`` 名字导入， 无需硬性重命名。
"""

from ..engine.models import *  # noqa: F401,F403
from ..engine.models import (  # noqa: F401
    AidataCaseMetadata,
    AidataService,
    AidataServiceCommand,
    AidataServiceConnection,
    AidataServiceFile,
    AidataTaskInput,
    AidataTaskOutput,
)
