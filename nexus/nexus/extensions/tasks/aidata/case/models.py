"""兼容性 shim：aidata.case.models → aidata.engine.models。

类型化 I/O 模型（``AidataCaseMetadata`` / ``AidataService`` / ……）目前仍位于
``engine.models``；此 shim 允许 ``case.spec``（以及任何将 ``case`` 包视为思考单元的
下游消费者）通过本地 ``.models`` 名称导入，而无需硬性重命名。
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
