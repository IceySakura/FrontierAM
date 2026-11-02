"""应用生命周期：启动时初始化七彩石、加载业务配置并启用热更新。

启动顺序有严格依赖：
1. `init_rainbow()` 建立客户端（七彩石是硬依赖，失败即启动失败）；
2. `load_initial()` 首次拉取 provider 配置——必须在开始接流量前完成，
   否则请求会打到一个凭据为空的服务上；
3. `start_watch()` 注册变更监听，之后配置可热更新。

三步任一失败都会让 startup 抛异常、Pod 起不来，由 K8s 重启重试，
而不是带着不完整的配置进入 ready 状态。
"""

from __future__ import annotations

import logging

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.types import Lifespan

from infra.config import init_rainbow
from label.config import LabelConfigHolder


logger = logging.getLogger(__name__)


def create_lifespan(holder: LabelConfigHolder) -> Lifespan[FastAPI]:
    """构建 lifespan：在每个 worker 进程内初始化七彩石并加载配置。

    七彩石 SDK 的底层 client 不可跨 fork 复用，因此必须在 lifespan 内
    初始化（每个 worker 进程各跑一次），而非在 `main()` 里提前建好。
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        await init_rainbow()
        # 首次加载不容错：配置未就绪就不该开始接流量。
        await holder.load_initial()
        app.state.watching = await holder.start_watch()
        logger.info('startup complete, config ready (watching=%s)', app.state.watching)
        yield

    return lifespan
