from __future__ import annotations

from fastapi import FastAPI

from infra.config import ConfigHolder
from infra.middleware import TraceHeaderMiddleware
from label.api import create_router
from label.config import LabelConfig, LabelConfigHolder, load_provider
from label.exception_handlers import register_exception_handlers
from label.lifespan import create_lifespan
from label.services.sandbox import SandboxService


def create_app(config: LabelConfig | None = None) -> FastAPI:
    """组装 FastAPI 应用：配置容器、服务、路由、异常处理与生命周期。

    `config` 只需包含启动期参数；provider 业务配置由 lifespan 在
    七彩石客户端就绪后加载，未传入时从环境变量读取。

    配置以 `ConfigHolder` 承载并传给依赖方，依赖方通过 `current`
    惰性读取，因此七彩石热更新后无需重建应用或重启进程。
    """
    holder: LabelConfigHolder = ConfigHolder(config or LabelConfig.from_env(), field='provider', loader=load_provider)
    sandbox_service = SandboxService(holder)

    app = FastAPI(title=holder.current.service_name, lifespan=create_lifespan(holder))
    app.state.config_holder = holder
    app.state.sandbox_service = sandbox_service

    # ty 对 starlette 的 _MiddlewareFactory[ParamSpec] 协议匹配有局限，
    # 其自带的 GZipMiddleware/CORSMiddleware 同样误报，故在此抑制。
    app.add_middleware(TraceHeaderMiddleware)  # ty: ignore[invalid-argument-type]
    register_exception_handlers(app)
    app.include_router(create_router())
    return app
