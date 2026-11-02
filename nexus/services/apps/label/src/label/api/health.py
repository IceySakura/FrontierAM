from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from label.config import LabelConfigHolder
from label.schemas.health import ConfigSnapshotResponse, HealthResponse


router = APIRouter(prefix='/health', tags=['health'])


def get_config_holder(request: Request) -> LabelConfigHolder:
    """从 app.state 取出启动时创建的配置容器。"""
    return request.app.state.config_holder


def get_watching(request: Request) -> bool:
    """热更新是否已启用，由 lifespan 在启动时写入。"""
    return bool(getattr(request.app.state, 'watching', False))


ConfigHolderDep = Annotated[LabelConfigHolder, Depends(get_config_holder)]
WatchingDep = Annotated[bool, Depends(get_watching)]


def _mask(value: str | None) -> str | None:
    """敏感值只保留前 8 位与长度，便于比对是否变更而不泄露完整凭据。"""
    if not value:
        return value
    return f'{value[:8]}...({len(value)})'


@router.get('/live', response_model=HealthResponse)
async def live(holder: ConfigHolderDep) -> HealthResponse:
    return HealthResponse(status='ok', service=holder.current.service_name)


@router.get('/ready', response_model=HealthResponse)
async def ready(holder: ConfigHolderDep) -> HealthResponse:
    return HealthResponse(status='ready', service=holder.current.service_name)


@router.get('/config', response_model=ConfigSnapshotResponse)
async def config_snapshot(holder: ConfigHolderDep, watching: WatchingDep) -> ConfigSnapshotResponse:
    """当前生效配置摘要，敏感值只回显前缀，用于确认热更新是否生效。"""
    provider = holder.current.provider
    return ConfigSnapshotResponse(
        watching=watching,
        ags={
            'secret_id': _mask(provider.ags.secret_id),
            'tool_id': provider.ags.tool_id,
            'region': provider.ags.region,
        },
        devcloud={'app_id': provider.devcloud.app_id, 'project': provider.devcloud.project},
        smartrun={'token': _mask(provider.smartrun.token), 'duration': provider.smartrun.duration},
    )
