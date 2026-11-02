from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from label.schemas.sandbox import (
    SandboxAcquireRequest,
    SandboxAcquireResponse,
    SandboxCallExecutorRequest,
    SandboxCallExecutorResponse,
    SandboxReleaseRequest,
    SandboxReleaseResponse,
)
from label.services.sandbox import SandboxService


router = APIRouter(prefix='/api/v1/label/sandbox', tags=['label-sandbox'])


def get_sandbox_service(request: Request) -> SandboxService:
    """从 app.state 取出启动时创建的单例 SandboxService。"""
    return request.app.state.sandbox_service


SandboxServiceDep = Annotated[SandboxService, Depends(get_sandbox_service)]


@router.post('/acquire', response_model=SandboxAcquireResponse)
async def acquire(request: SandboxAcquireRequest, service: SandboxServiceDep) -> SandboxAcquireResponse:
    data = await service.acquire(request)
    return SandboxAcquireResponse(data=data)


@router.post('/release', response_model=SandboxReleaseResponse)
async def release(request: SandboxReleaseRequest, service: SandboxServiceDep) -> SandboxReleaseResponse:
    await service.release(request)
    return SandboxReleaseResponse()


@router.post('/call_executor', response_model=SandboxCallExecutorResponse)
async def call_executor(request: SandboxCallExecutorRequest, service: SandboxServiceDep) -> SandboxCallExecutorResponse:
    data = await service.call_executor(request)
    return SandboxCallExecutorResponse(data=data)
