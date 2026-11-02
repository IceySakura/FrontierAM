from fastapi import APIRouter

from label.api.health import router as health_router
from label.api.sandbox import router as sandbox_router


def create_router() -> APIRouter:
    router = APIRouter()
    router.include_router(health_router)
    router.include_router(sandbox_router)
    return router


__all__ = ['create_router']
