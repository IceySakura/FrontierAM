from __future__ import annotations

from aiohttp import web

from nexus.distributed.gateways.application.routes.artifact_content import (
    download_internal_artifact_content,
    upload_artifact_content,
)
from nexus.distributed.gateways.application.routes.artifact_dependency_build import (
    report_dependency_build_failure,
    stage_dependency_build_wheelhouse_metadata,
)
from nexus.distributed.gateways.application.routes.artifact_uploads import upload_internal_artifact_content


def register_artifact_routes(app: web.Application) -> None:
    app.router.add_put('/v1/artifact-uploads/{upload_id}/content', upload_artifact_content)
    app.router.add_put('/v1/internal/artifacts/{artifact_kind}', upload_internal_artifact_content)
    app.router.add_get('/v1/internal/artifacts/{artifact_id}/content', download_internal_artifact_content)
    app.router.add_post(
        '/v1/internal/dependency-build/wheelhouse-metadata',
        stage_dependency_build_wheelhouse_metadata,
    )
    app.router.add_post('/v1/internal/dependency-build/failure', report_dependency_build_failure)


__all__ = ['register_artifact_routes']
