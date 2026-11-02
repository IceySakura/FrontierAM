from __future__ import annotations

import aiohttp

from aiohttp import web
from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.distributed.access.auth import AuthorizationPolicy
from nexus.distributed.access.trace_database import TraceDatabaseAccountManager
from nexus.distributed.artifacts.admission import TransferAdmission
from nexus.distributed.artifacts.spool import ArtifactSpool
from nexus.distributed.artifacts.store import ArtifactStore, ArtifactUploadCoordinator
from nexus.distributed.artifacts.tickets import ArtifactContentTicketSigner, UploadTicketSigner
from nexus.distributed.config import DistributedConfig
from nexus.distributed.observability.metrics import OpenTelemetryExecutionLogMetrics


SESSION_FACTORY_KEY = web.AppKey('session_factory', async_sessionmaker)
CONFIG_KEY = web.AppKey('config', DistributedConfig)
TRACE_DATABASE_ACCOUNT_MANAGER_KEY = web.AppKey('trace_database_account_manager', TraceDatabaseAccountManager)
HTTP_CLIENT_KEY = web.AppKey('http_client', aiohttp.ClientSession)
AUTHORIZATION_KEY = web.AppKey('authorization', AuthorizationPolicy)
UPLOAD_TICKET_SIGNER_KEY = web.AppKey('upload_ticket_signer', UploadTicketSigner)
ARTIFACT_CONTENT_TICKET_SIGNER_KEY = web.AppKey('artifact_content_ticket_signer', ArtifactContentTicketSigner)
ARTIFACT_STORE_KEY = web.AppKey('artifact_store', ArtifactStore)
ARTIFACT_UPLOAD_COORDINATOR_KEY = web.AppKey('artifact_upload_coordinator', ArtifactUploadCoordinator)
ARTIFACT_DOWNLOAD_ADMISSION_KEY = web.AppKey('artifact_download_admission', TransferAdmission)
ARTIFACT_SPOOL_KEY = web.AppKey('artifact_spool', ArtifactSpool)
EXECUTION_LOG_METRICS_KEY = web.AppKey('execution_log_metrics', OpenTelemetryExecutionLogMetrics)


__all__ = [
    'ARTIFACT_CONTENT_TICKET_SIGNER_KEY',
    'ARTIFACT_DOWNLOAD_ADMISSION_KEY',
    'ARTIFACT_STORE_KEY',
    'ARTIFACT_SPOOL_KEY',
    'ARTIFACT_UPLOAD_COORDINATOR_KEY',
    'AUTHORIZATION_KEY',
    'CONFIG_KEY',
    'EXECUTION_LOG_METRICS_KEY',
    'HTTP_CLIENT_KEY',
    'SESSION_FACTORY_KEY',
    'TRACE_DATABASE_ACCOUNT_MANAGER_KEY',
    'UPLOAD_TICKET_SIGNER_KEY',
]
