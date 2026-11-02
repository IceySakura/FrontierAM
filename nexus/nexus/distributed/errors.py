from __future__ import annotations

import re


_PUBLIC_ERROR_CODE_PATTERN = re.compile(r'^[a-z][a-z0-9_.:-]{0,127}$')


class DistributedDomainError(Exception):
    """A domain failure whose code and optional detail are safe to expose."""

    default_code = 'domain_error'
    status = 409
    retryable = False

    def __init__(
        self,
        code: str | None = None,
        *,
        detail: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        resolved_code = code or self.default_code
        if _PUBLIC_ERROR_CODE_PATTERN.fullmatch(resolved_code) is None:
            raise ValueError('domain_error_code_invalid')
        self.code = resolved_code
        self.detail = detail
        if retryable is not None:
            self.retryable = retryable
        super().__init__(resolved_code)


class ResourceNotFoundError(DistributedDomainError, KeyError):
    default_code = 'not_found'
    status = 404


class DomainConflictError(DistributedDomainError, ValueError):
    default_code = 'conflict'
    status = 409


class FencingError(DomainConflictError):
    default_code = 'operation_fenced'


class DomainValidationError(DistributedDomainError, ValueError):
    default_code = 'invalid_request'
    status = 422


class ResourceGoneError(DistributedDomainError):
    default_code = 'resource_gone'
    status = 410


class TemporaryUnavailableError(DistributedDomainError):
    default_code = 'temporarily_unavailable'
    status = 503
    retryable = True


class AdmissionRejectedError(DistributedDomainError):
    default_code = 'admission_rejected'
    status = 429
    retryable = True


__all__ = [
    'AdmissionRejectedError',
    'DistributedDomainError',
    'DomainConflictError',
    'DomainValidationError',
    'FencingError',
    'ResourceGoneError',
    'ResourceNotFoundError',
    'TemporaryUnavailableError',
]
