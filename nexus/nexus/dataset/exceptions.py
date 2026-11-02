"""Dataset client exceptions."""


class DatasetError(Exception):
    """Base exception for Nexus dataset operations."""


class DatasetConfigurationError(DatasetError):
    """Raised when required dataset configuration is missing."""


class DatasetAuthenticationError(DatasetError):
    """Raised when dataset service authentication fails."""


class DatasetClientError(DatasetError):
    """Raised when UC or IO service requests fail."""

    def __init__(
        self,
        message: str,
        *,
        trace_id: str | None = None,
        code: str | None = None,
        http_status: int | None = None,
    ):
        super().__init__(message)
        self.trace_id = trace_id
        self.code = code
        self.http_status = http_status


class UcAssetNotFoundError(DatasetClientError):
    """Raised when a UC Nexus asset is missing."""
