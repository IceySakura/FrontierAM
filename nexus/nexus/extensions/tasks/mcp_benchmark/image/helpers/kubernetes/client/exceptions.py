# ruff: noqa: N818


class ApiException(Exception):
    def __init__(self, status: int = 500, reason: str = '', body: str | None = None):
        super().__init__(reason or body or f'Kubernetes API error {status}')
        self.status = status
        self.reason = reason or body or f'Kubernetes API error {status}'
        self.body = body


__all__ = ['ApiException']
