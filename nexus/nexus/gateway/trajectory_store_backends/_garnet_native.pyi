from collections.abc import Sequence

hiredis_version: str

class NativeBuffer:
    """Own one native reply and expose its bytes through the buffer protocol."""

class NativeGarnetClient:
    def __init__(self, host: str, port: int) -> None: ...
    def mget(self, keys: Sequence[str]) -> list[NativeBuffer | None]: ...
    def close(self) -> None: ...
