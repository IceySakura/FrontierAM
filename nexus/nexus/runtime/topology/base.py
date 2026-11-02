"""Provider-neutral topology and service handles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from nexus.runtime.topology.models import TopologyStatus


if TYPE_CHECKING:
    from nexus.runtime.executors.models.command import RunCommandResponse


class ServiceHandle(ABC):
    """Minimal exec/copy surface for one service in a topology."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the topology-local service name."""

    @abstractmethod
    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: float = 180,
        stdin: str | None = None,
    ) -> RunCommandResponse:
        """Execute a shell command in the service."""

    @abstractmethod
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """Copy a Host file into the service."""

    @abstractmethod
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Copy a Host directory into the service."""

    @abstractmethod
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """Copy one service file to the Host."""

    @abstractmethod
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Copy one service directory to the Host."""

    @abstractmethod
    async def stop(self, *, timeout: int = 30) -> None:
        """Stop the service while leaving the topology alive."""


class TopologyHandle(ABC):
    """Lifecycle owner for a multi-service task environment."""

    @property
    @abstractmethod
    def topology_id(self) -> str:
        """Return the provider-local topology identifier."""

    @abstractmethod
    async def stage(self) -> None:
        """Stage the frozen topology definition and its task-local inputs."""

    @abstractmethod
    async def start(self) -> None:
        """Create and start the topology."""

    @abstractmethod
    async def wait_ready(self) -> TopologyStatus:
        """Wait until every required service reaches its ready terminal state."""

    @abstractmethod
    async def inspect(self) -> TopologyStatus:
        """Return a normalized current status snapshot."""

    @abstractmethod
    def service(self, name: str) -> ServiceHandle:
        """Return a lightweight handle for one service."""

    @abstractmethod
    async def delete(self) -> None:
        """Delete the topology and all attempt-scoped resources."""


__all__ = ['ServiceHandle', 'TopologyHandle']
