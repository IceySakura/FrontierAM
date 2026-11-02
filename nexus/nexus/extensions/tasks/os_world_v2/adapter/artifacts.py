"""Re-export the V2 artifact-service publisher for adapter-local callers."""

from nexus.extensions.tasks.os_world_v2.data.artifacts import RunArtifacts


__all__ = ['RunArtifacts']
