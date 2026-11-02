"""Repository profiles for SWE-Cosmos evaluation."""

from nexus.extensions.tasks.swe.swe_cosmos.profiles.base import RepoProfile
from nexus.extensions.tasks.swe.swe_cosmos.profiles.registry import registry


__all__ = ['RepoProfile', 'registry']
