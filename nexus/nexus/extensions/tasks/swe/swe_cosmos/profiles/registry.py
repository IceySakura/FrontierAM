"""Profile registry: dispatch a reference repo_dict to its RepoProfile subclass."""

from typing import Any

from nexus.extensions.tasks.swe.swe_cosmos.profiles.base import RepoProfile


class Registry:
    """Registry for RepoProfile instances, keyed by the repo_dict `dataset` field."""

    def get(self, repo_dict: dict[str, Any]) -> RepoProfile:
        """Build a RepoProfile from a reference instance dict (the per-row repo_dict)."""
        dataset_field = repo_dict.get('dataset') or 'SWE-Cosmos/SWE-Env'
        dataset = dataset_field.split('/', 1)[-1].lower()
        if dataset == 'swe-env':
            from nexus.extensions.tasks.swe.swe_cosmos.profiles.swe_env import SWEEnvProfile

            return SWEEnvProfile.from_dict(repo_dict)
        raise ValueError(f'Unsupported SWE-Cosmos dataset: {dataset}')


registry = Registry()
