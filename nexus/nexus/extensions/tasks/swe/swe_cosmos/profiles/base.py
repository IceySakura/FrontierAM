"""Slimmed RepoProfile base for SWE-Cosmos evaluation in nexus. Ported from swecosmos.profiles.base, keeping only what the nexus eval path needs.

Removed vs upstream: `image_name`, `repo_name`, `clone`, `initialize_container`, `extract_entities`, `get_test_files`, `get_test_cmd`, and `get_test_function`, etc. Repo cleanup and patch application now live in the dataset handler, which drives the runtime directly.

Kept (and made abstract so new profiles can extend): `log_parser`, and new method `get_eval_script` that returns the eval script body the handler writes into the container.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from nexus.extensions.tasks.swe.swe_cosmos.utils import (
    LANGUAGE_TO_EXTS,
    SUPPORTED_LANGUAGES,
)


@dataclass
class RepoProfile(ABC):
    """Base class for SWE-Cosmos repository profiles."""

    dataset: str | None = None
    split: str | None = None
    ref_instance_id: str | None = None
    language: str | None = None
    exts: list[str] = field(default_factory=list)
    workdir: str = '/testbed'
    user: str = 'root'

    # Official GitHub repository identity, ${owner}/${repo}.
    owner: str | None = None
    repo: str | None = None
    commit: str | None = None

    # Validation and evaluation arguments
    timeout: int = 900
    timeout_ref: int = 1800
    min_testing: bool = False
    min_pregold: bool = False

    # Test command
    docker_image: str | None = None
    test_cmd: str | None = None
    test_function: str | None = None

    # init_patch == repo_dict.patch, init_test_patch == repo_dict.test_patch.
    # init_test_patch is applied only when present AND apply_init_test_patch is True.
    init_patch: str | None = None
    init_test_patch: str | None = None
    apply_init_test_patch: bool = True

    def __post_init__(self) -> None:
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f'Unsupported language {self.language!r}. Must be one of: {sorted(SUPPORTED_LANGUAGES)}')
        if not self.exts:
            self.exts = list(LANGUAGE_TO_EXTS[self.language])

    # ---- Class methods ----
    @classmethod
    def from_dict(cls, repo_dict: dict[str, Any]) -> 'RepoProfile':
        """Build a profile from a reference instance dict (the per-row repo_dict)."""
        raise NotImplementedError(f'{cls.__name__} does not support from_dict()')

    # ---- Eval script helpers ----
    @abstractmethod
    def get_eval_script(self, instance: dict) -> str:
        """Return a complete eval shell script with sentinel markers around the test output.

        Subclasses must implement this to produce a script like:
            #!/bin/bash
            git checkout {bug_commit} -- {test_files}
            set -x
            : 'TEST_OUTPUT_START'
            <actual test command>
            : 'TEST_OUTPUT_END'
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not implement get_test_cmd()')

    @abstractmethod
    def log_parser(self, log: str | None) -> dict[str, str]:
        """Parse a test log into {test_id: status}."""
        raise NotImplementedError(f'{self.__class__.__name__} does not support log_parser()')
