"""
Base repository profile class.

This module defines the abstract base class for repository profiles that specify
installation and testing configurations for different repositories.

Note: This is a simplified version for SWE task evaluation only.
Methods related to image building, mirror creation, and entity extraction have been removed.
"""

from abc import ABC, ABCMeta, abstractmethod
from collections import UserDict
from dataclasses import dataclass, field
from typing import Any

from nexus.extensions.tasks.swe.swe_smith.constants import (
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_PATCH,
)


# Supported file extensions for code entities
SUPPORTED_EXTS = ['.py', '.go', '.rs', '.js', '.ts', '.c', '.cpp', '.java', '.cs', '.php']


class SingletonMeta(ABCMeta):
    _instances: dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass
class RepoProfile(ABC, metaclass=SingletonMeta):
    """
    Base class for repository profiles that define installation and testing specifications.

    This class provides a language-agnostic interface for repository configuration,
    allowing different languages (Python, Go, Rust, etc.) to have their own
    installation and testing patterns while maintaining a consistent API.

    Note: This is a simplified version for SWE task evaluation only.
    """

    exts: list[str] = field(default_factory=lambda: SUPPORTED_EXTS)
    eval_sets: set[str] = field(default_factory=set)

    # Install + Test specifications
    timeout: int = 90  # timeout (sec) for running test suite for a single instance
    timeout_ref: int = 900  # timeout for running entire test suite

    # `min_testing`: If set, then subset of tests (not all) are run for post-bug validation
    # Affects get_test_cmd, get_valid_report
    min_testing: bool = False

    # `min_pregold`: If set, then for pre-bug validation, individual runs are
    # performed instead of running the entire test suite
    min_pregold: bool = False

    # START: Properties ###

    @property
    def repo_name(self) -> str:
        return f'{self.owner}__{self.repo}.{self.commit[:8]}'

    @property
    def mirror_name(self) -> str:
        # Keep for registry key compatibility
        return f'swesmith-x/{self.repo_name}'

    # END: Properties ###

    # START: Methods that *may* require (re-)implementation ###

    def _is_test_path(self, root: str, file: str) -> bool:
        """Check whether the file path corresponds to a testing related file"""
        if len(self.exts) > 1 and not any(file.endswith(ext) for ext in self.exts):
            return False
        if file.lower().startswith('test') or file.rsplit('.', 1)[0].endswith('test'):
            return True
        dirs = root.split('/')
        return bool(any(x in dirs for x in ['tests', 'test', 'specs']))

    def get_test_files(self, instance: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Given an instance, return files corresponding to F2P, P2P test files"""
        return [], []

    def get_test_cmd(self, instance: dict[str, Any], f2p_only: bool = False) -> tuple[str, list[str]]:
        """
        Get the test command for this instance.

        For SWE task evaluation, we primarily use the test_cmd with test files
        from the instance's FAIL_TO_PASS and PASS_TO_PASS fields.
        """
        assert instance[KEY_INSTANCE_ID].rsplit('.', 1)[0] == self.repo_name, (
            f'WARNING: {instance[KEY_INSTANCE_ID]} not from {self.repo_name}'
        )
        test_command = self.test_cmd

        if f2p_only:
            f2p_files, _ = self.get_test_files(instance)
            test_command += f' {" ".join(f2p_files)}'
            return test_command, f2p_files

        if self.min_testing and FAIL_TO_PASS in instance:
            f2p_files, p2p_files = self.get_test_files(instance)
            if len(f2p_files + p2p_files) > 0:
                test_command += f' {" ".join(f2p_files + p2p_files)}'
            return test_command, f2p_files + p2p_files

        if not self.min_testing or KEY_PATCH not in instance:
            # If min testing is not enabled or there's no patch
            # return test command as is (usually = run whole test suite)
            return test_command, []

        # For min_testing with patch but no FAIL_TO_PASS, try to infer test files
        # This path requires cloning the repo, which is not available in simplified version
        # Fall back to running the whole test suite
        return test_command, []

    # END: Methods that *may* require (re-)implementation ###

    # START: Properties, Methods that require implementation ###

    owner: str = ''
    repo: str = ''
    commit: str = ''
    test_cmd: str = ''

    @abstractmethod
    def log_parser(self, log: str) -> dict[str, str]:
        """Parse test output logs and extract relevant information."""
        pass

    # END: Properties, Methods that require implementation ###


# MARK: Profile Registry ###


class Registry(UserDict[str, type]):
    """A registry mapping repo/mirror names to RepoProfile subclasses."""

    def __init__(self) -> None:
        super().__init__()

    def register_profile(self, profile_class: type) -> None:
        """Register a RepoProfile subclass (except base types)."""
        # Skip base types
        if profile_class.__name__ in {
            'RepoProfile',
            'PythonProfile',
            'GoProfile',
            'RustProfile',
            'JavaScriptProfile',
            'CProfile',
            'CppProfile',
            'JavaProfile',
            'CSharpProfile',
            'PhpProfile',
        }:
            return
        # Create temporary instance to get properties
        p = profile_class()
        self.data[p.repo_name] = profile_class
        self.data[p.mirror_name] = profile_class

    def get(self, key: str, default: Any = None) -> RepoProfile:  # type: ignore[override]
        """Get a profile instance by mirror name or repo name."""
        cls = self.data.get(key)
        if cls is None:
            if default is not None:
                return default
            raise KeyError(f'No profile registered for key: {key}')
        return cls()

    def get_from_inst(self, instance: dict[str, Any]) -> RepoProfile:
        """Get a profile class by a SWE-smith instance dict."""
        key = instance.get('repo', instance[KEY_INSTANCE_ID].rsplit('.', 1)[0])
        return self.get(key)

    def keys(self) -> Any:
        return self.data.keys()

    def values(self) -> list[RepoProfile]:  # type: ignore[override]
        return [cls() for cls in set(self.data.values())]


# Global registry instance that can be shared across modules
registry = Registry()
