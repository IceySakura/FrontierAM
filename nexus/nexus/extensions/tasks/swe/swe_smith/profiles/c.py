import re

from dataclasses import dataclass, field

from nexus.extensions.tasks.swe.swe_smith.constants import TestStatus
from nexus.extensions.tasks.swe.swe_smith.profiles.base import RepoProfile, registry


@dataclass
class CProfile(RepoProfile):
    """
    Profile for C repositories.
    """

    exts: list[str] = field(default_factory=lambda: ['.c', '.h'])


@dataclass
class Jqb9e19de76(CProfile):
    owner: str = 'jqlang'
    repo: str = 'jq'
    commit: str = 'b9e19de76e6e19d044007ead65d164710dc98877'
    test_cmd: str = 'make check'
    eval_sets: set[str] = field(default_factory=lambda: {'SWE-bench/SWE-bench_Multilingual'})

    def log_parser(self, log: str) -> dict[str, str]:
        test_status_map = {}
        pattern = r'^\s*(PASS|FAIL):\s(.+)$'
        for line in log.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                status, test_name = match.groups()
                if status == 'PASS':
                    test_status_map[test_name] = TestStatus.PASSED.value
                elif status == 'FAIL':
                    test_status_map[test_name] = TestStatus.FAILED.value
        return test_status_map


@dataclass
class Valkeyfc7c04e4(CProfile):
    owner: str = 'valkey-io'
    repo: str = 'valkey'
    commit: str = 'fc7c04e4f8ba86dfbac1ec059db457fb44ed0a2d'
    test_cmd: str = 'TERM=dumb ./runtest --durable'
    eval_sets: set[str] = field(default_factory=lambda: {'SWE-bench/SWE-bench_Multilingual'})

    def log_parser(self, log: str) -> dict[str, str]:
        test_status_map = {}
        pattern = r'^\[(ok|err|skip|ignore)\]:\s(.+?)(?:\s\((\d+\s*m?s)\))?$'
        for line in log.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                status, test_name, _duration = match.groups()
                if status == 'ok':
                    test_status_map[test_name] = TestStatus.PASSED.value
                elif status == 'err':
                    # Strip out file path information from failed test names
                    test_name = re.sub(r'\s+in\s+\S+$', '', test_name)
                    test_status_map[test_name] = TestStatus.FAILED.value
                elif status == 'skip' or status == 'ignore':
                    test_status_map[test_name] = TestStatus.SKIPPED.value
        return test_status_map


# Register all C profiles with the global registry
for _name, obj in list(globals().items()):
    if isinstance(obj, type) and issubclass(obj, CProfile) and obj.__name__ != 'CProfile':
        registry.register_profile(obj)
