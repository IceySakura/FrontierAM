from dataclasses import dataclass, field

from nexus.extensions.tasks.swe.swe_smith.constants import TestStatus
from nexus.extensions.tasks.swe.swe_smith.profiles.base import RepoProfile, registry


@dataclass
class CSharpProfile(RepoProfile):
    """
    Profile for CSharp repositories.
    """

    exts: list[str] = field(default_factory=lambda: ['.cs'])


@dataclass
class VirtualClient0bb16489(CSharpProfile):
    owner: str = 'microsoft'
    repo: str = 'VirtualClient'
    commit: str = '0bb16489e29d2b8ae18b1187ade52cda4eae68bd'
    test_cmd: str = './build-test.sh'

    def _is_test_path(self, root: str, file: str) -> bool:
        return file.endswith('Tests.cs') or root.endswith('.UnitTests') or root.endswith('.FunctionalTests')

    def log_parser(self, log: str) -> dict[str, str]:
        test_status_map = {}
        for line in log.split('\n'):
            line = line.strip()
            for prefix, status in [
                ('Passed', TestStatus.PASSED.value),
                ('Failed', TestStatus.FAILED.value),
                ('Skipped', TestStatus.SKIPPED.value),
            ]:
                if line.startswith(prefix):
                    test_name = line.split()[1]
                    test_status_map[test_name] = status
                    break
        return test_status_map


# Register all CSharp profiles with the global registry
for _name, obj in list(globals().items()):
    if isinstance(obj, type) and issubclass(obj, CSharpProfile) and obj.__name__ != 'CSharpProfile':
        registry.register_profile(obj)
