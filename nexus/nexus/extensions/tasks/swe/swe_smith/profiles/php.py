import re

from dataclasses import dataclass, field

from nexus.extensions.tasks.swe.swe_smith.constants import TestStatus
from nexus.extensions.tasks.swe.swe_smith.profiles.base import RepoProfile, registry


@dataclass
class PhpProfile(RepoProfile):
    """
    Profile for PHP repositories.
    """

    exts: list[str] = field(default_factory=lambda: ['.php'])
    test_cmd: str = 'vendor/bin/phpunit --testdox --colors=never'


@dataclass
class Dbal(PhpProfile):
    owner: str = 'doctrine'
    repo: str = 'dbal'
    commit: str = 'acb68b388b2577bb211bb26dc22d20a8ad93d97d'

    def log_parser(self, log: str) -> dict[str, str]:
        test_status_map = {}
        passed_pattern = re.compile(r'^\s*✔\s*(.+)$')
        failed_pattern = re.compile(r'^\s*✘\s*(.+)$')
        skipped_pattern = re.compile(r'^\s*↩\s*(.+)$')
        for line in log.split('\n'):
            for pattern, status in (
                (passed_pattern, TestStatus.PASSED.value),
                (failed_pattern, TestStatus.FAILED.value),
                (skipped_pattern, TestStatus.SKIPPED.value),
            ):
                match = pattern.match(line)
                if match:
                    test_name = match.group(1).strip()
                    test_status_map[test_name] = status
                    break
        return test_status_map


# Register all PHP profiles with the global registry
for _name, obj in list(globals().items()):
    if isinstance(obj, type) and issubclass(obj, PhpProfile) and obj.__name__ != 'PhpProfile':
        registry.register_profile(obj)
