import re

from dataclasses import dataclass, field

from nexus.extensions.tasks.swe.swe_smith.constants import TestStatus
from nexus.extensions.tasks.swe.swe_smith.profiles.base import RepoProfile, registry


@dataclass
class JavaProfile(RepoProfile):
    """
    Profile for Java repositories.
    """

    exts: list[str] = field(default_factory=lambda: ['.java'])


@dataclass
class Gsondd2fe59c(JavaProfile):
    owner: str = 'google'
    repo: str = 'gson'
    commit: str = 'dd2fe59c0d3390b2ad3dd365ed6938a5c15844cb'
    test_cmd: str = (
        'mvn test -B -T 1C -Dsurefire.useFile=false -Dsurefire.printSummary=true -Dsurefire.reportFormat=plain'
    )
    eval_sets: set[str] = field(default_factory=lambda: {'SWE-bench/SWE-bench_Multilingual'})

    def log_parser(self, log: str) -> dict[str, str]:
        test_status_map = {}
        pattern = r'^\[(INFO|ERROR)\]\s+(.*?)\s+--\s+Time elapsed:\s+([\d.]+)\s'
        for line in log.split('\n'):
            if line.endswith('<<< FAILURE!') and line.startswith('[ERROR]'):
                test_name = re.match(pattern, line)
                if test_name is None:
                    continue
                test_status_map[test_name.group(2)] = TestStatus.FAILED.value
            elif any(line.startswith(s) for s in ['[INFO]', '[ERROR]']) and 'Time elapsed:' in line:
                test_name = re.match(pattern, line)
                if test_name is None:
                    continue
                test_status_map[test_name.group(2)] = TestStatus.PASSED.value
        return test_status_map


# Register all Java profiles with the global registry
for _name, obj in list(globals().items()):
    if isinstance(obj, type) and issubclass(obj, JavaProfile) and obj.__name__ != 'JavaProfile':
        registry.register_profile(obj)
