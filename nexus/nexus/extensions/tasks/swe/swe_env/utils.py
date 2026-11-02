"""Utility constants and helpers ported from SWE-Env for nexus-side evaluation."""

import re

from pathlib import Path
from typing import Any


# Where the eval script body is materialized inside the sandbox.
_EVAL_SCRIPT_PATH = '/tmp/eval.sh'


# Sentinel markers wrapping the test output inside the eval script.
# These MUST stay byte-identical to swe_env so that read_test_output can locate the block in the `set -x` trace (each marker is printed as `+ : '<marker>'`).
TEST_OUTPUT_START = '>>>>> Start Test Output'
TEST_OUTPUT_END = '>>>>> End Test Output'
TESTS_TIMEOUT = '>>>>> Tests Timed Out'


SUPPORTED_LANGUAGES = set[str]([
    'c',
    'cpp',
    'csharp',
    'golang',
    'java',
    'javascript',
    'typescript',
    'php',
    'python',
    'ruby',
    'rust',
])

LANGUAGE_TO_EXTS: dict[str, list[str]] = {
    'c': ['.c', '.h'],
    'cpp': ['.cpp', '.cc', '.c', '.h'],
    'csharp': ['.cs'],
    'golang': ['.go'],
    'java': ['.java'],
    'javascript': ['.js'],
    'typescript': ['.js', '.ts', '.tsx'],
    'php': ['.php'],
    'python': ['.py'],
    'ruby': ['.rb'],
    'rust': ['.rs'],
}

LANGUAGE_MAPPINGS: dict[str, str] = {
    'c': 'c',
    'cc': 'cpp',
    'c/c++': 'cpp',
    'c++': 'cpp',
    'cplus': 'cpp',
    'c/cplus': 'cpp',
    'c plus plus': 'cpp',
    'cplusplus': 'cpp',
    'c#': 'csharp',
    'cs': 'csharp',
    'c sharp': 'csharp',
    'go': 'golang',
    'javascript/typescript': 'typescript',
    'javascript / typescript': 'typescript',
    'js': 'javascript',
    'ts': 'typescript',
    'js/ts': 'typescript',
    'js / ts': 'typescript',
    'tsx': 'typescript',
    'py': 'python',
    'rb': 'ruby',
    'rs': 'rust',
}

_EXIT_CODE_RE = re.compile(r'>>>>>>>>>> Evaluation Exit Code: (\d+) <<<<<<<<<<')


def language_normalizer(language: str) -> str:
    return LANGUAGE_MAPPINGS.get(language.lower().lstrip('.'), language.lower())


def language_to_ext(language: str | None) -> str:
    """Return the primary file extension (without the dot) for a language."""
    if not language:
        return 'py'
    exts = LANGUAGE_TO_EXTS.get(language_normalizer(language))
    return exts[0].lstrip('.') if exts else 'py'


def read_test_output(filepath: str) -> tuple[str, bool]:
    """Extract test output between sentinel markers from a test log file.

    Returns (content, found) where found indicates whether both sentinel markers were present in order.
    """
    content = Path(filepath).read_text(encoding='utf-8', errors='replace')
    start_sep = f"+ : '{TEST_OUTPUT_START}'"
    end_sep = f"+ : '{TEST_OUTPUT_END}'"
    si = content.find(start_sep)
    ei = content.find(end_sep)
    if si == -1 or ei == -1 or si > ei:
        return content, False
    return content[si + len(start_sep) : ei], True


def get_eval_report(test_log_path: str) -> dict[str, Any]:
    """Grade a single instance from its test log.

    The returned shape (resolved/status/tests_status) is what nexus expects per-instance, and tests_status carries the F2P/P2P success/failure split used for partial scoring.
    """

    def log_parser(log: str | None) -> dict[str, str]:
        """Determine the test status of the overall test (special for SWE-Env, only one test case)."""
        matched = _EXIT_CODE_RE.search(log or '')
        if matched and int(matched.group(1)) == 0:
            return {'all': 'PASSED'}
        return {'all': 'FAILED'}

    report: dict[str, Any] = {}
    test_output, _ = read_test_output(test_log_path)
    test_sm = log_parser(test_output)
    tests_status: dict[str, dict[str, list[str]]] = {
        'FAIL_TO_PASS': {'success': [], 'failure': []},
        'PASS_TO_PASS': {'success': [], 'failure': []},
    }

    fail_to_pass = ['all']
    for tc in fail_to_pass:
        bucket = 'success' if test_sm.get(tc) == 'PASSED' else 'failure'
        tests_status['FAIL_TO_PASS'][bucket].append(tc)

    # pass_to_pass = []
    # for tc in pass_to_pass:
    #     bucket = 'success' if test_sm.get(tc) == 'PASSED' else 'failure'
    #     tests_status['PASS_TO_PASS'][bucket].append(tc)

    resolved = len(tests_status['FAIL_TO_PASS']['failure']) == 0 and len(tests_status['PASS_TO_PASS']['failure']) == 0
    report['resolved'] = resolved
    report['status'] = 'resolved' if resolved else 'unresolved'
    report['tests_status'] = tests_status
    return report


def postprocess_eval_script(eval_script: str) -> str:
    """Post-process a pre-built eval_script to fix common issues.

    Fixes applied:
    1. Remove `set -u` (nounset) to avoid unbound variable errors when heredoc parsing leaves variables unset due to special characters in filenames.
    2. Inject conda/Nix environment activation for Python/pytest scripts that lack it, so `pytest` and `python` resolve correctly in container images.
    3. Inject GRADLE_OPTS proxy configuration so Gradle builds can reach external repositories through the container's HTTPS proxy.
    4. Disable HTTP/2 multiplexing for Cargo to avoid flaky downloads behind proxies (CARGO_HTTP_MULTIPLEXING=false).
    """
    # 1. Remove -u from set flags (e.g. "set -uxo pipefail" -> "set -xo pipefail")
    eval_script = re.sub(r'^(set\s+-\w*)u(\w*)', r'\1\2', eval_script, flags=re.MULTILINE)
    # Clean up bare "set -" left when -u was the only flag (e.g. "set -u" -> "set -xo pipefail")
    eval_script = re.sub(r'^set\s+-\s*$', 'set -xo pipefail', eval_script, flags=re.MULTILINE)
    eval_script = re.sub(r'^set\s+-\s+', 'set ', eval_script, flags=re.MULTILINE)

    # 2. Inject GRADLE_OPTS proxy configuration before pre_install / eval commands
    if 'gradlew' in eval_script and 'GRADLE_OPTS' not in eval_script:
        gradle_proxy_snippet = (
            'if [ -n "$https_proxy" ]; then'
            " PROXY_HOST=$(echo $https_proxy | sed 's|http://||;s|https://||' | cut -d: -f1);"
            " PROXY_PORT=$(echo $https_proxy | sed 's|http://||;s|https://||' | cut -d: -f2);"
            ' export GRADLE_OPTS="-Dhttps.proxyHost=$PROXY_HOST -Dhttps.proxyPort=$PROXY_PORT'
            ' -Dhttp.proxyHost=$PROXY_HOST -Dhttp.proxyPort=$PROXY_PORT";'
            ' fi\n'
        )
        eval_script = re.sub(
            r'^(cd\s+/testbed.*\n)',
            r'\1' + gradle_proxy_snippet,
            eval_script,
            count=1,
            flags=re.MULTILINE,
        )

    # 3. Inject CARGO_HTTP_MULTIPLEXING=false for Cargo builds
    if 'cargo' in eval_script and 'CARGO_HTTP_MULTIPLEXING' not in eval_script:
        eval_script = re.sub(
            r'^(cd\s+/testbed.*\n)',
            r'\1' + 'export CARGO_HTTP_MULTIPLEXING=false\n',
            eval_script,
            count=1,
            flags=re.MULTILINE,
        )

    # 4. Inject environment activation if the script uses pytest but has no activation
    # uses_pytest = 'pytest' in eval_script
    # has_conda_activate = 'conda activate' in eval_script
    # has_nix_setup = '/nix/' in eval_script

    # if uses_pytest and not has_conda_activate:
    #     # Insert env activation after the first "cd /testbed" or "cd /" line
    #     env_activation = (
    #         'source /opt/miniconda3/bin/activate 2>/dev/null && conda activate testbed 2>/dev/null || true\n'
    #     )
    #     # Also add Nix detection
    #     if not has_nix_setup:
    #         env_activation += (
    #             'if [ -d /nix ]; then'
    #             ' export PATH="/nix/runtime/nix-env/bin:$PATH";'
    #             ' NIX_PYTHON=$(find /nix -name "python3" -path "*/bin/*" 2>/dev/null | head -1);'
    #             ' if [ -n "$NIX_PYTHON" ]; then'
    #             '   NIX_BIN=$(dirname "$NIX_PYTHON");'
    #             '   export PATH="$NIX_BIN:$PATH";'
    #             ' fi;'
    #             ' fi\n'
    #         )
    #     # Insert after "cd /testbed" line
    #     eval_script = re.sub(
    #         r'^(cd\s+/testbed.*\n)',
    #         r'\1' + env_activation,
    #         eval_script,
    #         count=1,
    #         flags=re.MULTILINE,
    #     )

    return eval_script
