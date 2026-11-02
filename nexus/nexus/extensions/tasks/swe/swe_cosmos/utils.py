"""Utility constants and helpers ported from SWE-Cosmos for nexus-side evaluation.

Only the pieces required to grade a SWE-Cosmos instance are ported here. Synthesis-time machinery (tree-sitter entity extraction, language inference from a live image, threadpool harness, container lifecycle) is intentionally left out; nexus owns sandbox lifecycle and parallelism.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from unidiff import PatchSet

    from nexus.extensions.tasks.swe.swe_cosmos.profiles.base import RepoProfile


# Stashed onto the instance by eval_runtime_init, consumed by get_eval_script.
_BUG_COMMIT_KEY = '_swecosmos_bug_commit'
_REVERT_FILES_KEY = '_swecosmos_revert_files'

# Where the eval script body is materialized inside the sandbox.
_EVAL_SCRIPT_PATH = '/tmp/eval.sh'


# Sentinel markers wrapping the test output inside the eval script.
# These MUST stay byte-identical to swecosmos.profiles.base so that read_test_output can locate the block in the `set -x` trace (each marker is printed as `+ : '<marker>'`).
TEST_OUTPUT_START = '>>>>> Start Test Output'
TEST_OUTPUT_END = '>>>>> End Test Output'
TESTS_TIMEOUT = '>>>>> Tests Timed Out'


# Language normalization, copied verbatim from swecosmos.bugmaker.adapters so the RepoProfile validation and the gitignore-rule key match synthesis-side conventions.
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


def language_normalizer(language: str) -> str:
    return LANGUAGE_MAPPINGS.get(language.lower().lstrip('.'), language.lower())


def language_to_ext(language: str | None) -> str:
    """Return the primary file extension (without the dot) for a language.

    Built on SWE-Cosmos's language_normalizer + LANGUAGE_TO_EXTS so the gitignore-rule key the swe_task framework consumes matches synthesis conventions. Defaults to 'py'.
    """
    if not language:
        return 'py'
    exts = LANGUAGE_TO_EXTS.get(language_normalizer(language))
    return exts[0].lstrip('.') if exts else 'py'


def get_patch_set(patch: str | None) -> 'PatchSet | list':
    """Parse a unified diff into a unidiff.PatchSet, returning [] on failure.

    Ported from swecosmos.bugmaker.code_utils.get_patch_set (including the trailing-newline retry), so the eval path enumerates touched files via PatchedFile.path exactly as synthesis did.
    """
    if not patch:
        return []

    # Deferred so the package imports without unidiff: only the eval path needs it, and `swe`'s host deps are installed per-module.
    from unidiff import PatchSet

    try:
        return PatchSet(patch)
    except Exception:
        pass
    try:
        return PatchSet(patch + '\n')
    except Exception:
        return []


def read_test_output(filepath: str) -> tuple[str, bool]:
    """Extract test output between sentinel markers from a test log file.

    Ported from swecosmos.harness.utils.read_test_output. Returns (content, found) where found indicates whether both sentinel markers were present in order.
    """
    content = Path(filepath).read_text(encoding='utf-8', errors='replace')
    start_sep = f"+ : '{TEST_OUTPUT_START}'"
    end_sep = f"+ : '{TEST_OUTPUT_END}'"
    si = content.find(start_sep)
    ei = content.find(end_sep)
    if si == -1 or ei == -1 or si > ei:
        return content, False
    return content[si + len(start_sep) : ei], True


def get_eval_report(
    instance: dict[str, Any],
    rp: 'RepoProfile',
    test_log_path: str,
    f2p_only: bool = False,
) -> dict[str, Any]:
    """Grade a single instance from its test log.

    Ported from swecosmos.harness.evaluate.get_eval_report. The returned shape (resolved/status/tests_status) is what nexus expects per-instance, and tests_status carries the F2P/P2P success/failure split used for partial scoring.
    """
    report: dict[str, Any] = {}
    test_output, _ = read_test_output(test_log_path)
    test_sm = rp.log_parser(test_output)
    tests_status: dict[str, dict[str, list[str]]] = {
        'FAIL_TO_PASS': {'success': [], 'failure': []},
        'PASS_TO_PASS': {'success': [], 'failure': []},
    }
    for tc in instance.get('FAIL_TO_PASS', []):
        bucket = 'success' if test_sm.get(tc) == 'PASSED' else 'failure'
        tests_status['FAIL_TO_PASS'][bucket].append(tc)
    for tc in instance.get('PASS_TO_PASS', []):
        bucket = 'success' if test_sm.get(tc) == 'PASSED' else 'failure'
        tests_status['PASS_TO_PASS'][bucket].append(tc)
    resolved = len(tests_status['FAIL_TO_PASS']['failure']) == 0 and (
        f2p_only or len(tests_status['PASS_TO_PASS']['failure']) == 0
    )
    report['resolved'] = resolved
    report['status'] = 'resolved' if resolved else 'unresolved'
    report['tests_status'] = tests_status
    return report
