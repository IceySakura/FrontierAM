"""Dataset-specific input cleanup for the SWE HITL recipe."""

import re


def preprocess_problem_statement(text: str, preset: str) -> str:
    """Apply dataset-specific cleanup to problem statements."""
    if preset == 'rebench_v2':
        return _strip_rebench_v2_suffix(text)
    return text


def preprocess_eval_script(script: str, preset: str) -> str:
    """Remove reference-test patch application blocks from eval-script context."""
    if preset == 'rebench_v2':
        return _strip_eval_script_patches(script)
    return script


_REBENCH_V2_SUFFIX = re.compile(
    r'\n*Can you help me implement the necessary changes to the repository'
    r'.*$',
    re.DOTALL,
)

_RE_GIT_CHECKOUT_FILES = re.compile(
    r'^git checkout [0-9a-f]+ .+$',
    re.MULTILINE,
)

_RE_RM_TEST_FILE = re.compile(
    r'^rm -f .+$',
    re.MULTILINE,
)

_RE_GIT_APPLY_HEREDOC = re.compile(
    r"^git apply.*<<'(EOF\w*)'\n.*?^\1$",
    re.MULTILINE | re.DOTALL,
)


def _strip_rebench_v2_suffix(text: str) -> str:
    return _REBENCH_V2_SUFFIX.sub('', text).rstrip()


def _strip_eval_script_patches(script: str) -> str:
    script = _RE_GIT_APPLY_HEREDOC.sub('', script)
    script = _RE_GIT_CHECKOUT_FILES.sub('', script)
    script = _RE_RM_TEST_FILE.sub('', script)
    script = re.sub(r'\n{3,}', '\n\n', script)
    return script.strip()
