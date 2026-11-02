"""Unit tests for empty-submission detection (no live shell needed).

Empty/newline-only submissions are synthesized as the reserved id-0 no-op so
they never write phantom blank lines that would stall the start poll for 5s.
Whitespace-only input is ALSO empty: bash does not invoke execute_command for a
blank/whitespace-only line, so no cmd file is ever written; we synthesize id-0.
"""

from nexus.runtime.executors.terminal.session import _is_empty_submission


def test_empty_string_is_empty():
    assert _is_empty_submission('') is True


def test_newline_only_is_empty():
    assert _is_empty_submission('\n') is True
    assert _is_empty_submission('\n\n') is True


def test_whitespace_only_is_empty():
    # Bash does not call execute_command for whitespace-only input, so no
    # cmd_<id>.json is produced. Treat it as a reserved id-0 no-op.
    assert _is_empty_submission('   ') is True
    assert _is_empty_submission('\t') is True
    assert _is_empty_submission('  \n  ') is True


def test_real_command_is_not_empty():
    assert _is_empty_submission('echo hi') is False
    assert _is_empty_submission('a=1\nb=2') is False
    assert _is_empty_submission('  echo hi') is False  # Non-whitespace input is executable.
