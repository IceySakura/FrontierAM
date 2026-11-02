import os

import pytest

from nexus.utils.platform import is_windows


if is_windows():
    collect_ignore = [
        'test_aggregate.py',
        'test_async_detection.py',
        'test_builtin.py',
        'test_builtin_src_present.py',
        'test_capture.py',
        'test_capture_integrity.py',
        'test_empty_submission.py',
        'test_interactive.py',
        'test_keys.py',
        'test_multiline_batch.py',
        'test_pty_io.py',
        'test_recovery.py',
        'test_screen.py',
        'test_session.py',
        'test_session_registry.py',
        'test_timeout.py',
    ]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip patched-Bash integration cases unless its path is explicit."""
    if os.environ.get('NEXUS_BASH_BIN'):
        return

    missing_bash = pytest.mark.skip(reason='NEXUS_BASH_BIN is required for patched-Bash integration tests')
    for item in items:
        if item.get_closest_marker('integration') is not None:
            item.add_marker(missing_bash)
