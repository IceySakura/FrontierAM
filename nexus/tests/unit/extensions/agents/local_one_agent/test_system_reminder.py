"""Tests for local_one_agent system-reminder helpers (agent.py)."""

from nexus.extensions.agents.local_one_agent.agent import (
    _append_system_reminder,
    _build_permission_switch_reminder,
    _wrap_system_reminder,
)
from nexus.extensions.agents.local_one_agent.context import (
    LocalContext,
    LocalPermission,
    LocalRuntime,
    LocalWorkspace,
)


def _local_ctx(*, os_name: str = 'macos', shell: str = 'zsh') -> LocalContext:
    """Minimal LocalContext; only ``runtime.os``/``runtime.shell`` affect reminder wording."""
    return LocalContext(
        workspace=LocalWorkspace(path='/workspace', scope='conversation'),
        permission=LocalPermission(level='basic'),
        runtime=LocalRuntime(os=os_name, arch='x86_64', shell=shell),
    )


class TestWrapSystemReminder:
    def test_wraps_text_in_tag(self):
        assert _wrap_system_reminder('hello') == '<system-reminder>\nhello\n</system-reminder>'


class TestAppendSystemReminder:
    def test_none_reminder_returns_content_unchanged(self):
        assert _append_system_reminder('do the thing', None) == 'do the thing'

    def test_str_content_joined_with_blank_line(self):
        assert _append_system_reminder('do the thing', 'REMINDER') == 'do the thing\n\nREMINDER'

    def test_list_content_appends_text_part_without_mutating(self):
        parts = [{'type': 'text', 'text': 'do the thing'}]
        result = _append_system_reminder(parts, 'REMINDER')
        assert result == [
            {'type': 'text', 'text': 'do the thing'},
            {'type': 'text', 'text': 'REMINDER'},
        ]
        assert parts == [{'type': 'text', 'text': 'do the thing'}]

    def test_list_content_none_reminder_returns_same_object(self):
        parts = [{'type': 'text', 'text': 'do the thing'}]
        assert _append_system_reminder(parts, None) is parts


class TestBuildPermissionSwitchReminder:
    def test_no_previous_level_returns_none(self):
        assert _build_permission_switch_reminder(LocalPermission(level='basic'), _local_ctx()) is None

    def test_same_level_returns_none(self):
        perm = LocalPermission(level='basic', previous_level='basic')
        assert _build_permission_switch_reminder(perm, _local_ctx()) is None

    def test_switch_to_readonly(self):
        perm = LocalPermission(level='readonly', previous_level='basic')
        text = _build_permission_switch_reminder(perm, _local_ctx())
        assert text is not None
        assert '只读模式' in text

    def test_switch_to_basic(self):
        perm = LocalPermission(level='basic', previous_level='readonly')
        text = _build_permission_switch_reminder(perm, _local_ctx())
        assert text is not None
        assert '编辑模式' in text

    def test_windows_powershell_uses_powershell_wording(self):
        perm = LocalPermission(level='basic', previous_level='readonly')
        posix_text = _build_permission_switch_reminder(perm, _local_ctx())
        powershell_text = _build_permission_switch_reminder(perm, _local_ctx(os_name='windows', shell='powershell'))
        assert powershell_text is not None
        assert powershell_text != posix_text
