"""Shared platform checks do not classify Cygwin as native Windows."""

from types import SimpleNamespace

import pytest

from nexus.utils import platform


@pytest.mark.parametrize(
    ('name', 'windows'),
    [
        ('win32', True),
        ('cygwin', False),
        ('linux', False),
        ('linux2', False),
        ('darwin', False),
        ('freebsd14', False),
    ],
)
def test_host_platform_predicate(monkeypatch, name, windows):
    monkeypatch.setattr(platform, 'sys', SimpleNamespace(platform=name))
    assert platform.is_windows() is windows


def test_platform_is_checked_at_call_time(monkeypatch):
    host = SimpleNamespace(platform='win32')
    monkeypatch.setattr(platform, 'sys', host)
    assert platform.is_windows()
    host.platform = 'linux'
    assert not platform.is_windows()
