"""Platform dispatch and native horizontal wheel ABI without desktop input."""

import ctypes

from unittest.mock import Mock

import pytest

from nexus.runtime.server import gui


@pytest.mark.parametrize('delta', [120, -120, 2])
def test_windows_horizontal_uses_hwheel_and_keeps_raw_delta(monkeypatch, delta):
    native = Mock()
    library = Mock(mouse_event=native)
    loader = Mock(return_value=library)
    monkeypatch.setattr(ctypes, 'WinDLL', loader, raising=False)
    monkeypatch.setattr(gui, 'is_windows', lambda: True)
    backend = Mock()
    gui.PyAutoGUIBackend(backend).scroll(dx=delta, dy=-120)
    loader.assert_called_once_with('user32')
    native.assert_called_once_with(0x1000, 0, 0, ctypes.c_ulong(delta).value, 0)
    assert native.argtypes == [ctypes.c_ulong] * 4 + [ctypes.c_size_t]
    assert native.restype is None
    backend.hscroll.assert_not_called()
    backend.vscroll.assert_called_once_with(-120)


def test_windows_zero_horizontal_does_not_load_native_library(monkeypatch):
    loader = Mock()
    monkeypatch.setattr(ctypes, 'WinDLL', loader, raising=False)
    monkeypatch.setattr(gui, 'is_windows', lambda: True)
    backend = Mock()
    gui.PyAutoGUIBackend(backend).scroll(dy=120)
    loader.assert_not_called()
    backend.vscroll.assert_called_once_with(120)


def test_non_windows_scroll_still_uses_pyautogui(monkeypatch):
    monkeypatch.setattr(gui, 'is_windows', lambda: False)
    backend = Mock()
    gui.PyAutoGUIBackend(backend).scroll(dx=-120, dy=120)
    backend.hscroll.assert_called_once_with(-120)
    backend.vscroll.assert_called_once_with(120)
