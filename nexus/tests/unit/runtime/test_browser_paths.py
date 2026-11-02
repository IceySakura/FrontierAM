from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.runtime.executors import browser_paths
from nexus.utils.platform import is_windows


def available(monkeypatch, paths, platform='win32'):
    monkeypatch.setattr(browser_paths, 'is_windows', lambda: platform == 'win32')
    monkeypatch.setattr(Path, 'is_file', lambda path: str(path) in paths)


def test_explicit_path_has_priority(monkeypatch):
    available(monkeypatch, {'chosen', 'bundled'})
    assert browser_paths.resolve_browser_executable('chosen', 'bundled') == 'chosen'


def test_invalid_explicit_path_does_not_fallback(monkeypatch):
    available(monkeypatch, {'bundled'})
    with pytest.raises(FileNotFoundError, match='missing'):
        browser_paths.resolve_browser_executable('missing', 'bundled')


def test_windows_prefers_playwright(monkeypatch):
    available(monkeypatch, {'bundled'})
    assert browser_paths.resolve_browser_executable(None, 'bundled') == 'bundled'


def test_windows_falls_back_in_candidate_order(monkeypatch):
    available(monkeypatch, {'edge', 'chrome'})
    monkeypatch.setattr(browser_paths, '_windows_browser_candidates', lambda: map(Path, ['missing', 'chrome', 'edge']))
    assert browser_paths.resolve_browser_executable(None, 'bundled') == 'chrome'


@pytest.mark.skipif(not is_windows(), reason='Windows registry discovery')
def test_windows_discovers_chrome_before_edge(monkeypatch):
    import winreg

    def missing_key(*args):
        raise FileNotFoundError

    monkeypatch.setattr(winreg, 'OpenKey', missing_key)
    monkeypatch.setenv('ProgramFiles', 'C:/BrowserTest')
    monkeypatch.delenv('ProgramFiles(x86)', raising=False)
    monkeypatch.delenv('LOCALAPPDATA', raising=False)
    candidates = list(browser_paths._windows_browser_candidates())
    assert [path.name for path in candidates] == ['chrome.exe', 'msedge.exe']


def test_windows_missing_browser_is_actionable(monkeypatch):
    available(monkeypatch, set())
    monkeypatch.setattr(browser_paths, '_windows_browser_candidates', lambda: [])
    with pytest.raises(FileNotFoundError, match='Set executable_path'):
        browser_paths.resolve_browser_executable(None, 'bundled')


@pytest.mark.parametrize('nix_exists', [True, False])
def test_linux_preserves_nix_then_playwright(monkeypatch, nix_exists):
    nix = '/nix/runtime/nix-env/bin/chromium'
    # Path string formatting follows the host OS even when platform is mocked.
    available(monkeypatch, {str(Path(nix)), 'bundled'} if nix_exists else {'bundled'}, platform='linux')
    assert browser_paths.resolve_browser_executable(None, 'bundled') == (nix if nix_exists else 'bundled')


@pytest.mark.asyncio
async def test_failed_launch_cleans_up_and_allows_retry(monkeypatch):
    from nexus.runtime.executors import browser
    from nexus.runtime.executors.models.browser import CreateBrowserSessionRequest

    monkeypatch.setattr(browser, '_PLAYWRIGHT', None)
    monkeypatch.setattr(browser, '_BROWSER', None)
    monkeypatch.setattr(browser, 'BROWSER_SESSIONS', {})
    monkeypatch.setattr(browser, 'resolve_browser_executable', lambda *_: 'selected')
    pw = MagicMock()
    pw.stop = AsyncMock()
    launched = MagicMock()
    context = MagicMock()
    context.new_page = AsyncMock()
    context.close = AsyncMock()
    launched.new_context = AsyncMock(return_value=context)
    pw.chromium.launch = AsyncMock(side_effect=[RuntimeError('launch failed'), launched])
    manager = MagicMock()
    manager.start = AsyncMock(return_value=pw)
    monkeypatch.setattr(browser, 'async_playwright', lambda: manager)
    with pytest.raises(RuntimeError, match='launch failed'):
        await browser.create_browser_session(CreateBrowserSessionRequest())
    assert browser._PLAYWRIGHT is None
    assert browser._BROWSER is None
    pw.stop.assert_awaited_once()
    result = await browser.create_browser_session(CreateBrowserSessionRequest())
    assert result.session_id in browser.BROWSER_SESSIONS
    assert pw.chromium.launch.await_count == 2
