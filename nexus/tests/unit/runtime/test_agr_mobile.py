"""Unit tests for AGRMobileRuntime and AGRMobileProvider.

All external dependencies (e2b, appium) are mocked, so these tests run
without any real sandbox or Appium installation.
"""

from __future__ import annotations

import base64
import importlib
import importlib.machinery
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Stub out appium modules BEFORE importing anything that might trigger them.
# This allows tests to run without Appium-Python-Client installed.
# ---------------------------------------------------------------------------


def _make_appium_stubs() -> None:
    """Insert lightweight appium stubs into sys.modules for whatever the real package lacks."""
    # appium.options.common — provides AppiumOptions in Appium 3+
    _ensure_mock_attr('appium.options.common', 'AppiumOptions')
    _ensure_mock_attr('appium.options.android', 'UiAutomator2Options')
    _ensure_mock_attr('appium.webdriver.appium_connection', 'AppiumConnection')
    _ensure_mock_attr('appium.webdriver.client_config', 'AppiumClientConfig')

    # appium.webdriver.common.touch_action — removed in Appium 3+; stub it so
    # any stray import in test helpers doesn't explode
    _ensure_mock_module(
        'appium.webdriver.common.touch_action',
        TouchAction=MagicMock(name='TouchAction'),
    )


def _import_or_stub(dotted_path: str, *, is_package: bool | None = None):
    """Return the real module when importable, else register a stub under ``dotted_path``.

    Never shadow a working submodule: replacing e.g. ``appium.webdriver.appium_connection``
    with a bare MagicMock breaks the real ``appium.webdriver`` package import, because the
    import machinery then looks for ``__spec__`` on the mock.

    When *is_package* is ``None`` (the default), the stub is created as a **non-package**
    module.  Pass ``is_package=True`` only for paths that genuinely represent packages.
    """
    if dotted_path in sys.modules:
        return sys.modules[dotted_path]
    try:
        return importlib.import_module(dotted_path)
    except Exception:  # noqa: BLE001  missing/incompatible optional dependency
        mod = MagicMock(name=dotted_path)
        treat_as_package = bool(is_package)
        mod.__spec__ = importlib.machinery.ModuleSpec(dotted_path, loader=None, is_package=treat_as_package)
        if treat_as_package:
            mod.__path__ = []
        sys.modules[dotted_path] = mod
        return mod


def _ensure_mock_attr(dotted_path: str, attr: str) -> None:
    """Make sure ``dotted_path.attr`` exists (as a MagicMock) in sys.modules."""
    mod = _import_or_stub(dotted_path)
    if getattr(mod, attr, None) is None:
        setattr(mod, attr, MagicMock(name=attr))


def _ensure_mock_module(dotted_path: str, **attrs: Any) -> None:
    """Ensure ``dotted_path`` exists in sys.modules with given attributes."""
    mod = _import_or_stub(dotted_path)
    for name, val in attrs.items():
        if getattr(mod, name, None) is None:
            setattr(mod, name, val)


_make_appium_stubs()

from nexus.runtime.providers.agr_mobile.client import AGRMobileRuntime  # noqa: E402
from nexus.runtime.providers.agr_mobile import AGRMobileProvider  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime(**kwargs) -> AGRMobileRuntime:
    defaults = dict(
        sandbox_id='test-sandbox-123',
        api_key='test-api-key',
        domain='ap-guangzhou.tencentags.com',
    )
    defaults.update(kwargs)
    return AGRMobileRuntime(**defaults)


def _make_mock_driver() -> MagicMock:
    """Build a minimal mock Appium WebDriver."""
    driver = MagicMock()
    driver.get_screenshot_as_png.return_value = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
    driver.get_window_size.return_value = {'width': 1080, 'height': 1920}
    driver.switch_to.active_element.send_keys = MagicMock()
    driver.capabilities = {
        'deviceName': 'Pixel Test',
        'platformVersion': '14',
        'automationName': 'UiAutomator2',
    }
    driver.current_package = 'com.android.settings'
    driver.current_activity = '.Settings'
    return driver


def _make_mock_sandbox() -> MagicMock:
    sandbox = MagicMock()
    return sandbox


class _FakeCreatedSandbox:
    def __init__(self, sandbox_id: str = 'created-sandbox-123') -> None:
        self.sandbox_id = sandbox_id
        self.kill_calls = 0

    def kill(self) -> bool:
        self.kill_calls += 1
        return True


# ---------------------------------------------------------------------------
# AGRMobileRuntime — construction
# ---------------------------------------------------------------------------


class TestAGRMobileRuntimeInit:
    def test_fields_stored(self):
        rt = _make_runtime(appium_port=4724, ready_timeout=60)
        assert rt.sandbox_id == 'test-sandbox-123'
        assert rt.api_key == 'test-api-key'
        assert rt.domain == 'ap-guangzhou.tencentags.com'
        assert rt.appium_port == 4724
        assert rt.ready_timeout == 60
        assert not rt.connected

    def test_appium_url_with_domain(self):
        rt = _make_runtime()
        url = rt._get_appium_url()
        assert 'test-sandbox-123' in url
        assert 'ap-guangzhou.tencentags.com' in url
        assert str(rt.appium_port) in url

    def test_appium_url_without_domain(self):
        rt = AGRMobileRuntime(sandbox_id='s1', domain=None)
        url = rt._get_appium_url()
        assert url == f'http://localhost:{AGRMobileRuntime.APPIUM_PORT}'


# ---------------------------------------------------------------------------
# AGRMobileRuntime — connect / disconnect
# ---------------------------------------------------------------------------


class TestAGRMobileRuntimeConnect:
    @pytest.fixture
    def mock_sandbox(self):
        return _make_mock_sandbox()

    @pytest.fixture
    def mock_driver(self):
        return _make_mock_driver()

    async def test_connect_calls_sandbox_connect(self, mock_sandbox, mock_driver):
        rt = _make_runtime()

        with (
            patch('nexus.runtime.providers.agr_mobile.client._check_e2b_dependency') as mock_e2b,
            patch('nexus.runtime.providers.agr_mobile.client._check_appium_dependency') as mock_appium,
            patch.object(rt, '_wait_for_appium_ready', new_callable=AsyncMock) as mock_wait,
        ):
            MockSandbox = MagicMock()
            MockSandbox.connect.return_value = mock_sandbox
            mock_e2b.return_value = MockSandbox

            mock_webdriver = MagicMock()
            mock_webdriver.Remote.return_value = mock_driver
            mock_appium.return_value = mock_webdriver

            result = await rt.connect()

        assert result is True
        assert rt.connected
        MockSandbox.connect.assert_called_once_with(
            'test-sandbox-123',
            api_key='test-api-key',
            domain='ap-guangzhou.tencentags.com',
        )
        mock_wait.assert_awaited_once()

    async def test_disconnect_calls_driver_quit(self, mock_driver):
        rt = _make_runtime()
        rt._driver = mock_driver
        rt._sandbox = _make_mock_sandbox()
        rt._connected = True

        result = await rt.disconnect()

        assert result is True
        assert not rt.connected
        assert rt._driver is None
        mock_driver.quit.assert_called_once()

    async def test_disconnect_when_not_connected_is_safe(self):
        rt = _make_runtime()
        # Should not raise even if driver is None
        result = await rt.disconnect()
        assert result is True

    async def test_call_executor_raises_when_not_connected(self):
        rt = _make_runtime()
        with pytest.raises(RuntimeError, match='not connected'):
            await rt.call_executor('gui_click', {'x': 100, 'y': 200})


# ---------------------------------------------------------------------------
# AGRMobileRuntime — call_executor actions
# ---------------------------------------------------------------------------


class TestAGRMobileRuntimeExecutors:
    @pytest.fixture
    def connected_runtime(self) -> AGRMobileRuntime:
        rt = _make_runtime()
        rt._driver = _make_mock_driver()
        rt._connected = True
        return rt

    async def test_gui_click(self, connected_runtime):
        result = await connected_runtime.call_executor('gui_click', {'x': 100, 'y': 200})
        assert result == {'success': True}
        connected_runtime._driver.assert_not_called()  # tap via TouchAction

    async def test_gui_double_click(self, connected_runtime):
        result = await connected_runtime.call_executor('gui_double_click', {'x': 50, 'y': 80})
        assert result == {'success': True}

    async def test_gui_long_click_default_duration(self, connected_runtime):
        result = await connected_runtime.call_executor('gui_long_click', {'x': 10, 'y': 20})
        assert result == {'success': True}

    async def test_gui_long_click_custom_duration(self, connected_runtime):
        result = await connected_runtime.call_executor('gui_long_click', {'x': 10, 'y': 20, 'duration': 2000})
        assert result == {'success': True}

    async def test_gui_drag(self, connected_runtime):
        result = await connected_runtime.call_executor('gui_drag', {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 500})
        assert result == {'success': True}
        connected_runtime._driver.swipe.assert_called_once_with(0, 0, 100, 500, 300)

    async def test_gui_drag_custom_duration(self, connected_runtime):
        result = await connected_runtime.call_executor(
            'gui_drag', {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 500, 'duration': 1000}
        )
        assert result == {'success': True}
        connected_runtime._driver.swipe.assert_called_once_with(0, 0, 100, 500, 1000)

    async def test_gui_type(self, connected_runtime):
        result = await connected_runtime.call_executor('gui_type', {'text': 'hello'})
        assert result == {'success': True}
        connected_runtime._driver.switch_to.active_element.send_keys.assert_called_once_with('hello')

    async def test_gui_press_key_back(self, connected_runtime):
        result = await connected_runtime.call_executor('gui_press_key', {'key': 'Back'})
        assert result == {'success': True}
        connected_runtime._driver.press_keycode.assert_called_once_with(4)

    async def test_gui_press_key_home(self, connected_runtime):
        result = await connected_runtime.call_executor('gui_press_key', {'key': 'Home'})
        assert result == {'success': True}
        connected_runtime._driver.press_keycode.assert_called_once_with(3)

    async def test_gui_press_key_unsupported_raises(self, connected_runtime):
        result = await connected_runtime.call_executor('gui_press_key', {'key': 'F13'})
        assert result['success'] is False
        assert 'error' in result

    async def test_gui_screenshot_returns_base64_png(self, connected_runtime):
        result = await connected_runtime.call_executor('gui_screenshot', {})
        assert result['success'] is True
        assert result['format'] == 'png'
        # Verify base64 decode round-trip
        decoded = base64.b64decode(result['image'])
        assert decoded == connected_runtime._driver.get_screenshot_as_png.return_value
        assert result['width'] == 1080
        assert result['height'] == 1920

    async def test_gui_screenshot_caches_screen_size(self, connected_runtime):
        await connected_runtime.call_executor('gui_screenshot', {})
        # Second call should NOT call get_window_size again
        connected_runtime._driver.get_window_size.reset_mock()
        await connected_runtime.call_executor('gui_screenshot', {})
        connected_runtime._driver.get_window_size.assert_not_called()

    async def test_gui_get_screen_size(self, connected_runtime):
        result = await connected_runtime.call_executor('gui_get_screen_size', {})
        assert result == {'success': True, 'width': 1080, 'height': 1920}
        connected_runtime._driver.get_window_size.assert_called_once()

    async def test_gui_get_device_info(self, connected_runtime):
        connected_runtime._driver.execute_script.side_effect = [
            'Physical size: 1080x1920',
            'Physical density: 420',
        ]
        result = await connected_runtime.call_executor('gui_get_device_info', {})
        assert result == {
            'success': True,
            'device_name': 'Pixel Test',
            'platform_version': '14',
            'automation_name': 'UiAutomator2',
            'window_width': 1080,
            'window_height': 1920,
            'wm_size': 'Physical size: 1080x1920',
            'wm_density': 'Physical density: 420',
            'current_package': 'com.android.settings',
            'current_activity': '.Settings',
        }

    async def test_gui_launch_app(self, connected_runtime):
        result = await connected_runtime.call_executor('gui_launch_app', {'app': 'com.example.app'})
        assert result == {'success': True}
        connected_runtime._driver.activate_app.assert_called_once_with('com.example.app')

    async def test_gui_close_app(self, connected_runtime):
        result = await connected_runtime.call_executor('gui_close_app', {'app': 'com.example.app'})
        assert result == {'success': True}
        connected_runtime._driver.terminate_app.assert_called_once_with('com.example.app')

    async def test_gui_reset_apps(self, connected_runtime):
        connected_runtime._driver.execute_script.side_effect = [
            '\n'.join([
                'mActivityComponent=com.example.one/.MainActivity',
                'packageName=com.example.one',
                'mActivityComponent=com.android.launcher3/.Launcher',
            ]),
            'Recent tasks:\n  * Recent #0: Task{123 #9 type=standard A=10123:com.example.two}',
            '',
            '',
        ]
        result = await connected_runtime.call_executor('gui_reset_apps', {})
        assert result == {
            'success': True,
            'stopped_packages': ['com.example.one', 'com.example.two'],
        }
        assert connected_runtime._driver.execute_script.call_args_list == [
            (
                (
                    'mobile: shell',
                    {
                        'command': 'dumpsys',
                        'args': ['activity', 'activities'],
                    },
                ),
            ),
            (
                (
                    'mobile: shell',
                    {
                        'command': 'dumpsys',
                        'args': ['activity', 'recents'],
                    },
                ),
            ),
            (
                (
                    'mobile: shell',
                    {'command': 'am', 'args': ['force-stop', 'com.example.one']},
                ),
            ),
            (
                (
                    'mobile: shell',
                    {'command': 'am', 'args': ['force-stop', 'com.example.two']},
                ),
            ),
        ]

    async def test_unsupported_executor_returns_error(self, connected_runtime):
        result = await connected_runtime.call_executor('gui_nonexistent', {})
        assert result['success'] is False
        assert 'Unsupported executor' in result['error']

    async def test_executor_exception_returns_error(self, connected_runtime):
        connected_runtime._driver.swipe.side_effect = RuntimeError('connection lost')
        result = await connected_runtime.call_executor('gui_drag', {'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10})
        assert result['success'] is False
        assert 'connection lost' in result['error']

    async def test_call_function_raises(self, connected_runtime):
        with pytest.raises(NotImplementedError):
            await connected_runtime.call_function(lambda: None)

    async def test_call_tool_raises(self, connected_runtime):
        with pytest.raises(NotImplementedError):
            await connected_runtime.call_tool('some.path', {}, None)


# ---------------------------------------------------------------------------
# AGRMobileProvider
# ---------------------------------------------------------------------------


class TestAGRMobileProvider:
    def test_default_fields(self):
        provider = AGRMobileProvider(api_key='k', domain='d')
        assert provider.api_key == 'k'
        assert provider.domain == 'd'
        assert provider.appium_port == AGRMobileRuntime.APPIUM_PORT
        assert provider.ready_timeout == AGRMobileRuntime.APPIUM_READY_TIMEOUT

    async def test_acquire_raises_without_sandbox_locator(self):
        provider = AGRMobileProvider(api_key='k', domain='d')
        with pytest.raises(ValueError, match='sandbox_id or sandbox_template is required'):
            await provider.acquire('main')

    async def test_acquire_raises_when_template_create_has_no_api_key(self):
        provider = AGRMobileProvider(domain='d', sandbox_template='mobile-v1')
        with pytest.raises(ValueError, match='api_key is required when creating a sandbox via sandbox_template'):
            await provider.acquire('main')

    async def test_acquire_creates_template_sandbox_and_connects_runtime(self):
        provider = AGRMobileProvider(api_key='k', domain='example.com', sandbox_template='mobile-v1')

        mock_runtime = AsyncMock(spec=AGRMobileRuntime)
        mock_runtime.connected = True

        created_sandbox = _FakeCreatedSandbox()

        with (
            patch('nexus.runtime.providers.agr_mobile.provider._load_e2b_sandbox_class') as mock_loader,
            patch(
                'nexus.runtime.providers.agr_mobile.provider.AGRMobileRuntime', return_value=mock_runtime
            ) as MockRuntime,
        ):
            mock_loader.return_value = MagicMock(create=MagicMock(return_value=created_sandbox))
            result = await provider.acquire('main')

        mock_loader.return_value.create.assert_called_once_with(
            template='mobile-v1',
            timeout=1800,
            api_key='k',
            request_timeout=1800.0,
            domain='example.com',
        )
        MockRuntime.assert_called_once_with(
            sandbox_id='created-sandbox-123',
            api_key='k',
            domain='example.com',
            appium_port=AGRMobileRuntime.APPIUM_PORT,
            ready_timeout=AGRMobileRuntime.APPIUM_READY_TIMEOUT,
        )
        mock_runtime.connect.assert_awaited_once()
        assert result is mock_runtime
        assert provider._created_sandboxes['main'] is created_sandbox

    async def test_acquire_creates_and_connects_runtime(self):
        provider = AGRMobileProvider(api_key='k', domain='example.com')

        mock_runtime = AsyncMock(spec=AGRMobileRuntime)
        mock_runtime.connected = True

        with patch(
            'nexus.runtime.providers.agr_mobile.provider.AGRMobileRuntime', return_value=mock_runtime
        ) as MockRuntime:
            result = await provider.acquire('main', sandbox_id='sb-001')

        MockRuntime.assert_called_once_with(
            sandbox_id='sb-001',
            api_key='k',
            domain='example.com',
            appium_port=AGRMobileRuntime.APPIUM_PORT,
            ready_timeout=AGRMobileRuntime.APPIUM_READY_TIMEOUT,
        )
        mock_runtime.connect.assert_awaited_once()
        assert result is mock_runtime
        assert provider._active_runtimes['main'] is mock_runtime

    async def test_acquire_params_override_config(self):
        provider = AGRMobileProvider(api_key='default-key', domain='default.com')

        mock_runtime = AsyncMock(spec=AGRMobileRuntime)
        with patch(
            'nexus.runtime.providers.agr_mobile.provider.AGRMobileRuntime', return_value=mock_runtime
        ) as MockRuntime:
            await provider.acquire('main', sandbox_id='sb-002', api_key='override-key', domain='override.com')

        _, kwargs = MockRuntime.call_args
        assert kwargs['api_key'] == 'override-key'
        assert kwargs['domain'] == 'override.com'

    async def test_release_calls_disconnect(self):
        provider = AGRMobileProvider(api_key='k', domain='d')

        mock_runtime = AsyncMock(spec=AGRMobileRuntime)
        provider._active_runtimes['main'] = mock_runtime

        await provider.release('main')

        mock_runtime.disconnect.assert_awaited_once()
        assert 'main' not in provider._active_runtimes

    async def test_release_kills_provider_created_sandbox(self):
        provider = AGRMobileProvider(api_key='k', domain='d')

        mock_runtime = AsyncMock(spec=AGRMobileRuntime)
        created_sandbox = _FakeCreatedSandbox()
        provider._active_runtimes['main'] = mock_runtime
        provider._created_sandboxes['main'] = created_sandbox

        await provider.release('main')

        mock_runtime.disconnect.assert_awaited_once()
        assert created_sandbox.kill_calls == 1
        assert 'main' not in provider._created_sandboxes

    async def test_release_nonexistent_name_is_safe(self):
        provider = AGRMobileProvider(api_key='k', domain='d')
        # Should not raise
        await provider.release('nonexistent')

    def test_registered_in_registry(self):
        from nexus.runtime.providers.registry import runtime_provider_registry

        assert 'agr_mobile' in runtime_provider_registry
