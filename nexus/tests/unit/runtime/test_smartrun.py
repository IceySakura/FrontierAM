"""Unit tests for SmartRunRuntime and SmartRunRuntimeProvider. All HTTP calls are mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime(**kwargs):
    from nexus.runtime.providers.smartrun.client import SmartRunRuntime

    defaults = {
        'device_id': '00008020-000E6D863C01002E',
        'session_id': 'session-abc123',
        'screen_size': (1125, 2436),
        'platform': 'ios',
        'api_base_url': 'https://smartrun.woa.com',
        'token': 'mcp_test_token',
    }
    defaults.update(kwargs)
    return SmartRunRuntime(**defaults)


class TestSmartRunRuntimeInit:
    def test_fields_stored(self):
        rt = _make_runtime()
        assert rt.device_id == '00008020-000E6D863C01002E'
        assert rt.session_id == 'session-abc123'
        assert rt.screen_size == (1125, 2436)
        assert rt.platform == 'ios'
        assert not rt.connected

    async def test_not_connected_raises_on_executor(self):
        rt = _make_runtime()
        with pytest.raises(RuntimeError, match='not connected'):
            await rt.call_executor('gui_click', {'x': 100, 'y': 200})


class TestSmartRunRuntimeExecutors:
    """Test call_executor routing. All _call() calls are mocked."""

    @pytest.fixture
    def rt(self):
        rt = _make_runtime()
        rt._connected = True
        return rt

    @pytest.fixture
    def mock_call(self, rt):
        with patch.object(rt, '_call', new_callable=AsyncMock) as m:
            yield m

    async def test_gui_get_screen_size_no_http(self, rt, mock_call):
        result = await rt.call_executor('gui_get_screen_size', {})
        assert result == {'success': True, 'width': 1125, 'height': 2436}
        mock_call.assert_not_called()

    async def test_gui_click(self, rt, mock_call):
        mock_call.return_value = {'success': True}
        result = await rt.call_executor('gui_click', {'x': 100, 'y': 200})
        assert result == {'success': True}
        mock_call.assert_awaited_once_with(
            'tap',
            {'device_id': rt.device_id, 'session_id': rt.session_id, 'json_output': True, 'x': 100, 'y': 200},
        )

    async def test_gui_double_click(self, rt, mock_call):
        mock_call.return_value = {'success': True}
        result = await rt.call_executor('gui_double_click', {'x': 50, 'y': 80})
        assert result == {'success': True}
        mock_call.assert_awaited_once_with(
            'double_tap',
            {'device_id': rt.device_id, 'session_id': rt.session_id, 'json_output': True, 'x': 50, 'y': 80},
        )

    async def test_gui_long_click_default_duration(self, rt, mock_call):
        mock_call.return_value = {'success': True}
        result = await rt.call_executor('gui_long_click', {'x': 10, 'y': 20})
        assert result == {'success': True}
        call_args = mock_call.call_args[0]
        assert call_args[0] == 'long_press'
        assert call_args[1]['duration'] == 1000

    async def test_gui_long_click_custom_duration(self, rt, mock_call):
        mock_call.return_value = {'success': True}
        await rt.call_executor('gui_long_click', {'x': 10, 'y': 20, 'duration': 2000})
        assert mock_call.call_args[0][1]['duration'] == 2000

    async def test_gui_drag(self, rt, mock_call):
        mock_call.return_value = {'success': True}
        result = await rt.call_executor('gui_drag', {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 500})
        assert result == {'success': True}
        call_args = mock_call.call_args[0]
        assert call_args[0] == 'swipe'
        assert call_args[1]['start_x'] == 0
        assert call_args[1]['end_y'] == 500
        assert call_args[1]['duration'] == 300

    async def test_gui_drag_custom_duration(self, rt, mock_call):
        mock_call.return_value = {'success': True}
        await rt.call_executor('gui_drag', {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 500, 'duration': 1000})
        assert mock_call.call_args[0][1]['duration'] == 1000

    async def test_gui_type(self, rt, mock_call):
        mock_call.return_value = {'success': True}
        result = await rt.call_executor('gui_type', {'text': 'hello world'})
        assert result == {'success': True}
        mock_call.assert_awaited_once_with(
            'input_text',
            {'device_id': rt.device_id, 'session_id': rt.session_id, 'json_output': True, 'text': 'hello world'},
        )

    async def test_gui_press_key_home(self, rt, mock_call):
        mock_call.return_value = {'success': True}
        result = await rt.call_executor('gui_press_key', {'key': 'Home'})
        assert result == {'success': True}
        mock_call.assert_awaited_once_with(
            'press_home',
            {'device_id': rt.device_id, 'session_id': rt.session_id, 'json_output': True},
        )

    async def test_gui_press_key_back(self, rt, mock_call):
        mock_call.return_value = {'success': True}
        result = await rt.call_executor('gui_press_key', {'key': 'Back'})
        assert result == {'success': True}
        mock_call.assert_awaited_once_with(
            'press_back',
            {'device_id': rt.device_id, 'session_id': rt.session_id, 'json_output': True},
        )

    async def test_gui_press_key_unsupported(self, rt, mock_call):
        result = await rt.call_executor('gui_press_key', {'key': 'F13'})
        assert result['success'] is False
        assert 'unsupported key' in result['error']
        mock_call.assert_not_called()

    async def test_gui_launch_app(self, rt, mock_call):
        mock_call.return_value = {'success': True}
        result = await rt.call_executor('gui_launch_app', {'app': 'com.example.app'})
        assert result == {'success': True}
        mock_call.assert_awaited_once_with(
            'launch_app',
            {
                'device_id': rt.device_id,
                'session_id': rt.session_id,
                'json_output': True,
                'package_name': 'com.example.app',
            },
        )

    async def test_gui_close_app(self, rt, mock_call):
        mock_call.return_value = {'success': True}
        result = await rt.call_executor('gui_close_app', {'app': 'com.example.app'})
        assert result == {'success': True}
        mock_call.assert_awaited_once_with(
            'stop_app',
            {
                'device_id': rt.device_id,
                'session_id': rt.session_id,
                'json_output': True,
                'package_name': 'com.example.app',
            },
        )

    async def test_unsupported_executor_returns_error(self, rt, mock_call):
        result = await rt.call_executor('gui_nonexistent', {})
        assert result['success'] is False
        assert 'unsupported executor' in result['error']

    async def test_screenshot_downloads_and_encodes(self, rt, mock_call):
        import base64

        fake_image = b'\xff\xd8\xff' + b'\x00' * 100  # fake JPEG bytes

        mock_image_resp = MagicMock()
        mock_image_resp.content = fake_image

        with patch('nexus.runtime.providers.smartrun.client.http_client') as mock_http:
            mock_http.post = AsyncMock(
                return_value=MagicMock(
                    status_code=200,
                    text='data: {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"{\\"success\\":true,\\"screenshot_url\\":\\"https://cdn.example.com/shot.jpg\\"}"}]}}',
                )
            )
            mock_http.get = AsyncMock(return_value=mock_image_resp)
            rt2 = _make_runtime()
            rt2._connected = True
            result = await rt2.call_executor('gui_screenshot', {})

        assert result['success'] is True
        assert result['format'] == 'jpg'
        assert result['width'] == 1125
        assert result['height'] == 2436
        decoded = base64.b64decode(result['image'])
        assert decoded == fake_image

    async def test_invoke_raises(self, rt):
        from nexus.runtime.models import InvokeRequest

        req = InvokeRequest(import_path='mod::fn')
        with pytest.raises(NotImplementedError):
            await rt._invoke(req)

    async def test_call_tool_raises(self, rt):
        with pytest.raises(NotImplementedError):
            await rt.call_tool('mod::fn', {}, None)


class TestSmartRunRuntimeCall:
    """Test _call() HTTP parsing logic."""

    @pytest.fixture
    def rt(self):
        rt = _make_runtime()
        rt._connected = True
        return rt

    def _mock_sse_response(self, tool_result: dict) -> MagicMock:
        import json

        text = json.dumps(tool_result)
        sse_body = f'id: evt-1\ndata: {{"jsonrpc":"2.0","id":1,"result":{{"content":[{{"type":"text","text":{json.dumps(text)}}}]}}}}\n'
        resp = MagicMock()
        resp.status_code = 200
        resp.text = sse_body
        return resp

    async def test_call_parses_success(self, rt):
        with patch('nexus.runtime.providers.smartrun.client.http_client') as mock_http:
            mock_http.post = AsyncMock(return_value=self._mock_sse_response({'success': True, 'duration_ms': 100}))
            result = await rt._call('screenshot', {'device_id': 'x', 'session_id': 'y'})

        assert result == {'success': True, 'duration_ms': 100}

    async def test_call_raises_on_is_error(self, rt):
        import json

        err_content = [{'type': 'text', 'text': '设备被占用'}]
        sse_body = f'data: {{"jsonrpc":"2.0","id":1,"result":{{"isError":true,"content":{json.dumps(err_content)}}}}}\n'
        resp = MagicMock()
        resp.status_code = 200
        resp.text = sse_body

        with patch('nexus.runtime.providers.smartrun.client.http_client') as mock_http:
            mock_http.post = AsyncMock(return_value=resp)
            with pytest.raises(RuntimeError, match='设备被占用'):
                await rt._call('screenshot', {})

    async def test_call_raises_on_http_error(self, rt):
        resp = MagicMock()
        resp.status_code = 401
        resp.text = 'Unauthorized'

        with patch('nexus.runtime.providers.smartrun.client.http_client') as mock_http:
            mock_http.post = AsyncMock(return_value=resp)
            with pytest.raises(RuntimeError, match='HTTP error 401'):
                await rt._call('screenshot', {})

    async def test_call_raises_on_no_data_line(self, rt):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = 'id: evt-1\n'  # no data: line

        with patch('nexus.runtime.providers.smartrun.client.http_client') as mock_http:
            mock_http.post = AsyncMock(return_value=resp)
            with pytest.raises(RuntimeError, match='no data line'):
                await rt._call('screenshot', {})


class TestSmartRunRuntimeProviderInit:
    def test_token_from_arg(self):
        from nexus.runtime.providers.smartrun.provider import SmartRunRuntimeProvider

        p = SmartRunRuntimeProvider(token='mcp_xxx')
        assert p.token == 'mcp_xxx'

    def test_token_from_env(self, monkeypatch):
        monkeypatch.setenv('SMARTRUN_TOKEN', 'env_token')
        from nexus.runtime.providers.smartrun.provider import SmartRunRuntimeProvider

        p = SmartRunRuntimeProvider()
        assert p.token == 'env_token'

    def test_missing_token_raises(self, monkeypatch):
        monkeypatch.delenv('SMARTRUN_TOKEN', raising=False)
        from nexus.runtime.providers.smartrun.provider import SmartRunRuntimeProvider

        with pytest.raises(ValueError, match='token'):
            SmartRunRuntimeProvider()

    def test_default_fields(self):
        from nexus.runtime.providers.smartrun.provider import SmartRunRuntimeProvider

        p = SmartRunRuntimeProvider(token='t')
        assert p.api_base_url == 'https://smartrun.woa.com'
        assert p.duration == 1800
        assert p.startup_jitter_max == 0.0

    def test_registered_in_registry(self):
        import nexus.runtime.providers.smartrun  # noqa: F401 — triggers registration

        from nexus.runtime.providers.registry import runtime_provider_registry

        assert 'smartrun' in runtime_provider_registry


def _make_assign_response(
    device_id: str = '00008020-000E6D863C01002E',
    session_id: str = 'session-abc',
    screen_size: str = '1125x2436',
    platform: str = 'ios',
) -> dict:
    return {
        'device': {
            'device_id': device_id,
            'platform': platform,
            'screen_size': screen_size,
            'brand': 'Apple',
            'model': 'iPhone11,2',
            'os_version': '18.7.9',
            'is_online': True,
            'is_occupied': True,
        },
        'session': {
            'session_id': session_id,
            'auto_renew': True,
            'expire_at': 9999999999,
        },
        'success': True,
    }


class TestSmartRunRuntimeProviderAcquire:
    def _make_provider(self, **kwargs):
        from nexus.runtime.providers.smartrun.provider import SmartRunRuntimeProvider

        return SmartRunRuntimeProvider(token='mcp_test', **kwargs)

    async def test_acquire_assigns_and_connects(self):
        from nexus.runtime.providers.smartrun.client import SmartRunRuntime

        provider = self._make_provider()
        assign_resp = _make_assign_response()

        mock_runtime = AsyncMock(spec=SmartRunRuntime)
        mock_runtime.connected = True

        with (
            patch.object(provider, '_call_mcp', new_callable=AsyncMock, return_value=assign_resp),
            patch(
                'nexus.runtime.providers.smartrun.provider.SmartRunRuntime', return_value=mock_runtime
            ) as mock_runtime_cls,
        ):
            result = await provider.acquire('ep0')

        mock_runtime_cls.assert_called_once_with(
            device_id='00008020-000E6D863C01002E',
            session_id='session-abc',
            screen_size=(1125, 2436),
            platform='ios',
            api_base_url='https://smartrun.woa.com',
            token='mcp_test',
        )
        mock_runtime.connect.assert_awaited_once()
        assert result is mock_runtime
        assert provider._session_ids['ep0'] == 'session-abc'
        assert provider._device_ids['ep0'] == '00008020-000E6D863C01002E'

    async def test_acquire_passes_device_id_filter(self):
        provider = self._make_provider()
        assign_resp = _make_assign_response()

        with (
            patch.object(provider, '_call_mcp', new_callable=AsyncMock, return_value=assign_resp) as mock_call,
            patch('nexus.runtime.providers.smartrun.provider.SmartRunRuntime', return_value=AsyncMock()),
        ):
            await provider.acquire('ep0', device_id='00008020-000E6D863C01002E')

        call_args = mock_call.call_args[0]
        assert call_args[0] == 'assign_device'
        assert call_args[1]['device_id'] == '00008020-000E6D863C01002E'

    async def test_acquire_no_device_id_omits_field(self):
        provider = self._make_provider(platform='ios')
        assign_resp = _make_assign_response()

        with (
            patch.object(provider, '_call_mcp', new_callable=AsyncMock, return_value=assign_resp) as mock_call,
            patch('nexus.runtime.providers.smartrun.provider.SmartRunRuntime', return_value=AsyncMock()),
        ):
            await provider.acquire('ep0')

        call_args = mock_call.call_args[0][1]
        assert 'device_id' not in call_args
        assert call_args.get('platform') == 'ios'

    async def test_acquire_reuses_existing_runtime(self):
        provider = self._make_provider()
        mock_runtime = AsyncMock()
        provider._active_runtimes['ep0'] = mock_runtime
        provider._session_ids['ep0'] = 'session-abc'

        with patch.object(provider, '_call_mcp', new_callable=AsyncMock) as mock_call:
            result = await provider.acquire('ep0')

        assert result is mock_runtime
        mock_call.assert_not_called()

    async def test_acquire_parses_malformed_screen_size(self):
        provider = self._make_provider()
        assign_resp = _make_assign_response(screen_size='invalid')

        with (
            patch.object(provider, '_call_mcp', new_callable=AsyncMock, return_value=assign_resp),
            patch(
                'nexus.runtime.providers.smartrun.provider.SmartRunRuntime', return_value=AsyncMock()
            ) as mock_runtime_cls,
        ):
            await provider.acquire('ep0')

        # Falls back to (1080, 1920)
        call_kwargs = mock_runtime_cls.call_args[1]
        assert call_kwargs['screen_size'] == (1080, 1920)

    async def test_acquire_raises_on_assign_failure(self):
        provider = self._make_provider()

        with patch.object(provider, '_call_mcp', new_callable=AsyncMock, side_effect=RuntimeError('OCCUPIED')):
            with pytest.raises(RuntimeError, match='OCCUPIED'):
                await provider.acquire('ep0')

        assert 'ep0' not in provider._session_ids
        assert 'ep0' not in provider._device_ids

    async def test_release_calls_release_device(self):
        provider = self._make_provider()
        mock_runtime = AsyncMock()
        provider._active_runtimes['ep0'] = mock_runtime
        provider._session_ids['ep0'] = 'session-abc'
        provider._device_ids['ep0'] = 'dev-001'

        with patch.object(provider, '_call_mcp', new_callable=AsyncMock, return_value={'success': True}) as mock_call:
            await provider.release('ep0')

        mock_call.assert_awaited_once_with('release_device', {'device_id': 'dev-001', 'json_output': True})
        assert 'ep0' not in provider._session_ids
        assert 'ep0' not in provider._device_ids
        assert 'ep0' not in provider._active_runtimes

    async def test_release_tolerates_api_failure(self):
        provider = self._make_provider()
        provider._device_ids['ep0'] = 'dev-001'

        with patch.object(provider, '_call_mcp', new_callable=AsyncMock, side_effect=RuntimeError('timeout')):
            # Should not raise
            await provider.release('ep0')

        assert 'ep0' not in provider._device_ids

    async def test_release_nonexistent_is_safe(self):
        provider = self._make_provider()
        await provider.release('nonexistent')  # Must not raise

    async def test_get_runtime_id_returns_session_id(self):
        provider = self._make_provider()
        provider._session_ids['ep0'] = 'session-xyz'
        assert provider.get_runtime_id('ep0') == 'session-xyz'
        assert provider.get_runtime_id('missing') is None

    async def test_get_device_id(self):
        provider = self._make_provider()
        provider._device_ids['ep0'] = 'dev-001'
        assert provider.get_device_id('ep0') == 'dev-001'
        assert provider.get_device_id('missing') is None
