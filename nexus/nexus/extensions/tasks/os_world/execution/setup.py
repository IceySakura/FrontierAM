"""OSWorld setup action support for the local desktop runtime."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import posixpath
import re
import shlex
import tempfile
import uuid

from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import aiohttp

from nexus.extensions.tasks.os_world.execution.cdp import connect_chromium
from nexus.extensions.tasks.os_world.execution.commands import command_to_shell, osworld_user_command
from nexus.extensions.tasks.os_world.execution.guest_network import OSWORLD_PROXY_SERVER
from nexus.runtime.executors.models.file import FileItem
from nexus.utils.async_http import http_client_with_trust_env as http_client


logger = logging.getLogger(__name__)


_DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / 'nexus-osworld-cache'
_DOWNLOAD_RETRY_ATTEMPTS = 3
_DOWNLOAD_TIMEOUT_SECONDS = 300
_EXECUTE_UNTIL_FAILURE_LIMIT = 5
_EXECUTE_UNTIL_RETRY_DELAY_SECONDS = 0.3
OSWORLD_PROXY_SERVER_ARG = f'--proxy-server={OSWORLD_PROXY_SERVER}'
_TINYPROXY_CONFIG_PATH = '/tmp/tinyproxy.conf'
_TINYPROXY_LOG_PATH = '/tmp/osworld-tinyproxy.log'
# The AGS /execute endpoint enforces a ~120s server-side cap, so the whole probe
# loop must finish well under it: 6 * (12 + 3) = 90s worst case.
_PROXY_PROBE_URL = 'http://example.com'
_PROXY_PROBE_ATTEMPTS = 6
_PROXY_PROBE_INTERVAL_SECONDS = 3
_PROXY_PROBE_CURL_TIMEOUT_SECONDS = 12
_PROXY_PROBE_COMMAND_TIMEOUT_SECONDS = 110
_CHROME_CDP_SETUP_RETRY_ATTEMPTS = 15
_CHROME_CDP_SETUP_RETRY_DELAY_SECONDS = 5.0


def _single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _replace_setup_placeholders(command: str, *, client_password: str, width: int, height: int) -> str:
    return (
        command
        .replace('{CLIENT_PASSWORD}', client_password)
        .replace('{SCREEN_WIDTH_HALF}', str(width // 2))
        .replace('{SCREEN_HEIGHT_HALF}', str(height // 2))
        .replace('{SCREEN_WIDTH}', str(width))
        .replace('{SCREEN_HEIGHT}', str(height))
    )


def _replace_command_placeholders(
    command: str | list[Any],
    *,
    client_password: str,
    width: int,
    height: int,
) -> str | list[str]:
    if isinstance(command, list):
        return [
            _replace_setup_placeholders(str(item), client_password=client_password, width=width, height=height)
            for item in command
        ]
    return _replace_setup_placeholders(str(command), client_password=client_password, width=width, height=height)


# ── postconfig placeholder support (shared with evaluator.py) ──────────────────────
#
# Setup and evaluator.postconfig share placeholder substitution for
# ``{CLIENT_PASSWORD}`` and ``{SCREEN_*}``. Keep resolution in one helper so
# both phases execute the declared command with the same runtime values.


def replace_command_placeholders(
    command: str | list[Any],
    *,
    client_password: str,
    width: int,
    height: int,
) -> str | list[str]:
    """Public alias of the setup-side placeholder substitution, for postconfig reuse."""
    return _replace_command_placeholders(command, client_password=client_password, width=width, height=height)


async def screen_size_for_placeholders(runtime: Any) -> tuple[int, int]:
    """Screen size for {SCREEN_*} placeholders; falls back to OSWorld's 1280x800 default."""
    try:
        return await _screen_size(runtime)
    except Exception:  # noqa: BLE001 - a probe failure must not fail the whole evaluation
        logger.warning('OSWorld screen size probe failed; using 1280x800 for placeholders', exc_info=True)
        return 1280, 800


def _is_chrome_command(command: str) -> bool:
    return Path(command).name in {'google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser'}


def add_chrome_proxy_arg(command: str | list[Any], *, use_proxy: bool) -> str | list[str]:
    from nexus.extensions.tasks.os_world.execution.browser_launch import add_local_cdp_origins

    command = add_local_cdp_origins(command)
    if not use_proxy:
        return command
    if isinstance(command, list):
        if not command or not _is_chrome_command(str(command[0])):
            return command
        if any(str(item).startswith('--proxy-server=') for item in command):
            return [str(item) for item in command]
        return [*(str(item) for item in command), OSWORLD_PROXY_SERVER_ARG]

    try:
        parts = shlex.split(command)
    except ValueError:
        return command
    if not parts or not _is_chrome_command(parts[0]) or any(part.startswith('--proxy-server=') for part in parts):
        return command
    return f'{command} {shlex.quote(OSWORLD_PROXY_SERVER_ARG)}'


def osworld_runtime_use_proxy(runtime: Any) -> bool:
    return bool(getattr(runtime, 'osworld_use_proxy', False))


def _urls_match_for_tab_close(expected: str, actual: str) -> bool:
    expected_parts = urlparse(expected)
    actual_parts = urlparse(actual)
    expected_host = (expected_parts.hostname or '').removeprefix('www.').lower()
    actual_host = (actual_parts.hostname or '').removeprefix('www.').lower()
    expected_path = expected_parts.path.rstrip('/')
    actual_path = actual_parts.path.rstrip('/')
    return expected_host == actual_host and expected_path == actual_path


async def _run_chrome_cdp_setup(runtime: Any, operation: Any) -> None:
    last_error: Exception | None = None
    for attempt in range(_CHROME_CDP_SETUP_RETRY_ATTEMPTS):
        try:
            async with connect_chromium(runtime) as browser:
                await operation(browser)
            return
        except Exception as exc:
            last_error = exc
            if attempt + 1 < _CHROME_CDP_SETUP_RETRY_ATTEMPTS:
                await asyncio.sleep(_CHROME_CDP_SETUP_RETRY_DELAY_SECONDS)
    raise RuntimeError(f'Chrome CDP setup failed after {_CHROME_CDP_SETUP_RETRY_ATTEMPTS} attempts: {last_error}')


async def _open_chrome_tabs(runtime: Any, urls: list[str]) -> None:
    async def _open(browser: Any) -> None:
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        initial_pages = list(context.pages)
        for url in urls:
            page = await context.new_page()
            try:
                await page.goto(url, wait_until='commit', timeout=60000)
            except Exception:
                continue
        # Keep the standard task contract: close only initial about:blank pages.
        for page in initial_pages:
            if not page.is_closed() and page.url == 'about:blank':
                await page.close()

    await _run_chrome_cdp_setup(runtime, _open)


async def _close_chrome_tabs(runtime: Any, urls: list[str]) -> None:
    async def _close(browser: Any) -> None:
        for url in urls:
            for context in browser.contexts:
                page = next(
                    (
                        item
                        for item in context.pages
                        if not item.is_closed() and _urls_match_for_tab_close(url, item.url)
                    ),
                    None,
                )
                if page is not None:
                    await page.close()
                    break

    await _run_chrome_cdp_setup(runtime, _close)


async def _screen_size(runtime: Any) -> tuple[int, int]:
    response = await runtime.gui_get_screen_size()
    width = int(getattr(response, 'width', None) or 1280)
    height = int(getattr(response, 'height', None) or 800)
    return width, height


def _response_value(response: Any, field: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(field, default)
    return getattr(response, field, default)


def _download_cache_path(cache_dir: str | Path, url: str, path: str) -> Path:
    basename = posixpath.basename(path.rstrip('/')) or 'download'
    return Path(cache_dir) / f'{uuid.uuid5(uuid.NAMESPACE_URL, url)}_{basename}'


def _rewrite_hf_url(url: str) -> str:
    endpoint = os.environ.get('HF_ENDPOINT', '').rstrip('/')
    if endpoint and 'huggingface.co' in url:
        return url.replace('https://huggingface.co', endpoint).replace('http://huggingface.co', endpoint)
    return url


async def _download_url_bytes(url: str) -> bytes:
    response = await http_client.get(
        _rewrite_hf_url(url), timeout=aiohttp.ClientTimeout(total=_DOWNLOAD_TIMEOUT_SECONDS)
    )
    response.raise_for_status()
    return response.content


async def _write_bytes_atomic(path: Path, content: bytes) -> None:
    def _write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')
        try:
            tmp_path.write_bytes(content)
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    await asyncio.to_thread(_write)


async def _write_text_atomic(path: Path, content: str) -> None:
    def _write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')
        try:
            tmp_path.write_text(content, encoding='utf-8')
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    await asyncio.to_thread(_write)


async def _ensure_cached_download(url: str, path: str, cache_dir: str | Path) -> tuple[Path, bool]:
    cache_path = _download_cache_path(cache_dir, url, path)
    if cache_path.exists():
        return cache_path, True

    last_error: Exception | None = None
    for attempt in range(_DOWNLOAD_RETRY_ATTEMPTS):
        try:
            await _write_bytes_atomic(cache_path, await _download_url_bytes(url))
            return cache_path, False
        except Exception as exc:
            last_error = exc
            if attempt + 1 < _DOWNLOAD_RETRY_ATTEMPTS:
                await asyncio.sleep(0.5 * (attempt + 1))

    raise RuntimeError('Failed to download OSWorld setup asset') from last_error


def _upload_results(response: Any) -> list[Any]:
    return list(_response_value(response, 'results', []) or [])


def _dump_upload_result(item: Any) -> dict[str, object]:
    if hasattr(item, 'model_dump'):
        return cast(dict[str, object], item.model_dump())
    return {
        'path': str(_response_value(item, 'path', '')),
        'success': bool(_response_value(item, 'success', False)),
        'message': _response_value(item, 'message'),
    }


async def _upload_cached_file(runtime: Any, *, cache_path: Path, path: str) -> dict[str, object]:
    if hasattr(runtime, 'setup_upload_file'):
        try:
            response = await runtime.setup_upload_file(cache_path, path)
            status_code = int(_response_value(response, 'status_code', 0) or 0)
            success = status_code == 200
            message = 'File uploaded successfully' if success else str(_response_value(response, 'text', ''))[:500]
            return {
                'path': path,
                'success': success,
                'cache_path': str(cache_path),
                'upload_results': [{'path': path, 'success': success, 'message': message}],
            }
        except NotImplementedError:
            pass

    content = await asyncio.to_thread(cache_path.read_bytes)
    encoded = base64.b64encode(content).decode('ascii')
    response = await runtime.upload_file([FileItem(path=path, content=encoded, encoding='base64')], overwrite=True)
    upload_results = _upload_results(response)
    return {
        'path': path,
        'success': bool(upload_results)
        and all(bool(_response_value(item, 'success', False)) for item in upload_results),
        'cache_path': str(cache_path),
        'upload_results': [_dump_upload_result(item) for item in upload_results],
    }


def _load_proxy_config(proxy_config_file: str | Path) -> dict[str, object]:
    try:
        raw = json.loads(Path(proxy_config_file).read_text(encoding='utf-8'))
    except Exception as exc:
        raise ValueError('Failed to load OSWorld proxy config file') from exc

    if isinstance(raw, dict):
        raw = raw['proxies'] if isinstance(raw.get('proxies'), list) else [raw]
    if not isinstance(raw, list):
        raise ValueError('OSWorld proxy config must be an object or a list of objects')

    for item in raw:
        if not isinstance(item, dict):
            continue
        host = str(item.get('host') or '').strip()
        port = item.get('port')
        if not host or port is None:
            continue
        return {
            'host': host,
            'port': int(port),
            'username': item.get('username'),
            'password': item.get('password'),
            'protocol': str(item.get('protocol') or 'http'),
        }

    raise ValueError('OSWorld proxy config does not contain a usable proxy entry')


def _proxy_authority(proxy: dict[str, object]) -> str:
    host = str(proxy['host'])
    port = int(cast(int | str, proxy['port']))
    username = proxy.get('username')
    password = proxy.get('password')
    if username and password:
        return f'{username}:{password}@{host}:{port}'
    return f'{host}:{port}'


def _tinyproxy_config(proxy: dict[str, object], *, direct_domains: tuple[str, ...] = ()) -> str:
    for domain in direct_domains:
        if not re.fullmatch(r'\.?[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+', domain):
            raise ValueError('Direct proxy domains must be lowercase DNS names, optionally prefixed with a dot')
    return '\n'.join([
        'Port 18888',
        'Listen 127.0.0.1',
        'Allow 127.0.0.1',
        f'Upstream {proxy["protocol"]} {_proxy_authority(proxy)}',
        *(f'Upstream none "{domain}"' for domain in dict.fromkeys(direct_domains)),
        '',
    ])


async def _upload_proxy_config(runtime: Any, content: str) -> dict[str, object]:
    encoded = base64.b64encode(content.encode('utf-8')).decode('ascii')
    if hasattr(runtime, 'setup_upload_bytes'):
        response = await runtime.setup_upload_bytes(
            _TINYPROXY_CONFIG_PATH,
            content.encode('utf-8'),
            filename='tinyproxy.conf',
        )
        status_code = int(_response_value(response, 'status_code', _response_value(response, 'status', 0)) or 0)
        return {
            'path': _TINYPROXY_CONFIG_PATH,
            'success': status_code == 200,
            'status_code': status_code,
        }

    if hasattr(runtime, 'upload_file'):
        response = await runtime.upload_file(
            [FileItem(path=_TINYPROXY_CONFIG_PATH, content=encoded, encoding='base64')],
            overwrite=True,
        )
        upload_results = _upload_results(response)
        return {
            'path': _TINYPROXY_CONFIG_PATH,
            'success': bool(upload_results)
            and all(bool(_response_value(item, 'success', False)) for item in upload_results),
            'upload_results': [_dump_upload_result(item) for item in upload_results],
        }

    return {'path': _TINYPROXY_CONFIG_PATH, 'success': False, 'error': 'Runtime does not support file upload'}


def _tinyproxy_install_command(client_password: str) -> str:
    """Install tinyproxy and disable the auto-started default-config service.

    Kept separate from the start command: on the official OSWorld image the Debian
    ``tinyproxy`` package auto-creates and starts a systemd ``tinyproxy.service``
    (listening on 8888 with no upstream). Running install + start in one command
    would let that service hang the shell, so this command installs, then stops and
    disables the default service before we start our own instance from an explicit
    config. ``pkill -x tinyproxy`` matches by exact process name only, so it cannot
    SIGTERM the wrapping ``bash -lc`` shell (a plain ``pkill -f`` would).
    """
    password = shlex.quote(client_password)
    return f"""set -e
if ! command -v tinyproxy >/dev/null 2>&1; then
  if [ "$(id -u)" = "0" ]; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y tinyproxy
  elif command -v sudo >/dev/null 2>&1; then
    printf '%s\\n' {password} | sudo -S apt-get update
    printf '%s\\n' {password} | sudo -S env DEBIAN_FRONTEND=noninteractive apt-get install -y tinyproxy
  else
    exit 127
  fi
fi
if [ "$(id -u)" = "0" ]; then
  systemctl stop tinyproxy >/dev/null 2>&1 || true
  systemctl disable tinyproxy >/dev/null 2>&1 || true
elif command -v sudo >/dev/null 2>&1; then
  printf '%s\\n' {password} | sudo -S systemctl stop tinyproxy >/dev/null 2>&1 || true
  printf '%s\\n' {password} | sudo -S systemctl disable tinyproxy >/dev/null 2>&1 || true
fi
pkill -x tinyproxy >/dev/null 2>&1 || true
echo tinyproxy-install-done
"""


def _tinyproxy_launch_command() -> list[str]:
    """Command list passed to the keep-alive launch endpoint for our tinyproxy.

    ``tinyproxy -d`` stays in the foreground; the launch endpoint runs it under
    ``subprocess.Popen`` so it keeps running after the request returns. Logs go to
    the config log path so the request itself captures no output.
    """
    return ['tinyproxy', '-c', _TINYPROXY_CONFIG_PATH, '-d']


def _tinyproxy_fallback_start_command() -> str:
    """setsid/nohup keep-alive start used when the runtime has no launch endpoint."""
    return (
        f'pkill -x tinyproxy >/dev/null 2>&1 || true; '
        f'setsid tinyproxy -c {_TINYPROXY_CONFIG_PATH} -d >{_TINYPROXY_LOG_PATH} 2>&1 & disown; '
        f'echo tinyproxy-launch-done'
    )


def _proxy_bashrc_command() -> str:
    """Write only localhost proxy env vars to the user's bashrc (no credentials)."""
    return f"""if [ -d /home/user ]; then
  cat >> /home/user/.bashrc <<'EOF'
export http_proxy={OSWORLD_PROXY_SERVER}
export https_proxy={OSWORLD_PROXY_SERVER}
export HTTP_PROXY={OSWORLD_PROXY_SERVER}
export HTTPS_PROXY={OSWORLD_PROXY_SERVER}
EOF
  chown user:user /home/user/.bashrc 2>/dev/null || true
fi
echo tinyproxy-bashrc-done
"""


def _proxy_probe_command() -> str:
    """Poll the local tinyproxy port until a request through it succeeds."""
    curl = (
        f'curl -sS -o /dev/null -w "%{{http_code}}" '
        f'--max-time {_PROXY_PROBE_CURL_TIMEOUT_SECONDS} '
        f'-x {shlex.quote(OSWORLD_PROXY_SERVER)} {shlex.quote(_PROXY_PROBE_URL)}'
    )
    return f"""for attempt in $(seq 1 {_PROXY_PROBE_ATTEMPTS}); do
  code="$({curl} 2>/dev/null || true)"
  if [ "$code" = "200" ] || [ "$code" = "204" ] || [ "$code" = "301" ] || [ "$code" = "302" ]; then
    echo "proxy-probe-ok $code"
    exit 0
  fi
  sleep {_PROXY_PROBE_INTERVAL_SECONDS}
done
echo "proxy-probe-failed last_code=${{code:-none}}"
exit 1
"""


async def _launch_tinyproxy(runtime: Any, user_command_fn: Any) -> dict[str, object]:
    """Start tinyproxy in a keep-alive way, preferring the runtime launch endpoint."""
    if hasattr(runtime, 'setup_launch'):
        try:
            response = await runtime.setup_launch(_tinyproxy_launch_command())
        except NotImplementedError:
            pass
        else:
            status_code = int(_response_value(response, 'status_code', _response_value(response, 'status', 0)) or 0)
            text = str(_response_value(response, 'text', '') or '')
            success = status_code == 200
            return {
                'launch_method': 'setup_launch',
                'success': success,
                'status_code': status_code,
                'stdout': text if success else '',
                'stderr': '' if success else text[:500],
            }

    result = await _run_command(
        runtime,
        _tinyproxy_fallback_start_command(),
        timeout=30,
        user_command_fn=user_command_fn,
    )
    return {'launch_method': 'shell', **result}


async def run_proxy_setup(
    runtime: Any,
    *,
    proxy_config_file: str | Path,
    user_command_fn: Any = osworld_user_command,
    client_password: str = 'password',
    direct_domains: tuple[str, ...] = (),
) -> dict[str, object]:
    proxy = _load_proxy_config(proxy_config_file)
    upload_result = await _upload_proxy_config(runtime, _tinyproxy_config(proxy, direct_domains=direct_domains))
    if not upload_result.get('success'):
        return {'type': 'proxy', 'success': False, 'upload': upload_result}

    install_result = await _run_command(
        runtime,
        _tinyproxy_install_command(client_password),
        timeout=300,
        user_command_fn=user_command_fn,
        display_command=_tinyproxy_install_command('{CLIENT_PASSWORD}'),
    )
    if install_result.get('success') is not True:
        return {
            'type': 'proxy',
            'success': False,
            'stage': 'install',
            'proxy_server': OSWORLD_PROXY_SERVER,
            'config_path': _TINYPROXY_CONFIG_PATH,
            'upload': upload_result,
            **install_result,
        }

    launch_result = await _launch_tinyproxy(runtime, user_command_fn)
    bashrc_result = await _run_command(
        runtime,
        _proxy_bashrc_command(),
        timeout=30,
        user_command_fn=user_command_fn,
    )
    probe_result = await _run_command(
        runtime,
        _proxy_probe_command(),
        timeout=_PROXY_PROBE_COMMAND_TIMEOUT_SECONDS,
        user_command_fn=user_command_fn,
    )
    success = probe_result.get('success') is True
    return {
        'type': 'proxy',
        'success': success,
        'stage': 'probe' if not success else 'ready',
        'proxy_server': OSWORLD_PROXY_SERVER,
        'config_path': _TINYPROXY_CONFIG_PATH,
        'upload': upload_result,
        'install': install_result,
        'launch': launch_result,
        'bashrc': bashrc_result,
        **probe_result,
    }


async def _run_command(
    runtime: Any,
    command: str,
    *,
    timeout: int = 60,
    user_command_fn: Any = osworld_user_command,
    display_command: str | None = None,
) -> dict[str, object]:
    response = await runtime.run_command(command=user_command_fn(command), timeout=timeout)
    return_code = _response_value(response, 'return_code')
    if return_code is None:
        return_code = _response_value(response, 'returncode')
    return {
        'command': display_command or command,
        'success': return_code == 0,
        'return_code': return_code,
        'stdout': _response_value(response, 'stdout', '') or _response_value(response, 'output', ''),
        'stderr': _response_value(response, 'stderr', '') or _response_value(response, 'error', ''),
    }


async def _run_close_window(
    runtime: Any,
    parameters: dict[str, Any],
    *,
    user_command_fn: Any = osworld_user_command,
) -> dict[str, object]:
    window_name = str(parameters.get('window_name') or '')
    if not window_name:
        raise ValueError('OSWorld close_window requires window_name')

    strict = bool(parameters.get('strict', False))
    by_class = bool(parameters.get('by_class', False))
    base_result: dict[str, object] = {'window_name': window_name, 'strict': strict, 'by_class': by_class}

    if hasattr(runtime, 'setup_close_window'):
        try:
            response = await runtime.setup_close_window(window_name, strict=strict, by_class=by_class)
        except NotImplementedError:
            pass
        except Exception as exc:
            return {**base_result, 'success': False, 'error': f'{type(exc).__name__}: {exc}'}
        else:
            status_code = int(_response_value(response, 'status_code', _response_value(response, 'status', 0)) or 0)
            text = str(_response_value(response, 'text', '') or '')
            success = status_code == 200
            return {
                **base_result,
                'success': success,
                'status_code': status_code,
                'stdout': text if success else '',
                'stderr': '' if success else text[:500],
            }

    flags = '-' + ('x' if by_class else '') + ('F' if strict else '') + 'c'
    result = await _run_command(
        runtime,
        f'wmctrl {flags} {shlex.quote(window_name)}',
        timeout=30,
        user_command_fn=user_command_fn,
    )
    return {**base_result, **result}


async def _run_activate_window(
    runtime: Any,
    parameters: dict[str, Any],
    *,
    user_command_fn: Any = osworld_user_command,
) -> dict[str, object]:
    window_name = str(parameters.get('window_name') or '')
    if not window_name:
        raise ValueError('OSWorld activate_window requires window_name')

    strict = bool(parameters.get('strict', False))
    by_class = bool(parameters.get('by_class', False))
    base_result: dict[str, object] = {'window_name': window_name, 'strict': strict, 'by_class': by_class}

    if hasattr(runtime, 'setup_activate_window'):
        try:
            response = await runtime.setup_activate_window(window_name, strict=strict, by_class=by_class)
        except NotImplementedError:
            pass
        except Exception as exc:
            return {**base_result, 'success': False, 'error': f'{type(exc).__name__}: {exc}'}
        else:
            status_code = int(_response_value(response, 'status_code', _response_value(response, 'status', 0)) or 0)
            text = str(_response_value(response, 'text', '') or '')
            success = status_code == 200
            return {
                **base_result,
                'success': success,
                'status_code': status_code,
                'stdout': text if success else '',
                'stderr': '' if success else text[:500],
            }

    flags = '-' + ('x' if by_class else '') + ('F' if strict else '') + 'a'
    result = await _run_command(
        runtime,
        f'wmctrl {flags} {shlex.quote(window_name)}',
        timeout=30,
        user_command_fn=user_command_fn,
    )
    return {**base_result, **result}


def _coerce_until(parameters: dict[str, Any]) -> dict[str, Any]:
    until = parameters.get('until') or {}
    if not isinstance(until, dict):
        raise ValueError('OSWorld setup execute until must be an object')
    return until


def _until_matches(result: dict[str, object], until: dict[str, Any]) -> bool:
    if not until:
        return True
    if 'returncode' in until and result.get('return_code') == until['returncode']:
        return True
    if 'stdout' in until and str(until['stdout']) in str(result.get('stdout', '')):
        return True
    return 'stderr' in until and str(until['stderr']) in str(result.get('stderr', ''))


async def _write_execute_cache_files(
    result: dict[str, object],
    parameters: dict[str, Any],
    cache_dir: str | Path | None,
) -> None:
    base_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
    stdout_name = parameters.get('stdout')
    if stdout_name:
        await _write_text_atomic(base_dir / str(stdout_name), str(result.get('stdout', '')))
    stderr_name = parameters.get('stderr')
    if stderr_name:
        await _write_text_atomic(base_dir / str(stderr_name), str(result.get('stderr', '')))


# ── setup execute: permission-denied fallback to the server's own user ────────────────
#
# GUI setup uses the desktop user's X session. A command that fails with an
# explicit permission error may retry once as the server user, whose permissions
# depend on the image. Other failures and successful commands are not replayed.
_PERMISSION_DENIED_MARKERS = ('permission denied', 'operation not permitted', 'not permitted')


async def _run_execute(
    runtime: Any,
    command: str,
    parameters: dict[str, Any],
    *,
    timeout: int,
    user_command_fn: Any,
    cache_dir: str | Path | None,
    display_command: str | None = None,
) -> dict[str, object]:
    until = _coerce_until(parameters)
    invocation_failures = 0
    attempts = 0
    last_error = ''

    while True:
        attempts += 1
        try:
            result = await _run_command(
                runtime,
                command,
                timeout=timeout,
                user_command_fn=user_command_fn,
                display_command=display_command,
            )
        except Exception as exc:
            invocation_failures += 1
            last_error = f'{type(exc).__name__}: {exc}'
            result = None
        else:
            await _write_execute_cache_files(result, parameters, cache_dir)
            if _until_matches(result, until):
                return {**result, 'attempts': attempts}

        if not until or invocation_failures >= _EXECUTE_UNTIL_FAILURE_LIMIT:
            if result is not None:
                return {**result, 'attempts': attempts}
            return {
                'command': command,
                'success': False,
                'return_code': None,
                'stdout': '',
                'stderr': '',
                'attempts': attempts,
                'error': last_error,
            }

        await asyncio.sleep(_EXECUTE_UNTIL_RETRY_DELAY_SECONDS)


async def _run_download(
    runtime: Any,
    parameters: dict[str, Any],
    *,
    user_command_fn: Any = osworld_user_command,
    cache_dir: str | Path | None = None,
) -> dict[str, object]:
    if hasattr(runtime, 'setup_upload_file') or hasattr(runtime, 'upload_file'):
        try:
            return await _run_host_cached_download(runtime, parameters, cache_dir=cache_dir or _DEFAULT_CACHE_DIR)
        except NotImplementedError:
            pass
    return await _run_vm_download(runtime, parameters, user_command_fn=user_command_fn)


async def _run_host_cached_download(
    runtime: Any,
    parameters: dict[str, Any],
    *,
    cache_dir: str | Path,
) -> dict[str, object]:
    files = parameters.get('files', [])
    if not isinstance(files, list):
        raise ValueError('OSWorld download parameters.files must be a list')

    results: list[dict[str, object]] = []
    for file_spec in files:
        if not isinstance(file_spec, dict):
            raise ValueError('OSWorld download file spec must be an object')
        path = str(file_spec['path'])
        url = str(file_spec['url'])
        try:
            cache_path, cache_hit = await _ensure_cached_download(url, path, cache_dir)
            result = await _upload_cached_file(runtime, cache_path=cache_path, path=path)
            results.append({'cache_hit': cache_hit, **result})
        except NotImplementedError:
            raise
        except Exception as exc:
            results.append({'path': path, 'success': False, 'error': f'{type(exc).__name__}: {exc}'})
    return {'files': results}


async def _run_vm_download(
    runtime: Any,
    parameters: dict[str, Any],
    *,
    user_command_fn: Any = osworld_user_command,
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for file_spec in parameters.get('files', []):
        path = str(file_spec['path'])
        url = str(file_spec['url'])
        parent = shlex.quote(posixpath.dirname(path) or '.')
        command = f'mkdir -p {parent} && curl -L --retry 3 -o {shlex.quote(path)} {_single_quote(url)}'
        results.append(await _run_command(runtime, command, timeout=300, user_command_fn=user_command_fn))
    return {'files': results}


async def _run_update_browse_history(
    runtime: Any,
    parameters: dict[str, Any],
    *,
    user_command_fn: Any = osworld_user_command,
) -> dict[str, object]:
    history = parameters.get('history')
    if not isinstance(history, list):
        raise ValueError('update_browse_history expects history to be a list')

    history_json = json.dumps(history, ensure_ascii=False)
    history_b64 = base64.b64encode(history_json.encode('utf-8')).decode('ascii')
    command = f"""python3 - <<'PY'
import base64
import json
import os
import shutil
import sqlite3
import time
import urllib.request

from datetime import datetime, timedelta
from pathlib import Path

history = json.loads(base64.b64decode({history_b64!r}).decode('utf-8'))
db_url = (
    'https://huggingface.co/datasets/xlangai/ubuntu_osworld_file_cache/resolve/main/'
    'chrome/44ee5668-ecd5-4366-a6ce-c1c9b8d4e938/history_empty.sqlite?download=true'
)
cache_path = Path('/tmp/osworld-history-empty.sqlite')
if not cache_path.exists():
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(db_url, timeout=60) as response:
                cache_path.write_bytes(response.read())
            break
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2)
    else:
        raise RuntimeError(f'failed to download empty Chrome History DB: {{last_error}}')

db_path = Path('/tmp/osworld-history-new.sqlite')
shutil.copyfile(cache_path, db_path)
epoch_start = datetime(1601, 1, 1)
with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    for item in history:
        visit_time = datetime.now() - timedelta(seconds=float(item['visit_time_from_now_in_seconds']))
        chrome_timestamp = int((visit_time - epoch_start).total_seconds() * 1000000)
        cursor.execute(
            'INSERT INTO urls (url, title, visit_count, typed_count, last_visit_time, hidden) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (str(item['url']), str(item['title']), 1, 0, chrome_timestamp, 0),
        )
        url_id = cursor.lastrowid
        cursor.execute(
            'INSERT INTO visits (url, visit_time, from_visit, transition, segment_id, visit_duration) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (url_id, chrome_timestamp, 0, 805306368, 0, 0),
        )
    conn.commit()

history_path = Path(os.environ.get('HOME', '/home/user')) / '.config' / 'google-chrome' / 'Default' / 'History'
history_path.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(db_path, history_path)
os.chmod(history_path, 0o600)
print(json.dumps({{'history_count': len(history), 'path': str(history_path)}}))
PY"""
    return {
        'type': 'update_browse_history',
        **await _run_command(runtime, command, timeout=300, user_command_fn=user_command_fn),
    }


def _download_file_succeeded(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    return cast(dict[str, object], item).get('success') is True


async def run_setup_action(
    runtime: Any,
    action: dict[str, Any],
    *,
    user_command_fn: Any = osworld_user_command,
    download_fn: Any = None,
    execute_timeout: int = 60,
    cache_dir: str | Path | None = None,
    client_password: str = 'password',
    use_proxy: bool = False,
) -> dict[str, object]:
    """Run one OSWorld setup or postconfig action against a runtime.

    The action dispatch is reusable across desktop tasks (OSWorld, CUA-Gym, ...).
    Callers inject the environment-specific bits:
      - ``user_command_fn``: wraps a shell command for the session user + DISPLAY
        (OSWorld uses :99 by default; CUA-Gym injects a :0 / cwd-bound wrapper).
      - ``download_fn``: ``async (runtime, parameters) -> {'files': [...]}``; defaults
        to OSWorld's URL download. CUA-Gym injects bundle-file upload.
      - ``execute_timeout``: timeout for ``execute``/``command`` actions.
    """
    action_type = str(action.get('type', ''))
    parameters = action.get('parameters') or {}
    if not isinstance(parameters, dict):
        raise ValueError(f'OSWorld setup parameters must be an object for {action_type!r}')

    if action_type == 'sleep':
        seconds = float(parameters.get('seconds', 0))
        await asyncio.sleep(seconds)
        return {'type': action_type, 'success': True, 'seconds': seconds}

    if action_type == 'download':
        if download_fn is None:
            result = await _run_download(runtime, parameters, user_command_fn=user_command_fn, cache_dir=cache_dir)
        else:
            result = await download_fn(runtime, parameters)
        files = result['files']
        success = isinstance(files, list) and all(_download_file_succeeded(item) for item in files)
        return {'type': action_type, 'success': success, **result}

    if action_type == 'update_browse_history':
        return await _run_update_browse_history(runtime, parameters, user_command_fn=user_command_fn)

    if action_type == 'close_window':
        return {'type': action_type, **await _run_close_window(runtime, parameters, user_command_fn=user_command_fn)}

    if action_type == 'activate_window':
        return {'type': action_type, **await _run_activate_window(runtime, parameters, user_command_fn=user_command_fn)}

    width, height = await _screen_size(runtime)

    if action_type in {'execute', 'command'}:
        raw_command = parameters.get('command')
        if raw_command is None:
            raise ValueError(f'Missing command for OSWorld setup action: {action_type}')
        prepare_command = getattr(runtime, 'prepare_setup_command', None)
        if callable(prepare_command):
            raw_command = await prepare_command(raw_command)
        command = command_to_shell(
            _replace_command_placeholders(
                raw_command,
                client_password=client_password,
                width=width,
                height=height,
            )
        )
        display_command = command_to_shell(
            _replace_command_placeholders(
                raw_command,
                client_password='{CLIENT_PASSWORD}',
                width=width,
                height=height,
            )
        )
        return {
            'type': action_type,
            **await _run_execute(
                runtime,
                command,
                parameters,
                timeout=execute_timeout,
                user_command_fn=user_command_fn,
                cache_dir=cache_dir,
                display_command=display_command,
            ),
        }

    if action_type == 'launch':
        raw_command = parameters.get('command')
        if raw_command is None:
            raise ValueError('Missing command for OSWorld launch setup action')
        # Official V1 tasks start a socat relay for plain VMs. AGS images already
        # own this exact port mapping; never launch a competing listener.
        relay_parts = [str(part) for part in raw_command] if isinstance(raw_command, list) else shlex.split(raw_command)
        if getattr(runtime, 'cdp_proxy_mode', None) in {'image', 'auto'} and relay_parts == [
            'socat',
            'tcp-listen:9222,fork',
            'tcp:localhost:1337',
        ]:
            checked = await _run_command(
                runtime,
                'systemctl is-active --quiet cdp_proxy.service && '
                'systemctl show cdp_proxy.service --property=ExecStart --value',
                timeout=10,
                user_command_fn=user_command_fn,
            )
            description = str(checked.get('stdout') or '')
            if checked.get('success') is True and all(
                part in description for part in ('osworld-cdp-proxy', '9222', '1337')
            ):
                return {'type': action_type, 'success': True, 'reused_image_cdp_proxy': True}
            raise RuntimeError('Expected image-owned CDP relay 9222→1337 is unavailable; refusing duplicate socat')
        raw_command = add_chrome_proxy_arg(raw_command, use_proxy=use_proxy)
        command = command_to_shell(
            _replace_command_placeholders(
                raw_command,
                client_password='{CLIENT_PASSWORD}',
                width=width,
                height=height,
            )
        )
        result = await _run_command(
            runtime,
            f'nohup {command} >/tmp/osworld-launch.log 2>&1 &',
            user_command_fn=user_command_fn,
        )
        if parameters.get('wait_for_cdp', False):
            if result.get('success') is not True:
                raise RuntimeError('Chrome launch failed before CDP readiness check')

            # Verify an actual WebSocket connection, not merely an HTTP port.
            # Reuse the existing bounded setup retry and AGS/proxy routing.
            async def _connected(browser: Any) -> None:
                _ = browser.contexts

            await _run_chrome_cdp_setup(runtime, _connected)
        return {'type': action_type, **result}

    if action_type == 'open':
        path = str(parameters['path'])
        command = f'xdg-open {shlex.quote(path)} >/tmp/osworld-open.log 2>&1 &'
        return {'type': action_type, **await _run_command(runtime, command, user_command_fn=user_command_fn)}

    if action_type == 'chrome_open_tabs':
        urls = parameters.get('urls_to_open') or parameters.get('urls') or []
        if not isinstance(urls, list):
            raise ValueError('chrome_open_tabs expects urls_to_open to be a list')
        await _open_chrome_tabs(runtime, [str(url) for url in urls])
        return {'type': action_type, 'success': True}

    if action_type == 'chrome_close_tabs':
        urls = parameters.get('urls_to_close') or parameters.get('urls') or []
        if not isinstance(urls, list):
            raise ValueError('chrome_close_tabs expects urls_to_close to be a list')
        await _close_chrome_tabs(runtime, [str(url) for url in urls])
        return {'type': action_type, 'success': True}

    if action_type == 'copyfile':
        src = str(parameters['source'])
        dest = str(parameters['dest'])
        result = await _run_command(runtime, f'cp -f {src!r} {dest!r}', user_command_fn=user_command_fn)
        return {'type': action_type, 'success': result.get('return_code') == 0, **result}

    if action_type == 'upload_file':
        src = str(parameters['source'])
        dest = str(parameters.get('dest', posixpath.basename(src)))
        if not posixpath.isabs(dest):
            dest = f'/tmp/{dest}'
        cache_path = Path(cache_dir or _DEFAULT_CACHE_DIR) / posixpath.basename(src)
        if not cache_path.exists():
            raise FileNotFoundError(f'upload_file source not found in cache: {src} (expected at {cache_path})')
        response = await runtime.setup_upload_file(cache_path, dest)
        success = getattr(response, 'status_code', 200) == 200
        return {'type': action_type, 'success': success, 'dest': dest}

    if action_type == 'python':
        code = str(parameters.get('code', ''))
        result = await _run_command(
            runtime,
            f'python3 -c {shlex.quote(code)}',
            user_command_fn=user_command_fn,
        )
        return {'type': action_type, 'success': result.get('return_code') == 0, **result}

    raise NotImplementedError(f'Unsupported OSWorld setup type: {action_type}')


async def run_setup_actions(
    runtime: Any,
    actions: list[dict[str, Any]] | None,
    *,
    user_command_fn: Any = osworld_user_command,
    cache_dir: str | Path | None = None,
    client_password: str = 'password',
    use_proxy: bool = False,
) -> list[dict[str, object]]:
    """Run OSWorld setup actions sequentially."""
    results = []
    for action in actions or []:
        results.append(
            await run_setup_action(
                runtime,
                action,
                user_command_fn=user_command_fn,
                cache_dir=cache_dir,
                client_password=client_password,
                use_proxy=use_proxy,
            )
        )
    return results
