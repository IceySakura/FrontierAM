"""Capture a PNG inside the supplied runtime and download it as base64.

WebGL uses Xvfb, X11, and headed Chromium. The executable path is pinned to
the evaluation image; update it when the image browser version changes.
"""

from __future__ import annotations

import base64
import logging
import textwrap
import uuid

from typing import Any


logger = logging.getLogger(__name__)

CHROMIUM_PATH_IN_CONTAINER = '/opt/ms-playwright/chromium-1208/chrome-linux64/chrome'

# Chromium launch flags for the in-container xvfb X server (--ozone-platform=x11).
# The playwright-mcp (agent-browse) path has its own equivalent set in
# agent_setup/claude_config.py::_CHROMIUM_LAUNCH_ARGS (headless variant) — keep the
# SwiftShader-WebGL + sandbox-bypass groups consistent if you tune rendering.
_LAUNCH_ARGS: list[str] = [
    '--no-sandbox',
    '--disable-setuid-sandbox',
    '--disable-gpu-sandbox',
    '--disable-dev-shm-usage',
    '--password-store=basic',
    '--use-mock-keychain',
    '--ozone-platform=x11',
    '--enable-unsafe-swiftshader',
    '--use-gl=angle',
    '--use-angle=swiftshader-webgl',
    '--mute-audio',
    '--hide-scrollbars',
    '--disable-extensions',
    '--disable-background-networking',
    '--disable-component-update',
    '--disable-default-apps',
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-sync',
    '--metrics-recording-only',
    '--disable-features=Translate,TranslateUI,OptimizationHints,MediaRouter',
]

_VIEWPORT = {'width': 1920, 'height': 1080}
_NAV_TIMEOUT_MS = 30000
_INITIAL_WAIT_MS = 5000
_POST_SCROLL_WAIT_MS = 2000


def _build_playwright_script(url: str, out_path: str) -> str:
    """Build the inline python script that runs inside the container.

    Flow: clear cache → goto → wait states → initial wait → scroll to bottom &
    back → screenshot → write to ``out_path`` as raw PNG bytes. The caller pulls
    it back via ``download_file`` (b64-encoded on the wire). We don't stream base64
    through stdout because ``xvfb-run`` / ``bash -c`` can drop the trailing byte on
    long stdout writes, corrupting the b64.
    """
    scroll_js = """
        async () => {
            const delay = ms => new Promise(r => setTimeout(r, ms));
            const height = document.body.scrollHeight;
            const step = window.innerHeight;
            for (let y = 0; y <= height; y += step) {
                window.scrollTo(0, y);
                await delay(200);
            }
            await delay(1000);
            window.scrollTo(0, 0);
            await delay(500);
        }
    """
    return textwrap.dedent(f"""
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path={CHROMIUM_PATH_IN_CONTAINER!r},
                headless=False,
                args={_LAUNCH_ARGS!r},
            )
            ctx = browser.new_context(viewport={_VIEWPORT!r}, ignore_https_errors=True)
            page = ctx.new_page()
            cdp = ctx.new_cdp_session(page)
            try:
                try:
                    cdp.send('Network.clearBrowserCache')
                except Exception:
                    pass
                try:
                    page.goto({url!r}, wait_until='domcontentloaded', timeout={_NAV_TIMEOUT_MS})
                except PWTimeout:
                    pass
                for state, tmo in (('load', 10000), ('networkidle', 10000)):
                    try:
                        page.wait_for_load_state(state, timeout=tmo)
                    except PWTimeout:
                        break
                page.wait_for_timeout({_INITIAL_WAIT_MS})
                page.evaluate({scroll_js!r})
                page.wait_for_timeout({_POST_SCROLL_WAIT_MS})
                shot = page.screenshot(full_page=True, timeout=60000)
                with open({out_path!r}, 'wb') as f:
                    f.write(shot)
                print('OK', len(shot))
            finally:
                ctx.close()
                browser.close()
    """).strip()


async def capture_screenshot_in_container(
    runtime: Any,
    url: str,
    *,
    timeout: float = 180.0,
) -> str | None:
    """Take a full-page playwright screenshot of ``url`` inside ``runtime``.

    Returns base64 PNG on success, ``None`` on failure.

    Runtime must be an already-connected gongfeng ``RemoteRuntime`` (see
    ``_provision_runtime`` in ``judges/capabilities/agent.py``).
    """
    from nexus.runtime.executors.command import run_command
    from nexus.runtime.executors.file import download_file
    from nexus.runtime.executors.models.command import RunCommandRequest
    from nexus.runtime.executors.models.file import DownloadFileRequest, FileItem

    uid = uuid.uuid4().hex[:8]
    remote_png = f'/tmp/screenshot_{uid}.png'
    script = _build_playwright_script(url, remote_png)
    b64_script = base64.b64encode(script.encode()).decode()
    # xvfb-run supplies a virtual X display; -a picks a free display number
    # (avoids collision when multiple screenshots run concurrently in the
    # same container in the future).
    cmd = f'xvfb-run -a --server-args="-screen 0 1920x1080x24" bash -c "echo {b64_script} | base64 -d | python3"'
    result = await runtime.invoke(run_command, RunCommandRequest(command=cmd, timeout=timeout))
    if result.return_code != 0:
        logger.warning(
            'screenshot in container failed url=%s rc=%s stderr=%s',
            url,
            result.return_code,
            (result.stderr or '')[:400],
        )
        return None
    if not (result.stdout or '').strip().startswith('OK'):
        logger.warning(
            'screenshot in container: no OK marker for %s (stdout=%s stderr=%s)',
            url,
            (result.stdout or '')[:200],
            (result.stderr or '')[:200],
        )
        return None

    dl_result = await runtime.invoke(
        download_file,
        DownloadFileRequest(files=[FileItem(path=remote_png, encoding='base64')]),
    )
    if dl_result.errors:
        logger.warning('download screenshot failed for %s: %s', url, dl_result.errors)
        return None
    if not dl_result.files:
        logger.warning('download screenshot returned no files for %s', url)
        return None
    return dl_result.files[0].content


__all__ = [
    'CHROMIUM_PATH_IN_CONTAINER',
    'capture_screenshot_in_container',
]
