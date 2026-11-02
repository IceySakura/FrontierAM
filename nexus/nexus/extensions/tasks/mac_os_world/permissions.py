"""Dismiss only the infrastructure screen-capture consent dialog before task setup."""

from __future__ import annotations

import asyncio
import json
import shlex

from typing import Any


# Match both the system process and the requester; never dismiss application/task dialogs.
_CAPTURE_DIALOG_SCRIPT = r"""
var se = Application('System Events');
var ps = se.processes.whose({name: 'UserNotificationCenter'})();
var result = {matched: 0, clicked: 0, buttons: []};
for (var i = 0; i < ps.length; i++) {
    var windows = ps[i].windows();
    for (var j = 0; j < windows.length; j++) {
        var elements = windows[j].entireContents();
        var text = '';
        var buttons = [];
        for (var k = 0; k < elements.length; k++) {
            var item = elements[k];
            if (item.role() === 'AXStaticText') text += ' ' + item.value();
            if (item.role() === 'AXButton') buttons.push(item);
        }
        if (text.indexOf('GUICloneVMService') < 0 ||
            !/screen|屏幕|画面|экран|الشاشة/i.test(text)) continue;
        result.matched++;
        var allow = buttons.filter(function(b) {
            return ['Allow', '允许', '許可', 'Разрешить', 'السماح'].indexOf(b.name()) >= 0;
        });
        if (__CLICK__ && allow.length === 1) {
            var label = allow[0].name();
            allow[0].click();
            result.clicked++;
            result.buttons.push(label);
        }
    }
}
JSON.stringify(result);
"""


async def _capture_dialog(runtime: Any, *, click: bool) -> dict[str, Any]:
    script = _CAPTURE_DIALOG_SCRIPT.replace('__CLICK__', 'true' if click else 'false')
    response = await runtime.run_command(command='osascript -l JavaScript -e ' + shlex.quote(script), timeout=30)
    if response.return_code != 0:
        raise RuntimeError(f'Screen-capture permission probe failed: {response.stderr or response.stdout}')
    return json.loads(response.stdout or '')


async def prepare_screen_capture_permission(runtime: Any) -> dict[str, Any]:
    """Trigger the capture request, click the exact consent, then verify it disappeared."""
    screenshot = await runtime.gui_screenshot()
    if not screenshot.success or not screenshot.image:
        raise RuntimeError(f'Cannot initialize screen capture: {screenshot.error}')
    await asyncio.sleep(1)
    attempts = []
    for _ in range(3):
        observation = await _capture_dialog(runtime, click=True)
        attempts.append(observation)
        await asyncio.sleep(1)
        screenshot = await runtime.gui_screenshot()
        if not screenshot.success or not screenshot.image:
            raise RuntimeError(f'Cannot verify screen capture: {screenshot.error}')
        remaining = await _capture_dialog(runtime, click=False)
        if remaining['matched'] == 0:
            return {'success': True, 'clicked': sum(x['clicked'] for x in attempts), 'attempts': attempts}
    raise RuntimeError('GUICloneVMService screen-capture consent remains visible after initialization')
