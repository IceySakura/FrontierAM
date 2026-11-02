"""Desktop information getters."""

from __future__ import annotations

import json
import os
import re
import shlex

from typing import Any

from nexus.extensions.tasks.os_world.evaluation.registry import getter_registry
from nexus.extensions.tasks.os_world.getters.runtime import (
    cache_dir_from_config,
    get_file_bytes,
    response_stdout,
    run_bash,
    write_bytes,
)


@getter_registry.register()
async def get_vm_screen_size(runtime: Any, config: dict[str, Any]) -> dict[str, int | None]:
    if hasattr(runtime, 'screen_size'):
        data = await runtime.screen_size()
        if isinstance(data, dict):
            return {'width': data.get('width'), 'height': data.get('height')}
    if hasattr(runtime, 'gui_get_screen_size'):
        response = await runtime.gui_get_screen_size()
        return {'width': getattr(response, 'width', None), 'height': getattr(response, 'height', None)}
    return {'width': None, 'height': None}


@getter_registry.register()
async def get_vm_window_size(runtime: Any, config: dict[str, Any]) -> dict[str, int | None]:
    app_class_name = str(config['app_class_name'])
    script = f"""
set -e
window_id="$(xdotool search --class {shlex.quote(app_class_name)} | head -n 1 || true)"
if [ -z "$window_id" ]; then
  exit 1
fi
xwininfo -id "$window_id" | awk '/Width:|Height:/ {{print $1, $2}}'
"""
    result = await run_bash(runtime, script, timeout=30)
    output = response_stdout(result)
    width = height = None
    for line in output.splitlines():
        key, _, value = line.partition(':')
        if key.strip() == 'Width':
            width = int(value.strip())
        elif key.strip() == 'Height':
            height = int(value.strip())
    return {'width': width, 'height': height}


@getter_registry.register()
async def get_vm_wallpaper(runtime: Any, config: dict[str, Any]) -> str:
    cache_path = cache_dir_from_config(config) / str(config['dest'])
    result = await run_bash(
        runtime,
        r"""
uri="$(gsettings get org.gnome.desktop.background picture-uri 2>/dev/null | sed "s/^'//; s/'$//")"
uri="${uri#file://}"
printf '%s' "$uri"
""",
        timeout=30,
    )
    remote_path = response_stdout(result).strip()
    content = await get_file_bytes(runtime, remote_path) if remote_path else None
    await write_bytes(cache_path, content or b'')
    return str(cache_path)


@getter_registry.register()
async def get_list_directory(runtime: Any, config: dict[str, Any]) -> dict[str, Any]:
    path = str(config['path'])
    script = (
        "python3 - <<'PY'\n"
        'import json, os, sys\n'
        f'root={path!r}\n'
        'def node(p):\n'
        '    item={"name": os.path.basename(p) or p}\n'
        '    if os.path.isdir(p):\n'
        '        item["children"]=[node(os.path.join(p, name)) for name in sorted(os.listdir(p))]\n'
        '    return item\n'
        'print(json.dumps(node(root)))\n'
        'PY'
    )
    result = await run_bash(runtime, script, timeout=30)
    try:
        return json.loads(response_stdout(result))
    except json.JSONDecodeError:
        return {'name': os.path.basename(path), 'children': []}


def _parse_file_uri(raw: str) -> str:
    text = raw.strip().strip("'")
    return re.sub(r'^file://', '', text)
