"""VLC getters."""

from __future__ import annotations

import collections
import json
import shlex

from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree

from nexus.extensions.tasks.os_world.evaluation.errors import EvaluatorInfrastructureError
from nexus.extensions.tasks.os_world.evaluation.registry import getter_registry
from nexus.extensions.tasks.os_world.execution.commands import osworld_user_command
from nexus.extensions.tasks.os_world.getters.file import get_vm_file
from nexus.extensions.tasks.os_world.getters.runtime import (
    cache_dir_from_config,
    response_returncode,
    response_stdout,
    run_bash,
    runtime_headers,
    runtime_port_url,
    write_bytes,
)
from nexus.utils.async_http import http_client


@getter_registry.register()
async def get_vlc_config(runtime: Any, config: dict[str, Any]) -> str | None:
    dest = str(config.get('dest', 'vlcrc'))
    result = await get_vm_file(runtime, {**config, 'path': '/home/user/.config/vlc/vlcrc', 'dest': dest})
    return str(result) if result is not None else None


@getter_registry.register()
async def get_vlc_playing_info(runtime: Any, config: dict[str, Any]) -> str | None:
    dest = str(config.get('dest', 'status.xml'))
    cache_path = cache_dir_from_config(config) / dest
    # MPRIS exposes the actual GUI instances, unlike the image's fixed HTTP
    # endpoint which can remain attached to a dummy pause-only VLC process.
    probe = Path(__file__).with_name('vlc_mpris_probe.py').read_text()
    command = _vlc_probe_command(probe)
    result = await run_bash(runtime, command, timeout=30)
    observed: dict[str, Any]
    try:
        decoded = json.loads(response_stdout(result))
        if not isinstance(decoded, dict):
            raise ValueError('invalid probe response')
        observed = decoded
    except ValueError:
        observed = {'probe_error': 'invalid_response'}
    if response_returncode(result) not in (0, None):
        observed = {'probe_error': 'command_failed'}
    players = observed.get('players', [])
    if not isinstance(players, list) or any(not isinstance(p, dict) for p in players):
        raise EvaluatorInfrastructureError('VLC returned invalid player state')
    active = [p for p in players if p.get('state') == 'playing' and p.get('url') and not _placeholder(p['url'])]
    # Selection is independent of the expected filename/URL. Do not choose
    # whichever player happens to match the answer key.
    distinct = {(p['state'], p['url']) for p in active}
    if len(distinct) > 1:
        raise EvaluatorInfrastructureError(
            'Multiple VLC instances are playing different media; observation is ambiguous'
        )
    if active:
        content = _player_xml(active[0])
        await write_bytes(cache_path, content)
        return str(cache_path)

    base_url = runtime_port_url(runtime, int(getattr(runtime, 'vlc_port', 8080)))
    if base_url:
        response = await http_client.get(
            f'{base_url}/requests/status.xml',
            auth=None,
            headers=runtime_headers(runtime),
        )
        if response.ok and _real_http_media(response.content):
            await write_bytes(cache_path, response.content)
            return str(cache_path)

    # Fallback for runtimes where VLC is only reachable from inside the VM.
    result = await run_bash(
        runtime,
        'curl --noproxy "*" -fsS --max-time 10 --user :password http://127.0.0.1:8080/requests/status.xml',
        timeout=15,
    )
    output = response_stdout(result)
    if output and _real_http_media(output.encode('utf-8')):
        await write_bytes(cache_path, output.encode('utf-8'))
        return str(cache_path)
    if observed.get('probe_error'):
        raise EvaluatorInfrastructureError('VLC session observation failed and no real HTTP media state was available')
    # Successful discovery with no playing VLC is a real negative state, not
    # a transport failure. ffplay and other players intentionally do not count.
    return None


def _vlc_probe_command(probe: str) -> str:
    command = '/usr/bin/python3 -c ' + shlex.quote(probe)
    # AGS already executes as the desktop user. runuser is root-only; use it
    # solely for runtimes that execute commands as root.
    return f'if [ "$(id -u)" = 0 ]; then {osworld_user_command(command)}; else {command}; fi'


def _placeholder(value: str) -> bool:
    return value.startswith(('pause:', 'vlc://pause:', 'vlc://quit'))


def _real_http_media(content: bytes) -> bool:
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError:
        return False
    values = [node.text or '' for node in root.findall('information/category[@name="meta"]/info')]
    return bool(root.find('state') is not None and values and not any(_placeholder(v) for v in values))


def _player_xml(player: dict) -> bytes:
    root = ElementTree.Element('root')
    ElementTree.SubElement(root, 'state').text = player['state']
    info = ElementTree.SubElement(ElementTree.SubElement(root, 'information'), 'category', name='meta')
    url = player['url']
    for key, value in {
        'filename': unquote(urlparse(url).path.rsplit('/', 1)[-1]),
        'url': url,
        'title': player.get('title', ''),
    }.items():
        ElementTree.SubElement(info, 'info', name=key).text = value
    return ElementTree.tostring(root, encoding='utf-8')


@getter_registry.register()
async def get_default_video_player(runtime: Any, config: dict[str, Any]) -> str:
    extensions = [
        '3gp',
        '3gpp',
        'avi',
        'divx',
        'dv',
        'fli',
        'flv',
        'mp2t',
        'mp4',
        'mp4v-es',
        'mpeg',
        'mpeg-system',
        'msvideo',
        'ogg',
        'quicktime',
        'vnd.divx',
        'vnd.mpegurl',
        'vnd.rn-realvideo',
        'webm',
        'x-anim',
        'x-avi',
        'x-flc',
        'x-fli',
        'x-flv',
        'x-m4v',
        'x-matroska',
        'x-mpeg',
        'x-mpeg-system',
        'x-mpeg2',
        'x-ms-asf',
        'x-ms-asf-plugin',
        'x-ms-asx',
        'x-ms-wm',
        'x-ms-wmv',
        'x-ms-wmx',
        'x-ms-wvx',
        'x-msvideo',
        'x-nsv',
        'x-ogm',
        'x-ogm+ogg',
        'x-theora',
        'x-theora+ogg',
    ]
    script = '\n'.join(f'xdg-mime query default video/{ext} 2>/dev/null || true' for ext in extensions)
    result = await run_bash(runtime, script, timeout=30)
    apps = [line.strip() for line in response_stdout(result).splitlines() if line.strip()]
    return collections.Counter(apps).most_common(1)[0][0] if apps else 'unknown'
