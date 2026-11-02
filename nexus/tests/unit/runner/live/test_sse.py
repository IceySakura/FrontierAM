import json

from nexus.runner.live.sse import sse_bytes
from nexus.runner.trace import RunTraceStreamEvent


def test_sse_bytes_encodes_projection_event_and_prefixes_runtime_urls() -> None:
    event = RunTraceStreamEvent(
        id=7,
        type='runtime.ready',
        time_unix_nano=1_783_468_800_000_000_000,
        data={
            'runtimes': {
                'main': {
                    'proxy_url': '/v1/runs/run-1/runtimes/main/proxy',
                    'debug_gui_url': '/v1/runs/run-1/runtimes/main/proxy/debug/gui',
                }
            }
        },
    )

    payload = sse_bytes(event, base_path='/nexus/live').decode()

    assert payload.startswith('id: 7\nevent: runtime.ready\ndata: ')
    data = json.loads(payload.split('data: ', 1)[1])
    assert data['data']['runtimes']['main'] == {
        'proxy_url': '/nexus/live/v1/runs/run-1/runtimes/main/proxy',
        'debug_gui_url': '/nexus/live/v1/runs/run-1/runtimes/main/proxy/debug/gui',
    }
