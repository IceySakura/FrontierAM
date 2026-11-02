import base64

import pytest

from aiohttp.test_utils import TestClient, TestServer

from nexus.dashboard.server import create_app
from nexus.gateway.observers.file import FileTrajectoryObserver
from nexus.gateway.trajectory import Trajectory, TrajectoryNode


@pytest.mark.asyncio
@pytest.mark.parametrize('compress', [True, False])
async def test_images_preserved_inline_through_export_and_detail(tmp_path, compress):
    # The backend transports image bytes without decoding them.
    encoded = base64.b64encode(b'opaque image payload').decode('ascii')
    uri = f'data:image/png;base64,{encoded}'
    message = {
        'role': 'user',
        'content': [
            {'type': 'image_url', 'image_url': {'url': uri}},
            {'type': 'input_image', 'image_url': uri},
            {'type': 'tool_result', 'tool_use_id': 'screenshot', 'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': encoded}},
            ]},
        ],
    }
    trajectory = Trajectory(session_id='inline-images', roots=[0], nodes={
        0: TrajectoryNode(
            node_id=0, message_id=0, parent=None, children=[], generated_by='client', original_message=message,
        ),
    })
    before = trajectory.dump()
    FileTrajectoryObserver(str(tmp_path), compress=compress).on_final(trajectory)
    assert trajectory.dump() == before
    assert len(list(tmp_path.iterdir())) == 2
    async with TestClient(TestServer(create_app(tmp_path))) as client:
        catalog = await (await client.get('/api/trajectories')).json()
        assert catalog['total'] == 1
        response = await client.get('/api/trajectories/' + catalog['items'][0]['id'])
        assert response.status == 200
        assert (await response.json())['trajectory'] == before
