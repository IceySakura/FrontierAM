import json
import re

import pytest

from aiohttp import web
from aiohttp.test_utils import make_mocked_request
from pydantic import ValidationError

from nexus.distributed.config import DistributedConfig
from nexus.distributed.gateways.common.keys import CONFIG_KEY
from nexus.distributed.gateways.control.routes.operations import console_ui


@pytest.mark.parametrize('enabled', [None, False, True])
async def test_console_welcome_animation_requires_server_opt_in(enabled):
    app = web.Application()
    app[CONFIG_KEY] = DistributedConfig(
        **({} if enabled is None else {'cloud_console_welcome_animation_enabled': enabled})
    )
    request = make_mocked_request('GET', '/v1/console?welcomeAnimationEnabled=true', app=app)

    response = await console_ui(request)

    assert response.text is not None
    match = re.search(r'window\.__NEXUS_DISTRIBUTED_CONSOLE__ = (.*);', response.text)
    assert match is not None
    assert json.loads(match.group(1))['welcomeAnimationEnabled'] is (enabled is True)
    assert response.headers['Cache-Control'] == 'no-store'


@pytest.mark.parametrize(
    'url', [None, 'https://docs.example.com/nexus/start?lang=zh#runs', 'http://docs.internal/guide/']
)
async def test_console_guide_url_uses_server_config(url):
    app = web.Application()
    app[CONFIG_KEY] = DistributedConfig(**({} if url is None else {'cloud_console_guide_url': url}))
    request = make_mocked_request('GET', '/v1/console?guideUrl=https://ignored.example/', app=app)

    response = await console_ui(request)

    assert response.text is not None
    match = re.search(r'window\.__NEXUS_DISTRIBUTED_CONSOLE__ = (.*);', response.text)
    assert match is not None
    assert json.loads(match.group(1))['guideUrl'] == (url or 'https://nexus.pages.woa.com/cloud/overview/')


@pytest.mark.parametrize(
    'url', ['', '/guide', '//example.com', 'javascript:alert(1)', 'data:text/html,test', 'https://']
)
def test_console_guide_url_rejects_non_http_links(url):
    with pytest.raises(ValidationError, match='cloud_console_guide_url'):
        DistributedConfig(cloud_console_guide_url=url)
