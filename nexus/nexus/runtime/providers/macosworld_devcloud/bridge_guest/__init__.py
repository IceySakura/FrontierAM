"""Guest-only entry, also distributed as the standalone macosworld_bridge package."""

from nexus.runtime.bridge.app import create_bridge_asgi_app as _asgi
from nexus.runtime.bridge.app import create_bridge_fastapi_app as _fastapi

from .protocol import responses_to_chat


def _options(options):
    options = dict(options)
    if options.get('api_protocol') == 'responses-chat':
        options['api_protocol'] = 'responses'
        options['responses_converter'] = responses_to_chat
    return options


def create_bridge_fastapi_app(**options):
    return _fastapi(**_options(options))


def create_bridge_asgi_app(**options):
    return _asgi(**_options(options))
