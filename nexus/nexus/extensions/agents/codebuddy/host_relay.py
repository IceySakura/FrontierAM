"""Public CodeBuddy session extension that routes approvals and tools to Host services.

Select it with ``extension_factory='nexus.extensions.agents.codebuddy.host_relay::host_relay_extension'``.
The session id is the Host routing channel, the same key as the session's LLM channel.

``extension_config`` keys:

- ``relay_permissions``: forward deferred ACP permissions to the Host; pair with
  ``permission_policy='defer'``.
- ``relay_updates``: push every ACP ``session/update`` of a prompt sent with ``relay_turn_id``
  to the Host as it arrives, so the Host can report progress while the prompt is still running
  (see ``AcpUpdateRelay``).
- ``host_tools``: ``{server_name, tools, request_timeout?, tool_timeouts?}``; serves the
  Host-declared MCP ``tools`` on a loopback HTTP MCP server and forwards every call.
- ``client_questions``: let CodeBuddy ask the user. ``initialize`` declares form elicitation and
  CodeBuddy's ``AskUserQuestion`` (``CLIENT_QUESTION_CAPABILITIES``); ``elicitation/create`` is
  forwarded to the Host (see ``AcpElicitationRelay``) and ``AskUserQuestion`` arrives as a relayed
  permission the Host answers with ``ext_requests``. Requires ``relay_permissions``. Without it,
  ``ext_requests`` in permission replies are ignored.

The extension never assigns handlers to the connection; the session runtime does. A WorkBuddy
``session/fork`` shares its source's connection, so it enters this extension again under its own
session id, gets its own Host channels, and has its handlers bound to the fork's ACP session.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from nexus.extensions.agents.codebuddy.session_runtime import CodeBuddySessionExtension
from nexus.extensions.agents.installed_common.host_rpc import (
    AcpElicitationRelay,
    AcpPermissionRelay,
    AcpUpdateRelay,
    HostToolClient,
)
from nexus.extensions.agents.installed_common.host_tool_mcp import HostToolMCPServer


_CONFIG_KEYS = frozenset({'relay_permissions', 'relay_updates', 'host_tools', 'client_questions'})
_HOST_TOOLS_KEYS = frozenset({'server_name', 'tools', 'request_timeout', 'tool_timeouts'})
CLIENT_QUESTION_CAPABILITIES: dict[str, Any] = {
    'elicitation': {'form': {}},
    '_meta': {'codebuddy.ai': {'question': True}},
}
"""ACP ``clientCapabilities`` under which CodeBuddy exposes ``AskUserQuestion`` and form elicitation."""


def _parse_config(config: dict[str, Any]) -> tuple[bool, bool, dict[str, Any] | None, bool]:
    unknown = set(config) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f'Unknown host_relay_extension config keys: {sorted(unknown)}')
    relay_permissions = config.get('relay_permissions', False)
    if not isinstance(relay_permissions, bool):
        raise ValueError(f'relay_permissions must be a bool, got {relay_permissions!r}')
    relay_updates = config.get('relay_updates', False)
    if not isinstance(relay_updates, bool):
        raise ValueError(f'relay_updates must be a bool, got {relay_updates!r}')
    client_questions = config.get('client_questions', False)
    if not isinstance(client_questions, bool):
        raise ValueError(f'client_questions must be a bool, got {client_questions!r}')
    if client_questions and not relay_permissions:
        raise ValueError('client_questions needs relay_permissions: AskUserQuestion arrives as a permission request')
    host_tools = config.get('host_tools')
    if host_tools is not None:
        if not isinstance(host_tools, dict):
            raise ValueError(f'host_tools must be a mapping, got {type(host_tools).__name__}')
        unknown = set(host_tools) - _HOST_TOOLS_KEYS
        if unknown:
            raise ValueError(f'Unknown host_tools config keys: {sorted(unknown)}')
        missing = {'server_name', 'tools'} - set(host_tools)
        if missing:
            raise ValueError(f'host_tools config is missing required keys: {sorted(missing)}')
    if not relay_permissions and not relay_updates and host_tools is None:
        raise ValueError(
            'host_relay_extension needs relay_permissions, relay_updates or host_tools; omit the extension otherwise'
        )
    return relay_permissions, relay_updates, host_tools, client_questions


@asynccontextmanager
async def host_relay_extension(
    *, session_id: str, sio: Any, connection: Any, config: dict[str, Any]
) -> AsyncIterator[CodeBuddySessionExtension]:
    """Bind this session's Host channels for approvals, questions, updates and tools until the session closes."""
    relay_permissions, relay_updates, host_tools, client_questions = _parse_config(config)
    async with AsyncExitStack() as stack:
        permission_handler = None
        on_turn_end = None
        if relay_permissions:
            relay = AcpPermissionRelay(
                sio=sio, channel_id=session_id, connection=connection, client_questions=client_questions
            )
            stack.callback(relay.close)
            permission_handler = relay.request
            on_turn_end = relay.end_turn
        elicitation_handler = None
        client_capabilities = None
        if client_questions:
            elicitation_relay = AcpElicitationRelay(sio=sio, channel_id=session_id)
            stack.callback(elicitation_relay.close)
            elicitation_handler = elicitation_relay.request
            client_capabilities = CLIENT_QUESTION_CAPABILITIES
        update_relay = None
        if relay_updates:
            update_relay = AcpUpdateRelay(sio=sio, channel_id=session_id)
            # Exit runs in reverse: stop listening first, then flush what is already queued.
            stack.push_async_callback(update_relay.aclose)
            stack.callback(connection.add_update_listener(update_relay.publish))
        mcp_servers: tuple[dict[str, Any], ...] = ()
        if host_tools is not None:
            client = HostToolClient(sio=sio, channel_id=session_id)
            stack.callback(client.close)
            server = HostToolMCPServer(call_host=client.call, **host_tools)
            spec = await server.start()
            stack.push_async_callback(server.close)
            mcp_servers = (spec,)
        yield CodeBuddySessionExtension(
            permission_handler=permission_handler,
            mcp_servers=mcp_servers,
            update_relay=update_relay,
            client_capabilities=client_capabilities,
            elicitation_handler=elicitation_handler,
            on_turn_end=on_turn_end,
        )


__all__ = ['host_relay_extension']
