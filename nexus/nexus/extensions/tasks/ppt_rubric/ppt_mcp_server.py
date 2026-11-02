"""The 12 slide tools served to an installed codex harness over stdio MCP.

A thin protocol shell around ``ppt_tools``: every ``tools/call`` dispatches to the
same registered functions the built-in loop runs, so both arms of a ``ppt_topic``
row see identical tool descriptions, input schemas, budget keys, render constants
and deck-persist behaviour -- only the transport into the model differs (loop
tool registry vs MCP stdio). No tool behaviour is re-implemented here.

Process shape: codex spawns this module inside the sandbox as
``${NEXUS_WORKER_PYTHON} -m nexus.extensions.tasks.ppt_rubric.ppt_mcp_server``
and talks MCP over stdin/stdout using the minimal JSON-RPC loop validated by the
ppt12mcp prototype (initialize echoes the client's protocolVersion or the client
silently drops every tool; one JSON object per line; images travel as
``{"type": "image", "data": <b64>, "mimeType": ...}`` content parts).

There is no host-side ToolContext in this process. The per-row state the tools
expect -- reference slide, code workdir, image dump dir, deck filename -- is
seeded from ``PPT12_*`` env vars into a plain session dict, re-wrapped into a
fresh immutable ``ToolContext`` on every call, and each call's
``context_updates`` are absorbed back into the session.

Env contract (values built by ``codex_ppt12_mcp.build_ppt12_mcp_server``):

  PPT12_REFERENCE_PNG   source slide path   -> tool_state['image']
  PPT12_WORKSPACE       the row's working_dir -> tool_state['pptx_code_workdir']
  PPT12_DECK_FILENAME   deck name resolved inside the workspace -> deck_filename
  PPT12_DUMP_DIR        tool image dumps    -> tool_state image dump dir
  PPT12_MCP_LOG         append-only protocol log (positive-control channel)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback

from typing import Any

from nexus.core.tool.types import ToolContext

from .ppt_tools import (
    CODE_WORKDIR_STATE_KEY,
    IMAGE_STATE_KEY,
    PPT12_MODEL_NAMES,
    PPT12_TOOLS,
    load_ppt12,
)


SERVER_NAME = 'ppt12'
SERVER_VERSION = '1.0.0'
DEFAULT_PROTOCOL_VERSION = '2025-11-25'

REFERENCE_PNG_ENV = 'PPT12_REFERENCE_PNG'
WORKSPACE_ENV = 'PPT12_WORKSPACE'
DECK_FILENAME_ENV = 'PPT12_DECK_FILENAME'
DUMP_DIR_ENV = 'PPT12_DUMP_DIR'
MCP_LOG_ENV = 'PPT12_MCP_LOG'

MAX_TEXT_LEN = 100_000
_TRUNCATION_NOTE = '\n[... truncated ...]'

# ``data:<mime>;base64,<payload>`` image parts -> MCP image content items.
_DATA_URL_RE = re.compile(r'^data:(?P<mime>[\w.+/-]+);base64,(?P<data>.+)$', re.DOTALL)
_UNDECODABLE_IMAGE_NOTE = (
    '[tool produced an image that could not be decoded for transport; its measured numbers are in the text parts]'
)


class ToolCallError(Exception):
    """A tools/call that must be reported to the model as an error result."""


def _data_url_to_image_part(part: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an OpenAI ``image_url`` part to an MCP ``image`` content item.

    Returns None for anything that is not a decodable data URL (http links,
    malformed base64); the caller substitutes a text note instead of silently
    dropping the slot.
    """
    url = part.get('image_url')
    url = url.get('url') if isinstance(url, dict) else None
    match = _DATA_URL_RE.match(url or '')
    if match is None:
        return None
    try:
        base64.b64decode(match.group('data'), validate=True)
    except (ValueError, TypeError):
        return None
    return {'type': 'image', 'data': match.group('data'), 'mimeType': match.group('mime')}


def mcp_content(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce nexus tool output parts to MCP content items.

    Follows the prototype's rules: text parts pass through with a hard length
    cap, unknown shapes with a ``text`` member degrade to text, and an empty
    output still produces one visible item instead of nothing.
    """
    content: list[dict[str, Any]] = []
    for part in parts:
        ptype = part.get('type')
        if ptype == 'text':
            text = str(part.get('text') or '')
            if len(text) > MAX_TEXT_LEN:
                text = text[:MAX_TEXT_LEN] + _TRUNCATION_NOTE
            content.append({'type': 'text', 'text': text})
        elif ptype == 'image_url':
            image = _data_url_to_image_part(part)
            content.append(image or {'type': 'text', 'text': _UNDECODABLE_IMAGE_NOTE})
        elif ptype == 'image':
            content.append(dict(part))
        elif 'text' in part:
            content.append({'type': 'text', 'text': str(part['text'])[:MAX_TEXT_LEN]})
    return content or [{'type': 'text', 'text': '(no output)'}]


def tool_schemas() -> list[dict[str, Any]]:
    """Return the MCP tools/list entries for the 12 slide tools.

    name and description come verbatim from ``__tool_meta__`` and inputSchema is
    the registry's OpenAI function ``parameters`` schema -- byte-for-byte the
    contract the model saw in training, with ``$defs`` already resolved. The one
    deliberate deviation: render_slide's ``code`` is not required over MCP, so a
    prompt can teach the prototype's "build the deck yourself, then call
    render_slide without code" workflow; a call with no code renders the deck
    already sitting at the workspace path.
    """
    registry = load_ppt12(PPT12_TOOLS)
    entries: list[dict[str, Any]] = []
    for model_name in PPT12_MODEL_NAMES:
        meta = registry['visual/ppt_' + model_name].__tool_meta__
        schema = dict(meta.schema['function']['parameters'])
        if meta.name == 'render_slide' and schema.get('required'):
            schema['required'] = [name for name in schema['required'] if name != 'code']
            if not schema['required']:
                schema.pop('required')
        entries.append({'name': meta.name, 'description': meta.description, 'inputSchema': schema})
    return entries


class Ppt12McpServer:
    """One stdio MCP server process serving one task row's 12 slide tools."""

    def __init__(self, env: dict[str, str] | None = None) -> None:
        self._env = dict(os.environ if env is None else env)
        self._session: dict[str, Any] = {}
        self._tools: dict[str, Any] = {}
        self._log_path = (self._env.get(MCP_LOG_ENV) or '').strip()
        self._tool_configs = {
            'visual_agent_ppt12': {
                'deck_filename': (self._env.get(DECK_FILENAME_ENV) or '').strip() or 'slide.pptx',
            },
        }
        self._seed_reference()
        self._seed_workdir()
        self._seed_dump_dir()

    # -- session bootstrap -------------------------------------------------

    def _seed_reference(self) -> None:
        path = (self._env.get(REFERENCE_PNG_ENV) or '').strip()
        if not path:
            return
        try:
            from PIL import Image

            with Image.open(path) as image:
                image.load()
                self._session[IMAGE_STATE_KEY] = image
        except Exception as exc:  # noqa: BLE001 - degrade like _seed_slide_tools, do not die
            self._log(f'reference seed failed for {path}: {type(exc).__name__}: {exc}')

    def _seed_workdir(self) -> None:
        path = (self._env.get(WORKSPACE_ENV) or '').strip()
        if path and os.path.isdir(path):
            self._session[CODE_WORKDIR_STATE_KEY] = path

    def _seed_dump_dir(self) -> None:
        path = (self._env.get(DUMP_DIR_ENV) or '').strip()
        if not path:
            return
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            self._log(f'dump dir {path} unusable ({exc}); tools will fall back to a temp dir')
            return
        from .ppt_tools.image_tools import _DUMP_DIR_KEY

        self._session[_DUMP_DIR_KEY] = path

    def _context(self) -> ToolContext:
        return ToolContext(tool_state=self._session, tool_configs=self._tool_configs)

    def _absorb(self, updates: dict[str, Any]) -> None:
        self._session.update(updates)

    # -- protocol ----------------------------------------------------------

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Process one JSON-RPC request; return the response or None for notifications."""
        method = request.get('method')
        rid = request.get('id')
        if method == 'initialize':
            return {
                'jsonrpc': '2.0',
                'id': rid,
                'result': {
                    # Echo the client's own version: replying with an older one
                    # made the prototype's client silently drop every tool.
                    'protocolVersion': (request.get('params') or {}).get('protocolVersion', DEFAULT_PROTOCOL_VERSION),
                    'capabilities': {'tools': {'listChanged': False}},
                    'serverInfo': {'name': SERVER_NAME, 'version': SERVER_VERSION},
                },
            }
        if method == 'notifications/initialized':
            self._log('initialized handshake complete')
            return None
        if method == 'tools/list':
            return {'jsonrpc': '2.0', 'id': rid, 'result': {'tools': tool_schemas()}}
        if method == 'tools/call':
            params = request.get('params') or {}
            name = str(params.get('name') or '')
            args = params.get('arguments') or {}
            self._log(f'call {name} args={json.dumps(args, default=str)[:300]}')
            try:
                content = asyncio.run(self._dispatch(name, args if isinstance(args, dict) else {}))
                result: dict[str, Any] = {'content': content}
            except ToolCallError as exc:
                result = {'content': [{'type': 'text', 'text': str(exc)}], 'isError': True}
            except Exception as exc:  # noqa: BLE001 - one call must never kill the server
                self._log(f'tool {name} raised: {traceback.format_exc()[-1500:]}')
                result = {
                    'content': [{'type': 'text', 'text': f'Error: {type(exc).__name__}: {exc}'}],
                    'isError': True,
                }
            return {'jsonrpc': '2.0', 'id': rid, 'result': result}
        if method == 'ping':
            return {'jsonrpc': '2.0', 'id': rid, 'result': {}}
        if rid is not None:
            return {
                'jsonrpc': '2.0',
                'id': rid,
                'error': {'code': -32601, 'message': f'unknown method {method}'},
            }
        return None

    # -- dispatch ----------------------------------------------------------

    async def _dispatch(self, name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
        registry = load_ppt12(PPT12_TOOLS)
        tool = registry.get('visual/ppt_' + name)
        if tool is None:
            raise ToolCallError(f'Error: unknown tool {name}')
        meta = tool.__tool_meta__
        if name == 'render_slide' and not str(args.get('code') or '').strip():
            return await self._render_existing_deck()
        params = meta.input_model.model_validate(args)
        response = await tool(params, self._context())
        self._absorb(response.context_updates)
        return mcp_content(response.output)

    async def _render_existing_deck(self) -> list[dict[str, Any]]:
        """render_slide with no code renders the deck already at the workspace path.

        The prototype's second workflow: the model builds the deck with its own
        shell commands, then asks for a render of what is on disk. Uses the same
        rasterisation chain (soffice -> pdf -> pdftoppm 96dpi) and the same
        page-note shape as a coded render_slide call.
        """
        from PIL import Image

        from .ppt_tools import compile_tools
        from .ppt_tools.image_tools import RENDER_KEY, _image_note

        workspace = (self._env.get(WORKSPACE_ENV) or '').strip()
        deck_filename = self._tool_configs['visual_agent_ppt12']['deck_filename']
        deck_path = os.path.join(workspace, deck_filename) if workspace else deck_filename
        if not os.path.isfile(deck_path):
            raise ToolCallError(
                f'Error: render_slide without code renders the deck already in the working directory, '
                f'but there is no file at {deck_path}. Pass code, or write the deck first.'
            )
        work = tempfile.mkdtemp(prefix='pptx_render_')
        try:
            pngs, render_log, n_pages = await compile_tools._pptx_to_pngs(deck_path, work)
            if not pngs:
                raise ToolCallError(f'The .pptx at {deck_path} was found but rendering failed.\n{render_log}')
            images = []
            for path in pngs[: compile_tools._MAX_PAGES_RETURNED]:
                image = Image.open(path).convert('RGB')
                image.load()
                images.append(image)
        finally:
            shutil.rmtree(work, ignore_errors=True)

        context = self._context()
        parts: list[dict[str, Any]] = [
            {
                'type': 'text',
                'text': (
                    f'Rendered {n_pages} page(s) at {compile_tools.RASTER_DPI}dpi '
                    f'({images[0].size[0]}x{images[0].size[1]}px) from {deck_path}.'
                    + (f' Showing the first {len(images)}.' if n_pages > len(images) else '')
                ),
            }
        ]
        for idx, image in enumerate(images, start=1):
            note = _image_note(context, image, f'render_slide_page{idx}')
            self._absorb(note['updates'])
            parts.append({'type': 'text', 'text': f'Page {idx}: rendered {image.size[0]}x{image.size[1]}px.'})
            parts.append(note['part'])
        self._absorb({RENDER_KEY: images[0]})
        return mcp_content(parts)

    # -- plumbing ----------------------------------------------------------

    def serve(self, requests: Any, respond: Any) -> None:
        """Read JSON-RPC lines from *requests*, writing each response via *respond*."""
        self._log(f'server starting pid={os.getpid()}')
        for line in requests:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = self.handle(request)
            if response is not None:
                respond(response)

    def _log(self, message: str) -> None:
        if not self._log_path:
            return
        try:
            with open(self._log_path, 'a', encoding='utf-8') as handle:
                handle.write(f'{time.time():.3f} {message}\n')
        except Exception:  # noqa: BLE001 - logging must never break the protocol
            pass


def send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + '\n')
    sys.stdout.flush()


def main() -> None:
    Ppt12McpServer().serve(sys.stdin, send)


if __name__ == '__main__':
    main()
