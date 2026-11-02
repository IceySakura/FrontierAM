
from __future__ import annotations

import base64
import io
import sys

from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from PIL import Image  # noqa: E402

from nexus.extensions.tasks.ppt_rubric.ppt_mcp_server import (  # noqa: E402
    DECK_FILENAME_ENV,
    DUMP_DIR_ENV,
    Ppt12McpServer,
    REFERENCE_PNG_ENV,
    WORKSPACE_ENV,
    _UNDECODABLE_IMAGE_NOTE,
    _data_url_to_image_part,
    mcp_content,
)
from nexus.extensions.tasks.ppt_rubric.ppt_tools import (  # noqa: E402
    PPT12_MODEL_NAMES,
    PPT12_TOOLS,
    load_ppt12,
)


def _png_bytes(size: tuple[int, int] = (320, 180), colour: str = '#336699') -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', size, colour).save(buf, format='PNG')
    return buf.getvalue()


def _call(server: Ppt12McpServer, name: str, args: dict) -> dict:
    response = server.handle({'jsonrpc': '2.0', 'id': 7, 'method': 'tools/call', 'params': {'name': name, 'arguments': args}})
    assert response is not None  # tools/call always answers
    return response


def _reference_server(tmp_path: Path, **env: str) -> Ppt12McpServer:
    png = tmp_path / 'reference.png'
    png.write_bytes(_png_bytes())
    values = {
        REFERENCE_PNG_ENV: str(png),
        WORKSPACE_ENV: str(tmp_path),
        DECK_FILENAME_ENV: 'slide.pptx',
        DUMP_DIR_ENV: str(tmp_path / 'ppt12_toolimg'),
    }
    values.update(env)
    return Ppt12McpServer(env=values)


def _tools_list() -> list[dict]:
    response = Ppt12McpServer(env={}).handle({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'})
    assert response is not None
    return response['result']['tools']


class TestToolsListMatchesTheLoopContract:
    """The alignment core: MCP must not mint a second tool contract. Names,
    descriptions and parameter schemas come verbatim from the registry the
    built-in loop serves -- byte-for-byte what the model saw in training."""

    def test_twelve_tools_in_registry_order_with_loop_names(self):
        tools = _tools_list()
        assert [tool['name'] for tool in tools] == list(PPT12_MODEL_NAMES)
        assert len(tools) == 12

    def test_descriptions_and_schemas_come_verbatim_from_the_registry(self):
        registry = load_ppt12(PPT12_TOOLS)
        for entry in _tools_list():
            meta = registry['visual/ppt_' + entry['name']].__tool_meta__
            assert entry['description'] == meta.description, entry['name']
            expected = meta.schema['function']['parameters']
            if meta.name != 'render_slide':
                assert entry['inputSchema'] == expected, entry['name']
                continue
            # The one deliberate deviation: render_slide's `code` is optional over
            # MCP so the "build the deck yourself, render without code" workflow
            # stays callable. Everything else about the schema is untouched.
            assert entry['inputSchema'].get('required') is None
            assert {key: value for key, value in entry['inputSchema'].items() if key != 'required'} == {
                key: value for key, value in expected.items() if key != 'required'
            }

    def test_input_schemas_carry_no_refs_or_defs(self):
        for entry in _tools_list():
            assert '$defs' not in entry['inputSchema'], entry['name']
            assert '$ref' not in entry['inputSchema'], entry['name']


class TestProtocolLoop:
    def test_initialize_echoes_the_client_protocol_version(self):
        response = Ppt12McpServer(env={}).handle({
            'jsonrpc': '2.0',
            'id': 0,
            'method': 'initialize',
            'params': {'protocolVersion': '2025-06-18'},
        })
        assert response is not None
        assert response['result']['protocolVersion'] == '2025-06-18'

    def test_notifications_produce_no_response(self):
        server = Ppt12McpServer(env={})
        assert server.handle({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) is None
        assert server.handle({'jsonrpc': '2.0', 'method': 'notifications/cancelled'}) is None

    def test_ping_answers_and_unknown_methods_report_minus_32601(self):
        server = Ppt12McpServer(env={})
        ping = server.handle({'jsonrpc': '2.0', 'id': 2, 'method': 'ping'})
        assert ping is not None
        assert ping['result'] == {}
        error = server.handle({'jsonrpc': '2.0', 'id': 3, 'method': 'resources/list'})
        assert error is not None
        assert error['error']['code'] == -32601

    def test_serve_skips_bad_lines_and_answers_good_ones(self, tmp_path):
        server = _reference_server(tmp_path)
        responses: list[dict] = []
        server.serve(['not json', '', '{"jsonrpc":"2.0","id":9,"method":"tools/list"}'], responses.append)
        assert len(responses) == 1
        assert responses[0]['id'] == 9


class TestToolsCall:
    def test_page_info_round_trip_measures_the_seeded_reference(self, tmp_path):
        server = _reference_server(tmp_path)
        response = _call(server, 'page_info', {})
        content = response['result']['content']
        assert response['result'].get('isError') is None
        assert content[0]['type'] == 'text'
        assert 'page_info:' in content[0]['text']
        assert '320 x 180 px' in content[0]['text']
        # Session absorb: the budget key from context_updates persisted.
        assert 'visual_agent.ppt12.n_colour_samples' in server._session

    def test_text_blocks_round_trip_accumulates_records_across_calls(self, tmp_path):
        server = _reference_server(tmp_path)
        assert 'isError' not in _call(server, 'page_info', {})['result']
        response = _call(server, 'text_blocks', {})
        assert 'isError' not in response['result']
        # Flat-colour reference: the real tool reply, same bytes the loop would get.
        assert 'No text blocks found' in response['result']['content'][0]['text']
        assert [record['tool'] for record in server._session['pptx_tool_records']] == ['page_info', 'text_blocks']

    def test_inspect_region_returns_an_mcp_image_part(self, tmp_path):
        server = _reference_server(tmp_path)
        response = _call(server, 'inspect_region', {'bbox': [0, 0, 120, 90]})
        images = [part for part in response['result']['content'] if part['type'] == 'image']
        assert len(images) == 1
        assert images[0]['mimeType'] == 'image/png'
        Image.open(io.BytesIO(base64.b64decode(images[0]['data'])))  # round-trips back to a PNG
        assert (tmp_path / 'ppt12_toolimg').exists()

    def test_unknown_tool_is_an_error_result_not_a_crash(self, tmp_path):
        response = _call(_reference_server(tmp_path), 'no_such_tool', {})
        assert response['result']['isError'] is True
        assert 'unknown tool' in response['result']['content'][0]['text']

    def test_render_slide_without_code_and_without_a_deck_reports_the_fix(self, tmp_path):
        response = _call(_reference_server(tmp_path), 'render_slide', {})
        assert response['result']['isError'] is True
        text = response['result']['content'][0]['text']
        assert 'no file at' in text
        assert 'slide.pptx' in text


class TestImagePartConversion:
    def test_data_url_image_part_round_trip(self):
        data = base64.b64encode(_png_bytes(size=(4, 4))).decode('utf-8')
        part = {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{data}'}}
        assert _data_url_to_image_part(part) == {'type': 'image', 'data': data, 'mimeType': 'image/png'}

    def test_remote_urls_cannot_become_image_parts(self):
        part = {'type': 'image_url', 'image_url': {'url': 'https://example.invalid/x.png'}}
        assert _data_url_to_image_part(part) is None
        assert mcp_content([part]) == [{'type': 'text', 'text': _UNDECODABLE_IMAGE_NOTE}]

    def test_malformed_base64_degrades_to_a_text_note(self):
        part = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,@@not-b64@@'}}
        assert _data_url_to_image_part(part) is None
        assert mcp_content([part])[0]['type'] == 'text'

    def test_long_text_is_truncated_and_empty_output_stays_visible(self):
        truncated = mcp_content([{'type': 'text', 'text': 'x' * 300_000}])[0]['text']
        assert len(truncated) < 200_000
        assert truncated.endswith('[... truncated ...]')
        assert mcp_content([]) == [{'type': 'text', 'text': '(no output)'}]
