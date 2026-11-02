"""Tests for Anthropic proxy document (PDF) content block support.

Regression coverage for Claude Code's Read tool turning a PDF into a
``document`` block with a base64 source. The proxy previously rejected such
requests at pydantic validation (HTTP 422) because ``document`` was missing
from the ``ContentBlock`` union and had no converter branch.
"""

from nexus.runtime.server.services.anthropic_proxy.converter import anthropic_to_openai, convert_document_to_openai
from nexus.runtime.server.services.anthropic_proxy.models import DocumentContent, DocumentSource, MessagesRequest


PDF_BASE64 = 'JVBERi0xLjQKJVBERg=='


def _pdf_document_block():
    return {'type': 'document', 'source': {'type': 'base64', 'media_type': 'application/pdf', 'data': PDF_BASE64}}


def _read_tool_use_block():
    return {'type': 'tool_use', 'id': 'toolu_1', 'name': 'Read', 'input': {'file_path': '/workspace/x.pdf'}}


def test_pdf_document_block_in_tool_result_passes_validation():
    tool_result_msg = {
        'role': 'user',
        'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_1', 'content': [_pdf_document_block()]}],
    }
    assistant_msg = {'role': 'assistant', 'content': [_read_tool_use_block()]}
    payload = {'model': 'claude-sonnet-4-6', 'max_tokens': 1024, 'messages': [assistant_msg, tool_result_msg]}
    request = MessagesRequest.model_validate(payload)
    document = request.messages[1].content[0].content[0]
    assert isinstance(document, DocumentContent)
    assert document.source.media_type == 'application/pdf'
    assert document.source.data == PDF_BASE64


def test_pdf_document_block_as_top_level_user_block_passes_validation():
    payload = {
        'model': 'claude-sonnet-4-6',
        'max_tokens': 1024,
        'messages': [{'role': 'user', 'content': [_pdf_document_block()]}],
    }
    request = MessagesRequest.model_validate(payload)
    block = request.messages[0].content[0]
    assert isinstance(block, DocumentContent)
    assert block.source.type == 'base64'


def test_convert_document_base64_produces_openai_file_part():
    part = convert_document_to_openai({'type': 'base64', 'media_type': 'application/pdf', 'data': PDF_BASE64})
    assert part == {
        'type': 'file',
        'file': {'filename': 'document.pdf', 'file_data': f'data:application/pdf;base64,{PDF_BASE64}'},
    }


def test_convert_document_uses_title_as_filename():
    src = DocumentSource(type='base64', media_type='application/pdf', data=PDF_BASE64)
    part = convert_document_to_openai(src, title='report.pdf')
    assert part['type'] == 'file'
    assert part['file']['filename'] == 'report.pdf'


def test_convert_document_text_source_falls_back_to_text():
    part = convert_document_to_openai({'type': 'text', 'media_type': 'text/plain', 'text': 'hello world'})
    assert part == {'type': 'text', 'text': 'hello world'}


def test_convert_document_url_source_falls_back_to_text():
    part = convert_document_to_openai({'type': 'url', 'url': 'https://example.com/a.pdf'})
    assert part == {'type': 'text', 'text': '[Document: https://example.com/a.pdf]'}


def test_convert_document_without_data_falls_back_to_text():
    part = convert_document_to_openai({'type': 'base64', 'media_type': 'application/pdf'})
    assert part['type'] == 'text'
    assert 'application/pdf' in part['text']


def test_anthropic_to_openai_pdf_tool_result_becomes_file_part():
    tr_content = [{'type': 'text', 'text': 'PDF loaded'}, _pdf_document_block()]
    tool_result_msg = {
        'role': 'user',
        'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_1', 'content': tr_content}],
    }
    assistant_msg = {'role': 'assistant', 'content': [_read_tool_use_block()]}
    payload = {'model': 'claude-sonnet-4-6', 'max_tokens': 1024, 'messages': [assistant_msg, tool_result_msg]}
    openai_request = anthropic_to_openai(MessagesRequest.model_validate(payload))
    tool_messages = [m for m in openai_request['messages'] if m['role'] == 'tool']
    assert len(tool_messages) == 1
    parts = tool_messages[0]['content']
    file_parts = [p for p in parts if p.get('type') == 'file']
    assert len(file_parts) == 1
    assert file_parts[0]['file']['file_data'] == f'data:application/pdf;base64,{PDF_BASE64}'
    assert any(p.get('type') == 'text' and p.get('text') == 'PDF loaded' for p in parts)


def test_anthropic_to_openai_top_level_document_becomes_file_part():
    user_msg = {'role': 'user', 'content': [{'type': 'text', 'text': 'inspect this'}, _pdf_document_block()]}
    payload = {'model': 'claude-sonnet-4-6', 'max_tokens': 1024, 'messages': [user_msg]}
    openai_request = anthropic_to_openai(MessagesRequest.model_validate(payload))
    user_messages = [m for m in openai_request['messages'] if m['role'] == 'user']
    assert len(user_messages) == 1
    parts = user_messages[0]['content']
    assert any(p.get('type') == 'file' for p in parts)
