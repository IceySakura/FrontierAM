from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nexus.core import OpenAIConfig, TokenizerPanicError
from nexus.extensions.openai_clients.hytuner import HyTunerClient


PanicException = type('PanicException', (BaseException,), {'__module__': 'pyo3_runtime'})


def _build_panicking_client() -> HyTunerClient:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = 'rendered prompt'
    tokenizer.encode.side_effect = PanicException('NormalizedString bad split')
    return HyTunerClient(OpenAIConfig(client_args={'tokenizer': tokenizer}, request_args={}))


@pytest.mark.asyncio
async def test_create_completion_converts_tokenizer_panic_to_runtime_error(caplog: pytest.LogCaptureFixture) -> None:
    client = _build_panicking_client()

    with pytest.raises(TokenizerPanicError, match='Failed to tokenize request prompt'):
        await client.create_completion(messages=[{'role': 'user', 'content': 'question'}])

    assert 'operation=request prompt' in caplog.text
    assert "rendered_text='rendered prompt'" in caplog.text


@pytest.mark.asyncio
async def test_count_tokens_converts_tokenizer_panic_to_runtime_error() -> None:
    client = _build_panicking_client()

    with pytest.raises(TokenizerPanicError, match='Failed to tokenize token-count prompt'):
        await client.count_tokens(messages=[{'role': 'user', 'content': 'question'}])
