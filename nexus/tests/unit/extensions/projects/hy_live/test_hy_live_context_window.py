"""The shell reads the midbrain window and the last prompt size for hylive."""

from types import SimpleNamespace

from nexus.extensions.projects.hy_live.hylive_shell import _midbrain_context_window, _prompt_tokens


def test_prompt_tokens_reads_a_usage_object_or_mapping() -> None:
    assert _prompt_tokens(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=12))) == 12
    assert _prompt_tokens({'usage': {'prompt_tokens': 4}}) == 4
    assert _prompt_tokens(SimpleNamespace()) is None
    assert _prompt_tokens(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=True))) is None


def test_context_window_uses_an_explicit_request_arg() -> None:
    client = SimpleNamespace(config=SimpleNamespace(request_args={'max_input_tokens': 128_000, 'model': 'm'}))
    assert _midbrain_context_window(client) == 128_000
    client = SimpleNamespace(config=SimpleNamespace(request_args={'context_window': 32_000, 'model': 'm'}))
    assert _midbrain_context_window(client) == 32_000


def test_context_window_is_unknown_without_a_model_or_a_catalog_entry() -> None:
    assert _midbrain_context_window(SimpleNamespace()) is None
    client = SimpleNamespace(config=SimpleNamespace(request_args={'model': 'not-a-catalog-name'}))
    assert _midbrain_context_window(client) is None
