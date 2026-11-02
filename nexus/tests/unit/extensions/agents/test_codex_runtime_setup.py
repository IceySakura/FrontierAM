from pathlib import Path

import pytest

from nexus.extensions.agents.codex import runtime_setup


def test_resolve_codex_home_uses_stable_name_when_explicitly_configured() -> None:
    assert (
        runtime_setup._resolve_codex_home('training-parity')
        == Path.home() / runtime_setup.CODEX_HOME_DIR_NAME / 'training-parity'
    )


def test_resolve_codex_home_keeps_random_per_run_default() -> None:
    first = runtime_setup._resolve_codex_home(None)
    second = runtime_setup._resolve_codex_home(None)

    assert first.parent == Path.home() / runtime_setup.CODEX_HOME_DIR_NAME
    assert second.parent == Path.home() / runtime_setup.CODEX_HOME_DIR_NAME
    assert first != second


@pytest.mark.parametrize('name', ['', '.', '..', 'nested/home', '/tmp/home', 'contains space'])
def test_resolve_codex_home_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError, match='codex_home_name'):
        runtime_setup._resolve_codex_home(name)
