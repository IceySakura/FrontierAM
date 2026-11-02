"""OSWorld harness assets regressions."""

from __future__ import annotations

import json

from pathlib import Path

import pytest

import nexus.extensions.tasks.os_world as os_world

from nexus.extensions.tasks.os_world.execution.claude_code import _guide_text
from nexus.extensions.tasks.os_world.execution.codex_profile import ASSETS, OSWorldCodexOptions


def test_claude_guides_are_task_owned(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    root = Path(os_world.__file__).resolve().parent / 'prompts' / 'claude_code'
    assert _guide_text('v1') == (root / 'computer_use_guide.md').read_text()
    assert _guide_text('v2') == (root / 'computer_use_guide_v2.md').read_text()
    assert 'Claude Code' in _guide_text('v2')
    with pytest.raises(ValueError, match='filename'):
        _guide_text(filename='../other.md')


@pytest.mark.parametrize(
    ('preset', 'model'),
    [
        ('gpt6_catalog_v8', 'gpt-6-astra'),
        ('gpt56_sol_catalog', 'gpt-5.6-sol'),
    ],
)
def test_codex_presets_are_task_owned(monkeypatch, tmp_path, preset, model):
    monkeypatch.chdir(tmp_path)
    options = OSWorldCodexOptions(
        user_prompt='',
        model=model,
        guide_version='v1',
        model_catalog_preset=preset,
    )
    assert Path(os_world.__file__).resolve().parent / 'assets' / 'codex' == ASSETS
    assert options.guide_text == (ASSETS / 'guide_codex_linux.md').read_text()
    assert any(entry['slug'] == model for entry in json.loads(options.model_catalog_json)['models'])


def test_codex_explicit_assets_override_even_invalid_selectors():
    catalog = (ASSETS / 'gpt6_catalog_v8.json').read_text()
    options = OSWorldCodexOptions(
        user_prompt='',
        guide_text='',
        guide_version='unused',
        model_catalog_json=catalog,
        model_catalog_preset='unused',
    )
    assert options.guide_text == ''
    assert json.loads(options.model_catalog_json)['models']
    with pytest.raises(ValueError, match='validation error'):
        OSWorldCodexOptions(
            user_prompt='',
            guide_text='custom',
            model_catalog_json='invalid',
            model_catalog_preset='gpt6_catalog_v8',
        )
