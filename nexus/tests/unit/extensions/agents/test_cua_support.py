import base64
import importlib
import json
import sys

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.agents.cua import cua_support as support


def test_prompt_and_attribute_access():
    assert support._get({'x': 1}, 'x') == 1
    assert support._get(SimpleNamespace(x=2), 'x') == 2
    assert (
        support._instruction_text(
            [
                {'type': 'text', 'text': 'first'},
                {'type': 'image'},
                SimpleNamespace(type='text', text='second'),
            ]
        )
        == 'first\nsecond'
    )


@pytest.mark.asyncio
async def test_screenshot_decode_and_missing_image():
    raw = b'\x89PNG\r\n\x1a\nfixture'
    runtime = SimpleNamespace(gui_screenshot=AsyncMock(return_value={'image': support._b64(raw)}))
    assert await support._screenshot_bytes(runtime) == raw
    runtime.gui_screenshot.return_value = {'error': 'unavailable'}
    with pytest.raises(RuntimeError, match='no image'):
        await support._screenshot_bytes(runtime)


def test_trajectory_layout_and_result(tmp_path, monkeypatch):
    monkeypatch.setenv('CUA_TRAJ_DUMP_DIR', '/must-not-be-read')
    assert support._traj_dir(SimpleNamespace()) is None
    directory = support._traj_dir(
        SimpleNamespace(
            trajectory_dir=str(tmp_path / 'chrome__case'),
            osworld_domain='chrome',
            osworld_task_id='case',
        )
    )
    assert directory == str(tmp_path / 'chrome__case')
    support._init_traj_dir(directory)
    support._save_screenshot(directory, 'obs_001.png', b'png')
    support._append_jsonl(directory, 'trajectory.jsonl', {'text': '中文'})
    support._save_result(directory, {'score': 1})
    assert (Path(directory) / 'screenshots/obs_001.png').read_bytes() == b'png'
    assert json.loads((Path(directory) / 'trajectory.jsonl').read_text()) == {'text': '中文'}
    assert json.loads((Path(directory) / 'result.json').read_text()) == {'score': 1}
    explicit = support._traj_dir(SimpleNamespace(trajectory_dir=str(tmp_path / 'explicit')))
    assert explicit == str(tmp_path / 'explicit')


def test_bare_agents_import_without_another_model_agent():
    for name in ('cua_gpt_agent', 'cua_anthropic_agent'):
        module = importlib.import_module(f'nexus.extensions.agents.cua.{name}.agent')
        assert module._traj_dir is support._traj_dir
    assert 'nexus.extensions.agents.cua_kimi_agent.agent' not in sys.modules


def test_text_input_adapter_keeps_unicode_and_ascii_paths():
    from nexus.extensions.agents.cua.cua_text_input import text_input_code

    assert "typewrite('hello'" in text_input_code('hello')
    assert base64.b64encode('中文'.encode()).decode() in text_input_code('中文')


def test_cua_module_discovery():
    from nexus.core.module_meta import discover_all_modules

    module = next(m for m in discover_all_modules(include_compiled_manifest=False) if m.name == 'cua')
    assert module.objects['agents'] == {
        'cua_gpt_agent': 'nexus.extensions.agents.cua.cua_gpt_agent.agent',
        'cua_anthropic_agent': 'nexus.extensions.agents.cua.cua_anthropic_agent.agent',
        'macosworld_cua_gpt': 'nexus.extensions.agents.cua.macosworld_cua.gpt',
        'macosworld_cua_anthropic': 'nexus.extensions.agents.cua.macosworld_cua.anthropic',
        'macosworld_vlm': 'nexus.extensions.agents.cua.macosworld_vlm.agent',
    }
