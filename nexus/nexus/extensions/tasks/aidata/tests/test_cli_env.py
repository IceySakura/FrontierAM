from __future__ import annotations

import yaml

from nexus.extensions.tasks.aidata import aidata


def test_load_env_into_matches_cases_then_repo_precedence(tmp_path, monkeypatch) -> None:
    cases_root = tmp_path / 'tasks' / 'cases'
    cases_root.mkdir(parents=True)
    repo_root = tmp_path / 'repo'
    repo_root.mkdir()

    (cases_root.parent / '.env').write_text(
        '\n'.join([
            'FROM_CASES=cases',
            'SHARED=cases',
            'EXPORTED=cases',
            'QUOTED="quoted value"',
            '',
        ]),
        encoding='utf-8',
    )
    (repo_root / '.env').write_text(
        '\n'.join([
            'FROM_REPO=repo',
            'SHARED=repo',
            'EXPORTED=repo',
            '',
        ]),
        encoding='utf-8',
    )
    monkeypatch.setattr(aidata, 'CASES_DIR', cases_root)
    monkeypatch.setattr(aidata, 'REPO_ROOT', repo_root)

    env = {'EXPORTED': 'shell'}
    aidata._load_env_into(env)

    assert env['EXPORTED'] == 'shell'
    assert env['SHARED'] == 'cases'
    assert env['FROM_CASES'] == 'cases'
    assert env['FROM_REPO'] == 'repo'
    assert env['QUOTED'] == 'quoted value'


def test_anthropic_template_routes_api_key_to_selected_model() -> None:
    template = yaml.safe_load(aidata.TEMPLATE_ANTHROPIC.read_text(encoding='utf-8'))

    assert template['openai_clients']['main']['config']['client_args']['api_key'] == (
        '${oc.env:SEARCH_OPENAI_API_KEY}?provider=aws_third&model=${oc.env:SEARCH_OPENAI_MODEL}'
    )
