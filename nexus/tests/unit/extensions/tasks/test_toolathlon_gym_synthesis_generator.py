from __future__ import annotations

import json

from nexus.extensions.tasks.toolathlon_gym import synthesis_generator


def test_synthesis_generator_preserves_bound_task_config(monkeypatch, tmp_path) -> None:
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(
        json.dumps(
            {
                'env_id': 'env-1',
                'task_index': 2,
                'dataset': 'mock-only',
                'nexus': {
                    'task_name': 'identity',
                    'task_config': {'value': 'ok', 'data_binding': {'task_spec_sha256': 'bound'}},
                },
            }
        )
        + '\n'
    )
    monkeypatch.setenv('SYNTHESIS_MANIFEST_JSONL', str(manifest))
    monkeypatch.setattr(synthesis_generator, '_build_openai_clients', lambda: {})

    item = next(synthesis_generator.generate_configs())

    assert item.run_id == 'bound-env-1-t2'
    assert item.config.task_config['data_binding']['task_spec_sha256'] == 'bound'
