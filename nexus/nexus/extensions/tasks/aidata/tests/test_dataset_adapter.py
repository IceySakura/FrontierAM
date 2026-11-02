from __future__ import annotations

import base64
import json
import stat

import pytest

from nexus.extensions.tasks.aidata import dataset_adapter as adapter


def _instance(case_id: str = 'work/poc/demo') -> dict:
    return {
        'instance_id': 'demo',
        'case_id': case_id,
        'files': [
            {
                'path': 'task.yaml',
                'encoding': 'utf-8',
                'content': 'task_id: demo\ninstruction: hello\n',
            },
            {
                'path': 'data/blob.bin',
                'encoding': 'base64',
                'content': base64.b64encode(b'\x00demo\xff').decode('ascii'),
            },
            {
                'path': 'run-test.sh',
                'encoding': 'utf-8',
                'content': '#!/usr/bin/env bash\nexit 0\n',
                'mode': '755',
            },
        ],
    }


def test_materialize_instance_record_restores_case_bundle(tmp_path) -> None:
    restored = adapter.materialize_instance_record(
        _instance(),
        tmp_path,
    )

    assert restored.instance_id == 'demo'
    assert restored.case_id == 'work/poc/demo'
    assert restored.cases_root == tmp_path / 'cases'
    assert restored.case_dir == tmp_path / 'cases' / 'work' / 'poc' / 'demo'
    assert restored.instance_path == tmp_path / 'instance.json'

    assert (restored.case_dir / 'task.yaml').read_text(encoding='utf-8') == 'task_id: demo\ninstruction: hello\n'
    assert (restored.case_dir / 'data' / 'blob.bin').read_bytes() == b'\x00demo\xff'
    assert stat.S_IMODE((restored.case_dir / 'run-test.sh').stat().st_mode) == 0o755
    assert restored.instance_path.read_text(encoding='utf-8').startswith('{\n  "instance_id": "demo"')


def test_materialize_rejects_path_traversal(tmp_path) -> None:
    record = _instance()
    record['files'][0]['path'] = '../escape.yaml'

    with pytest.raises(ValueError, match='unsafe files\\[\\]\\.path'):
        adapter.materialize_instance_record(record, tmp_path)


def test_materialize_rejects_absolute_case_id(tmp_path) -> None:
    record = _instance(case_id='/tmp/demo')

    with pytest.raises(ValueError, match='unsafe case_id'):
        adapter.materialize_instance_record(record, tmp_path)


def test_materialize_rejects_unknown_encoding(tmp_path) -> None:
    record = _instance()
    record['files'][0]['encoding'] = 'gzip'

    with pytest.raises(ValueError, match='unsupported encoding'):
        adapter.materialize_instance_record(record, tmp_path)


def test_materialized_case_exposes_run_env(tmp_path) -> None:
    restored = adapter.materialize_instance_record(_instance(), tmp_path)

    assert restored.run_env() == {
        'AIDATA_CASES_ROOT': str(tmp_path / 'cases'),
        'AIDATA_CASE_IDS': 'work/poc/demo',
    }


def test_cli_materializes_and_writes_case_id_file(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    instance_json = tmp_path / 'instance.json'
    output_dir = tmp_path / 'out'
    instance_json.write_text(json.dumps(_instance()), encoding='utf-8')

    assert adapter.main([str(instance_json), str(output_dir)]) == 0

    assert (output_dir / 'case_id.txt').read_text(encoding='utf-8') == 'work/poc/demo\n'
    generator = output_dir / 'generator.py'
    assert generator.is_file()
    assert "os.environ['AIDATA_CASES_ROOT']" in generator.read_text(encoding='utf-8')
    assert (output_dir / 'cases' / 'work' / 'poc' / 'demo' / 'task.yaml').is_file()
    out = capsys.readouterr().out
    assert 'case_id=work/poc/demo' in out
    assert f'case_id_file={output_dir / "case_id.txt"}' in out
    assert f'generator_file={generator}' in out
