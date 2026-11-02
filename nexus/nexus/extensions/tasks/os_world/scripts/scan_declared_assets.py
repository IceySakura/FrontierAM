"""Scan cached task definitions without a model, sandbox, or network access."""

from __future__ import annotations

import argparse
import hashlib
import json

from pathlib import Path

from nexus.extensions.tasks.os_world.data.prefetch import _declared_assets


def scan(cache_root: Path, cases_file: Path) -> dict:
    """Use captured Taiji cases as the exact task membership, not directory contents."""
    envelope = json.loads(cases_file.read_text(encoding='utf-8'))['data']
    cases = envelope['cases']
    if envelope.get('truncated') or envelope['total'] != len(cases):
        raise ValueError('Case export must be complete')
    results = []
    seen = set()
    for case in cases:
        payload = case['payload']
        payload = json.loads(payload) if isinstance(payload, str) else payload
        config = payload['nexus_request']['payload']['run_input']
        task_id = config['task_config']['task_id']
        if not isinstance(task_id, str) or not task_id or Path(task_id).name != task_id or task_id in {'.', '..'}:
            raise ValueError('Task id must be a single directory name')
        if task_id in seen:
            raise ValueError(f'Duplicate task id: {task_id}')
        seen.add(task_id)
        path = cache_root / task_id / 'task.json'
        row = {'task_id': task_id, 'question_id': case['question_id']}
        try:
            raw = path.read_bytes()
            task = json.loads(raw)
            if task.get('id') != task_id:
                raise ValueError('Cached task id does not match case')
            if task.get('instruction') != payload['meta_data_']['instruction']:
                raise ValueError('Cached instruction does not match case')
            declared = _declared_assets(task)
            row.update(
                status='ok',
                task_json_sha256=hashlib.sha256(raw).hexdigest(),
                declared={key: sorted(value) for key, value in declared.items()},
            )
        except Exception as exc:  # Report each invalid definition without skipping the rest.
            row.update(status='error', error_type=type(exc).__name__, error=str(exc))
        results.append(row)
    return {
        'scope': 'cached task.json declaration parsing; not remote availability or full setup/evaluator validation',
        'total': len(results),
        'passed': sum(row['status'] == 'ok' for row in results),
        'failed': sum(row['status'] == 'error' for row in results),
        'tasks': results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path, required=True, help='Directory containing <task_id>/task.json')
    parser.add_argument('--cases', type=Path, required=True, help='Complete Taiji case-detail JSON response')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = scan(args.cache_root, args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('total', 'passed', 'failed')}))
    raise SystemExit(1 if report['failed'] else 0)


if __name__ == '__main__':
    main()
