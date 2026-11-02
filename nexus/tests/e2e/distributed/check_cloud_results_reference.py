"""Compare the installed exporter with a supplied reference script on fixed API/DB inputs.

This supplements live exports; it is not a live-service test.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib.util
import json
import sys

from collections import Counter
from pathlib import Path
from typing import Any, cast

from tests.e2e.distributed.compare_cloud_results import fingerprint

from nexus.distributed.export import results
from nexus.distributed.export.types import TraceResult


class Api:
    def __init__(self, statuses, *, partial):
        self.partial = partial
        self.items = [
            {
                'item_id': f'item-{i:04}',
                'status': status,
                'current_attempt_no': i % 3,
                'current_child_execution_id': None if status == 'cancelled' else f'child-{i:04}',
                'metadata': {'label': '中文', '_nexus': {'original_run_id': f'Original-{len(statuses) - i:04}'}},
            }
            for i, status in enumerate(statuses)
        ]
        self.outputs = {
            item['current_child_execution_id']: {
                'success': item['status'] == 'succeeded',
                'error': None,
                'task_output': {'resolved': True, 'nested': {'extra': {'time_unix_nano': 123456789}}},
                'agent_output': {'content': '你好\n"quoted" \\ backslash'},
                'processor_results': {
                    'chat': {'trajectories': [{'messages': [{'role': 'assistant', 'content': '完成'}]}]}
                },
                'extra': {'must_not_add_a_top_level_export_field': True},
            }
            for item in self.items
            if item['status'] == 'succeeded'
        }

    async def get_execution(self, execution_id):
        return {
            'execution_id': execution_id,
            'spec_type': 'nexus_batch' if execution_id == 'batch' else 'nexus_single',
            'status': 'running' if self.partial else 'failed',
            'attempt_count': 1,
            'parent_execution_id': None if execution_id == 'batch' else 'batch',
            'error_json': {'type': 'ExpectedFailure'},
        }

    async def get_batch_summary(self, execution_id):
        return {
            'generation_status': 'complete',
            'submission_closed': True,
            'summary_version': 1,
            'total_items': len(self.items),
            'materialized_count': len(self.items),
            'terminal_items': sum(x['status'] in results._TERMINAL_ITEM_STATUSES for x in self.items),
            'item_status_counts': dict(Counter(x['status'] for x in self.items)),
            'task_list_manifest': {'batch_metadata': {'dataset': 'fixture', 'nested': {'b': 2, 'a': 1}}},
        }

    async def get_batch_items(self, execution_id, cursor):
        start = 0 if cursor is None else next(i + 1 for i, item in enumerate(self.items) if item['item_id'] == cursor)
        items = self.items[start : start + 2]
        return {
            'items': copy.deepcopy(items),
            'next_cursor': items[-1]['item_id'] if start + 2 < len(self.items) else None,
        }

    async def get_result(self, execution_id):
        return copy.deepcopy(self.outputs.get(execution_id))

    async def get_control_events(self, execution_id):
        return []


class Reader:
    def __init__(self, api, result_type, *, fallback):
        self.api = api
        self.result_type = result_type
        self.fallback = fallback

    async def read_many(self, ids):
        return {
            execution_id: self.result_type(
                output=None if self.fallback else copy.deepcopy(self.api.outputs.get(execution_id)),
                start_time='trace-start',
                end_time='trace-end',
            )
            for execution_id in ids
        }

    async def iter_results(self, ids):
        for execution_id in ids:
            yield await self.read_many([execution_id])


async def run(reference, output):
    reports = []
    cases = {
        'success': (['succeeded'] * 5, False),
        'mixed_terminal': (['succeeded', 'cancelled', 'failed', 'killed', 'lost', 'expired'], False),
        'partial': (['pending', 'running', 'succeeded', 'cancelled', 'submitted', 'failed', 'succeeded'], True),
        'partial_empty': (['pending', 'running', 'submitted'], True),
        'empty_batch': ([], False),
        'multiple_pages': (['succeeded', 'cancelled', 'failed'] * 171, False),
    }
    for name, (statuses, partial) in cases.items():
        for fallback in (False, True):
            directory = output / f'{name}-fallback-{fallback}'
            directory.mkdir(parents=True, exist_ok=True)
            checksums = []
            for label, module, result_type in (
                ('reference', reference, reference.TraceResult),
                ('candidate', results, TraceResult),
            ):
                api = Api(statuses, partial=partial)
                path = directory / f'{label}.jsonl'
                await module.export_execution(
                    api=cast(Any, api),
                    execution_id='batch',
                    output=path,
                    force=False,
                    trace_reader=Reader(api, result_type, fallback=fallback),
                    allow_partial=partial,
                )
                checksums.append(fingerprint(path))
            sidecars = [
                (directory / f'{label}.jsonl.batch_metadata.json').read_bytes() for label in ('reference', 'candidate')
            ]
            equal = checksums[0]['md5'] == checksums[1]['md5'] and sidecars[0] == sidecars[1]
            reports.append({
                'case': directory.name,
                'equal': equal,
                'reference': checksums[0],
                'candidate': checksums[1],
            })
            print(json.dumps({'case': directory.name, 'equal': equal, 'md5': checksums[0]['md5']}), flush=True)
    (output / 'report.json').write_text(json.dumps(reports, indent=2) + '\n')
    return 0 if all(report['equal'] for report in reports) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('reference_export', args.reference)
    assert spec is not None
    assert spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    return asyncio.run(run(reference, args.output))


if __name__ == '__main__':
    raise SystemExit(main())
