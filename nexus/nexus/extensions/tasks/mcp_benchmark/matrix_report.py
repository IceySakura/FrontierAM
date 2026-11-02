from __future__ import annotations

import os
import sys


if __package__ in {None, ''}:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    while _SCRIPT_DIR in sys.path:
        sys.path.remove(_SCRIPT_DIR)

import argparse
import importlib.util
import json
import re

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


if __package__:
    from .constants import WORKSPACE_MCP_ROOT
    from .manifests import load_atlas_manifest_rows, load_toolathlon_manifest_rows, load_universe_manifest_rows
else:
    _MODULE_ROOT = Path(__file__).resolve().parent

    def _ensure_pkg(name: str, path: Path | None = None) -> None:
        if name in sys.modules:
            return
        module = ModuleType(name)
        if path is not None:
            module.__path__ = [str(path)]
        sys.modules[name] = module

    def _load_module(module_name: str, path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Unable to load module from {path}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    _PROJECT_ROOT = _MODULE_ROOT.parents[3]
    _ensure_pkg('nexus', _PROJECT_ROOT / 'nexus')
    _ensure_pkg('nexus.extensions', _PROJECT_ROOT / 'nexus' / 'extensions')
    _ensure_pkg('nexus.extensions.tasks', _PROJECT_ROOT / 'nexus' / 'extensions' / 'tasks')
    _ensure_pkg('nexus.extensions.tasks.mcp_benchmark', _MODULE_ROOT)

    _CONSTANTS = _load_module('nexus.extensions.tasks.mcp_benchmark.constants', _MODULE_ROOT / 'constants.py')
    WORKSPACE_MCP_ROOT = _CONSTANTS.WORKSPACE_MCP_ROOT

    _MANIFESTS = _load_module('nexus.extensions.tasks.mcp_benchmark.manifests', _MODULE_ROOT / 'manifests.py')
    load_toolathlon_manifest_rows = _MANIFESTS.load_toolathlon_manifest_rows
    load_atlas_manifest_rows = _MANIFESTS.load_atlas_manifest_rows
    load_universe_manifest_rows = _MANIFESTS.load_universe_manifest_rows


DATASET_ORDER = {
    'toolathlon': 0,
    'atlas': 1,
    'universe': 2,
}

SERVER_MODE_ORDER = {
    'api': 0,
}

SERVER_MODE_META = {
    'api': {'label': 'API', 'color': '#1d4ed8'},
}

TOOL_KIND_META = {
    'local': {
        'label': 'Local',
        'short': 'LOC',
        'color': '#0f766e',
        'description': 'Local stdio / workspace tools',
    },
    'browser': {
        'label': 'Browser',
        'short': 'WEB',
        'color': '#b45309',
        'description': 'Browser-driven MCP servers',
    },
    'remote': {
        'label': 'Remote',
        'short': 'API',
        'color': '#2563eb',
        'description': 'Remote API backed MCP servers',
    },
}

RUN_STATE_META = {
    'passed': {'label': 'Passed', 'color': '#1d7a5f'},
    'eval-failed': {'label': 'Eval Failed', 'color': '#c2410c'},
    'run-error': {'label': 'Run Error', 'color': '#b42318'},
    'not-run': {'label': 'Not Run', 'color': '#64748b'},
}

DATASET_META = {
    'toolathlon': {'label': 'Toolathlon', 'color': '#0f766e'},
    'atlas': {'label': 'Atlas', 'color': '#c2410c'},
    'universe': {'label': 'Universe', 'color': '#2563eb'},
}

LOCAL_TOOLS = {
    'arxiv-latex',
    'arxiv_local',
    'calculator',
    'cli-mcp-server',
    'date',
    'desktop-commander',
    'echo',
    'excel',
    'fetch',
    'filesystem',
    'git',
    'memory',
    'mcp-code-executor',
    'mcp-server-code-runner',
    'pdf-tools',
    'pptx',
    'terminal',
    'time',
    'word',
}

BROWSER_TOOLS = {
    'playwright',
    'playwright_with_chunk',
}

ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
WHITESPACE_RE = re.compile(r'\s+')
SCRIPT_CLOSE_RE = re.compile(r'</(script)', re.IGNORECASE)


def _normalize_whitespace(text: str, *, limit: int | None = None) -> str:
    normalized = WHITESPACE_RE.sub(' ', text).strip()
    if limit is not None and len(normalized) > limit:
        return normalized[: limit - 3].rstrip() + '...'
    return normalized


def _read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='ignore')


def _load_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _dedupe_keep_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in values if item))


def _tool_kind(tool_name: str) -> str:
    if tool_name in BROWSER_TOOLS:
        return 'browser'
    if tool_name in LOCAL_TOOLS:
        return 'local'
    return 'remote'


def _tool_short(tool_name: str) -> str:
    parts = [part for part in re.split(r'[^a-zA-Z0-9]+', tool_name) if part]
    if len(parts) >= 2:
        joined = ''.join(part[0].upper() for part in parts)
        return joined[:4]
    if not parts:
        return '?'
    return parts[0][:4].upper()


def _task_record(
    *,
    dataset: str,
    server_mode: str,
    case_id: str,
    prompt: str,
    server_names: list[str],
    source_path: str,
    category: str | None = None,
) -> dict[str, Any]:
    ordered_servers = _dedupe_keep_order(server_names)
    return {
        'dataset': dataset,
        'server_mode': server_mode,
        'case_id': case_id,
        'run_id': f'{dataset}-{server_mode}-{case_id}',
        'prompt': prompt,
        'server_names': ordered_servers,
        'server_count': len(ordered_servers),
        'source_path': source_path,
        'category': category,
    }


def _collect_toolathlon_tasks(server_modes: list[str]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for row in load_toolathlon_manifest_rows():
        for server_mode in server_modes:
            tasks.append(
                _task_record(
                    dataset='toolathlon',
                    server_mode=server_mode,
                    case_id=row['case_id'],
                    prompt=_normalize_whitespace(row['prompt']),
                    server_names=list(row['server_names']),
                    source_path=row['source_path'],
                )
            )
    return tasks


def _collect_atlas_tasks(server_modes: list[str]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for row in load_atlas_manifest_rows():
        for server_mode in server_modes:
            tasks.append(
                _task_record(
                    dataset='atlas',
                    server_mode=server_mode,
                    case_id=row['case_id'],
                    prompt=_normalize_whitespace(row['prompt']),
                    server_names=list(row['server_names']),
                    source_path=row['source_path'],
                    category=None,
                )
            )
    return tasks


def _collect_universe_tasks(server_modes: list[str]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for row in load_universe_manifest_rows():
        for server_mode in server_modes:
            tasks.append(
                _task_record(
                    dataset='universe',
                    server_mode=server_mode,
                    case_id=row['case_id'],
                    prompt=_normalize_whitespace(row['prompt']),
                    server_names=list(row['server_names']),
                    source_path=row['source_path'],
                    category=row.get('category'),
                )
            )
    return tasks


def _normalize_server_modes(server_modes: list[str] | None) -> list[str]:
    requested = server_modes or list(SERVER_MODE_ORDER)
    normalized = [mode for mode in SERVER_MODE_ORDER if mode in requested]
    return normalized or ['api']


def collect_all_tasks(datasets: list[str] | None = None, server_modes: list[str] | None = None) -> list[dict[str, Any]]:
    requested = set(datasets or DATASET_ORDER)
    modes = _normalize_server_modes(server_modes)
    tasks: list[dict[str, Any]] = []
    if 'toolathlon' in requested:
        tasks.extend(_collect_toolathlon_tasks(modes))
    if 'atlas' in requested:
        tasks.extend(_collect_atlas_tasks(modes))
    if 'universe' in requested:
        tasks.extend(_collect_universe_tasks(modes))
    tasks.sort(
        key=lambda item: (
            DATASET_ORDER.get(item['dataset'], 99),
            SERVER_MODE_ORDER.get(item['server_mode'], 99),
            item['case_id'],
        )
    )
    return tasks


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _strip_ansi(text: str | None) -> str | None:
    if text is None:
        return None
    return ANSI_ESCAPE_RE.sub('', text)


def _summarize_error(text: str | None, *, limit: int = 240) -> str | None:
    cleaned = _normalize_whitespace(_strip_ansi(text or '') or '', limit=limit)
    return cleaned or None


def _run_timestamp(start_time: str | None, end_time: str | None, fallback_mtime: float) -> float:
    end_dt = _parse_iso_datetime(end_time)
    if end_dt is not None:
        return end_dt.timestamp()
    start_dt = _parse_iso_datetime(start_time)
    if start_dt is not None:
        return start_dt.timestamp()
    return fallback_mtime


def _infer_eval_excerpt(task_output: dict[str, Any]) -> str | None:
    passed = task_output.get('passed')
    if passed is not False:
        return None
    score = task_output.get('score')
    eval_details = task_output.get('eval_details') or {}
    if isinstance(eval_details, dict):
        for key in ('error', 'reason', 'stderr', 'stdout'):
            value = eval_details.get(key)
            if value:
                return _summarize_error(str(value))
        claims = eval_details.get('claims')
        if isinstance(claims, list):
            unsupported = [item for item in claims if item.get('verdict') != 'FULLY_SUPPORTED']
            if unsupported:
                return f'{len(unsupported)} claim(s) not fully supported'
        results = eval_details.get('results')
        if isinstance(results, list):
            failed = [item for item in results if not item.get('passed', False)]
            if failed:
                return f'{len(failed)} evaluator check(s) failed'
    if isinstance(score, (int, float)):
        return f'evaluation score {float(score):.2f}'
    return 'evaluation failed'


def _derive_run_state(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return 'not-run'
    if summary.get('status') != 'success':
        return 'run-error'
    if summary.get('passed') is False:
        return 'eval-failed'
    if summary.get('passed') is True:
        return 'passed'
    return 'run-error'


def _guess_run_identity(run_id: str, metadata: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    dataset = metadata.get('dataset')
    server_mode = metadata.get('server_mode')
    case_id = metadata.get('case_id')
    if dataset and case_id:
        normalized_mode = str(server_mode or 'api')
        if normalized_mode not in SERVER_MODE_ORDER:
            normalized_mode = 'api'
        return str(dataset), normalized_mode, str(case_id)
    if '-' not in run_id:
        return None, None, None
    dataset, server_mode, case_id = run_id.split('-', 2) if run_id.count('-') >= 2 else (run_id, '', '')
    if dataset in DATASET_ORDER and server_mode in SERVER_MODE_ORDER and case_id:
        return dataset, server_mode, case_id
    dataset, _, case_id = run_id.partition('-')
    if dataset in DATASET_ORDER and case_id:
        return dataset, 'api', case_id
    return None, None, None


def _load_export_run(run_dir: Path) -> dict[str, Any] | None:
    status_path = run_dir / 'status.json'
    if not status_path.exists():
        return None

    status = _load_json(status_path)
    metadata_path = run_dir / 'metadata.json'
    metadata = _load_json(metadata_path) if metadata_path.exists() else {}
    task_output_path = run_dir / 'task_output.json'
    task_output = _load_json(task_output_path) if task_output_path.exists() else {}

    dataset, server_mode, case_id = _guess_run_identity(run_dir.name, metadata)
    if not dataset or not server_mode or not case_id:
        return None

    start_time = status.get('start_time')
    end_time = status.get('end_time')
    run_status = status.get('status')
    passed = task_output.get('passed') if task_output else None
    score = task_output.get('score') if task_output else None
    error_excerpt = (
        _summarize_error(status.get('error')) if run_status != 'success' else _infer_eval_excerpt(task_output)
    )

    return {
        'run_id': f'{dataset}-{server_mode}-{case_id}',
        'dataset': dataset,
        'server_mode': server_mode,
        'case_id': case_id,
        'status': run_status,
        'passed': passed,
        'score': score,
        'start_time': start_time,
        'end_time': end_time,
        'error_excerpt': error_excerpt,
        'path': str(run_dir),
        'source': 'export-run',
        'timestamp': _run_timestamp(start_time, end_time, status_path.stat().st_mtime),
    }


def _load_loose_result(path: Path) -> dict[str, Any] | None:
    data = _load_json(path)
    if not isinstance(data, dict):
        return None
    task_output = data.get('task_output', data)
    if task_output is None:
        return None
    if not isinstance(task_output, dict):
        return None
    dataset = task_output.get('dataset')
    server_mode = task_output.get('server_mode') or 'api'
    case_id = task_output.get('case_id')
    if dataset not in DATASET_ORDER or server_mode not in SERVER_MODE_ORDER or not case_id:
        return None
    return {
        'run_id': f'{dataset}-{server_mode}-{case_id}',
        'dataset': dataset,
        'server_mode': server_mode,
        'case_id': case_id,
        'status': 'success',
        'passed': task_output.get('passed'),
        'score': task_output.get('score'),
        'start_time': None,
        'end_time': None,
        'error_excerpt': _infer_eval_excerpt(task_output),
        'path': str(path),
        'source': 'loose-result',
        'timestamp': path.stat().st_mtime,
    }


def discover_latest_runs(workspace_root: Path) -> dict[str, dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for status_path in sorted(workspace_root.glob('**/runs/*/status.json')):
        summary = _load_export_run(status_path.parent)
        if summary is not None:
            candidates[summary['run_id']].append(summary)

    for loose_path in sorted(workspace_root.glob('*.result.json')):
        summary = _load_loose_result(loose_path)
        if summary is not None:
            candidates[summary['run_id']].append(summary)

    for loose_path in sorted(workspace_root.glob('result_*.json')):
        summary = _load_loose_result(loose_path)
        if summary is not None:
            candidates[summary['run_id']].append(summary)

    latest_runs: dict[str, dict[str, Any]] = {}
    source_rank = {'loose-result': 0, 'export-run': 1}
    for run_id, entries in candidates.items():
        best = max(entries, key=lambda item: (item['timestamp'], source_rank.get(item['source'], -1)))
        best = dict(best)
        best['attempts'] = len(entries)
        latest_runs[run_id] = best
    return latest_runs


def _build_tool_list(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    datasets_by_tool: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        for tool_name in task['server_names']:
            counts[tool_name] += 1
            datasets_by_tool[tool_name].add(task['dataset'])

    tools = []
    for tool_name in counts:
        kind = _tool_kind(tool_name)
        kind_meta = TOOL_KIND_META[kind]
        tools.append({
            'name': tool_name,
            'kind': kind,
            'label': kind_meta['label'],
            'kind_short': kind_meta['short'],
            'color': kind_meta['color'],
            'short': _tool_short(tool_name),
            'task_count': counts[tool_name],
            'datasets': sorted(datasets_by_tool[tool_name], key=lambda item: DATASET_ORDER.get(item, 99)),
        })
    kind_order = {'browser': 0, 'remote': 1, 'local': 2}
    tools.sort(key=lambda item: (kind_order.get(item['kind'], 99), item['name']))
    return tools


def build_report_data(
    workspace_root: Path,
    datasets: list[str] | None = None,
    server_modes: list[str] | None = None,
) -> dict[str, Any]:
    tasks = collect_all_tasks(datasets, server_modes)
    latest_runs = discover_latest_runs(workspace_root)

    report_tasks: list[dict[str, Any]] = []
    for task in tasks:
        run_summary = latest_runs.get(task['run_id'])
        run_state = _derive_run_state(run_summary)
        run_payload = {
            'has_run': run_summary is not None,
            'state': run_state,
            'status': run_summary.get('status') if run_summary is not None else None,
            'passed': run_summary.get('passed') if run_summary is not None else None,
            'score': run_summary.get('score') if run_summary is not None else None,
            'attempts': run_summary.get('attempts', 0) if run_summary is not None else 0,
            'start_time': run_summary.get('start_time') if run_summary is not None else None,
            'end_time': run_summary.get('end_time') if run_summary is not None else None,
            'error_excerpt': run_summary.get('error_excerpt') if run_summary is not None else None,
            'path': run_summary.get('path') if run_summary is not None else None,
            'source': run_summary.get('source') if run_summary is not None else None,
            'timestamp': run_summary.get('timestamp') if run_summary is not None else 0,
        }
        report_tasks.append({**task, 'run': run_payload})

    tools = _build_tool_list(report_tasks)
    task_count = len(report_tasks)
    tool_count = len(tools)
    dataset_counts = Counter(task['dataset'] for task in report_tasks)
    server_mode_counts = Counter(task['server_mode'] for task in report_tasks)
    run_state_counts = Counter(task['run']['state'] for task in report_tasks)
    run_count = sum(1 for task in report_tasks if task['run']['has_run'])

    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'workspace_root': str(workspace_root),
        'server_modes': _normalize_server_modes(server_modes),
        'tasks': report_tasks,
        'tools': tools,
        'summary': {
            'task_count': task_count,
            'tool_count': tool_count,
            'dataset_counts': dict(dataset_counts.items()),
            'server_mode_counts': dict(server_mode_counts.items()),
            'run_state_counts': dict(run_state_counts.items()),
            'run_count': run_count,
        },
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="icon" href="data:,">
<title>__TITLE__</title>
<style>
  :root {
    --paper: #f5efe4;
    --paper-strong: #efe3d1;
    --ink: #1f2937;
    --ink-dim: #5f6b7a;
    --navy: #10243a;
    --navy-soft: #1d3557;
    --line: #d8c9b6;
    --card: rgba(255, 250, 244, 0.92);
    --shadow: 0 24px 60px rgba(16, 36, 58, 0.16);
    --passed: #1d7a5f;
    --eval-failed: #c2410c;
    --run-error: #b42318;
    --not-run: #64748b;
    --tool-local: #0f766e;
    --tool-browser: #b45309;
    --tool-remote: #2563eb;
    --dataset-toolathlon: #0f766e;
    --dataset-atlas: #c2410c;
    --dataset-universe: #2563eb;
  }

  * {
    box-sizing: border-box;
  }

  html, body {
    margin: 0;
    min-height: 100%;
    background:
      radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 28%),
      radial-gradient(circle at top right, rgba(180, 83, 9, 0.10), transparent 24%),
      linear-gradient(180deg, #fbf7f1 0%, var(--paper) 100%);
    color: var(--ink);
    font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
  }

  body {
    padding: 22px;
  }

  .shell {
    max-width: 100%;
  }

  .hero {
    position: sticky;
    top: 0;
    z-index: 120;
    margin-bottom: 14px;
    padding: 18px 22px;
    border: 1px solid rgba(16, 36, 58, 0.10);
    border-radius: 20px;
    background:
      linear-gradient(135deg, rgba(16, 36, 58, 0.98), rgba(29, 53, 87, 0.95)),
      radial-gradient(circle at top right, rgba(255,255,255,0.10), transparent 32%);
    color: #f8fafc;
    box-shadow: var(--shadow);
    backdrop-filter: blur(16px);
  }

  .hero-top {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: flex-start;
    flex-wrap: wrap;
  }

  .eyebrow {
    font-size: 11px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: rgba(248, 250, 252, 0.68);
    margin-bottom: 8px;
  }

  h1 {
    margin: 0;
    font-size: 28px;
    line-height: 1.1;
    font-weight: 700;
  }

  .hero-subtitle {
    margin-top: 10px;
    max-width: 980px;
    color: rgba(248, 250, 252, 0.78);
    font-size: 14px;
    line-height: 1.5;
  }

  .stats {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 16px;
  }

  .stat-card {
    min-width: 118px;
    padding: 10px 12px;
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.08);
  }

  .stat-label {
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(248, 250, 252, 0.64);
  }

  .stat-value {
    margin-top: 6px;
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
  }

  .stamp {
    font-size: 11px;
    color: rgba(248, 250, 252, 0.64);
    white-space: nowrap;
  }

  .controls,
  .legend {
    border: 1px solid var(--line);
    border-radius: 18px;
    background: var(--card);
    box-shadow: 0 14px 34px rgba(16, 36, 58, 0.08);
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    padding: 14px 16px;
    margin-bottom: 12px;
  }

  .control-group {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .control-group label {
    color: var(--ink-dim);
    font-size: 12px;
    font-weight: 600;
  }

  .control-group select,
  .control-group input {
    height: 36px;
    border-radius: 10px;
    border: 1px solid var(--line);
    background: #fffdfa;
    color: var(--ink);
    padding: 0 12px;
    font-size: 12px;
    font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
    min-width: 140px;
  }

  .control-group input {
    min-width: 260px;
  }

  .control-group button {
    height: 36px;
    border-radius: 10px;
    border: 1px solid rgba(16, 36, 58, 0.16);
    background: linear-gradient(180deg, #fffdfa 0%, #f1e6d8 100%);
    color: var(--navy);
    padding: 0 14px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 14px 18px;
    align-items: center;
    padding: 12px 16px;
    margin-bottom: 14px;
  }

  .legend-title {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-dim);
    font-weight: 700;
  }

  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 12px;
    color: var(--ink);
  }

  .legend-swatch {
    width: 12px;
    height: 12px;
    border-radius: 3px;
    border: 1px solid rgba(16, 36, 58, 0.12);
    flex: 0 0 auto;
  }

  .matrix-card {
    border: 1px solid var(--line);
    border-radius: 20px;
    background: rgba(255, 252, 247, 0.96);
    box-shadow: var(--shadow);
    overflow: hidden;
  }

  .matrix-wrap {
    overflow: auto;
    max-height: calc(100vh - 330px);
  }

  table {
    border-collapse: separate;
    border-spacing: 0;
    width: max-content;
    min-width: 100%;
    font-size: 11px;
  }

  thead {
    position: sticky;
    top: 0;
    z-index: 40;
  }

  th,
  td {
    border-right: 1px solid rgba(216, 201, 182, 0.9);
    border-bottom: 1px solid rgba(216, 201, 182, 0.9);
  }

  thead th {
    background: linear-gradient(180deg, #f9f3ea 0%, #f0e5d6 100%);
  }

  thead th:first-child {
    position: sticky;
    left: 0;
    z-index: 70;
    min-width: 390px;
    max-width: 390px;
    width: 390px;
    padding: 14px;
    text-align: left;
    vertical-align: middle;
    box-shadow: 1px 0 0 rgba(216, 201, 182, 0.9);
  }

  thead .tool-th {
    position: relative;
    width: 28px;
    min-width: 28px;
    max-width: 28px;
    height: 176px;
    padding: 0;
    vertical-align: bottom;
    cursor: pointer;
    user-select: none;
    transition: background 0.15s ease;
  }

  thead .tool-th:hover {
    background: linear-gradient(180deg, #f3e7d8 0%, #eadcc9 100%);
  }

  thead .tool-th.sorted {
    background: linear-gradient(180deg, #ede1d1 0%, #e1d1bb 100%);
  }

  .tool-label {
    position: absolute;
    left: 50%;
    bottom: 12px;
    transform: translateX(-50%) rotate(-65deg);
    transform-origin: bottom left;
    white-space: nowrap;
    font-size: 10px;
    font-weight: 700;
    font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
  }

  .tool-count {
    position: absolute;
    top: 8px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 9px;
    color: var(--ink-dim);
    font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
  }

  .sort-pill {
    position: absolute;
    top: 22px;
    left: 50%;
    transform: translateX(-50%);
    min-width: 18px;
    padding: 2px 4px;
    border-radius: 999px;
    background: rgba(16, 36, 58, 0.10);
    color: var(--navy);
    font-size: 9px;
    font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
  }

  thead .kind-row th {
    height: 28px;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  thead .kind-row th:first-child {
    position: sticky;
    left: 0;
    z-index: 70;
    padding: 8px 14px;
    font-size: 10px;
    color: var(--ink-dim);
    background: linear-gradient(180deg, #f9f3ea 0%, #f0e5d6 100%);
  }

  .kind-chip {
    display: block;
    text-align: center;
    font-weight: 800;
    font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
  }

  tbody td:first-child {
    position: sticky;
    left: 0;
    z-index: 20;
    width: 390px;
    min-width: 390px;
    max-width: 390px;
    padding: 12px 14px;
    background: rgba(255, 251, 246, 0.98);
    box-shadow: 1px 0 0 rgba(216, 201, 182, 0.9);
  }

  tbody tr:hover td {
    background: rgba(16, 36, 58, 0.03);
  }

  tbody tr:hover td:first-child {
    background: rgba(255, 249, 242, 0.98);
  }

  tbody tr.state-passed td:first-child {
    box-shadow: inset 4px 0 0 var(--passed), 1px 0 0 rgba(216, 201, 182, 0.9);
  }

  tbody tr.state-eval-failed td:first-child {
    box-shadow: inset 4px 0 0 var(--eval-failed), 1px 0 0 rgba(216, 201, 182, 0.9);
  }

  tbody tr.state-run-error td:first-child {
    box-shadow: inset 4px 0 0 var(--run-error), 1px 0 0 rgba(216, 201, 182, 0.9);
  }

  tbody tr.state-not-run td:first-child {
    box-shadow: inset 4px 0 0 var(--not-run), 1px 0 0 rgba(216, 201, 182, 0.9);
  }

  .task-card {
    display: grid;
    gap: 8px;
  }

  .task-head {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .task-id {
    font-size: 13px;
    font-weight: 800;
    color: var(--navy);
    font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
    word-break: break-word;
  }

  .dataset-chip,
  .state-chip,
  .meta-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-height: 22px;
    padding: 0 9px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
    border: 1px solid transparent;
    white-space: nowrap;
  }

  .dataset-toolathlon {
    color: var(--dataset-toolathlon);
    background: rgba(15, 118, 110, 0.10);
    border-color: rgba(15, 118, 110, 0.20);
  }

  .dataset-atlas {
    color: var(--dataset-atlas);
    background: rgba(194, 65, 12, 0.10);
    border-color: rgba(194, 65, 12, 0.18);
  }

  .dataset-universe {
    color: var(--dataset-universe);
    background: rgba(37, 99, 235, 0.10);
    border-color: rgba(37, 99, 235, 0.18);
  }

  .state-passed {
    color: var(--passed);
    background: rgba(29, 122, 95, 0.10);
    border-color: rgba(29, 122, 95, 0.16);
  }

  .state-eval-failed {
    color: var(--eval-failed);
    background: rgba(194, 65, 12, 0.10);
    border-color: rgba(194, 65, 12, 0.16);
  }

  .state-run-error {
    color: var(--run-error);
    background: rgba(180, 35, 24, 0.10);
    border-color: rgba(180, 35, 24, 0.16);
  }

  .state-not-run {
    color: var(--not-run);
    background: rgba(100, 116, 139, 0.10);
    border-color: rgba(100, 116, 139, 0.16);
  }

  .meta-chip {
    color: var(--ink-dim);
    background: rgba(16, 36, 58, 0.05);
    border-color: rgba(16, 36, 58, 0.08);
  }

  .task-subline {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }

  .task-prompt,
  .task-tools,
  .task-error {
    font-size: 12px;
    line-height: 1.45;
  }

  .task-prompt {
    color: var(--ink);
  }

  .task-tools {
    color: var(--ink-dim);
    font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
  }

  .task-error {
    color: #8c2f15;
  }

  tbody td.matrix-cell {
    width: 28px;
    min-width: 28px;
    max-width: 28px;
    height: 28px;
    padding: 0;
    text-align: center;
    background: rgba(255, 252, 247, 0.92);
  }

  .matrix-cell.active {
    background: rgba(16, 36, 58, 0.02);
  }

  .cell-dot {
    display: inline-block;
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid rgba(16, 36, 58, 0.08);
  }

  .kind-local {
    background: rgba(15, 118, 110, 0.16);
  }

  .kind-local .cell-dot {
    background: rgba(15, 118, 110, 0.88);
  }

  .kind-browser {
    background: rgba(180, 83, 9, 0.14);
  }

  .kind-browser .cell-dot {
    background: rgba(180, 83, 9, 0.88);
  }

  .kind-remote {
    background: rgba(37, 99, 235, 0.14);
  }

  .kind-remote .cell-dot {
    background: rgba(37, 99, 235, 0.88);
  }

  .empty-state {
    padding: 40px 20px;
    text-align: center;
    color: var(--ink-dim);
    font-size: 14px;
  }

  .footer-note {
    padding: 12px 16px 16px;
    color: var(--ink-dim);
    font-size: 11px;
    border-top: 1px solid rgba(216, 201, 182, 0.9);
    background: rgba(248, 242, 233, 0.82);
  }

  @media (max-width: 920px) {
    body {
      padding: 12px;
    }

    .hero {
      padding: 16px;
      border-radius: 16px;
    }

    h1 {
      font-size: 23px;
    }

    .control-group input {
      min-width: 180px;
    }

    thead th:first-child,
    tbody td:first-child {
      min-width: 320px;
      max-width: 320px;
      width: 320px;
    }
  }
</style>
</head>
<body>
<div class="shell">
  <section class="hero">
    <div class="hero-top">
      <div>
        <div class="eyebrow">MCP benchmark coverage</div>
        <h1>__TITLE__</h1>
        <div class="hero-subtitle">
          One row per task across Toolathlon, Atlas, and Universe. Columns are MCP servers required by each task.
          Row badges reflect the latest exported run state discovered under <code>workspace/mcp/export_*</code>.
        </div>
      </div>
      <div class="stamp" id="generatedAt"></div>
    </div>
    <div class="stats" id="statsBar"></div>
  </section>

  <section class="controls">
    <div class="control-group">
      <label for="datasetFilter">Dataset</label>
      <select id="datasetFilter">
        <option value="all">All datasets</option>
        <option value="toolathlon">Toolathlon</option>
        <option value="atlas">Atlas</option>
        <option value="universe">Universe</option>
      </select>
    </div>
    <div class="control-group">
      <label for="statusFilter">Run State</label>
      <select id="statusFilter">
        <option value="all">All states</option>
        <option value="passed">Passed</option>
        <option value="eval-failed">Eval failed</option>
        <option value="run-error">Run error</option>
        <option value="not-run">Not run</option>
      </select>
    </div>
    <div class="control-group">
      <label for="serverModeFilter">Server Mode</label>
      <select id="serverModeFilter">
        <option value="all">All modes</option>
        <option value="api">API</option>
      </select>
    </div>
    <div class="control-group">
      <label for="sortMode">Sort</label>
      <select id="sortMode">
        <option value="default">Dataset + mode + case id</option>
        <option value="status">Run state</option>
        <option value="score">Score desc</option>
        <option value="updated">Latest run time</option>
        <option value="tool-count">MCP count desc</option>
      </select>
    </div>
    <div class="control-group">
      <label for="searchInput">Search</label>
      <input id="searchInput" type="text" placeholder="case id / prompt / tool / path">
    </div>
    <div class="control-group">
      <button id="resetButton" type="button">Reset</button>
    </div>
  </section>

  <section class="legend">
    <span class="legend-title">Run state</span>
    <span class="legend-item"><span class="legend-swatch" style="background: var(--passed)"></span>Passed</span>
    <span class="legend-item"><span class="legend-swatch" style="background: var(--eval-failed)"></span>Eval failed</span>
    <span class="legend-item"><span class="legend-swatch" style="background: var(--run-error)"></span>Run error</span>
    <span class="legend-item"><span class="legend-swatch" style="background: var(--not-run)"></span>Not run</span>
    <span class="legend-title">Tool kind</span>
    <span class="legend-item"><span class="legend-swatch" style="background: var(--tool-browser)"></span>Browser</span>
    <span class="legend-item"><span class="legend-swatch" style="background: var(--tool-remote)"></span>Remote API</span>
    <span class="legend-item"><span class="legend-swatch" style="background: var(--tool-local)"></span>Local stdio</span>
  </section>

  <section class="matrix-card">
    <div class="matrix-wrap">
      <table>
        <thead>
          <tr id="headerRow"></tr>
          <tr class="kind-row" id="kindRow"></tr>
        </thead>
        <tbody id="matrixBody"></tbody>
      </table>
    </div>
    <div class="footer-note" id="footerNote"></div>
  </section>
</div>

<script>
const REPORT = __REPORT_JSON__;
const RUN_STATE_META = __RUN_STATE_JSON__;
const TOOL_KIND_META = __TOOL_KIND_JSON__;
const DATASET_META = __DATASET_JSON__;
const SERVER_MODE_META = __SERVER_MODE_JSON__;
const DATASET_ORDER = { toolathlon: 0, atlas: 1, universe: 2 };
const SERVER_MODE_ORDER = { api: 0 };
const RUN_STATE_ORDER = { passed: 0, "eval-failed": 1, "run-error": 2, "not-run": 3 };

const state = {
  dataset: "all",
  serverMode: "all",
  status: "all",
  sort: "default",
  search: "",
  toolSort: null,
};

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function compact(value, limit = 180) {
  const text = String(value ?? "").replace(/\\s+/g, " ").trim();
  if (text.length <= limit) {
    return text;
  }
  return text.slice(0, limit - 3).trimEnd() + "...";
}

function formatScore(score) {
  return typeof score === "number" ? score.toFixed(2) : "n/a";
}

function formatTimestamp(value) {
  if (!value) {
    return "n/a";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return date.toLocaleString();
}

function datasetMeta(dataset) {
  return DATASET_META[dataset] || { label: dataset, color: "#334155" };
}

function serverModeMeta(serverMode) {
  return SERVER_MODE_META[serverMode] || { label: serverMode, color: "#475569" };
}

function runMeta(runState) {
  return RUN_STATE_META[runState] || { label: runState, color: "#64748b" };
}

function toolMeta(toolName) {
  return REPORT.tools.find((tool) => tool.name === toolName);
}

function taskMatches(task) {
  if (state.dataset !== "all" && task.dataset !== state.dataset) {
    return false;
  }
  if (state.serverMode !== "all" && task.server_mode !== state.serverMode) {
    return false;
  }
  if (state.status !== "all" && task.run.state !== state.status) {
    return false;
  }
  if (!state.search) {
    return true;
  }
  const haystack = [
    task.dataset,
    task.server_mode,
    task.case_id,
    task.prompt,
    task.source_path,
    task.category,
    task.server_names.join(" "),
    task.run.error_excerpt,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return haystack.includes(state.search);
}

function compareDefault(a, b) {
  const datasetDelta = (DATASET_ORDER[a.dataset] ?? 99) - (DATASET_ORDER[b.dataset] ?? 99);
  if (datasetDelta !== 0) {
    return datasetDelta;
  }
  const serverModeDelta = (SERVER_MODE_ORDER[a.server_mode] ?? 99) - (SERVER_MODE_ORDER[b.server_mode] ?? 99);
  if (serverModeDelta !== 0) {
    return serverModeDelta;
  }
  return a.case_id.localeCompare(b.case_id);
}

function compareSelectedSort(a, b) {
  switch (state.sort) {
    case "status": {
      const delta = (RUN_STATE_ORDER[a.run.state] ?? 99) - (RUN_STATE_ORDER[b.run.state] ?? 99);
      if (delta !== 0) {
        return delta;
      }
      break;
    }
    case "score": {
      const as = typeof a.run.score === "number" ? a.run.score : -1;
      const bs = typeof b.run.score === "number" ? b.run.score : -1;
      if (as !== bs) {
        return bs - as;
      }
      break;
    }
    case "updated": {
      const at = typeof a.run.timestamp === "number" ? a.run.timestamp : 0;
      const bt = typeof b.run.timestamp === "number" ? b.run.timestamp : 0;
      if (at !== bt) {
        return bt - at;
      }
      break;
    }
    case "tool-count": {
      if (a.server_count !== b.server_count) {
        return b.server_count - a.server_count;
      }
      break;
    }
    default:
      break;
  }
  return compareDefault(a, b);
}

function compareToolSort(a, b) {
  if (!state.toolSort) {
    return 0;
  }
  const av = a.server_names.includes(state.toolSort.tool) ? 1 : 0;
  const bv = b.server_names.includes(state.toolSort.tool) ? 1 : 0;
  if (av === bv) {
    return 0;
  }
  return state.toolSort.dir === "desc" ? bv - av : av - bv;
}

function getVisibleTasks() {
  return REPORT.tasks
    .filter(taskMatches)
    .slice()
    .sort((a, b) => compareToolSort(a, b) || compareSelectedSort(a, b));
}

function getVisibleTools(tasks) {
  const used = new Set();
  tasks.forEach((task) => task.server_names.forEach((tool) => used.add(tool)));
  return REPORT.tools.filter((tool) => used.has(tool.name));
}

function buildSortLabel(toolName) {
  if (!state.toolSort || state.toolSort.tool !== toolName) {
    return "";
  }
  return state.toolSort.dir === "desc" ? "on" : "off";
}

function toolSortHint(toolName) {
  if (!state.toolSort || state.toolSort.tool !== toolName) {
    return "";
  }
  return state.toolSort.dir === "desc" ? "DESC" : "ASC";
}

function buildHeader(tools) {
  const headerRow = document.getElementById("headerRow");
  const kindRow = document.getElementById("kindRow");
  headerRow.innerHTML = "";
  kindRow.innerHTML = "";

  const rootHeader = document.createElement("th");
  rootHeader.innerHTML = `
    <div class="eyebrow" style="color: var(--ink-dim); margin: 0 0 8px;">Task / Latest run</div>
    <div style="font-size: 20px; font-weight: 800; color: var(--navy);">Task x MCP Matrix</div>
    <div style="margin-top: 8px; color: var(--ink-dim); font-size: 12px; line-height: 1.45;">
      Click any MCP header to float matching tasks to the top. Filtered views hide columns that are unused by the current task subset.
    </div>
  `;
  headerRow.appendChild(rootHeader);

  const kindHeader = document.createElement("th");
  kindHeader.textContent = "Tool kind";
  kindRow.appendChild(kindHeader);

  for (const tool of tools) {
    const th = document.createElement("th");
    th.className = "tool-th" + (buildSortLabel(tool.name) ? " sorted" : "");
    th.title = `${tool.name} (${tool.task_count} task(s))\nKind: ${TOOL_KIND_META[tool.kind].description}`;
    th.innerHTML = `
      <span class="tool-count">${tool.task_count}</span>
      <span class="sort-pill">${toolSortHint(tool.name)}</span>
      <span class="tool-label" style="color: ${tool.color};">${escapeHtml(tool.name)}</span>
    `;
    th.addEventListener("click", () => {
      if (!state.toolSort || state.toolSort.tool !== tool.name) {
        state.toolSort = { tool: tool.name, dir: "desc" };
      } else if (state.toolSort.dir === "desc") {
        state.toolSort = { tool: tool.name, dir: "asc" };
      } else {
        state.toolSort = null;
      }
      render();
    });
    headerRow.appendChild(th);

    const kindTh = document.createElement("th");
    kindTh.title = `${TOOL_KIND_META[tool.kind].label}: ${TOOL_KIND_META[tool.kind].description}`;
    kindTh.innerHTML = `<span class="kind-chip" style="color:${tool.color};">${escapeHtml(tool.kind_short)}</span>`;
    kindRow.appendChild(kindTh);
  }
}

function buildTaskTitle(task) {
  const lines = [
    `${task.dataset}/${task.server_mode}/${task.case_id}`,
    `Server mode: ${serverModeMeta(task.server_mode).label}`,
    `Run state: ${runMeta(task.run.state).label}`,
    task.run.has_run ? `Runtime status: ${task.run.status}` : "Runtime status: no exported run found",
    task.run.has_run ? `Attempts discovered: ${task.run.attempts}` : null,
    typeof task.run.score === "number" ? `Score: ${formatScore(task.run.score)}` : null,
    task.run.start_time ? `Started: ${formatTimestamp(task.run.start_time)}` : null,
    task.run.end_time ? `Ended: ${formatTimestamp(task.run.end_time)}` : null,
    task.category ? `Category: ${task.category}` : null,
    `MCPs: ${task.server_names.join(", ")}`,
    `Source: ${task.source_path}`,
    task.run.path ? `Run artifact: ${task.run.path}` : null,
    task.run.error_excerpt ? `Error: ${task.run.error_excerpt}` : null,
    `Prompt: ${task.prompt}`,
  ];
  return lines.filter(Boolean).join("\\n");
}

function buildBody(tasks, tools) {
  const body = document.getElementById("matrixBody");
  body.innerHTML = "";

  if (!tasks.length) {
    const emptyRow = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = Math.max(2, tools.length + 1);
    cell.className = "empty-state";
    cell.textContent = "No tasks match the current filters.";
    emptyRow.appendChild(cell);
    body.appendChild(emptyRow);
    return;
  }

  for (const task of tasks) {
    const tr = document.createElement("tr");
    tr.className = `state-${task.run.state}`;

    const tdTask = document.createElement("td");
    tdTask.title = buildTaskTitle(task);
    const dataset = datasetMeta(task.dataset);
    const serverMode = serverModeMeta(task.server_mode);
    const run = runMeta(task.run.state);
    const scoreChip = typeof task.run.score === "number"
      ? `<span class="meta-chip">score ${formatScore(task.run.score)}</span>`
      : "";
    const attemptsChip = task.run.has_run
      ? `<span class="meta-chip">${task.run.attempts > 1 ? `attempts ${task.run.attempts}` : "1 attempt"}</span>`
      : `<span class="meta-chip">no export</span>`;
    const categoryChip = task.category ? `<span class="meta-chip">${escapeHtml(task.category)}</span>` : "";
    const serverModeChip = `<span class="meta-chip" style="color:${serverMode.color}; border-color:${serverMode.color}22;">${escapeHtml(serverMode.label)}</span>`;
    const errorLine = task.run.error_excerpt
      ? `<div class="task-error">${escapeHtml(compact(task.run.error_excerpt, 170))}</div>`
      : "";
    tdTask.innerHTML = `
      <div class="task-card">
        <div class="task-head">
          <span class="dataset-chip dataset-${task.dataset}">${escapeHtml(dataset.label)}</span>
          <span class="task-id">${escapeHtml(task.case_id)}</span>
        </div>
        <div class="task-subline">
          <span class="state-chip state-${task.run.state}">${escapeHtml(run.label)}</span>
          ${serverModeChip}
          ${attemptsChip}
          <span class="meta-chip">mcp ${task.server_count}</span>
          ${scoreChip}
          ${categoryChip}
        </div>
        <div class="task-prompt">${escapeHtml(compact(task.prompt, 190))}</div>
        <div class="task-tools">${escapeHtml(task.server_names.join(", "))}</div>
        ${errorLine}
      </div>
    `;
    tr.appendChild(tdTask);

    for (const tool of tools) {
      const td = document.createElement("td");
      td.className = "matrix-cell";
      if (task.server_names.includes(tool.name)) {
        td.classList.add("active", `kind-${tool.kind}`);
        td.title = `${task.dataset}/${task.server_mode}/${task.case_id}\\nTool: ${tool.name}\\nKind: ${TOOL_KIND_META[tool.kind].label}\\nRun: ${run.label}`;
        td.innerHTML = `<span class="cell-dot"></span>`;
      }
      tr.appendChild(td);
    }

    body.appendChild(tr);
  }
}

function updateStats(tasks, tools) {
  const statsBar = document.getElementById("statsBar");
  const counts = {
    total: REPORT.tasks.length,
    visible: tasks.length,
    visibleTools: tools.length,
    passed: tasks.filter((task) => task.run.state === "passed").length,
    evalFailed: tasks.filter((task) => task.run.state === "eval-failed").length,
    runError: tasks.filter((task) => task.run.state === "run-error").length,
    notRun: tasks.filter((task) => task.run.state === "not-run").length,
    api: tasks.filter((task) => task.server_mode === "api").length,
  };

  const cards = [
    ["Tasks", counts.total],
    ["Visible", counts.visible],
    ["API", counts.api],
    ["Tools", `${tools.length}/${REPORT.tools.length}`],
    ["Passed", counts.passed],
    ["Eval Failed", counts.evalFailed],
    ["Run Error", counts.runError],
    ["Not Run", counts.notRun],
  ];

  statsBar.innerHTML = cards
    .map(([label, value]) => `
      <div class="stat-card">
        <div class="stat-label">${escapeHtml(label)}</div>
        <div class="stat-value">${escapeHtml(value)}</div>
      </div>
    `)
    .join("");
}

function updateFooter(tasks, tools) {
  const footer = document.getElementById("footerNote");
  const sortNote = state.toolSort
    ? `Tool sort: ${state.toolSort.tool} (${state.toolSort.dir}).`
    : "Tool sort: none.";
  footer.textContent =
    `Showing ${tasks.length} task(s) and ${tools.length} visible MCP column(s). ${sortNote} ` +
    `Run artifacts are resolved from the latest timestamp among exported runs and loose result files under ${REPORT.workspace_root}.`;
}

function render() {
  const tasks = getVisibleTasks();
  const tools = getVisibleTools(tasks);
  buildHeader(tools);
  buildBody(tasks, tools);
  updateStats(tasks, tools);
  updateFooter(tasks, tools);
}

function bindControls() {
  document.getElementById("datasetFilter").addEventListener("change", (event) => {
    state.dataset = event.target.value;
    render();
  });
  document.getElementById("serverModeFilter").addEventListener("change", (event) => {
    state.serverMode = event.target.value;
    render();
  });
  document.getElementById("statusFilter").addEventListener("change", (event) => {
    state.status = event.target.value;
    render();
  });
  document.getElementById("sortMode").addEventListener("change", (event) => {
    state.sort = event.target.value;
    render();
  });
  document.getElementById("searchInput").addEventListener("input", (event) => {
    state.search = event.target.value.trim().toLowerCase();
    render();
  });
  document.getElementById("resetButton").addEventListener("click", () => {
    state.dataset = "all";
    state.serverMode = "all";
    state.status = "all";
    state.sort = "default";
    state.search = "";
    state.toolSort = null;
    document.getElementById("datasetFilter").value = "all";
    document.getElementById("serverModeFilter").value = "all";
    document.getElementById("statusFilter").value = "all";
    document.getElementById("sortMode").value = "default";
    document.getElementById("searchInput").value = "";
    render();
  });
}

document.getElementById("generatedAt").textContent = `Generated ${formatTimestamp(REPORT.generated_at)}`;
bindControls();
render();
</script>
</body>
</html>
"""


def build_html(report: dict[str, Any], *, title: str) -> str:
    report_json = SCRIPT_CLOSE_RE.sub(r'<\\/\\1', json.dumps(report, ensure_ascii=False))
    run_state_json = json.dumps(RUN_STATE_META, ensure_ascii=False)
    tool_kind_json = json.dumps(TOOL_KIND_META, ensure_ascii=False)
    dataset_json = json.dumps(DATASET_META, ensure_ascii=False)
    server_mode_json = json.dumps(SERVER_MODE_META, ensure_ascii=False)
    return (
        HTML_TEMPLATE
        .replace('__TITLE__', title)
        .replace('__REPORT_JSON__', report_json)
        .replace('__RUN_STATE_JSON__', run_state_json)
        .replace('__TOOL_KIND_JSON__', tool_kind_json)
        .replace('__DATASET_JSON__', dataset_json)
        .replace('__SERVER_MODE_JSON__', server_mode_json)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate a task x MCP matrix report for the MCP benchmark datasets.')
    parser.add_argument(
        '--output',
        type=Path,
        default=WORKSPACE_MCP_ROOT / 'task_mcp_matrix.html',
        help='Output HTML path.',
    )
    parser.add_argument(
        '--workspace-root',
        type=Path,
        default=WORKSPACE_MCP_ROOT,
        help='Workspace root containing exported MCP benchmark runs.',
    )
    parser.add_argument(
        '--datasets',
        default='toolathlon,atlas,universe',
        help='Comma-separated dataset names to include.',
    )
    parser.add_argument(
        '--server-modes',
        default='api',
        help='Comma-separated server modes to include.',
    )
    parser.add_argument(
        '--title',
        default='Nexus MCP Benchmark Matrix',
        help='HTML page title.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(',') if item.strip()]
    server_modes = [item.strip() for item in args.server_modes.split(',') if item.strip()]
    report = build_report_data(args.workspace_root, datasets, server_modes)
    html = build_html(report, title=args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding='utf-8')
    print(
        f'Wrote {args.output} with {report["summary"]["task_count"]} task(s), '
        f'{report["summary"]["tool_count"]} tool(s), and {report["summary"]["run_count"]} discovered run(s).'
    )


if __name__ == '__main__':
    main()
