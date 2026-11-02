#!/usr/bin/env python3
"""Taiji Analysis Benchmark — Evaluation Frontend.

A lightweight Flask app for browsing, comparing, and annotating benchmark results.

Usage:
    python3 -m nexus.extensions.tasks.taiji_analysis.eval_frontend.app \
        --results-dir /path/to/nexus_results \
        --port 8090

Features:
    - Dashboard: pass rate comparison across models, difficulty distribution
    - Case Browser: filter/search cases, view prediction vs ground truth
    - Multi-Model Compare: side-by-side comparison for a single case
    - Annotation: mark judge errors (false positive / false negative)
    - Hard Case Analysis: focus on all-fail items for judge calibration
"""

import argparse
import json
import re
import sqlite3

from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request


app = Flask(__name__, template_folder='templates', static_folder='static')

RESULTS_DIR: Path = Path('.')
ANNOTATIONS_FILE: Path = Path('annotations.json')

EPISODES_DIR = Path('/apdcephfs_gy6/share_304144884/hunyuan/debug_agent/taiji_replay/episodes')


def get_benchmark_versions() -> list[dict]:
    """Scan EPISODES_DIR for dataset_v*/ folders and build version list."""
    versions = []
    for d in sorted(EPISODES_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith('dataset_v'):
            continue
        parquet = d / 'dataset.parquet'
        dataset_json = d / 'dataset.json'
        if not parquet.exists() and not dataset_json.exists():
            continue
        m = re.match(r'dataset_v(\d+)', d.name)
        version = f'v{m.group(1)}' if m else d.name.replace('dataset_', '')
        if dataset_json.exists():
            with open(dataset_json) as f:
                cases = len(json.load(f))
        else:
            cases = 0
        mtime = datetime.fromtimestamp(d.stat().st_mtime).strftime('%Y-%m-%d')
        versions.append({
            'version': version,
            'date': mtime,
            'cases': cases,
            'description': d.name,
            'dataset': f'episodes/{d.name}/dataset.parquet',
        })
    return versions


EXCLUDE_EXPERIMENTS = {'hy3_results0709'}

ONLINE_GT_PARQUET = Path(
    '/apdcephfs_gy6/share_304144884/hunyuan/debug_agent/taiji_replay/episodes/dataset_v3_0720_218_onlineGT/dataset.parquet'
)
NEWRUN_GT_PARQUET = Path(
    '/apdcephfs_gy6/share_304144884/hunyuan/debug_agent/taiji_replay/episodes/dataset_v3_0720_218_newrunGT/dataset.parquet'
)
NEWRUN_GT_DATASET_JSON = Path(
    '/apdcephfs_gy6/share_304144884/hunyuan/debug_agent/taiji_replay/episodes/dataset_v3_0720_218_newrunGT/dataset.json'
)

_online_gt_cache: dict[str, str] | None = None
_newrun_gt_cache: dict[str, str] | None = None
_case_meta_cache: dict[str, dict] | None = None


def _load_gt_parquet(path: Path) -> dict[str, str]:
    """Load GT answers from parquet keyed by instance_id (== run_id)."""
    result = {}
    if path.exists():
        import pandas as pd

        df = pd.read_parquet(path, columns=['instance_id', 'answer'])
        for _, row in df.iterrows():
            result[str(row['instance_id'])] = row['answer'] or ''
    return result


def get_online_gt() -> dict[str, str]:
    """Load online GT answers keyed by instance_id."""
    global _online_gt_cache
    if _online_gt_cache is None:
        _online_gt_cache = _load_gt_parquet(ONLINE_GT_PARQUET)
    return _online_gt_cache


def get_newrun_gt() -> dict[str, str]:
    """Load newrun GT answers keyed by instance_id."""
    global _newrun_gt_cache
    if _newrun_gt_cache is None:
        _newrun_gt_cache = _load_gt_parquet(NEWRUN_GT_PARQUET)
    return _newrun_gt_cache


def get_case_meta() -> dict[str, dict]:
    """Load question/url from newrun GT dataset.json, keyed by run_id."""
    global _case_meta_cache
    if _case_meta_cache is None:
        _case_meta_cache = {}
        if NEWRUN_GT_DATASET_JSON.exists():
            with open(NEWRUN_GT_DATASET_JSON) as f:
                for item in json.load(f):
                    rid = str(item['run_id'])
                    _case_meta_cache[rid] = {
                        'question': item.get('query', ''),
                        'url': item.get('url', ''),
                        'sender': item.get('sender', ''),
                    }
    return _case_meta_cache


def get_experiments() -> list[dict]:
    """List all experiment folders (each has results.jsonl + metrics.json)."""
    experiments = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if d.name in EXCLUDE_EXPERIMENTS:
            continue
        metrics_file = d / 'metrics.json'
        results_file = d / 'results.jsonl'
        if metrics_file.exists() and results_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            experiments.append({
                'name': d.name,
                'path': str(d),
                'metrics': metrics,
            })
        # Check for sub-folders (like results_filter0714/opus4.8_xhigh/)
        for sub in sorted(d.iterdir()):
            if not sub.is_dir():
                continue
            sub_metrics = sub / 'metrics.json'
            sub_results = sub / 'results.jsonl'
            if sub_metrics.exists() and sub_results.exists():
                with open(sub_metrics) as f:
                    metrics = json.load(f)
                experiments.append({
                    'name': f'{d.name}/{sub.name}',
                    'path': str(sub),
                    'metrics': metrics,
                })
    return experiments


def load_results(exp_path: str) -> list[dict]:
    """Load all results from an experiment's results.jsonl."""
    results = []
    with open(Path(exp_path) / 'results.jsonl') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def _find_db(exp_path: Path) -> Path | None:
    """Find the taiji_bench DB file in an experiment directory."""
    dbs = list(exp_path.glob('taiji_bench*.db'))
    return dbs[0] if dbs else None


def load_case_detail(exp_path: str, run_id: str) -> dict | None:
    """Load full case detail from DB if available, else from jsonl."""
    db_path = _find_db(Path(exp_path))
    if db_path:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
        conn.close()
        if row:
            return {
                'run_id': row['run_id'],
                'status': row['status'],
                'start_time': row['start_time'],
                'end_time': row['end_time'],
                'error': row['error'],
                'config': json.loads(row['config']) if row['config'] else {},
                'task_output': json.loads(row['task_output']) if row['task_output'] else {},
                'agent_output': json.loads(row['agent_output']) if row['agent_output'] else {},
                'metadata': json.loads(row['metadata']) if row['metadata'] else {},
            }
    # Fallback to jsonl
    results = load_results(exp_path)
    for r in results:
        if str(r['run_id']) == str(run_id):
            return r
    return None


def load_annotations() -> dict:
    """Load annotations from JSON file."""
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE) as f:
            return json.load(f)
    return {}


def save_annotations(annotations: dict) -> None:
    """Save annotations to JSON file."""
    ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)


# ─── Routes ──────────────────────────────────────────────────────────────────


PRESET_USERS = ['lunahyhuang', 'nickklau']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/users')
def api_users():
    return jsonify(PRESET_USERS)


@app.route('/api/versions')
def api_versions():
    return jsonify(get_benchmark_versions())


@app.route('/api/experiments')
def api_experiments():
    return jsonify(get_experiments())


@app.route('/api/all_cases')
def api_all_cases():
    """Return all benchmark cases with content info and scores across experiments."""
    experiments = get_experiments()
    if not experiments:
        return jsonify([])

    # Load scores from all experiments
    exp_scores: dict[str, dict[str, float]] = {}
    for exp in experiments:
        scores = {}
        results = load_results(exp['path'])
        for r in results:
            score = r.get('task_output', {}).get('score', 0) if r.get('task_output') else 0
            scores[str(r['run_id'])] = score
        exp_scores[exp['name']] = scores

    # Use first experiment to get case list
    first_exp = experiments[0]
    results = load_results(first_exp['path'])

    # Load question + url from dataset.json
    case_meta = get_case_meta()

    # Determine version membership for each run_id
    # v0 = all 978 (original), v1 = 390 (current set), v2 = 186 (filter0714 subset)
    v2_ids: set[str] = set()
    filter_dir = RESULTS_DIR / 'results_filter0714'
    if filter_dir.exists():
        for sub in filter_dir.iterdir():
            sub_results = sub / 'results.jsonl'
            if sub_results.exists():
                with open(sub_results) as f:
                    for line in f:
                        v2_ids.add(str(json.loads(line)['run_id']))
                break

    # Load exclude reasons for the 14 cases removed from v2
    # (13 inaccessible + 1 bad GT). Stored in exclude_reasons.json if available.
    exclude_reasons: dict[str, str] = {}
    exclude_file = RESULTS_DIR / 'exclude_reasons.json'
    if exclude_file.exists():
        with open(exclude_file) as f:
            exclude_reasons = json.load(f)

    cases = []
    for r in results:
        rid = str(r['run_id'])
        meta = r.get('metadata', {})
        task_output = r.get('task_output', {}) or {}
        reference = task_output.get('reference', '')
        prediction = task_output.get('prediction', '')
        cm = case_meta.get(rid, {})
        question = cm.get('question', '')
        versions = ['v0', 'v1']
        if rid in v2_ids:
            versions.append('v2')
        case_data = {
            'run_id': rid,
            'inst_id': meta.get('inst_id', ''),
            'sender': cm.get('sender', '') or meta.get('sender', ''),
            'question': question[:200],
            'url': cm.get('url', ''),
            'reference_summary': _extract_summary(reference),
            'prediction_summary': _extract_summary(prediction),
            'versions': versions,
            'scores': {name: scores.get(rid, None) for name, scores in exp_scores.items()},
        }
        if rid in exclude_reasons:
            case_data['exclude_reason'] = exclude_reasons[rid]
        cases.append(case_data)

    return jsonify({'cases': cases, 'experiments': [e['name'] for e in experiments]})


@app.route('/api/results/<path:exp_name>')
def api_results(exp_name):
    exp_path = RESULTS_DIR / exp_name
    if not exp_path.exists():
        return jsonify({'error': 'not found'}), 404
    results = load_results(str(exp_path))
    return jsonify(results)


@app.route('/api/case/<path:exp_name>/<run_id>')
def api_case(exp_name, run_id):
    exp_path = RESULTS_DIR / exp_name
    if not exp_path.exists():
        return jsonify({'error': 'not found'}), 404
    detail = load_case_detail(str(exp_path), run_id)
    if not detail:
        return jsonify({'error': 'case not found'}), 404
    return jsonify(detail)


@app.route('/api/compare/<run_id>')
def api_compare(run_id):
    """Get results for a single run_id across all experiments."""
    experiments = get_experiments()
    results = {}
    for exp in experiments:
        exp_results = load_results(exp['path'])
        for r in exp_results:
            if str(r['run_id']) == str(run_id):
                results[exp['name']] = r
                break
    cm = get_case_meta().get(str(run_id), {})
    return jsonify({'results': results, 'question': cm.get('question', '')})


@app.route('/api/annotations', methods=['GET'])
def api_get_annotations():
    return jsonify(load_annotations())


@app.route('/api/annotations', methods=['POST'])
def api_save_annotation():
    data = request.json
    annotations = load_annotations()
    run_id = data.get('run_id')
    annotator = data.get('annotator', '')
    if not run_id:
        return jsonify({'error': 'run_id required'}), 400
    if not annotator:
        return jsonify({'error': 'annotator required — please select a user first'}), 403
    existing = annotations.get(run_id, {})
    existing.update({
        'label': data.get('label', existing.get('label', '')),
        'note': data.get('note', existing.get('note', '')),
        'annotator': annotator,
        'timestamp': data.get('timestamp', ''),
    })
    if 'gt_source' in data:
        existing['gt_source'] = data['gt_source']
    if 'benchmark_quality' in data:
        existing['benchmark_quality'] = data['benchmark_quality']
    annotations[run_id] = existing
    save_annotations(annotations)
    return jsonify({'ok': True})


@app.route('/api/model_feedback', methods=['POST'])
def api_save_model_feedback():
    """Save per-model feedback (score 0/1 + note) for a case."""
    data = request.json
    run_id = data.get('run_id')
    model = data.get('model')
    annotator = data.get('annotator', '')
    if not run_id or not model:
        return jsonify({'error': 'run_id and model required'}), 400
    if not annotator:
        return jsonify({'error': 'annotator required'}), 403

    annotations = load_annotations()
    existing = annotations.setdefault(run_id, {})
    model_fb = existing.setdefault('model_feedback', {})
    model_fb[model] = {
        'score': data.get('score'),
        'note': data.get('note', ''),
        'judge_comment': data.get('judge_comment', ''),
        'annotator': annotator,
        'timestamp': data.get('timestamp', ''),
    }
    save_annotations(annotations)
    return jsonify({'ok': True})


@app.route('/api/hard_cases')
def api_hard_cases():
    """Get all-fail cases with prediction vs reference summaries."""
    experiments = get_experiments()
    # Use first experiment with 'filter' in name, or first available
    filter_exps = [e for e in experiments if 'filter' in e['name']]
    if not filter_exps:
        filter_exps = experiments[:1]
    if not filter_exps:
        return jsonify([])

    # Load all experiments' scores
    all_scores = {}
    for exp in experiments:
        if 'filter' in exp['name'] or exp['name'].endswith('results0713'):
            scores = {}
            results = load_results(exp['path'])
            for r in results:
                score = r.get('task_output', {}).get('score', 0) if r.get('task_output') else 0
                scores[str(r['run_id'])] = score
            all_scores[exp['name']] = scores

    if not all_scores:
        return jsonify([])

    # Find common IDs where all models scored 0
    common_ids = set.intersection(*[set(s.keys()) for s in all_scores.values()])
    hard_cases = []
    for rid in sorted(common_ids):
        if not any(all_scores[m].get(rid, 0) >= 1.0 for m in all_scores):
            # Load detail from first experiment
            detail = load_case_detail(filter_exps[0]['path'], rid)
            if detail:
                task_output = detail.get('task_output', {})
                config = detail.get('config', {})
                task_config = config.get('task_config', {})
                hard_cases.append({
                    'run_id': rid,
                    'inst_id': detail.get('metadata', {}).get('inst_id', task_config.get('inst_id', '')),
                    'question': task_config.get('question', '')[:200],
                    'prediction_summary': _extract_summary(task_output.get('prediction', '')),
                    'reference_summary': _extract_summary(
                        task_output.get('reference', '') or task_config.get('answer', '')
                    ),
                    'score': task_output.get('score', 0),
                })
    return jsonify(hard_cases)


CASE_TAG_DIMENSIONS = {
    'root_cause': [
        'OOM',
        'NCCL/hang',
        'node_fault',
        'code_bug',
        'config_error',
        'disk_full',
        'performance',
        'timeout',
        'kill_rule',
        'other',
    ],
    'difficulty': ['easy', 'medium', 'hard'],
    'task_type': ['ppo', 'sft', 'eval', 'ci', 'other'],
}

TAGS_FILE: Path = Path('tags.json')


def load_tags() -> dict:
    """Load case tags from JSON file."""
    if TAGS_FILE.exists():
        with open(TAGS_FILE) as f:
            return json.load(f)
    return {}


def save_tags(tags: dict) -> None:
    """Save case tags to JSON file."""
    TAGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TAGS_FILE, 'w') as f:
        json.dump(tags, f, indent=2, ensure_ascii=False)


@app.route('/api/tags', methods=['GET'])
def api_get_tags():
    return jsonify({'dimensions': CASE_TAG_DIMENSIONS, 'tags': load_tags()})


@app.route('/api/tags', methods=['POST'])
def api_save_tag():
    data = request.json
    run_id = data.get('run_id')
    if not run_id:
        return jsonify({'error': 'run_id required'}), 400
    tags = load_tags()
    if run_id not in tags:
        tags[run_id] = {}
    dimension = data.get('dimension', '')
    value = data.get('value', '')
    if dimension and value:
        tags[run_id][dimension] = value
    elif dimension and not value:
        tags[run_id].pop(dimension, None)
    save_tags(tags)
    return jsonify({'ok': True})


@app.route('/api/stats/by_tag')
def api_stats_by_tag():
    """Compute pass rate broken down by tag dimensions, across all experiments."""
    experiments = get_experiments()
    tags = load_tags()

    if not tags:
        return jsonify({'message': 'No tags yet. Tag some cases first.', 'dimensions': {}})

    # Load scores per experiment
    exp_scores = {}
    for exp in experiments:
        scores = {}
        results = load_results(exp['path'])
        for r in results:
            score = r.get('task_output', {}).get('score', 0) if r.get('task_output') else 0
            scores[str(r['run_id'])] = score
        exp_scores[exp['name']] = scores

    # For each dimension, group by tag value and compute pass rate per experiment
    result = {}
    for dim in CASE_TAG_DIMENSIONS:
        dim_stats = {}
        # Group run_ids by tag value
        value_groups: dict[str, list[str]] = {}
        for run_id, case_tags in tags.items():
            val = case_tags.get(dim)
            if val:
                value_groups.setdefault(val, []).append(run_id)

        if not value_groups:
            continue

        for val, run_ids in sorted(value_groups.items()):
            exp_results: dict[str, dict] = {}
            for exp_name, scores in exp_scores.items():
                matched = [scores[rid] for rid in run_ids if rid in scores]
                if matched:
                    pass_rate = sum(1 for s in matched if s >= 1.0) / len(matched)
                    exp_results[exp_name] = {
                        'pass_rate': round(pass_rate, 4),
                        'n': len(matched),
                    }
            dim_stats[val] = {'count': len(run_ids), 'experiments': exp_results}

        result[dim] = dim_stats

    return jsonify({'dimensions': result})


@app.route('/api/judge_reasoning')
def api_judge_reasoning():
    """Return judge reasoning for all experiments."""
    reasoning_file = RESULTS_DIR / 'judge_reasoning.json'
    if reasoning_file.exists():
        with open(reasoning_file) as f:
            return jsonify(json.load(f))
    return jsonify({})


@app.route('/api/online_gt/<run_id>')
def api_online_gt(run_id):
    """Return the online GT answer for a given run_id."""
    gt = get_online_gt()
    answer = gt.get(str(run_id), '')
    return jsonify({'run_id': run_id, 'answer': answer})


@app.route('/api/newrun_gt/<run_id>')
def api_newrun_gt(run_id):
    """Return the newrun GT answer for a given run_id."""
    gt = get_newrun_gt()
    answer = gt.get(str(run_id), '')
    return jsonify({'run_id': run_id, 'answer': answer})


@app.route('/api/exclude_reasons', methods=['GET'])
def api_get_exclude_reasons():
    exclude_file = RESULTS_DIR / 'exclude_reasons.json'
    if exclude_file.exists():
        with open(exclude_file) as f:
            return jsonify(json.load(f))
    return jsonify({})


@app.route('/api/trace/<path:exp_name>/<run_id>')
def api_trace(exp_name, run_id):
    """Return trace spans for a run from the DB."""
    exp_path = RESULTS_DIR / exp_name
    db_path = _find_db(exp_path)
    if not db_path:
        return jsonify({'error': 'no DB found'}), 404

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        'SELECT event_type, span_id, event_data, time_unix_nano FROM run_trace_events WHERE run_id = ? ORDER BY time_unix_nano',
        (run_id,),
    ).fetchall()
    conn.close()

    # Build span objects from events
    spans: dict[str, dict] = {}
    for row in rows:
        data = json.loads(row['event_data']) if row['event_data'] else {}
        span_id = data.get('span_id') or row['span_id']
        if span_id not in spans:
            spans[span_id] = {'span_id': span_id, 'events': []}
        evt = {'type': row['event_type'], 'time': row['time_unix_nano'], 'data': data}
        spans[span_id]['events'].append(evt)

    # Extract structured info per span
    result = []
    for span_id, span in spans.items():
        info: dict = {
            'span_id': span_id,
            'name': '',
            'parent_span_id': '',
            'inputs': None,
            'outputs': None,
            'start_time': 0,
            'end_time': 0,
        }
        for evt in span['events']:
            d = evt['data']
            if evt['type'] == 'start':
                info['name'] = d.get('name', '')
                info['parent_span_id'] = d.get('parent_span_id', '')
                info['start_time'] = evt['time']
                attrs = d.get('attributes') or {}
                if 'inputs' in attrs:
                    info['inputs'] = attrs['inputs']
            elif evt['type'] == 'update':
                attrs = d.get('attributes') or {}
                if 'outputs' in attrs:
                    info['outputs'] = attrs['outputs']
            elif evt['type'] == 'end':
                info['end_time'] = evt['time']
        result.append(info)

    # Sort by start_time
    result.sort(key=lambda s: s['start_time'])
    return jsonify(result)


@app.route('/api/exclude_reasons', methods=['POST'])
def api_save_exclude_reason():
    data = request.json
    run_id = data.get('run_id')
    reason = data.get('reason', '')
    if not run_id:
        return jsonify({'error': 'run_id required'}), 400
    exclude_file = RESULTS_DIR / 'exclude_reasons.json'
    reasons = {}
    if exclude_file.exists():
        with open(exclude_file) as f:
            reasons = json.load(f)
    if reason:
        reasons[run_id] = reason
    else:
        reasons.pop(run_id, None)
    with open(exclude_file, 'w') as f:
        json.dump(reasons, f, indent=2, ensure_ascii=False)
    return jsonify({'ok': True})


NEWRUN_DATASET_JSON = Path(
    '/apdcephfs_gy6/share_304144884/hunyuan/debug_agent/taiji_replay/episodes/dataset_v1_newrun_0720_390/dataset.json'
)


@app.route('/api/gt_selection_stats')
def api_gt_selection_stats():
    """Return GT selection progress stats."""
    annotations = load_annotations()
    newrun_gt = get_newrun_gt()
    total = len(newrun_gt)
    selected = sum(1 for a in annotations.values() if a.get('gt_source'))
    by_source: dict[str, int] = {}
    for a in annotations.values():
        src = a.get('gt_source', '')
        if src:
            by_source[src] = by_source.get(src, 0) + 1
    return jsonify({
        'total': total,
        'selected': selected,
        'remaining': total - selected,
        'by_source': by_source,
    })


@app.route('/api/export_dataset', methods=['POST'])
def api_export_dataset():
    """Export a new dataset with GT replaced based on annotations.

    For each case:
    - If gt_source == 'newrun': keep the newrun answer (default)
    - If gt_source == 'online': use the original online GT from task_output.reference
    - If gt_source == '<exp_name>': use prediction from that experiment
    - If no gt_source set: keep newrun answer as default
    """
    from datetime import datetime

    import pandas as pd

    annotations = load_annotations()
    get_newrun_gt()
    experiments = get_experiments()

    online_gt = get_online_gt()

    # Load predictions from all experiments
    exp_predictions: dict[str, dict[str, str]] = {}
    for exp in experiments:
        preds = {}
        results = load_results(exp['path'])
        for r in results:
            pred = (r.get('task_output') or {}).get('prediction', '')
            if pred:
                preds[str(r['run_id'])] = pred
        exp_predictions[exp['name']] = preds

    # Load base dataset json
    if not NEWRUN_DATASET_JSON.exists():
        return jsonify({'error': f'Base dataset not found: {NEWRUN_DATASET_JSON}'}), 500

    with open(NEWRUN_DATASET_JSON) as f:
        dataset = json.load(f)

    # Apply GT selections
    replaced = 0
    for record in dataset:
        rid = str(record.get('instance_id', ''))
        ann = annotations.get(rid, {})
        gt_source = ann.get('gt_source', '')

        if not gt_source or gt_source == 'newrun':
            continue
        if gt_source == 'online':
            new_answer = online_gt.get(rid, '')
        else:
            new_answer = exp_predictions.get(gt_source, {}).get(rid, '')

        if new_answer:
            record['answer'] = new_answer
            record['historical_report'] = new_answer
            replaced += 1

    # Write output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = (
        Path('/apdcephfs_gy6/share_304144884/hunyuan/debug_agent/taiji_replay/episodes')
        / f'dataset_v1_curated_{timestamp}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write json
    json_path = output_dir / 'dataset.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Write parquet
    df = pd.DataFrame(dataset)
    parquet_path = output_dir / 'dataset.parquet'
    df.to_parquet(parquet_path, index=False)

    return jsonify({
        'ok': True,
        'output_dir': str(output_dir),
        'total': len(dataset),
        'replaced': replaced,
        'kept_newrun': len(dataset) - replaced,
    })


def _extract_summary(text: str) -> str:
    """Extract one-line summary from a report."""
    for line in text.split('\n'):
        if '一句话总结' in line or '\U0001f4a1' in line:
            return line.strip()[:300]
    # Fallback: first non-empty line
    for line in text.split('\n'):
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('---'):
            return line[:300]
    return text[:200]


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    global RESULTS_DIR, ANNOTATIONS_FILE, TAGS_FILE

    parser = argparse.ArgumentParser(description='Taiji Analysis Benchmark Eval Frontend')
    parser.add_argument(
        '--results-dir',
        default='/apdcephfs_gy6/share_304144884/hunyuan/debug_agent/taiji_replay/nexus_results',
        help='Directory containing experiment result folders',
    )
    parser.add_argument('--port', type=int, default=8090, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    args = parser.parse_args()

    RESULTS_DIR = Path(args.results_dir)
    ANNOTATIONS_FILE = RESULTS_DIR / 'annotations.json'
    TAGS_FILE = RESULTS_DIR / 'tags.json'

    print(f'Results dir: {RESULTS_DIR}')
    print(f'Annotations: {ANNOTATIONS_FILE}')
    print(f'Starting server on {args.host}:{args.port}')

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == '__main__':
    main()
