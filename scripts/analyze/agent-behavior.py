"""Behaviour of the agents in the v6.3 board cells, from the run records (results/OPEN_ITEMS.md, 11-26).

For every board cell (the attempt the board shows) of the visible models: time used and stop reason; requests, output
and reasoning tokens and the share of the wall time spent waiting for the model (ATR records); the harness's shell
commands by kind (compile, local check, timing, reading the task files, test submissions) and how many failed; file
edits; the submission's languages and lines; for iterative cells the public feedback trajectory; and the cell's outcome
class (format or build failure, resource failure, valid below the weak anchor, scored).

Usage: agent-behavior.py   -> var/raw/agent-behavior.json and per-model summaries on stdout
"""

import io
import json
import re
import statistics
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/site'))
STORE = ROOT / 'var/stores/store-v63'
TAGS = {'gpt-6.2-sol': 'sol62', 'gpt-6-astra': 'astra', 'gpt-6.1-sol': 'sol', 'claude-opus-5-5': 'opus', 'hy4': 'hy4',
        'qwen4-max': 'qwen4', 'qwen3.8-max-0902': 'qwen', 'minimax-m3.1-flash': 'minimax'}
TASKS = {'conic': ('conic', 'v3'), 'exponential-v3': ('exponential', 'v3'), 'galaxy-gravity-v4': ('galaxy-gravity', 'v4'),
         'polarization-demosaicking-v3': ('polarization-demosaicking', 'v3'), 'trajectory-covariance-v2': ('trajectory-covariance', 'v2'),
         'species-tree-v3': ('species-tree', 'v3'), 'ihvp-sequence-v4': ('ihvp-sequence', 'v4'),
         'blackbox-cubature-v2': ('blackbox-cubature', 'v2'), 'continuum-estimation': ('continuum-estimation', 'v1'),
         'fluorescence-deconvolution': ('fluorescence-deconvolution', 'v1')}
SETTING = {'单轮': 'single', '迭代': 'iterative'}
KINDS = [('submit_test', r'submit\s+--test'), ('submit_status', r'submit\s+--status'), ('submit_final', r'submit\s+--final'),
         ('compile', r'\b(gcc|g\+\+|clang\+?\+?|c\+\+|make|cmake|cargo|rustc|gfortran|nvcc|ninja)\b|build\.sh'),
         ('local_check', r'check\w*\.py|self_?check|run_tests|pytest|verify|validat|evaluat|score\w*\.py|compare|rmse|error'),
         ('timing', r'(^|[\s;(&|])(/usr/bin/)?time\s|perf\s+stat|hyperfine|timeit|perf_counter'),
         ('read_task', r'\b(cat|head|tail|sed|less|more|grep|rg|ls)\b.*(problem\.md|README|API\.md|INSTRUCTION|RULES|statement|samples|public-dev)')]
LANG = {'.c': 'C/C++', '.cc': 'C/C++', '.cpp': 'C/C++', '.cxx': 'C/C++', '.h': 'C/C++', '.hpp': 'C/C++', '.py': 'Python',
        '.rs': 'Rust', '.f90': 'Fortran', '.f': 'Fortran', '.jl': 'Julia', '.go': 'Go', '.sh': 'shell'}


def commands_of(cli: Path) -> tuple[list[tuple[str, bool]], int]:
    """(command, failed) per shell call and the number of file edits, from a Codex or Claude Code event log."""
    commands, edits, pending = [], 0, {}
    for line in cli.open(errors='replace'):
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        item = event.get('item') or {}
        if event.get('type') == 'item.completed' and item.get('type') == 'command_execution':      # Codex
            commands.append((str(item.get('command') or ''), item.get('exit_code') not in (0, None)))
        elif event.get('type') == 'item.completed' and item.get('type') == 'file_change':
            edits += 1
        message = event.get('message') if isinstance(event.get('message'), dict) else {}
        for block in message.get('content') if isinstance(message.get('content'), list) else []:   # Claude Code
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'tool_use':
                if block.get('name') == 'Bash':
                    pending[block.get('id')] = str((block.get('input') or {}).get('command') or '')
                elif block.get('name') in ('Write', 'Edit', 'MultiEdit', 'NotebookEdit'):
                    edits += 1
            elif block.get('type') == 'tool_result' and block.get('tool_use_id') in pending:
                commands.append((pending.pop(block['tool_use_id']), bool(block.get('is_error'))))
    commands += [(c, False) for c in pending.values()]
    return commands, edits


def submission_of(submission: str | None) -> dict:
    path = STORE / 'submissions' / str(submission) / 'source.tar'
    if not submission or not path.exists():
        return {}
    lines, files = {}, 0
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            suffix = Path(member.name).suffix.lower()
            if not member.isfile() or suffix not in LANG or member.size > 5_000_000:
                continue
            files += 1
            text = archive.extractfile(member).read().decode(errors='replace')
            lines[LANG[suffix]] = lines.get(LANG[suffix], 0) + sum(1 for l in io.StringIO(text) if l.strip())
    main = max((k for k in lines if k != 'shell'), key=lines.get, default=None)
    return {'files': files, 'lines': lines, 'main_language': main, 'total_lines': sum(lines.values())}


def outcome(row: dict) -> str:
    raw, score = str(row.get('raw') or ''), row.get('score')
    if raw.startswith('无效提交') or raw.startswith('未测'):
        if re.search(r'README|run\.sh|build|invalid_source|nonzero_exit|构建|提交检查|required', raw):
            return 'format_or_build'
        if re.search(r'timeout|超时|cpu|CPU|内存|memory|OOM|limit|超限', raw):
            return 'resource'
        return 'invalid_other' if raw.startswith('无效提交') else 'unmeasured'
    if score is None:
        return 'unmeasured'
    return 'below_weak' if score == 0 else 'scored'


def main() -> None:
    cells = []
    for task, (family, version) in TASKS.items():
        meta = json.loads((ROOT / 'tasks' / family / 'meta.json').read_text())
        group = next(g for g in meta['result_groups'] if g.get('protocol') == 'v6.3' and g.get('version') == version)
        for row in group['results']:
            if row['model'] not in TAGS:
                continue
            setting, n = SETTING[row['setting']], row.get('attempt', 1)
            log = ROOT / f'var/stores/store/reruns/v63-{TAGS[row["model"]]}-{setting}-{task}-{n}.log'
            finished = [json.loads(l) for l in log.read_text().splitlines() if 'develop_finished' in l] if log.exists() else []
            if not finished:
                continue
            run = STORE / 'dev-runs' / finished[-1]['run']
            result = json.loads((run / 'develop_result.json').read_text())
            responses = [json.loads(l) for f in run.glob('atr/*/events.jsonl') for l in f.open() if '"kind": "response"' in l]
            usage = [r.get('usage') or {} for r in responses]
            commands, edits = commands_of(run / 'cli.log') if (run / 'cli.log').exists() else ([], 0)
            kinds = {k: sum(1 for c, _ in commands if re.search(p, c)) for k, p in KINDS}
            tests = [e for e in result.get('evaluations') or [] if e.get('kind') == 'test']
            public = [e.get('score') for e in tests if e.get('score') is not None]
            seconds = result.get('development_seconds') or 0
            model_s = sum(r.get('duration_ms') or 0 for r in responses) / 1000
            output = sum(u.get('output', 0) for u in usage)
            cells.append({
                'task': task, 'model': row['model'], 'setting': setting, 'attempt': n, 'score': row.get('score'),
                'outcome': outcome(row), 'raw': row.get('raw'), 'minutes': seconds / 60, 'stop': result.get('agent_stop_reason'),
                'requests': len(responses), 'output_tokens': output, 'reasoning_tokens': sum(u.get('reasoning', 0) for u in usage),
                'thinking_chars': sum(r.get('thinking_chars') or 0 for r in responses),
                'model_share': model_s / seconds if seconds else None, 'tokens_per_s': output / model_s if model_s else None,
                'commands': len(commands), 'failed_commands': sum(1 for _, f in commands if f), 'edits': edits, **kinds,
                'tests': len(tests), 'first_test_minute': tests[0]['requested_at_seconds'] / 60 if tests else None,
                'public_first': public[0] if public else None, 'public_best': max(public) if public else None,
                'public_last': public[-1] if public else None,
                'submission': submission_of(row.get('submission'))})
    out = ROOT / 'var/raw/agent-behavior.json'
    out.write_text(json.dumps(cells, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'cells': len(cells), 'out': str(out.relative_to(ROOT))}))


if __name__ == '__main__':
    main()
