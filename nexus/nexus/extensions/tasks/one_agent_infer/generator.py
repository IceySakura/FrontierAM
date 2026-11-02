"""YB OneAgentBench batch generator for nexus run-batch.

Generates BatchItem configs from benchmark JSONL files.

Key feature: **multi-turn session support**.
Records sharing the same `session_id` are grouped into a single BatchItem
with a `turns` list.  The task executes turns sequentially within one sandbox,
passing history (query + answer only) between turns.

Single-turn records (session with 1 turn, or no session_id) produce a
BatchItem with `turns` containing one entry — fully backward compatible.

Environment variables:
    YB_BENCH_PATH: Path to benchmark root directory (containing the jsonl and files/)
    YB_BENCH_CONFIG: Path to base config YAML (optional, uses built-in default)
"""

from __future__ import annotations

import json
import logging
import os
import re

from collections import OrderedDict
from pathlib import Path

from omegaconf import OmegaConf

from nexus import BatchItem


logger = logging.getLogger(__name__)


def _benchmark_root(input_path: str | None = None) -> Path:
    """Resolve benchmark root directory."""
    if input_path:
        return Path(input_path)
    env = os.getenv('YB_BENCH_PATH')
    if env:
        return Path(env)
    candidates = [
        Path(__file__).resolve().parents[4]
        / 'yb-agent-openbench'
        / 'benchmarks'
        / 'releases'
        / 'YB_OneAgentBench_V0.1',
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError('Benchmark root not found. Set YB_BENCH_PATH or pass --input-path.')


def _split_time_and_tz(time_str: str) -> tuple[str, str]:
    """Split a bench ``time`` string into ``(date_part, tz_name)``.

    Bench records store time with the timezone already embedded, e.g.
    ``'2026-03-12 Thursday (Asia/Shanghai, UTC+8)'``. ``prompt_builder`` expects
    a pure date in ``cur_date`` and appends ' (<tz>, <utc_offset>)' on its own,
    so forwarding the raw string would render the timezone twice.

    Returns the leading date portion and the IANA timezone name (first token
    inside the trailing parentheses). When no timezone parenthetical is present,
    returns ``(time_str, '')`` unchanged.
    """
    if not time_str:
        return '', ''
    m = re.search(r'\s*\(([^,)]+)(?:,[^)]*)?\)\s*$', time_str)
    if not m:
        return time_str.strip(), ''
    date_part = time_str[: m.start()].strip()
    tz_name = m.group(1).strip()
    return date_part, tz_name


def _config_path(base_config_path: str | Path | None = None) -> Path:
    """Resolve config YAML path."""
    if base_config_path:
        p = Path(base_config_path)
        if p.exists():
            return p
        raise FileNotFoundError(f'Config not found: {p}')
    env = os.getenv('YB_BENCH_CONFIG')
    if env:
        return Path(env)
    default = Path(__file__).parent / 'config' / 'base.yaml'
    if default.exists():
        return default
    raise FileNotFoundError('Config not found. Set YB_BENCH_CONFIG or create config/base.yaml.')


def _load_base_config(path: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


def _load_benchmark(root: Path) -> list[dict]:
    """Load benchmark JSONL file."""
    env_jsonl = os.getenv('YB_BENCH_JSONL')
    jsonl_path = Path(env_jsonl) if env_jsonl else root / 'YB_OneAgentBench_V0.1_benchmark.jsonl'

    if not jsonl_path.exists():
        raise FileNotFoundError(f'Benchmark file not found: {jsonl_path}')

    records = []
    with jsonl_path.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSONL line {idx + 1}: {exc}') from exc
            records.append(data)
    return records


def _group_by_session(records: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group records by session_id, preserving order of first appearance.

    Returns list of (session_key, [records sorted by turn]).
    Records without session_id use task_id as session_key.
    """
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for r in records:
        key = r.get('session_id') or r.get('task_id', '')
        groups.setdefault(key, []).append(r)

    result = []
    for key, items in groups.items():
        items.sort(key=lambda x: x.get('turn') or 1)
        result.append((key, items))
    return result


def _record_to_turn(record: dict) -> dict:
    """Convert a benchmark record to a turn dict for task_config.turns."""
    input_file_list = record.get('input_file_list', [])
    if isinstance(input_file_list, str):
        input_file_list = json.loads(input_file_list)

    turn = {
        'turn': record.get('turn') or 1,
        'task_id': record['task_id'],
        'query': record['query'],
        'input_file_list': input_file_list,
        'expected_output': record.get('expected_output', ''),
        'L1': record.get('L1', ''),
        'L2': record.get('L2', ''),
        'difficulty': record.get('difficulty', ''),
    }

    pre_parsed = record.get('pre_parsed_doc_blocks')
    if pre_parsed:
        turn['pre_parsed_doc_blocks'] = pre_parsed

    # CI data may have conversation history embedded in the record
    history = record.get('history')
    if history:
        turn['history'] = history

    # ``mode`` controls how the turn is dispatched in task.py.
    # Currently supported:
    #   - ``'resume_trajectory'``: history ends in a ``tool`` message and the
    #     agent should continue from there with no new user message.
    # When omitted (default), task.py runs the standard
    # "upload files → build doc_blocks → user_prompt → run agent" flow.
    mode = record.get('mode')
    if mode:
        turn['mode'] = mode

    # Starting doc id for this turn — used by _run_session when the history
    # already contains [user_doc N] blocks (see converter) so the current
    # turn's new files don't restart numbering from 1.
    if 'session_doc_index_start' in record:
        turn['session_doc_index_start'] = record['session_doc_index_start']

    return turn


class OneAgentInferGenerator:
    """Yields BatchItem instances from YB OneAgentBench benchmark.

    Multi-turn sessions are merged into a single BatchItem with
    task_config.turns containing all turns in order.
    """

    def __init__(
        self,
        input_path: str | None = None,
        base_config_path: str | Path | None = None,
        limit: int | None = None,
    ) -> None:
        self.benchmark_root = _benchmark_root(input_path)
        self.base_config = _load_base_config(_config_path(base_config_path))
        env_limit = os.getenv('YB_BENCH_LIMIT')
        self.limit = int(env_limit) if env_limit else limit

    def __iter__(self):
        records = _load_benchmark(self.benchmark_root)
        sessions = _group_by_session(records)

        if self.limit is not None:
            sessions = sessions[: self.limit]

        for i, (session_key, items) in enumerate(sessions):
            config = dict(self.base_config)
            first = items[0]

            turns = [_record_to_turn(r) for r in items]

            # Bench ``time`` already embeds the timezone, e.g.
            #   '2026-03-12 Thursday (Asia/Shanghai, UTC+8)'
            # but prompt_builder expects ``cur_date`` to be a *pure* date and
            # appends ' (<tz>, <utc_offset>)' itself. Passing the raw string
            # would duplicate the timezone in the system prompt. Split it here:
            # keep the date part as ``cur_date`` and forward the IANA tz name as
            # ``timezone`` so the builder renders the timezone exactly once.
            cur_date, tz_name = _split_time_and_tz(first.get('time', ''))

            config['task_config'] = {
                'task_id': first['task_id'],
                'session_id': session_key,
                'turns': turns,
                'benchmark_root': str(self.benchmark_root),
                'timeout': 600 * len(turns),  # scale timeout with turn count
                'terminal_type': first.get('terminal', 'web'),
                'meta': {
                    k: v
                    for k, v in {
                        'cur_date': cur_date,
                        'timezone': tz_name,
                        'user_location': first.get('location', ''),
                    }.items()
                    if v
                },
            }

            # ENABLE_AUQ=false → exclude ask_user_question from agent tools
            enable_auq = os.getenv('ENABLE_AUQ', 'true').strip().lower()
            if enable_auq == 'false':
                from nexus.extensions.agents.yb_one_agent.prompt_builder import DEFAULT_TOOLS

                agent_tools = sorted(t for t in DEFAULT_TOOLS if t != 'ask_user_question')
                config.setdefault('agent_config', {})['tools'] = agent_tools

            yield BatchItem(
                config=config,  # type: ignore[invalid-argument-type]
                run_id=f'yb_bench_{session_key}',
                metadata={
                    'index': i,
                    'session_id': session_key,
                    'task_id': first['task_id'],
                    'n_turns': len(turns),
                    'L1': first.get('L1', ''),
                    'L2': first.get('L2', ''),
                    'difficulty': first.get('difficulty', ''),
                },
            )

    @staticmethod
    def total_count() -> int:
        root = _benchmark_root(None)
        records = _load_benchmark(root)
        sessions = _group_by_session(records)
        total = len(sessions)
        env_limit = os.getenv('YB_BENCH_LIMIT')
        if env_limit:
            return min(total, int(env_limit))
        return total

    @staticmethod
    def metadata() -> dict:
        return {
            'description': 'YB OneAgentBench evaluation (multi-turn session support)',
            'experiment': 'one_agent_infer',
        }


# ── public API (expected by GeneratorLoader) ────────────────────────


def generate_configs(
    input_path: str | None = None,
    base_config_path: str | Path | None = None,
):
    yield from OneAgentInferGenerator(
        input_path=input_path,
        base_config_path=base_config_path,
    )


def get_total_count() -> int:
    return OneAgentInferGenerator.total_count()


def get_batch_metadata() -> dict:
    return OneAgentInferGenerator.metadata()


__all__ = ['generate_configs', 'get_total_count', 'get_batch_metadata']
