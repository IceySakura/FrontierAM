"""Build the single-round-trip payload that ``collect_verifier_outputs`` parses."""

import base64
import json

from typing import Any


def _encode(text: str) -> str:
    return base64.b64encode(text.encode('utf-8')).decode('ascii') if text else ''


def collected_stdout(
    *,
    reward: str = '',
    reward_json: str = '',
    ctrf: Any = None,
    result: Any = None,
    log: str = '',
    log_size: int | None = None,
) -> str:
    lines = [
        f'reward.txt {_encode(reward)}',
        f'reward.json {_encode(reward_json)}',
        f'ctrf.json {_encode(json.dumps(ctrf) if ctrf is not None else "")}',
        f'verifier_result.json {_encode(json.dumps(result) if result is not None else "")}',
        f'test.txt.size {len(log.encode("utf-8")) if log_size is None else log_size}',
        f'test.txt {_encode(log)}',
    ]
    return '\n'.join(lines) + '\n'
