"""Compare ordered result JSONL files, ignoring only top-level lifecycle times.

Usage: uv run python tests/e2e/distributed/compare_cloud_results.py old.jsonl new.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re

from pathlib import Path
from typing import Any


def fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.md5()  # Compatibility checksum, not a security primitive.
    rows = []
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            record.pop('start_time', None)
            record.pop('end_time', None)
            # Preserve every remaining key, nested key order, array order and row order.
            normalized = (
                json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n'
            ).encode()
            digest.update(normalized)
            rows.append({'run_id': record['run_id'], 'md5': hashlib.md5(normalized).hexdigest()})
    return {'md5': digest.hexdigest(), 'count': len(rows), 'bytes': path.stat().st_size, 'rows': rows}


_JSON_STRING = rb'"(?:[^"\\]|\\.)*"'
_HEADER = re.compile(
    rb'^\{"run_id": ('
    + _JSON_STRING
    + rb'), "status": ('
    + _JSON_STRING
    + rb'), "start_time": (?:null|'
    + _JSON_STRING
    + rb'), "end_time": (?:null|'
    + _JSON_STRING
    + rb'), '
)


def fingerprint_streaming(path: Path) -> dict[str, Any]:
    """Hash validated standard exporter output without parsing large record bodies.

    This is stricter than fingerprint(): even whitespace outside the excluded
    fields participates in the checksum. Unsupported headers fail explicitly.
    """
    digest = hashlib.md5()
    rows = []
    with path.open('rb') as stream:
        while header := stream.readline(4096):
            match = _HEADER.match(header)
            if match is None:
                raise ValueError('Streaming comparison requires the standard exporter field order and spacing')
            normalized = b'{"run_id": ' + match[1] + b', "status": ' + match[2] + b', ' + header[match.end() :]
            row_digest = hashlib.md5(normalized)
            digest.update(normalized)
            chunk = header
            while not chunk.endswith(b'\n'):
                chunk = stream.readline(8 * 1024 * 1024)
                if not chunk:
                    raise ValueError('Incomplete JSONL record')
                row_digest.update(chunk)
                digest.update(chunk)
            rows.append({'run_id': json.loads(match[1]), 'md5': row_digest.hexdigest()})
    return {'md5': digest.hexdigest(), 'count': len(rows), 'bytes': path.stat().st_size, 'rows': rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--streaming', action='store_true', help='Stream validated standard exporter JSONL bytes')
    args = parser.parse_args()
    checksum = fingerprint_streaming if args.streaming else fingerprint
    reference = checksum(args.reference)
    candidate = checksum(args.candidate)
    equal = reference['md5'] == candidate['md5'] and reference['count'] == candidate['count']
    print(json.dumps({'equal': equal, 'reference': reference, 'candidate': candidate}, ensure_ascii=False, indent=2))
    return 0 if equal else 1


if __name__ == '__main__':
    raise SystemExit(main())
