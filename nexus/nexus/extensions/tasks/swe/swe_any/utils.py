"""Structured result protocol for Eval-authored SWE-Any test runners."""

from __future__ import annotations

import json
import re

from typing import Any


RUNNER_PROTOCOL_VERSION = 1
RESULTS_START = '>>>>> Start SWE-Any Structured Results'
RESULTS_END = '>>>>> End SWE-Any Structured Results'
TEST_STATUSES = frozenset({'PASSED', 'FAILED', 'ERROR', 'SKIPPED', 'XFAIL', 'XPASS'})
SUMMARY_STATUSES = frozenset({'completed', 'runner_error'})
_STATUS_SEVERITY = {
    'PASSED': 0,
    'XFAIL': 1,
    'SKIPPED': 2,
    'XPASS': 3,
    'FAILED': 4,
    'ERROR': 5,
}


class RunnerProtocolError(ValueError):
    """The runner result stream does not satisfy the SWE-Any JSONL contract."""


def parse_runner_results(text: str) -> dict[str, Any]:
    """Validate JSONL emitted by ``test_runner.sh`` and return tests + summary.

    Blank lines are ignored. Every non-blank line must be one JSON object and exactly one terminal
    summary is required. Runners should emit each opaque test id once, but pytest/JUnit may report
    one logical test separately for call and teardown failures. Duplicate ids are therefore merged
    conservatively to their most severe status instead of invalidating an otherwise usable oracle.
    """
    tests: dict[str, str] = {}
    duplicate_test_ids: set[str] = set()
    test_record_count = 0
    summary: dict[str, Any] | None = None
    for line_no, raw in enumerate((text or '').splitlines(), start=1):
        if not raw.strip():
            continue
        if summary is not None:
            raise RunnerProtocolError(f'line {line_no}: summary must be the final record')
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RunnerProtocolError(f'line {line_no}: invalid JSON: {exc.msg}') from exc
        if not isinstance(record, dict):
            raise RunnerProtocolError(f'line {line_no}: record must be an object')
        record_type = record.get('type')
        if record_type == 'test':
            test_id = record.get('id')
            status = record.get('status')
            if not isinstance(test_id, str) or not test_id.strip():
                raise RunnerProtocolError(f'line {line_no}: test id must be a non-empty string')
            if status not in TEST_STATUSES:
                raise RunnerProtocolError(f'line {line_no}: invalid test status {status!r}')
            test_record_count += 1
            if test_id in tests:
                duplicate_test_ids.add(test_id)
                if _STATUS_SEVERITY[status] > _STATUS_SEVERITY[tests[test_id]]:
                    tests[test_id] = status
            else:
                tests[test_id] = status
        elif record_type == 'summary':
            if summary is not None:
                raise RunnerProtocolError(f'line {line_no}: duplicate summary')
            status = record.get('status')
            if status not in SUMMARY_STATUSES:
                raise RunnerProtocolError(f'line {line_no}: invalid summary status {status!r}')
            for field in ('error_kind', 'message'):
                value = record.get(field)
                if value is not None and not isinstance(value, str):
                    raise RunnerProtocolError(f'line {line_no}: {field} must be a string or null')
            summary = record
        else:
            raise RunnerProtocolError(f'line {line_no}: unknown record type {record_type!r}')
    if summary is None:
        raise RunnerProtocolError('missing summary record')
    if summary['status'] == 'completed':
        count = summary.get('count')
        if isinstance(count, bool) or not isinstance(count, int) or count != test_record_count:
            raise RunnerProtocolError(f'completed summary count must equal {test_record_count}')
    elif tests:
        raise RunnerProtocolError('runner_error stream must not contain test records')
    return {
        'tests': tests,
        'summary': summary,
        'duplicate_test_ids': sorted(duplicate_test_ids),
    }


def extract_runner_results(log: str) -> dict[str, Any]:
    """Extract and parse the one structured result block embedded in an eval log."""
    lines = (log or '').splitlines()
    starts = [index for index, line in enumerate(lines) if line == RESULTS_START]
    ends = [index for index, line in enumerate(lines) if line == RESULTS_END]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise RunnerProtocolError('structured result markers not found')
    return parse_runner_results('\n'.join(lines[starts[0] + 1 : ends[0]]))


def derive_structured_reports(pre: dict[str, str], post: dict[str, str]) -> dict[str, list[str]]:
    """Derive F2P/P2P/P2F from two validated runner status maps.

    Post-gold PASS is the oracle anchor. A missing pre-gold id counts as non-pass (collection may
    fail in the masked state). A pre-gold PASS that is absent or non-pass post-gold is a regression.
    """
    report: dict[str, list[str]] = {'FAIL_TO_PASS': [], 'PASS_TO_PASS': [], 'PASS_TO_FAIL': []}
    for test_id, post_status in post.items():
        if post_status == 'PASSED':
            bucket = 'PASS_TO_PASS' if pre.get(test_id) == 'PASSED' else 'FAIL_TO_PASS'
            report[bucket].append(test_id)
    for test_id, pre_status in pre.items():
        if pre_status == 'PASSED' and post.get(test_id) != 'PASSED':
            report['PASS_TO_FAIL'].append(test_id)
    for key in report:
        report[key].sort()
    return report


_HUNK_RE = re.compile(r'^@@ -(\S+) \+(\S+) @@')


def _hunk_side_count(range_spec: str) -> int | None:
    """Line count of one side of a hunk range: ``start`` (implicit 1) or ``start,count``.

    Returns ``None`` when *count* is not an int (e.g. agent prose that happens to look like a
    hunk header) — callers treat that as "not a real hunk" instead of raising.
    """
    _, _, count = range_spec.partition(',')
    if not count:
        return 1
    try:
        return int(count)
    except ValueError:
        return None


def _header_path(raw: str) -> str:
    """A ``--- ``/``+++ `` header value → repo-relative path (drop the ``a/``/``b/`` prefix + trailing tab)."""
    path = raw.split('\t', 1)[0].strip()
    if path.startswith(('a/', 'b/')):
        path = path[2:]
    return path


def patch_files(patch: str) -> set[str]:
    """Repo-relative paths a unified diff modifies.

    Reads file headers only — ``diff --git`` plus the ``--- ``/``+++ `` pair — and skips hunk BODY
    lines by counting each hunk's declared line counts (the way ``git apply`` does), so body content
    that happens to start with ``--- ``/``+++ `` is never mistaken for a header. Whole-file deletions
    (``+++ /dev/null``) are recorded via their ``--- a/x`` path, and whole-file creates
    (``--- /dev/null``) via the ``+++ b/x`` path — both of which ``git apply`` accepts headerless.
    A line that looks like ``@@ … @@`` but has a non-integer count is ignored (not a hunk), so
    agent-authored prose cannot raise out of this helper.
    """
    files: set[str] = set()
    old_left = new_left = 0  # counted lines still owed by the current hunk (0/0 => outside a hunk)
    prev_old = ''  # last `--- ` path, used to name a `+++ /dev/null` whole-file deletion
    for line in patch.splitlines():
        if old_left <= 0 and new_left <= 0:
            if line.startswith('diff --git '):
                _head, sep, tail = line.partition(' b/')
                if sep:
                    files.add(tail.strip())
            elif line.startswith('--- '):
                prev_old = _header_path(line[4:])
            elif line.startswith('+++ '):
                path = _header_path(line[4:])
                if path and path != '/dev/null':
                    files.add(path)
                elif path == '/dev/null' and prev_old and prev_old != '/dev/null':
                    files.add(prev_old)  # whole-file deletion: the file is the ``---`` side
            else:
                m = _HUNK_RE.match(line)
                if m:
                    old_c = _hunk_side_count(m.group(1))
                    new_c = _hunk_side_count(m.group(2))
                    if old_c is None or new_c is None:
                        continue  # prose, not a hunk — stay outside
                    old_left, new_left = old_c, new_c
        else:
            # Inside a hunk body: consume by leading char so the counts decide when the hunk ends.
            head = line[:1]
            if head == '-':
                old_left -= 1
            elif head == '+':
                new_left -= 1
            elif head != '\\':  # a '\ No newline' marker belongs to the prior line; don't count it
                old_left -= 1
                new_left -= 1
    return files


__all__ = [
    'RESULTS_END',
    'RESULTS_START',
    'RUNNER_PROTOCOL_VERSION',
    'RunnerProtocolError',
    'derive_structured_reports',
    'extract_runner_results',
    'parse_runner_results',
    'patch_files',
]
