"""Fixture that repairs the Ghostscript pipe-protocol desync in matplotlib's image comparison.

## The bug

matplotlib compares ``[pdf]``/``[eps]`` baseline images by keeping one long-lived Ghostscript
process and driving it over pipes (``lib/matplotlib/testing/compare.py``, ``_GSConverter``).
Each request is answered by a ``GS>`` prompt (or ``GS<n>`` when *n* items are left on the
PostScript stack). On the failure path the converter tries to clear that stack::

    err = self._read_until(b"GS")
    stack = self._read_until(b">")
    if stack or not os.path.exists(dest):
        stack_size = int(stack[1:]) if stack else 0
        self._proc.stdin.write(b"pop\\n" * stack_size)   # <-- responses never read
        raise ImageComparisonFailure(...)

Every ``pop`` makes Ghostscript emit another prompt, but the code raises immediately and never
consumes them. Those prompts stay buffered in the pipe, so the *next* conversion reads a stale
prompt belonging to the previous exchange -- reader and writer are permanently off by one.

Verified against Ghostscript 9.55.0 in the instance image. After provoking one failed
conversion, every subsequent conversion reports success while the output file is never created::

    round 0: stack=b'<2' out=False
    round 1: stack=b'<1' out=False      # 12/12 conversions silently did nothing

The backlog of unread prompts is finite. Once it drains, ``_read_until`` blocks on
``self._proc.stdout.read(1)`` waiting for a byte that will never arrive: both pytest and ``gs``
park in ``pipe_wait`` at 0% CPU and the run wedges until an external timeout kills it. Observed
in the oracle canary on ``matplotlib__matplotlib-14623`` -- one run in twelve hung for the full
eval budget while the other eleven finished in ~80s. Because it takes a failed conversion to
seed the desync, it surfaces as a rare flake rather than a deterministic break.

## The fix

Drain the response to each ``pop`` so the protocol stays aligned, and drop the interpreter
outright if draining does not restore a clean prompt -- a fresh ``gs`` is started lazily on the
next call, which is always safer than reusing one whose state is unknown. This is a genuine
repair of the failure path: no timeouts, no watchdogs, and healthy conversions are untouched.

Verified in the instance image -- after three provoked failures, eight consecutive conversions
all return ``stack=b''`` with the output file present (pre-fix: all silently no-ops).

Scope: matplotlib instances only, and it edits test-harness plumbing rather than library code
under test, so it cannot mask a regression in the patch being graded.
"""

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_bench.test_spec import get_test_directives


_MATPLOTLIB_REPO = 'matplotlib/matplotlib'

# Test-harness file the fixture rewrites; the caller restores it after the run.
MATPLOTLIB_GHOSTSCRIPT_COMPARE_FILE = 'lib/matplotlib/testing/compare.py'

MATPLOTLIB_GHOSTSCRIPT_DESYNC_FIXTURE_COMMAND = """# matplotlib Ghostscript pipe desync fixture
python - <<'EOF_SWE_BENCH_MPL_GS_DESYNC'
import os
import sys

path = os.path.join('lib', 'matplotlib', 'testing', 'compare.py')
if not os.path.exists(path):
    print('matplotlib gs desync fixture: compare.py not found; skipping', file=sys.stderr)
    sys.exit(0)

with open(path, encoding='utf-8') as fh:
    source = fh.read()

if '# SWE-bench: drain pop responses' in source:
    sys.exit(0)

needle = '''        if stack or not os.path.exists(dest):
            stack_size = int(stack[1:]) if stack else 0
            self._proc.stdin.write(b"pop\\\\n" * stack_size)
'''

replacement = '''        if stack or not os.path.exists(dest):
            stack_size = int(stack[1:]) if stack else 0
            # SWE-bench: drain pop responses. Ghostscript answers every `pop` with another
            # prompt. Upstream writes the pops and raises without reading them, so the unread
            # prompts stay in the pipe and the next conversion consumes a stale one -- reader
            # and writer end up permanently off by one, and once the backlog drains the
            # blocking read(1) in _read_until waits forever (pytest and gs both in pipe_wait).
            # Consume one prompt per pop to keep the protocol aligned; if that does not leave a
            # clean prompt, discard the interpreter so the next call starts a fresh one.
            self._proc.stdin.write(b"pop\\\\n" * stack_size)
            try:
                self._proc.stdin.flush()
                for _ in range(stack_size):
                    self._read_until(b"GS")
                    if self._read_until(b">"):
                        raise _ConverterError
            except (_ConverterError, OSError, ValueError):
                try:
                    self._proc.kill()
                    self._proc.wait()
                except OSError:
                    pass
                self._proc = None
'''

if needle not in source:
    print('matplotlib gs desync fixture: failure path not in expected form; skipping',
          file=sys.stderr)
    sys.exit(0)

with open(path, 'w', encoding='utf-8') as fh:
    fh.write(source.replace(needle, replacement, 1))
print('matplotlib gs desync fixture: applied')
EOF_SWE_BENCH_MPL_GS_DESYNC"""


def needs_matplotlib_ghostscript_desync_fixture(instance: SWEbenchInstance) -> bool:
    """Return True when *instance* runs matplotlib tests that can reach the gs converter.

    Kept permissive within matplotlib: the desync is reachable from any suite with
    ``[pdf]``/``[eps]`` baseline comparisons, so gating on one instance id would leave siblings
    exposed. ``mpl_toolkits`` suites use ``image_comparison`` too. Other datasets are untouched.
    """
    if instance.get('repo') != _MATPLOTLIB_REPO:
        return False
    if instance.get('dataset_type') != 'swe-bench':
        return False

    try:
        directives = get_test_directives(instance)
    except Exception:  # noqa: BLE001 - a malformed instance is handled by the caller
        return False

    return any(str(d).startswith(('lib/matplotlib/tests/', 'lib/mpl_toolkits/')) for d in directives)


def get_matplotlib_ghostscript_desync_fixture_command(instance: SWEbenchInstance) -> str | None:
    """Return the shell snippet to inject, or ``None`` when the instance does not need it."""
    if not needs_matplotlib_ghostscript_desync_fixture(instance):
        return None
    return MATPLOTLIB_GHOSTSCRIPT_DESYNC_FIXTURE_COMMAND
