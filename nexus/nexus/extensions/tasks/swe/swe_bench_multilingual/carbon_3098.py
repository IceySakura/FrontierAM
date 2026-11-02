"""Exact-case microsecond-precision repair for briannesbitt__carbon-3098.

``Construct > From serialization`` (a PASS_TO_PASS entry) fails for roughly 1% of runs with a
one-microsecond difference in ``CarbonInterval::$f``::

    -    'f' => 0.129872
    +    'f' => 0.129873

Root cause is PHP itself, not the gold patch: ``DateInterval::$f`` truncates instead of rounding on
assignment, so ``$f = $microseconds / 1_000_000`` lands one microsecond low for the ~1.1% of values
whose quotient is not exactly representable (``130541 / 1e6 * 1e6 == 130540.99999999999``). The loss
is cumulative -- each assignment drops another microsecond. Removing the gold patch does not help
(measured 27 failures in 2000 runs without it versus 21 with it), and the test is wall-clock
dependent (``new Carbon('-3 Days')`` / ``new Carbon('today')``), which is what makes it flaky.

Two changes are needed, and they fix *different* failure directions:

1. ``ConstructTest.php`` -- drop the ``PHP_VERSION >= 8.3`` early return in ``assertSameIntervals``.
   This is what fixes the -1µs failures. Upstream already has tolerance logic for exactly this case,
   but the early return skips it on PHP 8.3 and compares whole objects instead. The skipped branch is
   the only thing that can absorb the loss in the nested ``rawInterval``/``originalInput``: those are
   plain ``DateInterval`` objects rebuilt by PHP's own ``__unserialize``, and in the serialised
   payload ``rawInterval`` is a *reference* (``r:``) to ``originalInput``, so Carbon has nowhere to
   intervene. Deleting the early return lets the existing tolerance run; no assertion is weakened,
   and every other check in the helper is untouched.
2. ``CarbonInterval.php`` -- nudge the two integer-microsecond assignments by half a microsecond so
   the truncation lands on the intended value. This is what fixes the +1µs failures: the tolerance
   block enabled by change 1 aligns the two intervals with
   ``$expected->microseconds = $actual->microseconds``, which routes through the ``microsecond``
   setter and used to lose *another* microsecond, overshooting in the opposite direction.

Attribution measured over full-file runs (37 tests each):

    baseline (gold + test patch only)   21 / 2000   1.05%
    change 1 alone                       9 / 3000   0.30%   remainder all +1µs
    change 2 alone                      24 / 3000   0.80%   remainder all -1µs
    both                                 0 / 5000   0%

Deliberately *not* attempted: compensating ``$this->f = $years->f`` on the
``diffAsCarbonInterval`` path, or repairing the nested intervals inside
``CarbonInterval::__unserialize``. Both were measured and neither reaches zero (35/3000 and 24/3000),
because the reference-shared nested ``DateInterval`` is rebuilt entirely inside PHP.

Note that the half-microsecond nudge relies on the setter truncating: ``130541.5 / 1e6`` is stored as
``0.130541``. Verified exhaustively -- all 1,000,000 microsecond values round-trip exactly, with no
value landing high. ``$f`` is always non-negative (sign lives in ``invert``, and
``solveNegativeInterval`` multiplies back to positive), so the nudge never has to handle a negative
input.

Safety: FAIL_TO_PASS (``Construct > Make``) still fails without the gold patch after these changes
(verified: ``DateMalformedIntervalStringException: Unknown or bad format (PT13.000001S)``), so the
repair does not mask what the instance is meant to test. Both edits are idempotent and skip silently
when their anchor is missing.
"""

from nexus.extensions.tasks.swe.constants import SWEbenchInstance


INSTANCE_ID = 'briannesbitt__carbon-3098'
REPO = 'briannesbitt/carbon'
DATASET_TYPE = 'swe-bench-multilingual'
WORKDIR = '/testbed'

_SOURCE_FILE = 'src/Carbon/CarbonInterval.php'
_TEST_FILE = 'tests/CarbonInterval/ConstructTest.php'

# The gold patch renames Carbon:: to CarbonInterface:: at one of the two assignment sites, so both
# spellings are accepted. Pre-patch the file only has the Carbon:: form.
_ROUND_MICROSECONDS = f"""python3 - <<'EOF_CARBON_3098_ROUND'
from pathlib import Path

path = Path('{WORKDIR}/{_SOURCE_FILE}')
try:
    source = path.read_text()
except OSError:
    raise SystemExit(0)

patched = source
for const in ('Carbon', 'CarbonInterface'):
    for var in ('$value', '$microseconds'):
        old = f'$this->f = {{var}} / {{const}}::MICROSECONDS_PER_SECOND;'
        new = f'$this->f = ((int) {{var}} + 0.5) / {{const}}::MICROSECONDS_PER_SECOND;'
        if new not in patched:
            patched = patched.replace(old, new)

if patched != source:
    path.write_text(patched)
EOF_CARBON_3098_ROUND"""

_APPLY_EXISTING_TOLERANCE = f"""python3 - <<'EOF_CARBON_3098_TOLERANCE'
from pathlib import Path

path = Path('{WORKDIR}/{_TEST_FILE}')
try:
    source = path.read_text()
except OSError:
    raise SystemExit(0)

early_return = (
    '        if (PHP_VERSION >= 8.3) {{\\n'
    '            $this->assertEquals($expected, $actual);\\n'
    '\\n'
    '            return;\\n'
    '        }}\\n'
)
if early_return in source:
    path.write_text(source.replace(early_return, '', 1))
EOF_CARBON_3098_TOLERANCE"""


def is_target_instance(instance: SWEbenchInstance) -> bool:
    """Return whether this exact-case repair applies."""
    return (
        instance.get('instance_id') == INSTANCE_ID
        # The dataset spells the repo lowercase; compare case-insensitively so a re-cased export
        # cannot silently disable the repair.
        and (instance.get('repo') or '').lower() == REPO
    )


def make_setup_commands() -> list[str]:
    """Return the commands that make the microsecond comparisons deterministic.

    These must run after the test patch is applied and before the tests execute.
    """
    return [
        '# Nexus exact-case microsecond determinism for carbon-3098',
        _ROUND_MICROSECONDS,
        _APPLY_EXISTING_TOLERANCE,
    ]
