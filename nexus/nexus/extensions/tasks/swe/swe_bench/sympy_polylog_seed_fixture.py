"""Fixture that makes ``test_polylog_values`` deterministic instead of randomly sampled.

## The bug

The test added by this instance's ``test_patch`` draws its own sample points with an unseeded
``random``::

    def test_polylog_values():
        import random
        ...
        for s in [-1, 0, 1]:
            for _ in range(10):
                z = random.uniform(-5, 5) + I*random.uniform(-5, 5)
                assert Abs(polylog(s, z).evalf() - polylog(s, z, evaluate=False).evalf()) < 1e-15

Thirty fresh complex points per run, each required to agree to ``1e-15`` between the auto-evaluated
and ``evaluate=False`` paths. That tolerance sits right against double precision: the machine
epsilon is ~2.2e-16, so a passing run typically has only ~4x headroom.

Measured in the instance image over 12000 sampled ``(s, z)`` points with the gold patch applied:

* 2 points (0.017%) exceed ``1e-15``
* worst observed difference **1.421e-14** -- 14x the tolerance
* distribution: p50 = 0, p99 = 2.29e-16, p99.9 = 4.58e-16

So almost every point is exact or near-exact, and a rare draw blows straight through the bound. The
canary saw this as 1 failure in 19 runs of the full suite.

The trigger is the mantissa, not a region of the complex plane: hand-writing the worst point with a
short decimal (``1.08 - 0.05j``) gives a difference of exactly 0, while the randomly drawn
``1.0804324833712666 - 0.05193831736437815j`` gives 1.421e-14.

## Why seeding rather than loosening the bound

Relaxing ``1e-15`` would also stop the flake, but it weakens what the test checks -- the tolerance is
the assertion. Seeding removes only the run-to-run variation and leaves the test exactly as strict
for the points it does check.

## Seed choice

Candidate seeds were measured in the image with the gold patch applied, taking the worst of the 30
drawn points plus the 6 fixed ones:

======  ==================  ========
seed    worst difference    headroom
======  ==================  ========
0       2.289e-16           4.4x
1       2.220e-16           4.5x
42      2.289e-16           4.4x
7       2.483e-16           4.0x
1234    **1.110e-16**       **9.0x**
======  ==================  ========

1234 is used because double the headroom is the difference between "passes here" and "passes on a
host whose libm or mpmath rounds slightly differently".

## FAIL_TO_PASS is preserved

``test_polylog_values`` is this instance's only FAIL_TO_PASS test, so the fixture must not turn it
green on its own. Verified in the image with the test patch applied and the gold patch reverted:
seeds 0, 1 and 42 all still fail (the pre-patch ``polylog(-1, z)`` is simply wrong, independent of
which points are drawn). Seeding decides *which* points are checked, not *whether* the implementation
is correct.

The eval script already restores the test file with ``git checkout <base_commit>
sympy/functions/special/tests/test_zeta_functions.py`` after the run, so this fixture does not need
its own cleanup step.
"""

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_bench.constants import KEY_INSTANCE_ID


_INSTANCE_ID = 'sympy__sympy-13852'
_REPO = 'sympy/sympy'

# Anchored on the test's own `import random`, which the test patch adds inside the function body.
# Matching that exact indented line means an upstream reshuffle skips the fixture instead of
# injecting the seed into the wrong scope, and the marker keeps re-runs idempotent.
SYMPY_POLYLOG_SEED_FIXTURE_COMMAND = """# sympy test_polylog_values determinism fixture
python - <<'EOF_SWE_BENCH_SYMPY_POLYLOG_SEED'
import sys

from pathlib import Path

path = Path('sympy/functions/special/tests/test_zeta_functions.py')
needle = '    import random\\n'
replacement = '    import random\\n    random.seed(1234)  # swe-bench: pin sampled points\\n'
MARKER = 'random.seed(1234)'

try:
    text = path.read_text()
except OSError:
    print('sympy polylog seed fixture: %s not found; skipping' % path, file=sys.stderr)
    raise SystemExit(0)

if MARKER in text:
    raise SystemExit(0)

# The test patch introduces exactly one `import random`, inside test_polylog_values. More than one
# would mean the layout changed and the seed could land in the wrong function.
if text.count(needle) != 1:
    print('sympy polylog seed fixture: %s not in expected form; skipping' % path, file=sys.stderr)
    raise SystemExit(0)

path.write_text(text.replace(needle, replacement, 1))
print('sympy polylog seed fixture: applied')
EOF_SWE_BENCH_SYMPY_POLYLOG_SEED"""


def needs_sympy_polylog_seed_fixture(instance: SWEbenchInstance) -> bool:
    """Return whether this exact-case repair applies."""
    return (
        instance.get(KEY_INSTANCE_ID) == _INSTANCE_ID
        # Compare case-insensitively so a re-cased export cannot silently disable the repair.
        and (instance.get('repo') or '').lower() == _REPO
        and instance.get('dataset_type') == 'swe-bench'
    )


def get_sympy_polylog_seed_fixture_command(instance: SWEbenchInstance) -> str | None:
    """Return the shell snippet to inject, or ``None`` when the instance does not need it."""
    if not needs_sympy_polylog_seed_fixture(instance):
        return None
    return SYMPY_POLYLOG_SEED_FIXTURE_COMMAND
