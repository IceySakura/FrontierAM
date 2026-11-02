"""Fixture that stops ``is_prime`` from recursing forever on already-evaluated numbers.

## The bug

Asking a numeric ``Pow`` or ``Mul`` whether it is prime can recurse until the stack is exhausted,
because both implementations delegate to a re-evaluated copy of themselves without checking that
re-evaluation actually produced something different.

``Pow._eval_is_prime`` (``sympy/core/power.py``)::

    if self.is_number:
        return self.doit().is_prime

For an already-evaluated power such as ``sqrt(2)``, ``doit()`` returns ``self`` -- verified in the
instance image: ``sqrt(Rational(2)).doit() is sqrt(Rational(2))`` is ``True``. So this asks the same
object the same question forever.

``Mul._eval_is_prime`` (``sympy/core/mul.py``) has the same shape, and its own comment shows the
problem was known but only half-addressed::

    # Note: `doit()` was not used due to test failing (Infinite Recursion)
    r = S.One
    for arg in self.args:
        r *= arg
    return r.is_prime

Re-multiplying the args can land back on an equal ``Mul``, and the intermediate products it builds
along the way trigger the same query again. The captured stack cycles between the two::

    sympy/core/mul.py:1335, in _eval_is_prime      -> r.is_prime
    sympy/core/assumptions.py:290, in _ask
    sympy/core/mul.py:1334, in _eval_is_prime      -> r *= arg  (builds a new Mul)
    ...  648 frames  ...
    RecursionError: maximum recursion depth exceeded

Both sites must be fixed: patching only ``mul.py`` left the failure rate unchanged at 20/20 runs,
because the cycle simply routes through ``Pow`` instead.

## Why it surfaces as a flake

``sympy/utilities/runtests.py`` seeds ``PYTHONHASHSEED`` randomly when it is unset, and the run
header records it -- the failing and passing canary batches show ``PYTHONHASHSEED=2351912622`` and
``3918649500`` respectively. The seed changes dict/set iteration order, which changes the assumption
evaluation path, which decides both *whether* the recursion triggers and *which* test it lands on.

Measured in the instance image over 25 unseeded runs of the graded files: ``RecursionError`` appeared
in **24**, but it usually hit ``test_powers_Integer``, ``test_simplify_ratio`` or
``test_mod_inverse`` -- none of which are graded, so the instance still passed. Occasionally it lands
on ``test_bug_sqrt``, which *is* in PASS_TO_PASS, and then the instance fails. That is the 1-in-18
failure rate seen in the canary. ``PYTHONHASHSEED=42`` and ``31337`` both reproduce the graded hit.

The defect is environmental, not caused by the candidate patch: at the bare ``base_commit`` with no
patch at all, ``RecursionError`` still appeared in 6 of 8 runs, and the gold patch does not touch
``power.py`` or ``mul.py``.

## Safety

The guards only add an early ``return None`` ("cannot determine") on the path that used to recurse;
every other branch is untouched, so no assumption becomes stronger. Verified in the image: with the
gold patch reverted but these guards in place, both FAIL_TO_PASS tests (``test_equality``,
``test_comparisons_with_unknown_type``) still fail -- the repair cannot mask what the instance
grades. With the gold patch applied, all 91 graded tests match the passing baseline exactly and
``RecursionError`` drops to 0 of 20 runs.
"""

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_bench.constants import KEY_INSTANCE_ID


_INSTANCE_ID = 'sympy__sympy-13091'
_REPO = 'sympy/sympy'

# Library files the fixture rewrites; the caller restores them after the run.
SYMPY_IS_PRIME_RECURSION_FILES = ('sympy/core/power.py', 'sympy/core/mul.py')

# Both needles are matched verbatim against the files as shipped (verified byte-for-byte in the
# instance image). Anchoring on the full block means a reshuffled upstream skips instead of being
# half-patched, and the marker check keeps it idempotent.
SYMPY_IS_PRIME_RECURSION_FIXTURE_COMMAND = """# sympy is_prime infinite recursion fixture
python - <<'EOF_SWE_BENCH_SYMPY_IS_PRIME'
import sys

from pathlib import Path

MARKER = 'SWE-bench: guard against is_prime self-recursion'

power_path = Path('sympy/core/power.py')
mul_path = Path('sympy/core/mul.py')

power_needle = '''        if self.is_number:
            return self.doit().is_prime
'''

power_replacement = '''        if self.is_number:
            # SWE-bench: guard against is_prime self-recursion. doit() returns self for an
            # already-evaluated power such as sqrt(2), so delegating asks the same object the
            # same question until the stack is exhausted.
            evaluated = self.doit()
            if evaluated is not self and evaluated != self:
                return evaluated.is_prime
'''

mul_needle = '''            r = S.One
            for arg in self.args:
                r *= arg
            return r.is_prime
'''

mul_replacement = '''            r = S.One
            for arg in self.args:
                r *= arg
            # SWE-bench: guard against is_prime self-recursion. Re-multiplying the args can land
            # back on an equal Mul, and the intermediate products trigger the same query again.
            if r is self or r == self:
                return None
            return r.is_prime
'''

edits = ((power_path, power_needle, power_replacement), (mul_path, mul_needle, mul_replacement))

sources = {}
for path, needle, _ in edits:
    try:
        text = path.read_text()
    except OSError:
        print('sympy is_prime recursion fixture: %s not found; skipping' % path, file=sys.stderr)
        raise SystemExit(0)
    if MARKER in text:
        raise SystemExit(0)
    if needle not in text:
        print('sympy is_prime recursion fixture: %s not in expected form; skipping' % path,
              file=sys.stderr)
        raise SystemExit(0)
    sources[path] = text

# Only write once both sites matched: fixing one alone does not stop the recursion, it just reroutes
# the cycle through the other class.
for path, needle, replacement in edits:
    path.write_text(sources[path].replace(needle, replacement, 1))

print('sympy is_prime recursion fixture: applied')
EOF_SWE_BENCH_SYMPY_IS_PRIME"""


def needs_sympy_is_prime_recursion_fixture(instance: SWEbenchInstance) -> bool:
    """Return whether this exact-case repair applies."""
    return (
        instance.get(KEY_INSTANCE_ID) == _INSTANCE_ID
        # Compare case-insensitively so a re-cased export cannot silently disable the repair.
        and (instance.get('repo') or '').lower() == _REPO
        and instance.get('dataset_type') == 'swe-bench'
    )


def get_sympy_is_prime_recursion_fixture_command(instance: SWEbenchInstance) -> str | None:
    """Return the shell snippet to inject, or ``None`` when the instance does not need it."""
    if not needs_sympy_is_prime_recursion_fixture(instance):
        return None
    return SYMPY_IS_PRIME_RECURSION_FIXTURE_COMMAND
