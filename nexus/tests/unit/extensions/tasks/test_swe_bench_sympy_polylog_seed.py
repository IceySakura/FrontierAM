import subprocess
import sys

from pathlib import Path

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_bench import dataset_handler, sympy_polylog_seed_fixture
from nexus.extensions.tasks.swe.swe_bench.constants import START_TEST_OUTPUT


_INSTANCE_ID = 'sympy__sympy-13852'
_TEST_FILE = 'sympy/functions/special/tests/test_zeta_functions.py'
_MARKER = '# sympy test_polylog_values determinism fixture'
_HEREDOC = 'EOF_SWE_BENCH_SYMPY_POLYLOG_SEED'

# The shape the test patch leaves behind: `import random` indented inside the test body, followed by
# the unseeded draws. Verbatim from the instance image.
_TEST_FILE_STUB = """def test_polylog_expansion():
    assert myexpand(polylog(0, z), z/(1 - z))


def test_polylog_values():
    import random
    assert polylog(2, 2) == pi**2/4 - I*pi*log(2)
    for s in [-1, 0, 1]:
        for _ in range(10):
            z = random.uniform(-5, 5) + I*random.uniform(-5, 5)
            assert Abs(polylog(s, z).evalf() - polylog(s, z, evaluate=False).evalf()) < 1e-15
"""


def _target_instance(*, instance_id: str = _INSTANCE_ID, repo: str = 'sympy/sympy') -> SWEbenchInstance:
    return {
        'instance_id': instance_id,
        'repo': repo,
        'dataset_type': 'swe-bench',
        'version': '1.1',
        'base_commit': 'c935e1d106743efd5bf0705fbeedbd18fadff4dc',
        'environment_setup_commit': 'c935e1d106743efd5bf0705fbeedbd18fadff4dc',
        'workdir': '/testbed',
        'test_patch': f'diff --git a/{_TEST_FILE} b/{_TEST_FILE}\n',
        'FAIL_TO_PASS': ['test_polylog_values'],
        'PASS_TO_PASS': ['test_zeta_eval'],
    }


def _snippet_for(target: Path) -> str:
    """Return the fixture's embedded python, retargeted at ``target``.

    The terminator is matched at line start: the heredoc name also appears on the opening
    ``<<'NAME'`` line, so searching for the bare name finds that one and yields an empty body.
    """
    command = sympy_polylog_seed_fixture.SYMPY_POLYLOG_SEED_FIXTURE_COMMAND
    start = command.index(f"python - <<'{_HEREDOC}'")
    body_start = command.index('\n', start) + 1
    body_end = command.index(f'\n{_HEREDOC}', body_start)
    body = command[body_start:body_end]
    return body.replace(f"Path('{_TEST_FILE}')", f'Path({str(target)!r})')


def _run_snippet(source: str, tmp_path: Path) -> str:
    """Execute the fixture's embedded python against a stub file and return the result.

    The snippet is pulled out of the shipped command string rather than re-typed, so the test
    exercises exactly what runs in the sandbox.
    """
    target = tmp_path / 'test_zeta_functions.py'
    target.write_text(source)
    subprocess.run([sys.executable, '-c', _snippet_for(target)], check=True, cwd=tmp_path)
    return target.read_text()


def test_polylog_seed_fixture_applies_to_target_instance() -> None:
    assert sympy_polylog_seed_fixture.needs_sympy_polylog_seed_fixture(_target_instance())
    assert sympy_polylog_seed_fixture.get_sympy_polylog_seed_fixture_command(_target_instance()) is not None


def test_polylog_seed_fixture_skips_other_instances() -> None:
    """Sibling sympy instances have their own flakes; do not touch their test files."""
    other = _target_instance(instance_id='sympy__sympy-13091')

    assert not sympy_polylog_seed_fixture.needs_sympy_polylog_seed_fixture(other)
    assert sympy_polylog_seed_fixture.get_sympy_polylog_seed_fixture_command(other) is None


def test_polylog_seed_fixture_skips_foreign_repo() -> None:
    """An instance id collision across repos must not enable the repair."""
    other = _target_instance(repo='not-sympy/sympy')

    assert not sympy_polylog_seed_fixture.needs_sympy_polylog_seed_fixture(other)


def test_polylog_seed_fixture_is_injected_before_the_test_command() -> None:
    script = dataset_handler.get_eval_script(_target_instance())

    assert _MARKER in script
    # Two orderings matter: the anchor line (`import random`) arrives with the test patch, and the
    # seed must be in place before the tests run.
    assert script.index('git apply') < script.index(_MARKER)
    assert script.index(_MARKER) < script.index(f": '{START_TEST_OUTPUT}'")


def test_polylog_seed_fixture_relies_on_the_scripts_own_test_file_restore() -> None:
    """No extra cleanup is added, because the eval script already restores the test file.

    The script checks the file out twice on its own -- once to reset before applying the test patch,
    once after the run -- and the trailing one is what undoes the seeding. Asserting the fixture adds
    no third checkout guards against redundant cleanup, and asserting the trailing one exists guards
    against it disappearing without anyone noticing this fixture depended on it.
    """
    script = dataset_handler.get_eval_script(_target_instance())
    restore = f'git checkout c935e1d106743efd5bf0705fbeedbd18fadff4dc {_TEST_FILE}'

    assert script.count(restore) == 2
    # The last checkout must come after the tests, so the seeded file never survives the run.
    assert script.rindex(restore) > script.index(_MARKER)


def test_polylog_seed_snippet_pins_the_seed(tmp_path: Path) -> None:
    patched = _run_snippet(_TEST_FILE_STUB, tmp_path)

    assert 'random.seed(1234)' in patched
    # Must land inside the test body, right after its own import -- not at module scope, where it
    # would be overwritten by whatever runs next.
    assert '    import random\n    random.seed(1234)' in patched


def test_polylog_seed_snippet_is_idempotent(tmp_path: Path) -> None:
    """Eval may retry, and a second seed line would be harmless but signals a broken guard."""
    once = _run_snippet(_TEST_FILE_STUB, tmp_path)
    twice = _run_snippet(once, tmp_path)

    assert twice == once
    assert twice.count('random.seed(1234)') == 1


def test_polylog_seed_snippet_skips_when_anchor_is_ambiguous(tmp_path: Path) -> None:
    """Two `import random` lines mean the layout changed and the seed could land in the wrong test."""
    ambiguous = _TEST_FILE_STUB + '\n\ndef test_something_else():\n    import random\n    pass\n'

    result = _run_snippet(ambiguous, tmp_path)

    assert 'random.seed' not in result


def test_polylog_seed_snippet_is_noop_without_anchor(tmp_path: Path) -> None:
    """Assert the file is never written when the anchor is absent.

    Read-only makes a stray write fail loudly instead of passing silently; comparing content alone
    would succeed even without the guard, since str.replace is already a no-op on a missing needle.
    """
    unrelated = 'def test_polylog_values():\n    assert True\n'
    target = tmp_path / 'test_zeta_functions.py'
    target.write_text(unrelated)
    target.chmod(0o444)

    subprocess.run([sys.executable, '-c', _snippet_for(target)], check=True, cwd=tmp_path)

    assert target.read_text() == unrelated
