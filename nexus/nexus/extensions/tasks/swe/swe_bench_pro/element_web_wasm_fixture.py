"""Fixture that isolates element-web's WebAssembly composer tests into their own jest process.

## The bug

``@matrix-org/matrix-wysiwyg`` ships a Rust/WASM editor core. When the objects it hands to
JavaScript are garbage collected, its finalizer runs on the WASM side and can throw::

    recursive use of an object detected which would lead to unsafe aliasing in rust
      at Q.wbg.__wbindgen_throw (node_modules/@matrix-org/matrix-wysiwyg/dist/matrix-wysiwyg.umd.cjs)
      at null.<anonymous> (wasm://wasm/002a3d1a)
      at FinalizationRegistry.cleanupSome (<anonymous>)

``FinalizationRegistry.cleanupSome`` is the giveaway: the throw is delivered at an arbitrary
garbage-collection point, not at the call site. This instance runs 13 test files through
``jest --maxWorkers=1``, so they all share one Node process. The WASM objects left behind by
``WysiwygComposer-test`` are therefore collected *while a later, unrelated test file is running*,
and that test is the one that fails.

Measured in the instance image over 10 runs of the graded file list, all 13 files together:

    every single run emitted the WASM error
    6 runs finished but with 1-2 unrelated tests broken
    4 runs died outright (41 / 135 / 150 / 150 test lines instead of 203, no "Tests:" summary)

Which test gets hit is random -- observed victims include
``InviteDialog > should lookup inputs which look like email addresses``,
``InviteDialog > should add pasted values``, ``Pill > should show a tooltip with the room Id`` and
``AutocompleteInput > should render custom suggestion element``. In the oracle canary this surfaced
as ``should lookup inputs which look like email addresses`` failing in one batch out of ten, and (in
an earlier batch) as 168 of 197 graded tests failing at once when the process died early.

The candidate patch touches ``MatrixChat.tsx``, ``SDKContext.ts``, ``usePermalinkMember.ts``,
``UserProfilesStore.ts`` and ``LruCache.ts``; the test patch touches ``Pill-test``,
``SdkContext-test``, ``UserProfilesStore-test`` and ``LruCache-test``. Neither goes anywhere near
``InviteDialog`` or the composer, and FAIL_TO_PASS passes 36/36 -- the broken tests are bystanders.

## The fix

Run the composer test file in a jest process of its own and the remaining files in another, then
concatenate both logs for the parser. No test is skipped, added, or reordered within its file, and
the parser sees exactly the same lines it always did -- there is simply no longer a shared heap for
the WASM finalizer to corrupt.

Verified in the instance image, 6 consecutive runs of the split form: zero WASM errors, zero broken
tests, and 174 + 29 = 203 test lines every time (the same 203 the single-process form produces on a
lucky run).
"""

import shlex

from collections.abc import Mapping
from typing import Any


ELEMENT_WEB_WASM_FIXTURE_MARKER = '# SWE-bench-pro element-web WASM composer isolation fixture'

_INSTANCE_ID = 'instance_element-hq__element-web-aec454dd6feeb93000380523cbb0b3681c0275fd-vnan'
_REPO = 'element-hq/element-web'

# The file whose WASM objects poison the shared process. Matched as a suffix so a ``.ts``/``.tsx``
# discrepancy in the dataset's file list cannot silently disable the split.
_WASM_TEST_STEM = 'test/components/views/rooms/wysiwyg_composer/components/WysiwygComposer-test'


def needs_element_web_wasm_fixture(instance: Mapping[str, Any]) -> bool:
    """Return whether this exact-case isolation applies."""
    return (
        instance.get('instance_id') == _INSTANCE_ID
        # Compare case-insensitively so a re-cased export cannot silently disable the fixture.
        and (instance.get('repo') or '').lower() == _REPO
    )


def split_selected_tests(selected_tests: list[str]) -> tuple[list[str], list[str]]:
    """Split *selected_tests* into (everything else, the WASM composer files).

    Returns two lists so the caller can run them as separate jest invocations. Order within each
    list is preserved.
    """
    wasm = [t for t in selected_tests if _WASM_TEST_STEM in t]
    rest = [t for t in selected_tests if _WASM_TEST_STEM not in t]
    return rest, wasm


def get_element_web_wasm_fixture_test_commands(
    instance: Mapping[str, Any],
    path: str,
    selected_tests: list[str],
) -> str:
    """Return test commands that run the composer file in its own jest process.

    Falls back to an empty string when the fixture does not apply, or when the file list does not
    actually contain both groups -- with nothing to separate, the default single invocation is
    already correct and this fixture must not change it.
    """
    if not needs_element_web_wasm_fixture(instance):
        return ''

    rest, wasm = split_selected_tests(selected_tests)
    if not rest or not wasm:
        return ''

    run_script = shlex.quote(f'{path}/run_script.sh')
    parser = shlex.quote(f'{path}/parser.py')
    stdout_log = shlex.quote(f'{path}/stdout.log')
    stderr_log = shlex.quote(f'{path}/stderr.log')
    test_output = shlex.quote(f'{path}/test_output.txt')
    rest_stdout = shlex.quote(f'{path}/stdout.rest.log')
    rest_stderr = shlex.quote(f'{path}/stderr.rest.log')
    wasm_stdout = shlex.quote(f'{path}/stdout.wasm.log')
    wasm_stderr = shlex.quote(f'{path}/stderr.wasm.log')
    rest_arg = shlex.quote(','.join(rest))
    wasm_arg = shlex.quote(','.join(wasm))

    # `|| true` on each invocation: a non-zero jest exit is normal here (the graded run tolerates
    # individual test failures and lets the parser decide), and this script has no `set -e` of its
    # own, so mirroring the default behaviour keeps both logs intact for the parser.
    return f"""
{ELEMENT_WEB_WASM_FIXTURE_MARKER}
# Two jest processes: the WASM composer file cannot then be garbage collected inside another
# file's test run. Logs are concatenated so the parser sees the usual combined output.
bash {run_script} {rest_arg} > {rest_stdout} 2> {rest_stderr} || true
bash {run_script} {wasm_arg} > {wasm_stdout} 2> {wasm_stderr} || true
cat {rest_stdout} {wasm_stdout} > {stdout_log}
cat {rest_stderr} {wasm_stderr} > {stderr_log}
# run parsing script
python {parser} {stdout_log} {stderr_log} {test_output}
""".strip()
