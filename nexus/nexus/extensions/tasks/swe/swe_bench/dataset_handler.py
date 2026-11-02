import json

from typing import Any, cast

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_bench.constants import (
    END_TEST_OUTPUT,
    KEY_INSTANCE_ID,
    MAP_REPO_TO_EXT,
    MAP_REPO_VERSION_TO_SPECS,
    START_TEST_OUTPUT,
    TEST_SPHINX,
)
from nexus.extensions.tasks.swe.swe_bench.grading import get_eval_report
from nexus.extensions.tasks.swe.swe_bench.matplotlib_ghostscript_desync_fixture import (
    MATPLOTLIB_GHOSTSCRIPT_COMPARE_FILE,
    get_matplotlib_ghostscript_desync_fixture_command,
)
from nexus.extensions.tasks.swe.swe_bench.requests_internal_httpbin import (
    REQUESTS_INTERNAL_HTTPBIN_TEST_COMMAND,
    REQUESTS_INTERNAL_HTTPBIN_TEST_FILE,
    get_requests_internal_httpbin_fixture_command,
)
from nexus.extensions.tasks.swe.swe_bench.sphinx_latex_images_fixture import (
    SPHINX_LATEX_IMAGES_TEST_FILE,
    get_sphinx_latex_images_fixture_command,
)
from nexus.extensions.tasks.swe.swe_bench.sphinx_linkcheck_fixture import (
    SPHINX_LINKCHECK_LOCAL_RESPONSE_FIXTURE_COMMAND,
    SPHINX_LINKCHECK_TEST_FILE,
    needs_sphinx_linkcheck_local_response_fixture,
)
from nexus.extensions.tasks.swe.swe_bench.sympy_is_prime_recursion_fixture import (
    SYMPY_IS_PRIME_RECURSION_FILES,
    get_sympy_is_prime_recursion_fixture_command,
)
from nexus.extensions.tasks.swe.swe_bench.sympy_polylog_seed_fixture import (
    get_sympy_polylog_seed_fixture_command,
)
from nexus.extensions.tasks.swe.swe_bench.test_spec import TestSpec, get_test_directives, make_eval_script_list


_MATPLOTLIB_HTTPS_INSTANCE_ID = 'matplotlib__matplotlib-20488'
_MATPLOTLIB_HTTPS_REPO = 'matplotlib/matplotlib'
_MATPLOTLIB_HTTPS_VERSION = '3.4'
_MATPLOTLIB_HTTPS_BASE_COMMIT = 'b7ce415c15eb39b026a097a2865da73fbcf15c9c'
_MATPLOTLIB_HTTPS_TEST_FILE = 'lib/matplotlib/tests/test_image.py'
_MATPLOTLIB_HTTPS_TEST = f'{_MATPLOTLIB_HTTPS_TEST_FILE}::test_https_imread_smoketest'
_MATPLOTLIB_LOCAL_URL_FIXTURE_COMMAND = """# SWE-bench matplotlib local URL response fixture
if ! git apply --verbose - <<'EOF_SWE_BENCH_MATPLOTLIB_LOCAL_URL_FIXTURE'
diff --git a/lib/matplotlib/tests/test_image.py b/lib/matplotlib/tests/test_image.py
--- a/lib/matplotlib/tests/test_image.py
+++ b/lib/matplotlib/tests/test_image.py
@@ -1148,9 +1148,44 @@
\x20
 @pytest.mark.network
 @pytest.mark.flaky
-def test_https_imread_smoketest():
+def test_https_imread_smoketest(monkeypatch):
+    import io
+    import ssl
+    import urllib.request
+    from pathlib import Path
+
+    url = 'https://matplotlib.org/1.5.0/_static/logo2.png'
+    image_path = (
+        Path(__file__).parent
+        / 'baseline_images/pngsuite/basn3p04.png'
+    )
+    payload = image_path.read_bytes()
+
+    class NonSeekableResponse:
+        def __enter__(self):
+            return self
+
+        def __exit__(self, exc_type, exc_value, traceback):
+            return False
+
+        def seek(self, *args):
+            raise io.UnsupportedOperation
+
+        def read(self):
+            return payload
+
+    def fake_urlopen(request_url, *, context):
+        assert request_url == url
+        assert context is not None
+        assert context.check_hostname is True
+        assert context.verify_mode == ssl.CERT_REQUIRED
+        return NonSeekableResponse()
+
+    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
+    expected = mimage.imread(image_path)
     with _api.suppress_matplotlib_deprecation_warning():
-        v = mimage.imread('https://matplotlib.org/1.5.0/_static/logo2.png')
+        actual = mimage.imread(url)
+    np.testing.assert_array_equal(actual, expected)
\x20
\x20
 # A basic ndarray subclass that implements a quantity
EOF_SWE_BENCH_MATPLOTLIB_LOCAL_URL_FIXTURE
then
    echo 'Failed to apply matplotlib local URL response fixture' >&2
    exit 1
fi"""


def _needs_matplotlib_local_url_fixture(instance: SWEbenchInstance) -> bool:
    if instance.get(KEY_INSTANCE_ID) != _MATPLOTLIB_HTTPS_INSTANCE_ID:
        return False

    expected_fields = {
        'repo': _MATPLOTLIB_HTTPS_REPO,
        'dataset_type': 'swe-bench',
        'version': _MATPLOTLIB_HTTPS_VERSION,
        'base_commit': _MATPLOTLIB_HTTPS_BASE_COMMIT,
    }
    for field, expected in expected_fields.items():
        if instance.get(field) != expected:
            raise ValueError(f'{_MATPLOTLIB_HTTPS_INSTANCE_ID}: unexpected {field}')

    pass_to_pass: Any = instance.get('PASS_TO_PASS')
    if isinstance(pass_to_pass, str):
        try:
            pass_to_pass = json.loads(pass_to_pass)
        except json.JSONDecodeError as e:
            raise ValueError(f'{_MATPLOTLIB_HTTPS_INSTANCE_ID}: invalid PASS_TO_PASS') from e
    if not isinstance(pass_to_pass, list) or pass_to_pass.count(_MATPLOTLIB_HTTPS_TEST) != 1:
        raise ValueError(f'{_MATPLOTLIB_HTTPS_INSTANCE_ID}: unexpected PASS_TO_PASS')

    if not isinstance(instance.get('test_patch'), str) or get_test_directives(instance) != [
        _MATPLOTLIB_HTTPS_TEST_FILE
    ]:
        raise ValueError(f'{_MATPLOTLIB_HTTPS_INSTANCE_ID}: unexpected test_patch directive')

    return True


def get_eval_script(instance: SWEbenchInstance) -> str:
    requests_internal_httpbin_fixture = get_requests_internal_httpbin_fixture_command(instance)
    add_matplotlib_fixture = _needs_matplotlib_local_url_fixture(instance)
    add_sphinx_linkcheck_fixture = needs_sphinx_linkcheck_local_response_fixture(instance)
    sphinx_latex_images_fixture = get_sphinx_latex_images_fixture_command(instance)
    sympy_is_prime_fixture = get_sympy_is_prime_recursion_fixture_command(instance)
    sympy_polylog_seed_fixture = get_sympy_polylog_seed_fixture_command(instance)
    matplotlib_ghostscript_desync_fixture = get_matplotlib_ghostscript_desync_fixture_command(instance)
    repo = instance['repo']
    version = instance.get('version')
    base_commit = instance['base_commit']
    test_patch = instance['test_patch']

    env_name = 'testbed'
    repo_directory = f'/{env_name}'
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]

    eval_script_list = make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch)
    if requests_internal_httpbin_fixture is not None:
        instance_id = instance[KEY_INSTANCE_ID]
        original_test_command = specs.get('test_cmd')
        if not isinstance(original_test_command, str):
            raise ValueError(f'{instance_id}: unexpected original test command')
        expected_test_command = f'{original_test_command} {REQUESTS_INTERNAL_HTTPBIN_TEST_FILE}'
        try:
            end_test_output = eval_script_list.index(f": '{END_TEST_OUTPUT}'")
        except ValueError as e:
            raise ValueError(f'{instance_id}: missing end test output marker') from e
        if end_test_output == 0 or eval_script_list[end_test_output - 1] != expected_test_command:
            raise ValueError(f'{instance_id}: unexpected original test command')
        eval_script_list[end_test_output - 1] = (
            f'{REQUESTS_INTERNAL_HTTPBIN_TEST_COMMAND} {REQUESTS_INTERNAL_HTTPBIN_TEST_FILE}'
        )
        try:
            start_test_output = eval_script_list.index(f": '{START_TEST_OUTPUT}'")
        except ValueError as e:
            raise ValueError(f'{instance_id}: missing start test output marker') from e
        if start_test_output >= end_test_output:
            raise ValueError(f'{instance_id}: unexpected test output marker order')
        eval_script_list.insert(start_test_output, requests_internal_httpbin_fixture)
        end_test_output = eval_script_list.index(f": '{END_TEST_OUTPUT}'")
        eval_script_list.insert(end_test_output + 1, f'git checkout {base_commit} requests/cacert.pem')
    if add_matplotlib_fixture:
        start_test_output = eval_script_list.index(f": '{START_TEST_OUTPUT}'")
        eval_script_list.insert(start_test_output, _MATPLOTLIB_LOCAL_URL_FIXTURE_COMMAND)
    if sympy_is_prime_fixture is not None:
        # Insert before the test command so the guards are in place for the run, and restore the
        # library files afterwards so the graded tree is left byte-identical.
        start_test_output = eval_script_list.index(f": '{START_TEST_OUTPUT}'")
        eval_script_list.insert(start_test_output, sympy_is_prime_fixture)
        end_test_output = eval_script_list.index(f": '{END_TEST_OUTPUT}'")
        eval_script_list.insert(
            end_test_output + 1, f'git checkout {base_commit} {" ".join(SYMPY_IS_PRIME_RECURSION_FILES)}'
        )
    if sympy_polylog_seed_fixture is not None:
        # Insert before the test command. Position matters twice over: the line it anchors on
        # (`import random`) is introduced by the test patch, which is applied earlier in the script,
        # and the test file is already restored by the trailing `git checkout <base_commit> <test
        # file>` that the eval script emits -- so no extra cleanup step is needed here.
        start_test_output = eval_script_list.index(f": '{START_TEST_OUTPUT}'")
        eval_script_list.insert(start_test_output, sympy_polylog_seed_fixture)
    if matplotlib_ghostscript_desync_fixture is not None:
        # Insert before the test command so the repaired converter is in place for the run, and
        # restore compare.py afterwards so the graded tree is left byte-identical.
        start_test_output = eval_script_list.index(f": '{START_TEST_OUTPUT}'")
        eval_script_list.insert(start_test_output, matplotlib_ghostscript_desync_fixture)
        end_test_output = eval_script_list.index(f": '{END_TEST_OUTPUT}'")
        eval_script_list.insert(
            end_test_output + 1, f'git checkout {base_commit} {MATPLOTLIB_GHOSTSCRIPT_COMPARE_FILE}'
        )
    if add_sphinx_linkcheck_fixture:
        start_test_output = eval_script_list.index(f": '{START_TEST_OUTPUT}'")
        eval_script_list.insert(start_test_output, SPHINX_LINKCHECK_LOCAL_RESPONSE_FIXTURE_COMMAND)
        end_test_output = eval_script_list.index(f": '{END_TEST_OUTPUT}'")
        eval_script_list.insert(end_test_output + 1, f'git checkout {base_commit} {SPHINX_LINKCHECK_TEST_FILE}')
    if sphinx_latex_images_fixture is not None:
        instance_id = instance[KEY_INSTANCE_ID]
        start_marker = f": '{START_TEST_OUTPUT}'"
        end_marker = f": '{END_TEST_OUTPUT}'"
        sleep_command = 'sleep ${SWE_BENCH_TEST_SLEEP:-0}'
        test_command = f'{TEST_SPHINX} {SPHINX_LATEX_IMAGES_TEST_FILE}'
        cleanup_command = f'git checkout {base_commit} {SPHINX_LATEX_IMAGES_TEST_FILE}'
        apply_test_patch_command = f"git apply -v - <<'EOF_114329324912'\n{test_patch}\nEOF_114329324912"

        if specs.get('test_cmd') != TEST_SPHINX:
            raise ValueError(f'{instance_id}: unexpected configured test command')
        if eval_script_list.count(start_marker) != 1 or eval_script_list.count(end_marker) != 1:
            raise ValueError(f'{instance_id}: unexpected test output markers')
        start_test_output = eval_script_list.index(start_marker)
        end_test_output = eval_script_list.index(end_marker)
        if start_test_output >= end_test_output:
            raise ValueError(f'{instance_id}: unexpected test output marker order')
        if (
            end_test_output != start_test_output + 3
            or eval_script_list.count(sleep_command) != 1
            or eval_script_list[start_test_output + 1] != sleep_command
            or eval_script_list[start_test_output + 2] != test_command
            or eval_script_list.count(test_command) != 1
        ):
            raise ValueError(f'{instance_id}: unexpected test command sequence')
        if (
            start_test_output == 0
            or eval_script_list[start_test_output - 1] != apply_test_patch_command
            or eval_script_list.count(apply_test_patch_command) != 1
        ):
            raise ValueError(f'{instance_id}: unexpected official test patch placement')
        if (
            end_test_output + 1 >= len(eval_script_list)
            or eval_script_list[end_test_output + 1] != cleanup_command
            or eval_script_list.count(cleanup_command) != 2
        ):
            raise ValueError(f'{instance_id}: unexpected test cleanup')

        eval_script_list.insert(start_test_output, sphinx_latex_images_fixture)

        start_test_output = eval_script_list.index(start_marker)
        end_test_output = eval_script_list.index(end_marker)
        if (
            eval_script_list[start_test_output - 1] != sphinx_latex_images_fixture
            or start_test_output >= end_test_output
            or eval_script_list[end_test_output + 1] != cleanup_command
        ):
            raise ValueError(f'{instance_id}: unexpected fixture placement')
        generated_script = '\n'.join(eval_script_list)
        if generated_script.count('# SWE-bench Sphinx LaTeX images exact-case local response fixture') != 1:
            raise ValueError(f'{instance_id}: unexpected fixture marker count')
    return '\n'.join(['#!/bin/bash', 'set -uxo pipefail'] + eval_script_list) + '\n'


def get_eval_report_map(
    instance: SWEbenchInstance, prediction: dict[str, str], test_log_path: str, include_tests_status: bool
) -> dict[str, Any]:
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance['repo']
    version = instance.get('version')

    def _from_json_or_obj(key: str) -> Any:
        """If key points to string, load with json"""
        instance_dict = cast(dict[str, Any], instance)
        if key not in instance_dict:
            # If P2P, F2P keys not found, it's a validation instance
            return []
        if isinstance(instance_dict[key], str):
            return json.loads(instance_dict[key])
        return instance_dict[key]

    pass_to_pass = _from_json_or_obj('PASS_TO_PASS')
    fail_to_pass = _from_json_or_obj('FAIL_TO_PASS')

    test_spec = TestSpec(
        instance_id=instance_id,
        repo=repo,
        version=version,
        eval_script_list=[],
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language=MAP_REPO_TO_EXT[repo],
    )

    return get_eval_report(test_spec, prediction, test_log_path, include_tests_status)


def get_problem_statement(instance: SWEbenchInstance) -> str:
    return instance.get('problem_statement') or instance.get('problem') or ''


def get_repo_language(instance: SWEbenchInstance) -> str | None:
    return MAP_REPO_TO_EXT.get(instance['repo'], None)
