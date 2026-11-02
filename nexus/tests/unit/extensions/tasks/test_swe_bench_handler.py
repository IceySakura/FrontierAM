import json

import pytest

from nexus.extensions.tasks.swe.swe_bench import dataset_handler
from nexus.extensions.tasks.swe.swe_bench.constants import START_TEST_OUTPUT


_MATPLOTLIB_INSTANCE_ID = 'matplotlib__matplotlib-20488'
_MATPLOTLIB_BASE_COMMIT = 'b7ce415c15eb39b026a097a2865da73fbcf15c9c'
_MATPLOTLIB_HTTPS_TEST = 'lib/matplotlib/tests/test_image.py::test_https_imread_smoketest'
_MATPLOTLIB_TEST_FILE = 'lib/matplotlib/tests/test_image.py'
_LOCAL_URL_FIXTURE_MARKER = '# SWE-bench matplotlib local URL response fixture'
_TEST_PATCH = """diff --git a/lib/matplotlib/tests/test_image.py b/lib/matplotlib/tests/test_image.py
--- a/lib/matplotlib/tests/test_image.py
+++ b/lib/matplotlib/tests/test_image.py
@@ -1233,2 +1233,3 @@ def test_imshow_quantitynd():
     fig.canvas.draw()
+    # unit-test official patch marker
 @check_figures_equal(extensions=['png'])
"""


def _instance(**overrides: object) -> dict[str, object]:
    instance: dict[str, object] = {
        'instance_id': _MATPLOTLIB_INSTANCE_ID,
        'repo': 'matplotlib/matplotlib',
        'dataset_type': 'swe-bench',
        'version': '3.4',
        'base_commit': _MATPLOTLIB_BASE_COMMIT,
        'test_patch': _TEST_PATCH,
        'FAIL_TO_PASS': ['lib/matplotlib/tests/test_image.py::test_huge_range_log[png--1]'],
        'PASS_TO_PASS': [
            'lib/matplotlib/tests/test_image.py::test_image_interps[png]',
            _MATPLOTLIB_HTTPS_TEST,
            'lib/matplotlib/tests/test_image.py::test_nonuniform_and_pcolor[png]',
        ],
    }
    instance.update(overrides)
    return instance


@pytest.mark.parametrize('pass_to_pass_as_json', [False, True])
def test_matplotlib_https_test_is_replaced_with_local_non_seekable_response(
    pass_to_pass_as_json: bool,
) -> None:
    instance = _instance()
    if pass_to_pass_as_json:
        instance['PASS_TO_PASS'] = json.dumps(instance['PASS_TO_PASS'])

    script = dataset_handler.get_eval_script(instance)

    assert script.count(_LOCAL_URL_FIXTURE_MARKER) == 1
    assert script.index('# unit-test official patch marker') < script.index(_LOCAL_URL_FIXTURE_MARKER)
    assert script.index(_LOCAL_URL_FIXTURE_MARKER) < script.index(f": '{START_TEST_OUTPUT}'")
    assert f'pytest -rA {_MATPLOTLIB_TEST_FILE}' in script
    assert '-k ' not in script

    fixture = script[script.index(_LOCAL_URL_FIXTURE_MARKER) : script.index(f": '{START_TEST_OUTPUT}'")]
    assert 'def test_https_imread_smoketest(monkeypatch):' in fixture
    assert 'https://matplotlib.org/1.5.0/_static/logo2.png' in fixture
    assert 'monkeypatch.setattr' in fixture
    assert 'urllib.request' in fixture
    assert 'context' in fixture
    assert 'context.check_hostname' in fixture
    assert 'context.verify_mode' in fixture
    assert 'io.UnsupportedOperation' in fixture
    assert 'baseline_images/pngsuite/basn3p04.png' in fixture
    assert 'assert_array_equal' in fixture
    assert 'exit 1' in fixture
    assert '_create_unverified_context' not in fixture
    assert 'CERT_NONE' not in fixture
    assert 'PYTHONHTTPSVERIFY' not in fixture


def test_matplotlib_https_fixture_is_not_injected_for_other_instance() -> None:
    script = dataset_handler.get_eval_script(_instance(instance_id='matplotlib__matplotlib-20676'))

    assert _LOCAL_URL_FIXTURE_MARKER not in script
    assert f'pytest -rA {_MATPLOTLIB_TEST_FILE}' in script


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('repo', 'django/django'),
        ('dataset_type', 'swe-bench-plus-plus'),
        ('version', '3.3'),
        ('base_commit', '0000000000000000000000000000000000000000'),
        (
            'PASS_TO_PASS',
            [
                'lib/matplotlib/tests/test_image.py::test_image_interps[png]',
                'lib/matplotlib/tests/test_image.py::test_nonuniform_and_pcolor[png]',
            ],
        ),
        (
            'PASS_TO_PASS',
            [
                _MATPLOTLIB_HTTPS_TEST,
                _MATPLOTLIB_HTTPS_TEST,
            ],
        ),
        ('test_patch', _TEST_PATCH.replace(_MATPLOTLIB_TEST_FILE, 'lib/matplotlib/tests/test_axes.py')),
    ],
)
def test_matplotlib_target_metadata_drift_fails_closed(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=_MATPLOTLIB_INSTANCE_ID):
        dataset_handler.get_eval_script(_instance(**{field: value}))


def test_matplotlib_https_fixture_generation_is_idempotent() -> None:
    instance = _instance()

    first = dataset_handler.get_eval_script(instance)
    second = dataset_handler.get_eval_script(instance)

    assert first == second
    assert first.count(_LOCAL_URL_FIXTURE_MARKER) == 1
