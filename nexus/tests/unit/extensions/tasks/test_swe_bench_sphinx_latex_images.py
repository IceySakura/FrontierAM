import ast
import copy
import json

from typing import Any, cast

import pytest

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_bench import dataset_handler, sphinx_latex_images_fixture
from nexus.extensions.tasks.swe.swe_bench.constants import (
    END_TEST_OUTPUT,
    SPECS_SPHINX,
    START_TEST_OUTPUT,
    TEST_SPHINX,
)


_INSTANCE_ID = 'sphinx-doc__sphinx-10435'
_BASE_COMMIT = 'f1061c012e214f16fd8790dec3c283d787e3daa8'
_ENVIRONMENT_SETUP_COMMIT = '60775ec4c4ea08509eee4b564cbf90f316021aff'
_TEST_FILE = 'tests/test_build_latex.py'
_PAYLOAD_FILE = 'tests/roots/test-images/rimg.png'
_FIXTURE_MARKER = '# SWE-bench Sphinx LaTeX images exact-case local response fixture'
_PYTHON_LOGO_URL = 'https://www.python.org/static/img/python-logo.png'
_MISSING_IMAGE_URL = 'https://www.google.com/NOT_EXIST.PNG'
_TEST_PATCH = r"""diff --git a/tests/test_build_latex.py b/tests/test_build_latex.py
--- a/tests/test_build_latex.py
+++ b/tests/test_build_latex.py
@@ -1623,7 +1623,7 @@ def test_latex_code_role(app):
         r'\PYG{p}{)}'
         r'\PYG{p}{:} '
         r'\PYG{k}{pass}')
-    assert (r'Inline \sphinxcode{\sphinxupquote{' + '\n' +
-            common_content + '\n}} code block') in content
+    assert (r'Inline \sphinxcode{\sphinxupquote{%' + '\n' +
+            common_content + '%\n}} code block') in content
     assert (r'\begin{sphinxVerbatim}[commandchars=\\\{\}]' +
             '\n' + common_content + '\n' + r'\end{sphinxVerbatim}') in content
"""
_FAIL_TO_PASS = ['tests/test_build_latex.py::test_latex_code_role']
_PASS_TO_PASS = [
    'tests/test_build_latex.py::test_writer',
    'tests/test_build_latex.py::test_latex_warnings',
    'tests/test_build_latex.py::test_latex_basic',
    'tests/test_build_latex.py::test_latex_basic_manual',
    'tests/test_build_latex.py::test_latex_basic_howto',
    'tests/test_build_latex.py::test_latex_basic_manual_ja',
    'tests/test_build_latex.py::test_latex_basic_howto_ja',
    'tests/test_build_latex.py::test_latex_theme',
    'tests/test_build_latex.py::test_latex_theme_papersize',
    'tests/test_build_latex.py::test_latex_theme_options',
    'tests/test_build_latex.py::test_latex_additional_settings_for_language_code',
    'tests/test_build_latex.py::test_latex_additional_settings_for_greek',
    'tests/test_build_latex.py::test_latex_title_after_admonitions',
    'tests/test_build_latex.py::test_latex_release',
    'tests/test_build_latex.py::test_numref',
    'tests/test_build_latex.py::test_numref_with_prefix1',
    'tests/test_build_latex.py::test_numref_with_prefix2',
    'tests/test_build_latex.py::test_numref_with_language_ja',
    'tests/test_build_latex.py::test_latex_obey_numfig_is_false',
    'tests/test_build_latex.py::test_latex_obey_numfig_secnum_depth_is_zero',
    'tests/test_build_latex.py::test_latex_obey_numfig_secnum_depth_is_two',
    'tests/test_build_latex.py::test_latex_obey_numfig_but_math_numfig_false',
    'tests/test_build_latex.py::test_latex_add_latex_package',
    'tests/test_build_latex.py::test_babel_with_no_language_settings',
    'tests/test_build_latex.py::test_babel_with_language_de',
    'tests/test_build_latex.py::test_babel_with_language_ru',
    'tests/test_build_latex.py::test_babel_with_language_tr',
    'tests/test_build_latex.py::test_babel_with_language_ja',
    'tests/test_build_latex.py::test_babel_with_unknown_language',
    'tests/test_build_latex.py::test_polyglossia_with_language_de',
    'tests/test_build_latex.py::test_polyglossia_with_language_de_1901',
    'tests/test_build_latex.py::test_footnote',
    'tests/test_build_latex.py::test_reference_in_caption_and_codeblock_in_footnote',
    'tests/test_build_latex.py::test_footnote_referred_multiple_times',
    'tests/test_build_latex.py::test_latex_show_urls_is_inline',
    'tests/test_build_latex.py::test_latex_show_urls_is_footnote',
    'tests/test_build_latex.py::test_latex_show_urls_is_no',
    'tests/test_build_latex.py::test_latex_show_urls_footnote_and_substitutions',
    'tests/test_build_latex.py::test_image_in_section',
    'tests/test_build_latex.py::test_latex_logo_if_not_found',
    'tests/test_build_latex.py::test_toctree_maxdepth_manual',
    'tests/test_build_latex.py::test_toctree_maxdepth_howto',
    'tests/test_build_latex.py::test_toctree_not_found',
    'tests/test_build_latex.py::test_toctree_without_maxdepth',
    'tests/test_build_latex.py::test_toctree_with_deeper_maxdepth',
    'tests/test_build_latex.py::test_latex_toplevel_sectioning_is_None',
    'tests/test_build_latex.py::test_latex_toplevel_sectioning_is_part',
    'tests/test_build_latex.py::test_latex_toplevel_sectioning_is_part_with_howto',
    'tests/test_build_latex.py::test_latex_toplevel_sectioning_is_chapter',
    'tests/test_build_latex.py::test_latex_toplevel_sectioning_is_chapter_with_howto',
    'tests/test_build_latex.py::test_latex_toplevel_sectioning_is_section',
    'tests/test_build_latex.py::test_latex_table_tabulars',
    'tests/test_build_latex.py::test_latex_table_longtable',
    'tests/test_build_latex.py::test_latex_table_complex_tables',
    'tests/test_build_latex.py::test_latex_table_custom_template_caseA',
    'tests/test_build_latex.py::test_latex_table_custom_template_caseB',
    'tests/test_build_latex.py::test_latex_table_custom_template_caseC',
    'tests/test_build_latex.py::test_latex_raw_directive',
    'tests/test_build_latex.py::test_latex_images',
    'tests/test_build_latex.py::test_latex_index',
    'tests/test_build_latex.py::test_latex_equations',
    'tests/test_build_latex.py::test_latex_image_in_parsed_literal',
    'tests/test_build_latex.py::test_latex_nested_enumerated_list',
    'tests/test_build_latex.py::test_latex_thebibliography',
    'tests/test_build_latex.py::test_latex_glossary',
    'tests/test_build_latex.py::test_latex_labels',
    'tests/test_build_latex.py::test_latex_figure_in_admonition',
    'tests/test_build_latex.py::test_default_latex_documents',
    'tests/test_build_latex.py::test_index_on_title',
    'tests/test_build_latex.py::test_texescape_for_non_unicode_supported_engine',
    'tests/test_build_latex.py::test_texescape_for_unicode_supported_engine',
    'tests/test_build_latex.py::test_latex_elements_extrapackages',
    'tests/test_build_latex.py::test_latex_nested_tables',
    'tests/test_build_latex.py::test_latex_container',
]


def _target_instance(*, lists_as_json: bool = False) -> SWEbenchInstance:
    instance: SWEbenchInstance = {
        'instance_id': _INSTANCE_ID,
        'repo': 'sphinx-doc/sphinx',
        'dataset_type': 'swe-bench',
        'version': '5.0',
        'base_commit': _BASE_COMMIT,
        'environment_setup_commit': _ENVIRONMENT_SETUP_COMMIT,
        'workdir': '/testbed',
        'test_patch': _TEST_PATCH,
        'FAIL_TO_PASS': copy.deepcopy(_FAIL_TO_PASS),
        'PASS_TO_PASS': copy.deepcopy(_PASS_TO_PASS),
    }
    if lists_as_json:
        instance['FAIL_TO_PASS'] = json.dumps(instance['FAIL_TO_PASS'])
        instance['PASS_TO_PASS'] = json.dumps(instance['PASS_TO_PASS'])
    return instance


def _same_version_non_target(instance_id: str, base_commit: str, test_file: str) -> SWEbenchInstance:
    test_name = test_file.removeprefix('tests/').removesuffix('.py')
    return {
        'instance_id': instance_id,
        'repo': 'sphinx-doc/sphinx',
        'dataset_type': 'swe-bench',
        'version': '5.0',
        'base_commit': base_commit,
        'environment_setup_commit': _ENVIRONMENT_SETUP_COMMIT,
        'workdir': '/testbed',
        'test_patch': (
            f'diff --git a/{test_file} b/{test_file}\n'
            f'--- a/{test_file}\n'
            f'+++ b/{test_file}\n'
            '@@ -1 +1,2 @@\n'
            ' # synthetic same-version negative control\n'
            '+# official patch marker\n'
        ),
        'FAIL_TO_PASS': [f'{test_file}::test_{test_name}_target'],
        'PASS_TO_PASS': [f'{test_file}::test_{test_name}_regression'],
    }


def test_sphinx_latex_images_canonical_case_fingerprints_are_locked() -> None:
    case = sphinx_latex_images_fixture._SPHINX_LATEX_IMAGES_CASE

    assert case.version == '5.0'
    assert case.base_commit == _BASE_COMMIT
    assert case.environment_setup_commit == _ENVIRONMENT_SETUP_COMMIT
    assert case.workdir == '/testbed'
    assert case.test_patch_sha256 == '31b719cc3d54f539d1e5e3632599048a8d7e13878091477c75157942f429ab44'
    assert case.fail_to_pass_count == 1
    assert case.fail_to_pass_sha256 == '81af166cecf0c41065a07bb3fe17b405de1114831216e0adac88b1d19034f117'
    assert case.pass_to_pass_count == 74
    assert case.pass_to_pass_sha256 == '840febde1ac2cf6af4b5e617c37abdc8edbc01dd44565b39f089b9553991e563'
    assert case.post_test_file_sha256 == 'f9251251619f8c31f43f469ebb91b1eb458b5a8a6ebbe3e3890d72c0fb8fba34'
    assert case.payload_sha256 == '3ff5183c3b40ebf3bc4f9c8f734a72948800f59f690d161576c251b34ba629c3'
    assert case.directives == (_TEST_FILE,)
    assert sphinx_latex_images_fixture.SPHINX_LATEX_IMAGES_TEST_FILE == _TEST_FILE
    assert sphinx_latex_images_fixture.SPHINX_LATEX_IMAGES_PAYLOAD_FILE == _PAYLOAD_FILE


@pytest.mark.parametrize('lists_as_json', [False, True])
def test_sphinx_10435_uses_exact_case_local_responses(lists_as_json: bool) -> None:
    script = dataset_handler.get_eval_script(_target_instance(lists_as_json=lists_as_json))

    assert script.count(_FIXTURE_MARKER) == 1
    assert script.index("common_content + '%\\n}} code block") < script.index(_FIXTURE_MARKER)
    start_test_output = script.index(f": '{START_TEST_OUTPUT}'")
    end_test_output = script.index(f": '{END_TEST_OUTPUT}'")
    fixture_start = script.index(_FIXTURE_MARKER)
    cleanup = script.index(f'git checkout {_BASE_COMMIT} {_TEST_FILE}', end_test_output)
    test_command = f'{TEST_SPHINX} {_TEST_FILE}'

    assert fixture_start < start_test_output < script.index(test_command) < end_test_output < cleanup
    assert script[start_test_output:end_test_output].count(test_command) == 1
    assert '-k ' not in script[start_test_output:end_test_output]
    assert 'sed -i' not in script
    assert 'httpbin.org' not in script
    assert 'tests/roots/test-images/index.rst' not in script
    assert _PYTHON_LOGO_URL in script
    assert _MISSING_IMAGE_URL in script
    assert 'requests.Response()' in script
    assert 'response.status_code = 200' in script
    assert 'response.status_code = 404' in script
    assert _PAYLOAD_FILE in script
    assert 'unexpected_requests.append' in script
    assert 'pytest.fail' in script
    assert 'request.node.name' not in script
    assert "monkeypatch.setattr(sphinx_images.requests, 'get', fake_get)" in script
    assert 'original_get' not in script
    assert '/etc/hosts' not in script
    assert 'verify=False' not in script
    assert 'CERT_NONE' not in script
    assert 'Session.send' not in script
    assert 'HTTPAdapter.send' not in script
    assert 'socket.' not in script
    assert 'serve_forever' not in script
    assert 'subprocess.Popen' not in script
    assert 'pytest.skip' not in script
    assert 'xfail' not in script
    assert 'tempfile.NamedTemporaryFile' in script
    assert 'os.replace' in script
    assert case_hashes_are_in_setup(script)


def case_hashes_are_in_setup(script: str) -> bool:
    case = sphinx_latex_images_fixture._SPHINX_LATEX_IMAGES_CASE
    return (
        case.post_test_file_sha256 in script
        and case.payload_sha256 in script
        and 'unexpected existing Sphinx LaTeX images fixture marker' in script
    )


def test_sphinx_latex_images_injected_fixture_source_is_valid_and_fail_closed() -> None:
    source = sphinx_latex_images_fixture._SPHINX_LATEX_IMAGES_TEST_FIXTURE

    ast.parse(source)
    assert '@pytest.fixture(autouse=True)' in source
    assert 'yield' in source
    assert 'finally:' in source
    assert 'unexpected_requests = []' in source
    assert 'unexpected_requests.append' in source
    assert 'raise RuntimeError' in source
    assert 'pytest.fail' in source
    assert _PYTHON_LOGO_URL in source
    assert _MISSING_IMAGE_URL in source
    assert 'requests.Response()' in source
    assert "monkeypatch.setattr(sphinx_images.requests, 'get', fake_get)" in source
    assert 'original_get' not in source
    assert 'request.node.name' not in source


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('repo', 'django/django'),
        ('dataset_type', 'swe-bench-plus-plus'),
        ('version', '5.1'),
        ('base_commit', '0000000000000000000000000000000000000000'),
        ('environment_setup_commit', '0000000000000000000000000000000000000000'),
        ('workdir', '/workspace'),
        ('test_patch', _TEST_PATCH + '\n'),
        ('FAIL_TO_PASS', []),
        ('FAIL_TO_PASS', [_FAIL_TO_PASS[0], _FAIL_TO_PASS[0]]),
        ('FAIL_TO_PASS', [1]),
        ('PASS_TO_PASS', []),
        ('PASS_TO_PASS', [_PASS_TO_PASS[0], _PASS_TO_PASS[0], *_PASS_TO_PASS[2:]]),
        ('PASS_TO_PASS', [1, *_PASS_TO_PASS[1:]]),
        ('PASS_TO_PASS', [*_PASS_TO_PASS[1:], _PASS_TO_PASS[0]]),
        ('test_patch', _TEST_PATCH.replace(_TEST_FILE, 'tests/test_build_html.py')),
    ],
)
def test_sphinx_10435_metadata_drift_fails_closed(field: str, value: object) -> None:
    instance = _target_instance()
    cast(dict[str, Any], instance)[field] = value

    with pytest.raises(ValueError, match=_INSTANCE_ID):
        dataset_handler.get_eval_script(instance)


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('FAIL_TO_PASS', '{'),
        ('FAIL_TO_PASS', 'null'),
        ('FAIL_TO_PASS', '{}'),
        ('PASS_TO_PASS', '{'),
        ('PASS_TO_PASS', 'null'),
        ('PASS_TO_PASS', '{}'),
    ],
)
def test_sphinx_10435_invalid_test_status_json_fails_closed(field: str, value: str) -> None:
    instance = _target_instance()
    cast(dict[str, Any], instance)[field] = value

    with pytest.raises(ValueError, match=_INSTANCE_ID):
        dataset_handler.get_eval_script(instance)


@pytest.mark.parametrize(
    ('instance_id', 'base_commit', 'test_file'),
    [
        ('sphinx-doc__sphinx-10466', 'cab2d93076d0cca7c53fac885f927dde3e2a5fec', 'tests/test_build_gettext.py'),
        ('sphinx-doc__sphinx-10323', '31eba1a76dd485dc633cae48227b46879eda5df4', 'tests/test_directive_code.py'),
    ],
)
def test_sphinx_latex_images_fixture_does_not_leak_to_same_version_cases(
    instance_id: str,
    base_commit: str,
    test_file: str,
) -> None:
    instance = _same_version_non_target(instance_id, base_commit, test_file)

    before = dataset_handler.get_eval_script(instance)
    dataset_handler.get_eval_script(_target_instance())
    after = dataset_handler.get_eval_script(instance)

    assert before == after
    assert _FIXTURE_MARKER not in after
    assert 'httpbin.org' not in after
    assert 'sed -i' not in after
    assert _PYTHON_LOGO_URL not in after
    assert _MISSING_IMAGE_URL not in after
    assert f'{TEST_SPHINX} {test_file}' in after
    assert f'git checkout {_BASE_COMMIT} {_TEST_FILE}' not in after


def test_sphinx_50_no_longer_has_version_wide_latex_images_workaround() -> None:
    command = SPECS_SPHINX['5.0']['test_cmd']

    assert command == TEST_SPHINX
    assert 'httpbin.org' not in command
    assert 'sed -i' not in command
    assert 'test-images/index.rst' not in command
    assert _TEST_FILE not in command


def test_sphinx_latex_images_fixture_generation_is_idempotent() -> None:
    instance = _target_instance()

    first = dataset_handler.get_eval_script(instance)
    second = dataset_handler.get_eval_script(instance)

    assert first == second
    assert first.count(_FIXTURE_MARKER) == 1


@pytest.mark.parametrize(
    'drift',
    ['missing_start', 'duplicate_start', 'reversed_markers', 'sleep', 'test_command', 'cleanup'],
)
def test_sphinx_10435_eval_script_shape_drift_fails_closed(
    drift: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_make_eval_script_list = dataset_handler.make_eval_script_list

    def make_drifted_eval_script_list(*args: object, **kwargs: object) -> list[str]:
        commands = original_make_eval_script_list(*args, **kwargs)
        start_marker = f": '{START_TEST_OUTPUT}'"
        end_marker = f": '{END_TEST_OUTPUT}'"
        start = commands.index(start_marker)
        end = commands.index(end_marker)
        if drift == 'missing_start':
            commands.pop(start)
        elif drift == 'duplicate_start':
            commands.insert(start, start_marker)
        elif drift == 'reversed_markers':
            commands[start], commands[end] = commands[end], commands[start]
        elif drift == 'sleep':
            commands[start + 1] = 'sleep 1'
        elif drift == 'test_command':
            commands[end - 1] = f'{TEST_SPHINX} {_TEST_FILE} -k test_latex_code_role'
        elif drift == 'cleanup':
            commands[end + 1] = 'git status --short'
        return commands

    monkeypatch.setattr(dataset_handler, 'make_eval_script_list', make_drifted_eval_script_list)

    with pytest.raises(ValueError, match=_INSTANCE_ID):
        dataset_handler.get_eval_script(_target_instance())
