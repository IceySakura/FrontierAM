# ruff: noqa: W293

import copy
import hashlib
import json

from typing import Any, TypedDict, cast

import pytest

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_bench import dataset_handler
from nexus.extensions.tasks.swe.swe_bench.constants import END_TEST_OUTPUT, START_TEST_OUTPUT


_FIXTURE_MARKER = '# SWE-bench Sphinx linkcheck local response fixture'
_TEST_FILE = 'tests/test_build_linkcheck.py'


class _CaseData(TypedDict):
    version: str
    base_commit: str
    test_patch_sha256: str
    test_patch: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    official_patch_marker: str
    directives: list[str]


_CASES: dict[str, _CaseData] = {
    'sphinx-doc__sphinx-7985': {
        'version': '3.2',
        'base_commit': 'f30284ef926ebaf04b176f21b421e2dffc679792',
        'test_patch_sha256': 'a7838eb1fc35e0604d8cf83187b1005c5cc3c8b3323341fd8b52cf7b278d34d7',
        'test_patch': r"""diff --git a/tests/roots/test-linkcheck/links.txt b/tests/roots/test-linkcheck/links.txt
--- a/tests/roots/test-linkcheck/links.txt
+++ b/tests/roots/test-linkcheck/links.txt
@@ -11,6 +11,8 @@ Some additional anchors to exercise ignore code
 * `Example Bar invalid <https://www.google.com/#top>`_
 * `Example anchor invalid <http://www.sphinx-doc.org/en/1.7/intro.html#does-not-exist>`_
 * `Complete nonsense <https://localhost:7777/doesnotexist>`_
+* `Example valid local file <conf.py>`_
+* `Example invalid local file <path/to/notfound>`_
 
 .. image:: https://www.google.com/image.png
 .. figure:: https://www.google.com/image2.png
diff --git a/tests/test_build_linkcheck.py b/tests/test_build_linkcheck.py
--- a/tests/test_build_linkcheck.py
+++ b/tests/test_build_linkcheck.py
@@ -30,7 +30,9 @@ def test_defaults(app, status, warning):
     # images should fail
     assert "Not Found for url: https://www.google.com/image.png" in content
     assert "Not Found for url: https://www.google.com/image2.png" in content
-    assert len(content.splitlines()) == 5
+    # looking for local file should fail
+    assert "[broken] path/to/notfound" in content
+    assert len(content.splitlines()) == 6
 
 
 @pytest.mark.sphinx('linkcheck', testroot='linkcheck', freshenv=True)
@@ -47,8 +49,8 @@ def test_defaults_json(app, status, warning):
                  "info"]:
         assert attr in row
 
-    assert len(content.splitlines()) == 8
-    assert len(rows) == 8
+    assert len(content.splitlines()) == 10
+    assert len(rows) == 10
     # the output order of the rows is not stable
     # due to possible variance in network latency
     rowsby = {row["uri"]:row for row in rows}
@@ -69,7 +71,7 @@ def test_defaults_json(app, status, warning):
     assert dnerow['uri'] == 'https://localhost:7777/doesnotexist'
     assert rowsby['https://www.google.com/image2.png'] == {
         'filename': 'links.txt',
-        'lineno': 16,
+        'lineno': 18,
         'status': 'broken',
         'code': 0,
         'uri': 'https://www.google.com/image2.png',
@@ -92,7 +94,8 @@ def test_defaults_json(app, status, warning):
                        'https://localhost:7777/doesnotexist',
                        'http://www.sphinx-doc.org/en/1.7/intro.html#',
                        'https://www.google.com/image.png',
-                       'https://www.google.com/image2.png']
+                       'https://www.google.com/image2.png',
+                       'path/to/notfound']
                    })
 def test_anchors_ignored(app, status, warning):
     app.builder.build_all()
""",
        'FAIL_TO_PASS': [
            'tests/test_build_linkcheck.py::test_defaults',
            'tests/test_build_linkcheck.py::test_anchors_ignored',
        ],
        'PASS_TO_PASS': [
            'tests/test_build_linkcheck.py::test_defaults_json',
            'tests/test_build_linkcheck.py::test_auth',
            'tests/test_build_linkcheck.py::test_linkcheck_request_headers',
        ],
        'official_patch_marker': '+* `Example valid local file <conf.py>`_',
        'directives': [_TEST_FILE],
    },
    'sphinx-doc__sphinx-8269': {
        'version': '3.3',
        'base_commit': '1e2ccd8f0eca0870cf6f8fce6934e2da8eba9b72',
        'test_patch_sha256': '90b91b5a11cf12c3b66e09e3565db97eab511467a004f991d06daf90b532ea14',
        'test_patch': r'''diff --git a/tests/roots/test-linkcheck-localserver/conf.py b/tests/roots/test-linkcheck-localserver/conf.py
new file mode 100644
--- /dev/null
+++ b/tests/roots/test-linkcheck-localserver/conf.py
@@ -0,0 +1,2 @@
+exclude_patterns = ['_build']
+linkcheck_anchors = True
diff --git a/tests/roots/test-linkcheck-localserver/index.rst b/tests/roots/test-linkcheck-localserver/index.rst
new file mode 100644
--- /dev/null
+++ b/tests/roots/test-linkcheck-localserver/index.rst
@@ -0,0 +1 @@
+`local server <http://localhost:7777/#anchor>`_
diff --git a/tests/test_build_linkcheck.py b/tests/test_build_linkcheck.py
--- a/tests/test_build_linkcheck.py
+++ b/tests/test_build_linkcheck.py
@@ -8,8 +8,10 @@
     :license: BSD, see LICENSE for details.
 """
 
+import http.server
 import json
 import re
+import threading
 from unittest import mock
 import pytest
 
@@ -106,6 +108,21 @@ def test_anchors_ignored(app, status, warning):
     # expect all ok when excluding #top
     assert not content
 
+@pytest.mark.sphinx('linkcheck', testroot='linkcheck-localserver', freshenv=True)
+def test_raises_for_invalid_status(app, status, warning):
+    server_thread = HttpServerThread(InternalServerErrorHandler, daemon=True)
+    server_thread.start()
+    try:
+        app.builder.build_all()
+    finally:
+        server_thread.terminate()
+    content = (app.outdir / 'output.txt').read_text()
+    assert content == (
+        "index.rst:1: [broken] http://localhost:7777/#anchor: "
+        "500 Server Error: Internal Server Error "
+        "for url: http://localhost:7777/\n"
+    )
+
 
 @pytest.mark.sphinx(
     'linkcheck', testroot='linkcheck', freshenv=True,
@@ -160,3 +177,22 @@ def test_linkcheck_request_headers(app, status, warning):
                 assert headers["X-Secret"] == "open sesami"
             else:
                 assert headers["Accept"] == "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8"
+
+
+class HttpServerThread(threading.Thread):
+    def __init__(self, handler, *args, **kwargs):
+        super().__init__(*args, **kwargs)
+        self.server = http.server.HTTPServer(("localhost", 7777), handler)
+
+    def run(self):
+        self.server.serve_forever(poll_interval=0.01)
+
+    def terminate(self):
+        self.server.shutdown()
+        self.server.server_close()
+        self.join()
+
+
+class InternalServerErrorHandler(http.server.BaseHTTPRequestHandler):
+    def do_GET(self):
+        self.send_error(500, "Internal Server Error")
''',
        'FAIL_TO_PASS': ['tests/test_build_linkcheck.py::test_raises_for_invalid_status'],
        'PASS_TO_PASS': [
            'tests/test_build_linkcheck.py::test_defaults',
            'tests/test_build_linkcheck.py::test_defaults_json',
            'tests/test_build_linkcheck.py::test_anchors_ignored',
            'tests/test_build_linkcheck.py::test_auth',
            'tests/test_build_linkcheck.py::test_linkcheck_request_headers',
        ],
        'official_patch_marker': '+def test_raises_for_invalid_status(app, status, warning):',
        'directives': [
            'tests/roots/test-linkcheck-localserver/conf.py',
            'tests/roots/test-linkcheck-localserver/index.rst',
            _TEST_FILE,
        ],
    },
    'sphinx-doc__sphinx-8475': {
        'version': '3.4',
        'base_commit': '3ea1ec84cc610f7a9f4f6b354e264565254923ff',
        'test_patch_sha256': '805ddf0fe1635f793420f7e16d069d216c3078aa2d0f7d70900c75809b3e0d9a',
        'test_patch': r"""diff --git a/tests/test_build_linkcheck.py b/tests/test_build_linkcheck.py
--- a/tests/test_build_linkcheck.py
+++ b/tests/test_build_linkcheck.py
@@ -382,3 +382,31 @@ def test_connect_to_selfsigned_nonexistent_cert_file(app):
         "uri": "https://localhost:7777/",
         "info": "Could not find a suitable TLS CA certificate bundle, invalid path: does/not/exist",
     }
+
+
+@pytest.mark.sphinx('linkcheck', testroot='linkcheck-localserver', freshenv=True)
+def test_TooManyRedirects_on_HEAD(app):
+    class InfiniteRedirectOnHeadHandler(http.server.BaseHTTPRequestHandler):
+        def do_HEAD(self):
+            self.send_response(302, "Found")
+            self.send_header("Location", "http://localhost:7777/")
+            self.end_headers()
+
+        def do_GET(self):
+            self.send_response(200, "OK")
+            self.end_headers()
+            self.wfile.write(b"ok\n")
+
+    with http_server(InfiniteRedirectOnHeadHandler):
+        app.builder.build_all()
+
+    with open(app.outdir / 'output.json') as fp:
+        content = json.load(fp)
+    assert content == {
+        "code": 0,
+        "status": "working",
+        "filename": "index.rst",
+        "lineno": 1,
+        "uri": "http://localhost:7777/",
+        "info": "",
+    }
""",
        'FAIL_TO_PASS': ['tests/test_build_linkcheck.py::test_TooManyRedirects_on_HEAD'],
        'PASS_TO_PASS': [
            'tests/test_build_linkcheck.py::test_defaults',
            'tests/test_build_linkcheck.py::test_defaults_json',
            'tests/test_build_linkcheck.py::test_anchors_ignored',
            'tests/test_build_linkcheck.py::test_raises_for_invalid_status',
            'tests/test_build_linkcheck.py::test_auth_header_uses_first_match',
            'tests/test_build_linkcheck.py::test_auth_header_no_match',
            'tests/test_build_linkcheck.py::test_linkcheck_request_headers',
            'tests/test_build_linkcheck.py::test_linkcheck_request_headers_no_slash',
            'tests/test_build_linkcheck.py::test_linkcheck_request_headers_default',
            'tests/test_build_linkcheck.py::test_follows_redirects_on_HEAD',
            'tests/test_build_linkcheck.py::test_follows_redirects_on_GET',
            'tests/test_build_linkcheck.py::test_invalid_ssl',
            'tests/test_build_linkcheck.py::test_connect_to_selfsigned_fails',
            'tests/test_build_linkcheck.py::test_connect_to_selfsigned_with_tls_verify_false',
            'tests/test_build_linkcheck.py::test_connect_to_selfsigned_with_tls_cacerts',
            'tests/test_build_linkcheck.py::test_connect_to_selfsigned_with_requests_env_var',
            'tests/test_build_linkcheck.py::test_connect_to_selfsigned_nonexistent_cert_file',
        ],
        'official_patch_marker': '+def test_TooManyRedirects_on_HEAD(app):',
        'directives': [_TEST_FILE],
    },
}


def _instance(instance_id: str, *, lists_as_json: bool = False) -> SWEbenchInstance:
    case = _CASES[instance_id]
    instance: SWEbenchInstance = {
        'instance_id': instance_id,
        'repo': 'sphinx-doc/sphinx',
        'dataset_type': 'swe-bench',
        'version': case['version'],
        'base_commit': case['base_commit'],
        'test_patch': case['test_patch'],
        'FAIL_TO_PASS': copy.deepcopy(case['FAIL_TO_PASS']),
        'PASS_TO_PASS': copy.deepcopy(case['PASS_TO_PASS']),
    }
    if lists_as_json:
        instance['FAIL_TO_PASS'] = json.dumps(instance['FAIL_TO_PASS'])
        instance['PASS_TO_PASS'] = json.dumps(instance['PASS_TO_PASS'])
    return instance


def _non_target_instance(instance_id: str, version: str, base_commit: str) -> SWEbenchInstance:
    test_patch = """diff --git a/tests/test_placeholder.py b/tests/test_placeholder.py
--- a/tests/test_placeholder.py
+++ b/tests/test_placeholder.py
@@ -1 +1,2 @@
 def test_placeholder():
+    pass
"""
    return {
        'instance_id': instance_id,
        'repo': 'sphinx-doc/sphinx',
        'dataset_type': 'swe-bench',
        'version': version,
        'base_commit': base_commit,
        'test_patch': test_patch,
        'FAIL_TO_PASS': ['tests/test_placeholder.py::test_placeholder'],
        'PASS_TO_PASS': [],
    }


@pytest.mark.parametrize('instance_id', _CASES)
def test_sphinx_linkcheck_family_uses_exact_case_local_response_fixture(instance_id: str) -> None:
    case = _CASES[instance_id]
    script = dataset_handler.get_eval_script(_instance(instance_id))

    assert script.count(_FIXTURE_MARKER) == 1
    assert script.index(case['official_patch_marker']) < script.index(_FIXTURE_MARKER)
    assert script.index(_FIXTURE_MARKER) < script.index(f": '{START_TEST_OUTPUT}'")
    assert script.index(f": '{END_TEST_OUTPUT}'") < script.rindex(f'git checkout {case["base_commit"]} {_TEST_FILE}')

    fixture = script[script.index(_FIXTURE_MARKER) : script.index(f": '{START_TEST_OUTPUT}'")]
    assert 'test_defaults' in fixture
    assert 'test_defaults_json' in fixture
    assert 'test_anchors_ignored' in fixture
    assert 'sphinx.builders.linkcheck' in fixture
    assert 'requests.Response' in fixture
    assert '_content_consumed' in fixture
    assert 'https://www.w3.org/TR/2006/REC-xml-names-20060816/' in fixture
    assert 'https://www.google.com/' in fixture
    assert 'http://www.sphinx-doc.org/en/1.7/intro.html' in fixture
    assert 'https://www.google.com/image.png' in fixture
    assert 'https://www.google.com/image2.png' in fixture
    assert 'Not Found' in fixture
    assert 'localhost' in fixture
    assert 'unexpected external' in fixture.lower()

    assert 'httpbin.org' not in script
    assert 'NOT FOUND for url:' not in script
    assert 'pytest.mark.skip' not in fixture
    assert 'pytest.skip' not in fixture
    assert 'xfail' not in fixture
    assert 'Session.send' not in fixture
    assert 'verify=False' not in fixture
    assert 'CERT_NONE' not in fixture
    assert '_create_unverified_context' not in fixture
    assert '/etc/hosts' not in fixture
    assert 'socket.' not in fixture
    assert '-k ' not in script
    for directive in case['directives']:
        assert directive in script


@pytest.mark.parametrize('instance_id', _CASES)
def test_sphinx_linkcheck_canonical_test_patch_hashes_are_locked(instance_id: str) -> None:
    case = _CASES[instance_id]

    assert hashlib.sha256(case['test_patch'].encode()).hexdigest() == case['test_patch_sha256']


@pytest.mark.parametrize('lists_as_json', [False, True])
def test_sphinx_linkcheck_target_accepts_json_and_list_test_status(lists_as_json: bool) -> None:
    script = dataset_handler.get_eval_script(_instance('sphinx-doc__sphinx-8269', lists_as_json=lists_as_json))

    assert script.count(_FIXTURE_MARKER) == 1


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('repo', 'django/django'),
        ('dataset_type', 'swe-bench-plus-plus'),
        ('version', '3.2'),
        ('base_commit', '0000000000000000000000000000000000000000'),
        ('FAIL_TO_PASS', []),
        (
            'FAIL_TO_PASS',
            [
                'tests/test_build_linkcheck.py::test_raises_for_invalid_status',
                'tests/test_build_linkcheck.py::test_raises_for_invalid_status',
            ],
        ),
        ('PASS_TO_PASS', []),
        (
            'PASS_TO_PASS',
            [
                'tests/test_build_linkcheck.py::test_defaults',
                'tests/test_build_linkcheck.py::test_defaults',
            ],
        ),
        ('test_patch', _CASES['sphinx-doc__sphinx-8269']['test_patch'] + '\n'),
        (
            'test_patch',
            _CASES['sphinx-doc__sphinx-8269']['test_patch'].replace(
                'tests/test_build_linkcheck.py', 'tests/test_build_other.py'
            ),
        ),
    ],
)
def test_sphinx_linkcheck_target_metadata_drift_fails_closed(field: str, value: object) -> None:
    instance = _instance('sphinx-doc__sphinx-8269')
    cast(dict[str, Any], instance)[field] = value

    with pytest.raises(ValueError, match='sphinx-doc__sphinx-8269'):
        dataset_handler.get_eval_script(instance)


@pytest.mark.parametrize(
    ('instance_id', 'version', 'base_commit'),
    [
        ('sphinx-doc__sphinx-7889', '3.2', 'ec9af606c6cfa515f946d74da9b51574f2f9b16f'),
        ('sphinx-doc__sphinx-8265', '3.3', 'b428cd2404675475a5c3dc2a2b0790ba57676202'),
        ('sphinx-doc__sphinx-8459', '3.4', '68aa4fb29e7dfe521749e1e14f750d7afabb3481'),
    ],
)
def test_sphinx_linkcheck_fixture_does_not_leak_to_same_version_cases(
    instance_id: str,
    version: str,
    base_commit: str,
) -> None:
    instance = _non_target_instance(instance_id, version, base_commit)

    before = dataset_handler.get_eval_script(instance)
    dataset_handler.get_eval_script(_instance('sphinx-doc__sphinx-8269'))
    after = dataset_handler.get_eval_script(instance)

    assert before == after
    assert _FIXTURE_MARKER not in after
    assert 'httpbin.org' not in after
    assert 'www\\.google\\.com' not in after
    assert 'NOT FOUND for url:' not in after


@pytest.mark.parametrize('instance_id', _CASES)
def test_sphinx_linkcheck_fixture_generation_is_idempotent(instance_id: str) -> None:
    instance = _instance(instance_id)

    first = dataset_handler.get_eval_script(instance)
    second = dataset_handler.get_eval_script(instance)

    assert first == second
    assert first.count(_FIXTURE_MARKER) == 1
