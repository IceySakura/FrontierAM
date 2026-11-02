import hashlib
import json

from dataclasses import dataclass
from typing import Any

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_bench.constants import KEY_INSTANCE_ID
from nexus.extensions.tasks.swe.swe_bench.test_spec import get_test_directives


@dataclass(frozen=True)
class _SphinxLinkcheckCase:
    version: str
    base_commit: str
    test_patch_sha256: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    directives: tuple[str, ...]


SPHINX_LINKCHECK_TEST_FILE = 'tests/test_build_linkcheck.py'
_SPHINX_LINKCHECK_CASES = {
    'sphinx-doc__sphinx-7985': _SphinxLinkcheckCase(
        version='3.2',
        base_commit='f30284ef926ebaf04b176f21b421e2dffc679792',
        test_patch_sha256='a7838eb1fc35e0604d8cf83187b1005c5cc3c8b3323341fd8b52cf7b278d34d7',
        fail_to_pass=(
            'tests/test_build_linkcheck.py::test_defaults',
            'tests/test_build_linkcheck.py::test_anchors_ignored',
        ),
        pass_to_pass=(
            'tests/test_build_linkcheck.py::test_defaults_json',
            'tests/test_build_linkcheck.py::test_auth',
            'tests/test_build_linkcheck.py::test_linkcheck_request_headers',
        ),
        directives=(SPHINX_LINKCHECK_TEST_FILE,),
    ),
    'sphinx-doc__sphinx-8269': _SphinxLinkcheckCase(
        version='3.3',
        base_commit='1e2ccd8f0eca0870cf6f8fce6934e2da8eba9b72',
        test_patch_sha256='90b91b5a11cf12c3b66e09e3565db97eab511467a004f991d06daf90b532ea14',
        fail_to_pass=('tests/test_build_linkcheck.py::test_raises_for_invalid_status',),
        pass_to_pass=(
            'tests/test_build_linkcheck.py::test_defaults',
            'tests/test_build_linkcheck.py::test_defaults_json',
            'tests/test_build_linkcheck.py::test_anchors_ignored',
            'tests/test_build_linkcheck.py::test_auth',
            'tests/test_build_linkcheck.py::test_linkcheck_request_headers',
        ),
        directives=(
            'tests/roots/test-linkcheck-localserver/conf.py',
            'tests/roots/test-linkcheck-localserver/index.rst',
            SPHINX_LINKCHECK_TEST_FILE,
        ),
    ),
    'sphinx-doc__sphinx-8475': _SphinxLinkcheckCase(
        version='3.4',
        base_commit='3ea1ec84cc610f7a9f4f6b354e264565254923ff',
        test_patch_sha256='805ddf0fe1635f793420f7e16d069d216c3078aa2d0f7d70900c75809b3e0d9a',
        fail_to_pass=('tests/test_build_linkcheck.py::test_TooManyRedirects_on_HEAD',),
        pass_to_pass=(
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
        ),
        directives=(SPHINX_LINKCHECK_TEST_FILE,),
    ),
}


def _load_test_status(instance: SWEbenchInstance, instance_id: str, key: str) -> list[str]:
    value: Any = instance.get(key)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f'{instance_id}: invalid {key}') from e

    if not isinstance(value, list) or not all(isinstance(test, str) for test in value):
        raise ValueError(f'{instance_id}: unexpected {key}')
    if len(value) != len(set(value)):
        raise ValueError(f'{instance_id}: duplicate {key}')
    return value


def needs_sphinx_linkcheck_local_response_fixture(instance: SWEbenchInstance) -> bool:
    instance_id = instance.get(KEY_INSTANCE_ID) or ''
    case = _SPHINX_LINKCHECK_CASES.get(instance_id)
    if case is None:
        return False

    expected_fields = {
        'repo': 'sphinx-doc/sphinx',
        'dataset_type': 'swe-bench',
        'version': case.version,
        'base_commit': case.base_commit,
    }
    for field, expected in expected_fields.items():
        if instance.get(field) != expected:
            raise ValueError(f'{instance_id}: unexpected {field}')

    test_patch = instance.get('test_patch')
    if not isinstance(test_patch, str):
        raise ValueError(f'{instance_id}: unexpected test_patch')
    test_patch_sha256 = hashlib.sha256(test_patch.encode()).hexdigest()
    if test_patch_sha256 != case.test_patch_sha256:
        raise ValueError(f'{instance_id}: unexpected test_patch')

    fail_to_pass = _load_test_status(instance, instance_id, 'FAIL_TO_PASS')
    if fail_to_pass != list(case.fail_to_pass):
        raise ValueError(f'{instance_id}: unexpected FAIL_TO_PASS')

    pass_to_pass = _load_test_status(instance, instance_id, 'PASS_TO_PASS')
    if pass_to_pass != list(case.pass_to_pass):
        raise ValueError(f'{instance_id}: unexpected PASS_TO_PASS')

    if get_test_directives(instance) != list(case.directives):
        raise ValueError(f'{instance_id}: unexpected test_patch directives')

    return True


SPHINX_LINKCHECK_LOCAL_RESPONSE_FIXTURE_COMMAND = r"""# SWE-bench Sphinx linkcheck local response fixture
if ! cat >> tests/test_build_linkcheck.py <<'EOF_SWE_BENCH_SPHINX_LINKCHECK_LOCAL_RESPONSE_FIXTURE'


@pytest.fixture(autouse=True)
def _swe_bench_sphinx_linkcheck_local_responses(request, monkeypatch):
    if request.node.name not in {'test_defaults', 'test_defaults_json', 'test_anchors_ignored'}:
        return

    from urllib.parse import urlsplit

    import requests
    import sphinx.builders.linkcheck

    original_requests = {
        'GET': sphinx.builders.linkcheck.requests.get,
        'HEAD': sphinx.builders.linkcheck.requests.head,
    }
    response_specs = {
        (
            'GET',
            'https://www.w3.org/TR/2006/REC-xml-names-20060816/',
        ): (200, 'OK', b'<html><body><a id="defaulting"></a></body></html>'),
        (
            'GET',
            'http://www.sphinx-doc.org/en/1.4.8/tutorial.html',
        ): (200, 'OK', b'<html><body><a id="install-sphinx"></a></body></html>'),
        (
            'GET',
            'https://www.google.com/',
        ): (200, 'OK', b'<html><body>google fixture</body></html>'),
        (
            'GET',
            'https://www.google.com',
        ): (200, 'OK', b'<html><body>google fixture</body></html>'),
        (
            'HEAD',
            'https://www.google.com/',
        ): (200, 'OK', b''),
        (
            'HEAD',
            'https://www.google.com',
        ): (200, 'OK', b''),
        (
            'GET',
            'http://www.sphinx-doc.org/en/1.7/intro.html',
        ): (200, 'OK', b'<html><body>sphinx intro fixture</body></html>'),
        (
            'HEAD',
            'https://www.google.com/image.png',
        ): (404, 'Not Found', b''),
        (
            'GET',
            'https://www.google.com/image.png',
        ): (404, 'Not Found', b''),
        (
            'HEAD',
            'https://www.google.com/image2.png',
        ): (404, 'Not Found', b''),
        (
            'GET',
            'https://www.google.com/image2.png',
        ): (404, 'Not Found', b''),
    }

    def make_response(method, url, status_code, reason, content):
        response = requests.Response()
        response.status_code = status_code
        response.reason = reason
        response.url = url
        response.encoding = 'utf-8'
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response._content = content
        response._content_consumed = True
        response.request = requests.Request(method=method, url=url).prepare()
        return response

    def dispatch(method, url, *args, **kwargs):
        hostname = urlsplit(url).hostname
        if hostname in {'localhost', '127.0.0.1', '::1'}:
            return original_requests[method](url, *args, **kwargs)

        try:
            status_code, reason, content = response_specs[(method, url)]
        except KeyError:
            raise RuntimeError(f'unexpected external {method} request: {url}')
        return make_response(method, url, status_code, reason, content)

    def get(url, *args, **kwargs):
        return dispatch('GET', url, *args, **kwargs)

    def head(url, *args, **kwargs):
        return dispatch('HEAD', url, *args, **kwargs)

    monkeypatch.setattr(sphinx.builders.linkcheck.requests, 'get', get)
    monkeypatch.setattr(sphinx.builders.linkcheck.requests, 'head', head)
EOF_SWE_BENCH_SPHINX_LINKCHECK_LOCAL_RESPONSE_FIXTURE
then
    echo 'Failed to append Sphinx linkcheck local response fixture' >&2
    exit 1
fi"""
