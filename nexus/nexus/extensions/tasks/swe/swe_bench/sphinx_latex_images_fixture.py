import hashlib
import json

from dataclasses import dataclass
from typing import Any

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_bench.constants import KEY_INSTANCE_ID
from nexus.extensions.tasks.swe.swe_bench.test_spec import get_test_directives


@dataclass(frozen=True)
class _SphinxLatexImagesCase:
    version: str
    base_commit: str
    environment_setup_commit: str
    workdir: str
    test_patch_sha256: str
    fail_to_pass_count: int
    fail_to_pass_sha256: str
    pass_to_pass_count: int
    pass_to_pass_sha256: str
    post_test_file_sha256: str
    payload_sha256: str
    directives: tuple[str, ...]


_SPHINX_LATEX_IMAGES_INSTANCE_ID = 'sphinx-doc__sphinx-10435'
SPHINX_LATEX_IMAGES_TEST_FILE = 'tests/test_build_latex.py'
SPHINX_LATEX_IMAGES_PAYLOAD_FILE = 'tests/roots/test-images/rimg.png'
_SPHINX_LATEX_IMAGES_FIXTURE_MARKER = '# SWE-bench Sphinx LaTeX images exact-case local response fixture'
_SPHINX_LATEX_IMAGES_CASE = _SphinxLatexImagesCase(
    version='5.0',
    base_commit='f1061c012e214f16fd8790dec3c283d787e3daa8',
    environment_setup_commit='60775ec4c4ea08509eee4b564cbf90f316021aff',
    workdir='/testbed',
    test_patch_sha256='31b719cc3d54f539d1e5e3632599048a8d7e13878091477c75157942f429ab44',
    fail_to_pass_count=1,
    fail_to_pass_sha256='81af166cecf0c41065a07bb3fe17b405de1114831216e0adac88b1d19034f117',
    pass_to_pass_count=74,
    pass_to_pass_sha256='840febde1ac2cf6af4b5e617c37abdc8edbc01dd44565b39f089b9553991e563',
    post_test_file_sha256='f9251251619f8c31f43f469ebb91b1eb458b5a8a6ebbe3e3890d72c0fb8fba34',
    payload_sha256='3ff5183c3b40ebf3bc4f9c8f734a72948800f59f690d161576c251b34ba629c3',
    directives=(SPHINX_LATEX_IMAGES_TEST_FILE,),
)

_SPHINX_LATEX_IMAGES_TEST_FIXTURE = r"""# SWE-bench Sphinx LaTeX images exact-case local response fixture
import pytest


@pytest.fixture(autouse=True)
def _swe_bench_sphinx_latex_images_local_responses(monkeypatch):
    from pathlib import Path

    import requests
    import sphinx.transforms.post_transforms.images as sphinx_images

    python_logo_url = 'https://www.python.org/static/img/python-logo.png'
    missing_image_url = 'https://www.google.com/NOT_EXIST.PNG'
    payload = (Path(__file__).parent / 'roots/test-images/rimg.png').read_bytes()
    unexpected_requests = []

    def fake_get(url, *args, **kwargs):
        if url == python_logo_url:
            response = requests.Response()
            response.status_code = 200
            response.reason = 'OK'
            response.url = url
            response.headers['Content-Type'] = 'image/png'
            response._content = payload
            response._content_consumed = True
            response.request = requests.Request(method='GET', url=url).prepare()
            return response

        if url == missing_image_url:
            response = requests.Response()
            response.status_code = 404
            response.reason = 'Not Found'
            response.url = url
            response.headers['Content-Type'] = 'image/png'
            response._content = b''
            response._content_consumed = True
            response.request = requests.Request(method='GET', url=url).prepare()
            return response

        unexpected_requests.append(url)
        raise RuntimeError(f'unexpected Sphinx image request: {url}')

    monkeypatch.setattr(sphinx_images.requests, 'get', fake_get)
    try:
        yield
    finally:
        if unexpected_requests:
            pytest.fail(f'unexpected Sphinx image requests: {unexpected_requests!r}')
"""


def _load_test_status(instance: SWEbenchInstance, instance_id: str, key: str) -> list[str]:
    value: Any = instance.get(key)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f'{instance_id}: invalid {key}') from e

    if not isinstance(value, list) or not all(isinstance(test, str) for test in value):
        raise ValueError(f'{instance_id}: unexpected {key}')
    return value


def _canonical_test_status_sha256(tests: list[str]) -> str:
    encoded = json.dumps(tests, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def _make_fixture_command() -> str:
    case = _SPHINX_LATEX_IMAGES_CASE
    fixture_source = repr(_SPHINX_LATEX_IMAGES_TEST_FIXTURE)
    return f"""# Install the exact-case Sphinx LaTeX images fixture atomically.
if ! python - <<'EOF_SWE_BENCH_SPHINX_LATEX_IMAGES_FIXTURE'
import hashlib
import os
import stat
import tempfile

from pathlib import Path


test_path = Path({SPHINX_LATEX_IMAGES_TEST_FILE!r})
payload_path = Path({SPHINX_LATEX_IMAGES_PAYLOAD_FILE!r})
expected_test_sha256 = {case.post_test_file_sha256!r}
expected_payload_sha256 = {case.payload_sha256!r}
fixture_marker = (
    '# SWE-bench Sphinx LaTeX images exact-case '
    'local response fixture'
).encode()
fixture_source = {fixture_source}.encode()

original = test_path.read_bytes()
if hashlib.sha256(original).hexdigest() != expected_test_sha256:
    raise RuntimeError('unexpected post-official-patch Sphinx LaTeX test file')
if fixture_marker in original:
    raise RuntimeError('unexpected existing Sphinx LaTeX images fixture marker')

payload = payload_path.read_bytes()
if hashlib.sha256(payload).hexdigest() != expected_payload_sha256:
    raise RuntimeError('unexpected Sphinx LaTeX images payload')

original_mode = stat.S_IMODE(test_path.stat().st_mode)
temporary_path = None
try:
    with tempfile.NamedTemporaryFile(
        mode='wb',
        dir=test_path.parent,
        prefix=f'.{{test_path.name}}.',
        suffix='.tmp',
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(original)
        if original and not original.endswith(b'\\n'):
            temporary.write(b'\\n')
        temporary.write(b'\\n')
        temporary.write(fixture_source)
        if not fixture_source.endswith(b'\\n'):
            temporary.write(b'\\n')
        temporary.flush()
        os.fsync(temporary.fileno())

    os.chmod(temporary_path, original_mode)
    os.replace(temporary_path, test_path)
    temporary_path = None
    directory_fd = os.open(test_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    if temporary_path is not None:
        temporary_path.unlink(missing_ok=True)
EOF_SWE_BENCH_SPHINX_LATEX_IMAGES_FIXTURE
then
    git checkout {case.base_commit} {SPHINX_LATEX_IMAGES_TEST_FILE}
    exit 1
fi"""


def get_sphinx_latex_images_fixture_command(instance: SWEbenchInstance) -> str | None:
    instance_id = instance.get(KEY_INSTANCE_ID) or ''
    if instance_id != _SPHINX_LATEX_IMAGES_INSTANCE_ID:
        return None

    case = _SPHINX_LATEX_IMAGES_CASE
    expected_fields = {
        'repo': 'sphinx-doc/sphinx',
        'dataset_type': 'swe-bench',
        'version': case.version,
        'base_commit': case.base_commit,
        'environment_setup_commit': case.environment_setup_commit,
        'workdir': case.workdir,
    }
    for field, expected in expected_fields.items():
        if instance.get(field) != expected:
            raise ValueError(f'{instance_id}: unexpected {field}')

    test_patch = instance.get('test_patch')
    if not isinstance(test_patch, str):
        raise ValueError(f'{instance_id}: unexpected test_patch')
    if hashlib.sha256(test_patch.encode()).hexdigest() != case.test_patch_sha256:
        raise ValueError(f'{instance_id}: unexpected test_patch')

    fail_to_pass = _load_test_status(instance, instance_id, 'FAIL_TO_PASS')
    if (
        len(fail_to_pass) != case.fail_to_pass_count
        or _canonical_test_status_sha256(fail_to_pass) != case.fail_to_pass_sha256
    ):
        raise ValueError(f'{instance_id}: unexpected FAIL_TO_PASS')

    pass_to_pass = _load_test_status(instance, instance_id, 'PASS_TO_PASS')
    if (
        len(pass_to_pass) != case.pass_to_pass_count
        or _canonical_test_status_sha256(pass_to_pass) != case.pass_to_pass_sha256
    ):
        raise ValueError(f'{instance_id}: unexpected PASS_TO_PASS')

    if get_test_directives(instance) != list(case.directives):
        raise ValueError(f'{instance_id}: unexpected test_patch directives')

    command = _make_fixture_command()
    if command.count(_SPHINX_LATEX_IMAGES_FIXTURE_MARKER) != 1:
        raise ValueError(f'{instance_id}: unexpected fixture marker count')
    return command
